"""Sync PESA-Fat stage-3 measurement workbooks into the NVITK database.

Reads per-subject Excel files produced by CT-PET v5 and Dixon v5 stage 3, upserts
long-form rows into ``image_measurements``, and rebuilds the SQLite catalog index once
at the end.

Use ``--from-source local`` for workstation result roots (``layout_local``) or
``--from-source sge`` to SFTP files from cluster storage (``layout_cluster``) into
the local mirror before publishing.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click

from nvitk.cluster.remote_transfer import (
    download_remote_file,
    remote_path_exists,
    resolve_cluster_host,
    sftp_session,
)
from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.db_publish import publish_stage3_excel, resolve_repo
from nvitk.pipes.pesa_fat.common.paths import (
    CLUSTER_HOST_ALIASES,
    BatchLayout,
    layout_cluster,
    layout_local,
    parse_subjects,
)
from nvitk.pipes.pesa_fat.common.stage3_batch_summary import aggregate_stage3_summary
from nvitk.pipes.pesa_fat.qc.measurements_table import stage3_measurements_xlsx_path

log = Logger()

PIPELINE_CHOICES = ("ct-pet-v5", "dixon-v5")
PesaFatQcPipeline = Literal["ct-pet-v5", "dixon-v5"]
FromSource = Literal["local", "sge"]


@dataclass
class SyncResult:
    published: int = 0
    skipped: int = 0
    downloaded: int = 0
    errors: list[str] = field(default_factory=list)


def _normalize_pipelines(pipelines: str) -> list[PesaFatQcPipeline]:
    pipes = [p.strip().lower() for p in pipelines.split(",") if p.strip()]
    bad = set(pipes) - set(PIPELINE_CHOICES)
    if bad:
        raise click.BadParameter(f"Unknown pipelines {bad}. Valid: {PIPELINE_CHOICES}")
    return pipes  # type: ignore[return-value]


def _resolve_ssh_credentials(
    *,
    remote_host: str | None,
    remote_user: str | None,
) -> tuple[str, str, str]:
    host = (remote_host or os.environ.get("NVITK_SGE_SSH_HOST", "")).strip()
    user = (remote_user or os.environ.get("NVITK_SGE_SSH_USER", "")).strip()
    password = os.environ.get("NVITK_SGE_SSH_PASSWORD", "")
    if not host:
        host = click.prompt("SSH hostname (short name or IP)")
    if not user:
        user = click.prompt("SSH user")
    if not password:
        password = getpass.getpass("SSH password: ")
    host_resolved = resolve_cluster_host(CLUSTER_HOST_ALIASES.get(host, host))
    return host_resolved, user, password


def _measurement_pairs(
    lay: BatchLayout,
    subjects: list[str],
    pipelines: list[PesaFatQcPipeline],
) -> list[tuple[str, PesaFatQcPipeline, Path]]:
    pairs: list[tuple[str, PesaFatQcPipeline, Path]] = []
    for subject in subjects:
        for pipeline in pipelines:
            pairs.append((subject, pipeline, stage3_measurements_xlsx_path(lay, subject, pipeline)))
    return pairs


def _download_measurements_from_cluster(
    *,
    cluster_lay: BatchLayout,
    local_lay: BatchLayout,
    subjects: list[str],
    pipelines: list[PesaFatQcPipeline],
    host: str,
    user: str,
    password: str,
    dry_run: bool,
) -> tuple[list[tuple[str, PesaFatQcPipeline, Path]], SyncResult]:
    """SFTP stage-3 workbooks from cluster paths into the local results mirror."""
    result = SyncResult()
    ready: list[tuple[str, PesaFatQcPipeline, Path]] = []
    remote_pairs = _measurement_pairs(cluster_lay, subjects, pipelines)
    local_pairs = _measurement_pairs(local_lay, subjects, pipelines)
    if len(remote_pairs) != len(local_pairs):
        raise RuntimeError("internal error: remote/local measurement pair mismatch")

    if dry_run:
        for (subject, pipeline, remote_path), (_subject2, _pipeline2, local_path) in zip(
            remote_pairs, local_pairs, strict=True
        ):
            log.info("[dry-run] would download %s -> %s (%s)", remote_path, local_path, pipeline)
            ready.append((subject, pipeline, local_path))
        result.downloaded = len(ready)
        return ready, result

    with sftp_session(host=host, user=user, password=password) as (_client, sftp):
        for (subject, pipeline, remote_path), (_subject2, _pipeline2, local_path) in zip(
            remote_pairs, local_pairs, strict=True
        ):
            remote_s = str(remote_path)
            if not remote_path_exists(sftp, remote_s):
                msg = f"{subject} / {pipeline}: remote file missing ({remote_s})"
                log.warning(msg)
                result.errors.append(msg)
                result.skipped += 1
                continue
            try:
                download_remote_file(sftp, remote_s, local_path)
                result.downloaded += 1
                ready.append((subject, pipeline, local_path))
                log.info("Downloaded %s -> %s", remote_s, local_path)
            except OSError as exc:
                msg = f"{subject} / {pipeline}: download failed ({exc})"
                log.warning(msg)
                result.errors.append(msg)
                result.skipped += 1
    return ready, result


def _publish_measurements(
    *,
    batch: str,
    pairs: list[tuple[str, PesaFatQcPipeline, Path]],
    dry_run: bool,
    skip_db: bool,
) -> SyncResult:
    result = SyncResult()
    if skip_db:
        for subject, pipeline, path in pairs:
            if path.is_file():
                log.info("[skip-db] found %s (%s / %s)", path, subject, pipeline)
                result.published += 1
            else:
                msg = f"{subject} / {pipeline}: local file missing ({path})"
                log.warning(msg)
                result.errors.append(msg)
                result.skipped += 1
        return result

    repo = resolve_repo()
    for subject, pipeline, path in pairs:
        if not path.is_file():
            msg = f"{subject} / {pipeline}: local file missing ({path})"
            log.warning(msg)
            result.errors.append(msg)
            result.skipped += 1
            continue
        if dry_run:
            log.info("[dry-run] would publish %s (%s / %s)", path, subject, pipeline)
            result.published += 1
            continue
        try:
            rows = publish_stage3_excel(
                subject_uid=subject,
                excel_path=path,
                pipeline=pipeline,
                repo=repo,
                build_sqlite_index=False,
                source_batch_id=batch,
            )
            if rows.empty:
                result.skipped += 1
            else:
                result.published += 1
                log.info("Published %d rows for %s / %s", len(rows), subject, pipeline)
        except Exception as exc:
            msg = f"{subject} / {pipeline}: DB publish failed ({exc})"
            log.warning(msg)
            result.errors.append(msg)
            result.skipped += 1

    if not dry_run and not skip_db and result.published > 0:
        log.info("Rebuilding SQLite catalog index …")
        repo.build_sqlite_index()
        log.info("SQLite index rebuilt at %s", repo.sqlite.db_path)
    return result


def sync_measurements(
    batch: str,
    subjects: list[str],
    *,
    pipelines: list[PesaFatQcPipeline],
    from_source: FromSource = "local",
    results_root: Path | None = None,
    remote_host: str | None = None,
    remote_user: str | None = None,
    dry_run: bool = False,
    skip_db: bool = False,
    aggregate: bool = True,
) -> SyncResult:
    """Sync stage-3 measurement workbooks for *subjects* into the NVITK DB."""
    local_lay = layout_local(batch, results_root=results_root)
    if not subjects:
        subjects = list(local_lay.iter_subjects())
    if not subjects:
        raise click.ClickException(
            f"No subjects to sync for batch {batch!r} (use --subjects or check nifti layout)."
        )

    overall = SyncResult()

    if from_source == "local":
        pairs = _measurement_pairs(local_lay, subjects, pipelines)
        pub = _publish_measurements(batch=batch, pairs=pairs, dry_run=dry_run, skip_db=skip_db)
        overall.published += pub.published
        overall.skipped += pub.skipped
        overall.errors.extend(pub.errors)
        target_lay = local_lay
    else:
        cluster_lay = layout_cluster(batch, results_root=results_root)
        host, user, password = _resolve_ssh_credentials(
            remote_host=remote_host,
            remote_user=remote_user,
        )
        pairs, dl = _download_measurements_from_cluster(
            cluster_lay=cluster_lay,
            local_lay=local_lay,
            subjects=subjects,
            pipelines=pipelines,
            host=host,
            user=user,
            password=password,
            dry_run=dry_run,
        )
        overall.downloaded += dl.downloaded
        overall.skipped += dl.skipped
        overall.errors.extend(dl.errors)
        pub = _publish_measurements(batch=batch, pairs=pairs, dry_run=dry_run, skip_db=skip_db)
        overall.published += pub.published
        overall.skipped += pub.skipped
        overall.errors.extend(pub.errors)
        target_lay = local_lay

    if aggregate and not dry_run and not skip_db and overall.published > 0:
        for pipeline in pipelines:
            try:
                aggregate_stage3_summary(target_lay, subjects, pipeline)
            except Exception as exc:
                msg = f"batch summary aggregation failed ({pipeline}): {exc}"
                log.warning(msg)
                overall.errors.append(msg)

    log.info(
        "Sync complete: published=%d skipped=%d downloaded=%d errors=%d",
        overall.published,
        overall.skipped,
        overall.downloaded,
        len(overall.errors),
    )
    return overall


@click.command("nvitk-pesa-fat-sync-measurements")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week2').")
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated PESA* subjects (default: all under local nifti batch).",
)
@click.option(
    "--pipelines",
    default=",".join(PIPELINE_CHOICES),
    show_default=True,
    help="Comma-separated pipelines (ct-pet-v5, dixon-v5).",
)
@click.option(
    "--from-source",
    "from_source",
    type=click.Choice(["local", "sge"]),
    default="local",
    show_default=True,
    help="Read measurements from local result roots or download from cluster via SFTP.",
)
@click.option(
    "--results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override local results root (download target for --from-source sge).",
)
@click.option("--remote-host", default=None, help="Cluster SSH host (alias ok; or NVITK_SGE_SSH_HOST).")
@click.option("--remote-user", default=None, help="Cluster SSH user (or NVITK_SGE_SSH_USER).")
@click.option("--dry-run", is_flag=True, help="Log actions without SFTP or DB writes.")
@click.option("--skip-db", is_flag=True, help="Download/verify files only; do not publish to DB.")
@click.option(
    "--no-aggregate",
    is_flag=True,
    default=False,
    help="Skip rebuilding batch SummaryCodebook xlsx after sync.",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    subjects: str | None,
    pipelines: str,
    from_source: str,
    results_root: Path | None,
    remote_host: str | None,
    remote_user: str | None,
    dry_run: bool,
    skip_db: bool,
    no_aggregate: bool,
    log_level: str,
) -> None:
    """Sync CT-PET and Dixon stage-3 measurements into the NVITK database."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    pipes = _normalize_pipelines(pipelines)
    local_lay = layout_local(batch, results_root=results_root)
    subj_list = parse_subjects(subjects) or list(local_lay.iter_subjects())
    if not subj_list:
        raise click.ClickException(
            f"No subjects found for batch {batch!r} (use --subjects or check local nifti-root)."
        )

    sync_measurements(
        batch,
        subj_list,
        pipelines=pipes,
        from_source=from_source,  # type: ignore[arg-type]
        results_root=results_root,
        remote_host=remote_host,
        remote_user=remote_user,
        dry_run=dry_run,
        skip_db=skip_db,
        aggregate=not no_aggregate,
    )


if __name__ == "__main__":
    main()
