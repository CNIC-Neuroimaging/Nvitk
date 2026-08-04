"""Publish qvtpy stage-6 measurements into ``image_measurements`` (pipeline ``4dflow_v3``).

Scans an ``--output-root`` for subjects with stage-6 CSV outputs and upserts them into
the DB. Use ``--from-sge`` to SFTP ``loc_measurements.csv`` and
``vessel_hemodynamics.csv`` from the cluster results tree into a temporary local
directory (removed after publish). ``--from-source`` selects which DataRepo to write
to (``local`` settings repo vs. the ``sge`` cluster dataset root).

PITC/PWV metrics are published from ``vessel_hemodynamics.csv`` via
:func:`nvitk.pipes.qvtpy.common.db_publish.publish_stage6`
(``pitc_slope``, ``pitc_intercept``, ``pwv``, ``pwv_fielding_xcor``, ``damping_index``).
"""

from __future__ import annotations

import getpass
import os
import shutil
import tempfile
from pathlib import Path

import click

from nvitk.cluster.remote_transfer import (
    download_remote_file,
    remote_path_exists,
    resolve_cluster_host,
    sftp_session,
)
from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.common.db_publish import publish_stage6, resolve_repo
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER
from nvitk.pipes.qvtpy.util.io.paths import CLUSTER_HOST_ALIASES, layout_cluster

log = Logger()

_STAGE6_FILES: tuple[str, ...] = (
    "loc_measurements.csv",
    "vessel_hemodynamics.csv",
)
_STAGE7_FILES: tuple[str, ...] = (STAGE7_SKIP_MARKER,)


def _stage6_dir(output_root: Path, subject: str) -> Path:
    """Stage 6 (measure) output directory for *subject* under *output_root*."""
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def _stage7_dir(output_root: Path, subject: str) -> Path:
    """Stage 7 (morphometrics) output directory for *subject* under *output_root*."""
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE7_MORPHOMETRICS_DIR


def _subjects_with_stage6(output_root: Path) -> list[str]:
    """Subjects under *output_root* with at least one stage-6 measurement CSV."""
    if not output_root.is_dir():
        return []
    out: list[str] = []
    for subj_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        s6 = _stage6_dir(output_root, subj_dir.name)
        if (s6 / "loc_measurements.csv").is_file() or (
            s6 / "vessel_hemodynamics.csv"
        ).is_file():
            out.append(subj_dir.name)
    return out


def _subjects_with_stage7(output_root: Path) -> list[str]:
    """Subjects under *output_root* with a completed stage-7 skip marker."""
    if not output_root.is_dir():
        return []
    out: list[str] = []
    for subj_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        if (_stage7_dir(output_root, subj_dir.name) / STAGE7_SKIP_MARKER).is_file():
            out.append(subj_dir.name)
    return out


def _subjects_with_stage6_or_stage7(output_root: Path) -> list[str]:
    """Union of subjects with stage-6 measurements and/or a completed stage-7 marker."""
    return sorted(set(_subjects_with_stage6(output_root)) | set(_subjects_with_stage7(output_root)))


def _resolve_ssh_credentials(
    *,
    remote_host: str | None,
    remote_user: str | None,
) -> tuple[str, str, str]:
    """Resolve (host, user, password) from args, env vars, or interactive prompts, resolving
    *host* through :data:`~nvitk.pipes.qvtpy.util.io.paths.CLUSTER_HOST_ALIASES`."""
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


def _download_stage6_from_sge(
    *,
    subjects: list[str],
    cluster_results_root: Path,
    local_temp_root: Path,
    host: str,
    user: str,
    password: str,
) -> list[str]:
    """SFTP stage-6 CSVs into ``local_temp_root``; return subjects with at least one file."""
    ready: list[str] = []
    with sftp_session(host=host, user=user, password=password) as (_client, sftp):
        for subject in subjects:
            remote_dir = _stage6_dir(cluster_results_root, subject)
            local_dir = _stage6_dir(local_temp_root, subject)
            local_dir.mkdir(parents=True, exist_ok=True)
            got_any = False
            for name in _STAGE6_FILES:
                remote_s = str(remote_dir / name)
                if not remote_path_exists(sftp, remote_s):
                    log.warning("[%s] remote missing: %s", subject, remote_s)
                    continue
                local_path = local_dir / name
                try:
                    download_remote_file(sftp, remote_s, local_path)
                    got_any = True
                    log.info("Downloaded %s -> %s", remote_s, local_path)
                except OSError as exc:
                    log.warning("[%s] download failed for %s: %s", subject, name, exc)
            if got_any:
                ready.append(subject)
            else:
                log.warning("[%s] no stage6 measurement files on cluster", subject)
    return ready


def _download_stage6_and_stage7_from_sge(
    *,
    subjects: list[str],
    cluster_results_root: Path,
    local_temp_root: Path,
    host: str,
    user: str,
    password: str,
) -> list[str]:
    """SFTP stage-6 CSVs and stage-7 Excel into ``local_temp_root``."""
    ready: list[str] = []
    with sftp_session(host=host, user=user, password=password) as (_client, sftp):
        for subject in subjects:
            got_any = False
            remote_s6 = _stage6_dir(cluster_results_root, subject)
            local_s6 = _stage6_dir(local_temp_root, subject)
            local_s6.mkdir(parents=True, exist_ok=True)
            for name in _STAGE6_FILES:
                remote_s = str(remote_s6 / name)
                if not remote_path_exists(sftp, remote_s):
                    continue
                local_path = local_s6 / name
                try:
                    download_remote_file(sftp, remote_s, local_path)
                    got_any = True
                    log.info("Downloaded %s -> %s", remote_s, local_path)
                except OSError as exc:
                    log.warning("[%s] download failed for %s: %s", subject, name, exc)

            remote_s7 = _stage7_dir(cluster_results_root, subject)
            local_s7 = _stage7_dir(local_temp_root, subject)
            local_s7.mkdir(parents=True, exist_ok=True)
            for name in _STAGE7_FILES:
                remote_s = str(remote_s7 / name)
                if not remote_path_exists(sftp, remote_s):
                    continue
                local_path = local_s7 / name
                try:
                    download_remote_file(sftp, remote_s, local_path)
                    got_any = True
                    log.info("Downloaded %s -> %s", remote_s, local_path)
                except OSError as exc:
                    log.warning("[%s] download failed for %s: %s", subject, name, exc)

            if got_any:
                ready.append(subject)
            else:
                log.warning("[%s] no stage6/stage7 measurement files on cluster", subject)
    return ready


@click.command("qvtpy-sync-measurements")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
    help="Local results root (required unless --from-sge).",
)
@click.option(
    "--subjects",
    default="",
    help="Comma-separated subject ids (default: all with stage6 under --output-root).",
)
@click.option(
    "--from-source",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Target DataRepo: local settings repo or the SGE cluster dataset root.",
)
@click.option(
    "--from-sge",
    is_flag=True,
    default=False,
    help=(
        "SFTP stage6 CSVs from the cluster results tree into a temporary local "
        "directory, publish, then delete the temp files."
    ),
)
@click.option("--remote-host", default=None, help="SSH hostname or CLUSTER_HOST_ALIASES key.")
@click.option("--remote-user", default=None, help="SSH username.")
@click.option(
    "--cluster-results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cluster results root (default: layout_cluster().results_root).",
)
@click.option("--build-sqlite-index/--no-build-sqlite-index", default=True, show_default=True)
def main(
    output_root: Path | None,
    subjects: str,
    from_source: str,
    from_sge: bool,
    remote_host: str | None,
    remote_user: str | None,
    cluster_results_root: Path | None,
    build_sqlite_index: bool,
) -> None:
    """CLI entry point (``qvtpy-sync-measurements``): publish stage-6 measurement CSVs (optionally
    fetched from the cluster over SFTP first) into ``image_measurements``."""
    Logger()
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    temp_root: Path | None = None
    publish_root: Path

    try:
        if from_sge:
            cluster = layout_cluster(
                results_root=cluster_results_root if cluster_results_root else None
            )
            remote_results = Path(
                cluster_results_root
                if cluster_results_root is not None
                else cluster.results_root
            )
            if not subject_list:
                raise click.UsageError(
                    "--from-sge requires --subjects (cluster subject discovery is not listed locally)."
                )
            host, user, password = _resolve_ssh_credentials(
                remote_host=remote_host,
                remote_user=remote_user,
            )
            temp_root = Path(tempfile.mkdtemp(prefix="nvitk_qvtpy_sync_sge_"))
            log.info("SFTP staging directory: %s", temp_root)
            subject_list = _download_stage6_from_sge(
                subjects=subject_list,
                cluster_results_root=remote_results,
                local_temp_root=temp_root,
                host=host,
                user=user,
                password=password,
            )
            publish_root = temp_root
        else:
            if output_root is None:
                raise click.UsageError("--output-root is required unless --from-sge is set.")
            publish_root = output_root
            if not subject_list:
                subject_list = _subjects_with_stage6(publish_root)

        if not subject_list:
            log.warning("No subjects with stage6 measurements to publish.")
            return

        repo = resolve_repo(prefer_sge=(from_source.lower() == "sge"))
        total = 0
        for subject in subject_list:
            rows = publish_stage6(
                subject_uid=subject,
                stage6_dir=_stage6_dir(publish_root, subject),
                repo=repo,
                build_sqlite_index=False,
            )
            total += int(len(rows))
            log.info("[%s] published %d image_measurements row(s)", subject, len(rows))
        if build_sqlite_index:
            repo.build_sqlite_index()
        log.info(
            "qvtpy sync: %d row(s) across %d subject(s)%s",
            total,
            len(subject_list),
            " (from-sge)" if from_sge else "",
        )
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            log.info("Removed temporary SFTP staging directory %s", temp_root)


__all__ = ["main"]


if __name__ == "__main__":
    main()
