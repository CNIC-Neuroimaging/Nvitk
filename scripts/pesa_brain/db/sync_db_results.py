#!/usr/bin/env python3
"""Publish qvtpy stage-6 measurement CSVs into ``image_measurements``.

Reads ``loc_measurements.csv`` / ``vessel_hemodynamics.csv`` under
``--results-path/<subject>/qvtpy/stage6_measure/`` and upserts long-form rows
into the NVITK dataset (default DB from ``.nvitk/settings.json``). Rebuilds the
SQLite index at the end.

Examples::

    # All subjects with stage-6 outputs under the local results root
    python scripts/pesa_brain/db/sync_db_results.py \\
        --results-path /data/RESULTS/QVTPy

    # Explicit subjects
    python scripts/pesa_brain/db/sync_db_results.py \\
        --results-path /data/RESULTS/QVTPy \\
        --subjects PESA5745609,PESA123

    # Cohort alias (subjects from cohort_membership ∩ stage-6 on disk)
    python scripts/pesa_brain/db/sync_db_results.py \\
        --results-path /data/RESULTS/QVTPy \\
        --subjects PESA-Brain

    # SFTP stage-6 CSVs from the cluster, then publish
    python scripts/pesa_brain/db/sync_db_results.py \\
        --from-sge --subjects PESA5745609,PESA123
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.common.db_publish import (
    QVTPY_PIPELINE_ID,
    publish_stage6,
    resolve_repo,
)
from nvitk.pipes.qvtpy.stage0_download import load_subjects
from nvitk.pipes.qvtpy.sync_measurements import (
    _download_stage6_from_sge,
    _resolve_ssh_credentials,
    _stage6_dir,
    _subjects_with_stage6,
)
from nvitk.pipes.qvtpy.util.io.paths import layout_cluster, layout_local
from nvitk.db.xnat import parse_subject_tokens
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token

log = Logger()


def _resolve_subject_list(
    *,
    subjects: str | None,
    subjects_file: Path | None,
    results_path: Path | None,
    from_sge: bool,
) -> list[str]:
    """Resolve subject ids from CLI, cohort alias, or on-disk stage-6 discovery."""
    if subjects_file is not None:
        return load_subjects(subjects=None, subjects_file=subjects_file)

    if subjects is None or not str(subjects).strip():
        if from_sge:
            raise click.UsageError(
                "--from-sge requires --subjects (or --subjects-file); "
                "remote subject discovery is not listed locally."
            )
        if results_path is None:
            raise click.UsageError("--results-path is required when --subjects is omitted.")
        found = _subjects_with_stage6(results_path)
        if not found:
            log.warning("No subjects with stage6 measurements under %s", results_path)
        return found

    tokens = parse_subject_tokens(subjects)
    if len(tokens) == 1 and resolve_xnat_project_cohort_token(tokens[0]) is not None:
        cohort_id = tokens[0].strip()
        repo = resolve_repo(prefer_sge=False)
        uids = repo._cohort_subject_uid_set(cohort_id)
        if not uids:
            raise click.UsageError(
                f"Cohort {cohort_id!r} has no subjects in cohort_membership "
                "(or the table is missing)."
            )
        cohort_list = sorted(uids)
        if from_sge:
            return cohort_list
        if results_path is None:
            return cohort_list
        on_disk = set(_subjects_with_stage6(results_path))
        filtered = [s for s in cohort_list if s in on_disk]
        missing = len(cohort_list) - len(filtered)
        if missing:
            log.info(
                "Cohort %s: %d subject(s) with stage6 under %s "
                "(%d cohort members lack stage6 on disk)",
                cohort_id,
                len(filtered),
                results_path,
                missing,
            )
        return filtered

    return load_subjects(subjects=subjects, subjects_file=None)


@click.command("sync-db-results")
@click.option(
    "--results-path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Local qvtpy results root containing <subject>/qvtpy/stage6_measure/. "
        "Defaults to layout_local().results_root when not using --from-sge."
    ),
)
@click.option(
    "--subjects",
    default=None,
    help=(
        "Comma/space-separated subject ids, or a single cohort alias "
        "(e.g. PESA-Brain). Omit to publish every subject with stage6 under "
        "--results-path."
    ),
)
@click.option(
    "--subjects-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional .txt/.csv/.xlsx subject list (mutually exclusive with --subjects).",
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
@click.option(
    "--from-source",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Target DataRepo: local settings repo or the SGE cluster dataset root.",
)
@click.option("--remote-host", default=None, help="SSH hostname or CLUSTER_HOST_ALIASES key.")
@click.option("--remote-user", default=None, help="SSH username.")
@click.option(
    "--cluster-results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override cluster results root (default: layout_cluster().results_root).",
)
@click.option(
    "--build-sqlite-index/--no-build-sqlite-index",
    default=True,
    show_default=True,
    help="Rebuild the SQLite catalog index after publishing.",
)
def main(
    results_path: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    from_sge: bool,
    from_source: str,
    remote_host: str | None,
    remote_user: str | None,
    cluster_results_root: Path | None,
    build_sqlite_index: bool,
) -> None:
    """Sync qvtpy stage-6 results into image_measurements and rebuild SQLite."""
    if subjects is not None and subjects_file is not None:
        raise click.UsageError("Provide at most one of --subjects or --subjects-file.")

    temp_root: Path | None = None
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
            subject_list = _resolve_subject_list(
                subjects=subjects,
                subjects_file=subjects_file,
                results_path=None,
                from_sge=True,
            )
            host, user, password = _resolve_ssh_credentials(
                remote_host=remote_host,
                remote_user=remote_user,
            )
            temp_root = Path(tempfile.mkdtemp(prefix="nvitk_pesa_sync_sge_"))
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
            if results_path is None:
                results_path = Path(layout_local().results_root)
            publish_root = Path(results_path)
            if not publish_root.is_dir():
                raise click.UsageError(f"--results-path not found: {publish_root}")
            subject_list = _resolve_subject_list(
                subjects=subjects,
                subjects_file=subjects_file,
                results_path=publish_root,
                from_sge=False,
            )

        if not subject_list:
            log.warning("No subjects with stage6 measurements to publish.")
            return

        repo = resolve_repo(prefer_sge=(from_source.lower() == "sge"))
        total = 0
        n_ok = 0

        log.ensure_progress()
        progress_task = log.progress(f"Publishing stage6 measurements for {len(subject_list)} subjects", total=len(subject_list))

        for subject in subject_list:
            s6 = _stage6_dir(publish_root, subject)
            rows = publish_stage6(
                subject_uid=subject,
                stage6_dir=s6,
                repo=repo,
                build_sqlite_index=False,
            )
            n_rows = int(len(rows))
            total += n_rows
            if n_rows:
                n_ok += 1
            log.info("[%s] published %d image_measurements row(s) from %s", subject, n_rows, s6)
            log.update_progress(progress_task, advance=1)

        log.stop_progress()

        if build_sqlite_index:
            idx = repo.build_sqlite_index()
            log.info("Rebuilt SQLite index: %s", idx)

        log.info(
            "sync-db-results: %d row(s) across %d/%d subject(s) "
            "(pipeline=%s%s)",
            total,
            n_ok,
            len(subject_list),
            QVTPY_PIPELINE_ID,
            ", from-sge" if from_sge else "",
        )
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            log.info("Removed temporary SFTP staging directory %s", temp_root)


if __name__ == "__main__":
    main()
