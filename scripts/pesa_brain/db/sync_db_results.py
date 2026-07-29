#!/usr/bin/env python3
"""Publish qvtpy stage-6 hemodynamics and stage-7 morphometrics into ``image_measurements``.

Stage 6 (``loc_measurements.csv`` / ``vessel_hemodynamics.csv``) → pipeline ``4dflow_v3``.
Stage 7 (``case_metrics_donut_tree.xlsx``) → pipeline ``tof_morpho_v1``.

Examples::

    python scripts/pesa_brain/db/sync_db_results.py \\
        --results-path /data/RESULTS/QVTPy

    python scripts/pesa_brain/db/sync_db_results.py \\
        --results-path /data/RESULTS/QVTPy --skip-existing

    python scripts/pesa_brain/db/sync_db_results.py \\
        --from-sge --subjects PESA5745609,PESA123
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.xnat import parse_subject_tokens
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token
from nvitk.pipes.qvtpy.common.db_publish import (
    QVTPY_PIPELINE_ID,
    _UPSERT_KEY,
    build_image_measurement_rows_from_stage6,
    resolve_repo,
)
from nvitk.pipes.qvtpy.common.morpho_db_publish import (
    TOF_MORPHO_PIPELINE_ID,
    build_image_measurement_rows_from_stage7,
)
from nvitk.pipes.qvtpy.stage0_download import load_subjects
from nvitk.pipes.qvtpy.sync_measurements import (
    _download_stage6_and_stage7_from_sge,
    _resolve_ssh_credentials,
    _stage6_dir,
    _stage7_dir,
    _subjects_with_stage6_or_stage7,
)
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER
from nvitk.pipes.qvtpy.util.io.paths import layout_cluster, layout_local

log = Logger()


def _resolve_subject_list(
    *,
    subjects: str | None,
    subjects_file: Path | None,
    results_path: Path | None,
    from_sge: bool,
) -> list[str]:
    """Resolve subject ids from CLI, cohort alias, or on-disk stage-6/7 discovery."""
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
        found = _subjects_with_stage6_or_stage7(results_path)
        if not found:
            log.warning("No subjects with stage6/stage7 under %s", results_path)
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
        on_disk = set(_subjects_with_stage6_or_stage7(results_path))
        filtered = [s for s in cohort_list if s in on_disk]
        missing = len(cohort_list) - len(filtered)
        if missing:
            log.info(
                "Cohort %s: %d subject(s) with stage6/7 under %s "
                "(%d cohort members lack outputs on disk)",
                cohort_id,
                len(filtered),
                results_path,
                missing,
            )
        return filtered

    return load_subjects(subjects=subjects, subjects_file=None)


def _existing_subjects(repo, pipeline_id: str) -> set[str]:
    try:
        df = repo.get(
            "image_measurements",
            columns=["subject_uid"],
            filters={"pipeline_id": pipeline_id},
            cohort_id=False,
        )
        if df is None or df.empty:
            return set()
        return set(df["subject_uid"].astype(str).unique())
    except Exception as exc:
        log.warning(
            "Could not query existing %s subjects (%s); treating as empty.",
            pipeline_id,
            exc,
        )
        return set()


@click.command("sync-db-results")
@click.option(
    "--results-path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Local qvtpy results root containing <subject>/qvtpy/stage6_measure/ "
        "and/or stage7_morphometrics/. Defaults to layout_local().results_root "
        "when not using --from-sge."
    ),
)
@click.option(
    "--subjects",
    default=None,
    help=(
        "Comma/space-separated subject ids, or a single cohort alias "
        "(e.g. PESA-Brain). Omit to publish every subject with stage6/7 under "
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
        "SFTP stage6 CSVs and stage7 Excel from the cluster results tree into a "
        "temporary local directory, publish, then delete the temp files."
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
    "--skip-existing/--no-skip-existing",
    default=False,
    show_default=True,
    help=(
        "Skip stage6 publish for subjects already in 4dflow_v3, and stage7 publish "
        "for subjects already in tof_morpho_v1 (independently)."
    ),
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
    skip_existing: bool,
    build_sqlite_index: bool,
) -> None:
    """Sync qvtpy stage-6/7 results into image_measurements and rebuild SQLite."""
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
            subject_list = _download_stage6_and_stage7_from_sge(
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
            log.warning("No subjects with stage6/stage7 measurements to publish.")
            return

        repo = resolve_repo(prefer_sge=(from_source.lower() == "sge"))

        skip_s6: set[str] = set()
        skip_s7: set[str] = set()
        if skip_existing:
            skip_s6 = _existing_subjects(repo, QVTPY_PIPELINE_ID)
            skip_s7 = _existing_subjects(repo, TOF_MORPHO_PIPELINE_ID)
            if skip_s6:
                log.info(
                    "skip-existing: %d subject(s) already have pipeline=%s",
                    len(skip_s6),
                    QVTPY_PIPELINE_ID,
                )
            if skip_s7:
                log.info(
                    "skip-existing: %d subject(s) already have pipeline=%s",
                    len(skip_s7),
                    TOF_MORPHO_PIPELINE_ID,
                )

        frames_s6: list[pd.DataFrame] = []
        frames_s7: list[pd.DataFrame] = []
        n_ok_s6 = 0
        n_ok_s7 = 0
        n_skip_s6 = 0
        n_skip_s7 = 0

        log.ensure_progress()
        progress_task = log.progress(
            f"Collecting stage6/7 rows for {len(subject_list)} subjects",
            total=len(subject_list),
        )

        for subject in subject_list:
            s6 = _stage6_dir(publish_root, subject)
            s7 = _stage7_dir(publish_root, subject)
            has_s6 = (s6 / "loc_measurements.csv").is_file() or (
                s6 / "vessel_hemodynamics.csv"
            ).is_file()
            has_s7 = (s7 / STAGE7_SKIP_MARKER).is_file()

            if has_s6:
                if subject in skip_s6:
                    n_skip_s6 += 1
                    log.info("[%s] skip stage6 (already in %s)", subject, QVTPY_PIPELINE_ID)
                else:
                    rows6 = build_image_measurement_rows_from_stage6(
                        subject_uid=subject, stage6_dir=s6,
                    )
                    if not rows6.empty:
                        frames_s6.append(rows6)
                        n_ok_s6 += 1
                    log.info("[%s] stage6: %d row(s) from %s", subject, len(rows6), s6)

            if has_s7:
                if subject in skip_s7:
                    n_skip_s7 += 1
                    log.info("[%s] skip stage7 (already in %s)", subject, TOF_MORPHO_PIPELINE_ID)
                else:
                    rows7 = build_image_measurement_rows_from_stage7(
                        subject_uid=subject, stage7_dir=s7,
                    )
                    if not rows7.empty:
                        frames_s7.append(rows7)
                        n_ok_s7 += 1
                    log.info("[%s] stage7: %d row(s) from %s", subject, len(rows7), s7)

            log.update_progress(progress_task, advance=1)

        log.stop_progress()

        if not frames_s6 and not frames_s7:
            log.warning(
                "No measurement rows collected "
                "(skipped stage6=%d, stage7=%d) — nothing to publish.",
                n_skip_s6,
                n_skip_s7,
            )
            return

        n_s6_rows = 0
        n_s7_rows = 0
        all_batches: list[pd.DataFrame] = []
        if frames_s6:
            batch6 = pd.concat(frames_s6, ignore_index=True)
            n_s6_rows = len(batch6)
            all_batches.append(batch6)
            log.info(
                "Collected %d stage6 row(s) for %d subject(s) (pipeline=%s)",
                n_s6_rows,
                n_ok_s6,
                QVTPY_PIPELINE_ID,
            )

        if frames_s7:
            batch7 = pd.concat(frames_s7, ignore_index=True)
            n_s7_rows = len(batch7)
            all_batches.append(batch7)
            log.info(
                "Collected %d stage7 row(s) for %d subject(s) (pipeline=%s)",
                n_s7_rows,
                n_ok_s7,
                TOF_MORPHO_PIPELINE_ID,
            )

        combined = pd.concat(all_batches, ignore_index=True)
        log.info(
            "Writing %d total row(s) (stage6=%d, stage7=%d) in a single upsert …",
            len(combined),
            n_s6_rows,
            n_s7_rows,
        )
        repo.upsert_table(
            "image_measurements",
            combined,
            key_columns=_UPSERT_KEY,
            provenance={
                "importer": "qvtpy_sync_db_results",
                "pipelines": [QVTPY_PIPELINE_ID, TOF_MORPHO_PIPELINE_ID],
            },
            build_sqlite_index=False,
        )

        if build_sqlite_index:
            idx = repo.build_sqlite_index()
            log.info("Rebuilt SQLite index: %s", idx)

        log.info(
            "sync-db-results: stage6=%d rows/%d subjects (%s), "
            "stage7=%d rows/%d subjects (%s)%s",
            n_s6_rows,
            n_ok_s6,
            QVTPY_PIPELINE_ID,
            n_s7_rows,
            n_ok_s7,
            TOF_MORPHO_PIPELINE_ID,
            ", from-sge" if from_sge else "",
        )
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            log.info("Removed temporary SFTP staging directory %s", temp_root)


if __name__ == "__main__":
    main()
