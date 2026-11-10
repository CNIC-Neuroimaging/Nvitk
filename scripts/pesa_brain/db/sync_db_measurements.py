#!/usr/bin/env python3
"""Publish qvtpy stage-6 hemodynamics and stage-7 morphometrics into ``image_measurements``.

Stage 6 hemodynamics (``loc_measurements.csv`` / ``vessel_hemodynamics.csv``) -> pipeline
``4dflow_v4``. Stage 7 morphometrics (``morphometrics_report.csv``, rebuilt from
``vessel_summary.csv`` / ``lesions.csv`` / ``volumetry.csv`` when missing; the legacy
``case_metrics_donut_tree.xlsx`` aggregation for runs without ``vessel_summary.csv``) -> pipeline
``tof_morpho_v2``. ``--measurements`` publishes both (default) or only one of them; each published
pipeline is registered as its modality's default in the dataset catalog, as the stage publishers do.

``--hemodynamics-pipeline`` / ``--morpho-pipeline`` publish under another version the dataset
catalog already has (an id or an alias of that modality, e.g. ``4dflow_v3`` / ``v3``): for
re-uploading results of an earlier pipeline version. Those rows leave the catalog's default as it is.

A subject's stage-7 rows replace all its earlier rows of the morphometrics pipeline (the report's
variable set is the whole set, as in ``publish_stage7``); stage-6 rows are upserted, or replace the
subject's rows of the hemodynamics pipeline with ``--drop-existing``.

LOC hemodynamics include surviving PCOMMs (``LPCOMM`` / ``RPCOMM``) when present in
``loc_measurements.csv``; ACOMM is never measured. This sync script publishes all
LOC vessel rows without a vessel blacklist: PCOMM counts are logged in the summary.

Examples::

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --results-path /data/RESULTS/QVTPy

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --results-path /data/RESULTS/QVTPy --skip-existing

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --results-path /data/RESULTS/QVTPy --subjects PESA5745609 --drop-existing

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --from-sge --subjects PESA5745609,PESA123

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --results-path /data/RESULTS/QVTPy --measurements morphometrics

    python scripts/pesa_brain/db/sync_db_measurements.py \\
        --results-path /data/RESULTS/QVTPy_v3 --measurements hemodynamics \\
        --hemodynamics-pipeline 4dflow_v3
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
    count_pcomm_rows,
    qvtpy_pipeline_catalog_spec,
    register_default_pipeline,
    resolve_repo,
)
from nvitk.pipes.qvtpy.common.morpho_db_publish import (
    TOF_MORPHO_PIPELINE_ID,
    build_image_measurement_rows_from_stage7,
    tof_morpho_pipeline_catalog_spec,
)
from nvitk.pipes.qvtpy.stage0_download import load_subjects
from nvitk.pipes.qvtpy.sync_measurements import (
    _download_stage6_and_stage7_from_sge,
    _resolve_ssh_credentials,
    _stage6_dir,
    _stage7_dir,
    _subjects_with_stage6,
    _subjects_with_stage6_or_stage7,
    _subjects_with_stage7,
)
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER
from nvitk.pipes.qvtpy.util.io.paths import layout_cluster, layout_local

log = Logger()

#: ``--measurements`` choices -> (stage 6, stage 7).
_MEASUREMENTS: dict[str, tuple[bool, bool]] = {
    "all": (True, True),
    "hemodynamics": (True, False),
    "morphometrics": (False, True),
}


def _resolve_pipeline(repo, selector: str | None, spec: dict) -> tuple[str, bool]:
    """``(pipeline_id, is_current)`` of *selector* for the modality of *spec* (a publisher's catalog
    entry). ``None`` or the publisher's own id is the current version, registered as the default
    on publish even if the catalog does not have it yet; anything else must be an id or alias the
    dataset catalog knows for that modality."""
    current = str(spec["pipeline_id"])
    if selector is None or not selector.strip() or selector.strip() == current:
        return current, True
    try:
        ids = repo.catalog.resolve_pipeline_selector(selector.strip(), modality=spec["modality"])
    except Exception as exc:  # noqa: BLE001 (ManifestError and friends: a bad selector)
        raise click.BadParameter(f"{selector!r}: {exc}") from exc
    if not ids or len(ids) != 1:
        known = [e.get("pipeline_id") for e in repo.catalog.list_pipelines(spec["modality"])]
        raise click.BadParameter(
            f"{selector!r} is not one {spec['modality']} pipeline of the dataset catalog "
            f"(known: {', '.join(map(str, known)) or 'none'})."
        )
    return str(ids[0]), str(ids[0]) == current


def _as_pipeline(rows: pd.DataFrame, repo, pipeline_id: str, current: str) -> pd.DataFrame:
    """*rows* (built under the publisher's *current* id) relabelled as *pipeline_id*, with that
    catalog entry's ``pipeline_name``."""
    if rows.empty or pipeline_id == current:
        return rows
    rows = rows.copy()
    rows["pipeline_id"] = pipeline_id
    entry = next((e for e in repo.catalog.list_pipelines() if str(e.get("pipeline_id")) == pipeline_id), {})
    if "pipeline_name" in rows.columns and entry.get("pipeline_name"):
        rows["pipeline_name"] = entry["pipeline_name"]
    return rows


def _subjects_on_disk(results_path: Path, *, stage6: bool, stage7: bool) -> list[str]:
    if stage6 and stage7:
        return _subjects_with_stage6_or_stage7(results_path)
    return _subjects_with_stage6(results_path) if stage6 else _subjects_with_stage7(results_path)


def _resolve_subject_list(
    *,
    subjects: str | None,
    subjects_file: Path | None,
    results_path: Path | None,
    from_sge: bool,
    stage6: bool = True,
    stage7: bool = True,
) -> list[str]:
    """Resolve subject ids from CLI, cohort alias, or on-disk discovery of the selected stages."""
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
        found = _subjects_on_disk(results_path, stage6=stage6, stage7=stage7)
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
        on_disk = set(_subjects_on_disk(results_path, stage6=stage6, stage7=stage7))
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


def _drop_pipeline_measurements_for_subjects(
    repo,
    scopes: dict[str, set[str]],
) -> dict[str, int]:
    """Remove all ``image_measurements`` rows of each ``pipeline_id -> subject_uids`` in *scopes*,
    in a single rewrite.

    Returns the dropped rows per pipeline. Writes Parquet without rebuilding SQLite
    (caller may rebuild after the subsequent upsert).
    """
    scopes = {pid: uids for pid, uids in scopes.items() if uids}
    dropped = {pid: 0 for pid in scopes}
    if not scopes:
        return dropped
    try:
        existing = repo.get("image_measurements", cohort_id=False, use_sqlite=False)
    except Exception as exc:
        log.warning(
            "Could not read image_measurements for drop-existing (%s); skipping drop.",
            exc,
        )
        return dropped
    if existing is None or existing.empty:
        return dropped
    if "pipeline_id" not in existing.columns or "subject_uid" not in existing.columns:
        log.warning(
            "image_measurements missing pipeline_id/subject_uid; skipping drop-existing."
        )
        return dropped

    pipelines = existing["pipeline_id"].astype(str)
    subject_col = existing["subject_uid"].astype(str)
    drop_mask = pd.Series(False, index=existing.index)
    for pid, uids in scopes.items():
        hit = pipelines.eq(str(pid)) & subject_col.isin(uids)
        dropped[pid] = int(hit.sum())
        drop_mask |= hit
    n_drop = int(drop_mask.sum())
    if n_drop == 0:
        return dropped

    kept = existing.loc[~drop_mask].reset_index(drop=True)
    repo.write_table(
        "image_measurements",
        kept,
        provenance={
            "importer": "qvtpy_sync_db_measurements",
            "action": "drop-existing",
            "pipelines": sorted(scopes),
            "n_subjects": {pid: len(uids) for pid, uids in scopes.items()},
            "n_dropped": dropped,
        },
        build_sqlite_index=False,
    )
    return dropped


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
    "--measurements",
    type=click.Choice(sorted(_MEASUREMENTS), case_sensitive=False),
    default="all",
    show_default=True,
    help=(
        "Which results to publish: all (stage-6 hemodynamics and stage-7 morphometrics), "
        "hemodynamics (stage 6 only) or morphometrics (stage 7 only)."
    ),
)
@click.option(
    "--hemodynamics-pipeline",
    default=QVTPY_PIPELINE_ID,
    show_default=True,
    help=(
        "Pipeline the stage-6 rows go to: the current one (registered as the default), or another "
        "4D-flow pipeline id/alias of the dataset catalog (e.g. 4dflow_v3, v3)."
    ),
)
@click.option(
    "--morpho-pipeline",
    default=TOF_MORPHO_PIPELINE_ID,
    show_default=True,
    help=(
        "Pipeline the stage-7 rows go to: the current one (registered as the default), or another "
        "TOF pipeline id/alias of the dataset catalog (e.g. tof_morpho_v1)."
    ),
)
@click.option(
    "--from-sge",
    is_flag=True,
    default=False,
    help=(
        "SFTP the stage6 CSVs and stage7 report files from the cluster results tree into a "
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
        "Skip stage6 publish for subjects already in the hemodynamics pipeline, and stage7 "
        "publish for subjects already in the morphometrics pipeline (independently)."
    ),
)
@click.option(
    "--drop-existing/--no-drop-existing",
    default=False,
    show_default=True,
    help=(
        "Before uploading stage6 rows, delete all existing image_measurements for "
        "those subjects under the hemodynamics pipeline (so orphaned variables/regions "
        "from prior runs are removed). Stage7 rows always replace the subject's rows of the "
        "morphometrics pipeline. Mutually exclusive with --skip-existing."
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
    measurements: str,
    hemodynamics_pipeline: str,
    morpho_pipeline: str,
    from_sge: bool,
    from_source: str,
    remote_host: str | None,
    remote_user: str | None,
    cluster_results_root: Path | None,
    skip_existing: bool,
    drop_existing: bool,
    build_sqlite_index: bool,
) -> None:
    """Sync qvtpy stage-6/7 results into image_measurements and rebuild SQLite."""
    do_s6, do_s7 = _MEASUREMENTS[measurements.lower()]
    stages = " + ".join(n for n, on in (("stage6", do_s6), ("stage7", do_s7)) if on)
    if subjects is not None and subjects_file is not None:
        raise click.UsageError("Provide at most one of --subjects or --subjects-file.")
    if skip_existing and drop_existing:
        raise click.UsageError("Provide at most one of --skip-existing or --drop-existing.")

    # The pipelines resolve before anything is downloaded, so a wrong one fails at once.
    repo = resolve_repo(prefer_sge=(from_source.lower() == "sge"))
    spec6, spec7 = qvtpy_pipeline_catalog_spec(), tof_morpho_pipeline_catalog_spec()
    pid6, current6 = _resolve_pipeline(repo, hemodynamics_pipeline if do_s6 else None, spec6)
    pid7, current7 = _resolve_pipeline(repo, morpho_pipeline if do_s7 else None, spec7)
    for pid, current, on in ((pid6, current6, do_s6), (pid7, current7, do_s7)):
        if on and not current:
            log.info("Publishing under %s (an earlier version: the catalog default is not changed)", pid)

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
                stage6=do_s6,
                stage7=do_s7,
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
                stage6=do_s6,
                stage7=do_s7,
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
                stage6=do_s6,
                stage7=do_s7,
            )

        if not subject_list:
            log.warning("No subjects with %s measurements to publish.", stages)
            return

        skip_s6: set[str] = set()
        skip_s7: set[str] = set()
        if skip_existing:
            skip_s6 = _existing_subjects(repo, pid6) if do_s6 else set()
            skip_s7 = _existing_subjects(repo, pid7) if do_s7 else set()
            if skip_s6:
                log.info(
                    "skip-existing: %d subject(s) already have pipeline=%s",
                    len(skip_s6),
                    pid6,
                )
            if skip_s7:
                log.info(
                    "skip-existing: %d subject(s) already have pipeline=%s",
                    len(skip_s7),
                    pid7,
                )

        frames_s6: list[pd.DataFrame] = []
        frames_s7: list[pd.DataFrame] = []
        n_ok_s6 = 0
        n_ok_s7 = 0
        n_skip_s6 = 0
        n_skip_s7 = 0
        subjects_s6: set[str] = set()
        subjects_s7: set[str] = set()

        log.ensure_progress()
        progress_task = log.progress(
            f"Collecting {stages} rows for {len(subject_list)} subjects",
            total=len(subject_list),
        )

        for subject in subject_list:
            s6 = _stage6_dir(publish_root, subject)
            s7 = _stage7_dir(publish_root, subject)
            has_s6 = do_s6 and (
                (s6 / "loc_measurements.csv").is_file()
                or (s6 / "vessel_hemodynamics.csv").is_file()
            )
            has_s7 = do_s7 and (s7 / STAGE7_SKIP_MARKER).is_file()

            if has_s6:
                if subject in skip_s6:
                    n_skip_s6 += 1
                    log.info("[%s] skip stage6 (already in %s)", subject, pid6)
                else:
                    rows6 = _as_pipeline(
                        build_image_measurement_rows_from_stage6(subject_uid=subject, stage6_dir=s6),
                        repo, pid6, QVTPY_PIPELINE_ID,
                    )
                    if not rows6.empty:
                        frames_s6.append(rows6)
                        subjects_s6.add(str(subject))
                        n_ok_s6 += 1
                    log.info("[%s] stage6: %d row(s) from %s", subject, len(rows6), s6)

            if has_s7:
                if subject in skip_s7:
                    n_skip_s7 += 1
                    log.info("[%s] skip stage7 (already in %s)", subject, pid7)
                else:
                    rows7 = _as_pipeline(
                        build_image_measurement_rows_from_stage7(subject_uid=subject, stage7_dir=s7),
                        repo, pid7, TOF_MORPHO_PIPELINE_ID,
                    )
                    if not rows7.empty:
                        frames_s7.append(rows7)
                        subjects_s7.add(str(subject))
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

        # Stage 7 always replaces the subject's rows (publish_stage7): a variable the report no
        # longer has must not survive from an earlier run.
        scopes = {pid7: subjects_s7}
        if drop_existing:
            scopes[pid6] = subjects_s6
        for pid, n_dropped in _drop_pipeline_measurements_for_subjects(repo, scopes).items():
            log.info(
                "Replacing: removed %d row(s) of pipeline=%s for %d subject(s)",
                n_dropped,
                pid,
                len(scopes[pid]),
            )

        # Register the current pipelines as their modality's default, as the stage publishers do.
        if frames_s6 and current6:
            register_default_pipeline(repo, spec6)
        if frames_s7 and current7:
            register_default_pipeline(repo, spec7)

        n_s6_rows = 0
        n_s7_rows = 0
        all_batches: list[pd.DataFrame] = []
        if frames_s6:
            batch6 = pd.concat(frames_s6, ignore_index=True)
            n_s6_rows = len(batch6)
            all_batches.append(batch6)
            pcomm = count_pcomm_rows(batch6)
            log.info(
                "Collected %d stage6 row(s) for %d subject(s) (pipeline=%s); "
                "PCOMM: LPCOMM=%d RPCOMM=%d",
                n_s6_rows,
                n_ok_s6,
                pid6,
                pcomm["LPCOMM"],
                pcomm["RPCOMM"],
            )

        if frames_s7:
            batch7 = pd.concat(frames_s7, ignore_index=True)
            n_s7_rows = len(batch7)
            all_batches.append(batch7)
            log.info(
                "Collected %d stage7 row(s) for %d subject(s) (pipeline=%s)",
                n_s7_rows,
                n_ok_s7,
                pid7,
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
                "pipelines": [
                    pid
                    for pid, frames in ((pid6, frames_s6), (pid7, frames_s7))
                    if frames
                ],
            },
            build_sqlite_index=False,
        )

        if build_sqlite_index:
            idx = repo.build_sqlite_index()
            log.info("Rebuilt SQLite index: %s", idx)

        log.info(
            "sync-db-results: stage6=%d rows/%d subjects (%s), "
            "stage7=%d rows/%d subjects (%s)%s%s",
            n_s6_rows,
            n_ok_s6,
            pid6,
            n_s7_rows,
            n_ok_s7,
            pid7,
            ", from-sge" if from_sge else "",
            ", drop-existing" if drop_existing else "",
        )
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            log.info("Removed temporary SFTP staging directory %s", temp_root)


if __name__ == "__main__":
    main()
