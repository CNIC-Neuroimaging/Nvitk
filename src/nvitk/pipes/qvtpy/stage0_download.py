"""qvtpy stage 0 (download): XNAT -> DICOM into ``DEFAULT_DICOM_ROOT``.

Lean direct downloader for the sequences needed by qvtpy stage 0 conversion
(:mod:`nvitk.pipes.qvtpy.stage0_convert`). Reuses :mod:`nvitk.db.xnat`
primitives (``connect_xnat``, ``classify_scan``, ``download_scan_dicoms``)
without requiring a :class:`~nvitk.db.repo.DataRepo`.

Default sequences: ``TOF``, ``4DFLOW_AP``, ``4DFLOW_RL``, ``4DFLOW_FH``.

Per-subject layout under ``--dicom-root`` (lowercase slot names, aligned with
:func:`nvitk.db.xnat.xnat_sequence_to_asset_slot`)::

    {subject}/tof/
    {subject}/4dflow_ap/
    {subject}/4dflow_rl/
    {subject}/4dflow_fh/

When invoked from the qvtpy master runner (:mod:`nvitk.pipes.qvtpy.run`) this
step is **opt-in** (``stage0_d``); ``stage0_convert`` runs by default.

**Outputs**

- ``{dicom_root}/{subject}/{tof|4dflow_ap|4dflow_rl|4dflow_fh}/`` DICOM file trees.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import click
import pandas as pd

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.db.xnat import (
    classify_scan,
    connect_xnat,
    download_scan_dicoms,
    parse_subject_tokens,
    requested_sequence_set,
    resolve_subject_labels,
)
from nvitk.db.xnat_config import (
    XnatConnectionConfig,
    load_xnat_profile,
    resolve_xnat_connection,
)

from . import config as cfg

log = Logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEQUENCES: tuple[str, ...] = ("TOF", "4DFLOW_AP", "4DFLOW_RL", "4DFLOW_FH")

SLOT_DIRS: dict[str, str] = {
    "TOF": "tof",
    "4DFLOW_AP": "4dflow_ap",
    "4DFLOW_RL": "4dflow_rl",
    "4DFLOW_FH": "4dflow_fh",
}


# ---------------------------------------------------------------------------
# Subject list loading
# ---------------------------------------------------------------------------

_SUBJECT_COLUMN_CANDIDATES: tuple[str, ...] = (
    "subject",
    "subject_id",
    "subject_uid",
    "pesa",
    "pesa_id",
    "id",
)


# ---------------------------------------------------------------------------
# Subject list parsing
# ---------------------------------------------------------------------------


def _normalize_header(value: str) -> str:
    """Lowercase *value* with non-alphanumeric characters stripped, for header matching."""
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _detect_subject_column(columns: Iterable[str]) -> str | None:
    """Name of the column in *columns* matching a known subject-id header, or None."""
    norm = {_normalize_header(c): c for c in columns}
    for candidate in _SUBJECT_COLUMN_CANDIDATES:
        key = _normalize_header(candidate)
        if key in norm:
            return norm[key]
    return None


def _read_subjects_dataframe(path: Path) -> list[str]:
    """Sorted unique subject IDs read from a CSV/XLSX *path*, using the auto-detected
    subject column (or the first non-empty column as a fallback)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, dtype=str, keep_default_na=False)
        except ImportError as exc:
            raise click.ClickException(
                f"Reading {path.suffix} requires openpyxl. Install with: pip install openpyxl"
            ) from exc
    else:
        raise click.ClickException(f"Unsupported subjects-file suffix: {path.suffix!r}")

    if df.empty:
        return []

    columns = [str(c) for c in df.columns]
    target = _detect_subject_column(columns)
    if target is None:
        for c in columns:
            values = [v for v in df[c].astype(str).tolist() if v and v.strip()]
            if values:
                target = c
                break

    if target is None:
        return []

    values = [str(v).strip().strip('"') for v in df[target].tolist() if str(v).strip()]
    return sorted(set(values))


def load_subjects(
    *,
    subjects: str | None,
    subjects_file: str | Path | None,
) -> list[str]:
    """Resolve a subject list from CLI inputs.

    Exactly one of *subjects* (comma/whitespace separated string) or
    *subjects_file* (``.txt`` / ``.csv`` / ``.xlsx``) must be provided.
    For tabular files the subject column is auto-detected from the candidates
    in :data:`_SUBJECT_COLUMN_CANDIDATES`, falling back to the first non-empty
    column.
    """
    if (subjects is None and subjects_file is None) or (
        subjects is not None and subjects_file is not None
    ):
        raise click.ClickException(
            "Provide exactly one of --subjects or --subjects-file."
        )

    if subjects is not None:
        ids = parse_subject_tokens(subjects)
        return sorted(set(ids))

    path = Path(subjects_file).expanduser().resolve()
    if not path.exists():
        raise click.ClickException(f"subjects-file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".xlsx", ".xls"):
        return _read_subjects_dataframe(path)
    return resolve_subject_labels(subjects_file=path)


def resolve_subjects_for_xnat_pipeline(
    *,
    subjects: str | None,
    subjects_file: str | Path | None,
    xnat_config: XnatConnectionConfig,
    database_root: str | Path | None = None,
) -> tuple[list[str], XnatConnectionConfig]:
    """Resolve subjects for qvtpy when sourcing DICOMs from XNAT.

    When *subjects* is a single cohort alias (e.g. ``PESA-Brain``), expand to all
    subject labels in that XNAT project and override ``xnat_config.project``.

    When *database_root* is set, pre-filter to subjects with all qvtpy sequences
    indexed in the catalog ``scans`` table (TOF + 3× 4DFlow).
    """
    from nvitk.db.xnat import list_xnat_project_subject_labels
    from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token
    from nvitk.pipes.qvtpy.util.io.db_subject_filter import (
        filter_subjects_by_qvtpy_scan_availability,
    )

    if subjects_file is not None:
        labels = load_subjects(subjects=subjects, subjects_file=subjects_file)
        conn = xnat_config
    elif subjects is None:
        raise click.ClickException(
            "Provide exactly one of --subjects or --subjects-file."
        )
    else:
        project_id = resolve_xnat_project_cohort_token(subjects)
        if project_id is None:
            labels = load_subjects(subjects=subjects, subjects_file=None)
            conn = xnat_config
        else:
            conn = replace(xnat_config, project=project_id)
            labels = list_xnat_project_subject_labels(conn)
            if not labels:
                raise click.ClickException(
                    f"No subjects found in XNAT project {project_id!r} "
                    f"(from cohort alias {subjects!r})."
                )
            log.info(
                f"XNAT cohort alias {subjects!r} -> project {project_id!r} "
                f"({len(labels)} subject(s))"
            )

    if database_root is not None:
        try:
            labels = filter_subjects_by_qvtpy_scan_availability(
                labels,
                database_root=database_root,
                project_id=conn.project,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if not labels:
            raise click.ClickException(
                "No subjects remain after database scan pre-filter "
                f"(project={conn.project!r})."
            )

    return labels, conn


# ---------------------------------------------------------------------------
# XNAT helpers (lean: no DataRepo)
# ---------------------------------------------------------------------------


def _coalesce_attr(obj: Any, *names: str) -> Any:
    """First non-None value among *names* on *obj* (calling it if it's a method)."""
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None:
                return value
    return None


def _resolve_subject(project: Any, label: str) -> Any | None:
    """XNAT subject object in *project* matching *label* (by dict key or label/id/name), or None."""
    subjects_map = getattr(project, "subjects", None) or {}
    if label in subjects_map:
        return subjects_map[label]
    for _key, subj in subjects_map.items():
        uid = str(_coalesce_attr(subj, "label", "id", "name") or _key)
        if uid == label:
            return subj
    return None


def _dir_has_files(directory: Path) -> bool:
    """True if *directory* exists and contains at least one regular file, recursively."""
    if not directory.is_dir():
        return False
    for child in directory.iterdir():
        if child.is_file():
            return True
        if child.is_dir() and _dir_has_files(child):
            return True
    return False


def sequence_slot_dir(sequence: str) -> str:
    """Lowercase slot folder name for a sequence key (e.g. ``TOF`` -> ``tof``)."""
    slot = SLOT_DIRS.get(sequence)
    if slot is None:
        slot = re.sub(r"[^0-9a-z]+", "_", sequence.lower()).strip("_") or "unknown"
    return slot


def local_subject_dicoms_complete(
    dicom_root: str | Path,
    subject: str,
    sequences: Iterable[str],
) -> bool:
    """Return True when every requested sequence has files under the local subject tree."""
    subj_dir = Path(dicom_root).expanduser() / subject
    for sequence in sequences:
        if not _dir_has_files(subj_dir / sequence_slot_dir(sequence)):
            return False
    return True


def collect_local_subject_dicoms(
    dicom_root: str | Path,
    subject: str,
    sequences: Iterable[str],
) -> dict[str, list[Path]]:
    """Build a :func:`download_subject`-shaped result from an on-disk local tree."""
    subj_dir = Path(dicom_root).expanduser() / subject
    out: dict[str, list[Path]] = {}
    for sequence in sequences:
        slot_dir = subj_dir / sequence_slot_dir(sequence)
        if _dir_has_files(slot_dir):
            out[sequence] = sorted(p for p in slot_dir.rglob("*") if p.is_file())
        else:
            out[sequence] = []
    return out


# ---------------------------------------------------------------------------
# Per-subject and batch download
# ---------------------------------------------------------------------------


def download_subject(
    xnat_session: Any,
    project_id: str,
    subject_label: str,
    *,
    dicom_root: Path,
    sequences: set[str],
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """Download requested-sequence DICOMs for one subject.

    Returns ``{sequence_key: [files]}`` for every sequence in *sequences*; a
    sequence that was unavailable, skipped, or failed will map to an empty list.
    """
    project = xnat_session.projects[project_id]
    subject = _resolve_subject(project, subject_label)
    if subject is None:
        raise LookupError(
            f"Subject {subject_label!r} not found in XNAT project {project_id!r}"
        )

    out: dict[str, list[Path]] = {seq: [] for seq in sequences}

    experiments = list(getattr(subject, "experiments", {}).values())
    for experiment in experiments:
        scans = list(getattr(experiment, "scans", {}).values())
        for scan in scans:
            series_description = str(
                _coalesce_attr(scan, "series_description", "type", "label") or ""
            )
            quality = str(_coalesce_attr(scan, "quality") or "")
            classification = classify_scan(series_description, quality)
            if classification is None:
                continue

            sequence = classification["sequence"]
            if sequence not in sequences:
                log.debug(
                    f"[{subject_label}] skipping scan {series_description!r} "
                    f"(classified as {sequence!r}, not requested)"
                )
                continue

            slot = sequence_slot_dir(sequence)

            scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
            target_dir = dicom_root / subject_label / slot

            if skip_existing and _dir_has_files(target_dir):
                log.info(
                    f"[{subject_label}] {sequence} (scan_id={scan_id}): "
                    f"skip-existing -> {target_dir}"
                )
                existing = sorted(p for p in target_dir.rglob("*") if p.is_file())
                if existing:
                    out[sequence] = existing
                continue

            log.info(
                f"[{subject_label}] downloading {sequence} (scan_id={scan_id}) "
                f"-> {target_dir}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                files = download_scan_dicoms(scan, target_dir)
            except Exception as exc:
                log.warning(
                    f"[{subject_label}] failed downloading {sequence} "
                    f"(scan_id={scan_id}): {exc}"
                )
                continue
            out[sequence] = files

    return out


def run_download(
    subjects: list[str],
    *,
    dicom_root: str | Path,
    xnat_config: XnatConnectionConfig,
    sequences: Iterable[str] | None = None,
    skip_existing: bool = True,
    skip_existing_downloads: bool = False,
    report: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    """Download requested DICOM sequences for a list of subjects.

    A single XNAT session is opened for the whole batch. Errors on a single
    subject are logged and do not abort the batch.
    """
    seq_set: set[str] = set(sequences) if sequences else set(DEFAULT_SEQUENCES)
    root = Path(dicom_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    log.info(f"qvtpy stage0_download | subjects={len(subjects)}")
    log.info(f"  dicom_root : {root}")
    log.info(f"  sequences  : {sorted(seq_set)}")
    log.info(f"  xnat       : {xnat_config.server} / {xnat_config.project}")

    log.info(f"  skip_existing_downloads: {skip_existing_downloads}")

    results: dict[str, dict[str, list[Path]]] = {}
    with connect_xnat(xnat_config) as session:
        for subject_label in subjects:
            try:
                if skip_existing_downloads and local_subject_dicoms_complete(
                    root, subject_label, seq_set
                ):
                    log.info(
                        f"[{subject_label}] all requested sequences present locally "
                        "— skip XNAT download"
                    )
                    results[subject_label] = collect_local_subject_dicoms(
                        root, subject_label, seq_set
                    )
                    continue

                per_seq_skip = skip_existing_downloads or skip_existing
                results[subject_label] = download_subject(
                    session,
                    xnat_config.project,
                    subject_label,
                    dicom_root=root,
                    sequences=seq_set,
                    skip_existing=per_seq_skip,
                )
            except LookupError as exc:
                log.warning(f"[{subject_label}] {exc}")
                results[subject_label] = {seq: [] for seq in seq_set}
            except Exception as exc:
                log.warning(f"[{subject_label}] unexpected error: {exc}")
                results[subject_label] = {seq: [] for seq in seq_set}

    if report:
        required_slots: list[str] = []
        for seq in sorted(seq_set):
            slot = SLOT_DIRS.get(seq) or re.sub(r"[^0-9a-z]+", "_", seq.lower()).strip("_")
            if slot and slot not in required_slots:
                required_slots.append(slot)
        print_qc_report(root, subjects, required_slots)

    return results


def print_qc_report_from_results(
    results: dict[str, dict[str, list[Path]]],
    subjects: Iterable[str],
    sequences: Iterable[str],
) -> dict[str, Any]:
    """Print QC summary from in-memory download results (no local tree required)."""
    seq_list = sorted(set(sequences))
    slot_by_seq: dict[str, str] = {}
    for seq in seq_list:
        slot = SLOT_DIRS.get(seq) or re.sub(r"[^0-9a-z]+", "_", seq.lower()).strip("_")
        if slot:
            slot_by_seq[seq] = slot

    subj_list = sorted(set(subjects))
    complete: list[str] = []
    incomplete: dict[str, list[str]] = {}

    for subj in subj_list:
        by_seq = results.get(subj, {})
        missing: list[str] = []
        for seq in seq_list:
            slot = slot_by_seq.get(seq, seq.lower())
            if not by_seq.get(seq):
                missing.append(f"{slot}[missing or empty]")
        if not missing:
            complete.append(subj)
        else:
            incomplete[subj] = missing

    bar = "=" * 60
    print(bar)
    print("DICOM completeness report (from download results)")
    slots = [slot_by_seq[s] for s in seq_list if s in slot_by_seq]
    print(f"  sequences req : {' '.join(slots)}")
    print(bar)
    print(f"Subjects scanned : {len(subj_list)}")
    print(f"Complete         : {len(complete)}")
    print(f"Incomplete       : {len(incomplete)}")
    print()

    if incomplete:
        print("-- Incomplete subjects (missing sequences) --")
        width = max((len(s) for s in incomplete.keys()), default=0)
        for subj in sorted(incomplete.keys()):
            print(f"  {subj:<{width}}  ->  {' '.join(incomplete[subj])}")
        print()

    if complete:
        print("-- Complete subjects --")
        for subj in complete:
            print(f"  {subj}")
        print()

    return {
        "complete": complete,
        "incomplete": incomplete,
        "n_subjects": len(subj_list),
    }


# ---------------------------------------------------------------------------
# QC report
# ---------------------------------------------------------------------------


def print_qc_report(
    dicom_root: str | Path,
    subjects: Iterable[str],
    required_slots: Iterable[str],
) -> dict[str, Any]:
    """Print a brief QC report of the downloaded DICOM trees.

    For each subject under *dicom_root* checks that each ``required_slots``
    subdirectory exists and contains at least one regular file. Returns a
    summary dict (also useful for tests).
    """
    root = Path(dicom_root)
    slots = list(dict.fromkeys(required_slots))
    subj_list = sorted(set(subjects))

    complete: list[str] = []
    incomplete: dict[str, list[str]] = {}

    for subj in subj_list:
        subj_dir = root / subj
        missing: list[str] = []
        for slot in slots:
            d = subj_dir / slot
            if not d.is_dir():
                missing.append(f"{slot}[missing dir]")
                continue
            if not _dir_has_files(d):
                missing.append(f"{slot}[empty]")
        if not missing:
            complete.append(subj)
        else:
            incomplete[subj] = missing

    bar = "=" * 60
    print(bar)
    print("DICOM completeness report")
    print(f"  root          : {root}")
    print(f"  sequences req : {' '.join(slots)}")
    print(bar)
    print(f"Subjects scanned : {len(subj_list)}")
    print(f"Complete         : {len(complete)}")
    print(f"Incomplete       : {len(incomplete)}")
    print()

    if incomplete:
        print("-- Incomplete subjects (missing sequences) --")
        width = max((len(s) for s in incomplete.keys()), default=0)
        for subj in sorted(incomplete.keys()):
            print(f"  {subj:<{width}}  ->  {' '.join(incomplete[subj])}")
        print()

    if complete:
        print("-- Complete subjects --")
        for subj in complete:
            print(f"  {subj}")

    return {
        "root": str(root),
        "required_slots": slots,
        "complete": complete,
        "incomplete": incomplete,
        "total": len(subj_list),
    }


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.command("qvtpy-stage0-download")
@backend_click_option()
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination root for downloaded DICOMs (default: local layout from config).",
)
@click.option(
    "--subjects",
    default=None,
    help=(
        "Comma/space separated subject IDs, or a cohort alias "
        "(e.g. PESA-Brain expands to all XNAT PESA_Brain subjects)."
    ),
)
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Text/CSV/XLSX file with subject IDs (one per line for .txt; subject column auto-detected for tabular files).",
)
@click.option(
    "--sequences",
    default=",".join(DEFAULT_SEQUENCES),
    show_default=True,
    help="Comma-separated sequence keys. Use '4DFLOW' as shorthand for AP+RL+FH.",
)
@click.option(
    "--xnat-config",
    "xnat_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="XNAT profile (YAML/JSON). Falls back to NVITK_XNAT_CONFIG / ~/.config/nvitk/xnat.*.",
)
@click.option("--server", type=str, default=None, help="XNAT server URL (override).")
@click.option("--project", type=str, default=None, help="XNAT project (override).")
@click.option("--user", type=str, default=None, help="XNAT username (override).")
@click.option("--password", type=str, default=None, help="XNAT password (override).")
@click.option(
    "--netrc-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional netrc file for authentication.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip a sequence when its destination directory already has files.",
)
@click.option(
    "--skip-existing-downloads",
    is_flag=True,
    default=False,
    help=(
        "When all requested sequences are already on disk under the local subject "
        "tree, skip the XNAT download for that subject. Missing sequences are "
        "still fetched from XNAT."
    ),
)
@click.option(
    "--database",
    "database_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root with indexed scans (pre-filter TOF + 3× 4DFlow availability).",
)
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="Print a brief QC report of downloaded sequences after the batch.",
)
def main(
    dicom_root: Path,
    subjects: str | None,
    subjects_file: Path | None,
    sequences: str,
    xnat_config_path: Path | None,
    server: str | None,
    project: str | None,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    skip_existing: bool,
    skip_existing_downloads: bool,
    report: bool,
    database_root: Path | None,
) -> None:
    """CLI entry point (``qvtpy-stage0-download``): resolve subjects and XNAT connection, then
    download the requested DICOM sequences for each subject into the local layout."""
    Logger()

    from nvitk.pipes.qvtpy.util.io.paths import layout_local

    local_paths = layout_local(dicom_root=dicom_root if dicom_root else None)

    profile = load_xnat_profile(xnat_config_path)
    conn = resolve_xnat_connection(
        profile,
        server=server,
        project=project,
        user=user,
        password=password,
        netrc_file=str(netrc_file) if netrc_file else None,
    )

    subject_list, conn = resolve_subjects_for_xnat_pipeline(
        subjects=subjects,
        subjects_file=subjects_file,
        xnat_config=conn,
        database_root=database_root,
    )
    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    seq_set = requested_sequence_set(sequences) or set(DEFAULT_SEQUENCES)

    run_download(
        subject_list,
        dicom_root=local_paths.dicom_root,
        xnat_config=conn,
        sequences=seq_set,
        skip_existing=skip_existing,
        skip_existing_downloads=skip_existing_downloads,
        report=report,
    )


__all__ = [
    "DEFAULT_SEQUENCES",
    "SLOT_DIRS",
    "collect_local_subject_dicoms",
    "download_subject",
    "load_subjects",
    "local_subject_dicoms_complete",
    "resolve_subjects_for_xnat_pipeline",
    "print_qc_report",
    "print_qc_report_from_results",
    "run_download",
    "sequence_slot_dir",
    "main",
]


if __name__ == "__main__":
    main()
