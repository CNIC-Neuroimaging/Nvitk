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
step is **opt-in** via ``--with-download``; ``stage0_convert`` continues to run
by default.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import click
import pandas as pd

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


def _normalize_header(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).lower())


def _detect_subject_column(columns: Iterable[str]) -> str | None:
    norm = {_normalize_header(c): c for c in columns}
    for candidate in _SUBJECT_COLUMN_CANDIDATES:
        key = _normalize_header(candidate)
        if key in norm:
            return norm[key]
    return None


def _read_subjects_dataframe(path: Path) -> list[str]:
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


# ---------------------------------------------------------------------------
# XNAT helpers (lean: no DataRepo)
# ---------------------------------------------------------------------------


def _coalesce_attr(obj: Any, *names: str) -> Any:
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
    subjects_map = getattr(project, "subjects", None) or {}
    if label in subjects_map:
        return subjects_map[label]
    for _key, subj in subjects_map.items():
        uid = str(_coalesce_attr(subj, "label", "id", "name") or _key)
        if uid == label:
            return subj
    return None


def _dir_has_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for child in directory.iterdir():
        if child.is_file():
            return True
        if child.is_dir() and _dir_has_files(child):
            return True
    return False


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

            slot = SLOT_DIRS.get(sequence)
            if slot is None:
                slot = re.sub(r"[^0-9a-z]+", "_", sequence.lower()).strip("_") or "unknown"

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

    results: dict[str, dict[str, list[Path]]] = {}
    with connect_xnat(xnat_config) as session:
        for subject_label in subjects:
            try:
                results[subject_label] = download_subject(
                    session,
                    xnat_config.project,
                    subject_label,
                    dicom_root=root,
                    sequences=seq_set,
                    skip_existing=skip_existing,
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
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_DICOM_ROOT,
    show_default=True,
    help="Destination root for downloaded DICOMs.",
)
@click.option(
    "--subjects",
    default=None,
    help="Comma/space separated subject IDs (e.g. 'PESA0001,PESA0002').",
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
    report: bool,
) -> None:
    Logger()

    subject_list = load_subjects(subjects=subjects, subjects_file=subjects_file)
    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    seq_set = requested_sequence_set(sequences) or set(DEFAULT_SEQUENCES)

    profile = load_xnat_profile(xnat_config_path)
    conn = resolve_xnat_connection(
        profile,
        server=server,
        project=project,
        user=user,
        password=password,
        netrc_file=str(netrc_file) if netrc_file else None,
    )

    run_download(
        subject_list,
        dicom_root=dicom_root,
        xnat_config=conn,
        sequences=seq_set,
        skip_existing=skip_existing,
        report=report,
    )


__all__ = [
    "DEFAULT_SEQUENCES",
    "SLOT_DIRS",
    "download_subject",
    "load_subjects",
    "print_qc_report",
    "run_download",
    "main",
]


if __name__ == "__main__":
    main()
