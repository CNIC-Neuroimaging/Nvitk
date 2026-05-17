"""Black-blood stage 0 (download): XNAT -> DICOM for VWI_BB (BrainVIEW T1W).

Downloads one BrainVIEW variant per subject (strong > default > weak) into
``{dicom_root}/{subject}/vwi_bb/``.
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
    resolve_subject_labels,
    select_preferred_vwi_bb_scan,
)
from nvitk.db.xnat_config import (
    XnatConnectionConfig,
    load_xnat_profile,
    resolve_xnat_connection,
)

from . import config as cfg

log = Logger()

DEFAULT_SEQUENCES: tuple[str, ...] = ("VWI_BB",)
VWI_BB_SLOT = cfg.STAGE0_DICOM_SLOT

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
    if (subjects is None and subjects_file is None) or (
        subjects is not None and subjects_file is not None
    ):
        raise click.ClickException(
            "Provide exactly one of --subjects or --subjects-file."
        )

    if subjects is not None:
        return sorted(set(parse_subject_tokens(subjects)))

    path = Path(subjects_file).expanduser().resolve()
    if not path.exists():
        raise click.ClickException(f"subjects-file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".xlsx", ".xls"):
        return _read_subjects_dataframe(path)
    return resolve_subject_labels(subjects_file=path)


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


def download_subject(
    xnat_session: Any,
    project_id: str,
    subject_label: str,
    *,
    dicom_root: Path,
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """Download preferred VWI_BB DICOMs for one subject."""
    project = xnat_session.projects[project_id]
    subject = _resolve_subject(project, subject_label)
    if subject is None:
        raise LookupError(
            f"Subject {subject_label!r} not found in XNAT project {project_id!r}"
        )

    out: dict[str, list[Path]] = {"VWI_BB": []}
    target_dir = dicom_root / subject_label / VWI_BB_SLOT

    if skip_existing and _dir_has_files(target_dir):
        log.info(
            f"[{subject_label}] VWI_BB: skip-existing -> {target_dir}"
        )
        existing = sorted(p for p in target_dir.rglob("*") if p.is_file())
        if existing:
            out["VWI_BB"] = existing
        return out

    candidates: list[dict[str, Any]] = []
    for experiment in getattr(subject, "experiments", {}).values():
        for scan in getattr(experiment, "scans", {}).values():
            series_description = str(
                _coalesce_attr(scan, "series_description", "type", "label") or ""
            )
            quality = str(_coalesce_attr(scan, "quality") or "")
            classification = classify_scan(series_description, quality)
            if classification is None or classification.get("sequence") != "VWI_BB":
                continue
            scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
            candidates.append(
                {
                    "scan": scan,
                    "variant": classification.get("variant"),
                    "scan_id": scan_id,
                    "series_description": series_description,
                }
            )

    chosen = select_preferred_vwi_bb_scan(candidates, subject_label=subject_label)
    if chosen is None:
        log.warning(f"[{subject_label}] no VWI_BB (BrainVIEW) scan found on XNAT")
        return out

    scan = chosen["scan"]
    variant = chosen.get("variant")
    scan_id = chosen.get("scan_id")
    log.info(
        f"[{subject_label}] downloading VWI_BB variant={variant!r} "
        f"(scan_id={scan_id}) -> {target_dir}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = download_scan_dicoms(scan, target_dir)
    except Exception as exc:
        log.warning(f"[{subject_label}] VWI_BB download failed: {exc}")
        return out
    out["VWI_BB"] = files
    return out


def run_download(
    subjects: list[str],
    *,
    dicom_root: str | Path,
    xnat_config: XnatConnectionConfig,
    skip_existing: bool = True,
    report: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    root = Path(dicom_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    log.info(f"black_blood stage0_download | subjects={len(subjects)}")
    log.info(f"  dicom_root : {root}")
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
                    skip_existing=skip_existing,
                )
            except LookupError as exc:
                log.warning(f"[{subject_label}] {exc}")
                results[subject_label] = {"VWI_BB": []}
            except Exception as exc:
                log.warning(f"[{subject_label}] unexpected error: {exc}")
                results[subject_label] = {"VWI_BB": []}

    if report:
        print_qc_report(root, subjects, [VWI_BB_SLOT])

    return results


def print_qc_report(
    dicom_root: str | Path,
    subjects: Iterable[str],
    required_slots: list[str],
) -> dict[str, Any]:
    root = Path(dicom_root)
    subj_list = list(subjects)
    complete: list[str] = []
    incomplete: list[str] = []

    for subj in subj_list:
        ok = all(_dir_has_files(root / subj / slot) for slot in required_slots)
        if ok:
            complete.append(subj)
        else:
            incomplete.append(subj)

    print()
    print("DICOM completeness report (black_blood VWI_BB)")
    print(f"  root: {root}")
    print(f"  required slots: {required_slots}")
    print(f"  complete: {len(complete)} / {len(subj_list)}")
    if incomplete:
        print("-- Incomplete --")
        for subj in incomplete:
            print(f"  {subj}")

    return {
        "root": str(root),
        "required_slots": required_slots,
        "complete": complete,
        "incomplete": incomplete,
        "total": len(subj_list),
    }


@click.command("nvitk-pesa-brain-bb-download")
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
    help="Destination root for downloaded DICOMs.",
)
@click.option("--subjects", default=None)
@click.option("--subjects-file", type=click.Path(path_type=Path), default=None)
@click.option(
    "--xnat-config",
    "xnat_config_path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option("--server", type=str, default=None)
@click.option("--project", type=str, default=None)
@click.option("--user", type=str, default=None)
@click.option("--password", type=str, default=None)
@click.option("--netrc-file", type=click.Path(path_type=Path), default=None)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
@click.option("--report", is_flag=True, default=False)
def main(
    dicom_root: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    xnat_config_path: Path | None,
    server: str | None,
    project: str | None,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    skip_existing: bool,
    report: bool,
) -> None:
    """Download VWI_BB (BrainVIEW) DICOMs from XNAT."""
    subject_list = load_subjects(subjects=subjects, subjects_file=subjects_file)
    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    root = dicom_root or cfg.DEFAULT_DICOM_ROOT
    if root is None:
        raise click.ClickException(
            "Set --dicom-root or black_blood.config.DEFAULT_DICOM_ROOT."
        )

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
        dicom_root=root,
        xnat_config=conn,
        skip_existing=skip_existing,
        report=report,
    )


if __name__ == "__main__":
    main()
