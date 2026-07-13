#!/usr/bin/env python3
"""Print a DICOM download completeness report for qvtpy stage-0 layout.

Checks that each subject has the expected sequence slot directories
(``tof``, ``4dflow_ap``, …) with at least one file.

Examples::

    # All subject folders under the default local PESA-Brain DICOM root
    python scripts/dicom_qc_report.py

    # Explicit root and subject list
    python scripts/dicom_qc_report.py /data/DICOM --subjects PESA5745609,PESA123

    # Every subject folder under a path
    python scripts/dicom_qc_report.py /data/DICOM

    # All subjects in the XNAT PESA-Brain project (needs XNAT config)
    python scripts/dicom_qc_report.py --subjects PESA-Brain --xnat-config .nvitk/xnat.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nvitk.db.xnat import requested_sequence_set
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token
from nvitk.pipes.qvtpy.stage0_download import (
    DEFAULT_SEQUENCES,
    load_subjects,
    print_qc_report,
    resolve_subjects_for_xnat_pipeline,
    sequence_slot_dir,
)
from nvitk.pipes.qvtpy.util.paths import LOCAL_DEFAULT_DICOM_ROOT


def _sequences_to_slots(sequences: str) -> list[str]:
    seq_set = requested_sequence_set(sequences)
    if not seq_set:
        seq_set = set(DEFAULT_SEQUENCES)
    slots: list[str] = []
    for seq in sorted(seq_set):
        slot = sequence_slot_dir(seq)
        if slot not in slots:
            slots.append(slot)
    return slots


def _list_subjects_from_dicom_root(dicom_root: Path) -> list[str]:
    if not dicom_root.is_dir():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    return sorted(
        p.name
        for p in dicom_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def resolve_report_subjects(
    *,
    dicom_root: Path,
    subjects: str | None,
    subjects_file: Path | None,
    xnat_config_path: Path | None,
    database_root: Path | None,
) -> list[str]:
    """Resolve subject labels for the QC report."""
    if subjects_file is not None:
        return load_subjects(subjects=None, subjects_file=subjects_file)

    if subjects is not None:
        if resolve_xnat_project_cohort_token(subjects) is not None:
            profile = load_xnat_profile(xnat_config_path)
            conn = resolve_xnat_connection(profile)
            labels, _conn = resolve_subjects_for_xnat_pipeline(
                subjects=subjects,
                subjects_file=None,
                xnat_config=conn,
                database_root=database_root,
            )
            return labels
        return load_subjects(subjects=subjects, subjects_file=None)

    return _list_subjects_from_dicom_root(dicom_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assess qvtpy DICOM download completeness under a DICOM root.",
    )
    parser.add_argument(
        "dicom_root",
        nargs="?",
        type=Path,
        default=LOCAL_DEFAULT_DICOM_ROOT,
        help=(
            "Root containing per-subject folders "
            f"(default: {LOCAL_DEFAULT_DICOM_ROOT})."
        ),
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help=(
            "Comma/space separated subject IDs, or a cohort alias "
            "(e.g. PESA-Brain expands via XNAT)."
        ),
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=None,
        help="Text/CSV/XLSX file with subject IDs.",
    )
    parser.add_argument(
        "--sequences",
        default=",".join(DEFAULT_SEQUENCES),
        help=f"Comma-separated sequence keys (default: {','.join(DEFAULT_SEQUENCES)}). "
        "4DFLOW expands to AP+RL+FH.",
    )
    parser.add_argument(
        "--xnat-config",
        type=Path,
        default=None,
        help="XNAT profile for cohort aliases (PESA-Brain, …).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Optional dataset root to pre-filter subjects by indexed scans.",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with code 1 when any subject is incomplete.",
    )
    args = parser.parse_args(argv)

    dicom_root = args.dicom_root.expanduser().resolve()
    if args.subjects_file is not None and args.subjects is not None:
        parser.error("Provide only one of --subjects or --subjects-file.")

    try:
        subject_list = resolve_report_subjects(
            dicom_root=dicom_root,
            subjects=args.subjects,
            subjects_file=args.subjects_file,
            xnat_config_path=args.xnat_config,
            database_root=args.database,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not subject_list:
        print(f"No subjects to report under {dicom_root}", file=sys.stderr)
        return 2

    slots = _sequences_to_slots(args.sequences)
    summary = print_qc_report(dicom_root, subject_list, slots)

    if args.fail_on_incomplete and summary["incomplete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
