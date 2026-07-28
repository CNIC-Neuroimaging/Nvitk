#!/usr/bin/env python3
"""Print a qvtpy pipeline completeness report (per-stage status per subject).

Checks that each subject has the expected on-disk artifacts for the requested
pipeline stages (same markers used by ``skip_existing`` in the stage runners).

For ``stage0_c`` under ``--nifti-root``, each subject must have:

- ``4DFlow/{AP,RL,FH}/*_m.nii*`` and ``*_ph.nii*``
- ``TOF/TOF.nii*``
- ``4DFlow/Angiography_3D.nii*`` and ``4DFlow/ComplexDifference_3D.nii*``

Examples::

    # All subject folders under the default local results root
    python scripts/pesa_brain/reports/qvtpy_qc_report.py

    # Explicit roots and subject list
    python scripts/pesa_brain/reports/qvtpy_qc_report.py \\
        --results-root /data/RESULTS/QVTPy \\
        --nifti-root /data/NIFTI \\
        --subjects PESA5745609,PESA123

    # Default stages through stage6; include optional stage4t and stage7
    python scripts/pesa_brain/reports/qvtpy_qc_report.py --stages stage0_c,stage1,stage2,stage3,stage4,stage5,stage6,stage7

    # All subjects in the XNAT PESA-Brain project (needs XNAT config).
    # Percentages use only subjects with all 4 qvtpy sequences; full cohort size is shown.
    python scripts/pesa_brain/reports/qvtpy_qc_report.py --subjects PESA-Brain --xnat-config .nvitk/xnat.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nvitk.db.xnat import requested_sequence_set
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token
from nvitk.pipes.qvtpy.stages import STAGE_CONVERT, STAGE_DOWNLOAD
from nvitk.pipes.qvtpy.stage0_download import (
    DEFAULT_SEQUENCES,
    load_subjects,
    resolve_subjects_for_xnat_pipeline,
)
from nvitk.pipes.qvtpy.util.io.paths import (
    LOCAL_DEFAULT_DICOM_ROOT,
    LOCAL_DEFAULT_NIFTI_ROOT,
    LOCAL_DEFAULT_RESULTS_ROOT,
    layout_local,
)
from nvitk.pipes.qvtpy.util.io.qc_report import (
    DEFAULT_STAGES,
    ReportSubjects,
    filter_subjects_with_required_sequences,
    parse_stages,
    print_qc_report,
)


def _list_subjects_from_results_root(results_root: Path) -> list[str]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    return sorted(
        p.name
        for p in results_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def resolve_report_subjects(
    *,
    results_root: Path,
    subjects: str | None,
    subjects_file: Path | None,
    xnat_config_path: Path | None,
    database_root: Path | None,
    dicom_root: Path,
    nifti_root: Path,
    sequences: set[str],
) -> ReportSubjects:
    """Resolve subject labels for the QC report."""
    if subjects_file is not None:
        labels = load_subjects(subjects=None, subjects_file=subjects_file)
        return ReportSubjects(subjects=labels)

    if subjects is not None:
        cohort_token = resolve_xnat_project_cohort_token(subjects)
        if cohort_token is not None:
            profile = load_xnat_profile(xnat_config_path)
            conn = resolve_xnat_connection(profile)
            all_labels, conn = resolve_subjects_for_xnat_pipeline(
                subjects=subjects,
                subjects_file=None,
                xnat_config=conn,
                database_root=None,
            )
            eligible, source = filter_subjects_with_required_sequences(
                all_labels,
                sequences=sequences,
                dicom_root=dicom_root,
                nifti_root=nifti_root,
                database_root=database_root,
                project_id=conn.project,
            )
            return ReportSubjects(
                subjects=eligible,
                cohort_label=subjects.strip(),
                cohort_total=len(all_labels),
                excluded_no_sequences=len(all_labels) - len(eligible),
                sequence_source=source,
            )
        return ReportSubjects(
            subjects=load_subjects(subjects=subjects, subjects_file=None)
        )

    return ReportSubjects(subjects=_list_subjects_from_results_root(results_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assess qvtpy pipeline stage completeness under a results root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Root containing per-subject qvtpy outputs "
            f"(default: {LOCAL_DEFAULT_RESULTS_ROOT})."
        ),
    )
    parser.add_argument(
        "--nifti-root",
        type=Path,
        default=None,
        help=(
            "NIfTI root for stage0_c checks (AP/RL/FH m+ph, TOF, "
            "Angiography_3D, ComplexDifference_3D) "
            f"(default: {LOCAL_DEFAULT_NIFTI_ROOT})."
        ),
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=None,
        help=(
            "DICOM root for stage0_d checks "
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
        "--stages",
        default=DEFAULT_STAGES,
        help=(
            "Comma-separated pipeline stages to check "
            f"(default: {DEFAULT_STAGES})."
        ),
    )
    parser.add_argument(
        "--sequences",
        default=",".join(DEFAULT_SEQUENCES),
        help=f"Comma-separated sequence keys for stage0_d (default: {','.join(DEFAULT_SEQUENCES)}).",
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
        help=(
            "Optional dataset root to filter cohort subjects by indexed scans "
            "(TOF + 3× 4DFlow). Without this, local DICOM or NIfTI is used."
        ),
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with code 1 when any subject is incomplete.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-subject stage matrix and full failure details.",
    )
    args = parser.parse_args(argv)

    if args.subjects_file is not None and args.subjects is not None:
        parser.error("Provide only one of --subjects or --subjects-file.")

    try:
        stages = parse_stages(args.stages)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = layout_local(
        dicom_root=args.dicom_root,
        nifti_root=args.nifti_root,
        results_root=args.results_root,
    )
    results_root = paths.results_root.expanduser().resolve()
    nifti_root = paths.nifti_root.expanduser().resolve()
    dicom_root = paths.dicom_root.expanduser().resolve()
    seq_set = requested_sequence_set(args.sequences) or set(DEFAULT_SEQUENCES)

    try:
        cohort = resolve_report_subjects(
            results_root=results_root,
            subjects=args.subjects,
            subjects_file=args.subjects_file,
            xnat_config_path=args.xnat_config,
            database_root=args.database,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            sequences=seq_set,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    subject_list = cohort.subjects
    if not subject_list:
        if cohort.cohort_total:
            print(
                f"No subjects with all required sequences "
                f"(0/{cohort.cohort_total} in {cohort.cohort_label})",
                file=sys.stderr,
            )
        else:
            print(f"No subjects to report under {results_root}", file=sys.stderr)
        return 2

    needs_dicom = STAGE_DOWNLOAD in stages
    needs_nifti = STAGE_CONVERT in stages
    dicom_for_report = dicom_root if needs_dicom else None
    nifti_for_report = nifti_root if needs_nifti else None

    summary = print_qc_report(
        subject_list,
        stages,
        results_root=results_root,
        nifti_root=nifti_for_report,
        dicom_root=dicom_for_report,
        dicom_sequences=seq_set,
        verbose=args.verbose,
        cohort_label=cohort.cohort_label,
        cohort_total=cohort.cohort_total,
        excluded_no_sequences=cohort.excluded_no_sequences,
        sequence_source=cohort.sequence_source,
    )

    if args.fail_on_incomplete and summary["incomplete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
