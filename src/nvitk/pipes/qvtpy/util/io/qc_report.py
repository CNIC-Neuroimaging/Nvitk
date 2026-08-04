"""Filesystem QC for qvtpy pipeline stages (per-subject completion markers).

Each stage is considered complete when the same output artifacts that
``skip_existing`` checks in the stage runner are present on disk.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import click

from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stages import (
    DEFAULT_STAGES,
    STAGE_CENTERLINE,
    STAGE_CONVERT,
    STAGE_DOWNLOAD,
    STAGE_EICAB,
    STAGE_LOC,
    STAGE_MEASURE,
    STAGE_MORPHOMETRICS,
    STAGE_REG,
    STAGE_SEG,
    STAGE_SEG_T,
    parse_stages as _parse_stages_spec,
)
from nvitk.pipes.qvtpy.stage0_convert import REQUIRED_DERIVED_FILES, REQUIRED_FLOW_DIRS
from nvitk.pipes.qvtpy.stage0_download import (
    DEFAULT_SEQUENCES,
    local_subject_dicoms_complete,
    sequence_slot_dir,
)
from nvitk.pipes.qvtpy.stage1_eicab import _output_has_segmentation
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER

_DEFAULT_SUBJECT_PREVIEW = 3


@dataclass(frozen=True)
class ReportSubjects:
    """Resolved subject list plus optional XNAT cohort metadata."""

    subjects: list[str]
    cohort_label: str | None = None
    cohort_total: int | None = None
    excluded_no_sequences: int = 0
    sequence_source: str | None = None


@dataclass(frozen=True)
class StageCheck:
    """Completion status for one pipeline stage on one subject."""

    stage: str
    complete: bool
    detail: str


def parse_stages(spec: str) -> list[str]:
    """Parse a comma-separated stage list into canonical ids in pipeline order."""
    try:
        return _parse_stages_spec(spec)
    except click.ClickException as exc:
        raise ValueError(str(exc)) from exc


def _glob_first(directory: Path, *patterns: str) -> Path | None:
    """First file in *directory* matching any of *patterns* (patterns tried in order)."""
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def _eicab_dir(results_root: Path, subject: str) -> Path:
    """Stage 1 (eICAB) output directory for *subject* under *results_root*."""
    return results_root / subject / cfg.STAGE1_EICAB_DIR


def _qvt_stage_dir(results_root: Path, subject: str, stage_dir: str) -> Path:
    """qvtpy *stage_dir* output directory for *subject* under *results_root*."""
    return results_root / subject / cfg.QVT_SUBDIR / stage_dir


def _nifti_convert_complete(nifti_root: Path, subject: str) -> bool:
    """True if *subject*'s stage-0 NIfTI conversion outputs are all present."""
    return _check_stage0_c(nifti_root, subject).complete


def filter_subjects_with_required_sequences(
    subjects: Iterable[str],
    *,
    sequences: Iterable[str],
    dicom_root: Path | None = None,
    nifti_root: Path | None = None,
    database_root: Path | None = None,
    project_id: str | None = None,
) -> tuple[list[str], str]:
    """Keep subjects with all requested qvtpy input sequences on disk or in the catalog."""
    seq_list = list(sequences)
    subject_list = sorted({s for s in subjects if s})

    if database_root is not None and project_id is not None:
        from nvitk.pipes.qvtpy.util.io.db_subject_filter import (
            filter_subjects_by_qvtpy_scan_availability,
        )

        filtered = filter_subjects_by_qvtpy_scan_availability(
            subject_list,
            database_root=database_root,
            project_id=project_id,
        )
        return filtered, "indexed scans (database)"

    if dicom_root is not None:
        filtered = [
            subj
            for subj in subject_list
            if local_subject_dicoms_complete(dicom_root, subj, seq_list)
        ]
        return filtered, "local DICOM"

    if nifti_root is not None:
        filtered = [
            subj for subj in subject_list if _nifti_convert_complete(nifti_root, subj)
        ]
        return filtered, "local NIfTI"

    return subject_list, "none"


def _check_stage0_d(
    dicom_root: Path,
    subject: str,
    *,
    sequences: Iterable[str],
) -> StageCheck:
    """Stage 0 (download) completion: all requested DICOM sequences present on disk."""
    seq_list = list(sequences)
    if local_subject_dicoms_complete(dicom_root, subject, seq_list):
        return StageCheck(STAGE_DOWNLOAD, True, "ok")
    missing: list[str] = []
    subj_dir = dicom_root / subject
    for seq in seq_list:
        slot = sequence_slot_dir(seq)
        slot_dir = subj_dir / slot
        if not slot_dir.is_dir():
            missing.append(f"{slot}[missing dir]")
        elif not any(p.is_file() for p in slot_dir.rglob("*")):
            missing.append(f"{slot}[empty]")
    return StageCheck(STAGE_DOWNLOAD, False, " ".join(missing) or "incomplete")


def _check_stage0_c(nifti_root: Path, subject: str) -> StageCheck:
    """Stage 0 (convert) completion: required 4D-flow/TOF/derived NIfTI files all present."""
    subj_dir = nifti_root / subject
    missing: list[str] = []

    flow_root = subj_dir / "4DFlow"
    for d in REQUIRED_FLOW_DIRS:
        dd = flow_root / d
        if not dd.is_dir():
            missing.extend([f"{d}_m[missing dir]", f"{d}_ph[missing dir]"])
            continue
        if _glob_first(dd, "*_m.nii.gz", "*_m.nii") is None:
            missing.append(f"{d}_m[missing]")
        if _glob_first(dd, "*_ph.nii.gz", "*_ph.nii") is None:
            missing.append(f"{d}_ph[missing]")

    tof_dir = subj_dir / "TOF"
    if not tof_dir.is_dir():
        missing.append("TOF[missing dir]")
    elif _glob_first(tof_dir, "TOF.nii.gz", "TOF.nii") is None:
        missing.append("TOF[missing]")

    for stem in REQUIRED_DERIVED_FILES:
        if not (
            (flow_root / f"{stem}.nii.gz").is_file()
            or (flow_root / f"{stem}.nii").is_file()
        ):
            missing.append(f"{stem}[missing]")

    if not missing:
        return StageCheck(STAGE_CONVERT, True, "ok")
    return StageCheck(STAGE_CONVERT, False, " ".join(missing))


def _check_stage1(results_root: Path, subject: str) -> StageCheck:
    """Stage 1 (eICAB) completion: an eICAB segmentation NIfTI exists."""
    out_dir = _eicab_dir(results_root, subject)
    if _output_has_segmentation(out_dir):
        return StageCheck(STAGE_EICAB, True, "ok")
    return StageCheck(STAGE_EICAB, False, "no eICAB segmentation NIfTI")


def _check_stage2(results_root: Path, subject: str) -> StageCheck:
    """Stage 2 (registration) completion: ``registration_meta.json`` exists."""
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE2_REGISTRATION_DIR) / "registration_meta.json"
    if marker.is_file():
        return StageCheck(STAGE_REG, True, "ok")
    return StageCheck(STAGE_REG, False, "missing registration_meta.json")


def _check_stage3(results_root: Path, subject: str) -> StageCheck:
    """Stage 3 (centerline) completion: ``centerline_meta.json`` exists."""
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE3_CENTERLINE_DIR) / "centerline_meta.json"
    if marker.is_file():
        return StageCheck(STAGE_CENTERLINE, True, "ok")
    return StageCheck(STAGE_CENTERLINE, False, "missing centerline_meta.json")


def _check_stage4(results_root: Path, subject: str) -> StageCheck:
    """Stage 4 (segmentation) completion: segmentation NIfTI and metadata JSON both exist."""
    out_dir = _qvt_stage_dir(results_root, subject, cfg.STAGE4_SEG_DIR)
    seg = out_dir / "seg_4dflow.nii.gz"
    meta = out_dir / "segmentation_meta.json"
    if seg.is_file() and meta.is_file():
        return StageCheck(STAGE_SEG, True, "ok")
    missing: list[str] = []
    if not seg.is_file():
        missing.append("seg_4dflow.nii.gz")
    if not meta.is_file():
        missing.append("segmentation_meta.json")
    return StageCheck(STAGE_SEG, False, ", ".join(missing))


def _check_stage4t(results_root: Path, subject: str) -> StageCheck:
    """Stage 4t (temporal segmentation) completion: 4D seg volume, temporal summary, and every
    per-timepoint metadata JSON all exist."""
    out_dir = _qvt_stage_dir(results_root, subject, cfg.STAGE4T_SEG_DIR)
    seg = out_dir / "seg_4dflow_4d.nii.gz"
    summary = out_dir / "temporal_seg_summary.json"
    if not seg.is_file():
        return StageCheck(STAGE_SEG_T, False, "missing seg_4dflow_4d.nii.gz")
    if not summary.is_file():
        return StageCheck(STAGE_SEG_T, False, "missing temporal_seg_summary.json")
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        n_t = int(payload.get("n_timepoints", payload.get("n_frames", 0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return StageCheck(STAGE_SEG_T, False, "invalid temporal_seg_summary.json")
    if n_t < 1:
        return StageCheck(STAGE_SEG_T, False, "temporal_seg_summary.json has no timepoints")
    missing_meta = [
        f"segmentation_meta_t{t:02d}.json"
        for t in range(n_t)
        if not (out_dir / f"segmentation_meta_t{t:02d}.json").is_file()
    ]
    if missing_meta:
        return StageCheck(STAGE_SEG_T, False, "missing " + ", ".join(missing_meta))
    return StageCheck(STAGE_SEG_T, True, "ok")


def _check_stage5(results_root: Path, subject: str) -> StageCheck:
    """Stage 5 (LOC generation) completion: ``locs.csv`` exists."""
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE5_LOC_DIR) / "locs.csv"
    if marker.is_file():
        return StageCheck(STAGE_LOC, True, "ok")
    return StageCheck(STAGE_LOC, False, "missing locs.csv")


def _check_stage6(results_root: Path, subject: str) -> StageCheck:
    """Stage 6 (measure) completion: ``loc_measurements.csv`` exists."""
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE6_MEASURE_DIR) / "loc_measurements.csv"
    if marker.is_file():
        return StageCheck(STAGE_MEASURE, True, "ok")
    return StageCheck(STAGE_MEASURE, False, "missing loc_measurements.csv")


def _check_stage7(results_root: Path, subject: str) -> StageCheck:
    """Stage 7 (morphometrics) completion: the stage-7 skip/done marker exists."""
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE7_MORPHOMETRICS_DIR) / STAGE7_SKIP_MARKER
    if marker.is_file():
        return StageCheck(STAGE_MORPHOMETRICS, True, "ok")
    return StageCheck(STAGE_MORPHOMETRICS, False, f"missing {STAGE7_SKIP_MARKER}")


def check_subject_stages(
    subject: str,
    stages: Iterable[str],
    *,
    results_root: Path,
    nifti_root: Path | None = None,
    dicom_root: Path | None = None,
    dicom_sequences: Iterable[str] | None = None,
) -> list[StageCheck]:
    """Return completion status for *subject* across the requested *stages*."""
    seqs = list(dicom_sequences or DEFAULT_SEQUENCES)
    checks: list[StageCheck] = []
    for stage in stages:
        if stage == STAGE_DOWNLOAD:
            if dicom_root is None:
                checks.append(StageCheck(stage, False, "dicom_root not provided"))
            else:
                checks.append(_check_stage0_d(dicom_root, subject, sequences=seqs))
        elif stage == STAGE_CONVERT:
            if nifti_root is None:
                checks.append(StageCheck(stage, False, "nifti_root not provided"))
            else:
                checks.append(_check_stage0_c(nifti_root, subject))
        elif stage == STAGE_EICAB:
            checks.append(_check_stage1(results_root, subject))
        elif stage == STAGE_REG:
            checks.append(_check_stage2(results_root, subject))
        elif stage == STAGE_CENTERLINE:
            checks.append(_check_stage3(results_root, subject))
        elif stage == STAGE_SEG:
            checks.append(_check_stage4(results_root, subject))
        elif stage == STAGE_SEG_T:
            checks.append(_check_stage4t(results_root, subject))
        elif stage == STAGE_LOC:
            checks.append(_check_stage5(results_root, subject))
        elif stage == STAGE_MEASURE:
            checks.append(_check_stage6(results_root, subject))
        elif stage == STAGE_MORPHOMETRICS:
            checks.append(_check_stage7(results_root, subject))
        else:
            checks.append(StageCheck(stage, False, f"unknown stage {stage!r}"))
    return checks


def _stage_short_label(stage: str) -> str:
    """Compact label for *stage* (``"stage1"`` -> ``"s1"``) used in table headers."""
    return stage.replace("stage", "s")


def _compact_subject_list(subjects: Iterable[str], *, max_show: int = _DEFAULT_SUBJECT_PREVIEW) -> str:
    """Comma-joined subject list, truncated to *max_show* names with a ``"(+N more)"`` suffix."""
    names = sorted(subjects)
    if not names:
        return "(none)"
    if len(names) <= max_show:
        return ", ".join(names)
    shown = ", ".join(names[:max_show])
    return f"{shown} (+{len(names) - max_show} more)"


def _pct(numerator: int, denominator: int) -> str:
    """Format ``numerator/denominator`` as a right-aligned percentage string (``"n/a"`` if 0/0)."""
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:5.1f}%"


def _print_stage_summary(
    stage_list: list[str],
    per_subject: dict[str, list[StageCheck]],
    total: int,
) -> dict[str, dict[str, int]]:
    """Print per-stage ok/fail counts; return numeric summary."""
    stage_stats: dict[str, dict[str, int]] = {}
    print("-- Stage completion --")
    print(f"  {'stage':<12} {'ok':>4} {'fail':>5} {'pct':>7}")
    for stage in stage_list:
        ok = sum(
            1
            for checks in per_subject.values()
            for c in checks
            if c.stage == stage and c.complete
        )
        fail = total - ok
        stage_stats[stage] = {"ok": ok, "fail": fail, "total": total}
        print(f"  {stage:<12} {ok:4d} {fail:5d} {_pct(ok, total):>7}")
    print()
    return stage_stats


def _print_failure_summary(
    stage_list: list[str],
    per_subject: dict[str, list[StageCheck]],
) -> None:
    """Aggregate failure reasons per stage (no per-subject listing)."""
    reasons_by_stage: dict[str, Counter[str]] = {stage: Counter() for stage in stage_list}
    for checks in per_subject.values():
        for check in checks:
            if not check.complete:
                reasons_by_stage[check.stage][check.detail] += 1

    failing_stages = [stage for stage in stage_list if reasons_by_stage[stage]]
    if not failing_stages:
        return

    print("-- Failures by stage --")
    for stage in failing_stages:
        counter = reasons_by_stage[stage]
        n_fail = sum(counter.values())
        print(f"  {stage} ({n_fail} subject(s)):")
        for detail, count in counter.most_common():
            print(f"    {detail} — {count}")
    print()


def _print_incomplete_cohorts(
    stage_list: list[str],
    per_subject: dict[str, list[StageCheck]],
    *,
    max_show: int = _DEFAULT_SUBJECT_PREVIEW,
) -> None:
    """Group incomplete subjects by which stages failed."""
    cohorts: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for subj, checks in per_subject.items():
        failed = _failed_stages_for_subject(checks, stage_list)
        if failed:
            cohorts[failed].append(subj)

    if not cohorts:
        return

    print("-- Incomplete cohorts --")
    for failed_stages, subjects in sorted(cohorts.items(), key=lambda item: (-len(item[1]), item[0])):
        label = ", ".join(failed_stages)
        preview = _compact_subject_list(subjects, max_show=max_show)
        print(f"  failed [{label}] ({len(subjects)}): {preview}")
    print()


def _failed_stages_for_subject(
    checks: list[StageCheck],
    stage_list: list[str],
) -> tuple[str, ...]:
    """Ordered tuple of *stage_list* stage ids that are incomplete for this subject's *checks*."""
    by_stage = {c.stage: c for c in checks}
    return tuple(stage for stage in stage_list if not by_stage[stage].complete)


def _print_verbose_details(
    stage_list: list[str],
    per_subject: dict[str, list[StageCheck]],
    *,
    complete: list[str],
    incomplete: dict[str, list[str]],
) -> None:
    """Per-subject matrix and failure lines (opt-in)."""
    if incomplete:
        print("-- Per-subject status (verbose) --")
        subj_width = max(len(s) for s in incomplete)
        stage_width = max(len(_stage_short_label(s)) for s in stage_list)
        header = f"{'subject':<{subj_width}}  " + "  ".join(
            f"{_stage_short_label(s):>{stage_width}}" for s in stage_list
        )
        print(header)
        print("-" * len(header))
        for subj in sorted(incomplete):
            checks = {c.stage: c for c in per_subject[subj]}
            cells = [
                "ok" if checks[stage].complete else "FAIL"
                for stage in stage_list
            ]
            line = f"{subj:<{subj_width}}  " + "  ".join(
                f"{cell:>{stage_width}}" for cell in cells
            )
            print(line)
        print()

        print("-- Failure details (verbose) --")
        for subj in sorted(incomplete):
            for item in incomplete[subj]:
                print(f"  {subj}: {item}")
        print()

    if complete:
        print(f"-- Fully complete ({len(complete)}) --")
        print(f"  {_compact_subject_list(complete, max_show=8)}")
        print()


def print_qc_report(
    subjects: Iterable[str],
    stages: Iterable[str],
    *,
    results_root: Path,
    nifti_root: Path | None = None,
    dicom_root: Path | None = None,
    dicom_sequences: Iterable[str] | None = None,
    verbose: bool = False,
    max_subject_preview: int = _DEFAULT_SUBJECT_PREVIEW,
    cohort_label: str | None = None,
    cohort_total: int | None = None,
    excluded_no_sequences: int = 0,
    sequence_source: str | None = None,
) -> dict[str, Any]:
    """Print a summarized stage completion report and return a summary dict."""
    stage_list = list(stages)
    subj_list = sorted({s for s in subjects if s})
    total = len(subj_list)
    seq_list = list(dicom_sequences or DEFAULT_SEQUENCES)

    per_subject: dict[str, list[StageCheck]] = {}
    complete: list[str] = []
    incomplete: dict[str, list[str]] = {}

    for subj in subj_list:
        checks = check_subject_stages(
            subj,
            stage_list,
            results_root=results_root,
            nifti_root=nifti_root,
            dicom_root=dicom_root,
            dicom_sequences=dicom_sequences,
        )
        per_subject[subj] = checks
        failed = [f"{c.stage}: {c.detail}" for c in checks if not c.complete]
        if failed:
            incomplete[subj] = failed
        else:
            complete.append(subj)

    bar = "=" * 72
    print(bar)
    print("qvtpy pipeline QC report")
    print(f"  results_root : {results_root}")
    if nifti_root is not None:
        print(f"  nifti_root   : {nifti_root}")
    if dicom_root is not None:
        print(f"  dicom_root   : {dicom_root}")
    print(f"  stages       : {', '.join(stage_list)}")
    print(bar)
    n_complete = len(complete)
    n_incomplete = len(incomplete)
    seq_label = ", ".join(seq_list)
    if cohort_label is not None and cohort_total is not None:
        source = sequence_source or "required sequences"
        print(f"Cohort {cohort_label} : {cohort_total} in project")
        print(
            f"  scored {total} with all required sequences ({source}; {seq_label})"
        )
        if excluded_no_sequences:
            print(f"  excluded {excluded_no_sequences} without required sequences")
        print(
            f"  pipeline : {n_complete} complete ({_pct(n_complete, total).strip()}) | "
            f"{n_incomplete} incomplete ({_pct(n_incomplete, total).strip()})"
        )
    else:
        print(
            f"Subjects : {total} total | "
            f"{n_complete} complete ({_pct(n_complete, total).strip()}) | "
            f"{n_incomplete} incomplete ({_pct(n_incomplete, total).strip()})"
        )
    print()

    stage_stats = _print_stage_summary(stage_list, per_subject, total)
    _print_failure_summary(stage_list, per_subject)
    _print_incomplete_cohorts(
        stage_list,
        per_subject,
        max_show=max_subject_preview,
    )

    if verbose:
        _print_verbose_details(
            stage_list,
            per_subject,
            complete=complete,
            incomplete=incomplete,
        )
    elif incomplete:
        print("Use --verbose for per-subject status and full subject lists.")
        print()

    return {
        "results_root": str(results_root),
        "stages": stage_list,
        "complete": complete,
        "incomplete": incomplete,
        "stage_stats": stage_stats,
        "cohort_label": cohort_label,
        "cohort_total": cohort_total,
        "excluded_no_sequences": excluded_no_sequences,
        "sequence_source": sequence_source,
        "per_subject": {
            subj: [{"stage": c.stage, "complete": c.complete, "detail": c.detail} for c in checks]
            for subj, checks in per_subject.items()
        },
        "total": total,
    }


__all__ = [
    "DEFAULT_STAGES",
    "ReportSubjects",
    "StageCheck",
    "check_subject_stages",
    "filter_subjects_with_required_sequences",
    "parse_stages",
    "print_qc_report",
]
