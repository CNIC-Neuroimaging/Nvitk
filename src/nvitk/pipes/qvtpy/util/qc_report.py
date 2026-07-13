"""Filesystem QC for qvtpy pipeline stages (per-subject completion markers).

Each stage is considered complete when the same output artifacts that
``skip_existing`` checks in the stage runner are present on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.run import (
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
    _STAGE_ALIASES,
    _STAGES_ORDERED,
)
from nvitk.pipes.qvtpy.stage0_convert import REQUIRED_FLOW_DIRS
from nvitk.pipes.qvtpy.stage0_download import (
    DEFAULT_SEQUENCES,
    local_subject_dicoms_complete,
    sequence_slot_dir,
)
from nvitk.pipes.qvtpy.stage1_eicab import _output_has_segmentation
from nvitk.pipes.qvtpy.util.morpho_paths import STAGE7_SKIP_MARKER


@dataclass(frozen=True)
class StageCheck:
    """Completion status for one pipeline stage on one subject."""

    stage: str
    complete: bool
    detail: str


def parse_stages(spec: str) -> list[str]:
    """Parse a comma-separated stage list into canonical ids in pipeline order."""
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--stages cannot be empty.")
    canonical: set[str] = set()
    for tok in tokens:
        key = tok.replace("-", "_")
        if key not in _STAGE_ALIASES:
            valid = ", ".join(sorted(set(_STAGE_ALIASES.keys())))
            raise ValueError(f"Unknown stage {tok!r}. Valid: {valid}.")
        canonical.add(_STAGE_ALIASES[key])
    return [s for s in _STAGES_ORDERED if s in canonical]


def _glob_first(directory: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def _eicab_dir(results_root: Path, subject: str) -> Path:
    return results_root / subject / cfg.STAGE1_EICAB_DIR


def _qvt_stage_dir(results_root: Path, subject: str, stage_dir: str) -> Path:
    return results_root / subject / cfg.QVT_SUBDIR / stage_dir


def _check_stage0_d(
    dicom_root: Path,
    subject: str,
    *,
    sequences: Iterable[str],
) -> StageCheck:
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

    if not missing:
        return StageCheck(STAGE_CONVERT, True, "ok")
    return StageCheck(STAGE_CONVERT, False, " ".join(missing))


def _check_stage1(results_root: Path, subject: str) -> StageCheck:
    out_dir = _eicab_dir(results_root, subject)
    if _output_has_segmentation(out_dir):
        return StageCheck(STAGE_EICAB, True, "ok")
    return StageCheck(STAGE_EICAB, False, "no eICAB segmentation NIfTI")


def _check_stage2(results_root: Path, subject: str) -> StageCheck:
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE2_REGISTRATION_DIR) / "registration_meta.json"
    if marker.is_file():
        return StageCheck(STAGE_REG, True, "ok")
    return StageCheck(STAGE_REG, False, "missing registration_meta.json")


def _check_stage3(results_root: Path, subject: str) -> StageCheck:
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE3_CENTERLINE_DIR) / "centerline_meta.json"
    if marker.is_file():
        return StageCheck(STAGE_CENTERLINE, True, "ok")
    return StageCheck(STAGE_CENTERLINE, False, "missing centerline_meta.json")


def _check_stage4(results_root: Path, subject: str) -> StageCheck:
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
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE5_LOC_DIR) / "locs.csv"
    if marker.is_file():
        return StageCheck(STAGE_LOC, True, "ok")
    return StageCheck(STAGE_LOC, False, "missing locs.csv")


def _check_stage6(results_root: Path, subject: str) -> StageCheck:
    marker = _qvt_stage_dir(results_root, subject, cfg.STAGE6_MEASURE_DIR) / "loc_measurements.csv"
    if marker.is_file():
        return StageCheck(STAGE_MEASURE, True, "ok")
    return StageCheck(STAGE_MEASURE, False, "missing loc_measurements.csv")


def _check_stage7(results_root: Path, subject: str) -> StageCheck:
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
    return stage.replace("stage", "s")


def print_qc_report(
    subjects: Iterable[str],
    stages: Iterable[str],
    *,
    results_root: Path,
    nifti_root: Path | None = None,
    dicom_root: Path | None = None,
    dicom_sequences: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Print a per-subject stage completion report and return a summary dict."""
    stage_list = list(stages)
    subj_list = sorted({s for s in subjects if s})

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
    print(f"Subjects scanned : {len(subj_list)}")
    print(f"Fully complete   : {len(complete)}")
    print(f"Incomplete       : {len(incomplete)}")
    print()

    if incomplete:
        print("-- Incomplete subjects (failed stages) --")
        subj_width = max(len(s) for s in incomplete)
        stage_width = max(len(_stage_short_label(s)) for s in stage_list)
        header = f"{'subject':<{subj_width}}  " + "  ".join(
            f"{_stage_short_label(s):>{stage_width}}" for s in stage_list
        )
        print(header)
        print("-" * len(header))
        for subj in sorted(incomplete):
            checks = {c.stage: c for c in per_subject[subj]}
            cells: list[str] = []
            for stage in stage_list:
                check = checks[stage]
                cells.append("ok" if check.complete else "FAIL")
            line = f"{subj:<{subj_width}}  " + "  ".join(
                f"{cell:>{stage_width}}" for cell in cells
            )
            print(line)
        print()
        print("-- Failure details --")
        for subj in sorted(incomplete):
            for item in incomplete[subj]:
                print(f"  {subj}: {item}")
        print()

    if complete:
        print("-- Fully complete subjects --")
        for subj in complete:
            print(f"  {subj}")
        print()

    return {
        "results_root": str(results_root),
        "stages": stage_list,
        "complete": complete,
        "incomplete": incomplete,
        "per_subject": {
            subj: [{"stage": c.stage, "complete": c.complete, "detail": c.detail} for c in checks]
            for subj, checks in per_subject.items()
        },
        "total": len(subj_list),
    }


__all__ = [
    "DEFAULT_STAGES",
    "StageCheck",
    "check_subject_stages",
    "parse_stages",
    "print_qc_report",
]
