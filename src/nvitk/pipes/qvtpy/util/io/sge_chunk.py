"""QVTpy-specific SGE chunk helpers (stage counts for per-user job limits)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Canonical stage ids (match nvitk.pipes.qvtpy.stages constants).
STAGE_CONVERT = "stage0_c"
STAGE_EICAB = "stage1"
STAGE_REG = "stage2"
STAGE_CENTERLINE = "stage3"
STAGE_SEG = "stage4"
STAGE_SEG_T = "stage4t"
STAGE_LOC = "stage5"
STAGE_MEASURE = "stage6"
STAGE_MORPHOMETRICS = "stage7"

_SGE_RUN_STAGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("run_conv", STAGE_CONVERT),
    ("run_eicab", STAGE_EICAB),
    ("run_s2", STAGE_REG),
    ("run_s3", STAGE_CENTERLINE),
    ("run_s4", STAGE_SEG),
    ("run_s4t", STAGE_SEG_T),
    ("run_s5", STAGE_LOC),
    ("run_s6", STAGE_MEASURE),
    ("run_s7", STAGE_MORPHOMETRICS),
)


def _enabled_stage_ids(stage_runs: dict[str, bool]) -> list[str]:
    return [stage_id for flag, stage_id in _SGE_RUN_STAGE_PAIRS if stage_runs.get(flag, False)]


def pending_sge_stage_ids(
    subject: str,
    *,
    stage_runs: dict[str, bool],
    skip_processed: bool,
    results_root: Path,
    nifti_root: Path,
) -> list[str]:
    """Return enabled stage ids that still need work for *subject*."""
    enabled = _enabled_stage_ids(stage_runs)
    if not skip_processed or not enabled:
        return enabled

    from nvitk.pipes.qvtpy.util.io.qc_report import check_subject_stages

    checks = check_subject_stages(
        subject,
        enabled,
        results_root=results_root,
        nifti_root=nifti_root,
    )
    complete = {c.stage for c in checks if c.complete}
    return [stage_id for stage_id in enabled if stage_id not in complete]


def count_sge_stages_per_subject(
    *,
    run_conv: bool,
    run_eicab: bool,
    run_s2: bool,
    run_s3: bool,
    run_s4: bool,
    run_s4t: bool,
    run_s5: bool,
    run_s6: bool,
    run_s7: bool,
) -> int:
    """Maximum number of array *tasks* (stages) per subject when all are enabled.

    Master SGE submit emits one array job per subject; this count is the task
    range size (``-t 1-N``), not the number of ``qsub`` jobs.
    """
    return len(
        _enabled_stage_ids(
            {
                "run_conv": run_conv,
                "run_eicab": run_eicab,
                "run_s2": run_s2,
                "run_s3": run_s3,
                "run_s4": run_s4,
                "run_s4t": run_s4t,
                "run_s5": run_s5,
                "run_s6": run_s6,
                "run_s7": run_s7,
            }
        )
    )


def count_sge_stages_for_subject(
    subject: str,
    *,
    stage_runs: dict[str, bool],
    skip_processed: bool,
    results_root: Path,
    nifti_root: Path,
) -> int:
    """Pending stage *tasks* for *subject* (respects ``--skip-processed``)."""
    return len(
        pending_sge_stage_ids(
            subject,
            stage_runs=stage_runs,
            skip_processed=skip_processed,
            results_root=results_root,
            nifti_root=nifti_root,
        )
    )


def stage_runs_from_emit_kwargs(kwargs: dict[str, Any]) -> dict[str, bool]:
    """Build the stage-run map used by :func:`pending_sge_stage_ids`."""
    return {
        "run_conv": bool(kwargs.get("run_conv")),
        "run_eicab": bool(kwargs.get("run_eicab")),
        "run_s2": bool(kwargs.get("run_s2")),
        "run_s3": bool(kwargs.get("run_s3")),
        "run_s4": bool(kwargs.get("run_s4")),
        "run_s4t": bool(kwargs.get("run_s4t")),
        "run_s5": bool(kwargs.get("run_s5")),
        "run_s6": bool(kwargs.get("run_s6")),
        "run_s7": bool(kwargs.get("run_s7")),
    }


def filter_subjects_pending_work(
    subjects: list[str],
    *,
    stage_runs: dict[str, bool],
    skip_processed: bool,
    results_root: Path,
    nifti_root: Path,
) -> tuple[list[str], list[str]]:
    """Split *subjects* into those with pending work vs already complete."""
    if not skip_processed:
        return list(subjects), []
    pending: list[str] = []
    skipped: list[str] = []
    for subj in subjects:
        if pending_sge_stage_ids(
            subj,
            stage_runs=stage_runs,
            skip_processed=True,
            results_root=results_root,
            nifti_root=nifti_root,
        ):
            pending.append(subj)
        else:
            skipped.append(subj)
    return pending, skipped


__all__ = [
    "STAGE_CENTERLINE",
    "STAGE_CONVERT",
    "STAGE_EICAB",
    "STAGE_LOC",
    "STAGE_MEASURE",
    "STAGE_MORPHOMETRICS",
    "STAGE_REG",
    "STAGE_SEG",
    "STAGE_SEG_T",
    "count_sge_stages_for_subject",
    "count_sge_stages_per_subject",
    "filter_subjects_pending_work",
    "pending_sge_stage_ids",
    "stage_runs_from_emit_kwargs",
]
