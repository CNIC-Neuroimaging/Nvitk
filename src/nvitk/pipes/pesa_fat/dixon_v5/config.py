"""Single source of truth for the Dixon v5 pipeline.

Analogous to :mod:`nvitk.pipes.pesa_fat.ct_pet_v5.config` but for the MR
Dixon pipeline. Centralises per-region TotalSegmentator task lists, output
directory fragments, SGE defaults and Singularity container / model paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nvitk.pipes.pesa_fat.common.paths import DEFAULT_MODEL_ROOT
from nvitk.pipes.pesa_fat.dixon_v5.labels import (
    HEAD_LABELS,
    LEGS_LABELS,
    THORAX_LABELS,
)


# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "dixon-v5"
SGE_JOB_PREFIX: str = "PESAFat_DIXON"


# ---------------------------------------------------------------------------
# Filesystem fragments
# ---------------------------------------------------------------------------

STAGE1_DIR: str = "res_segmentation_dixon"     # <subj>/DIXON_<REGION>/<task>.nii.gz
STAGE2_DIR: str = "res_post_processing_dixon"  # <subj>/{HEAD,THORAX,LEGS}.nii.gz
STAGE3_DIR: str = "res_measure_dixon"          # <batch>_SummaryCodebook.xlsx

# Expected Dixon input stems: DIXON_<REGION>_<SUFFIX>.nii[.gz]
INPUT_PREFIX: str = "DIXON"
INPUT_SUFFIXES: tuple[str, ...] = ("FAT", "WATER", "FAT_FRACTION", "T2STAR")


# ---------------------------------------------------------------------------
# TotalSegmentator task plan (per Dixon region)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MrTask:
    """A MR TotalSegmentator task and which Dixon contrast feeds it."""

    name: str
    input_suffix: str                # FAT / WATER / ...
    roi_subset: tuple[str, ...] = ()


HEAD_TASKS: tuple[MrTask, ...] = (
    MrTask("total_mr", input_suffix="FAT", roi_subset=("autochthon_left", "autochthon_right")),
)

THORAX_TASKS: tuple[MrTask, ...] = (
    MrTask(
        "total_mr",
        input_suffix="FAT",
        roi_subset=(
            "spleen",
            "kidney_right",
            "kidney_left",
            "liver",
            "pancreas",
            "autochthon_left",
            "autochthon_right",
        ),
    ),
    MrTask("tissue_types_mr", input_suffix="FAT"),
    MrTask("body_mr", input_suffix="FAT"),
    MrTask("vertebrae_mr", input_suffix="WATER"),
)

LEGS_TASKS: tuple[MrTask, ...] = (
    MrTask("thigh_shoulder_muscles_mr", input_suffix="FAT"),
    MrTask("tissue_types_mr", input_suffix="FAT"),
    MrTask("body_mr", input_suffix="FAT"),
)


REGIONS: dict[str, tuple[MrTask, ...]] = {
    "HEAD": HEAD_TASKS,
    "THORAX": THORAX_TASKS,
    "LEGS": LEGS_TASKS,
}

REGION_ORDER: tuple[str, ...] = ("HEAD", "THORAX", "LEGS")


# ---------------------------------------------------------------------------
# Stage-3 measurement specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DixonMeasureSpec:
    """One Dixon stage-3 measurement column group.

    ``prefix`` is the Excel column prefix; each metric in ``metrics`` becomes
    a column ``f"{prefix}_{metric}"``. ``region`` selects which stage-2 mask
    and which Dixon-contrast maps are used; ``mask_file`` is the stage-2
    label file (under ``RESULTS/<batch>/res_post_processing_dixon/<SUBJECT>``)
    and ``label_ids`` are the integer labels contributing to the mask (a
    tuple with >1 entries is treated as the union).
    """

    prefix: str
    region: str                    # "HEAD" | "THORAX" | "LEGS"
    mask_file: str                 # "HEAD.nii" | "THORAX.nii" | "LEGS.nii"
    label_ids: tuple[int, ...]
    metrics: tuple[str, ...] = ("VOL", "FF", "T2", "R2")


def _lr_triplet(
    prefix_fmt: str,
    region: str,
    mask_file: str,
    labels: dict[str, int],
    key_l: str,
    key_r: str,
    *,
    metrics: tuple[str, ...] = ("VOL", "FF", "T2", "R2"),
) -> tuple[DixonMeasureSpec, ...]:
    """Emit (L, R, LR-union) specs for a bilateral structure."""
    return (
        DixonMeasureSpec(
            prefix_fmt.format(side="L"),
            region,
            mask_file,
            (labels[key_l],),
            metrics=metrics,
        ),
        DixonMeasureSpec(
            prefix_fmt.format(side="R"),
            region,
            mask_file,
            (labels[key_r],),
            metrics=metrics,
        ),
        DixonMeasureSpec(
            prefix_fmt.format(side="LR"),
            region,
            mask_file,
            (labels[key_l], labels[key_r]),
            metrics=metrics,
        ),
    )


MEASURE_SPECS: tuple[DixonMeasureSpec, ...] = (
    # HEAD: paravertebral (PVM) L, R, LR
    *_lr_triplet(
        "DIXON_H_PVM_{side}", "HEAD", "HEAD.nii", HEAD_LABELS, "H_PVM_L", "H_PVM_R",
    ),

    # THORAX: liver (with WF), pancreas, kidneys L/R/LR, paravertebral L/R/LR,
    #         and bone-narrow vertebrae L3/L4.
    DixonMeasureSpec(
        "DIXON_LIVER",
        "THORAX",
        "THORAX.nii",
        (THORAX_LABELS["LIVER"],),
        metrics=("VOL", "FF", "T2", "R2", "WF"),
    ),
    DixonMeasureSpec(
        "DIXON_PANCREAS",
        "THORAX",
        "THORAX.nii",
        (THORAX_LABELS["PANCREAS"],),
    ),
    *_lr_triplet(
        "DIXON_KIDNEY_{side}",
        "THORAX",
        "THORAX.nii",
        THORAX_LABELS,
        "KIDNEY_L",
        "KIDNEY_R",
    ),
    *_lr_triplet(
        "DIXON_T_PVM_{side}",
        "THORAX",
        "THORAX.nii",
        THORAX_LABELS,
        "T_PVM_L",
        "T_PVM_R",
    ),
    DixonMeasureSpec(
        "DIXON_BN_L3", "THORAX", "THORAX.nii", (THORAX_LABELS["BN_L3"],),
    ),
    DixonMeasureSpec(
        "DIXON_BN_L4", "THORAX", "THORAX.nii", (THORAX_LABELS["BN_L4"],),
    ),

    # LEGS: quadriceps (QM) L, R, LR
    *_lr_triplet(
        "DIXON_L_QM_{side}",
        "LEGS",
        "LEGS.nii",
        LEGS_LABELS,
        "L_QM_L",
        "L_QM_R",
    ),
)


# ``WF`` is computed relative to the mean water-map signal over this union
# of paravertebral muscle (thorax) + quadriceps (legs) labels. Defined here
# (as anatomic references) so stage 3 has a single source of truth.
WF_REFERENCE_LABELS: tuple[tuple[str, str, str], ...] = (
    # (region, mask_file, label_key)
    ("THORAX", "THORAX.nii", "T_PVM_L"),
    ("THORAX", "THORAX.nii", "T_PVM_R"),
    ("LEGS",   "LEGS.nii",   "L_QM_L"),
    ("LEGS",   "LEGS.nii",   "L_QM_R"),
)


# ---------------------------------------------------------------------------
# SGE / Singularity defaults
# ---------------------------------------------------------------------------

SGE_PROJECT: str = "GPU"
SGE_ACCOUNT: str = "Prod"
SGE_NGPU: int = 1
SGE_H_VMEM: str = "50G"
SGE_QUEUE: str | None = None

SGE_CPU_H_VMEM: str = "32G"
SGE_CPU_NGPU: int = 0

SGE_LOG_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/logs/PESAFat")
SGE_ERR_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/errs/PESAFat")

CONTAINER_PATH: Path = Path(
    "/data3/BIOIT_IMAGE/Containers/gpu-pesa-fat_v2025.5.27.sif"
)
MODELS_PATH: Path = DEFAULT_MODEL_ROOT


__all__ = [
    "PIPELINE_NAME",
    "SGE_JOB_PREFIX",
    "STAGE1_DIR",
    "STAGE2_DIR",
    "STAGE3_DIR",
    "INPUT_PREFIX",
    "INPUT_SUFFIXES",
    "MrTask",
    "HEAD_TASKS",
    "THORAX_TASKS",
    "LEGS_TASKS",
    "REGIONS",
    "REGION_ORDER",
    "DixonMeasureSpec",
    "MEASURE_SPECS",
    "WF_REFERENCE_LABELS",
    "SGE_PROJECT",
    "SGE_ACCOUNT",
    "SGE_NGPU",
    "SGE_H_VMEM",
    "SGE_QUEUE",
    "SGE_CPU_H_VMEM",
    "SGE_CPU_NGPU",
    "SGE_LOG_DIR",
    "SGE_ERR_DIR",
    "CONTAINER_PATH",
    "MODELS_PATH",
]
