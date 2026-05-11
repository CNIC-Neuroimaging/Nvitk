"""
DIXON v5 configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.pipes.pesa_fat.common.paths import DEFAULT_MODEL_ROOT
from nvitk.pipes.pesa_fat.dixon_v5.labels import (
    HEAD_LABELS,
    LEGS_LABELS,
    THORAX_LABELS,
)


# ---------------------------------------------------------------------------
# Pipeline identity (Just naming, nothing important)
# ---------------------------------------------------------------------------

PIPELINE_NAME = "dixon-v5"
SGE_JOB_PREFIX = "PESAFat_DIXON"


# ---------------------------------------------------------------------------
# Filesystem fragments
# ---------------------------------------------------------------------------

STAGE1_DIR = "res_segmentation_dixon"     # <subj>/DIXON_<REGION>/<task>.nii.gz
STAGE2_DIR = "res_post_processing_dixon"  # <subj>/{HEAD,THORAX,LEGS}.nii.gz
STAGE3_DIR = "res_measure_dixon"          # <batch>_SummaryCodebook.xlsx

# Expected Dixon input stems: DIXON_<REGION>_<SUFFIX>.nii[.gz]
INPUT_PREFIX = "DIXON"
INPUT_SUFFIXES = ("FAT", "WATER", "FAT_FRACTION", "T2STAR")


# ---------------------------------------------------------------------------
# TotalSegmentator task plan (per Dixon region)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MrTask:
    """MR TotalSegmentator task and which DIXON sequence it uses."""

    name: str
    input_suffix: str                # FAT / WATER / ...
    roi_subset: tuple[str, ...] = ()


HEAD_TASKS = (
    MrTask("total_mr", input_suffix="FAT", roi_subset=("autochthon_left", "autochthon_right")),
)

THORAX_TASKS = (
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

LEGS_TASKS = (
    MrTask("thigh_shoulder_muscles_mr", input_suffix="FAT"),
    MrTask("tissue_types_mr", input_suffix="FAT"),
    MrTask("body_mr", input_suffix="FAT"),
)


REGIONS = {
    "HEAD": HEAD_TASKS,
    "THORAX": THORAX_TASKS,
    "LEGS": LEGS_TASKS,
}

REGION_ORDER = ("HEAD", "THORAX", "LEGS")


# ---------------------------------------------------------------------------
# Stage-3 measurement specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DixonMeasureSpec:
    """DIXON stage-3 measurement definition.

    ``prefix`` is the Excel column prefix; each metric in ``metrics`` becomes
      a column ``f"{prefix}_{metric}"``. 
    ``region`` selects which stage-2 mask and which Dixon-contrast maps are used; 
    ``mask_file`` is the stage-2 label file 
      (under ``RESULTS/<batch>/res_post_processing_dixon/<SUBJECT>``)
    ``label_ids`` are the integer labels contributing to the mask 
      (atuple with >1 entries is treated as the union).
    """

    prefix: str
    region: str                    # "HEAD" | "THORAX" | "LEGS"
    mask_file: str                 # "HEAD.nii" | "THORAX.nii" | "LEGS.nii"
    label_ids: tuple[int, ...]
    metrics: tuple[str, ...] = ("VOL", "FF", "T2", "R2", "NSlices")


def _lr_triplet(
    prefix_fmt: str,
    region: str,
    mask_file: str,
    labels: dict[str, int],
    key_l: str,
    key_r: str,
    *,
    metrics: tuple[str, ...] = ("VOL", "FF", "T2", "R2", "NSlices"),
) -> tuple[DixonMeasureSpec, ...]:
    """Emit (L, R, LR-union) specs for a bilateral structures."""
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


MEASURE_SPECS = (
    # # HEAD: paravertebral (PVM) L, R, LR
    # *_lr_triplet(
    #     "DIXON_H_PVM_{side}", "HEAD", "HEAD.nii", HEAD_LABELS, "H_PVM_L", "H_PVM_R",
    # ),

    # THORAX: liver (with WF), pancreas, kidneys L/R/LR, paravertebral L/R/LR,
    #         and bone-narrow vertebrae L3/L4.
    DixonMeasureSpec(
        "DIXON_LIVER",
        "THORAX",
        "THORAX.nii",
        (THORAX_LABELS["LIVER"],),
        metrics=("VOL", "FF", "T2", "R2", "WF", "NSlices"),
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


# ``WF`` is computed (for LIVER only) as:
#
#     WF = mean(water_map_liver) / mean(water_map_PVM) * 100
#
# where the denominator is the mean water-map signal pooled over both
# paravertebral muscle masks (T_PVM_L + T_PVM_R) in the THORAX region.
WF_REFERENCE_LABELS = (
    # (region, mask_file, label_key)
    ("THORAX", "THORAX.nii", "T_PVM_L"),
    ("THORAX", "THORAX.nii", "T_PVM_R"),
)


# ---------------------------------------------------------------------------
# SGE / Singularity defaults
# ---------------------------------------------------------------------------

SGE_PROJECT     = "GPU"
SGE_ACCOUNT     = "Prod"
SGE_NGPU        = 1
SGE_H_VMEM      = "50G"
SGE_QUEUE       = None
 
SGE_CPU_H_VMEM  = "32G"
SGE_CPU_NGPU    = 0
 
SGE_LOG_DIR     = Path("/data3/BIOIT_IMAGE/BioImaging/env/logs/PESAFatV5")
SGE_ERR_DIR     = Path("/data3/BIOIT_IMAGE/BioImaging/env/errs/PESAFatV5")

CONTAINER_PATH  = Path("/data3/BIOIT_IMAGE/Containers/gpu-pesa-fat_v2025.5.27.sif")
MODELS_PATH     = DEFAULT_MODEL_ROOT

_pipe_dx = _sj.merged_pipeline_flat("pesa_fat_dixon")
_paths_dx = _sj.paths_section()
if (v := _pipe_dx.get("sge_project")) is not None:
    SGE_PROJECT = str(v)
if (v := _pipe_dx.get("sge_account")) is not None:
    SGE_ACCOUNT = str(v)
if (v := _pipe_dx.get("sge_ngpu")) is not None:
    SGE_NGPU = int(v)
if (v := _pipe_dx.get("sge_h_vmem")) is not None:
    SGE_H_VMEM = str(v)
if "sge_queue" in _pipe_dx:
    SGE_QUEUE = _pipe_dx["sge_queue"]
if (v := _pipe_dx.get("sge_cpu_h_vmem")) is not None:
    SGE_CPU_H_VMEM = str(v)
if (v := _pipe_dx.get("sge_cpu_ngpu")) is not None:
    SGE_CPU_NGPU = int(v)
_lg_dx, _er_dx = _sj.resolve_log_err_dirs(
    paths=_paths_dx,
    pipe=_pipe_dx,
    fallback_log=SGE_LOG_DIR,
    fallback_err=SGE_ERR_DIR,
)
SGE_LOG_DIR, SGE_ERR_DIR = _lg_dx, _er_dx
if (v := _pipe_dx.get("default_sge_container_root") or _pipe_dx.get("container_path")):
    CONTAINER_PATH = Path(os.path.expanduser(str(v)))
if (v := _pipe_dx.get("default_sge_model_root") or _pipe_dx.get("models_path")):
    MODELS_PATH = Path(os.path.expanduser(str(v)))


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
