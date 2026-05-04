"""Single source of truth for the CT-PET v5 pipeline.

Everything a stage, runner or orchestrator needs to know about this pipeline
lives here: filesystem fragments (stage subdirectory names, expected NIfTI
inputs), TotalSegmentator task plan, SGE defaults and the Singularity
container / model paths used when the pipeline is submitted to a cluster.

Modifying the pipeline configuration (adding a new TS task, renaming a stage
output, changing SGE resources) should only require edits to this file and,
when new stage files are added, to :mod:`run` and :mod:`pyproject.toml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nvitk.pipes.pesa_fat.common.paths import DEFAULT_MODEL_ROOT
from nvitk.pipes.pesa_fat.ct_pet_v5.labels import (
    BODY_LABELS,
    FAT_LABELS,
    FAT_BATCH_LABELS,
    MO_LABELS,
    MUSCLES_LABELS,
    ORGANS_LABELS,
)


# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "ct-pet-v5"
SGE_JOB_PREFIX: str = "PESAFat_CTPET"


# ---------------------------------------------------------------------------
# Filesystem fragments (relative to BatchLayout.results_dir / nifti_dir)
# ---------------------------------------------------------------------------

STAGE1_DIR: str = "res_segmentation_ct"        # <subj>/CT/<task>.nii.gz
STAGE2_DIR: str = "res_post_processing_ct"     # <subj>/CT/{MO,FAT,BODY,ORGANS,MUSCLES}.nii.gz
STAGE3_DIR: str = "res_measure_suv"            # <batch>_SummaryCodebook.xlsx

# Per-subject CT input file stem under ``BatchLayout.nifti_dir / <subject>``.
INPUT_STEM: str = "CT"
PET_STEM: str = "PT"


# ---------------------------------------------------------------------------
# TotalSegmentator task plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CtTask:
    """A TotalSegmentator task with its optional ROI subset."""

    name: str
    roi_subset: tuple[str, ...] = ()


CT_TASKS: tuple[CtTask, ...] = (
    CtTask(
        "total",
        (
            "vertebrae_L4",
            "vertebrae_L3",
            "spleen",
            "kidney_right",
            "kidney_left",
            "liver",
            "pancreas",
            "autochthon_left",
            "autochthon_right",
            "small_bowel",
            "colon",
            "urinary_bladder",
        ),
    ),
    CtTask("tissue_types"),
    CtTask("thigh_shoulder_muscles"),
    CtTask("body"),
)


# ---------------------------------------------------------------------------
# Stage-3 measurement specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuvSpec:
    """SUV measurement spec.

    ``column_prefix`` defines the Excel column prefix, with one column per
    entry in the ``(suffix, stat)`` pairs in :data:`SUV_STATS`. ``mask_file``
    is the stage-2 label file name (e.g. ``"MUSCLES.nii"``). ``label_ids``
    holds the integer labels whose union forms the measurement mask.
    """

    column_prefix: str
    mask_file: str
    label_ids: tuple[int, ...]


@dataclass(frozen=True)
class VolSpec:
    """Volume measurement spec (single column)."""

    column: str
    mask_file: str
    label_ids: tuple[int, ...]


# (column suffix -> Measurer.suv stat name)
SUV_STATS: tuple[tuple[str, str], ...] = (
    ("SUVMAX", "max"),
    ("SUVmean", "mean"),
    ("SUV95p", "p95"),
    ("SUV99p", "p99"),
)


SUV_SPECS: tuple[SuvSpec, ...] = (
    # Vertebrae / MO
    SuvSpec("L3", "MO.nii", (MO_LABELS["L3"],)),
    SuvSpec("L4", "MO.nii", (MO_LABELS["L4"],)),
    SuvSpec("MO", "MO.nii", (MO_LABELS["L3"], MO_LABELS["L4"])),
    # Fat
    SuvSpec("GRASA_V", "FAT.nii", (FAT_LABELS["GRASA_V"],)),
    SuvSpec("GRASA_SC", "FAT.nii", (FAT_LABELS["GRASA_SC"],)),
    # Fat Batch
    SuvSpec("GRASA_V_BATCH", "FAT_BATCH.nii", (FAT_BATCH_LABELS["GRASA_V_BATCH"],)),
    SuvSpec("GRASA_SC_BATCH", "FAT_BATCH.nii", (FAT_BATCH_LABELS["GRASA_SC_BATCH"],)),
    # Organs
    SuvSpec("HIGADO", "ORGANS.nii", (ORGANS_LABELS["HIGADO"],)),
    SuvSpec("BAZO", "ORGANS.nii", (ORGANS_LABELS["BAZO"],)),
    SuvSpec("PANCREAS", "ORGANS.nii", (ORGANS_LABELS["PANCREAS"],)),
    # Muscles (hemisphere-split with bilateral union)
    SuvSpec("CUADRICEPS_L", "MUSCLES.nii", (MUSCLES_LABELS["CUADRICEPS_L"],)),
    SuvSpec("CUADRICEPS_R", "MUSCLES.nii", (MUSCLES_LABELS["CUADRICEPS_R"],)),
    SuvSpec(
        "CUADRICEPS_LR",
        "MUSCLES.nii",
        (MUSCLES_LABELS["CUADRICEPS_L"], MUSCLES_LABELS["CUADRICEPS_R"]),
    ),
    SuvSpec("PARAVERTEBRAL_L", "MUSCLES.nii", (MUSCLES_LABELS["PARAVERTEBRAL_L"],)),
    SuvSpec("PARAVERTEBRAL_R", "MUSCLES.nii", (MUSCLES_LABELS["PARAVERTEBRAL_R"],)),
    SuvSpec(
        "PARAVERTEBRAL_LR",
        "MUSCLES.nii",
        (MUSCLES_LABELS["PARAVERTEBRAL_L"], MUSCLES_LABELS["PARAVERTEBRAL_R"]),
    ),
    SuvSpec("DELTOIDES_L", "MUSCLES.nii", (MUSCLES_LABELS["DELTOIDES_L"],)),
    SuvSpec("DELTOIDES_R", "MUSCLES.nii", (MUSCLES_LABELS["DELTOIDES_R"],)),
    SuvSpec(
        "DELTOIDES_LR",
        "MUSCLES.nii",
        (MUSCLES_LABELS["DELTOIDES_L"], MUSCLES_LABELS["DELTOIDES_R"]),
    ),
    SuvSpec("TRAPECIOS", "MUSCLES.nii", (MUSCLES_LABELS["TRAPECIOS"],)),
)


VOL_SPECS: tuple[VolSpec, ...] = (
    VolSpec("GRASA_V_VOL", "FAT.nii", (FAT_LABELS["GRASA_V"],)),
    VolSpec("GRASA_SC_VOL", "FAT.nii", (FAT_LABELS["GRASA_SC"],)),
    VolSpec("GRASA_V_BATCH_VOL", "FAT_BATCH.nii", (FAT_BATCH_LABELS["GRASA_V_BATCH"],)),
    VolSpec("GRASA_SC_BATCH_VOL", "FAT_BATCH.nii", (FAT_BATCH_LABELS["GRASA_SC_BATCH"],)),
    VolSpec("CORP_VOL", "BODY.nii", (BODY_LABELS["BODY"],)),
    VolSpec("HIGADO_VOL", "ORGANS.nii", (ORGANS_LABELS["HIGADO"],)),
    VolSpec("BAZO_VOL", "ORGANS.nii", (ORGANS_LABELS["BAZO"],)),
    VolSpec("PANCREAS_VOL", "ORGANS.nii", (ORGANS_LABELS["PANCREAS"],)),
    VolSpec("CUADRICEPS_L_VOL", "MUSCLES.nii", (MUSCLES_LABELS["CUADRICEPS_L"],)),
    VolSpec("CUADRICEPS_R_VOL", "MUSCLES.nii", (MUSCLES_LABELS["CUADRICEPS_R"],)),
    VolSpec("PARAVERTEBRAL_L_VOL", "MUSCLES.nii", (MUSCLES_LABELS["PARAVERTEBRAL_L"],)),
    VolSpec("PARAVERTEBRAL_R_VOL", "MUSCLES.nii", (MUSCLES_LABELS["PARAVERTEBRAL_R"],)),
    VolSpec("DELTOIDES_L_VOL", "MUSCLES.nii", (MUSCLES_LABELS["DELTOIDES_L"],)),
    VolSpec("DELTOIDES_R_VOL", "MUSCLES.nii", (MUSCLES_LABELS["DELTOIDES_R"],)),
    VolSpec("TRAPECIOS_VOL", "MUSCLES.nii", (MUSCLES_LABELS["TRAPECIOS"],)),
    VolSpec("MO_L3_VOL", "MO.nii", (MO_LABELS["L3"],)),
    VolSpec("MO_L4_VOL", "MO.nii", (MO_LABELS["L4"],)),
)


# ---------------------------------------------------------------------------
# SGE / Singularity defaults
# ---------------------------------------------------------------------------

SGE_PROJECT: str = "GPU"
SGE_ACCOUNT: str = "Prod"
SGE_NGPU: int = 1
SGE_H_VMEM: str = "50G"
SGE_QUEUE: str | None = None

# CPU-only stages (0, 2, 3) relax GPU requirements.
SGE_CPU_H_VMEM: str = "32G"
SGE_CPU_NGPU: int = 0

SGE_LOG_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/logs/PESAFatV5")
SGE_ERR_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/errs/PESAFatV5")

CONTAINER_PATH: Path = Path("/data3/BIOIT_IMAGE/Containers/nvitk_v2026.04.21.sif")
MODELS_PATH: Path = DEFAULT_MODEL_ROOT


__all__ = [
    "PIPELINE_NAME",
    "SGE_JOB_PREFIX",
    "STAGE1_DIR",
    "STAGE2_DIR",
    "STAGE3_DIR",
    "INPUT_STEM",
    "PET_STEM",
    "CtTask",
    "CT_TASKS",
    "SuvSpec",
    "VolSpec",
    "SUV_STATS",
    "SUV_SPECS",
    "VOL_SPECS",
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
