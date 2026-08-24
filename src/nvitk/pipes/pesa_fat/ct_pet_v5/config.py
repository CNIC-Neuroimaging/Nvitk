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

import os
from dataclasses import dataclass
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths
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

SGE_CPU_H_VMEM: str = "32G"
SGE_CPU_NGPU: int = 0

SGE_LOG_DIR: Path = Path(tempfile.gettempdir()) / "nvitk-sge" / "logs" / "PESAFat-CT-PET"
SGE_ERR_DIR: Path = Path(tempfile.gettempdir()) / "nvitk-sge" / "errs" / "PESAFat-CT-PET"

CONTAINER_PATH: Path | None = None  # sge.json: paths.nvitk_container
MODELS_PATH: Path = DEFAULT_MODEL_ROOT

def _apply_config() -> None:
    """Merge ``sge.json`` over this module's defaults.

    Run once at import and again whenever the configuration directory is redirected,
    so a late ``--config-dir`` reaches these constants too.
    """
    global CONTAINER_PATH, SGE_ACCOUNT, SGE_CPU_H_VMEM, SGE_CPU_NGPU, SGE_ERR_DIR, SGE_H_VMEM, SGE_LOG_DIR, SGE_NGPU, SGE_PROJECT, SGE_QUEUE, _er_ct, _lg_ct, _paths_ct, _pipe_ct
    _pipe_ct = _sj.merged_pipeline_flat("pesa_fat_ct_pet")
    _paths_ct = _sj.paths_section()
    if (v := _pipe_ct.get("sge_project")) is not None:
        SGE_PROJECT = str(v)
    if (v := _pipe_ct.get("sge_account")) is not None:
        SGE_ACCOUNT = str(v)
    if (v := _pipe_ct.get("sge_ngpu")) is not None:
        SGE_NGPU = int(v)
    if (v := _pipe_ct.get("sge_h_vmem")) is not None:
        SGE_H_VMEM = str(v)
    if "sge_queue" in _pipe_ct:
        SGE_QUEUE = _pipe_ct["sge_queue"]
    if (v := _pipe_ct.get("sge_cpu_h_vmem")) is not None:
        SGE_CPU_H_VMEM = str(v)
    if (v := _pipe_ct.get("sge_cpu_ngpu")) is not None:
        SGE_CPU_NGPU = int(v)
    _lg_ct, _er_ct = _sj.resolve_log_err_dirs(
        paths=_paths_ct,
        pipe=_pipe_ct,
        fallback_log=SGE_LOG_DIR,
        fallback_err=SGE_ERR_DIR,
    )
    SGE_LOG_DIR, SGE_ERR_DIR = _lg_ct, _er_ct
    CONTAINER_PATH = _sj.resolve_nvitk_container(pipe=_pipe_ct, fallback=CONTAINER_PATH)


_apply_config()
config_paths.register_reload_hook(_apply_config)
if (v := _pipe_ct.get("default_sge_model_root") or _pipe_ct.get("models_path")):
    MODELS_PATH = Path(os.path.expanduser(str(v)))


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
