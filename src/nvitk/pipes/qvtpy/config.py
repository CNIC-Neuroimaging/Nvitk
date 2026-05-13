"""qvtpy configuration (stage0: DICOM -> NIfTI + reorg; stage1: eICAB).

Single source of truth for host-side paths, stage directory names, SGE
defaults, and the pipeline Singularity image. Optional overrides come from
``.nvitk/sge.json`` (see :mod:`nvitk.cluster.sge_json`) under the ``qvtpy``
pipeline key.
"""

from __future__ import annotations

import os
from pathlib import Path

from nvitk.cluster import sge_json as _sj


# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "qvtpy"
SGE_JOB_PREFIX: str = "QVTPY"


# ---------------------------------------------------------------------------
# Roots (host-side)
# ---------------------------------------------------------------------------

DEFAULT_DICOM_ROOT   = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/DICOM")
DEFAULT_NIFTI_ROOT   = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/NIFTI")
DEFAULT_RESULTS_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/RESULTS/res_QVTPy")


# ---------------------------------------------------------------------------
# Stage directory fragments
# ---------------------------------------------------------------------------

STAGE0_DIR: str = "res_convert_qvtpy"
STAGE1_EICAB_DIR: str = "eicab"
QVT_SUBDIR: str = "qvtpy"
STAGE2_REGISTRATION_DIR: str = "stage2_registration"
STAGE3_CENTERLINE_DIR: str = "stage3_centerline"
STAGE4_SEG_DIR: str = "stage4_4dflow_segmentation"
STAGE5_LOC_DIR: str = "stage5_loc_generation"
STAGE6_MEASURE_DIR: str = "stage6_measure"


# ---------------------------------------------------------------------------
# SGE defaults (overridable via .nvitk/sge.json `pipelines.qvtpy`)
# ---------------------------------------------------------------------------

SGE_PROJECT: str = "MCC_GPU"
SGE_ACCOUNT: str = "MCC_GPU"
SGE_NGPU: int = 0
SGE_H_VMEM: str = "30G"
SGE_QUEUE: str | None = None

SGE_LOG_DIR: Path = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/logs/QVTPY")
SGE_ERR_DIR: Path = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/errs/QVTPY")

# Default location for emitted submission scripts (parallels eicab.config).
SGE_SCRIPTS_DIR: Path = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/")

CONTAINER_PATH: Path = Path("/data3/BIOIT_IMAGE/Containers/nvitk_v2026.04.21.sif")


_pipe = _sj.merged_pipeline_flat("qvtpy")
_paths = _sj.paths_section()
if (v := _pipe.get("sge_project")) is not None:
    SGE_PROJECT = str(v)
if (v := _pipe.get("sge_account")) is not None:
    SGE_ACCOUNT = str(v)
if (v := _pipe.get("sge_ngpu")) is not None:
    SGE_NGPU = int(v)
if (v := _pipe.get("sge_h_vmem")) is not None:
    SGE_H_VMEM = str(v)
if "sge_queue" in _pipe:
    SGE_QUEUE = _pipe["sge_queue"]

_lg_qvt, _er_qvt = _sj.resolve_log_err_dirs(
    paths=_paths,
    pipe=_pipe,
    fallback_log=SGE_LOG_DIR,
    fallback_err=SGE_ERR_DIR,
)
SGE_LOG_DIR, SGE_ERR_DIR = _lg_qvt, _er_qvt

if (v := _pipe.get("default_sge_scripts_dir")):
    SGE_SCRIPTS_DIR = Path(os.path.expanduser(str(v)))
if (v := _pipe.get("container_path")):
    CONTAINER_PATH = Path(os.path.expanduser(str(v)))


__all__ = [
    "PIPELINE_NAME",
    "SGE_JOB_PREFIX",
    "DEFAULT_DICOM_ROOT",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "STAGE0_DIR",
    "STAGE1_EICAB_DIR",
    "QVT_SUBDIR",
    "STAGE2_REGISTRATION_DIR",
    "STAGE3_CENTERLINE_DIR",
    "STAGE4_SEG_DIR",
    "STAGE5_LOC_DIR",
    "STAGE6_MEASURE_DIR",
    "SGE_PROJECT",
    "SGE_ACCOUNT",
    "SGE_NGPU",
    "SGE_H_VMEM",
    "SGE_QUEUE",
    "SGE_LOG_DIR",
    "SGE_ERR_DIR",
    "SGE_SCRIPTS_DIR",
    "CONTAINER_PATH",
]
