"""qvtpy configuration (stage0: DICOM -> NIfTI + reorg).

This is intentionally minimal for now. The goal is to mirror the PESA-Fat
pipeline structure: one config module as a single source of truth for paths,
SGE defaults, and stage directory names.
"""

from __future__ import annotations

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

DEFAULT_DICOM_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/DICOM")
DEFAULT_NIFTI_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/NIFTI")
DEFAULT_RESULTS_ROOT = Path("/data3/BIOIT_IMAGE/PESA-Brain/RESULTS")


# ---------------------------------------------------------------------------
# Stage directory fragments
# ---------------------------------------------------------------------------

STAGE0_DIR: str = "res_convert_qvtpy"
STAGE1_EICAB_DIR: str = "eicab"


# ---------------------------------------------------------------------------
# SGE defaults (placeholder; adapt later)
# ---------------------------------------------------------------------------

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
if (v := _pipe.get("sge_queue")) is not None:
    SGE_QUEUE = str(v)

_lg_qvt, _er_qvt = _sj.resolve_log_err_dirs(
    paths=_paths,
    pipe=_pipe,
    fallback_log=SGE_LOG_DIR,
    fallback_err=SGE_ERR_DIR,
)
SGE_LOG_DIR, SGE_ERR_DIR = _lg_qvt, _er_qvt
if (v := _pipe.get("default_sge_scripts_dir")):
    SGE_SCRIPTS_DIR = Path(os.path.expanduser(str(v)))

CONTAINER_PATH: Path = Path("/data3/BIOIT_IMAGE/Containers/nvitk_v2026.04.21.sif")


__all__ = [
    "PIPELINE_NAME",
    "SGE_JOB_PREFIX",
    "DEFAULT_DICOM_ROOT",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "STAGE0_DIR",
    "STAGE1_EICAB_DIR",
    "SGE_PROJECT",
    "SGE_ACCOUNT",
    "SGE_CPU_H_VMEM",
    "SGE_CPU_NGPU",
    "SGE_QUEUE",
    "SGE_LOG_DIR",
    "SGE_ERR_DIR",
    "CONTAINER_PATH",
]

