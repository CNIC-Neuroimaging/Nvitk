"""qvtpy configuration (stage0: DICOM -> NIfTI + reorg).

This is intentionally minimal for now. The goal is to mirror the PESA-Fat
pipeline structure: one config module as a single source of truth for paths,
SGE defaults, and stage directory names.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "qvtpy"
SGE_JOB_PREFIX: str = "QVTPY"


# ---------------------------------------------------------------------------
# Roots (host-side)
# ---------------------------------------------------------------------------

# TODO: update defaults to your desired dataset roots.
DEFAULT_DICOM_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/DICOM")
DEFAULT_NIFTI_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/NIFTI")
DEFAULT_RESULTS_ROOT = Path("/data3/BIOIT_IMAGE/PESA-Brain/RESULTS")


# ---------------------------------------------------------------------------
# Stage directory fragments
# ---------------------------------------------------------------------------

STAGE0_DIR: str = "res_convert_qvtpy"

# eICAB (stage1): outputs live under ``{DEFAULT_RESULTS_ROOT}/{subject}/{STAGE1_EICAB_DIR}/``.
STAGE1_EICAB_DIR: str = "eicab"


# ---------------------------------------------------------------------------
# SGE defaults (placeholder; adapt later)
# ---------------------------------------------------------------------------

SGE_PROJECT: str = "MCC_GPU"
SGE_ACCOUNT: str = "MCC_GPU"
SGE_CPU_H_VMEM: str = "16G"
SGE_CPU_NGPU: int = 0
SGE_QUEUE: str | None = None

SGE_LOG_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/logs/QVTPY")
SGE_ERR_DIR: Path = Path("/data3/BIOIT_IMAGE/BioImaging/env/errs/QVTPY")

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

