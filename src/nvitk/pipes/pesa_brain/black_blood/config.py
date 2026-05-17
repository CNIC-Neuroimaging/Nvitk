"""Black-blood pipeline defaults."""

from __future__ import annotations

from pathlib import Path

# Host paths (aligned with qvtpy layout; no qvtpy import).
_PESA_BRAIN_DATA = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain")

DEFAULT_DICOM_ROOT: Path = _PESA_BRAIN_DATA / "DATA" / "DICOM"
DEFAULT_NIFTI_ROOT: Path = _PESA_BRAIN_DATA / "DATA" / "NIFTI"
DEFAULT_RESULTS_ROOT: Path | None = _PESA_BRAIN_DATA / "RESULTS" / "res_PESABrain"

# qvtpy TOF / eICAB (notebooks/reports/qvtpy methods layout).
DEFAULT_QVTPY_NIFTI_ROOT: Path = _PESA_BRAIN_DATA / "DATA" / "NIFTI"
DEFAULT_QVTPY_RESULTS_ROOT: Path = _PESA_BRAIN_DATA / "RESULTS" / "res_QVTPy"

DEFAULT_EICAB_RESULTS_ROOT: Path | None = DEFAULT_QVTPY_RESULTS_ROOT

VWI_BB_REL_PATH: str | None = "BlackBlood/vwi_bb.nii.gz"

STAGE0_DICOM_SLOT: str = "vwi_bb"
EICAB_SUBDIR: str = "eicab"
QVTPY_EICAB_SUBDIR: str = EICAB_SUBDIR
PIPELINE_SUBDIR: str = "pesa_brain"
BLACK_BLOOD_SUBDIR: str = "black_blood"
STAGE1_REG_DIR: str = "stage1_registration"
STAGE2_SEG_DIR: str = "stage2_bb_segmentation"

SGE_LOG_DIR: Path | None = None
SGE_ERR_DIR: Path | None = None
CONTAINER_PATH: Path | None = None
