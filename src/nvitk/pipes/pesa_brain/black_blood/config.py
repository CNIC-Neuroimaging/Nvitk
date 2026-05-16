"""Black-blood pipeline defaults (fill paths before batch use)."""

from __future__ import annotations

from pathlib import Path

# User-filled roots (None until configured).
DEFAULT_NIFTI_ROOT: Path | None = None
DEFAULT_RESULTS_ROOT: Path | None = None
DEFAULT_EICAB_RESULTS_ROOT: Path | None = None

# Relative path under ``{nifti_root}/{subject}/`` for WVI (e.g. ``BlackBlood/WVI.nii.gz``).
WVI_REL_PATH: str | None = None

EICAB_SUBDIR: str = "eicab"
PIPELINE_SUBDIR: str = "pesa_brain"
BLACK_BLOOD_SUBDIR: str = "black_blood"
STAGE1_REG_DIR: str = "stage1_registration"
STAGE2_SEG_DIR: str = "stage2_bb_segmentation"

# Optional cluster placeholders (unused in v1 local runner).
SGE_LOG_DIR: Path | None = None
SGE_ERR_DIR: Path | None = None
CONTAINER_PATH: Path | None = None
