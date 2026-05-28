"""gPET pipeline defaults.

The defaults mirror PESA-Fat roots so `gpetpy` can run on the same workstation /
filesystem without extra config.
"""

from __future__ import annotations

from pathlib import Path

from .layout import (
    DEFAULT_DICOM_ROOT as _DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT as _DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT as _DEFAULT_RESULTS_ROOT,
)

DEFAULT_DICOM_ROOT: Path = _DEFAULT_DICOM_ROOT
DEFAULT_NIFTI_ROOT: Path = _DEFAULT_NIFTI_ROOT
DEFAULT_RESULTS_ROOT: Path = _DEFAULT_RESULTS_ROOT
PIPELINE_SUBDIR: str = "gpetpy"

SGE_LOG_DIR: Path | None = None
SGE_ERR_DIR: Path | None = None
CONTAINER_PATH: Path | None = None
