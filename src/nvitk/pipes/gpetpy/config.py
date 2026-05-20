"""gPET pipeline defaults (not implemented)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_NIFTI_ROOT: Path | None = None
DEFAULT_RESULTS_ROOT: Path | None = None
PIPELINE_SUBDIR: str = "gpetpy"

SGE_LOG_DIR: Path | None = None
SGE_ERR_DIR: Path | None = None
CONTAINER_PATH: Path | None = None
