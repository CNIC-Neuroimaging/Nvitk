"""Default paths and layout constants for the black-blood (vwi_bb) pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import lazy_config

# ──────────────────────────────────────────────────────────────────────────────
# Host data roots (bbtpy defaults; no qvtpy import)
# ──────────────────────────────────────────────────────────────────────────────

#: ``sge.json`` section holding this pipeline's data roots.
PIPELINE_PATHS_ID = "bbtpy_paths"


def _pipe_paths() -> dict:
    """The ``pipelines.bbtpy_paths`` block of ``sge.json`` (empty when unset)."""
    return _sj.pipeline_section(PIPELINE_PATHS_ID)


def _opt_root(key: str):
    """A configured root as a path, or ``None``.

    ``None`` rather than an error: these are read at import time as Click option defaults, so
    an unconfigured machine must still be able to print ``--help``. ``run.py`` reports the
    missing root by name when it actually needs one.
    """
    raw = _pipe_paths().get(key)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


_RESOLVERS: dict[str, lazy_config.Resolver] = {
    "DEFAULT_DICOM_ROOT": lambda: _opt_root("dicom_root"),
    "DEFAULT_NIFTI_ROOT": lambda: _opt_root("nifti_root"),
    "DEFAULT_RESULTS_ROOT": lambda: _opt_root("results_root"),
    "DEFAULT_QVTPY_NIFTI_ROOT": lambda: _opt_root("qvtpy_nifti_root") or _opt_root("nifti_root"),
    "DEFAULT_QVTPY_RESULTS_ROOT": lambda: _opt_root("qvtpy_results_root"),
    "DEFAULT_EICAB_RESULTS_ROOT": lambda: (
        _opt_root("eicab_results_root") or _opt_root("qvtpy_results_root")
    ),
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)

# ---- Relative NIfTI paths and stage directories ------------------------------

VWI_BB_REL_PATH: str | None = "BlackBlood/vwi_bb.nii.gz"

STAGE0_DICOM_SLOT: str = "vwi_bb"
EICAB_SUBDIR: str = "eicab"
QVTPY_EICAB_SUBDIR: str = EICAB_SUBDIR
PIPELINE_SUBDIR: str = "bbtpy"
STAGE1_REG_DIR: str = "stage1_registration"
STAGE2_SEG_DIR: str = "stage2_bb_segmentation"

# ──────────────────────────────────────────────────────────────────────────────
# Batch / container (optional overrides)
# ──────────────────────────────────────────────────────────────────────────────

SGE_LOG_DIR: Path | None = None
SGE_ERR_DIR: Path | None = None
CONTAINER_PATH: Path | None = None
