"""qvtpy host configuration: data roots, stage paths, and SGE defaults.

**Inputs (environment / JSON)**

- Default DICOM, NIfTI, and results roots on the analysis host.
- Optional ``.nvitk/sge.json`` ``pipelines.qvtpy`` block (see :mod:`nvitk.cluster.sge_json`).

**Outputs (constants consumed by stages)**

- ``STAGE*_DIR`` — subdirectory names under ``<results>/<subject>/qvtpy/``.
- ``SGE_*`` — project, queue, log/err dirs, and ``CONTAINER_PATH`` for cluster jobs.

Stage 0 conversion and stage 1 eICAB use these paths; later stages read/write under
``DEFAULT_RESULTS_ROOT`` / ``DEFAULT_NIFTI_ROOT`` unless overridden on the CLI.

Every value here is read from ``sge.json`` on first use — there are no installation paths
in this file, and nothing is resolved at import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import lazy_config
from nvitk.pipes.qvtpy.util.io import paths as _paths_mod


# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "qvtpy"
SGE_JOB_PREFIX: str = "QVTPY"


# ---------------------------------------------------------------------------
# Roots (cluster defaults; see util.paths for local vs cluster layout)
# ---------------------------------------------------------------------------

from nvitk.pipes.qvtpy.util.io.paths import resolve_totalseg_model_dir  # noqa: E402


# ---------------------------------------------------------------------------
# Stage directory fragments
# ---------------------------------------------------------------------------

STAGE0_DIR: str = "res_convert_qvtpy"
STAGE1_EICAB_DIR: str = "eicab"
QVT_SUBDIR: str = "qvtpy"
STAGE2_REGISTRATION_DIR: str = "stage2_registration"
STAGE3_CENTERLINE_DIR: str = "stage3_centerline"
STAGE4_SEG_DIR: str = "stage4_4dflow_segmentation"
STAGE4T_SEG_DIR: str = "stage4t_4dflow_t_segmentation"
STAGE5_LOC_DIR: str = "stage5_loc_generation"
STAGE6_MEASURE_DIR: str = "stage6_measure"
STAGE7_MORPHOMETRICS_DIR: str = "stage7_morphometrics"


# ---------------------------------------------------------------------------
# SGE defaults (overridable via .nvitk/sge.json `pipelines.qvtpy`)
# ---------------------------------------------------------------------------

#: Portable fallbacks for scratch locations. Everything else must come from ``sge.json`` —
#: an unset value resolves to ``None`` rather than to somebody else's filesystem.
_FALLBACK_SGE_ROOT = Path(tempfile.gettempdir()) / "nvitk-sge"


def _pipe() -> dict:
    """``defaults`` overlaid with ``pipelines.qvtpy`` from ``sge.json``."""
    return _sj.merged_pipeline_flat(PIPELINE_NAME)


def _log_err() -> tuple[Path, Path]:
    """Resolved (log, err) directories for this pipeline."""
    return _sj.resolve_log_err_dirs(
        paths=_sj.paths_section(),
        pipe=_pipe(),
        fallback_log=_FALLBACK_SGE_ROOT / "logs" / SGE_JOB_PREFIX,
        fallback_err=_FALLBACK_SGE_ROOT / "errs" / SGE_JOB_PREFIX,
    )


def _opt_path(value) -> Path | None:
    """A configured value as an expanded path, or ``None`` when unset."""
    if value is None or not str(value).strip():
        return None
    return Path(os.path.expanduser(str(value).strip()))


_ROOT_KEYS = (
    "DEFAULT_DICOM_ROOT", "DEFAULT_NIFTI_ROOT", "DEFAULT_RESULTS_ROOT",
    "DEFAULT_TOTALSEG_MODEL_ROOT", "LOCAL_DEFAULT_DICOM_ROOT", "LOCAL_DEFAULT_NIFTI_ROOT",
    "LOCAL_DEFAULT_RESULTS_ROOT", "LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT",
    "CLUSTER_HOST_ALIASES",
)

_RESOLVERS: dict[str, lazy_config.Resolver] = {
    "SGE_PROJECT": lambda: str(_pipe().get("sge_project", "")) or None,
    "SGE_ACCOUNT": lambda: str(_pipe().get("sge_account", "")) or None,
    "SGE_NGPU": lambda: int(_pipe().get("sge_ngpu") or 0),
    "SGE_H_VMEM": lambda: str(_pipe().get("sge_h_vmem", "")) or None,
    "SGE_QUEUE": lambda: _pipe().get("sge_queue"),
    "SGE_LOG_DIR": lambda: _log_err()[0],
    "SGE_ERR_DIR": lambda: _log_err()[1],
    "SGE_SCRIPTS_DIR": lambda: (
        _opt_path(_pipe().get("default_sge_scripts_dir"))
        or _opt_path(_sj.paths_section().get("sge_scripts_dir"))
        or _FALLBACK_SGE_ROOT / "scripts"
    ),
    "CONTAINER_PATH": lambda: _sj.resolve_nvitk_container(pipe=_pipe()),
    "NVITK_SRC_DIR": lambda: _sj.resolve_nvitk_src_dir(),
    # Data roots are owned by util.io.paths; forwarded here so `cfg.X` keeps working and stays
    # live rather than snapshotting at import.
    **{name: (lambda n=name: getattr(_paths_mod, n)) for name in _ROOT_KEYS},
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)


__all__ = [
    "PIPELINE_NAME",
    "SGE_JOB_PREFIX",
    "CLUSTER_HOST_ALIASES",
    "DEFAULT_DICOM_ROOT",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_TOTALSEG_MODEL_ROOT",
    "LOCAL_DEFAULT_DICOM_ROOT",
    "LOCAL_DEFAULT_NIFTI_ROOT",
    "LOCAL_DEFAULT_RESULTS_ROOT",
    "LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT",
    "resolve_totalseg_model_dir",
    "STAGE0_DIR",
    "STAGE1_EICAB_DIR",
    "QVT_SUBDIR",
    "STAGE2_REGISTRATION_DIR",
    "STAGE3_CENTERLINE_DIR",
    "STAGE4_SEG_DIR",
    "STAGE4T_SEG_DIR",
    "STAGE5_LOC_DIR",
    "STAGE6_MEASURE_DIR",
    "STAGE7_MORPHOMETRICS_DIR",
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
