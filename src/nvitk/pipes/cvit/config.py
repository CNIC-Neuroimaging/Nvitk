"""
CViT host configuration: model / training defaults and SGE settings.

**Inputs (JSON)**

- ``sge.json`` ``pipelines.cvit`` — SGE project/account/memory, log and err dirs, container
  path (see :mod:`nvitk.cluster.sge_json`).
- ``sge.json`` ``pipelines.cvit_paths`` — ``local_*`` / ``cluster_*`` data roots
  (see :mod:`nvitk.pipes.cvit.util.paths`).

Every SGE / path value is read on first use, so ``--config-dir`` is honoured and an unconfigured
machine can still print ``--help``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import lazy_config
from nvitk.pipes.cvit.util import paths as _paths_mod

# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "cvit"
SGE_JOB_PREFIX: str = "CVIT"

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET_NAME: str = "CViT"
DEFAULT_NUM_FOLDS: int = 5
DEFAULT_FOLD_SEED: int = 12345

# ---------------------------------------------------------------------------
# Architecture defaults (see nvitk.nn.cvit.config for the full field reference)
# ---------------------------------------------------------------------------

DEFAULT_ARCH: str = "CViTB"
DEFAULT_TOKENIZER: str = "hierarchical"
DEFAULT_DECODER: str = "unet"
DEFAULT_SKIPS: str = "all"

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_LOSS: str = "dice_ce"
#: nnU-Net configuration trained by default (``2d`` is available for 2D datasets).
DEFAULT_CONFIGURATION: str = "3d_fullres"
#: Baseline planner whose fingerprint-derived spacing / patch size / normalisation CViT plans
#: inherit.
DEFAULT_BASELINE_PLANNER: str = "nnUNetPlannerResEncM"
DEFAULT_LR: float = 3e-4
DEFAULT_WARMUP_EPOCHS: int = 50
DEFAULT_PROBE_EVERY: int = 25

# ---------------------------------------------------------------------------
# Self-supervised defaults
# ---------------------------------------------------------------------------

DEFAULT_SSL: str = "none"
DEFAULT_MASK_RATIO: float = 0.6
#: nnssl spacing style (``median`` keeps fine structures; ``onemmiso`` resamples to 1 mm).
DEFAULT_SSL_CONFIG: str = "median"
DEFAULT_SSL_PATCH: tuple[int, int, int] = (128, 128, 128)

# ---------------------------------------------------------------------------
# Attention probe defaults
# ---------------------------------------------------------------------------

DEFAULT_PROBE_INTERVENTIONS: tuple[str, ...] = (
    "full", "attn_off", "attn_local", "attn_uniform", "transformer_off", "tokens_off", "skips_off",
)
#: Radii (mm) for ``attn_local``; each is one intervention.
DEFAULT_LOCAL_RADII_MM: tuple[float, ...] = (10.0, 30.0)

# ---------------------------------------------------------------------------
# SGE (overridable via sge.json `pipelines.cvit`)
# ---------------------------------------------------------------------------

_FALLBACK_SGE_ROOT = Path(tempfile.gettempdir()) / "nvitk-sge"


def _pipe() -> dict:
    """``defaults`` overlaid with ``pipelines.cvit`` from ``sge.json``."""
    return _sj.merged_pipeline_flat(PIPELINE_NAME)


def _log_err() -> tuple[Path, Path]:
    return _sj.resolve_log_err_dirs(
        paths=_sj.paths_section(),
        pipe=_pipe(),
        fallback_log=_FALLBACK_SGE_ROOT / "logs" / SGE_JOB_PREFIX,
        fallback_err=_FALLBACK_SGE_ROOT / "errs" / SGE_JOB_PREFIX,
    )


def _opt_path(value) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(os.path.expanduser(str(value).strip()))


_ROOT_KEYS: tuple[str, ...] = (
    *(f"DEFAULT_{k.upper()}" for k in _paths_mod.ROOT_KEYS),
    *(f"LOCAL_DEFAULT_{k.upper()}" for k in _paths_mod.ROOT_KEYS),
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
    **{name: (lambda n=name: getattr(_paths_mod, n)) for name in _ROOT_KEYS},
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)


__all__ = [
    "CONTAINER_PATH",
    "DEFAULT_ARCH",
    "DEFAULT_BASELINE_PLANNER",
    "DEFAULT_CONFIGURATION",
    "DEFAULT_DATASET_NAME",
    "DEFAULT_DECODER",
    "DEFAULT_FOLD_SEED",
    "DEFAULT_LOCAL_RADII_MM",
    "DEFAULT_LOSS",
    "DEFAULT_LR",
    "DEFAULT_MASK_RATIO",
    "DEFAULT_NUM_FOLDS",
    "DEFAULT_PROBE_EVERY",
    "DEFAULT_PROBE_INTERVENTIONS",
    "DEFAULT_SKIPS",
    "DEFAULT_SSL",
    "DEFAULT_SSL_CONFIG",
    "DEFAULT_SSL_PATCH",
    "DEFAULT_TOKENIZER",
    "DEFAULT_WARMUP_EPOCHS",
    "NVITK_SRC_DIR",
    "PIPELINE_NAME",
    "SGE_ACCOUNT",
    "SGE_ERR_DIR",
    "SGE_H_VMEM",
    "SGE_JOB_PREFIX",
    "SGE_LOG_DIR",
    "SGE_NGPU",
    "SGE_PROJECT",
    "SGE_QUEUE",
    "SGE_SCRIPTS_DIR",
    *_ROOT_KEYS,
]
