"""ToPBrain host configuration: data roots, model defaults, and SGE settings.

**Inputs (JSON)**

- ``sge.json`` ``pipelines.topbrain`` — SGE project/account/memory, log and err dirs,
  container path (see :mod:`nvitk.cluster.sge_json`).
- ``sge.json`` ``pipelines.topbrain_paths`` — ``local_*`` / ``cluster_*`` data roots
  (see :mod:`nvitk.pipes.topbrain.util.paths`).

**Outputs (constants consumed by stages)**

- ``DEFAULT_*`` / ``LOCAL_DEFAULT_*`` — data roots, forwarded from ``util.paths``.
- ``SGE_*`` — project, account, memory, log/err dirs and ``CONTAINER_PATH``.
- Training and pre-training defaults (trainer names, patch/batch sizes, folds).

Every value is read from ``sge.json`` on first use — there are no installation paths in this
file and nothing is resolved at import time, so ``--config-dir`` is honoured.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import lazy_config
from nvitk.pipes.topbrain.util import paths as _paths_mod

# ---------------------------------------------------------------------------
# Pipeline identity
# ---------------------------------------------------------------------------

PIPELINE_NAME: str = "topbrain"
SGE_JOB_PREFIX: str = "TOPBRAIN"

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------

#: Label set used unless ``--label-set`` says otherwise. ``ta36`` is the only modality-agnostic
#: set and the one the 2026 track scores.
DEFAULT_LABEL_SET: str = "ta36"

#: Number of cross-validation folds. Grouped by patient id, so each fold holds out 5 of the 25
#: patients — both their CT and their MR.
DEFAULT_NUM_FOLDS: int = 5

#: Seed for the fold assignment. Fixed so a re-run reproduces the same splits.
DEFAULT_FOLD_SEED: int = 12345

# ---------------------------------------------------------------------------
# Intensity harmonisation (stage 0)
# ---------------------------------------------------------------------------

#: CTA window in Hounsfield units. The upper bound is deliberately high enough to keep bone
#: distinguishable from contrast-filled lumen: the TA36 infraclinoid ICA classes (35/36) run
#: through the carotid canal *inside* bone, so a tighter window that saturates bone would erase
#: the very contrast the hardest two classes depend on.
DEFAULT_CT_WINDOW: tuple[float, float] = (300.0, 600.0)

#: Percentiles used to robust-scale MRA. TOF has no standardised scale and a hard floor at 0,
#: so a fixed window is meaningless and percentiles are used instead.
DEFAULT_MR_PERCENTILES: tuple[float, float] = (0.5, 99.5)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

# Planning is not configured here. Stage 2 delegates it to ``nnUNetv2_preprocess_like_nnssl``,
# which derives the plans — including their name, which embeds the chosen spacing — from the
# pre-trained checkpoint. Stages 3-5 read the resulting identifier back from stage 2's
# provenance rather than reconstructing it; see
# :func:`nvitk.pipes.topbrain.stage2_train.resolve_trained_run`.

#: Supervised loss, by :mod:`~nvitk.pipes.topbrain.util.losses` registry name. Deliberately the
#: plain nnU-Net objective: the topology losses are only meaningful measured against it.
DEFAULT_LOSS: str = "dice_ce"

#: Self-supervised trainer class (nnssl). SparK masks in the encoder rather than the input, so
#: it transfers better to a downstream encoder than plain MAE.
DEFAULT_SSL_TRAINER: str = "SparkMAETrainer"

#: nnssl spacing style for pre-training. **Not** ``onemmiso``: sub-millimetre vessels such as
#: AChA, OA and Pcom do not survive resampling to 1 mm isotropic.
DEFAULT_SSL_CONFIG: str = "median"

#: Self-supervised loss, by registry name (nnssl's ``MAEMSELoss``).
DEFAULT_SSL_LOSS: str = "mse"

# ---------------------------------------------------------------------------
# SGE defaults (overridable via sge.json `pipelines.topbrain`)
# ---------------------------------------------------------------------------

#: Portable fallbacks for scratch locations. Everything else must come from ``sge.json`` — an
#: unset value resolves to ``None`` rather than to somebody else's filesystem.
_FALLBACK_SGE_ROOT = Path(tempfile.gettempdir()) / "nvitk-sge"


def _pipe() -> dict:
    """``defaults`` overlaid with ``pipelines.topbrain`` from ``sge.json``."""
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


#: Root names forwarded from ``util.paths`` so ``cfg.X`` keeps working and stays live rather
#: than snapshotting at import.
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
    "DEFAULT_CT_WINDOW",
    "DEFAULT_FOLD_SEED",
    "DEFAULT_LABEL_SET",
    "DEFAULT_LOSS",
    "DEFAULT_MR_PERCENTILES",
    "DEFAULT_NUM_FOLDS",
    "DEFAULT_SSL_CONFIG",
    "DEFAULT_SSL_LOSS",
    "DEFAULT_SSL_TRAINER",
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
