"""
Filesystem layout for the CViT pipeline (local workstation vs cluster vs container).

Description
-----------
Every root comes from ``sge.json``'s ``pipelines.cvit_paths`` section (``local_<key>`` /
``cluster_<key>``) or from the matching ``--*-root`` CLI flag; there are no installation paths in
this file. A root that is needed but unconfigured raises
:class:`~nvitk.core.config_paths.ConfigError` naming the key. Configuration is read on use, not on
import, so ``--config-dir`` is honoured.

Precedence follows the repository convention (``docs/configuration.md``): under ``--submit
local`` the CLI flag wins; under ``--submit sge`` the ``cluster_*`` value wins, because a flag
typed on the workstation holds a host path.

Data layout
-----------
::

    <data_root>/                                   # your labelled data (read-only)
    <nnunet_raw|nnunet_preprocessed|nnunet_results>/DatasetXXX_<Name>/
    <nnssl_raw|nnssl_preprocessed|nnssl_results>/DatasetYYY_<Name>Corpus/
    <corpus_root>/                                 # harmonised unlabelled corpus volumes
    <results_root>/stage*_*/                       # bundles, metrics, probe, predictions, exports
    <model_root>/                                  # optional external checkpoints

nnU-Net and nnssl never share a directory, so either side can be wiped independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths, lazy_config

#: ``sge.json`` section holding this pipeline's data roots.
PIPELINE_PATHS_ID = "cvit_paths"

#: Root keys of ``pipelines.cvit_paths`` (without the ``local_`` / ``cluster_`` prefix).
ROOT_KEYS: tuple[str, ...] = (
    "data_root",
    "nnunet_raw",
    "nnunet_preprocessed",
    "nnunet_results",
    "nnssl_raw",
    "nnssl_preprocessed",
    "nnssl_results",
    "corpus_root",
    "results_root",
    "model_root",
)

# ──────────────────────────────────────────────────────────────────────────────
# Stage output subdirectories (under ``results_root``)
# ──────────────────────────────────────────────────────────────────────────────

STAGE0_DATAPREP_DIR = "stage0_dataprep"
STAGE1_PRETRAIN_DIR = "stage1_pretrain"
STAGE2_TRAIN_DIR = "stage2_train"
STAGE3_EVAL_DIR = "stage3_evaluate"
STAGE3B_PROBE_DIR = "stage3b_probe"
STAGE4_INFER_DIR = "stage4_infer"
STAGE5_EXPORT_DIR = "stage5_export"

#: Default nnU-Net / nnssl dataset ids — in the 6xx range, clear of public datasets and of
#: topbrain's 5xx ids that may share the same ``nnUNet_raw``.
DEFAULT_DATASET_ID = 601
DEFAULT_CORPUS_ID = 611


def dataset_folder_name(dataset_id: int, name: str) -> str:
    """nnU-Net / nnssl folder name, ``Dataset601_Name``."""
    if not 1 <= int(dataset_id) <= 999:
        raise ValueError(f"Dataset id must be in 1..999, got {dataset_id}.")
    clean = "".join(c for c in str(name) if c.isalnum() or c in "-")
    if not clean:
        raise ValueError(f"Dataset name {name!r} has no alphanumeric characters.")
    return f"Dataset{int(dataset_id):03d}_{clean}"


# ──────────────────────────────────────────────────────────────────────────────
# Config access (lazy)
# ──────────────────────────────────────────────────────────────────────────────


def _pipe_paths() -> dict:
    """The ``pipelines.cvit_paths`` block (empty when absent)."""
    return _sj.pipeline_section(PIPELINE_PATHS_ID)


def _opt_root(key: str) -> Path | None:
    """A configured root, or ``None`` when unset (never raises; used for Click defaults)."""
    raw = _pipe_paths().get(key)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


_RESOLVERS: dict[str, lazy_config.Resolver] = {
    **{f"DEFAULT_{k.upper()}": (lambda k=k: _opt_root(f"cluster_{k}")) for k in ROOT_KEYS},
    **{f"LOCAL_DEFAULT_{k.upper()}": (lambda k=k: _opt_root(f"local_{k}")) for k in ROOT_KEYS},
    "CLUSTER_HOST_ALIASES": lambda: _sj.merge_cluster_host_aliases(
        {}, _sj.paths_section(), _pipe_paths()
    ),
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Resolved layout
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CViTPaths:
    """Resolved roots for one execution context."""

    data_root: Path
    nnunet_raw: Path
    nnunet_preprocessed: Path
    nnunet_results: Path
    nnssl_raw: Path
    nnssl_preprocessed: Path
    nnssl_results: Path
    corpus_root: Path
    results_root: Path
    model_root: Path

    # ---- nnU-Net dataset accessors -------------------------------------------------------
    def nnunet_raw_dir(self, dataset: str) -> Path:
        return self.nnunet_raw / dataset

    def nnunet_preprocessed_dir(self, dataset: str) -> Path:
        return self.nnunet_preprocessed / dataset

    def nnunet_results_dir(self, dataset: str) -> Path:
        return self.nnunet_results / dataset

    # ---- nnssl accessors -----------------------------------------------------------------
    def nnssl_raw_dir(self, corpus: str) -> Path:
        return self.nnssl_raw / corpus

    def nnssl_preprocessed_dir(self, corpus: str) -> Path:
        return self.nnssl_preprocessed / corpus

    # ---- stage outputs -------------------------------------------------------------------
    def stage_dir(self, name: str) -> Path:
        """A stage output directory under ``results_root``."""
        return self.results_root / name

    def ensure_dirs(self, *dirs: Path) -> None:
        """``mkdir -p`` each of *dirs*."""
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}


def _root_from_config(key: str, *, prefix: str, fallback: Path | None) -> Path:
    """Resolve one root honouring the flag-vs-config precedence (see module docstring)."""
    raw = _pipe_paths().get(f"{prefix}_{key}")
    if prefix == "cluster" and raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return Path(fallback)
    return Path(
        os.path.expanduser(
            str(
                config_paths.require(
                    raw,
                    key=f"pipelines.{PIPELINE_PATHS_ID}.{prefix}_{key}",
                    hint="Set it in sge.json, or pass the matching --*-root flag.",
                )
            ).strip()
        )
    )


def _layout(prefix: str, **overrides: Path | None) -> CViTPaths:
    return CViTPaths(**{
        key: _root_from_config(key, prefix=prefix, fallback=overrides.get(key)) for key in ROOT_KEYS
    })


def layout_local(**overrides: Path | None) -> CViTPaths:
    """Roots on the workstation (``local_*`` keys); CLI flags win."""
    return _layout("local", **overrides)


def layout_cluster(**overrides: Path | None) -> CViTPaths:
    """Roots on the cluster (``cluster_*`` keys); configured values win."""
    return _layout("cluster", **overrides)


#: Stand-in for a root a container job did not bind — fails on an obviously fake path.
UNAVAILABLE_ROOT = Path("/nonexistent/cvit-root-not-bound-in-container")


def layout_from_roots(**roots: Path | str | None) -> CViTPaths:
    """A layout from explicit roots only (worker processes, tests); unset roots are unavailable."""
    return CViTPaths(**{
        key: Path(roots[key]) if roots.get(key) is not None else UNAVAILABLE_ROOT / key
        for key in ROOT_KEYS
    })


__all__ = [
    "CViTPaths",
    "DEFAULT_CORPUS_ID",
    "DEFAULT_DATASET_ID",
    "PIPELINE_PATHS_ID",
    "ROOT_KEYS",
    "STAGE0_DATAPREP_DIR",
    "STAGE1_PRETRAIN_DIR",
    "STAGE2_TRAIN_DIR",
    "STAGE3B_PROBE_DIR",
    "STAGE3_EVAL_DIR",
    "STAGE4_INFER_DIR",
    "STAGE5_EXPORT_DIR",
    "UNAVAILABLE_ROOT",
    "dataset_folder_name",
    "layout_cluster",
    "layout_from_roots",
    "layout_local",
]
