"""Defaults for TotalSegmentator CLI (local + SGE).

Override via CLI flags or edit this module. Optional site overrides in
``.nvitk/sge.json`` (see :mod:`nvitk.cluster.sge_json`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths

CONTAINER_PATH: Path | None = None  # sge.json: paths.nvitk_container
MODELS_DIR: Path | None = None  # sge.json: pipelines.totalsegmentator.default_sge_model_root

SGE_PROJECT = "MCC_GPU"
SGE_ACCOUNT = "MCC_GPU"
SGE_NGPU = 1
SGE_H_VMEM = "50G"
SGE_QUEUE = None
SGE_LOG_DIR = Path(tempfile.gettempdir()) / "nvitk-sge" / "logs" / "totalseg"
SGE_ERR_DIR = Path(tempfile.gettempdir()) / "nvitk-sge" / "errs" / "totalseg"

DEFAULT_SGE_SCRIPTS_DIR = Path(tempfile.gettempdir()) / "nvitk-sge" / "scripts"

def _apply_config() -> None:
    """Merge ``sge.json`` over this module's defaults.

    Run once at import and again whenever the configuration directory is redirected,
    so a late ``--config-dir`` reaches these constants too.
    """
    global CLUSTER_HOST_ALIASES, CONTAINER_PATH, DEFAULT_NVITK_SRC_DIR, DEFAULT_SGE_SCRIPTS_DIR, MODELS_DIR, SGE_ACCOUNT, SGE_ERR_DIR, SGE_H_VMEM, SGE_LOG_DIR, SGE_NGPU, SGE_PROJECT, SGE_QUEUE, _paths, _pipe, er, lg
    DEFAULT_NVITK_SRC_DIR = _sj.resolve_nvitk_src_dir()

    CLUSTER_HOST_ALIASES = {}

    _pipe = _sj.merged_pipeline_flat("totalsegmentator")
    _paths = _sj.paths_section()
    if (v := _paths.get("nvitk_src_dir")):
        DEFAULT_NVITK_SRC_DIR = Path(os.path.expanduser(str(v)))
    if (v := _pipe.get("sge_project")) is not None:
        SGE_PROJECT = str(v)
    if (v := _pipe.get("sge_account")) is not None:
        SGE_ACCOUNT = str(v)
    if (v := _pipe.get("sge_ngpu")) is not None:
        SGE_NGPU = int(v)
    if (v := _pipe.get("sge_h_vmem")) is not None:
        SGE_H_VMEM = str(v)
    if "sge_queue" in _pipe:
        SGE_QUEUE = _pipe["sge_queue"]
    lg, er = _sj.resolve_log_err_dirs(
        paths=_paths,
        pipe=_pipe,
        fallback_log=SGE_LOG_DIR,
        fallback_err=SGE_ERR_DIR,
    )
    SGE_LOG_DIR, SGE_ERR_DIR = lg, er
    if (v := _pipe.get("default_sge_scripts_dir")):
        DEFAULT_SGE_SCRIPTS_DIR = Path(os.path.expanduser(str(v)))
    CONTAINER_PATH = _sj.resolve_nvitk_container(pipe=_pipe, fallback=CONTAINER_PATH)
    if (v := _pipe.get("default_sge_model_root") or _pipe.get("models_dir")):
        MODELS_DIR = Path(os.path.expanduser(str(v)))
    CLUSTER_HOST_ALIASES = _sj.merge_cluster_host_aliases(
        CLUSTER_HOST_ALIASES, _paths, _pipe
    )


_apply_config()
config_paths.register_reload_hook(_apply_config)

__all__ = [
    "CLUSTER_HOST_ALIASES",
    "CONTAINER_PATH",
    "DEFAULT_NVITK_SRC_DIR",
    "DEFAULT_SGE_SCRIPTS_DIR",
    "MODELS_DIR",
    "SGE_ACCOUNT",
    "SGE_ERR_DIR",
    "SGE_H_VMEM",
    "SGE_LOG_DIR",
    "SGE_NGPU",
    "SGE_PROJECT",
    "SGE_QUEUE",
]
