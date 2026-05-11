"""Defaults for TotalSegmentator CLI (local + SGE).

Override via CLI flags or edit this module. Optional site overrides in
``.nvitk/sge.json`` (see :mod:`nvitk.cluster.sge_json`).
"""

from __future__ import annotations

import os
from pathlib import Path

from nvitk.cluster import sge_json as _sj

CONTAINER_PATH = Path("/images/BIOIT_IMAGE/nvitk_v2026.04.21.sif")
MODELS_DIR = Path("/references/AI_models/totalsegmentator")

SGE_PROJECT = "MCC_GPU"
SGE_ACCOUNT = "MCC_GPU"
SGE_NGPU = 1
SGE_H_VMEM = "50G"
SGE_QUEUE = None
SGE_LOG_DIR = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/logs/totalseg")
SGE_ERR_DIR = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/errs/totalseg")

DEFAULT_SGE_SCRIPTS_DIR = Path("/data3/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS/")

DEFAULT_NVITK_SRC_DIR = Path("/data3/BIOIT_IMAGE/nvitk/src")

CLUSTER_HOST_ALIASES: dict[str, str] = {}

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
if (v := _pipe.get("default_sge_container_root") or _pipe.get("container_path")):
    CONTAINER_PATH = Path(os.path.expanduser(str(v)))
if (v := _pipe.get("default_sge_model_root") or _pipe.get("models_dir")):
    MODELS_DIR = Path(os.path.expanduser(str(v)))
CLUSTER_HOST_ALIASES = _sj.merge_cluster_host_aliases(
    CLUSTER_HOST_ALIASES, _paths, _pipe
)

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
