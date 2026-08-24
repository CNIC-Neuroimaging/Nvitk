"""Defaults for eICAB TOF / Circle-of-Willis segmentation (local + SGE).

Override paths via CLI flags; if :data:`CONTAINER_PATH` does not exist, local/SGE
runs fail with a clear error unless ``--container`` points to a valid image.

Optional site overrides: copy ``.nvitk/sge.example.json`` to ``.nvitk/sge.json``
(see :mod:`nvitk.cluster.sge_json`). Under ``pipelines.eicab``,
``default_sge_container_root`` is the outer Singularity image used for
``singularity exec`` (``PIPELINE_CONTAINER_PATH``); ``container_path`` (or
``--container``) is the inner eICAB inference ``.sif`` passed to ``singularity run``.
``default_vasculature_host_dir`` sets the host tree for Neuro/vasculature2 (see
``--vasculature-dir``). eICAB weights live inside the eICAB container; there is no
separate model-weights bind for this tool.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths

# eICAB Singularity image (override with --container).
CONTAINER_PATH: Path | None = None  # sge.json: pipelines.eicab.container_path
DEFAULT_TMP_DIR = Path("~/local_tmp")
# OMP/BLAS thread cap inside the eICAB container (no qsub -pe required).
EICAB_THREAD_LIMIT: int | None = None
# Bind VED multiscale NIfTIs to node-local /data_tmp during SGE runs (NFS-safe).
EICAB_LOCAL_METRIC_SCRATCH = True
# Node-local scratch root for VED metric_space (this cluster: /data_tmp).
EICAB_METRIC_SCRATCH_ROOT = tempfile.gettempdir()  # sge.json: pipelines.eicab.eicab_metric_scratch_root
# Optional qsub -pe smp N (cluster-specific; omit unless your queue supports it).
SGE_PE_SMP: int | None = None

PIPELINE_CONTAINER_PATH = CONTAINER_PATH
SGE_PROJECT = "MCC"
SGE_ACCOUNT = "Prod"
SGE_NGPU = 0
SGE_H_VMEM = "25G"
SGE_QUEUE = None
SGE_LOG_DIR = Path(tempfile.gettempdir()) / "nvitk-sge" / "logs" / "eICAB"
SGE_ERR_DIR = Path(tempfile.gettempdir()) / "nvitk-sge" / "errs" / "eICAB"

DEFAULT_SGE_SCRIPTS_DIR: Path = Path(tempfile.gettempdir()) / "nvitk-sge" / "scripts"

# Host tree bind-mounted to ``/programs/Neuro/vasculature2`` (legacy eICAB layout).
DEFAULT_VASCULATURE_HOST_DIR: Path | None = None  # sge.json: pipelines.eicab.default_vasculature_host_dir

CLUSTER_HOST_ALIASES: dict[str, str] = {}

def _apply_config() -> None:
    """Merge ``sge.json`` over this module's defaults.

    Run once at import and again whenever the configuration directory is redirected,
    so a late ``--config-dir`` reaches these constants too.
    """
    global CLUSTER_HOST_ALIASES, CONTAINER_PATH, DEFAULT_SGE_SCRIPTS_DIR, DEFAULT_TMP_DIR, DEFAULT_VASCULATURE_HOST_DIR, EICAB_LOCAL_METRIC_SCRATCH, EICAB_METRIC_SCRATCH_ROOT, EICAB_THREAD_LIMIT, PIPELINE_CONTAINER_PATH, SGE_ACCOUNT, SGE_ERR_DIR, SGE_H_VMEM, SGE_LOG_DIR, SGE_NGPU, SGE_PE_SMP, SGE_PROJECT, SGE_QUEUE, _paths, _pipe, er, lg
    _pipe = _sj.merged_pipeline_flat("eicab")
    _paths = _sj.paths_section()
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
    if (v := _pipe.get("container_path")):
        CONTAINER_PATH = Path(os.path.expanduser(str(v)))
    if (v := _pipe.get("default_sge_container_root") or _pipe.get("pipeline_container_path")):
        PIPELINE_CONTAINER_PATH = Path(os.path.expanduser(str(v)))
    elif "default_sge_container_root" not in _pipe and "pipeline_container_path" not in _pipe:
        PIPELINE_CONTAINER_PATH = CONTAINER_PATH
    if (v := _pipe.get("default_tmp_dir")):
        DEFAULT_TMP_DIR = Path(os.path.expanduser(str(v)))
    if (v := _pipe.get("eicab_thread_limit")) is not None:
        EICAB_THREAD_LIMIT = int(v)
    if (v := _pipe.get("eicab_local_metric_scratch")) is not None:
        EICAB_LOCAL_METRIC_SCRATCH = bool(v)
    if (v := _pipe.get("eicab_metric_scratch_root")) is not None:
        EICAB_METRIC_SCRATCH_ROOT = str(v).rstrip("/") or tempfile.gettempdir()
    if (v := _pipe.get("sge_pe_smp")) is not None:
        SGE_PE_SMP = int(v)
    if (v := _pipe.get("default_vasculature_host_dir")):
        DEFAULT_VASCULATURE_HOST_DIR = Path(os.path.expanduser(str(v)))
    CLUSTER_HOST_ALIASES = _sj.merge_cluster_host_aliases(
        CLUSTER_HOST_ALIASES, _paths, _pipe
    )


_apply_config()
config_paths.register_reload_hook(_apply_config)

__all__ = [
    "CLUSTER_HOST_ALIASES",
    "CONTAINER_PATH",
    "DEFAULT_SGE_SCRIPTS_DIR",
    "DEFAULT_TMP_DIR",
    "DEFAULT_VASCULATURE_HOST_DIR",
    "EICAB_LOCAL_METRIC_SCRATCH",
    "EICAB_METRIC_SCRATCH_ROOT",
    "EICAB_THREAD_LIMIT",
    "PIPELINE_CONTAINER_PATH",
    "SGE_ACCOUNT",
    "SGE_ERR_DIR",
    "SGE_H_VMEM",
    "SGE_LOG_DIR",
    "SGE_NGPU",
    "SGE_PE_SMP",
    "SGE_PROJECT",
    "SGE_QUEUE",
]
