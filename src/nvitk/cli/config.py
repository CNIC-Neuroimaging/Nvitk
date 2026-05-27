"""Defaults for module-level image-tool CLIs; overlays from ``.nvitk/sge.json``."""

from __future__ import annotations

from pathlib import Path

from nvitk.cluster import sge_json as _sj

PIPELINE_ID = "image_tools"

# Host paths (override via sge.json paths / pipelines.image_tools)
DEFAULT_CONTAINER = Path("/data3/BIOIT_IMAGE/Containers/nvitk_v2026.05.27.sif")
DEFAULT_MODELS = Path("/references/AI_models")

SGE_PROJECT = "GPU"
SGE_ACCOUNT = "Prod"
SGE_QUEUE: str | None = None
SGE_NGPU = 0
SGE_H_VMEM = "32G"
SGE_JOB_PREFIX = "nvitk-tool"

SGE_SCRIPTS_DIR = Path("/tmp/nvitk-sge/scripts")
SGE_LOG_DIR = Path("/tmp/nvitk-sge/logs")
SGE_ERR_DIR = Path("/tmp/nvitk-sge/errs")

NVITK_SRC_DIR = Path(__file__).resolve().parents[2]

_paths = _sj.paths_section()
_pipe = _sj.merged_pipeline_flat(PIPELINE_ID)

if v := _paths.get("nvitk_src_dir"):
    NVITK_SRC_DIR = Path(str(v))
DEFAULT_CONTAINER = _sj.resolve_nvitk_container(pipe=_pipe, fallback=DEFAULT_CONTAINER)
if v := _pipe.get("default_sge_model_root") or _pipe.get("sge_model_root"):
    DEFAULT_MODELS = Path(str(v))
if v := _pipe.get("sge_project"):
    SGE_PROJECT = str(v)
if v := _pipe.get("sge_account"):
    SGE_ACCOUNT = str(v)
if v := _pipe.get("sge_queue"):
    SGE_QUEUE = str(v) if v else None
if v := _pipe.get("sge_ngpu") is not None:
    SGE_NGPU = int(v)
if v := _pipe.get("sge_h_vmem"):
    SGE_H_VMEM = str(v)
if v := _pipe.get("sge_job_prefix"):
    SGE_JOB_PREFIX = str(v)
if v := _paths.get("sge_scripts_dir"):
    SGE_SCRIPTS_DIR = Path(str(v))
if v := _pipe.get("default_sge_scripts_dir"):
    SGE_SCRIPTS_DIR = Path(str(v))

SGE_LOG_DIR, SGE_ERR_DIR = _sj.resolve_log_err_dirs(
    paths=_paths,
    pipe=_pipe,
    fallback_log=SGE_LOG_DIR,
    fallback_err=SGE_ERR_DIR,
)
