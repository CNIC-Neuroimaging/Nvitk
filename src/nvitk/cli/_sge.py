"""SGE submission helpers for module-level image-tool CLIs."""

from __future__ import annotations

import shlex
from datetime import datetime
from pathlib import Path

from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
    write_script_header,
)
from nvitk.cli import config as cfg


def cluster_paths(
    *,
    data_root: Path,
    output_root: Path,
    container: Path | None = None,
    models: Path | None = None,
) -> ClusterPaths:
    return ClusterPaths(
        src=cfg.NVITK_SRC_DIR,
        container=container or cfg.DEFAULT_CONTAINER,
        models=models or cfg.DEFAULT_MODELS,
        data_root=data_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )


def default_resources(*, gpu: bool = False) -> SgeResources:
    if gpu:
        return SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=max(1, int(cfg.SGE_NGPU) or 1),
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        )
    return SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=0,
        h_vmem=cfg.SGE_H_VMEM,
        queue=cfg.SGE_QUEUE,
    )


def build_worker_command(
    module_path: str,
    subcommand: str,
    *,
    container_input: str,
    container_output: str,
    extra_args: list[str],
) -> str:
    """Build python command run inside Singularity (container paths)."""
    src = "/nvitk/src"
    script = f"{src}/nvitk/cli/{module_path}"
    parts = [
        "python",
        shlex.quote(script),
        shlex.quote(subcommand),
        "-i",
        shlex.quote(container_input),
        "-o",
        shlex.quote(container_output),
    ]
    parts.extend(shlex.quote(a) for a in extra_args)
    return " ".join(parts)


def submit_tool_job(
    *,
    job_name: str,
    python_cmd: str,
    data_root: Path,
    output_root: Path,
    gpu: bool = False,
    emit: object | None = None,
) -> str | None:
    paths = cluster_paths(data_root=data_root, output_root=output_root)
    paths.ensure_dirs()
    spec = StageSpec(
        job_name=job_name,
        python_cmd=python_cmd,
        resources=default_resources(gpu=gpu),
        binds=SingularityBinds(),
        use_nv=gpu,
        extra_env={"NVITK_BACKEND": "cupy" if gpu else "numpy"},
    )
    return submit_stage(spec, paths, emit=emit)


def emit_submit_script(
    *,
    script_path: Path,
    stages: list[tuple[str, str]],
    data_root: Path,
    output_root: Path,
    gpu: bool = False,
) -> Path:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    paths = cluster_paths(data_root=data_root, output_root=output_root)
    paths.ensure_dirs()
    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(fh)
        for job_name, python_cmd in stages:
            submit_tool_job(
                job_name=job_name,
                python_cmd=python_cmd,
                data_root=data_root,
                output_root=output_root,
                gpu=gpu,
                emit=fh,
            )
    return script_path


def default_emit_path(tool: str, subcommand: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.SGE_SCRIPTS_DIR / f"submit_{cfg.SGE_JOB_PREFIX}_{tool}_{subcommand}_{ts}.sh"
