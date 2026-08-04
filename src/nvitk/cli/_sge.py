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
    python_module_argv,
    submit_stage,
    write_script_header,
)
from nvitk.cluster import sge_json
from nvitk.cli import config as cfg
from nvitk.core.click_backend import sge_backend_env


def cluster_paths(
    *,
    data_root: Path,
    output_root: Path,
    container: Path | None = None,
    models: Path | None = None,
    nvitk_src: Path | None = None,
) -> ClusterPaths:
    """Build a :class:`~nvitk.cluster.sge.ClusterPaths` for a CLI job, filling container/models/source
    paths from :mod:`nvitk.cli.config` defaults where not overridden."""
    src = nvitk_src or sge_json.resolve_nvitk_src_dir(fallback=cfg.NVITK_SRC_DIR)
    return ClusterPaths(
        src=src,
        container=container or cfg.DEFAULT_CONTAINER,
        models=models or cfg.DEFAULT_MODELS,
        data_root=data_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )


def default_resources(*, gpu: bool = False) -> SgeResources:
    """Default :class:`~nvitk.cluster.sge.SgeResources` for a CLI job, requesting a GPU slot when
    *gpu* is True (else CPU-only)."""
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
    stem = module_path.removesuffix(".py")
    module = f"nvitk.cli.{stem}"
    parts = [
        *python_module_argv(module),
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
    """Submit (or, if *emit* is given, append to that script file handle instead of submitting) one
    SGE stage running *python_cmd* under Singularity, using default resources/binds for *gpu*."""
    paths = cluster_paths(data_root=data_root, output_root=output_root)
    paths.ensure_dirs()
    binds = SingularityBinds()
    spec = StageSpec(
        job_name=job_name,
        python_cmd=python_cmd,
        resources=default_resources(gpu=gpu),
        binds=binds,
        use_nv=gpu,
        extra_env=sge_backend_env(binds.src, "cupy" if gpu else "numpy"),
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
    """Write a qsub shell script at *script_path* containing one job stage per ``(job_name,
    python_cmd)`` in *stages*, sharing the header and cluster paths."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    paths = cluster_paths(data_root=data_root, output_root=output_root)
    paths.ensure_dirs()
    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=paths.log_dir,
            err_dir=paths.err_dir,
            title="nvitk image_tools CLI",
        )
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
    """Timestamped default path for an emitted submit script for *tool*/*subcommand* under
    ``cfg.SGE_SCRIPTS_DIR``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.SGE_SCRIPTS_DIR / f"submit_{cfg.SGE_JOB_PREFIX}_{tool}_{subcommand}_{ts}.sh"
