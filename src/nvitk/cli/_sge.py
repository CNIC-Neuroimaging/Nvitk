"""SGE submission helpers for module-level image-tool CLIs."""

from __future__ import annotations

import shlex
from datetime import datetime
from pathlib import Path

from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResourceOverrides,
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


def default_resources(
    *,
    gpu: bool = False,
    overrides: SgeResourceOverrides | None = None,
) -> SgeResources:
    """Configured :class:`~nvitk.cluster.sge.SgeResources` for a CLI job, GPU or CPU-only.

    *overrides* replaces individual fields for this one submission -- what the GUI submit
    dialog sends when the operator picks a different project or memory size. The configured
    values stay the default, so a caller that passes nothing behaves exactly as before.
    """
    base = SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=max(1, int(cfg.SGE_NGPU) or 1) if gpu else 0,
        h_vmem=cfg.SGE_H_VMEM,
        queue=cfg.SGE_QUEUE,
    )
    return overrides.apply(base) if overrides is not None else base


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
    models: Path | None = None,
    extra_env: dict[str, str] | None = None,
    overrides: SgeResourceOverrides | None = None,
) -> str | None:
    """Submit (or, if *emit* is given, append to that script file handle instead of submitting) one
    SGE stage running *python_cmd* under Singularity, using default resources/binds for *gpu*.

    *models* adds a ``-B`` bind for a weights directory the tool needs, and
    *extra_env* exports additional variables inside the container — a tool whose
    weights live outside ``image_tools`` (TotalSegmentator) needs both."""
    # No ensure_dirs() here: these are cluster paths and cluster storage is not mounted
    # locally. submit_stage() creates them on the direct (on-cluster) submission path, and
    # write_script_header() emits the mkdir -p for the emit path.
    paths = cluster_paths(data_root=data_root, output_root=output_root, models=models)
    binds = SingularityBinds()
    env = dict(sge_backend_env(binds.src, "cupy" if gpu else "numpy"))
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    spec = StageSpec(
        job_name=job_name,
        python_cmd=python_cmd,
        resources=default_resources(gpu=gpu, overrides=overrides),
        binds=binds,
        use_nv=gpu,
        extra_env=env,
    )
    return submit_stage(spec, paths, emit=emit)


def emit_submit_script(
    *,
    script_path: Path,
    stages: list[tuple[str, str]],
    data_root: Path,
    output_root: Path,
    gpu: bool = False,
    models: Path | None = None,
    extra_env: dict[str, str] | None = None,
    overrides: SgeResourceOverrides | None = None,
) -> Path:
    """Write a qsub shell script at *script_path* containing one job stage per ``(job_name,
    python_cmd)`` in *stages*, sharing the header, cluster paths and resource request."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    paths = cluster_paths(data_root=data_root, output_root=output_root, models=models)
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
                models=models,
                extra_env=extra_env,
                overrides=overrides,
            )
    return script_path


def default_emit_path(tool: str, subcommand: str) -> Path:
    """Timestamped *local* path for an emitted submit script for *tool*/*subcommand*.

    Deliberately not ``cfg.SGE_SCRIPTS_DIR``: that is a cluster directory, and cluster storage
    is not mounted on the workstation, so creating it here would make a local directory of the
    same name and the cluster would never see the script. The script is written locally and
    published with :func:`nvitk.cluster.sge_remote.publish_sge_driver_script`.
    """
    from nvitk.cluster.sge_remote import local_sge_staging_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return local_sge_staging_dir() / f"submit_{cfg.SGE_JOB_PREFIX}_{tool}_{subcommand}_{ts}.sh"
