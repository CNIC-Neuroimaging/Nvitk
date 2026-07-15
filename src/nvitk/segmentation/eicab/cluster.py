"""SGE submission for eICAB (host ``singularity run``, optional nvitk post-steps)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TextIO

from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    SgeResources,
    StageSpec,
    build_singularity_command,
    python_module_argv,
    submit_host_stage,
)

from .runner import build_eicab_singularity_argv

# In-container mount for eICAB work dirs outside the standard /nvitk/output/ tree.
EICAB_WORK_MOUNT = "/eicab_work"


def _require_under(child: Path, root: Path, label: str) -> Path:
    c = child.resolve()
    r = root.resolve()
    try:
        return c.relative_to(r)
    except ValueError as exc:
        raise ValueError(
            f"{label} path {c} must be under {r} for cluster bind mounts."
        ) from exc


def _is_under(child: Path, root: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_run_job_python_cmd(
    *,
    input_container_path: str,
    output_container_path: str,
    tmp_container_path: str,
    eicab_container_container_path: str,
    resolution: float,
    device: str,
    simple_segmentation: bool,
    attention: bool,
    keep_aux_outputs: bool,
    post_process_eicab: bool,
    backend: str,
    binds: SingularityBinds,
) -> str:
    """Legacy in-container ``run_job`` argv string (nested Singularity; prefer host path)."""
    module = "nvitk.segmentation.eicab.run_job"
    parts: list[str] = [
        *python_module_argv(module),
        "--input",
        shlex.quote(input_container_path),
        "--output",
        shlex.quote(output_container_path),
        "--tmp",
        shlex.quote(tmp_container_path),
        "--eicab-container",
        shlex.quote(eicab_container_container_path),
        "--resolution",
        str(resolution),
        "--device",
        device,
    ]
    if simple_segmentation:
        parts.append("--simple-segmentation")
    if attention:
        parts.append("--attention")
    if keep_aux_outputs:
        parts.append("--keep-aux-outputs")
    if post_process_eicab:
        parts.append("--post-process-eicab")
    else:
        parts.append("--no-post-process-eicab")
    parts.extend(["--backend", shlex.quote(str(backend).strip().lower())])
    return " ".join(parts)


def build_nvitk_exec_shell_cmd(
    *,
    pipeline_container: Path,
    src_dir: Path,
    data_root: Path,
    output_root: Path,
    python_cmd: str,
    use_nv: bool = False,
) -> str:
    """``singularity exec`` nvitk image on the cluster host (Python worker, no nesting)."""
    binds = SingularityBinds()
    spec = StageSpec(
        job_name="_nvitk_exec",
        python_cmd=python_cmd,
        binds=binds,
        use_nv=use_nv,
        extra_env={"PYTHONPATH": binds.src},
    )
    paths = ClusterPaths(
        src=src_dir,
        container=pipeline_container,
        models=None,
        data_root=data_root,
        output_root=output_root,
        log_dir=Path("/tmp"),
        err_dir=Path("/tmp"),
    )
    return build_singularity_command(spec, paths)


def build_nvitk_exec_on_work_dir(
    *,
    work_dir: Path,
    pipeline_container: Path,
    src_dir: Path,
    python_cmd: str,
    use_nv: bool = False,
) -> str:
    """``singularity exec`` nvitk with *work_dir* bound at :data:`EICAB_WORK_MOUNT`."""
    binds = SingularityBinds()
    inner = f"export PYTHONPATH={binds.src} && {python_cmd}".strip()
    nv = "--nv " if use_nv else ""
    wd = work_dir.resolve()
    parts: list[str] = [
        f"singularity exec {nv}",
        f"-B {shlex.quote(str(src_dir.resolve()))}:{shlex.quote(binds.src)} ",
        f"-B {shlex.quote(str(wd))}:{shlex.quote(EICAB_WORK_MOUNT)} ",
        f"{shlex.quote(str(pipeline_container.resolve()))} bash -c ",
        shlex.quote(inner),
    ]
    return "".join(parts)


def build_eicab_sync_shell_cmd(*, scratch_dir: Path, final_dir: Path) -> str:
    """Copy finished eICAB outputs from cluster scratch to the NFS results tree."""
    scratch = shlex.quote(str(scratch_dir.resolve()))
    final = shlex.quote(str(final_dir.resolve()))
    return f"mkdir -p {final} && rsync -a {scratch}/ {final}/"


def build_eicab_prune_shell_cmd(
    *,
    work_dir: Path,
    output_root: Path,
    pipeline_container: Path,
    src_dir: Path,
    data_root: Path,
) -> str:
    """Prune auxiliary eICAB outputs inside the nvitk container."""
    binds = SingularityBinds()
    py_tpl = (
        "from pathlib import Path; "
        "from nvitk.segmentation.eicab.runner import segmentation_outputs_to_keep, "
        "prune_eicab_outputs; "
        f"p=Path({{c_out}}); "
        "k=segmentation_outputs_to_keep(p); "
        "prune_eicab_outputs(p, keep_aux_outputs=False, keep_paths=k)"
    )
    if _is_under(work_dir, output_root):
        rel_out = _require_under(work_dir, output_root, "output")
        c_out = json.dumps(f"{binds.output}{rel_out.as_posix()}")
        py_cmd = py_tpl.format(c_out=c_out)
        inner = f"export PYTHONPATH={binds.src} && python -c {shlex.quote(py_cmd)}"
        return build_nvitk_exec_shell_cmd(
            pipeline_container=pipeline_container,
            src_dir=src_dir,
            data_root=data_root,
            output_root=output_root,
            python_cmd=inner,
        )
    c_out = json.dumps(EICAB_WORK_MOUNT)
    py_cmd = py_tpl.format(c_out=c_out)
    return build_nvitk_exec_on_work_dir(
        work_dir=work_dir,
        pipeline_container=pipeline_container,
        src_dir=src_dir,
        python_cmd=f"python -c {shlex.quote(py_cmd)}",
    )


def build_eicab_postprocess_shell_cmd(
    *,
    work_dir: Path,
    output_root: Path,
    pipeline_container: Path,
    src_dir: Path,
    data_root: Path,
    backend: str,
) -> str:
    """ICA post-process step inside the nvitk container."""
    binds = SingularityBinds()
    if _is_under(work_dir, output_root):
        rel_out = _require_under(work_dir, output_root, "output")
        c_out = json.dumps(f"{binds.output}{rel_out.as_posix()}")
        py_cmd = (
            "from pathlib import Path; "
            "from nvitk.pipes.qvtpy.util.eicab_postprocess import postprocess_eicab_directory; "
            f"postprocess_eicab_directory(Path({c_out}))"
        )
        inner = (
            f"export PYTHONPATH={binds.src} && "
            f"export NVITK_BACKEND={shlex.quote(str(backend).strip().lower())} && "
            f"python -c {shlex.quote(py_cmd)}"
        )
        return build_nvitk_exec_shell_cmd(
            pipeline_container=pipeline_container,
            src_dir=src_dir,
            data_root=data_root,
            output_root=output_root,
            python_cmd=inner,
        )
    c_out = json.dumps(EICAB_WORK_MOUNT)
    py_cmd = (
        "from pathlib import Path; "
        "from nvitk.pipes.qvtpy.util.eicab_postprocess import postprocess_eicab_directory; "
        f"postprocess_eicab_directory(Path({c_out}))"
    )
    inner = (
        f"export NVITK_BACKEND={shlex.quote(str(backend).strip().lower())} && "
        f"python -c {shlex.quote(py_cmd)}"
    )
    return build_nvitk_exec_on_work_dir(
        work_dir=work_dir,
        pipeline_container=pipeline_container,
        src_dir=src_dir,
        python_cmd=inner,
    )


def build_eicab_host_shell_cmd(
    *,
    input_nifti: Path,
    output_dir: Path,
    tmp_dir: Path,
    eicab_container: Path,
    src_dir: Path,
    pipeline_container: Path,
    input_root: Path,
    output_root: Path,
    vasculature_host: Path,
    resolution: float,
    device: str,
    simple_segmentation: bool,
    attention: bool,
    keep_aux_outputs: bool,
    post_process_eicab: bool,
    backend: str,
    scratch_output_dir: Path | None = None,
) -> str:
    """Full stage1 host command: eICAB ``singularity run`` + optional nvitk follow-ups.

    When *scratch_output_dir* is set, eICAB inference and follow-up steps run on
    cluster-local scratch; finished outputs are rsynced to *output_dir* (NFS) at
    the end. The job must execute entirely on one compute node (normal SGE behaviour).
    """
    dev = device.lower()
    if dev == "gpu":
        dev = "cuda"

    final_dir = output_dir.resolve()
    work_dir = scratch_output_dir.resolve() if scratch_output_dir is not None else final_dir
    work_tmp = tmp_dir.resolve()

    prep = (
        f"mkdir -p {shlex.quote(str(work_dir))} "
        f"&& mkdir -p {shlex.quote(str(work_tmp))} "
        f"&& mkdir -p {shlex.quote(str(final_dir))}"
    )
    steps: list[str] = [
        prep,
        shlex.join(
            build_eicab_singularity_argv(
                input_nifti,
                work_dir,
                tmp_dir=work_tmp,
                container=eicab_container,
                resolution=resolution,
                simple_segmentation=simple_segmentation,
                attention=attention,
                device=dev,
                vasculature_host_path=vasculature_host,
            )
        ),
    ]
    if not keep_aux_outputs:
        steps.append(
            build_eicab_prune_shell_cmd(
                work_dir=work_dir,
                output_root=output_root,
                pipeline_container=pipeline_container,
                src_dir=src_dir,
                data_root=input_root,
            )
        )
    if post_process_eicab:
        steps.append(
            build_eicab_postprocess_shell_cmd(
                work_dir=work_dir,
                output_root=output_root,
                pipeline_container=pipeline_container,
                src_dir=src_dir,
                data_root=input_root,
                backend=backend,
            )
        )
    if scratch_output_dir is not None:
        steps.append(
            build_eicab_sync_shell_cmd(scratch_dir=work_dir, final_dir=final_dir)
        )
        steps.append(f"rm -rf {shlex.quote(str(work_dir))}")
    return " && ".join(steps)


def cluster_paths(
    *,
    src_dir: Path,
    pipeline_container: Path,
    input_root: Path,
    output_root: Path,
    log_dir: Path,
    err_dir: Path,
) -> ClusterPaths:
    """Cluster log/bind metadata for eICAB SGE jobs."""
    return ClusterPaths(
        src=src_dir,
        container=pipeline_container,
        models=None,
        data_root=input_root,
        output_root=output_root,
        log_dir=log_dir,
        err_dir=err_dir,
    )


def submit_eicab_job(
    *,
    job_name: str,
    input_nifti: Path,
    output_dir: Path,
    tmp_dir: Path,
    eicab_container: Path,
    src_dir: Path,
    pipeline_container: Path,
    input_root: Path,
    output_root: Path,
    vasculature_host: Path,
    log_dir: Path,
    err_dir: Path,
    resolution: float,
    device: str,
    simple_segmentation: bool,
    attention: bool,
    keep_aux_outputs: bool,
    post_process_eicab: bool = True,
    backend: str = "gpu",
    resources: SgeResources,
    scratch_output_dir: Path | None = None,
    hold_jid: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit one eICAB job on the cluster host (or emit a bash block when *emit* is set)."""
    dev = device.lower()
    if dev == "gpu":
        dev = "cuda"
    use_nv = dev == "cuda"

    host_shell_cmd = build_eicab_host_shell_cmd(
        input_nifti=input_nifti.resolve(),
        output_dir=output_dir.resolve(),
        tmp_dir=tmp_dir.resolve(),
        eicab_container=eicab_container.resolve(),
        src_dir=src_dir.resolve(),
        pipeline_container=pipeline_container.resolve(),
        input_root=input_root.resolve(),
        output_root=output_root.resolve(),
        vasculature_host=vasculature_host.resolve(),
        resolution=resolution,
        device=device,
        simple_segmentation=simple_segmentation,
        attention=attention,
        keep_aux_outputs=keep_aux_outputs,
        post_process_eicab=post_process_eicab,
        backend=backend,
        scratch_output_dir=scratch_output_dir,
    )

    paths = cluster_paths(
        src_dir=src_dir,
        pipeline_container=pipeline_container,
        input_root=input_root,
        output_root=output_root,
        log_dir=log_dir,
        err_dir=err_dir,
    )

    spec = StageSpec(
        job_name=job_name,
        python_cmd="",
        resources=resources,
        use_nv=use_nv,
    )
    return submit_host_stage(
        spec,
        paths,
        host_shell_cmd,
        hold_jid=hold_jid,
        dry_run=dry_run,
        emit=emit,
    )


__all__ = [
    "EICAB_WORK_MOUNT",
    "build_eicab_host_shell_cmd",
    "build_eicab_sync_shell_cmd",
    "build_run_job_python_cmd",
    "cluster_paths",
    "submit_eicab_job",
]
