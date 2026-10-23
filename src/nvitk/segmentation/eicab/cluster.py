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

from .runner import (
    build_eicab_singularity_shell_cmd,
    metric_scratch_cleanup_shell,
    metric_scratch_prep_shell,
)


def _require_under(child: Path, root: Path, label: str) -> Path:
    """Resolve *child* and assert it lives under *root* (path-traversal guard); return it."""
    c = child.resolve()
    r = root.resolve()
    try:
        return c.relative_to(r)
    except ValueError as exc:
        raise ValueError(
            f"{label} path {c} must be under {r} for cluster bind mounts."
        ) from exc


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


def build_eicab_prune_shell_cmd(
    *,
    output_dir: Path,
    output_root: Path,
    pipeline_container: Path,
    src_dir: Path,
    data_root: Path,
) -> str:
    """Prune auxiliary eICAB outputs inside the nvitk container."""
    binds = SingularityBinds()
    rel_out = _require_under(output_dir, output_root, "output")
    c_out = f"{binds.output}{rel_out.as_posix()}"
    py_cmd = (
        "from pathlib import Path; "
        "from nvitk.segmentation.eicab.runner import segmentation_outputs_to_keep, "
        "prune_eicab_outputs; "
        f"p=Path({json.dumps(c_out)}); "
        "k=segmentation_outputs_to_keep(p); "
        "prune_eicab_outputs(p, keep_aux_outputs=False, keep_paths=k)"
    )
    inner = f"export PYTHONPATH={binds.src} && python -c {shlex.quote(py_cmd)}"
    return build_nvitk_exec_shell_cmd(
        pipeline_container=pipeline_container,
        src_dir=src_dir,
        data_root=data_root,
        output_root=output_root,
        python_cmd=inner,
    )


def build_eicab_postprocess_shell_cmd(
    *,
    output_dir: Path,
    output_root: Path,
    pipeline_container: Path,
    src_dir: Path,
    data_root: Path,
    backend: str,
    siphon: bool = True,
    distal: bool = True,
) -> str:
    """The eICAB post-process step inside the nvitk container (*siphon*: include the ICA siphon
    correction)."""
    binds = SingularityBinds()
    rel_out = _require_under(output_dir, output_root, "output")
    c_out = f"{binds.output}{rel_out.as_posix()}"
    py_cmd = (
        "from pathlib import Path; "
        "from nvitk.pipes.qvtpy.util.eicab.eicab_postprocess import postprocess_eicab_directory; "
        f"postprocess_eicab_directory(Path({json.dumps(c_out)}), siphon={bool(siphon)}, distal={bool(distal)})"
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


def existing_eicab_shell_test(output_dir: Path) -> str:
    """POSIX shell test: true when *output_dir* already holds a non-empty eICAB CW mask and
    ``*_resampled`` TOF (the post-process inputs)."""
    out_q = shlex.quote(str(output_dir.resolve()))
    has = (
        "_nvitk_has() { for _f in \"$@\"; do [ -s \"$_f\" ] && return 0; done; return 1; }"
    )
    return (
        f"{has}; _nvitk_has {out_q}/*_eICAB_CW.nii.gz {out_q}/*_eICAB_CW.nii "
        f"&& _nvitk_has {out_q}/*_resampled.nii.gz {out_q}/*_resampled.nii"
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
    pp_siphon: bool = True,
    pp_distal: bool = True,
    thread_limit: int | None = None,
    sge_pe_smp: int | None = None,
    local_metric_scratch: bool = False,
    whole_brain: bool = True,
    reuse_existing: bool = False,
) -> str:
    """Full stage1 host command: eICAB ``singularity run`` + optional nvitk follow-ups.

    *whole_brain* False stops eICAB once the CW mask is written (no WB flood fill).
    *reuse_existing* checks at run time for an existing CW mask + ``TOF_resampled`` in
    *output_dir*: when found, eICAB and the prune are skipped and only the post-process runs.
    Either branch is announced on the job's stdout.
    """
    dev = device.lower()
    if dev == "gpu":
        dev = "cuda"

    if sge_pe_smp:
        cpu_expr = f"${{NSLOTS:-{sge_pe_smp}}}"
    elif thread_limit:
        cpu_expr = str(thread_limit)
    else:
        cpu_expr = None
    out_q = shlex.quote(str(output_dir.resolve()))
    tmp_q = shlex.quote(str(tmp_dir.resolve()))
    # Wipe stale VED scale files from prior failed runs (shape-mismatch source).
    # ved_cwd on NFS is only a fallback; with local scratch, CWD is rebound from /data_tmp.
    prep = (
        f"mkdir -p {out_q} && mkdir -p {tmp_q} "
        f"&& rm -rf {out_q}/metric_space "
        f"&& mkdir -p {out_q}/metric_space "
        f"&& rm -rf {tmp_q}/ved_cwd && mkdir -p {tmp_q}/ved_cwd"
    )
    scratch_root: str | None = None
    if local_metric_scratch:
        from .config import EICAB_METRIC_SCRATCH_ROOT

        scratch_root = EICAB_METRIC_SCRATCH_ROOT
        prep = f"{prep} && {metric_scratch_prep_shell(scratch_root)}"
    steps: list[str] = [
        prep,
        build_eicab_singularity_shell_cmd(
            input_nifti,
            output_dir,
            tmp_dir=tmp_dir,
            container=eicab_container,
            resolution=resolution,
            simple_segmentation=simple_segmentation,
            attention=attention,
            device=dev,
            vasculature_host_path=vasculature_host,
            cpu_limit_shell_expr=cpu_expr,
            nvitk_src_dir=src_dir,
            local_metric_scratch=local_metric_scratch,
            metric_scratch_root=scratch_root,
            cw_only=not whole_brain,
        )
    ]
    if not keep_aux_outputs:
        steps.append(
            build_eicab_prune_shell_cmd(
                output_dir=output_dir,
                output_root=output_root,
                pipeline_container=pipeline_container,
                src_dir=src_dir,
                data_root=input_root,
            )
        )
    scope = "CW only, no whole-brain flood fill" if not whole_brain else "CW + whole brain"
    eicab_msg = shlex.quote(f"[nvitk stage1] running eICAB ({scope}) -> {output_dir}")
    if reuse_existing:
        found_msg = shlex.quote(
            f"[nvitk stage1] existing eICAB CW mask + TOF_resampled in {output_dir}: "
            "eICAB skipped, post-process only"
        )
        missing_msg = shlex.quote(
            f"[nvitk stage1] no eICAB CW mask in {output_dir}: falling back to eICAB ({scope})"
        )
        steps = [
            f"if {existing_eicab_shell_test(output_dir)}; then echo {found_msg}; "
            f"else echo {missing_msg} && {' && '.join(steps)}; fi"
        ]
    else:
        steps.insert(0, f"echo {eicab_msg}")
    if post_process_eicab:
        steps.append(
            build_eicab_postprocess_shell_cmd(
                output_dir=output_dir,
                output_root=output_root,
                pipeline_container=pipeline_container,
                src_dir=src_dir,
                data_root=input_root,
                backend=backend,
                siphon=pp_siphon,
                distal=pp_distal,
            )
        )
    body = " && ".join(steps)
    if not local_metric_scratch or scratch_root is None:
        return body
    # Always remove only this job's nvitk_eicab_<JOB_ID> dir (success or failure).
    cleanup = metric_scratch_cleanup_shell(scratch_root)
    return (
        "{ "
        f"{body} ; "
        "_nvitk_eicab_rc=$? ; "
        f"{cleanup} ; "
        "exit $_nvitk_eicab_rc ; "
        "}"
    )


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
    pp_siphon: bool = True,
    pp_distal: bool = True,
    backend: str = "gpu",
    resources: SgeResources,
    thread_limit: int | None = None,
    sge_pe_smp: int | None = None,
    local_metric_scratch: bool = False,
    whole_brain: bool = True,
    reuse_existing: bool = False,
    hold_jid: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit one eICAB job on the cluster host (or emit a bash block when *emit* is set).

    *whole_brain* / *reuse_existing*: see :func:`build_eicab_host_shell_cmd`.
    """
    dev = device.lower()
    if dev == "gpu":
        dev = "cuda"
    use_nv = dev == "cuda"

    job_resources = resources
    if sge_pe_smp and not resources.pe_smp:
        job_resources = SgeResources(
            project=resources.project,
            account=resources.account,
            ngpu=resources.ngpu,
            h_vmem=resources.h_vmem,
            queue=resources.queue,
            pe_smp=sge_pe_smp,
        )

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
        pp_siphon=pp_siphon,
        pp_distal=pp_distal,
        backend=backend,
        thread_limit=thread_limit,
        sge_pe_smp=sge_pe_smp,
        local_metric_scratch=local_metric_scratch,
        whole_brain=whole_brain,
        reuse_existing=reuse_existing,
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
        resources=job_resources,
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
    "build_eicab_host_shell_cmd",
    "existing_eicab_shell_test",
    "build_run_job_python_cmd",
    "cluster_paths",
    "submit_eicab_job",
]
