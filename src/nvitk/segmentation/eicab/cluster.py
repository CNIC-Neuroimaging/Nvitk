"""SGE submission for eICAB via nvitk ``StageSpec`` + pipeline Singularity image."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TextIO

from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    SgeResources,
    StageSpec,
    submit_stage,
)

from . import config as cfg


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


def _dedupe_binds(pairs: list[tuple[Path, str]]) -> tuple[tuple[Path, str], ...]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Path, str]] = []
    for h, m in pairs:
        key = (str(h.resolve()), m)
        if key in seen:
            continue
        seen.add(key)
        out.append((h, m))
    return tuple(out)


def _map_host_to_outer_container_path(
    host_file: Path,
    roots: tuple[tuple[Path, str], ...],
) -> str:
    """Path as seen inside the outer pipeline container (under standard binds)."""
    h = host_file.resolve()
    for root, prefix in roots:
        try:
            rel = h.relative_to(root.resolve())
            return f"{prefix}{rel.as_posix()}"
        except ValueError:
            continue
    return str(h)


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
    binds: SingularityBinds,
) -> str:
    module = "nvitk.segmentation.eicab.run_job"
    script = f"{binds.src}{module.replace('.', '/')}.py"
    parts: list[str] = [
        "python",
        shlex.quote(script),
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
    return " ".join(parts)


def cluster_paths(
    *,
    src_dir: Path,
    pipeline_container: Path,
    input_root: Path,
    output_root: Path,
    log_dir: Path,
    err_dir: Path,
) -> ClusterPaths:
    """eICAB has no separate model weights bind (weights live inside the eICAB .sif)."""
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
    resources: SgeResources,
    hold_jid: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit one eICAB job (or emit a bash block when *emit* is set)."""
    binds = SingularityBinds()
    rel_in = _require_under(input_nifti, input_root, "input")
    rel_out = _require_under(output_dir, output_root, "output")

    tmp_res = tmp_dir.resolve()
    out_root_res = output_root.resolve()
    if _is_under(tmp_res, out_root_res):
        rel_tmp = _require_under(tmp_dir, output_root, "tmp")
        c_tmp = f"{binds.output}{rel_tmp.as_posix()}"
    else:
        c_tmp = str(tmp_res)

    extra_binds: list[tuple[Path, str]] = [
        (vasculature_host.resolve(), "/programs/Neuro/vasculature2"),
    ]
    if not _is_under(tmp_res, out_root_res):
        extra_binds.append((tmp_res, str(tmp_res)))

    ec_resolved = eicab_container.resolve()
    mount_roots = (input_root, output_root, src_dir)
    if not any(_is_under(ec_resolved, r) for r in mount_roots):
        ec_parent = ec_resolved.parent
        extra_binds.append((ec_parent, str(ec_parent)))

    extra_host_binds = _dedupe_binds(extra_binds)

    roots_map: tuple[tuple[Path, str], ...] = (
        (input_root.resolve(), binds.data),
        (output_root.resolve(), binds.output),
        (src_dir.resolve(), binds.src),
    )
    c_eicab = _map_host_to_outer_container_path(ec_resolved, roots_map)

    c_in = f"{binds.data}{rel_in.as_posix()}"
    c_out = f"{binds.output}{rel_out.as_posix()}"

    dev = device.lower()
    if dev == "gpu":
        dev = "cuda"
    use_nv = dev == "cuda"

    python_cmd = build_run_job_python_cmd(
        input_container_path=c_in,
        output_container_path=c_out,
        tmp_container_path=c_tmp,
        eicab_container_container_path=c_eicab,
        resolution=resolution,
        device=dev,
        simple_segmentation=simple_segmentation,
        attention=attention,
        keep_aux_outputs=keep_aux_outputs,
        binds=binds,
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
        python_cmd=python_cmd,
        resources=resources,
        binds=binds,
        use_nv=use_nv,
        extra_env={"PYTHONPATH": str(binds.src)},
        extra_host_binds=extra_host_binds,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, dry_run=dry_run, emit=emit)


__all__ = [
    "build_run_job_python_cmd",
    "cluster_paths",
    "submit_eicab_job",
]
