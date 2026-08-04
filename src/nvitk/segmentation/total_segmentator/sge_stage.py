"""Submit TotalSegmentator on SGE using PESA-Fat :class:`StageSpec` (emit script or qsub)."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TextIO

from nvitk.cluster.sge import (
    ClusterPaths as PesaClusterPaths,
    SingularityBinds,
    SgeResources,
    StageSpec,
    submit_stage,
)

from .cluster import ClusterPaths, SegmentationJob


def _require_under(path: Path, root: Path, label: str) -> Path:
    """Resolve *path* and assert it lives under *root* (path-traversal guard); return it."""
    p = path.resolve()
    r = root.resolve()
    try:
        return p.relative_to(r)
    except ValueError as exc:
        raise ValueError(
            f"{label} ({p}) must be under mount root {r} for SGE binds."
        ) from exc


def build_inference_python_cmd(
    job: SegmentationJob,
    paths: ClusterPaths,
    binds: SingularityBinds,
) -> str:
    """``bash`` wrapper command executed inside the outer Singularity image."""
    rel_in = _require_under(job.subject_input, paths.input_root, "--input")
    rel_out = _require_under(job.subject_output, paths.output_root, "--output")
    script_path = f"{binds.src}nvitk/segmentation/total_segmentator/_inference.sh"
    subset_str = " ".join(job.roi_subset)
    return (
        f"bash {shlex.quote(script_path)} "
        f"--input {shlex.quote(f'{binds.data}{rel_in.as_posix()}')} "
        f"--output {shlex.quote(f'{binds.output}{rel_out.as_posix()}')} "
        f"--task {shlex.quote(job.task)} "
        f"--subset {shlex.quote(subset_str)} "
        f"--mode {shlex.quote(job.label_mode)} "
        f"--backend {shlex.quote(job.backend)}"
    )


def to_pesa_cluster_paths(paths: ClusterPaths) -> PesaClusterPaths:
    """Adapt TotalSegmentator :class:`ClusterPaths` to the PESA pipeline's path bundle."""
    return PesaClusterPaths(
        src=paths.src,
        container=paths.container,
        models=paths.models,
        data_root=paths.input_root,
        output_root=paths.output_root,
        log_dir=paths.log_dir,
        err_dir=paths.err_dir,
    )


def submit_totalsegmentator_stage(
    job: SegmentationJob,
    paths: ClusterPaths,
    *,
    resources: SgeResources,
    hold_jid: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """One TotalSegmentator qsub stage (or emit a bash block)."""
    binds = SingularityBinds()
    python_cmd = build_inference_python_cmd(job, paths, binds)
    pesa_paths = to_pesa_cluster_paths(paths)
    spec = StageSpec(
        job_name=job.job_name,
        python_cmd=python_cmd,
        resources=resources,
        binds=binds,
        use_nv=True,
        extra_env={
            "PYTHONPATH": str(binds.src),
            "TOTALSEG_HOME_DIR": str(binds.models),
        },
    )
    return submit_stage(spec, pesa_paths, hold_jid=hold_jid, dry_run=dry_run, emit=emit)


__all__ = [
    "build_inference_python_cmd",
    "submit_totalsegmentator_stage",
    "to_pesa_cluster_paths",
]
