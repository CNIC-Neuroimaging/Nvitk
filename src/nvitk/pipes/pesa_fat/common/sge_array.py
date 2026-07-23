"""Shared SGE array-job helpers for PESA-Fat CT-PET / Dixon pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

from nvitk.cluster.sge import (
    ArrayTaskSpec,
    ClusterPaths,
    SgeResources,
    StageSpec,
    build_singularity_command,
    submit_array_job,
)
from nvitk.pipes.pesa_fat.common.paths import BatchLayout

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sge_pesa_array_resources(
    stages_sel: list[str],
    *,
    device: str,
    project: str,
    account: str,
    queue: str | None,
    h_vmem_stage1: str,
    h_vmem_cpu: str,
    ngpu: int,
    cpu_ngpu: int,
) -> tuple[SgeResources, bool]:
    """Padded resources for a per-subject PESA-Fat stage1–3 array job.

    Pads ``h_vmem`` to stage1 when stage1 is among *stages_sel*; otherwise uses
    the CPU-stage mem. ``ngpu`` / ``use_nv`` follow *device* (same as today's
    per-stage behaviour when ``device=gpu``).
    """
    want_gpu = str(device).strip().lower() == "gpu"
    h_vmem = h_vmem_stage1 if "stage1" in stages_sel else h_vmem_cpu
    return (
        SgeResources(
            project=project,
            account=account,
            ngpu=int(ngpu) if want_gpu else int(cpu_ngpu),
            h_vmem=h_vmem,
            queue=queue,
        ),
        True,  # use_nv matches historical StageSpec.use_nv=True for pipe stages
    )


def array_marker_dir(lay: BatchLayout, job_prefix: str, subject: str) -> Path:
    """Host path for per-subject array done-markers."""
    safe_prefix = _SAFE.sub("_", job_prefix).strip("._-") or "pipe"
    safe_subj = _SAFE.sub("_", subject).strip("._-") or "subject"
    return lay.results_dir / ".sge_array_markers" / safe_prefix / safe_subj


def submit_subject_stage_array(
    *,
    subject: str,
    job_prefix: str,
    stages_sel: list[str],
    stage_specs: list[StageSpec],
    paths: ClusterPaths,
    resources: SgeResources,
    use_nv: bool = True,
    base_hold: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
    marker_dir: Path,
) -> list[str]:
    """Submit one SGE array job for *stages_sel*; return ``[jid]``.

    Each :class:`StageSpec` is turned into a singularity shell command as one
    array task. Aggregate / QC callers may hold on the single returned jid.
    """
    if not stages_sel:
        return []
    if len(stage_specs) != len(stages_sel):
        raise ValueError("stage_specs length must match stages_sel")

    tasks = [
        ArrayTaskSpec(
            stage_id=stage_id,
            shell_cmd=build_singularity_command(spec, paths),
        )
        for stage_id, spec in zip(stages_sel, stage_specs, strict=True)
    ]
    safe_subj = _SAFE.sub("_", subject).strip("._-") or "subject"
    job_name = f"{job_prefix}_{safe_subj}"[:63]
    jid = submit_array_job(
        job_name=job_name,
        resources=resources,
        paths=paths,
        tasks=tasks,
        marker_dir=marker_dir,
        task_concurrency=1,
        use_nv=use_nv,
        hold_jid=base_hold,
        dry_run=dry_run,
        emit=emit,
    )
    return [jid]


__all__ = [
    "array_marker_dir",
    "sge_pesa_array_resources",
    "submit_subject_stage_array",
]
