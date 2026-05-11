"""
SGE + Singularity job templating for TotalSegmentator on an HPC cluster.

The :func:`build_qsub_command` helper emits the exact shell command that the
BioImaging CT-PET / Dixon pipelines used to submit via ``qsub``. The Python
pipeline entry points use this when called with ``--submit sge`` instead of
running TotalSegmentator in-process.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class SingularityBinds:
    """Container bind-points used by the PESA-Fat cluster pipeline."""

    src: str = "/nvitk/src/"
    data: str = "/nvitk/data/"
    output: str = "/nvitk/output/"
    models: str = "/nvitk/models/"


@dataclass
class SgeResources:
    """SGE submission resources."""

    project: str = "MCC_GPU"
    account: str = "MCC_GPU"
    ngpu: int = 1
    h_vmem: str = "50G"
    queue: str | None = None


@dataclass
class ClusterPaths:
    """Host-side paths that must exist before submission."""

    src: Path
    container: Path
    models: Path
    input_root: Path
    output_root: Path
    log_dir: Path
    err_dir: Path

    inference_script: str = "nvitk/segmentation/total_segmentator/_inference.sh"
    """Relative path under *src* of the helper wrapping the ``TotalSegmentator`` CLI."""

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.err_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SegmentationJob:
    """One SGE job: segment *input* into *output* with *task* (optionally restricted to *roi_subset*)."""

    job_name: str
    subject_input: Path
    subject_output: Path
    task: str
    roi_subset: Sequence[str] = field(default_factory=list)
    label_mode: str = "multilabel"
    backend: str = "gpu"


def build_singularity_command(
    job: SegmentationJob,
    paths: ClusterPaths,
    binds: SingularityBinds | None = None,
) -> str:
    """
    Build the ``singularity exec --nv ... bash -c '...'`` inner command string.

    The returned string is what BioImaging piped into ``qsub`` via ``echo "$cmd" | qsub ...``.
    """
    binds = binds or SingularityBinds()
    rel_in = job.subject_input.resolve().relative_to(paths.input_root.resolve())
    rel_out = job.subject_output.resolve().relative_to(paths.output_root.resolve())
    bind_input = f"{binds.data}{rel_in.as_posix()}"
    bind_output = f"{binds.output}{rel_out.as_posix()}"
    subset_str = " ".join(job.roi_subset)

    script_path = f"{binds.src}{paths.inference_script}"

    cmd = (
        "singularity exec --nv "
        f"-B {paths.src}:{binds.src} "
        f"-B {paths.input_root}:{binds.data} "
        f"-B {paths.output_root}:{binds.output} "
        f"-B {paths.models}:{binds.models} "
        f"{paths.container} bash -c "
        + shlex.quote(
            f'export TOTALSEG_HOME_DIR="{binds.models}" && bash {script_path} '
            f"--input {bind_input} "
            f"--output {bind_output} "
            f"--task {job.task} "
            f'--subset "{subset_str}" '
            f"--mode {job.label_mode} "
            f"--backend {job.backend}"
        )
    )
    return cmd


def build_qsub_command(
    job: SegmentationJob,
    paths: ClusterPaths,
    resources: SgeResources | None = None,
) -> list[str]:
    """
    Build the ``qsub`` argv for *job*.

    The ``singularity`` command is expected to be piped into ``stdin`` (the same
    pattern the BioImaging scripts used).
    """
    resources = resources or SgeResources()
    log_file = paths.log_dir / f"{job.job_name}.log"
    err_file = paths.err_dir / f"{job.job_name}.err"

    argv = [
        "qsub",
        "-P", resources.project,
        "-terse",
        "-N", job.job_name,
        "-A", resources.account,
        "-l", f"ngpu={resources.ngpu}",
        "-l", f"h_vmem={resources.h_vmem}",
        "-o", str(log_file),
        "-e", str(err_file),
    ]
    if resources.queue:
        argv.extend(["-q", resources.queue])
    return argv


def submit_job(
    job: SegmentationJob,
    paths: ClusterPaths,
    *,
    resources: SgeResources | None = None,
    binds: SingularityBinds | None = None,
    dry_run: bool = False,
) -> str:
    """
    Submit *job* to SGE by piping the Singularity command to ``qsub`` (like the legacy shell scripts).

    Returns the SGE job id (or ``'DRY_RUN'`` when *dry_run* is True).
    """
    if dry_run:
        build_singularity_command(job, paths, binds)
        build_qsub_command(job, paths, resources)
        return "DRY_RUN"

    paths.ensure_dirs()
    inner_cmd = build_singularity_command(job, paths, binds)
    qsub_cmd = build_qsub_command(job, paths, resources)

    result = subprocess.run(
        qsub_cmd,
        input=inner_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def submit_jobs(
    jobs: Iterable[SegmentationJob],
    paths: ClusterPaths,
    *,
    resources: SgeResources | None = None,
    binds: SingularityBinds | None = None,
    dry_run: bool = False,
) -> list[str]:
    return [submit_job(j, paths, resources=resources, binds=binds, dry_run=dry_run) for j in jobs]


__all__ = [
    "SingularityBinds",
    "SgeResources",
    "ClusterPaths",
    "SegmentationJob",
    "build_singularity_command",
    "build_qsub_command",
    "submit_job",
    "submit_jobs",
]
