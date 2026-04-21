"""
Dedicated ``nvitk-totalseg`` CLI for the TotalSegmentator integration.

Runs either locally (thin wrapper over :func:`run_totalsegmentator`) or on an
SGE cluster by submitting a Singularity-wrapped ``TotalSegmentator`` command.

Examples
--------
Run locally::

    nvitk-totalseg \
        --input ct.nii.gz --output out/ \
        --task total --roi-subset vertebrae_L4 --roi-subset vertebrae_L3

Submit to SGE (dry-run)::

    nvitk-totalseg \
        --input /nfs/input/SUBJ --output /nfs/output/SUBJ \
        --task total_mr --backend sge \
        --container /path/to/TotalSegmentator.sif \
        --models-dir /data_local/ai_models/imaging/TotalSegmentator/v2.0.0/ \
        --src-dir /nfs/src --input-root /nfs/input --output-root /nfs/output \
        --log-dir /nfs/logs --err-dir /nfs/errs \
        --project GPU --account Prod --ngpu 1 --h-vmem 50G \
        --job-name totalseg_SUBJ --dry-run
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger

from .class_maps import AVAILABLE_TASKS
from .cluster import (
    ClusterPaths,
    SegmentationJob,
    SgeResources,
    SingularityBinds,
    submit_job,
)
from .runner import run_totalsegmentator


log = Logger()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    required=True,
    help="NIfTI file or directory with subject image(s).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the multilabel segmentation.",
)
@click.option(
    "--task",
    type=click.Choice(list(AVAILABLE_TASKS), case_sensitive=True),
    required=True,
    help="TotalSegmentator task name.",
)
@click.option(
    "--roi-subset",
    "roi_subset",
    multiple=True,
    help="Restrict to a subset of ROIs (repeat the flag for multiple ROIs).",
)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu"], case_sensitive=False),
    default="gpu",
    show_default=True,
    help="Device for local inference (ignored for --backend sge).",
)
@click.option(
    "--backend",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Run locally via TotalSegmentator CLI or submit a Singularity/qsub job.",
)
@click.option(
    "--multilabel/--per-class",
    "multilabel",
    default=True,
    show_default=True,
    help="Emit a single multilabel NIfTI or one file per class.",
)
@click.option(
    "--statistics/--no-statistics",
    "statistics",
    default=True,
    show_default=True,
    help="Pass --statistics to TotalSegmentator.",
)
@click.option("--fast", is_flag=True, help="Pass --fast to TotalSegmentator.")
@click.option("--preview", is_flag=True, help="Pass --preview to TotalSegmentator.")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional TOTALSEG_HOME_DIR for the subprocess (local backend).",
)
# ── SGE options ──
@click.option(
    "--container",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Path to the Singularity container with TotalSegmentator.",
)
@click.option(
    "--models-dir",
    "models_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path bind-mounted as the models directory inside the container.",
)
@click.option(
    "--src-dir",
    "src_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path to the nvitk source tree (mounted at /PESAFat/src/).",
)
@click.option(
    "--input-root",
    "input_root",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path mounted as the data root (parent of --input).",
)
@click.option(
    "--output-root",
    "output_root",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path mounted as the output root (parent of --output).",
)
@click.option(
    "--log-dir",
    "log_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path for qsub stdout logs (.log files).",
)
@click.option(
    "--err-dir",
    "err_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path for qsub stderr logs (.err files).",
)
@click.option("--project", default="GPU", show_default=True, help="(sge) qsub -P project.")
@click.option("--account", default="Prod", show_default=True, help="(sge) qsub -A account.")
@click.option("--ngpu", default=1, show_default=True, type=int, help="(sge) qsub -l ngpu.")
@click.option("--h-vmem", default="50G", show_default=True, help="(sge) qsub -l h_vmem.")
@click.option("--queue", default=None, help="(sge) optional qsub -q queue.")
@click.option("--job-name", default=None, help="(sge) SGE job name (defaults to totalseg_<task>).")
@click.option("--dry-run", is_flag=True, help="(sge) Build and echo the command but do not submit.")
def main(
    input_path: Path,
    output_path: Path,
    task: str,
    roi_subset: tuple[str, ...],
    device: str,
    backend: str,
    multilabel: bool,
    statistics: bool,
    fast: bool,
    preview: bool,
    model_dir: Path | None,
    container: Path | None,
    models_dir: Path | None,
    src_dir: Path | None,
    input_root: Path | None,
    output_root: Path | None,
    log_dir: Path | None,
    err_dir: Path | None,
    project: str,
    account: str,
    ngpu: int,
    h_vmem: str,
    queue: str | None,
    job_name: str | None,
    dry_run: bool,
) -> None:
    """Run TotalSegmentator either locally or on SGE."""
    Logger(level="INFO")

    backend_l = backend.lower()
    if backend_l == "local":
        log.info(
            f"Running TotalSegmentator locally: input={input_path} output={output_path} "
            f"task={task} device={device}"
        )
        result = run_totalsegmentator(
            input=input_path,
            output=output_path,
            task=task,
            device=device.lower(),
            roi_subset=list(roi_subset) if roi_subset else None,
            multilabel=multilabel,
            statistics=statistics,
            fast=fast,
            preview=preview,
            model_dir=model_dir,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"TotalSegmentator exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr}"
            )
        return

    missing = [
        name
        for name, val in (
            ("--container", container),
            ("--models-dir", models_dir),
            ("--src-dir", src_dir),
            ("--input-root", input_root),
            ("--output-root", output_root),
            ("--log-dir", log_dir),
            ("--err-dir", err_dir),
        )
        if val is None
    ]
    if missing:
        raise click.UsageError(
            "--backend sge requires: " + ", ".join(missing)
        )

    paths = ClusterPaths(
        src=src_dir,  # type: ignore[arg-type]
        container=container,  # type: ignore[arg-type]
        models=models_dir,  # type: ignore[arg-type]
        input_root=input_root,  # type: ignore[arg-type]
        output_root=output_root,  # type: ignore[arg-type]
        log_dir=log_dir,  # type: ignore[arg-type]
        err_dir=err_dir,  # type: ignore[arg-type]
    )
    resources = SgeResources(
        project=project,
        account=account,
        ngpu=ngpu,
        h_vmem=h_vmem,
        queue=queue,
    )
    binds = SingularityBinds()
    job = SegmentationJob(
        job_name=job_name or f"totalseg_{task}",
        subject_input=input_path,
        subject_output=output_path,
        task=task,
        roi_subset=list(roi_subset),
        label_mode="multilabel" if multilabel else "single_labels",
        backend="gpu" if device.lower() == "gpu" else "cpu",
    )

    job_id = submit_job(
        job,
        paths,
        resources=resources,
        binds=binds,
        dry_run=dry_run,
    )
    if dry_run:
        # Show what would have been submitted.
        from .cluster import build_qsub_command, build_singularity_command

        qsub_cmd = build_qsub_command(job, paths, resources)
        inner_cmd = build_singularity_command(job, paths, binds)
        click.echo("=== qsub ===")
        click.echo(" ".join(qsub_cmd))
        click.echo("=== singularity (piped to qsub stdin) ===")
        click.echo(inner_cmd)
        click.echo("=== job id ===")
    click.echo(job_id)


if __name__ == "__main__":
    main()  # pragma: no cover
