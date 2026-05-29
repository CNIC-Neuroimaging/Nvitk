"""
Dedicated ``nvitk-totalseg`` CLI for the TotalSegmentator integration.

Runs either locally (thin wrapper over :func:`run_totalsegmentator`) or on an
SGE cluster via ``qsub`` + ``singularity exec`` (PESA-Fat style), optionally
emitting a bash script and running it over SSH.
"""

from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.sge import SgeResources, write_script_header

from .class_maps import AVAILABLE_TASKS
from .cluster import ClusterPaths, SegmentationJob
from . import config as ts_cfg
from .runner import run_totalsegmentator
from .sge_stage import submit_totalsegmentator_stage


log = Logger()


def _default_emit_script(task: str) -> Path:
    ts_cfg.DEFAULT_SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ts_cfg.DEFAULT_SGE_SCRIPTS_DIR / f"submit_totalseg_{task}_{ts}.sh"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option()
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
    help="Device for local inference; maps to TotalSegmentator --backend on SGE.",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Run locally via TotalSegmentator CLI or prepare/submit on SGE.",
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
    help="(sge) Outer Singularity image (GPU stack with TotalSegmentator CLI).",
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
    help="(sge) Host path mounted as the data root; --input must lie under it.",
)
@click.option(
    "--output-root",
    "output_root",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host path mounted as the output root; --output must lie under it.",
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
@click.option(
    "--project",
    default=None,
    help="(sge) qsub -P project (default from total_segmentator.config).",
)
@click.option(
    "--account",
    default=None,
    help="(sge) qsub -A account (default from total_segmentator.config).",
)
@click.option(
    "--ngpu",
    default=None,
    type=int,
    help="(sge) qsub -l ngpu (default from total_segmentator.config).",
)
@click.option(
    "--h-vmem",
    default=None,
    help="(sge) qsub -l h_vmem (default from total_segmentator.config).",
)
@click.option("--queue", default=None, help="(sge) optional qsub -q queue.")
@click.option("--job-name", default=None, help="(sge) SGE job name (defaults to totalseg_<task>).")
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Bash submission script path (default: total_segmentator.config).",
)
@click.option(
    "--direct-submit",
    is_flag=True,
    help="(sge) Submit with qsub immediately instead of writing a bash script first.",
)
@click.option(
    "--no-remote",
    is_flag=True,
    help="(sge) After writing the submission script, do not run it via SSH.",
)
@click.option("--remote-host", default=None, help="(sge) SSH host or alias.")
@click.option("--remote-user", default=None, help="(sge) SSH username.")
@click.option("--display-progress/--no-progress", default=True, help="Display output in terminal.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="(sge) With --direct-submit: do not actually qsub. Otherwise ignored for script mode.",
)
def main(
    input_path: Path,
    output_path: Path,
    task: str,
    roi_subset: tuple[str, ...],
    device: str,
    submit: str,
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
    project: str | None,
    account: str | None,
    ngpu: int | None,
    h_vmem: str | None,
    queue: str | None,
    job_name: str | None,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    display_progress: bool,
    dry_run: bool,
) -> None:
    """Run TotalSegmentator either locally or on SGE."""
    Logger(level="INFO")

    submit_l = submit.lower()
    if submit_l == "local":
        log.info(
            "Running TotalSegmentator locally: input=%s output=%s task=%s device=%s",
            input_path,
            output_path,
            task,
            device,
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
            capture_output=not display_progress,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"TotalSegmentator exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr}"
            )
        return

    # SGE defaults
    in_root = (
        Path(input_root)
        if input_root is not None
        else (input_path.parent if input_path.is_file() else input_path)
    )
    out_root = Path(output_root) if output_root is not None else output_path.parent
    src_p = Path(src_dir) if src_dir is not None else ts_cfg.DEFAULT_NVITK_SRC_DIR
    cont = Path(container) if container is not None else ts_cfg.CONTAINER_PATH
    models_p = Path(models_dir) if models_dir is not None else ts_cfg.MODELS_DIR
    log_p = Path(log_dir) if log_dir is not None else ts_cfg.SGE_LOG_DIR
    err_p = Path(err_dir) if err_dir is not None else ts_cfg.SGE_ERR_DIR

    if not cont.is_file():
        raise click.ClickException(
            f"Singularity container not found: {cont}. Pass --container or fix "
            "`nvitk.segmentation.total_segmentator.config.CONTAINER_PATH`."
        )

    paths = ClusterPaths(
        src=src_p,
        container=cont,
        models=models_p,
        input_root=in_root,
        output_root=out_root,
        log_dir=log_p,
        err_dir=err_p,
    )
    resources = SgeResources(
        project=project or ts_cfg.SGE_PROJECT,
        account=account or ts_cfg.SGE_ACCOUNT,
        ngpu=ngpu if ngpu is not None else ts_cfg.SGE_NGPU,
        h_vmem=h_vmem or ts_cfg.SGE_H_VMEM,
        queue=queue if queue is not None else ts_cfg.SGE_QUEUE,
    )
    job = SegmentationJob(
        job_name=job_name or f"totalseg_{task}",
        subject_input=input_path,
        subject_output=output_path,
        task=task,
        roi_subset=list(roi_subset),
        label_mode="multilabel" if multilabel else "single_labels",
        backend="gpu" if device.lower() == "gpu" else "cpu",
    )

    if direct_submit:
        jid = submit_totalsegmentator_stage(
            job,
            paths,
            resources=resources,
            hold_jid=None,
            dry_run=dry_run,
            emit=None,
        )
        click.echo(jid)
        return

    script_path = Path(emit_script) if emit_script is not None else _default_emit_script(task)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=log_p,
            err_dir=err_p,
            title=f"totalseg task={task} job={job.job_name}",
        )
        submit_totalsegmentator_stage(
            job,
            paths,
            resources=resources,
            hold_jid=None,
            dry_run=False,
            emit=fh,
        )
    log.info("Wrote SGE submission script: %s", script_path)

    if dry_run:
        log.info("Dry-run: script written; skipping SSH execution.")
        return

    if no_remote:
        log.info(
            "Skipping remote SSH (--no-remote). Run on the login node: bash %s",
            script_path,
        )
        return

    log.reset(restart_progress=False)
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host_resolved = ts_cfg.CLUSTER_HOST_ALIASES.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass("SSH password: ")
    ok = run_sge_script_ssh(host_resolved, user, password, script_path)
    if not ok:
        log.warning(
            "Remote execution did not complete successfully. Run manually: bash %s",
            script_path,
        )


if __name__ == "__main__":
    main()
