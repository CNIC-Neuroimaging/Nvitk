"""``nvitk-eicab`` — Circle-of-Willis / TOF eICAB segmentation (local or SGE)."""

from __future__ import annotations

import getpass
import re
from datetime import datetime
from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
import nvitk

from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.sge import SgeResources, write_script_header

from . import config as cfg
from .cluster import submit_eicab_job
from .runner import eicab_tmp_dir, run_eicab


log = Logger()

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _input_subject_key(input_path: Path) -> str:
    """Subject-like token from a TOF NIfTI filename."""
    stem = input_path.name
    if stem.lower().endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.lower().endswith(".nii"):
        stem = stem[: -len(".nii")]
    return stem or input_path.parent.name


def _default_nvitk_src_dir() -> Path:
    """Host directory mounted at ``/nvitk/src/`` (contains a ``nvitk/`` package tree)."""
    return Path(nvitk.__file__).resolve().parent.parent


def _default_emit_script(input_path: Path) -> Path:
    stem = _SAFE.sub("_", input_path.stem.replace(".nii", ""))[:60]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.DEFAULT_SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.DEFAULT_SGE_SCRIPTS_DIR / f"submit_eicab_{stem}_{ts}.sh"


@click.command("nvitk-eicab", context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="TOF/MRA NIfTI (.nii / .nii.gz), raw (not skull-stripped).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for segmentation NIfTI(s).",
)
@click.option("--resolution", type=float, default=0.5, show_default=True)
@click.option("--simple-segmentation", is_flag=True)
@click.option("--attention", is_flag=True)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Inference device (gpu is an alias for cuda).",
)
@click.option(
    "--container",
    "eicab_container",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the eICAB Singularity image (default from eicab.config).",
)
@click.option(
    "--tmp-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host temp directory bind-mounted to /tmp in the eICAB container "
    "(local default: ~/local_tmp; sge default: <output>/.eicab_tmp). "
    "Shared bases (e.g. /data_tmp) are namespaced per subject.",
)
@click.option(
    "--keep-aux-outputs",
    is_flag=True,
    default=False,
    help="Keep legacy auxiliary folders/files (default: only CoW/WB NIfTIs).",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Run locally or prepare/submit an SGE job (script + optional SSH).",
)
@click.option(
    "--pipeline-container",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Outer Singularity image with Python + singularity client.",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host tree mounted at /nvitk/src/ (default: parent of the installed nvitk package).",
)
@click.option(
    "--input-root",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host root mounted at /nvitk/data/ (default: parent of --input if it is a file).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host root mounted at /nvitk/output/ (default: parent of --output). "
    "Default tmp is under --output; use --tmp-dir for a separate host path (e.g. /data_tmp).",
)
@click.option(
    "--vasculature-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host tree bind-mounted to /programs/Neuro/vasculature2 (required; default from "
    "eicab.config DEFAULT_VASCULATURE_HOST_DIR, overridable via pipelines.eicab in sge.json).",
)
@click.option("--log-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stdout logs.")
@click.option("--err-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stderr logs.")
@click.option("--project", default=None, help="(sge) qsub -P (default from eicab.config).")
@click.option("--account", default=None, help="(sge) qsub -A (default from eicab.config).")
@click.option("--ngpu", default=None, type=int, help="(sge) qsub -l ngpu (default from eicab.config).")
@click.option("--h-vmem", default=None, help="(sge) qsub -l h_vmem (default from eicab.config).")
@click.option("--queue", default=None, help="(sge) optional qsub -q.")
@click.option("--job-name", default=None, help="(sge) Job name (default derived from input).")
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Bash submission script path (default: under eicab.config DEFAULT_SGE_SCRIPTS_DIR).",
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
@click.option("--dry-run", is_flag=True, help="(sge) Build command only (no qsub / no SSH).")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    input_path: Path,
    output_path: Path,
    resolution: float,
    simple_segmentation: bool,
    attention: bool,
    device: str,
    eicab_container: Path | None,
    tmp_dir: Path | None,
    keep_aux_outputs: bool,
    submit: str,
    pipeline_container: Path | None,
    src_dir: Path | None,
    input_root: Path | None,
    output_root: Path | None,
    vasculature_dir: Path | None,
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
    log_level: str,
) -> None:
    """Run eICAB locally (singularity run) or submit to SGE (wrapped pipeline container)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    ec = Path(eicab_container) if eicab_container is not None else cfg.CONTAINER_PATH
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )

    if submit.lower() == "local":
        if not vas_host.is_dir():
            raise click.ClickException(
                f"Vasculature tools directory not found or not a directory: {vas_host}. "
                "eICAB requires this tree on PATH inside the container (see legacy "
                "run_eicab_inference.sh). Pass --vasculature-dir or set "
                "pipelines.eicab.default_vasculature_host_dir in .nvitk/sge.json."
            )
        if not ec.is_file():
            raise click.ClickException(
                f"eICAB container not found: {ec}. Pass --container or fix "
                "`nvitk.segmentation.eicab.config.CONTAINER_PATH`."
            )
        if tmp_dir is None:
            tmp = cfg.DEFAULT_TMP_DIR
        else:
            tmp = eicab_tmp_dir(
                output_path,
                tmp_dir=tmp_dir,
                subject_key=_input_subject_key(input_path),
            )
        output_path.mkdir(parents=True, exist_ok=True)
        run_eicab(
            input_path,
            output_path,
            resolution=resolution,
            simple_segmentation=simple_segmentation,
            attention=attention,
            device=device,
            container=ec,
            tmp_dir=tmp,
            keep_aux_outputs=keep_aux_outputs,
            vasculature_host_path=vas_host,
            capture_output=not display_progress,
        )
        log.info("eICAB local run finished: output -> %s", output_path)
        return

    # SGE
    in_root = (
        Path(input_root)
        if input_root is not None
        else (input_path.parent if input_path.is_file() else input_path)
    )
    out_root = Path(output_root) if output_root is not None else output_path.parent
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    tmp = eicab_tmp_dir(
        output_path,
        tmp_dir=tmp_dir,
        subject_key=_input_subject_key(input_path),
    )

    pc = Path(pipeline_container) if pipeline_container is not None else cfg.PIPELINE_CONTAINER_PATH
    ld = Path(log_dir) if log_dir is not None else cfg.SGE_LOG_DIR
    ed = Path(err_dir) if err_dir is not None else cfg.SGE_ERR_DIR

    if not ec.is_file():
        raise click.ClickException(
            f"eICAB container not found: {ec}. Pass --container with a valid host path "
            "(extra bind mounts are added automatically when the image is outside "
            "the input/output/src trees)."
        )
    if not pc.is_file():
        raise click.ClickException(
            f"Pipeline container not found: {pc}. Pass --pipeline-container or fix config."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    resources = SgeResources(
        project=project or cfg.SGE_PROJECT,
        account=account or cfg.SGE_ACCOUNT,
        ngpu=ngpu if ngpu is not None else cfg.SGE_NGPU,
        h_vmem=h_vmem or cfg.SGE_H_VMEM,
        queue=queue if queue is not None else cfg.SGE_QUEUE,
    )
    jn = job_name or f"eicab_{_SAFE.sub('_', input_path.stem)[:40]}"

    if direct_submit:
        jid = submit_eicab_job(
            job_name=jn,
            input_nifti=input_path.resolve(),
            output_dir=output_path.resolve(),
            tmp_dir=tmp.resolve(),
            eicab_container=ec.resolve(),
            src_dir=src_p.resolve(),
            pipeline_container=pc.resolve(),
            input_root=in_root.resolve(),
            output_root=out_root.resolve(),
            vasculature_host=vas_host,
            log_dir=ld,
            err_dir=ed,
            resolution=resolution,
            device=device,
            simple_segmentation=simple_segmentation,
            attention=attention,
            keep_aux_outputs=keep_aux_outputs,
            resources=resources,
            hold_jid=None,
            dry_run=dry_run,
            emit=None,
        )
        click.echo(jid)
        return

    script_path = Path(emit_script) if emit_script is not None else _default_emit_script(input_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=ld,
            err_dir=ed,
            title=f"eICAB job={jn} input={input_path.name}",
        )
        submit_eicab_job(
            job_name=jn,
            input_nifti=input_path.resolve(),
            output_dir=output_path.resolve(),
            tmp_dir=tmp.resolve(),
            eicab_container=ec.resolve(),
            src_dir=src_p.resolve(),
            pipeline_container=pc.resolve(),
            input_root=in_root.resolve(),
            output_root=out_root.resolve(),
            vasculature_host=vas_host,
            log_dir=ld,
            err_dir=ed,
            resolution=resolution,
            device=device,
            simple_segmentation=simple_segmentation,
            attention=attention,
            keep_aux_outputs=keep_aux_outputs,
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
        log.info("Skipping remote SSH (--no-remote). Run on the login node: bash %s", script_path)
        return

    log.reset(restart_progress=False)
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host_resolved = cfg.CLUSTER_HOST_ALIASES.get(host_key, host_key)
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
