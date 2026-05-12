"""qvtpy pipeline master.

This mirrors the PESA-Fat runner shape: local execution vs. SGE submission via
a bash script + SSH to the cluster login node.

Stages (select with ``--stages``; default ``stage0_c,stage1``)
--------------------------------------------------------------
- ``stage0_d`` (``stage0_download``) -- XNAT -> DICOM into ``--dicom-root``
  (see :mod:`nvitk.pipes.qvtpy.stage0_download`). Requires ``--subjects`` or
  ``--subjects-file``. Runs locally even when ``--submit sge`` (XNAT pull is
  network/credentials bound and isn't a cluster job).
- ``stage0_c`` (``stage0_convert``) -- DICOM -> NIfTI conversion + reorg
  (see :mod:`nvitk.pipes.qvtpy.stage0_convert`).
- ``stage1`` (``stage1_eicab``) -- eICAB CoW/TOF segmentation on the
  reorganized ``TOF/TOF.nii.gz`` per subject
  (see :mod:`nvitk.pipes.qvtpy.stage1_eicab`).

SGE submission (``--submit sge``) writes one bash script under
``DEFAULT_SGE_SCRIPTS_DIR`` (one block per subject per stage, with
``-hold_jid`` chaining stage1 onto stage0_c) and then SSHes the script to a
login node via :func:`nvitk.cluster.remote_submit.run_sge_script_ssh`.
"""

from __future__ import annotations

import getpass
import shlex
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
    write_script_header,
)
from nvitk.core.logger import Logger
from nvitk.segmentation.eicab import config as eicab_cfg

from . import config as cfg
from . import stage0_convert, stage0_download, stage1_eicab

log = Logger()


# ---------------------------------------------------------------------------
# Stage selection
# ---------------------------------------------------------------------------

STAGE_DOWNLOAD = "stage0_d"
STAGE_CONVERT = "stage0_c"
STAGE_EICAB = "stage1"

_STAGE_ALIASES: dict[str, str] = {
    "stage0_d": STAGE_DOWNLOAD,
    "stage0d": STAGE_DOWNLOAD,
    "stage0_download": STAGE_DOWNLOAD,
    "download": STAGE_DOWNLOAD,
    "stage0_c": STAGE_CONVERT,
    "stage0c": STAGE_CONVERT,
    "stage0_convert": STAGE_CONVERT,
    "stage0": STAGE_CONVERT,
    "convert": STAGE_CONVERT,
    "stage1": STAGE_EICAB,
    "stage1_eicab": STAGE_EICAB,
    "eicab": STAGE_EICAB,
}

_ALL_STAGES: tuple[str, ...] = (STAGE_DOWNLOAD, STAGE_CONVERT, STAGE_EICAB)
DEFAULT_STAGES: str = f"{STAGE_CONVERT},{STAGE_EICAB}"


def _parse_stages(spec: str) -> list[str]:
    """Normalize ``--stages`` to canonical names, preserving pipeline order."""
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise click.ClickException("--stages cannot be empty.")
    canonical: set[str] = set()
    for tok in tokens:
        key = tok.replace("-", "_")
        if key not in _STAGE_ALIASES:
            raise click.ClickException(
                f"Unknown stage {tok!r}. Valid: {', '.join(_ALL_STAGES)}."
            )
        canonical.add(_STAGE_ALIASES[key])
    return [s for s in _ALL_STAGES if s in canonical]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_subjects(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _default_nvitk_src_dir() -> Path:
    """Host directory mounted at ``/nvitk/src/`` (contains a ``nvitk/`` package tree)."""
    return Path(nvitk.__file__).resolve().parent.parent


def _default_submit_script_path() -> Path:
    """Default bash submission script path under :data:`cfg.SGE_SCRIPTS_DIR`."""
    cfg.SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.SGE_SCRIPTS_DIR / f"submit_qvtpy_{ts}.sh"


def _emit_stage0_convert(
    fh: TextIO,
    subject: str,
    *,
    dicom_root: Path,
    nifti_root: Path,
    container: Path,
    src_dir: Path,
    compute_phase_derived: bool,
    skip_existing: bool,
) -> str:
    """Append a stage0_c SGE block to *fh*; returns the bash jid var ref."""
    binds = SingularityBinds()
    script = f"{binds.src}nvitk/pipes/qvtpy/stage0_convert.py"
    cmd_parts: list[str] = [
        "python",
        shlex.quote(script),
        "--subject",
        shlex.quote(subject),
        "--dicom-root",
        shlex.quote(binds.data),
        "--nifti-root",
        shlex.quote(binds.output),
    ]
    if compute_phase_derived:
        cmd_parts.append("--compute-phase-derived")
    if skip_existing:
        cmd_parts.append("--skip-existing")
    python_cmd = " ".join(cmd_parts)

    paths = ClusterPaths(
        src=src_dir,
        container=container,
        models=None,
        data_root=dicom_root,
        output_root=nifti_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage0c_{subject}",
        python_cmd=python_cmd,
        resources=SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=cfg.SGE_NGPU,
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env={"PYTHONPATH": str(binds.src)},
    )
    return submit_stage(spec, paths, emit=fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("nvitk-qvtpy")
@click.option("--dicom-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_DICOM_ROOT)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option(
    "--stages",
    "stages_spec",
    default=DEFAULT_STAGES,
    show_default=True,
    help=(
        "Comma-separated stages to run. Choices: "
        f"{', '.join(_ALL_STAGES)} (aliases: stage0_download, stage0_convert, stage1_eicab)."
    ),
)
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated subject list. If omitted, falls back to listing subfolders of the input root for the first selected stage.",
)
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Text/CSV/XLSX file with subject IDs.",
)
@click.option("--submit", type=click.Choice(["local", "sge"]), default="local", show_default=True)
@click.option(
    "--container",
    type=click.Path(path_type=Path),
    default=cfg.CONTAINER_PATH,
    help="Pipeline Singularity image used to run stage0_c on SGE.",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host tree mounted at /nvitk/src/ (default: parent of the installed nvitk package).",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--compute-phase-derived", is_flag=True, default=False)
@click.option(
    "--sequences",
    default=",".join(stage0_download.DEFAULT_SEQUENCES),
    show_default=True,
    help="Sequences to download (only used when stages include stage0_d).",
)
@click.option(
    "--xnat-config",
    "xnat_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="XNAT profile (YAML/JSON). Falls back to NVITK_XNAT_CONFIG / ~/.config/nvitk/xnat.*.",
)
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="Print a brief QC report after stage0_d (ignored otherwise).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_RESULTS_ROOT,
    show_default=True,
    help="Parent directory for stage1 outputs.",
)
@click.option(
    "--eicab-container",
    type=click.Path(path_type=Path),
    default=None,
    help="eICAB Singularity image override (stage1; default from eicab.config).",
)
@click.option(
    "--vasculature-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host tree mounted at /programs/Neuro/vasculature2 (stage1; default from eicab.config).",
)
@click.option(
    "--eicab-device",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="eICAB inference device (stage1).",
)
@click.option(
    "--eicab-resolution",
    type=float,
    default=0.5,
    show_default=True,
    help="eICAB target isotropic resolution in mm (stage1).",
)
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Bash submission script path (default: under eicab.config.DEFAULT_SGE_SCRIPTS_DIR).",
)
@click.option(
    "--no-remote",
    is_flag=True,
    help="(sge) After writing the submission script, do not run it via SSH.",
)
@click.option("--remote-host", default=None, help="(sge) SSH hostname or alias from CLUSTER_HOST_ALIASES.")
@click.option("--remote-user", default=None, help="(sge) SSH username (else prompt).")
def main(
    dicom_root: Path,
    nifti_root: Path,
    stages_spec: str,
    subjects: str | None,
    subjects_file: Path | None,
    submit: str,
    container: Path,
    src_dir: Path | None,
    skip_existing: bool,
    compute_phase_derived: bool,
    sequences: str,
    xnat_config_path: Path | None,
    report: bool,
    output_root: Path,
    eicab_container: Path | None,
    vasculature_dir: Path | None,
    eicab_device: str,
    eicab_resolution: float,
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
) -> None:
    Logger()

    stages = _parse_stages(stages_spec)
    run_dl = STAGE_DOWNLOAD in stages
    run_conv = STAGE_CONVERT in stages
    run_eicab = STAGE_EICAB in stages
    log.info(f"qvtpy | stages={','.join(stages)} | submit={submit}")

    if subjects or subjects_file:
        subject_list = stage0_download.load_subjects(
            subjects=subjects,
            subjects_file=subjects_file,
        )
    else:
        if run_dl:
            raise click.ClickException(
                "stage0_d (XNAT download) requires --subjects or --subjects-file."
            )
        fallback_root = dicom_root if run_conv else nifti_root
        subject_list = _iter_subjects(fallback_root)
        if not subject_list:
            raise click.ClickException(
                f"No subjects to process (looked under {fallback_root}). "
                "Pass --subjects / --subjects-file or populate that root."
            )

    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    # stage0_d always runs locally (XNAT pull is network/credentials-bound).
    if run_dl:
        if submit == "sge":
            log.info("stage0_d (XNAT download) always runs locally; SGE script will cover stage0_c/stage1.")
        from nvitk.db.xnat import requested_sequence_set
        from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

        profile = load_xnat_profile(xnat_config_path)
        conn = resolve_xnat_connection(profile)
        seq_set = requested_sequence_set(sequences) or set(stage0_download.DEFAULT_SEQUENCES)
        stage0_download.run_download(
            subject_list,
            dicom_root=dicom_root,
            xnat_config=conn,
            sequences=seq_set,
            skip_existing=skip_existing,
            report=report,
        )

    if submit == "local":
        for subj in subject_list:
            if run_conv:
                stage0_convert.run_subject(
                    subj,
                    dicom_root=dicom_root,
                    nifti_root=nifti_root,
                    compute_phase_derived=compute_phase_derived,
                    skip_existing=skip_existing,
                )
            if run_eicab:
                try:
                    stage1_eicab.run_subject(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        skip_existing=skip_existing,
                        resolution=eicab_resolution,
                        device=eicab_device,
                        eicab_container=eicab_container,
                        vasculature_dir=vasculature_dir,
                    )
                except (FileNotFoundError, OSError) as exc:
                    log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")
        return

    # SGE
    if not (run_conv or run_eicab):
        log.info("Nothing to submit to SGE (only stage0_d was selected). Done.")
        return

    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    script_path = Path(emit_script) if emit_script is not None else _default_submit_script_path()
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=cfg.SGE_LOG_DIR,
            err_dir=cfg.SGE_ERR_DIR,
            title=f"qvtpy stages={','.join(stages)} n_subjects={len(subject_list)}",
        )
        for subj in subject_list:
            stage0_jid: str | None = None
            if run_conv:
                try:
                    stage0_jid = _emit_stage0_convert(
                        fh,
                        subj,
                        dicom_root=dicom_root,
                        nifti_root=nifti_root,
                        container=container,
                        src_dir=src_p,
                        compute_phase_derived=compute_phase_derived,
                        skip_existing=skip_existing,
                    )
                except Exception as exc:
                    log.warning(f"[{subj}] stage0_c emit skipped: {exc}")

            if run_eicab:
                try:
                    stage1_eicab.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        skip_existing=skip_existing,
                        resolution=eicab_resolution,
                        device=eicab_device,
                        eicab_container=eicab_container,
                        pipeline_container=None,
                        src_dir=src_p,
                        vasculature_dir=vasculature_dir,
                        hold_jid=stage0_jid or None,
                        dry_run=False,
                        emit=fh,
                    )
                except (FileNotFoundError, OSError) as exc:
                    log.warning(f"[{subj}] stage1 eICAB emit skipped: {exc}")

    log.info("=" * 78)
    log.info(f"qvtpy SGE script written: {script_path}")
    log.info(f"On the cluster login node: bash {script_path}")
    log.info("=" * 78)

    if no_remote:
        log.info("Skipping remote SSH (--no-remote).")
        return

    log.reset(restart_progress=False)
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host_resolved = eicab_cfg.CLUSTER_HOST_ALIASES.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass("SSH password: ")
    ok = run_sge_script_ssh(host_resolved, user, password, script_path)
    if not ok:
        log.warning(
            f"Remote execution did not complete successfully. Run manually: bash {script_path}"
        )


__all__ = ["main", "DEFAULT_STAGES", "STAGE_DOWNLOAD", "STAGE_CONVERT", "STAGE_EICAB"]
