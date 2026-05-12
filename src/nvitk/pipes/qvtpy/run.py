"""qvtpy pipeline master.

This mirrors the PESA-Fat runner shape: local execution vs. SGE submission.

Stages (select with ``--stages``; default ``stage0_c,stage1``)
--------------------------------------------------------------
- ``stage0_d`` (``stage0_download``) — XNAT -> DICOM into ``--dicom-root``
  (see :mod:`nvitk.pipes.qvtpy.stage0_download`). Requires ``--subjects`` or
  ``--subjects-file``.
- ``stage0_c`` (``stage0_convert``) — DICOM -> NIfTI conversion and
  reorganization (see :mod:`nvitk.pipes.qvtpy.stage0_convert`).
- ``stage1`` (``stage1_eicab``) — eICAB CoW/TOF segmentation on the
  reorganized ``TOF/TOF.nii.gz`` per subject
  (see :mod:`nvitk.pipes.qvtpy.stage1_eicab`). In ``--submit sge`` mode each
  stage1 job is held on the corresponding stage0_c job id (when stage0_c
  also runs in the same invocation).
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import click

from nvitk.core.logger import Logger

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


def _qsub_stage0_convert(
    *,
    subject: str,
    dicom_root: Path,
    nifti_root: Path,
    compute_phase_derived: bool,
    skip_existing: bool,
    container: Path,
) -> str:
    """Submit one stage0_convert job. Minimal implementation; can be upgraded later."""
    job = f"{cfg.SGE_JOB_PREFIX}_stage0c_{subject}"
    cfg.SGE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SGE_ERR_DIR.mkdir(parents=True, exist_ok=True)

    inner = " ".join(
        [
            "python",
            "-m",
            "nvitk.pipes.qvtpy.stage0_convert",
            "--subject",
            shlex.quote(subject),
            "--dicom-root",
            shlex.quote(str(dicom_root)),
            "--nifti-root",
            shlex.quote(str(nifti_root)),
            "--compute-phase-derived" if compute_phase_derived else "",
            "--skip-existing" if skip_existing else "",
        ]
    ).strip()

    sing = (
        "singularity exec "
        + "--nv " * 0
        + "-B "
        + shlex.quote(str(dicom_root))
        + ":/DICOM "
        + "-B "
        + shlex.quote(str(nifti_root))
        + ":/NIFTI "
        + shlex.quote(str(container))
        + " bash -c "
        + shlex.quote(inner)
    )

    argv = [
        "qsub",
        "-P",
        cfg.SGE_PROJECT,
        "-terse",
        "-N",
        job,
        "-A",
        cfg.SGE_ACCOUNT,
        "-l",
        f"ngpu={cfg.SGE_CPU_NGPU}",
        "-l",
        f"h_vmem={cfg.SGE_CPU_H_VMEM}",
        "-o",
        str(cfg.SGE_LOG_DIR / f"{job}.log"),
        "-e",
        str(cfg.SGE_ERR_DIR / f"{job}.err"),
    ]
    if cfg.SGE_QUEUE:
        argv += ["-q", cfg.SGE_QUEUE]

    proc = subprocess.run(argv, input=sing, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"qsub failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


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
@click.option("--container", type=click.Path(path_type=Path), default=cfg.CONTAINER_PATH)
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
def main(
    dicom_root: Path,
    nifti_root: Path,
    stages_spec: str,
    subjects: str | None,
    subjects_file: Path | None,
    submit: str,
    container: Path,
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

    if run_dl:
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
    for subj in subject_list:
        stage0_jid: str | None = None
        if run_conv:
            stage0_jid = _qsub_stage0_convert(
                subject=subj,
                dicom_root=dicom_root,
                nifti_root=nifti_root,
                compute_phase_derived=compute_phase_derived,
                skip_existing=skip_existing,
                container=container,
            )
            log.info(f"[{subj}] submitted stage0_c jid={stage0_jid}")

        if run_eicab:
            try:
                eicab_jid = stage1_eicab.submit_subject_sge(
                    subj,
                    nifti_root=nifti_root,
                    output_root=output_root,
                    skip_existing=skip_existing,
                    resolution=eicab_resolution,
                    device=eicab_device,
                    eicab_container=eicab_container,
                    vasculature_dir=vasculature_dir,
                    hold_jid=stage0_jid or None,
                )
                if eicab_jid:
                    log.info(
                        f"[{subj}] submitted stage1 eICAB jid={eicab_jid}"
                        + (f" (hold_jid={stage0_jid})" if stage0_jid else "")
                    )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")


__all__ = ["main", "DEFAULT_STAGES", "STAGE_DOWNLOAD", "STAGE_CONVERT", "STAGE_EICAB"]
