"""qvtpy pipeline master (stage0 only for now).

This mirrors the PESA-Fat runner shape: local execution vs. SGE submission.
Currently only stage0 (conversion + reorg) is implemented.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Iterable

import click

from nvitk.core.logger import Logger

from . import config as cfg
from . import stage0_convert

log = Logger()


def _iter_subjects(dicom_root: Path) -> list[str]:
    if not dicom_root.exists():
        return []
    return sorted([p.name for p in dicom_root.iterdir() if p.is_dir()])


def _qsub_stage0(
    *,
    subject: str,
    dicom_root: Path,
    nifti_root: Path,
    compute_phase_derived: bool,
    skip_existing: bool,
    container: Path,
) -> str:
    """Submit one stage0 job. Minimal implementation; can be upgraded later."""
    job = f"{cfg.SGE_JOB_PREFIX}_stage0_{subject}"
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


@click.command("nvitk-qvtpy")
@click.option("--dicom-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_DICOM_ROOT)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option("--subjects", default=None, help="Comma-separated subject list (default: all dirs under dicom-root).")
@click.option("--submit", type=click.Choice(["local", "sge"]), default="local", show_default=True)
@click.option("--container", type=click.Path(path_type=Path), default=cfg.CONTAINER_PATH)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--compute-phase-derived", is_flag=True, default=False)
def main(
    dicom_root: Path,
    nifti_root: Path,
    subjects: str | None,
    submit: str,
    container: Path,
    skip_existing: bool,
    compute_phase_derived: bool,
) -> None:
    Logger()

    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
    else:
        subject_list = _iter_subjects(dicom_root)

    if not subject_list:
        raise click.ClickException(f"No subjects found under dicom_root={dicom_root}")

    if submit == "local":
        for subj in subject_list:
            stage0_convert.run_subject(
                subj,
                dicom_root=dicom_root,
                nifti_root=nifti_root,
                compute_phase_derived=compute_phase_derived,
                skip_existing=skip_existing,
            )
        return

    # SGE
    for subj in subject_list:
        jid = _qsub_stage0(
            subject=subj,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            compute_phase_derived=compute_phase_derived,
            skip_existing=skip_existing,
            container=container,
        )
        log.info(f"[{subj}] submitted stage0 jid={jid}")


__all__ = ["main"]

