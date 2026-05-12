"""qvtpy stage 1: eICAB Circle-of-Willis / TOF segmentation on stage0 NIfTI layout.

Expects per-subject folders under ``--nifti-root`` as produced by
:mod:`nvitk.pipes.qvtpy.stage0_convert`, in particular ``TOF/TOF.nii.gz`` (or
``.nii``). Writes eICAB outputs under ``--output-root/<subject>/<eicab-subdir>/``
(default subdir from :data:`nvitk.pipes.qvtpy.config.STAGE1_EICAB_DIR`).

Two submission modes (mirroring stage0 + ``nvitk-eicab``):

- ``--submit local`` (default): :func:`nvitk.segmentation.eicab.runner.run_eicab`
  via ``singularity run`` on the current host.
- ``--submit sge``: per-subject SGE job built with
  :func:`nvitk.segmentation.eicab.cluster.submit_eicab_job`
  (canonical pipeline-container wrapper with all proper bind mounts).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.sge import SgeResources, write_script_header
from nvitk.core.logger import Logger
from nvitk.segmentation.eicab import config as eicab_cfg
from nvitk.segmentation.eicab.cluster import submit_eicab_job
from nvitk.segmentation.eicab.runner import run_eicab

from . import config as cfg

log = Logger()

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _default_nvitk_src_dir() -> Path:
    """Host directory mounted at ``/nvitk/src/`` (contains a ``nvitk/`` package tree)."""
    return Path(nvitk.__file__).resolve().parent.parent


def _default_emit_script(subject: str) -> Path:
    stem = _SAFE.sub("_", subject)[:60] or "subject"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eicab_cfg.DEFAULT_SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return eicab_cfg.DEFAULT_SGE_SCRIPTS_DIR / f"submit_qvtpy_eicab_{stem}_{ts}.sh"


def _iter_subjects_nifti(nifti_root: Path) -> list[str]:
    """Subject ids: one folder per subject under *nifti_root*."""
    if not nifti_root.exists():
        return []
    return sorted(p.name for p in nifti_root.iterdir() if p.is_dir())


def find_tof_volume(subject_nifti_dir: Path) -> Path | None:
    """Return path to TOF NIfTI after stage0 reorganization, or None."""
    tof_dir = subject_nifti_dir / "TOF"
    if not tof_dir.is_dir():
        return None
    for name in ("TOF.nii.gz", "TOF.nii"):
        p = tof_dir / name
        if p.is_file():
            return p
    candidates = sorted(
        p
        for p in tof_dir.iterdir()
        if p.is_file() and (p.suffix == ".nii" or p.name.endswith(".nii.gz"))
    )
    return candidates[0] if candidates else None


def _output_has_segmentation(out_dir: Path) -> bool:
    if not out_dir.is_dir():
        return False
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.endswith(".nii.gz") or p.suffix == ".nii":
            return True
    return False


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    skip_existing: bool = False,
    resolution: float = 0.5,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    eicab_container: Path | None = None,
    tmp_dir: Path | None = None,
    vasculature_dir: Path | None = None,
    keep_aux_outputs: bool = False,
) -> Path:
    """Run eICAB locally for one subject. Returns the eICAB output directory."""
    subj_nifti = nifti_root / subject
    if not subj_nifti.is_dir():
        raise FileNotFoundError(f"NIfTI subject dir not found: {subj_nifti}")

    tof = find_tof_volume(subj_nifti)
    if tof is None:
        raise FileNotFoundError(
            f"No TOF NIfTI under {subj_nifti / 'TOF'} (expected TOF/TOF.nii.gz from stage0)."
        )

    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    out_dir = output_root / subject / subdir
    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    container = Path(eicab_container) if eicab_container is not None else eicab_cfg.CONTAINER_PATH
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(eicab_cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )

    if skip_existing and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return out_dir

    log.info(f"qvtpy stage1 eICAB (local) | subject={subject}")
    log.info(f"  input : {tof}")
    log.info(f"  output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_eicab(
        tof,
        out_dir,
        resolution=resolution,
        simple_segmentation=simple_segmentation,
        attention=attention,
        device=device,
        container=container,
        tmp_dir=tmp,
        keep_aux_outputs=keep_aux_outputs,
        vasculature_host_path=vas_host,
        capture_output=False,
    )
    return out_dir


# ---------------------------------------------------------------------------
# SGE submission
# ---------------------------------------------------------------------------


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    skip_existing: bool = False,
    resolution: float = 0.5,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    eicab_container: Path | None = None,
    pipeline_container: Path | None = None,
    src_dir: Path | None = None,
    tmp_dir: Path | None = None,
    vasculature_dir: Path | None = None,
    log_dir: Path | None = None,
    err_dir: Path | None = None,
    keep_aux_outputs: bool = False,
    resources: SgeResources | None = None,
    hold_jid: str | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
    job_name: str | None = None,
) -> str:
    """Submit one stage1 eICAB SGE job. Returns the qsub job id (or '' when *emit* is set)."""
    subj_nifti = nifti_root / subject
    if not subj_nifti.is_dir():
        raise FileNotFoundError(f"NIfTI subject dir not found: {subj_nifti}")

    tof = find_tof_volume(subj_nifti)
    if tof is None:
        raise FileNotFoundError(
            f"No TOF NIfTI under {subj_nifti / 'TOF'} (expected TOF/TOF.nii.gz from stage0)."
        )

    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    out_dir = output_root / subject / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return ""

    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    eicab_c = Path(eicab_container) if eicab_container is not None else eicab_cfg.CONTAINER_PATH
    pipeline_c = Path(pipeline_container) if pipeline_container is not None else eicab_cfg.PIPELINE_CONTAINER_PATH
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(eicab_cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )
    ld = Path(log_dir) if log_dir is not None else eicab_cfg.SGE_LOG_DIR
    ed = Path(err_dir) if err_dir is not None else eicab_cfg.SGE_ERR_DIR

    res = resources or SgeResources(
        project=eicab_cfg.SGE_PROJECT,
        account=eicab_cfg.SGE_ACCOUNT,
        ngpu=eicab_cfg.SGE_NGPU,
        h_vmem=eicab_cfg.SGE_H_VMEM,
        queue=eicab_cfg.SGE_QUEUE,
    )
    jn = job_name or f"{cfg.SGE_JOB_PREFIX}_stage1_eicab_{_SAFE.sub('_', subject)[:40]}"

    log.info(f"qvtpy stage1 eICAB (sge) | subject={subject}")
    log.info(f"  input : {tof}")
    log.info(f"  output: {out_dir}")

    return submit_eicab_job(
        job_name=jn,
        input_nifti=tof.resolve(),
        output_dir=out_dir.resolve(),
        tmp_dir=tmp.resolve(),
        eicab_container=eicab_c.resolve(),
        src_dir=src_p.resolve(),
        pipeline_container=pipeline_c.resolve(),
        input_root=nifti_root.resolve(),
        output_root=output_root.resolve(),
        vasculature_host=vas_host,
        log_dir=ld,
        err_dir=ed,
        resolution=resolution,
        device=device,
        simple_segmentation=simple_segmentation,
        attention=attention,
        keep_aux_outputs=keep_aux_outputs,
        resources=res,
        hold_jid=hold_jid,
        dry_run=dry_run,
        emit=emit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("qvtpy-stage1-eicab")
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_NIFTI_ROOT,
    show_default=True,
    help="Root of per-subject NIfTI trees (stage0 output).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_RESULTS_ROOT,
    show_default=True,
    help="Parent directory for per-subject eICAB outputs.",
)
@click.option(
    "--eicab-subdir",
    type=str,
    default=None,
    help=f"Subdirectory under output-root/<subject>/ (default: {cfg.STAGE1_EICAB_DIR!r}).",
)
@click.option(
    "--subject",
    default=None,
    help="Single subject id. If omitted, all subfolders of --nifti-root are processed.",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Run locally or submit per-subject SGE jobs (canonical eICAB pipeline-container wrapper).",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--resolution", type=float, default=0.5, show_default=True)
@click.option("--simple-segmentation", is_flag=True, default=False)
@click.option("--attention", is_flag=True, default=False)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
)
@click.option(
    "--container",
    "eicab_container",
    type=click.Path(path_type=Path),
    default=None,
    help="eICAB Singularity image (default: nvitk.segmentation.eicab.config.CONTAINER_PATH).",
)
@click.option(
    "--pipeline-container",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Outer pipeline Singularity image (default: eicab.config.PIPELINE_CONTAINER_PATH).",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host tree mounted at /nvitk/src/ (default: parent of the installed nvitk package).",
)
@click.option(
    "--tmp-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Temp directory for eICAB (default: <output>/<subject>/<eicab-subdir>/.eicab_tmp).",
)
@click.option(
    "--vasculature-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host tree bind-mounted to /programs/Neuro/vasculature2 (eICAB requirement).",
)
@click.option("--log-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stdout logs.")
@click.option("--err-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stderr logs.")
@click.option(
    "--keep-aux-outputs",
    is_flag=True,
    default=False,
    help="Keep auxiliary eICAB intermediates (default: prune to CoW/WB NIfTIs).",
)
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Write a bash submission script instead of qsubbing directly.",
)
@click.option("--dry-run", is_flag=True, default=False, help="(sge) Build the command but do not qsub.")
def main(
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None,
    subject: str | None,
    submit: str,
    skip_existing: bool,
    resolution: float,
    simple_segmentation: bool,
    attention: bool,
    device: str,
    eicab_container: Path | None,
    pipeline_container: Path | None,
    src_dir: Path | None,
    tmp_dir: Path | None,
    vasculature_dir: Path | None,
    log_dir: Path | None,
    err_dir: Path | None,
    keep_aux_outputs: bool,
    emit_script: Path | None,
    dry_run: bool,
) -> None:
    Logger()

    subjects = [subject] if subject else _iter_subjects_nifti(nifti_root)
    if not subjects:
        raise click.ClickException(
            f"No subject folders found under nifti_root={nifti_root}. "
            "Pass --subject or point --nifti-root at a directory of subject subfolders."
        )

    mode = submit.lower()
    if mode == "local":
        for subj in subjects:
            try:
                run_subject(
                    subj,
                    nifti_root=nifti_root,
                    output_root=output_root,
                    eicab_subdir=eicab_subdir,
                    skip_existing=skip_existing,
                    resolution=resolution,
                    simple_segmentation=simple_segmentation,
                    attention=attention,
                    device=device,
                    eicab_container=eicab_container,
                    tmp_dir=tmp_dir,
                    vasculature_dir=vasculature_dir,
                    keep_aux_outputs=keep_aux_outputs,
                )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")
        return

    fh: TextIO | None = None
    script_path: Path | None = None
    if emit_script is not None:
        script_path = Path(emit_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(script_path, "w", encoding="utf-8")
        write_script_header(
            fh,
            log_dir=Path(log_dir) if log_dir is not None else eicab_cfg.SGE_LOG_DIR,
            err_dir=Path(err_dir) if err_dir is not None else eicab_cfg.SGE_ERR_DIR,
            title=f"qvtpy stage1 eICAB (n={len(subjects)})",
        )

    try:
        for subj in subjects:
            try:
                jid = submit_subject_sge(
                    subj,
                    nifti_root=nifti_root,
                    output_root=output_root,
                    eicab_subdir=eicab_subdir,
                    skip_existing=skip_existing,
                    resolution=resolution,
                    simple_segmentation=simple_segmentation,
                    attention=attention,
                    device=device,
                    eicab_container=eicab_container,
                    pipeline_container=pipeline_container,
                    src_dir=src_dir,
                    tmp_dir=tmp_dir,
                    vasculature_dir=vasculature_dir,
                    log_dir=log_dir,
                    err_dir=err_dir,
                    keep_aux_outputs=keep_aux_outputs,
                    dry_run=dry_run,
                    emit=fh,
                )
                if jid:
                    log.info(f"[{subj}] stage1 eICAB submitted jid={jid}")
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")
    finally:
        if fh is not None:
            fh.close()
            log.info(f"Wrote SGE submission script: {script_path}")


__all__ = ["find_tof_volume", "run_subject", "submit_subject_sge", "main"]


if __name__ == "__main__":
    main()
