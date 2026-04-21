"""PESA-Fat batch master.

Single entry point that drives the *whole* PESA-Fat batch — stage 0
(DICOM -> NIfTI + renaming), then the ct-pet-v5 and dixon-v5 pipelines
(stages 1 / 2 / 3).

Two execution modes:

* ``--submit local`` — stage 0 is run in-process, then each pipeline's
  master (:mod:`ct_pet_v5.run` / :mod:`dixon_v5.run`) is called with
  ``--submit local`` for the selected stages.
* ``--submit sge`` — for every subject we submit a stage-0 SGE job
  (``qsub`` wrapping ``singularity exec python -m stage0_convert``) and
  capture its jid. Each pipeline's per-subject chain is then submitted
  with ``-hold_jid`` pointing at that jid, so stages 1-2-3 wait for the
  subject's own stage 0 to finish. The CT-PET and Dixon chains of a given
  subject run in parallel after stage 0.

Examples
--------

Local dry-run of the whole batch::

    nvitk-pesa-fat --batch 202602_Week4

Full SGE submission (all subjects, both pipelines, all stages)::

    nvitk-pesa-fat --batch 202602_Week4 --submit sge \
        --src-dir /home/imarcoss/nvitk/src

Skip stage 0, run only dixon v5 for a subset locally::

    nvitk-pesa-fat --batch 202602_Week4 \
        --subjects PESA001,PESA002 \
        --pipelines dixon-v5 \
        --stages stage1,stage2,stage3
"""

from __future__ import annotations

import shlex
from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT,
    BatchLayout,
    layout,
    parse_subjects,
)
from nvitk.pipes.pesa_fat.common.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
)
from nvitk.pipes.pesa_fat.common import stage0_convert
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ctpet_cfg
from nvitk.pipes.pesa_fat.ct_pet_v5 import run as ctpet_run
from nvitk.pipes.pesa_fat.dixon_v5 import config as dixon_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import run as dixon_run


log = Logger()


PIPELINE_CHOICES = ("ct-pet-v5", "dixon-v5")
STAGE_CHOICES = ("stage0", "stage1", "stage2", "stage3")


# ---------------------------------------------------------------------------
# Stage 0 on SGE
# ---------------------------------------------------------------------------


_STAGE0_BIND_DICOM = "/PESAFat/DICOM/"
_STAGE0_BIND_NIFTI = "/PESAFat/NIFTI/"


def _stage0_python_cmd(subject: str, lay: BatchLayout, log_level: str) -> str:
    """Stage 0 inside the container reads from the mounted DICOM root and
    writes to the mounted NIfTI root; use container paths here."""
    return " ".join(
        [
            "python",
            "-m",
            "nvitk.pipes.pesa_fat.common.stage0_convert",
            "--batch",
            shlex.quote(lay.batch),
            "--subject",
            shlex.quote(subject),
            "--dicom-root",
            shlex.quote(_STAGE0_BIND_DICOM),
            "--nifti-root",
            shlex.quote(_STAGE0_BIND_NIFTI),
            "--log-level",
            log_level,
        ]
    )


def _stage0_cluster_paths(lay: BatchLayout, container: Path, src_dir: Path) -> ClusterPaths:
    return ClusterPaths(
        src=src_dir,
        container=container,
        models=lay.model_root,
        data_root=lay.dicom_root,     # bind dicom root so dcm2nii can read inputs
        output_root=lay.nifti_root,   # write converted NIfTIs here
        log_dir=ctpet_cfg.SGE_LOG_DIR,
        err_dir=ctpet_cfg.SGE_ERR_DIR,
    )


def _submit_stage0(
    subject: str,
    lay: BatchLayout,
    *,
    container: Path,
    src_dir: Path,
    dry_run: bool,
    log_level: str,
) -> str:
    """Submit a single stage-0 SGE job and return its jid (or 'DRY_RUN')."""
    paths = _stage0_cluster_paths(lay, container, src_dir)
    binds = SingularityBinds(data=_STAGE0_BIND_DICOM, output=_STAGE0_BIND_NIFTI)
    spec = StageSpec(
        job_name=f"PESAFat_stage0_{subject}",
        python_cmd=_stage0_python_cmd(subject, lay, log_level),
        resources=SgeResources(
            project=ctpet_cfg.SGE_PROJECT,
            account=ctpet_cfg.SGE_ACCOUNT,
            ngpu=ctpet_cfg.SGE_CPU_NGPU,
            h_vmem=ctpet_cfg.SGE_CPU_H_VMEM,
            queue=ctpet_cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
    )
    jid = submit_stage(spec, paths, dry_run=dry_run)
    log.info(f"[{subject}] stage0 submitted -> {jid}")
    return jid


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _subject_list(lay: BatchLayout, subjects_arg: str | None, source: str) -> list[str]:
    parsed = parse_subjects(subjects_arg)
    if parsed:
        return parsed
    if source == "dicom":
        if not lay.dicom_dir.exists():
            raise click.ClickException(
                f"DICOM batch directory not found: {lay.dicom_dir}"
            )
        subs = [d.name for d in lay.subject_dicom_dirs()]
    else:
        if not lay.nifti_dir.exists():
            raise click.ClickException(
                f"NIfTI batch directory not found: {lay.nifti_dir}"
            )
        subs = list(lay.iter_subjects())
    if not subs:
        raise click.ClickException(
            f"No PESA* subjects found under {lay.dicom_dir if source == 'dicom' else lay.nifti_dir}"
        )
    return subs


def _run_local(
    lay: BatchLayout,
    subjects: list[str],
    pipelines: list[str],
    stages_sel: list[str],
    *,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    regions: tuple[str, ...],
    compress: bool,
) -> None:
    if "stage0" in stages_sel:
        log.info("=" * 78)
        log.info(f"STAGE 0 (local) | batch={lay.batch} | {len(subjects)} subject(s)")
        log.info("=" * 78)
        for subj in subjects:
            try:
                stage0_convert.run_subject(subj, lay, compress=compress)
            except Exception as exc:
                log.error(f"[{subj}] stage 0 failed: {exc}")

    pipe_stages = [s for s in stages_sel if s != "stage0"]
    if not pipe_stages:
        return

    if "ct-pet-v5" in pipelines:
        log.info("=" * 78)
        log.info(f"CT-PET v5 (local) | stages={pipe_stages}")
        log.info("=" * 78)
        ctpet_run._run_local(
            lay,
            subjects,
            pipe_stages,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
        )

    if "dixon-v5" in pipelines:
        log.info("=" * 78)
        log.info(f"Dixon v5 (local) | stages={pipe_stages}")
        log.info("=" * 78)
        dixon_run._run_local(
            lay,
            subjects,
            pipe_stages,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
            regions=regions,
        )


def _run_sge(
    lay: BatchLayout,
    subjects: list[str],
    pipelines: list[str],
    stages_sel: list[str],
    *,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    regions: tuple[str, ...],
    container: Path,
    src_dir: Path,
    dry_run: bool,
    log_level: str,
) -> None:
    run_stage0 = "stage0" in stages_sel
    pipe_stages = [s for s in stages_sel if s != "stage0"]

    stage0_jids: dict[str, str | None] = {s: None for s in subjects}
    if run_stage0:
        log.info("=" * 78)
        log.info(f"STAGE 0 (SGE) | batch={lay.batch} | {len(subjects)} subject(s)")
        log.info("=" * 78)
        for subj in subjects:
            stage0_jids[subj] = _submit_stage0(
                subj,
                lay,
                container=container,
                src_dir=src_dir,
                dry_run=dry_run,
                log_level=log_level,
            )

    if not pipe_stages:
        return

    for subj in subjects:
        base_hold = stage0_jids.get(subj)
        if "ct-pet-v5" in pipelines:
            ctpet_run.submit_subject_chain(
                subj,
                lay,
                pipe_stages,
                container=container,
                src_dir=src_dir,
                backend=backend,
                device=device,
                model_dir=model_dir,
                overwrite=overwrite,
                base_hold=base_hold,
                dry_run=dry_run,
                log_level=log_level,
            )
        if "dixon-v5" in pipelines:
            dixon_run.submit_subject_chain(
                subj,
                lay,
                pipe_stages,
                container=container,
                src_dir=src_dir,
                backend=backend,
                device=device,
                model_dir=model_dir,
                overwrite=overwrite,
                regions=regions,
                base_hold=base_hold,
                dry_run=dry_run,
                log_level=log_level,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("nvitk-pesa-fat")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week4').")
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated PESA* subjects (default: every subject in the batch).",
)
@click.option(
    "--pipelines",
    default=",".join(PIPELINE_CHOICES),
    show_default=True,
    help="Comma-separated pipelines to run.",
)
@click.option(
    "--stages",
    default=",".join(STAGE_CHOICES),
    show_default=True,
    help="Comma-separated stages to run (stage0 is shared).",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"]),
    default="local",
    show_default=True,
)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--backend",
    type=click.Choice(["cupy", "numpy"], case_sensitive=False),
    default="cupy",
    show_default=True,
    help="Array backend for stages 2/3.",
)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu"]),
    default="gpu",
    show_default=True,
    help="Device for stage 1 (TotalSegmentator).",
)
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True, help="Re-run stage 1 even when outputs exist.")
@click.option(
    "--dixon-regions",
    default=",".join(dixon_cfg.REGION_ORDER),
    show_default=True,
    help="Comma-separated Dixon regions for stage 1.",
)
@click.option(
    "--no-compress",
    is_flag=True,
    help="Stage 0 writes uncompressed .nii (default is .nii.gz).",
)
@click.option(
    "--container",
    type=click.Path(path_type=Path),
    default=None,
    help="Singularity container (default from config).",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host path to the nvitk source tree (required for --submit sge).",
)
@click.option("--dry-run", is_flag=True, help="(sge) Print commands but do not submit.")
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--debug", is_flag=True, help="Debug mode.")
def main(
    batch: str,
    subjects: str | None,
    pipelines: str,
    stages: str,
    submit: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    dixon_regions: str,
    no_compress: bool,
    container: Path | None,
    src_dir: Path | None,
    dry_run: bool,
    log_level: str,
    debug: bool,
) -> None:
    """Run the full PESA-Fat pipeline (stage 0 + ct-pet-v5 + dixon-v5) for a batch."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    if debug:
        try:
            import debugpy
            debugpy.listen(("localhost", 5678))
        except Exception as exc:
            warn(
                f'debugpy not available. Continuing without debugpy: \n'
                f'Exception: {exc}\n'
            )

    pipelines_sel = [p.strip().lower() for p in pipelines.split(",") if p.strip()]
    unknown_pipes = set(pipelines_sel) - set(PIPELINE_CHOICES)
    if unknown_pipes:
        raise click.BadParameter(
            f"Unknown pipelines {unknown_pipes}. Valid: {PIPELINE_CHOICES}"
        )

    stages_sel = [s.strip().lower() for s in stages.split(",") if s.strip()]
    unknown_stages = set(stages_sel) - set(STAGE_CHOICES)
    if unknown_stages:
        raise click.BadParameter(
            f"Unknown stages {unknown_stages}. Valid: {STAGE_CHOICES}"
        )

    region_tuple = tuple(r.strip().upper() for r in dixon_regions.split(",") if r.strip())
    unknown_regions = set(region_tuple) - set(dixon_cfg.REGIONS)
    if unknown_regions:
        raise click.BadParameter(
            f"Unknown dixon regions {unknown_regions}. Valid: {tuple(dixon_cfg.REGIONS)}"
        )

    lay = layout(
        batch,
        dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
        model_root=model_dir,
    )

    source = "dicom" if "stage0" in stages_sel else "nifti"
    subj_list = _subject_list(lay, subjects, source)

    if submit == "local":
        _run_local(
            lay,
            subj_list,
            pipelines_sel,
            stages_sel,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
            regions=region_tuple,
            compress=not no_compress,
        )
        log.info("=" * 78)
        log.info(f"PESA-Fat batch '{batch}' complete.")
        log.info("=" * 78)
        return

    if container is None:
        container = ctpet_cfg.CONTAINER_PATH
    if src_dir is None:
        raise click.UsageError("--src-dir is required for --submit sge")

    _run_sge(
        lay,
        subj_list,
        pipelines_sel,
        stages_sel,
        backend=backend,
        device=device,
        model_dir=model_dir,
        overwrite=overwrite,
        regions=region_tuple,
        container=container,
        src_dir=src_dir,
        dry_run=dry_run,
        log_level=log_level,
    )
    log.info("=" * 78)
    log.info(f"PESA-Fat batch '{batch}' submitted.")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
