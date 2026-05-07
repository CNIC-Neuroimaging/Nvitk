"""PESA-Fat batch master.

Single entry point that drives the *whole* PESA-Fat batch — stage 0
(DICOM -> NIfTI + renaming), then the ct-pet-v5 and dixon-v5 pipelines
(stages 1–3), and optional stage 4 (local HTML QC report).

Two execution modes:

* ``--submit local`` — stage 0 is run in-process, then each pipeline's
  master (:mod:`ct_pet_v5.run` / :mod:`dixon_v5.run`) is called with
  ``--submit local`` for the selected stages.
* ``--submit sge`` — for every subject we submit a stage-0 SGE job
  (``qsub`` wrapping ``singularity exec python -m stage0_convert``) and
  capture its jid. Each pipeline's per-subject chain is then submitted
  with ``-hold_jid`` pointing at that jid, so stages 1-2-3 wait for the
  subject's own stage 0 to finish. The CT-PET and Dixon chains of a given
  subject run in parallel after stage 0. Optional ``stage4`` appends a CPU
  QC job that holds on all stage-3 jids when stage 3 is part of the script.

Examples
--------

Local dry-run of the whole batch::

    nvitk-pesa-fat --batch 202602_Week4

SGE: write ``SCRIPTS_CLUSTER/submit_<batch>.sh`` (default path), then try SSH
to run it on the login node (install ``nvitk[cluster]`` for Paramiko)::

    nvitk-pesa-fat --batch 202602_Week4 --submit sge

Override script path with ``--emit-script``; skip SSH with ``--no-remote``.
Default ``--src-dir`` is :data:`nvitk.pipes.pesa_fat.common.paths.DEFAULT_NVITK_SRC_DIR`.

Skip stage 0, run only dixon v5 for a subset locally::

    nvitk-pesa-fat --batch 202602_Week4 \
        --subjects PESA001,PESA002 \
        --pipelines dixon-v5 \
        --stages stage1,stage2,stage3
"""

from __future__ import annotations

import getpass
import shlex
from pathlib import Path
from typing import TextIO

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    CLUSTER_HOST_ALIASES,
    DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT,
    DEFAULT_NVITK_SRC_DIR,
    DEFAULT_RESULTS_ROOT,
    BatchLayout,
    default_submit_script_path,
    layout,
    parse_subjects,
)
from nvitk.pipes.pesa_fat.common.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
    write_script_header,
)
from nvitk.pipes.pesa_fat.common import stage0_convert
from nvitk.pipes.pesa_fat.common.remote_submit import try_run_script_via_ssh
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ctpet_cfg
from nvitk.pipes.pesa_fat.ct_pet_v5 import run as ctpet_run
from nvitk.pipes.pesa_fat.dixon_v5 import config as dixon_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import run as dixon_run
from nvitk.pipes.pesa_fat.common.stage4_qc import run_qc as run_stage4_qc


log = Logger()


PIPELINE_CHOICES = ("ct-pet-v5", "dixon-v5")
STAGE_CHOICES = ("stage0", "stage1", "stage2", "stage3", "stage4")


# ---------------------------------------------------------------------------
# Stage 0 on SGE
# ---------------------------------------------------------------------------


_STAGE0_BIND_DICOM = "/PESAFat/DICOM/"
_STAGE0_BIND_NIFTI = "/PESAFat/NIFTI/"
_STAGE0_MODULE = "nvitk.pipes.pesa_fat.common.stage0_convert"
_AGGREGATE_MODULE = "nvitk.pipes.pesa_fat.common.batch_stage3_aggregate"
_STAGE4_QC_MODULE = "nvitk.pipes.pesa_fat.common.stage4_qc"


def _module_to_script(binds_src: str, module: str) -> str:
    """Convert ``nvitk.pipes.foo.bar`` to the container-side script path
    ``<binds_src>nvitk/pipes/foo/bar.py`` so workers can invoke the file
    directly (``python <path>.py ...``) instead of ``python -m <module>``.
    """
    return f"{binds_src}{module.replace('.', '/')}.py"


def _stage0_python_cmd(subject: str, lay: BatchLayout, log_level: str) -> str:
    """Stage 0 inside the container reads from the mounted DICOM root and
    writes to the mounted NIfTI root; use container paths here."""
    script = _module_to_script(SingularityBinds().src, _STAGE0_MODULE)
    return " ".join(
        [
            "python",
            shlex.quote(script),
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


def _pipeline_cluster_paths(lay: BatchLayout, container: Path, src_dir: Path) -> ClusterPaths:
    """Host paths for stages 1–3 and batch stage-3 aggregation (NIfTI + RESULTS)."""
    return ClusterPaths(
        src=src_dir,
        container=container,
        models=lay.model_root,
        data_root=lay.nifti_root,
        output_root=lay.results_root,
        log_dir=ctpet_cfg.SGE_LOG_DIR,
        err_dir=ctpet_cfg.SGE_ERR_DIR,
    )


def _stage4_qc_python_cmd(
    lay: BatchLayout,
    subjects_csv: str,
    pipelines_csv: str,
    log_level: str,
) -> str:
    binds = SingularityBinds()
    script = _module_to_script(binds.src, _STAGE4_QC_MODULE)
    return " ".join(
        [
            "python",
            shlex.quote(script),
            "--batch",
            shlex.quote(lay.batch),
            "--subjects",
            shlex.quote(subjects_csv),
            "--pipelines",
            shlex.quote(pipelines_csv),
            "--nifti-root",
            shlex.quote(binds.data),
            "--results-root",
            shlex.quote(binds.output),
            "--log-level",
            shlex.quote(log_level),
        ]
    )


def _aggregate_python_cmd(
    lay: BatchLayout,
    subjects_csv: str,
    pipelines_csv: str,
    log_level: str,
) -> str:
    binds = SingularityBinds()
    script = _module_to_script(binds.src, _AGGREGATE_MODULE)
    return " ".join(
        [
            "python",
            shlex.quote(script),
            "--batch",
            shlex.quote(lay.batch),
            "--subjects",
            shlex.quote(subjects_csv),
            "--pipelines",
            shlex.quote(pipelines_csv),
            "--dicom-root",
            shlex.quote(binds.data),
            "--nifti-root",
            shlex.quote(binds.data),
            "--results-root",
            shlex.quote(binds.output),
            "--model-dir",
            shlex.quote(binds.models),
            "--log-level",
            shlex.quote(log_level),
        ]
    )


def _emit_batch_aggregate_stage(
    emit: TextIO,
    lay: BatchLayout,
    subjects: list[str],
    pipelines_agg: list[str],
    *,
    container: Path,
    src_dir: Path,
    hold_jid: list[str],
    log_level: str,
) -> None:
    """Append a CPU qsub that merges per-subject stage-3 xlsx after all stage3 jobs."""
    if not hold_jid or not pipelines_agg:
        return
    paths = _pipeline_cluster_paths(lay, container, src_dir)
    binds = SingularityBinds()
    spec = StageSpec(
        job_name=f"PESAFat_batch_aggregate_{lay.batch}",
        python_cmd=_aggregate_python_cmd(
            lay,
            ",".join(subjects),
            ",".join(pipelines_agg),
            log_level,
        ),
        resources=SgeResources(
            project=ctpet_cfg.SGE_PROJECT,
            account=ctpet_cfg.SGE_ACCOUNT,
            ngpu=ctpet_cfg.SGE_CPU_NGPU,
            h_vmem=ctpet_cfg.SGE_CPU_H_VMEM,
            queue=ctpet_cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env={
            "PYTHONPATH": str(binds.src),
            "TOTALSEG_HOME_DIR": str(binds.models),
        },
    )
    submit_stage(spec, paths, hold_jid=hold_jid, dry_run=False, emit=emit)


def _emit_stage4_qc_stage(
    emit: TextIO,
    lay: BatchLayout,
    subjects: list[str],
    pipelines_qc: list[str],
    *,
    container: Path,
    src_dir: Path,
    hold_jid: list[str] | None,
    log_level: str,
) -> None:
    """Append a CPU qsub that builds the HTML QC report after stage 3 (or immediately)."""
    if not pipelines_qc:
        return
    paths = _pipeline_cluster_paths(lay, container, src_dir)
    binds = SingularityBinds()
    spec = StageSpec(
        job_name=f"PESAFat_stage4_qc_{lay.batch}",
        python_cmd=_stage4_qc_python_cmd(
            lay,
            ",".join(subjects),
            ",".join(pipelines_qc),
            log_level,
        ),
        resources=SgeResources(
            project=ctpet_cfg.SGE_PROJECT,
            account=ctpet_cfg.SGE_ACCOUNT,
            ngpu=ctpet_cfg.SGE_CPU_NGPU,
            h_vmem=ctpet_cfg.SGE_CPU_H_VMEM,
            queue=ctpet_cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env={
            "PYTHONPATH": str(binds.src),
            "TOTALSEG_HOME_DIR": str(binds.models),
        },
    )
    submit_stage(
        spec,
        paths,
        hold_jid=hold_jid if hold_jid else None,
        dry_run=False,
        emit=emit,
    )


def _submit_stage0(
    subject: str,
    lay: BatchLayout,
    *,
    container: Path,
    src_dir: Path,
    dry_run: bool,
    log_level: str,
    device: str = "gpu",
    emit: TextIO | None = None,
) -> str:
    """Submit a single stage-0 SGE job and return its jid (or a shell-variable
    reference when *emit* is set, or ``'DRY_RUN'``)."""
    paths = _stage0_cluster_paths(lay, container, src_dir)
    binds = SingularityBinds(data=_STAGE0_BIND_DICOM, output=_STAGE0_BIND_NIFTI)
    ngpu = ctpet_cfg.SGE_NGPU if device == "gpu" else ctpet_cfg.SGE_CPU_NGPU
    spec = StageSpec(
        job_name=f"PESAFat_stage0_{subject}",
        python_cmd=_stage0_python_cmd(subject, lay, log_level),
        resources=SgeResources(
            project=ctpet_cfg.SGE_PROJECT,
            account=ctpet_cfg.SGE_ACCOUNT,
            ngpu=ngpu,
            h_vmem=ctpet_cfg.SGE_CPU_H_VMEM,
            queue=ctpet_cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=True,
        extra_env={
            "PYTHONPATH": str(binds.src),
            "TOTALSEG_HOME_DIR": str(binds.models),
        }
    )
    jid = submit_stage(spec, paths, dry_run=dry_run, emit=emit)
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
    exclude_ureter: bool = False,
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

    pipe_stages = [s for s in stages_sel if s not in ("stage0", "stage4")]
    if not pipe_stages and "stage4" not in stages_sel:
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
            exclude_ureter=exclude_ureter,
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

    if "stage4" in stages_sel:
        log.info("=" * 78)
        log.info("STAGE 4 QC (local HTML report)")
        log.info("=" * 78)
        run_stage4_qc(
            lay.batch,
            subjects,
            pipelines=list(pipelines),
            nifti_root=lay.nifti_root,
            results_root=lay.results_root,
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
    emit: TextIO | None = None,
    exclude_ureter: bool = False,
) -> None:
    run_stage0 = "stage0" in stages_sel
    pipe_stages = [s for s in stages_sel if s not in ("stage0", "stage4")]

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
                device=device,
                emit=emit,
            )

    stage3_hold_refs: list[str] = []

    if pipe_stages:
        for subj in subjects:
            base_hold = stage0_jids.get(subj)
            if "ct-pet-v5" in pipelines:
                jids_ct = ctpet_run.submit_subject_chain(
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
                    emit=emit,
                    exclude_ureter=exclude_ureter,
                )
                if "stage3" in pipe_stages and jids_ct:
                    stage3_hold_refs.append(jids_ct[-1])
            if "dixon-v5" in pipelines:
                jids_dx = dixon_run.submit_subject_chain(
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
                    emit=emit,
                )
                if "stage3" in pipe_stages and jids_dx:
                    stage3_hold_refs.append(jids_dx[-1])
    elif "stage4" not in stages_sel:
        return

    if (
        emit is not None
        and "stage3" in pipe_stages
        and stage3_hold_refs
    ):
        pipelines_agg = [p for p in pipelines if p in ("ct-pet-v5", "dixon-v5")]
        _emit_batch_aggregate_stage(
            emit,
            lay,
            subjects,
            pipelines_agg,
            container=container,
            src_dir=src_dir,
            hold_jid=stage3_hold_refs,
            log_level=log_level,
        )

    if emit is not None and "stage4" in stages_sel:
        pipelines_qc = [p for p in pipelines if p in ("ct-pet-v5", "dixon-v5")]
        stage4_hold = (
            stage3_hold_refs if ("stage3" in pipe_stages and stage3_hold_refs) else None
        )
        _emit_stage4_qc_stage(
            emit,
            lay,
            subjects,
            pipelines_qc,
            container=container,
            src_dir=src_dir,
            hold_jid=stage4_hold,
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
@click.option("--overwrite", is_flag=True, default=True, help="Re-run stage 1 even when outputs exist.")
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
    default=DEFAULT_NVITK_SRC_DIR,
    show_default=True,
    help="Host path to the nvitk source tree (Singularity bind for --submit sge).",
)
@click.option("--dry-run", is_flag=True, help="(unused for sge script mode) Legacy flag.")
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Bash submission script path. Default: SCRIPTS_CLUSTER/submit_<batch>.sh.",
)
@click.option(
    "--no-remote",
    is_flag=True,
    help="(sge) After writing the script, do not prompt for SSH or run it remotely.",
)
@click.option(
    "--remote-host",
    default=None,
    help="(sge) SSH hostname or alias from CLUSTER_HOST_ALIASES (else prompt).",
)
@click.option(
    "--remote-user",
    default=None,
    help="(sge) SSH username (else prompt).",
)
@click.option("--exclude-ureter", is_flag=True, default=False, help="Exclude ureter from the fat mask.")
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
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    log_level: str,
    debug: bool,
    exclude_ureter: bool = False,
) -> None:
    """Run the full PESA-Fat pipeline (stage 0 + ct-pet-v5 + dixon-v5) for a batch."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    if debug:
        try:
            import debugpy
            debugpy.listen(("localhost", 5678))
        except Exception as exc:
            log.warning(
                "debugpy not available. Continuing without debugpy. Exception: %s",
                exc,
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
            exclude_ureter=exclude_ureter,
        )
        log.info("=" * 78)
        log.info(f"PESA-Fat batch '{batch}' complete.")
        log.info("=" * 78)
        return

    if container is None:
        container = ctpet_cfg.CONTAINER_PATH
    src_resolved = Path(src_dir) if src_dir is not None else DEFAULT_NVITK_SRC_DIR

    script_path = emit_script if emit_script is not None else default_submit_script_path(batch)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=ctpet_cfg.SGE_LOG_DIR,
            err_dir=ctpet_cfg.SGE_ERR_DIR,
            title=f"batch={batch} pipelines={','.join(pipelines_sel)}",
        )
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
            src_dir=src_resolved,
            dry_run=dry_run,
            log_level=log_level,
            emit=fh,
            exclude_ureter=exclude_ureter,
        )
    log.info("=" * 78)
    log.info("PESA-Fat batch '%s' script written: %s", batch, script_path)
    log.info("On the cluster login node: bash %s", script_path)
    log.info("=" * 78)

    if no_remote:
        return

    log.reset(restart_progress=False)
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host_resolved = CLUSTER_HOST_ALIASES.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass("SSH password: ")
    ok = try_run_script_via_ssh(host_resolved, user, password, script_path)
    if not ok:
        log.warning(
            "Remote execution did not complete successfully. Run manually on the "
            "cluster: bash %s",
            script_path,
        )


if __name__ == "__main__":
    main()
