"""CT-PET v5 pipeline master.

Entry point for the whole CT-PET v5 pipeline (stages 1-3). Dispatches each
stage either **locally** (in-process loop over subjects) or on **SGE** (one
array job per subject; tasks = stages with ``-tc 1`` + done-markers). Stage 0
is *not* part of this master — it is orchestrated by
:mod:`nvitk.pipes.pesa_fat.run_batch` and its ``jid`` can be forwarded here via
``--base-hold``.

Examples
--------
Local run for one subject, only stage 2::

    nvitk-pesa-fat-ctpet --batch 202602_Week4 --subjects PESA001 \
        --stages stage2 --submit local

Submit the full chain of every subject on the cluster::

    nvitk-pesa-fat-ctpet --batch 202602_Week4 --submit sge \
        --container /data3/BIOIT_IMAGE/Containers/gpu-pesa-fat_v2025.5.27.sif \
        --src-dir /path/to/nvitk/src

The ``--base-hold`` flag should be set when a stage 0 job is already in
queue, so every subject's array waits for its stage 0 sibling.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Callable, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT,
    BatchLayout,
    layout,
    parse_subjects,
)
from nvitk.pipes.pesa_fat.common import stage0_convert
from nvitk.pipes.pesa_fat.common.xnat_inputs import XnatPesaFatRequest, download_pesa_fat_dicoms_from_xnat
from nvitk.pipes.pesa_fat.common.db_publish import publish_stage3_excel
from nvitk.pipes.pesa_fat.common.sge_array import (
    array_marker_dir,
    sge_pesa_array_resources,
    submit_subject_stage_array,
)
from nvitk.pipes.pesa_fat.common.sge_db import pesa_fat_sge_db_submission
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    write_script_header,
)
from nvitk.pipes.pesa_fat.common.stage3_batch_summary import aggregate_stage3_summary
from nvitk.pipes.pesa_fat.ct_pet_v5 import (
    config as cfg,
    stage1_segment,
    stage2_postprocess,
    stage3_measure,
)


log = Logger()
INPUT_SOURCE_CHOICES = ("paths", "xnat")


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


STAGE_MODULES: dict[str, str] = {
    "stage1": "nvitk.pipes.pesa_fat.ct_pet_v5.stage1_segment",
    "stage2": "nvitk.pipes.pesa_fat.ct_pet_v5.stage2_postprocess",
    "stage3": "nvitk.pipes.pesa_fat.ct_pet_v5.stage3_measure",
}


def _local_runner(stage: str) -> Callable[..., Any]:
    """Return the local ``run_subject`` callable for *stage*; raises ``ValueError`` for unknown stages."""
    if stage == "stage1":
        return stage1_segment.run_subject
    if stage == "stage2":
        return stage2_postprocess.run_subject
    if stage == "stage3":
        return stage3_measure.run_subject
    raise ValueError(f"Unknown stage '{stage}'")


def _stage_resources(stage: str, *, device: str = "gpu") -> SgeResources:
    """Per-stage SGE resource request.

    Stage 1 always uses the GPU h_vmem config. When *device* is ``"gpu"``,
    every stage requests ``ngpu=SGE_NGPU`` (so stages 2/3 run on a GPU node
    too); when ``"cpu"``, every stage requests ``ngpu=SGE_CPU_NGPU``.
    """
    ngpu = cfg.SGE_NGPU if device == "gpu" else cfg.SGE_CPU_NGPU
    h_vmem = cfg.SGE_H_VMEM if stage == "stage1" else cfg.SGE_CPU_H_VMEM
    return SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=ngpu,
        h_vmem=h_vmem,
        queue=cfg.SGE_QUEUE,
    )


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def _run_local(
    lay: BatchLayout,
    subjects: list[str],
    stages_sel: list[str],
    *,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    exclude_ureter: bool = True,
) -> None:
    """Run the requested CT-PET v5 stages for every subject sequentially in-process, publishing stage-3
    results to the DB and aggregating the batch summary once stage-3 has run for any subject."""
    for subj in subjects:
        log.info(f"=== CT-PET v5 LOCAL | subject={subj} | stages={stages_sel} ===")
        for s in stages_sel:
            try:
                if s == "stage1":
                    stage1_segment.run_subject(
                        subj, lay, device=device, model_dir=model_dir, overwrite=overwrite
                    )
                elif s == "stage2":
                    stage2_postprocess.run_subject(subj, lay, backend=backend, exclude_ureter=exclude_ureter)
                elif s == "stage3":
                    stage3_measure.run_subject(subj, lay, backend=backend)
                    try:
                        per_subject = lay.results_dir / cfg.STAGE3_DIR / "per_subject" / f"{subj}.xlsx"
                        if per_subject.exists():
                            publish_stage3_excel(
                                subject_uid=subj,
                                excel_path=per_subject,
                                pipeline="ct-pet-v5",
                                source_batch_id=lay.batch,
                            )
                    except Exception as exc:
                        import traceback
                        log.warning(traceback.format_exc())
                        log.warning("DB publish skipped for %s (ct-pet-v5): %s", subj, exc)
            except Exception as exc:
                import traceback
                log.warning(traceback.format_exc())
                log.error(f"[{subj}] {s} failed: {exc}")

    if "stage3" in stages_sel:
        _aggregate_stage3(lay, subjects)


def _aggregate_stage3(lay: BatchLayout, subjects: list[str]) -> Path | None:
    """Concatenate per-subject stage-3 xlsx files into ``<batch>_SummaryCodebook.xlsx``."""
    return aggregate_stage3_summary(lay, subjects, "ct-pet-v5")


# ---------------------------------------------------------------------------
# SGE execution
# ---------------------------------------------------------------------------


def _build_python_cmd(
    stage: str,
    subject: str,
    lay: BatchLayout,
    *,
    binds: SingularityBinds,
    backend: str,
    device: str,
    overwrite: bool,
    log_level: str,
    exclude_ureter: bool = True,
) -> str:
    """Build the ``python -m nvitk.pipes...stageX --batch ... --subject ...`` cmd.

    Uses *in-container* paths so the worker sees the bind-mounted locations:

    * ``--nifti-root``   -> ``binds.data``      (e.g. ``/PESAFat/data/``)
    * ``--results-root`` -> ``binds.output``    (e.g. ``/PESAFat/output/``)
    * ``--model-dir``    -> ``binds.models``    (e.g. ``/models/``)
    * ``--dicom-root``   is not actually used by stages 1-3, but we still set
      it to ``binds.data`` so :class:`BatchLayout` stays consistent.
    """
    module = STAGE_MODULES[stage]
    c_dicom = binds.data
    c_nifti = binds.data
    c_out = binds.output
    c_model = binds.models

    parts: list[str] = [
        *python_module_argv(module),
        "--batch",
        shlex.quote(lay.batch),
        "--subject",
        shlex.quote(subject),
        "--dicom-root",
        shlex.quote(c_dicom),
        "--nifti-root",
        shlex.quote(c_nifti),
        "--results-root",
        shlex.quote(c_out),
        "--log-level",
        log_level,
        "--no-exclude-ureter" if not exclude_ureter and stage == "stage2" else "",
    ]
    if stage == "stage1":
        parts += ["--device", device, "--model-dir", shlex.quote(c_model)]
        if overwrite:
            parts.append("--overwrite")
    else:
        parts += ["--backend", backend]
    return " ".join(parts)


def _cluster_paths(lay: BatchLayout, container: Path, src_dir: Path) -> ClusterPaths:
    """Build the :class:`~nvitk.cluster.sge.ClusterPaths` (container/model/data/output roots) for
    submitting CT-PET v5 SGE jobs from *lay*."""
    return ClusterPaths(
        src=src_dir,
        container=container,
        models=lay.model_root,
        data_root=lay.nifti_root,
        output_root=lay.results_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )


def submit_subject_chain(
    subject: str,
    lay: BatchLayout,
    stages_sel: list[str],
    *,
    container: Path,
    src_dir: Path,
    backend: str = "cupy",
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = False,
    base_hold: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    emit: TextIO | None = None,
    exclude_ureter: bool = True,
) -> list[str]:
    """Submit one CT-PET v5 SGE array job for a *single* subject.

    Array tasks are the selected stages in order (``-tc 1`` + done-markers).
    *base_hold* is applied to the whole array (e.g. the subject's stage-0 jid).

    Returns ``[jid]`` (one array job id) so batch aggregate/QC can hold on it.
    """
    if not stages_sel:
        return []

    paths = _cluster_paths(lay, container, src_dir)
    if model_dir is not None:
        paths = ClusterPaths(
            src=paths.src,
            container=paths.container,
            models=model_dir,
            data_root=paths.data_root,
            output_root=paths.output_root,
            log_dir=paths.log_dir,
            err_dir=paths.err_dir,
        )
    binds = SingularityBinds()
    db_env, db_binds = pesa_fat_sge_db_submission()

    specs: list[StageSpec] = []
    for s in stages_sel:
        resources = _stage_resources(s, device=device)
        stage_env = {
            "PYTHONPATH": str(binds.src),
            "TOTALSEG_HOME_DIR": str(binds.models),
        }
        stage_binds: tuple[tuple[Path, str], ...] = ()
        if s == "stage3" and db_env:
            stage_env.update(db_env)
            stage_binds = db_binds
        specs.append(
            StageSpec(
                job_name=f"{cfg.SGE_JOB_PREFIX}_{s}_{subject}",
                python_cmd=_build_python_cmd(
                    s,
                    subject,
                    lay,
                    binds=binds,
                    backend=backend,
                    device=device,
                    overwrite=overwrite,
                    log_level=log_level,
                    exclude_ureter=exclude_ureter,
                ),
                resources=resources,
                binds=binds,
                use_nv=True,
                extra_env=stage_env,
                extra_host_binds=stage_binds,
            )
        )

    array_resources, use_nv = sge_pesa_array_resources(
        stages_sel,
        device=device,
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        queue=cfg.SGE_QUEUE,
        h_vmem_stage1=cfg.SGE_H_VMEM,
        h_vmem_cpu=cfg.SGE_CPU_H_VMEM,
        ngpu=cfg.SGE_NGPU,
        cpu_ngpu=cfg.SGE_CPU_NGPU,
    )
    jids = submit_subject_stage_array(
        subject=subject,
        job_prefix=cfg.SGE_JOB_PREFIX,
        stages_sel=stages_sel,
        stage_specs=specs,
        paths=paths,
        resources=array_resources,
        use_nv=use_nv,
        base_hold=base_hold,
        dry_run=dry_run,
        emit=emit,
        marker_dir=array_marker_dir(lay, cfg.SGE_JOB_PREFIX, subject),
    )
    log.info(f"[{subject}] CT-PET v5 SGE array jids: {jids} (tasks={stages_sel})")
    return jids


def _run_sge(
    lay: BatchLayout,
    subjects: list[str],
    stages_sel: list[str],
    *,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    container: Path,
    src_dir: Path,
    base_hold: str | None,
    dry_run: bool,
    log_level: str,
    emit: TextIO | None = None,
    exclude_ureter: bool = True,
) -> dict[str, list[str]]:
    """Submit an SGE job chain (or array) for the requested stages for every subject; returns
    ``{subject: [job_id, ...]}``."""
    all_jids: dict[str, list[str]] = {}
    for subj in subjects:
        all_jids[subj] = submit_subject_chain(
            subj,
            lay,
            stages_sel,
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
    return all_jids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("nvitk-pesa-fat-ctpet")
@backend_click_option()
@click.option(
    "--batch",
    required=False,
    default="auto",
    show_default=True,
    help="Batch name (e.g. '202602_Week4'). Use 'auto' with --input-source xnat.",
)
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated PESA* subjects (default: all under nifti-root).",
)
@click.option(
    "--stages",
    default="stage1,stage2,stage3",
    show_default=True,
    help="Comma-separated stages to run.",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"]),
    default="local",
    show_default=True,
)
@click.option(
    "--base-hold",
    default=None,
    help="SGE jid(s) to wait on before stage 1 (e.g. stage 0 jid per subject).",
)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--input-source",
    type=click.Choice(INPUT_SOURCE_CHOICES, case_sensitive=False),
    default="paths",
    show_default=True,
    help="Input source: on-disk DICOM paths, or download DICOMs from XNAT + run stage0 before stage1-3 (local only).",
)
@click.option(
    "--xnat-config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional XNAT config file (JSON/YAML). If omitted, uses NVITK_XNAT_CONFIG or ~/.config/nvitk/xnat.*",
)
@click.option(
    "--xnat-session",
    default=None,
    help="XNAT experiment label to use (if omitted, uses newest experiment that matches required sequences).",
)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu"]),
    default="gpu",
    show_default=True,
)
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True, help="Re-run stage 1 even when outputs exist.")
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
@click.option(
    "--dry-run",
    is_flag=True,
    help="(sge) Print the qsub+singularity commands but do not submit.",
)
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Write a self-contained bash submission script to this path "
         "instead of submitting. Run it on the cluster login node with "
         "`bash <script>`; only qsub + singularity are required there.",
)
@click.option("--log-level", default="INFO", show_default=True)
@click.option("--debug", is_flag=True, help="Debug mode.")
@click.option(
    "--exclude-ureter/--no-exclude-ureter",
    default=True,
    help="Exclude PET ureter from BATCH visceral fat labels (default: on).",
)
def main(
    batch: str,
    subjects: str | None,
    stages: str,
    submit: str,
    base_hold: str | None,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    input_source: str,
    xnat_config: Path | None,
    xnat_session: str | None,
    backend: str,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    container: Path | None,
    src_dir: Path | None,
    dry_run: bool,
    emit_script: Path | None,
    log_level: str,
    debug: bool,
    exclude_ureter: bool = True,
) -> None:
    """CT-PET v5 pipeline master (local or SGE dispatch)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    if debug:
        try:
            import debugpy
            debugpy.listen(("localhost", 5678))
        except Exception as exc:
            log.warning(f'debugpy not available. Continuing without debugpy: \nException: {exc}')

    if str(batch).strip().lower() == "auto" and str(input_source).strip().lower() != "xnat":
        raise click.ClickException("--batch auto is only valid with --input-source xnat.")
    lay = layout(
        batch,
        dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
        model_root=model_dir or cfg.MODELS_PATH,
    )

    stages_sel = [s.strip() for s in stages.split(",") if s.strip()]
    unknown = set(stages_sel) - set(STAGE_MODULES)
    if unknown:
        raise click.BadParameter(
            f"Unknown stages {unknown}. Valid: {tuple(STAGE_MODULES)}"
        )

    subj_list = parse_subjects(subjects) or list(lay.iter_subjects())
    if not subj_list:
        raise click.ClickException(
            f"No PESA* subjects found under {lay.nifti_dir} (use --subjects to override)."
        )

    if submit == "local":
        src = str(input_source).strip().lower()
        if src == "xnat":
            req = XnatPesaFatRequest(project_id="IA_PET_V5", session_label=xnat_session)
            dl = download_pesa_fat_dicoms_from_xnat(
                batch=batch,
                subjects=subj_list,
                dicom_root=lay.dicom_root,
                xnat_config_path=xnat_config,
                request=req,
            )
            # Recompute per-subject layout batch (may be auto-derived).
            for subj in subj_list:
                subj_batch, _ = dl.get(subj, (lay.batch, {}))
                lay_s = lay if subj_batch == lay.batch else layout(
                    subj_batch,
                    dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
                    nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
                    results_root=results_root or DEFAULT_RESULTS_ROOT,
                    model_root=model_dir or cfg.MODELS_PATH,
                )
                stage0_convert.run_subject(subj, lay_s, compress=True)
        _run_local(
            lay,
            subj_list,
            stages_sel,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
            exclude_ureter=exclude_ureter,
        )
        return

    if container is None:
        container = cfg.CONTAINER_PATH
    if src_dir is None:
        raise click.UsageError("--src-dir is required for --submit sge")

    if emit_script is not None:
        emit_script.parent.mkdir(parents=True, exist_ok=True)
        with open(emit_script, "w", encoding="utf-8") as fh:
            write_script_header(
                fh,
                log_dir=cfg.SGE_LOG_DIR,
                err_dir=cfg.SGE_ERR_DIR,
                title=f"ct-pet-v5 batch={batch}",
            )
            _run_sge(
                lay,
                subj_list,
                stages_sel,
                backend=backend,
                device=device,
                model_dir=model_dir,
                overwrite=overwrite,
                container=container,
                src_dir=src_dir,
                base_hold=base_hold,
                dry_run=False,
                log_level=log_level,
                emit=fh,
                exclude_ureter=exclude_ureter,
            )
        log.info(f"Wrote submission script: {emit_script}")
        return

    _run_sge(
        lay,
        subj_list,
        stages_sel,
        backend=backend,
        device=device,
        model_dir=model_dir,
        overwrite=overwrite,
        container=container,
        src_dir=src_dir,
        base_hold=base_hold,
        dry_run=dry_run,
        log_level=log_level,
        exclude_ureter=exclude_ureter,
    )


if __name__ == "__main__":
    main()
