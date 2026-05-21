"""Dixon v5 pipeline master.

Entry point for the whole Dixon v5 pipeline (stages 1-3). Same shape as
:mod:`nvitk.pipes.pesa_fat.ct_pet_v5.run`: dispatches each stage either
locally (in-process loop) or on SGE (one chain per subject with
``-hold_jid`` between stages). ``--base-hold`` should be set when a stage 0
job is already in queue so every subject's stage 1 waits on its stage 0.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Callable, TextIO

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
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_chain,
    write_script_header,
)
from nvitk.pipes.pesa_fat.common.stage3_batch_summary import aggregate_stage3_summary
from nvitk.pipes.pesa_fat.dixon_v5 import (
    config as cfg,
    stage1_segment,
    stage2_postprocess,
    stage3_measure,
)


log = Logger()


STAGE_MODULES: dict[str, str] = {
    "stage1": "nvitk.pipes.pesa_fat.dixon_v5.stage1_segment",
    "stage2": "nvitk.pipes.pesa_fat.dixon_v5.stage2_postprocess",
    "stage3": "nvitk.pipes.pesa_fat.dixon_v5.stage3_measure",
}


def _local_runner(stage: str) -> Callable[..., Any]:
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
    regions: tuple[str, ...],
) -> None:
    for subj in subjects:
        log.info(f"=== Dixon v5 LOCAL | subject={subj} | stages={stages_sel} ===")
        for s in stages_sel:
            try:
                if s == "stage1":
                    stage1_segment.run_subject(
                        subj,
                        lay,
                        device=device,
                        model_dir=model_dir,
                        overwrite=overwrite,
                        regions=regions,
                    )
                elif s == "stage2":
                    stage2_postprocess.run_subject(subj, lay, backend=backend)
                elif s == "stage3":
                    stage3_measure.run_subject(subj, lay, backend=backend)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                log.error(f"[{subj}] {s} failed: {exc}")

    if "stage3" in stages_sel:
        _aggregate_stage3(lay, subjects)


def _aggregate_stage3(lay: BatchLayout, subjects: list[str]) -> Path | None:
    return aggregate_stage3_summary(lay, subjects, "dixon-v5")


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
    regions: tuple[str, ...],
    log_level: str,
) -> str:
    """Build the per-stage ``python -m ...`` command using *in-container* paths.

    See :func:`nvitk.pipes.pesa_fat.ct_pet_v5.run._build_python_cmd` for the
    path remapping rationale.
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
    ]
    if stage == "stage1":
        parts += [
            "--device", device,
            "--regions", shlex.quote(",".join(regions)),
            "--model-dir", shlex.quote(c_model),
        ]
        if overwrite:
            parts.append("--overwrite")
    else:
        parts += ["--backend", backend]
    return " ".join(parts)


def _cluster_paths(lay: BatchLayout, container: Path, src_dir: Path) -> ClusterPaths:
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
    regions: tuple[str, ...] = cfg.REGION_ORDER,
    base_hold: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    emit: TextIO | None = None,
) -> list[str]:
    """Submit the Dixon v5 SGE chain for a *single* subject.

    Stages are submitted in order with ``-hold_jid`` linking them. Returns
    the list of jids (one per stage, in ``stages_sel`` order).
    """
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

    specs: list[StageSpec] = []
    for s in stages_sel:
        resources = _stage_resources(s, device=device)
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
                    regions=regions,
                    log_level=log_level,
                ),
                resources=resources,
                binds=binds,
                use_nv=True,
                extra_env={
                    "PYTHONPATH": str(binds.src),
                    "TOTALSEG_HOME_DIR": str(binds.models),
                }
            )
        )
    jids = submit_chain(specs, paths, base_hold=base_hold, dry_run=dry_run, emit=emit)
    log.info(f"[{subject}] Dixon v5 SGE chain jids: {jids}")
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
    regions: tuple[str, ...],
    container: Path,
    src_dir: Path,
    base_hold: str | None,
    dry_run: bool,
    log_level: str,
    emit: TextIO | None = None,
) -> dict[str, list[str]]:
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
            regions=regions,
            base_hold=base_hold,
            dry_run=dry_run,
            log_level=log_level,
            emit=emit,
        )
    return all_jids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("nvitk-pesa-fat-dixon")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week4').")
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
    "--regions",
    default=",".join(cfg.REGION_ORDER),
    show_default=True,
    help="Comma-separated regions for stage 1 (HEAD, THORAX, LEGS).",
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
    "--backend",
    type=click.Choice(["cupy", "numpy"], case_sensitive=False),
    default="cupy",
    show_default=True,
)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu"]),
    default="gpu",
    show_default=True,
)
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
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
@click.option("--dry-run", is_flag=True)
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
def main(
    batch: str,
    subjects: str | None,
    stages: str,
    regions: str,
    submit: str,
    base_hold: str | None,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
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
) -> None:
    """Dixon v5 pipeline master (local or SGE dispatch)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    if debug:
        try:
            import debugpy
            debugpy.listen(("localhost", 5678))
        except Exception as exc:
            log.warning(f'debugpy not available. Continuing without debugpy: \nException: {exc}')

    lay = layout(
        batch,
        dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
        model_root=model_dir or cfg.MODELS_PATH,
    )

    stages_sel = [s.strip() for s in stages.split(",") if s.strip()]
    unknown_stages = set(stages_sel) - set(STAGE_MODULES)
    if unknown_stages:
        raise click.BadParameter(
            f"Unknown stages {unknown_stages}. Valid: {tuple(STAGE_MODULES)}"
        )

    region_tuple = tuple(r.strip().upper() for r in regions.split(",") if r.strip())
    unknown_regions = set(region_tuple) - set(cfg.REGIONS)
    if unknown_regions:
        raise click.BadParameter(
            f"Unknown regions {unknown_regions}. Valid: {tuple(cfg.REGIONS)}"
        )

    subj_list = parse_subjects(subjects) or list(lay.iter_subjects())
    if not subj_list:
        raise click.ClickException(
            f"No PESA* subjects found under {lay.nifti_dir} (use --subjects to override)."
        )

    if submit == "local":
        _run_local(
            lay,
            subj_list,
            stages_sel,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
            regions=region_tuple,
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
                title=f"dixon-v5 batch={batch}",
            )
            _run_sge(
                lay,
                subj_list,
                stages_sel,
                backend=backend,
                device=device,
                model_dir=model_dir,
                overwrite=overwrite,
                regions=region_tuple,
                container=container,
                src_dir=src_dir,
                base_hold=base_hold,
                dry_run=False,
                log_level=log_level,
                emit=fh,
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
        regions=region_tuple,
        container=container,
        src_dir=src_dir,
        base_hold=base_hold,
        dry_run=dry_run,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
