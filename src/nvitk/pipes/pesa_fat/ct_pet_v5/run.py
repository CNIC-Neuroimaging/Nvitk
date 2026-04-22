"""CT-PET v5 pipeline master.

Entry point for the whole CT-PET v5 pipeline (stages 1-3). Dispatches each
stage either **locally** (in-process loop over subjects) or on **SGE** (one
per-subject chain with ``-hold_jid`` between stages). Stage 0 is *not* part
of this master — it is orchestrated by :mod:`nvitk.pipes.pesa_fat.run_batch`
and its ``jid`` can be forwarded here via ``--base-hold``.

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
queue, so every subject's stage 1 waits for its stage 0 sibling.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Callable, TextIO

import click
import pandas as pd

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
    submit_chain,
    write_script_header,
)
from nvitk.pipes.pesa_fat.ct_pet_v5 import (
    config as cfg,
    stage1_segment,
    stage2_postprocess,
    stage3_measure,
)


log = Logger()


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


STAGE_MODULES: dict[str, str] = {
    "stage1": "nvitk.pipes.pesa_fat.ct_pet_v5.stage1_segment",
    "stage2": "nvitk.pipes.pesa_fat.ct_pet_v5.stage2_postprocess",
    "stage3": "nvitk.pipes.pesa_fat.ct_pet_v5.stage3_measure",
}


def _local_runner(stage: str) -> Callable[..., Any]:
    if stage == "stage1":
        return stage1_segment.run_subject
    if stage == "stage2":
        return stage2_postprocess.run_subject
    if stage == "stage3":
        return stage3_measure.run_subject
    raise ValueError(f"Unknown stage '{stage}'")


def _stage_resources(stage: str) -> SgeResources:
    """Stage 1 requests a GPU; stages 2/3 are CPU-only."""
    if stage == "stage1":
        return SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=cfg.SGE_NGPU,
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        )
    return SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=cfg.SGE_CPU_NGPU,
        h_vmem=cfg.SGE_CPU_H_VMEM,
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
) -> None:
    for subj in subjects:
        log.info(f"=== CT-PET v5 LOCAL | subject={subj} | stages={stages_sel} ===")
        for s in stages_sel:
            try:
                if s == "stage1":
                    stage1_segment.run_subject(
                        subj, lay, device=device, model_dir=model_dir, overwrite=overwrite
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
    """Concatenate per-subject stage-3 xlsx files into ``<batch>_SummaryCodebook.xlsx``."""
    per_subject = lay.results_dir / cfg.STAGE3_DIR / "per_subject"
    if not per_subject.exists():
        log.warning(f"Stage 3 aggregation skipped: {per_subject} not found")
        return None

    rows: list[dict[str, Any]] = []
    for subj in subjects:
        f = per_subject / f"{subj}.xlsx"
        if not f.exists():
            log.warning(f"Stage 3 aggregation: {f} missing")
            continue
        df = pd.read_excel(f)
        if not df.empty:
            rows.append(df.iloc[0].to_dict())

    if not rows:
        log.warning("Stage 3 aggregation: no per-subject rows collected")
        return None

    out = lay.results_dir / cfg.STAGE3_DIR / f"{lay.batch}_SummaryCodebook.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(rows, columns=stage3_measure.column_order())
    df_all.to_excel(out, index=False)
    log.info(f"Stage 3 aggregate written: {out}")
    return out


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
        "python",
        "-m",
        module,
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
        parts += ["--device", device, "--model-dir", shlex.quote(c_model)]
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
    base_hold: str | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    emit: TextIO | None = None,
) -> list[str]:
    """Submit the CT-PET v5 SGE chain for a *single* subject.

    Stages are submitted in order with ``-hold_jid`` linking them. The first
    stage holds on *base_hold* (e.g. the subject's stage-0 jid).

    Returns the list of jids (one per stage, in ``stages_sel`` order).
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
        resources = _stage_resources(s)
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
                ),
                resources=resources,
                binds=binds,
                use_nv=resources.ngpu > 0,
                extra_env={
                    "PYTHONPATH": str(binds.src + "src/"),
                    "TOTALSEG_HOME_DIR": str(binds.models),
                }
            )
        )
    jids = submit_chain(specs, paths, base_hold=base_hold, dry_run=dry_run, emit=emit)
    log.info(f"[{subject}] CT-PET v5 SGE chain jids: {jids}")
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
            base_hold=base_hold,
            dry_run=dry_run,
            log_level=log_level,
            emit=emit,
        )
    return all_jids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("nvitk-pesa-fat-ctpet")
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
def main(
    batch: str,
    subjects: str | None,
    stages: str,
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
    """CT-PET v5 pipeline master (local or SGE dispatch)."""
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
        _run_local(
            lay,
            subj_list,
            stages_sel,
            backend=backend,
            device=device,
            model_dir=model_dir,
            overwrite=overwrite,
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
    )


if __name__ == "__main__":
    main()
