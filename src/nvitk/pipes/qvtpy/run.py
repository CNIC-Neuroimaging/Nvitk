"""qvtpy pipeline master.

This mirrors the PESA-Fat runner shape: local execution vs. SGE submission via
a bash script + SSH to the cluster login node.

Stages (select with ``--stages``; default ``stage0_c,stage1``)
--------------------------------------------------------------
- ``stage0_d`` — XNAT → DICOM (local only).
- ``stage0_c`` — DICOM → NIfTI + reorg + optional ``phase2volume`` derivatives.
- ``stage1`` — eICAB on ``TOF/TOF.nii.gz``.
- ``stage2`` — eICAB ``TOF_resampled`` → 4D flow rigid FLIRT (NiPype FSL; fixed = Angiography_3D or CD).
- ``stage3`` — eICAB in 4Dflow space + arterial/venous centerlines (``--eicab-mask``, geometry venous branches).
- ``stage4`` — Local CD crop per vessel + threshold + optional region growing → ``seg_4dflow``.
- ``stage4t`` — Same as stage 4 on each ``ComplexDifference_4D`` frame → ``seg_4dflow_4d`` (opt-in).
- ``stage5`` — QVTplus-style LOC CSV (arterial + venous).
- ``stage6`` — Per-LOC masked-plane flow / PI / RI from phase volumes.

SGE: per subject, stages emit in order with ``-hold_jid`` chaining (see
:func:`nvitk.cluster.sge.submit_stage`).
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
    python_module_argv,
    submit_stage,
    write_script_header,
)
from nvitk.core.click_backend import backend_click_option, sge_backend_env
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.segmentation.eicab import config as eicab_cfg

from . import config as cfg
from . import (
    stage0_convert,
    stage0_download,
    stage1_eicab,
    stage2_registration,
    stage3_centerline,
    stage4_4dflow_segmentation,
    stage4t_4dflow_t_segmentation,
    stage5_loc_generation,
    stage6_measure,
)


log = Logger()

# ---------------------------------------------------------------------------
# Stage identifiers and aliases
# ---------------------------------------------------------------------------

STAGE_DOWNLOAD = "stage0_d"
STAGE_CONVERT = "stage0_c"
STAGE_EICAB = "stage1"
STAGE_REG = "stage2"
STAGE_CENTERLINE = "stage3"
STAGE_SEG = "stage4"
STAGE_SEG_T = "stage4t"
STAGE_LOC = "stage5"
STAGE_MEASURE = "stage6"

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
    "stage2": STAGE_REG,
    "stage2_registration": STAGE_REG,
    "registration": STAGE_REG,
    "stage3": STAGE_CENTERLINE,
    "stage3_centerline": STAGE_CENTERLINE,
    "centerline": STAGE_CENTERLINE,
    "stage4": STAGE_SEG,
    "stage4_4dflow_segmentation": STAGE_SEG,
    "segmentation": STAGE_SEG,
    "stage4t": STAGE_SEG_T,
    "stage4t_4dflow_t_segmentation": STAGE_SEG_T,
    "segmentation_t": STAGE_SEG_T,
    "seg_t": STAGE_SEG_T,
    "stage5": STAGE_LOC,
    "stage5_loc_generation": STAGE_LOC,
    "loc": STAGE_LOC,
    "stage6": STAGE_MEASURE,
    "stage6_measure": STAGE_MEASURE,
    "measure": STAGE_MEASURE,
}

_STAGES_ORDERED: tuple[str, ...] = (
    STAGE_DOWNLOAD,
    STAGE_CONVERT,
    STAGE_EICAB,
    STAGE_REG,
    STAGE_CENTERLINE,
    STAGE_SEG,
    STAGE_SEG_T,
    STAGE_LOC,
    STAGE_MEASURE,
)

_ALL_STAGES: tuple[str, ...] = _STAGES_ORDERED
DEFAULT_STAGES: str = f"{STAGE_CONVERT},{STAGE_EICAB},{STAGE_REG},{STAGE_CENTERLINE},{STAGE_SEG},{STAGE_LOC},{STAGE_MEASURE}"

_STAGE_LABELS: dict[str, str] = {
    STAGE_DOWNLOAD: "XNAT download",
    STAGE_CONVERT: "DICOM → NIfTI",
    STAGE_EICAB: "eICAB (TOF)",
    STAGE_REG: "FLIRT TOF → 4D flow",
    STAGE_CENTERLINE: "centerlines + venous",
    STAGE_SEG: "CD segmentation (4D)",
    STAGE_SEG_T: "CD segmentation (4D+t)",
    STAGE_LOC: "LOC generation",
    STAGE_MEASURE: "flow measurement",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_stages(spec: str) -> list[str]:
    """Parse ``--stages`` comma list into canonical stage ids in pipeline order."""
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise click.ClickException("--stages cannot be empty.")
    canonical: set[str] = set()
    for tok in tokens:
        key = tok.replace("-", "_")
        if key not in _STAGE_ALIASES:
            raise click.ClickException(
                f"Unknown stage {tok!r}. Valid: {', '.join(sorted(set(_STAGE_ALIASES.keys())))}."
            )
        canonical.add(_STAGE_ALIASES[key])
    return [s for s in _STAGES_ORDERED if s in canonical]


def _iter_subjects(root: Path) -> list[str]:
    """Sorted subject folder names under *root* (one directory per subject)."""
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _default_nvitk_src_dir() -> Path:
    """Host tree mounted at ``/nvitk/src/`` in SGE Singularity jobs."""
    return cfg.NVITK_SRC_DIR


def _default_submit_script_path() -> Path:
    """Timestamped bash script under :data:`~nvitk.pipes.qvtpy.config.SGE_SCRIPTS_DIR`."""
    cfg.SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.SGE_SCRIPTS_DIR / f"submit_qvtpy_{ts}.sh"


# ---------------------------------------------------------------------------
# SGE script emission (per-stage submit_stage)
# ---------------------------------------------------------------------------


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
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    backend: str = "gpu",
) -> str:
    """Append one stage0_c :func:`~nvitk.cluster.sge.submit_stage` block; return job id."""
    binds = SingularityBinds()
    cmd_parts: list[str] = [
        *python_module_argv("nvitk.pipes.qvtpy.stage0_convert"),
        "--backend",
        shlex.quote(backend),
        "--subject",
        shlex.quote(subject),
        "--dicom-root",
        shlex.quote(binds.data),
        "--nifti-root",
        shlex.quote(binds.output),
        "--phase-bg-poly-order",
        str(int(phase_bg_poly_order)),
        "--phase-bg-static-percentile",
        str(float(phase_bg_static_percentile)),
    ]
    if compute_phase_derived:
        cmd_parts.append("--compute-phase-derived")
    if phase_background_correction:
        cmd_parts.append("--phase-background-correction")
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
        extra_env=sge_backend_env(binds.src, backend),
    )
    return submit_stage(spec, paths, emit=fh)


# ---------------------------------------------------------------------------
# Master CLI (local + SGE)
# ---------------------------------------------------------------------------


@click.command("nvitk-qvtpy")
@backend_click_option()
@click.option("--dicom-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_DICOM_ROOT)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option(
    "--stages",
    "stages_spec",
    default=DEFAULT_STAGES,
    show_default=True,
    help="Comma-separated stages (see module docstring for names and aliases).",
)
@click.option("--subjects", default=None, help="Comma-separated subject list.")
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
    help="Pipeline Singularity image for SGE stages.",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host tree mounted at /nvitk/src/.",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_RESULTS_ROOT,
    show_default=True,
    help="Results root (eICAB + qvtpy stages).",
)
@click.option("--emit-script", type=click.Path(path_type=Path), default=None)
@click.option("--no-remote", is_flag=True)
@click.option("--remote-host", default=None)
@click.option("--remote-user", default=None)
# --- stage 0 ---
@click.option("--compute-phase-derived", is_flag=True, default=True)
@click.option(
    "--phase-background-correction/--no-phase-background-correction",
    "phase_background_correction",
    is_flag=True,
    default=True,
    show_default=True,
    help="phase2volume: polynomial velocity background correction (default on).",
)
@click.option("--phase-bg-poly-order", type=int, default=2, show_default=True)
@click.option("--phase-bg-static-percentile", type=float, default=25.0, show_default=True)
@click.option(
    "--sequences",
    default=",".join(stage0_download.DEFAULT_SEQUENCES),
    show_default=True,
    help="Sequences for stage0_d.",
)
@click.option("--xnat-config", "xnat_config_path", type=click.Path(path_type=Path), default=None)
@click.option("--report", is_flag=True, default=False)
@click.option("--report-derived", is_flag=True, default=False)
# --- stage 1 ---
@click.option("--eicab-container", type=click.Path(path_type=Path), default=None)
@click.option("--vasculature-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--eicab-device",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
)
@click.option("--eicab-resolution", type=float, default=0.5, show_default=True)
@click.option(
    "--post-process-eicab/--no-post-process-eicab",
    default=True,
    show_default=True,
    help="Stage1: Otsu ICA resegment + ICA region growing after eICAB.",
)
@click.option(
    '--only-pp/--no-only-pp',
    default=False,
    show_default=True,
    help="Stage1: Only run ICA post-process on existing eICAB outputs.",
)
# --- stage 2 ---
@click.option(
    "--stage2-reference",
    type=click.Choice(["angio", "cd"]),
    default="angio",
    show_default=True,
    help="Fixed image for FLIRT (stage2).",
)
@click.option("--stage2-dof", type=int, default=6, show_default=True)
@click.option("--stage2-cost", type=str, default="normmi", show_default=True)
# --- stage 3 ---
@click.option(
    "--eicab-mask",
    type=click.Choice(["cw", "wb"], case_sensitive=False),
    default="cw",
    show_default=True,
    help="Stage3: prefer Circle-of-Willis or whole-brain eICAB mask.",
)
@click.option("--cd-up-thresh", type=float, default=None, help="Stage3: CD sliding-threshold upper fraction.")
@click.option(
    "--cd-shift-hm/--no-cd-shift-hm",
    default=None,
    help="Stage3: FWHM shift on threshold curve (default on).",
)
@click.option("--venous-min-component-frac", type=float, default=0.005, show_default=True)
@click.option("--eicab-min-island-fraction", type=float, default=0.05, show_default=True)
@click.option("--eicab-bridge-open-radius", type=int, default=0, show_default=True)
@click.option("--venous-min-branch-points", type=int, default=5, show_default=True)
# --- stage 4 ---
@click.option("--crop-padding-bbox", type=int, default=3, show_default=True, help="Stage4: bbox padding (vox).")
@click.option(
    "--4dflow-thr-algorithm",
    "thr_algorithm_4dflow",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="otsu",
    show_default=True,
    help="Stage4: local threshold on CD crop.",
)
@click.option(
    "--region-growing/--no-region-growing",
    default=True,
    show_default=True,
    help="Stage4: grow labels into unassigned high-CD voxels.",
)
@click.option("--rg-intensity-frac", type=float, default=0.45, show_default=True, help="Stage4: RG intensity factor (default vessels).")
@click.option("--rg-intensity-frac-explore", type=float, default=0.05, show_default=True, help="Stage4: RG frac for ACA/MCA/PCA.")
@click.option("--cl-barrier-radius", type=int, default=3, show_default=True, help="Stage4: dilate other centerlines (vox).")
@click.option("--rg-barrier-radius", type=int, default=1, show_default=True, help="Stage4: dilate other seg during RG (vox).")
@click.option(
    "--aca-sequential-grow/--no-aca-sequential-grow",
    default=True,
    show_default=True,
    help="Stage4: LACA then RACA grow; overlap corrected at junction when needed.",
)
@click.option(
    "--aca-overlap-min-voxels",
    type=int,
    default=10,
    show_default=True,
    help="Stage4: min LACA∩RACA voxels to apply convergence correction at AComm.",
)
@click.option(
    "--acomm-junction-radius",
    type=int,
    default=10,
    show_default=True,
    help="Stage4: Voronoi-split overlap only within this many vox of AComm junction.",
)
# --- stage 5 ---
@click.option(
    "--loc-arterial-strategy",
    type=click.Choice(["qvtpy", "midpoint"]),
    default="qvtpy",
    show_default=True,
)
@click.option("--cross-section-radius-vox", type=float, default=10.0, show_default=True)
@click.option("--loc-endpoint-inset-frac", type=float, default=0.08, show_default=True, help="Stage5: dual LOC inset from polyline ends.")
# --- stage 6 ---
@click.option(
    "--measure-resegment/--no-measure-resegment",
    default=True,
    show_default=True,
)
@click.option(
    "--measure-thr-algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="lthr",
    show_default=True,
    help="Stage6: in-plane threshold when --measure-resegment.",
)
@click.option("--cross-section-res", type=int, default=0, show_default=True)
@click.option("--cross-section-plane-interp", type=int, default=2, show_default=True)
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
    output_root: Path,
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    compute_phase_derived: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    sequences: str,
    xnat_config_path: Path | None,
    report: bool,
    report_derived: bool,
    eicab_container: Path | None,
    vasculature_dir: Path | None,
    eicab_device: str,
    eicab_resolution: float,
    post_process_eicab: bool,
    only_pp: bool,
    stage2_reference: str,
    stage2_dof: int,
    stage2_cost: str,
    eicab_mask: str,
    cd_up_thresh: float | None,
    cd_shift_hm: bool | None,
    venous_min_component_frac: float,
    eicab_min_island_fraction: float,
    eicab_bridge_open_radius: int,
    venous_min_branch_points: int,
    crop_padding_bbox: int,
    thr_algorithm_4dflow: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    loc_arterial_strategy: str,
    cross_section_radius_vox: float,
    loc_endpoint_inset_frac: float,
    measure_resegment: bool,
    measure_thr_algorithm: str,
    cross_section_res: int,
    cross_section_plane_interp: int,
    backend: str,
) -> None:
    Logger()

    stages = _parse_stages(stages_spec)
    run_dl = STAGE_DOWNLOAD in stages
    run_conv = STAGE_CONVERT in stages
    run_eicab = STAGE_EICAB in stages
    run_s2 = STAGE_REG in stages
    run_s3 = STAGE_CENTERLINE in stages
    run_s4 = STAGE_SEG in stages
    run_s4t = STAGE_SEG_T in stages
    run_s5 = STAGE_LOC in stages
    run_s6 = STAGE_MEASURE in stages

    log.info(f"qvtpy | stages={','.join(stages)} | submit={submit}")

    if subjects or subjects_file:
        subject_list = stage0_download.load_subjects(
            subjects=subjects,
            subjects_file=subjects_file,
        )
    else:
        if run_dl:
            raise click.ClickException("stage0_d requires --subjects or --subjects-file.")
        fallback_root = dicom_root if run_conv else nifti_root
        subject_list = _iter_subjects(fallback_root)
        if not subject_list:
            raise click.ClickException(
                f"No subjects to process (looked under {fallback_root}). "
                "Pass --subjects / --subjects-file or populate that root."
            )

    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    active_stages = (
        stages if submit == "local" else [s for s in stages if s == STAGE_DOWNLOAD]
    )

    with PipelineRunTracker(
        log,
        "qvtpy",
        subject_list,
        active_stages,
        stage_labels=_STAGE_LABELS,
    ) as run:
        if run_dl:
            if submit == "sge":
                log.info("stage0_d runs locally; SGE covers other selected stages.")
            from nvitk.db.xnat import requested_sequence_set
            from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

            profile = load_xnat_profile(xnat_config_path)
            conn = resolve_xnat_connection(profile)
            seq_set = requested_sequence_set(sequences) or set(
                stage0_download.DEFAULT_SEQUENCES
            )

            def _download() -> None:
                stage0_download.run_download(
                    subject_list,
                    dicom_root=dicom_root,
                    xnat_config=conn,
                    sequences=seq_set,
                    skip_existing=skip_existing,
                    report=report,
                )

            run.run_stage(
                "(cohort)",
                STAGE_DOWNLOAD,
                _download,
                detail=f"{len(subject_list)} subject(s)",
            )

        if submit == "local":
            for subj in subject_list:
                if run_conv:
                    run.run_stage(
                        subj,
                        STAGE_CONVERT,
                        lambda s=subj: stage0_convert.run_subject(
                            s,
                            dicom_root=dicom_root,
                            nifti_root=nifti_root,
                            compute_phase_derived=compute_phase_derived,
                            skip_existing=skip_existing,
                            phase_background_correction=phase_background_correction,
                            phase_bg_poly_order=phase_bg_poly_order,
                            phase_bg_static_percentile=phase_bg_static_percentile,
                        ),
                    )
                if run_eicab:
                    run.run_stage(
                        subj,
                        STAGE_EICAB,
                        lambda s=subj: stage1_eicab.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            resolution=eicab_resolution,
                            device=eicab_device,
                            eicab_container=eicab_container,
                            vasculature_dir=vasculature_dir,
                            post_process_eicab=post_process_eicab,
                            only_pp=only_pp,
                        ),
                    )
                if run_s2:
                    run.run_stage(
                        subj,
                        STAGE_REG,
                        lambda s=subj: stage2_registration.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            reference=stage2_reference,  # type: ignore[arg-type]
                            dof=stage2_dof,
                            cost=stage2_cost,
                        ),
                    )
                if run_s3:
                    run.run_stage(
                        subj,
                        STAGE_CENTERLINE,
                        lambda s=subj: stage3_centerline.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            eicab_mask=eicab_mask.lower(),  # type: ignore[arg-type]
                            cd_up_thresh=cd_up_thresh,
                            cd_shift_hm=cd_shift_hm,
                            venous_min_component_frac=venous_min_component_frac,
                            eicab_min_island_fraction=eicab_min_island_fraction,
                            eicab_bridge_open_radius=eicab_bridge_open_radius,
                            venous_min_branch_points=venous_min_branch_points,
                        ),
                    )
                if run_s4:
                    run.run_stage(
                        subj,
                        STAGE_SEG,
                        lambda s=subj: stage4_4dflow_segmentation.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            crop_padding_bbox=crop_padding_bbox,
                            thr_algorithm=thr_algorithm_4dflow.lower(),  # type: ignore[arg-type]
                            region_growing=region_growing,
                            rg_intensity_frac=rg_intensity_frac,
                            rg_intensity_frac_explore=rg_intensity_frac_explore,
                            cl_barrier_radius=cl_barrier_radius,
                            rg_barrier_radius=rg_barrier_radius,
                            aca_sequential_grow=aca_sequential_grow,
                            aca_overlap_min_voxels=aca_overlap_min_voxels,
                            acomm_junction_radius=acomm_junction_radius,
                        ),
                    )
                if run_s4t:
                    run.run_stage(
                        subj,
                        STAGE_SEG_T,
                        lambda s=subj: stage4t_4dflow_t_segmentation.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            crop_padding_bbox=crop_padding_bbox,
                            thr_algorithm=thr_algorithm_4dflow.lower(),  # type: ignore[arg-type]
                            region_growing=region_growing,
                            rg_intensity_frac=rg_intensity_frac,
                            rg_intensity_frac_explore=rg_intensity_frac_explore,
                            cl_barrier_radius=cl_barrier_radius,
                            rg_barrier_radius=rg_barrier_radius,
                            aca_sequential_grow=aca_sequential_grow,
                            aca_overlap_min_voxels=aca_overlap_min_voxels,
                            acomm_junction_radius=acomm_junction_radius,
                        ),
                    )
                if run_s5:
                    run.run_stage(
                        subj,
                        STAGE_LOC,
                        lambda s=subj: stage5_loc_generation.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            loc_arterial_strategy=loc_arterial_strategy,
                            cross_section_radius_vox=cross_section_radius_vox,
                            venous_min_component_frac=venous_min_component_frac,
                            loc_endpoint_inset_frac=loc_endpoint_inset_frac,
                        ),
                    )
                if run_s6:
                    run.run_stage(
                        subj,
                        STAGE_MEASURE,
                        lambda s=subj: stage6_measure.run_subject(
                            s,
                            nifti_root=nifti_root,
                            output_root=output_root,
                            skip_existing=skip_existing,
                            cross_section_radius_vox=cross_section_radius_vox,
                            measure_resegment=measure_resegment,
                            measure_thr_algorithm=measure_thr_algorithm.lower(),  # type: ignore[arg-type]
                            cross_section_res=cross_section_res,
                            cross_section_plane_interp=cross_section_plane_interp,
                        ),
                    )

    if submit == "local":
        if run_conv and report:
            stage0_convert.print_nifti_qc_report(
                nifti_root, subject_list, check_derived=report_derived
            )
        return

    cluster_stages = (
        run_conv or run_eicab or run_s2 or run_s3 or run_s4 or run_s4t or run_s5 or run_s6
    )
    if not cluster_stages:
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
            prev_jid: str | None = None
            if run_conv:
                try:
                    prev_jid = _emit_stage0_convert(
                        fh,
                        subj,
                        dicom_root=dicom_root,
                        nifti_root=nifti_root,
                        container=container,
                        src_dir=src_p,
                        compute_phase_derived=compute_phase_derived,
                        skip_existing=skip_existing,
                        phase_background_correction=phase_background_correction,
                        phase_bg_poly_order=phase_bg_poly_order,
                        phase_bg_static_percentile=phase_bg_static_percentile,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage0_c emit skipped: {exc}")

            if run_eicab:
                try:
                    jid = stage1_eicab.submit_subject_sge(
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
                        post_process_eicab=post_process_eicab,
                        hold_jid=prev_jid,
                        backend=backend,
                        dry_run=False,
                        emit=fh,
                        only_pp=only_pp,
                    )
                    prev_jid = jid or prev_jid
                except (FileNotFoundError, OSError) as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage1 eICAB emit skipped: {exc}")

            if run_s2:
                try:
                    prev_jid = stage2_registration.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        reference=stage2_reference,  # type: ignore[arg-type]
                        dof=stage2_dof,
                        cost=stage2_cost,
                        hold_jid=prev_jid,
                        backend=backend,
                        emit=fh,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage2 emit skipped: {exc}")

            if run_s3:
                try:
                    prev_jid = stage3_centerline.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        hold_jid=prev_jid,
                        emit=fh,
                        eicab_mask=eicab_mask,
                        cd_up_thresh=cd_up_thresh,
                        cd_shift_hm=cd_shift_hm,
                        venous_min_component_frac=venous_min_component_frac,
                        eicab_min_island_fraction=eicab_min_island_fraction,
                        eicab_bridge_open_radius=eicab_bridge_open_radius,
                        venous_min_branch_points=venous_min_branch_points,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage3 emit skipped: {exc}")

            if run_s4:
                try:
                    prev_jid = stage4_4dflow_segmentation.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        hold_jid=prev_jid,
                        emit=fh,
                        crop_padding_bbox=crop_padding_bbox,
                        thr_algorithm=thr_algorithm_4dflow,
                        region_growing=region_growing,
                        rg_intensity_frac=rg_intensity_frac,
                        rg_intensity_frac_explore=rg_intensity_frac_explore,
                        cl_barrier_radius=cl_barrier_radius,
                        rg_barrier_radius=rg_barrier_radius,
                        aca_sequential_grow=aca_sequential_grow,
                        aca_overlap_min_voxels=aca_overlap_min_voxels,
                        acomm_junction_radius=acomm_junction_radius,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage4 emit skipped: {exc}")

            if run_s4t:
                try:
                    prev_jid = stage4t_4dflow_t_segmentation.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        hold_jid=prev_jid,
                        emit=fh,
                        crop_padding_bbox=crop_padding_bbox,
                        thr_algorithm=thr_algorithm_4dflow,
                        region_growing=region_growing,
                        rg_intensity_frac=rg_intensity_frac,
                        rg_intensity_frac_explore=rg_intensity_frac_explore,
                        cl_barrier_radius=cl_barrier_radius,
                        rg_barrier_radius=rg_barrier_radius,
                        aca_sequential_grow=aca_sequential_grow,
                        aca_overlap_min_voxels=aca_overlap_min_voxels,
                        acomm_junction_radius=acomm_junction_radius,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage4t emit skipped: {exc}")

            if run_s5:
                try:
                    prev_jid = stage5_loc_generation.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        hold_jid=prev_jid,
                        emit=fh,
                        loc_arterial_strategy=loc_arterial_strategy,
                        cross_section_radius_vox=cross_section_radius_vox,
                        venous_min_component_frac=venous_min_component_frac,
                        loc_endpoint_inset_frac=loc_endpoint_inset_frac,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage5 emit skipped: {exc}")

            if run_s6:
                try:
                    stage6_measure.submit_subject_sge(
                        subj,
                        nifti_root=nifti_root,
                        output_root=output_root,
                        container=container,
                        src_dir=src_p,
                        skip_existing=skip_existing,
                        hold_jid=prev_jid,
                        emit=fh,
                        cross_section_radius_vox=cross_section_radius_vox,
                        measure_resegment=measure_resegment,
                        measure_thr_algorithm=measure_thr_algorithm,
                        cross_section_res=cross_section_res,
                        cross_section_plane_interp=cross_section_plane_interp,
                        backend=backend,
                    )
                except Exception as exc:
                    import traceback
                    log.exception(traceback.format_exc())
                    log.exception(f"[{subj}] stage6 emit skipped: {exc}")

    log.info("=" * 78)
    log.info(f"qvtpy SGE script written: {script_path}")
    log.info(f"On the cluster login node: bash {script_path}")
    log.info("=" * 78)

    if run_conv and report:
        log.info("NIfTI report skipped: conversion is queued on the cluster.")

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


__all__ = [
    "main",
    "DEFAULT_STAGES",
    "STAGE_DOWNLOAD",
    "STAGE_CONVERT",
    "STAGE_EICAB",
    "STAGE_REG",
    "STAGE_CENTERLINE",
    "STAGE_SEG",
    "STAGE_SEG_T",
    "STAGE_LOC",
    "STAGE_MEASURE",
]


if __name__ == "__main__":
    main()
