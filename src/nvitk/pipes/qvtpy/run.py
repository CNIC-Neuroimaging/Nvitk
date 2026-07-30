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
- ``stage7`` — TOF eICAB morphometrics (centerline caliber / tortuosity / stenosis).
- ``stage8_xnat_upload`` — upload ``eicab/`` and ``qvtpy/`` results to XNAT session resources (local or cluster via SSH fetch).

SGE: per subject, one ``qsub -t`` array job (task = pending stage) with
``-tc 1`` and done-markers between tasks (see
:func:`nvitk.cluster.sge.emit_array_job_block`).
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.remote_submit import run_sge_script_ssh_capture
from nvitk.cluster.sge_chunk import (
    drip_submit_subjects,
    parse_sge_submission_job_ids,
    wait_for_sge_job_ids,
    warn_if_chunk_exceeds_sge_limit,
)
from nvitk.cluster.sge_remote import publish_sge_driver_script, resolve_sge_script_paths
from nvitk.cluster.sge import (
    ArrayTaskSpec,
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    build_singularity_command,
    emit_array_job_block,
    python_module_argv,
    submit_stage,
    write_script_header,
)
from nvitk.core.click_backend import backend_click_option, sge_backend_env
from nvitk.measure.hemodynamics import QUALITY_THRESH_DEFAULT
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_qvtpy_array_resources,
    sge_qvtpy_stage_resources,
    sge_stage_use_nv,
)
from nvitk.pipes.qvtpy.util.io.sge_chunk import (
    count_sge_stages_for_subject,
    count_sge_stages_per_subject,
    filter_subjects_pending_work,
    pending_sge_stage_ids,
    stage_runs_from_emit_kwargs,
)
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.segmentation.eicab import config as eicab_cfg

from . import config as cfg
from . import stage0_download
from .stages import (
    DEFAULT_STAGES,
    STAGE_CENTERLINE,
    STAGE_CONVERT,
    STAGE_DOWNLOAD,
    STAGE_EICAB,
    STAGE_LABELS as _STAGE_LABELS,
    STAGE_LOC,
    STAGE_MEASURE,
    STAGE_MORPHOMETRICS,
    STAGE_REG,
    STAGE_SEG,
    STAGE_SEG_T,
    STAGE_XNAT_UPLOAD,
    parse_stages as _parse_stages,
)


log = Logger()
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _iter_subjects(root: Path) -> list[str]:
    """Sorted subject folder names under *root* (one directory per subject)."""
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _xnat_convert_subject(
    subject: str,
    *,
    xnat_config,
    sequences: set[str],
    dicom_root: Path,
    nifti_root: Path,
    save_dicoms: bool,
    convert_kwargs: dict,
) -> Path:
    """Download one subject's DICOMs from XNAT then convert to NIfTI.

    When *save_dicoms* is true the DICOMs are persisted under *dicom_root*;
    otherwise they are downloaded into a temporary directory that is removed
    once conversion finishes (NIfTI outputs only).
    """
    import tempfile

    # Download-only flags must not be forwarded to stage0_convert.run_subject.
    conv_kw = {
        k: v
        for k, v in convert_kwargs.items()
        if k != "skip_existing_downloads"
    }
    skip_existing_downloads = bool(
        convert_kwargs.get("skip_existing_downloads", False)
    )

    from . import stage0_convert

    if save_dicoms:
        stage0_download.run_download(
            [subject],
            dicom_root=dicom_root,
            xnat_config=xnat_config,
            sequences=sequences,
            skip_existing=conv_kw.get("skip_existing", False),
            skip_existing_downloads=skip_existing_downloads,
        )
        return stage0_convert.run_subject(
            subject,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            **conv_kw,
        )

    with tempfile.TemporaryDirectory(prefix=f"qvtpy_dicom_{subject}_") as tmp:
        tmp_root = Path(tmp)
        stage0_download.run_download(
            [subject],
            dicom_root=tmp_root,
            xnat_config=xnat_config,
            sequences=sequences,
            skip_existing=False,
        )
        return stage0_convert.run_subject(
            subject,
            dicom_root=tmp_root,
            nifti_root=nifti_root,
            **conv_kw,
        )


def _default_nvitk_src_dir() -> Path:
    """Host tree mounted at ``/nvitk/src/`` in SGE Singularity jobs."""
    return cfg.NVITK_SRC_DIR


# ---------------------------------------------------------------------------
# SGE script emission (per-stage submit_stage)
# ---------------------------------------------------------------------------


def _build_stage0_convert_command(
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
    """Host shell command for stage0_c (Singularity + python module)."""
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
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_backend_env(binds.src, backend),
    )
    return build_singularity_command(spec, paths)


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
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_backend_env(binds.src, backend),
    )
    return submit_stage(spec, paths, emit=fh)


def _emit_qvtpy_sge_subjects_for_chunk(
    fh: TextIO,
    chunk_subjects: list[str],
    *,
    run_conv: bool,
    run_eicab: bool,
    run_s2: bool,
    run_s3: bool,
    run_s4: bool,
    run_s4t: bool,
    run_s5: bool,
    run_s6: bool,
    run_s7: bool,
    dicom_root_eff: Path,
    nifti_root_eff: Path,
    output_root_eff: Path,
    container: Path,
    src_p: Path,
    backend: str,
    skip_existing: bool,
    compute_phase_derived: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    eicab_resolution: float,
    eicab_device: str,
    eicab_thread_limit: int | None,
    eicab_sge_pe_smp: int | None,
    eicab_local_metric_scratch: bool | None,
    eicab_container: Path | None,
    vasculature_dir: Path | None,
    post_process_eicab: bool,
    only_pp: bool,
    stage2_reference: str,
    stage2_dof: int,
    stage2_cost: str,
    eicab_mask: str,
    eicab_prefer_pp: bool,
    cd_up_thresh: float | None,
    cd_shift_hm: bool | None,
    venous_min_component_frac: float,
    eicab_min_island_fraction: float,
    eicab_bridge_open_radius: int,
    venous_min_branch_points: int,
    pcomm_min_points: int,
    arterial_branch_min_points: int,
    venous_brain_mask: bool,
    totalseg_model_dir: Path | None,
    crop_padding_bbox: int,
    thr_algorithm_4dflow: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    rg_intensity_frac_aca: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    distal_flow_expand: bool,
    distal_hyst_low_factor: float,
    distal_hyst_high_factor: float,
    distal_thicken_iter: int,
    distal_max_image_frac: float,
    distal_lr_halfspace_slack: int,
    loc_arterial_strategy: str,
    cross_section_radius_vox: float,
    loc_endpoint_inset_frac: float,
    distal_locs: bool,
    measure_resegment: bool,
    measure_thr_algorithm: str,
    cross_section_res: int,
    cross_section_plane_interp: int,
    cs_supersampling: bool,
    save_plots: bool,
    skip_processed: bool = False,
    pitc_stride: int = 1,
    pitc_quality_thresh: float = QUALITY_THRESH_DEFAULT,
    pitc_quality_metric: str = "stdv_from_mean",
    pitc_measure_resegment: bool = False,
    pitc_label_constrain: bool = True,
) -> int:
    """Append one SGE array ``qsub`` per subject; return array jobs emitted."""
    from . import (
        stage1_eicab,
        stage2_registration,
        stage3_centerline,
        stage4_4dflow_segmentation,
        stage4t_4dflow_t_segmentation,
        stage5_loc_generation,
        stage6_measure,
        stage7_morphometrics,
    )

    stage_runs = {
        "run_conv": run_conv,
        "run_eicab": run_eicab,
        "run_s2": run_s2,
        "run_s3": run_s3,
        "run_s4": run_s4,
        "run_s4t": run_s4t,
        "run_s5": run_s5,
        "run_s6": run_s6,
        "run_s7": run_s7,
    }
    jobs_emitted = 0
    for subj in chunk_subjects:
        if skip_processed:
            pending_list = pending_sge_stage_ids(
                subj,
                stage_runs=stage_runs,
                skip_processed=True,
                results_root=output_root_eff,
                nifti_root=nifti_root_eff,
            )
            if not pending_list:
                log.info(f"[{subj}] skip-processed: all requested stages complete")
                continue
            pending = set(pending_list)
        else:
            pending = None

        def _should_emit(stage_id: str) -> bool:
            return pending is None or stage_id in pending

        tasks: list[ArrayTaskSpec] = []

        if run_conv and _should_emit(STAGE_CONVERT):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_CONVERT,
                        _build_stage0_convert_command(
                            subj,
                            dicom_root=dicom_root_eff,
                            nifti_root=nifti_root_eff,
                            container=container,
                            src_dir=src_p,
                            compute_phase_derived=compute_phase_derived,
                            skip_existing=skip_existing,
                            phase_background_correction=phase_background_correction,
                            phase_bg_poly_order=phase_bg_poly_order,
                            phase_bg_static_percentile=phase_bg_static_percentile,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage0_c emit skipped: {exc}")
        elif run_conv and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_CONVERT} (outputs present)")

        if run_eicab and _should_emit(STAGE_EICAB):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_EICAB,
                        stage1_eicab.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            resolution=eicab_resolution,
                            device=eicab_device,
                            eicab_container=eicab_container,
                            pipeline_container=container,
                            src_dir=src_p,
                            sge_pe_smp=eicab_sge_pe_smp,
                            thread_limit=eicab_thread_limit,
                            local_metric_scratch=eicab_local_metric_scratch,
                            vasculature_dir=vasculature_dir,
                            post_process_eicab=post_process_eicab,
                            backend=backend,
                            only_pp=only_pp,
                        ),
                    )
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage1 eICAB emit skipped: {exc}")
        elif run_eicab and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_EICAB} (outputs present)")

        if run_s2 and _should_emit(STAGE_REG):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_REG,
                        stage2_registration.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            reference=stage2_reference,
                            dof=stage2_dof,
                            cost=stage2_cost,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage2 emit skipped: {exc}")
        elif run_s2 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_REG} (outputs present)")

        if run_s3 and _should_emit(STAGE_CENTERLINE):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_CENTERLINE,
                        stage3_centerline.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            eicab_mask=eicab_mask,
                            eicab_prefer_pp=eicab_prefer_pp,
                            cd_up_thresh=cd_up_thresh,
                            cd_shift_hm=cd_shift_hm,
                            venous_min_component_frac=venous_min_component_frac,
                            eicab_min_island_fraction=eicab_min_island_fraction,
                            eicab_bridge_open_radius=eicab_bridge_open_radius,
                            venous_min_branch_points=venous_min_branch_points,
                            pcomm_min_points=pcomm_min_points,
                            arterial_branch_min_points=arterial_branch_min_points,
                            venous_brain_mask=venous_brain_mask,
                            totalseg_model_dir=totalseg_model_dir,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage3 emit skipped: {exc}")
        elif run_s3 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_CENTERLINE} (outputs present)")

        if run_s4 and _should_emit(STAGE_SEG):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_SEG,
                        stage4_4dflow_segmentation.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            crop_padding_bbox=crop_padding_bbox,
                            thr_algorithm=thr_algorithm_4dflow,
                            region_growing=region_growing,
                            rg_intensity_frac=rg_intensity_frac,
                            rg_intensity_frac_explore=rg_intensity_frac_explore,
                            rg_intensity_frac_aca=rg_intensity_frac_aca,
                            cl_barrier_radius=cl_barrier_radius,
                            rg_barrier_radius=rg_barrier_radius,
                            aca_sequential_grow=aca_sequential_grow,
                            aca_overlap_min_voxels=aca_overlap_min_voxels,
                            acomm_junction_radius=acomm_junction_radius,
                            distal_flow_expand=distal_flow_expand,
                            distal_hyst_low_factor=distal_hyst_low_factor,
                            distal_hyst_high_factor=distal_hyst_high_factor,
                            distal_thicken_iter=distal_thicken_iter,
                            distal_max_image_frac=distal_max_image_frac,
                            distal_lr_halfspace_slack=distal_lr_halfspace_slack,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage4 emit skipped: {exc}")
        elif run_s4 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_SEG} (outputs present)")

        if run_s4t and _should_emit(STAGE_SEG_T):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_SEG_T,
                        stage4t_4dflow_t_segmentation.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
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
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage4t emit skipped: {exc}")
        elif run_s4t and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_SEG_T} (outputs present)")

        if run_s5 and _should_emit(STAGE_LOC):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_LOC,
                        stage5_loc_generation.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            loc_arterial_strategy=loc_arterial_strategy,
                            cross_section_radius_vox=cross_section_radius_vox,
                            venous_min_component_frac=venous_min_component_frac,
                            loc_endpoint_inset_frac=loc_endpoint_inset_frac,
                            distal_locs=distal_locs,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage5 emit skipped: {exc}")
        elif run_s5 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_LOC} (outputs present)")

        if run_s6 and _should_emit(STAGE_MEASURE):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_MEASURE,
                        stage6_measure.build_subject_sge_command(
                            subj,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            cross_section_radius_vox=cross_section_radius_vox,
                            measure_resegment=measure_resegment,
                            measure_thr_algorithm=measure_thr_algorithm,
                            cross_section_res=cross_section_res,
                            cross_section_plane_interp=cross_section_plane_interp,
                            cs_supersampling=cs_supersampling,
                            pitc_stride=pitc_stride,
                            pitc_quality_thresh=pitc_quality_thresh,
                            pitc_quality_metric=pitc_quality_metric,
                            pitc_measure_resegment=pitc_measure_resegment,
                            pitc_label_constrain=pitc_label_constrain,
                            save_plots=save_plots,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage6 emit skipped: {exc}")
        elif run_s6 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_MEASURE} (outputs present)")

        if run_s7 and _should_emit(STAGE_MORPHOMETRICS):
            try:
                tasks.append(
                    ArrayTaskSpec(
                        STAGE_MORPHOMETRICS,
                        stage7_morphometrics.build_subject_sge_command(
                            subj,
                            output_root=output_root_eff,
                            container=container,
                            src_dir=src_p,
                            skip_existing=skip_existing,
                            backend=backend,
                        ),
                    )
                )
            except Exception as exc:
                import traceback

                log.exception(traceback.format_exc())
                log.exception(f"[{subj}] stage7 emit skipped: {exc}")
        elif run_s7 and skip_processed:
            log.info(f"[{subj}] skip-processed: {STAGE_MORPHOMETRICS} (outputs present)")

        if not tasks:
            continue

        include_eicab = any(t.stage_id == STAGE_EICAB for t in tasks)
        resources, use_nv = sge_qvtpy_array_resources(
            backend,
            include_eicab=include_eicab,
            eicab_device=eicab_device,
            eicab_pe_smp=eicab_sge_pe_smp,
        )
        subj_token = _sge_script_subject_token(subj)
        job_name = f"{cfg.SGE_JOB_PREFIX}_{subj_token}"[:63]
        marker_dir = output_root_eff / subj / cfg.QVT_SUBDIR / ".sge_array_markers"
        paths = ClusterPaths(
            src=src_p,
            container=container,
            models=None,
            data_root=nifti_root_eff,
            output_root=output_root_eff,
            log_dir=cfg.SGE_LOG_DIR,
            err_dir=cfg.SGE_ERR_DIR,
        )
        stage_ids = ",".join(t.stage_id for t in tasks)
        log.info(
            f"[{subj}] SGE array: {len(tasks)} task(s) [{stage_ids}] "
            f"job_name={job_name}"
        )
        try:
            emit_array_job_block(
                fh,
                job_name=job_name,
                resources=resources,
                paths=paths,
                tasks=tasks,
                marker_dir=marker_dir,
                task_concurrency=1,
                use_nv=use_nv,
            )
            jobs_emitted += 1
        except Exception as exc:
            import traceback

            log.exception(traceback.format_exc())
            log.exception(f"[{subj}] array emit skipped: {exc}")

    return jobs_emitted



_SGE_SCRIPT_SUBJECT_TOKEN = re.compile(r"[^\w.-]+")


def _sge_script_subject_token(subj: str) -> str:
    tok = _SGE_SCRIPT_SUBJECT_TOKEN.sub("_", subj).strip("._-")
    return tok or "subject"


def _local_stage_pending(
    subj: str,
    stage_id: str,
    *,
    skip_processed: bool,
    output_root: Path,
    nifti_root: Path,
) -> bool:
    if not skip_processed:
        return True
    from nvitk.pipes.qvtpy.util.io.qc_report import check_subject_stages

    checks = check_subject_stages(
        subj,
        [stage_id],
        results_root=output_root,
        nifti_root=nifti_root,
    )
    if checks and checks[0].complete:
        log.info(f"[{subj}] skip-processed: {stage_id} (outputs present)")
        return False
    return True


def _submit_qvtpy_sge_subjects_remote(
    subjects: list[str],
    *,
    script_basename: str,
    title: str,
    emit_kwargs: dict,
    ssh_host: str,
    ssh_user: str,
    ssh_password: str,
) -> tuple[int, list[str]]:
    """Write, upload, and run an SGE driver for *subjects*; return ``(exit_code, job_ids)``."""
    local_script_path, remote_script_path = resolve_sge_script_paths(
        None,
        remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
        default_basename=script_basename,
    )
    with open(local_script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=cfg.SGE_LOG_DIR,
            err_dir=cfg.SGE_ERR_DIR,
            title=title,
        )
        _emit_qvtpy_sge_subjects_for_chunk(fh, subjects, **emit_kwargs)

    log.info(f"  local script : {local_script_path}")
    log.info(f"  cluster path : {remote_script_path}")

    cluster_exec_path = publish_sge_driver_script(
        local_script_path,
        remote_script_path,
        host=ssh_host,
        user=ssh_user,
        password=ssh_password,
    )
    exit_code, stdout, stderr = run_sge_script_ssh_capture(
        ssh_host,
        ssh_user,
        ssh_password,
        cluster_exec_path,
        local_script_path=local_script_path,
    )
    job_ids = parse_sge_submission_job_ids(stdout, stderr)
    return exit_code, job_ids


# ---------------------------------------------------------------------------
# Master CLI (local + SGE)
# ---------------------------------------------------------------------------


@click.command("nvitk-qvtpy")
@backend_click_option()
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override DICOM root (default: local or cluster layout from config).",
)
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override NIfTI root (default: local or cluster layout from config).",
)
@click.option(
    "--stages",
    "stages_spec",
    default=DEFAULT_STAGES,
    show_default=True,
    help="Comma-separated stages (see module docstring for names and aliases).",
)
@click.option(
    "--subjects",
    default=None,
    help=(
        "Comma-separated subject list, or a cohort alias with --from-source xnat "
        "(e.g. PESA-Brain runs all subjects in the XNAT PESA_Brain project)."
    ),
)
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Text/CSV/XLSX file with subject IDs.",
)
@click.option(
    "--subjects-2-exclude",
    "subjects_2_exclude",
    default=None,
    help=(
        "Comma/whitespace-separated subject IDs to drop from the resolved run list "
        "(after --subjects / --subjects-file / cohort expansion). Applies to local and SGE."
    ),
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
@click.option("--skip-existing", is_flag=True, default=False, help=(
    "Skip stages whose outputs already exist on disk. On SGE, subjects/stages with "
    "complete outputs are not submitted (same completion checks as the QC report)."
))
@click.option(
    "--skip-processed",
    is_flag=True,
    default=False,
    help=(
        "Skip stages whose completion markers already exist on disk "
        "(per subject, per stage; uses the same checks as the QC report)."
    ),
)
@click.option(
    "--skip-existing-downloads",
    is_flag=True,
    default=False,
    help=(
        "Skip XNAT download when all requested sequences already exist under the "
        "local subject DICOM tree; fetch only missing sequences otherwise."
    ),
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Results root (default: local or cluster layout from config).",
)
@click.option("--emit-script", type=click.Path(path_type=Path), default=None)
@click.option("--no-remote", is_flag=True)
@click.option("--remote-host", default=None)
@click.option("--remote-user", default=None)
@click.option(
    "--sge-subject-chunk-size",
    type=int,
    default=900,
    show_default=True,
    help=(
        "(sge) Subjects in the first submission batch (must fit ~1000 jobs/user). "
        "Remaining subjects are drip-fed when queue capacity allows. "
        "Value 1 runs one subject at a time (waits for each job to finish)."
    ),
)
@click.option(
    "--sge-serial-subjects",
    is_flag=True,
    help=(
        "(sge) Wait for each subject's jobs to finish before submitting the next "
        "(default when --sge-subject-chunk-size=1)."
    ),
)
@click.option(
    "--sge-chunk-poll-interval",
    type=float,
    default=120.0,
    show_default=True,
    help="(sge) Seconds between SSH qstat polls while drip-feeding subjects.",
)
@click.option(
    "--sge-chunk-wait-timeout",
    type=float,
    default=None,
    help="(sge) Max seconds for the drip-feed loop (default: unlimited).",
)
@click.option(
    "--sge-job-margin",
    type=int,
    default=10,
    show_default=True,
    help="(sge) Reserve this many job slots below the per-user cap when drip-feeding.",
)
# --- stage 0 ---
@click.option(
    "--from-source",
    type=click.Choice(["local", "xnat"], case_sensitive=False),
    default="local",
    show_default=True,
    help=(
        "DICOM source for stage0_c. 'xnat' downloads from the XNAT PESA-Brain "
        "project before conversion. (Distinct from pesa_fat's --input-source.)"
    ),
)
@click.option(
    "--save-dicoms/--no-save-dicoms",
    default=False,
    show_default=True,
    help=(
        "With --from-source xnat: persist DICOMs under --dicom-root when true; "
        "when false download to a temp dir, convert, then delete (NIfTI only)."
    ),
)
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
@click.option(
    "--database",
    "database_root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Dataset catalog root with indexed XNAT scans (enables pre-filter: subject "
        "must have TOF + 4DFLOW_AP/RL/FH in the scans table). Omit to download "
        "requested sequences for all resolved subjects."
    ),
)
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
    "--eicab-thread-limit",
    type=int,
    default=None,
    help=(
        "(sge, stage1) Cap OMP/BLAS threads inside the eICAB container. "
        "Default: pipelines.eicab.eicab_thread_limit in .nvitk/sge.json."
    ),
)
@click.option(
    "--eicab-sge-pe-smp",
    type=int,
    default=None,
    help=(
        "(sge, stage1) Optional qsub -pe smp N (cluster-specific). "
        "Leave unset unless your queue supports the smp parallel environment."
    ),
)
@click.option(
    "--eicab-local-metric-scratch/--no-eicab-local-metric-scratch",
    default=None,
    help=(
        "(sge, stage1) Write VED multiscale NIfTIs to node-local /data_tmp "
        "(bind over /output/metric_space). Default from .nvitk/sge.json."
    ),
)
@click.option(
    "--post-process-eicab/--no-post-process-eicab",
    default=False,
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
@click.option(
    "--eicab-prefer-pp/--no-eicab-prefer-pp",
    default=True,
    show_default=True,
    help="Stage3: use stage1 *_pp eICAB mask when available.",
)
@click.option("--cd-up-thresh", type=float, default=None, help="Stage3: CD sliding-threshold upper fraction.")
@click.option(
    "--cd-shift-hm/--no-cd-shift-hm",
    default=None,
    help="Stage3: FWHM shift on threshold curve (default on).",
)
@click.option("--venous-min-component-frac", type=float, default=0.005, show_default=True)
@click.option("--eicab-min-island-fraction", type=float, default=0.005, show_default=True)
@click.option("--eicab-bridge-open-radius", type=int, default=0, show_default=True)
@click.option("--venous-min-branch-points", type=int, default=30, show_default=True)
@click.option(
    "--pcomm-min-points",
    type=int,
    default=5,
    show_default=True,
    help="Stage3: drop LPCOMM/RPCOMM centerlines shorter than this (filters tiny FPs).",
)
@click.option(
    "--arterial-branch-min-points",
    type=int,
    default=10,
    show_default=True,
    help="Stage3: min points to keep MCA/ACA/PCA bifurcation side branches.",
)
@click.option(
    "--venous-brain-mask/--no-venous-brain-mask",
    default=True,
    show_default=True,
    help="Stage3: restrict venous candidates to TotalSegmentator brain on Angiography_3D.",
)
@click.option(
    "--totalseg-model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Stage3: TotalSegmentator weights (default: qvtpy config model_root).",
)
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
@click.option("--rg-intensity-frac-explore", type=float, default=0.25, show_default=True, help="Stage4: RG frac for MCA/PCA.")
@click.option("--rg-intensity-frac-aca", type=float, default=0.35, show_default=True, help="Stage4: RG frac for ACA.")
@click.option("--cl-barrier-radius", type=int, default=5, show_default=True, help="Stage4: dilate other centerlines (vox).")
@click.option("--rg-barrier-radius", type=int, default=3, show_default=True, help="Stage4: dilate other seg during RG (vox).")
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
@click.option(
    "--distal-flow-expand/--no-distal-flow-expand",
    default=True,
    show_default=True,
    help=(
        "Stage4: after region growing, expand MCA/ACA/PCA into a Frangi+hysteresis "
        "vessel tree via watershed (eICAB-inspired, Python-only). Default ON."
    ),
)
@click.option(
    "--distal-hyst-low-factor",
    type=float,
    default=3,
    show_default=True,
    help="Stage4: distal GMM hysteresis low factor (higher → thinner tree; 3.5–4.5).",
)
@click.option(
    "--distal-hyst-high-factor",
    type=float,
    default=0.5,
    show_default=True,
    help="Stage4: distal GMM hysteresis high factor.",
)
@click.option(
    "--distal-thicken-iter",
    type=int,
    default=0,
    show_default=True,
    help="Stage4: lumen thicken iterations (0=thinnest; 1 can look blobbier).",
)
@click.option(
    "--distal-max-image-frac",
    type=float,
    default=0.08,
    show_default=True,
    help="Stage4: max voxels claimed by distal expand as fraction of image.",
)
@click.option(
    "--distal-lr-halfspace-slack",
    type=int,
    default=2,
    show_default=True,
    help="Stage4: L/R midline slack so contralateral ACA/MCA/PCA claims are blocked.",
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
@click.option(
    "--distal-locs/--no-distal-locs",
    default=False,
    show_default=True,
    help=(
        "Stage5: for MCA/ACA/PCA also emit distal (fin) LOCs past the stage-3 "
        "A1/M1/P1 span. Default off: one init LOC per vessel."
    ),
)
# --- stage 6 ---
@click.option(
    "--measure-resegment/--no-measure-resegment",
    default=False,
    show_default=True,
    help="Stage6 LOC: in-plane resegmentation (default off; upsample stage-4 masks).",
)
@click.option(
    "--measure-thr-algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="otsu",
    show_default=True,
    help="Stage6: in-plane threshold when --measure-resegment.",
)
@click.option("--cross-section-res", type=int, default=0, show_default=True)
@click.option("--cross-section-plane-interp", type=int, default=1, show_default=True)
@click.option(
    "--cs-supersampling/--no-cs-supersampling",
    default=True,
    show_default=True,
    help="Stage6: supersample oblique cross-section grid (~4×) for intensity + masks.",
)
@click.option(
    "--save-plots/--no-save-plots",
    default=True,
    show_default=True,
    help="Stage6: render paper-style PITC/PWV/flow figures + per-region PITC branch masks.",
)
@click.option("--pitc-stride", type=int, default=1, show_default=True)
@click.option(
    "--pitc-quality-thresh",
    type=float,
    default=QUALITY_THRESH_DEFAULT,
    show_default=True,
)
@click.option(
    "--pitc-quality-metric",
    type=click.Choice(["stdv_from_mean", "waveform"], case_sensitive=False),
    default="stdv_from_mean",
    show_default=True,
)
@click.option(
    "--pitc-measure-resegment/--no-pitc-measure-resegment",
    default=False,
    show_default=True,
    help="Stage6 PITC/PWV: in-plane resegmentation (default off).",
)
@click.option(
    "--pitc-label-constrain/--no-pitc-label-constrain",
    default=True,
    show_default=True,
)
# --- stage 8 (XNAT upload) ---
@click.option(
    "--xnat-upload-require-stages",
    default="stage2,stage3,stage4,stage5,stage6,stage7",
    show_default=True,
    help="(stage8) QVTpy stages required before uploading the qvtpy resource.",
)
@click.option("--xnat-upload-eicab/--no-xnat-upload-eicab", default=True, show_default=True)
@click.option("--xnat-upload-qvtpy/--no-xnat-upload-qvtpy", default=True, show_default=True)
@click.option(
    "--xnat-upload-skip-existing/--xnat-upload-overwrite-existing",
    default=True,
    show_default=True,
    help="(stage8) Skip or overwrite existing XNAT session resources.",
)
@click.option("--xnat-upload-dry-run", is_flag=True, default=False)
def main(
    dicom_root: Path | None,
    nifti_root: Path | None,
    stages_spec: str,
    subjects: str | None,
    subjects_file: Path | None,
    subjects_2_exclude: str | None,
    submit: str,
    container: Path,
    src_dir: Path | None,
    skip_existing: bool,
    skip_processed: bool,
    skip_existing_downloads: bool,
    output_root: Path | None,
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    from_source: str,
    save_dicoms: bool,
    compute_phase_derived: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    sequences: str,
    xnat_config_path: Path | None,
    database_root: Path | None,
    report: bool,
    report_derived: bool,
    eicab_container: Path | None,
    vasculature_dir: Path | None,
    eicab_device: str,
    eicab_resolution: float,
    eicab_thread_limit: int | None,
    eicab_sge_pe_smp: int | None,
    eicab_local_metric_scratch: bool | None,
    post_process_eicab: bool,
    only_pp: bool,
    stage2_reference: str,
    stage2_dof: int,
    stage2_cost: str,
    eicab_mask: str,
    eicab_prefer_pp: bool,
    cd_up_thresh: float | None,
    cd_shift_hm: bool | None,
    venous_min_component_frac: float,
    eicab_min_island_fraction: float,
    eicab_bridge_open_radius: int,
    venous_min_branch_points: int,
    pcomm_min_points: int,
    arterial_branch_min_points: int,
    venous_brain_mask: bool,
    totalseg_model_dir: Path | None,
    crop_padding_bbox: int,
    thr_algorithm_4dflow: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    rg_intensity_frac_aca: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    distal_flow_expand: bool,
    distal_hyst_low_factor: float,
    distal_hyst_high_factor: float,
    distal_thicken_iter: int,
    distal_max_image_frac: float,
    distal_lr_halfspace_slack: int,
    loc_arterial_strategy: str,
    cross_section_radius_vox: float,
    loc_endpoint_inset_frac: float,
    distal_locs: bool,
    measure_resegment: bool,
    measure_thr_algorithm: str,
    cross_section_res: int,
    cross_section_plane_interp: int,
    cs_supersampling: bool,
    save_plots: bool,
    backend: str,
    pitc_stride: int,
    pitc_quality_thresh: float,
    pitc_quality_metric: str,
    pitc_measure_resegment: bool,
    pitc_label_constrain: bool,
    sge_subject_chunk_size: int,
    sge_serial_subjects: bool,
    sge_chunk_poll_interval: float,
    sge_chunk_wait_timeout: float | None,
    sge_job_margin: int,
    xnat_upload_require_stages: str,
    xnat_upload_eicab: bool,
    xnat_upload_qvtpy: bool,
    xnat_upload_skip_existing: bool,
    xnat_upload_dry_run: bool,
) -> None:
    Logger()

    from nvitk.pipes.qvtpy.util.io.paths import layout_cluster, layout_local, resolve_totalseg_model_dir

    local_paths = layout_local(
        dicom_root=cfg.LOCAL_DEFAULT_DICOM_ROOT if dicom_root is None else dicom_root,
        nifti_root=cfg.LOCAL_DEFAULT_NIFTI_ROOT if nifti_root is None else nifti_root,
        results_root=cfg.LOCAL_DEFAULT_RESULTS_ROOT if output_root is None else output_root,
    )
    cluster_paths = layout_cluster(
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=cfg.DEFAULT_RESULTS_ROOT if output_root is None else output_root,
    )
    if submit == "sge":
        dicom_root_eff = cluster_paths.dicom_root
        nifti_root_eff = cluster_paths.nifti_root
        output_root_eff = cluster_paths.results_root
        dicom_download_root = local_paths.dicom_root
    else:
        dicom_root_eff = local_paths.dicom_root
        nifti_root_eff = local_paths.nifti_root
        output_root_eff = local_paths.results_root
        dicom_download_root = local_paths.dicom_root

    log.info(f"Layout | submit={submit}")
    log.info(f"  local   dicom={local_paths.dicom_root} nifti={local_paths.nifti_root}")
    log.info(f"  cluster dicom={cluster_paths.dicom_root} nifti={cluster_paths.nifti_root}")
    log.info(f"  active  dicom={dicom_root_eff} nifti={nifti_root_eff} results={output_root_eff}")

    totalseg_model_dir_eff = (
        Path(totalseg_model_dir)
        if totalseg_model_dir is not None
        else resolve_totalseg_model_dir(prefer_cluster=(submit == "sge"))
    )
    log.info(f"  totalseg models={totalseg_model_dir_eff} ({'cluster' if submit == 'sge' else 'local'})")

    eicab_thread_eff = (
        eicab_thread_limit
        if eicab_thread_limit is not None
        else eicab_cfg.EICAB_THREAD_LIMIT
    )
    eicab_pe_smp_eff = eicab_sge_pe_smp
    eicab_metric_scratch_eff = (
        eicab_local_metric_scratch
        if eicab_local_metric_scratch is not None
        else eicab_cfg.EICAB_LOCAL_METRIC_SCRATCH
    )

    if submit == "sge" and from_source.lower() == "xnat":
        log.info(f"  XNAT download target (local): {dicom_download_root}")

    stages = _parse_stages(stages_spec)
    run_dl = STAGE_DOWNLOAD in stages
    run_conv = STAGE_CONVERT in stages
    run_eicab = STAGE_EICAB in stages
    if submit == "sge" and run_eicab:
        if eicab_thread_eff is not None:
            log.info(f"  eicab thread limit (container): {eicab_thread_eff}")
        if eicab_pe_smp_eff is not None:
            log.info(f"  eicab sge pe smp (qsub -pe): {eicab_pe_smp_eff}")
        if eicab_metric_scratch_eff:
            log.info(
                "  eicab metric scratch: "
                f"{eicab_cfg.EICAB_METRIC_SCRATCH_ROOT} (node-local VED scale NIfTIs)"
            )
    run_s2 = STAGE_REG in stages
    run_s3 = STAGE_CENTERLINE in stages
    run_s4 = STAGE_SEG in stages
    run_s4t = STAGE_SEG_T in stages
    run_s5 = STAGE_LOC in stages
    run_s6 = STAGE_MEASURE in stages
    run_s7 = STAGE_MORPHOMETRICS in stages
    run_s8 = STAGE_XNAT_UPLOAD in stages

    use_xnat = from_source.lower() == "xnat"
    # if use_xnat and run_conv and submit == "sge":
    #     # SGE stages read DICOMs from the cluster tree, so the download must
    #     # persist locally regardless of --save-dicoms; auto-insert stage0_d.
    #     run_dl = True
    #     if not save_dicoms:
    #         log.info(
    #             "--from-source xnat with --submit sge: per-subject XNAT download -> "
    #             f"cluster SFTP ({cluster_paths.dicom_root}), then delete local staging."
    #         )

    log.info(f"qvtpy | stages={','.join(stages)} | submit={submit}")

    if skip_existing and not skip_processed:
        log.info(
            "--skip-existing: checking completion markers per stage before scheduling "
            "(same checks as QC report)."
        )
    skip_processed = skip_processed or skip_existing

    xnat_conn = None
    if use_xnat or run_s8:
        from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

        profile = load_xnat_profile(xnat_config_path)
        xnat_conn = resolve_xnat_connection(profile)

    if subjects or subjects_file:
        from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token

        cohort_alias = (
            resolve_xnat_project_cohort_token(subjects)
            if subjects and not subjects_file
            else None
        )
        if use_xnat or (run_s8 and cohort_alias):
            if xnat_conn is None:
                raise click.ClickException("XNAT connection required for subject resolution.")
            subject_list, xnat_conn = stage0_download.resolve_subjects_for_xnat_pipeline(
                subjects=subjects,
                subjects_file=subjects_file,
                xnat_config=xnat_conn,
                database_root=database_root,
            )
        else:
            if subjects and not subjects_file and resolve_xnat_project_cohort_token(subjects):
                raise click.ClickException(
                    f"--subjects {subjects!r} is an XNAT cohort alias; "
                    "use --from-source xnat to expand it to all project subjects."
                )
            subject_list = stage0_download.load_subjects(
                subjects=subjects,
                subjects_file=subjects_file,
            )
    else:
        if run_dl:
            raise click.ClickException("stage0_d requires --subjects or --subjects-file.")
        fallback_root = dicom_root_eff if run_conv else nifti_root_eff
        subject_list = _iter_subjects(fallback_root)
        if not subject_list:
            raise click.ClickException(
                f"No subjects to process (looked under {fallback_root}). "
                "Pass --subjects / --subjects-file or populate that root."
            )

    if not subject_list:
        raise click.ClickException("No subjects resolved from inputs.")

    if subjects_2_exclude:
        from nvitk.db.xnat import parse_subject_tokens

        exclude = set(parse_subject_tokens(subjects_2_exclude))
        if exclude:
            before = len(subject_list)
            subject_list = [s for s in subject_list if s not in exclude]
            n_dropped = before - len(subject_list)
            log.info(
                f"--subjects-2-exclude: dropped {n_dropped}/{before} subject(s) "
                f"({len(subject_list)} remaining)"
            )
            if not subject_list:
                raise click.ClickException(
                    "No subjects left after --subjects-2-exclude."
                )

    ssh_host_resolved: str | None = None
    ssh_user: str | None = None
    ssh_password: str | None = None

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
                log.info("stage0_d streams locally to cluster; SGE covers other selected stages.")
            from nvitk.db.xnat import requested_sequence_set

            if xnat_conn is None:
                raise click.ClickException("stage0_d with --from-source xnat requires XNAT credentials.")
            seq_set = requested_sequence_set(sequences) or set(
                stage0_download.DEFAULT_SEQUENCES
            )

            if submit == "sge" and use_xnat and subject_list:
                from nvitk.pipes.qvtpy.util.io.cluster_upload import (
                    prompt_ssh_credentials,
                    stream_subjects_xnat_to_cluster,
                )

                ssh_host_resolved, ssh_user, ssh_password = prompt_ssh_credentials(
                    remote_host=remote_host,
                    remote_user=remote_user,
                    host_aliases=cfg.CLUSTER_HOST_ALIASES,
                )

                def _stream_xnat_to_cluster() -> None:
                    results = stream_subjects_xnat_to_cluster(
                        subject_list,
                        local_paths=local_paths,
                        cluster_paths=cluster_paths,
                        xnat_config=xnat_conn,
                        sequences=seq_set,
                        host=ssh_host_resolved,
                        user=ssh_user,
                        password=ssh_password,
                        skip_existing=skip_existing,
                        skip_existing_downloads=skip_existing_downloads,
                        delete_local_after_upload=True,
                    )
                    if report:
                        stage0_download.print_qc_report_from_results(
                            results, subject_list, seq_set
                        )

                run.run_stage(
                    "(cohort)",
                    STAGE_DOWNLOAD,
                    _stream_xnat_to_cluster,
                    detail=(
                        f"{len(subject_list)} subject(s) "
                        f"XNAT -> {cluster_paths.dicom_root} (local staging cleared)"
                    ),
                )
            else:
                def _download() -> None:
                    stage0_download.run_download(
                        subject_list,
                        dicom_root=dicom_download_root,
                        xnat_config=xnat_conn,
                        sequences=seq_set,
                        skip_existing=skip_existing,
                        skip_existing_downloads=skip_existing_downloads,
                        report=report,
                    )

                run.run_stage(
                    "(cohort)",
                    STAGE_DOWNLOAD,
                    _download,
                    detail=f"{len(subject_list)} subject(s)",
                )

        xnat_seq_set: set[str] = set()
        if use_xnat and run_conv and submit == "local":
            from nvitk.db.xnat import requested_sequence_set

            if xnat_conn is None:
                raise click.ClickException("XNAT connection required for --from-source xnat.")
            xnat_seq_set = requested_sequence_set(sequences) or set(
                stage0_download.DEFAULT_SEQUENCES
            )

        if submit == "local":
            from . import (
                stage0_convert,
                stage1_eicab,
                stage2_registration,
                stage3_centerline,
                stage4_4dflow_segmentation,
                stage4t_4dflow_t_segmentation,
                stage5_loc_generation,
                stage6_measure,
                stage7_morphometrics,
            )

            _local_skip = dict(
                skip_processed=skip_processed,
                output_root=output_root_eff,
                nifti_root=nifti_root_eff,
            )
            for subj in subject_list:
                if run_conv and _local_stage_pending(subj, STAGE_CONVERT, **_local_skip):
                    convert_kwargs = dict(
                        compute_phase_derived=compute_phase_derived,
                        skip_existing=skip_existing,
                        phase_background_correction=phase_background_correction,
                        phase_bg_poly_order=phase_bg_poly_order,
                        phase_bg_static_percentile=phase_bg_static_percentile,
                    )
                    if use_xnat:
                        convert_kwargs["skip_existing_downloads"] = (
                            skip_existing_downloads
                        )
                        run.run_stage(
                            subj,
                            STAGE_CONVERT,
                            lambda s=subj, kw=convert_kwargs: _xnat_convert_subject(
                                s,
                                xnat_config=xnat_conn,
                                sequences=xnat_seq_set,
                                dicom_root=dicom_root_eff,
                                nifti_root=nifti_root_eff,
                                save_dicoms=save_dicoms,
                                convert_kwargs=kw,
                            ),
                        )
                    else:
                        run.run_stage(
                            subj,
                            STAGE_CONVERT,
                            lambda s=subj, kw=convert_kwargs: stage0_convert.run_subject(
                                s,
                                dicom_root=dicom_root_eff,
                                nifti_root=nifti_root_eff,
                                **kw,
                            ),
                        )
                if run_eicab and _local_stage_pending(subj, STAGE_EICAB, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_EICAB,
                        lambda s=subj: stage1_eicab.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            resolution=eicab_resolution,
                            device=eicab_device,
                            eicab_container=eicab_container,
                            vasculature_dir=vasculature_dir,
                            post_process_eicab=post_process_eicab,
                            only_pp=only_pp,
                        ),
                    )
                if run_s2 and _local_stage_pending(subj, STAGE_REG, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_REG,
                        lambda s=subj: stage2_registration.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            reference=stage2_reference,
                            dof=stage2_dof,
                            cost=stage2_cost,
                        ),
                    )
                if run_s3 and _local_stage_pending(subj, STAGE_CENTERLINE, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_CENTERLINE,
                        lambda s=subj: stage3_centerline.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            eicab_mask=eicab_mask.lower(),
                            eicab_prefer_pp=eicab_prefer_pp,
                            cd_up_thresh=cd_up_thresh,
                            cd_shift_hm=cd_shift_hm,
                            venous_min_component_frac=venous_min_component_frac,
                            eicab_min_island_fraction=eicab_min_island_fraction,
                            eicab_bridge_open_radius=eicab_bridge_open_radius,
                            venous_min_branch_points=venous_min_branch_points,
                            pcomm_min_points=pcomm_min_points,
                            arterial_branch_min_points=arterial_branch_min_points,
                            venous_brain_mask=venous_brain_mask,
                            totalseg_model_dir=totalseg_model_dir_eff,
                            totalseg_device=backend,
                        ),
                    )
                if run_s4 and _local_stage_pending(subj, STAGE_SEG, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_SEG,
                        lambda s=subj: stage4_4dflow_segmentation.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            crop_padding_bbox=crop_padding_bbox,
                            thr_algorithm=thr_algorithm_4dflow.lower(),
                            region_growing=region_growing,
                            rg_intensity_frac=rg_intensity_frac,
                            rg_intensity_frac_explore=rg_intensity_frac_explore,
                            rg_intensity_frac_aca=rg_intensity_frac_aca,
                            cl_barrier_radius=cl_barrier_radius,
                            rg_barrier_radius=rg_barrier_radius,
                            aca_sequential_grow=aca_sequential_grow,
                            aca_overlap_min_voxels=aca_overlap_min_voxels,
                            acomm_junction_radius=acomm_junction_radius,
                            distal_flow_expand=distal_flow_expand,
                            distal_hyst_low_factor=distal_hyst_low_factor,
                            distal_hyst_high_factor=distal_hyst_high_factor,
                            distal_thicken_iter=distal_thicken_iter,
                            distal_max_image_frac=distal_max_image_frac,
                            distal_lr_halfspace_slack=distal_lr_halfspace_slack,
                        ),
                    )
                if run_s4t and _local_stage_pending(subj, STAGE_SEG_T, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_SEG_T,
                        lambda s=subj: stage4t_4dflow_t_segmentation.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            crop_padding_bbox=crop_padding_bbox,
                            thr_algorithm=thr_algorithm_4dflow.lower(),
                            region_growing=region_growing,
                            rg_intensity_frac=rg_intensity_frac,
                            rg_intensity_frac_explore=rg_intensity_frac_explore,
                            rg_intensity_frac_aca=rg_intensity_frac_aca,
                            cl_barrier_radius=cl_barrier_radius,
                            rg_barrier_radius=rg_barrier_radius,
                            aca_sequential_grow=aca_sequential_grow,
                            aca_overlap_min_voxels=aca_overlap_min_voxels,
                            acomm_junction_radius=acomm_junction_radius,
                        ),
                    )
                if run_s5 and _local_stage_pending(subj, STAGE_LOC, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_LOC,
                        lambda s=subj: stage5_loc_generation.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            loc_arterial_strategy=loc_arterial_strategy,
                            cross_section_radius_vox=cross_section_radius_vox,
                            venous_min_component_frac=venous_min_component_frac,
                            loc_endpoint_inset_frac=loc_endpoint_inset_frac,
                            distal_locs=distal_locs,
                        ),
                    )
                if run_s6 and _local_stage_pending(subj, STAGE_MEASURE, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_MEASURE,
                        lambda s=subj: stage6_measure.run_subject(
                            s,
                            nifti_root=nifti_root_eff,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                            cross_section_radius_vox=cross_section_radius_vox,
                            measure_resegment=measure_resegment,
                            measure_thr_algorithm=measure_thr_algorithm.lower(),
                            cross_section_res=cross_section_res,
                            cross_section_plane_interp=cross_section_plane_interp,
                            cs_supersampling=cs_supersampling,
                            pitc_stride=pitc_stride,
                            pitc_quality_thresh=pitc_quality_thresh,
                            pitc_quality_metric=pitc_quality_metric.lower(),
                            pitc_measure_resegment=pitc_measure_resegment,
                            pitc_label_constrain=pitc_label_constrain,
                            save_plots=save_plots,
                        ),
                    )
                if run_s7 and _local_stage_pending(subj, STAGE_MORPHOMETRICS, **_local_skip):
                    run.run_stage(
                        subj,
                        STAGE_MORPHOMETRICS,
                        lambda s=subj: stage7_morphometrics.run_subject(
                            s,
                            output_root=output_root_eff,
                            skip_existing=skip_existing,
                        ),
                    )

    if run_s8:
        from nvitk.pipes.qvtpy.util.io.xnat_upload import parse_require_stages, run_xnat_upload

        if xnat_conn is None:
            raise click.ClickException("stage8_xnat_upload requires XNAT credentials.")
        try:
            required_xnat_stages = parse_require_stages(xnat_upload_require_stages)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        results_source: str = "local" if submit == "local" else "cluster"
        remote_results: Path | None = None
        if results_source == "cluster":
            remote_results = cluster_paths.results_root
            if not (ssh_host_resolved and ssh_user and ssh_password):
                from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials

                ssh_host_resolved, ssh_user, ssh_password = prompt_ssh_credentials(
                    remote_host=remote_host,
                    remote_user=remote_user,
                    host_aliases=cfg.CLUSTER_HOST_ALIASES,
                )

        run_xnat_upload(
            subject_list,
            output_root=output_root_eff,
            xnat_config=xnat_conn,
            required_stages=required_xnat_stages,
            upload_eicab=xnat_upload_eicab,
            upload_qvtpy=xnat_upload_qvtpy,
            overwrite=not xnat_upload_skip_existing,
            skip_existing=xnat_upload_skip_existing,
            dry_run=xnat_upload_dry_run,
            results_source=results_source,
            ssh_host=ssh_host_resolved,
            ssh_user=ssh_user,
            ssh_password=ssh_password,
            remote_results_root=remote_results,
        )

    if submit == "local":
        if run_conv and report:
            from . import stage0_convert

            stage0_convert.print_nifti_qc_report(
                nifti_root_eff, subject_list, check_derived=report_derived
            )
        return

    cluster_stages = (
        run_conv or run_eicab or run_s2 or run_s3 or run_s4 or run_s4t or run_s5 or run_s6 or run_s7
    )
    if not cluster_stages:
        log.info("Nothing to submit to SGE (only stage0_d was selected). Done.")
        return

    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_runs_map = {
        "run_conv": run_conv,
        "run_eicab": run_eicab,
        "run_s2": run_s2,
        "run_s3": run_s3,
        "run_s4": run_s4,
        "run_s4t": run_s4t,
        "run_s5": run_s5,
        "run_s6": run_s6,
        "run_s7": run_s7,
    }
    if skip_processed:
        subject_list, skipped_complete = filter_subjects_pending_work(
            subject_list,
            stage_runs=stage_runs_map,
            skip_processed=True,
            results_root=output_root_eff,
            nifti_root=nifti_root_eff,
        )
        if skipped_complete:
            log.info(
                f"  skipped {len(skipped_complete)} subject(s) with all requested "
                "stages already complete"
            )
        if not subject_list:
            log.info("All subjects already complete for requested stages; nothing to submit to SGE.")
            return

    # One SGE array job per subject (tasks = pending stages).
    jobs_per_subject = 1
    stages_per_subject = count_sge_stages_per_subject(
        run_conv=run_conv,
        run_eicab=run_eicab,
        run_s2=run_s2,
        run_s3=run_s3,
        run_s4=run_s4,
        run_s4t=run_s4t,
        run_s5=run_s5,
        run_s6=run_s6,
        run_s7=run_s7,
    )
    chunk_size = max(1, int(sge_subject_chunk_size))
    serial_subjects = bool(sge_serial_subjects) or chunk_size == 1
    warn_if_chunk_exceeds_sge_limit(
        chunk_size, jobs_per_subject, margin=int(sge_job_margin)
    )
    initial_subjects = subject_list[:chunk_size]
    remaining_subjects = subject_list[chunk_size:]
    drip_mode = "serial (one subject at a time)" if serial_subjects else "capacity-based"
    log.info(
        f"SGE submit: initial batch {len(initial_subjects)} subject(s), "
        f"drip {len(remaining_subjects)} more at {jobs_per_subject} array job(s)/subject "
        f"(up to {stages_per_subject} stage task(s)/subject; {drip_mode})"
    )

    emit_kwargs = dict(
        run_conv=run_conv,
        run_eicab=run_eicab,
        run_s2=run_s2,
        run_s3=run_s3,
        run_s4=run_s4,
        run_s4t=run_s4t,
        run_s5=run_s5,
        run_s6=run_s6,
        run_s7=run_s7,
        dicom_root_eff=dicom_root_eff,
        nifti_root_eff=nifti_root_eff,
        output_root_eff=output_root_eff,
        container=container,
        src_p=src_p,
        backend=backend,
        skip_existing=skip_existing,
        compute_phase_derived=compute_phase_derived,
        phase_background_correction=phase_background_correction,
        phase_bg_poly_order=phase_bg_poly_order,
        phase_bg_static_percentile=phase_bg_static_percentile,
        eicab_resolution=eicab_resolution,
        eicab_device=eicab_device,
        eicab_thread_limit=eicab_thread_eff,
        eicab_sge_pe_smp=eicab_pe_smp_eff,
        eicab_local_metric_scratch=eicab_metric_scratch_eff,
        eicab_container=eicab_container,
        vasculature_dir=vasculature_dir,
        post_process_eicab=post_process_eicab,
        only_pp=only_pp,
        stage2_reference=stage2_reference,
        stage2_dof=stage2_dof,
        stage2_cost=stage2_cost,
        eicab_mask=eicab_mask,
        eicab_prefer_pp=eicab_prefer_pp,
        cd_up_thresh=cd_up_thresh,
        cd_shift_hm=cd_shift_hm,
        venous_min_component_frac=venous_min_component_frac,
        eicab_min_island_fraction=eicab_min_island_fraction,
        eicab_bridge_open_radius=eicab_bridge_open_radius,
        venous_min_branch_points=venous_min_branch_points,
        pcomm_min_points=pcomm_min_points,
        arterial_branch_min_points=arterial_branch_min_points,
        venous_brain_mask=venous_brain_mask,
        totalseg_model_dir=totalseg_model_dir_eff,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm_4dflow=thr_algorithm_4dflow,
        region_growing=region_growing,
        rg_intensity_frac=rg_intensity_frac,
        rg_intensity_frac_explore=rg_intensity_frac_explore,
        rg_intensity_frac_aca=rg_intensity_frac_aca,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
        distal_flow_expand=distal_flow_expand,
        distal_hyst_low_factor=distal_hyst_low_factor,
        distal_hyst_high_factor=distal_hyst_high_factor,
        distal_thicken_iter=distal_thicken_iter,
        distal_max_image_frac=distal_max_image_frac,
        distal_lr_halfspace_slack=distal_lr_halfspace_slack,
        loc_arterial_strategy=loc_arterial_strategy,
        cross_section_radius_vox=cross_section_radius_vox,
        loc_endpoint_inset_frac=loc_endpoint_inset_frac,
        distal_locs=distal_locs,
        measure_resegment=measure_resegment,
        measure_thr_algorithm=measure_thr_algorithm,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        cs_supersampling=cs_supersampling,
        save_plots=save_plots,
        skip_processed=skip_processed,
        pitc_stride=pitc_stride,
        pitc_quality_thresh=pitc_quality_thresh,
        pitc_quality_metric=pitc_quality_metric,
        pitc_measure_resegment=pitc_measure_resegment,
        pitc_label_constrain=pitc_label_constrain,
    )

    if run_conv and report:
        log.info("NIfTI report skipped: conversion is queued on the cluster.")

    if no_remote:
        initial_base = (
            Path(emit_script).name
            if emit_script is not None
            else f"submit_qvtpy_{ts}_initial.sh"
        )
        local_script_path, remote_script_path = resolve_sge_script_paths(
            None,
            remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
            default_basename=initial_base,
        )
        with open(local_script_path, "w", encoding="utf-8") as fh:
            write_script_header(
                fh,
                log_dir=cfg.SGE_LOG_DIR,
                err_dir=cfg.SGE_ERR_DIR,
                title=(
                    f"qvtpy stages={','.join(stages)} initial "
                    f"n_subjects={len(initial_subjects)}"
                ),
            )
            _emit_qvtpy_sge_subjects_for_chunk(fh, initial_subjects, **emit_kwargs)
        log.info(f"Initial batch script: {local_script_path}")
        log.info(f"Cluster path: {remote_script_path}")

        if remaining_subjects:
            drip_dir = local_script_path.parent / f"{ts}_drip"
            drip_dir.mkdir(parents=True, exist_ok=True)
            for subj in remaining_subjects:
                token = _sge_script_subject_token(subj)
                drip_local, drip_remote = resolve_sge_script_paths(
                    None,
                    remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
                    default_basename=f"submit_qvtpy_{ts}_subj_{token}.sh",
                )
                drip_local = drip_dir / drip_local.name
                with open(drip_local, "w", encoding="utf-8") as fh:
                    write_script_header(
                        fh,
                        log_dir=cfg.SGE_LOG_DIR,
                        err_dir=cfg.SGE_ERR_DIR,
                        title=f"qvtpy stages={','.join(stages)} subject={subj}",
                    )
                    _emit_qvtpy_sge_subjects_for_chunk(fh, [subj], **emit_kwargs)
            log.info(
                f"Drip scripts for {len(remaining_subjects)} subject(s) in {drip_dir} "
                "(submit manually when queue capacity allows)."
            )
        log.info("Skipping remote SSH (--no-remote). Upload and run scripts on the cluster.")
        return

    log.reset(restart_progress=False)
    if not (ssh_host_resolved and ssh_user and ssh_password):
        from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials

        ssh_host_resolved, ssh_user, ssh_password = prompt_ssh_credentials(
            remote_host=remote_host,
            remote_user=remote_user,
            host_aliases=cfg.CLUSTER_HOST_ALIASES,
        )

    initial_base = (
        Path(emit_script).name if emit_script is not None else f"submit_qvtpy_{ts}_initial.sh"
    )
    log.info("=" * 78)
    log.info(f"SGE initial batch: {len(initial_subjects)} subject(s)")
    log.info("=" * 78)
    exit_code, job_ids = _submit_qvtpy_sge_subjects_remote(
        initial_subjects,
        script_basename=initial_base,
        title=(
            f"qvtpy stages={','.join(stages)} initial n_subjects={len(initial_subjects)}"
        ),
        emit_kwargs=emit_kwargs,
        ssh_host=ssh_host_resolved,
        ssh_user=ssh_user,
        ssh_password=ssh_password,
    )
    log.info(f"Initial batch: parsed {len(job_ids)} SGE job id(s) from submission output.")
    if exit_code != 0:
        log.error(f"Initial batch submission exited {exit_code}; stopping.")
        return

    if serial_subjects and job_ids and remaining_subjects:
        log.info(
            f"SGE serial: waiting for initial batch ({len(job_ids)} job(s)) "
            "before drip-feeding remaining subjects."
        )
        if not wait_for_sge_job_ids(
            ssh_host_resolved,
            ssh_user,
            ssh_password,
            job_ids,
            poll_interval=float(sge_chunk_poll_interval),
            wait_timeout=sge_chunk_wait_timeout,
        ):
            log.error("SGE serial: initial batch wait timed out; stopping.")
            return

    if not remaining_subjects:
        return

    def _submit_drip_subject(subj: str) -> tuple[bool, list[str]]:
        token = _sge_script_subject_token(subj)
        code, drip_ids = _submit_qvtpy_sge_subjects_remote(
            [subj],
            script_basename=f"submit_qvtpy_{ts}_subj_{token}.sh",
            title=f"qvtpy stages={','.join(stages)} subject={subj}",
            emit_kwargs=emit_kwargs,
            ssh_host=ssh_host_resolved,
            ssh_user=ssh_user,
            ssh_password=ssh_password,
        )
        log.info(f"Drip {subj}: parsed {len(drip_ids)} SGE job id(s).")
        return code == 0, drip_ids

    def _jobs_for_subject(subj: str) -> int:
        """Queue slots: one array job per subject with pending stage work."""
        if not skip_processed:
            return jobs_per_subject
        n_stages = count_sge_stages_for_subject(
            subj,
            stage_runs=stage_runs_from_emit_kwargs(emit_kwargs),
            skip_processed=True,
            results_root=output_root_eff,
            nifti_root=nifti_root_eff,
        )
        return 1 if n_stages > 0 else 0

    ok = drip_submit_subjects(
        remaining_subjects,
        _submit_drip_subject,
        host=ssh_host_resolved,
        user=ssh_user,
        password=ssh_password,
        jobs_per_subject=_jobs_for_subject,
        poll_interval=float(sge_chunk_poll_interval),
        loop_timeout=sge_chunk_wait_timeout,
        margin=int(sge_job_margin),
        serial=serial_subjects,
        wait_timeout=sge_chunk_wait_timeout,
    )
    if not ok:
        log.error("SGE drip submission stopped before all subjects were queued.")


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
    "STAGE_MORPHOMETRICS",
    "STAGE_XNAT_UPLOAD",
]


if __name__ == "__main__":
    main()
