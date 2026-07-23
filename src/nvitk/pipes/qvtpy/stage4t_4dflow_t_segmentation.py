"""qvtpy stage 4t: per-timepoint local CD segmentation → ``seg_4dflow_4d``.

**Inputs**

- ``ComplexDifference_4D`` (requires stage-0 ``--compute-phase-derived``), stage-3 centerlines.

**Outputs**

- ``seg_4dflow_4d.nii.gz``, per-timepoint ``segmentation_meta_t*.json``, ``temporal_seg_summary.json``.
"""

from __future__ import annotations

import json
import shlex
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO

import click

import nvitk
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stage4_4dflow_segmentation import EICAB_IN_4DFLOW_NIFTI
from nvitk.pipes.qvtpy.util.centerline_io import (
    centerline_meta_path,
    centerlines_mask_path,
    export_centerlines_from_segmentation,
    load_venous_centerlines,
)
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    ThrAlgorithm,
    VESSEL_EXTRA_PADDING,
    _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    _ACOMM_JUNCTION_RADIUS_DEFAULT,
    _DEFAULT_RG_INTENSITY_FRAC,
    _RG_INTENSITY_FRAC_EXPLORE,
    build_seg_4dflow_local,
    vessel_stats_to_dict,
)

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage4t_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4T_SEG_DIR


def _cd4d_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_4D.nii.gz"
    if p.is_file():
        return p
    p2 = nifti_root / subject / "4DFlow" / "ComplexDifference_4D.nii"
    if p2.is_file():
        return p2
    raise FileNotFoundError(
        f"Missing ComplexDifference_4D for {subject} (run stage0_c with --compute-phase-derived)"
    )


def _meta_path_for_timepoint(out_dir: Path, t: int) -> Path:
    return out_dir / f"segmentation_meta_t{t:02d}.json"


def _outputs_complete(out_dir: Path, n_timepoints: int) -> bool:
    if not (out_dir / "seg_4dflow_4d.nii.gz").is_file():
        return False
    if not (out_dir / "temporal_seg_summary.json").is_file():
        return False
    return all(_meta_path_for_timepoint(out_dir, t).is_file() for t in range(n_timepoints))


# ---------------------------------------------------------------------------
# Temporal QC helpers
# ---------------------------------------------------------------------------


def dice_binary(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """Dice coefficient for two boolean masks; ``None`` if both empty."""
    a = as_backend_array(mask_a).astype(bool)
    b = as_backend_array(mask_b).astype(bool)
    inter = int(np.count_nonzero(a & b))
    sa = int(np.count_nonzero(a))
    sb = int(np.count_nonzero(b))
    if sa == 0 and sb == 0:
        return None
    if sa + sb == 0:
        return None
    return float(2.0 * inter / (sa + sb))


def multilabel_dice(seg_a: np.ndarray, seg_b: np.ndarray) -> float | None:
    """Mean Dice over labels present in either volume (foreground union)."""
    labels = sorted(
        int(v)
        for v in np.unique(np.concatenate([seg_a.ravel(), seg_b.ravel()]))
        if int(v) > 0
    )
    if not labels:
        return None
    scores: list[float] = []
    for lid in labels:
        d = dice_binary(seg_a == lid, seg_b == lid)
        if d is not None:
            scores.append(d)
    if not scores:
        return None
    return float(np.mean(np.array(scores)))


def build_temporal_seg_summary(
    seg_stack: np.ndarray,
    *,
    subject: str,
    complex_difference_4d: str,
    centerlines_mask: str,
    eicab_in_4dflow: str | None,
    label_ids: list[int],
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
) -> dict[str, Any]:
    """Summarize per-label stability across temporal segmentations."""
    if seg_stack.ndim != 4:
        raise ValueError(f"seg_stack must be 4D, got shape {seg_stack.shape}")
    n_t = int(seg_stack.shape[3])
    per_label: dict[str, Any] = {}
    multilabel_dice_vs_t0: list[float | None] = []

    for lid in label_ids:
        counts: list[int] = []
        dice_vs_t0: list[float | None] = []
        ref = seg_stack[..., 0] == lid
        for t in range(n_t):
            m = seg_stack[..., t] == lid
            counts.append(int(np.count_nonzero(m)))
            dice_vs_t0.append(dice_binary(ref, m))

        pair_scores: list[float] = []
        for t_a, t_b in combinations(range(n_t), 2):
            d = dice_binary(seg_stack[..., t_a] == lid, seg_stack[..., t_b] == lid)
            if d is not None:
                pair_scores.append(d)
        mean_pairwise = float(np.mean(np.array(pair_scores))) if pair_scores else None

        per_label[str(lid)] = {
            "voxel_count": counts,
            "dice_vs_t0": dice_vs_t0,
            "mean_pairwise_dice": mean_pairwise,
        }

    for t in range(n_t):
        multilabel_dice_vs_t0.append(multilabel_dice(seg_stack[..., 0], seg_stack[..., t]))

    return {
        "subject": subject,
        "complex_difference_4d": complex_difference_4d,
        "centerlines_mask": centerlines_mask,
        "eicab_in_4dflow": eicab_in_4dflow,
        "n_timepoints": n_t,
        "crop_padding_bbox": int(crop_padding_bbox),
        "vessel_extra_padding": int(VESSEL_EXTRA_PADDING),
        "thr_algorithm": thr_algorithm,
        "region_growing": bool(region_growing),
        "rg_intensity_frac": float(rg_intensity_frac),
        "rg_intensity_frac_explore": float(rg_intensity_frac_explore),
        "cl_barrier_radius": int(cl_barrier_radius),
        "rg_barrier_radius": int(rg_barrier_radius),
        "aca_sequential_grow": bool(aca_sequential_grow),
        "aca_overlap_min_voxels": int(aca_overlap_min_voxels),
        "acomm_junction_radius": int(acomm_junction_radius),
        "per_label": per_label,
        "multilabel_dice_vs_t0": multilabel_dice_vs_t0,
    }


def _segmentation_meta_for_timepoint(
    *,
    subject: str,
    timepoint: int,
    n_timepoints: int,
    complex_difference_4d: str,
    cl_path: Path,
    eicab_in_4dflow: str | None,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    aca_sequential_grow_info: dict[str, Any] | None,
    vessels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-timepoint ``segmentation_meta`` JSON payload (mirrors stage-4 fields)."""
    return {
        "subject": subject,
        "timepoint": int(timepoint),
        "n_timepoints": int(n_timepoints),
        "complex_difference_4d": complex_difference_4d,
        "centerlines_mask": str(cl_path),
        "eicab_in_4dflow": eicab_in_4dflow,
        "crop_padding_bbox": int(crop_padding_bbox),
        "vessel_extra_padding": int(VESSEL_EXTRA_PADDING),
        "thr_algorithm": thr_algorithm,
        "region_growing": bool(region_growing),
        "rg_intensity_frac": float(rg_intensity_frac),
        "rg_intensity_frac_explore": float(rg_intensity_frac_explore),
        "rg_intensity_frac_explore_labels": "ACA,MCA,PCA (ids 4-9)",
        "rg_skip_labels": "all venous (ids 31-34)",
        "venous_region_growing": False,
        "comm_segmentation_strategy": "centerline_rg_only",
        "post_threshold_clean": "largest_cc_per_label",
        "cl_barrier_radius": int(cl_barrier_radius),
        "rg_barrier_radius": int(rg_barrier_radius),
        "aca_sequential_grow": bool(aca_sequential_grow),
        "aca_overlap_min_voxels": int(aca_overlap_min_voxels),
        "acomm_junction_radius": int(acomm_junction_radius),
        "aca_sequential_grow_info": aca_sequential_grow_info,
        "vessels": vessels,
    }


# ---------------------------------------------------------------------------
# Stage 4t: per-timepoint segmentation
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
) -> Path:
    """Segment each CD time frame; return stage-4t output directory."""
    s3 = _stage3_dir(output_root, subject)
    cl_path = centerlines_mask_path(s3)
    if not cl_path.is_file():
        raise FileNotFoundError(f"Missing {cl_path} (run stage3)")

    eicab_path = s3 / EICAB_IN_4DFLOW_NIFTI
    eicab_qvtpy = None
    if eicab_path.is_file():
        eicab_qvtpy = as_backend_array(imread(eicab_path).data).astype(np.int32, copy=False)
    else:
        log.warning(f"[{subject}] stage4t: missing {eicab_path}; ACA Voronoi uses centerlines only")

    cd4d_path = _cd4d_path(nifti_root, subject)
    cd_img = imread(cd4d_path)
    cd4d = as_backend_array(cd_img.data).astype(np.float64)
    if cd4d.ndim != 4:
        raise ValueError(f"Expected 4D ComplexDifference for {subject}, got shape {cd4d.shape}")

    n_t = int(cd4d.shape[3])
    out_dir = _stage4t_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "seg_4dflow_4d.nii.gz"
    summary_path = out_dir / "temporal_seg_summary.json"

    if skip_existing and _outputs_complete(out_dir, n_t):
        log.info(f"[{subject}] stage4t seg: skip -> {out_dir}")
        return out_dir

    cl_img = imread(cl_path)
    centerlines_mask = as_backend_array(cl_img.data).astype(np.int32, copy=False)

    label_ids = sorted(int(v) for v in np.unique(centerlines_mask) if int(v) > 0)
    seg_stack = np.zeros((*cd4d.shape[:3], n_t), dtype=np.int32)
    cd4d_str = str(cd4d_path)
    eicab_str = str(eicab_path) if eicab_path.is_file() else None

    build_kw = dict(
        centerlines_mask=centerlines_mask,
        eicab_qvtpy=eicab_qvtpy,
        crop_padding_bbox=int(crop_padding_bbox),
        thr_algorithm=thr_algorithm,
        region_growing=bool(region_growing),
        rg_intensity_frac=float(rg_intensity_frac),
        rg_intensity_frac_explore=float(rg_intensity_frac_explore),
        cl_barrier_radius=int(cl_barrier_radius),
        rg_barrier_radius=int(rg_barrier_radius),
        aca_sequential_grow=bool(aca_sequential_grow),
        aca_overlap_min_voxels=int(aca_overlap_min_voxels),
        acomm_junction_radius=int(acomm_junction_radius),
    )

    vertebral_split_t0 = None
    for t in range(n_t):
        result = build_seg_4dflow_local(cd4d[..., t], **build_kw)
        seg_stack[..., t] = as_backend_array(result.segmentation).astype(np.int32, copy=False)
        if t == 0 and result.vertebral_split is not None:
            vertebral_split_t0 = result.vertebral_split
        meta_t = _segmentation_meta_for_timepoint(
            subject=subject,
            timepoint=t,
            n_timepoints=n_t,
            complex_difference_4d=cd4d_str,
            cl_path=cl_path,
            eicab_in_4dflow=eicab_str,
            crop_padding_bbox=crop_padding_bbox,
            thr_algorithm=thr_algorithm,
            region_growing=region_growing,
            rg_intensity_frac=rg_intensity_frac,
            rg_intensity_frac_explore=rg_intensity_frac_explore,
            cl_barrier_radius=cl_barrier_radius,
            rg_barrier_radius=rg_barrier_radius,
            aca_sequential_grow=aca_sequential_grow,
            aca_overlap_min_voxels=aca_overlap_min_voxels,
            acomm_junction_radius=acomm_junction_radius,
            aca_sequential_grow_info=(
                None
                if result.aca_sequential_grow is None
                else result.aca_sequential_grow.as_dict()
            ),
            vessels=[vessel_stats_to_dict(st) for st in result.vessel_stats],
        )
        _meta_path_for_timepoint(out_dir, t).write_text(
            json.dumps(meta_t, indent=2),
            encoding="utf-8",
        )

    ref_meta = dict(cd_img.metadata or {})
    ref_meta["axes"] = "XYZT"
    ref_meta["shape"] = tuple(seg_stack.shape)
    imsave(seg_path, seg_stack, metadata=ref_meta)

    summary = build_temporal_seg_summary(
        seg_stack,
        subject=subject,
        complex_difference_4d=cd4d_str,
        centerlines_mask=str(cl_path),
        eicab_in_4dflow=eicab_str,
        label_ids=label_ids,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm,
        region_growing=region_growing,
        rg_intensity_frac=rg_intensity_frac,
        rg_intensity_frac_explore=rg_intensity_frac_explore,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if vertebral_split_t0 is not None:
        (out_dir / "vertebral_split.json").write_text(
            json.dumps(vertebral_split_t0.as_dict(), indent=2),
            encoding="utf-8",
        )

    # venous_polylines: dict[str, Any] = {}
    # venous_label_by_name: dict[str, int] = {}
    # if centerline_meta_path(s3).is_file():
    #     meta3 = json.loads(centerline_meta_path(s3).read_text(encoding="utf-8"))
    #     venous_label_by_name = {
    #         str(k): int(v) for k, v in (meta3.get("venous_label_by_name") or {}).items()
    #     }
    #     venous_polylines = load_venous_centerlines(s3, min_points=3, meta=meta3)
    # export_centerlines_from_segmentation(
    #     seg_stack[..., 0],
    #     out_dir,
    #     metadata=ref_meta,
    #     venous_polylines=venous_polylines,
    #     venous_label_by_name=venous_label_by_name,
    # )

    log.info(f"[{subject}] stage4t segmentation ({n_t} frames) -> {seg_path}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _stage4t_cli_options(func):
    func = click.option("--subject", required=True)(func)
    func = click.option("--nifti-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--output-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--skip-existing", is_flag=True, default=False)(func)
    func = click.option("--crop-padding-bbox", type=int, default=3, show_default=True)(func)
    func = click.option(
        "--4dflow-thr-algorithm",
        "thr_algorithm",
        type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
        default="lsthr",
        show_default=True,
    )(func)
    func = click.option(
        "--region-growing/--no-region-growing",
        default=True,
        show_default=True,
    )(func)
    func = click.option(
        "--rg-intensity-frac",
        type=float,
        default=_DEFAULT_RG_INTENSITY_FRAC,
        show_default=True,
        help="RG gate: grow_thresh = max(mean(CD_seeds)*frac, local_thr). Lower = more growth.",
    )(func)
    func = click.option(
        "--rg-intensity-frac-explore",
        type=float,
        default=_RG_INTENSITY_FRAC_EXPLORE,
        show_default=True,
        help="RG frac for ACA/MCA/PCA (lower = explore more).",
    )(func)
    func = click.option("--cl-barrier-radius", type=int, default=2, show_default=True)(func)
    func = click.option("--rg-barrier-radius", type=int, default=3, show_default=True)(func)
    func = click.option(
        "--aca-sequential-grow/--no-aca-sequential-grow",
        default=True,
        show_default=True,
        help="Grow LACA then RACA; second ACA ignores first as RG barrier; fix overlap at junction.",
    )(func)
    func = click.option(
        "--aca-overlap-min-voxels",
        type=int,
        default=_ACA_OVERLAP_MIN_VOXELS_DEFAULT,
        show_default=True,
        help="Min LACA∩RACA voxels to apply convergence correction at AComm.",
    )(func)
    func = click.option(
        "--acomm-junction-radius",
        type=int,
        default=_ACOMM_JUNCTION_RADIUS_DEFAULT,
        show_default=True,
        help="Vox: only overlap within this distance of AComm junction is Voronoi-split.",
    )(func)
    return func


def _subject_sge_spec(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: str = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    backend: str = "gpu",
) -> tuple[StageSpec, ClusterPaths]:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage4t_4dflow_t_segmentation"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--crop-padding-bbox",
        str(int(crop_padding_bbox)),
        "--4dflow-thr-algorithm",
        shlex.quote(str(thr_algorithm).lower()),
        "--rg-intensity-frac",
        str(float(rg_intensity_frac)),
        "--rg-intensity-frac-explore",
        str(float(rg_intensity_frac_explore)),
        "--cl-barrier-radius",
        str(int(cl_barrier_radius)),
        "--rg-barrier-radius",
        str(int(rg_barrier_radius)),
        "--aca-overlap-min-voxels",
        str(int(aca_overlap_min_voxels)),
        "--acomm-junction-radius",
        str(int(acomm_junction_radius)),
    ]
    if region_growing:
        parts.append("--region-growing")
    else:
        parts.append("--no-region-growing")
    if aca_sequential_grow:
        parts.append("--aca-sequential-grow")
    else:
        parts.append("--no-aca-sequential-grow")
    if skip_existing:
        parts.append("--skip-existing")
    python_cmd = " ".join(parts)
    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=None,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage4t_{subject}",
        python_cmd=python_cmd,
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: str = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    backend: str = "gpu",
) -> str:
    """Return the host shell command for one stage4t array/SGE task."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm,
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
    return build_singularity_command(spec, paths)


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
    crop_padding_bbox: int = 3,
    thr_algorithm: str = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-4t SGE job. Returns qsub job id."""
    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm,
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
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage4t-seg")
@backend_click_option()
@_stage4t_cli_options
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm.lower(),
        region_growing=region_growing,
        rg_intensity_frac=rg_intensity_frac,
        rg_intensity_frac_explore=rg_intensity_frac_explore,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
    )


__all__ = [
    "build_subject_sge_command",
    "build_temporal_seg_summary",
    "dice_binary",
    "main",
    "multilabel_dice",
    "run_subject",
    "submit_subject_sge",
]


if __name__ == "__main__":
    main()
