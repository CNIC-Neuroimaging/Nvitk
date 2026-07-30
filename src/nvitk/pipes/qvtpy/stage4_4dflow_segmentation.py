"""qvtpy stage 4: per-vessel local CD threshold + region growing → ``seg_4dflow``.

**Inputs**

- ``ComplexDifference_3D``, stage-3 ``centerlines_mask``, optional ``eicab_in_4dflow``.

**Outputs**

- ``seg_4dflow.nii.gz``, ``segmentation_meta.json``, stage-4 centerline exports.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, TextIO

import click

import nvitk
from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.centerline.centerline_io import (
    centerline_meta_path,
    centerlines_mask_path,
    export_centerlines_from_segmentation,
    load_arterial_centerlines,
    load_venous_centerlines,
)
from nvitk.pipes.qvtpy.util.segmentation.vessel_cd_segmentation import (
    ThrAlgorithm,
    VESSEL_EXTRA_PADDING,
    _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    _ACOMM_JUNCTION_RADIUS_DEFAULT,
    _DEFAULT_RG_INTENSITY_FRAC,
    _RG_INTENSITY_FRAC_ACA,
    _RG_INTENSITY_FRAC_EXPLORE,
    _RG_MAX_GROW_FRAC_DEFAULT,
    _RG_MAX_IMAGE_FRAC_DEFAULT,
    build_seg_4dflow_local,
    vessel_stats_to_dict,
)

setup(globals())

log = Logger()

EICAB_IN_4DFLOW_NIFTI = "eicab_in_4dflow.nii.gz"
EICAB_IN_4DFLOW_EICAB_IDS_NIFTI = "eicab_in_4dflow_eicab_ids.nii.gz"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage4_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR


def _cd_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii.gz"
    if p.is_file():
        return p
    p2 = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Missing ComplexDifference_3D for {subject}")


def _segmentation_meta(
    *,
    subject: str,
    nifti_root: Path,
    cl_path: Path,
    eicab_in_4dflow: str | None,
    eicab_in_4dflow_eicab_ids: str | None,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    rg_intensity_frac_aca: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    rg_max_grow_frac: float,
    rg_max_image_frac: float,
    venous_region_growing: bool,
    segment_acomm: bool,
    aca_sequential_grow_info: dict[str, Any] | None,
    vessels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "subject": subject,
        "complex_difference": str(_cd_path(nifti_root, subject)),
        "centerlines_mask": str(cl_path),
        "eicab_in_4dflow": eicab_in_4dflow,
        "eicab_in_4dflow_eicab_ids": eicab_in_4dflow_eicab_ids,
        "crop_padding_bbox": int(crop_padding_bbox),
        "vessel_extra_padding": int(VESSEL_EXTRA_PADDING),
        "thr_algorithm": thr_algorithm,
        "region_growing": bool(region_growing),
        "rg_intensity_frac": float(rg_intensity_frac),
        "rg_intensity_frac_explore": float(rg_intensity_frac_explore),
        "rg_intensity_frac_aca": float(rg_intensity_frac_aca),
        "rg_intensity_frac_explore_labels": "MCA,PCA (ids 7-9)",
        "rg_intensity_frac_aca_labels": "ACA (ids 4-5)",
        "eicab_rg_barrier_label_ids": [15, 16],
        "eicab_rg_barrier_vessels": "PCA,basilar",
        "rg_skip_labels": "STRV only (id 32)" if venous_region_growing else "all venous (ids 31-34)",
        "venous_region_growing": bool(venous_region_growing),
        "rg_max_grow_frac": float(rg_max_grow_frac),
        "rg_max_image_frac": float(rg_max_image_frac),
        "segment_acomm": bool(segment_acomm),
        "eicab_pcomm_barrier_label_ids": [15, 16],
        "comm_segmentation_strategy": "centerline_plus_threshold_rg",
        "post_threshold_clean": "largest_cc_per_label",
        "centerlines_from_seg": "centerlines_mask_4dflow.nii.gz",
        "cl_barrier_radius": int(cl_barrier_radius),
        "rg_barrier_radius": int(rg_barrier_radius),
        "aca_sequential_grow": bool(aca_sequential_grow),
        "aca_overlap_min_voxels": int(aca_overlap_min_voxels),
        "acomm_junction_radius": int(acomm_junction_radius),
        "aca_sequential_grow_info": aca_sequential_grow_info,
        "vessels": vessels,
    }


# ---------------------------------------------------------------------------
# Stage 4: local CD crop + threshold + largest-CC clean + region growing
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
    rg_intensity_frac_aca: float = _RG_INTENSITY_FRAC_ACA,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    rg_max_grow_frac: float = _RG_MAX_GROW_FRAC_DEFAULT,
    rg_max_image_frac: float = _RG_MAX_IMAGE_FRAC_DEFAULT,
    venous_region_growing: bool = True,
    segment_acomm: bool = False,
    distal_flow_expand: bool = False,
    distal_hyst_low_factor: float = 3.5,
    distal_hyst_high_factor: float = 0.5,
    distal_thicken_iter: int = 0,
    distal_max_image_frac: float = 0.006,
    distal_lr_halfspace_slack: int = 2,
) -> Path:
    """Build multilabel 4D-flow segmentation locally; return stage-4 output directory."""
    s3 = _stage3_dir(output_root, subject)
    cl_path = centerlines_mask_path(s3)
    if not cl_path.is_file():
        raise FileNotFoundError(f"Missing {cl_path} (run stage3)")

    eicab_path = s3 / EICAB_IN_4DFLOW_NIFTI
    eicab_qvtpy = None
    if eicab_path.is_file():
        eicab_qvtpy = as_backend_array(imread(eicab_path).data).astype(np.int32, copy=False)
    else:
        log.warning(f"[{subject}] stage4: missing {eicab_path}; ACA Voronoi uses centerlines only")

    native_eicab_path = s3 / EICAB_IN_4DFLOW_EICAB_IDS_NIFTI
    eicab_native = None
    if native_eicab_path.is_file():
        eicab_native = as_backend_array(imread(native_eicab_path).data).astype(np.int32, copy=False)
    else:
        log.warning(
            f"[{subject}] stage4: missing {native_eicab_path}; "
            "PCA/basilar RG will omit native eICAB SCA barriers (re-run stage3)"
        )

    out_dir = _stage4_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "seg_4dflow.nii.gz"
    meta_path = out_dir / "segmentation_meta.json"
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        log.info(f"[{subject}] stage4 seg: skip -> {out_dir}")
        return out_dir

    cd_img = imread(_cd_path(nifti_root, subject))
    cd = as_backend_array(cd_img.data).astype(np.float64)
    ref_meta = dict(cd_img.metadata or {})

    cl_img = imread(cl_path)
    centerlines_mask = as_backend_array(cl_img.data).astype(np.int32, copy=False)

    log.step(
        f"per-vessel local CD seg (thr={thr_algorithm}, "
        f"RG={region_growing}, rg_frac={rg_intensity_frac})"
    )
    if bool(distal_flow_expand):
        log.step(
            f"[{subject}] distal-flow-expand ON "
            f"(vessel-tree watershed; "
            f"hyst_low={float(distal_hyst_low_factor):.2f}, "
            f"hyst_high={float(distal_hyst_high_factor):.2f}, "
            f"thicken={int(distal_thicken_iter)}, "
            f"max_frac={float(distal_max_image_frac):.4f})"
        )
    else:
        log.info(f"[{subject}] distal-flow-expand OFF (default); skipping distal pass")

    result = build_seg_4dflow_local(
        cd,
        centerlines_mask,
        eicab_qvtpy=eicab_qvtpy,
        eicab_native=eicab_native,
        crop_padding_bbox=int(crop_padding_bbox),
        thr_algorithm=thr_algorithm,
        region_growing=bool(region_growing),
        rg_intensity_frac=float(rg_intensity_frac),
        rg_intensity_frac_explore=float(rg_intensity_frac_explore),
        rg_intensity_frac_aca=float(rg_intensity_frac_aca),
        cl_barrier_radius=int(cl_barrier_radius),
        rg_barrier_radius=int(rg_barrier_radius),
        aca_sequential_grow=bool(aca_sequential_grow),
        aca_overlap_min_voxels=int(aca_overlap_min_voxels),
        acomm_junction_radius=int(acomm_junction_radius),
        rg_max_grow_frac=float(rg_max_grow_frac),
        rg_max_image_frac=float(rg_max_image_frac),
        venous_region_growing=bool(venous_region_growing),
        segment_acomm=bool(segment_acomm),
        distal_flow_expand=bool(distal_flow_expand),
        distal_hyst_low_factor=float(distal_hyst_low_factor),
        distal_hyst_high_factor=float(distal_hyst_high_factor),
        distal_thicken_iter=int(distal_thicken_iter),
        distal_max_image_frac=float(distal_max_image_frac),
        distal_lr_halfspace_slack=int(distal_lr_halfspace_slack),
    )

    imsave(seg_path, result.segmentation, metadata=ref_meta)
    if result.vertebral_split is not None:
        (out_dir / "vertebral_split.json").write_text(
            json.dumps(result.vertebral_split.as_dict(), indent=2),
            encoding="utf-8",
        )

    venous_polylines: dict[str, Any] = {}
    venous_label_by_name: dict[str, int] = {}
    prefer_arterial: dict[int, Any] = {}
    if centerline_meta_path(s3).is_file():
        meta3 = json.loads(centerline_meta_path(s3).read_text(encoding="utf-8"))
        venous_label_by_name = {
            str(k): int(v) for k, v in (meta3.get("venous_label_by_name") or {}).items()
        }
        venous_polylines = load_venous_centerlines(s3, min_points=3, meta=meta3)
        if centerlines_mask_path(s3).is_file():
            prefer_arterial = load_arterial_centerlines(s3, min_points=3, meta=meta3)
        stage3_arterial_labels = {
            int(x) for x in (meta3.get("arterial_labels") or [])
        }
        # Honor stage-3 PCOMM drops: never keep a mask/CL for a PCOMM that was
        # filtered by pcomm_min_points.
        from nvitk.pipes.qvtpy.labels import QVTPY_PCOMM_IDS

        seg_arr = as_backend_array(result.segmentation)
        cleared = []
        for lid in QVTPY_PCOMM_IDS:
            if int(lid) in stage3_arterial_labels:
                continue
            if int(np.count_nonzero(seg_arr == int(lid))) == 0:
                continue
            seg_arr[seg_arr == int(lid)] = 0
            cleared.append(int(lid))
        if cleared:
            result.segmentation = seg_arr
            imsave(seg_path, seg_arr, metadata=ref_meta)
            log.info(
                f"[{subject}] stage4: cleared dropped PCOMM mask label(s): {cleared}"
            )
    export_centerlines_from_segmentation(
        result.segmentation,
        out_dir,
        metadata=ref_meta,
        venous_polylines=venous_polylines,
        venous_label_by_name=venous_label_by_name,
        prefer_polylines=prefer_arterial,
    )

    meta_doc = _segmentation_meta(
                subject=subject,
                nifti_root=nifti_root,
                cl_path=cl_path,
                eicab_in_4dflow=str(eicab_path) if eicab_path.is_file() else None,
                eicab_in_4dflow_eicab_ids=(
                    str(native_eicab_path) if native_eicab_path.is_file() else None
                ),
                crop_padding_bbox=crop_padding_bbox,
                thr_algorithm=thr_algorithm,
                region_growing=region_growing,
                rg_intensity_frac=rg_intensity_frac,
                rg_intensity_frac_explore=rg_intensity_frac_explore,
                rg_intensity_frac_aca=rg_intensity_frac_aca,
                cl_barrier_radius=cl_barrier_radius,
                rg_barrier_radius=rg_barrier_radius,
                aca_sequential_grow=aca_sequential_grow,
                aca_overlap_min_voxels=aca_overlap_min_voxels,
                acomm_junction_radius=acomm_junction_radius,
                rg_max_grow_frac=rg_max_grow_frac,
                rg_max_image_frac=rg_max_image_frac,
                venous_region_growing=venous_region_growing,
                segment_acomm=segment_acomm,
                aca_sequential_grow_info=(
                    None
                    if result.aca_sequential_grow is None
                    else result.aca_sequential_grow.as_dict()
                ),
                vessels=[vessel_stats_to_dict(st) for st in result.vessel_stats],
            )
    meta_doc["distal_flow_expand"] = bool(distal_flow_expand)
    meta_doc["distal_method"] = "frangi_hysteresis_watershed"
    meta_doc["distal_hyst_low_factor"] = float(distal_hyst_low_factor)
    meta_doc["distal_hyst_high_factor"] = float(distal_hyst_high_factor)
    meta_doc["distal_thicken_iter"] = int(distal_thicken_iter)
    meta_doc["distal_max_image_frac"] = float(distal_max_image_frac)
    meta_doc["distal_lr_halfspace_slack"] = int(distal_lr_halfspace_slack)
    if result.distal_expand is not None:
        meta_doc["distal_expand"] = result.distal_expand
        de = result.distal_expand
        log.info(
            f"[{subject}] distal expand meta: "
            f"method={de.get('method')}, "
            f"labels={list((de.get('labels') or {}).keys())}"
        )
    meta_path.write_text(
        json.dumps(meta_doc, indent=2),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage4 segmentation -> {seg_path}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _stage4_cli_options(func):
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
        help="RG frac for MCA/PCA (lower = explore more).",
    )(func)
    func = click.option(
        "--rg-intensity-frac-aca",
        type=float,
        default=_RG_INTENSITY_FRAC_ACA,
        show_default=True,
        help="RG frac for ACA sequential grow (lower = explore more; default 0.25).",
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
    func = click.option(
        "--rg-max-grow-frac",
        type=float,
        default=_RG_MAX_GROW_FRAC_DEFAULT,
        show_default=True,
        help="Roll back a region grow if grown voxels exceed this multiple of the seed.",
    )(func)
    func = click.option(
        "--rg-max-image-frac",
        type=float,
        default=_RG_MAX_IMAGE_FRAC_DEFAULT,
        show_default=True,
        help="Roll back a region grow if the label exceeds this fraction of the volume.",
    )(func)
    func = click.option(
        "--venous-region-growing/--no-venous-region-growing",
        default=True,
        show_default=True,
        help="Conservative CD region growing for venous sinuses (SSSV/LTSV/RTSV).",
    )(func)
    func = click.option(
        "--segment-acomm/--no-segment-acomm",
        default=False,
        show_default=True,
        help="Grow AComm as its own label; off by default (used only for ACA L/R split).",
    )(func)
    func = click.option(
        "--distal-flow-expand/--no-distal-flow-expand",
        default=False,
        show_default=True,
        help=(
            "After region growing, expand MCA/ACA/PCA into a Frangi+hysteresis "
            "vessel tree via watershed (eICAB-inspired, Python-only). Default OFF."
        ),
    )(func)
    func = click.option(
        "--distal-hyst-low-factor",
        type=float,
        default=3.5,
        show_default=True,
        help=(
            "Distal GMM hysteresis low factor (higher → thinner tree; try 3.5–4.5 to reduce blobs)."
        ),
    )(func)
    func = click.option(
        "--distal-hyst-high-factor",
        type=float,
        default=0.5,
        show_default=True,
        help=(
            "Distal GMM hysteresis high factor. Lower → more high seeds / thicker tree; "
            "raise slightly to thin."
        ),
    )(func)
    func = click.option(
        "--distal-thicken-iter",
        type=int,
        default=0,
        show_default=True,
        help=(
            "Optional binary dilations of the Frangi tree inside a high-CD gate. "
            "0 = thinnest (recommended); 1 recovers lumen but can look blobbier."
        ),
    )(func)
    func = click.option(
        "--distal-max-image-frac",
        type=float,
        default=0.006,
        show_default=True,
        help="Cap on total voxels claimed by distal expand (fraction of image).",
    )(func)
    func = click.option(
        "--distal-lr-halfspace-slack",
        type=int,
        default=2,
        show_default=True,
        help="L/R midline slack (voxels) so ACA/MCA/PCA cannot claim the contralateral side.",
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
    rg_intensity_frac_aca: float = _RG_INTENSITY_FRAC_ACA,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    rg_max_grow_frac: float = _RG_MAX_GROW_FRAC_DEFAULT,
    rg_max_image_frac: float = _RG_MAX_IMAGE_FRAC_DEFAULT,
    venous_region_growing: bool = True,
    segment_acomm: bool = False,
    distal_flow_expand: bool = False,
    distal_hyst_low_factor: float = 3.5,
    distal_hyst_high_factor: float = 0.5,
    distal_thicken_iter: int = 0,
    distal_max_image_frac: float = 0.006,
    distal_lr_halfspace_slack: int = 2,
    backend: str = "gpu",
) -> tuple[StageSpec, ClusterPaths]:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage4_4dflow_segmentation"),
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
        "--rg-intensity-frac-aca",
        str(float(rg_intensity_frac_aca)),
        "--cl-barrier-radius",
        str(int(cl_barrier_radius)),
        "--rg-barrier-radius",
        str(int(rg_barrier_radius)),
        "--aca-overlap-min-voxels",
        str(int(aca_overlap_min_voxels)),
        "--acomm-junction-radius",
        str(int(acomm_junction_radius)),
        "--rg-max-grow-frac",
        str(float(rg_max_grow_frac)),
        "--rg-max-image-frac",
        str(float(rg_max_image_frac)),
    ]
    if region_growing:
        parts.append("--region-growing")
    else:
        parts.append("--no-region-growing")
    if aca_sequential_grow:
        parts.append("--aca-sequential-grow")
    else:
        parts.append("--no-aca-sequential-grow")
    if venous_region_growing:
        parts.append("--venous-region-growing")
    else:
        parts.append("--no-venous-region-growing")
    if segment_acomm:
        parts.append("--segment-acomm")
    else:
        parts.append("--no-segment-acomm")
    if distal_flow_expand:
        parts.append("--distal-flow-expand")
        parts.extend(
            [
                "--distal-hyst-low-factor",
                str(float(distal_hyst_low_factor)),
                "--distal-hyst-high-factor",
                str(float(distal_hyst_high_factor)),
                "--distal-thicken-iter",
                str(int(distal_thicken_iter)),
                "--distal-max-image-frac",
                str(float(distal_max_image_frac)),
                "--distal-lr-halfspace-slack",
                str(int(distal_lr_halfspace_slack)),
            ]
        )
    else:
        parts.append("--no-distal-flow-expand")
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
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage4_{subject}",
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
    rg_intensity_frac_aca: float = _RG_INTENSITY_FRAC_ACA,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    rg_max_grow_frac: float = _RG_MAX_GROW_FRAC_DEFAULT,
    rg_max_image_frac: float = _RG_MAX_IMAGE_FRAC_DEFAULT,
    venous_region_growing: bool = True,
    segment_acomm: bool = False,
    distal_flow_expand: bool = False,
    distal_hyst_low_factor: float = 3.5,
    distal_hyst_high_factor: float = 0.5,
    distal_thicken_iter: int = 0,
    distal_max_image_frac: float = 0.006,
    distal_lr_halfspace_slack: int = 2,
    backend: str = "gpu",
) -> str:
    """Return the host shell command for one stage4 array/SGE task."""
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
        rg_intensity_frac_aca=rg_intensity_frac_aca,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
        rg_max_grow_frac=rg_max_grow_frac,
        rg_max_image_frac=rg_max_image_frac,
        venous_region_growing=venous_region_growing,
        segment_acomm=segment_acomm,
        distal_flow_expand=distal_flow_expand,
        distal_hyst_low_factor=distal_hyst_low_factor,
        distal_hyst_high_factor=distal_hyst_high_factor,
        distal_thicken_iter=distal_thicken_iter,
        distal_max_image_frac=distal_max_image_frac,
        distal_lr_halfspace_slack=distal_lr_halfspace_slack,
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
    rg_intensity_frac_aca: float = _RG_INTENSITY_FRAC_ACA,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    rg_max_grow_frac: float = _RG_MAX_GROW_FRAC_DEFAULT,
    rg_max_image_frac: float = _RG_MAX_IMAGE_FRAC_DEFAULT,
    venous_region_growing: bool = True,
    segment_acomm: bool = False,
    distal_flow_expand: bool = False,
    distal_hyst_low_factor: float = 3.5,
    distal_hyst_high_factor: float = 0.5,
    distal_thicken_iter: int = 0,
    distal_max_image_frac: float = 0.006,
    distal_lr_halfspace_slack: int = 2,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-4 SGE job. Returns qsub job id."""
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
        rg_intensity_frac_aca=rg_intensity_frac_aca,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
        rg_max_grow_frac=rg_max_grow_frac,
        rg_max_image_frac=rg_max_image_frac,
        venous_region_growing=venous_region_growing,
        segment_acomm=segment_acomm,
        distal_flow_expand=distal_flow_expand,
        distal_hyst_low_factor=distal_hyst_low_factor,
        distal_hyst_high_factor=distal_hyst_high_factor,
        distal_thicken_iter=distal_thicken_iter,
        distal_max_image_frac=distal_max_image_frac,
        distal_lr_halfspace_slack=distal_lr_halfspace_slack,
        backend=backend,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage4-seg")
@backend_click_option()
@_stage4_cli_options
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
    rg_intensity_frac_aca: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    rg_max_grow_frac: float,
    rg_max_image_frac: float,
    venous_region_growing: bool,
    segment_acomm: bool,
    distal_flow_expand: bool,
    distal_hyst_low_factor: float,
    distal_hyst_high_factor: float,
    distal_thicken_iter: int,
    distal_max_image_frac: float,
    distal_lr_halfspace_slack: int,
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
        rg_intensity_frac_aca=rg_intensity_frac_aca,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
        rg_max_grow_frac=rg_max_grow_frac,
        rg_max_image_frac=rg_max_image_frac,
        venous_region_growing=venous_region_growing,
        segment_acomm=segment_acomm,
        distal_flow_expand=distal_flow_expand,
        distal_hyst_low_factor=distal_hyst_low_factor,
        distal_hyst_high_factor=distal_hyst_high_factor,
        distal_thicken_iter=distal_thicken_iter,
        distal_max_image_frac=distal_max_image_frac,
        distal_lr_halfspace_slack=distal_lr_halfspace_slack,
    )


__all__ = [
    "EICAB_IN_4DFLOW_EICAB_IDS_NIFTI",
    "EICAB_IN_4DFLOW_NIFTI",
    "main",
    "run_subject",
    "build_subject_sge_command",
    "submit_subject_sge",
]


if __name__ == "__main__":
    main()
