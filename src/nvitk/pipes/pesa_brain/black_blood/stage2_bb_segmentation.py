"""
Black-blood stage 2: eICAB centerlines in native ``vwi_bb`` space + BB segmentation.

Workflow
--------
1. Warp eICAB CW/WB labels into ``vwi_bb`` grid (stage1 ``tof_to_vwi_bb.mat`` when needed).
2. Rasterize centerlines → ``centerlines_mask.nii.gz``; write ``eicab_bb_in_vwi_bb.nii.gz``.
3. Optional BB smoothing, then eICAB-guided hypointense lumen segmentation per vessel.
4. Write ``seg_bb.nii.gz`` with ``vwi_bb`` affine.
"""

from __future__ import annotations

import json
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.io.imageio import imread
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import run_bb_segmentation
from nvitk.pipes.pesa_brain.black_blood.util.vwi_preprocess import VwiPreprocess, preprocess_vwi_bb
from nvitk.pipes.pesa_brain.black_blood.util.centerlines_from_eicab import (
    EICAB_BB_IN_VWI_BB_NIFTI,
    build_centerlines_from_eicab,
)
from nvitk.pipes.pesa_brain.black_blood.util.eicab_masks import EicabMaskKind

log = Logger()


def _load_stage1_meta(output_root: Path, subject: str) -> dict:
    meta_path = paths.registration_meta_path(output_root, subject)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing stage1 registration_meta.json for {subject}: {meta_path}"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_results_root: Path,
    skip_existing: bool = False,
    vwi_bb_rel: str | None = None,
    eicab_subdir: str | None = None,
    eicab_mask: EicabMaskKind = "cw",
    bbox_padding: int = 8,
    eicab_prior_dilate: int = 3,
    centerline_max_dist: int = 12,
    ica_centerline_max_dist: int = 6,
    lumen_intensity_frac: float = 1.2,
    lumen_percentile: float = 55.0,
    ica_lumen_intensity_frac: float = 1.05,
    ica_lumen_percentile: float = 42.0,
    rg_intensity_frac: float = 1.0,
    rg_constraint: bool = True,
    min_component_frac: float = 0.005,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 2,
    min_centerline_points: int = 5,
    vwi_preprocess: VwiPreprocess = "median",
    vwi_median_size: int = 3,
    vwi_gaussian_sigma: float = 0.8,
) -> Path:
    """Build centerlines and eICAB-guided BB segmentation in native ``vwi_bb`` space."""
    _load_stage1_meta(output_root, subject)
    vwi_bb_ref = paths.vwi_bb_path(nifti_root, subject, vwi_bb_rel=vwi_bb_rel)
    mat = paths.registration_matrix_path(output_root, subject)

    mask_res = paths.eicab_mask_resolution(
        eicab_results_root,
        subject,
        eicab_mask=eicab_mask,
        eicab_subdir=eicab_subdir,
    )
    out_dir = paths.stage2_dir(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"pesa_brain stage2 | subject={subject} strategy=eicab_guided_centerline_lumen "
        f"space=vwi_bb eicab_mask={mask_res.used} (requested={mask_res.requested})"
    )

    log.step("building centerlines from eICAB in vwi_bb space")
    build_centerlines_from_eicab(
        mask_res.path,
        vwi_bb_ref,
        out_dir,
        transform_mat=mat,
        min_points=min_centerline_points,
        skip_existing=skip_existing,
        eicab_mask_info=mask_res,
    )

    vwi_img = imread(vwi_bb_ref)
    cl_img = imread(out_dir / "centerlines_mask.nii.gz")
    eicab_path = out_dir / EICAB_BB_IN_VWI_BB_NIFTI
    if not eicab_path.is_file():
        raise FileNotFoundError(
            f"Missing {eicab_path.name} for {subject}; re-run centerline build."
        )
    eicab_img = imread(eicab_path)

    ref_shape = tuple(int(x) for x in vwi_img.data.shape[:3])
    for name, img in (
        ("centerlines_mask", cl_img),
        ("eicab_bb_in_vwi_bb", eicab_img),
    ):
        if tuple(img.data.shape[:3]) != ref_shape:
            raise ValueError(
                f"vwi_bb shape {ref_shape} != {name} shape {img.data.shape[:3]} for {subject}"
            )

    wvi_data = preprocess_vwi_bb(
        vwi_img.data,
        vwi_preprocess,
        median_size=vwi_median_size,
        gaussian_sigma=vwi_gaussian_sigma,
    )
    if vwi_preprocess != "none":
        log.step(f"preprocessed vwi_bb ({vwi_preprocess}) for segmentation")

    log.step("running eICAB-guided hypointense lumen segmentation")
    run_bb_segmentation(
        wvi_data,
        cl_img.data,
        eicab_img.data,
        out_dir,
        bbox_padding=bbox_padding,
        eicab_prior_dilate=eicab_prior_dilate,
        centerline_max_dist=centerline_max_dist,
        ica_centerline_max_dist=ica_centerline_max_dist,
        lumen_intensity_frac=lumen_intensity_frac,
        lumen_percentile=lumen_percentile,
        ica_lumen_intensity_frac=ica_lumen_intensity_frac,
        ica_lumen_percentile=ica_lumen_percentile,
        rg_intensity_frac=rg_intensity_frac,
        rg_constraint=rg_constraint,
        min_component_frac=min_component_frac,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        metadata=dict(vwi_img.metadata or {}),
        skip_existing=skip_existing,
    )
    return out_dir
