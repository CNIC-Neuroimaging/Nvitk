"""Black-blood stage 2: centerlines in vwi_bb space + BB artery segmentation."""

from __future__ import annotations

import json
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.io.imageio import imread
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    SegStrategy,
    ThrAlgorithm,
    run_bb_segmentation,
)
from nvitk.pipes.pesa_brain.black_blood.util.centerlines_from_eicab import (
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
    seg_strategy: SegStrategy,
    skip_existing: bool = False,
    vwi_bb_rel: str | None = None,
    eicab_subdir: str | None = None,
    eicab_mask: EicabMaskKind = "cw",
    thr_algorithm: ThrAlgorithm = "otsu",
    crop_padding_bbox: int = 3,
    cl_barrier_radius: int = 2,
    min_component_frac: float = 0.005,
    rg_intensity_frac: float = 0.45,
    rg_barrier_radius: int = 2,
    min_centerline_points: int = 5,
) -> Path:
    """Build centerlines and segment in native vwi_bb space."""
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
        f"pesa_brain stage2 | subject={subject} strategy={seg_strategy} "
        f"space=vwi_bb eicab_mask={mask_res.used} (requested={mask_res.requested})"
    )

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

    run_bb_segmentation(
        vwi_img.data,
        cl_img.data,
        out_dir,
        strategy=seg_strategy,
        thr_algorithm=thr_algorithm,
        crop_padding_bbox=crop_padding_bbox,
        cl_barrier_radius=cl_barrier_radius,
        min_component_frac=min_component_frac,
        rg_intensity_frac=rg_intensity_frac,
        rg_barrier_radius=rg_barrier_radius,
        skip_existing=skip_existing,
    )
    return out_dir
