"""
Black-blood stage 2: eICAB in ``vwi_bb`` space + mask-based lumen segmentation.

Workflow
--------
1. Warp eICAB CW/WB to ``vwi_bb``; write ``eicab_bb_in_vwi_bb.nii.gz`` and centerlines (QC).
2. Optional ``vwi_bb`` smoothing.
3. Per vessel: dilated eICAB ROI → hypointense threshold (ROI intensities only) → ``seg_bb``.
"""

from __future__ import annotations

import json
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.io.imageio import imread
from nvitk.pipes.bbtpy.util import paths
from nvitk.pipes.bbtpy.util.bb_vessel_segmentation import (
    ThrAlgorithm,
    run_bb_segmentation,
)
from nvitk.pipes.bbtpy.util.centerlines_from_eicab import (
    EICAB_BB_IN_VWI_BB_NIFTI,
    build_centerlines_from_eicab,
)
from nvitk.pipes.bbtpy.util.eicab_masks import EicabMaskKind
from nvitk.pipes.bbtpy.util.vwi_preprocess import VwiPreprocess, preprocess_vwi_bb

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
    eicab_dilate: int = 4,
    thr_algorithm: ThrAlgorithm = "lsthr",
    min_component_frac: float = 0.005,
    min_centerline_points: int = 5,
    vwi_preprocess: VwiPreprocess = "none",
    vwi_median_size: int = 3,
    vwi_gaussian_sigma: float = 0.8,
) -> Path:
    """Build eICAB/centerline artifacts and mask-threshold BB segmentation."""
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
        f"bbtpy stage2 | subject={subject} strategy=eicab_mask_hypointense_threshold "
        f"thr={thr_algorithm} eicab_mask={mask_res.used} (requested={mask_res.requested})"
    )

    log.step("warping eICAB and building centerlines in vwi_bb space")
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
    eicab_path = out_dir / EICAB_BB_IN_VWI_BB_NIFTI
    if not eicab_path.is_file():
        raise FileNotFoundError(
            f"Missing {eicab_path.name} for {subject}; re-run centerline build."
        )
    eicab_img = imread(eicab_path)
    if tuple(vwi_img.data.shape[:3]) != tuple(eicab_img.data.shape[:3]):
        raise ValueError(
            f"vwi_bb shape {vwi_img.data.shape[:3]} != eicab_bb "
            f"{eicab_img.data.shape[:3]} for {subject}"
        )

    wvi_data = preprocess_vwi_bb(
        vwi_img.data,
        vwi_preprocess,
        median_size=vwi_median_size,
        gaussian_sigma=vwi_gaussian_sigma,
    )
    if vwi_preprocess != "none":
        log.step(f"preprocessed vwi_bb ({vwi_preprocess})")

    log.step("running per-vessel eICAB-mask hypointense thresholding")
    run_bb_segmentation(
        wvi_data,
        eicab_img.data,
        out_dir,
        eicab_dilate=eicab_dilate,
        thr_algorithm=thr_algorithm,
        min_component_frac=min_component_frac,
        metadata=dict(vwi_img.metadata or {}),
        skip_existing=skip_existing,
    )
    return out_dir
