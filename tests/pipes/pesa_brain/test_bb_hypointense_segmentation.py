"""Black-blood hypointense region-growing on native BB intensities."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    build_seg_bb_centerline_growth,
    run_bb_segmentation,
)
from nvitk.segmentation.region_growing import region_grow_into_label_volume


def _synthetic_dark_tube(shape: tuple[int, int, int] = (40, 40, 40)) -> tuple[np.ndarray, np.ndarray]:
    """Bright background with a dark cylindrical lumen along z."""
    wvi = np.full(shape, 220.0, dtype=np.float64)
    cl = np.zeros(shape, dtype=np.int32)
    cx, cy = shape[0] // 2, shape[1] // 2
    for z in range(8, shape[2] - 8):
        for x in range(cx - 3, cx + 4):
            for y in range(cy - 3, cy + 4):
                wvi[x, y, z] = 40.0
        cl[cx, cy, z] = 1
    return wvi, cl


def test_hypointense_rg_grows_dark_not_bright() -> None:
    wvi, cl = _synthetic_dark_tube()
    seg = cl.copy()
    region_grow_into_label_volume(
        seg, wvi, 1, intensity_frac=0.45, polarity="hypointense"
    )
    lumen = wvi < 100.0
    overlap = np.count_nonzero((seg == 1) & lumen)
    bright = np.count_nonzero((seg == 1) & ~lumen)
    assert overlap > 10
    assert bright < overlap


def test_hyperintense_rg_on_dark_tube_prefers_bright_background() -> None:
    wvi, cl = _synthetic_dark_tube()
    seg = cl.copy()
    region_grow_into_label_volume(
        seg, wvi, 1, intensity_frac=0.45, polarity="hyperintense"
    )
    grown = seg == 1
    assert np.mean(wvi[grown]) > 150.0


def test_centerline_growth_pipeline_targets_dark_lumen() -> None:
    wvi, cl = _synthetic_dark_tube()
    result = build_seg_bb_centerline_growth(wvi, cl, rg_intensity_frac=0.45)
    lumen = wvi < 100.0
    overlap = np.count_nonzero((result.seg == 1) & lumen)
    assert overlap > 10


def test_run_bb_segmentation_writes_metadata(tmp_path) -> None:
    wvi, cl = _synthetic_dark_tube()
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    meta = {"affine": affine, "spacing": (2.0, 2.0, 2.0)}
    run_bb_segmentation(wvi, cl, tmp_path, metadata=meta)
    from nvitk.io.imageio import imread

    seg_img = imread(tmp_path / "seg_bb.nii.gz")
    assert tuple(seg_img.data.shape[:3]) == wvi.shape
    assert np.allclose(seg_img.metadata.get("affine"), affine)
