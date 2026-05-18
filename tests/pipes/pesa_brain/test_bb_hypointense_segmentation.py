"""Black-blood hypointense region-growing (native BB intensities)."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    build_seg_bb_centerline_growth,
)
from nvitk.segmentation.region_growing import region_grow_into_label_volume


def _synthetic_dark_tube(shape: tuple[int, int, int] = (40, 40, 40)) -> tuple[np.ndarray, np.ndarray]:
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


def test_hypointense_lower_frac_grows_less() -> None:
    wvi, cl = _synthetic_dark_tube()
    seg_lo = cl.copy()
    seg_hi = cl.copy()
    n_lo = region_grow_into_label_volume(
        seg_lo, wvi, 1, intensity_frac=0.35, polarity="hypointense"
    )
    n_hi = region_grow_into_label_volume(
        seg_hi, wvi, 1, intensity_frac=0.75, polarity="hypointense"
    )
    assert n_lo < n_hi


def test_hyperintense_rg_on_dark_tube_prefers_bright_background() -> None:
    wvi, cl = _synthetic_dark_tube()
    seg = cl.copy()
    region_grow_into_label_volume(
        seg, wvi, 1, intensity_frac=0.45, polarity="hyperintense"
    )
    assert np.mean(wvi[seg == 1]) > 150.0


def test_centerline_growth_pipeline_targets_dark_lumen() -> None:
    wvi, cl = _synthetic_dark_tube()
    result = build_seg_bb_centerline_growth(wvi, cl, rg_intensity_frac=0.45)
    lumen = wvi < 100.0
    assert np.count_nonzero((result.seg == 1) & lumen) > 10
