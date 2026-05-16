"""Tests for centerline-growth BB segmentation."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.labels import BB_LICA
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    build_seg_bb_centerline_growth,
)


def test_centerline_growth_with_barrier() -> None:
    shape = (16, 16, 16)
    wvi = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    wvi[4:12, 8, 8] = 80.0
    clm[6, 8, 8] = BB_LICA
    clm[7, 8, 8] = BB_LICA

    result = build_seg_bb_centerline_growth(
        wvi,
        clm,
        rg_intensity_frac=0.4,
        cl_barrier_radius=1,
        rg_barrier_radius=1,
    )
    assert int(np.count_nonzero(result.seg == BB_LICA)) > 2
