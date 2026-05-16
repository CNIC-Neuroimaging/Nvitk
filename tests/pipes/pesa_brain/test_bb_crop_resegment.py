"""Tests for crop-resegment BB segmentation."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.labels import BB_LICA
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    build_seg_bb_crop_resegment,
)


def test_crop_resegment_synthetic_tube() -> None:
    shape = (24, 24, 24)
    wvi = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    for z in range(8, 16):
        wvi[11:14, 11:14, z] = 100.0
        clm[12, 12, z] = BB_LICA

    result = build_seg_bb_crop_resegment(
        wvi,
        clm,
        thr_algorithm="lsthr",
        crop_padding_bbox=2,
        cl_barrier_radius=1,
        min_component_frac=0.0,
    )
    assert int(np.count_nonzero(result.seg == BB_LICA)) > 0
