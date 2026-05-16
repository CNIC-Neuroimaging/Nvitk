"""Tests for centerline rasterization from synthetic multilabel."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.labels import BB_LICA
from nvitk.pipes.pesa_brain.black_blood.util.centerlines_from_eicab import (
    rasterize_centerlines_mask,
)
from nvitk.morphology.centerline import compute_centerlines


def test_rasterize_synthetic_tube_centerlines() -> None:
    shape = (20, 20, 20)
    labels = np.zeros(shape, dtype=np.int32)
    for z in range(5, 15):
        labels[10, 10, z] = BB_LICA

    centerlines = compute_centerlines(
        labels,
        centerline_mask=labels > 0,
        labels=[BB_LICA],
        min_points=3,
    )
    assert BB_LICA in centerlines
    mask = rasterize_centerlines_mask(shape, centerlines)
    assert int(np.count_nonzero(mask == BB_LICA)) >= 3
