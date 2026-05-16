"""Tests for nvitk.segmentation.region_growing."""

from __future__ import annotations

import numpy as np

from nvitk.segmentation.region_growing import (
    region_grow_binary_mask,
    region_grow_into_label_volume,
)


def test_region_grow_into_label_volume() -> None:
    cd = np.zeros((10, 10, 10), dtype=np.float64)
    cd[4:7, 4:7, 4:7] = 50.0
    seg = np.zeros((10, 10, 10), dtype=np.int32)
    seg[5, 5, 5] = 1
    n = region_grow_into_label_volume(seg, cd, 1, intensity_frac=0.5, abs_floor=0.0)
    assert n > 0
    assert int(np.count_nonzero(seg == 1)) > 1


def test_region_grow_binary_mask() -> None:
    cd = np.zeros((8, 8, 8), dtype=np.float64)
    cd[2:6, 2:6, 2:6] = 40.0
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[3, 3, 3] = True
    n = region_grow_binary_mask(mask, cd, intensity_frac=0.5, abs_floor=0.0)
    assert n > 0
    assert mask[4, 4, 4]
