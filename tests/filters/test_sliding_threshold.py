"""Tests for nvitk.filters.sliding_threshold."""

from __future__ import annotations

import numpy as np

from nvitk.filters.sliding_threshold import (
    binary_mask_sliding_threshold_2d,
    binary_mask_sliding_threshold_3d,
)


def test_sliding_threshold_3d_on_bright_blob() -> None:
    vol = np.zeros((20, 20, 20), dtype=np.float64)
    vol[8:12, 8:12, 8:12] = 100.0
    mask, opt = binary_mask_sliding_threshold_3d(vol, med_filt_flag=False)
    assert opt >= 0.0
    assert mask[10, 10, 10]
    assert int(np.count_nonzero(mask)) > 0
    assert int(np.count_nonzero(mask)) < vol.size


def test_sliding_threshold_2d_on_bright_disk() -> None:
    sl = np.zeros((32, 32), dtype=np.float64)
    yy, xx = np.ogrid[:32, :32]
    sl[(yy - 16) ** 2 + (xx - 16) ** 2 < 36] = 50.0
    mask = binary_mask_sliding_threshold_2d(sl)
    assert mask[16, 16]
    assert int(np.count_nonzero(mask)) < sl.size
