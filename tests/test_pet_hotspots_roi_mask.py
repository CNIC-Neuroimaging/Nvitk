"""Tests for PET hotspot ROI mask selection."""

from __future__ import annotations

import numpy as np

from nvitk.viz.pet_hotspots import _roi_mask


def test_roi_mask_binary_mask_ignores_original_label_ids() -> None:
    """After label-mode binarization, ROI is all voxels > 0 (not original label values)."""
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    roi = _roi_mask(mask, [2, 3, 5])
    assert roi.sum() == 8
    assert np.array_equal(roi, mask > 0)


def test_roi_mask_multilabel_image_uses_label_ids() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 2
    mask[2, 2, 2] = 3
    roi = _roi_mask(mask, [2, 3])
    assert roi.sum() == 2
    assert roi[1, 1, 1]
    assert roi[2, 2, 2]
    assert not roi[0, 0, 0]
