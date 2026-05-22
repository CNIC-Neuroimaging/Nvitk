"""Tests for convex hull and EDT helpers."""

import numpy as np

from nvitk.segmentation.hull_edt import convex_hull_slicewise, distance_transform


def test_convex_hull_slicewise_fills_cavity():
    vol = np.zeros((8, 8, 4), dtype=np.uint8)
    vol[2:6, 2:6, 1] = 1
    hull = convex_hull_slicewise(vol, axis=2)
    data = np.asarray(hull.data if hasattr(hull, "data") else hull)
    assert int(data[..., 1].sum()) >= int(vol[..., 1].sum())


def test_distance_transform_tube():
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    vol[5, 5, 2:8] = 1
    tube = distance_transform(vol, spacing=(1.0, 1.0, 1.0), radius_mm=2.0)
    data = tube.data if hasattr(tube, "data") else tube
    assert int(np.asarray(data).sum()) > 0
