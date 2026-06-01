"""Napari 4D dims order (pre-GIF-export behaviour)."""

from __future__ import annotations

import numpy as np

from nvitk.gui.core.orientation import napari_affine_for_display, napari_dim_order


def test_napari_dim_order_xyzt():
    aff = np.diag([0.5, 0.5, 1.0, 1.0]).astype(float)
    assert napari_dim_order("XYZT", aff, 4) == (3, 0, 1, 2)


def test_napari_affine_decouples_time_from_spatial():
    aff = np.eye(4)
    aff[0, 3] = 99.0
    aff[3, 0] = 88.0
    out = napari_affine_for_display(aff, (10, 11, 12, 15), "XYZT", {"t_res": 1.0})
    assert out is not None
    assert out[0, 3] == 0.0
    assert out[3, 0] == 0.0
    assert out[3, 3] == 1.0
