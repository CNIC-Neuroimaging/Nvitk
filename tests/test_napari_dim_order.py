"""Tests for Napari 4D dim ordering (time vs spatial axes)."""

from nvitk.gui.orientation import napari_dim_order


def test_xyzt_puts_time_first_then_xyz():
    order = napari_dim_order("XYZT", None, 4)
    assert order == (3, 0, 1, 2)


def test_xyz_3d_superior_first():
    order = napari_dim_order("XYZ", None, 3)
    assert len(order) == 3
    assert set(order) == {0, 1, 2}
