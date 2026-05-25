"""3D Napari dim order matches post-transpose in-plane layout."""

from nvitk.gui.orientation import axial_dim_order, napari_dim_order_3d


def test_napari_dim_order_3d_swaps_in_plane_axes():
    assert napari_dim_order_3d(None, 3) == (2, 1, 0)
    assert axial_dim_order(None, 3) == (2, 0, 1)
