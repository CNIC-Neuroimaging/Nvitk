"""4D display: scale-only load path and dim ranges."""

import numpy as np

from nvitk.gui.orientation import napari_dim_order, napari_scale_for_display, prepare_for_napari


def test_xyzt_dim_order():
    assert napari_dim_order("XYZT", None, 4) == (3, 0, 1, 2)


def test_prepare_for_napari_4d_uses_scale_not_affine():
    aff = np.eye(4)
    aff[3, 0] = 0.8
    aff[0, 3] = -43.0
    data = np.zeros((256, 256, 101, 15))
    md = {"x_res": 0.78, "y_res": 0.78, "z_res": 0.78, "t_res": 0.8}
    out, display_affine, scale = prepare_for_napari(
        data, aff, axes="XYZT", metadata=md
    )
    assert display_affine is None
    assert scale == (0.78, 0.78, 0.78, 0.8)


def test_napari_scale_for_display_defaults():
    scale = napari_scale_for_display((10, 20, 30, 5), "XYZT", {})
    assert scale == (1.0, 1.0, 1.0, 1.0)


def test_napari_viewer_4d_temporal_range_with_scale():
    pytest = __import__("pytest")
    napari = pytest.importorskip("napari")

    data = np.zeros((256, 256, 101, 15))
    scale = napari_scale_for_display(data.shape, "XYZT", {"t_res": 0.8})
    viewer = napari.Viewer(show=False)
    viewer.add_image(data, scale=scale, axis_labels=("X", "Y", "Z", "T"))
    viewer.dims.order = (3, 0, 1, 2)
    viewer.dims.ndisplay = 3
    r = viewer.dims.range[3]
    n_steps = (r.stop - r.start) / r.step + 1
    assert abs(n_steps - 15.0) < 0.01
    viewer.close()
