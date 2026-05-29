"""Tests for Napari display orientation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nvitk.gui.core.orientation import reorient_layer_for_view


class _FakeEvents:
    def __init__(self) -> None:
        self.affine_calls = 0
        self.data_calls = 0

    def affine(self, value=None) -> None:
        self.affine_calls += 1

    def data(self, value=None) -> None:
        self.data_calls += 1


class _FakeLayer:
    def __init__(self, data: np.ndarray, affine: np.ndarray, *, axes: str | None = None) -> None:
        self.data = data
        self.affine = affine
        self.axis_labels = tuple(axes) if axes else None
        self.metadata = {"axes": axes} if axes else {}
        self.events = _FakeEvents()


def test_reorient_flip_only_ras_to_las_mirrors_affine_not_data() -> None:
    import nibabel as nib

    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[0, :, :] = 1.0
    data[-1, :, :] = 2.0
    layer = _FakeLayer(data, np.diag([1.0, 1.0, 1.0, 1.0]), axes="XYZ")

    previous, new_axes = reorient_layer_for_view(layer, "LAS")

    assert previous == "RAS"
    assert new_axes is None
    assert np.allclose(layer.data, data)
    assert nib.aff2axcodes(layer.affine) == ("L", "A", "S")
    assert layer.affine[0, 0] < 0
    assert layer.events.affine_calls == 1
    assert layer.events.data_calls == 0


def test_reorient_perm_ras_to_ars_reorders_data() -> None:
    import nibabel as nib

    data = np.arange(10 * 12 * 14, dtype=np.float32).reshape(10, 12, 14)
    layer = _FakeLayer(data.copy(), np.diag([1.0, 1.0, 1.0, 1.0]), axes="XYZ")

    previous, new_axes = reorient_layer_for_view(layer, "ARS")

    assert previous == "RAS"
    assert layer.data.shape == (12, 10, 14)
    assert nib.aff2axcodes(layer.affine) == ("A", "R", "S")
    assert new_axes == "YXZ"
    assert layer.events.data_calls == 1
