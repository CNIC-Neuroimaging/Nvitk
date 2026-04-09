from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nvitk.core import available_backends
from nvitk.core.exceptions import BackendUnavailableError
from nvitk.io import imread, imsave
from nvitk.types import Image


def _has_cupy_backend() -> bool:
    return "cupy" in available_backends()


def test_image_basics_and_metadata():
    arr = np.arange(24).reshape(2, 3, 4)
    img = Image(arr, metadata={"x_res": 0.7}, axes="XYZ", name="demo")

    assert img.backend == "numpy"
    assert img.shape == (2, 3, 4)
    assert img.ndim == 3
    assert img.metadata["axes"] == "XYZ"
    assert img.metadata["shape"] == (2, 3, 4)
    assert img.name == "demo"


def test_image_slice_returns_image_with_updated_axes():
    arr = np.arange(24).reshape(2, 3, 4)
    img = Image(arr, axes="XYZ")
    sl = img[:, :, 1]

    assert isinstance(sl, Image)
    assert sl.shape == (2, 3)
    assert sl.axes == "XY"


def test_image_scalar_index_returns_scalar():
    img = Image(np.arange(24).reshape(2, 3, 4), axes="XYZ")
    value = img[1, 2, 3]
    assert np.isscalar(value)


def test_image_arithmetic_keeps_image_type():
    img = Image(np.ones((2, 2), dtype=np.float32), axes="XY")
    out = img + 2

    assert isinstance(out, Image)
    assert np.allclose(np.asarray(out), np.full((2, 2), 3.0))
    assert out.axes == "XY"


def test_image_numpy_interop():
    img = Image(np.arange(6).reshape(2, 3), axes="XY")
    out = np.asarray(img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert np.array_equal(out, np.arange(6).reshape(2, 3))


def test_image_to_backend_numpy():
    img = Image(np.arange(5), axes="X")
    cpu_img = img.to_backend("numpy")
    assert isinstance(cpu_img, Image)
    assert cpu_img.backend == "numpy"
    assert np.array_equal(np.asarray(cpu_img), np.arange(5))


def test_image_to_cupy_behavior():
    img = Image(np.arange(5), axes="X")
    if _has_cupy_backend():
        gpu_img = img.to_cupy(strict=True)
        assert gpu_img.backend == "cupy"
        assert np.array_equal(np.asarray(gpu_img), np.arange(5))
    else:
        with pytest.raises(BackendUnavailableError):
            img.to_cupy(strict=True)


def test_image_copy_and_setitem():
    img = Image(np.zeros((2, 2), dtype=np.int32), axes="XY")
    other = img.copy()
    other[0, 1] = 7
    assert img[0, 1] == 0
    assert other[0, 1] == 7


def test_image_affine_property():
    img = Image(np.zeros((3, 3, 3), dtype=np.float32), axes="XYZ")
    aff = np.eye(4, dtype=float)
    aff[0, 3] = 12.5
    img.affine = aff

    assert img.affine is not None
    assert img.affine.shape == (4, 4)
    assert img.affine[0, 3] == 12.5


def test_image_spacing_property_syncs_resolution_fields():
    img = Image(np.zeros((3, 3, 3), dtype=np.float32), axes="XYZ")
    img.spacing = (0.7, 0.8, 1.5)

    assert img.spacing == (0.7, 0.8, 1.5)
    assert img.metadata["x_res"] == 0.7
    assert img.metadata["y_res"] == 0.8
    assert img.metadata["z_res"] == 1.5


def test_image_pet_and_dixon_tag_helpers():
    md = {
        "Modality": "PT",
        "RadiopharmaceuticalStartTime": "101500",
        "DixonWaterFraction": 0.75,
        "echo_time_ms": 2.3,
    }
    img = Image(np.zeros((4, 4, 4)), metadata=md, axes="XYZ")

    assert img.is_pet is True
    assert "RadiopharmaceuticalStartTime" in img.pet_tags
    assert "DixonWaterFraction" in img.dixon_tags
    assert "echo_time_ms" in img.dixon_tags


def test_image_metadata_helpers():
    img = Image(np.zeros((2, 2, 2)), metadata={"PatientID": "SUB-001"}, axes="XYZ")
    assert img.get_meta("PatientID") == "SUB-001"
    assert img.has_meta("PatientID")
    assert img.get_meta("Modality", "NA") == "NA"

    img.set_meta("Modality", "MR")
    assert img.get_meta("Modality") == "MR"
    picked = img.select_meta(["PatientID", "Modality", "Missing"])
    assert picked == {"PatientID": "SUB-001", "Modality": "MR"}


def test_imread_returns_image_with_metadata(tmp_path: Path):
    pytest.importorskip("nibabel")
    path = tmp_path / "sample.nii.gz"
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    imsave(path, data, metadata={"axes": "XYZ", "Modality": "MR", "PatientID": "SUB-01"})

    img = imread(path)
    assert isinstance(img, Image)
    assert img.axes == "XYZ"
    assert img.get_meta("Modality") == "MR"
    assert img.get_meta("PatientID") == "SUB-01"


def test_image_constructor_from_path(tmp_path: Path):
    pytest.importorskip("nibabel")
    path = tmp_path / "series.nii.gz"
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    imsave(path, data, metadata={"axes": "XYZ", "PatientID": "SUB-99"})

    img = Image(path)
    assert isinstance(img, Image)
    assert img.shape == (2, 2, 2)
    assert img.axes == "XYZ"
    assert img.get_meta("PatientID") == "SUB-99"
