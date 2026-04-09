from __future__ import annotations

import numpy as np
import pytest

from nvitk.core import available_backends
from nvitk.core.array import as_backend_array, ensure_same_backend, is_cupy_array, to_cupy, to_numpy
from nvitk.core.exceptions import BackendUnavailableError


def _has_cupy_backend() -> bool:
    return "cupy" in available_backends()


def test_tutorial_to_numpy():
    """Tutorial: normalize plain python data to numpy."""
    arr = to_numpy([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]


def test_tutorial_choose_backend_explicitly():
    """Tutorial: ask for numpy or gpu arrays from the same helper."""
    cpu = as_backend_array([1, 2, 3], backend="numpy")
    assert isinstance(cpu, np.ndarray)

    gpu = as_backend_array([1, 2, 3], backend="gpu")
    if _has_cupy_backend():
        assert is_cupy_array(gpu)
    else:
        assert isinstance(gpu, np.ndarray)


def test_tutorial_to_cupy_strict_mode():
    """Tutorial: strict=True means 'error instead of fallback'."""
    if _has_cupy_backend():
        arr = to_cupy(np.arange(3), strict=True)
        assert is_cupy_array(arr)
        assert np.array_equal(to_numpy(arr), np.arange(3))
    else:
        with pytest.raises(BackendUnavailableError):
            to_cupy(np.arange(3), strict=True)


def test_tutorial_convert_multiple_arrays_together():
    """Tutorial: keep multiple arrays on one backend with ensure_same_backend."""
    a, b = ensure_same_backend([1, 2], np.array([3, 4]), backend="numpy")
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
