"""GUI-specific backend helpers: layer arrays and Napari-safe outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import is_cupy_array, set_global_backend, using


def setup_tool_backend(use_gpu: bool) -> str:
    """Set process default backend for a tool run. Returns resolved name (``numpy`` or ``cupy``)."""
    if use_gpu:
        return set_global_backend("cupy", allow_fallback=True)
    return set_global_backend("cpu", allow_fallback=False)


def layer_data_for_tool(data: Any, *, use_gpu: bool) -> Any:
    """Coerce layer data for tool input: NumPy on CPU, backend array on GPU."""
    if use_gpu:
        from nvitk.core.array import as_backend_array

        return as_backend_array(data)
    return to_numpy(data)


def napari_array(out: Any) -> np.ndarray:
    """Always return a NumPy array suitable for Napari (no implicit CuPy conversion)."""
    # Only unwrap nvitk Image wrappers — CuPy/NumPy arrays also have a ``.data`` memptr.
    from nvitk.types import Image

    if isinstance(out, Image):
        out = out.data
    if isinstance(out, tuple) and out:
        out = out[0]
    if is_cupy_array(out):
        return to_numpy(out)
    try:
        arr = np.asarray(out)
    except TypeError:
        return to_numpy(out)
    if arr.dtype == object:
        return to_numpy(out)
    return arr


def run_with_backend(use_gpu: bool):
    """Context manager: ``using('cupy')`` or ``using('numpy')`` for tool body."""
    name = "cupy" if use_gpu else "numpy"
    return using(name, allow_fallback=use_gpu)
