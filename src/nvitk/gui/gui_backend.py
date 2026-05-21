"""GUI helpers aligned with the global nvitk backend (:mod:`nvitk.core.backend`).

The active backend is set by ``nvitk-gui --backend cpu|gpu``, the dock GPU toggle
(:mod:`nvitk.gui.gpu_toggle`), or :func:`~nvitk.core.backend.set_default_backend`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import get_global_backend, is_cupy_array, using


def gpu_enabled() -> bool:
    """True when the global backend is CuPy."""
    return get_global_backend() == "cupy"


def layer_data_for_tool(data: Any) -> Any:
    """Coerce layer data for tool input using the global backend."""
    if gpu_enabled():
        from nvitk.core.array import as_backend_array

        return as_backend_array(data)
    return to_numpy(data)


def napari_array(out: Any) -> np.ndarray:
    """Always return a NumPy array suitable for Napari (no implicit CuPy conversion)."""
    from nvitk.types import Image

    if isinstance(out, Image):
        out = out.data
    if isinstance(out, tuple) and out:
        out = out[0]
    if is_cupy_array(out):
        return to_numpy(out)
    try:
        arr = to_numpy(out)
    except TypeError:
        return to_numpy(out)
    return arr.astype(np.float64)


def run_with_backend():
    """Context manager scoped to the current global backend."""
    return using(get_global_backend(), allow_fallback=True)
