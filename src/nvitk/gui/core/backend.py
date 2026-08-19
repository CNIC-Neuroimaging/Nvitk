"""GUI helpers aligned with the global nvitk backend (:mod:`nvitk.core.backend`).

The active backend is set by ``nvitk-gui --backend cpu|gpu``, the dock GPU toggle
(:mod:`nvitk.gui.tools.gpu_toggle`), or :func:`~nvitk.core.backend.set_default_backend`.
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


def napari_label_array(out: Any) -> np.ndarray:
    """
    NumPy array for Napari that **keeps its integer dtype**, for label-map results.

    :func:`napari_array` casts to ``float64``, which is right for intensities and wrong for a label
    map: the ids survive the round trip numerically, but the layer can no longer be added as a
    Napari ``Labels`` layer without a second cast, and every downstream ``unique`` comes back as
    floats. Tools that return a parcellation use this instead.

    A non-integer result is left alone rather than rounded — silently quantising a float map into
    label ids would invent a parcellation that the operation never produced.
    """
    from nvitk.types import Image

    if isinstance(out, Image):
        out = out.data
    if isinstance(out, tuple) and out:
        out = out[0]
    arr = to_numpy(out)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return arr
    return arr.astype(np.float64)


def run_with_backend():
    """Context manager scoped to the current global backend."""
    return using(get_global_backend(), allow_fallback=True)
