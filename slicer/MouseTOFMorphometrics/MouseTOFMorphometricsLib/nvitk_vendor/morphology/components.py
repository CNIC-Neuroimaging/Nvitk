"""Connected-component labelling — NumPy/SciPy stand-in for ``nvitk.morphology.components``.

Hand-written (not synced). Only :func:`label_connected` is reachable from the
morphometrics pipeline (``polyline_graph``); it reproduces upstream exactly —
``ndi.label`` with a ``generate_binary_structure`` element and *connectivity*
clamped to ``[1, ndim]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def _label_structure(ndim: int, connectivity: int) -> Any:
    """Structuring element for ``ndi.label``, with *connectivity* clamped to ``[1, ndim]``."""
    conn = max(1, min(int(connectivity), ndim))
    return ndi.generate_binary_structure(ndim, conn)


def label_connected(mask: Any, *, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Label connected components of a binary *mask*; returns ``(labeled, num_features)``."""
    arr = np.asarray(getattr(mask, "data", mask)).astype(bool, copy=False)
    labeled, num = ndi.label(arr, structure=_label_structure(int(arr.ndim), connectivity))
    return labeled, int(num)


def remove_small_components(mask: Any, *, min_size: int, connectivity: int = 1) -> np.ndarray:
    """Drop connected components with fewer than *min_size* voxels."""
    arr = np.asarray(getattr(mask, "data", mask)).astype(bool, copy=False)
    if int(np.count_nonzero(arr)) == 0:
        return arr
    labeled, num = label_connected(arr, connectivity=connectivity)
    if num == 0:
        return np.zeros_like(arr, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep = np.array([i for i in range(1, len(counts)) if int(counts[i]) >= int(min_size)], dtype=labeled.dtype)
    if keep.size == 0:
        return np.zeros_like(arr, dtype=bool)
    return np.isin(labeled, keep)


def largest_component(mask: Any, *, connectivity: int = 1) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    arr = np.asarray(getattr(mask, "data", mask)).astype(bool, copy=False)
    labeled, num = label_connected(arr, connectivity=connectivity)
    if num <= 1:
        return arr
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


__all__ = ["label_connected", "largest_component", "remove_small_components"]
