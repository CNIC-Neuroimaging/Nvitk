"""Binary mask logical operators (union, intersection, subtract, …)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Image


def _as_bool(mask: Image | Any) -> np.ndarray:
    arr = to_numpy(mask.data if isinstance(mask, Image) else mask)
    return np.asarray(arr > 0, dtype=bool)


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    if isinstance(original, Image):
        return original.with_data(data)
    return data


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes must match; got {a.shape} vs {b.shape}.")


def mask_union(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Voxels where either mask is foreground (OR)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a | b).astype(np.uint8))


def mask_intersection(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Voxels where both masks are foreground (AND)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a & b).astype(np.uint8))


def mask_subtract(
    mask_a: Image | Any,
    mask_b: Image | Any,
    *,
    keep_overlap: bool = False,
) -> Image | Any:
    """Foreground in *mask_a* not in *mask_b* (A \\ B), or overlap only if *keep_overlap*."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    out = (a & b) if keep_overlap else (a & ~b)
    return _wrap_like(mask_a, out.astype(np.uint8))


def mask_xor(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Symmetric difference (A ⊕ B)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a ^ b).astype(np.uint8))


def mask_complement(
    mask: Image | Any,
    within: Image | Any | None = None,
) -> Image | Any:
    """Logical NOT of *mask*; optional *within* ROI limits the complement region."""
    a = _as_bool(mask)
    if within is None:
        return _wrap_like(mask, (~a).astype(np.uint8))
    w = _as_bool(within)
    _check_same_shape(a, w)
    return _wrap_like(mask, (w & ~a).astype(np.uint8))


__all__ = [
    "mask_union",
    "mask_intersection",
    "mask_subtract",
    "mask_xor",
    "mask_complement",
]
