"""Internal helpers shared by the measure primitives."""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def resolve_array(img: Image | Any) -> Any:
    """Return the voxel array for *img*, or *img* itself when already an array."""
    return img.data if isinstance(img, Image) else img


def resolve_spacing(img: Image | Any, spacing: tuple[float, ...] | None) -> tuple[float, ...]:
    """
    Return a spacing tuple for *img*.

    Resolution order:
    1. explicit *spacing* argument;
    2. ``img.spacing`` when *img* is an :class:`Image`;
    3. :class:`ValueError`.
    """
    if spacing is not None:
        return tuple(float(s) for s in spacing)
    if isinstance(img, Image) and img.spacing is not None:
        return tuple(float(s) for s in img.spacing)
    raise ValueError("Spacing is required (pass explicitly or via an Image with metadata spacing).")


def bool_mask(mask: Image | Any) -> Any:
    """Boolean-cast *mask* preserving backend (NumPy/CuPy)."""
    arr = resolve_array(mask)
    if hasattr(arr, "astype"):
        return arr.astype(bool)
    return as_backend_array(arr).astype(bool)


def ensure_same_shape(a: Image | Any, b: Image | Any) -> None:
    sa = resolve_array(a).shape
    sb = resolve_array(b).shape
    if tuple(sa) != tuple(sb):
        raise ValueError(f"Shape mismatch: {sa} vs {sb}")
