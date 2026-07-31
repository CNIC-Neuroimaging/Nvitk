"""Binary mask logical operators and mask→image intensity apply."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.types import Image


def _as_bool(mask: Image | Any) -> np.ndarray:
    arr = to_numpy(mask.data if isinstance(mask, Image) else mask)
    return as_backend_array(arr > 0).astype(bool)


def _foreground_mask(
    mask: Image | Any,
    *,
    label_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Boolean foreground from a mask / label map.

    Empty *label_ids* → any nonzero voxel. Otherwise only listed label ids.
    """
    arr = to_numpy(mask.data if isinstance(mask, Image) else mask)
    if label_ids:
        ids = np.asarray([int(x) for x in label_ids], dtype=arr.dtype)
        return as_backend_array(np.isin(arr, ids))
    return as_backend_array(arr != 0)


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


def apply_mask_to_image(
    image: Image | Any,
    mask: Image | Any,
    *,
    mode: str = "keep_inside",
    fill_value: float = 0.0,
    label_ids: Sequence[int] | None = None,
) -> Image | Any:
    """Apply a segmentation mask to an intensity image.

    Parameters
    ----------
    image
        Intensity volume (active layer).
    mask
        Segmentation / binary mask on the same grid as *image*.
    mode
        ``keep_inside`` — retain voxels where the mask is foreground; fill the rest.
        ``keep_outside`` — retain voxels where the mask is background; fill the rest.
    fill_value
        Value written into discarded voxels (default ``0``).
    label_ids
        Optional label ids treated as foreground; ``None`` / empty → any nonzero.
    """
    vol = to_numpy(image.data if isinstance(image, Image) else image)
    fg = to_numpy(_foreground_mask(mask, label_ids=label_ids))
    if fg.shape != vol.shape:
        raise ValueError(
            f"Mask shape {fg.shape} must match image shape {vol.shape}."
        )
    mode_key = str(mode or "keep_inside").strip().lower().replace("-", "_")
    if mode_key in ("keep_inside", "inside", "in"):
        keep = fg
    elif mode_key in ("keep_outside", "outside", "out"):
        keep = ~fg
    else:
        raise ValueError(
            f"Unknown mask apply mode {mode!r}; use 'keep_inside' or 'keep_outside'."
        )
    out = np.array(vol, copy=True, dtype=np.result_type(vol.dtype, np.float32))
    fill = np.asarray(fill_value, dtype=out.dtype)
    out[~keep] = fill
    # Preserve original dtype when fill fits (e.g. integer volumes + fill 0).
    if np.can_cast(fill, vol.dtype, casting="safe") or (
        np.issubdtype(vol.dtype, np.integer) and float(fill_value) == int(fill_value)
    ):
        try:
            out = out.astype(vol.dtype, copy=False)
        except (TypeError, ValueError):
            pass
    return _wrap_like(image, as_backend_array(out))


__all__ = [
    "apply_mask_to_image",
    "mask_union",
    "mask_intersection",
    "mask_subtract",
    "mask_xor",
    "mask_complement",
]
