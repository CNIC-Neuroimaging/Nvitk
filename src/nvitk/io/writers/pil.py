"""Save 2D or ``YXC`` arrays with Pillow (format from file extension)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from .._common import reorder_axes


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def write_pil(
    path: str,
    data: Any,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Write raster image; floats are scaled to uint8 when needed."""
    try:
        from PIL import Image as PILImage
    except Exception as exc:
        raise BackendUnavailableError('Pillow is not installed. Please install it with "pip install pillow".') from exc

    metadata = dict(metadata or {})
    arr = to_numpy(data)

    axes_prev = metadata.get("axes")
    if axes and axes_prev and axes_prev != axes:
        arr = reorder_axes(arr, axes_prev, axes)
        metadata["axes"] = axes

    if arr.ndim not in (2, 3):
        raise ValidationError(f"PIL writer supports 2D or 3D arrays, got ndim={arr.ndim}")
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4):
        raise ValidationError(f"Expected last channel dim to be 1, 3 or 4 for PIL output, got {arr.shape[-1]}")

    arr_u8 = _to_uint8(arr)
    img = PILImage.fromarray(arr_u8.squeeze() if arr_u8.ndim == 3 and arr_u8.shape[-1] == 1 else arr_u8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out), **kwargs)
