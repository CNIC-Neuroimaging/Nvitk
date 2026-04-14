"""Write arrays as MetaImage via SimpleITK (spacing, origin, direction from metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


def write_mha(
    path: str,
    data: Any,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Save *data* as ``.mha`` / ``.mhd``; expects ``ZYX`` (or reordered) for 3D volumes."""
    if sitk is None:
        raise BackendUnavailableError('SimpleITK is not installed. Please install it with "pip install SimpleITK".')

    metadata = dict(metadata or {})
    arr = to_numpy(data)

    axes_prev = metadata.get("axes")
    # For SITK GetImageFromArray, expected order for 3D is ZYX.
    target_axes = axes or axes_prev or ("ZYX" if arr.ndim == 3 else "".join(f"D{i}" for i in range(arr.ndim)))
    if axes_prev and axes_prev != target_axes:
        arr = reorder_axes(arr, axes_prev, target_axes)
        metadata["axes"] = target_axes

    img = sitk.GetImageFromArray(arr)

    spacing = metadata.get("spacing")
    if spacing is None:
        spacing = tuple(
            v for v in (metadata.get("x_res"), metadata.get("y_res"), metadata.get("z_res")) if v is not None
        )
    if spacing:
        img.SetSpacing(tuple(float(v) for v in spacing))

    origin = metadata.get("origin")
    if origin:
        img.SetOrigin(tuple(float(v) for v in origin))

    direction = metadata.get("direction")
    if direction:
        img.SetDirection(tuple(float(v) for v in direction))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out), **kwargs)
