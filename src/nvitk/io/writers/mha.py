"""Write arrays as MetaImage via SimpleITK (spacing, origin, direction from metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError

from .._common import itk_geometry_from_affine, reorder_axes

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

    # SimpleITK's GetImageFromArray reads a 3D array as (z, y, x) and nothing else, so the array
    # must arrive in that order. The previous version derived the target from the metadata and so
    # left an "XYZ" array -- everything read from NIfTI -- untransposed, which wrote the volume
    # with its axes reversed while the spacing stayed in x, y, z order. The result opened
    # squashed, with the three views showing the wrong planes.
    axes_prev = metadata.get("axes")
    if arr.ndim == 3:
        target_axes = axes or "ZYX"
    else:
        target_axes = axes or axes_prev or "".join(f"D{i}" for i in range(arr.ndim))
    if axes_prev and axes_prev != target_axes:
        arr = reorder_axes(arr, axes_prev, target_axes)
    metadata["axes"] = target_axes

    img = sitk.GetImageFromArray(arr)

    # ITK geometry, in ITK's own (x, y, z) / LPS convention. Explicit keys win: a volume read
    # from MetaImage or NRRD already carries them that way. Otherwise they are derived from the
    # affine, which for anything read from NIfTI is RAS and needs the flip -- skipping it mirrors
    # left against right without touching a voxel.
    spacing = metadata.get("spacing")
    origin = metadata.get("origin")
    direction = metadata.get("direction")
    affine = metadata.get("affine")
    if affine is not None and not (spacing and origin and direction):
        derived = itk_geometry_from_affine(
            affine, axes_prev or "XYZ", world=str(metadata.get("world", "ras")),
        )
        spacing = spacing or derived[0]
        origin = origin or derived[1]
        direction = direction or derived[2]

    if spacing is None:
        spacing = tuple(
            v for v in (metadata.get("x_res"), metadata.get("y_res"), metadata.get("z_res")) if v is not None
        )
    if spacing:
        img.SetSpacing(tuple(float(v) for v in spacing))
    if origin:
        img.SetOrigin(tuple(float(v) for v in origin))
    if direction:
        img.SetDirection(tuple(float(v) for v in direction))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out), **kwargs)
