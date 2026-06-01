"""Intensity projections along a volume axis."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Image


_PROJECTION_METHODS = frozenset({"max", "mean", "avg", "average", "median", "min", "std", "sum"})


def project_along_axis(volume: np.ndarray, axis: int, method: str) -> np.ndarray:
    """Collapse *volume* along *axis* using *method* (max, mean, median, min, std, sum)."""
    how = str(method).strip().lower()
    if how in ("mean", "avg", "average"):
        return np.mean(volume, axis=axis)
    if how == "max":
        return np.max(volume, axis=axis)
    if how == "median":
        return np.median(volume, axis=axis)
    if how == "min":
        return np.min(volume, axis=axis)
    if how == "std":
        return np.std(volume, axis=axis)
    if how == "sum":
        return np.sum(volume, axis=axis)
    raise ValueError(
        f"Unknown projection {method!r}; use one of max, mean, median, min, std, sum."
    )


def _project_affine(affine: np.ndarray | None, axis: int, out_ndim: int) -> np.ndarray | None:
    if affine is None:
        return None
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4) or out_ndim < 2:
        return None
    spatial_keep = [i for i in range(3) if i != axis]
    if len(spatial_keep) < out_ndim:
        return None
    new_aff = np.eye(4, dtype=float)
    for j, i in enumerate(spatial_keep[:out_ndim]):
        new_aff[:3, j] = aff[:3, i]
        new_aff[j, 3] = float(aff[i, 3])
    return new_aff


def _permute_axes_string(axes: str | None, axis: int) -> str | None:
    if axes is None or len(axes) <= axis:
        return None
    return axes[:axis] + axes[axis + 1 :]


def project_volume(
    image: Image,
    *,
    axis: int,
    method: str = "max",
) -> Image:
    """Return a new :class:`~nvitk.types.Image` projected along *axis*."""
    how = str(method).strip().lower()
    if how not in _PROJECTION_METHODS:
        raise ValueError(f"Unsupported projection method: {method!r}")

    data = to_numpy(image.data, copy=False)
    ndim = int(data.ndim)
    if ndim < 3:
        raise ValueError(f"Projection requires at least 3 dimensions, got ndim={ndim}.")
    if axis < 0 or axis >= ndim:
        raise ValueError(f"projection axis {axis} out of range for ndim={ndim}.")

    projected = project_along_axis(data, axis, how)
    out_ndim = int(projected.ndim)

    md = dict(image.metadata or {})
    aff = md.get("affine")
    if aff is not None:
        new_aff = _project_affine(np.asarray(aff, dtype=float), axis, out_ndim)
        if new_aff is not None:
            md["affine"] = new_aff
            for i, key in enumerate(("x_res", "y_res", "z_res")):
                if i < out_ndim:
                    md[key] = float(np.linalg.norm(new_aff[:3, i]))
            if out_ndim >= 3:
                md["spacing"] = tuple(md.get(f"{c}_res", 1.0) for c in ("x", "y", "z")[:out_ndim])

    new_axes = _permute_axes_string(image.axes, axis)
    if new_axes is not None and len(new_axes) == out_ndim:
        md["axes"] = new_axes
    else:
        md.pop("axes", None)
        new_axes = None

    md["shape"] = tuple(int(x) for x in projected.shape)
    suffix = how if how not in ("avg", "average") else "mean"
    name = image.name
    if name:
        name = f"{name}_proj{suffix}_ax{axis}"

    return Image(
        data=projected,
        metadata=md,
        axes=new_axes,
        name=name,
        orientation=image.orientation,
    )
