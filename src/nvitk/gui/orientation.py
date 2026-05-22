"""Napari display: file affine + axial viewing along Superior axis."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
from nvitk.core.array import to_numpy


@contextmanager
def suppress_nonorthogonal_slice_warning() -> Iterator[None]:
    """Hide Napari's oblique-acquisition slice warning (display is still approximate)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Non-orthogonal slicing.*",
            category=UserWarning,
        )
        yield


def superior_voxel_axis(affine: np.ndarray | None, ndim: int = 3) -> int:
    """Array axis index whose +direction is closest to patient Superior."""
    if affine is None or ndim < 3:
        return min(2, ndim - 1)
    try:
        import nibabel as nib
    except Exception:
        return min(2, ndim - 1)

    aff = to_numpy(affine).astype(float)
    try:
        codes = nib.orientations.aff2axcodes(aff[:3, :3])
    except Exception:
        return min(2, ndim - 1)

    for i, code in enumerate(codes[:3]):
        if str(code).upper() == "S":
            return i
    for i, code in enumerate(codes[:3]):
        if str(code).upper() == "I":
            return i
    return min(2, ndim - 1)


def axial_dim_order(affine: np.ndarray | None, ndim: int = 3) -> tuple[int, ...]:
    """Dims order with Superior first so 2D mode steps axially through the volume."""
    sup = superior_voxel_axis(affine, ndim)
    rest = [i for i in range(ndim) if i != sup]
    return (sup, *rest)


def _axes_string_from_layer(layer: Any) -> str | None:
    labels = getattr(layer, "axis_labels", None)
    if labels is not None and len(labels) == int(getattr(layer.data, "ndim", 0)):
        return "".join(str(l) for l in labels)
    meta = getattr(layer, "metadata", None) or {}
    nv = meta.get("nvitk_metadata") if isinstance(meta, dict) else None
    if isinstance(nv, dict) and nv.get("axes"):
        return str(nv["axes"])
    if isinstance(meta, dict) and meta.get("axes"):
        return str(meta["axes"])
    return None


def napari_dim_order(
    axes: str | None,
    affine: np.ndarray | None,
    ndim: int,
) -> tuple[int, ...]:
    """
    Napari ``dims.order``: non-displayed axes first (time), then spatial axes to render.

    For ``XYZT`` data this yields ``(T, X, Y, Z)`` so the time slider drives axis ``T`` and
    the 3D view uses array ``X``, ``Y``, ``Z`` (no superior-first shuffle on 4D+).
    """
    if ndim <= 3:
        return axial_dim_order(affine, ndim)

    from nvitk.io._common import default_nifti_axes

    ax = (axes or default_nifti_axes(ndim)).upper()
    if len(ax) != ndim:
        return axial_dim_order(affine, ndim)

    time_axes = [i for i, ch in enumerate(ax) if ch in ("T", "C")]
    spatial_axes = [i for i, ch in enumerate(ax) if ch in "XYZ"]
    if not time_axes:
        return axial_dim_order(affine, ndim)
    if len(spatial_axes) < 3:
        return tuple(time_axes) + axial_dim_order(affine, len(spatial_axes) or ndim)

    return tuple(time_axes + spatial_axes)


def prepare_for_napari(
    data: np.ndarray,
    affine: np.ndarray | None,
    *,
    radiological: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float, float] | None]:
    """Pass through array and file affine (no voxel reorientation)."""
    _ = radiological
    if affine is not None:
        aff = to_numpy(affine).astype(float)
        if aff.shape != (4, 4):
            affine = None
        else:
            affine = aff
    return to_numpy(data), affine, None


def _lr_voxel_axis_for_radiological(affine: np.ndarray | None, ndim: int) -> int | None:
    """Voxel axis to flip for radiological in-plane L/R (patient R on screen left)."""
    if affine is None or ndim < 3:
        return None
    try:
        import nibabel as nib
    except Exception:
        return None
    try:
        codes = nib.orientations.aff2axcodes(to_numpy(affine).astype(float)[:3, :3])
    except Exception:
        return None
    for i, code in enumerate(codes[:ndim]):
        if str(code).upper() == "R":
            return i
    for i, code in enumerate(codes[:ndim]):
        if str(code).upper() == "L":
            return i
    return None


def _apply_voxel_axis_flip(layer: Any, axis: int) -> None:
    """Flip one voxel axis via affine (keeps world coordinates consistent)."""
    aff = getattr(layer, "affine", None)
    if aff is None:
        return
    aff = to_numpy(aff).astype(float).copy()
    if aff.shape != (4, 4):
        return
    shape = getattr(layer, "data", None)
    if shape is None:
        return
    n = int(shape.shape[axis]) if axis < len(shape.shape) else 0
    if n <= 1:
        return
    flip = np.eye(4)
    flip[axis, axis] = -1.0
    flip[axis, 3] = float(n - 1)
    layer.affine = aff @ flip


def configure_viewer_for_layer(
    viewer: Any,
    layer: Any,
    *,
    radiological: bool = True,
) -> None:
    """Axial-friendly dims for 3D; preserve 4D+ volumes (time/other axes untouched)."""
    if getattr(layer, "data", None) is None or layer.data.ndim < 3:
        return
    try:
        aff = getattr(layer, "affine", None)
        aff_arr = to_numpy(aff).astype(float) if aff is not None else None
        ndim = int(layer.data.ndim)
        axes_str = _axes_string_from_layer(layer)
        order = napari_dim_order(axes_str, aff_arr, ndim)
        shape = layer.data.shape

        if ndim == 3:
            sup = order[0]
            viewer.dims.ndisplay = 2
            viewer.dims.order = order
            mid = int(shape[sup] // 2) if sup < len(shape) else 0
            point = [int(round(float(x))) for x in viewer.dims.point]
            if len(point) < ndim:
                point = list(point) + [0] * (ndim - len(point))
            point[0] = mid
            viewer.dims.point = tuple(point[:ndim])
            if radiological and aff_arr is not None:
                lr = _lr_voxel_axis_for_radiological(aff_arr, ndim)
                if lr is not None and lr != sup:
                    _apply_voxel_axis_flip(layer, lr)
        else:
            viewer.dims.ndisplay = 3
            viewer.dims.order = order
            if axes_str and len(axes_str) == ndim:
                viewer.dims.axis_labels = tuple(axes_str)
            point = [0] * ndim
            if axes_str and len(axes_str) == ndim:
                for i, ch in enumerate(axes_str.upper()):
                    if ch == "Z" and i < len(shape):
                        point[i] = int(shape[i] // 2)
            viewer.dims.point = tuple(point[:ndim])
        try:
            viewer.camera.angles = (0, 0, 0)
        except Exception:
            pass
    except Exception:
        pass
