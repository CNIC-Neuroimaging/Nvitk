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


def napari_dim_order_3d(affine: np.ndarray | None, ndim: int = 3) -> tuple[int, ...]:
    """
    Dims order for 3D axial viewing in Napari (matches in-plane layout after Ctrl+T).

    ``axial_dim_order`` alone yields ``(S, A, R)``; Napari's default display matches
    ``(S, R, A)`` for typical RAS NIfTI — the same result as ``dims.transpose()`` once,
    without calling transpose on load.
    """
    sup = superior_voxel_axis(affine, ndim)
    rest = [i for i in range(ndim) if i != sup]
    if len(rest) >= 2:
        return (sup, rest[1], rest[0])
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


def _resolution_from_metadata(metadata: dict[str, Any] | None, axis_char: str) -> float | None:
    md = metadata or {}
    key = {
        "X": "x_res",
        "Y": "y_res",
        "Z": "z_res",
        "T": "t_res",
        "C": "t_res",
    }.get(axis_char.upper())
    if key is None:
        return None
    val = md.get(key)
    if val is None and axis_char.upper() in ("T", "C"):
        val = md.get("temporal_resolution")
    if val is None:
        return None
    return float(val)


def napari_scale_for_display(
    shape: tuple[int, ...],
    axes: str | None,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    """Diagonal voxel spacing per array axis (used for 4D+ instead of affine)."""
    from nvitk.io._common import default_nifti_axes

    ndim = len(shape)
    axes_u = (axes or default_nifti_axes(ndim)).upper()
    if len(axes_u) != ndim:
        axes_u = default_nifti_axes(ndim)
    return tuple(_resolution_from_metadata(metadata, ch) or 1.0 for ch in axes_u)


def napari_affine_for_display(
    affine: np.ndarray | None,
    shape: tuple[int, ...],
    axes: str | None,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """
    Affine for Napari display.

    For 4D+ volumes, decouple the temporal axis from spatial world coordinates so the
    time slider spans voxel indices (e.g. 15 phases), not a distorted world extent.
    """
    from nvitk.io._common import default_nifti_axes

    ndim = len(shape)
    if ndim <= 3:
        if affine is None:
            return None
        aff = to_numpy(affine).astype(float)
        return aff if aff.shape == (4, 4) else None

    axes_u = (axes or default_nifti_axes(ndim)).upper()
    if len(axes_u) != ndim:
        axes_u = default_nifti_axes(ndim)

    time_axes = [i for i, ch in enumerate(axes_u) if ch in ("T", "C")]
    if not time_axes:
        if affine is None:
            return None
        aff = to_numpy(affine).astype(float)
        return aff if aff.shape == (4, 4) else None

    t_ix = time_axes[0]
    md = metadata or {}
    t_scale = _resolution_from_metadata(md, axes_u[t_ix]) or 1.0

    if affine is not None:
        aff = to_numpy(affine).astype(float).copy()
        if aff.shape != (4, 4):
            aff = np.eye(4, dtype=float)
    else:
        aff = np.eye(4, dtype=float)
        for i, ch in enumerate(axes_u):
            if ch in "XYZ":
                s = _resolution_from_metadata(md, ch)
                if s is not None:
                    aff[i, i] = float(s)

    # Drop spatial↔temporal cross-terms (common 4DFlow bug: X extent mapped onto T slider).
    aff[0:3, t_ix] = 0.0
    aff[t_ix, 0:3] = 0.0
    aff[t_ix, t_ix] = float(t_scale)
    if t_ix == 3:
        aff[3, 0:3] = 0.0
        aff[3, 3] = float(t_scale)
    else:
        aff[3, t_ix] = 0.0
    return aff


def prepare_for_napari(
    data: np.ndarray,
    affine: np.ndarray | None,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    radiological: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, ...] | None]:
    """
    Prepare layer display transforms.

    4D+ uses ``scale`` only (no affine). Oblique 4DFlow affines couple spatial axes into
    the temporal world axis and Napari then reports ~60 bogus time steps instead of 15.
    The file affine is preserved in metadata as ``affine_source`` for export.
    """
    _ = radiological
    data = to_numpy(data)
    shape = tuple(int(x) for x in data.shape)
    if data.ndim > 3:
        return data, None, napari_scale_for_display(shape, axes, metadata)
    display_affine = napari_affine_for_display(affine, shape, axes, metadata)
    return data, display_affine, None


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


def _layer_display_scale(layer: Any, ndim: int) -> tuple[float, ...]:
    scale = getattr(layer, "scale", None)
    if scale is not None and len(scale) >= ndim:
        return tuple(float(scale[i]) for i in range(ndim))
    return (1.0,) * ndim


def _metadata_for_layer(layer: Any) -> dict[str, Any]:
    meta = getattr(layer, "metadata", None) or {}
    if not isinstance(meta, dict):
        return {}
    nv = meta.get("nvitk_metadata")
    return nv if isinstance(nv, dict) else meta


def ensure_4d_scale_only_layer(layer: Any) -> None:
    """
    Force 4D+ layers to diagonal scale (no oblique file affine).

    Oblique 4DFlow affines break the time slider and cause out-of-bounds indices.
    """
    data = getattr(layer, "data", None)
    if data is None:
        return
    ndim = int(data.ndim)
    if ndim <= 3:
        return
    axes_str = _axes_string_from_layer(layer)
    sc = napari_scale_for_display(tuple(int(x) for x in data.shape), axes_str, _metadata_for_layer(layer))
    layer.scale = sc
    layer.affine = np.eye(4, dtype=float)


def _synchronize_4d_dims(
    viewer: Any,
    layer: Any,
    *,
    axes_str: str | None,
    shape: tuple[int, ...],
) -> None:
    """Force Napari dims range/point from array shape (15 phases, not ~62)."""
    from napari.components.dims import RangeTuple

    ndim = len(shape)
    sc = _layer_display_scale(layer, ndim)
    ranges = []
    point: list[float] = []
    for i, n in enumerate(shape):
        step = sc[i]
        stop = float(max(0, n - 1)) * step
        ranges.append(RangeTuple(0.0, stop, step))
        point.append(0.0)
    if axes_str and len(axes_str) == ndim:
        for i, ch in enumerate(axes_str.upper()):
            if ch == "Z":
                point[i] = float(max(0, shape[i] - 1)) / 2.0 * sc[i]
    viewer.dims.range = tuple(ranges)
    viewer.dims.point = tuple(point)


def configure_viewer_for_layer(
    viewer: Any,
    layer: Any,
    *,
    radiological: bool = False,
    configure_dims: bool | None = None,
) -> None:
    """Axial-friendly dims for 3D; preserve 4D+ volumes (time/other axes untouched).

    Radiological L/R affine flips are off by default so raw images align with label
    masks opened from the same NIfTI grid (file affine, no extra X mirror).

    Set ``configure_dims=False`` when adding further layers (e.g. tool outputs) so the
    user's ndisplay, axis order, transpose, and camera are left unchanged.
    """
    if getattr(layer, "data", None) is None or layer.data.ndim < 3:
        return
    if configure_dims is None:
        configure_dims = len(viewer.layers) <= 1
    try:
        aff = getattr(layer, "affine", None)
        aff_arr = to_numpy(aff).astype(float) if aff is not None else None
        ndim = int(layer.data.ndim)
        axes_str = _axes_string_from_layer(layer)
        shape = layer.data.shape

        if ndim > 3:
            ensure_4d_scale_only_layer(layer)
        if not configure_dims:
            return

        if ndim == 3:
            order = napari_dim_order_3d(aff_arr, ndim)
            sup = order[0]
            viewer.dims.ndisplay = 2
            viewer.dims.order = order
            mid = int(shape[sup] // 2) if sup < len(shape) else 0
            point = [int(round(float(x))) for x in viewer.dims.point]
            if len(point) < ndim:
                point = list(point) + [0] * (ndim - len(point))
            point[sup] = mid
            viewer.dims.point = tuple(point[:ndim])
            if radiological and aff_arr is not None:
                lr = _lr_voxel_axis_for_radiological(aff_arr, ndim)
                if lr is not None and lr != sup:
                    _apply_voxel_axis_flip(layer, lr)
        else:
            order = napari_dim_order(axes_str, aff_arr, ndim)
            viewer.dims.ndisplay = 3
            viewer.dims.order = order
            if axes_str and len(axes_str) == ndim:
                viewer.dims.axis_labels = tuple(axes_str)
            _synchronize_4d_dims(viewer, layer, axes_str=axes_str, shape=shape)
        try:
            viewer.camera.angles = (0, 0, 0)
        except Exception:
            pass
    except Exception:
        pass
