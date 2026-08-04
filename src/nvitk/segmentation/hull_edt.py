"""Convex hull and Euclidean distance transform helpers for binary masks."""

from __future__ import annotations

from typing import Any, Tuple

from nvitk.core import setup
from nvitk.core.backend import using
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.types import Image

setup(globals())


def _as_bool_mask(mask: Image | Any) -> np.ndarray:
    """Backend boolean foreground (``> 0``) from an :class:`Image` or array."""
    arr = mask.data if isinstance(mask, Image) else mask
    return as_backend_array(arr > 0).astype(bool)


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    """Re-wrap *data* as an :class:`Image` when *original* was one; else return *data*."""
    if isinstance(original, Image):
        return original.with_data(data)
    return data


def convex_hull_slicewise(mask: Image | Any, *, axis: int = -1) -> Image | Any:
    """
    Apply a 2D convex hull independently on each slice orthogonal to *axis*.

    Non-zero voxels in each slice are filled to their planar convex hull.
    """
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise ImportError(
            "convex_hull_slicewise requires scikit-image (skimage.morphology.convex_hull_image)."
        ) from exc

    m = _as_bool_mask(mask).astype(np.uint8)
    out = m.copy()
    axis_n = int(axis) % m.ndim
    for i in range(m.shape[axis_n]):
        idx: list[slice | int] = [slice(None)] * m.ndim
        idx[axis_n] = i
        sl = out[tuple(idx)]
        if sl.any():
            _aux = convex_hull_image(to_numpy(sl))
            out[tuple(idx)] = as_backend_array(_aux)
    return _wrap_like(mask, as_backend_array(out).astype(np.uint8))


def convex_hull_3d(mask: Image | Any) -> Image | Any:
    """Fill the 3D convex hull of all foreground voxels."""
    m = _as_bool_mask(mask)
    coords = np.argwhere(m)
    if coords.shape[0] < 4:
        raw = mask.data if isinstance(mask, Image) else mask
        return _wrap_like(mask, np.zeros_like(raw, dtype=np.uint8))

    with using("cpu"):
        coords = to_numpy(coords)
        hull = scipy.spatial.ConvexHull(coords)
    
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        grid = np.mgrid[
            mins[0] : maxs[0],
            mins[1] : maxs[1],
            mins[2] : maxs[2],
        ]
        pts = np.stack([g.ravel() for g in grid], axis=1).astype(np.float64)
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]
        # hull.equations rows are (normal_vector, offset) for halfspaces: A x + b <= 0.
        # Broadcast b across all points: (n_planes, 1) + (n_planes, n_points).
        inside = np.all((A @ pts.T) + b[:, None] <= 1e-6, axis=0)
        sub = inside.reshape(grid[0].shape).astype(np.uint8)
        out = np.zeros(m.shape, dtype=np.uint8)
        out[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]] = sub
    return _wrap_like(mask, as_backend_array(out))


def distance_transform(
    mask: Image | Any,
    *,
    spacing: Tuple[float, float, float] | None = None,
    radius_mm: float | None = None,
) -> Image | Any:
    """
    Euclidean distance transform of the background (outside foreground).

    When *radius_mm* is set, return a binary tube (distance ≤ *radius_mm*).
    Otherwise return float32 distance (mm if *spacing* is provided).
    """
    m = _as_bool_mask(mask)
    inv = ~m
    sp = spacing
    if sp is None and isinstance(mask, Image) and mask.spacing is not None:
        sp = tuple(float(x) for x in mask.spacing[:3])
    if sp is not None and len(sp) >= 3:
        dist = ndi.distance_transform_edt(inv, sampling=sp[:3])
    else:
        dist = ndi.distance_transform_edt(inv)

    if radius_mm is not None and float(radius_mm) > 0:
        out = (dist <= float(radius_mm)).astype(np.uint8)
    else:
        out = as_backend_array(dist).astype(np.float32)
    return _wrap_like(mask, out)


__all__ = [
    "convex_hull_slicewise",
    "convex_hull_3d",
    "distance_transform",
]
