"""Rotate 2D/3D image arrays around a spatial axis."""

from __future__ import annotations

from typing import Any

import numpy as _host_np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def _plane_axes(axis: int) -> tuple[int, int]:
    """Return the two axes spanning the plane orthogonal to *axis*."""
    if axis == 0:
        return (1, 2)
    if axis == 1:
        return (0, 2)
    if axis == 2:
        return (0, 1)
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")


def _rotation_matrix_3d(axis: int, degrees: float) -> _host_np.ndarray:
    th = float(_host_np.radians(degrees))
    c = float(_host_np.cos(th))
    s = float(_host_np.sin(th))
    if axis == 0:
        return _host_np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis == 1:
        return _host_np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis == 2:
        return _host_np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")


def _affine_after_center_rotation(
    affine: _host_np.ndarray,
    shape: tuple[int, ...],
    axis: int,
    degrees: float,
) -> _host_np.ndarray:
    """Update voxel→world affine after a reshape=False rotation about the array center."""
    A = _host_np.asarray(affine, dtype=float).copy()
    if A.shape != (4, 4) or len(shape) < 3:
        return A
    center = (_host_np.asarray(shape[:3], dtype=float) - 1.0) * 0.5
    # Output voxel x_new samples input at R(-θ)(x_new − c) + c.
    Rm = _rotation_matrix_3d(axis, -float(degrees))
    T = _host_np.eye(4, dtype=float)
    T[:3, :3] = Rm
    T[:3, 3] = (_host_np.eye(3, dtype=float) - Rm) @ center
    return A @ T


def rotate_volume(
    image: Image | Any,
    angle_degrees: float,
    *,
    axis: int = 2,
    order: int = 1,
    reshape: bool = False,
    mode: str = "constant",
    cval: float = 0.0,
) -> Image | Any:
    """
    Rotate a 2D or 3D image by *angle_degrees* around *axis*.

    Parameters
    ----------
    image
        :class:`~nvitk.types.Image` or ndarray-like volume.
    angle_degrees
        Counter-clockwise rotation (degrees) in the plane orthogonal to *axis*
        (``scipy.ndimage.rotate`` convention).
    axis
        Axis to rotate around for 3D data (``0``, ``1``, or ``2``; default Z).
        Ignored for 2D (always in-plane).
    order
        Spline interpolation order. Use ``0`` for label masks.
    reshape
        If ``True``, expand the canvas to fit the rotated content (affine is
        left unchanged). If ``False`` (default), keep the original shape and
        update the affine when present.
    mode, cval
        Boundary fill for ``scipy.ndimage.rotate``.
    """
    is_image = isinstance(image, Image)
    data = image.data if is_image else image
    arr = as_backend_array(data)
    ndim = int(arr.ndim)
    if ndim not in (2, 3):
        raise ValueError(f"rotate_volume expects 2D or 3D data; got ndim={ndim}")

    angle = float(angle_degrees)
    if abs(angle) < 1e-12:
        return image if is_image else arr

    if ndim == 2:
        plane = (0, 1)
        rot_axis = 2
    else:
        rot_axis = int(axis)
        plane = _plane_axes(rot_axis)

    rotated = ndi.rotate(
        arr,
        angle,
        axes=plane,
        reshape=bool(reshape),
        order=int(order),
        mode=str(mode),
        cval=float(cval),
        prefilter=int(order) > 0,
    )

    if not is_image:
        return rotated

    out = image.with_data(rotated)
    if reshape:
        # Canvas size changed; keep spacing but drop a stale orientation claim.
        meta = dict(out.metadata or {})
        meta.pop("orientation", None)
        out.metadata = meta
        return out

    affine = image.affine
    if affine is not None and ndim == 3:
        new_aff = _affine_after_center_rotation(
            to_numpy(affine), tuple(int(s) for s in arr.shape), rot_axis, angle
        )
        out.metadata["affine"] = new_aff
        # Orientation codes may no longer match after an arbitrary rotation.
        out.metadata.pop("orientation", None)
    return out


__all__ = ["rotate_volume"]
