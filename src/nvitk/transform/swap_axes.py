"""Swap or permute axes of an image volume."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _host_np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def _permute_axes_string(axes: str | None, order: Sequence[int]) -> str | None:
    if axes is None or len(axes) != len(order):
        return None
    return "".join(axes[i] for i in order)


def _permute_affine_columns(
    affine: _host_np.ndarray | None,
    order: Sequence[int],
) -> _host_np.ndarray | None:
    """Permute voxel-axis columns of a 4×4 affine for a spatial axis reorder."""
    if affine is None:
        return None
    A = _host_np.asarray(affine, dtype=float).copy()
    if A.shape != (4, 4):
        return A
    spatial = [i for i in order if 0 <= i < 3]
    if len(spatial) == 3 and sorted(spatial) == [0, 1, 2]:
        new = _host_np.eye(4, dtype=float)
        for new_i, old_i in enumerate(spatial):
            new[:3, new_i] = A[:3, old_i]
        new[:3, 3] = A[:3, 3]
        return new
    if len(order) == 2 and all(0 <= i < 3 for i in order):
        i, j = int(order[0]), int(order[1])
        A[:, [i, j]] = A[:, [j, i]]
        A[3, :] = (0.0, 0.0, 0.0, 1.0)
        return A
    return A


def permute_axes(
    image: Image | Any,
    order: Sequence[int],
) -> Image | Any:
    """
    Reorder array axes to *order*.

    Parameters
    ----------
    image
        :class:`~nvitk.types.Image` or ndarray-like.
    order
        New axis order. Must be a permutation of ``0..ndim-1``, or of
        ``0..k-1`` to permute only the leading *k* axes (e.g. spatial of 4D).
    """
    is_image = isinstance(image, Image)
    data = image.data if is_image else image
    arr = as_backend_array(data)
    ndim = int(arr.ndim)
    ord_t = tuple(int(i) for i in order)
    if len(ord_t) != len(set(ord_t)):
        raise ValueError(f"order must not contain duplicates; got {ord_t}")
    if any(i < 0 or i >= ndim for i in ord_t):
        raise ValueError(f"order indices must be in [0, {ndim}); got {ord_t}")
    if len(ord_t) != ndim:
        if sorted(ord_t) != list(range(len(ord_t))):
            raise ValueError(
                f"partial order must be a permutation of 0..{len(ord_t) - 1}; got {ord_t}"
            )
        ord_t = tuple(list(ord_t) + list(range(len(ord_t), ndim)))

    if ord_t == tuple(range(ndim)):
        return image if is_image else arr

    transposed = np.transpose(arr, ord_t)
    if not is_image:
        return transposed

    new_axes = None
    if image.axes is not None and len(image.axes) == ndim:
        new_axes = _permute_axes_string(image.axes, ord_t)

    out = image.with_data(transposed, axes=new_axes)

    spacing = image.spacing
    if spacing is not None and len(spacing) >= ndim:
        sp = list(spacing)
        out.spacing = tuple(sp[i] for i in ord_t) + tuple(sp[ndim:])
    elif spacing is not None and len(spacing) >= 3 and set(ord_t[:3]) == {0, 1, 2}:
        sp = list(spacing)
        out.spacing = tuple(sp[i] for i in ord_t[:3]) + tuple(sp[3:])

    if image.affine is not None and ndim >= 2:
        if ndim >= 3 and set(ord_t[:3]) == {0, 1, 2}:
            out.metadata["affine"] = _permute_affine_columns(to_numpy(image.affine), ord_t[:3])
        else:
            out.metadata["affine"] = _permute_affine_columns(to_numpy(image.affine), ord_t[:2])
        out.metadata.pop("orientation", None)
    return out


def swap_axes(
    image: Image | Any,
    axis0: int,
    axis1: int,
) -> Image | Any:
    """
    Swap two axes of a 2D/3D/4D image.

    Parameters
    ----------
    image
        :class:`~nvitk.types.Image` or ndarray-like.
    axis0, axis1
        Axis indices to exchange (no-op if equal).
    """
    is_image = isinstance(image, Image)
    data = image.data if is_image else image
    arr = as_backend_array(data)
    ndim = int(arr.ndim)
    a0, a1 = int(axis0), int(axis1)
    if a0 == a1:
        return image if is_image else arr
    for a in (a0, a1):
        if a < 0 or a >= ndim:
            raise ValueError(f"axis {a} out of range for ndim={ndim}")
    order = list(range(ndim))
    order[a0], order[a1] = order[a1], order[a0]
    return permute_axes(image, order)


__all__ = ["permute_axes", "swap_axes"]
