"""
Split bilateral label masks into left/right hemispheres.

Two strategies are provided:

- :func:`split_lr_by_cc` — connected-components driven: keeps the *n* largest
  CCs in *mask* and assigns Left/Right from their world-space x centroids. This
  is what ct-pet-v5 and dixon-v5 use for the `deltoid` TotalSegmentator label
  which ships bilaterally without native L/R classes.
- :func:`split_lr_by_midline` — geometric fallback: split on a sagittal plane
  (specified by voxel x or auto-detected from the affine origin).

Both functions work for images in any voxel orientation: the sign of the
affine's first column determines which voxel-space side maps to anatomical
"left" in world coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as _host_np

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def _centroid_world_x(coords_voxel: _host_np.ndarray, affine: _host_np.ndarray) -> float:
    """Project a voxel-space centroid onto the world x axis using *affine*."""
    homog = _host_np.asarray(
        [coords_voxel[0], coords_voxel[1], coords_voxel[2], 1.0], dtype=float
    )
    return float((_host_np.asarray(affine, dtype=float) @ homog)[0])


def split_lr_by_cc(mask: Image, *, n: int = 2, structure: Any = None) -> tuple[Image, Image]:
    """
    Split a bilateral binary *mask* into ``(left, right)`` images using connected components.

    The top-*n* CCs (by size) are kept and labeled L/R via their centroids'
    world-space x coordinate (computed with ``mask.affine``). The convention
    used is the radiology/anatomical one: **Left** means the subject's left
    side, which in a right-handed RAS/LAS coordinate system has the *lower* x
    world coordinate; in LPS/LPI it has the *higher* x coordinate. This is
    derived from the sign of ``affine[0, 0]`` so it works with either
    convention.

    Parameters
    ----------
    mask
        3D binary :class:`~nvitk.types.Image`. Its :attr:`Image.affine` is required.
    n
        How many components to keep (default 2, i.e. left + right).
    structure
        Connectivity structure passed to ``ndi.label``.

    Returns
    -------
    (Image, Image)
        Two images: ``(left, right)``. If fewer than 2 CCs are found, the
        missing one is returned as an empty mask.
    """
    if mask.ndim != 3:
        raise ValueError("split_lr_by_cc expects a 3D mask.")
    affine = mask.affine
    if affine is None:
        raise ValueError("split_lr_by_cc requires mask.affine to be set.")

    data = mask.data
    labeled, num = ndi.label(data, structure=structure)
    if num == 0:
        empty = np.zeros_like(data, dtype=np.uint8)
        return mask.with_data(empty.copy()), mask.with_data(empty.copy())

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    # Tiny array (N+1 ints); host hop is cheap and makes Python sort easy.
    sizes_h = to_numpy(sizes)
    top_ids = _host_np.argsort(sizes_h)[-n:][::-1]
    top_ids = [int(i) for i in top_ids if int(sizes_h[int(i)]) > 0]

    if len(top_ids) == 0:
        empty = np.zeros_like(data, dtype=np.uint8)
        return mask.with_data(empty.copy()), mask.with_data(empty.copy())

    # Small (<=n) list of 3-tuples; ok to keep host-side.
    affine_np = _host_np.asarray(to_numpy(affine), dtype=float)
    world_x: list[float | None] = []
    for cc_id in top_ids:
        coords = ndi.center_of_mass(np.ones_like(labeled), labeled, cc_id)
        coords_np = _host_np.asarray(to_numpy(np.asarray(coords)))
        world_x.append(_centroid_world_x(coords_np, affine_np))

    if len(top_ids) == 1:
        top_ids_opt: list[int | None] = [top_ids[0], None]
        world_x = [world_x[0], None]
    else:
        top_ids_opt = [top_ids[0], top_ids[1]]

    if world_x[1] is None:
        left_id, right_id = top_ids_opt[0], None
    else:
        if world_x[0] <= world_x[1]:
            left_id, right_id = top_ids_opt[0], top_ids_opt[1]
        else:
            left_id, right_id = top_ids_opt[1], top_ids_opt[0]

    left = (
        (labeled == left_id).astype(np.uint8)
        if left_id is not None
        else np.zeros_like(data, dtype=np.uint8)
    )
    right = (
        (labeled == right_id).astype(np.uint8)
        if right_id is not None
        else np.zeros_like(data, dtype=np.uint8)
    )

    return mask.with_data(left), mask.with_data(right)


def split_lr_by_midline(mask: Image, *, plane_x: int | None = None) -> tuple[Image, Image]:
    """
    Split a bilateral mask by a sagittal (YZ) plane at voxel x = *plane_x*.

    When *plane_x* is None, the plane is chosen so that its world-x matches the
    volume's world-x midpoint (i.e. the geometric midline along the image).

    Returns ``(left, right)`` following the same lower-world-x -> LEFT
    convention as :func:`split_lr_by_cc`.
    """
    if mask.ndim != 3:
        raise ValueError("split_lr_by_midline expects a 3D mask.")

    data = mask.data
    if plane_x is None:
        affine = mask.affine
        nx = data.shape[0]
        if affine is None:
            plane_x = nx // 2
        else:
            affine_np = _host_np.asarray(to_numpy(affine), dtype=float)
            origin = affine_np[:3, 3]
            step = affine_np[:3, 0]
            if float(step[0]) == 0.0:
                plane_x = nx // 2
            else:
                world_min = float(origin[0])
                world_max = float(origin[0] + step[0] * (nx - 1))
                target_world = (world_min + world_max) / 2.0
                plane_x = int(round((target_world - float(origin[0])) / float(step[0])))
                plane_x = max(0, min(nx - 1, plane_x))

    affine = mask.affine
    flip = False
    if affine is not None and float(to_numpy(affine)[0, 0]) < 0:
        flip = True

    lower_half = np.zeros_like(data, dtype=np.uint8)
    upper_half = np.zeros_like(data, dtype=np.uint8)
    lower_half[:plane_x, :, :] = data[:plane_x, :, :]
    upper_half[plane_x:, :, :] = data[plane_x:, :, :]

    left_arr = upper_half if flip else lower_half
    right_arr = lower_half if flip else upper_half

    return (
        mask.with_data(left_arr.astype(np.uint8)),
        mask.with_data(right_arr.astype(np.uint8)),
    )


__all__ = ["split_lr_by_cc", "split_lr_by_midline"]
