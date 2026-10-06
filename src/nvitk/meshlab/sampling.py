"""Images on surfaces and surfaces on image grids: probing and distance fields.

- :func:`sample_image` / :func:`sample_on_surface` — read an image's values at
  points or at a mesh's vertices (ParaView's *Probe*): colour a vessel wall by
  PET uptake, a surface by the label it touches. Optionally along the normal,
  averaged or maximised over a depth (a wall's thickness).
- :func:`signed_distance_image` — a closed mesh as a signed distance map on an
  image grid (mm; negative inside).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nvitk.types import Mesh, PointCloud

#: How :func:`sample_on_surface` combines the samples taken along each normal.
DEPTH_MODES: tuple[str, ...] = ("mean", "max", "min")


def sample_image(points: np.ndarray, data: np.ndarray, affine: np.ndarray, *, order: int = 1,
                 outside: float = np.nan) -> np.ndarray:
    """Values of a 3D image (array + 4x4 voxel→world affine) at world *points*.

    *order* 0 for label maps (nearest), 1 for intensities (trilinear).
    """
    from scipy.ndimage import map_coordinates

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    ijk = (np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(np.asarray(affine, dtype=float)).T)[:, :3]
    arr = np.asarray(data)
    vals = map_coordinates(arr.astype(np.float32) if order > 0 else arr, ijk.T, order=int(order),
                           mode="constant", cval=np.nan if order > 0 else 0)
    shape = np.asarray(arr.shape[:3])
    out = vals.astype(float)
    outside_mask = np.any((ijk < -0.5) | (ijk > shape - 0.5), axis=1)
    out[outside_mask] = outside
    return out


def sample_on_surface(
    mesh: Mesh | PointCloud,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    order: int = 1,
    depth: float = 0.0,
    samples: int = 5,
    mode: str = "mean",
) -> np.ndarray:
    """Per-vertex (per-point) image values, optionally across a depth along the normal.

    With *depth* > 0 each vertex reads *samples* points from ``−depth`` to
    ``+depth`` mm along its normal and combines them with *mode* — a wall's mean
    or peak value rather than the value exactly on the boundary.
    """
    if isinstance(mesh, Mesh):
        pts = mesh.vertices
        normals = mesh.vertex_normals if depth > 0 and mesh.n_faces else None
    else:
        pts = mesh.points
        normals = mesh.normals if depth > 0 else None
    if normals is None or depth <= 0:
        return sample_image(pts, data, affine, order=order)
    offsets = np.linspace(-float(depth), float(depth), max(2, int(samples)))
    stack = np.stack([sample_image(pts + normals * o, data, affine, order=order) for o in offsets])
    if mode == "max":
        return np.nanmax(stack, axis=0)
    if mode == "min":
        return np.nanmin(stack, axis=0)
    return np.nanmean(stack, axis=0)


def signed_distance_image(mesh: Mesh, shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    """Signed distance (mm) from a closed mesh on a grid: negative inside, positive outside.

    Voxel-accurate (the inside is filled, then Euclidean distance transforms run
    on both sides with the grid's spacing).
    """
    from scipy.ndimage import distance_transform_edt

    from nvitk.meshlab.voxelize import mesh_to_mask

    aff = np.asarray(affine, dtype=float)
    inside = np.asarray(mesh_to_mask(mesh, affine=aff, shape=tuple(int(v) for v in shape)).data) > 0
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    if not inside.any():
        return distance_transform_edt(np.ones(inside.shape, bool), sampling=spacing).astype(np.float32)
    out_d = distance_transform_edt(~inside, sampling=spacing)
    in_d = distance_transform_edt(inside, sampling=spacing)
    return np.where(inside, -in_d, out_d).astype(np.float32)


__all__ = ["DEPTH_MODES", "sample_image", "sample_on_surface", "signed_distance_image"]
