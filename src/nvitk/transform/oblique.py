"""Oblique reslicing utilities (backend-aware map_coordinates).

This module uses `nvitk.core.backend.setup` to expose `np` and `ndi` for either
NumPy+SciPy or CuPy+cupyx.scipy.ndimage depending on the current backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nvitk.core.backend import setup
from nvitk.core import as_backend_array

setup(globals())


@dataclass(frozen=True)
class ObliquePlaneCoords:
    """Precomputed ``map_coordinates`` sampling grid for one oblique plane."""

    coords: Any
    radius_vox: float
    res: int


def oblique_plane_coords(
    center_xyz,
    u_xyz,
    v_xyz,
    *,
    radius_vox: float,
    res: int,
) -> ObliquePlaneCoords:
    """Build a reusable coordinate grid for :func:`oblique_slice_with_coords`."""
    r = float(radius_vox)
    n = int(res)
    center = as_backend_array(center_xyz)
    u = as_backend_array(u_xyz)
    v = as_backend_array(v_xyz)

    lin = as_backend_array(np.linspace(-r, r, n))
    xx, yy = np.meshgrid(lin, lin, indexing="xy")
    pts = center[None, None, :] + xx[..., None] * u[None, None, :] + yy[..., None] * v[None, None, :]
    coords = as_backend_array([pts[..., 0], pts[..., 1], pts[..., 2]])
    return ObliquePlaneCoords(coords=coords, radius_vox=r, res=n)


def oblique_slice_with_coords(
    vol,
    plane: ObliquePlaneCoords,
    *,
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
):
    """Sample *vol* on a plane using a precomputed :class:`ObliquePlaneCoords` grid."""
    return ndi.map_coordinates(
        as_backend_array(vol),
        plane.coords,
        order=int(order),
        mode=mode,
        cval=float(cval),
    )


def oblique_slice(
    vol,
    *,
    center_xyz,
    u_xyz,
    v_xyz,
    radius_vox: float,
    res: int,
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
):
    """Sample `vol` on an oblique plane in voxel coordinates (x,y,z).

    Parameters
    ----------
    vol
        3D array (NumPy or CuPy).
    center_xyz, u_xyz, v_xyz
        3-vectors in voxel coordinates. `u` and `v` span the plane.
    radius_vox
        Half-size of the square plane in voxels.
    res
        Output resolution (res × res).
    order
        Interpolation order (0 for masks, 1 for images).
    """
    plane = oblique_plane_coords(
        center_xyz,
        u_xyz,
        v_xyz,
        radius_vox=radius_vox,
        res=res,
    )
    return oblique_slice_with_coords(
        vol,
        plane,
        order=order,
        mode=mode,
        cval=cval,
    )


__all__ = ["ObliquePlaneCoords", "oblique_plane_coords", "oblique_slice", "oblique_slice_with_coords"]

