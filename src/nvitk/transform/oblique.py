"""Oblique reslicing utilities (backend-aware map_coordinates).

This module uses `nvitk.core.backend.setup` to expose `np` and `ndi` for either
NumPy+SciPy or CuPy+cupyx.scipy.ndimage depending on the current backend.
"""

from __future__ import annotations

from nvitk.core.backend import setup
from nvitk.core import as_backend_array

setup(globals())


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
    r = float(radius_vox)
    n = int(res)
    center = as_backend_array(center_xyz)
    u = as_backend_array(u_xyz)
    v = as_backend_array(v_xyz)

    lin = as_backend_array(np.linspace(-r, r, n))
    xx, yy = np.meshgrid(lin, lin, indexing="xy")
    pts = center[None, None, :] + xx[..., None] * u[None, None, :] + yy[..., None] * v[None, None, :]

    # map_coordinates expects coords per axis: (x, y, z)
    coords = as_backend_array([pts[..., 0], pts[..., 1], pts[..., 2]])
    return ndi.map_coordinates(as_backend_array(vol), coords, order=int(order), mode=mode, cval=float(cval))


__all__ = ["oblique_slice"]

