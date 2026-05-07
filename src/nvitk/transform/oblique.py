"""Oblique reslicing utilities (backend-aware map_coordinates).

This module uses `nvitk.core.backend.setup` to expose `np` and `ndi` for either
NumPy+SciPy or CuPy+cupyx.scipy.ndimage depending on the current backend.
"""

from __future__ import annotations

import numpy as _np

from nvitk.core.backend import setup

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
    center = _np.asarray(center_xyz, dtype=_np.float32)
    u = _np.asarray(u_xyz, dtype=_np.float32)
    v = _np.asarray(v_xyz, dtype=_np.float32)

    lin = _np.linspace(-r, r, n, dtype=_np.float32)
    xx, yy = _np.meshgrid(lin, lin, indexing="xy")
    pts = center[None, None, :] + xx[..., None] * u[None, None, :] + yy[..., None] * v[None, None, :]

    # map_coordinates expects coords per axis: (x, y, z)
    coords = [pts[..., 0], pts[..., 1], pts[..., 2]]
    return ndi.map_coordinates(vol, coords, order=int(order), mode=mode, cval=float(cval))


__all__ = ["oblique_slice"]

