"""Spatial polynomial background correction for PC-MRI velocity (QVTplus-style).

Fits a low-order polynomial in normalized voxel coordinates to each velocity
component using voxels with low speed on the **temporal mean** field, evaluates
the fit on the full grid, then subtracts that 3D background from the mean
field and from **each** time frame (MATLAB-style: correct vMean and each frame).

Uses :func:`nvitk.core.backend.setup` so ``np`` follows the active NumPy or CuPy
backend (see :mod:`nvitk.core.backend`).
"""

from __future__ import annotations

from nvitk.core.backend import setup, using
from nvitk.core.array import as_backend_array, to_numpy

setup(globals())


def _poly_design_matrix(xyz, order: int):
    """xyz: (N, 3) in [-1, 1]. Returns (N, n_terms)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    cols = [np.ones_like(x), x, y, z]
    if order >= 2:
        cols.extend([x * x, y * y, z * z, x * y, x * z, y * z])
    if order >= 3:
        cols.extend(
            [
                x * x * x,
                y * y * y,
                z * z * z,
                x * x * y,
                x * x * z,
                y * y * x,
                y * y * z,
                z * z * x,
                z * z * y,
                x * y * z,
            ]
        )
    return np.stack(cols, axis=1)


def fit_polynomial_background_3vector(
    vx_mean,
    vy_mean,
    vz_mean,
    *,
    spatial_order: int = 2,
    static_percentile: float = 25.0,
    max_voxels: int = 12000,
):
    """Return (bg_x, bg_y, bg_z) same shape as inputs, fitted on low-speed voxels."""
    if spatial_order not in (2, 3):
        raise ValueError("spatial_order must be 2 or 3")
    if vx_mean.shape != vy_mean.shape or vx_mean.shape != vz_mean.shape:
        raise ValueError("vx_mean, vy_mean, vz_mean must match shape")

    nx, ny, nz = vx_mean.shape
    speed = np.sqrt(vx_mean**2 + vy_mean**2 + vz_mean**2)
    thr = np.percentile(speed, static_percentile)
    mask = speed.ravel() < thr
    idx = np.flatnonzero(mask)
    if idx.size < (spatial_order + 1) ** 3 + 10:
        idx = np.flatnonzero(speed.ravel() <= np.percentile(speed, 50.0))

    if idx.size > max_voxels:
        with using('cpu'):
            perm = np.random.default_rng(0).permutation(idx.size)[:max_voxels]
            idx = idx[perm]
        idx = as_backend_array(idx)

    ix = np.arange(nx, dtype=np.float64) / max(nx - 1, 1) * 2.0 - 1.0
    iy = np.arange(ny, dtype=np.float64) / max(ny - 1, 1) * 2.0 - 1.0
    iz = np.arange(nz, dtype=np.float64) / max(nz - 1, 1) * 2.0 - 1.0
    zz, yy, xx = np.meshgrid(iz, iy, ix, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    A_full = _poly_design_matrix(coords, spatial_order)

    A = A_full[idx]
    bvx = vx_mean.ravel()[idx]
    bvy = vy_mean.ravel()[idx]
    bvz = vz_mean.ravel()[idx]

    cx, *_ = np.linalg.lstsq(A, bvx, rcond=None)
    cy, *_ = np.linalg.lstsq(A, bvy, rcond=None)
    cz, *_ = np.linalg.lstsq(A, bvz, rcond=None)

    bg_x = (A_full @ cx).reshape(nx, ny, nz)
    bg_y = (A_full @ cy).reshape(nx, ny, nz)
    bg_z = (A_full @ cz).reshape(nx, ny, nz)
    return bg_x, bg_y, bg_z


def subtract_mean_background_from_temporal(
    vx,
    vy,
    vz,
    bg_x,
    bg_y,
    bg_z,
):
    """Subtract 3D (bg_*) from each frame of 4D velocity components."""
    if vx.ndim != 4:
        raise ValueError("expected (nx,ny,nz,nt) temporal velocity")
    return (
        vx - bg_x[..., np.newaxis],
        vy - bg_y[..., np.newaxis],
        vz - bg_z[..., np.newaxis],
    )
