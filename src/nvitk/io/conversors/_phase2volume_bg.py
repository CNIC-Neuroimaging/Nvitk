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


def fit_polynomial_background_scalar(
    volume,
    *,
    spatial_order: int = 2,
    static_percentile: float = 25.0,
    max_voxels: int = 12000,
):
    """Return a 3D polynomial background field fitted on low-intensity voxels of *volume*."""
    if spatial_order not in (2, 3):
        raise ValueError("spatial_order must be 2 or 3")
    vol = as_backend_array(volume).astype(np.float64)
    nx, ny, nz = vol.shape
    flat = vol.ravel()
    thr = np.percentile(flat, float(static_percentile))
    mask = flat < thr
    idx = np.flatnonzero(mask)
    if idx.size < (spatial_order + 1) ** 3 + 10:
        idx = np.flatnonzero(flat <= np.percentile(flat, 50.0))

    if idx.size > max_voxels:
        with using("cpu"):
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
    b = flat[idx]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    bg = (A_full @ coef).reshape(nx, ny, nz)
    return bg


def subtract_polynomial_background_4d_scalar(
    volume_4d,
    *,
    spatial_order: int = 2,
    static_percentile: float = 25.0,
):
    """Per-frame spatial polynomial background subtraction on a 4D scalar stack."""
    vol = as_backend_array(volume_4d).astype(np.float64)
    if vol.ndim != 4:
        raise ValueError("expected (nx, ny, nz, nt) volume")
    out = vol.copy()
    for t in range(int(vol.shape[3])):
        bg = fit_polynomial_background_scalar(
            vol[..., t],
            spatial_order=int(spatial_order),
            static_percentile=float(static_percentile),
        )
        out[..., t] = np.maximum(vol[..., t] - bg, 0.0)
    return out


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


# ---------------------------------------------------------------------------
# Scalar background on 3D and 3D+t images (any time axis)
# ---------------------------------------------------------------------------

#: The monomials ``x**a * y**b * z**c`` of the background polynomial, in the
#: order of :func:`_poly_design_matrix`'s columns.
_EXPONENTS = {
    2: (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
    ),
}
_EXPONENTS[3] = _EXPONENTS[2] + (
    (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
    (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1),
)


def _unit_axis(n: int):
    """Voxel indices along an axis of length *n*, mapped onto [-1, 1]."""
    return np.arange(n, dtype=np.float64) / max(n - 1, 1) * 2.0 - 1.0


def _background_sample(flat, *, static_percentile: float, signed: bool, mask_flat, order: int,
                       max_voxels: int):
    """Flat indices of the voxels the background is fitted on: the lowest
    *static_percentile* % of the values (of the absolute values when *signed*),
    inside *mask_flat* when given, at most *max_voxels* of them."""
    level = np.abs(flat) if signed else flat
    usable = np.isfinite(level)
    if mask_flat is not None:
        usable &= mask_flat
    candidates = np.flatnonzero(usable)
    if candidates.size == 0:
        raise ValueError("No voxels to fit the background on (is the mask empty?).")
    values = level[candidates]
    idx = candidates[values < np.percentile(values, float(static_percentile))]
    if idx.size < len(_EXPONENTS[order]) + 10:
        idx = candidates[values <= np.percentile(values, 50.0)]
    if idx.size > max_voxels:
        with using("cpu"):
            perm = np.random.default_rng(0).permutation(int(idx.size))[:max_voxels]
        idx = idx[as_backend_array(perm)]
    return idx


def polynomial_background(
    volume,
    *,
    order: int = 2,
    static_percentile: float = 25.0,
    signed: bool = False,
    mask=None,
    max_voxels: int = 12000,
):
    """The smooth spatial background of a 3D *volume*: a polynomial of *order*
    (2 or 3) in the voxel coordinates (each axis mapped onto [-1, 1]), fitted by
    least squares on the background voxels — the lowest *static_percentile* % of
    the values (of their absolute values when *signed*, as for a phase or
    velocity map), inside *mask* when one is given.

    Only the sampled voxels go into the fit, and the polynomial is evaluated one
    axis at a time, so no design matrix the size of the volume is built.
    """
    if int(order) not in _EXPONENTS:
        raise ValueError("order must be 2 or 3")
    order = int(order)
    vol = as_backend_array(volume).astype(np.float64)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {vol.ndim}D")
    nx, ny, nz = vol.shape
    flat = vol.ravel()
    mask_flat = None
    if mask is not None:
        mask_arr = as_backend_array(mask)
        if tuple(mask_arr.shape) != tuple(vol.shape):
            raise ValueError(f"mask shape {tuple(mask_arr.shape)} does not match the volume {tuple(vol.shape)}")
        mask_flat = mask_arr.ravel() != 0
    idx = _background_sample(flat, static_percentile=static_percentile, signed=signed,
                             mask_flat=mask_flat, order=order, max_voxels=int(max_voxels))
    i = idx // (ny * nz)
    j = (idx // nz) % ny
    k = idx % nz
    x = i * (2.0 / max(nx - 1, 1)) - 1.0
    y = j * (2.0 / max(ny - 1, 1)) - 1.0
    z = k * (2.0 / max(nz - 1, 1)) - 1.0
    design = np.stack([x ** a * y ** b * z ** c for a, b, c in _EXPONENTS[order]], axis=1)
    coef, *_ = np.linalg.lstsq(design, flat[idx], rcond=None)
    ax, ay, az = _unit_axis(nx), _unit_axis(ny), _unit_axis(nz)
    background = np.zeros((nx, ny, nz), dtype=np.float64)
    for weight, (a, b, c) in zip(coef, _EXPONENTS[order]):
        background += weight * ((ax ** a)[:, None, None] * (ay ** b)[None, :, None] * (az ** c)[None, None, :])
    return background


def subtract_polynomial_background(
    volume,
    *,
    time_axis: int | None = None,
    per_frame: bool = True,
    order: int = 2,
    static_percentile: float = 25.0,
    signed: bool = False,
    clip_negative: bool | None = None,
    mask=None,
    max_voxels: int = 12000,
    return_background: bool = False,
):
    """*volume* with its spatial polynomial background (:func:`polynomial_background`)
    subtracted — the correction qvtpy applies to the complex difference.

    A 3D volume is fitted on itself. A 3D+t image (*time_axis* names its time
    axis, any position) is corrected frame by frame: with *per_frame* each frame
    gets its own fit (qvtpy's ``ComplexDifference_4D`` correction), otherwise one
    background is fitted on the temporal mean and taken off every frame (the
    scheme of its velocity correction: steadier on noisy frames).

    *clip_negative* (default: when not *signed*) floors the result at 0, as a
    magnitude such as the complex difference cannot be negative. *mask* (3D,
    the spatial shape) limits where the background is fitted.

    Returns the corrected image (float64 for float64 input, else float32), and
    with *return_background* also the background: 3D, or one per frame.
    """
    clip = (not signed) if clip_negative is None else bool(clip_negative)
    shape = tuple(volume.shape)
    out_dtype = np.float64 if np.dtype(volume.dtype) == np.dtype("float64") else np.float32
    fit = dict(order=order, static_percentile=static_percentile, signed=signed, mask=mask,
               max_voxels=max_voxels)

    def _corrected(frame, background):
        out = as_backend_array(frame).astype(np.float64) - background
        return np.maximum(out, 0.0) if clip else out

    if time_axis is None:
        if len(shape) != 3:
            raise ValueError(f"expected a 3D image (or give time_axis for 3D+t), got {len(shape)}D")
        background = polynomial_background(volume, **fit)
        out = _corrected(volume, background).astype(out_dtype)
        return (out, background) if return_background else out

    if len(shape) != 4:
        raise ValueError(f"expected a 3D+t image, got {len(shape)}D")
    t_ax = int(time_axis) % 4
    n_t = int(shape[t_ax])

    def _frame_index(t: int) -> tuple:
        index = [slice(None)] * 4
        index[t_ax] = t
        return tuple(index)

    out = np.empty(shape, dtype=out_dtype)
    if per_frame:
        backgrounds = np.empty(shape, dtype=out_dtype) if return_background else None
        for t in range(n_t):
            frame = as_backend_array(volume[_frame_index(t)])
            background = polynomial_background(frame, **fit)
            out[_frame_index(t)] = _corrected(frame, background).astype(out_dtype)
            if backgrounds is not None:
                backgrounds[_frame_index(t)] = background.astype(out_dtype)
        return (out, backgrounds) if return_background else out

    mean = None
    for t in range(n_t):
        frame = as_backend_array(volume[_frame_index(t)]).astype(np.float64)
        mean = frame if mean is None else mean + frame
    background = polynomial_background(mean / float(n_t), **fit)
    for t in range(n_t):
        out[_frame_index(t)] = _corrected(volume[_frame_index(t)], background).astype(out_dtype)
    return (out, background) if return_background else out
