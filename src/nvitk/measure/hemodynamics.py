"""Phase-contrast / 4D Flow hemodynamic indices and velocity helpers.

PC-MRI velocity conventions match :mod:`nvitk.io.conversors.phase2volume`.
PI/RI definitions follow QVTplus-style ratios on time-resolved flow or velocity
series (see :func:`pulsatility_index`, :func:`resistivity_index`).

Uses :func:`nvitk.core.backend.setup` so ``np`` follows the active NumPy or CuPy
backend; inputs are coerced with :func:`~nvitk.core.array.as_backend_array`.
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup

setup(globals())


def pulsatility_index(flow_t, *, eps: float = 1e-9):
    """PI = (max_t - min_t) / mean(|flow|) per row (QVTplus-style on flow)."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1)
    mn = np.min(x, axis=1)
    mu = np.mean(np.abs(x), axis=1)
    return (np.abs(mx - mn) / np.maximum(mu, eps)).astype(np.float64)


def resistivity_index(flow_t, *, eps: float = 1e-9):
    """RI = (max_t - min_t) / max(|flow|) per row."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1)
    mn = np.min(x, axis=1)
    den = np.maximum(np.max(np.abs(x), axis=1), eps)
    return (np.abs(mx - mn) / den).astype(np.float64)


def mean_flow_ml_s(flow_t, temporal_resolution_s: float | None):
    """Time-mean flow proxy (same units as *flow_t* per frame)."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.mean(x, axis=1)


def velocity_mm_s_from_phases(ap, rl, fh):
    """PC velocity components in mm/s (same convention as :mod:`nvitk.io.conversors.phase2volume`)."""
    ap = as_backend_array(ap).astype(np.float64)
    rl = as_backend_array(rl).astype(np.float64)
    fh = as_backend_array(fh).astype(np.float64)
    vx = -rl * 10.0
    vy = -ap * 10.0
    vz = fh * 10.0
    return vx, vy, vz


def through_plane_velocity_series(
    vx,
    vy,
    vz,
    *,
    i: int,
    j: int,
    k: int,
    tangent,
):
    """Time series of velocity projected onto unit *tangent* at voxel (i,j,k)."""
    t = as_backend_array(tangent).astype(np.float64).reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)
    nt = int(vx.shape[3])
    out = np.empty(nt, dtype=np.float64)
    for ti in range(nt):
        v = np.array([vx[i, j, k, ti], vy[i, j, k, ti], vz[i, j, k, ti]], dtype=np.float64)
        out[ti] = float(np.dot(v, t))
    return out


__all__ = [
    "mean_flow_ml_s",
    "pulsatility_index",
    "resistivity_index",
    "through_plane_velocity_series",
    "velocity_mm_s_from_phases",
]
