"""Oblique cross-section sampling and in-plane vessel segmentation (QVTplus-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology.centerline import centerline_tangents
from nvitk.transform.oblique import oblique_slice

setup(globals())

# ---------------------------------------------------------------------------
# Defaults (MATLAB segment_cross_section_thresh fusion weights)
# ---------------------------------------------------------------------------

_FUSE_WEIGHTS = (0.2, 0.8, 0.2)
_DEFAULT_RADIUS_VOX = 10.0
_DEFAULT_INTERP_VALS = 4


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrossSectionResult:
    """In-plane segmentation and metrics at one centerline station."""

    mask_2d: np.ndarray
    area_mm2: float
    circularity: float
    center_xyz: np.ndarray
    tangent: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pixel_spacing_mm: tuple[float, float]
    plane_res: int


# ---------------------------------------------------------------------------
# Oblique plane geometry
# ---------------------------------------------------------------------------


def plane_basis_from_tangent(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit vectors ``u``, ``v`` spanning the plane normal to *tangent*."""
    t = as_backend_array(tangent).astype(np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    t = t / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(t, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(t, ref)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-9:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = u / u_norm
    v = np.cross(t, u)
    v = v / (float(np.linalg.norm(v)) + 1e-12)
    return u.astype(np.float64), v.astype(np.float64)


def tilt_corrected_spacing_mm(
    voxel_spacing: tuple[float, float, float],
    tangent: np.ndarray,
) -> tuple[float, float]:
    """In-plane pixel spacing with tilt correction (MATLAB pixelSpace intent)."""
    sx, sy, sz = (float(voxel_spacing[0]), float(voxel_spacing[1]), float(voxel_spacing[2]))
    res = min(sx, sy, sz)
    t = as_backend_array(tangent).astype(np.float64).reshape(3)
    t = t / (float(np.linalg.norm(t)) + 1e-12)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    sin_ang = float(np.linalg.norm(np.cross(t, z_hat)))
    sin_ang = min(max(sin_ang, 0.0), 1.0)
    slice_sp = sz
    px = res + (slice_sp - res) * sin_ang
    return px, px


# ---------------------------------------------------------------------------
# In-plane sliding threshold + segmentation
# ---------------------------------------------------------------------------


def _normalize_slice(sl: Any) -> Any:
    sl = as_backend_array(sl).astype(np.float64)
    hi = float(np.max(sl))
    if hi <= 0.0:
        return np.zeros_like(sl)
    return sl / hi


def _sliding_threshold_2d(
    fused: np.ndarray,
    *,
    step: float = 0.001,
    up_thresh: float = 0.8,
    smf: int = 90,
) -> np.ndarray:
    """2D sliding-threshold binary mask on normalized fused contrast."""
    img = as_backend_array(fused).astype(np.float64)
    max_val = float(np.max(img))
    if max_val <= 0.0:
        return np.zeros(img.shape, dtype=bool)

    x = np.arange(0.0, up_thresh + step * 0.5, step, dtype=np.float64)
    sval = np.array([float(np.count_nonzero(img > (max_val * n))) for n in x], dtype=np.float64)
    smf = int(max(1, smf))
    kernel = np.ones(smf, dtype=np.float64) / float(smf)
    y = np.convolve(sval, kernel, mode="same")
    ymax = float(np.max(y))
    if ymax <= 0.0:
        return np.zeros(img.shape, dtype=bool)
    y = y / ymax
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    curvature = np.maximum(ddy, 0.0)
    idx = int(np.argmax(curvature))
    opt_frac = float(x[idx])
    thresh = max_val * opt_frac
    return (img > thresh).astype(bool, copy=False)


def segment_in_plane(
    mag_sl: np.ndarray,
    cd_sl: np.ndarray,
    vel_sl: np.ndarray,
    *,
    min_component_fraction: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Fuse slices, threshold, keep CC closest to plane center. Returns ``(mask, circularity)``."""
    from nvitk.morphology.components import (
        keep_component_closest_to_center,
        label_connected,
        remove_small_components,
    )

    w_mag, w_cd, w_vel = _FUSE_WEIGHTS
    fused = (
        w_mag * _normalize_slice(mag_sl)
        + w_cd * _normalize_slice(cd_sl)
        + w_vel * _normalize_slice(vel_sl)
    )
    binary = _sliding_threshold_2d(fused)
    n_fg = int(np.count_nonzero(binary))
    if n_fg == 0:
        return np.zeros(binary.shape, dtype=bool), 0.0
    min_size = max(1, int(round(float(min_component_fraction) * n_fg)))
    binary = as_backend_array(
        remove_small_components(binary, min_size=min_size, connectivity=1)
    ).astype(bool, copy=False)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=bool), 0.0

    labeled, _ = label_connected(binary, connectivity=1)
    best_mask = as_backend_array(
        keep_component_closest_to_center(labeled)
    ).astype(bool, copy=False)

    circularity = _circularity_proxy(best_mask)
    return best_mask.astype(bool, copy=False), float(circularity)


def _circularity_proxy(mask: np.ndarray) -> float:
    """R_in^2 / R_out^2 style proxy from distance transform (MATLAB diam_val intent)."""
    m = as_backend_array(mask.astype(bool, copy=False))
    if not np.any(m):
        return 0.0
    dist = ndi.distance_transform_edt(m)
    rin = float(np.max(dist[m])) if np.any(m) else 0.0
    # Approximate outer radius from mask extent.
    yi, xi = np.nonzero(m)
    cy, cx = float(np.mean(yi)), float(np.mean(xi))
    rout = float(np.max(np.sqrt((yi - cy) ** 2 + (xi - cx) ** 2))) + 1e-6
    if rout <= 0.0:
        return 0.0
    return float((rin / rout) ** 2)


# ---------------------------------------------------------------------------
# Per-station and along-polyline segmentation
# ---------------------------------------------------------------------------


def _plane_res(radius_vox: float, interp_vals: int, cross_section_res: int) -> int:
    if cross_section_res > 0:
        return int(cross_section_res)
    r = float(radius_vox)
    iv = max(1, int(interp_vals))
    return max(8, int(round(2.0 * r * iv)) + 1)


def segment_at_point(
    center_xyz: np.ndarray,
    tangent: np.ndarray,
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float = _DEFAULT_RADIUS_VOX,
    interp_vals: int = _DEFAULT_INTERP_VALS,
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
) -> CrossSectionResult:
    """Sample oblique plane and segment vessel cross-section at *center_xyz*."""
    u, v = plane_basis_from_tangent(tangent)
    res = _plane_res(radius_vox, interp_vals, cross_section_res)
    order = int(plane_interp_order)
    center = as_backend_array(center_xyz).astype(np.float64).reshape(3)
    mag_sl = oblique_slice(
        mag,
        center_xyz=center,
        u_xyz=u,
        v_xyz=v,
        radius_vox=radius_vox,
        res=res,
        order=order,
    )
    cd_sl = oblique_slice(
        cd,
        center_xyz=center,
        u_xyz=u,
        v_xyz=v,
        radius_vox=radius_vox,
        res=res,
        order=order,
    )
    vel_sl = oblique_slice(
        vel_mag,
        center_xyz=center,
        u_xyz=u,
        v_xyz=v,
        radius_vox=radius_vox,
        res=res,
        order=order,
    )
    mask, circ = segment_in_plane(mag_sl, cd_sl, vel_sl)
    px, py = tilt_corrected_spacing_mm(voxel_spacing, tangent)
    scale = (2.0 * float(radius_vox) + 1.0) / (2.0 * float(radius_vox) * float(interp_vals) + 1.0)
    d_area = (px / 10.0) * (py / 10.0) * (scale**2)
    area_mm2 = float(np.count_nonzero(mask)) * d_area * 100.0  # cm^2 -> mm^2
    return CrossSectionResult(
        mask_2d=mask,
        area_mm2=area_mm2,
        circularity=circ,
        center_xyz=center,
        tangent=as_backend_array(tangent).astype(np.float64).reshape(3),
        u=u,
        v=v,
        pixel_spacing_mm=(px, py),
        plane_res=res,
    )


def segment_along_polyline(
    points_xyz: np.ndarray,
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    stride: int = 1,
    radius_vox: float = _DEFAULT_RADIUS_VOX,
    **kwargs: Any,
) -> list[CrossSectionResult]:
    """Segment at stations along an ordered centerline polyline."""
    pts = as_backend_array(points_xyz).astype(np.float64)
    if pts.shape[0] < 2:
        return []
    tangents = centerline_tangents(pts, k_half=2)
    out: list[CrossSectionResult] = []
    step = max(1, int(stride))
    for idx in range(0, pts.shape[0], step):
        res = segment_at_point(
            pts[idx],
            tangents[idx],
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
            **kwargs,
        )
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# Masked-plane flow (stage 6)
# ---------------------------------------------------------------------------


def masked_plane_velocity_series(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    result: CrossSectionResult,
    *,
    plane_interp_order: int = 1,
) -> np.ndarray:
    """Through-plane velocity time series: masked mean on the oblique plane per frame."""
    vx = as_backend_array(vx)
    vy = as_backend_array(vy)
    vz = as_backend_array(vz)
    if vx.ndim != 4:
        raise ValueError("Expected 4D velocity components (x,y,z,t).")
    nt = int(vx.shape[3])
    t_hat = result.tangent / (float(np.linalg.norm(result.tangent)) + 1e-12)
    mask = result.mask_2d.astype(bool, copy=False)
    if not np.any(mask):
        return np.zeros(nt, dtype=np.float64)

    out = np.empty(nt, dtype=np.float64)
    for ti in range(nt):
        vxsl = oblique_slice(
            vx[..., ti],
            center_xyz=result.center_xyz,
            u_xyz=result.u,
            v_xyz=result.v,
            radius_vox=_DEFAULT_RADIUS_VOX,
            res=result.plane_res,
            order=int(plane_interp_order),
        )
        vysl = oblique_slice(
            vy[..., ti],
            center_xyz=result.center_xyz,
            u_xyz=result.u,
            v_xyz=result.v,
            radius_vox=_DEFAULT_RADIUS_VOX,
            res=result.plane_res,
            order=int(plane_interp_order),
        )
        vzsl = oblique_slice(
            vz[..., ti],
            center_xyz=result.center_xyz,
            u_xyz=result.u,
            v_xyz=result.v,
            radius_vox=_DEFAULT_RADIUS_VOX,
            res=result.plane_res,
            order=int(plane_interp_order),
        )
        v_through = vxsl * t_hat[0] + vysl * t_hat[1] + vzsl * t_hat[2]
        vals = v_through[mask]
        out[ti] = float(np.mean(vals)) if vals.size else 0.0
    return out


def flow_series_ml_s(velocity_ts: np.ndarray, area_mm2: float) -> np.ndarray:
    """Q(t) = v_plane(t) * area / 1000 (ml/s)."""
    return as_backend_array(velocity_ts).astype(np.float64) * (float(area_mm2) / 1000.0)


__all__ = [
    "CrossSectionResult",
    "flow_series_ml_s",
    "masked_plane_velocity_series",
    "plane_basis_from_tangent",
    "segment_along_polyline",
    "segment_at_point",
    "segment_in_plane",
    "tilt_corrected_spacing_mm",
]
