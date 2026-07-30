"""Oblique cross-section sampling and in-plane vessel segmentation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]

from nvitk.core.array import as_backend_array
from nvitk.core.backend import map_in_thread_pool, setup
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_2d
from nvitk.morphology.centerline import centerline_tangents
from nvitk.transform.oblique import (
    ObliquePlaneCoords,
    oblique_plane_coords,
    oblique_slice,
    oblique_slice_with_coords,
)

setup(globals())

# ---------------------------------------------------------------------------
# Defaults (MATLAB segment_cross_section_thresh)
# ---------------------------------------------------------------------------

_FUSE_WEIGHTS = (0.2, 0.8, 0.2)
_DEFAULT_RADIUS_VOX = 10.0
_DEFAULT_INTERP_VALS = 4


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
    radius_vox: float = _DEFAULT_RADIUS_VOX
    intensity_2d: np.ndarray | None = None


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


def _normalize_slice(sl: Any) -> Any:
    sl = as_backend_array(sl).astype(np.float64)
    hi = float(np.max(sl))
    if hi <= 0.0:
        return np.zeros_like(sl)
    return sl / hi


def segment_in_plane_cd_only(
    cd_sl: np.ndarray,
    *,
    min_component_fraction: float = 0.05,
    thr_algorithm: ThrAlgorithm = "lsthr",
) -> tuple[np.ndarray, float]:
    """Threshold a single complex-difference (or intensity) oblique slice in-plane."""
    from nvitk.morphology.components import (
        keep_component_closest_to_center,
        label_connected,
        remove_small_components,
    )

    fused = _normalize_slice(cd_sl)
    if thr_algorithm == "otsu":
        pos = fused[fused > 0]
        if pos.size < 2:
            return np.zeros(fused.shape, dtype=bool), 0.0
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError("otsu requires scikit-image") from exc
        try:
            t = float(threshold_otsu(pos))
        except ValueError:
            return np.zeros(fused.shape, dtype=bool), 0.0
        binary = (fused > t).astype(bool, copy=False)
    else:
        binary = binary_mask_sliding_threshold_2d(
            fused,
            shift_hm_flag=(thr_algorithm == "lthr"),
        )
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
    return best_mask.astype(bool, copy=False), float(_circularity_proxy(best_mask))


def cross_section_at_point(
    center_xyz: np.ndarray,
    tangent: np.ndarray,
    *,
    cd: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float = _DEFAULT_RADIUS_VOX,
    interp_vals: int = _DEFAULT_INTERP_VALS,
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
    cs_supersampling: bool = False,
    measure_resegment: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    volume_seg: np.ndarray | None = None,
    volume_label_id: int = 0,
) -> CrossSectionResult:
    """Oblique cross-section at a centerline point using a single 3D intensity volume.

    Intensity and mask are always sampled on the same in-plane grid. When
    *cs_supersampling* is on, that grid is finer: the intensity crop is linearly
    interpolated onto it, and (without resegmentation) the volume mask is
    nearest-neighbor sampled onto the same fine grid so overlays coincide.
    """
    u, v = plane_basis_from_tangent(tangent)
    res_meas = resolve_plane_res(
        radius_vox,
        cross_section_res=cross_section_res,
        interp_vals=interp_vals,
        supersampling=cs_supersampling,
    )
    center = as_backend_array(center_xyz).astype(np.float64).reshape(3)
    tang = as_backend_array(tangent).astype(np.float64).reshape(3)

    plane_meas = oblique_plane_coords(
        center, u, v, radius_vox=radius_vox, res=res_meas
    )

    if measure_resegment:
        order_meas = int(plane_interp_order)
        cd_sl = oblique_slice_with_coords(cd, plane_meas, order=order_meas)
        mask, circ = segment_in_plane_cd_only(cd_sl, thr_algorithm=thr_algorithm)
        intensity_2d = cd_sl
    else:
        if volume_seg is None:
            raise ValueError("volume_seg is required when measure_resegment is False")
        # Upsample the raw crop onto the (possibly supersampled) plane; keep
        # label masks as nearest-neighbor so overlays share the same grid.
        intensity_order = 1 if cs_supersampling else 0
        intensity_2d = oblique_slice_with_coords(cd, plane_meas, order=intensity_order)
        seg_sl = oblique_slice_with_coords(
            volume_seg.astype(np.float32), plane_meas, order=0
        )
        mask = _mask_from_volume_seg_slice(seg_sl, volume_label_id)
        circ = _circularity_proxy(mask)

    area_mm2 = _cross_section_area_mm2(
        mask,
        voxel_spacing=voxel_spacing,
        tangent=tang,
        plane_res=res_meas,
        radius_vox=radius_vox,
    )
    return CrossSectionResult(
        mask_2d=mask,
        area_mm2=area_mm2,
        circularity=float(circ),
        center_xyz=center,
        tangent=tang,
        u=u,
        v=v,
        pixel_spacing_mm=tilt_corrected_spacing_mm(voxel_spacing, tang),
        plane_res=res_meas,
        radius_vox=float(radius_vox),
        intensity_2d=as_backend_array(intensity_2d),
    )


def segment_in_plane(
    mag_sl: np.ndarray,
    cd_sl: np.ndarray,
    vel_sl: np.ndarray,
    *,
    min_component_fraction: float = 0.05,
    thr_algorithm: ThrAlgorithm = "lsthr",
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
    if thr_algorithm == "otsu":
        pos = fused[fused > 0]
        if pos.size < 2:
            return np.zeros(fused.shape, dtype=bool), 0.0
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError("otsu requires scikit-image") from exc
        try:
            t = float(threshold_otsu(pos))
        except ValueError:
            return np.zeros(fused.shape, dtype=bool), 0.0
        binary = (fused > t).astype(bool, copy=False)
    else:
        binary = binary_mask_sliding_threshold_2d(
            fused,
            shift_hm_flag=(thr_algorithm == "lthr"),
        )
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


def segment_in_plane_label_constrained(
    mag_sl: np.ndarray,
    cd_sl: np.ndarray,
    vel_sl: np.ndarray,
    label_mask_sl: np.ndarray,
    *,
    min_component_fraction: float = 0.05,
    thr_algorithm: ThrAlgorithm = "lsthr",
) -> tuple[np.ndarray, float, bool]:
    """In-plane QVT fusion mask intersected with a multilabel plane slice.

    Returns ``(mask, circularity, used_label_fallback)``.
    """
    from nvitk.morphology.components import (
        keep_component_closest_to_center,
        label_connected,
    )

    mask, _ = segment_in_plane(
        mag_sl,
        cd_sl,
        vel_sl,
        min_component_fraction=min_component_fraction,
        thr_algorithm=thr_algorithm,
    )
    label_bool = as_backend_array(label_mask_sl).astype(bool, copy=False)
    if not np.any(label_bool):
        return mask, _circularity_proxy(mask), False
    merged = mask & label_bool
    if np.any(merged):
        labeled, _ = label_connected(merged, connectivity=1)
        best_mask = as_backend_array(
            keep_component_closest_to_center(labeled)
        ).astype(bool, copy=False)
        return best_mask, _circularity_proxy(best_mask), False
    labeled, _ = label_connected(label_bool, connectivity=1)
    best_mask = as_backend_array(
        keep_component_closest_to_center(labeled)
    ).astype(bool, copy=False)
    return best_mask, _circularity_proxy(best_mask), True


def _circularity_proxy(mask: np.ndarray) -> float:
    """R_in^2 / R_out^2 style proxy from distance transform (MATLAB diam_val intent)."""
    m = as_backend_array(mask.astype(bool, copy=False))
    if not np.any(m):
        return 0.0
    dist = ndi.distance_transform_edt(m)
    rin = float(np.max(dist[m])) if np.any(m) else 0.0
    yi, xi = np.nonzero(m)
    cy, cx = float(np.mean(yi)), float(np.mean(xi))
    rout = float(np.max(np.sqrt((yi - cy) ** 2 + (xi - cx) ** 2))) + 1e-6
    if rout <= 0.0:
        return 0.0
    return float((rin / rout) ** 2)


def _plane_res(radius_vox: float, interp_vals: int, cross_section_res: int) -> int:
    if cross_section_res > 0:
        return int(cross_section_res)
    r = float(radius_vox)
    iv = max(1, int(interp_vals))
    return max(8, int(round(2.0 * r * iv)) + 1)


def _plane_res_nearest(radius_vox: float, cross_section_res: int) -> int:
    """One sample per voxel across the disk (no supersampling)."""
    if cross_section_res > 0:
        return int(cross_section_res)
    r = float(radius_vox)
    return max(8, int(np.ceil(2.0 * r)) + 1)


def resolve_plane_res(
    radius_vox: float,
    *,
    cross_section_res: int = 0,
    interp_vals: int = _DEFAULT_INTERP_VALS,
    supersampling: bool = False,
) -> int:
    """Oblique plane grid size; explicit *cross_section_res* overrides supersampling."""
    if int(cross_section_res) > 0:
        return int(cross_section_res)
    if supersampling:
        return _plane_res(radius_vox, interp_vals, 0)
    return _plane_res_nearest(radius_vox, 0)


def _mask_from_volume_seg_slice(
    seg_sl: np.ndarray,
    volume_label_id: int,
    *,
    keep_central_cc: bool = True,
) -> np.ndarray:
    """Label mask on an oblique seg slice; optionally keep CC nearest plane center."""
    from nvitk.morphology.components import (
        keep_component_closest_to_center,
        label_connected,
    )

    mask = as_backend_array(np.round(seg_sl) == int(volume_label_id)).astype(bool)
    if not keep_central_cc or not np.any(mask):
        return mask
    labeled, n_cc = label_connected(mask, connectivity=1)
    if n_cc <= 1:
        return mask
    return as_backend_array(keep_component_closest_to_center(labeled)).astype(bool)


def _cross_section_area_mm2(
    mask: np.ndarray,
    *,
    voxel_spacing: tuple[float, float, float],
    tangent: np.ndarray,
    plane_res: int,
    radius_vox: float,
) -> float:
    px, py = tilt_corrected_spacing_mm(voxel_spacing, tangent)
    native_span = 2.0 * float(radius_vox) + 1.0
    scale = native_span / float(max(1, int(plane_res)))
    d_area = (px / 10.0) * (py / 10.0) * (scale**2)
    return float(np.count_nonzero(mask)) * d_area * 100.0


def cross_section_at_loc(
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
    cs_supersampling: bool = False,
    measure_resegment: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    volume_seg: np.ndarray | None = None,
    volume_label_id: int = 0,
    label_constrain: bool = False,
) -> CrossSectionResult:
    """Oblique cross-section at a LOC; resegment in-plane or sample stage-4 mask."""
    u, v = plane_basis_from_tangent(tangent)
    res = resolve_plane_res(
        radius_vox,
        cross_section_res=cross_section_res,
        interp_vals=interp_vals,
        supersampling=cs_supersampling,
    )
    order = int(plane_interp_order)
    center = as_backend_array(center_xyz).astype(np.float64).reshape(3)
    tang = as_backend_array(tangent).astype(np.float64).reshape(3)

    plane_meas = oblique_plane_coords(
        center, u, v, radius_vox=radius_vox, res=res
    )

    if measure_resegment:
        cd_sl = oblique_slice_with_coords(cd, plane_meas, order=order)
        mag_sl = oblique_slice_with_coords(mag, plane_meas, order=order)
        vel_sl = oblique_slice_with_coords(vel_mag, plane_meas, order=order)
        if label_constrain and volume_seg is not None:
            seg_sl = oblique_slice_with_coords(
                volume_seg.astype(np.float32), plane_meas, order=0
            )
            label_mask = _mask_from_volume_seg_slice(seg_sl, volume_label_id)
            mask, circ, _fallback = segment_in_plane_label_constrained(
                mag_sl,
                cd_sl,
                vel_sl,
                label_mask,
                thr_algorithm=thr_algorithm,
            )
        else:
            mask, circ = segment_in_plane(
                mag_sl,
                cd_sl,
                vel_sl,
                thr_algorithm=thr_algorithm,
            )
    else:
        if volume_seg is None:
            raise ValueError("volume_seg is required when measure_resegment is False")
        # Match GUI display: upsample intensity on the fine plane; NN for labels.
        intensity_order = 1 if cs_supersampling else 0
        cd_sl = oblique_slice_with_coords(cd, plane_meas, order=intensity_order)
        seg_sl = oblique_slice_with_coords(
            volume_seg.astype(np.float32), plane_meas, order=0
        )
        mask = _mask_from_volume_seg_slice(seg_sl, volume_label_id)
        circ = _circularity_proxy(mask)

    area_mm2 = _cross_section_area_mm2(
        mask,
        voxel_spacing=voxel_spacing,
        tangent=tang,
        plane_res=res,
        radius_vox=radius_vox,
    )
    return CrossSectionResult(
        mask_2d=mask,
        area_mm2=area_mm2,
        circularity=float(circ),
        center_xyz=center,
        tangent=tang,
        u=u,
        v=v,
        pixel_spacing_mm=tilt_corrected_spacing_mm(voxel_spacing, tang),
        plane_res=res,
        radius_vox=float(radius_vox),
        intensity_2d=as_backend_array(cd_sl),
    )


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
    thr_algorithm: ThrAlgorithm = "lsthr",
) -> CrossSectionResult:
    """Sample oblique plane and segment vessel cross-section at *center_xyz*."""
    return cross_section_at_loc(
        center_xyz,
        tangent,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
        interp_vals=interp_vals,
        cross_section_res=cross_section_res,
        plane_interp_order=plane_interp_order,
        measure_resegment=True,
        thr_algorithm=thr_algorithm,
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


def _velocity_frame_workers(nt: int) -> int:
    """Thread count for per-frame velocity reslicing (0/1 = sequential)."""
    if nt < 6:
        return 1
    env = os.environ.get("NVITK_CROSS_SECTION_FRAME_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu, nt))


def _through_plane_frame_mean(
    ti: int,
    *,
    vx: Any,
    vy: Any,
    vz: Any,
    plane: ObliquePlaneCoords,
    order: int,
    t_hat: np.ndarray,
    mask: np.ndarray,
) -> float:
    vxsl = oblique_slice_with_coords(vx[..., ti], plane, order=order)
    vysl = oblique_slice_with_coords(vy[..., ti], plane, order=order)
    vzsl = oblique_slice_with_coords(vz[..., ti], plane, order=order)
    v_through = vxsl * t_hat[0] + vysl * t_hat[1] + vzsl * t_hat[2]
    vals = as_backend_array(v_through)[mask]
    return float(np.mean(vals)) if vals.size else 0.0


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
    t_hat = (as_backend_array(result.tangent)).astype(np.float64)
    t_hat = t_hat / (float(np.linalg.norm(t_hat)) + 1e-12)
    mask = (as_backend_array(result.mask_2d)).astype(bool, copy=False)
    if not np.any(mask):
        return np.zeros(nt, dtype=np.float64)

    plane = oblique_plane_coords(
        result.center_xyz,
        result.u,
        result.v,
        radius_vox=float(result.radius_vox),
        res=int(result.plane_res),
    )
    order = int(plane_interp_order)
    workers = _velocity_frame_workers(nt)
    if workers <= 1:
        out = np.empty(nt, dtype=np.float64)
        for ti in range(nt):
            out[ti] = _through_plane_frame_mean(
                ti,
                vx=vx,
                vy=vy,
                vz=vz,
                plane=plane,
                order=order,
                t_hat=t_hat,
                mask=mask,
            )
        return out

    def _frame_task(ti: int) -> float:
        return _through_plane_frame_mean(
            ti,
            vx=vx,
            vy=vy,
            vz=vz,
            plane=plane,
            order=order,
            t_hat=t_hat,
            mask=mask,
        )

    vals = map_in_thread_pool(_frame_task, range(nt), max_workers=workers)
    return np.asarray(vals, dtype=np.float64)


def flow_series_ml_s(velocity_ts: np.ndarray, area_mm2: float) -> np.ndarray:
    """Q(t) = v_plane(t) * area / 1000 (ml/s)."""
    return as_backend_array(velocity_ts).astype(np.float64) * (float(area_mm2) / 1000.0)


__all__ = [
    "CrossSectionResult",
    "ThrAlgorithm",
    "cross_section_at_loc",
    "cross_section_at_point",
    "flow_series_ml_s",
    "masked_plane_velocity_series",
    "resolve_plane_res",
    "segment_along_polyline",
    "segment_at_point",
    "segment_in_plane",
    "segment_in_plane_label_constrained",
    "segment_in_plane_cd_only",
    "tilt_corrected_spacing_mm",
]
