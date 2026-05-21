"""Mask-based hemodynamic indices (pseudo-LOC and voxel-averaged QC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.measure.cross_section import (
    ThrAlgorithm,
    cross_section_at_loc,
    flow_series_ml_s,
    masked_plane_velocity_series,
)
from nvitk.measure.hemodynamics import (
    pulsatility_index,
    resistivity_index,
    velocity_mm_s_from_phases,
)
from nvitk.morphology.centerline import centerline_tangents, compute_centerlines

setup(globals())

MaskMethod = Literal["pseudo_loc", "voxel_avg", "both"]


@dataclass(frozen=True)
class MaskHemodynamicsResult:
    method: str
    label_id: int
    pi: float
    ri: float
    mean_velocity_mm_s: float | None = None
    mean_flow_ml_s: float | None = None
    cross_section_area_mm2: float | None = None
    note: str = ""


def _pca_flow_axis(coords_zyx: np.ndarray) -> np.ndarray:
    """Dominant axis from mask voxel coordinates (z,y,x)."""
    c = coords_zyx.astype(np.float64)
    c = c - np.mean(c, axis=0)
    if c.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def mask_voxel_averaged_hemodynamics(
    mask: np.ndarray,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    *,
    label_id: int = 1,
) -> MaskHemodynamicsResult:
    """
    Velocity-only PI/RI from mean through-plane velocity along PCA axis.

    No cross-sectional area; not interchangeable with catheter-style flow.
    """
    m = to_numpy(mask)
    roi = m == int(label_id)
    if not np.any(roi):
        raise ValueError(f"Label {label_id} not found in mask.")
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)
    coords = np.argwhere(roi)
    axis = _pca_flow_axis(coords)
    nt = int(vx.shape[3])
    series = np.empty(nt, dtype=np.float64)
    for t in range(nt):
        vox = np.stack([vx[..., t][roi], vy[..., t][roi], vz[..., t][roi]], axis=1)
        series[t] = float(np.mean(vox @ axis))
    flow_2d = series.reshape(1, -1)
    return MaskHemodynamicsResult(
        method="voxel_avg",
        label_id=int(label_id),
        pi=float(pulsatility_index(flow_2d)[0]),
        ri=float(resistivity_index(flow_2d)[0]),
        mean_velocity_mm_s=float(np.mean(np.abs(series))),
        note="Velocity-only PI/RI (no cross-sectional area).",
    )


def mask_pseudo_loc_hemodynamics(
    mask: np.ndarray,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    label_id: int = 1,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    measure_resegment: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
    volume_seg: np.ndarray | None = None,
) -> MaskHemodynamicsResult:
    """Pseudo-LOC at mid-centerline station with oblique cross-section flow."""
    m = to_numpy(mask)
    lines = compute_centerlines(m, labels=[int(label_id)], min_points=5)
    if int(label_id) not in lines:
        raise ValueError(f"Could not compute centerline for label {label_id}.")
    pts = as_backend_array(lines[int(label_id)]).astype(np.float64)
    mid = int(pts.shape[0] // 2)
    center = pts[mid]
    tang = centerline_tangents(pts, k_half=2)[mid]
    xs = cross_section_at_loc(
        center,
        tang,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
        cross_section_res=cross_section_res,
        plane_interp_order=int(plane_interp_order),
        measure_resegment=measure_resegment,
        thr_algorithm=thr_algorithm,
        volume_seg=volume_seg,
        volume_label_id=int(label_id),
    )
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)
    vel_ts = masked_plane_velocity_series(
        vx, vy, vz, xs, plane_interp_order=int(plane_interp_order)
    )
    area_mm2 = float(xs.area_mm2)
    flow_ts = flow_series_ml_s(vel_ts, area_mm2)
    flow_2d = flow_ts.reshape(1, -1)
    return MaskHemodynamicsResult(
        method="pseudo_loc",
        label_id=int(label_id),
        pi=float(pulsatility_index(flow_2d)[0]),
        ri=float(resistivity_index(flow_2d)[0]),
        mean_velocity_mm_s=float(np.mean(np.abs(vel_ts))),
        mean_flow_ml_s=float(np.mean(np.abs(flow_ts))),
        cross_section_area_mm2=area_mm2,
        note="Mid-centerline pseudo-LOC with oblique cross-section.",
    )


def measure_mask_hemodynamics(
    mask: np.ndarray,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    *,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    label_id: int = 1,
    method: MaskMethod = "both",
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    measure_resegment: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    volume_seg: np.ndarray | None = None,
) -> list[MaskHemodynamicsResult]:
    """Run one or both mask hemodynamic methods for *label_id*."""
    ap_a = as_backend_array(ap).astype(np.float64)
    rl_a = as_backend_array(rl).astype(np.float64)
    fh_a = as_backend_array(fh).astype(np.float64)
    vx, vy, vz = velocity_mm_s_from_phases(ap_a, rl_a, fh_a)
    vm = vel_mag if vel_mag is not None else np.sqrt(vx[..., 0] ** 2 + vy[..., 0] ** 2 + vz[..., 0] ** 2)
    cd_a = cd if cd is not None else vm
    mag_a = mag if mag is not None else cd_a

    results: list[MaskHemodynamicsResult] = []
    if method in ("pseudo_loc", "both"):
        results.append(
            mask_pseudo_loc_hemodynamics(
                mask,
                ap_a,
                rl_a,
                fh_a,
                mag=mag_a,
                cd=cd_a,
                vel_mag=vm,
                label_id=label_id,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
                measure_resegment=measure_resegment,
                thr_algorithm=thr_algorithm,
                volume_seg=volume_seg,
            )
        )
    if method in ("voxel_avg", "both"):
        results.append(
            mask_voxel_averaged_hemodynamics(mask, ap_a, rl_a, fh_a, label_id=label_id)
        )
    return results


__all__ = [
    "MaskHemodynamicsResult",
    "MaskMethod",
    "mask_pseudo_loc_hemodynamics",
    "mask_voxel_averaged_hemodynamics",
    "measure_mask_hemodynamics",
]
