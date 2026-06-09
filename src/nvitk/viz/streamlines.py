"""Headless flow tracing: instantaneous streamlines and time-varying pathlines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from nvitk.core.array import to_numpy

IntegrationDirection = Literal["forward", "backward", "both"]
SeedMode = Literal["volume", "planar"]
TraceMode = Literal["streamlines", "pathlines"]
ColorMetric = Literal["speed", "integration_time", "arc_length", "fixed"]
PlaneSide = Literal["min", "max"]


@dataclass(frozen=True)
class FlowTraceParams:
    """Parameters for mask-seeded streamlines or pathlines."""

    n_seeds: int = 64
    max_length: float = 35.0
    stream_seed: int | None = 42
    label_ids: tuple[int, ...] | None = None
    trace_mode: TraceMode = "streamlines"
    integration_direction: IntegrationDirection = "forward"
    seed_mode: SeedMode = "planar"
    seed_plane_axis: int = 2
    seed_plane_side: PlaneSide = "min"
    dt_seconds: float = 1.0
    resample_paths: bool = False
    resample_spacing_vox: float = 0.5


def _mask_roi(mask: np.ndarray, label_ids: Sequence[int] | None) -> np.ndarray:
    m = to_numpy(mask)
    if label_ids:
        roi = np.zeros(m.shape, dtype=bool)
        for lid in label_ids:
            roi |= m == int(lid)
        return roi
    return m > 0


def _mask_velocity_to_roi(
    velocity: np.ndarray,
    mask: np.ndarray,
    label_ids: Sequence[int] | None,
) -> np.ndarray:
    """Zero velocity outside the vessel ROI."""
    vel = to_numpy(velocity).astype(np.float32, copy=True)
    roi = _mask_roi(mask, label_ids)
    if vel.ndim == 4:
        vel[~roi] = 0.0
    elif vel.ndim == 5:
        vel[~roi, ...] = 0.0
    return vel


def _points_inside_roi(points: np.ndarray, roi: np.ndarray) -> np.ndarray:
    pts = to_numpy(points).astype(np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    ii = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, roi.shape[0] - 1)
    jj = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, roi.shape[1] - 1)
    kk = np.clip(np.round(pts[:, 2]).astype(np.int64), 0, roi.shape[2] - 1)
    return roi[ii, jj, kk]


def _clip_polyline_to_roi(poly: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    pts = to_numpy(poly).astype(np.float32, copy=False)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    inside = _points_inside_roi(pts, roi)
    if not bool(np.any(inside)):
        return None
    best_start = 0
    best_len = 0
    run_start = 0
    run_len = 0
    for i, ok in enumerate(inside):
        if ok:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
        else:
            run_len = 0
    if best_len < 2:
        return None
    return pts[best_start : best_start + best_len]


def clip_polylines_to_roi(
    polylines: Sequence[np.ndarray],
    mask: np.ndarray,
    *,
    label_ids: Sequence[int] | None = None,
) -> list[np.ndarray]:
    roi = _mask_roi(mask, label_ids)
    out: list[np.ndarray] = []
    for poly in polylines:
        clipped = _clip_polyline_to_roi(poly, roi)
        if clipped is not None:
            out.append(clipped)
    return out


def stream_seed_cloud(
    mask: np.ndarray,
    n_seeds: int,
    seed: int | None,
    *,
    label_ids: Sequence[int] | None = None,
    seed_mode: SeedMode = "volume",
    seed_plane_axis: int = 2,
    seed_plane_side: PlaneSide = "min",
) -> np.ndarray | None:
    """Random subsample of mask voxel indices (N, 3) for trace seeds."""
    if int(n_seeds) <= 0:
        return None
    roi = _mask_roi(mask, label_ids)
    coords = np.argwhere(roi)
    if coords.shape[0] == 0:
        return None
    if seed_mode == "planar":
        coords = _planar_seed_candidates(
            coords,
            axis=int(seed_plane_axis),
            side=str(seed_plane_side),
        )
        if coords.shape[0] == 0:
            return None
    n_eff = min(int(n_seeds), int(coords.shape[0]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(coords.shape[0], size=n_eff, replace=False)
    return coords[pick].astype(np.float32, copy=False)


def _planar_seed_candidates(
    coords: np.ndarray,
    *,
    axis: int,
    side: str,
) -> np.ndarray:
    ax = int(axis) % 3
    if side == "max":
        plane_val = int(np.max(coords[:, ax]))
    else:
        plane_val = int(np.min(coords[:, ax]))
    slab = coords[coords[:, ax] == plane_val]
    if slab.shape[0] >= 4:
        return slab
    for tol in (1, 2, 3):
        slab = coords[np.abs(coords[:, ax] - plane_val) <= tol]
        if slab.shape[0] >= 4:
            return slab
    return coords


def resample_polyline(
    poly: np.ndarray,
    *,
    spacing_vox: float,
) -> np.ndarray:
    """Resample a polyline to approximately uniform arc-length spacing."""
    pts = to_numpy(poly).astype(np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.astype(np.float32, copy=False)
    step = max(float(spacing_vox), 1e-3)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= step:
        return pts.astype(np.float32, copy=False)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.arange(0.0, total + 0.5 * step, step, dtype=np.float64)
    out = np.empty((samples.shape[0], 3), dtype=np.float64)
    j = 0
    for i, s in enumerate(samples):
        while j < seg.shape[0] - 1 and dist[j + 1] < s:
            j += 1
        d0, d1 = dist[j], dist[j + 1]
        if d1 <= d0 + 1e-9:
            out[i] = pts[j]
        else:
            t = (s - d0) / (d1 - d0)
            out[i] = (1.0 - t) * pts[j] + t * pts[j + 1]
    return out.astype(np.float32, copy=False)


def resample_polylines(
    polylines: Sequence[np.ndarray],
    *,
    spacing_vox: float,
) -> list[np.ndarray]:
    return [
        resample_polyline(poly, spacing_vox=spacing_vox)
        for poly in polylines
        if poly is not None and np.asarray(poly).shape[0] >= 2
    ]


def sample_vel_trilinear(frame: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Trilinear sample of a (X, Y, Z, 3) velocity frame at fractional voxel position."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    nx, ny, nz, _ = frame.shape
    x = float(np.clip(x, 0.0, nx - 1.001))
    y = float(np.clip(y, 0.0, ny - 1.001))
    z = float(np.clip(z, 0.0, nz - 1.001))
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    z1 = min(z0 + 1, nz - 1)
    fx, fy, fz = x - x0, y - y0, z - z0

    c000 = frame[x0, y0, z0, :]
    c100 = frame[x1, y0, z0, :]
    c010 = frame[x0, y1, z0, :]
    c110 = frame[x1, y1, z0, :]
    c001 = frame[x0, y0, z1, :]
    c101 = frame[x1, y0, z1, :]
    c011 = frame[x0, y1, z1, :]
    c111 = frame[x1, y1, z1, :]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    c = c0 * (1.0 - fz) + c1 * fz
    return np.asarray(c, dtype=np.float32)


def _polylines_from_polydata(points: np.ndarray, lines: np.ndarray) -> list[np.ndarray]:
    pts = to_numpy(points).astype(np.float64)
    conn = to_numpy(lines).astype(np.int64).ravel()
    if pts.size == 0 or conn.size == 0:
        return []
    polylines: list[np.ndarray] = []
    i = 0
    while i < conn.size:
        n = int(conn[i])
        if n <= 0:
            break
        ids = conn[i + 1 : i + 1 + n]
        polylines.append(pts[ids].astype(np.float32, copy=False))
        i += 1 + n
    return polylines


def compute_streamlines(
    velocity_xyz: np.ndarray,
    mask: np.ndarray,
    params: FlowTraceParams,
) -> list[np.ndarray]:
    """Instantaneous streamlines for one cardiac phase."""
    import pyvista as pv

    vel = to_numpy(velocity_xyz).astype(np.float32, copy=False)
    if vel.ndim != 4 or vel.shape[3] != 3:
        raise ValueError("velocity_xyz must have shape (X, Y, Z, 3).")
    vel = _mask_velocity_to_roi(vel, mask, params.label_ids)
    x, y, z, _ = vel.shape
    seeds = stream_seed_cloud(
        mask,
        params.n_seeds,
        params.stream_seed,
        label_ids=params.label_ids,
        seed_mode=params.seed_mode,
        seed_plane_axis=params.seed_plane_axis,
        seed_plane_side=params.seed_plane_side,
    )
    if seeds is None:
        return []

    grid = pv.ImageData(dimensions=(x, y, z), spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data["v"] = vel.reshape(-1, 3, order="F")
    seed_cloud = pv.PolyData(seeds)
    direction = str(params.integration_direction or "forward")
    if direction not in ("forward", "backward", "both"):
        direction = "forward"
    try:
        stream = grid.streamlines_from_source(
            seed_cloud,
            vectors="v",
            max_length=float(params.max_length),
            integration_direction=direction,
        )
    except Exception:
        return []
    if stream.n_points <= 0:
        return []
    polylines = _polylines_from_polydata(stream.points, stream.lines)
    polylines = clip_polylines_to_roi(polylines, mask, label_ids=params.label_ids)
    if params.resample_paths:
        polylines = resample_polylines(
            polylines, spacing_vox=float(params.resample_spacing_vox)
        )
        polylines = clip_polylines_to_roi(polylines, mask, label_ids=params.label_ids)
    return polylines


def compute_pathlines(
    velocity_xyzt: np.ndarray,
    mask: np.ndarray,
    params: FlowTraceParams,
    *,
    time_start: int,
) -> list[np.ndarray]:
    """Pathlines integrated forward through the time-varying velocity field."""
    vel = _mask_velocity_to_roi(velocity_xyzt, mask, params.label_ids)
    if vel.ndim != 5 or vel.shape[3] < 2 or vel.shape[4] != 3:
        return []
    _, _, _, n_time, _ = vel.shape
    seeds = stream_seed_cloud(
        mask,
        params.n_seeds,
        params.stream_seed,
        label_ids=params.label_ids,
        seed_mode=params.seed_mode,
        seed_plane_axis=params.seed_plane_axis,
        seed_plane_side=params.seed_plane_side,
    )
    if seeds is None:
        return []

    dt_eff = max(float(params.dt_seconds), 1e-6)
    n_steps = int(max(1, min(n_time - 1, np.ceil(float(params.max_length) / dt_eff))))
    tt0 = int(np.clip(time_start, 0, n_time - 1))
    roi = _mask_roi(mask, params.label_ids)
    polylines: list[np.ndarray] = []

    for s in seeds:
        p = s.astype(np.float32, copy=True)
        if not bool(roi[
            int(np.clip(round(p[0]), 0, roi.shape[0] - 1)),
            int(np.clip(round(p[1]), 0, roi.shape[1] - 1)),
            int(np.clip(round(p[2]), 0, roi.shape[2] - 1)),
        ]):
            continue
        seg = [p.copy()]
        tt = tt0
        for _ in range(n_steps):
            if tt >= n_time - 1:
                break
            v0 = sample_vel_trilinear(vel[..., tt, :], p)
            p1 = p + v0 * dt_eff
            tt1 = min(tt + 1, n_time - 1)
            v1 = sample_vel_trilinear(vel[..., tt1, :], p1)
            v = 0.5 * (v0 + v1)
            p = p + v * dt_eff
            ii = int(np.clip(round(float(p[0])), 0, roi.shape[0] - 1))
            jj = int(np.clip(round(float(p[1])), 0, roi.shape[1] - 1))
            kk = int(np.clip(round(float(p[2])), 0, roi.shape[2] - 1))
            if not bool(roi[ii, jj, kk]):
                break
            seg.append(p.copy())
            tt = tt1
        if len(seg) >= 2:
            polylines.append(np.stack(seg, axis=0).astype(np.float32, copy=False))

    if params.resample_paths:
        polylines = resample_polylines(
            polylines, spacing_vox=float(params.resample_spacing_vox)
        )
        polylines = clip_polylines_to_roi(polylines, mask, label_ids=params.label_ids)
    return polylines


def _speed_field(velocity_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(to_numpy(velocity_xyz).astype(np.float64), axis=-1)


def _sample_speed_at_points(points: np.ndarray, speed: np.ndarray) -> np.ndarray:
    pts = to_numpy(points).astype(np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    ii = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, speed.shape[0] - 1)
    jj = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, speed.shape[1] - 1)
    kk = np.clip(np.round(pts[:, 2]).astype(np.int64), 0, speed.shape[2] - 1)
    return speed[ii, jj, kk].astype(np.float64, copy=False)


def _arc_length_from_start(points: np.ndarray) -> np.ndarray:
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def vertex_scalars_for_polylines(
    polylines: Sequence[np.ndarray],
    *,
    velocity_xyz: np.ndarray | None = None,
    color_metric: ColorMetric = "speed",
    dt_seconds: float = 1.0,
    trace_mode: TraceMode = "streamlines",
) -> list[np.ndarray]:
    """Per-vertex scalar values for coloring."""
    metric = str(color_metric or "speed")
    out: list[np.ndarray] = []
    speed = _speed_field(velocity_xyz) if velocity_xyz is not None else None
    for poly in polylines:
        pts = to_numpy(poly).astype(np.float64)
        n = pts.shape[0]
        if n == 0:
            out.append(np.zeros((0,), dtype=np.float64))
            continue
        if metric == "fixed":
            out.append(np.zeros((n,), dtype=np.float64))
        elif metric in ("integration_time", "arc_length"):
            if trace_mode == "pathlines" and metric == "integration_time":
                out.append(np.arange(n, dtype=np.float64) * float(dt_seconds))
            else:
                out.append(_arc_length_from_start(pts))
        elif speed is not None:
            out.append(_sample_speed_at_points(pts, speed))
        else:
            out.append(np.zeros((n,), dtype=np.float64))
    return out


def streamline_mean_speeds(
    polylines: Sequence[np.ndarray],
    velocity_xyz: np.ndarray,
) -> np.ndarray:
    """Mean |v| (mm/s) along each polyline."""
    scalars = vertex_scalars_for_polylines(
        polylines, velocity_xyz=velocity_xyz, color_metric="speed"
    )
    return np.array([float(np.mean(s)) if s.size else 0.0 for s in scalars], dtype=np.float64)


# Backward-compatible alias
StreamlineParams = FlowTraceParams

__all__ = [
    "ColorMetric",
    "FlowTraceParams",
    "IntegrationDirection",
    "PlaneSide",
    "SeedMode",
    "StreamlineParams",
    "TraceMode",
    "clip_polylines_to_roi",
    "compute_pathlines",
    "compute_streamlines",
    "resample_polyline",
    "resample_polylines",
    "sample_vel_trilinear",
    "stream_seed_cloud",
    "streamline_mean_speeds",
    "vertex_scalars_for_polylines",
]
