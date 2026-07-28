"""Flexion-point detection and splitting along ordered centerline polylines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvitk.core.array import to_numpy


@dataclass(frozen=True)
class FlexionPoint:
    """One detected bend on a centerline polyline."""

    index: int
    xyz: tuple[float, float, float]
    turn_angle_deg: float
    cumulative_arc_mm: float = 0.0


def polyline_turning_angles_deg(
    points: np.ndarray,
    *,
    smooth_window: int = 3,
) -> np.ndarray:
    """Per-interior-vertex turning angle in degrees (0 at endpoints)."""
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n < 3:
        return out

    w = max(1, int(smooth_window))
    tang = np.zeros_like(pts)
    for i in range(n):
        i0 = max(0, i - w)
        i1 = min(n - 1, i + w)
        d = pts[i1] - pts[i0]
        norm = float(np.linalg.norm(d))
        tang[i] = d / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i in range(1, n - 1):
        t0 = tang[i - 1]
        t1 = tang[i + 1]
        cos_a = float(np.clip(np.dot(t0, t1), -1.0, 1.0))
        out[i] = float(np.degrees(np.arccos(cos_a)))
    return out


def detect_flexion_points(
    points: np.ndarray,
    *,
    min_turn_angle_deg: float = 45.0,
    min_separation_points: int = 8,
    smooth_window: int = 3,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[FlexionPoint]:
    """Return flexion points as local maxima of turning angle along *points*."""
    pts = to_numpy(points).astype(np.float64)
    n = int(pts.shape[0])
    if n < 3:
        return []

    angles = polyline_turning_angles_deg(pts, smooth_window=smooth_window)
    sx, sy, sz = (float(voxel_spacing[0]), float(voxel_spacing[1]), float(voxel_spacing[2]))
    seg_mm = np.linalg.norm(
        np.diff(pts, axis=0) * np.array([sx, sy, sz], dtype=np.float64),
        axis=1,
    )
    cum_mm = np.concatenate([[0.0], np.cumsum(seg_mm)])

    sep = max(1, int(min_separation_points))
    min_ang = float(min_turn_angle_deg)
    candidates: list[tuple[int, float]] = []

    for i in range(1, n - 1):
        if angles[i] < min_ang:
            continue
        left = angles[max(0, i - sep) : i]
        right = angles[i + 1 : min(n, i + sep + 1)]
        if angles[i] >= float(np.max(left) if left.size else 0.0) and angles[i] >= float(
            np.max(right) if right.size else 0.0
        ):
            candidates.append((i, float(angles[i])))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[int, float]] = []
    for idx, ang in candidates:
        if all(abs(idx - k) >= sep for k, _ in kept):
            kept.append((idx, ang))

    kept.sort(key=lambda x: x[0])
    return [
        FlexionPoint(
            index=int(idx),
            xyz=(float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2])),
            turn_angle_deg=float(ang),
            cumulative_arc_mm=float(cum_mm[idx]),
        )
        for idx, ang in kept
    ]


def split_polyline_at_indices(
    points: np.ndarray,
    cut_indices: list[int],
) -> list[np.ndarray]:
    """Split an ordered polyline at interior *cut_indices*."""
    pts = to_numpy(points).astype(np.float32)
    n = int(pts.shape[0])
    if n == 0:
        return []
    cuts = sorted({int(c) for c in cut_indices if 0 < int(c) < n})
    bounds = [0, *cuts, n]
    parts: list[np.ndarray] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            parts.append(pts[a:b].astype(np.float32, copy=False))
    return parts


def rasterize_polylines_to_volume(
    shape: tuple[int, int, int],
    segments: list[np.ndarray],
    label_ids: list[int],
) -> np.ndarray:
    """Paint polyline segments into a 3D int32 volume."""
    vol = np.zeros(shape, dtype=np.int32)
    for lid, seg in zip(label_ids, segments):
        if not seg.size:
            continue
        p = to_numpy(seg)
        for row in p:
            i = int(round(float(row[0])))
            j = int(round(float(row[1])))
            k = int(round(float(row[2])))
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                vol[i, j, k] = int(lid)
    return vol


__all__ = [
    "FlexionPoint",
    "detect_flexion_points",
    "polyline_turning_angles_deg",
    "rasterize_polylines_to_volume",
    "split_polyline_at_indices",
]
