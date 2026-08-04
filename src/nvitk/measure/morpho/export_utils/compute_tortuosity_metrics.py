#!/usr/bin/env python3
"""Compute literature-style tortuosity metrics from generated centerlines.

Accepts either a folder of .vtp centerline files or an existing metrics
workbook whose per-centerline sheets contain x/y/z coordinate columns.

Metrics:
- tortuosity_index = actual vessel length / endpoint Euclidean distance
- inflection_count_metric = bend_count * tortuosity_index
- sum_of_angles_metric = sum of bend-chain angles / vessel length

Edit the CONFIG block below for direct single-case runs; the batch runner
calls run() directly with the appropriate paths.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("Missing dependency: pandas.") from exc


# ============================================================================
# CONFIG FOR DIRECT RUN (paths must be supplied at runtime; no host defaults)
# ============================================================================
INPUT_PATH = None
OUTPUT_CSV = None   # None → <input folder>/tortuosity_metrics.csv
OUTPUT_XLSX = None  # None → <input folder>/tortuosity_metrics.xlsx
SAVE_POINTWISE_SHEETS = False
SAVE_DEBUG_VTPS = True
DEBUG_VTP_DIR = None  # None → <input folder>/tortuosity_debug_vtps
BEND_TOLERANCE_FRACTION = 0.07
BEND_MIN_TOLERANCE_MM = 0.50
RESAMPLE_STEP_MM = 0.2
MIN_SEGMENT_LENGTH_MM = 1e-6
INFLECTION_SMOOTH_SIGMA_MM = 2.0
MIN_INFLECTION_LOBE_MM = 2.0
# ============================================================================


SUMMARY_SHEET_RE = re.compile(r"^\d{2}_")


def safe_sheet_name(name: str) -> str:
    """Sanitize *name* for use as an Excel sheet name (illegal chars stripped, 31-char limit)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return (cleaned or "centerline")[:31]


def safe_filename(name: str) -> str:
    """Sanitize *name* into a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned or "centerline"


def vessel_sort_key(path_or_name) -> Tuple[int, str]:
    """Sort key ordering centerline files/sheets by their leading label id, then name."""
    name = Path(path_or_name).stem
    match = re.match(r"^(\d+)(?:\D|$)", name)
    if not match:
        match = re.search(r"(?:^|_)label[_-]?(\d+)(?:\D|$)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def remove_duplicate_points(points: np.ndarray, min_segment_length_mm: float) -> np.ndarray:
    """Drop consecutive points closer together than *min_segment_length_mm* (keeps the first of each run)."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > min_segment_length_mm:
            keep.append(i)
    return points[keep]


def cumulative_s(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each point along the polyline, starting at 0."""
    if len(points) == 0:
        return np.array([], dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def arc_length(points: np.ndarray) -> float:
    """Total polyline length (sum of consecutive-point distances); 0.0 for <2 points."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def median_segment_length(points: np.ndarray, fallback: float = 0.1) -> float:
    """Median distance between consecutive points (a robust proxy for the local sampling resolution)."""
    if len(points) < 2:
        return float(fallback)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-12)]
    return float(np.median(lengths)) if lengths.size else float(fallback)


def chord_length(points: np.ndarray) -> float:
    """Straight-line distance from the first to the last point; 0.0 for <2 points."""
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def resample_polyline_by_arclength(points: np.ndarray, step_mm: Optional[float]) -> np.ndarray:
    """Resample a polyline to (near-)equal arc-length steps of *step_mm* via linear interpolation."""
    points = np.asarray(points, dtype=float)
    if step_mm is None or step_mm <= 0 or len(points) < 2:
        return points
    s = cumulative_s(points)
    total = float(s[-1]) if len(s) else 0.0
    if total <= step_mm or total <= 1e-12:
        return points
    n_steps = int(np.floor(total / step_mm))
    target_s = np.arange(n_steps + 1, dtype=float) * float(step_mm)
    if target_s[-1] < total:
        target_s = np.concatenate([target_s, [total]])
    else:
        target_s[-1] = total
    resampled = np.empty((len(target_s), 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_s, s, points[:, dim])
    return resampled


def local_turn_angles(points: np.ndarray) -> np.ndarray:
    """Turn angle (degrees) at each interior point between its incoming and outgoing segment vectors."""
    n = len(points)
    angles = np.full(n, np.nan, dtype=float)
    if n < 3:
        return angles
    prev_vec = points[1:-1] - points[:-2]
    next_vec = points[2:] - points[1:-1]
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    next_norm = np.linalg.norm(next_vec, axis=1)
    valid = (prev_norm > 1e-12) & (next_norm > 1e-12)
    dots = np.zeros(len(prev_vec), dtype=float)
    dots[valid] = np.einsum("ij,ij->i", prev_vec[valid], next_vec[valid]) / (prev_norm[valid] * next_norm[valid])
    dots = np.clip(dots, -1.0, 1.0)
    middle = angles[1:-1]
    middle[valid] = np.degrees(np.arccos(dots[valid]))
    angles[1:-1] = middle
    return angles


def perpendicular_distance_to_chord(points: np.ndarray) -> np.ndarray:
    """Perpendicular distance of every point from the straight chord joining the path's first and last point."""
    points = np.asarray(points, dtype=float)
    distances = np.full(len(points), np.nan, dtype=float)
    if len(points) == 0:
        return distances
    if len(points) < 2:
        distances[:] = 0.0
        return distances
    start = points[0]
    end = points[-1]
    chord_vec = end - start
    chord_norm = np.linalg.norm(chord_vec)
    if chord_norm <= 1e-12:
        distances[:] = np.linalg.norm(points - start, axis=1)
        return distances
    rel = points - start
    cross = np.cross(rel, chord_vec)
    distances[:] = np.linalg.norm(cross, axis=1) / chord_norm
    return distances


def perpendicular_distance_to_line_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Perpendicular distance of every point from the finite segment ``[start, end]`` (clamped, not an infinite line)."""
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    seg = end - start
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(((points - start) @ seg) / seg_len2, 0.0, 1.0)
    proj = start + t[:, None] * seg
    return np.linalg.norm(points - proj, axis=1)


def gaussian_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a 1-D signal that may contain NaNs (filled by interpolation before convolving, then re-masked)."""
    if sigma <= 0 or len(arr) < 3:
        return arr.copy()
    valid = np.isfinite(arr)
    if valid.sum() < 3:
        return arr.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    idx_valid = np.where(valid)[0]
    filled = np.interp(np.arange(len(arr)), idx_valid, arr[idx_valid])
    padded = np.pad(filled, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[~valid] = np.nan
    return smoothed


def signed_curvature_3d(points: np.ndarray, smooth_sigma_points: float = 20.0) -> np.ndarray:
    """Signed curvature along a 3-D polyline, with a consistent sign convention picked via a global binormal reference.

    Curvature magnitude comes from consecutive tangent turning (binormal
    cross-products); since a 3-D curve's binormal direction is only defined up
    to sign at each point, an SVD over all binormals picks one dominant plane
    normal as the reference so the sign is stable along the whole path (a
    genuine 3-D torsion reversal will still show as an actual sign flip).
    Output is smoothed with :func:`gaussian_smooth_1d`.
    """
    n = len(points)
    if n < 3:
        return np.full(n, np.nan)
    tangents = np.empty_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents /= norms
    binormals_mid = np.cross(tangents[:-1], tangents[1:])
    b_norms = np.linalg.norm(binormals_mid, axis=1)
    significant = b_norms > 1e-10
    if significant.sum() >= 2:
        b_sig = binormals_mid[significant]
        centered = b_sig - b_sig.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        ref = vt[0]
    elif significant.sum() == 1:
        ref = binormals_mid[significant][0]
        ref /= np.linalg.norm(ref)
    else:
        return np.full(n, np.nan)
    signed_mid = binormals_mid @ ref
    signed_curv = np.full(n, np.nan)
    signed_curv[1:-1] = 0.5 * (signed_mid[:-1] + signed_mid[1:])
    signed_curv[0] = signed_mid[0]
    signed_curv[-1] = signed_mid[-1]
    return gaussian_smooth_1d(signed_curv, sigma=smooth_sigma_points)


def find_inflection_indices(points: np.ndarray, signed_curv: np.ndarray, min_lobe_length_mm: float = 2.0) -> List[int]:
    """Point indices where signed curvature changes sign (real S-bends), merging inflections closer than *min_lobe_length_mm*."""
    n = len(points)
    if n < 3:
        return []
    s = cumulative_s(points)
    valid = np.isfinite(signed_curv)
    raw: List[int] = []
    for i in range(n - 1):
        if valid[i] and valid[i + 1] and signed_curv[i] * signed_curv[i + 1] < 0:
            if abs(signed_curv[i]) <= abs(signed_curv[i + 1]):
                raw.append(i)
            else:
                raw.append(i + 1)
    raw = sorted(set(raw))
    raw = [i for i in raw if 0 < i < n - 1]
    if not raw:
        return []
    filtered: List[int] = [raw[0]]
    for idx in raw[1:]:
        if s[idx] - s[filtered[-1]] >= min_lobe_length_mm:
            filtered.append(idx)
        else:
            if abs(signed_curv[idx]) < abs(signed_curv[filtered[-1]]):
                filtered[-1] = idx
    return filtered


def point_deflection_angle_deg(start: np.ndarray, apex: np.ndarray, end: np.ndarray) -> float:
    """Deflection angle (degrees) of the path bending through *apex*: 180° minus the interior angle at *apex*."""
    ba = np.asarray(start, dtype=float) - np.asarray(apex, dtype=float)
    bc = np.asarray(end, dtype=float) - np.asarray(apex, dtype=float)
    ba_n = float(np.linalg.norm(ba))
    bc_n = float(np.linalg.norm(bc))
    if ba_n <= 1e-12 or bc_n <= 1e-12:
        return np.nan
    interior = float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (ba_n * bc_n), -1.0, 1.0))))
    return float(abs(180.0 - interior))


def bend_angles_for_indices(points: np.ndarray, bend_indices: List[int]) -> Dict[int, float]:
    """Deflection angle at each bend index, measured against its neighboring bend/endpoint keypoints."""
    if len(points) < 3:
        return {}
    bends = sorted(set(int(i) for i in bend_indices if 0 < int(i) < len(points) - 1))
    keypoints = [0] + bends + [len(points) - 1]
    return {
        bend_idx: point_deflection_angle_deg(points[keypoints[i]], points[bend_idx], points[keypoints[i + 2]])
        for i, bend_idx in enumerate(bends)
    }


def rdp_bend_events(points: np.ndarray) -> List[dict]:
    """Detect bend "lobes" via recursive Ramer-Douglas-Peucker simplification against a local tolerance.

    Each recursive split point whose deviation from its local chord exceeds a
    tolerance (scaled by that chord's length) is recorded as a candidate bend
    event; overlapping candidates closer than 10% of the total path length are
    then pruned, keeping the most prominent one in each cluster.
    """
    points = np.asarray(points, dtype=float)
    s = cumulative_s(points)
    events: List[dict] = []
    if len(points) < 3:
        return events

    def recurse(start: int, end: int) -> None:
        """RDP recursion: find the point farthest from the chord ``[start, end]``, split there if it exceeds tolerance."""
        if end - start < 2:
            return
        segment_points = points[start:end + 1]
        distances = perpendicular_distance_to_line_segment(segment_points, points[start], points[end])
        local_idx = int(np.argmax(distances))
        idx = int(start + local_idx)
        max_distance = float(distances[local_idx])
        segment_chord_length = float(np.linalg.norm(points[end] - points[start]))
        local_tolerance = max(float(BEND_MIN_TOLERANCE_MM), float(BEND_TOLERANCE_FRACTION) * segment_chord_length)
        if idx <= start or idx >= end or max_distance < local_tolerance:
            return
        events.append({
            "idx": idx,
            "lobe_start_idx": int(start), "lobe_end_idx": int(end),
            "lobe_length_mm": float(s[end] - s[start]),
            "lobe_apex_distance_mm": max_distance,
            "lobe_prominence_mm": max_distance,
            "bend_tolerance_mm": local_tolerance,
            "score": max_distance,
        })
        recurse(start, idx)
        recurse(idx, end)

    recurse(0, len(points) - 1)
    all_events = sorted(events, key=lambda row: int(row["idx"]))
    vessel_arc = float(s[-1]) if len(s) > 1 else 0.0
    min_sep = 0.10 * vessel_arc
    if not all_events or min_sep <= 0:
        return all_events
    by_prominence = sorted(all_events, key=lambda e: float(e["score"]), reverse=True)
    kept: List[dict] = []
    for event in by_prominence:
        arc = float(s[int(event["idx"])])
        if all(abs(arc - float(s[int(k["idx"])])) >= min_sep for k in kept):
            kept.append(event)
    return sorted(kept, key=lambda row: int(row["idx"]))


def write_debug_vtp(
    path: Path,
    centerline_name: str,
    points: np.ndarray,
    s: np.ndarray,
    angles: np.ndarray,
    perpendicular_distances: np.ndarray,
    bend_events: List[dict],
    max_bending_idx: Optional[int],
    inflection_indices: Optional[List[int]] = None,
) -> None:
    """Write a VTK debug point cloud marking endpoints, bend events, the max-bending point, and inflections
    (each tagged with its role, arc length, angle, and bend-lobe stats) for visual QC of tortuosity detection.
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError("Writing debug .vtp files requires VTK.") from exc

    debug_records = []
    event_by_idx = {int(event["idx"]): event for event in bend_events}

    def add_point(idx: int, role: str, role_code: int) -> None:
        """Record one debug point (with its role tag and metrics) at path index *idx*."""
        if idx < 0 or idx >= len(points):
            return
        event = event_by_idx.get(int(idx), {})
        debug_records.append({
            "point_index": int(idx), "role": role, "role_code": int(role_code),
            "s_mm": float(s[idx]) if idx < len(s) else np.nan,
            "turn_angle_deg": float(angles[idx]) if idx < len(angles) and np.isfinite(angles[idx]) else np.nan,
            "perpendicular_distance_to_chord_mm": float(perpendicular_distances[idx]) if idx < len(perpendicular_distances) and np.isfinite(perpendicular_distances[idx]) else np.nan,
            "bend_lobe_length_mm": float(event.get("lobe_length_mm", np.nan)),
            "bend_lobe_apex_distance_mm": float(event.get("lobe_apex_distance_mm", np.nan)),
            "bend_lobe_prominence_mm": float(event.get("lobe_prominence_mm", np.nan)),
            "bend_tolerance_mm": float(event.get("bend_tolerance_mm", np.nan)),
            "bend_angle_deg": float(event.get("bend_angle_deg", np.nan)),
        })

    add_point(0, "endpoint_start", 1)
    add_point(len(points) - 1, "endpoint_end", 2)
    max_is_bend = max_bending_idx is not None and int(max_bending_idx) in event_by_idx
    for event in bend_events:
        idx = int(event["idx"])
        if max_is_bend and idx == int(max_bending_idx):
            add_point(idx, "bend_and_max_bending_point", 6)
        else:
            add_point(idx, "bend", 3)
    if max_bending_idx is not None and not max_is_bend:
        add_point(int(max_bending_idx), "max_bending_point", 4)
    for idx in (inflection_indices or []):
        add_point(int(idx), "inflection_point", 5)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(len(debug_records))
    for i, record in enumerate(debug_records):
        xyz = points[record["point_index"]]
        vtk_points.SetPoint(i, float(xyz[0]), float(xyz[1]), float(xyz[2]))

    verts = vtk.vtkCellArray()
    for i in range(len(debug_records)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    lines = vtk.vtkCellArray()
    if len(debug_records) >= 2:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(verts)
    poly.SetLines(lines)

    numeric_arrays = [
        ("SourcePointIndex", [r["point_index"] for r in debug_records]),
        ("PointRoleCode", [r["role_code"] for r in debug_records]),
        ("ArcLengthMm", [r["s_mm"] for r in debug_records]),
        ("TurnAngleDeg", [r["turn_angle_deg"] for r in debug_records]),
        ("PerpendicularDistanceToChordMm", [r["perpendicular_distance_to_chord_mm"] for r in debug_records]),
        ("BendLobeLengthMm", [r["bend_lobe_length_mm"] for r in debug_records]),
        ("BendLobeApexDistanceMm", [r["bend_lobe_apex_distance_mm"] for r in debug_records]),
        ("BendLobeProminenceMm", [r["bend_lobe_prominence_mm"] for r in debug_records]),
        ("BendToleranceMm", [r["bend_tolerance_mm"] for r in debug_records]),
        ("BendAngleDeg", [r["bend_angle_deg"] for r in debug_records]),
    ]
    for arr_name, values in numeric_arrays:
        arr = numpy_support.numpy_to_vtk(np.asarray(values, dtype=np.float64), deep=True)
        arr.SetName(arr_name)
        poly.GetPointData().AddArray(arr)

    role_arr = vtk.vtkStringArray()
    role_arr.SetName("PointRole")
    role_arr.SetNumberOfValues(len(debug_records))
    for i, record in enumerate(debug_records):
        role_arr.SetValue(i, record["role"])
    poly.GetPointData().AddArray(role_arr)

    name_arr = vtk.vtkStringArray()
    name_arr.SetName("CenterlineName")
    name_arr.SetNumberOfValues(len(debug_records))
    for i in range(len(debug_records)):
        name_arr.SetValue(i, centerline_name)
    poly.GetPointData().AddArray(name_arr)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.Write()


def compute_metrics(name: str, points: np.ndarray) -> Tuple[dict, pd.DataFrame]:
    """Compute the full standalone tortuosity-metric report for one centerline.

    Cleans/resamples the polyline, then derives tortuosity index, signed
    curvature + inflection points, RDP bend events, per-bend deflection angles,
    and chord-distance-based bending. Returns ``(summary_dict, pointwise_df)``
    — a one-row summary plus a per-point table (for the debug VTP / detail sheet).
    """
    points = remove_duplicate_points(points, MIN_SEGMENT_LENGTH_MM)
    points = resample_polyline_by_arclength(points, RESAMPLE_STEP_MM)
    s = cumulative_s(points)
    length = arc_length(points)
    chord = chord_length(points)
    tortuosity_index = float(length / chord) if chord > 1e-12 else np.nan

    angles = local_turn_angles(points)
    step = float(RESAMPLE_STEP_MM) if RESAMPLE_STEP_MM and RESAMPLE_STEP_MM > 0 else median_segment_length(points)
    smooth_sigma_pts = float(INFLECTION_SMOOTH_SIGMA_MM) / step
    signed_curv = signed_curvature_3d(points, smooth_sigma_points=smooth_sigma_pts)
    inflection_idx = find_inflection_indices(points, signed_curv, min_lobe_length_mm=float(MIN_INFLECTION_LOBE_MM))

    bend_events = rdp_bend_events(points)
    bend_idx = [int(event["idx"]) for event in bend_events]
    bend_count = len(bend_idx)
    inflection_count_metric = float(bend_count * tortuosity_index) if np.isfinite(tortuosity_index) else np.nan

    bend_angle_by_idx = bend_angles_for_indices(points, bend_idx)
    bend_angle_sum_deg = float(np.nansum(list(bend_angle_by_idx.values()))) if bend_angle_by_idx else 0.0
    sum_of_angles_metric_deg_per_mm = float(bend_angle_sum_deg / length) if length > 1e-12 else np.nan
    bend_size_max_mm = float(np.max([event["lobe_apex_distance_mm"] for event in bend_events])) if bend_events else 0.0

    perpendicular_distances = perpendicular_distance_to_chord(points)
    finite_perp = perpendicular_distances[np.isfinite(perpendicular_distances)]
    max_bending_idx = int(np.nanargmax(perpendicular_distances)) if finite_perp.size else None

    summary = {
        "centerline": name,
        "length_mm": float(length),
        "chord_length_mm": float(chord),
        "tortuosity_index": tortuosity_index,
        "inflection_count_metric": inflection_count_metric,
        "bend_angle_sum_deg": bend_angle_sum_deg,
        "bend_count": bend_count,
        "bend_indices": ";".join(str(i) for i in bend_idx),
        "max_bending_index": max_bending_idx if max_bending_idx is not None else np.nan,
        "bending_length_mm": bend_size_max_mm,
        "sum_of_angles_metric_deg_per_mm": sum_of_angles_metric_deg_per_mm,
        "total_curvature_deg_per_mm": float(np.nansum(angles) / length) if length > 1e-12 else np.nan,
    }

    bend_lobe_length = np.full(len(points), np.nan, dtype=float)
    bend_lobe_apex_distance = np.full(len(points), np.nan, dtype=float)
    bend_lobe_prominence = np.full(len(points), np.nan, dtype=float)
    bend_tolerance = np.full(len(points), np.nan, dtype=float)
    bend_angle = np.full(len(points), np.nan, dtype=float)
    for event in bend_events:
        idx = int(event["idx"])
        bend_lobe_length[idx] = float(event["lobe_length_mm"])
        bend_lobe_apex_distance[idx] = float(event["lobe_apex_distance_mm"])
        bend_lobe_prominence[idx] = float(event.get("lobe_prominence_mm", np.nan))
        bend_tolerance[idx] = float(event.get("bend_tolerance_mm", np.nan))
        bend_angle[idx] = bend_angle_by_idx.get(idx, np.nan)

    pointwise = pd.DataFrame({
        "point_index": np.arange(len(points), dtype=int),
        "x_mm": points[:, 0] if len(points) else np.array([], dtype=float),
        "y_mm": points[:, 1] if len(points) else np.array([], dtype=float),
        "z_mm": points[:, 2] if len(points) else np.array([], dtype=float),
        "s_mm": s,
        "turn_angle_deg": angles,
        "signed_curvature_3d": signed_curv,
        "perpendicular_distance_to_chord_mm": perpendicular_distances,
        "is_inflection_point": np.isin(np.arange(len(points), dtype=int), np.asarray(inflection_idx, dtype=int) if inflection_idx else np.array([], dtype=int)).astype(int),
        "is_bend": np.isin(np.arange(len(points), dtype=int), np.asarray(bend_idx, dtype=int) if bend_idx else np.array([], dtype=int)).astype(int),
        "bend_angle_deg": bend_angle,
        "bend_lobe_length_mm": bend_lobe_length,
        "bend_lobe_apex_distance_mm": bend_lobe_apex_distance,
        "bend_lobe_prominence_mm": bend_lobe_prominence,
        "bend_tolerance_mm": bend_tolerance,
    })
    return summary, pointwise


def load_vtp_points(path: Path) -> np.ndarray:
    """Load the longest polyline's points from a centerline VTP file."""
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError("Reading .vtp files requires VTK.") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return np.empty((0, 3), dtype=float)
    all_points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    lines = poly.GetLines()
    if lines is None or poly.GetNumberOfLines() == 0:
        return all_points
    id_list = vtk.vtkIdList()
    lines.InitTraversal()
    best_ids = []
    while lines.GetNextCell(id_list):
        ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        if len(ids) > len(best_ids):
            best_ids = ids
    return all_points[np.asarray(best_ids, dtype=int)] if best_ids else all_points


def load_centerlines_from_vtp_folder(input_path: Path) -> Dict[str, np.ndarray]:
    """Load every ``.vtp`` centerline in a folder, keyed by vessel name (strips a ``_radius`` suffix)."""
    files = sorted(input_path.glob("*.vtp"), key=vessel_sort_key)
    if not files:
        raise FileNotFoundError(f"No .vtp files found in {input_path}")
    return {path.stem.replace("_radius", ""): load_vtp_points(path) for path in files}


def is_centerline_sheet(sheet_name: str, columns: Iterable[str]) -> bool:
    """True for a per-centerline data sheet (not a numbered summary sheet) that has x/y/z_mm columns."""
    if SUMMARY_SHEET_RE.match(sheet_name):
        return False
    cols = {str(c).lower() for c in columns}
    return {"x_mm", "y_mm", "z_mm"}.issubset(cols)


def load_centerlines_from_workbook(input_path: Path) -> Dict[str, np.ndarray]:
    """Load every per-centerline sheet's x/y/z points from a morphometrics workbook, keyed by sheet name."""
    xls = pd.ExcelFile(input_path)
    out: Dict[str, np.ndarray] = {}
    for sheet in sorted(xls.sheet_names, key=vessel_sort_key):
        header = pd.read_excel(xls, sheet_name=sheet, nrows=0)
        if not is_centerline_sheet(sheet, header.columns):
            continue
        df = pd.read_excel(xls, sheet_name=sheet, usecols=["x_mm", "y_mm", "z_mm"])
        points = df[["x_mm", "y_mm", "z_mm"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            out[sheet] = points
    if not out:
        raise RuntimeError(f"No per-centerline sheets with x_mm/y_mm/z_mm found in {input_path}")
    return out


def load_centerlines(input_path: Path) -> Dict[str, np.ndarray]:
    """Load centerlines from a VTP folder, a single VTP file, or a morphometrics workbook (dispatch by path type)."""
    if input_path.is_dir():
        return load_centerlines_from_vtp_folder(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return load_centerlines_from_workbook(input_path)
    if input_path.suffix.lower() == ".vtp":
        return {input_path.stem.replace("_radius", ""): load_vtp_points(input_path)}
    raise ValueError(f"Unsupported INPUT_PATH type: {input_path}")


def default_output_paths(input_path: Path, output_csv: Optional[str], output_xlsx: Optional[str]) -> Tuple[Path, Path]:
    """Resolve the CSV/XLSX report paths, defaulting next to *input_path* when not explicitly given."""
    base_dir = input_path if input_path.is_dir() else input_path.parent
    csv_path = Path(output_csv) if output_csv else base_dir / "tortuosity_metrics.csv"
    xlsx_path = Path(output_xlsx) if output_xlsx else base_dir / "tortuosity_metrics.xlsx"
    return csv_path, xlsx_path


def default_debug_vtp_dir(input_path: Path) -> Path:
    """Resolve the debug-VTP output directory, defaulting next to *input_path* when ``DEBUG_VTP_DIR`` is unset."""
    if DEBUG_VTP_DIR:
        return Path(DEBUG_VTP_DIR).expanduser().resolve()
    return (input_path if input_path.is_dir() else input_path.parent) / "tortuosity_debug_vtps"


def run(
    input_path: str,
    output_csv: Optional[str] = None,
    output_xlsx: Optional[str] = None,
    save_pointwise_sheets: bool = False,
) -> pd.DataFrame:
    """Full standalone pipeline: load centerlines, compute tortuosity metrics for each, and write CSV/XLSX reports
    (optionally saving a debug VTP per centerline and per-point detail sheets in the workbook).
    """
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"INPUT_PATH not found: {source}")

    centerlines = load_centerlines(source)
    rows = []
    pointwise_tables = {}
    debug_dir = default_debug_vtp_dir(source) if SAVE_DEBUG_VTPS else None

    for name, points in centerlines.items():
        if len(points) < 2:
            print(f"  [tortuosity] skipping {name}: fewer than 2 points")
            continue
        summary, pointwise = compute_metrics(name, points)
        rows.append(summary)
        pointwise_tables[name] = pointwise
        if debug_dir is not None:
            max_bending_idx = int(pointwise["perpendicular_distance_to_chord_mm"].idxmax())
            bend_events = []
            for idx, row in pointwise[pointwise["is_bend"].astype(bool)].iterrows():
                bend_events.append({
                    "idx": int(idx),
                    "lobe_length_mm": float(row["bend_lobe_length_mm"]),
                    "lobe_apex_distance_mm": float(row["bend_lobe_apex_distance_mm"]),
                    "lobe_prominence_mm": float(row["bend_lobe_prominence_mm"]),
                    "bend_tolerance_mm": float(row["bend_tolerance_mm"]),
                    "bend_angle_deg": float(row["bend_angle_deg"]),
                })
            inflection_idx_list = [int(idx) for idx in pointwise.index[pointwise["is_inflection_point"].astype(bool)]]
            write_debug_vtp(
                path=debug_dir / f"{safe_filename(name)}_tortuosity_debug.vtp",
                centerline_name=name,
                points=pointwise[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float),
                s=pointwise["s_mm"].to_numpy(dtype=float),
                angles=pointwise["turn_angle_deg"].to_numpy(dtype=float),
                perpendicular_distances=pointwise["perpendicular_distance_to_chord_mm"].to_numpy(dtype=float),
                bend_events=bend_events,
                max_bending_idx=max_bending_idx,
                inflection_indices=inflection_idx_list,
            )

    if not rows:
        raise RuntimeError("No centerline with at least 2 points was found.")

    summary_df = pd.DataFrame(rows)
    csv_path, xlsx_path = default_output_paths(source, output_csv, output_xlsx)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="00_Tortuosity", index=False)
        if save_pointwise_sheets:
            for name in sorted(pointwise_tables, key=vessel_sort_key):
                pointwise_tables[name].to_excel(writer, sheet_name=safe_sheet_name(name), index=False)

    print(f"  [tortuosity] {len(centerlines)} centerline(s) → {xlsx_path}")
    if debug_dir is not None:
        print(f"  [tortuosity] debug VTPs → {debug_dir}")
    return summary_df


def main() -> None:
    """Direct-run entry point: run the tortuosity pipeline per the module-level ``INPUT_PATH``/config block."""
    if not INPUT_PATH:
        raise SystemExit("Set INPUT_PATH or call run() with input_path from stage7 outputs.")
    run(
        input_path=INPUT_PATH,
        output_csv=OUTPUT_CSV,
        output_xlsx=OUTPUT_XLSX,
        save_pointwise_sheets=SAVE_POINTWISE_SHEETS,
    )


if __name__ == "__main__":
    main()
