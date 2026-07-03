"""Generic geometry, arc-length, tangent, resampling, and polyline helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nvitk.measure.morphometrics_config import (
    CENTERLINE_RESAMPLE_MIN_SEGMENT_MM,
    CENTERLINE_RESAMPLE_STEP_MM,
    DONUT_ARM_MAIN_OVERLAP_TOL_MM,
    DONUT_ARM_MIN_POINTS_AFTER_MAIN_OVERLAP_TRIM,
    RESAMPLE_CENTERLINES_BY_ARCLENGTH,
    TRIM_DONUT_ARM_OVERLAP_WITH_MAIN_CENTERLINE,
    VMTK_MIN_RETRIED_SEED_SEPARATION_MM,
)

def unit_vector(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def point_inside_mask_mm(pt_mm: np.ndarray, mask_bool: np.ndarray, spacing) -> bool:
    spacing = np.asarray(spacing, dtype=float)
    ijk = np.round(np.asarray(pt_mm, dtype=float) / spacing).astype(int)
    shape = np.array(mask_bool.shape, dtype=int)
    if np.any(ijk < 0) or np.any(ijk >= shape):
        return False
    return bool(mask_bool[tuple(ijk)])


def sample_edt_mm(dist_mm: np.ndarray, pt_mm: np.ndarray, spacing) -> float:
    spacing = np.asarray(spacing, dtype=float)
    ijk = np.round(np.asarray(pt_mm, dtype=float) / spacing).astype(int)
    ijk = np.clip(ijk, 0, np.array(dist_mm.shape) - 1)
    return float(dist_mm[tuple(ijk)])


def estimate_path_endpoint_tangent(path_mm: np.ndarray, at_start: bool = True, n_pts: int = 5):
    pts = np.asarray(path_mm, dtype=float)
    if len(pts) < 2:
        return None
    n_pts = max(2, min(int(n_pts), len(pts)))
    if at_start:
        seg = pts[:n_pts]
        tangent = seg[0] - seg[-1]
    else:
        seg = pts[-n_pts:]
        tangent = seg[-1] - seg[0]
    return unit_vector(tangent)


def march_endpoint_to_wall(start_pt_mm, tangent_out, mask_bool, spacing, step_mm=None, wall_tol_mm=0.35, max_steps=200):
    spacing = np.asarray(spacing, dtype=float)
    if step_mm is None:
        step_mm = float(np.min(spacing)) * 0.5
    tangent_out = unit_vector(tangent_out)
    if tangent_out is None:
        return np.empty((0, 3), dtype=float)
    dist_mm = ndi.distance_transform_edt(mask_bool, sampling=spacing)
    p = np.asarray(start_pt_mm, dtype=float).copy()
    if not point_inside_mask_mm(p, mask_bool, spacing):
        return np.empty((0, 3), dtype=float)
    new_pts = []
    for _ in range(int(max_steps)):
        p_next = p + tangent_out * step_mm
        if not point_inside_mask_mm(p_next, mask_bool, spacing):
            break
        d = sample_edt_mm(dist_mm, p_next, spacing)
        new_pts.append(p_next.copy())
        p = p_next
        if d <= wall_tol_mm:
            break
    return np.asarray(new_pts, dtype=float)


def extend_path_to_walls(path_mm, mask_bool, spacing, tangent_pts=5, step_mm=None, wall_tol_mm=0.35, max_steps=200):
    pts = np.asarray(path_mm, dtype=float)
    if len(pts) < 2:
        return pts
    t_start = estimate_path_endpoint_tangent(pts, at_start=True, n_pts=tangent_pts)
    prefix = march_endpoint_to_wall(pts[0], t_start, mask_bool, spacing, step_mm, wall_tol_mm, max_steps)
    t_end = estimate_path_endpoint_tangent(pts, at_start=False, n_pts=tangent_pts)
    suffix = march_endpoint_to_wall(pts[-1], t_end, mask_bool, spacing, step_mm, wall_tol_mm, max_steps)
    if len(prefix):
        pts = np.vstack([prefix[::-1], pts])
    if len(suffix):
        pts = np.vstack([pts, suffix])
    return pts


def arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return np.nan
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample_polyline_by_arclength(pts_mm: np.ndarray, step_mm: Optional[float]) -> np.ndarray:
    pts_mm = np.asarray(pts_mm, dtype=float)
    if step_mm is None or step_mm <= 0 or len(pts_mm) < 2:
        return pts_mm

    finite = np.isfinite(pts_mm).all(axis=1)
    pts_mm = pts_mm[finite]
    if len(pts_mm) < 2:
        return pts_mm

    keep = [0]
    min_seg = float(CENTERLINE_RESAMPLE_MIN_SEGMENT_MM)
    for i in range(1, len(pts_mm)):
        if np.linalg.norm(pts_mm[i] - pts_mm[keep[-1]]) > min_seg:
            keep.append(i)
    pts_mm = pts_mm[np.asarray(keep, dtype=int)]
    if len(pts_mm) < 2:
        return pts_mm

    total = arc_length(pts_mm)
    if not np.isfinite(total) or total <= 1e-12:
        return pts_mm

    n_segments = max(1, int(np.round(total / float(step_mm))))
    if n_segments == 1:
        return np.vstack([pts_mm[0], pts_mm[-1]])

    def walk_equal_chords(chord_mm: float) -> Tuple[np.ndarray, bool]:
        out = [pts_mm[0].copy()]
        seg_i = 0
        cursor = pts_mm[0].copy()
        eps = 1e-10
        for _ in range(n_segments - 1):
            center = out[-1]
            found = False
            while seg_i < len(pts_mm) - 1:
                a = cursor
                b = pts_mm[seg_i + 1]
                d = b - a
                aa = float(np.dot(d, d))
                if aa <= 1e-20:
                    seg_i += 1
                    cursor = pts_mm[seg_i].copy()
                    continue
                rel = a - center
                bb = 2.0 * float(np.dot(rel, d))
                cc = float(np.dot(rel, rel) - chord_mm * chord_mm)
                disc = bb * bb - 4.0 * aa * cc
                if disc >= -1e-12:
                    disc = max(0.0, disc)
                    roots = sorted(((-bb - np.sqrt(disc)) / (2.0 * aa), (-bb + np.sqrt(disc)) / (2.0 * aa)))
                    for t in roots:
                        if eps < t <= 1.0 + eps:
                            t = float(np.clip(t, 0.0, 1.0))
                            q = a + t * d
                            out.append(q)
                            cursor = q
                            found = True
                            break
                if found:
                    break
                seg_i += 1
                cursor = pts_mm[seg_i].copy()
            if not found:
                return np.vstack(out + [pts_mm[-1].copy()]), False
        out.append(pts_mm[-1].copy())
        return np.vstack(out), True

    def final_delta(chord_mm: float) -> Tuple[float, np.ndarray, bool]:
        candidate, ok = walk_equal_chords(chord_mm)
        if len(candidate) < n_segments + 1:
            return -chord_mm, candidate, False
        final_len = float(np.linalg.norm(candidate[-1] - candidate[-2]))
        return final_len - chord_mm, candidate, ok

    low = 0.0
    high = max(float(step_mm), total / n_segments)
    high_delta, high_candidate, high_ok = final_delta(high)
    while high_delta > 0.0 and high < total:
        high *= 1.5
        high_delta, high_candidate, high_ok = final_delta(high)

    best = high_candidate
    for _ in range(40):
        mid = 0.5 * (low + high)
        delta, candidate, ok = final_delta(mid)
        if ok:
            best = candidate
        if delta > 0.0:
            low = mid
        else:
            high = mid

    return best


def resample_generated_centerline_points(pts_mm: np.ndarray) -> np.ndarray:
    if not RESAMPLE_CENTERLINES_BY_ARCLENGTH:
        return np.asarray(pts_mm, dtype=float)
    return resample_polyline_by_arclength(pts_mm, CENTERLINE_RESAMPLE_STEP_MM)


def chord_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return np.nan
    return float(np.linalg.norm(pts[-1] - pts[0]))


def cumulative_s(pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.array([])
    if len(pts) == 1:
        return np.array([0.0])
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def point_at_arc_length(pts: np.ndarray, target_s: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.full(3, np.nan)
    if len(pts) == 1:
        return pts[0].copy()

    s = cumulative_s(pts)
    target_s = float(np.clip(target_s, 0.0, float(s[-1])))
    j = int(np.searchsorted(s, target_s))
    if j <= 0:
        return pts[0].copy()
    if j >= len(s):
        return pts[-1].copy()
    denom = s[j] - s[j - 1]
    if denom <= 1e-12:
        return pts[j].copy()
    a = (target_s - s[j - 1]) / denom
    return (1.0 - a) * pts[j - 1] + a * pts[j]


def trimmed_seed_pair_from_path(path_mm: np.ndarray, trim_start_mm: float, trim_end_mm: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pts = np.asarray(path_mm, dtype=float)
    if len(pts) < 2:
        return None
    s = cumulative_s(pts)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 0.0:
        return None
    trim_start_mm = max(0.0, float(trim_start_mm))
    trim_end_mm = trim_start_mm if trim_end_mm is None else max(0.0, float(trim_end_mm))
    if trim_start_mm + trim_end_mm >= total - float(VMTK_MIN_RETRIED_SEED_SEPARATION_MM):
        return None
    return point_at_arc_length(pts, trim_start_mm), point_at_arc_length(pts, total - trim_end_mm)


def mean_distance_to_polyline_points(query_pts: np.ndarray, ref_pts: np.ndarray) -> float:
    query_pts = np.asarray(query_pts, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)
    if len(query_pts) == 0 or len(ref_pts) == 0:
        return np.inf
    dist, _ = cKDTree(ref_pts).query(query_pts, k=1)
    return float(np.nanmean(dist))


def trim_polyline_overlap_with_reference_ends(
    pts: np.ndarray,
    reference_centerline_points: Optional[List[np.ndarray]],
    spacing,
    min_points_after_trim: Optional[int] = None,
    keep_original_if_too_short: bool = True,
) -> Tuple[np.ndarray, int, int, int, float]:
    pts = np.asarray(pts, dtype=float)
    min_points = int(min_points_after_trim) if min_points_after_trim is not None else int(DONUT_ARM_MIN_POINTS_AFTER_MAIN_OVERLAP_TRIM)
    refs = [np.asarray(x, dtype=float) for x in (reference_centerline_points or []) if len(x)]
    if (
        not TRIM_DONUT_ARM_OVERLAP_WITH_MAIN_CENTERLINE
        or len(pts) < min_points
        or not refs
    ):
        return pts, 0, 0, int(len(pts)), np.nan

    spacing = np.asarray(spacing, dtype=float)
    overlap_tol = (
        float(DONUT_ARM_MAIN_OVERLAP_TOL_MM)
        if DONUT_ARM_MAIN_OVERLAP_TOL_MM is not None
        else max(0.75, 2.0 * float(np.min(spacing)))
    )
    ref = np.vstack(refs)
    dist, _ = cKDTree(ref).query(pts, k=1)
    close = np.isfinite(dist) & (dist <= overlap_tol)

    start = 0
    while start < len(close) and close[start]:
        start += 1

    end = len(close)
    while end > start and close[end - 1]:
        end -= 1

    if end - start < min_points:
        if keep_original_if_too_short:
            return pts, 0, 0, int(len(pts)), overlap_tol
        return pts[0:0].copy(), int(start), int(len(pts) - end), 0, overlap_tol

    trimmed = pts[start:end]
    return trimmed, int(start), int(len(pts) - end), int(len(trimmed)), overlap_tol


def splice_skeleton_end_connectors(pts: np.ndarray, arm_pts_mm: np.ndarray, blend_mm: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    arm_pts_mm = np.asarray(arm_pts_mm, dtype=float)
    if len(pts) < 2 or len(arm_pts_mm) < 2:
        return pts

    out = pts.copy()
    d0 = float(np.linalg.norm(out[0] - arm_pts_mm[0]))
    if d0 > blend_mm:
        idx = int(np.argmin(np.linalg.norm(arm_pts_mm - out[0], axis=1)))
        if idx > 0:
            out = np.vstack([arm_pts_mm[:idx + 1], out[1:]])
        else:
            out[0] = arm_pts_mm[0]

    d1 = float(np.linalg.norm(out[-1] - arm_pts_mm[-1]))
    if d1 > blend_mm:
        idx = int(np.argmin(np.linalg.norm(arm_pts_mm - out[-1], axis=1)))
        if idx < len(arm_pts_mm) - 1:
            out = np.vstack([out[:-1], arm_pts_mm[idx:]])
        else:
            out[-1] = arm_pts_mm[-1]

    return out


def trim_polyline_ends_by_length(pts: np.ndarray, trim_start_mm: float, trim_end_mm: Optional[float] = None) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return pts
    trim_start_mm = max(0.0, float(trim_start_mm))
    trim_end_mm = trim_start_mm if trim_end_mm is None else max(0.0, float(trim_end_mm))
    s = cumulative_s(pts)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 0.0 or trim_start_mm + trim_end_mm >= 0.8 * total:
        return pts

    keep = (s >= trim_start_mm) & (s <= total - trim_end_mm)
    if keep.sum() < 2:
        return pts

    out = pts[keep]
    start_target = trim_start_mm
    end_target = total - trim_end_mm

    def interp_at(target_s: float) -> np.ndarray:
        j = int(np.searchsorted(s, target_s))
        if j <= 0:
            return pts[0]
        if j >= len(s):
            return pts[-1]
        denom = s[j] - s[j - 1]
        if denom <= 1e-12:
            return pts[j]
        a = (target_s - s[j - 1]) / denom
        return (1.0 - a) * pts[j - 1] + a * pts[j]

    start_pt = interp_at(start_target)
    end_pt = interp_at(end_target)
    if np.linalg.norm(out[0] - start_pt) > 1e-8:
        out = np.vstack([start_pt, out])
    if np.linalg.norm(out[-1] - end_pt) > 1e-8:
        out = np.vstack([out, end_pt])
    return out


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not mask.any():
        return []
    idx = np.where(mask)[0]
    runs = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev + 1))
        start = prev = value
    runs.append((start, prev + 1))
    return runs
