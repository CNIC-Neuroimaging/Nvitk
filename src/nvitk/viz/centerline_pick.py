"""Centerline polyline picking and plane-frame helpers (shared with flowshow)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from nvitk.core.array import to_numpy


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector; zero input yields zeros."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return np.zeros_like(v)
    return v / n


def tangent_window_indices(
    points: np.ndarray,
    idx: int,
    *,
    window: int = 5,
) -> tuple[int, int]:
    """Inclusive index range ``[a, b]`` used by ``tangent_from_centerline``."""
    w = int(window)
    if w not in (3, 5):
        w = 5
    k = 1 if w == 3 else 2
    n = int(np.asarray(points).shape[0])
    a = max(0, int(idx) - k)
    b = min(n - 1, int(idx) + k)
    return a, b


def tangent_from_centerline(points: np.ndarray, idx: int, *, window: int = 5) -> np.ndarray:
    """Tangent at ``points[idx]`` from neighbors (window 3 or 5)."""
    pts = np.asarray(points, dtype=np.float32)
    a, b = tangent_window_indices(pts, idx, window=window)
    if b == a:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return unit_vector(pts[b] - pts[a])


def frame_from_tangent(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane unit vectors ``u``, ``v`` orthogonal to *tangent*."""
    t = unit_vector(tangent)
    axes = np.eye(3, dtype=np.float32)
    dots = np.abs(axes @ t)
    a = axes[int(np.argmin(dots))]
    u = unit_vector(np.cross(t, a))
    v = unit_vector(np.cross(t, u))
    return u, v


@dataclass(frozen=True)
class CenterlinePick:
    """Nearest centerline station to a 3D pick point."""

    label: int
    index: int
    point: np.ndarray
    distance_sq: float


def _segment_vertex_index(i: int, n_pts: int, t: float) -> int:
    if t >= 0.5:
        return min(i + 1, n_pts - 1)
    return i


def nearest_centerline_on_segments(
    picked_xyz: np.ndarray,
    centerlines: Mapping[int, np.ndarray],
    *,
    max_distance_vox: float | None = None,
) -> CenterlinePick | None:
    """Snap a click to the closest point on any centerline segment (not only vertices)."""
    p = np.asarray(picked_xyz, dtype=np.float64).reshape(3)
    max_d2 = (
        float(max_distance_vox) ** 2
        if max_distance_vox is not None
        else float("inf")
    )
    best: CenterlinePick | None = None
    best_d2 = float("inf")
    for lbl, pts in centerlines.items():
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 2 or pts_arr.shape[1] != 3:
            continue
        n = pts_arr.shape[0]
        for i in range(n - 1):
            a, b = pts_arr[i], pts_arr[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                closest = a
                t = 0.0
            else:
                t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
                closest = a + t * ab
            d2 = float(np.sum((p - closest) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = CenterlinePick(
                    label=int(lbl),
                    index=_segment_vertex_index(i, n, t),
                    point=closest.astype(np.float32, copy=False),
                    distance_sq=d2,
                )
    if best is None or best_d2 > max_d2:
        return None
    return best


def nearest_centerline_along_ray(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    centerlines: Mapping[int, np.ndarray],
    *,
    max_distance_vox: float | None = None,
    anchor_xyz: np.ndarray | None = None,
    max_anchor_distance_vox: float | None = None,
) -> CenterlinePick | None:
    """
    Pick the centerline point closest to the view line through *ray_origin*.

    Uses the full line (both directions along *ray_direction*), so the result does not
    flip when the camera orientation sign changes. Optional *anchor_xyz* rejects snaps
    far from the click in 3D (limits grabbing parallel vessels).
    """
    o = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    d = unit_vector(np.asarray(ray_direction, dtype=np.float64).reshape(3))
    anchor = (
        np.asarray(anchor_xyz, dtype=np.float64).reshape(3)
        if anchor_xyz is not None
        else o
    )
    max_perp2 = (
        float(max_distance_vox) ** 2
        if max_distance_vox is not None
        else float("inf")
    )
    max_anchor2 = (
        float(max_anchor_distance_vox) ** 2
        if max_anchor_distance_vox is not None
        else float("inf")
    )
    best: CenterlinePick | None = None
    best_d2 = float("inf")
    for lbl, pts in centerlines.items():
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 2 or pts_arr.shape[1] != 3:
            continue
        n = pts_arr.shape[0]
        for i in range(n - 1):
            a, b = pts_arr[i], pts_arr[i + 1]
            seg = b - a
            denom = float(np.dot(seg, seg))
            if denom < 1e-12:
                continue
            # Closest point between infinite line (o + s*d) and segment (a + t*seg).
            w0 = o - a
            b_coef = float(np.dot(d, seg))
            c_coef = denom
            d_coef = float(np.dot(d, w0))
            e_coef = float(np.dot(seg, w0))
            det = c_coef - b_coef * b_coef
            if abs(det) < 1e-9:
                t_seg = float(np.clip(-e_coef / c_coef, 0.0, 1.0))
            else:
                t_seg = (e_coef - b_coef * d_coef) / det
                t_seg = float(np.clip(t_seg, 0.0, 1.0))
            closest = a + t_seg * seg
            hit = o + float(np.dot(closest - o, d)) * d
            d2_perp = float(np.sum((closest - hit) ** 2))
            if d2_perp >= best_d2 or d2_perp > max_perp2:
                continue
            # Reject only in-plane offset from click (ignore depth along the view line).
            diff = closest - anchor
            diff_orth = diff - float(np.dot(diff, d)) * d
            d2_anchor = float(np.sum(diff_orth**2))
            if d2_anchor > max_anchor2:
                continue
            best_d2 = d2_perp
            best = CenterlinePick(
                label=int(lbl),
                index=_segment_vertex_index(i, n, t_seg),
                point=closest.astype(np.float32, copy=False),
                distance_sq=d2_perp,
            )
    return best


def nearest_centerline_point(
    picked_xyz: np.ndarray,
    centerlines: Mapping[int, np.ndarray],
    *,
    max_distance_vox: float | None = None,
) -> CenterlinePick | None:
    """Map a 3D pick to the closest point on centerline polylines."""
    return nearest_centerline_on_segments(
        picked_xyz,
        centerlines,
        max_distance_vox=max_distance_vox,
    )


def choose_plane_normal_sense(
    tangent: np.ndarray,
    plane_center: np.ndarray,
    click_xyz: np.ndarray | None = None,
    *,
    centerline_pts: np.ndarray | None = None,
    index: int = 0,
) -> np.ndarray:
    """
    Resolve ±plane normal using centerline polyline orientation (one sense only).

    The tangent from finite differences is ambiguous; align it with the direction
    along the ordered centerline (index → index+1, or index−1 → index at the end).
    """
    _ = plane_center, click_xyz
    t = unit_vector(np.asarray(tangent, dtype=np.float64).reshape(3))
    if centerline_pts is None:
        return t.astype(np.float32)
    pts = np.asarray(centerline_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return t.astype(np.float32)
    idx = int(np.clip(index, 0, pts.shape[0] - 1))
    if idx < pts.shape[0] - 1:
        forward = unit_vector(pts[idx + 1] - pts[idx])
    else:
        forward = unit_vector(pts[idx] - pts[idx - 1])
    if float(np.dot(t, forward)) < 0.0:
        return (-t).astype(np.float32)
    return t.astype(np.float32)


def centerline_label_at_voxel(
    centerline_mask: np.ndarray,
    click_xyz: np.ndarray,
) -> int | None:
    """Skeleton mask label under the click (0 = background)."""
    mask = np.asarray(centerline_mask)
    if mask.ndim != 3:
        return None
    click = np.asarray(click_xyz, dtype=np.float64).reshape(3)
    i = int(np.clip(round(float(click[0])), 0, mask.shape[0] - 1))
    j = int(np.clip(round(float(click[1])), 0, mask.shape[1] - 1))
    k = int(np.clip(round(float(click[2])), 0, mask.shape[2] - 1))
    lbl = int(mask[i, j, k])
    return lbl if lbl != 0 else None


def centerline_label_near_click(
    centerline_mask: np.ndarray,
    click_xyz: np.ndarray,
    *,
    radius_vox: float = 2.0,
) -> int | None:
    """Most common skeleton label in a small neighborhood around the click."""
    mask = np.asarray(centerline_mask)
    if mask.ndim != 3:
        return None
    click = np.asarray(click_xyz, dtype=np.float64).reshape(3)
    r = max(1, int(np.ceil(float(radius_vox))))
    i0 = int(np.clip(round(float(click[0])), 0, mask.shape[0] - 1))
    j0 = int(np.clip(round(float(click[1])), 0, mask.shape[1] - 1))
    k0 = int(np.clip(round(float(click[2])), 0, mask.shape[2] - 1))
    counts: dict[int, int] = {}
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                i = int(np.clip(i0 + di, 0, mask.shape[0] - 1))
                j = int(np.clip(j0 + dj, 0, mask.shape[1] - 1))
                k = int(np.clip(k0 + dk, 0, mask.shape[2] - 1))
                lbl = int(mask[i, j, k])
                if lbl != 0:
                    counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _pick_label_candidates(
    click: np.ndarray,
    pts: np.ndarray,
    lbl: int,
    *,
    max_distance_vox: float,
    max_ray_distance_vox: float,
    max_anchor_distance_vox: float,
    ray_origin: np.ndarray | None,
    ray_direction: np.ndarray | None,
    use_view_line: bool,
) -> CenterlinePick | None:
    """View-line snap in 3D, else 3D distance from click."""
    one = {int(lbl): pts}
    if use_view_line and ray_origin is not None and ray_direction is not None:
        pr = nearest_centerline_along_ray(
            ray_origin,
            ray_direction,
            one,
            max_distance_vox=max_ray_distance_vox,
            anchor_xyz=click,
            max_anchor_distance_vox=max_anchor_distance_vox,
        )
        if pr is not None:
            return pr
    return nearest_centerline_on_segments(
        click,
        one,
        max_distance_vox=max_distance_vox,
    )


def pick_centerline(
    picked_xyz: np.ndarray,
    centerlines: Mapping[int, np.ndarray],
    *,
    max_distance_vox: float,
    max_ray_distance_vox: float | None = None,
    max_anchor_distance_vox: float | None = None,
    centerline_mask: np.ndarray | None = None,
    ray_origin: np.ndarray | None = None,
    ray_direction: np.ndarray | None = None,
    use_view_line: bool = False,
) -> CenterlinePick | None:
    """
    Snap the click to the nearest point on a centerline polyline (segment-wise).

    In 3D with *use_view_line*, snap to the view line through the click (perpendicular
    distance), then fall back to 3D distance if no segment is close enough to that line.

    When several vessels are close, each label is scored separately; the skeleton label
    near the cursor breaks ties only when picks are similarly close.
    """
    click = np.asarray(picked_xyz, dtype=np.float64).reshape(3)
    max_ray = (
        float(max_ray_distance_vox)
        if max_ray_distance_vox is not None
        else max(4.0, float(max_distance_vox) * 1.15)
    )
    max_anchor = (
        float(max_anchor_distance_vox)
        if max_anchor_distance_vox is not None
        else max(5.0, float(max_distance_vox) * 1.6)
    )
    hint_lbl = None
    if centerline_mask is not None:
        hint_lbl = centerline_label_near_click(centerline_mask, click, radius_vox=1.5)
        if hint_lbl is None:
            hint_lbl = centerline_label_at_voxel(centerline_mask, click)

    scored: list[tuple[float, CenterlinePick]] = []
    for lbl, pts in centerlines.items():
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 2 or pts_arr.shape[1] != 3:
            continue
        pick = _pick_label_candidates(
            click,
            pts_arr,
            int(lbl),
            max_distance_vox=max_distance_vox,
            max_ray_distance_vox=max_ray,
            max_anchor_distance_vox=max_anchor,
            ray_origin=ray_origin,
            ray_direction=ray_direction,
            use_view_line=use_view_line,
        )
        if pick is not None:
            scored.append((float(pick.distance_sq), pick))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    best_score, best = scored[0]
    if hint_lbl is None or len(scored) < 2:
        return best

    second_score = scored[1][0]
    if second_score > best_score * 2.5:
        return best

    hinted: CenterlinePick | None = None
    hinted_score = float("inf")
    for score, pick in scored:
        if int(pick.label) == int(hint_lbl) and score < hinted_score:
            hinted_score = score
            hinted = pick
    if hinted is None:
        return best
    if hinted_score <= best_score * 1.5:
        return hinted
    return best


def refine_pick_to_vertex_if_closer(
    pick: CenterlinePick,
    centerlines: Mapping[int, np.ndarray],
    click_xyz: np.ndarray,
    *,
    max_distance_vox: float = 2.5,
) -> CenterlinePick:
    """Prefer the nearest polyline vertex when the click is closer to it than the segment snap."""
    pts = centerlines.get(int(pick.label))
    if pts is None:
        return pick
    pts_arr = np.asarray(pts, dtype=np.float64)
    click = np.asarray(click_xyz, dtype=np.float64).reshape(3)
    pick_pt = np.asarray(pick.point, dtype=np.float64).reshape(3)
    d2_vertex = np.sum((pts_arr - click.reshape(1, 3)) ** 2, axis=1)
    vi = int(np.argmin(d2_vertex))
    max_d2 = float(max_distance_vox) ** 2
    if float(d2_vertex[vi]) > max_d2:
        return pick
    if float(d2_vertex[vi]) >= float(np.sum((pick_pt - click) ** 2)):
        return pick
    return CenterlinePick(
        label=int(pick.label),
        index=vi,
        point=pts_arr[vi].astype(np.float32, copy=False),
        distance_sq=float(d2_vertex[vi]),
    )


def smooth_polyline_display(
    points: np.ndarray,
    *,
    n_out: int | None = None,
) -> np.ndarray:
    """Optional spline resampling for smoother Tracks display (picking uses raw *points*)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return pts.astype(np.float32, copy=False)
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return pts.astype(np.float32, copy=False)
    try:
        tck, _ = splprep(pts.T, s=0, k=min(3, pts.shape[0] - 1))
        n = n_out if n_out is not None else max(pts.shape[0], int(pts.shape[0] * 1.5))
        n = max(4, int(n))
        u = np.linspace(0.0, 1.0, n)
        out = np.stack(splev(u, tck), axis=1)
        return out.astype(np.float32, copy=False)
    except Exception:
        return pts.astype(np.float32, copy=False)


__all__ = [
    "CenterlinePick",
    "frame_from_tangent",
    "nearest_centerline_along_ray",
    "nearest_centerline_on_segments",
    "nearest_centerline_point",
    "centerline_label_at_voxel",
    "centerline_label_near_click",
    "choose_plane_normal_sense",
    "pick_centerline",
    "refine_pick_to_vertex_if_closer",
    "smooth_polyline_display",
    "tangent_from_centerline",
    "tangent_window_indices",
    "unit_vector",
]
