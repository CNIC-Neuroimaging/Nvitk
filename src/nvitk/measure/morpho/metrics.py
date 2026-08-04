"""Shape, radius, tortuosity, curvature, torsion, and branchpoint metrics."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .geometry import arc_length, chord_length, cumulative_s, unit_vector
from .models import SkeletonTree, VesselInfo
from .skeleton import dijkstra_dist_from_root

def tortuosity_dm(pts: np.ndarray) -> float:
    """Distance-metric tortuosity: arc length / chord length (≥1; 1 = straight)."""
    L = arc_length(pts); C = chord_length(pts)
    return float(L / C) if np.isfinite(L) and np.isfinite(C) and C > 1e-8 else np.nan


def discrete_curvature(pts: np.ndarray) -> np.ndarray:
    """Menger curvature at each interior point of a 3-D polyline (NaN at the two endpoints)."""
    n = len(pts)
    kappa = np.full(n, np.nan)
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab = b - a; ac = c - a
        la = np.linalg.norm(ab); lac = np.linalg.norm(ac); lbc = np.linalg.norm(c - b)
        if la < 1e-8 or lac < 1e-8 or lbc < 1e-8:
            continue
        kappa[i] = np.linalg.norm(np.cross(ab, ac)) / (la * lac * lbc)
    return kappa


def discrete_torsion(pts: np.ndarray) -> np.ndarray:
    """Torsion of a 3-D polyline via arc-length derivatives (Frenet-frame formula, NaN where ill-conditioned)."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    torsion = np.full(n, np.nan)
    if n < 5:
        return torsion

    s = cumulative_s(pts)
    if len(np.unique(np.round(s, decimals=8))) < 5:
        return torsion

    try:
        r1 = np.gradient(pts, s, axis=0, edge_order=2)
        r2 = np.gradient(r1, s, axis=0, edge_order=2)
        r3 = np.gradient(r2, s, axis=0, edge_order=2)
    except Exception:
        return torsion

    cross12 = np.cross(r1, r2)
    denom = np.linalg.norm(cross12, axis=1) ** 2
    numer = np.einsum("ij,ij->i", cross12, r3)
    valid = np.isfinite(numer) & np.isfinite(denom) & (denom > 1e-12)
    torsion[valid] = numer[valid] / denom[valid]
    torsion[np.isfinite(denom) & (denom <= 1e-12)] = 0.0
    return torsion


def signed_turn_proxy(pts: np.ndarray) -> np.ndarray:
    """Per-point sign (+1/-1) of the local binormal relative to a median reference direction.

    Used to detect inflection points (sign flips) along a tortuous path.
    """
    n = len(pts); out = np.zeros(n)
    if n < 4:
        return out
    d = np.diff(pts, axis=0)
    dn = np.linalg.norm(d, axis=1, keepdims=True); dn[dn < 1e-12] = 1e-12
    t = d / dn
    b = np.cross(t[:-1], t[1:])
    bn = np.linalg.norm(b, axis=1, keepdims=True); bn[bn < 1e-12] = 1e-12
    b_unit = b / bn
    ref = np.median(b_unit, axis=0)
    rn = np.linalg.norm(ref)
    ref = ref / rn if rn >= 1e-8 else np.array([0.0, 0.0, 1.0])
    sgn = np.sign(b_unit @ ref)
    out[1:-1] = sgn; out[0] = out[1]; out[-1] = out[-2]
    return out


def smooth_1d(x: np.ndarray, win: int = 7) -> np.ndarray:
    """Reflect-padded moving-average smoothing of a 1-D signal with an odd window."""
    if win < 3:
        return x.copy()
    win = int(win) | 1
    kernel = np.ones(win) / win
    return np.convolve(np.pad(x, win // 2, mode="reflect"), kernel, mode="valid")


def inflection_count(kappa, turn_proxy, kappa_min, smooth_win) -> int:
    """Count sign changes of :func:`signed_turn_proxy` at points with curvature ≥ *kappa_min*."""
    k = smooth_1d(np.where(np.isfinite(kappa), kappa, 0.0), win=smooth_win)
    tp = np.where(np.isfinite(turn_proxy), turn_proxy, 0.0)
    valid = (k >= kappa_min) & (tp != 0)
    if valid.sum() < 3:
        return 0
    seq = tp[valid]
    comp = [seq[0]]
    for v in seq[1:]:
        if v != comp[-1]:
            comp.append(v)
    return max(0, len(comp) - 1)


def bend_peak_count(kappa, kappa_peak, smooth_win) -> int:
    """Count local curvature maxima at or above *kappa_peak* (smoothed curvature signal)."""
    k = smooth_1d(np.where(np.isfinite(kappa), kappa, 0.0), win=smooth_win)
    if len(k) < 3:
        return 0
    return sum(1 for i in range(1, len(k) - 1) if k[i] > k[i - 1] and k[i] >= k[i + 1] and k[i] >= kappa_peak)


def radius_from_edt(mask_bool, spacing, path_vox) -> np.ndarray:
    """Vessel radius (mm) at each path voxel, sampled from a spacing-aware distance transform."""
    dist = ndi.distance_transform_edt(mask_bool, sampling=spacing)
    return np.array([dist[i, j, k] for (i, j, k) in path_vox], dtype=float)


def radius_stats(r: np.ndarray) -> dict:
    """Summary radius statistics (mean/std/CV/min/percentiles) over finite values of *r*."""
    rr = r[np.isfinite(r)]
    if len(rr) == 0:
        return {"radius_mean_mm": np.nan, "radius_std_mm": np.nan, "radius_cv": np.nan, "radius_min_mm": np.nan, "radius_p05_mm": np.nan, "radius_p50_mm": np.nan, "radius_p95_mm": np.nan}
    mu = float(np.mean(rr)); sd = float(np.std(rr))
    return {
        "radius_mean_mm": mu,
        "radius_std_mm": sd,
        "radius_cv": float(sd / mu) if mu > 1e-8 else np.nan,
        "radius_min_mm": float(np.min(rr)),
        "radius_p05_mm": float(np.percentile(rr, 5)),
        "radius_p50_mm": float(np.percentile(rr, 50)),
        "radius_p95_mm": float(np.percentile(rr, 95)),
    }


def taper_slope(s, r) -> float:
    """Linear-fit slope of radius *r* vs arc length *s* (mm radius change per mm length)."""
    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 3:
        return np.nan
    a, _ = np.polyfit(s[mask], r[mask], 1)
    return float(a)


def vector_from_node_along_neighbor(tree: SkeletonTree, node: int, nbr: int, spacing, n_steps: int = 5) -> Optional[np.ndarray]:
    """Estimate local branch direction from a branchpoint along a neighbor chain."""
    spacing = np.asarray(spacing, dtype=float)
    prev = node
    cur = nbr
    pts = [tree.pts_vox[node].astype(float) * spacing]
    for _ in range(n_steps):
        pts.append(tree.pts_vox[cur].astype(float) * spacing)
        nexts = [x for x in tree.neighbors[cur] if x != prev]
        if len(nexts) != 1:
            break
        prev, cur = cur, nexts[0]
    v = pts[-1] - pts[0]
    return unit_vector(v)


def compute_branchpoint_metrics(label: int, component_id: int, tree: SkeletonTree, mask_bool: np.ndarray, spacing, vessel_info: VesselInfo) -> pd.DataFrame:
    """Per-branchpoint table: parent/daughter EDT radii, area-conservation ratio, and daughter-pair angles."""
    if tree.root is None:
        return pd.DataFrame()
    spacing = np.asarray(spacing, dtype=float)
    dist_root = tree.dist_from_root_mm if tree.dist_from_root_mm is not None else dijkstra_dist_from_root(tree, tree.root, spacing)
    edt = ndi.distance_transform_edt(mask_bool, sampling=spacing)
    rows = []
    for bp in tree.branchpoints:
        bp_mm = tree.pts_vox[bp].astype(float) * spacing
        parent_nbrs = [n for n in tree.neighbors[bp] if dist_root[n] < dist_root[bp]]
        daughter_nbrs = [n for n in tree.neighbors[bp] if dist_root[n] > dist_root[bp]]
        if not daughter_nbrs:
            continue
        parent_nbr = parent_nbrs[int(np.argmin([dist_root[x] for x in parent_nbrs]))] if parent_nbrs else None
        parent_radius = float(edt[tuple(map(int, tree.pts_vox[bp]))])
        if parent_nbr is not None:
            parent_probe = tree.pts_vox[parent_nbr]
            parent_radius = float(edt[tuple(map(int, parent_probe))])
        daughter_radii = []
        daughter_vectors = []
        for dn in daughter_nbrs:
            daughter_radii.append(float(edt[tuple(map(int, tree.pts_vox[dn]))]))
            v = vector_from_node_along_neighbor(tree, bp, dn, spacing, n_steps=5)
            if v is not None:
                daughter_vectors.append(v)
        angles = []
        for i in range(len(daughter_vectors)):
            for j in range(i + 1, len(daughter_vectors)):
                dot = float(np.clip(np.dot(daughter_vectors[i], daughter_vectors[j]), -1.0, 1.0))
                angles.append(float(np.degrees(np.arccos(dot))))
        daughter_radii = np.asarray(daughter_radii, dtype=float)
        area_ratio = float(np.nansum(daughter_radii ** 2) / (parent_radius ** 2)) if parent_radius > 1e-8 else np.nan
        rows.append({
            "label": int(label), "component_id": int(component_id), "vessel_name": vessel_info.name,
            "branchpoint_index": int(bp), "x_mm": float(bp_mm[0]), "y_mm": float(bp_mm[1]), "z_mm": float(bp_mm[2]),
            "degree": int(tree.degree[bp]), "n_daughters": int(len(daughter_nbrs)),
            "parent_radius_edt_mm": float(parent_radius),
            "daughter_radii_edt_mm_json": json.dumps([float(x) for x in daughter_radii]),
            "daughter_parent_radius_ratios_json": json.dumps([float(x / parent_radius) if parent_radius > 1e-8 else None for x in daughter_radii]),
            "daughter_area_ratio_sum_r2_over_parent_r2": area_ratio,
            "daughter_pair_angles_deg_json": json.dumps(angles),
            "daughter_pair_angle_min_deg": float(np.nanmin(angles)) if angles else np.nan,
            "daughter_pair_angle_max_deg": float(np.nanmax(angles)) if angles else np.nan,
        })
    return pd.DataFrame(rows)
