"""LOC (location of interest) selection heuristics aligned with QVTplus.

**Inputs**

- Arterial / venous centerline polylines, optional venous slab mask, contrast volumes.

**Outputs**

- :class:`LocRecord` rows (voxel, tangent, cross-section metrics) for ``locs.csv`` (stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import centerline_tangents
from nvitk.measure.cross_section import CrossSectionResult, segment_at_point
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_BASILAR,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_MCA_IDS,
    QVTPY_PCA_IDS,
    QVTPY_RACA,
    QVTPY_RICA,
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRV_REF = np.array([0.0, 1.0, 1.0], dtype=np.float64)

# Dual init/fin LOCs at ⅓ and ⅔ arc length (empty: ACAs use junction/CW stations).
QVTPY_DUAL_LOC_ARTERIAL_IDS: frozenset[int] = frozenset()
# MCA / PCA: dual init/fin LOCs at bifurcation-based stations (see pick_mca_pca_loc_indices).
QVTPY_MCA_PCA_BIFURCATION_LOC_IDS: frozenset[int] = frozenset(
    QVTPY_MCA_IDS | QVTPY_PCA_IDS
)
# ACA: init proximal to AComm (or CW eICAB mid), fin distal of junction / CW.
QVTPY_ACA_JUNCTION_LOC_IDS: frozenset[int] = frozenset(QVTPY_ACA_IDS)
# L/R ICA + basilar: one LOC each, ideally on a common axial (z) plane at max axial alignment.
QVTPY_ICA_BASILAR_SINGLE_LOC_IDS: frozenset[int] = frozenset(
    {QVTPY_LICA, QVTPY_RICA, QVTPY_BASILAR}
)
_ACA_PARENT_ICA: dict[int, int] = {
    int(QVTPY_LACA): int(QVTPY_LICA),
    int(QVTPY_RACA): int(QVTPY_RICA),
}


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocRecord:
    """One LOC: vessel id, polyline station, voxel index, tangent, optional cross-section QC."""

    vessel_id: int
    vessel_name: str
    segment_id: int
    centerline_index: int
    i: int
    j: int
    k: int
    centerline_x: float
    centerline_y: float
    centerline_z: float
    tangent_x: float
    tangent_y: float
    tangent_z: float
    loc_circularity: float = 0.0
    loc_cross_section_area_mm2: float = 0.0
    loc_role: str = "mid"


# ---------------------------------------------------------------------------
# Polyline utilities
# ---------------------------------------------------------------------------


def split_into_parts(points: np.ndarray, n_parts: int) -> list[np.ndarray]:
    pts = to_numpy(points)
    n = pts.shape[0]
    if n == 0:
        return []
    parts: list[np.ndarray] = []
    edges = np.linspace(0, n, int(n_parts) + 1, dtype=int)
    for p in range(int(n_parts)):
        seg = pts[edges[p] : edges[p + 1]]
        if seg.shape[0] > 0:
            parts.append(seg)
    return parts


def pick_equal_section_boundary_indices(n: int, n_sections: int) -> list[int]:
    """Return ``n_sections - 1`` boundary indices between equal-length index groups."""
    if n < 1 or n_sections < 2:
        return []
    edges = np.linspace(0, n, int(n_sections) + 1, dtype=int)
    boundaries = [int(edges[i]) for i in range(1, int(n_sections))]
    return [min(max(0, b), n - 1) for b in boundaries]


def polyline_cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc length along *points* (N,3), starting at 0."""
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def pick_index_at_arc_fraction(points: np.ndarray, frac: float) -> int:
    """Polyline index closest to *frac* of total arc length (0=start, 1=end)."""
    pts = to_numpy(points)
    n = pts.shape[0]
    if n < 1:
        return 0
    cum = polyline_cumulative_arc_length(pts)
    total = float(cum[-1])
    if total <= 0.0:
        return int(n // 2)
    target = float(np.clip(frac, 0.0, 1.0)) * total
    idx = int(np.searchsorted(cum, target, side="left"))
    return int(np.clip(idx, 0, n - 1))


def pick_index_arc_midpoint_between(points: np.ndarray, i0: int, i1: int) -> int:
    """Polyline index at the arc-length midpoint between indices *i0* and *i1*."""
    pts = to_numpy(points)
    n = pts.shape[0]
    if n < 1:
        return 0
    ia = int(np.clip(min(i0, i1), 0, n - 1))
    ib = int(np.clip(max(i0, i1), 0, n - 1))
    if ia == ib:
        return ia
    cum = polyline_cumulative_arc_length(pts)
    target = 0.5 * (float(cum[ia]) + float(cum[ib]))
    idx = int(np.searchsorted(cum, target, side="left"))
    return int(np.clip(idx, ia, ib))


def _voxel_tuple(row: np.ndarray) -> tuple[int, int, int]:
    return (
        int(round(float(row[0]))),
        int(round(float(row[1]))),
        int(round(float(row[2]))),
    )


def _nearest_polyline_index(
    points: np.ndarray,
    xyz: np.ndarray,
    *,
    max_dist_vox: float = 3.0,
) -> int | None:
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 1:
        return None
    d = np.linalg.norm(pts - to_numpy(xyz).reshape(1, 3), axis=1)
    i = int(np.argmin(d))
    if float(d[i]) > float(max_dist_vox):
        return None
    return i


def _skeleton_coords_from_seg(seg: np.ndarray, label_id: int) -> np.ndarray:
    from nvitk.morphology.centerline import skeletonize_binary

    roi = to_numpy(seg) == int(label_id)
    if not roi.any():
        return np.zeros((0, 3), dtype=np.float32)
    sk = to_numpy(skeletonize_binary(roi)) > 0
    return np.argwhere(sk).astype(np.float32)


def _chain_arc_length(path: list[tuple[int, int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    pts = np.asarray(path, dtype=np.float64)
    return float(polyline_cumulative_arc_length(pts)[-1])


def _longest_downstream_chain_from_junction(
    j1: tuple[int, int, int],
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]],
    deg: dict[tuple[int, int, int], int],
    proximal_neighbor: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    branches: list[list[tuple[int, int, int]]] = []
    for n in adj.get(j1, []):
        if n == proximal_neighbor:
            continue
        path: list[tuple[int, int, int]] = [j1, n]
        prev, cur = j1, n
        while deg.get(cur, 0) == 2:
            nbrs = [x for x in adj.get(cur, []) if x != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            path.append(nxt)
            prev, cur = cur, nxt
        branches.append(path)
    if not branches:
        return [j1]
    return max(branches, key=_chain_arc_length)


def _first_junction_or_endpoint_on_chain(
    chain: list[tuple[int, int, int]],
    deg: dict[tuple[int, int, int], int],
    *,
    min_degree: int = 3,
) -> tuple[int, int, int]:
    md = int(min_degree)
    for v in chain[1:]:
        if deg.get(v, 0) >= md:
            return v
    return chain[-1]


def _proximal_neighbor_at_junction(
    points: np.ndarray,
    j1_idx: int,
    j1_voxel: tuple[int, int, int],
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> tuple[int, int, int] | None:
    neighbors = adj.get(j1_voxel, [])
    if not neighbors:
        return None
    pts = to_numpy(points).astype(np.float64)
    if j1_idx > 0:
        ref = pts[j1_idx - 1]
    else:
        ref = pts[0]
    dists = [float(np.linalg.norm(np.asarray(n, dtype=np.float64) - ref)) for n in neighbors]
    return neighbors[int(np.argmin(dists))]


def pick_mca_pca_loc_indices(
    points: np.ndarray,
    *,
    seg: np.ndarray | None = None,
    label_id: int | None = None,
    junction_max_dist_vox: float = 3.0,
) -> tuple[int, int | None]:
    """Init/fin LOCs for MCA/PCA from skeleton bifurcations.

  * ≥1 bifurcation — init at arc midpoint start → first bifurcation; fin at arc
    midpoint first bifurcation → distal target on the longest downstream branch
    (first junction on that branch, or branch endpoint).
  * No bifurcation — init only at ¼ arc length from the vessel start; fin is ``None``.
    """
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    if n < 1:
        return 0, None

    junction_poly_indices: list[int] = []
    sk_adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] | None = None
    sk_deg: dict[tuple[int, int, int], int] | None = None

    if seg is not None and label_id is not None:
        from nvitk.morphology.polyline_graph import collapse_junction_clusters, skeleton_graph

        coords = _skeleton_coords_from_seg(seg, int(label_id))
        if coords.shape[0] >= 3:
            _nodes, sk_adj, sk_deg = skeleton_graph(coords)
            junction_reps = collapse_junction_clusters(_nodes, sk_deg, min_degree=3)
            for jv in junction_reps:
                ji = _nearest_polyline_index(
                    pts,
                    np.asarray(jv, dtype=np.float64),
                    max_dist_vox=junction_max_dist_vox,
                )
                if ji is not None and ji > 0:
                    junction_poly_indices.append(int(ji))
            junction_poly_indices = sorted(set(junction_poly_indices))

    if not junction_poly_indices:
        return pick_index_at_arc_fraction(pts, 0.25), None

    j1_idx = int(junction_poly_indices[0])
    init_idx = pick_index_arc_midpoint_between(pts, 0, j1_idx)

    j2_idx = n - 1
    if sk_adj is not None and sk_deg is not None:
        j1_voxel = _voxel_tuple(pts[j1_idx])
        if j1_voxel in sk_adj:
            prox = _proximal_neighbor_at_junction(pts, j1_idx, j1_voxel, sk_adj)
            if prox is not None:
                downstream = _longest_downstream_chain_from_junction(
                    j1_voxel, sk_adj, sk_deg, prox
                )
                distal = _first_junction_or_endpoint_on_chain(downstream, sk_deg)
                mapped = _nearest_polyline_index(
                    pts,
                    np.asarray(distal, dtype=np.float64),
                    max_dist_vox=junction_max_dist_vox,
                )
                if mapped is not None and mapped > j1_idx:
                    j2_idx = int(mapped)

    fin_idx = pick_index_arc_midpoint_between(pts, j1_idx, j2_idx)
    if fin_idx <= init_idx:
        fin_idx = int(np.clip(j1_idx + 1, 0, n - 1))
    return init_idx, fin_idx


def _nearest_polyline_index_unbounded(
    points: np.ndarray,
    target: np.ndarray,
) -> int:
    pts = to_numpy(points).astype(np.float64)
    t = np.asarray(target, dtype=np.float64).reshape(3)
    d2 = np.sum((pts - t) ** 2, axis=1)
    return int(np.argmin(d2))


def _aca_polyline_proximal_is_low_index(
    points: np.ndarray,
    *,
    eicab_qvtpy: np.ndarray | None,
    label_id: int,
) -> bool:
    """True when polyline index 0 is closer to the parent ICA than the distal end."""
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 2:
        return True
    parent = _ACA_PARENT_ICA.get(int(label_id))
    if parent is None or eicab_qvtpy is None:
        return True
    eq = to_numpy(eicab_qvtpy).astype(np.int32, copy=False)
    ica = np.argwhere(eq == int(parent))
    if ica.size == 0:
        return True
    com = ica.astype(np.float64).mean(axis=0)
    d0 = float(np.linalg.norm(pts[0] - com))
    d1 = float(np.linalg.norm(pts[-1] - com))
    return d0 <= d1


def _aca_cw_mask_indices(
    points: np.ndarray,
    eicab_qvtpy: np.ndarray,
    label_id: int,
) -> list[int]:
    """Polyline indices whose rounded voxel lies in the CW eICAB ACA label."""
    pts = to_numpy(points).astype(np.float64)
    eq = to_numpy(eicab_qvtpy).astype(np.int32, copy=False)
    out: list[int] = []
    for i in range(pts.shape[0]):
        ii, jj, kk = (
            int(round(float(pts[i, 0]))),
            int(round(float(pts[i, 1]))),
            int(round(float(pts[i, 2]))),
        )
        if (
            0 <= ii < eq.shape[0]
            and 0 <= jj < eq.shape[1]
            and 0 <= kk < eq.shape[2]
            and int(eq[ii, jj, kk]) == int(label_id)
        ):
            out.append(i)
    return out


def pick_aca_loc_indices(
    points: np.ndarray,
    *,
    label_id: int,
    eicab_qvtpy: np.ndarray | None = None,
    junction_ijk: tuple[int, int, int] | None = None,
    junction_max_dist_vox: float = 8.0,
) -> tuple[int, int | None, str]:
    """Init/fin LOCs for ACA relative to AComm junction or CW eICAB coverage.

    Indices are returned in the original polyline order. Init is the proximal
    station (A1 / CW mid); fin is distal of the junction / CW section.

    * Junction present — init = arc midpoint proximal→junction; fin = junction→distal.
    * No junction — init = arc midpoint of CW eICAB ACA section; fin = mid of distal
      remainder. Falls back to tertiles / single mid.
    """
    pts_orig = to_numpy(points).astype(np.float64)
    n = pts_orig.shape[0]
    if n < 1:
        return 0, None, "empty"
    if n < 3:
        return 0, None, "too_short"

    proximal_low = _aca_polyline_proximal_is_low_index(
        pts_orig, eicab_qvtpy=eicab_qvtpy, label_id=int(label_id)
    )
    # Work in proximal-first index space, then map back.
    pts = pts_orig if proximal_low else pts_orig[::-1].copy()

    def _to_orig(i: int) -> int:
        return int(i) if proximal_low else int(n - 1 - i)

    j_work: int | None = None
    method = "fallback_tertiles"

    if junction_ijk is not None:
        ji = _nearest_polyline_index(
            pts,
            np.asarray(junction_ijk, dtype=np.float64),
            max_dist_vox=float(junction_max_dist_vox),
        )
        if ji is None:
            ji = _nearest_polyline_index_unbounded(
                pts, np.asarray(junction_ijk, dtype=np.float64)
            )
        if 0 < int(ji) < n - 1:
            j_work = int(ji)
            method = "acomm_junction"

    if j_work is not None:
        init_w = pick_index_arc_midpoint_between(pts, 0, j_work)
        fin_w = pick_index_arc_midpoint_between(pts, j_work, n - 1)
        if fin_w <= init_w:
            fin_w = int(np.clip(j_work + 1, 0, n - 1))
        return _to_orig(init_w), _to_orig(fin_w), method

    if eicab_qvtpy is not None:
        cw_idx = _aca_cw_mask_indices(pts, eicab_qvtpy, int(label_id))
        if len(cw_idx) >= 1:
            prox_lo, prox_hi = int(cw_idx[0]), int(cw_idx[-1])
            init_w = pick_index_arc_midpoint_between(pts, prox_lo, prox_hi)
            if prox_hi < n - 2:
                fin_w = pick_index_arc_midpoint_between(pts, prox_hi, n - 1)
                if fin_w != init_w:
                    return _to_orig(init_w), _to_orig(fin_w), "cw_eicab_mid"
            return _to_orig(init_w), None, "cw_eicab_mid_only"

    dual = pick_dual_loc_indices(n, pts)
    if dual is None:
        return _to_orig(pick_mid_loc_index(n, pts)), None, "mid_fallback"
    return _to_orig(dual[0]), _to_orig(dual[1]), method


def _infer_aca_junction_ijk(
    eicab_qvtpy: np.ndarray | None,
) -> tuple[int, int, int] | None:
    """AComm COM snapped to nearest eICAB AComm voxel (qvtpy label space)."""
    if eicab_qvtpy is None:
        return None
    eq = to_numpy(eicab_qvtpy).astype(np.int32, copy=False)
    coords = np.argwhere(eq == int(QVTPY_ACOMM))
    if coords.size == 0:
        return None
    coords_f = coords.astype(np.float64)
    com = coords_f.mean(axis=0)
    d2 = np.sum((coords_f - com) ** 2, axis=1)
    best = coords[int(np.argmin(d2))]
    return (int(best[0]), int(best[1]), int(best[2]))


def pick_axial_alignment_index(
    points: np.ndarray,
    *,
    proximal_frac: float = 0.65,
) -> int:
    """Polyline index with strongest alignment to the superior (z) axis in the proximal segment."""
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    if n < 1:
        return 0
    tangents = to_numpy(centerline_tangents(pts, k_half=2)).astype(np.float64)
    limit = max(3, int(round(float(proximal_frac) * n)))
    scores = np.abs(tangents[:limit, 2])
    return int(np.argmax(scores))


def pick_index_near_z(
    points: np.ndarray,
    target_z: float,
    *,
    candidate_indices: np.ndarray | None = None,
) -> int:
    """Index whose z coordinate is closest to *target_z* (optionally restricted to candidates)."""
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    if n < 1:
        return 0
    if candidate_indices is None:
        pool = np.arange(n, dtype=np.int32)
    else:
        pool = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        pool = pool[(pool >= 0) & (pool < n)]
        if pool.size == 0:
            pool = np.arange(n, dtype=np.int32)
    z = pts[pool, 2]
    return int(pool[int(np.argmin(np.abs(z - float(target_z))))])


def select_ica_basilar_aligned_loc_indices(
    arterial_polylines: dict[int, np.ndarray],
    *,
    proximal_frac: float = 0.65,
    min_axial_score: float = 0.55,
) -> dict[int, int]:
    """Single LOC index per ICA/basilar vessel on a shared axial (z) plane.

    Picks the proximal station of strongest z-axis alignment on each vessel, then
    refines all three to the median z among those stations (before the ICA siphon).
    """
    present = [int(v) for v in QVTPY_ICA_BASILAR_SINGLE_LOC_IDS if int(v) in arterial_polylines]
    if not present:
        return {}

    anchor_idx: dict[int, int] = {}
    z_vals: list[float] = []
    for vid in present:
        pts = to_numpy(arterial_polylines[vid]).astype(np.float64)
        idx = pick_axial_alignment_index(pts, proximal_frac=proximal_frac)
        anchor_idx[vid] = idx
        z_vals.append(float(pts[idx, 2]))

    shared_z = float(np.median(z_vals))
    out: dict[int, int] = {}
    for vid in present:
        pts = to_numpy(arterial_polylines[vid]).astype(np.float64)
        tangents = to_numpy(centerline_tangents(pts, k_half=2)).astype(np.float64)
        limit = max(3, int(round(float(proximal_frac) * pts.shape[0])))
        axial = np.abs(tangents[:limit, 2])
        good = np.where(axial >= float(min_axial_score))[0]
        if good.size == 0:
            good = np.array([anchor_idx[vid]], dtype=np.int32)
        out[vid] = pick_index_near_z(pts, shared_z, candidate_indices=good)
    return out


def pick_dual_loc_indices(n: int, points: np.ndarray | None = None) -> tuple[int, int] | None:
    """Init/fin LOCs at ⅓ and ⅔ arc length (falls back to equal index tertiles)."""
    if points is not None:
        pts = to_numpy(points)
        if pts.shape[0] >= 3:
            init_idx = pick_index_at_arc_fraction(pts, 1.0 / 3.0)
            fin_idx = pick_index_at_arc_fraction(pts, 2.0 / 3.0)
            if init_idx < fin_idx:
                return init_idx, fin_idx
    if n < 3:
        return None
    bounds = pick_equal_section_boundary_indices(n, 3)
    if len(bounds) != 2:
        return None
    init_idx, fin_idx = bounds[0], bounds[1]
    if init_idx >= fin_idx:
        return None
    return init_idx, fin_idx


def pick_mid_loc_index(n: int, points: np.ndarray | None = None) -> int:
    """Mid LOC at 50% arc length (falls back to equal index halves)."""
    if points is not None:
        pts = to_numpy(points)
        if pts.shape[0] >= 1:
            return pick_index_at_arc_fraction(pts, 0.5)
    if n < 1:
        return 0
    bounds = pick_equal_section_boundary_indices(n, 2)
    if bounds:
        return int(bounds[0])
    return n // 2


def pick_endpoint_indices(
    n: int,
    *,
    inset_frac: float = 0.08,
    min_inset_pts: int = 5,
) -> tuple[int, int] | None:
    """Legacy inset endpoints (superseded by :func:`pick_dual_loc_indices`)."""
    del inset_frac, min_inset_pts
    return pick_dual_loc_indices(n)


def _vertex_in_mask(row: np.ndarray, mask: np.ndarray) -> bool:
    i, j, k = int(round(float(row[0]))), int(round(float(row[1]))), int(round(float(row[2])))
    return (
        0 <= i < mask.shape[0]
        and 0 <= j < mask.shape[1]
        and 0 <= k < mask.shape[2]
        and bool(mask[i, j, k])
    )


def pick_arc_midpoint_in_mask(
    points: np.ndarray,
    mask: np.ndarray | None,
    *,
    frac: float = 0.5,
) -> int:
    """Arc-length station along *points*; segments count when either endpoint is in *mask*."""
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    if n < 1:
        return 0
    if mask is None:
        return pick_index_at_arc_fraction(pts, frac)

    m = to_numpy(mask).astype(bool)
    in_mask = np.array([_vertex_in_mask(pts[i], m) for i in range(n)], dtype=bool)
    if not in_mask.any():
        return pick_index_at_arc_fraction(pts, frac)

    cum = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        seg_len = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if in_mask[i - 1] or in_mask[i]:
            cum[i] = cum[i - 1] + seg_len
        else:
            cum[i] = cum[i - 1]
    total = float(cum[-1])
    if total <= 0.0:
        return pick_index_at_arc_fraction(pts, frac)
    target = float(np.clip(frac, 0.0, 1.0)) * total
    idx = int(np.searchsorted(cum, target, side="left"))
    return int(np.clip(idx, 0, n - 1))


def pick_masked_midpoint(
    points: np.ndarray,
    mask: np.ndarray | None,
) -> int:
    """Venous-style midpoint at 50% arc length (optionally weighted by *mask*)."""
    return pick_arc_midpoint_in_mask(points, mask, frac=0.5)


def local_direction_alignment(points: np.ndarray, idx: int, *, window: int = 5) -> np.ndarray:
    """Unit principal direction along the polyline near index *idx* (SVD over a local window)."""
    pts = to_numpy(points).astype(np.float64)
    n = pts.shape[0]
    i0 = max(0, idx - window)
    i1 = min(n - 1, idx + window)
    seg = pts[i0 : i1 + 1]
    if seg.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = seg - np.mean(seg, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    return d / (float(np.linalg.norm(d)) + 1e-12)


def _z_std(points: np.ndarray) -> float:
    return float(np.std(to_numpy(points)[:, 2]))


# ---------------------------------------------------------------------------
# LocRecord construction + cross-section at station
# ---------------------------------------------------------------------------


def _record_from_polyline(
    points: np.ndarray,
    idx: int,
    *,
    vessel_id: int,
    vessel_name: str,
    segment_id: int = 0,
    loc_role: str = "mid",
    xs: CrossSectionResult | None = None,
) -> LocRecord:
    pts = to_numpy(points)
    idx = int(np.clip(idx, 0, pts.shape[0] - 1))
    tangents = centerline_tangents(pts, k_half=2)
    cx, cy, cz = float(pts[idx, 0]), float(pts[idx, 1]), float(pts[idx, 2])
    tx, ty, tz = float(tangents[idx, 0]), float(tangents[idx, 1]), float(tangents[idx, 2])
    return LocRecord(
        vessel_id=int(vessel_id),
        vessel_name=str(vessel_name),
        segment_id=int(segment_id),
        centerline_index=int(idx),
        i=int(round(cx)),
        j=int(round(cy)),
        k=int(round(cz)),
        centerline_x=cx,
        centerline_y=cy,
        centerline_z=cz,
        tangent_x=tx,
        tangent_y=ty,
        tangent_z=tz,
        loc_circularity=float(xs.circularity) if xs else 0.0,
        loc_cross_section_area_mm2=float(xs.area_mm2) if xs else 0.0,
        loc_role=str(loc_role),
    )


def _cross_section_at(
    points: np.ndarray,
    idx: int,
    *,
    mag: np.ndarray | None,
    cd: np.ndarray | None,
    vel_mag: np.ndarray | None,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float,
) -> CrossSectionResult | None:
    if mag is None or cd is None or vel_mag is None:
        return None
    pts = to_numpy(points)
    tangents = centerline_tangents(pts, k_half=2)
    idx = int(np.clip(idx, 0, pts.shape[0] - 1))
    return segment_at_point(
        pts[idx],
        tangents[idx],
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
    )


def select_main_vessel_loc(
    points: np.ndarray,
    *,
    vessel_id: int,
    vessel_name: str,
    mask: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
) -> LocRecord | None:
    """Single midpoint LOC on a polyline (masked arc-length when *mask* is set)."""
    pts = to_numpy(points)
    if pts.shape[0] < 3:
        return None
    idx = pick_masked_midpoint(pts, mask)
    xs = _cross_section_at(
        pts,
        idx,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
    )
    return _record_from_polyline(pts, idx, vessel_id=vessel_id, vessel_name=vessel_name, xs=xs)


# ---------------------------------------------------------------------------
# Venous LOC heuristics (SSSV / STRV / LTSV / RTSV)
# ---------------------------------------------------------------------------


def resolve_sssv_strv(
    sssv_pts: np.ndarray,
    strv_pts: np.ndarray,
    *,
    mask: np.ndarray | None,
) -> tuple[int, int]:
    """Return (sssv_idx, strv_idx) at arc-length midpoints (swap validated downstream)."""
    return (
        pick_arc_midpoint_in_mask(sssv_pts, mask, frac=0.5),
        pick_arc_midpoint_in_mask(strv_pts, mask, frac=0.5),
    )


def validate_sssv_strv_swap(
    sssv_pts: np.ndarray,
    strv_pts: np.ndarray,
    sssv_idx: int,
    strv_idx: int,
) -> tuple[int, int]:
    d_sssv = local_direction_alignment(sssv_pts, sssv_idx)
    d_strv = local_direction_alignment(strv_pts, strv_idx)
    ref = _STRV_REF / (float(np.linalg.norm(_STRV_REF)) + 1e-12)
    align_sssv = abs(float(np.dot(d_sssv, ref)))
    align_strv = abs(float(np.dot(d_strv, ref)))
    if align_sssv > align_strv:
        return strv_idx, sssv_idx
    return sssv_idx, strv_idx


def resolve_long_venous_segment(
    long_pts: np.ndarray,
    *,
    mask: np.ndarray | None,
    vertical: bool,
) -> int:
    """Arc-length midpoint (``vertical`` kept for API compatibility)."""
    del vertical
    return pick_arc_midpoint_in_mask(long_pts, mask, frac=0.5)


def select_venous_locs(
    venous_polylines: dict[str, np.ndarray],
    *,
    venous_mask: np.ndarray | None = None,
    name_to_id: dict[str, int] | None = None,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
) -> list[LocRecord]:
    """One LOC per venous vessel (SSSV/STRV/LTSV/RTSV) with geometry validation."""
    from nvitk.pipes.qvtpy.util.venous_heuristics import venous_name_to_label_id

    if not venous_polylines:
        return []

    indices: dict[str, int] = {}
    for name, pts in venous_polylines.items():
        if pts.shape[0] < 3:
            continue
        indices[name] = pick_arc_midpoint_in_mask(pts, venous_mask, frac=0.5)

    if NAME_SSSV in indices and NAME_STRV in indices and NAME_SSSV in venous_polylines and NAME_STRV in venous_polylines:
        si, ti = resolve_sssv_strv(
            venous_polylines[NAME_SSSV],
            venous_polylines[NAME_STRV],
            mask=venous_mask,
        )
        si, ti = validate_sssv_strv_swap(
            venous_polylines[NAME_SSSV],
            venous_polylines[NAME_STRV],
            si,
            ti,
        )
        indices[NAME_SSSV] = si
        indices[NAME_STRV] = ti

    for lat in (NAME_LTSV, NAME_RTSV):
        if lat in venous_polylines and NAME_SSSV in venous_polylines and lat in indices:
            s_pts = venous_polylines[NAME_SSSV]
            l_pts = venous_polylines[lat]
            if s_pts.shape[0] > 0 and l_pts.shape[0] > 0:
                if np.linalg.norm(np.mean(s_pts, axis=0) - np.mean(l_pts, axis=0)) < 5.0:
                    indices[lat] = resolve_long_venous_segment(
                        l_pts, mask=venous_mask, vertical=False
                    )

    out: list[LocRecord] = []
    for name, idx in indices.items():
        pts = venous_polylines[name]
        vid = venous_name_to_label_id(name, name_to_id)
        xs = _cross_section_at(
            pts,
            idx,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
        )
        out.append(
            _record_from_polyline(
                pts,
                idx,
                vessel_id=vid,
                vessel_name=name,
                xs=xs,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Combined arterial / venous LOC selection
# ---------------------------------------------------------------------------


def _arterial_loc_at_index(
    pts: np.ndarray,
    idx: int,
    *,
    vessel_id: int,
    vessel_name: str,
    segment_id: int,
    loc_role: str,
    mag: np.ndarray | None,
    cd: np.ndarray | None,
    vel_mag: np.ndarray | None,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float,
) -> LocRecord:
    xs = _cross_section_at(
        pts,
        idx,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
    )
    return _record_from_polyline(
        pts,
        idx,
        vessel_id=vessel_id,
        vessel_name=vessel_name,
        segment_id=segment_id,
        loc_role=loc_role,
        xs=xs,
    )


def _normalize_arterial_branches(
    arterial: dict[int, Any],
) -> dict[int, list[tuple[str, np.ndarray]]]:
    """Accept either the legacy single-polyline dict or the named-branch dict.

    Legacy ``{label: (N,3)}`` is wrapped as a single trunk named by the vessel;
    the branched ``{label: [(name, pts), ...]}`` form is passed through.
    """
    from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

    out: dict[int, list[tuple[str, np.ndarray]]] = {}
    for lid, val in arterial.items():
        if isinstance(val, list):
            out[int(lid)] = [(str(n), p) for n, p in val]
        else:
            out[int(lid)] = [(qvtpy_vessel_name(int(lid)), val)]
    return out


def select_arterial_locs(
    arterial_polylines: dict[int, Any],
    *,
    venous_mask: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    strategy: str = "qvtpy",
    endpoint_inset_frac: float = 0.08,
    arterial_seg: np.ndarray | None = None,
    eicab_qvtpy: np.ndarray | None = None,
) -> tuple[list[LocRecord], dict[str, Any]]:
    """Select arterial LOCs. Returns ``(records, meta_extra)`` for loc_meta.json.

    ``arterial_polylines`` may be the named-branch mapping
    ``{parent_label: [(branch_name, pts), ...]}`` (preferred) or the legacy
    ``{label: pts}`` single-polyline dict.

    LOC policy (``strategy="qvtpy"``):

    - ICA / basilar: one shared-z mid LOC on the trunk
    - MCA / PCA: exactly two LOCs (init/fin) via bifurcation arc midpoints on the trunk
    - ACA: exactly two LOCs (init/fin) via AComm / CW eICAB stations on the trunk
    - Other labels: one mid LOC on the trunk

    Stage-4 named side branches are kept for PITC/PWV (stage 6); they do **not**
    create extra stage-5 LOCs.
    """
    from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

    _ = (venous_mask, endpoint_inset_frac)  # API-stable; unused by current strategies
    branches = _normalize_arterial_branches(arterial_polylines)
    meta_extra: dict[str, Any] = {
        "dual_loc_fallback_vessels": [],
        "loc_boundary_method_dual": "arc_length_tertiles",
        "loc_boundary_method_mca_pca": "bifurcation_arc_midpoints",
        "loc_boundary_method_aca": "junction_or_cw_eicab_midpoints",
        "loc_boundary_method_mid": "arc_length_midpoint",
        "arterial_centerline_source": "stage4_seg_branches",
        "loc_per_branch": False,
        "arterial_branch_names": {},
        "arterial_seg_for_bifurcations": arterial_seg is not None,
        "aca_loc_methods": {},
    }
    out: list[LocRecord] = []
    use_dual = strategy == "qvtpy"
    aca_junction = _infer_aca_junction_ijk(eicab_qvtpy) if use_dual else None
    if aca_junction is not None:
        meta_extra["aca_junction_ijk"] = [int(x) for x in aca_junction]

    # ICA/basilar shared-z alignment uses the trunk polyline of each label.
    trunk_by_label = {
        int(lid): to_numpy(b[0][1]) for lid, b in branches.items() if b
    }
    ica_basilar_idx = (
        select_ica_basilar_aligned_loc_indices(trunk_by_label) if use_dual else {}
    )
    if ica_basilar_idx:
        meta_extra["ica_basilar_shared_z_locs"] = {
            str(k): int(v) for k, v in ica_basilar_idx.items()
        }

    for vid, branch_list in sorted(branches.items()):
        if not branch_list:
            continue
        trunk = to_numpy(branch_list[0][1])
        vname = qvtpy_vessel_name(vid)
        names_here = [str(n) for n, _p in branch_list]
        if names_here:
            meta_extra["arterial_branch_names"][vname] = names_here
        if trunk.shape[0] < 3:
            continue

        # ICA / basilar: single shared-z mid LOC on the trunk.
        if use_dual and int(vid) in QVTPY_ICA_BASILAR_SINGLE_LOC_IDS:
            if int(vid) not in ica_basilar_idx:
                continue
            out.append(
                _arterial_loc_at_index(
                    trunk,
                    ica_basilar_idx[int(vid)],
                    vessel_id=vid,
                    vessel_name=vname,
                    segment_id=0,
                    loc_role="mid",
                    mag=mag,
                    cd=cd,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                )
            )
            continue

        # MCA / PCA: exactly two LOCs (init/fin) from bifurcation stations.
        if use_dual and int(vid) in QVTPY_MCA_PCA_BIFURCATION_LOC_IDS:
            init_idx, fin_idx = pick_mca_pca_loc_indices(
                trunk,
                seg=arterial_seg,
                label_id=int(vid),
            )
            out.append(
                _arterial_loc_at_index(
                    trunk,
                    init_idx,
                    vessel_id=vid,
                    vessel_name=vname,
                    segment_id=0,
                    loc_role="init",
                    mag=mag,
                    cd=cd,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                )
            )
            if fin_idx is not None:
                out.append(
                    _arterial_loc_at_index(
                        trunk,
                        fin_idx,
                        vessel_id=vid,
                        vessel_name=vname,
                        segment_id=1,
                        loc_role="fin",
                        mag=mag,
                        cd=cd,
                        vel_mag=vel_mag,
                        voxel_spacing=voxel_spacing,
                        radius_vox=radius_vox,
                    )
                )
            continue

        # ACA: exactly two LOCs (init/fin) from AComm / CW stations.
        if use_dual and int(vid) in QVTPY_ACA_JUNCTION_LOC_IDS:
            init_idx, fin_idx, aca_method = pick_aca_loc_indices(
                trunk,
                label_id=int(vid),
                eicab_qvtpy=eicab_qvtpy,
                junction_ijk=aca_junction,
            )
            meta_extra["aca_loc_methods"][vname] = aca_method
            out.append(
                _arterial_loc_at_index(
                    trunk,
                    init_idx,
                    vessel_id=vid,
                    vessel_name=vname,
                    segment_id=0,
                    loc_role="init",
                    mag=mag,
                    cd=cd,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                )
            )
            if fin_idx is not None:
                out.append(
                    _arterial_loc_at_index(
                        trunk,
                        fin_idx,
                        vessel_id=vid,
                        vessel_name=vname,
                        segment_id=1,
                        loc_role="fin",
                        mag=mag,
                        cd=cd,
                        vel_mag=vel_mag,
                        voxel_spacing=voxel_spacing,
                        radius_vox=radius_vox,
                    )
                )
            continue

        # Remaining arteries (comm, VA, …): one mid LOC on the trunk.
        mid_idx = pick_mid_loc_index(trunk.shape[0], trunk)
        out.append(
            _arterial_loc_at_index(
                trunk,
                mid_idx,
                vessel_id=vid,
                vessel_name=vname,
                segment_id=0,
                loc_role="mid",
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
            )
        )

    return out, meta_extra


# ---- CSV / serialization -----------------------------------------------------


def loc_record_to_dict(rec: LocRecord) -> dict[str, float | int | str]:
    """Serialize :class:`LocRecord` for ``locs.csv`` / JSON."""
    return {
        "vessel_id": rec.vessel_id,
        "vessel_name": rec.vessel_name,
        "segment_id": rec.segment_id,
        "centerline_index": rec.centerline_index,
        "i": rec.i,
        "j": rec.j,
        "k": rec.k,
        "centerline_x": rec.centerline_x,
        "centerline_y": rec.centerline_y,
        "centerline_z": rec.centerline_z,
        "tangent_x": rec.tangent_x,
        "tangent_y": rec.tangent_y,
        "tangent_z": rec.tangent_z,
        "loc_circularity": rec.loc_circularity,
        "loc_cross_section_area_mm2": rec.loc_cross_section_area_mm2,
        "loc_role": rec.loc_role,
    }


__all__ = [
    "LocRecord",
    "loc_record_to_dict",
    "pick_aca_loc_indices",
    "pick_dual_loc_indices",
    "pick_mca_pca_loc_indices",
    "pick_index_arc_midpoint_between",
    "pick_axial_alignment_index",
    "select_ica_basilar_aligned_loc_indices",
    "pick_equal_section_boundary_indices",
    "pick_arc_midpoint_in_mask",
    "pick_index_at_arc_fraction",
    "pick_mid_loc_index",
    "polyline_cumulative_arc_length",
    "select_arterial_locs",
    "select_venous_locs",
]
