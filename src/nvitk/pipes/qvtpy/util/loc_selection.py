"""LOC (location of interest) selection heuristics aligned with QVTplus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import centerline_tangents
from nvitk.measure.cross_section import CrossSectionResult, segment_at_point
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_IDS,
    QVTPY_BASILAR,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_MCA_IDS,
    QVTPY_PCA_IDS,
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRV_REF = np.array([0.0, 1.0, 1.0], dtype=np.float64)

# ICA, ACA, MCA, PCA: dual init/fin LOCs; basilar and vertebral arteries use midpoint only.
QVTPY_DUAL_LOC_ARTERIAL_IDS: frozenset[int] = frozenset(
    (QVTPY_ICA_BASILAR_IDS - {QVTPY_BASILAR}) | QVTPY_ACA_IDS | QVTPY_MCA_IDS | QVTPY_PCA_IDS
)


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LocRecord:
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


def select_arterial_locs(
    arterial_polylines: dict[int, np.ndarray],
    *,
    venous_mask: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    strategy: str = "qvtpy",
    endpoint_inset_frac: float = 0.08,
) -> tuple[list[LocRecord], dict[str, Any]]:
    """Select arterial LOCs. Returns ``(records, meta_extra)`` for loc_meta.json."""
    from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

    meta_extra: dict[str, Any] = {
        "dual_loc_fallback_vessels": [],
        "loc_boundary_method_dual": "arc_length_tertiles",
        "loc_boundary_method_mid": "arc_length_midpoint",
        "arterial_centerline_source": "stage4_seg",
    }
    out: list[LocRecord] = []
    use_dual = strategy == "qvtpy"

    for vid, pts in sorted(arterial_polylines.items()):
        pts_np = to_numpy(pts)
        vname = qvtpy_vessel_name(vid)
        if pts_np.shape[0] < 3:
            continue

        if use_dual and int(vid) in QVTPY_DUAL_LOC_ARTERIAL_IDS:
            endpoints = pick_dual_loc_indices(pts_np.shape[0], pts_np)
            if endpoints is None:
                meta_extra["dual_loc_fallback_vessels"].append(vname)
                mid_idx = pick_mid_loc_index(pts_np.shape[0], pts_np)
                out.append(
                    _arterial_loc_at_index(
                        pts_np,
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
                continue
            init_idx, fin_idx = endpoints
            for seg_id, idx, role in (
                (0, init_idx, "init"),
                (1, fin_idx, "fin"),
            ):
                out.append(
                    _arterial_loc_at_index(
                        pts_np,
                        idx,
                        vessel_id=vid,
                        vessel_name=vname,
                        segment_id=seg_id,
                        loc_role=role,
                        mag=mag,
                        cd=cd,
                        vel_mag=vel_mag,
                        voxel_spacing=voxel_spacing,
                        radius_vox=radius_vox,
                    )
                )
            continue

        mid_idx = pick_mid_loc_index(pts_np.shape[0], pts_np)
        out.append(
            _arterial_loc_at_index(
                pts_np,
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
    "pick_dual_loc_indices",
    "pick_equal_section_boundary_indices",
    "pick_arc_midpoint_in_mask",
    "pick_index_at_arc_fraction",
    "pick_mid_loc_index",
    "polyline_cumulative_arc_length",
    "select_arterial_locs",
    "select_venous_locs",
]
