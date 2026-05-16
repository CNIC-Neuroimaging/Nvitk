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

# ICA, ACA, MCA, PCA: dual init/fin LOCs under qvtpy strategy.
QVTPY_DUAL_LOC_ARTERIAL_IDS: frozenset[int] = (
    QVTPY_ICA_BASILAR_IDS | QVTPY_ACA_IDS | QVTPY_MCA_IDS | QVTPY_PCA_IDS
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


def pick_endpoint_indices(
    n: int,
    *,
    inset_frac: float = 0.08,
    min_inset_pts: int = 5,
) -> tuple[int, int] | None:
    """Return ``(init_idx, fin_idx)`` near polyline ends, inset from extremes."""
    if n < 1:
        return None
    if n < 2 * min_inset_pts + 3:
        return None
    mid = n // 2
    init_idx = min(max(int(min_inset_pts), 0), mid)
    fin_from_end = max(int(min_inset_pts), int(round(n * float(inset_frac))))
    fin_idx = max(min(n - 1 - fin_from_end, n - 1 - int(min_inset_pts)), mid)
    fin_idx = min(max(fin_idx, mid), n - 1)
    if init_idx >= fin_idx:
        return None
    return init_idx, fin_idx


def pick_masked_midpoint(
    points: np.ndarray,
    mask: np.ndarray | None,
) -> int:
    """Index along polyline preferring in-mask midpoint."""
    pts = to_numpy(points)
    n = pts.shape[0]
    if n == 0:
        return 0
    if mask is None:
        return n // 2
    m = to_numpy(mask).astype(bool)
    inside = []
    for idx, row in enumerate(pts):
        i, j, k = int(round(row[0])), int(round(row[1])), int(round(row[2]))
        if 0 <= i < m.shape[0] and 0 <= j < m.shape[1] and 0 <= k < m.shape[2] and m[i, j, k]:
            inside.append(idx)
    if inside:
        return inside[len(inside) // 2]
    return n // 2


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
    """Return (sssv_idx, strv_idx) using SVD alignment vs STRV reference."""
    parts_s = split_into_parts(sssv_pts, 6)
    parts_t = split_into_parts(strv_pts, 6)
    if not parts_s or not parts_t:
        return pick_masked_midpoint(sssv_pts, mask), pick_masked_midpoint(strv_pts, mask)

    def _best_part(parts: list[np.ndarray], ref: np.ndarray, maximize: bool) -> np.ndarray:
        best = parts[0]
        best_sc = -np.inf if maximize else np.inf
        for p in parts:
            c = p - np.mean(p, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            d = vt[0] / (float(np.linalg.norm(vt[0])) + 1e-12)
            sc = float(np.dot(d, ref / (float(np.linalg.norm(ref)) + 1e-12)))
            if (maximize and sc > best_sc) or (not maximize and sc < best_sc):
                best_sc = sc
                best = p
        return best

    ref = _STRV_REF / (float(np.linalg.norm(_STRV_REF)) + 1e-12)
    strv_part = _best_part(parts_t, ref, maximize=True)
    sssv_part = _best_part(parts_s, ref, maximize=False)
    return pick_masked_midpoint(sssv_part, mask), pick_masked_midpoint(strv_part, mask)


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
    parts = split_into_parts(long_pts, 6)
    if not parts:
        return pick_masked_midpoint(long_pts, mask)
    scored = [( _z_std(p), i) for i, p in enumerate(parts)]
    scored.sort(key=lambda t: t[0], reverse=vertical)
    return pick_masked_midpoint(parts[scored[0][1]], mask)


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
        if name == NAME_SSSV:
            indices[name] = pick_masked_midpoint(pts, venous_mask)
        elif name == NAME_STRV:
            indices[name] = pick_masked_midpoint(pts, venous_mask)
        else:
            indices[name] = resolve_long_venous_segment(
                pts, mask=venous_mask, vertical=False
            )

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
    }
    out: list[LocRecord] = []
    use_dual = strategy == "qvtpy"

    for vid, pts in sorted(arterial_polylines.items()):
        pts_np = to_numpy(pts)
        vname = qvtpy_vessel_name(vid)
        if pts_np.shape[0] < 3:
            continue

        if use_dual and int(vid) in QVTPY_DUAL_LOC_ARTERIAL_IDS:
            endpoints = pick_endpoint_indices(
                pts_np.shape[0],
                inset_frac=float(endpoint_inset_frac),
            )
            if endpoints is None:
                meta_extra["dual_loc_fallback_vessels"].append(vname)
                rec = select_main_vessel_loc(
                    pts_np,
                    vessel_id=vid,
                    vessel_name=vname,
                    mask=venous_mask,
                    mag=mag,
                    cd=cd,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                )
                if rec:
                    out.append(rec)
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

        rec = select_main_vessel_loc(
            pts_np,
            vessel_id=vid,
            vessel_name=vname,
            mask=venous_mask,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
        )
        if rec:
            out.append(rec)

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
    "select_arterial_locs",
    "select_venous_locs",
]
