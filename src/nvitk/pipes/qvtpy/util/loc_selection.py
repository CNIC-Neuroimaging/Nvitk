"""LOC (location of interest) selection heuristics aligned with QVTplus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import centerline_tangents
from nvitk.pipes.qvtpy.util.cross_section import CrossSectionResult, segment_at_point
from nvitk.pipes.qvtpy.labels import (
    EICAB_BASILAR,
    EICAB_LICA,
    EICAB_RICA,
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
)

_STRV_REF = np.array([0.0, 1.0, 1.0], dtype=np.float64)


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


def _record_from_polyline(
    points: np.ndarray,
    idx: int,
    *,
    vessel_id: int,
    vessel_name: str,
    segment_id: int = 0,
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


def select_ica_ba_loc(
    polylines: dict[int, np.ndarray],
    *,
    ica_ids: tuple[int, ...] = (EICAB_LICA, EICAB_RICA),
    ba_id: int = EICAB_BASILAR,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    name_for_id: Any = None,
) -> list[LocRecord]:
    """Pick ICA/BA LOCs near common Z with best circularity."""
    from nvitk.pipes.qvtpy.labels import eicab_vessel_name

    targets = [i for i in (*ica_ids, ba_id) if i in polylines and polylines[i].shape[0] >= 3]
    if not targets:
        return []
    z_vals = [float(np.median(to_numpy(polylines[i])[:, 2])) for i in targets]
    z_common = float(np.median(z_vals))
    out: list[LocRecord] = []
    for vid in targets:
        pts = to_numpy(polylines[vid])
        z = pts[:, 2]
        order = np.argsort(np.abs(z - z_common))
        best_idx = int(order[0])
        best_circ = -1.0
        for idx in order[: min(15, order.size)]:
            xs = _cross_section_at(
                pts,
                int(idx),
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
            )
            circ = xs.circularity if xs else 0.0
            if circ >= best_circ:
                best_circ = circ
                best_idx = int(idx)
        vname = name_for_id(vid) if callable(name_for_id) else eicab_vessel_name(vid)
        xs_final = _cross_section_at(
            pts,
            best_idx,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
        )
        out.append(_record_from_polyline(pts, best_idx, vessel_id=vid, vessel_name=vname, xs=xs_final))
    return out


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


def select_arterial_locs(
    arterial_polylines: dict[int, np.ndarray],
    *,
    venous_mask: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    cd: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_vox: float = 10.0,
    strategy: str = "qvtplus",
) -> list[LocRecord]:
    from nvitk.pipes.qvtpy.labels import eicab_vessel_name

    if strategy != "qvtplus":
        out: list[LocRecord] = []
        for vid, pts in sorted(arterial_polylines.items()):
            rec = select_main_vessel_loc(
                pts,
                vessel_id=vid,
                vessel_name=eicab_vessel_name(vid),
                mask=venous_mask,
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
            )
            if rec:
                out.append(rec)
        return out

    ica_ba = {EICAB_LICA, EICAB_RICA, EICAB_BASILAR}
    ica_polys = {k: v for k, v in arterial_polylines.items() if k in ica_ba}
    rest = {k: v for k, v in arterial_polylines.items() if k not in ica_ba}

    out = select_ica_ba_loc(
        ica_polys,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
    )
    for vid, pts in sorted(rest.items()):
        rec = select_main_vessel_loc(
            pts,
            vessel_id=vid,
            vessel_name=eicab_vessel_name(vid),
            mask=venous_mask,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
        )
        if rec:
            out.append(rec)
    return out


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
    }


__all__ = [
    "LocRecord",
    "loc_record_to_dict",
    "select_arterial_locs",
    "select_venous_locs",
]
