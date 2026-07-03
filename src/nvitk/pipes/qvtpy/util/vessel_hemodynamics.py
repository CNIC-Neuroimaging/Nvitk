"""Vessel-level 4D-flow hemodynamics: PITC and PWV from dense centerline sampling.

This is the mask-aware qvtpy analogue of the QVTplus post-processing PITC/PWV
pipeline. Instead of the MATLAB geometric connectivity search over a single binary
complex-difference (CD) skeleton, it walks a deterministic label-aware tree rooted
at the left ICA, right ICA, and basilar artery, sampling oblique cross-sections
along each vessel's own multilabel ``seg_4dflow`` mask.

For each station along a vessel it records distance from the root, pulsatility
index (PI), waveform quality, cross-sectional area, and the flow time series. Those
feed:

- **PITC** — weighted linear fit of PI vs distance-from-root per root
  (:func:`~nvitk.measure.hemodynamics.pitc_fit`).
- **PWV** — Bjornfoot waveform-shift optimizer (default DB variable) and Fielding
  cross-correlation (QC), per root.
- **Damping index** — root PI vs downstream branch PI.

**Outputs** (consumed by :mod:`nvitk.pipes.qvtpy.stage6_measure`)

- ``profile_rows`` — one row per sampled station (``pitc_profile.csv``).
- ``summary_rows`` — one row per root and per branch (``vessel_hemodynamics.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.logger import Logger
from nvitk.measure.cross_section import (
    ThrAlgorithm,
    cross_section_at_point,
    flow_series_ml_s,
    masked_plane_velocity_series,
)
from nvitk.measure.hemodynamics import (
    QUALITY_THRESH_DEFAULT,
    accept_pwv,
    circular_cross_correlation_lag,
    damping_index,
    pitc_fit,
    pulsatility_index,
    pwv_bjornfoot_optimize,
    pwv_fielding_xcor,
    quality_weights,
    waveform_quality_score,
)
from nvitk.morphology.centerline import centerline_tangents
from nvitk.pipes.qvtpy.labels import (
    QVTPY_BASILAR,
    QVTPY_COMM_IDS,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_LPCA,
    QVTPY_LVA,
    QVTPY_RACA,
    QVTPY_RICA,
    QVTPY_RMCA,
    QVTPY_RPCA,
    QVTPY_RVA,
    qvtpy_vessel_name,
)

log = Logger()

_DEFAULT_STRIDE = 3
_DEFAULT_RADIUS_VOX = 10.0


@dataclass(frozen=True)
class RootGroup:
    """A PITC/PWV root vessel and its downstream branches."""

    region_id: str
    root_label: int
    branch_labels: tuple[int, ...]


# Label-aware tree (replaces QVTplus geometric SearchDist connectivity + manual
# Exclude/LR rows). Communicating arteries and PCAs on ICA trees are excluded
# from ICA PITC/PWV (PCAs belong under the basilar root only).
ROOT_GROUPS: tuple[RootGroup, ...] = (
    RootGroup("L_ICA", QVTPY_LICA, (QVTPY_LACA, QVTPY_LMCA)),
    RootGroup("R_ICA", QVTPY_RICA, (QVTPY_RACA, QVTPY_RMCA)),
    RootGroup("Basilar", QVTPY_BASILAR, (QVTPY_LPCA, QVTPY_RPCA, QVTPY_LVA, QVTPY_RVA)),
)

# Hard allow-list per root so mis-tagged or overlapping stations never enter ICA PITC.
PITC_GROUP_ALLOWED_IDS: dict[str, frozenset[int]] = {
    "L_ICA": frozenset({QVTPY_LICA, QVTPY_LACA, QVTPY_LMCA}),
    "R_ICA": frozenset({QVTPY_RICA, QVTPY_RACA, QVTPY_RMCA}),
    "Basilar": frozenset({QVTPY_BASILAR, QVTPY_LPCA, QVTPY_RPCA, QVTPY_LVA, QVTPY_RVA}),
}


@dataclass
class VesselHemodynamicsResult:
    """Station-level profile rows and per-region summary rows.

    ``region_plot_data`` is populated only when ``collect_plot_data=True`` and holds
    the per-station arrays (PI, quality, flow waveforms, PWV cross-correlation
    diagnostics) needed to render the paper-style measurement figures.
    """

    profile_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    region_plot_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    all_label_waveforms: dict[int, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _arc_length_mm(points_xyz: np.ndarray, voxel_spacing: tuple[float, float, float]) -> np.ndarray:
    """Cumulative arc length in mm along an ordered polyline (anisotropic spacing)."""
    pts = to_numpy(as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype="float64")
    sp = np.asarray(voxel_spacing, dtype="float64").reshape(3)
    diffs = np.diff(pts, axis=0) * sp[None, :]
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _orient_polyline(points_xyz: np.ndarray, anchor_xyz: np.ndarray) -> np.ndarray:
    """Return *points_xyz* ordered so index 0 is the endpoint nearest *anchor_xyz*."""
    pts = to_numpy(as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return pts
    anchor = np.asarray(anchor_xyz, dtype="float64").reshape(3)
    d_first = float(np.linalg.norm(pts[0] - anchor))
    d_last = float(np.linalg.norm(pts[-1] - anchor))
    return pts if d_first <= d_last else pts[::-1].copy()


def _root_proximal_anchor(points_xyz: np.ndarray) -> np.ndarray:
    """Proximal (inferior, min-Z) endpoint of a root vessel polyline."""
    pts = to_numpy(as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] == 0:
        return np.zeros(3, dtype="float64")
    endpoints = np.vstack([pts[0], pts[-1]])
    return endpoints[int(np.argmin(endpoints[:, 2]))]


# ---------------------------------------------------------------------------
# Station sampling
# ---------------------------------------------------------------------------


def _sample_vessel_stations(
    points_xyz: np.ndarray,
    label_id: int,
    *,
    distance_offset_mm: float,
    cd: np.ndarray,
    mag: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    volume_seg: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    stride: int,
    radius_vox: float,
) -> list[dict[str, Any]]:
    """Sample cross-sections along one vessel; return per-station metric dicts."""
    pts = to_numpy(as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return []
    tangents = to_numpy(centerline_tangents(pts, k_half=2))
    arc = _arc_length_mm(pts, voxel_spacing)
    step = max(1, int(stride))
    rows: list[dict[str, Any]] = []
    for idx in range(0, pts.shape[0], step):
        try:
            cs = cross_section_at_point(
                pts[idx],
                tangents[idx],
                cd=cd,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
                measure_resegment=False,
                volume_seg=volume_seg,
                volume_label_id=int(label_id),
            )
        except Exception as exc:  # noqa: BLE001 - keep profiling other stations
            log.warning(f"label {label_id}: cross-section failed at station {idx}: {exc}")
            continue
        if cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
            continue
        vel_ts = masked_plane_velocity_series(vx, vy, vz, cs)
        flow_ts = to_numpy(flow_series_ml_s(vel_ts, cs.area_mm2)).astype("float64")
        pi = float(pulsatility_index(flow_ts)[0])
        quality = waveform_quality_score(flow_ts)
        rows.append(
            {
                "vessel_id": int(label_id),
                "vessel_name": qvtpy_vessel_name(int(label_id)),
                "station_index": int(idx),
                "distance_mm": float(distance_offset_mm + arc[idx]),
                "pi": pi,
                "quality": float(quality),
                "area_mm2": float(cs.area_mm2),
                "flow_ts": flow_ts,
            }
        )
    return rows


def build_all_label_waveforms(
    centerlines: dict[int, np.ndarray],
    *,
    volume_seg: np.ndarray,
    cd: np.ndarray,
    mag: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float,
    region_waveforms: dict[int, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Midpoint flow waveforms for every labeled centerline (arterial + venous)."""
    from nvitk.pipes.qvtpy.labels import QVTPY_VENOUS_LABEL_IDS
    from nvitk.pipes.qvtpy.util.loc_selection import pick_mid_loc_index

    out: dict[int, dict[str, Any]] = dict(region_waveforms or {})
    for label_id in sorted(int(k) for k in centerlines.keys()):
        lid = int(label_id)
        if lid in out:
            continue
        pts = to_numpy(as_backend_array(centerlines[lid])).astype("float64")
        if pts.shape[0] < 2:
            continue
        idx = pick_mid_loc_index(pts.shape[0], pts)
        tangents = to_numpy(centerline_tangents(pts, k_half=2))
        is_venous = lid in QVTPY_VENOUS_LABEL_IDS
        cs = None
        for use_resegment in ((False, True) if is_venous else (False,)):
            try:
                cs = cross_section_at_point(
                    pts[idx],
                    tangents[idx],
                    cd=cd,
                    mag=mag,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                    measure_resegment=use_resegment,
                    volume_seg=volume_seg,
                    volume_label_id=lid,
                )
            except Exception:
                cs = None
                continue
            if cs is not None and cs.area_mm2 > 0.0 and bool(np.any(cs.mask_2d)):
                break
        if cs is None or cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
            continue
        vel_ts = masked_plane_velocity_series(vx, vy, vz, cs)
        flow_ts = to_numpy(flow_series_ml_s(vel_ts, cs.area_mm2)).astype("float64")
        out[lid] = {
            "vessel_name": qvtpy_vessel_name(lid),
            "mean": flow_ts,
            "std": np.zeros_like(flow_ts),
            "n_stations": 1,
        }
    return out


def _distance_offset_for_branch(
    root_pts: np.ndarray,
    branch_pts: np.ndarray,
    voxel_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Orient a branch away from the root distal end and return its distance offset.

    Offset = full root arc length + straight-line gap between the root distal
    endpoint and the branch proximal endpoint (in mm).
    """
    root_arc = _arc_length_mm(root_pts, voxel_spacing)
    root_distal = to_numpy(as_backend_array(root_pts)).astype("float64")[-1]
    branch_oriented = _orient_polyline(branch_pts, root_distal)
    sp = np.asarray(voxel_spacing, dtype="float64").reshape(3)
    gap = float(np.linalg.norm((branch_oriented[0] - root_distal) * sp))
    return branch_oriented, float(root_arc[-1] + gap)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_vessel_hemodynamics(
    centerlines: dict[int, np.ndarray],
    *,
    volume_seg: np.ndarray,
    cd: np.ndarray,
    mag: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    temporal_resolution_s: float | None,
    stride: int = _DEFAULT_STRIDE,
    radius_vox: float = _DEFAULT_RADIUS_VOX,
    quality_thresh: float = QUALITY_THRESH_DEFAULT,
    collect_plot_data: bool = False,
) -> VesselHemodynamicsResult:
    """Compute per-root PITC/PWV and per-branch damping from dense centerline sampling."""
    result = VesselHemodynamicsResult()
    cls = {int(k): to_numpy(as_backend_array(v)).astype("float64") for k, v in centerlines.items()}

    for group in ROOT_GROUPS:
        if group.root_label not in cls:
            continue
        root_pts = _orient_polyline(cls[group.root_label], _root_proximal_anchor(cls[group.root_label]))
        # Re-anchor so index 0 is the proximal (inferior) end -> distance grows distally.
        root_anchor = _root_proximal_anchor(root_pts)
        root_pts = _orient_polyline(root_pts, root_anchor)

        group_stations: list[dict[str, Any]] = []
        root_rows = _sample_vessel_stations(
            root_pts,
            group.root_label,
            distance_offset_mm=0.0,
            cd=cd,
            mag=mag,
            vel_mag=vel_mag,
            vx=vx,
            vy=vy,
            vz=vz,
            volume_seg=volume_seg,
            voxel_spacing=voxel_spacing,
            stride=stride,
            radius_vox=radius_vox,
        )
        for r in root_rows:
            r["root_region_id"] = group.region_id
        group_stations.extend(root_rows)

        branch_rows_by_label: dict[int, list[dict[str, Any]]] = {}
        for blabel in group.branch_labels:
            if int(blabel) in QVTPY_COMM_IDS:
                continue
            if blabel not in cls:
                continue
            branch_pts, offset = _distance_offset_for_branch(root_pts, cls[blabel], voxel_spacing)
            brows = _sample_vessel_stations(
                branch_pts,
                blabel,
                distance_offset_mm=offset,
                cd=cd,
                mag=mag,
                vel_mag=vel_mag,
                vx=vx,
                vy=vy,
                vz=vz,
                volume_seg=volume_seg,
                voxel_spacing=voxel_spacing,
                stride=stride,
                radius_vox=radius_vox,
            )
            for r in brows:
                r["root_region_id"] = group.region_id
            branch_rows_by_label[blabel] = brows
            group_stations.extend(brows)

        allowed = PITC_GROUP_ALLOWED_IDS[group.region_id]
        group_stations = [
            r
            for r in group_stations
            if int(r["vessel_id"]) in allowed
            and int(r["vessel_id"]) not in QVTPY_COMM_IDS
        ]
        root_rows = [r for r in root_rows if int(r["vessel_id"]) in allowed]
        branch_rows_by_label = {
            int(lid): [r for r in rows if int(r["vessel_id"]) in allowed]
            for lid, rows in branch_rows_by_label.items()
            if int(lid) in allowed
        }

        # Profile rows (drop the heavy flow_ts before persisting).
        for r in group_stations:
            result.profile_rows.append({k: v for k, v in r.items() if k != "flow_ts"})

        if len(group_stations) < 2:
            continue

        pis = np.array([r["pi"] for r in group_stations], dtype="float64")
        dists = np.array([r["distance_mm"] for r in group_stations], dtype="float64")
        quals = np.array([r["quality"] for r in group_stations], dtype="float64")
        fit = pitc_fit(pis, dists, quals, thresh=quality_thresh)

        pwv_result = _root_pwv(group_stations, temporal_resolution_s, quality_thresh=quality_thresh)

        if collect_plot_data:
            result.region_plot_data[group.region_id] = _collect_region_plot_data(
                group,
                group_stations,
                root_rows,
                branch_rows_by_label,
                fit,
                pwv_result,
                temporal_resolution_s,
                quality_thresh=quality_thresh,
            )

        result.summary_rows.append(
            {
                "region_id": group.region_id,
                "region_label": group.region_id,
                "row_kind": "root",
                "pitc_slope": fit["pitc_slope"],
                "pitc_intercept": fit["pitc_intercept"],
                "pitc_r2": fit["r2"],
                "pitc_n": fit["n"],
                "global_pi": fit["global_pi"],
                "pwv_bjornfoot_m_s": pwv_result["pwv_bjornfoot_m_s"],
                "pwv_fielding_m_s": pwv_result["pwv_fielding_m_s"],
                "pwv_r_fielding": pwv_result["pwv_r_fielding"],
                "pwv_n_stations": pwv_result["pwv_n_stations"],
                "damping_index": "",
            }
        )

        # Per-branch damping index (root mean PI vs branch mean PI).
        root_pi = float(np.mean([r["pi"] for r in root_rows])) if root_rows else float("nan")
        for blabel, brows in branch_rows_by_label.items():
            if not brows:
                continue
            branch_pi = float(np.mean([r["pi"] for r in brows]))
            di = damping_index(root_pi, branch_pi)
            result.summary_rows.append(
                {
                    "region_id": qvtpy_vessel_name(int(blabel)),
                    "region_label": qvtpy_vessel_name(int(blabel)),
                    "row_kind": "branch",
                    "pitc_slope": "",
                    "pitc_intercept": "",
                    "pitc_r2": "",
                    "pitc_n": "",
                    "global_pi": "",
                    "pwv_bjornfoot_m_s": "",
                    "pwv_fielding_m_s": "",
                    "pwv_r_fielding": "",
                    "pwv_n_stations": "",
                    "damping_index": di,
                }
            )

    if collect_plot_data and cls:
        merged_region_wf: dict[int, dict[str, Any]] = {}
        for region in result.region_plot_data.values():
            for label, wf in region.get("vessel_waveforms", {}).items():
                prev = merged_region_wf.get(int(label))
                if prev is None or int(wf["n_stations"]) > int(prev["n_stations"]):
                    merged_region_wf[int(label)] = wf
        result.all_label_waveforms = build_all_label_waveforms(
            cls,
            volume_seg=volume_seg,
            cd=cd,
            mag=mag,
            vel_mag=vel_mag,
            vx=vx,
            vy=vy,
            vz=vz,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
            region_waveforms=merged_region_wf,
        )

    return result


def _root_pwv(
    stations: list[dict[str, Any]],
    temporal_resolution_s: float | None,
    *,
    quality_thresh: float,
) -> dict[str, Any]:
    """Bjornfoot + Fielding PWV over high-quality stations ordered by distance."""
    empty = {
        "pwv_bjornfoot_m_s": "",
        "pwv_fielding_m_s": "",
        "pwv_r_fielding": "",
        "pwv_n_stations": 0,
    }
    if temporal_resolution_s is None or float(temporal_resolution_s) <= 0:
        return empty
    good = [r for r in stations if r["quality"] > float(quality_thresh)]
    good.sort(key=lambda r: r["distance_mm"])
    if len(good) < 3:
        return empty
    dist_m = np.array([r["distance_mm"] for r in good], dtype="float64") / 1000.0
    weights = np.array([r["quality"] for r in good], dtype="float64")
    flow_matrix = np.vstack([r["flow_ts"] for r in good]).astype("float64")

    bj = pwv_bjornfoot_optimize(dist_m, flow_matrix, float(temporal_resolution_s), weights=weights)
    fi = pwv_fielding_xcor(dist_m, flow_matrix, float(temporal_resolution_s), weights=weights)
    bj_val = bj["pwv_m_s"] if accept_pwv(bj["pwv_m_s"]) else ""
    fi_val = fi["pwv_m_s"] if accept_pwv(fi["pwv_m_s"]) else ""
    return {
        "pwv_bjornfoot_m_s": bj_val,
        "pwv_fielding_m_s": fi_val,
        "pwv_r_fielding": fi["r"],
        "pwv_n_stations": int(len(good)),
    }


def _time_to_upstroke_s(flow_ts: np.ndarray, tr: float) -> float:
    """Time (s) of maximal systolic upslope in a periodic flow waveform."""
    x = np.asarray(flow_ts, dtype="float64").reshape(-1)
    if x.size < 3:
        return float("nan")
    d = np.diff(x)
    return float(int(np.argmax(d)) * float(tr))


def _collect_region_plot_data(
    group: RootGroup,
    group_stations: list[dict[str, Any]],
    root_rows: list[dict[str, Any]],
    branch_rows_by_label: dict[int, list[dict[str, Any]]],
    fit: dict[str, Any],
    pwv_result: dict[str, Any],
    temporal_resolution_s: float | None,
    *,
    quality_thresh: float,
) -> dict[str, Any]:
    """Assemble per-station arrays for the PITC / PWV / flow-waveform figures."""
    stations = sorted(group_stations, key=lambda r: r["distance_mm"])
    distance_mm = np.array([r["distance_mm"] for r in stations], dtype="float64")
    pi = np.array([r["pi"] for r in stations], dtype="float64")
    quality = np.array([r["quality"] for r in stations], dtype="float64")

    # PWV cross-correlation diagnostics over high-quality stations (Fielding-style).
    good = [r for r in stations if r["quality"] > float(quality_thresh)]
    good.sort(key=lambda r: r["distance_mm"])
    pwv_dist_mm = np.array([r["distance_mm"] for r in good], dtype="float64")
    xcor_time_s = np.full(len(good), np.nan, dtype="float64")
    upstroke_s = np.full(len(good), np.nan, dtype="float64")
    w1 = np.zeros(len(good), dtype="float64")
    w2 = np.zeros(len(good), dtype="float64")
    tr = float(temporal_resolution_s) if temporal_resolution_s else 0.0
    if good and tr > 0:
        ref = np.asarray(good[0]["flow_ts"], dtype="float64")
        quals = np.array([r["quality"] for r in good], dtype="float64")
        w1 = np.asarray(quality_weights(quals, thresh=quality_thresh), dtype="float64")
        for i, r in enumerate(good):
            lag_frames, corr = circular_cross_correlation_lag(ref, r["flow_ts"])
            xcor_time_s[i] = lag_frames * tr
            upstroke_s[i] = _time_to_upstroke_s(np.asarray(r["flow_ts"], dtype="float64"), tr)
            w2[i] = abs(corr)

    # Per-vessel flow waveforms (mean +/- std across the vessel's own stations).
    vessel_waveforms: dict[int, dict[str, Any]] = {}
    for label, rows in [(group.root_label, root_rows), *branch_rows_by_label.items()]:
        flows = [np.asarray(r["flow_ts"], dtype="float64") for r in rows if r.get("flow_ts") is not None]
        if not flows:
            continue
        nt = min(f.size for f in flows)
        stack = np.vstack([f[:nt] for f in flows])
        vessel_waveforms[int(label)] = {
            "vessel_name": qvtpy_vessel_name(int(label)),
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
            "n_stations": int(stack.shape[0]),
        }

    return {
        "region_id": group.region_id,
        "distance_mm": distance_mm,
        "pi": pi,
        "quality": quality,
        "pitc_slope": fit.get("pitc_slope"),
        "pitc_intercept": fit.get("pitc_intercept"),
        "global_pi": fit.get("global_pi"),
        "quality_thresh": float(quality_thresh),
        "pwv_distance_mm": pwv_dist_mm,
        "pwv_xcor_time_s": xcor_time_s,
        "pwv_time_to_upstroke_s": upstroke_s,
        "pwv_weight_quality": w1,
        "pwv_weight_correlation": w2,
        "pwv_bjornfoot_m_s": pwv_result.get("pwv_bjornfoot_m_s"),
        "pwv_fielding_m_s": pwv_result.get("pwv_fielding_m_s"),
        "temporal_resolution_s": tr if tr > 0 else None,
        "vessel_waveforms": vessel_waveforms,
    }


__all__ = [
    "PITC_GROUP_ALLOWED_IDS",
    "ROOT_GROUPS",
    "RootGroup",
    "VesselHemodynamicsResult",
    "build_all_label_waveforms",
    "compute_vessel_hemodynamics",
]
