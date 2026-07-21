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

from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize

from nvitk.core.array import as_backend_array
from nvitk.core.backend import map_in_thread_pool
from nvitk.core.logger import Logger
from nvitk.measure.cross_section import (
    ThrAlgorithm,
    cross_section_at_loc,
    masked_plane_velocity_series,
)
from nvitk.measure.hemodynamics import (
    QUALITY_THRESH_DEFAULT,
    accept_pwv,
    cross_correlation_delay_seconds,
    damping_index,
    flow_pulsatile_ml_s,
    flow_per_heart_cycle_ml_s,
    bjornfoot_prepare_waveforms,
    pitc_fit,
    pulsatility_index_qvt,
    pwv_bjornfoot_optimize,
    pwv_fielding_xcor,
    quality_weights,
    station_quality_scores,
    stdv_from_mean_branch,
    time_to_upstroke_seconds,
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

_DEFAULT_STRIDE = 1
_DEFAULT_RADIUS_VOX = 10.0
_DEFAULT_BRANCH_WORKERS = max(1, min(4, os.cpu_count() or 4))


def _branch_sample_workers(n_branches: int) -> int:
    if n_branches <= 1:
        return 1
    env = os.environ.get("NVITK_PITC_BRANCH_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(_DEFAULT_BRANCH_WORKERS, n_branches))


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
class StationGeometryViz:
    """One sampled cross-section station for Napari overlays."""

    vessel_id: int
    vessel_name: str
    station_index: int
    centerline_x: float
    centerline_y: float
    centerline_z: float
    distance_mm: float
    pi: float
    quality: float
    area_mm2: float
    used_for_pwv: bool = False
    pwv_weight_area: float = float("nan")
    pwv_weight_quality: float = float("nan")
    pwv_xcor_time_s: float = float("nan")
    pwv_time_to_upstroke_s: float = float("nan")


@dataclass
class VesselGeometryViz:
    """Oriented centerline polyline and its stations for one vessel."""

    vessel_id: int
    vessel_name: str
    polyline_oriented: np.ndarray
    distance_offset_mm: float
    stations: list[StationGeometryViz] = field(default_factory=list)


@dataclass
class RegionGeometryViz:
    """Per-root geometry for PITC/PWV Napari visualization."""

    region_id: str
    root_label: int
    root_polyline_oriented: np.ndarray
    root_init_xyz: np.ndarray
    root_fin_xyz: np.ndarray
    pitc_slope: float
    pitc_intercept: float
    pitc_r2: float
    pitc_n: int
    global_pi: float
    quality_thresh: float
    pwv_bjornfoot_m_s: Any = ""
    pwv_fielding_m_s: Any = ""
    pwv_r_fielding: Any = ""
    pwv_n_stations: int = 0
    vessels: dict[int, VesselGeometryViz] = field(default_factory=dict)


@dataclass
class VesselHemodynamicsResult:
    """Station-level profile rows and per-region summary rows.

    ``region_plot_data`` is populated only when ``collect_plot_data=True`` and holds
    the per-station arrays (PI, quality, flow waveforms, PWV cross-correlation
    diagnostics) needed to render the paper-style measurement figures.

    ``geometry_by_region`` is populated alongside ``region_plot_data`` and holds
    oriented polylines, geometric root init/fin endpoints, and station XYZ for
    Napari overlays.

    ``volume_seg`` is the island-cleaned multilabel mask used for sampling (and
    for PITC branch mask export).
    """

    profile_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    region_plot_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    geometry_by_region: dict[str, RegionGeometryViz] = field(default_factory=dict)
    all_label_waveforms: dict[int, dict[str, Any]] = field(default_factory=dict)
    volume_seg: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _arc_length_mm(points_xyz: np.ndarray, voxel_spacing: tuple[float, float, float]) -> np.ndarray:
    """Cumulative arc length in mm along an ordered polyline (anisotropic spacing)."""
    pts = (as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype="float64")
    sp = as_backend_array(voxel_spacing).astype("float64").reshape(3)
    diffs = np.diff(pts, axis=0) * sp[None, :]
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([as_backend_array([0.0]), np.cumsum(as_backend_array(seg))])


def _orient_polyline(points_xyz: np.ndarray, anchor_xyz: np.ndarray) -> np.ndarray:
    """Return *points_xyz* ordered so index 0 is the endpoint nearest *anchor_xyz*."""
    pts = (as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return pts
    anchor = as_backend_array(anchor_xyz).astype("float64").reshape(3)
    d_first = float(np.linalg.norm(pts[0] - anchor))
    d_last = float(np.linalg.norm(pts[-1] - anchor))
    return pts if d_first <= d_last else pts[::-1].copy()


def _root_proximal_anchor(points_xyz: np.ndarray) -> np.ndarray:
    """Proximal (inferior, min-Z) endpoint of a root vessel polyline."""
    pts = (as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] == 0:
        return np.zeros(3, dtype="float64")
    endpoints = np.vstack([pts[0], pts[-1]])
    return endpoints[int(np.argmin(endpoints[:, 2]))]


# ---------------------------------------------------------------------------
# Station sampling
# ---------------------------------------------------------------------------


def _assign_branch_qualities(
    rows: list[dict[str, Any]],
    *,
    quality_metric: str,
) -> None:
    """Attach per-station Q scores along one vessel branch."""
    if not rows:
        return
    flow_matrix = np.vstack([as_backend_array(r["flow_ts"]).astype("float64") for r in rows])
    areas = np.array([float(r["area_mm2"]) for r in rows], dtype=np.float64)
    circs = np.array([float(r["circularity"]) for r in rows], dtype=np.float64)
    fpc = flow_matrix.mean(axis=1)
    if quality_metric == "stdv_from_mean":
        quals = stdv_from_mean_branch(fpc, areas, circs, flow_matrix)
    else:
        quals = station_quality_scores(flow_matrix, metric="waveform")
    for row, q in zip(rows, quals):
        row["quality"] = float(q)
        row["quality_metric"] = quality_metric


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
    measure_resegment: bool = True,
    label_constrain: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
    cs_supersampling: bool = False,
    quality_metric: str = "stdv_from_mean",
) -> list[dict[str, Any]]:
    """Sample cross-sections along one vessel; return per-station metric dicts."""
    pts = (as_backend_array(points_xyz)).astype("float64")
    if pts.shape[0] < 2:
        return []
    tangents = (centerline_tangents(pts, k_half=2))
    arc = _arc_length_mm(pts, voxel_spacing)
    step = max(1, int(stride))
    rows: list[dict[str, Any]] = []
    for idx in range(0, pts.shape[0], step):
        try:
            cs = cross_section_at_loc(
                pts[idx],
                tangents[idx],
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                voxel_spacing=voxel_spacing,
                radius_vox=radius_vox,
                cross_section_res=cross_section_res,
                plane_interp_order=plane_interp_order,
                cs_supersampling=cs_supersampling,
                measure_resegment=measure_resegment,
                thr_algorithm=thr_algorithm,
                volume_seg=volume_seg,
                volume_label_id=int(label_id),
                label_constrain=label_constrain,
            )
        except Exception as exc:  # noqa: BLE001 - keep profiling other stations
            log.warning(f"label {label_id}: cross-section failed at station {idx}: {exc}")
            continue
        if cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
            continue
        vel_ts = masked_plane_velocity_series(
            vx, vy, vz, cs, plane_interp_order=plane_interp_order
        )
        # Magnitude flow: tangent polarity must not flip PI / mean flow sign.
        flow_ts = np.abs(flow_pulsatile_ml_s(vel_ts, cs.area_mm2))
        pi = pulsatility_index_qvt(flow_ts)
        if not np.isfinite(pi):
            continue
        rows.append(
            {
                "vessel_id": int(label_id),
                "vessel_name": qvtpy_vessel_name(int(label_id)),
                "station_index": int(idx),
                "distance_mm": float(distance_offset_mm + arc[idx]),
                "pi": float(pi),
                "quality": 0.0,
                "quality_metric": quality_metric,
                "area_mm2": float(cs.area_mm2),
                "circularity": float(cs.circularity),
                "flow_mean_ml_s": flow_per_heart_cycle_ml_s(flow_ts),
                "flow_ts": flow_ts,
            }
        )
    _assign_branch_qualities(rows, quality_metric=quality_metric)
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
        pts = (as_backend_array(centerlines[lid])).astype("float64")
        if pts.shape[0] < 2:
            continue
        idx = pick_mid_loc_index(pts.shape[0], pts)
        tangents = (centerline_tangents(pts, k_half=2))
        is_venous = lid in QVTPY_VENOUS_LABEL_IDS
        cs = None
        for use_resegment in ((False, True) if is_venous else (False,)):
            try:
                cs = cross_section_at_loc(
                    pts[idx],
                    tangents[idx],
                    mag=mag,
                    cd=cd,
                    vel_mag=vel_mag,
                    voxel_spacing=voxel_spacing,
                    radius_vox=radius_vox,
                    measure_resegment=use_resegment,
                    volume_seg=volume_seg,
                    volume_label_id=lid,
                    label_constrain=not use_resegment,
                )
            except Exception:
                cs = None
                continue
            if cs is not None and cs.area_mm2 > 0.0 and bool(np.any(cs.mask_2d)):
                break
        if cs is None or cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
            continue
        vel_ts = masked_plane_velocity_series(vx, vy, vz, cs)
        flow_ts = np.abs(flow_pulsatile_ml_s(vel_ts, cs.area_mm2))
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
    root_distal = (as_backend_array(root_pts)).astype("float64")[-1]
    branch_oriented = _orient_polyline(branch_pts, root_distal)
    sp = as_backend_array(voxel_spacing).astype("float64").reshape(3)
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
    quality_metric: str = "stdv_from_mean",
    measure_resegment: bool = True,
    label_constrain: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
    cs_supersampling: bool = False,
    collect_plot_data: bool = False,
) -> VesselHemodynamicsResult:
    """Compute per-root PITC/PWV and per-branch damping from dense centerline sampling."""
    from nvitk.core.array import to_numpy
    from nvitk.pipes.qvtpy.util.mask_cleaning import clean_volume_seg_for_pitc

    result = VesselHemodynamicsResult()
    cls = {int(k): (as_backend_array(v)).astype("float64") for k, v in centerlines.items()}
    # Drop isolated label islands before station sampling / label-constrained CS.
    volume_seg = clean_volume_seg_for_pitc(volume_seg, cls)
    result.volume_seg = to_numpy(volume_seg).astype(np.int32, copy=False)

    sample_kw = dict[str, ndarray[_AnyShape, dtype[Any]] | tuple[float, float, float] | int | float | str](
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
        measure_resegment=measure_resegment,
        label_constrain=label_constrain,
        thr_algorithm=thr_algorithm,
        cross_section_res=cross_section_res,
        plane_interp_order=plane_interp_order,
        cs_supersampling=cs_supersampling,
        quality_metric=quality_metric,
    )

    for group in ROOT_GROUPS:
        if group.root_label not in cls:
            continue
        root_pts = _orient_polyline(cls[group.root_label], _root_proximal_anchor(cls[group.root_label]))
        # Re-anchor so index 0 is the proximal (inferior) end -> distance grows distally.
        root_anchor = _root_proximal_anchor(root_pts)
        root_pts = _orient_polyline(root_pts, root_anchor)

        group_stations: list[dict[str, Any]] = []
        vessel_polylines: dict[int, tuple[np.ndarray, float]] = {
            int(group.root_label): (root_pts.copy(), 0.0),
        }
        root_rows = _sample_vessel_stations(
            root_pts,
            group.root_label,
            distance_offset_mm=0.0,
            **sample_kw,
        )
        for r in root_rows:
            r["root_region_id"] = group.region_id
        group_stations.extend(root_rows)

        branch_rows_by_label: dict[int, list[dict[str, Any]]] = {}
        eligible_branches = [
            int(blabel)
            for blabel in group.branch_labels
            if int(blabel) not in QVTPY_COMM_IDS and blabel in cls
        ]

        def _sample_branch(blabel: int) -> tuple[int, list[dict[str, Any]], np.ndarray, float]:
            branch_pts, offset = _distance_offset_for_branch(
                root_pts, cls[blabel], voxel_spacing
            )
            brows = _sample_vessel_stations(
                branch_pts,
                blabel,
                distance_offset_mm=offset,
                **sample_kw,
            )
            for r in brows:
                r["root_region_id"] = group.region_id
            return blabel, brows, branch_pts, offset

        workers = _branch_sample_workers(len(eligible_branches))
        if workers <= 1:
            for blabel in eligible_branches:
                _, brows, branch_pts, offset = _sample_branch(blabel)
                vessel_polylines[int(blabel)] = (branch_pts.copy(), float(offset))
                branch_rows_by_label[blabel] = brows
                group_stations.extend(brows)
        else:
            for blabel, brows, branch_pts, offset in map_in_thread_pool(
                _sample_branch, eligible_branches, max_workers=workers
            ):
                vessel_polylines[int(blabel)] = (branch_pts.copy(), float(offset))
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
        n_used = int(fit.get("n") or 0)
        log.info(
            f"PITC {group.region_id}: slope={fit['pitc_slope']:.6g} "
            f"intercept={fit['pitc_intercept']:.6g} r2={fit['r2']:.4f} "
            f"global_pi={fit['global_pi']:.4f} "
            f"n_fit={n_used}/{len(group_stations)} (Q>{quality_thresh:g})"
        )

        pwv_result = _root_pwv(
            group_stations,
            temporal_resolution_s,
            quality_thresh=quality_thresh,
            region_id=group.region_id,
        )

        if collect_plot_data:
            plot_data = _collect_region_plot_data(
                group,
                group_stations,
                root_rows,
                branch_rows_by_label,
                fit,
                pwv_result,
                temporal_resolution_s,
                quality_thresh=quality_thresh,
            )
            result.region_plot_data[group.region_id] = plot_data
            result.geometry_by_region[group.region_id] = _build_region_geometry(
                group,
                group_stations,
                vessel_polylines,
                root_pts,
                fit,
                pwv_result,
                plot_data,
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
    region_id: str = "",
) -> dict[str, Any]:
    """Bjornfoot + Fielding PWV over high-quality stations ordered by distance."""
    tag = region_id or "root"
    empty = {
        "pwv_bjornfoot_m_s": "",
        "pwv_fielding_m_s": "",
        "pwv_r_fielding": "",
        "pwv_n_stations": 0,
    }
    if temporal_resolution_s is None or float(temporal_resolution_s) <= 0:
        log.warning(f"PWV {tag}: skipped (missing/invalid temporal_resolution_s)")
        return empty
    good = [r for r in stations if r["quality"] > float(quality_thresh)]
    good.sort(key=lambda r: r["distance_mm"])
    if len(good) < 3:
        log.warning(
            f"PWV {tag}: skipped (only {len(good)} stations with Q>{quality_thresh:g})"
        )
        return empty
    dist_m = np.array([r["distance_mm"] for r in good], dtype="float64") / 1000.0
    quals = np.array([r["quality"] for r in good], dtype="float64")
    areas = np.array([r["area_mm2"] for r in good], dtype="float64")
    flow_matrix = np.vstack([r["flow_ts"] for r in good]).astype("float64")
    tr = float(temporal_resolution_s)

    # QVTplus PWV(1): Bjornfoot tag=0 (area/scaling² on velocity); PWV(2): Fielding XCor
    # with the same Bjornfoot weights (enc_PWV.m / enc_PWV_WO).
    _, fielding_weights, _ = bjornfoot_prepare_waveforms(
        flow_matrix,
        areas,
        qualities=quals,
        thresh=quality_thresh,
        weight_mode="area",
    )

    bj = pwv_bjornfoot_optimize(
        dist_m,
        flow_matrix,
        tr,
        areas=areas,
        qualities=quals,
        quality_thresh=quality_thresh,
        weight_mode="area",
    )
    fi = pwv_fielding_xcor(
        dist_m, flow_matrix, tr, weights=fielding_weights
    )
    bj_ok = accept_pwv(bj["pwv_m_s"])
    fi_ok = accept_pwv(fi["pwv_m_s"])
    bj_val = bj["pwv_m_s"] if bj_ok else ""
    fi_val = fi["pwv_m_s"] if fi_ok else ""
    d_span_mm = float(dist_m[-1] - dist_m[0]) * 1000.0 if dist_m.size else 0.0
    log.info(
        f"PWV {tag}: n={len(good)} d_span={d_span_mm:.1f} mm tr={tr * 1e3:.3f} ms | "
        f"Fielding raw={fi['pwv_m_s']:.4g} m/s "
        f"(accept={fi_ok}, r={fi['r']:.4f}, n_fit={fi['n']}) | "
        f"Bjornfoot raw={bj['pwv_m_s']:.4g} m/s (accept={bj_ok}) | "
        f"CSV fielding={fi_val if fi_val != '' else 'empty'} "
        f"bjornfoot={bj_val if bj_val != '' else 'empty'}"
    )
    return {
        "pwv_bjornfoot_m_s": bj_val,
        "pwv_fielding_m_s": fi_val,
        "pwv_r_fielding": fi["r"],
        "pwv_n_stations": int(len(good)),
    }


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

    # PWV diagnostics over Q>thresh stations (QVTplus enc_PWV_XCor dual-weight figure).
    # W1 = Bjornfoot area/scaling² (tag=0); W2 = Dempsey quality (tag=1).
    good = [r for r in stations if r["quality"] > float(quality_thresh)]
    good.sort(key=lambda r: r["distance_mm"])
    pwv_dist_mm = np.array([r["distance_mm"] for r in good], dtype="float64")
    xcor_time_s = np.full(len(good), np.nan, dtype="float64")
    upstroke_s = np.full(len(good), np.nan, dtype="float64")
    w1 = np.zeros(len(good), dtype="float64")
    w2 = np.zeros(len(good), dtype="float64")
    tr = float(temporal_resolution_s) if temporal_resolution_s else 0.0
    if good and tr > 0:
        ref = as_backend_array(good[0]["flow_ts"]).astype("float64")
        flow_stack = np.vstack(
            [as_backend_array(r["flow_ts"]).astype("float64") for r in good]
        )
        areas = np.array([r["area_mm2"] for r in good], dtype="float64")
        quals = np.array([r["quality"] for r in good], dtype="float64")
        _, w1_arr, _ = bjornfoot_prepare_waveforms(
            flow_stack,
            areas,
            qualities=quals,
            thresh=quality_thresh,
            weight_mode="area",
        )
        w1 = as_backend_array(w1_arr).astype("float64")
        w2 = as_backend_array(
            quality_weights(quals, thresh=quality_thresh)
        ).astype("float64")
        for i, r in enumerate(good):
            delay_s, _corr = cross_correlation_delay_seconds(ref, r["flow_ts"], tr)
            xcor_time_s[i] = delay_s
            upstroke_s[i] = time_to_upstroke_seconds(
                as_backend_array(r["flow_ts"]).astype("float64"), tr
            )

    # Per-vessel flow waveforms (mean +/- std across the vessel's own stations).
    vessel_waveforms: dict[int, dict[str, Any]] = {}
    for label, rows in [(group.root_label, root_rows), *branch_rows_by_label.items()]:
        flows = [
            as_backend_array(r["flow_ts"]).astype("float64")
            for r in rows
            if r.get("flow_ts") is not None
        ]
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
        # W1 = Bjornfoot area weights; W2 = Dempsey quality (QVTplus enc_PWV_XCor).
        "pwv_weight_area": w1,
        "pwv_weight_quality": w2,
        "pwv_bjornfoot_m_s": pwv_result.get("pwv_bjornfoot_m_s"),
        "pwv_fielding_m_s": pwv_result.get("pwv_fielding_m_s"),
        "temporal_resolution_s": tr if tr > 0 else None,
        "vessel_waveforms": vessel_waveforms,
    }


def _build_region_geometry(
    group: RootGroup,
    group_stations: list[dict[str, Any]],
    vessel_polylines: dict[int, tuple[np.ndarray, float]],
    root_pts: np.ndarray,
    fit: dict[str, Any],
    pwv_result: dict[str, Any],
    plot_data: dict[str, Any],
    *,
    quality_thresh: float,
) -> RegionGeometryViz:
    """Build Napari-ready geometry: oriented polylines, init/fin, station XYZ."""
    from nvitk.core.array import to_numpy

    root_arr = to_numpy(root_pts).astype("float64", copy=False)
    pwv_meta_by_key: dict[tuple[int, int, float], int] = {}
    pwv_dist = to_numpy(plot_data.get("pwv_distance_mm", [])).astype("float64").reshape(-1)
    pwv_w1 = to_numpy(plot_data.get("pwv_weight_area", [])).astype("float64").reshape(-1)
    pwv_w2 = to_numpy(plot_data.get("pwv_weight_quality", [])).astype("float64").reshape(-1)
    pwv_xcor = to_numpy(plot_data.get("pwv_xcor_time_s", [])).astype("float64").reshape(-1)
    pwv_ttu = to_numpy(plot_data.get("pwv_time_to_upstroke_s", [])).astype("float64").reshape(-1)
    good_rows = [r for r in group_stations if r["quality"] > float(quality_thresh)]
    good_rows.sort(key=lambda r: r["distance_mm"])
    for i, r in enumerate(good_rows):
        pwv_meta_by_key[
            (int(r["vessel_id"]), int(r["station_index"]), float(r["distance_mm"]))
        ] = i

    vessels: dict[int, VesselGeometryViz] = {}
    for label_id, (poly, offset) in vessel_polylines.items():
        poly_arr = to_numpy(poly).astype("float64", copy=False)
        station_rows = [
            r for r in group_stations if int(r["vessel_id"]) == int(label_id)
        ]
        stations: list[StationGeometryViz] = []
        for r in station_rows:
            idx = int(r["station_index"])
            if idx < 0 or idx >= poly_arr.shape[0]:
                continue
            xyz = poly_arr[idx]
            key = (int(label_id), idx, float(r["distance_mm"]))
            pwv_i = pwv_meta_by_key.get(key)
            stations.append(
                StationGeometryViz(
                    vessel_id=int(label_id),
                    vessel_name=str(r.get("vessel_name") or qvtpy_vessel_name(int(label_id))),
                    station_index=idx,
                    centerline_x=float(xyz[0]),
                    centerline_y=float(xyz[1]),
                    centerline_z=float(xyz[2]),
                    distance_mm=float(r["distance_mm"]),
                    pi=float(r["pi"]),
                    quality=float(r["quality"]),
                    area_mm2=float(r["area_mm2"]),
                    used_for_pwv=pwv_i is not None,
                    pwv_weight_area=float(pwv_w1[pwv_i]) if pwv_i is not None and pwv_i < pwv_w1.size else float("nan"),
                    pwv_weight_quality=float(pwv_w2[pwv_i]) if pwv_i is not None and pwv_i < pwv_w2.size else float("nan"),
                    pwv_xcor_time_s=float(pwv_xcor[pwv_i]) if pwv_i is not None and pwv_i < pwv_xcor.size else float("nan"),
                    pwv_time_to_upstroke_s=float(pwv_ttu[pwv_i]) if pwv_i is not None and pwv_i < pwv_ttu.size else float("nan"),
                )
            )
        vessels[int(label_id)] = VesselGeometryViz(
            vessel_id=int(label_id),
            vessel_name=qvtpy_vessel_name(int(label_id)),
            polyline_oriented=poly_arr,
            distance_offset_mm=float(offset),
            stations=stations,
        )

    return RegionGeometryViz(
        region_id=group.region_id,
        root_label=int(group.root_label),
        root_polyline_oriented=root_arr,
        root_init_xyz=root_arr[0].copy(),
        root_fin_xyz=root_arr[-1].copy(),
        pitc_slope=float(fit.get("pitc_slope", float("nan"))),
        pitc_intercept=float(fit.get("pitc_intercept", float("nan"))),
        pitc_r2=float(fit.get("r2", float("nan"))),
        pitc_n=int(fit.get("n") or 0),
        global_pi=float(fit.get("global_pi", float("nan"))),
        quality_thresh=float(quality_thresh),
        pwv_bjornfoot_m_s=pwv_result.get("pwv_bjornfoot_m_s", ""),
        pwv_fielding_m_s=pwv_result.get("pwv_fielding_m_s", ""),
        pwv_r_fielding=pwv_result.get("pwv_r_fielding", ""),
        pwv_n_stations=int(pwv_result.get("pwv_n_stations") or 0),
        vessels=vessels,
    )


__all__ = [
    "PITC_GROUP_ALLOWED_IDS",
    "ROOT_GROUPS",
    "RegionGeometryViz",
    "RootGroup",
    "StationGeometryViz",
    "VesselGeometryViz",
    "VesselHemodynamicsResult",
    "build_all_label_waveforms",
    "compute_vessel_hemodynamics",
]
