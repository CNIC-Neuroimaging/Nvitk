"""VMTK centerline generation, validation, retries, and result handling."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from vtk.util import numpy_support

from .anatomy import orient_centerline_points_by_flow
from .caliber import (
    compute_siphon_mask,
    detect_enlargement_segments,
    detect_stenosis_segments,
    enlargement_pointwise,
    refresh_enlargement_summary_from_flags,
    resolve_stenosis_enlargement_overlap,
    segment_detail_json,
    select_caliber_detection_radius,
    stenosis_pointwise,
    stenosis_raw_percent,
    stenosis_total_length,
    stenosis_valid_mask_from_ends,
)
from nvitk.measure.morphometrics_config import (
    BEND_KAPPA_PEAK,
    CENTERLINE_RESAMPLE_STEP_MM,
    ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM,
    ENLARGEMENT_CENTERLINE_START_EXCLUDE_MM,
    ENLARGEMENT_EXCLUDE_END_MM,
    ENLARGEMENT_MIN_LEN_MM,
    ENLARGEMENT_SUPPORT_THRESHOLD_PCT,
    ENLARGEMENT_THRESHOLD_PCT,
    FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE,
    FINAL_CENTERLINE_OVERLAP_TOL_MM,
    INFLECT_KAPPA_MIN,
    INFLECT_SMOOTH_WIN,
    PRUNE_OVERLAPPING_FINAL_CENTERLINE_PREFIXES,
    RADIUS_SOURCE_FOR_CALIBER_DETECTION,
    RETRY_VMTK_WITH_TRIMMED_SEEDS,
    STENOSIS_CANDIDATE_EXCLUDE_END_MM,
    STENOSIS_EXCLUDE_END_MM,
    STENOSIS_MIN_LEN_MM,
    STENOSIS_SUPPORT_THRESHOLD_PCT,
    STENOSIS_THRESHOLD_PCT,
    SUPPRESS_ENLARGEMENT_NEAR_CENTERLINE_STARTS,
    VALIDATE_VMTK_CENTERLINE_GEOMETRY,
    VMTK_MAX_CENTERLINE_ENDPOINT_SEED_DIST_MM,
    VMTK_MIN_CENTERLINE_TO_SEED_PATH_LENGTH_RATIO,
    VMTK_MIN_VALID_CENTERLINE_POINTS,
    VMTK_REFERENCE_CONNECTION_TOL_MM,
    VMTK_SEED_TRIM_RETRY_MM,
    VMTK_VALIDATE_ENDPOINT_SEEDS,
    VMTK_VALIDATE_LENGTH_RATIO,
    VMTK_VALIDATE_REFERENCE_CONNECTION,
)
from .geometry import (
    arc_length,
    chord_length,
    cumulative_s,
    resample_generated_centerline_points,
    trimmed_seed_pair_from_path,
)
from .metrics import (
    bend_peak_count,
    discrete_curvature,
    discrete_torsion,
    inflection_count,
    radius_from_edt,
    radius_stats,
    signed_turn_proxy,
    taper_slope,
    tortuosity_dm,
)
from .models import VesselInfo
from .surface import (
    add_string_point_array,
    add_tree_metadata_point_arrays,
    build_polyline_polydata,
    build_radius_tube_polydata,
    compute_cross_section_radius,
    default_tree_point_metadata,
    extract_point_data_array,
    resample_point_data_by_arclength,
    save_vtp,
    snap_to_surface,
)

# nvitk-native centerlines (skeleton path + EDT MIS); pipeline API preserved.
VMTK_AVAILABLE = True


def _centerline_poly_from_path_mm(path_mm: np.ndarray) -> object:
    """Build VTK polyline from a mm path with discrete curvature/torsion arrays."""
    pts = resample_generated_centerline_points(np.asarray(path_mm, dtype=float))
    if len(pts) < 2:
        raise RuntimeError("centerline path has fewer than 2 points after resampling")
    kappa = discrete_curvature(pts)
    torsion = discrete_torsion(pts)
    radius_mis = np.full(len(pts), np.nan, dtype=float)
    return build_polyline_polydata(
        points=pts,
        arrays=[
            (kappa, "Curvature"),
            (torsion, "Torsion"),
            (radius_mis, "MaximumInscribedSphereRadius"),
        ],
    )


def centerline_poly_length_mm(centerline_poly) -> float:
    if centerline_poly is None or centerline_poly.GetNumberOfPoints() < 2:
        return np.nan
    pts = numpy_support.vtk_to_numpy(centerline_poly.GetPoints().GetData())
    return arc_length(pts)


def centerline_points_from_result(res: dict) -> np.ndarray:
    return np.column_stack([res["x_mm"], res["y_mm"], res["z_mm"]]).astype(float)


def run_vmtk_centerline_between_seeds(surface, seed_start_mm, seed_end_mm, spacing):
    _ = surface
    _ = spacing
    pts = np.vstack(
        [np.asarray(seed_start_mm, dtype=float), np.asarray(seed_end_mm, dtype=float)]
    )
    print(f"    native centerline seeds: src={np.round(pts[0], 2)}  tgt={np.round(pts[-1], 2)}")
    return _centerline_poly_from_path_mm(pts)


def centerline_points_from_polydata(centerline_poly) -> np.ndarray:
    if centerline_poly is None or centerline_poly.GetNumberOfPoints() == 0:
        return np.empty((0, 3), dtype=float)
    return numpy_support.vtk_to_numpy(centerline_poly.GetPoints().GetData()).astype(float)


def validate_vmtk_centerline_attempt(
    centerline_poly,
    seed_path_mm: np.ndarray,
    seed_start_mm: np.ndarray,
    seed_end_mm: np.ndarray,
    spacing,
    reference_centerline_points: Optional[List[np.ndarray]] = None,
    require_both_reference_endpoints_connected: bool = False,
) -> Tuple[bool, str]:
    if not VALIDATE_VMTK_CENTERLINE_GEOMETRY:
        return True, ""

    pts = centerline_points_from_polydata(centerline_poly)
    min_points = int(VMTK_MIN_VALID_CENTERLINE_POINTS)
    if len(pts) < min_points:
        return False, f"centerline has only {len(pts)} point(s), fewer than required {min_points}"

    spacing = np.asarray(spacing, dtype=float)
    seed_path_len = arc_length(seed_path_mm)
    cl_len = arc_length(pts)
    if (
        VMTK_VALIDATE_LENGTH_RATIO
        and
        np.isfinite(seed_path_len)
        and seed_path_len > 1e-8
        and np.isfinite(cl_len)
        and cl_len < float(VMTK_MIN_CENTERLINE_TO_SEED_PATH_LENGTH_RATIO) * seed_path_len
    ):
        return (
            False,
            f"centerline too short relative to seed path "
            f"({cl_len:.2f} mm vs {seed_path_len:.2f} mm)",
        )

    if VMTK_VALIDATE_ENDPOINT_SEEDS:
        endpoint_tol = (
            float(VMTK_MAX_CENTERLINE_ENDPOINT_SEED_DIST_MM)
            if VMTK_MAX_CENTERLINE_ENDPOINT_SEED_DIST_MM is not None
            else max(1.5, 3.0 * float(np.min(spacing)))
        )
        direct = max(float(np.linalg.norm(pts[0] - seed_start_mm)), float(np.linalg.norm(pts[-1] - seed_end_mm)))
        flipped = max(float(np.linalg.norm(pts[-1] - seed_start_mm)), float(np.linalg.norm(pts[0] - seed_end_mm)))
        if min(direct, flipped) > endpoint_tol:
            return (
                False,
                f"centerline endpoints far from requested seeds "
                f"(best max endpoint distance {min(direct, flipped):.2f} mm > {endpoint_tol:.2f} mm)",
            )

    ref_pts = [np.asarray(x, dtype=float) for x in (reference_centerline_points or []) if len(x)]
    if VMTK_VALIDATE_REFERENCE_CONNECTION and ref_pts:
        ref = np.vstack(ref_pts)
        ref_tol = (
            float(VMTK_REFERENCE_CONNECTION_TOL_MM)
            if VMTK_REFERENCE_CONNECTION_TOL_MM is not None
            else max(1.0, 2.5 * float(np.min(spacing)))
        )
        ref_tree = cKDTree(ref)
        end_dists, _ = ref_tree.query(np.vstack([pts[0], pts[-1]]), k=1)
        connection_dist = float(np.max(end_dists)) if require_both_reference_endpoints_connected else float(np.min(end_dists))
        if connection_dist > ref_tol:
            mode = "both endpoints connect" if require_both_reference_endpoints_connected else "at least one endpoint connects"
            return (
                False,
                f"centerline disconnected from existing vessel centerlines "
                f"({mode}; check distance {connection_dist:.2f} mm > {ref_tol:.2f} mm)",
            )

    return True, ""


def reference_centerline_points_from_results(path_results: List[dict]) -> List[np.ndarray]:
    refs = []
    for res in path_results:
        try:
            pts = centerline_points_from_result(res)
        except Exception:
            continue
        if len(pts):
            refs.append(pts)
    return refs


def refresh_centerline_result_metrics(res: dict) -> None:
    pts = centerline_points_from_result(res)
    s = cumulative_s(pts)
    radius = np.asarray(res.get("stenosis_detection_radius_mm", res["radius_mm"]), dtype=float)
    curvature = np.asarray(res["curvature_1_per_mm"], dtype=float)
    torsion = np.asarray(res["torsion_1_per_mm"], dtype=float)
    res["s_mm"] = s
    res["diameter_mm"] = 2.0 * radius
    res["length_mm"] = float(arc_length(pts))
    res["chord_length_mm"] = float(chord_length(pts))
    res["tortuosity_dm"] = float(tortuosity_dm(pts))
    res["turn_proxy"] = signed_turn_proxy(pts)
    k_valid = curvature[np.isfinite(curvature)]
    res["curvature_mean_1_per_mm"] = float(np.nanmean(curvature)) if k_valid.size else np.nan
    res["curvature_median_1_per_mm"] = float(np.nanmedian(curvature)) if k_valid.size else np.nan
    res["curvature_p95_1_per_mm"] = float(np.nanpercentile(k_valid, 95)) if k_valid.size else np.nan
    res["curvature_max_1_per_mm"] = float(np.nanmax(curvature)) if k_valid.size else np.nan
    res["inflection_count"] = int(inflection_count(curvature, res["turn_proxy"], INFLECT_KAPPA_MIN, INFLECT_SMOOTH_WIN))
    res["bend_peak_count"] = int(bend_peak_count(curvature, BEND_KAPPA_PEAK, INFLECT_SMOOTH_WIN))
    res.update(radius_stats(radius))
    res["taper_slope_mm_per_mm"] = float(taper_slope(s, radius))
    sten = detect_stenosis_segments(
        s=s, r=radius, threshold_pct=STENOSIS_THRESHOLD_PCT,
        min_segment_mm=STENOSIS_MIN_LEN_MM, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        pts=pts,
    )
    sten_pct_point, is_stenotic = stenosis_pointwise(
        s=s, r=radius, r_ref_per_point=sten.r_ref_per_point,
        threshold_pct=STENOSIS_THRESHOLD_PCT, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        min_segment_mm=STENOSIS_MIN_LEN_MM,
    )
    sten_raw_pct_point = stenosis_raw_percent(
        s=s, r=radius, r_ref_per_point=sten.r_ref_per_point,
        exclude_end_mm=0.0,
    )
    enlarg = detect_enlargement_segments(
        s=s, r=radius, threshold_pct=ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        pts=pts,
    )
    enlargement_pct_point, is_enlarged = enlargement_pointwise(
        s=s, r=radius, r_ref_per_point=enlarg.r_ref_per_point,
        threshold_pct=ENLARGEMENT_THRESHOLD_PCT, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM,
        pts=pts,
    )
    sten_pct_point, is_stenotic, enlargement_pct_point, is_enlarged = resolve_stenosis_enlargement_overlap(
        sten_pct_point, is_stenotic, enlargement_pct_point, is_enlarged
    )
    # CS-based enlargement binary flag (radius_mm is always cross-section radius)
    radius_cs = np.asarray(res["radius_mm"], dtype=float)
    enlarg_cs = detect_enlargement_segments(
        s=s, r=radius_cs, threshold_pct=ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        pts=pts,
    )
    _, is_enlarged_cs = enlargement_pointwise(
        s=s, r=radius_cs, r_ref_per_point=enlarg_cs.r_ref_per_point,
        threshold_pct=ENLARGEMENT_THRESHOLD_PCT, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, pts=pts,
    )
    is_enlarged_cs = np.where(is_stenotic == 1, 0, is_enlarged_cs).astype(int)
    siphon_mask = compute_siphon_mask(pts, s)
    res["siphon_mask_point"] = siphon_mask.astype(float)
    res["stenosis_percent_point"] = sten_pct_point
    res["stenosis_raw_percent_point"] = sten_raw_pct_point
    res["stenosis_core_candidate_point"] = (
        (sten_raw_pct_point >= STENOSIS_THRESHOLD_PCT) & np.isfinite(sten_raw_pct_point)
    ).astype(float)
    res["stenosis_support_candidate_point"] = (
        (sten_raw_pct_point >= min(STENOSIS_SUPPORT_THRESHOLD_PCT, STENOSIS_THRESHOLD_PCT))
        & np.isfinite(sten_raw_pct_point)
        & (s >= float(STENOSIS_EXCLUDE_END_MM))
    ).astype(float)
    res["is_stenotic"] = is_stenotic
    res["stenosis_reference_radius_point"] = sten.r_ref_per_point
    res["stenosis_threshold_radius_point"] = (1.0 - STENOSIS_THRESHOLD_PCT / 100.0) * sten.r_ref_per_point
    res["enlargement_percent_point"] = enlargement_pct_point
    res["is_enlarged"] = is_enlarged
    res["is_enlarged_mis"] = is_enlarged
    res["is_enlarged_cs"] = is_enlarged_cs
    res["enlargement_reference_radius_point"] = enlarg.r_ref_per_point
    res["enlargement_threshold_radius_point"] = (1.0 + ENLARGEMENT_THRESHOLD_PCT / 100.0) * enlarg.r_ref_per_point
    stenotic_radius = radius[(is_stenotic == 1) & np.isfinite(radius)]
    enlarged_radius = radius[(is_enlarged == 1) & np.isfinite(radius)]
    res["radius_ref_mm"] = float(sten.r_ref)
    res["radius_min_mm"] = float(sten.r_min)
    res["radius_min_stenotic_mm"] = float(np.min(stenotic_radius)) if stenotic_radius.size else np.nan
    res["stenosis_percent_max"] = float(sten.percent_stenosis_max)
    res["degree_of_stenosis_pct"] = float(sten.percent_stenosis_max)
    res["stenosis_length_total_mm"] = float(stenosis_total_length(s, is_stenotic))
    res["stenosis_segments_n"] = int(len(sten.segments_point_idx))
    res["stenosis_segments_point_idx"] = json.dumps(sten.segments_point_idx)
    res["stenosis_segments_detail_json"] = segment_detail_json(s, sten_pct_point, sten.segments_point_idx)
    res["enlargement_radius_ref_mm"] = float(enlarg.r_ref)
    res["enlargement_radius_max_mm"] = float(enlarg.r_max)
    res["radius_max_enlarged_mm"] = float(np.max(enlarged_radius)) if enlarged_radius.size else np.nan
    res["enlargement_percent_max"] = float(enlarg.percent_enlargement_max)
    res["enlargement_length_total_mm"] = float(stenosis_total_length(s, is_enlarged))
    res["enlargement_segments_n"] = int(len(enlarg.segments_point_idx))
    res["enlargement_segments_point_idx"] = json.dumps(enlarg.segments_point_idx)
    res["enlargement_segments_detail_json"] = segment_detail_json(s, enlargement_pct_point, enlarg.segments_point_idx)


def slice_centerline_result(res: dict, start_idx: int) -> dict:
    n = len(res["x_mm"])
    start_idx = int(np.clip(start_idx, 0, max(0, n - 1)))
    sliced = dict(res)
    per_point_keys = [
        "x_mm", "y_mm", "z_mm", "s_mm",
        "radius_mm", "diameter_mm", "maximum_inscribed_sphere_radius_mm", "stenosis_detection_radius_mm",
        "curvature_1_per_mm", "torsion_1_per_mm", "turn_proxy",
        "stenosis_raw_percent_point", "stenosis_core_candidate_point",
        "stenosis_support_candidate_point", "stenosis_percent_point", "is_stenotic",
        "stenosis_reference_radius_point", "stenosis_threshold_radius_point",
        "enlargement_raw_percent_point", "enlargement_core_candidate_point",
        "enlargement_support_candidate_point", "enlargement_percent_point",
        "is_enlarged", "is_enlarged_cs", "is_enlarged_mis",
        "enlargement_reference_radius_point", "enlargement_threshold_radius_point",
        "siphon_mask_point",
        "donut_loop_indices", "donut_arm_indices", "donut_arm_labels",
        "tree_point_labels", "tree_point_paths", "tree_point_depths",
    ]
    for key in per_point_keys:
        if key in sliced:
            sliced[key] = np.asarray(sliced[key])[start_idx:].copy()
    sliced["overlap_pruned_start_points"] = int(start_idx)
    refresh_centerline_result_metrics(sliced)
    return sliced


def save_centerline_result_vtps(res: dict, centerline_dir: Optional[str], centerline_radius_dir: Optional[str]) -> None:
    pts = centerline_points_from_result(res)
    if len(pts) < 2:
        return
    stenosis_binary = np.asarray(res["is_stenotic"], dtype=np.float64)
    enlargement_binary = np.asarray(res.get("is_enlarged", np.zeros(len(pts))), dtype=np.float64)
    enlargement_binary_cs = np.asarray(res.get("is_enlarged_cs", enlargement_binary), dtype=np.float64)
    enlargement_binary_mis = np.asarray(res.get("is_enlarged_mis", enlargement_binary), dtype=np.float64)
    tree_meta = default_tree_point_metadata(len(pts), res.get("tree_label", "trunk"), res.get("tree_path", ""))
    tree_depth_array = (
        np.asarray(res["tree_point_depths"], dtype=float)
        if "tree_point_depths" in res and len(res["tree_point_depths"]) == len(pts)
        else tree_meta["tree_depth"]
    )
    extra_arrays = []
    if "donut_loop_indices" in res and len(res["donut_loop_indices"]) == len(pts):
        extra_arrays.append((np.asarray(res["donut_loop_indices"], dtype=float), "DonutLoopIndex"))
    if "donut_arm_indices" in res and len(res["donut_arm_indices"]) == len(pts):
        extra_arrays.append((np.asarray(res["donut_arm_indices"], dtype=float), "DonutArmIndex"))
    if centerline_dir:
        poly = build_polyline_polydata(points=pts, arrays=[
            (res["curvature_1_per_mm"], "Curvature"),
            (res["torsion_1_per_mm"], "Torsion"),
            (res["radius_mm"], "EffectiveRadius"),
            (res["radius_mm"], "CrossSectionRadius"),
            (res.get("maximum_inscribed_sphere_radius_mm", np.full(len(pts), np.nan)), "MaximumInscribedSphereRadius"),
            (res.get("stenosis_detection_radius_mm", res["radius_mm"]), "StenosisDetectionRadius"),
            (res.get("stenosis_reference_radius_point", np.full(len(pts), np.nan)), "StenosisReferenceRadius"),
            (res.get("stenosis_threshold_radius_point", np.full(len(pts), np.nan)), "StenosisThresholdRadius"),
            (res.get("stenosis_raw_percent_point", np.full(len(pts), np.nan)), "StenosisRawPercent"),
            (res.get("stenosis_core_candidate_point", np.full(len(pts), np.nan)), "StenosisCoreCandidate"),
            (res.get("stenosis_support_candidate_point", np.full(len(pts), np.nan)), "StenosisSupportCandidate"),
            (res.get("stenosis_percent_point", np.full(len(pts), np.nan)), "StenosisPercent"),
            (stenosis_binary, "StenosisBinary"),
            (res.get("enlargement_reference_radius_point", np.full(len(pts), np.nan)), "EnlargementReferenceRadius"),
            (res.get("enlargement_threshold_radius_point", np.full(len(pts), np.nan)), "EnlargementThresholdRadius"),
            (res.get("enlargement_raw_percent_point", np.full(len(pts), np.nan)), "EnlargementRawPercent"),
            (res.get("enlargement_core_candidate_point", np.full(len(pts), np.nan)), "EnlargementCoreCandidate"),
            (res.get("enlargement_support_candidate_point", np.full(len(pts), np.nan)), "EnlargementSupportCandidate"),
            (res.get("enlargement_percent_point", np.full(len(pts), np.nan)), "EnlargementPercent"),
            (enlargement_binary, "EnlargementBinary"),
            (enlargement_binary_cs, "EnlargementBinaryCS"),
            (enlargement_binary_mis, "EnlargementBinaryMIS"),
            (res.get("siphon_mask_point", np.zeros(len(pts), dtype=float)), "SiphonMask"),
            (np.full(len(pts), res.get("overlap_pruned_start_points", 0), dtype=float), "OverlapPrunedStartPoints"),
            (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
        ] + extra_arrays)
        if "tree_point_labels" in res and len(res["tree_point_labels"]) == len(pts):
            arr = numpy_support.numpy_to_vtk(np.asarray(tree_depth_array, dtype=np.float64), deep=True)
            arr.SetName("TreeDepth")
            poly.GetPointData().AddArray(arr)
            add_string_point_array(poly, [str(x) for x in res["tree_point_labels"]], "TreeLabel")
            if "tree_point_paths" in res and len(res["tree_point_paths"]) == len(pts):
                add_string_point_array(poly, [str(x) for x in res["tree_point_paths"]], "TreePath")
            else:
                add_string_point_array(poly, [str(res.get("tree_path", ""))] * len(pts), "TreePath")
        else:
            add_tree_metadata_point_arrays(poly, tree_meta)
        if "donut_arm_labels" in res and len(res["donut_arm_labels"]) == len(pts):
            add_string_point_array(poly, [str(x) for x in res["donut_arm_labels"]], "DonutArmLabel")
        save_vtp(poly, os.path.join(centerline_dir, res["path_id"] + ".vtp"))
    if centerline_radius_dir:
        poly_radius = build_radius_tube_polydata(points=pts, radius=np.asarray(res["radius_mm"], dtype=float), arrays=[
            (res["curvature_1_per_mm"], "Curvature"),
            (res["torsion_1_per_mm"], "Torsion"),
            (res["radius_mm"], "EffectiveRadius"),
            (res["radius_mm"], "CrossSectionRadius"),
            (res.get("maximum_inscribed_sphere_radius_mm", np.full(len(pts), np.nan)), "MaximumInscribedSphereRadius"),
            (res.get("stenosis_detection_radius_mm", res["radius_mm"]), "StenosisDetectionRadius"),
            (res.get("stenosis_reference_radius_point", np.full(len(pts), np.nan)), "StenosisReferenceRadius"),
            (res.get("stenosis_threshold_radius_point", np.full(len(pts), np.nan)), "StenosisThresholdRadius"),
            (res.get("stenosis_raw_percent_point", np.full(len(pts), np.nan)), "StenosisRawPercent"),
            (res.get("stenosis_core_candidate_point", np.full(len(pts), np.nan)), "StenosisCoreCandidate"),
            (res.get("stenosis_support_candidate_point", np.full(len(pts), np.nan)), "StenosisSupportCandidate"),
            (res.get("stenosis_percent_point", np.full(len(pts), np.nan)), "StenosisPercent"),
            (stenosis_binary, "StenosisBinary"),
            (res.get("enlargement_reference_radius_point", np.full(len(pts), np.nan)), "EnlargementReferenceRadius"),
            (res.get("enlargement_threshold_radius_point", np.full(len(pts), np.nan)), "EnlargementThresholdRadius"),
            (res.get("enlargement_raw_percent_point", np.full(len(pts), np.nan)), "EnlargementRawPercent"),
            (res.get("enlargement_core_candidate_point", np.full(len(pts), np.nan)), "EnlargementCoreCandidate"),
            (res.get("enlargement_support_candidate_point", np.full(len(pts), np.nan)), "EnlargementSupportCandidate"),
            (res.get("enlargement_percent_point", np.full(len(pts), np.nan)), "EnlargementPercent"),
            (enlargement_binary, "EnlargementBinary"),
            (enlargement_binary_cs, "EnlargementBinaryCS"),
            (enlargement_binary_mis, "EnlargementBinaryMIS"),
            (res.get("siphon_mask_point", np.zeros(len(pts), dtype=float)), "SiphonMask"),
            (tree_depth_array, "TreeDepth"),
            (np.full(len(pts), res.get("overlap_pruned_start_points", 0), dtype=float), "OverlapPrunedStartPoints"),
            (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
        ] + extra_arrays)
        save_vtp(poly_radius, os.path.join(centerline_radius_dir, res["path_id"] + "_radius.vtp"))


def suppress_enlargements_near_centerline_starts(
    path_results: List[dict],
    spacing,
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
) -> int:
    if not SUPPRESS_ENLARGEMENT_NEAR_CENTERLINE_STARTS or len(path_results) <= 1:
        return 0
    spacing = np.asarray(spacing, dtype=float)
    exclude_mm = (
        float(ENLARGEMENT_CENTERLINE_START_EXCLUDE_MM)
        if ENLARGEMENT_CENTERLINE_START_EXCLUDE_MM is not None
        else max(1.5, 3.0 * float(np.min(spacing)))
    )
    starts = []
    for idx, res in enumerate(path_results):
        pts = centerline_points_from_result(res)
        if len(pts):
            starts.append((idx, pts[0]))
    if len(starts) <= 1:
        return 0

    changed = 0
    for idx, res in enumerate(path_results):
        pts = centerline_points_from_result(res)
        if len(pts) == 0:
            continue
        other_starts = np.asarray([p for j, p in starts if j != idx], dtype=float)
        if len(other_starts) == 0:
            continue
        flag = np.asarray(res.get("is_enlarged", np.zeros(len(pts), dtype=int)), dtype=int)
        if not np.any(flag == 1):
            continue
        dist, _ = cKDTree(other_starts).query(pts, k=1)
        suppress = np.isfinite(dist) & (dist <= exclude_mm) & (flag == 1)
        if not np.any(suppress):
            continue
        before = int(np.sum(flag == 1))
        flag[suppress] = 0
        res["is_enlarged"] = flag
        refresh_enlargement_summary_from_flags(res)
        after = int(np.sum(np.asarray(res.get("is_enlarged", []), dtype=int) == 1))
        changed += 1
        print(
            f"    [enlargement] Suppressed {before - after} bifurcation-near enlarged point(s) "
            f"in {res.get('path_id', '<unknown>')} (within {exclude_mm:.2f} mm of another centerline start)."
        )
        save_centerline_result_vtps(res, centerline_dir, centerline_radius_dir)
    return changed


def prune_overlapping_final_centerlines(
    path_results: List[dict],
    spacing,
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
) -> Tuple[List[dict], set, int]:
    if not PRUNE_OVERLAPPING_FINAL_CENTERLINE_PREFIXES or len(path_results) <= 1:
        return path_results, set(), 0

    spacing = np.asarray(spacing, dtype=float)
    overlap_tol = (
        float(FINAL_CENTERLINE_OVERLAP_TOL_MM)
        if FINAL_CENTERLINE_OVERLAP_TOL_MM is not None
        else max(0.75, 2.0 * float(np.min(spacing)))
    )
    ordered = sorted(path_results, key=lambda r: int(r.get("path_index", 0)))
    kept = []
    ref_points = []
    discarded = set()
    trimmed_count = 0

    for res in ordered:
        pts = centerline_points_from_result(res)
        if len(pts) < 2 or not ref_points:
            kept.append(res)
            if len(pts):
                ref_points.append(pts)
            continue

        ref = np.vstack(ref_points)
        dist, _ = cKDTree(ref).query(pts, k=1)
        close = np.isfinite(dist) & (dist <= overlap_tol)
        prefix_n = 0
        for is_close in close:
            if is_close:
                prefix_n += 1
            else:
                break

        if prefix_n <= 1:
            kept.append(res)
            ref_points.append(pts)
            continue

        start_idx = max(0, prefix_n - 1)
        if len(pts) - start_idx < int(FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE):
            discarded.add(str(res["path_id"]))
            print(
                f"    [final overlap] Discarding {res['path_id']}: only "
                f"{len(pts) - start_idx} point(s) after removing shared trunk prefix "
                f"({prefix_n} point(s), tol {overlap_tol:.2f} mm)."
            )
            continue

        trimmed = slice_centerline_result(res, start_idx)
        kept.append(trimmed)
        ref_points.append(centerline_points_from_result(trimmed))
        trimmed_count += 1
        print(
            f"    [final overlap] Trimmed shared trunk from {res['path_id']}: "
            f"removed {start_idx} point(s), kept {len(trimmed['x_mm'])} "
            f"(tol {overlap_tol:.2f} mm)."
        )

    if discarded:
        from .tree_regions import remove_discarded_tree_path_outputs

        remove_discarded_tree_path_outputs(discarded, centerline_dir, centerline_radius_dir)
    for res in kept:
        if int(res.get("overlap_pruned_start_points", 0)) > 0:
            save_centerline_result_vtps(res, centerline_dir, centerline_radius_dir)
    return kept, discarded, trimmed_count


def run_vmtk_centerline_for_path_with_tip_retries(
    surface,
    path_mm: np.ndarray,
    spacing,
    context: str = "",
    reference_centerline_points: Optional[List[np.ndarray]] = None,
    require_both_reference_endpoints_connected: bool = False,
):
    pts = np.asarray(path_mm, dtype=float)
    if len(pts) < 2:
        raise RuntimeError("Cannot run VMTK: seed path has fewer than 2 points.")

    attempts = [(0.0, pts[0], pts[-1])]
    if RETRY_VMTK_WITH_TRIMMED_SEEDS:
        for trim_mm in VMTK_SEED_TRIM_RETRY_MM:
            pair = trimmed_seed_pair_from_path(pts, float(trim_mm))
            if pair is not None:
                attempts.append((float(trim_mm), pair[0], pair[1]))

    errors = []
    for trim_mm, seed_start_mm, seed_end_mm in attempts:
        try:
            if trim_mm > 0:
                print(
                    f"    [centerline retry] {context} retry with both seeds trimmed "
                    f"{trim_mm:.2f} mm inward."
                )
                pair = trimmed_seed_pair_from_path(pts, float(trim_mm))
                if pair is None:
                    raise RuntimeError("trimmed seed pair unavailable")
                sub_pts = np.vstack([pair[0], pair[1]]).astype(float)
            else:
                sub_pts = pts
            centerline_poly = _centerline_poly_from_path_mm(sub_pts)
            ok, reason = validate_vmtk_centerline_attempt(
                centerline_poly=centerline_poly,
                seed_path_mm=pts,
                seed_start_mm=seed_start_mm,
                seed_end_mm=seed_end_mm,
                spacing=spacing,
                reference_centerline_points=reference_centerline_points,
                require_both_reference_endpoints_connected=require_both_reference_endpoints_connected,
            )
            if not ok:
                raise RuntimeError(reason)
            if trim_mm > 0:
                print(f"    [centerline retry] {context} succeeded after {trim_mm:.2f} mm seed trim.")
            return centerline_poly, seed_start_mm, seed_end_mm, trim_mm
        except Exception as e:
            errors.append(f"trim={trim_mm:.2f} mm: {e}")
            if trim_mm == 0:
                print(f"    [centerline retry] {context} original endpoint seeds failed: {e}")

    raise RuntimeError("Centerline generation failed for original and trimmed seeds. " + " | ".join(errors))


def analyze_centerline_poly(
    centerline_poly,
    surface,
    mask_cc: np.ndarray,
    spacing,
    vessel_info: VesselInfo,
    multilabel: np.ndarray,
    mapping: dict,
    force_start_to_end: bool = True,
    preferred_start_mm: Optional[np.ndarray] = None,
    save_centerline_vtp: Optional[str] = None,
    save_centerline_radius_vtp: Optional[str] = None,
) -> dict:
    spacing = np.asarray(spacing, dtype=float)
    pts = numpy_support.vtk_to_numpy(centerline_poly.GetPoints().GetData())
    kappa_vmtk = extract_point_data_array(centerline_poly, "Curvature", len(pts))
    torsion = extract_point_data_array(centerline_poly, "Torsion", len(pts))
    radius_mis = extract_point_data_array(centerline_poly, "MaximumInscribedSphereRadius", len(pts))

    if not force_start_to_end:
        pts_oriented, flipped = orient_centerline_points_by_flow(pts, vessel_info, multilabel, spacing, mapping)
        if flipped:
            print("    [direction] Reversed — flipping.")
            pts = pts_oriented
            kappa_vmtk = kappa_vmtk[::-1]
            torsion = torsion[::-1]
            radius_mis = radius_mis[::-1]
        else:
            print("    [direction] OK.")
    elif preferred_start_mm is not None and len(pts) >= 2:
        preferred_start_mm = np.asarray(preferred_start_mm, dtype=float)
        d_first = float(np.linalg.norm(pts[0] - preferred_start_mm))
        d_last = float(np.linalg.norm(pts[-1] - preferred_start_mm))
        if d_last < d_first:
            print(
                f"    [direction] Reversed to root seed "
                f"(first={d_first:.2f} mm, last={d_last:.2f} mm)."
            )
            pts = pts[::-1]
            kappa_vmtk = kappa_vmtk[::-1]
            torsion = torsion[::-1]
            radius_mis = radius_mis[::-1]
        else:
            print(f"    [direction] Root seed orientation OK (first={d_first:.2f} mm, last={d_last:.2f} mm).")

    original_n = len(pts)
    pts_before_resample = pts.copy()
    radius_mis_before_resample = radius_mis.copy()
    pts_resampled = resample_generated_centerline_points(pts)
    resampled_centerline = len(pts_resampled) != original_n or (
        len(pts_resampled) == original_n and len(pts) and not np.allclose(pts_resampled, pts)
    )
    if resampled_centerline:
        print(
            f"    [centerline] Resampled {original_n} -> {len(pts_resampled)} point(s) "
            f"at {float(CENTERLINE_RESAMPLE_STEP_MM):.3f} mm arc-length spacing."
        )
        pts = pts_resampled
        kappa_vmtk = discrete_curvature(pts)
        torsion = discrete_torsion(pts)
        radius_mis = resample_point_data_by_arclength(pts_before_resample, pts, radius_mis_before_resample)
    else:
        fallback_torsion = discrete_torsion(pts)
        torsion = np.where(np.isfinite(torsion), torsion, fallback_torsion)

    radius_cross, cross_section_area = compute_cross_section_radius(surface, pts)

    radius_area = radius_cross
    path_vox_cl = [tuple(np.clip(np.round(p / spacing).astype(int), 0, np.array(mask_cc.shape) - 1)) for p in pts]
    cs_valid = np.isfinite(radius_area) & (radius_area > 1e-6)
    if cs_valid.any():
        print(f"    Using cross-section radius ({cs_valid.sum()}/{len(radius_cross)} valid).")
        radius = radius_area.copy()
        if (~cs_valid).any():
            radius_edt = radius_from_edt(mask_cc.astype(bool), spacing, path_vox_cl)
            radius = np.where(cs_valid, radius_area, radius_edt)
    else:
        print("    Using EDT radius (cross-section unavailable).")
        radius = radius_from_edt(mask_cc.astype(bool), spacing, path_vox_cl)
        radius_area = radius.copy()
    detection_radius = select_caliber_detection_radius(radius_area, radius_mis)
    if str(RADIUS_SOURCE_FOR_CALIBER_DETECTION).lower() in {"maximum_inscribed_sphere", "mis", "vmtk"}:
        n_mis = int(np.sum(np.isfinite(radius_mis) & (radius_mis > 1e-6)))
        print(f"    Using maximum-inscribed-sphere radius for stenosis/enlargement detection ({n_mis}/{len(radius_mis)} valid).")

    s = cumulative_s(pts)
    siphon_mask = compute_siphon_mask(pts, s)
    L = arc_length(pts); C = chord_length(pts); T = tortuosity_dm(pts)
    kappa = kappa_vmtk if not np.all(np.isnan(kappa_vmtk)) else discrete_curvature(pts)
    k_valid = kappa[np.isfinite(kappa)]
    k_mean = float(np.nanmean(kappa)) if k_valid.size else np.nan
    k_median = float(np.nanmedian(kappa)) if k_valid.size else np.nan
    k_p95 = float(np.nanpercentile(k_valid, 95)) if k_valid.size else np.nan
    k_max = float(np.nanmax(kappa)) if k_valid.size else np.nan
    turn_proxy = signed_turn_proxy(pts)
    n_inflections = inflection_count(kappa, turn_proxy, INFLECT_KAPPA_MIN, INFLECT_SMOOTH_WIN)
    n_bends = bend_peak_count(kappa, BEND_KAPPA_PEAK, INFLECT_SMOOTH_WIN)
    r_stats = radius_stats(radius_area)
    taper = taper_slope(s, detection_radius)
    sten = detect_stenosis_segments(
        s=s, r=detection_radius, threshold_pct=STENOSIS_THRESHOLD_PCT,
        min_segment_mm=STENOSIS_MIN_LEN_MM, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        pts=pts,
    )
    sten_pct_point, is_stenotic = stenosis_pointwise(
        s=s, r=detection_radius, r_ref_per_point=sten.r_ref_per_point,
        threshold_pct=STENOSIS_THRESHOLD_PCT, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        min_segment_mm=STENOSIS_MIN_LEN_MM,
    )
    sten_raw_pct_point = stenosis_raw_percent(
        s=s, r=detection_radius, r_ref_per_point=sten.r_ref_per_point,
        exclude_end_mm=0.0,
    )
    candidate_valid = stenosis_valid_mask_from_ends(s, float(STENOSIS_CANDIDATE_EXCLUDE_END_MM))
    sten_core_candidate = (
        (sten_raw_pct_point >= STENOSIS_THRESHOLD_PCT) & np.isfinite(sten_raw_pct_point)
        & candidate_valid
    ).astype(np.float64)
    sten_support_candidate = (
        (sten_raw_pct_point >= min(STENOSIS_SUPPORT_THRESHOLD_PCT, STENOSIS_THRESHOLD_PCT))
        & np.isfinite(sten_raw_pct_point)
        & candidate_valid
    ).astype(np.float64)
    enlarg = detect_enlargement_segments(
        s=s, r=radius_area, threshold_pct=ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        pts=pts,
    )
    enlargement_pct_point, is_enlarged = enlargement_pointwise(
        s=s, r=radius_area, r_ref_per_point=enlarg.r_ref_per_point,
        threshold_pct=ENLARGEMENT_THRESHOLD_PCT, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM,
        pts=pts,
    )
    enlarg_raw_pct_point = np.full(len(pts), np.nan, dtype=float)
    enlarg_ref_valid = np.isfinite(radius_area) & np.isfinite(enlarg.r_ref_per_point) & (enlarg.r_ref_per_point > 1e-8)
    enlarg_raw_pct_point[enlarg_ref_valid] = (radius_area[enlarg_ref_valid] / enlarg.r_ref_per_point[enlarg_ref_valid] - 1.0) * 100.0
    enlarg_candidate_valid = stenosis_valid_mask_from_ends(s, float(ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM))
    enlarg_core_candidate = (
        (enlarg_raw_pct_point >= ENLARGEMENT_THRESHOLD_PCT) & np.isfinite(enlarg_raw_pct_point)
        & enlarg_candidate_valid
    ).astype(np.float64)
    enlarg_support_candidate = (
        (enlarg_raw_pct_point >= min(ENLARGEMENT_SUPPORT_THRESHOLD_PCT, ENLARGEMENT_THRESHOLD_PCT))
        & np.isfinite(enlarg_raw_pct_point)
        & enlarg_candidate_valid
    ).astype(np.float64)
    sten_pct_point, is_stenotic, enlargement_pct_point, is_enlarged = resolve_stenosis_enlargement_overlap(
        sten_pct_point, is_stenotic, enlargement_pct_point, is_enlarged
    )
    sten_total_len = stenosis_total_length(s, is_stenotic)
    stenotic_radius = detection_radius[(is_stenotic == 1) & np.isfinite(detection_radius)]
    enlarged_radius = radius_area[(is_enlarged == 1) & np.isfinite(radius_area)]
    stenosis_binary = is_stenotic.astype(np.float64)
    enlargement_binary = is_enlarged.astype(np.float64)
    # CS-based enlargement is already computed above (radius_area); MIS-based is new.
    is_enlarged_cs = is_enlarged
    enlarg_mis = detect_enlargement_segments(
        s=s, r=detection_radius, threshold_pct=ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        pts=pts,
    )
    _, is_enlarged_mis = enlargement_pointwise(
        s=s, r=detection_radius, r_ref_per_point=enlarg_mis.r_ref_per_point,
        threshold_pct=ENLARGEMENT_THRESHOLD_PCT, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, pts=pts,
    )
    is_enlarged_mis = np.where(is_stenotic == 1, 0, is_enlarged_mis).astype(int)

    if save_centerline_vtp:
        poly = build_polyline_polydata(points=pts, arrays=[
            (kappa_vmtk, "Curvature"), (torsion, "Torsion"),
            (radius_area, "EffectiveRadius"), (radius_area, "CrossSectionRadius"),
            (radius_mis, "MaximumInscribedSphereRadius"),
            (detection_radius, "StenosisDetectionRadius"),
            (cross_section_area, "CrossSectionArea"),
            (sten.r_ref_per_point, "StenosisReferenceRadius"),
            ((1.0 - STENOSIS_THRESHOLD_PCT / 100.0) * sten.r_ref_per_point, "StenosisThresholdRadius"),
            (sten_raw_pct_point, "StenosisRawPercent"),
            (sten_core_candidate, "StenosisCoreCandidate"),
            (sten_support_candidate, "StenosisSupportCandidate"),
            (sten_pct_point, "StenosisPercent"), (stenosis_binary, "StenosisBinary"),
            (enlarg.r_ref_per_point, "EnlargementReferenceRadius"),
            ((1.0 + ENLARGEMENT_THRESHOLD_PCT / 100.0) * enlarg.r_ref_per_point, "EnlargementThresholdRadius"),
            (enlarg_raw_pct_point, "EnlargementRawPercent"),
            (enlarg_core_candidate, "EnlargementCoreCandidate"),
            (enlarg_support_candidate, "EnlargementSupportCandidate"),
            (enlargement_pct_point, "EnlargementPercent"), (enlargement_binary, "EnlargementBinary"),
            (is_enlarged_cs.astype(np.float64), "EnlargementBinaryCS"),
            (is_enlarged_mis.astype(np.float64), "EnlargementBinaryMIS"),
            (siphon_mask.astype(np.float64), "SiphonMask"),
            (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
        ])
        add_tree_metadata_point_arrays(poly, default_tree_point_metadata(len(pts), "trunk"))
        save_vtp(poly, save_centerline_vtp)
    if save_centerline_radius_vtp:
        tree_meta = default_tree_point_metadata(len(pts), "trunk", "")
        poly_radius = build_radius_tube_polydata(points=pts, radius=radius, arrays=[
            (kappa_vmtk, "Curvature"), (torsion, "Torsion"),
            (radius, "EffectiveRadius"), (cross_section_area, "CrossSectionArea"),
            (radius_area, "CrossSectionRadius"),
            (radius_mis, "MaximumInscribedSphereRadius"),
            (detection_radius, "StenosisDetectionRadius"),
            (sten.r_ref_per_point, "StenosisReferenceRadius"),
            ((1.0 - STENOSIS_THRESHOLD_PCT / 100.0) * sten.r_ref_per_point, "StenosisThresholdRadius"),
            (sten_raw_pct_point, "StenosisRawPercent"),
            (sten_core_candidate, "StenosisCoreCandidate"),
            (sten_support_candidate, "StenosisSupportCandidate"),
            (sten_pct_point, "StenosisPercent"), (stenosis_binary, "StenosisBinary"),
            (enlarg.r_ref_per_point, "EnlargementReferenceRadius"),
            ((1.0 + ENLARGEMENT_THRESHOLD_PCT / 100.0) * enlarg.r_ref_per_point, "EnlargementThresholdRadius"),
            (enlarg_raw_pct_point, "EnlargementRawPercent"),
            (enlarg_core_candidate, "EnlargementCoreCandidate"),
            (enlarg_support_candidate, "EnlargementSupportCandidate"),
            (enlargement_pct_point, "EnlargementPercent"), (enlargement_binary, "EnlargementBinary"),
            (is_enlarged_cs.astype(np.float64), "EnlargementBinaryCS"),
            (is_enlarged_mis.astype(np.float64), "EnlargementBinaryMIS"),
            (siphon_mask.astype(np.float64), "SiphonMask"),
            (tree_meta["tree_depth"], "TreeDepth"),
            (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
        ])
        save_vtp(poly_radius, save_centerline_radius_vtp)

    return {
        "x_mm": pts[:, 0], "y_mm": pts[:, 1], "z_mm": pts[:, 2], "s_mm": s,
        "radius_mm": radius_area, "diameter_mm": 2.0 * radius_area,
        "maximum_inscribed_sphere_radius_mm": radius_mis,
        "stenosis_detection_radius_mm": detection_radius,
        "curvature_1_per_mm": kappa, "torsion_1_per_mm": torsion, "turn_proxy": turn_proxy,
        "stenosis_raw_percent_point": sten_raw_pct_point,
        "stenosis_core_candidate_point": sten_core_candidate,
        "stenosis_support_candidate_point": sten_support_candidate,
        "stenosis_percent_point": sten_pct_point, "is_stenotic": is_stenotic,
        "stenosis_reference_radius_point": sten.r_ref_per_point,
        "stenosis_threshold_radius_point": (1.0 - STENOSIS_THRESHOLD_PCT / 100.0) * sten.r_ref_per_point,
        "enlargement_raw_percent_point": enlarg_raw_pct_point,
        "enlargement_core_candidate_point": enlarg_core_candidate,
        "enlargement_support_candidate_point": enlarg_support_candidate,
        "enlargement_percent_point": enlargement_pct_point, "is_enlarged": is_enlarged,
        "is_enlarged_cs": is_enlarged_cs, "is_enlarged_mis": is_enlarged_mis,
        "siphon_mask_point": siphon_mask.astype(float),
        "enlargement_reference_radius_point": enlarg.r_ref_per_point,
        "enlargement_threshold_radius_point": (1.0 + ENLARGEMENT_THRESHOLD_PCT / 100.0) * enlarg.r_ref_per_point,
        "length_mm": float(L), "chord_length_mm": float(C), "tortuosity_dm": float(T),
        "curvature_mean_1_per_mm": float(k_mean), "curvature_median_1_per_mm": float(k_median),
        "curvature_p95_1_per_mm": float(k_p95), "curvature_max_1_per_mm": float(k_max),
        "inflection_count": int(n_inflections), "bend_peak_count": int(n_bends),
        "radius_ref_mm": float(sten.r_ref), "radius_min_mm": float(sten.r_min),
        "radius_min_stenotic_mm": float(np.min(stenotic_radius)) if stenotic_radius.size else np.nan,
        "stenosis_percent_max": float(sten.percent_stenosis_max),
        "degree_of_stenosis_pct": float(sten.percent_stenosis_max),
        "stenosis_length_total_mm": float(sten_total_len),
        "stenosis_segments_n": int(len(sten.segments_point_idx)),
        "stenosis_segments_point_idx": json.dumps(sten.segments_point_idx),
        "stenosis_segments_detail_json": segment_detail_json(s, sten_pct_point, sten.segments_point_idx),
        "enlargement_radius_ref_mm": float(enlarg.r_ref),
        "enlargement_radius_max_mm": float(enlarg.r_max),
        "enlargement_percent_max": float(enlarg.percent_enlargement_max),
        "enlargement_length_total_mm": float(stenosis_total_length(s, is_enlarged)),
        "enlargement_segments_n": int(len(enlarg.segments_point_idx)),
        "enlargement_segments_point_idx": json.dumps(enlarg.segments_point_idx),
        "enlargement_segments_detail_json": segment_detail_json(s, enlargement_pct_point, enlarg.segments_point_idx),
        "taper_slope_mm_per_mm": float(taper),
        "radius_max_enlarged_mm": float(np.max(enlarged_radius)) if enlarged_radius.size else np.nan,
        **r_stats,
    }
