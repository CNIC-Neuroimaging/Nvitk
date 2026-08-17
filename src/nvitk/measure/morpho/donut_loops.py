"""Donut-loop detection, isolated-arm VMTK, and loop component processing."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from vtk.util import numpy_support

from .anatomy import choose_root_endpoint, make_loop_region_path_id, make_path_id
from .anatomy_axes import MorphoContext, default_morpho_context
from .caliber import (
    detect_enlargement_segments,
    detect_stenosis_segments,
    enlargement_pointwise,
    resolve_stenosis_enlargement_overlap,
    segment_detail_json,
    select_caliber_detection_radius,
    stenosis_pointwise,
    stenosis_raw_percent,
    stenosis_total_length,
)
from .centerlines import (
    analyze_centerline_poly,
    centerline_points_from_result,
    deduplicate_path_results,
    prune_overlapping_final_centerlines,
    reference_centerline_points_from_results,
    run_vmtk_centerline_for_path_with_tip_retries,
    suppress_enlargements_near_centerline_starts,
)
from nvitk.measure.morphometrics_config import (
    CENTERLINE_RESAMPLE_STEP_MM,
    DONUT_ARM_ENDPOINT_BLEND_MM,
    DONUT_ARM_END_TRIM_RADIUS_FACTOR,
    DONUT_ARM_MASK_MARGIN_MM,
    ENLARGEMENT_EXCLUDE_END_MM,
    ENLARGEMENT_MIN_LEN_MM,
    ENLARGEMENT_THRESHOLD_PCT,
    EXPORT_ANATOMIC_SPLIT_CENTERLINES,
    FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE,
    PRESERVE_DONUT_ARM_ENDPOINTS,
    RUN_VMTK_ON_ISOLATED_DONUT_ARM,
    SAVE_DONUT_LOOP_DEBUG,
    SAVE_UNSELECTED_DONUT_ARM_SKELETON,
    SKELETON_MAX_STEPS,
    SKELETON_STEP_MM,
    SKELETON_TANGENT_NPTS,
    SKELETON_WALL_TOL_MM,
    STENOSIS_EXCLUDE_END_MM,
    STENOSIS_MIN_LEN_MM,
    STENOSIS_SUPPORT_THRESHOLD_PCT,
    STENOSIS_THRESHOLD_PCT,
    TRIM_DONUT_ARM_ENDS_BY_RADIUS,
    TRIM_DONUT_ARM_VMTK_TO_SKELETON_ENDPOINTS,
    USE_SKELETON_CONNECTORS_AFTER_TRIM,
)
from .geometry import (
    arc_length,
    cumulative_s,
    extend_path_to_walls,
    mean_distance_to_polyline_points,
    resample_generated_centerline_points,
    splice_skeleton_end_connectors,
    trim_polyline_ends_by_length,
    trim_polyline_overlap_with_reference_ends,
)
from .io_utils import safe_filename, safe_sheet_name
from .metrics import compute_branchpoint_metrics, discrete_curvature, discrete_torsion, radius_from_edt
from .models import SkeletonTree, VesselInfo
from .skeleton import dijkstra_dist_from_root, skeleton_total_graph_length_mm
from .export_utils.summaries import build_vessel_points_dataframe
from .volumetry import anatomy_provenance_fields, component_volumetry_fields
from .surface import (
    add_string_point_array,
    build_donut_loop_debug_polydata,
    build_polyline_polydata,
    compute_cross_section_radius,
    mask_to_surface,
    save_vtp,
)
from .tree_regions import ordered_terminal_path_records, tree_path_label_for_terminal
from .tree_segments import save_anatomic_fallback_centerlines

def isolate_donut_arm_mask(
    mask_cc: np.ndarray,
    tree: SkeletonTree,
    arm_nodes: List[int],
    competing_arm_nodes: List[int],
    spacing,
) -> np.ndarray:
    """Split *mask_cc* into the arm containing *arm_nodes*, by nearest-skeleton-node voting.

    Assigns each foreground voxel to whichever of the two competing arms'
    skeleton points is closer (with a small margin so ties favor the arm), then
    keeps only the connected component containing the arm's own seed voxels
    (majority vote if the seeds span more than one component).
    """
    spacing = np.asarray(spacing, dtype=float)
    mask_bool = mask_cc.astype(bool)
    vox = np.argwhere(mask_bool)
    if len(vox) == 0:
        return np.zeros_like(mask_bool, dtype=bool)

    arm_nodes = [int(x) for x in arm_nodes]
    competing_arm_nodes = [int(x) for x in competing_arm_nodes]
    arm_pts = tree.pts_vox[arm_nodes].astype(float) * spacing
    comp_pts = tree.pts_vox[competing_arm_nodes].astype(float) * spacing
    vox_mm = vox.astype(float) * spacing

    dist_arm, _ = cKDTree(arm_pts).query(vox_mm, k=1)
    dist_comp, _ = cKDTree(comp_pts).query(vox_mm, k=1)
    margin = float(DONUT_ARM_MASK_MARGIN_MM)
    keep = dist_arm <= (dist_comp + margin)

    arm_mask = np.zeros_like(mask_bool, dtype=bool)
    arm_mask[tuple(vox[keep].T)] = True

    seed_vox = tree.pts_vox[arm_nodes]
    seed_mask = np.zeros_like(mask_bool, dtype=bool)
    seed_mask[tuple(seed_vox.T)] = True
    labels, n = ndi.label(arm_mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return arm_mask

    seed_labels = labels[seed_mask & (labels > 0)]
    if len(seed_labels) == 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(np.argmax(sizes))

    chosen = int(pd.Series(seed_labels).mode().iloc[0])
    return labels == chosen


def run_isolated_donut_arm_vmtk(
    mask_cc: np.ndarray,
    tree: SkeletonTree,
    arm_nodes: List[int],
    competing_arm_nodes: List[int],
    spacing,
    path: str,
    loop_i: int,
    arm_i: int,
    region_name: str,
    reference_centerline_points: Optional[List[np.ndarray]] = None,
) -> Tuple[float, int, int, np.ndarray, dict]:
    """Extract a VMTK centerline for one donut-loop arm in isolation (its own surface, not the shared trunk mask).

    Isolating the arm mask (:func:`isolate_donut_arm_mask`) before running VMTK
    avoids the centerline snapping across the loop through the competing arm.
    Returns ``(length_mm, n_points, arm_mask_voxels, points, metrics_dict)``.
    """
    spacing = np.asarray(spacing, dtype=float)
    arm_mask = isolate_donut_arm_mask(mask_cc, tree, arm_nodes, competing_arm_nodes, spacing)
    if int(arm_mask.sum()) < 4:
        raise RuntimeError("isolated donut arm mask is too small")

    arm_surface = mask_to_surface(arm_mask, spacing)
    arm_pts_mm = tree.pts_vox[[int(x) for x in arm_nodes]].astype(float) * spacing
    centerline_poly, seed_start_mm, seed_end_mm, seed_trim_mm = run_vmtk_centerline_for_path_with_tip_retries(
        arm_surface,
        arm_pts_mm,
        spacing,
        context=f"isolated donut loop {loop_i} arm {arm_i}",
        reference_centerline_points=reference_centerline_points,
        require_both_reference_endpoints_connected=True,
    )
    pts = numpy_support.vtk_to_numpy(centerline_poly.GetPoints().GetData())

    if len(pts) >= 2:
        pts = pts.copy()
        if np.linalg.norm(pts[0] - seed_start_mm) > np.linalg.norm(pts[-1] - seed_start_mm):
            pts = pts[::-1]
        if TRIM_DONUT_ARM_VMTK_TO_SKELETON_ENDPOINTS and len(pts) >= 4:
            i_start = int(np.argmin(np.linalg.norm(pts - seed_start_mm, axis=1)))
            i_end = int(np.argmin(np.linalg.norm(pts - seed_end_mm, axis=1)))
            if i_start > i_end:
                pts = pts[::-1]
                i_start = len(pts) - 1 - i_start
                i_end = len(pts) - 1 - i_end
            if i_end > i_start:
                pts = pts[i_start:i_end + 1].copy()
        if TRIM_DONUT_ARM_ENDS_BY_RADIUS and len(pts) >= 3:
            dist_mm = ndi.distance_transform_edt(arm_mask.astype(bool), sampling=spacing)
            center_vox = np.clip(np.round(pts / spacing).astype(int), 0, np.array(arm_mask.shape) - 1)
            radii = dist_mm[tuple(center_vox.T)]
            valid_radii = radii[np.isfinite(radii) & (radii > 0)]
            trim_mm = float(np.nanmean(valid_radii) * DONUT_ARM_END_TRIM_RADIUS_FACTOR) if len(valid_radii) else 0.0
            if trim_mm > 0:
                pts = trim_polyline_ends_by_length(pts, trim_mm)
                print(f"    [donut] Trimmed isolated arm VMTK ends by {trim_mm:.2f} mm (mean radius).")
        if PRESERVE_DONUT_ARM_ENDPOINTS and USE_SKELETON_CONNECTORS_AFTER_TRIM and len(pts) >= 2:
            pts = splice_skeleton_end_connectors(pts, arm_pts_mm, float(DONUT_ARM_ENDPOINT_BLEND_MM))
        pts, main_overlap_trim_start_n, main_overlap_trim_end_n, _, main_overlap_tol = trim_polyline_overlap_with_reference_ends(
            pts,
            reference_centerline_points,
            spacing,
            min_points_after_trim=FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE,
            keep_original_if_too_short=False,
        )
        if main_overlap_trim_start_n or main_overlap_trim_end_n:
            print(
                f"    [donut] Removed main-vessel overlap from isolated arm: "
                f"{main_overlap_trim_start_n} start point(s), {main_overlap_trim_end_n} end point(s) "
                f"(tol {main_overlap_tol:.2f} mm)."
            )
        if len(pts) < int(FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE):
            raise RuntimeError(
                f"isolated donut arm left only {len(pts)} point(s) after overlap trimming "
                f"(minimum {FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE})"
            )
    else:
        main_overlap_trim_start_n = 0
        main_overlap_trim_end_n = 0
        main_overlap_tol = np.nan
    original_n = len(pts)
    pts = resample_generated_centerline_points(pts)
    if len(pts) != original_n:
        print(
            f"    [donut] Resampled isolated arm {original_n} -> {len(pts)} point(s) "
            f"at {float(CENTERLINE_RESAMPLE_STEP_MM):.3f} mm arc-length spacing."
        )
    length_mm = arc_length(pts)

    kappa = discrete_curvature(pts)
    torsion = discrete_torsion(pts)
    radius_area, cross_section_area = compute_cross_section_radius(arm_surface, pts)
    if not (np.isfinite(radius_area) & (radius_area > 1e-6)).any():
        path_vox_cl = [tuple(np.clip(np.round(p / spacing).astype(int), 0, np.array(arm_mask.shape) - 1)) for p in pts]
        radius_area = radius_from_edt(arm_mask.astype(bool), spacing, path_vox_cl)
        cross_section_area = np.pi * radius_area ** 2
    radius_mis = np.full(len(pts), np.nan, dtype=float)
    detection_radius = select_caliber_detection_radius(radius_area, radius_mis)
    s = cumulative_s(pts)
    sten = detect_stenosis_segments(
        s=s, r=detection_radius, threshold_pct=STENOSIS_THRESHOLD_PCT,
        min_segment_mm=STENOSIS_MIN_LEN_MM, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        pts=pts,
    )
    stenosis_pct_point, is_stenotic = stenosis_pointwise(
        s=s, r=detection_radius, r_ref_per_point=sten.r_ref_per_point,
        threshold_pct=STENOSIS_THRESHOLD_PCT, exclude_end_mm=STENOSIS_EXCLUDE_END_MM,
        min_segment_mm=STENOSIS_MIN_LEN_MM,
    )
    stenosis_raw_pct_point = stenosis_raw_percent(
        s=s, r=detection_radius, r_ref_per_point=sten.r_ref_per_point,
        exclude_end_mm=0.0,
    )
    stenosis_core_candidate = (
        (stenosis_raw_pct_point >= STENOSIS_THRESHOLD_PCT) & np.isfinite(stenosis_raw_pct_point)
    ).astype(np.float64)
    stenosis_support_candidate = (
        (stenosis_raw_pct_point >= min(STENOSIS_SUPPORT_THRESHOLD_PCT, STENOSIS_THRESHOLD_PCT))
        & np.isfinite(stenosis_raw_pct_point)
        & (s >= float(STENOSIS_EXCLUDE_END_MM))
    ).astype(np.float64)
    enlarg = detect_enlargement_segments(
        s=s, r=detection_radius, threshold_pct=ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        pts=pts,
    )
    enlargement_pct_point, is_enlarged = enlargement_pointwise(
        s=s, r=detection_radius, r_ref_per_point=enlarg.r_ref_per_point,
        threshold_pct=ENLARGEMENT_THRESHOLD_PCT, exclude_end_mm=ENLARGEMENT_EXCLUDE_END_MM,
        min_segment_mm=ENLARGEMENT_MIN_LEN_MM,
        pts=pts,
    )
    stenosis_pct_point, is_stenotic, enlargement_pct_point, is_enlarged = resolve_stenosis_enlargement_overlap(
        stenosis_pct_point, is_stenotic, enlargement_pct_point, is_enlarged
    )
    tree_label = f"loop{int(loop_i):02d}_arm{int(arm_i):02d}"
    tree_path = f"loop{int(loop_i):02d}.{int(arm_i):02d}"
    tree_depth = len([p for p in tree_path.split(".") if p])
    poly = build_polyline_polydata(points=pts, arrays=[
        (kappa, "Curvature"),
        (torsion, "Torsion"),
        (radius_area, "EffectiveRadius"),
        (radius_area, "CrossSectionRadius"),
        (radius_mis, "MaximumInscribedSphereRadius"),
        (detection_radius, "StenosisDetectionRadius"),
        (cross_section_area, "CrossSectionArea"),
        (sten.r_ref_per_point, "StenosisReferenceRadius"),
        ((1.0 - STENOSIS_THRESHOLD_PCT / 100.0) * sten.r_ref_per_point, "StenosisThresholdRadius"),
        (stenosis_raw_pct_point, "StenosisRawPercent"),
        (stenosis_core_candidate, "StenosisCoreCandidate"),
        (stenosis_support_candidate, "StenosisSupportCandidate"),
        (stenosis_pct_point, "StenosisPercent"),
        (is_stenotic.astype(np.float64), "StenosisBinary"),
        (enlarg.r_ref_per_point, "EnlargementReferenceRadius"),
        ((1.0 + ENLARGEMENT_THRESHOLD_PCT / 100.0) * enlarg.r_ref_per_point, "EnlargementThresholdRadius"),
        (enlargement_pct_point, "EnlargementPercent"),
        (is_enlarged.astype(np.float64), "EnlargementBinary"),
        (np.full(len(pts), loop_i, dtype=float), "DonutLoopIndex"),
        (np.full(len(pts), arm_i, dtype=float), "DonutArmIndex"),
        (np.full(len(pts), tree_depth, dtype=float), "TreeDepth"),
        (np.ones(len(pts), dtype=float), "IsIsolatedArmVMTK"),
        (np.full(len(pts), main_overlap_trim_start_n, dtype=float), "MainOverlapTrimStartPoints"),
        (np.full(len(pts), main_overlap_trim_end_n, dtype=float), "MainOverlapTrimEndPoints"),
        (np.full(len(pts), seed_trim_mm, dtype=float), "SeedTrimRetryMm"),
        (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
    ])
    add_string_point_array(poly, [region_name] * len(pts), "DonutArmLabel")
    add_string_point_array(poly, [tree_label] * len(pts), "TreeLabel")
    add_string_point_array(poly, [tree_path] * len(pts), "TreePath")
    save_vtp(poly, path)
    metrics = {
        "stenosis_percent_max": float(sten.percent_stenosis_max),
        "degree_of_stenosis_pct": float(sten.percent_stenosis_max),
        "stenosis_length_total_mm": float(stenosis_total_length(s, is_stenotic)),
        "stenosis_segments_n": int(len(sten.segments_point_idx)),
        "stenosis_segments_point_idx": json.dumps(sten.segments_point_idx),
        "stenosis_segments_detail_json": segment_detail_json(s, stenosis_pct_point, sten.segments_point_idx),
        "enlargement_percent_max": float(enlarg.percent_enlargement_max),
        "enlargement_length_total_mm": float(stenosis_total_length(s, is_enlarged)),
        "enlargement_segments_n": int(len(enlarg.segments_point_idx)),
        "enlargement_segments_point_idx": json.dumps(enlarg.segments_point_idx),
        "enlargement_segments_detail_json": segment_detail_json(s, enlargement_pct_point, enlarg.segments_point_idx),
    }
    return float(length_mm), int(len(pts)), int(arm_mask.sum()), pts, metrics


def process_donut_loop_component_vmtk(
    label: int,
    component_id: int,
    mask_cc: np.ndarray,
    multilabel: np.ndarray,
    spacing,
    vessel_info: VesselInfo,
    mapping: dict,
    tree: SkeletonTree,
    surface,
    loops: List[Tuple[int, int, List[int], List[int]]],
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
    region_centerline_dir: Optional[str],
    surface_dir: Optional[str],
    ctx: Optional[MorphoContext] = None,
) -> Tuple[
    List[dict], Dict[str, pd.DataFrame], dict, pd.DataFrame, List[dict],
    Dict[str, pd.DataFrame], List[dict], pd.DataFrame, List[dict],
]:
    """Full donut-loop-aware processing of one component: root selection, per-terminal centerlines,
    and per-loop selected/alternate-arm isolation, metrics, and export.

    Component skeletons whose topology includes cycles (from :func:`find_loop_branches`)
    need special handling because a plain root-to-terminal BFS path can only
    traverse one side of each loop; this function additionally runs isolated
    VMTK extraction on the losing ("alternate") arm of every loop so both sides
    get their own centerline and metrics. Same return shape as
    :func:`orchestration.process_component_tree_vmtk`.
    """
    spacing = np.asarray(spacing, dtype=float)
    ctx = ctx or default_morpho_context()
    axes = ctx.axes
    path_results = []
    point_sheets = {}
    loop_rows = []
    loop_region_summaries = []
    loop_region_sheets = {}
    saved_unselected_sidecars = set()
    loop_sidecar_reference_points = []

    if tree.endpoints and tree.root is None:
        tree.root = choose_root_endpoint(
            tree, vessel_info, multilabel, spacing, mapping,
            mask_bool=mask_cc.astype(bool), axes=axes,
        )
        tree.dist_from_root_mm = dijkstra_dist_from_root(tree, tree.root, spacing)

    if tree.root is None:
        raise RuntimeError("Donut loop mode requires a root endpoint to generate whole-vessel centerlines.")

    terminal_indices = [int(e) for e in tree.endpoints if int(e) != int(tree.root)]
    if not terminal_indices:
        terminal_indices = [int(max((node for loop in loops for node in (loop[1],)), key=lambda n: tree.dist_from_root_mm[n]))]
    terminal_path_records = ordered_terminal_path_records(tree, int(tree.root), terminal_indices, spacing)
    missing_terminals = sorted(set(int(t) for t in terminal_indices) - set(row["terminal_id"] for row in terminal_path_records))
    for terminal_id in missing_terminals:
        print(f"    [donut] Warning: no full-vessel skeleton path root->{terminal_id}; skipping terminal.")
    if terminal_path_records:
        order_text = ", ".join(f"{row['terminal_id']}:{row['length_mm']:.1f}mm" for row in terminal_path_records)
        print(f"    [donut] Terminal path order (longest first): {order_text}")

    step_mm = SKELETON_STEP_MM if SKELETON_STEP_MM is not None else float(np.min(spacing)) * 0.5
    wall_tol_mm = SKELETON_WALL_TOL_MM if SKELETON_WALL_TOL_MM is not None else max(0.25, float(np.min(spacing)) * 0.75)

    for path_i, path_record in enumerate(terminal_path_records, start=1):
        terminal_id = int(path_record["terminal_id"])
        pidx = path_record["path_indices"]
        path_mm = tree.pts_vox[pidx].astype(float) * spacing
        path_mm_ext = extend_path_to_walls(
            path_mm=path_mm,
            mask_bool=mask_cc.astype(bool),
            spacing=spacing,
            tangent_pts=SKELETON_TANGENT_NPTS,
            step_mm=step_mm,
            wall_tol_mm=wall_tol_mm,
            max_steps=SKELETON_MAX_STEPS,
        )
        seed_start_mm = path_mm_ext[0]
        seed_end_mm = path_mm_ext[-1]
        tree_label, tree_path = tree_path_label_for_terminal(tree, int(tree.root), int(terminal_id), spacing)
        path_id = make_path_id(label, vessel_info.name, component_id, path_i, int(terminal_id), tree_label=tree_label)
        defer_donut_full_path_vtp_export = bool(EXPORT_ANATOMIC_SPLIT_CENTERLINES)
        cl_path = os.path.join(centerline_dir, path_id + ".vtp") if centerline_dir and not defer_donut_full_path_vtp_export else None
        cl_radius_path = os.path.join(centerline_radius_dir, path_id + "_radius.vtp") if centerline_radius_dir and not defer_donut_full_path_vtp_export else None

        try:
            centerline_poly, actual_seed_start_mm, actual_seed_end_mm, seed_trim_mm = run_vmtk_centerline_for_path_with_tip_retries(
                surface,
                path_mm_ext,
                spacing,
                context=f"donut full vessel path {path_i}",
                reference_centerline_points=reference_centerline_points_from_results(path_results),
            )
        except Exception as e:
            print(
                f"    [vmtk] donut full vessel path {path_i}: "
                f"all VMTK seed attempts failed ({e}); skipping this path."
            )
            continue
        res = analyze_centerline_poly(
            centerline_poly=centerline_poly,
            surface=surface,
            mask_cc=mask_cc,
            spacing=spacing,
            vessel_info=vessel_info,
            multilabel=multilabel,
            mapping=mapping,
            force_start_to_end=True,
            preferred_start_mm=actual_seed_start_mm,
            save_centerline_vtp=cl_path,
            save_centerline_radius_vtp=cl_radius_path,
            axes=axes,
        )
        res.update({
            "label": int(label),
            "component_id": int(component_id),
            "path_id": path_id,
            "tree_label": tree_label,
            "tree_path": tree_path,
            "path_role": "root_to_terminal_branch",
            "path_index": int(path_i),
            "root_skeleton_index": int(tree.root),
            "terminal_skeleton_index": int(terminal_id),
            "tree_mode": True,
            "donut_loop_mode": True,
            "donut_loop_index": np.nan,
            "donut_arm_index": np.nan,
            "donut_gateway_a": np.nan,
            "donut_gateway_b": np.nan,
            "vmtk_seed_trim_retry_mm": float(seed_trim_mm),
            "seed_start_x_mm": float(actual_seed_start_mm[0]),
            "seed_start_y_mm": float(actual_seed_start_mm[1]),
            "seed_start_z_mm": float(actual_seed_start_mm[2]),
            "seed_end_x_mm": float(actual_seed_end_mm[0]),
            "seed_end_y_mm": float(actual_seed_end_mm[1]),
            "seed_end_z_mm": float(actual_seed_end_mm[2]),
        })

        cl_pts = centerline_points_from_result(res)
        donut_loop_indices = np.full(len(cl_pts), -1.0, dtype=float)
        donut_arm_indices = np.full(len(cl_pts), -1.0, dtype=float)
        donut_labels = np.array(["whole_vessel"] * len(cl_pts), dtype=object)

        for loop_i, (gateway_a, gateway_b, arm1_nodes, arm2_nodes) in enumerate(loops, start=1):
            arm1_pts_mm = tree.pts_vox[[int(x) for x in arm1_nodes]].astype(float) * spacing
            arm2_pts_mm = tree.pts_vox[[int(x) for x in arm2_nodes]].astype(float) * spacing
            d_arm1 = mean_distance_to_polyline_points(arm1_pts_mm, cl_pts)
            d_arm2 = mean_distance_to_polyline_points(arm2_pts_mm, cl_pts)
            selected_arm_i = 1 if d_arm1 <= d_arm2 else 2
            unselected_arm_i = 2 if selected_arm_i == 1 else 1
            selected_pts_mm = arm1_pts_mm if selected_arm_i == 1 else arm2_pts_mm
            selected_nodes = arm1_nodes if selected_arm_i == 1 else arm2_nodes
            unselected_pts_mm = arm2_pts_mm if selected_arm_i == 1 else arm1_pts_mm
            unselected_nodes = arm2_nodes if selected_arm_i == 1 else arm1_nodes
            dist_to_selected, _ = cKDTree(selected_pts_mm).query(cl_pts, k=1)
            near_selected = dist_to_selected <= max(0.75, 2.0 * float(np.min(spacing)))
            donut_loop_indices[near_selected] = float(loop_i)
            donut_arm_indices[near_selected] = float(selected_arm_i)
            donut_labels[near_selected] = f"loop{loop_i:02d}_selected_arm"

            gateway_a_mm = tree.pts_vox[gateway_a].astype(float) * spacing
            gateway_b_mm = tree.pts_vox[gateway_b].astype(float) * spacing
            loop_rows.append({
                "label": int(label),
                "component_id": int(component_id),
                "vessel_name": vessel_info.name,
                "loop_index": int(loop_i),
                "arm_index": int(selected_arm_i),
                "region_name": "selected_arm",
                "path_id": path_id,
                "gateway_a": int(gateway_a),
                "gateway_b": int(gateway_b),
                "gateway_a_x_mm": float(gateway_a_mm[0]),
                "gateway_a_y_mm": float(gateway_a_mm[1]),
                "gateway_a_z_mm": float(gateway_a_mm[2]),
                "gateway_b_x_mm": float(gateway_b_mm[0]),
                "gateway_b_y_mm": float(gateway_b_mm[1]),
                "gateway_b_z_mm": float(gateway_b_mm[2]),
                "n_skeleton_points": int(len(arm1_nodes if selected_arm_i == 1 else arm2_nodes)),
                "skeleton_arm_length_mm": float(arc_length(selected_pts_mm)),
                "centerline_length_mm": float(res["length_mm"]),
                "selected_by_vmtk": True,
                "nearest_distance_to_arm1_mm": float(d_arm1),
                "nearest_distance_to_arm2_mm": float(d_arm2),
                "node_indices": [int(x) for x in (arm1_nodes if selected_arm_i == 1 else arm2_nodes)],
            })

            if SAVE_UNSELECTED_DONUT_ARM_SKELETON and centerline_dir:
                sidecar_id = make_loop_region_path_id(label, vessel_info.name, component_id, loop_i, "alternate_arm")
                sidecar_saved = False
                if sidecar_id in saved_unselected_sidecars:
                    sidecar_len = np.nan
                    sidecar_n = 0
                    sidecar_mask_voxels = np.nan
                    sidecar_metrics = {}
                    sidecar_method = "already_saved"
                    sidecar_saved = True
                else:
                    sidecar_path = os.path.join(centerline_dir, sidecar_id + ".vtp")
                    sidecar_method = "isolated_arm_vmtk"
                    sidecar_mask_voxels = np.nan
                    sidecar_len = np.nan
                    sidecar_n = 0
                    sidecar_metrics = {}
                    if not RUN_VMTK_ON_ISOLATED_DONUT_ARM:
                        sidecar_method = "isolated_arm_vmtk_disabled"
                    else:
                        try:
                            sidecar_len, sidecar_n, sidecar_mask_voxels, sidecar_pts, sidecar_metrics = run_isolated_donut_arm_vmtk(
                                mask_cc=mask_cc,
                                tree=tree,
                                arm_nodes=unselected_nodes,
                                competing_arm_nodes=selected_nodes,
                                spacing=spacing,
                                path=sidecar_path,
                                loop_i=loop_i,
                                arm_i=unselected_arm_i,
                                region_name=f"loop{loop_i:02d}_alternate_arm_isolated_vmtk",
                                reference_centerline_points=[cl_pts] + loop_sidecar_reference_points,
                            )
                            sidecar_saved = True
                            saved_unselected_sidecars.add(sidecar_id)
                            loop_sidecar_reference_points.append(sidecar_pts)
                        except Exception as e:
                            sidecar_method = "isolated_arm_vmtk_failed"
                            print(
                                f"    [donut] Isolated-arm VMTK failed for loop {loop_i} "
                                f"arm{unselected_arm_i}: {e}; no skeleton fallback will be saved."
                            )
                loop_rows.append({
                    "label": int(label),
                    "component_id": int(component_id),
                    "vessel_name": vessel_info.name,
                    "loop_index": int(loop_i),
                    "arm_index": int(unselected_arm_i),
                    "region_name": "alternate_arm",
                    "path_id": sidecar_id,
                    "gateway_a": int(gateway_a),
                    "gateway_b": int(gateway_b),
                    "gateway_a_x_mm": float(gateway_a_mm[0]),
                    "gateway_a_y_mm": float(gateway_a_mm[1]),
                    "gateway_a_z_mm": float(gateway_a_mm[2]),
                    "gateway_b_x_mm": float(gateway_b_mm[0]),
                    "gateway_b_y_mm": float(gateway_b_mm[1]),
                    "gateway_b_z_mm": float(gateway_b_mm[2]),
                    "n_skeleton_points": int(len(unselected_nodes)),
                    "skeleton_arm_length_mm": float(arc_length(unselected_pts_mm)),
                    "centerline_length_mm": float(sidecar_len),
                    "smoothed_skeleton_points": int(sidecar_n),
                    "sidecar_method": sidecar_method,
                    "isolated_mask_voxels": sidecar_mask_voxels,
                    "sidecar_saved": bool(sidecar_saved),
                    "stenosis_percent_max": sidecar_metrics.get("stenosis_percent_max", np.nan),
                    "degree_of_stenosis_pct": sidecar_metrics.get("degree_of_stenosis_pct", np.nan),
                    "stenosis_length_total_mm": sidecar_metrics.get("stenosis_length_total_mm", np.nan),
                    "stenosis_segments_n": sidecar_metrics.get("stenosis_segments_n", 0),
                    "stenosis_segments_point_idx": sidecar_metrics.get("stenosis_segments_point_idx", "[]"),
                    "stenosis_segments_detail_json": sidecar_metrics.get("stenosis_segments_detail_json", "[]"),
                    "enlargement_percent_max": sidecar_metrics.get("enlargement_percent_max", np.nan),
                    "enlargement_length_total_mm": sidecar_metrics.get("enlargement_length_total_mm", np.nan),
                    "enlargement_segments_n": sidecar_metrics.get("enlargement_segments_n", 0),
                    "enlargement_segments_point_idx": sidecar_metrics.get("enlargement_segments_point_idx", "[]"),
                    "enlargement_segments_detail_json": sidecar_metrics.get("enlargement_segments_detail_json", "[]"),
                    "selected_by_vmtk": False,
                    "nearest_distance_to_arm1_mm": float(d_arm1),
                    "nearest_distance_to_arm2_mm": float(d_arm2),
                    "node_indices": [int(x) for x in unselected_nodes],
                })
                print(
                    f"    [donut] Full vessel {path_id}: VMTK followed loop {loop_i} arm{selected_arm_i} "
                    f"(mean skeleton-to-centerline dist arm1={d_arm1:.2f} mm, arm2={d_arm2:.2f} mm); "
                    f"saved unselected arm{unselected_arm_i} via {sidecar_method} "
                    f"{sidecar_len:.2f} mm ({sidecar_n} points)."
                )

        res["donut_loop_indices"] = donut_loop_indices.copy()
        res["donut_arm_indices"] = donut_arm_indices.copy()
        res["donut_arm_labels"] = donut_labels.copy()

        if cl_path:
            pts = centerline_points_from_result(res)
            stenosis_binary = np.asarray(res["is_stenotic"], dtype=np.float64)
            enlargement_binary = np.asarray(res.get("is_enlarged", np.zeros(len(pts))), dtype=np.float64)
            tree_labels = np.array([res.get("tree_label", "trunk")] * len(pts), dtype=object)
            tree_paths = np.array([res.get("tree_path", "")] * len(pts), dtype=object)
            tree_depths = np.full(len(pts), len([p for p in str(res.get("tree_path", "")).split(".") if p]), dtype=float)
            for region_name in np.unique(donut_labels):
                region_name = str(region_name)
                region_mask = donut_labels == region_name
                if "selected_arm" in region_name:
                    tree_labels[region_mask] = f"{res.get('tree_label', 'trunk')}_loop_selected"
            res["tree_point_labels"] = tree_labels.copy()
            res["tree_point_paths"] = tree_paths.copy()
            res["tree_point_depths"] = tree_depths.copy()
            labeled_poly = build_polyline_polydata(points=pts, arrays=[
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
                (res.get("enlargement_percent_point", np.full(len(pts), np.nan)), "EnlargementPercent"),
                (enlargement_binary, "EnlargementBinary"),
                (donut_loop_indices, "DonutLoopIndex"),
                (donut_arm_indices, "DonutArmIndex"),
                (tree_depths, "TreeDepth"),
                (np.arange(1, len(pts) + 1, dtype=np.float64), "PointIndex"),
            ])
            add_string_point_array(labeled_poly, donut_labels.tolist(), "DonutArmLabel")
            add_string_point_array(labeled_poly, tree_labels.tolist(), "TreeLabel")
            add_string_point_array(labeled_poly, tree_paths.tolist(), "TreePath")
            save_vtp(labeled_poly, cl_path)

        path_results.append(res)
        print(f"    [donut] Full vessel {path_id}: VMTK length={res['length_mm']:.2f} mm ({len(res['s_mm'])} points).")

    path_results, overlap_discarded_path_ids, n_overlap_trimmed = prune_overlapping_final_centerlines(
        path_results=path_results,
        spacing=spacing,
        centerline_dir=centerline_dir,
        centerline_radius_dir=centerline_radius_dir,
    )
    suppress_enlargements_near_centerline_starts(
        path_results=path_results,
        spacing=spacing,
        centerline_dir=centerline_dir,
        centerline_radius_dir=centerline_radius_dir,
    )
    point_sheets = {
        safe_sheet_name(res["path_id"]): build_vessel_points_dataframe("", label, vessel_info, res)
        for res in path_results
    }

    branch_df = compute_branchpoint_metrics(label, component_id, tree, mask_cc.astype(bool), spacing, vessel_info)
    loop_debug_df = pd.DataFrame([{k: v for k, v in row.items() if k != "node_indices"} | {
        "node_indices_json": json.dumps(row["node_indices"])
    } for row in loop_rows])

    if SAVE_DONUT_LOOP_DEBUG and region_centerline_dir and loop_rows:
        debug_poly = build_donut_loop_debug_polydata(loop_rows, tree, spacing)
        save_vtp(debug_poly, os.path.join(region_centerline_dir, safe_filename(f"{label}_{vessel_info.name}_comp{component_id:02d}_loop_debug_skeleton.vtp")))

    saved_anatomic = []
    split_results: List[dict] = []
    if EXPORT_ANATOMIC_SPLIT_CENTERLINES and centerline_dir:
        expected_anatomic = int(len(path_results))
        saved_anatomic, split_results = save_anatomic_fallback_centerlines(
            vessel_info=vessel_info,
            component_id=component_id,
            path_results=path_results,
            centerline_dir=centerline_dir,
            centerline_radius_dir=centerline_radius_dir,
            axes=axes,
            spacing=spacing,
        )
        if len(saved_anatomic) < expected_anatomic:
            print(
                f"    [anatomic sanity] Donut vessel expected at least {expected_anatomic} "
                f"main centerline VTP(s), but wrote {len(saved_anatomic)}."
            )

    tree_summary = {
        "label": int(label),
        "component_id": int(component_id),
        "vessel_name": vessel_info.name,
        "full_name": vessel_info.full_name,
        "side": vessel_info.side,
        "pair": vessel_info.pair or "",
        "territory": vessel_info.territory,
        "flow_from": vessel_info.flow_from,
        "tree_mode": True,
        "donut_loop_mode": True,
        "n_skeleton_voxels": int(len(tree.pts_vox)),
        "n_endpoints": int(len(tree.endpoints)),
        "n_terminals": int(len(tree.endpoints)),
        "n_branchpoints": int(len(tree.branchpoints)),
        "n_donut_loops": int(len(loops)),
        "n_donut_loop_regions": int(len(path_results)),
        "n_donut_loop_arms": int(sum(1 for row in loop_rows if str(row.get("region_name", "")) in {"selected_arm", "alternate_arm"})),
        "unique_skeleton_graph_length_mm": float(skeleton_total_graph_length_mm(tree, spacing)),
        "n_centerline_paths": int(len(path_results)),
        "n_centerline_paths_discarded_short": 0,
        "n_centerline_paths_discarded_spurious_arm": 0,
        "n_centerline_paths_discarded_overlap": int(len(overlap_discarded_path_ids)),
        "n_centerline_paths_overlap_trimmed": int(n_overlap_trimmed),
        "min_centerline_path_length_mm": np.nan,
        "centerline_paths_total_length_with_shared_trunks_mm": float(np.nansum([r["length_mm"] for r in path_results])) if path_results else np.nan,
        "root_skeleton_index": np.nan,
        "root_x_mm": np.nan,
        "root_y_mm": np.nan,
        "root_z_mm": np.nan,
        "n_anatomic_centerline_vtps_expected_min": int(len(path_results)),
        "n_anatomic_centerline_vtps_written": int(len(saved_anatomic)),
        "anatomic_centerline_vtps_written": ";".join(saved_anatomic),
    }
    tree_summary.update(anatomy_provenance_fields(ctx, vessel_info))
    tree_summary.update(component_volumetry_fields(
        mask_cc, spacing, surface,
        skeleton_length_mm=tree_summary["unique_skeleton_graph_length_mm"],
    ))

    if not split_results:
        # No anatomic export ran; deduplicate the measured paths so the shared
        # trunk of a loop component is still only counted once.
        split_results, _dropped = deduplicate_path_results(path_results, spacing)
        split_results = [dict(res, non_overlapping=True) for res in split_results]
    tree_summary["n_non_overlapping_segments"] = int(len(split_results))
    tree_summary["centerline_total_length_mm"] = float(
        np.nansum([r.get("length_mm", np.nan) for r in split_results])
    ) if split_results else np.nan

    return (
        path_results, point_sheets, tree_summary, branch_df, loop_region_summaries,
        loop_region_sheets, [], loop_debug_df, split_results,
    )

