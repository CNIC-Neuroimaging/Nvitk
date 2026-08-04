"""Orchestration: process one component and one full case."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nvitk.measure.morpho.anatomy import choose_root_endpoint, make_path_id
from nvitk.measure.morpho.centerlines import (
    analyze_centerline_poly,
    centerline_poly_length_mm,
    prune_overlapping_final_centerlines,
    reference_centerline_points_from_results,
    run_vmtk_centerline_for_path_with_tip_retries,
    suppress_enlargements_near_centerline_starts,
    VMTK_AVAILABLE,
)
from nvitk.measure.morphometrics_config import (
    CENTERLINE_OVERLAP_TOL_MM,
    CENTERLINE_RESAMPLE_STEP_MM,
    DISCARD_SHORT_CENTERLINE_PATHS,
    DISCARD_SHORT_TREE_ARMS,
    ENABLE_DONUT_LOOP_MODE,
    ENABLE_RECURSIVE_TREE_SEGMENTS,
    ENABLE_TREE_MODE,
    EXPORT_ANATOMIC_SPLIT_CENTERLINES,
    MIN_CENTERLINE_PATH_LENGTH_MM,
    MIN_TREE_ARM_LENGTH_MM,
    MIN_TREE_ENDPOINTS_FOR_TREE_MODE,
    PROCESS_ALL_CONNECTED_COMPONENTS,
    PROCESS_SELECTED_TAGS_ONLY,
    REFINE_SURFACE_FOR_VMTK,
    RESAMPLE_CENTERLINES_BY_ARCLENGTH,
    RETRY_VMTK_WITH_TRIMMED_SEEDS,
    SAVE_CENTERLINE_RADIUS,
    SAVE_CENTERLINES,
    SAVE_PRE_REFINED_SURFACES,
    SAVE_SURFACES,
    SELECTED_TAGS,
    SKELETON_MAX_STEPS,
    SKELETON_STEP_MM,
    SKELETON_TANGENT_NPTS,
    SKELETON_WALL_TOL_MM,
    SPLIT_BIFURCATING_TREE_REGIONS,
    SURFACE_SUBDIVISION_LEVELS,
    TREE_SEGMENT_ASSIGN_TOL_MM,
    USE_VMTK_FOR_DONUT_LOOP_PATH,
    VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM,
    VMTK_SEED_TRIM_RETRY_MM,
)
from nvitk.measure.morpho.donut_loops import process_donut_loop_component_vmtk
from nvitk.measure.morpho.geometry import extend_path_to_walls
from nvitk.measure.morpho.io_utils import (
    safe_filename,
    safe_sheet_name,
)
from nvitk.measure.morpho.labels_util import (
    empty_vessel_info,
    is_acoa_vessel,
)
from nvitk.measure.morpho.metrics import compute_branchpoint_metrics
from nvitk.measure.morpho.models import VesselInfo
from nvitk.measure.morpho.skeleton import (
    dijkstra_dist_from_root,
    find_loop_branches,
    skeleton_endpoints_and_path,
    skeleton_total_graph_length_mm,
    skeleton_tree_from_mask,
    skeletonize_mask,
)
from nvitk.measure.morpho.export_utils.summaries import (
    build_vessel_points_dataframe,
)
from nvitk.measure.morpho.surface import mask_to_surface, save_vtp
from nvitk.measure.morpho.tree_regions import (
    discarded_source_path_ids_from_regions,
    ordered_terminal_path_records,
    remove_discarded_tree_path_outputs,
    save_labeled_tree_path_centerlines,
    split_bifurcating_tree_centerlines,
    tree_path_label_for_terminal,
)
from nvitk.measure.morpho.tree_segments import (
    annotate_anatomic_tree_segments,
    build_connected_skeleton_edge_segments,
    build_recursive_tree_segments,
    save_anatomic_fallback_centerlines,
    save_anatomic_split_tree_centerlines,
    save_recursive_labeled_tree_path_centerlines,
)

def process_component_tree_vmtk(
    label: int,
    component_id: int,
    mask_cc: np.ndarray,
    multilabel: np.ndarray,
    spacing,
    vessel_info: VesselInfo,
    mapping: dict,
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
    region_centerline_dir: Optional[str],
    surface_dir: Optional[str],
) -> Tuple[List[dict], Dict[str, pd.DataFrame], dict, pd.DataFrame, List[dict], Dict[str, pd.DataFrame], List[dict], pd.DataFrame]:
    """Full per-component morphometrics pipeline: skeletonize → VMTK centerlines → metrics/export.

    Drives, for one connected component of one label: skeleton extraction and
    root selection, surface reconstruction, VMTK centerline generation between
    root and each terminal (with donut-loop handling and retry logic), stenosis/
    enlargement detection, recursive tree-segment splitting, and every VTP/point
    export. This is the single entry point :mod:`run_case` dispatches to workers.

    Returns
    -------
    tuple
        ``(path_results, point_sheets, tree_summary, branch_df, region_summaries,
        region_sheets, recursive_segments, donut_loop_df)`` — see the call sites in
        :mod:`run_case` for how each piece feeds the exported workbook.
    """
    if not VMTK_AVAILABLE:
        raise RuntimeError("Morphometrics centerline backend unavailable.")
    spacing = np.asarray(spacing, dtype=float)

    skel = skeletonize_mask(mask_cc.astype(bool))
    tree = skeleton_tree_from_mask(mask_cc.astype(bool), spacing=spacing)
    if len(tree.pts_vox) < 2:
        raise RuntimeError("Skeleton too short.")

    print(f"    Skeleton: {len(tree.pts_vox)} voxels, {len(tree.endpoints)} endpoints, {len(tree.branchpoints)} branchpoints")

    base_tree_name = safe_filename(f"{label}_{vessel_info.name}_comp{component_id:02d}")
    pre_refined_surface_path = (
        os.path.join(surface_dir, base_tree_name + "_pre_refined.vtp")
        if surface_dir and SAVE_PRE_REFINED_SURFACES
        else None
    )
    surface = mask_to_surface(mask_cc, spacing, pre_refined_surface_path=pre_refined_surface_path)
    surface_path = os.path.join(surface_dir, base_tree_name + ".vtp") if surface_dir else None
    if surface_path:
        save_vtp(surface, surface_path)

    if ENABLE_DONUT_LOOP_MODE:
        cycle_core, gateways, loops = find_loop_branches(tree.pts_vox, tree.neighbors, tree.degree)
        if loops:
            print(
                f"    [donut] Detected {len(loops)} loop(s): "
                f"cycle_core={len(cycle_core)} nodes, gateways={sorted(gateways)}"
            )
            return process_donut_loop_component_vmtk(
                label=label,
                component_id=component_id,
                mask_cc=mask_cc,
                multilabel=multilabel,
                spacing=spacing,
                vessel_info=vessel_info,
                mapping=mapping,
                tree=tree,
                surface=surface,
                loops=loops,
                centerline_dir=centerline_dir,
                centerline_radius_dir=centerline_radius_dir,
                region_centerline_dir=region_centerline_dir,
                surface_dir=surface_dir,
            )

    use_tree_mode = ENABLE_TREE_MODE and len(tree.endpoints) >= MIN_TREE_ENDPOINTS_FOR_TREE_MODE

    # If unbranched, preserve old behavior: longest path between two endpoints.
    if not use_tree_mode:
        ep_a_vox, ep_b_vox, path_vox = skeleton_endpoints_and_path(skel)
        if ep_a_vox is None:
            raise RuntimeError("Skeleton too short.")
        terminals = [0]
        root_idx = None
        path_indices = None
        path_vox_list = [path_vox]
        print(f"    Mode: single path. Endpoints={np.round(np.asarray(ep_a_vox)*spacing,2)} -> {np.round(np.asarray(ep_b_vox)*spacing,2)}")
    else:
        root_idx = choose_root_endpoint(tree, vessel_info, multilabel, spacing, mapping, mask_bool=mask_cc.astype(bool))
        tree.root = root_idx
        tree.dist_from_root_mm = dijkstra_dist_from_root(tree, root_idx, spacing)
        terminal_indices = [e for e in tree.endpoints if e != root_idx]
        terminal_path_records = ordered_terminal_path_records(tree, int(root_idx), [int(x) for x in terminal_indices], spacing)
        missing_terminals = sorted(set(int(t) for t in terminal_indices) - set(row["terminal_id"] for row in terminal_path_records))
        for term in missing_terminals:
            print(f"    Warning: no skeleton path root->{term}; skipping terminal.")
        terminals = [row["terminal_id"] for row in terminal_path_records]
        path_vox_list = [row["path_vox"] for row in terminal_path_records]
        order_text = ", ".join(f"{row['terminal_id']}:{row['length_mm']:.1f}mm" for row in terminal_path_records)
        print(f"    Mode: tree. Root endpoint={root_idx}; terminal paths={len(path_vox_list)}")
        if order_text:
            print(f"    Terminal path order (longest first): {order_text}")

    step_mm = SKELETON_STEP_MM if SKELETON_STEP_MM is not None else float(np.min(spacing)) * 0.5
    wall_tol_mm = SKELETON_WALL_TOL_MM if SKELETON_WALL_TOL_MM is not None else max(0.25, float(np.min(spacing)) * 0.75)

    path_results = []
    point_sheets = {}
    n_short_centerline_paths_discarded = 0
    # Prefer unique morphology skeleton edges (connected at junctions, no
    # overlapping trunks, all endpoints/roots covered). Fall back to the older
    # root-directed recursive walk only if edge extraction yields nothing.
    recursive_tree_segments = []
    if use_tree_mode and ENABLE_RECURSIVE_TREE_SEGMENTS:
        recursive_tree_segments = build_connected_skeleton_edge_segments(tree, spacing)
        if not recursive_tree_segments:
            recursive_tree_segments = build_recursive_tree_segments(tree, spacing)
        if recursive_tree_segments:
            recursive_tree_segments = annotate_anatomic_tree_segments(
                tree, recursive_tree_segments
            )
            print(
                f"    [tree segments] Connected skeleton-edge segments: "
                f"{len(recursive_tree_segments)}"
            )

    for path_i, path_vox in enumerate(path_vox_list, start=1):
        if len(path_vox) < 2:
            continue
        path_mm = np.asarray(path_vox, dtype=float) * spacing
        path_mm_ext = extend_path_to_walls(
            path_mm=path_mm, mask_bool=mask_cc.astype(bool), spacing=spacing,
            tangent_pts=SKELETON_TANGENT_NPTS, step_mm=step_mm,
            wall_tol_mm=wall_tol_mm, max_steps=SKELETON_MAX_STEPS,
        )
        seed_start_mm = path_mm_ext[0]
        seed_end_mm = path_mm_ext[-1]
        print(f"    Path {path_i}: seed start={np.round(seed_start_mm,2)} end={np.round(seed_end_mm,2)}")

        terminal_id = terminals[path_i - 1] if path_i - 1 < len(terminals) else path_i
        if use_tree_mode and root_idx is not None:
            tree_label, tree_path = tree_path_label_for_terminal(tree, int(root_idx), int(terminal_id), spacing)
        else:
            tree_label, tree_path = "trunk", ""
        path_id = make_path_id(label, vessel_info.name, component_id, path_i, int(terminal_id), tree_label=tree_label)
        defer_tree_vtp_export = bool(use_tree_mode and EXPORT_ANATOMIC_SPLIT_CENTERLINES)
        cl_path = os.path.join(centerline_dir, path_id + ".vtp") if centerline_dir and not defer_tree_vtp_export else None
        cl_radius_path = os.path.join(centerline_radius_dir, path_id + "_radius.vtp") if centerline_radius_dir and not defer_tree_vtp_export else None

        try:
            centerline_poly, actual_seed_start_mm, actual_seed_end_mm, seed_trim_mm = run_vmtk_centerline_for_path_with_tip_retries(
                surface,
                path_mm_ext,
                spacing,
                context=f"tree path {path_i}",
                reference_centerline_points=reference_centerline_points_from_results(path_results),
            )
        except Exception as e:
            print(
                f"    [vmtk] tree path {path_i}: all VMTK seed attempts failed "
                f"({e}); skipping this path."
            )
            continue
        centerline_length_mm = centerline_poly_length_mm(centerline_poly)
        centerline_n_points = centerline_poly.GetNumberOfPoints() if centerline_poly is not None else 0
        print(f"    Path {path_i}: generated centerline length={centerline_length_mm:.2f} mm ({centerline_n_points} points).")
        short_centerline_pruning_enabled = DISCARD_SHORT_CENTERLINE_PATHS and not is_acoa_vessel(vessel_info)
        if (
            short_centerline_pruning_enabled
            and np.isfinite(centerline_length_mm)
            and centerline_length_mm < MIN_CENTERLINE_PATH_LENGTH_MM
        ):
            n_short_centerline_paths_discarded += 1
            print(
                f"    Path {path_i}: discarded short centerline "
                f"({centerline_length_mm:.2f} mm < {MIN_CENTERLINE_PATH_LENGTH_MM:.2f} mm)."
            )
            continue

        res = analyze_centerline_poly(
            centerline_poly=centerline_poly, surface=surface, mask_cc=mask_cc,
            spacing=spacing, vessel_info=vessel_info, multilabel=multilabel, mapping=mapping,
            force_start_to_end=use_tree_mode,  # tree paths are seeded root -> terminal; do not reorient by downstream labels.
            preferred_start_mm=actual_seed_start_mm if use_tree_mode else None,
            save_centerline_vtp=cl_path, save_centerline_radius_vtp=cl_radius_path,
        )
        res.update({
            "label": int(label), "component_id": int(component_id), "path_id": path_id,
            "tree_label": tree_label, "tree_path": tree_path,
            "path_role": "trunk" if int(terminal_id) == 0 else "root_to_terminal_branch",
            "path_index": int(path_i), "root_skeleton_index": int(root_idx) if root_idx is not None else np.nan,
            "terminal_skeleton_index": int(terminal_id) if use_tree_mode else np.nan,
            "seed_start_x_mm": float(actual_seed_start_mm[0]), "seed_start_y_mm": float(actual_seed_start_mm[1]), "seed_start_z_mm": float(actual_seed_start_mm[2]),
            "seed_end_x_mm": float(actual_seed_end_mm[0]), "seed_end_y_mm": float(actual_seed_end_mm[1]), "seed_end_z_mm": float(actual_seed_end_mm[2]),
            "original_seed_start_x_mm": float(seed_start_mm[0]), "original_seed_start_y_mm": float(seed_start_mm[1]), "original_seed_start_z_mm": float(seed_start_mm[2]),
            "original_seed_end_x_mm": float(seed_end_mm[0]), "original_seed_end_y_mm": float(seed_end_mm[1]), "original_seed_end_z_mm": float(seed_end_mm[2]),
            "vmtk_seed_trim_retry_mm": float(seed_trim_mm),
            "tree_mode": bool(use_tree_mode),
        })
        path_results.append(res)

    branch_df = compute_branchpoint_metrics(label, component_id, tree, mask_cc.astype(bool), spacing, vessel_info)

    tree_summary = {
        "label": int(label), "component_id": int(component_id), "vessel_name": vessel_info.name,
        "full_name": vessel_info.full_name, "side": vessel_info.side, "pair": vessel_info.pair or "",
        "territory": vessel_info.territory, "flow_from": vessel_info.flow_from,
        "tree_mode": bool(use_tree_mode),
        "n_skeleton_voxels": int(len(tree.pts_vox)),
        "n_endpoints": int(len(tree.endpoints)),
        "n_terminals": int(max(0, len(tree.endpoints) - 1)) if use_tree_mode else int(min(2, len(tree.endpoints))),
        "n_branchpoints": int(len(tree.branchpoints)),
        "unique_skeleton_graph_length_mm": float(skeleton_total_graph_length_mm(tree, spacing)),
        "n_centerline_paths": int(len(path_results)),
        "n_centerline_paths_discarded_short": int(n_short_centerline_paths_discarded),
        "min_centerline_path_length_mm": (
            float(MIN_CENTERLINE_PATH_LENGTH_MM)
            if DISCARD_SHORT_CENTERLINE_PATHS and not is_acoa_vessel(vessel_info)
            else np.nan
        ),
        "centerline_paths_total_length_with_shared_trunks_mm": float(np.nansum([r["length_mm"] for r in path_results])) if path_results else np.nan,
        "root_skeleton_index": int(root_idx) if root_idx is not None else np.nan,
        "root_x_mm": float(tree.pts_vox[root_idx][0] * spacing[0]) if root_idx is not None else np.nan,
        "root_y_mm": float(tree.pts_vox[root_idx][1] * spacing[1]) if root_idx is not None else np.nan,
        "root_z_mm": float(tree.pts_vox[root_idx][2] * spacing[2]) if root_idx is not None else np.nan,
    }

    tree_region_summaries, tree_region_sheets, tree_regions = split_bifurcating_tree_centerlines(
        label=label,
        component_id=component_id,
        vessel_info=vessel_info,
        path_results=path_results if use_tree_mode else [],
        spacing=spacing,
        region_centerline_dir=region_centerline_dir,
    )
    discarded_path_ids = discarded_source_path_ids_from_regions(tree_regions)
    remove_discarded_tree_path_outputs(discarded_path_ids, centerline_dir, centerline_radius_dir)
    if discarded_path_ids:
        path_results = [res for res in path_results if res.get("path_id", "") not in discarded_path_ids]
        for path_id in discarded_path_ids:
            point_sheets.pop(safe_sheet_name(path_id), None)
        tree_summary["n_centerline_paths"] = int(len(path_results))
        tree_summary["n_centerline_paths_discarded_spurious_arm"] = int(len(discarded_path_ids))
        tree_summary["centerline_paths_total_length_with_shared_trunks_mm"] = (
            float(np.nansum([r["length_mm"] for r in path_results])) if path_results else np.nan
        )
    else:
        tree_summary["n_centerline_paths_discarded_spurious_arm"] = 0
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
    if overlap_discarded_path_ids:
        for path_id in overlap_discarded_path_ids:
            point_sheets.pop(safe_sheet_name(path_id), None)
        tree_summary["n_centerline_paths"] = int(len(path_results))
        tree_summary["n_centerline_paths_discarded_overlap"] = int(len(overlap_discarded_path_ids))
        tree_summary["centerline_paths_total_length_with_shared_trunks_mm"] = (
            float(np.nansum([r["length_mm"] for r in path_results])) if path_results else np.nan
        )
    else:
        tree_summary["n_centerline_paths_discarded_overlap"] = 0
    tree_summary["n_centerline_paths_overlap_trimmed"] = int(n_overlap_trimmed)

    point_sheets = {}
    for res in path_results:
        sheet_name = safe_sheet_name(res["path_id"])
        point_sheets[sheet_name] = build_vessel_points_dataframe("", label, vessel_info, res)

    saved_anatomic: List[str] = []
    if recursive_tree_segments:
        if EXPORT_ANATOMIC_SPLIT_CENTERLINES:
            saved_anatomic = save_anatomic_split_tree_centerlines(
                label=label,
                component_id=component_id,
                vessel_info=vessel_info,
                path_results=path_results,
                segments=recursive_tree_segments,
                centerline_dir=centerline_dir if use_tree_mode else None,
                centerline_radius_dir=centerline_radius_dir,
                spacing=spacing,
                root_idx=int(root_idx) if root_idx is not None else None,
                mask_cc=mask_cc.astype(bool),
            )
        else:
            save_recursive_labeled_tree_path_centerlines(
                path_results,
                recursive_tree_segments,
                centerline_dir if use_tree_mode else None,
                spacing,
            )
    else:
        if EXPORT_ANATOMIC_SPLIT_CENTERLINES and use_tree_mode:
            saved_anatomic = save_anatomic_fallback_centerlines(
                vessel_info=vessel_info,
                component_id=component_id,
                path_results=path_results,
                centerline_dir=centerline_dir,
                centerline_radius_dir=centerline_radius_dir,
            )
        else:
            save_labeled_tree_path_centerlines(path_results, tree_regions, centerline_dir if use_tree_mode else None)

    if EXPORT_ANATOMIC_SPLIT_CENTERLINES and use_tree_mode:
        expected_anatomic = int(len(path_results))
        saved_anatomic = list(dict.fromkeys(saved_anatomic))
        # Only fall back to full root→terminal exports when anatomic split wrote
        # nothing — stacking fallbacks on partial splits reintroduces overlapping trunks.
        if not saved_anatomic:
            print(
                "    [anatomic sanity] Anatomical export wrote no centerline VTP(s). "
                "Writing anatomical fallback centerline(s) so no surviving vessel path is lost."
            )
            fallback_saved = save_anatomic_fallback_centerlines(
                vessel_info=vessel_info,
                component_id=component_id,
                path_results=path_results,
                centerline_dir=centerline_dir,
                centerline_radius_dir=centerline_radius_dir,
            )
            saved_anatomic = list(dict.fromkeys(fallback_saved))
        elif len(saved_anatomic) < expected_anatomic:
            print(
                f"    [anatomic sanity] Anatomical export wrote {len(saved_anatomic)} "
                f"segment VTP(s) (paths={expected_anatomic}); keeping split segments only "
                "to avoid overlapping root→terminal duplicates."
            )
        tree_summary["n_anatomic_centerline_vtps_expected_min"] = expected_anatomic
        tree_summary["n_anatomic_centerline_vtps_written"] = int(len(saved_anatomic))
        tree_summary["anatomic_centerline_vtps_written"] = ";".join(saved_anatomic)

    return path_results, point_sheets, tree_summary, branch_df, tree_region_summaries, tree_region_sheets, recursive_tree_segments, pd.DataFrame()

