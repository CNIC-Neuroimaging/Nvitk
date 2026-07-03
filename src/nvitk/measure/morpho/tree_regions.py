"""Bifurcating tree-region splitting, labeling, pruning, and exports."""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .anatomy import make_tree_region_id, tree_region_metadata
from .centerlines import centerline_points_from_result
from nvitk.measure.morphometrics_config import (
    CENTERLINE_OVERLAP_MAX_GAP_POINTS,
    CENTERLINE_OVERLAP_TOL_MM,
    DISCARD_SHORT_TREE_ARMS,
    MIN_COMMON_BASE_POINTS,
    MIN_TREE_ARM_LENGTH_MM,
    SPLIT_BIFURCATING_TREE_REGIONS,
    TREE_REGION_CODES,
    TREE_REGION_ROLE_CODES,
    VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM,
)
from .geometry import arc_length
from .io_utils import safe_filename, safe_sheet_name
from .models import SkeletonTree, VesselInfo
from .skeleton import bfs_path_indices, dijkstra_dist_from_root
from .export_utils.summaries import build_tree_region_points_dataframe, tree_region_summary_from_points
from .surface import add_string_point_array, build_polyline_polydata, save_vtp

def discarded_source_path_ids_from_regions(regions: List[dict]) -> set:
    discarded = set()
    for region in regions:
        discarded.update(str(path_id) for path_id in region.get("discarded_source_path_ids", []))
    return discarded


def remove_discarded_tree_path_outputs(discarded_path_ids: set, centerline_dir: Optional[str], centerline_radius_dir: Optional[str]) -> None:
    for path_id in sorted(discarded_path_ids):
        candidates = []
        if centerline_dir:
            candidates.append(os.path.join(centerline_dir, path_id + ".vtp"))
        if centerline_radius_dir:
            candidates.append(os.path.join(centerline_radius_dir, path_id + "_radius.vtp"))
        removed = []
        for path in candidates:
            if os.path.exists(path):
                os.remove(path)
                removed.append(os.path.basename(path))
        if removed:
            print(f"    [tree regions] Removed discarded spurious path outputs: {', '.join(removed)}")


def min_tree_arm_length_for_vessel(vessel_info: VesselInfo) -> float:
    name = str(getattr(vessel_info, "name", "") or "")
    pair = str(getattr(vessel_info, "pair", "") or "")
    candidates = [name, pair]
    for key in candidates:
        if key in VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM:
            return float(VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM[key])
    return float(MIN_TREE_ARM_LENGTH_MM)


def ordered_terminal_path_records(tree: SkeletonTree, root_idx: int, terminal_indices: List[int], spacing) -> List[dict]:
    spacing = np.asarray(spacing, dtype=float)
    records = []
    for term in terminal_indices:
        pidx = bfs_path_indices(tree.neighbors, int(root_idx), int(term))
        if not pidx:
            continue
        pts_mm = tree.pts_vox[pidx].astype(float) * spacing
        records.append({
            "terminal_id": int(term),
            "path_indices": [int(x) for x in pidx],
            "path_vox": [tuple(map(int, tree.pts_vox[i])) for i in pidx],
            "length_mm": float(arc_length(pts_mm)),
        })
    records.sort(key=lambda row: (-float(row["length_mm"]) if np.isfinite(row["length_mm"]) else np.inf, int(row["terminal_id"])))
    return records


def tree_label_from_path_parts(parts: List[int]) -> str:
    if not parts:
        return "trunk"
    label = f"arm{int(parts[0]):02d}"
    for part in parts[1:]:
        label += f"_subarm{int(part):02d}"
    return label


def tree_path_label_for_terminal(tree: SkeletonTree, root_idx: int, terminal_idx: int, spacing) -> Tuple[str, str]:
    path = bfs_path_indices(tree.neighbors, int(root_idx), int(terminal_idx))
    if not path:
        return "arm00", ""
    dist_root = tree.dist_from_root_mm if tree.dist_from_root_mm is not None else dijkstra_dist_from_root(tree, int(root_idx), spacing)
    endpoints = set(int(x) for x in tree.endpoints)
    parts = []

    def child_extent(parent: int, child: int) -> float:
        seen = {int(parent)}
        queue = deque([int(child)])
        best = float(dist_root[child])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            if node in endpoints:
                best = max(best, float(dist_root[node]))
            for nbr in tree.neighbors[node]:
                if int(nbr) not in seen and dist_root[nbr] > dist_root[node]:
                    queue.append(int(nbr))
        return best

    for i in range(len(path) - 1):
        node = int(path[i])
        next_node = int(path[i + 1])
        downstream_children = [int(n) for n in tree.neighbors[node] if dist_root[n] > dist_root[node]]
        if len(downstream_children) <= 1:
            continue
        downstream_children.sort(key=lambda child: (-child_extent(node, child), child))
        if next_node in downstream_children:
            parts.append(downstream_children.index(next_node) + 1)

    branch_path = ".".join(str(x) for x in parts)
    return tree_label_from_path_parts(parts), branch_path


def branch_path_from_segment_name(segment_name: str) -> str:
    parts = [p for p in str(segment_name).split(".") if p and p != "S"]
    return ".".join(parts)


def branch_label_from_path(branch_path: str) -> str:
    parts = [int(p) for p in str(branch_path).split(".") if str(p).isdigit()]
    return tree_label_from_path_parts(parts)



def split_bifurcating_tree_centerlines(
    label: int,
    component_id: int,
    vessel_info: VesselInfo,
    path_results: List[dict],
    spacing,
    region_centerline_dir: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, pd.DataFrame], List[dict]]:
    if not SPLIT_BIFURCATING_TREE_REGIONS or len(path_results) != 2:
        if SPLIT_BIFURCATING_TREE_REGIONS and len(path_results) > 2:
            print(f"    [tree regions] Expected 2 paths, found {len(path_results)}; skipping base/arm split.")
        return [], {}, []

    spacing = np.asarray(spacing, dtype=float)
    overlap_tol = CENTERLINE_OVERLAP_TOL_MM if CENTERLINE_OVERLAP_TOL_MM is not None else max(0.5, 1.5 * float(np.min(spacing)))

    res1, res2 = sorted(path_results, key=lambda r: int(r.get("path_index", 0)))
    pts1 = centerline_points_from_result(res1)
    pts2 = centerline_points_from_result(res2)

    tree2 = cKDTree(pts2)
    dist12, idx12 = tree2.query(pts1, k=1)
    tree1 = cKDTree(pts1)
    dist21, idx21 = tree1.query(pts2, k=1)

    def overlapping_prefix_len(distances: np.ndarray, tol: float, max_gap_points: int) -> int:
        last_overlap = -1
        gap = 0
        for i, d in enumerate(distances):
            if np.isfinite(d) and d <= tol:
                last_overlap = i
                gap = 0
            else:
                gap += 1
                if gap > max_gap_points:
                    break
        return last_overlap + 1

    common_n1 = overlapping_prefix_len(dist12, overlap_tol, CENTERLINE_OVERLAP_MAX_GAP_POINTS)
    common_n2 = overlapping_prefix_len(dist21, overlap_tol, CENTERLINE_OVERLAP_MAX_GAP_POINTS)

    if common_n1 < MIN_COMMON_BASE_POINTS or common_n2 < MIN_COMMON_BASE_POINTS:
        head_n1 = min(10, len(dist12))
        head_n2 = min(10, len(dist21))
        print(
            f"    [tree regions] No shared base detected "
            f"(path1={common_n1} points, path2={common_n2} points within {overlap_tol:.2f} mm); skipping split."
        )
        print(
            f"    [tree regions] Nearest-distance diagnostics: "
            f"path1 first {head_n1} min/median/max="
            f"{np.nanmin(dist12[:head_n1]):.2f}/{np.nanmedian(dist12[:head_n1]):.2f}/{np.nanmax(dist12[:head_n1]):.2f} mm, "
            f"path2 first {head_n2} min/median/max="
            f"{np.nanmin(dist21[:head_n2]):.2f}/{np.nanmedian(dist21[:head_n2]):.2f}/{np.nanmax(dist21[:head_n2]):.2f} mm."
        )
        return [], {}, []

    base_tree_name = safe_filename(f"{label}_{vessel_info.name}_comp{component_id:02d}")
    base_id = make_tree_region_id(label, vessel_info.name, component_id, "trunk")
    arm1_id = make_tree_region_id(label, vessel_info.name, component_id, "branch01")
    arm2_id = make_tree_region_id(label, vessel_info.name, component_id, "branch02")

    base_idx1 = np.arange(common_n1, dtype=int)
    base_idx2 = idx12[:common_n1].astype(int)
    base_points = 0.5 * (pts1[base_idx1] + pts2[base_idx2])
    arm1_points = pts1[common_n1 - 1:]
    arm2_points = pts2[common_n2 - 1:]

    r1 = np.asarray(res1["radius_mm"])
    r2 = np.asarray(res2["radius_mm"])
    k1 = np.asarray(res1["curvature_1_per_mm"])
    k2 = np.asarray(res2["curvature_1_per_mm"])
    t1 = np.asarray(res1["torsion_1_per_mm"])
    t2 = np.asarray(res2["torsion_1_per_mm"])

    base_radius = 0.5 * (r1[base_idx1] + r2[base_idx2])
    base_curvature = 0.5 * (k1[base_idx1] + k2[base_idx2])
    base_torsion = 0.5 * (t1[base_idx1] + t2[base_idx2])

    arm1_length_mm = arc_length(arm1_points)
    arm2_length_mm = arc_length(arm2_points)
    min_tree_arm_length_mm = min_tree_arm_length_for_vessel(vessel_info)

    if DISCARD_SHORT_TREE_ARMS:
        arm1_short = np.isfinite(arm1_length_mm) and arm1_length_mm < min_tree_arm_length_mm
        arm2_short = np.isfinite(arm2_length_mm) and arm2_length_mm < min_tree_arm_length_mm
        if arm1_short ^ arm2_short:
            keep_region = "arm2" if arm1_short else "arm1"
            keep_role = "branch02" if arm1_short else "branch01"
            discarded_role = "branch01" if arm1_short else "branch02"
            keep_id = make_tree_region_id(label, vessel_info.name, component_id, f"trunk_plus_{keep_role}")
            if arm1_short:
                fused_points = np.vstack([base_points, arm2_points[1:]])
                fused_radius = np.concatenate([base_radius, r2[common_n2:]])
                fused_curvature = np.concatenate([base_curvature, k2[common_n2:]])
                fused_torsion = np.concatenate([base_torsion, t2[common_n2:]])
                fused_source_path_id = res2["path_id"]
                fused_source_idx = np.concatenate([base_idx2, np.arange(common_n2, len(pts2), dtype=int)])
                discarded_region = "branch01"
                discarded_length = arm1_length_mm
                discarded_source_path_id = res1["path_id"]
            else:
                fused_points = np.vstack([base_points, arm1_points[1:]])
                fused_radius = np.concatenate([base_radius, r1[common_n1:]])
                fused_curvature = np.concatenate([base_curvature, k1[common_n1:]])
                fused_torsion = np.concatenate([base_torsion, t1[common_n1:]])
                fused_source_path_id = res1["path_id"]
                fused_source_idx = np.concatenate([base_idx1, np.arange(common_n1, len(pts1), dtype=int)])
                discarded_region = "branch02"
                discarded_length = arm2_length_mm
                discarded_source_path_id = res2["path_id"]

            print(
                f"    [tree regions] Discarding spurious {discarded_region} "
                f"({discarded_length:.2f} mm < {min_tree_arm_length_mm:.2f} mm; "
                f"vessel={vessel_info.name}); "
                f"fused trunk with {keep_role}: {arc_length(fused_points):.2f} mm "
                f"({len(fused_points)} points)."
            )
            regions = [
                {
                    "region_id": keep_id,
                    "region": f"trunk_plus_{keep_role}",
                    "legacy_region": f"base_plus_{keep_region}",
                    "points": fused_points,
                    "source_path_id": fused_source_path_id,
                    "source_path_id_2": "",
                    "source_idx": fused_source_idx,
                    "source_idx_2": None,
                    "discarded_source_path_ids": [discarded_source_path_id],
                    "radius": fused_radius,
                    "curvature": fused_curvature,
                    "torsion": fused_torsion,
                }
            ]
        elif arm1_short and arm2_short:
            print(
                f"    [tree regions] Both independent arms are short "
                f"(arm1={arm1_length_mm:.2f} mm, arm2={arm2_length_mm:.2f} mm; "
                f"threshold={min_tree_arm_length_mm:.2f} mm; vessel={vessel_info.name}). "
                f"Keeping split for review."
            )
            regions = None
        else:
            regions = None
    else:
        regions = None

    if regions is None:
        regions = [
            {
                "region_id": base_id,
                "region": "trunk",
                "legacy_region": "common_base",
                "points": base_points,
                "source_path_id": res1["path_id"],
                "source_path_id_2": res2["path_id"],
                "source_idx": base_idx1,
                "source_idx_2": base_idx2,
                "radius": base_radius,
                "curvature": base_curvature,
                "torsion": base_torsion,
            },
            {
                "region_id": arm1_id,
                "region": "branch01",
                "legacy_region": "arm1",
                "points": arm1_points,
                "source_path_id": res1["path_id"],
                "source_path_id_2": "",
                "source_idx": np.arange(common_n1 - 1, len(pts1), dtype=int),
                "source_idx_2": None,
                "radius": r1[common_n1 - 1:],
                "curvature": k1[common_n1 - 1:],
                "torsion": t1[common_n1 - 1:],
            },
            {
                "region_id": arm2_id,
                "region": "branch02",
                "legacy_region": "arm2",
                "points": arm2_points,
                "source_path_id": res2["path_id"],
                "source_path_id_2": "",
                "source_idx": np.arange(common_n2 - 1, len(pts2), dtype=int),
                "source_idx_2": None,
                "radius": r2[common_n2 - 1:],
                "curvature": k2[common_n2 - 1:],
                "torsion": t2[common_n2 - 1:],
            },
        ]

    summaries = []
    sheets = {}
    for region in regions:
        points = region["points"]
        meta = tree_region_metadata(region["region"])
        summaries.append(tree_region_summary_from_points(
            label=label,
            component_id=component_id,
            vessel_info=vessel_info,
            tree_region_id=region["region_id"],
            tree_region=region["region"],
            points=points,
            source_path_id=region["source_path_id"],
            source_path_id_2=region["source_path_id_2"],
            tree_metadata=meta,
        ))
        sheets[safe_sheet_name(region["region_id"])] = build_tree_region_points_dataframe(
            case_id="",
            label=label,
            vessel_info=vessel_info,
            component_id=component_id,
            tree_region_id=region["region_id"],
            tree_region=region["region"],
            points=points,
            source_path_id=region["source_path_id"],
            source_point_indices=region["source_idx"],
            radius_mm=region["radius"],
            curvature=region["curvature"],
            torsion=region["torsion"],
            source_path_id_2=region["source_path_id_2"],
            source_point_indices_2=region["source_idx_2"],
            tree_metadata=meta,
        )
        if region_centerline_dir and len(points) >= 2:
            branch_depth = np.full(len(points), meta["tree_branch_depth"], dtype=float)
            poly = build_polyline_polydata(points=points, arrays=[
                (region["radius"], "EffectiveRadius"),
                (region["curvature"], "Curvature"),
                (region["torsion"], "Torsion"),
                (branch_depth, "TreeDepth"),
                (np.arange(1, len(points) + 1, dtype=np.float64), "PointIndex"),
            ])
            add_string_point_array(poly, [region["region"]] * len(points), "TreeLabel")
            add_string_point_array(poly, [meta["tree_branch_path"]] * len(points), "TreePath")
            save_vtp(poly, os.path.join(region_centerline_dir, region["region_id"] + ".vtp"))

    if len(regions) == 3 and regions[0]["region"] == "trunk":
        print(
            f"    [tree regions] Split into "
            f"trunk={arc_length(base_points):.2f} mm ({len(base_points)} points), "
            f"branch01={arm1_length_mm:.2f} mm ({len(arm1_points)} points), "
            f"branch02={arm2_length_mm:.2f} mm ({len(arm2_points)} points) "
            f"(overlap tol {overlap_tol:.2f} mm; matched prefixes path1={common_n1}, path2={common_n2})."
        )
    return summaries, sheets, regions


def save_labeled_tree_path_centerlines(path_results: List[dict], regions: List[dict], centerline_dir: Optional[str]) -> None:
    if not centerline_dir or not path_results or not regions:
        return

    for res in path_results:
        n = len(res["x_mm"])
        labels = np.array(["unassigned"] * n, dtype=object)
        codes = np.full(n, TREE_REGION_CODES["unassigned"], dtype=float)
        branch_depths = np.zeros(n, dtype=float)
        branch_paths = np.array([""] * n, dtype=object)
        path_id = res.get("path_id", "")

        for region in regions:
            region_name = str(region["region"])
            meta = tree_region_metadata(region_name)
            code = float(TREE_REGION_CODES.get(region_name, TREE_REGION_CODES["unassigned"]))
            if region.get("source_path_id") == path_id:
                idx = np.asarray(region["source_idx"], dtype=int)
                idx = idx[(idx >= 0) & (idx < n)]
                labels[idx] = region_name
                codes[idx] = code
                branch_depths[idx] = meta["tree_branch_depth"]
                branch_paths[idx] = meta["tree_branch_path"]
            if region.get("source_path_id_2") == path_id and region.get("source_idx_2") is not None:
                idx = np.asarray(region["source_idx_2"], dtype=int)
                idx = idx[(idx >= 0) & (idx < n)]
                labels[idx] = region_name
                codes[idx] = code
                branch_depths[idx] = meta["tree_branch_depth"]
                branch_paths[idx] = meta["tree_branch_path"]

        pts = centerline_points_from_result(res)
        stenosis_binary = np.asarray(res["is_stenotic"], dtype=np.float64)
        enlargement_binary = np.asarray(res.get("is_enlarged", np.zeros(n)), dtype=np.float64)
        poly = build_polyline_polydata(points=pts, arrays=[
            (res["curvature_1_per_mm"], "Curvature"),
            (res["torsion_1_per_mm"], "Torsion"),
            (res["radius_mm"], "EffectiveRadius"),
            (res["radius_mm"], "CrossSectionRadius"),
            (res.get("maximum_inscribed_sphere_radius_mm", np.full(n, np.nan)), "MaximumInscribedSphereRadius"),
            (res.get("stenosis_detection_radius_mm", res["radius_mm"]), "StenosisDetectionRadius"),
            (res.get("stenosis_reference_radius_point", np.full(n, np.nan)), "StenosisReferenceRadius"),
            (res.get("stenosis_threshold_radius_point", np.full(n, np.nan)), "StenosisThresholdRadius"),
            (res.get("stenosis_raw_percent_point", np.full(n, np.nan)), "StenosisRawPercent"),
            (res.get("stenosis_core_candidate_point", np.full(n, np.nan)), "StenosisCoreCandidate"),
            (res.get("stenosis_support_candidate_point", np.full(n, np.nan)), "StenosisSupportCandidate"),
            (res.get("stenosis_percent_point", np.full(n, np.nan)), "StenosisPercent"),
            (stenosis_binary, "StenosisBinary"),
            (res.get("enlargement_reference_radius_point", np.full(n, np.nan)), "EnlargementReferenceRadius"),
            (res.get("enlargement_threshold_radius_point", np.full(n, np.nan)), "EnlargementThresholdRadius"),
            (res.get("enlargement_percent_point", np.full(n, np.nan)), "EnlargementPercent"),
            (enlargement_binary, "EnlargementBinary"),
            (branch_depths, "TreeDepth"),
            (np.arange(1, n + 1, dtype=np.float64), "PointIndex"),
        ])
        add_string_point_array(poly, labels.tolist(), "TreeLabel")
        add_string_point_array(poly, branch_paths.tolist(), "TreePath")
        save_vtp(poly, os.path.join(centerline_dir, path_id + ".vtp"))
        assigned_n = int(np.sum(codes != TREE_REGION_CODES["unassigned"]))
        print(f"    [tree regions] Labeled {path_id}.vtp: {assigned_n}/{n} points assigned.")
