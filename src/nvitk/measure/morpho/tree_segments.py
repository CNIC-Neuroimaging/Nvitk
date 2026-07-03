"""Recursive tree segment construction, assignment, and anatomical exports."""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .anatomy import display_anatomic_segment_path
from .centerlines import centerline_points_from_result, save_centerline_result_vtps
from nvitk.measure.morphometrics_config import (
    ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS,
    EXPORT_ANATOMIC_SPLIT_CENTERLINES,
    REMOVE_ROOT_TO_TERMINAL_CENTERLINE_VTPS_AFTER_SPLIT,
    TREE_REGION_ROLE_CODES,
    TREE_SEGMENT_ASSIGN_TOL_MM,
)
from .geometry import arc_length, contiguous_true_runs, cumulative_s, resample_generated_centerline_points
from .io_utils import safe_filename
from .metrics import discrete_curvature, discrete_torsion
from .models import SkeletonTree, VesselInfo
from .skeleton import dijkstra_dist_from_root
from .surface import add_string_point_array, build_polyline_polydata, save_vtp
from .tree_regions import branch_label_from_path, branch_path_from_segment_name

def build_recursive_tree_segments(tree: SkeletonTree, spacing) -> List[dict]:
    if tree.root is None:
        return []
    spacing = np.asarray(spacing, dtype=float)
    dist_root = tree.dist_from_root_mm if tree.dist_from_root_mm is not None else dijkstra_dist_from_root(tree, tree.root, spacing)

    branchpoint_set = set(int(x) for x in tree.branchpoints)
    endpoints = set(int(x) for x in tree.endpoints)

    branch_node_to_cluster: Dict[int, int] = {}
    cluster_members: Dict[int, List[int]] = {}
    seen_branch_nodes = set()
    for bp in sorted(branchpoint_set, key=lambda node: (dist_root[node], node)):
        if bp in seen_branch_nodes:
            continue
        cluster_id = len(cluster_members) + 1
        queue_cluster = deque([bp])
        seen_branch_nodes.add(bp)
        members = []
        while queue_cluster:
            node = int(queue_cluster.popleft())
            members.append(node)
            branch_node_to_cluster[node] = cluster_id
            for nbr in tree.neighbors[node]:
                nbr = int(nbr)
                if nbr in branchpoint_set and nbr not in seen_branch_nodes:
                    seen_branch_nodes.add(nbr)
                    queue_cluster.append(nbr)
        cluster_members[cluster_id] = sorted(members, key=lambda node: (dist_root[node], node))

    def junction_key_for_node(node: int) -> str:
        node = int(node)
        if node in branch_node_to_cluster:
            return f"B{branch_node_to_cluster[node]}"
        if node in endpoints:
            return f"E{node}"
        return f"N{node}"

    def junction_members(junction_key: str) -> List[int]:
        if junction_key.startswith("B"):
            return cluster_members.get(int(junction_key[1:]), [])
        return [int(junction_key[1:])]

    def outgoing_edges(junction_key: str, parent_junction_key: Optional[str]) -> List[Tuple[int, int]]:
        junction_nodes = set(junction_members(junction_key))
        parent_nodes = set(junction_members(parent_junction_key)) if parent_junction_key else set()
        edges = []
        for node in sorted(junction_nodes, key=lambda value: (dist_root[value], value)):
            for nbr in tree.neighbors[node]:
                nbr = int(nbr)
                if nbr in junction_nodes or nbr in parent_nodes:
                    continue
                if dist_root[nbr] <= dist_root[node]:
                    continue
                edges.append((int(node), nbr))
        edges.sort(key=lambda edge: (dist_root[edge[1]], edge[1]))
        return edges

    segments = []
    root_junction = junction_key_for_node(int(tree.root))
    queue = deque([(root_junction, None, 0, "S")])
    seg_counter = 0
    visited_edges = set()

    while queue:
        start_junction, parent_junction, depth, prefix = queue.popleft()
        child_edges = outgoing_edges(start_junction, parent_junction)
        for child_i, (start_node, nbr) in enumerate(child_edges, start=1):
            edge_key = tuple(sorted((int(start_node), int(nbr))))
            if edge_key in visited_edges:
                continue
            chain = [int(start_node), int(nbr)]
            visited_edges.add(edge_key)
            prev = int(start_node)
            cur = int(nbr)
            while cur not in endpoints and cur not in branch_node_to_cluster:
                nexts = [
                    int(n)
                    for n in tree.neighbors[cur]
                    if int(n) != prev and dist_root[int(n)] > dist_root[cur]
                ]
                if not nexts:
                    break
                nexts.sort(key=lambda n: (dist_root[n], n))
                nxt = nexts[0]
                visited_edges.add(tuple(sorted((cur, nxt))))
                chain.append(nxt)
                prev, cur = cur, nxt

            end_junction = junction_key_for_node(cur)
            seg_counter += 1
            segment_name = f"{prefix}.{child_i}" if prefix else f"S{child_i}"
            branch_path = branch_path_from_segment_name(segment_name)
            branch_parts = [int(p) for p in branch_path.split(".") if p.isdigit()]
            parent_branch_path = ".".join(str(p) for p in branch_parts[:-1])
            branch_index = branch_parts[-1] if branch_parts else 0
            branch_depth = len(branch_parts)
            region_label = branch_label_from_path(branch_path)
            pts_mm = tree.pts_vox[chain].astype(float) * spacing
            segments.append({
                "segment_id": int(seg_counter),
                "segment_name": segment_name,
                "region_label": region_label,
                "tree_region_role": "branch",
                "tree_region_role_code": TREE_REGION_ROLE_CODES["branch"],
                "tree_branch_path": branch_path,
                "parent_tree_branch_path": parent_branch_path,
                "tree_branch_index": int(branch_index),
                "tree_branch_depth": int(branch_depth),
                "depth": int(branch_depth),
                "start_node": int(start_node),
                "end_node": int(cur),
                "start_junction": start_junction,
                "end_junction": end_junction,
                "node_indices": [int(x) for x in chain],
                "points": pts_mm,
                "length_mm": float(arc_length(pts_mm)),
                "is_terminal": bool(cur in tree.endpoints and cur != tree.root),
            })

            if cur in branch_node_to_cluster:
                queue.append((end_junction, start_junction, depth + 1, segment_name))

    return segments


def segment_child_z_score(segment: dict, lookahead_points: int = ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS) -> float:
    pts = np.asarray(segment.get("_anatomic_centerline_points_for_z", segment.get("points", [])), dtype=float)
    if len(pts) == 0:
        return np.nan
    start = 1 if len(pts) > 1 else 0
    stop = min(len(pts), start + int(max(1, lookahead_points)))
    vals = pts[start:stop, 2]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


def branch_suffixes_by_superior_inferior(child_segments: List[dict]) -> Dict[int, str]:
    children = [seg for seg in child_segments if seg is not None]
    if not children:
        return {}
    scored = []
    for seg in children:
        score = segment_child_z_score(seg)
        scored.append((float(score) if np.isfinite(score) else -np.inf, int(seg["segment_id"]), seg))
    scored.sort(key=lambda row: (row[0], row[1]))
    if len(scored) == 1:
        return {int(scored[0][2]["segment_id"]): ""}

    suffixes = {}
    suffixes[int(scored[0][2]["segment_id"])] = "i"
    suffixes[int(scored[-1][2]["segment_id"])] = "s"
    for rank, (_score, _seg_id, seg) in enumerate(scored[1:-1], start=2):
        suffixes[int(seg["segment_id"])] = f"b{rank:02d}"
    return suffixes


def annotate_anatomic_tree_segments(tree: SkeletonTree, segments: List[dict]) -> List[dict]:
    """Assign M1/M2s/M2i/... labels to directed skeleton segments."""
    if tree.root is None or not segments:
        return segments

    by_start: Dict[str, List[dict]] = {}
    for seg in segments:
        start_key = seg.get("start_junction", f"N{int(seg['start_node'])}")
        by_start.setdefault(start_key, []).append(seg)
    for children in by_start.values():
        children.sort(key=lambda seg: (segment_child_z_score(seg), int(seg["segment_id"])))

    annotated_by_id: Dict[int, dict] = {int(seg["segment_id"]): seg for seg in segments}
    queue = deque()
    root_children = by_start.get(f"E{int(tree.root)}", by_start.get(f"N{int(tree.root)}", []))
    if len(root_children) == 1:
        seg = root_children[0]
        seg.update({
            "anatomic_segment_name": "M1",
            "anatomic_segment_path": "M1",
            "anatomic_generation": 1,
            "anatomic_parent_path": "",
            "anatomic_branch_suffix": "",
        })
        queue.append(seg)
    else:
        suffixes = branch_suffixes_by_superior_inferior(root_children)
        for seg in root_children:
            suffix = suffixes.get(int(seg["segment_id"]), "")
            name = f"M1{suffix}" if suffix else "M1"
            seg.update({
                "anatomic_segment_name": name,
                "anatomic_segment_path": name,
                "anatomic_generation": 1,
                "anatomic_parent_path": "",
                "anatomic_branch_suffix": suffix,
            })
            queue.append(seg)

    visited_ids: set = set()
    while queue:
        parent_seg = queue.popleft()
        parent_id = int(parent_seg["segment_id"])
        if parent_id in visited_ids:
            continue
        visited_ids.add(parent_id)
        end_key = parent_seg.get("end_junction", f"N{int(parent_seg['end_node'])}")
        children = by_start.get(end_key, [])
        if not children:
            continue
        generation = int(parent_seg.get("anatomic_generation", 1)) + 1
        suffixes = branch_suffixes_by_superior_inferior(children)
        for child in children:
            if int(child["segment_id"]) in visited_ids:
                continue
            suffix = suffixes.get(int(child["segment_id"]), "")
            name = f"M{generation}{suffix}" if suffix else f"M{generation}"
            parent_path = str(parent_seg.get("anatomic_segment_path", parent_seg.get("region_label", "")))
            segment_path = name if parent_path == "M1" else (f"{parent_path}_{name}" if parent_path else name)
            child.update({
                "anatomic_segment_name": name,
                "anatomic_segment_path": segment_path,
                "anatomic_generation": generation,
                "anatomic_parent_path": parent_path,
                "anatomic_branch_suffix": suffix,
            })
            queue.append(child)

    for seg_id, seg in annotated_by_id.items():
        if "anatomic_segment_path" not in seg:
            fallback = str(seg.get("region_label", f"segment{seg_id:02d}"))
            seg.update({
                "anatomic_segment_name": fallback,
                "anatomic_segment_path": fallback,
                "anatomic_generation": int(seg.get("tree_branch_depth", 0)),
                "anatomic_parent_path": str(seg.get("parent_tree_branch_path", "")),
                "anatomic_branch_suffix": "",
            })
    return segments


def assign_points_to_recursive_segments(points: np.ndarray, segments: List[dict], assign_tol_mm: float) -> dict:
    n = len(points)
    assigned = {
        "tree_segment_label": np.array(["unassigned"] * n, dtype=object),
        "tree_segment_id": np.zeros(n, dtype=float),
        "tree_region_role": np.array(["unassigned"] * n, dtype=object),
        "tree_region_role_code": np.full(n, TREE_REGION_ROLE_CODES["unassigned"], dtype=float),
        "tree_branch_depth": np.zeros(n, dtype=float),
        "tree_branch_index": np.zeros(n, dtype=float),
        "tree_branch_path": np.array([""] * n, dtype=object),
        "parent_tree_branch_path": np.array([""] * n, dtype=object),
    }
    if not segments or n == 0:
        return assigned

    segment_points = []
    segment_values = {key: [] for key in assigned}
    for seg in segments:
        pts = np.asarray(seg["points"], dtype=float)
        if len(pts) == 0:
            continue
        segment_points.append(pts)
        segment_values["tree_segment_label"].extend([seg["region_label"]] * len(pts))
        segment_values["tree_segment_id"].extend([float(seg["segment_id"])] * len(pts))
        segment_values["tree_region_role"].extend([seg["tree_region_role"]] * len(pts))
        segment_values["tree_region_role_code"].extend([float(seg["tree_region_role_code"])] * len(pts))
        segment_values["tree_branch_depth"].extend([float(seg["tree_branch_depth"])] * len(pts))
        segment_values["tree_branch_index"].extend([float(seg["tree_branch_index"])] * len(pts))
        segment_values["tree_branch_path"].extend([seg["tree_branch_path"]] * len(pts))
        segment_values["parent_tree_branch_path"].extend([seg["parent_tree_branch_path"]] * len(pts))
    if not segment_points:
        return assigned

    all_segment_points = np.vstack(segment_points)
    segment_arrays = {
        key: np.asarray(values, dtype=object if ("path" in key or "label" in key or key == "tree_region_role") else float)
        for key, values in segment_values.items()
    }
    dist, idx = cKDTree(all_segment_points).query(points, k=1)
    is_assigned = np.isfinite(dist) & (dist <= assign_tol_mm)
    for key, values in segment_arrays.items():
        assigned[key][is_assigned] = values[idx[is_assigned]]
    return assigned


def save_recursive_labeled_tree_path_centerlines(
    path_results: List[dict],
    segments: List[dict],
    centerline_dir: Optional[str],
    spacing,
    skip_path_ids: Optional[set] = None,
) -> None:
    if not centerline_dir or not path_results or not segments:
        return
    skip_path_ids = skip_path_ids or set()
    spacing = np.asarray(spacing, dtype=float)
    assign_tol = TREE_SEGMENT_ASSIGN_TOL_MM if TREE_SEGMENT_ASSIGN_TOL_MM is not None else max(0.75, 2.0 * float(np.min(spacing)))
    for res in path_results:
        if res.get("path_id", "") in skip_path_ids:
            continue
        pts = centerline_points_from_result(res)
        assigned = assign_points_to_recursive_segments(pts, segments, assign_tol)
        n = len(pts)
        stenosis_binary = np.asarray(res["is_stenotic"], dtype=np.float64)
        enlargement_binary = np.asarray(res.get("is_enlarged", np.zeros(n)), dtype=np.float64)
        enlargement_binary_cs = np.asarray(res.get("is_enlarged_cs", enlargement_binary), dtype=np.float64)
        enlargement_binary_mis = np.asarray(res.get("is_enlarged_mis", enlargement_binary), dtype=np.float64)
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
            (enlargement_binary_cs, "EnlargementBinaryCS"),
            (enlargement_binary_mis, "EnlargementBinaryMIS"),
            (assigned["tree_branch_depth"], "TreeDepth"),
            (np.arange(1, n + 1, dtype=np.float64), "PointIndex"),
        ])
        add_string_point_array(poly, assigned["tree_segment_label"].tolist(), "TreeLabel")
        add_string_point_array(poly, assigned["tree_branch_path"].tolist(), "TreePath")
        save_vtp(poly, os.path.join(centerline_dir, res["path_id"] + ".vtp"))
        assigned_n = int(np.sum(assigned["tree_segment_id"] > 0))
        print(f"    [tree segments] Labeled {res['path_id']}.vtp: {assigned_n}/{n} points assigned (tol {assign_tol:.2f} mm).")


def remove_root_to_terminal_centerline_outputs(
    path_results: List[dict],
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
    keep_path_ids: Optional[set] = None,
) -> None:
    if not REMOVE_ROOT_TO_TERMINAL_CENTERLINE_VTPS_AFTER_SPLIT:
        return
    keep_path_ids = set(str(x) for x in (keep_path_ids or set()))
    for res in path_results:
        path_id = str(res.get("path_id", ""))
        if not path_id or path_id in keep_path_ids:
            continue
        candidates = []
        if centerline_dir:
            candidates.append(os.path.join(centerline_dir, path_id + ".vtp"))
        if centerline_radius_dir:
            candidates.append(os.path.join(centerline_radius_dir, path_id + "_radius.vtp"))
        for path in candidates:
            if os.path.exists(path):
                os.remove(path)


def collapse_unary_supported_anatomic_segments(
    segments: List[dict],
    best_chunks: Dict[int, dict],
    root_idx: Optional[int],
) -> List[dict]:
    """Collapse skeleton splits that are not supported by final kept paths."""
    supported = [seg for seg in segments if int(seg["segment_id"]) in best_chunks]
    if not supported:
        return []

    by_start: Dict[str, List[dict]] = {}
    incoming = set()
    for seg in supported:
        by_start.setdefault(str(seg["start_junction"]), []).append(seg)
        incoming.add(str(seg["end_junction"]))
    for children in by_start.values():
        children.sort(key=lambda seg: (segment_child_z_score(seg), int(seg["segment_id"])))

    roots: List[dict] = []
    if root_idx is not None:
        for key in (f"E{int(root_idx)}", f"N{int(root_idx)}"):
            roots.extend(by_start.get(key, []))
    if not roots:
        roots = [seg for seg in supported if str(seg["start_junction"]) not in incoming]
    if not roots:
        roots = sorted(supported, key=lambda seg: int(seg["segment_id"]))[:1]

    export_segments = []
    visited = set()

    def stitched_chain_points(chain: List[dict]) -> np.ndarray:
        parts = []
        for seg in chain:
            pts = np.asarray(best_chunks[int(seg["segment_id"])]["points"], dtype=float)
            if len(pts) == 0:
                continue
            if parts and np.linalg.norm(parts[-1][-1] - pts[0]) <= 1e-6:
                pts = pts[1:]
            if len(pts):
                parts.append(pts)
        return np.vstack(parts) if parts else np.empty((0, 3), dtype=float)

    def add_chain(start_seg: dict) -> None:
        chain = []
        cur = start_seg
        while int(cur["segment_id"]) not in visited:
            chain.append(cur)
            visited.add(int(cur["segment_id"]))
            children = [
                child for child in by_start.get(str(cur["end_junction"]), [])
                if int(child["segment_id"]) not in visited
            ]
            if len(children) != 1:
                break
            cur = children[0]

        if not chain:
            return

        first = chain[0]
        last = chain[-1]
        merged = dict(first)
        points = stitched_chain_points(chain)
        merged.update({
            "segment_id": int(first["segment_id"]),
            "source_segment_ids": [int(seg["segment_id"]) for seg in chain],
            "segment_name": "_".join(str(seg.get("segment_name", seg["segment_id"])) for seg in chain),
            "end_node": int(last["end_node"]),
            "end_junction": str(last["end_junction"]),
            "is_terminal": bool(last.get("is_terminal", False)),
            "points": points,
            "length_mm": float(arc_length(points)),
            "_anatomic_centerline_points_for_z": points,
            "_collapsed_from_n_segments": int(len(chain)),
        })
        export_segments.append(merged)

        for child in by_start.get(str(last["end_junction"]), []):
            if int(child["segment_id"]) not in visited:
                add_chain(child)

    for root in roots:
        if int(root["segment_id"]) not in visited:
            add_chain(root)
    for seg in sorted(supported, key=lambda row: int(row["segment_id"])):
        if int(seg["segment_id"]) not in visited:
            add_chain(seg)

    collapsed = sum(max(0, int(seg.get("_collapsed_from_n_segments", 1)) - 1) for seg in export_segments)
    if collapsed:
        print(f"    [anatomic segments] Collapsed {collapsed} unary/spurious skeleton split(s) after final path pruning.")
    return export_segments


def save_anatomic_split_tree_centerlines(
    label: int,
    component_id: int,
    vessel_info: VesselInfo,
    path_results: List[dict],
    segments: List[dict],
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
    spacing,
    root_idx: Optional[int] = None,
) -> List[str]:
    if not EXPORT_ANATOMIC_SPLIT_CENTERLINES or not centerline_dir or not path_results or not segments:
        return []

    spacing = np.asarray(spacing, dtype=float)
    assign_tol = TREE_SEGMENT_ASSIGN_TOL_MM if TREE_SEGMENT_ASSIGN_TOL_MM is not None else max(0.75, 2.0 * float(np.min(spacing)))
    segment_by_id = {int(seg["segment_id"]): seg for seg in segments}
    best_chunks: Dict[int, dict] = {}

    for res in path_results:
        pts = centerline_points_from_result(res)
        if len(pts) < 2:
            continue
        assigned = assign_points_to_recursive_segments(pts, segments, assign_tol)
        segment_ids = np.asarray(assigned["tree_segment_id"], dtype=float)
        for seg_id in sorted(segment_by_id):
            for start, end in contiguous_true_runs(segment_ids == float(seg_id)):
                if end - start < 2:
                    continue
                chunk_pts = pts[start:end]
                chunk_len = arc_length(chunk_pts)
                previous = best_chunks.get(seg_id)
                if previous is None or chunk_len > previous["length_mm"]:
                    best_chunks[seg_id] = {
                        "res": res,
                        "start": int(start),
                        "end": int(end),
                        "points": chunk_pts,
                        "length_mm": float(chunk_len),
                    }

    for seg_id, chunk in best_chunks.items():
        if seg_id in segment_by_id:
            segment_by_id[seg_id]["_anatomic_centerline_points_for_z"] = np.asarray(chunk["points"], dtype=float)
    export_segments = collapse_unary_supported_anatomic_segments(segments, best_chunks, root_idx)
    if not export_segments:
        print("    [anatomic segments] Warning: no final supported tree segments to export.")
        return []
    if root_idx is not None:
        annotate_anatomic_tree_segments(type("AnatomicSegmentTreeView", (), {"root": int(root_idx)})(), export_segments)

    saved = []
    vessel_token = safe_filename(str(vessel_info.name or f"label_{label}"))
    for seg in sorted(export_segments, key=lambda row: int(row["segment_id"])):
        seg_id = int(seg["segment_id"])
        source_segment_ids = [int(x) for x in seg.get("source_segment_ids", [seg_id])]

        source_parts = []
        value_parts: Dict[str, List[np.ndarray]] = {
            "is_stenotic": [],
            "is_enlarged": [],
            "is_enlarged_cs": [],
            "is_enlarged_mis": [],
            "radius_mm": [],
            "maximum_inscribed_sphere_radius_mm": [],
            "stenosis_detection_radius_mm": [],
            "stenosis_reference_radius_point": [],
            "stenosis_threshold_radius_point": [],
            "stenosis_raw_percent_point": [],
            "stenosis_core_candidate_point": [],
            "stenosis_support_candidate_point": [],
            "stenosis_percent_point": [],
            "enlargement_reference_radius_point": [],
            "enlargement_threshold_radius_point": [],
            "enlargement_percent_point": [],
        }
        for source_seg_id in source_segment_ids:
            chunk = best_chunks.get(source_seg_id)
            if chunk is None:
                continue
            part_pts = np.asarray(chunk["points"], dtype=float)
            if len(part_pts) == 0:
                continue
            drop_first = bool(source_parts and np.linalg.norm(source_parts[-1][-1] - part_pts[0]) <= 1e-6)
            if drop_first:
                part_pts = part_pts[1:]
            if len(part_pts) == 0:
                continue
            source_parts.append(part_pts)

            res = chunk["res"]
            start = int(chunk["start"])
            end = int(chunk["end"])
            defaults = {
                "is_stenotic": np.zeros(len(res["s_mm"])),
                "is_enlarged": np.zeros(len(res["s_mm"])),
                "is_enlarged_cs": np.zeros(len(res["s_mm"])),
                "is_enlarged_mis": np.zeros(len(res["s_mm"])),
                "radius_mm": np.full(len(res["s_mm"]), np.nan),
                "maximum_inscribed_sphere_radius_mm": np.full(len(res["s_mm"]), np.nan),
                "stenosis_detection_radius_mm": np.asarray(res.get("radius_mm", np.full(len(res["s_mm"]), np.nan)), dtype=float),
                "stenosis_reference_radius_point": np.full(len(res["s_mm"]), np.nan),
                "stenosis_threshold_radius_point": np.full(len(res["s_mm"]), np.nan),
                "stenosis_raw_percent_point": np.full(len(res["s_mm"]), np.nan),
                "stenosis_core_candidate_point": np.zeros(len(res["s_mm"]), dtype=float),
                "stenosis_support_candidate_point": np.zeros(len(res["s_mm"]), dtype=float),
                "stenosis_percent_point": np.full(len(res["s_mm"]), np.nan),
                "enlargement_reference_radius_point": np.full(len(res["s_mm"]), np.nan),
                "enlargement_threshold_radius_point": np.full(len(res["s_mm"]), np.nan),
                "enlargement_percent_point": np.full(len(res["s_mm"]), np.nan),
            }
            for key in value_parts:
                values = np.asarray(res.get(key, defaults[key]), dtype=float)[start:end]
                if drop_first:
                    values = values[1:]
                value_parts[key].append(values)

        if not source_parts:
            print(
                f"    [anatomic segments] Warning: no centerline points assigned to "
                f"{seg.get('anatomic_segment_path', seg_id)} (segment {seg_id})."
            )
            continue

        source_pts = np.vstack(source_parts)
        source_s = cumulative_s(source_pts)
        pts = resample_generated_centerline_points(source_pts)
        n = len(pts)
        target_s = cumulative_s(pts)
        raw_segment_path = str(seg.get("anatomic_segment_path", seg.get("region_label", f"segment{seg_id:02d}")))
        segment_path = safe_filename(display_anatomic_segment_path(raw_segment_path))
        path_id = safe_filename(f"{vessel_token}_{segment_path}")
        if component_id != 1:
            path_id = safe_filename(f"{path_id}_comp{component_id:02d}")

        def chain_array(key: str, default=np.nan, binary: bool = False) -> np.ndarray:
            if not value_parts.get(key):
                return np.full(n, default, dtype=float)
            sliced = np.concatenate(value_parts[key]).astype(float)
            if len(sliced) != len(source_pts):
                return np.full(n, default, dtype=float)
            unchanged_points = n == len(source_pts) and (n == 0 or np.allclose(pts, source_pts))
            if len(source_s) < 2 or source_s[-1] <= 1e-12 or unchanged_points:
                out = sliced.copy()
            else:
                valid = np.isfinite(sliced)
                if valid.sum() < 2:
                    out = np.full(n, default, dtype=float)
                else:
                    out = np.interp(target_s, source_s[valid], sliced[valid])
            if binary:
                return (out >= 0.5).astype(np.float64)
            return out

        stenosis_binary = chain_array("is_stenotic", default=0.0, binary=True)
        enlargement_binary = chain_array("is_enlarged", default=0.0, binary=True)
        enlargement_binary_cs = chain_array("is_enlarged_cs", default=0.0, binary=True)
        enlargement_binary_mis = chain_array("is_enlarged_mis", default=0.0, binary=True)
        radius = chain_array("radius_mm")
        maximum_inscribed_sphere_radius = chain_array("maximum_inscribed_sphere_radius_mm")
        stenosis_detection_radius = chain_array("stenosis_detection_radius_mm")
        stenosis_reference = chain_array("stenosis_reference_radius_point")
        stenosis_threshold = chain_array("stenosis_threshold_radius_point")
        stenosis_raw_percent = chain_array("stenosis_raw_percent_point")
        stenosis_core_candidate = chain_array("stenosis_core_candidate_point", default=0.0, binary=True)
        stenosis_support_candidate = chain_array("stenosis_support_candidate_point", default=0.0, binary=True)
        stenosis_percent = chain_array("stenosis_percent_point")
        enlargement_reference = chain_array("enlargement_reference_radius_point")
        enlargement_threshold = chain_array("enlargement_threshold_radius_point")
        curvature = discrete_curvature(pts)
        torsion = discrete_torsion(pts)
        enlargement_percent = chain_array("enlargement_percent_point")
        tree_depth = np.full(n, float(seg.get("anatomic_generation", seg.get("tree_branch_depth", 0))), dtype=float)

        poly = build_polyline_polydata(points=pts, arrays=[
            (curvature, "Curvature"),
            (torsion, "Torsion"),
            (radius, "EffectiveRadius"),
            (radius, "CrossSectionRadius"),
            (maximum_inscribed_sphere_radius, "MaximumInscribedSphereRadius"),
            (stenosis_detection_radius, "StenosisDetectionRadius"),
            (stenosis_reference, "StenosisReferenceRadius"),
            (stenosis_threshold, "StenosisThresholdRadius"),
            (stenosis_raw_percent, "StenosisRawPercent"),
            (stenosis_core_candidate, "StenosisCoreCandidate"),
            (stenosis_support_candidate, "StenosisSupportCandidate"),
            (stenosis_percent, "StenosisPercent"),
            (stenosis_binary, "StenosisBinary"),
            (enlargement_reference, "EnlargementReferenceRadius"),
            (enlargement_threshold, "EnlargementThresholdRadius"),
            (enlargement_percent, "EnlargementPercent"),
            (enlargement_binary, "EnlargementBinary"),
            (enlargement_binary_cs, "EnlargementBinaryCS"),
            (enlargement_binary_mis, "EnlargementBinaryMIS"),
            (tree_depth, "TreeDepth"),
            (np.arange(1, n + 1, dtype=np.float64), "PointIndex"),
            (np.full(n, seg_id, dtype=np.float64), "TreeSegmentId"),
        ])
        add_string_point_array(poly, [str(seg.get("anatomic_segment_name", segment_path))] * n, "TreeLabel")
        add_string_point_array(poly, [segment_path] * n, "TreePath")
        add_string_point_array(poly, [display_anatomic_segment_path(str(seg.get("anatomic_parent_path", "")))] * n, "ParentTreePath")
        save_vtp(poly, os.path.join(centerline_dir, path_id + ".vtp"))
        saved.append(path_id)

    remove_root_to_terminal_centerline_outputs(
        path_results,
        centerline_dir,
        centerline_radius_dir,
        keep_path_ids=set(saved),
    )
    if saved:
        print(f"    [anatomic segments] Saved {len(saved)} split centerline VTP(s): {', '.join(saved)}")
    return saved


def save_anatomic_fallback_centerlines(
    vessel_info: VesselInfo,
    component_id: int,
    path_results: List[dict],
    centerline_dir: Optional[str],
    centerline_radius_dir: Optional[str],
) -> List[str]:
    if not EXPORT_ANATOMIC_SPLIT_CENTERLINES or not centerline_dir or not path_results:
        return []

    vessel_token = safe_filename(str(vessel_info.name or "vessel"))
    paths = sorted(path_results, key=lambda row: float(row.get("length_mm", 0.0)), reverse=True)
    saved = []

    if len(paths) == 1:
        names = [(paths[0], "M1")]
    else:
        scored = []
        for res in paths:
            pts = centerline_points_from_result(res)
            lookahead = pts[1:min(len(pts), 1 + int(ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS))]
            z = float(np.nanmean(lookahead[:, 2])) if len(lookahead) else -np.inf
            scored.append((z, res))
        scored.sort(key=lambda row: row[0])
        suffix_by_id = {id(scored[0][1]): "M2i", id(scored[-1][1]): "M2s"}
        for rank, (_z, res) in enumerate(scored[1:-1], start=2):
            suffix_by_id[id(res)] = f"M2b{rank:02d}"
        names = [(res, suffix_by_id[id(res)]) for res in paths]

    for res, segment_name in names:
        path_id = safe_filename(f"{vessel_token}_{segment_name}")
        if int(component_id) != 1:
            path_id = safe_filename(f"{path_id}_comp{int(component_id):02d}")
        fallback = dict(res)
        fallback["path_id"] = path_id
        save_centerline_result_vtps(fallback, centerline_dir, centerline_radius_dir)
        saved.append(path_id)

    if saved:
        print(
            "    [anatomic segments] Fallback saved root-to-terminal centerline(s) "
            f"with anatomical names: {', '.join(saved)}"
        )
    return saved
