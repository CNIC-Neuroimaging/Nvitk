"""Vessel anatomy, root selection, flow orientation, and anatomical naming."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from nvitk.measure.morphometrics_config import EXPORT_ANATOMIC_SPLIT_CENTERLINES, TREE_REGION_ROLE_CODES
from .io_utils import safe_filename
from .models import SkeletonTree, VesselInfo

def nearest_dist_to_vessel_names(vessel_names: List[str], pt_mm: np.ndarray, multilabel: np.ndarray, spacing, mapping: dict) -> float:
    spacing = np.asarray(spacing, dtype=float)
    best = np.inf
    wanted = set(vessel_names or [])
    if not wanted:
        return best
    for label, info in mapping.items():
        if info.name not in wanted:
            continue
        vox = np.argwhere(multilabel == label)
        if len(vox) == 0:
            continue
        d = float(np.min(np.linalg.norm(vox.astype(float) * spacing - pt_mm, axis=1)))
        best = min(best, d)
    return best


def anatomical_endpoint_score(pt_mm: np.ndarray, rule: Optional[str]) -> float:
    if rule == "inferior":
        return float(pt_mm[2])
    if rule == "superior":
        return float(-pt_mm[2])
    if rule == "lateral_R":
        return float(pt_mm[0])
    if rule == "lateral_L":
        return float(-pt_mm[0])
    if rule == "anterior":
        return float(-pt_mm[1])
    if rule == "posterior":
        return float(pt_mm[1])
    return 0.0


def choose_root_endpoint(
    tree: SkeletonTree,
    vessel_info: VesselInfo,
    multilabel: np.ndarray,
    spacing,
    mapping: dict,
    mask_bool: Optional[np.ndarray] = None,
) -> int:
    """Choose proximal/root endpoint among all skeleton terminals."""
    if not tree.endpoints:
        raise RuntimeError("Cannot choose root: skeleton has no endpoints.")
    spacing = np.asarray(spacing, dtype=float)
    endpoint_pts_mm = np.asarray([tree.pts_vox[i].astype(float) * spacing for i in tree.endpoints])

    upstream = vessel_info.flow_from if vessel_info.flow_from and vessel_info.flow_from != "systemic" else ""
    if upstream:
        d = np.array([
            nearest_dist_to_vessel_names([upstream], p, multilabel, spacing, mapping)
            for p in endpoint_pts_mm
        ])
        if np.isfinite(d).any():
            best_local = int(np.nanargmin(d))
            root = int(tree.endpoints[best_local])
            print(f"    [tree root] closest to flow_from='{upstream}': endpoint={best_local} dist={d[best_local]:.2f} mm")
            return root

    # If no upstream label is present, use the existing anatomical fallback rule.
    rule = vessel_info.no_upstream_start
    if rule:
        scores = np.array([anatomical_endpoint_score(p, rule) for p in endpoint_pts_mm])
        root = int(tree.endpoints[int(np.argmin(scores))])
        print(f"    [tree root] anatomical fallback '{rule}'.")
        return root

    # Last fallback: largest EDT radius at endpoint, if mask is available.
    if mask_bool is not None:
        dist_mm = ndi.distance_transform_edt(mask_bool, sampling=spacing)
        radii = []
        for idx in tree.endpoints:
            ijk = tuple(map(int, tree.pts_vox[idx]))
            radii.append(float(dist_mm[ijk]))
        root = int(tree.endpoints[int(np.nanargmax(radii))])
        print("    [tree root] fallback: largest endpoint EDT radius.")
        return root

    print("    [tree root] fallback: first endpoint.")
    return int(tree.endpoints[0])


def orient_centerline_points_by_flow(
    pts: np.ndarray,
    vessel_info: VesselInfo,
    multilabel: np.ndarray,
    spacing,
    mapping: dict,
) -> Tuple[np.ndarray, bool]:
    spacing = np.asarray(spacing, dtype=float)
    downstream = [name for name in (vessel_info.flow_to or []) if name and not name.startswith("cortex") and name != "systemic"]
    upstream = vessel_info.flow_from if vessel_info.flow_from and vessel_info.flow_from != "systemic" else ""
    if len(pts) < 2:
        return pts, False
    d0_down = nearest_dist_to_vessel_names(downstream, pts[0], multilabel, spacing, mapping) if downstream else np.inf
    d1_down = nearest_dist_to_vessel_names(downstream, pts[-1], multilabel, spacing, mapping) if downstream else np.inf
    d0_up = nearest_dist_to_vessel_names([upstream], pts[0], multilabel, spacing, mapping) if upstream else np.inf
    d1_up = nearest_dist_to_vessel_names([upstream], pts[-1], multilabel, spacing, mapping) if upstream else np.inf
    downstream_valid = np.isfinite(d0_down) or np.isfinite(d1_down)
    upstream_valid = np.isfinite(d0_up) or np.isfinite(d1_up)
    if downstream_valid and upstream_valid:
        keep = (d0_up + d1_down) <= (d1_up + d0_down)
        print(f"    [direction] flow_from={upstream} flow_to={downstream}: start_up={d0_up:.1f} end_down={d1_down:.1f} | start_down={d0_down:.1f} end_up={d1_up:.1f}")
        return (pts, False) if keep else (pts[::-1], True)
    if downstream_valid:
        print(f"    [direction] flow_to={downstream}: start={d0_down:.1f} end={d1_down:.1f}")
        return (pts, False) if d1_down <= d0_down else (pts[::-1], True)
    if upstream_valid:
        print(f"    [direction] flow_from={upstream}: start={d0_up:.1f} end={d1_up:.1f}")
        return (pts, False) if d0_up <= d1_up else (pts[::-1], True)
    return pts, False


def make_path_id(label: int, vessel_name: str, component_id: int, path_i: int, terminal_i: int, tree_label: Optional[str] = None) -> str:
    prefix = f"{label}_{vessel_name}_comp{component_id:02d}"
    if EXPORT_ANATOMIC_SPLIT_CENTERLINES and int(terminal_i) == 0:
        suffix = "M1"
        if int(component_id) != 1:
            suffix += f"_comp{int(component_id):02d}"
        return safe_filename(f"{vessel_name}_{suffix}")
    if tree_label:
        return safe_filename(f"{prefix}_{tree_label}")
    if int(terminal_i) == 0:
        return safe_filename(f"{prefix}_trunk")
    return safe_filename(f"{prefix}_arm{path_i:02d}")


def display_anatomic_segment_path(segment_path: str) -> str:
    segment_path = str(segment_path or "")
    return segment_path[3:] if segment_path.startswith("M1_M") else segment_path


def make_loop_region_path_id(label: int, vessel_name: str, component_id: int, loop_i: int, region_name: str, region_i: int = 1) -> str:
    if str(region_name) == "alternate_arm":
        suffix = "loop_arm"
        if int(loop_i) != 1:
            suffix += f"_loop{int(loop_i):02d}"
        if int(region_i) != 1:
            suffix += f"_{int(region_i):02d}"
        if int(component_id) != 1:
            suffix += f"_comp{int(component_id):02d}"
        return safe_filename(f"{vessel_name}_{suffix}")
    clean_region = str(region_name).replace("alternate_arm", "loop_arm").replace("selected_arm", "loop_selected_arm")
    suffix = f"{clean_region}{region_i}" if region_i != 1 else clean_region
    return safe_filename(f"{label}_{vessel_name}_comp{component_id:02d}_loop{loop_i:02d}_{suffix}")


def make_tree_region_id(label: int, vessel_name: str, component_id: int, role: str) -> str:
    return safe_filename(f"{label}_{vessel_name}_comp{component_id:02d}_{role}")


def tree_region_metadata(region_name: str) -> dict:
    name = str(region_name or "unassigned")
    if name in {"trunk", "common_base"}:
        return {
            "tree_region_role": "trunk",
            "tree_region_role_code": TREE_REGION_ROLE_CODES["trunk"],
            "tree_branch_depth": 0,
            "tree_branch_index": 0,
            "tree_branch_path": "",
            "parent_tree_branch_path": "",
        }

    if name.startswith("trunk_plus_branch") or name.startswith("base_plus_arm"):
        suffix = name.replace("trunk_plus_branch", "").replace("base_plus_arm", "")
        branch_index = int(suffix) if suffix.isdigit() else 0
        return {
            "tree_region_role": "fused_trunk_branch",
            "tree_region_role_code": TREE_REGION_ROLE_CODES["fused_trunk_branch"],
            "tree_branch_depth": 1 if branch_index else 0,
            "tree_branch_index": branch_index,
            "tree_branch_path": str(branch_index) if branch_index else "",
            "parent_tree_branch_path": "",
        }

    if name.startswith("branch") or name.startswith("arm"):
        suffix = name.replace("branch", "").replace("arm", "")
        branch_index = int(suffix) if suffix.isdigit() else 0
        return {
            "tree_region_role": "branch",
            "tree_region_role_code": TREE_REGION_ROLE_CODES["branch"],
            "tree_branch_depth": 1 if branch_index else 0,
            "tree_branch_index": branch_index,
            "tree_branch_path": str(branch_index) if branch_index else "",
            "parent_tree_branch_path": "",
        }

    if name == "selected_arm":
        role = "donut_selected_arm"
    elif name == "alternate_arm":
        role = "donut_alternate_arm"
    else:
        role = "unassigned"
    return {
        "tree_region_role": role,
        "tree_region_role_code": TREE_REGION_ROLE_CODES[role],
        "tree_branch_depth": 0,
        "tree_branch_index": 0,
        "tree_branch_path": "",
        "parent_tree_branch_path": "",
    }
