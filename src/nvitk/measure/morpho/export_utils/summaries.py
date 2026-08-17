"""Summary rows, export dataframes, and aggregate asymmetry/hemisphere outputs."""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import pandas as pd

from nvitk.measure.morpho.anatomy import tree_region_metadata
from nvitk.measure.morphometrics_config import TREE_REGION_ROLE_CODES
from nvitk.measure.morpho.geometry import arc_length, chord_length, cumulative_s
from nvitk.measure.morpho.metrics import tortuosity_dm
from nvitk.measure.morpho.models import VesselInfo

PATH_SUMMARY_EXPORT_DROP_COLUMNS = [
    "case_id",
    "component_id",
    "path_id",
    "path_role",
    "tree_mode",
    "donut_loop_mode",
    "donut_loop_index",
    "donut_arm_index",
    "overlap_pruned_start_points",
    "root_skeleton_index",
    "terminal_skeleton_index",
    "vmtk_seed_trim_retry_mm",
    "side",
    "pair",
    "territory",
    "flow_from",
    "inflection_count",
    "bend_peak_count",
    "radius_ref_mm",
    "radius_min_mm",
    "enlargement_radius_ref_mm",
    "enlargement_radius_max_mm",
]

TREE_SUMMARY_EXPORT_DROP_COLUMNS = [
    "component_id",
    "full_name",
    "side",
    "pair",
    "territory",
    "flow_from",
    "tree_mode",
    "n_terminals",
    "n_branchpoints",
    "n_centerline_paths_discarded_short",
    "min_centerline_path_length_mm",
    # unique_skeleton_graph_length_mm is kept in the export: it is the reference
    # for checking that centerline_total_length_mm has not over-counted shared trunks.
    "root_skeleton_index",
    "root_x_mm",
    "root_y_mm",
    "root_z_mm",
    "n_centerline_paths_discarded_spurious_arm",
    "n_centerline_paths_discarded_overlap",
    "n_centerline_paths_overlap_trimmed",
    "case_id",
    "donut_loop_mode",
    "n_donut_loop_regions",
    "n_donut_loop_arms",
]

BRANCHPOINT_EXPORT_DROP_COLUMNS = [
    "case_id",
    "component_id",
    "branchpoint_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "degree",
]

def compute_vessel_summary_row(case_id: str, label: int, vessel_info: VesselInfo, res: dict) -> dict:
    """Flatten one path result's metrics + vessel identity into a single export-ready summary row."""
    return {
        "case_id": case_id,
        "label": label,
        "component_id": res.get("component_id", 1),
        "path_id": res.get("path_id", ""),
        "tree_label": res.get("tree_label", ""),
        "tree_path": res.get("tree_path", ""),
        "path_role": res.get("path_role", ""),
        "path_index": res.get("path_index", np.nan),
        "tree_mode": res.get("tree_mode", False),
        "donut_loop_mode": res.get("donut_loop_mode", False),
        "donut_loop_index": res.get("donut_loop_index", np.nan),
        "donut_arm_index": res.get("donut_arm_index", np.nan),
        "overlap_pruned_start_points": res.get("overlap_pruned_start_points", 0),
        "root_skeleton_index": res.get("root_skeleton_index", np.nan),
        "terminal_skeleton_index": res.get("terminal_skeleton_index", np.nan),
        "vmtk_seed_trim_retry_mm": res.get("vmtk_seed_trim_retry_mm", 0.0),
        "vessel_name": vessel_info.name,
        "full_name": vessel_info.full_name,
        "side": vessel_info.side,
        "pair": vessel_info.pair or "",
        "territory": vessel_info.territory,
        "flow_from": vessel_info.flow_from,
        "length_mm": res["length_mm"],
        "chord_length_mm": res["chord_length_mm"],
        "tortuosity_dm": res["tortuosity_dm"],
        "curvature_mean_1_per_mm": res["curvature_mean_1_per_mm"],
        "curvature_median_1_per_mm": res["curvature_median_1_per_mm"],
        "curvature_p95_1_per_mm": res["curvature_p95_1_per_mm"],
        "curvature_max_1_per_mm": res["curvature_max_1_per_mm"],
        "inflection_count": res["inflection_count"],
        "bend_peak_count": res["bend_peak_count"],
        "radius_mean_mm": res["radius_mean_mm"],
        "radius_std_mm": res["radius_std_mm"],
        "radius_cv": res["radius_cv"],
        "radius_min_mm": res["radius_min_mm"],
        "radius_min_stenotic_mm": res.get("radius_min_stenotic_mm", np.nan),
        "radius_p05_mm": res["radius_p05_mm"],
        "radius_p50_mm": res["radius_p50_mm"],
        "radius_p95_mm": res["radius_p95_mm"],
        "taper_slope_mm_per_mm": res["taper_slope_mm_per_mm"],
        "radius_ref_mm": res["radius_ref_mm"],
        "stenosis_percent_max": res["stenosis_percent_max"],
        "degree_of_stenosis_pct": res["degree_of_stenosis_pct"],
        "stenosis_length_total_mm": res["stenosis_length_total_mm"],
        "stenosis_segments_n": res["stenosis_segments_n"],
        "stenosis_segments_point_idx": res["stenosis_segments_point_idx"],
        "stenosis_segments_detail_json": res.get("stenosis_segments_detail_json", "[]"),
        "enlargement_radius_ref_mm": res.get("enlargement_radius_ref_mm", np.nan),
        "enlargement_radius_max_mm": res.get("enlargement_radius_max_mm", np.nan),
        "radius_max_enlarged_mm": res.get("radius_max_enlarged_mm", np.nan),
        "enlargement_percent_max": res.get("enlargement_percent_max", np.nan),
        "enlargement_length_total_mm": res.get("enlargement_length_total_mm", np.nan),
        "enlargement_segments_n": res.get("enlargement_segments_n", 0),
        "enlargement_segments_point_idx": res.get("enlargement_segments_point_idx", "[]"),
        "enlargement_segments_detail_json": res.get("enlargement_segments_detail_json", "[]"),
    }


def path_summary_export_dataframe(path_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal-only columns from the path summary table before writing to Excel."""
    if path_summary_df.empty:
        return path_summary_df
    return path_summary_df.drop(columns=PATH_SUMMARY_EXPORT_DROP_COLUMNS, errors="ignore")


def tree_summary_export_dataframe(tree_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal-only columns from the tree summary table before writing to Excel."""
    if tree_summary_df.empty:
        return tree_summary_df
    return tree_summary_df.drop(columns=TREE_SUMMARY_EXPORT_DROP_COLUMNS, errors="ignore")


def branchpoint_export_dataframe(branchpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal-only columns from the branchpoint table before writing to Excel."""
    if branchpoints_df.empty:
        return branchpoints_df
    return branchpoints_df.drop(columns=BRANCHPOINT_EXPORT_DROP_COLUMNS, errors="ignore")


def build_tree_region_points_dataframe(
    case_id: str,
    label: int,
    vessel_info: VesselInfo,
    component_id: int,
    tree_region_id: str,
    tree_region: str,
    points: np.ndarray,
    source_path_id: str,
    source_point_indices: np.ndarray,
    radius_mm: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    torsion: Optional[np.ndarray] = None,
    source_path_id_2: str = "",
    source_point_indices_2: Optional[np.ndarray] = None,
    tree_metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """Build the per-point export table for one tree-region region (points + geometry + tree metadata)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    s = cumulative_s(points)
    if radius_mm is None:
        radius_mm = np.full(n, np.nan)
    if curvature is None:
        curvature = np.full(n, np.nan)
    if torsion is None:
        torsion = np.full(n, np.nan)
    if source_point_indices_2 is None:
        source_point_indices_2 = np.full(n, -1, dtype=int)
    tree_metadata = tree_metadata or tree_region_metadata(tree_region)
    return pd.DataFrame({
        "case_id": case_id,
        "label": int(label),
        "component_id": int(component_id),
        "tree_region_id": tree_region_id,
        "tree_region": tree_region,
        "tree_region_role": tree_metadata["tree_region_role"],
        "tree_region_role_code": tree_metadata["tree_region_role_code"],
        "tree_branch_depth": tree_metadata["tree_branch_depth"],
        "tree_branch_index": tree_metadata["tree_branch_index"],
        "tree_branch_path": tree_metadata["tree_branch_path"],
        "parent_tree_branch_path": tree_metadata["parent_tree_branch_path"],
        "source_path_id": source_path_id,
        "source_path_id_2": source_path_id_2,
        "source_point_index": source_point_indices.astype(int),
        "source_point_index_2": source_point_indices_2.astype(int),
        "vessel_name": vessel_info.name,
        "full_name": vessel_info.full_name,
        "side": vessel_info.side,
        "pair": vessel_info.pair or "",
        "territory": vessel_info.territory,
        "point_index": np.arange(n, dtype=int),
        "x_mm": points[:, 0] if n else np.array([], dtype=float),
        "y_mm": points[:, 1] if n else np.array([], dtype=float),
        "z_mm": points[:, 2] if n else np.array([], dtype=float),
        "s_mm": s,
        "radius_mm": radius_mm,
        "diameter_mm": 2.0 * radius_mm,
        "curvature_1_per_mm": curvature,
        "torsion_1_per_mm": torsion,
        "length_mm": float(arc_length(points)),
    })


def tree_region_summary_from_points(
    label: int,
    component_id: int,
    vessel_info: VesselInfo,
    tree_region_id: str,
    tree_region: str,
    points: np.ndarray,
    source_path_id: str,
    source_path_id_2: str = "",
    tree_metadata: Optional[dict] = None,
) -> dict:
    """Summarize one tree-region's geometry (length, chord, tortuosity) as a single export row."""
    points = np.asarray(points, dtype=float)
    tree_metadata = tree_metadata or tree_region_metadata(tree_region)
    return {
        "label": int(label),
        "component_id": int(component_id),
        "tree_region_id": tree_region_id,
        "tree_region": tree_region,
        "tree_region_role": tree_metadata["tree_region_role"],
        "tree_region_role_code": tree_metadata["tree_region_role_code"],
        "tree_branch_depth": tree_metadata["tree_branch_depth"],
        "tree_branch_index": tree_metadata["tree_branch_index"],
        "tree_branch_path": tree_metadata["tree_branch_path"],
        "parent_tree_branch_path": tree_metadata["parent_tree_branch_path"],
        "source_path_id": source_path_id,
        "source_path_id_2": source_path_id_2,
        "vessel_name": vessel_info.name,
        "full_name": vessel_info.full_name,
        "side": vessel_info.side,
        "pair": vessel_info.pair or "",
        "territory": vessel_info.territory,
        "n_points": int(len(points)),
        "length_mm": float(arc_length(points)),
        "chord_length_mm": float(chord_length(points)),
        "tortuosity_dm": float(tortuosity_dm(points)),
    }


def recursive_tree_segment_summary_rows(
    case_id: str,
    label: int,
    component_id: int,
    vessel_info: VesselInfo,
    segments: List[dict],
) -> List[dict]:
    """Flatten recursive tree segments into export rows (identity, anatomic naming, geometry)."""
    rows = []
    for seg in segments:
        rows.append({
            "case_id": case_id,
            "label": int(label),
            "component_id": int(component_id),
            "vessel_name": vessel_info.name,
            "full_name": vessel_info.full_name,
            "side": vessel_info.side,
            "pair": vessel_info.pair or "",
            "territory": vessel_info.territory,
            "tree_segment_id": int(seg["segment_id"]),
            "tree_segment_label": seg["region_label"],
            "tree_segment_name": seg["segment_name"],
            "anatomic_segment_name": seg.get("anatomic_segment_name", ""),
            "anatomic_segment_path": seg.get("anatomic_segment_path", ""),
            "anatomic_parent_path": seg.get("anatomic_parent_path", ""),
            "anatomic_generation": int(seg.get("anatomic_generation", 0)),
            "anatomic_branch_suffix": seg.get("anatomic_branch_suffix", ""),
            "tree_region_role": seg.get("tree_region_role", "branch"),
            "tree_region_role_code": int(seg.get("tree_region_role_code", TREE_REGION_ROLE_CODES["branch"])),
            "tree_branch_depth": int(seg.get("tree_branch_depth", seg.get("depth", 0))),
            "tree_branch_index": int(seg.get("tree_branch_index", 0)),
            "tree_branch_path": seg.get("tree_branch_path", ""),
            "parent_tree_branch_path": seg.get("parent_tree_branch_path", ""),
            "depth": int(seg["depth"]),
            "start_skeleton_index": int(seg["start_node"]),
            "end_skeleton_index": int(seg["end_node"]),
            "start_junction": seg.get("start_junction", ""),
            "end_junction": seg.get("end_junction", ""),
            "is_terminal": bool(seg["is_terminal"]),
            "n_skeleton_points": int(len(seg["node_indices"])),
            "length_mm": float(seg["length_mm"]),
            "node_indices_json": json.dumps(seg["node_indices"]),
        })
    return rows


def build_vessel_points_dataframe(case_id: str, label: int, vessel_info: VesselInfo, res: dict) -> pd.DataFrame:
    """Per-point export table for one path: geometry, radius, curvature/torsion, and stenosis/enlargement flags."""
    is_stenotic = np.asarray(res["is_stenotic"], dtype=int)
    stenosis_percent_point = np.asarray(res["stenosis_percent_point"], dtype=float)
    stenosis_raw_percent_point = np.asarray(
        res.get("stenosis_raw_percent_point", np.full(len(res["s_mm"]), np.nan)),
        dtype=float,
    )
    stenosis_core_candidate_point = np.asarray(
        res.get("stenosis_core_candidate_point", np.zeros(len(res["s_mm"]), dtype=float)),
        dtype=float,
    )
    stenosis_support_candidate_point = np.asarray(
        res.get("stenosis_support_candidate_point", np.zeros(len(res["s_mm"]), dtype=float)),
        dtype=float,
    )
    is_enlarged = np.asarray(res.get("is_enlarged", np.zeros(len(res["s_mm"]), dtype=int)), dtype=int)
    enlargement_percent_point = np.asarray(
        res.get("enlargement_percent_point", np.full(len(res["s_mm"]), np.nan)),
        dtype=float,
    )
    vessel_df = pd.DataFrame({
        "point_index": np.arange(len(res["s_mm"]), dtype=int),
        "x_mm": res["x_mm"], "y_mm": res["y_mm"], "z_mm": res["z_mm"], "s_mm": res["s_mm"],
        "radius_mm": res["radius_mm"], "diameter_mm": res["diameter_mm"],
        "maximum_inscribed_sphere_radius_mm": res.get("maximum_inscribed_sphere_radius_mm", np.full(len(res["s_mm"]), np.nan)),
        "stenosis_detection_radius_mm": res.get("stenosis_detection_radius_mm", res["radius_mm"]),
        "curvature_1_per_mm": res["curvature_1_per_mm"], "torsion_1_per_mm": res["torsion_1_per_mm"],
        "stenosis_raw_percent_point": stenosis_raw_percent_point,
        "stenosis_core_candidate_point": stenosis_core_candidate_point,
        "stenosis_support_candidate_point": stenosis_support_candidate_point,
        "stenosis_percent_point": stenosis_percent_point,
        "is_stenotic": is_stenotic,
        "enlargement_percent_point": enlargement_percent_point,
        "is_enlarged": is_enlarged,
    })
    return vessel_df


def compute_lr_asymmetry(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel-pair left/right metrics, asymmetry index (``(L-R)/mean``), and L/R ratio.

    Groups the path summary by ``pair``, aggregates each side's paths (length-
    weighted mean for most metrics; sum/min/max for length/radius-min/degree
    metrics), and reports paired L/R values plus derived asymmetry columns.
    """
    if summary_df.empty or "pair" not in summary_df.columns:
        return pd.DataFrame()
    metrics = [
        "length_mm",
        "tortuosity_dm",
        "radius_mean_mm",
        "radius_p50_mm",
        "radius_min_mm",
        "curvature_mean_1_per_mm",
        "curvature_median_1_per_mm",
        "stenosis_percent_max",
        "enlargement_percent_max",
    ]

    def _join_unique(values) -> str:
        """Comma-join the distinct non-null string values in *values*, preserving first-seen order."""
        clean = []
        for value in values:
            if pd.isna(value):
                continue
            text = str(value)
            if text and text not in clean:
                clean.append(text)
        return ", ".join(clean)

    def _first_int(values) -> float:
        """First numeric value in *values* as an int, or NaN if none are numeric."""
        vals = pd.to_numeric(values, errors="coerce").dropna()
        return int(vals.iloc[0]) if len(vals) else np.nan

    def _weighted_mean(group: pd.DataFrame, metric: str) -> float:
        """Length-weighted mean of *metric* over *group* (unweighted mean if lengths are unusable)."""
        vals = pd.to_numeric(group.get(metric), errors="coerce")
        valid = vals.notna()
        if not valid.any():
            return np.nan
        weights = pd.to_numeric(group.get("length_mm"), errors="coerce").where(valid)
        good_weights = weights.notna() & np.isfinite(weights) & (weights > 0)
        if good_weights.any():
            return float(np.average(vals[good_weights], weights=weights[good_weights]))
        return float(vals[valid].mean())

    def _aggregate_metric(group: pd.DataFrame, metric: str) -> float:
        """Combine a metric across a side's paths with the appropriate reducer (sum/min/max/weighted-mean)."""
        if metric not in group.columns:
            return np.nan
        vals = pd.to_numeric(group[metric], errors="coerce").dropna()
        if vals.empty:
            return np.nan
        if metric == "length_mm":
            return float(vals.sum())
        if metric == "radius_min_mm":
            return float(vals.min())
        if metric in {"stenosis_percent_max", "enlargement_percent_max"}:
            return float(vals.max())
        return _weighted_mean(group, metric)

    def _aggregate_side(group: pd.DataFrame) -> dict:
        """Aggregate one side's (L or R) paths for a vessel pair into a single metric dict."""
        side = {
            "label": _first_int(group.get("label", pd.Series(dtype=float))),
            "name": _join_unique(group.get("vessel_name", pd.Series(dtype=object))),
            "n_paths": int(len(group)),
        }
        for metric in metrics:
            side[metric] = _aggregate_metric(group, metric)
        return side

    rows = []
    for pair_key, group in summary_df.groupby("pair"):
        if pd.isna(pair_key) or not str(pair_key):
            continue
        g_l = group[group["side"] == "L"]
        g_r = group[group["side"] == "R"]
        if g_l.empty or g_r.empty:
            continue
        left = _aggregate_side(g_l)
        right = _aggregate_side(g_r)
        row = {
            "pair": pair_key,
            "label_L": left["label"],
            "label_R": right["label"],
            "name_L": left["name"],
            "name_R": right["name"],
            "n_paths_L": left["n_paths"],
            "n_paths_R": right["n_paths"],
        }
        for m in metrics:
            lv = left[m]
            rv = right[m]
            denom = (lv + rv) / 2.0
            row[f"{m}_L"] = lv; row[f"{m}_R"] = rv
            row[f"{m}_AI"] = (lv - rv) / denom if np.isfinite(denom) and abs(denom) > 1e-12 else np.nan
            row[f"{m}_ratio_L_over_R"] = lv / rv if np.isfinite(rv) and abs(rv) > 1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hemispheric_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-hemisphere (L/R) rollup: total length, length-weighted tortuosity/radius, and lesion counts.

    One row per side, aggregating across every vessel/path on that side rather
    than pairing left against right (see :func:`compute_lr_asymmetry` for that).
    """
    if summary_df.empty:
        return pd.DataFrame()
    if "side" not in summary_df.columns:
        return pd.DataFrame()
    df = summary_df[summary_df["side"].isin(["L", "R"])].copy()
    if df.empty:
        return pd.DataFrame()

    def _numeric(group: pd.DataFrame, column: str) -> pd.Series:
        """Coerce *column* of *group* to numeric, dropping non-numeric/missing entries."""
        if column not in group.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(group[column], errors="coerce").dropna()

    def _length_weights(group: pd.DataFrame, valid_index: pd.Index) -> pd.Series:
        """Per-row path-length weights aligned to *valid_index* (falls back to equal weighting)."""
        if "length_mm" not in group.columns:
            return pd.Series(np.ones(len(valid_index)), index=valid_index, dtype=float)
        weights = pd.to_numeric(group.loc[valid_index, "length_mm"], errors="coerce")
        weights = weights.where(np.isfinite(weights) & (weights > 0), np.nan)
        if weights.notna().any():
            return weights
        return pd.Series(np.ones(len(valid_index)), index=valid_index, dtype=float)

    def _weighted_mean(group: pd.DataFrame, column: str) -> float:
        """Length-weighted mean of *column* over *group*."""
        values = _numeric(group, column)
        if values.empty:
            return np.nan
        weights = _length_weights(group, values.index)
        valid = weights.notna()
        if valid.any():
            return float(np.average(values.loc[valid], weights=weights.loc[valid]))
        return float(values.mean())

    def _weighted_percentile(group: pd.DataFrame, column: str, percentile: float) -> float:
        """Length-weighted percentile of *column* over *group* (falls back to an unweighted percentile)."""
        values = _numeric(group, column)
        if values.empty:
            return np.nan
        weights = _length_weights(group, values.index).fillna(0.0)
        order = np.argsort(values.to_numpy(dtype=float))
        sorted_values = values.to_numpy(dtype=float)[order]
        sorted_weights = weights.to_numpy(dtype=float)[order]
        total_weight = float(np.sum(sorted_weights))
        if total_weight <= 0:
            return float(np.percentile(sorted_values, percentile))
        cumulative = np.cumsum(sorted_weights)
        cutoff = (percentile / 100.0) * total_weight
        idx = int(np.searchsorted(cumulative, cutoff, side="left"))
        idx = min(max(idx, 0), len(sorted_values) - 1)
        return float(sorted_values[idx])

    def _sum(group: pd.DataFrame, column: str) -> float:
        """Sum of *column* over *group* (0.0 if empty)."""
        values = _numeric(group, column)
        return float(values.sum()) if len(values) else 0.0

    def _max(group: pd.DataFrame, column: str) -> float:
        """Max of *column* over *group* (NaN if empty)."""
        values = _numeric(group, column)
        return float(values.max()) if len(values) else np.nan

    rows = []
    for side, group in df.groupby("side"):
        stenosis_degrees = _numeric(group, "stenosis_percent_max")
        stenosis_degrees = stenosis_degrees[stenosis_degrees > 0]
        enlargement_degrees = _numeric(group, "enlargement_percent_max")
        enlargement_degrees = enlargement_degrees[enlargement_degrees > 0]
        row = {
            "side": side,
            "n_paths": int(len(group)),
            "vessel_names": ", ".join(sorted(set(str(v) for v in group.get("vessel_name", []) if pd.notna(v) and str(v)))),
            "length_total_mm": _sum(group, "length_mm"),
            "tortuosity_mean": _weighted_mean(group, "tortuosity_dm"),
            "tortuosity_p90": _weighted_percentile(group, "tortuosity_dm", 90),
            "radius_p10_mm": _weighted_percentile(group, "radius_p50_mm", 10),
            "radius_p50_mm": _weighted_percentile(group, "radius_p50_mm", 50),
            "radius_p90_mm": _weighted_percentile(group, "radius_p50_mm", 90),
            "stenosis_segments_n": int(_sum(group, "stenosis_segments_n")),
            "stenosis_degree_max_pct": _max(group, "stenosis_percent_max"),
            "stenosis_degree_p90_pct": float(np.percentile(stenosis_degrees, 90)) if len(stenosis_degrees) else np.nan,
            "stenosis_length_total_mm": _sum(group, "stenosis_length_total_mm"),
            "enlargement_segments_n": int(_sum(group, "enlargement_segments_n")),
            "enlargement_degree_max_pct": _max(group, "enlargement_percent_max"),
            "enlargement_degree_p90_pct": float(np.percentile(enlargement_degrees, 90)) if len(enlargement_degrees) else np.nan,
            "enlargement_length_total_mm": _sum(group, "enlargement_length_total_mm"),
        }
        rows.append(row)
    return pd.DataFrame(rows)



def compute_volumetry_summary(
    tree_summary_df: pd.DataFrame,
    input_volumetry: dict | None = None,
) -> pd.DataFrame:
    """Per-label volumetry rolled up from the per-component tree summaries.

    One row per label (components summed), plus a ``TOTAL`` row. Volumes come
    from voxel counts × voxel volume; ``mesh_volume_mm3`` / ``surface_area_mm2``
    come from the reconstructed surface and are ``NaN`` where VTK could not
    evaluate a closed mesh.

    ``input_volumetry`` (label -> ``{n_voxels, volume_mm3, n_components}``, as
    built by :func:`nvitk.measure.morpho.run_case.label_input_volumetry`) adds
    ``input_*`` columns describing the mask the pipeline was handed. The
    measured columns are always lower: they exclude components too small to
    skeletonize, and the mask itself has been Taubin-smoothed upstream.
    """
    if tree_summary_df is None or tree_summary_df.empty or "volume_mm3" not in tree_summary_df.columns:
        return pd.DataFrame()

    df = tree_summary_df.copy()

    def _first_str(values) -> str:
        for value in values:
            if pd.notna(value) and str(value):
                return str(value)
        return ""

    rows = []
    for label, group in df.groupby("label", dropna=False, sort=True):
        rows.append({
            "label": int(label),
            "vessel_name": _first_str(group.get("vessel_name", [])),
            "full_name": _first_str(group.get("full_name", [])),
            "side": _first_str(group.get("side", [])),
            "territory": _first_str(group.get("territory", [])),
            "species": _first_str(group.get("species", [])),
            "n_components": int(len(group)),
            "n_voxels": int(np.nansum(pd.to_numeric(group.get("n_voxels"), errors="coerce"))),
            "volume_mm3": float(np.nansum(pd.to_numeric(group.get("volume_mm3"), errors="coerce"))),
            "volume_ul": float(np.nansum(pd.to_numeric(group.get("volume_ul"), errors="coerce"))),
            "volume_cc": float(np.nansum(pd.to_numeric(group.get("volume_cc"), errors="coerce"))),
            "mesh_volume_mm3": float(np.nansum(pd.to_numeric(group.get("mesh_volume_mm3"), errors="coerce"))),
            "surface_area_mm2": float(np.nansum(pd.to_numeric(group.get("surface_area_mm2"), errors="coerce"))),
            "skeleton_length_mm": float(
                np.nansum(pd.to_numeric(group.get("unique_skeleton_graph_length_mm"), errors="coerce"))
            ),
            "equivalent_radius_mm": float(
                np.nanmean(pd.to_numeric(group.get("equivalent_radius_mm"), errors="coerce"))
            ),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if input_volumetry:
        out["input_n_components"] = [
            int(input_volumetry.get(int(lab), {}).get("n_components", 0)) for lab in out["label"]
        ]
        out["input_n_voxels"] = [
            int(input_volumetry.get(int(lab), {}).get("n_voxels", 0)) for lab in out["label"]
        ]
        out["input_volume_mm3"] = [
            float(input_volumetry.get(int(lab), {}).get("volume_mm3", np.nan)) for lab in out["label"]
        ]
        out["measured_volume_fraction"] = out["volume_mm3"] / out["input_volume_mm3"].replace(0, np.nan)

    total = {"label": "TOTAL", "vessel_name": "", "full_name": "", "side": "", "territory": ""}
    total["species"] = _first_str(out["species"])
    for column in (
        "n_components", "n_voxels", "volume_mm3", "volume_ul", "volume_cc",
        "mesh_volume_mm3", "surface_area_mm2", "skeleton_length_mm",
        "input_n_components", "input_n_voxels", "input_volume_mm3",
    ):
        if column in out.columns:
            total[column] = out[column].sum()
    total["equivalent_radius_mm"] = np.nan
    if "input_volume_mm3" in out.columns and total.get("input_volume_mm3"):
        total["measured_volume_fraction"] = total["volume_mm3"] / total["input_volume_mm3"]
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)
