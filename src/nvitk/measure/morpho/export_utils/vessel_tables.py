"""Compact morphometrics tables: one row per vessel, per segment, per lesion, and per subject.

``vessel_summary.csv`` (sheet ``Vessel_Summary``) is the primary output: one row per labelled
vessel (all its components and segments combined), with

- presence / components / segments / bifurcations;
- size: non-overlapping centerline length, main-path length / chord, volume, surface area;
- calibre (MIS and cross-section): mean / median / std / p05 / p95 / min / max diameter, proximal /
  distal (``FEATURE_END_WINDOW_MM``) and last 5 / 10 mm diameters, min/max ratio, taper,
  bifurcation-blanked mean / min;
- shape: arc/chord tortuosity (DM), tortuosity of the first 10 / last 10 / last 20 mm, bending
  length, smoothed curvature mean / p95 / max, total curvature ∫κ ds, curvature energy ∫κ² ds,
  sum of angles, torsion, turns, chord polar / azimuth angles, accumulated polar change;
- topology: branching angles and the Murray area ratio of its bifurcations;
- lesions: stenosis / enlargement counts, maxima, total length, and the high-confidence counts.

The main path is the longest measured root→terminal path of the vessel's largest component (its
proximal-to-distal course); lengths that describe the whole vessel tree use the non-overlapping
segments. See :mod:`nvitk.measure.morpho.vessel_features` for the descriptor definitions.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.measure import morphometrics_config as _cfg
from nvitk.measure.morpho.vessel_features import lesion_rows, path_features

_IDENTITY = ("case_id", "label", "vessel_name", "full_name", "side", "pair", "territory")


def _identity(case_id: str, label: int, vessel_info: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "label": int(label),
        "vessel_name": getattr(vessel_info, "name", f"label_{label}"),
        "full_name": getattr(vessel_info, "full_name", ""),
        "side": getattr(vessel_info, "side", ""),
        "pair": getattr(vessel_info, "pair", "") or "",
        "territory": getattr(vessel_info, "territory", ""),
    }


def _points(res: Mapping[str, Any]) -> np.ndarray:
    return np.column_stack([res["x_mm"], res["y_mm"], res["z_mm"]]).astype(float)


def _radii(res: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = len(res["x_mm"])
    mis = np.asarray(res.get("maximum_inscribed_sphere_radius_mm", np.full(n, np.nan)), dtype=float)
    cs = np.asarray(res.get("radius_mm", np.full(n, np.nan)), dtype=float)
    return mis, cs


def component_tables(
    *,
    case_id: str,
    label: int,
    component_id: int,
    vessel_info: Any,
    path_results: Sequence[Mapping[str, Any]],
    split_results: Sequence[Mapping[str, Any]],
    branch_df: pd.DataFrame,
    tree_summary: Mapping[str, Any],
    axes: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per-component rows (run inside the worker, where the per-point arrays live)."""
    junctions = (
        branch_df[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
        if branch_df is not None and not branch_df.empty and {"x_mm", "y_mm", "z_mm"} <= set(branch_df.columns)
        else np.empty((0, 3))
    )
    ident = _identity(case_id, label, vessel_info)
    contexts = [r for r in path_results if len(r.get("x_mm", [])) >= 2] or list(split_results)

    seg_list = [seg for seg in split_results if len(seg.get("x_mm", [])) >= 2]
    lesions = [
        {**ident, "component_id": int(component_id), **row}
        for row in lesion_rows(contexts, seg_list, junctions=junctions)
    ]
    segments = []
    for seg in seg_list:
        mis, cs = _radii(seg)
        name = str(seg.get("anatomic_segment_name") or seg.get("tree_label") or "")
        segments.append({
            **ident,
            "component_id": int(component_id),
            "segment": name,
            "path_id": str(seg.get("path_id", "")),
            "is_terminal": bool(seg.get("is_terminal", False)),
            "geometry_source": str(seg.get("geometry_source", "centerline")),
            **path_features(_points(seg), mis, cs, junctions=junctions, axes=axes),
            "n_stenoses": sum(1 for r in lesions if r["kind"] == "stenosis" and r["segment"] == name),
            "n_enlargements": sum(1 for r in lesions if r["kind"] == "enlargement" and r["segment"] == name),
        })

    main = max(contexts, key=lambda r: float(r.get("length_mm", 0.0) or 0.0), default=None)
    vessel: dict[str, Any] = {**ident, "component_id": int(component_id)}
    if main is not None:
        mis, cs = _radii(main)
        main_feats = path_features(_points(main), mis, cs, junctions=junctions, axes=axes)
        vessel.update({f"main_{k}" if k in ("length_mm", "chord_length_mm", "n_points") else k: v
                       for k, v in main_feats.items()})
    vessel["centerline_length_mm"] = float(sum(float(s.get("length_mm", 0.0) or 0.0) for s in segments))
    vessel["n_segments"] = len(segments)
    vessel["n_bifurcations"] = int(len(junctions))
    if branch_df is not None and not branch_df.empty:
        if "daughter_pair_angle_min_deg" in branch_df:
            vessel["branching_angle_min_deg"] = float(np.nanmin(branch_df["daughter_pair_angle_min_deg"]))
            vessel["branching_angle_max_deg"] = float(np.nanmax(branch_df["daughter_pair_angle_max_deg"]))
        if "daughter_area_ratio_sum_r2_over_parent_r2" in branch_df:
            vessel["murray_area_ratio_median"] = float(np.nanmedian(branch_df["daughter_area_ratio_sum_r2_over_parent_r2"]))
    for key in ("volume_mm3", "surface_area_mm2", "mesh_volume_mm3"):
        if key in tree_summary:
            vessel[key] = tree_summary[key]
    return {"segments": segments, "lesions": lesions, "vessels": [vessel]}


def _lesion_summary(lesions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("stenosis", "enlargement"):
        sub = lesions[lesions["kind"] == kind] if not lesions.empty else lesions
        out[f"n_{kind}"] = int(len(sub))
        out[f"n_{kind}_high_confidence"] = int((sub["confidence_class"] == "high").sum()) if len(sub) else 0
        out[f"{kind}_taper_percent_max"] = float(sub["taper_percent"].max()) if len(sub) else 0.0
        out[f"{kind}_diameter_percent_max"] = float(sub["diameter_percent"].max()) if len(sub) else np.nan
        out[f"{kind}_length_total_mm"] = float(sub["length_mm"].sum()) if len(sub) else 0.0
        if len(sub):
            worst = sub.sort_values("taper_percent", ascending=False).iloc[0]
            out[f"{kind}_worst_segment"] = worst["segment"]
            out[f"{kind}_worst_from_origin_mm"] = worst["start_s_from_vessel_origin_mm"]
            out[f"{kind}_worst_confidence"] = worst["confidence"]
    return out


def vessel_summary(
    vessel_rows: Iterable[Mapping[str, Any]],
    lesion_rows_: Iterable[Mapping[str, Any]],
    mapping: Mapping[int, Any] | None = None,
) -> pd.DataFrame:
    """One row per label: the largest component's main-path features, totals over all components,
    lesion summary. Labels of the topology that are absent get a ``present = False`` row."""
    vdf = pd.DataFrame(list(vessel_rows))
    ldf = pd.DataFrame(list(lesion_rows_))
    rows = []
    if not vdf.empty:
        for label, grp in vdf.groupby("label", sort=True):
            grp = grp.sort_values("centerline_length_mm", ascending=False)
            row = dict(grp.iloc[0])
            row.pop("component_id", None)
            row["present"] = True
            row["n_components"] = int(len(grp))
            for key in ("centerline_length_mm", "n_segments", "n_bifurcations", "volume_mm3", "surface_area_mm2"):
                if key in grp:
                    row[key] = float(np.nansum(grp[key].to_numpy(dtype=float)))
            sub = ldf[ldf["label"] == label] if not ldf.empty else ldf
            row.update(_lesion_summary(sub))
            rows.append(row)
    present = {int(r["label"]) for r in rows}
    for label, info in sorted((mapping or {}).items()):
        if int(label) not in present:
            rows.append({**_identity(rows[0]["case_id"] if rows else "", int(label), info), "present": False})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lead = [c for c in (*_IDENTITY, "present", "n_components") if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]].sort_values("label").reset_index(drop=True)


#: Circle-of-Willis variant segments of the eICAB topology (vessel_name → variant name).
_COW_VARIANT_VESSELS = {
    "LACA": "left_A1", "RACA": "right_A1", "ACOMM": "AComm",
    "LPCOMM": "left_PComm", "RPCOMM": "right_PComm",
    "LPCA_P1": "left_P1", "RPCA_P1": "right_P1",
}


def subject_summary(vessels: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: CoW variants (absent / hypoplastic / present), completeness, left/right
    asymmetry indices (2|L−R|/(L+R)) of paired vessels, totals and lesion counts."""
    if vessels.empty:
        return pd.DataFrame()
    row: dict[str, Any] = {"case_id": vessels["case_id"].dropna().iloc[0] if "case_id" in vessels else ""}
    by_name = {str(r["vessel_name"]): r for _, r in vessels.iterrows()}
    thr = float(_cfg.HYPOPLASIA_DIAMETER_MM)
    complete = 0
    for name, variant in _COW_VARIANT_VESSELS.items():
        r = by_name.get(name)
        if r is None or not bool(r.get("present", False)):
            state = "absent"
        else:
            d = float(r.get("diameter_median_mm", np.nan))
            state = "hypoplastic" if np.isfinite(d) and d < thr else "present"
            complete += state == "present"
        row[f"{variant}_status"] = state
        row[f"{variant}_diameter_median_mm"] = float(r.get("diameter_median_mm", np.nan)) if r is not None else np.nan
    row["cow_variant_segments_present"] = int(complete)
    row["cow_complete"] = bool(complete == len(_COW_VARIANT_VESSELS))
    present = vessels[vessels.get("present", True) == True]  # noqa: E712
    row["n_vessels_present"] = int(len(present))
    row["total_centerline_length_mm"] = float(np.nansum(present.get("centerline_length_mm", pd.Series(dtype=float))))
    for kind in ("stenosis", "enlargement"):
        for col in (f"n_{kind}", f"n_{kind}_high_confidence"):
            row[f"total_{col}"] = int(np.nansum(present.get(col, pd.Series(dtype=float))))
    for metric in ("main_length_mm", "diameter_median_mm", "tortuosity_dm", "curvature_mean_1_per_mm"):
        for pair, grp in present.groupby("pair"):
            if not pair or len(grp) != 2 or metric not in grp:
                continue
            sides = {str(s): float(v) for s, v in zip(grp["side"], grp[metric])}
            left, right = sides.get("L", np.nan), sides.get("R", np.nan)
            if np.isfinite(left) and np.isfinite(right) and (left + right) > 0:
                row[f"{pair}_{metric}_asymmetry_index"] = float(2.0 * abs(left - right) / (left + right))
    return pd.DataFrame([row])


__all__ = ["component_tables", "subject_summary", "vessel_summary"]
