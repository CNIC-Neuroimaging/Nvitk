"""Read a morphometrics run's outputs into flat rows for the Slicer results table.

Volumetry is read from ``volumetry.csv`` when present (written next to the
workbook precisely so no openpyxl round-trip is needed) and from the
``06_Volumetry`` sheet otherwise.
"""

from __future__ import annotations

import os
from typing import Any, Optional

EXCEL_NAME = "case_metrics_donut_tree.xlsx"
VOLUMETRY_CSV = "volumetry.csv"

VOLUMETRY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "Label"),
    ("vessel_name", "Vessel"),
    ("n_components", "CCs"),
    ("n_voxels", "Voxels"),
    ("volume_mm3", "Volume (mm³)"),
    ("volume_ul", "Volume (µL)"),
    ("surface_area_mm2", "Surface (mm²)"),
    ("skeleton_length_mm", "Skeleton (mm)"),
    ("equivalent_radius_mm", "Equiv. radius (mm)"),
    ("input_volume_mm3", "Input vol (mm³)"),
)

PATH_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "Label"),
    ("vessel_name", "Vessel"),
    ("tree_label", "Segment"),
    ("length_mm", "Length (mm)"),
    ("radius_p50_mm", "Radius p50 (mm)"),
    ("radius_mean_mm", "Radius mean (mm)"),
    ("tortuosity_dm", "Tortuosity"),
    # Human-calibrated detectors; shown for completeness, not trustworthy for mouse.
    ("stenosis_percent_max", "Stenosis max (%)"),
    ("enlargement_percent_max", "Enlargement max (%)"),
)

#: Per-vessel view: volumetry joined to length-weighted centerline metrics.
VESSEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("label", "Label"),
    ("vessel_name", "Vessel"),
    ("n_voxels", "Voxels"),
    ("volume_mm3", "Volume (mm³)"),
    ("surface_area_mm2", "Surface (mm²)"),
    ("length_mm", "Length (mm)"),
    ("n_segments", "Segments"),
    ("radius_mean_mm", "Radius mean (mm)"),
    ("equivalent_radius_mm", "Equiv. radius (mm)"),
    ("tortuosity_dm", "Tortuosity"),
)


def _read_frame(case_dir: str, sheet: str, csv_name: Optional[str] = None):
    """Load one result table as a DataFrame, preferring a sidecar CSV."""
    import pandas as pd

    if csv_name:
        csv_path = os.path.join(case_dir, csv_name)
        if os.path.isfile(csv_path):
            return pd.read_csv(csv_path)

    excel_path = os.path.join(case_dir, EXCEL_NAME)
    if not os.path.isfile(excel_path):
        return None
    try:
        return pd.read_excel(excel_path, sheet_name=sheet)
    except Exception:
        return None


def _format(value: Any) -> str:
    """Render one cell: blank for missing, 4 significant decimals for floats."""
    import math

    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.4g}"
    return str(value)


def _table(frame, columns) -> tuple[list[str], list[list[str]]]:
    if frame is None or frame.empty:
        return [], []
    present = [(key, header) for key, header in columns if key in frame.columns]
    headers = [header for _key, header in present]
    rows = [[_format(row.get(key)) for key, _header in present] for _idx, row in frame.iterrows()]
    return headers, rows


def volumetry_table(case_dir: str) -> tuple[list[str], list[list[str]]]:
    """``(headers, rows)`` for the per-label volumetry, including the TOTAL row."""
    return _table(_read_frame(case_dir, "06_Volumetry", VOLUMETRY_CSV), VOLUMETRY_COLUMNS)


def path_summary_table(case_dir: str, limit: int = 200) -> tuple[list[str], list[list[str]]]:
    """``(headers, rows)`` for the per-segment morphometrics, longest first.

    ``00_Path_Summary`` holds non-overlapping segments — each piece of vessel
    appears once, so summing ``length_mm`` gives the real tree length.
    """
    frame = _read_frame(case_dir, "00_Path_Summary")
    if frame is not None and not frame.empty and "length_mm" in frame.columns:
        frame = frame.sort_values("length_mm", ascending=False).head(int(limit))
    return _table(frame, PATH_COLUMNS)


def vessel_table(case_dir: str) -> tuple[list[str], list[list[str]]]:
    """``(headers, rows)`` per vessel: mask volumetry beside centerline metrics."""
    import numpy as np
    import pandas as pd

    vol = _read_frame(case_dir, "06_Volumetry", VOLUMETRY_CSV)
    paths = _read_frame(case_dir, "00_Path_Summary")
    if vol is None or vol.empty:
        return [], []
    vol = vol[vol["label"].astype(str) != "TOTAL"].copy()

    if paths is not None and not paths.empty and "vessel_name" in paths.columns:
        grouped = paths.groupby("vessel_name", dropna=False)
        agg = []
        for name, g in grouped:
            length = pd.to_numeric(g.get("length_mm"), errors="coerce").fillna(0.0)
            w = length.to_numpy(dtype=float)
            row = {"vessel_name": name, "length_mm": float(w.sum()), "n_segments": int(len(g))}
            for col in ("radius_mean_mm", "tortuosity_dm"):
                vals = pd.to_numeric(g.get(col), errors="coerce").to_numpy(dtype=float)
                m = np.isfinite(vals) & (w > 0)
                row[col] = float(np.average(vals[m], weights=w[m])) if m.any() else float("nan")
            agg.append(row)
        vol = vol.merge(pd.DataFrame(agg), on="vessel_name", how="left")
    return _table(vol, VESSEL_COLUMNS)


def anatomy_provenance(case_dir: str) -> dict:
    """What species / orientation / scaling the run actually resolved, for the status line."""
    frame = _read_frame(case_dir, "01_Tree_Summary")
    if frame is None or frame.empty:
        return {}
    out = {}
    for key in ("species", "orientation_axcodes", "length_scale"):
        if key in frame.columns and len(frame[key]):
            value = frame[key].iloc[0]
            out[key] = value
    return out


def count_result_vtps(case_dir: str) -> tuple[int, int]:
    """``(n_centerline_vtps, n_surface_vtps)`` written by the run."""
    import glob

    n_centerlines = len(glob.glob(os.path.join(case_dir, "centerlines", "*.vtp")))
    n_surfaces = len(glob.glob(os.path.join(case_dir, "surfaces", "*.vtp")))
    return n_centerlines, n_surfaces


__all__ = [
    "EXCEL_NAME",
    "PATH_COLUMNS",
    "VESSEL_COLUMNS",
    "VOLUMETRY_COLUMNS",
    "VOLUMETRY_CSV",
    "anatomy_provenance",
    "count_result_vtps",
    "path_summary_table",
    "vessel_table",
    "volumetry_table",
]
