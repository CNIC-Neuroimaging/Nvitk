#!/usr/bin/env python3
"""Generate one vessel-radius histogram per centerline sheet in a metrics workbook.

All sheets are processed together so the x-axis range, bin edges, and y-axis
limit are shared — making histograms directly comparable across vessels.

Edit the CONFIG block below for direct single-case runs; the batch runner
patches EXCEL_PATH and OUTPUT_DIR at runtime.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG FOR DIRECT RUN (paths must be supplied at runtime; no host defaults)
# ============================================================================
EXCEL_PATH = None
OUTPUT_DIR = None   # None → <excel folder>/radius_histograms
RADIUS_COLUMN = "radius_mm"
BIN_METHOD = "fd"   # "fd", "auto", "sturges", "sqrt", or integer as text e.g. "40"
MIN_BINS = 12
MAX_BINS = 80
FIG_DPI = 180
# ============================================================================


SUMMARY_SHEET_RE = re.compile(r"^\d{2}_")


def safe_filename(name: str) -> str:
    """Sanitize *name* into a filesystem-safe filename fragment (falls back to ``\"centerline\"``)."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return name.strip("_") or "centerline"


def vessel_sheet_sort_key(sheet_name: str) -> Tuple[int, str]:
    """Sort key ordering workbook sheets by their leading label id, then name."""
    match = re.match(r"^(\d+)(?:\D|$)", sheet_name)
    if not match:
        match = re.search(r"(?:^|_)label[_-]?(\d+)(?:\D|$)", sheet_name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), sheet_name
    return 10**9, sheet_name


def is_candidate_centerline_sheet(sheet_name: str, columns: List[str], radius_column: str) -> bool:
    """True for a per-centerline data sheet (not a numbered summary sheet) that has the radius column."""
    if SUMMARY_SHEET_RE.match(sheet_name):
        return False
    return radius_column in columns


def read_radius_sheets(excel_path: str, radius_column: str) -> Dict[str, np.ndarray]:
    """Read the positive, finite radius values from every per-centerline sheet of the morphometrics workbook."""
    xls = pd.ExcelFile(excel_path)
    radii_by_sheet: Dict[str, np.ndarray] = {}
    for sheet_name in sorted(xls.sheet_names, key=vessel_sheet_sort_key):
        header = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)
        if not is_candidate_centerline_sheet(sheet_name, list(header.columns), radius_column):
            continue
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[radius_column])
        radii = pd.to_numeric(df[radius_column], errors="coerce").to_numpy(dtype=float)
        radii = radii[np.isfinite(radii) & (radii > 0)]
        if radii.size:
            radii_by_sheet[sheet_name] = radii
    return radii_by_sheet


def choose_bin_edges(all_radii: np.ndarray, bin_method: str, min_bins: int, max_bins: int) -> np.ndarray:
    """Pick shared histogram bin edges across all vessels, so every sheet's plot uses the same x-axis.

    *bin_method* is either an explicit bin count or a NumPy binning strategy
    name (e.g. ``\"auto\"``); the resulting bin count is clamped to
    ``[min_bins, max_bins]``. A small padding is added around the data range.
    """
    r_min = float(np.min(all_radii))
    r_max = float(np.max(all_radii))
    if not np.isfinite(r_min) or not np.isfinite(r_max):
        raise ValueError("Radius values are not finite.")
    if np.isclose(r_min, r_max):
        pad = max(0.05, 0.1 * abs(r_min))
        return np.linspace(r_min - pad, r_max + pad, min_bins + 1)
    pad = 0.02 * (r_max - r_min)
    hist_min = max(0.0, r_min - pad)
    hist_max = r_max + pad
    if str(bin_method).isdigit():
        n_bins = int(bin_method)
    else:
        edges = np.histogram_bin_edges(all_radii, bins=bin_method, range=(hist_min, hist_max))
        n_bins = max(1, len(edges) - 1)
    n_bins = int(np.clip(n_bins, int(min_bins), int(max_bins)))
    return np.linspace(hist_min, hist_max, n_bins + 1)


def histogram_plan(radii_by_sheet: Dict[str, np.ndarray], bin_edges: np.ndarray) -> Tuple[int, pd.DataFrame]:
    """Compute each sheet's histogram counts/summary stats and the shared peak count (for a common y-axis)."""
    rows = []
    y_max = 0
    for sheet_name, radii in radii_by_sheet.items():
        counts, _ = np.histogram(radii, bins=bin_edges)
        y_max = max(y_max, int(counts.max()) if counts.size else 0)
        rows.append({
            "sheet_name": sheet_name,
            "n_points": int(radii.size),
            "radius_min_mm": float(np.min(radii)),
            "radius_p05_mm": float(np.percentile(radii, 5)),
            "radius_p50_mm": float(np.percentile(radii, 50)),
            "radius_p95_mm": float(np.percentile(radii, 95)),
            "radius_max_mm": float(np.max(radii)),
            "histogram_peak_count": int(counts.max()) if counts.size else 0,
        })
    return max(1, y_max), pd.DataFrame(rows)


def plot_histograms(
    radii_by_sheet: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    y_max: int,
    output_dir: str,
) -> None:
    """Render and save one radius-histogram PNG per sheet, using shared x/y limits across all plots."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    y_lim = max(1.0, y_max * 1.08)

    for sheet_name, radii in radii_by_sheet.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(radii, bins=bin_edges, color="#287C8E", edgecolor="white", linewidth=0.8)
        ax.set_title(sheet_name)
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Point count")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_lim)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{safe_filename(sheet_name)}_radius_histogram.png")
        fig.savefig(out_path, dpi=FIG_DPI)
        plt.close(fig)


def run(excel_path: str, output_dir: str) -> None:
    """Full pipeline: read radii from the morphometrics workbook, plot per-vessel histograms, and write a CSV summary."""
    excel_path = os.path.abspath(excel_path)
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    radii_by_sheet = read_radius_sheets(excel_path, RADIUS_COLUMN)
    if not radii_by_sheet:
        raise RuntimeError(f"No per-centerline sheets with column '{RADIUS_COLUMN}' in {excel_path}")

    all_radii = np.concatenate(list(radii_by_sheet.values()))
    bin_edges = choose_bin_edges(all_radii, BIN_METHOD, MIN_BINS, MAX_BINS)
    y_max, summary_df = histogram_plan(radii_by_sheet, bin_edges)
    plot_histograms(radii_by_sheet, bin_edges, y_max, output_dir)

    summary_df.insert(0, "histogram_bins_n", len(bin_edges) - 1)
    summary_df.insert(1, "histogram_radius_min_mm", float(bin_edges[0]))
    summary_df.insert(2, "histogram_radius_max_mm", float(bin_edges[-1]))
    summary_df.insert(3, "histogram_ylim_count", int(np.ceil(y_max * 1.08)))
    summary_path = os.path.join(output_dir, "radius_histogram_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"  [histograms] {len(radii_by_sheet)} sheet(s), "
          f"radius {bin_edges[0]:.3f}–{bin_edges[-1]:.3f} mm, "
          f"{len(bin_edges) - 1} bins → {output_dir}")


def main() -> None:
    """Direct-run entry point: run the histogram pipeline per the module-level ``EXCEL_PATH``/``OUTPUT_DIR`` config."""
    if not EXCEL_PATH:
        raise SystemExit("Set EXCEL_PATH or call run() with excel_path from stage7 outputs.")
    excel_path = os.path.abspath(EXCEL_PATH)
    output_dir = OUTPUT_DIR or os.path.join(os.path.dirname(excel_path), "radius_histograms")
    run(excel_path, os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
