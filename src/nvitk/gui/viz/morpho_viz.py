"""Napari debug visualization for qvtpy stage-7 TOF morphometrics centerlines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.gui.core.spatial import layer_affine
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER

log = Logger()

MORPHO_OVERLAY_META = "nvitk_morpho_overlay"
MORPHO_PATHS_LAYER = "Morpho centerlines"
MORPHO_POINTS_LAYER = "Morpho samples"

_COLOR_BY_ARRAYS: dict[str, tuple[str, ...]] = {
    "radius": ("EffectiveRadius", "CrossSectionRadius", "radius_mm"),
    "stenosis": ("StenosisPercent", "stenosis_percent_point"),
    "curvature": ("Curvature", "curvature_1_per_mm"),
}


def clear_morpho_layers(viewer: Any) -> None:
    """Remove prior morphometrics overlay layers."""
    for lyr in list(viewer.layers):
        meta = getattr(lyr, "metadata", {}) or {}
        if meta.get(MORPHO_OVERLAY_META):
            viewer.layers.remove(lyr)


def _world_mm_to_data(points_mm: np.ndarray, reference_layer: Any | None) -> np.ndarray:
    """Map physical mm polyline points into the reference layer's data coordinates."""
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    if reference_layer is None:
        return pts[:, :3]
    aff = layer_affine(reference_layer)
    if aff is None:
        return pts[:, :3]
    inv = np.linalg.inv(np.asarray(aff, dtype=np.float64))
    homog = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ inv.T)[:, :3]


def _load_vtp_polyline_with_arrays(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (Nx3 points along longest line, dict of point-data arrays)."""
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError("Reading .vtp files requires VTK.") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return np.empty((0, 3), dtype=np.float64), {}

    all_points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    lines = poly.GetLines()
    ids: list[int] = []
    if lines is not None and poly.GetNumberOfLines() > 0:
        id_list = vtk.vtkIdList()
        lines.InitTraversal()
        best: list[int] = []
        while lines.GetNextCell(id_list):
            cur = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
            if len(cur) > len(best):
                best = cur
        ids = best
    if not ids:
        ids = list(range(all_points.shape[0]))
    pts = all_points[np.asarray(ids, dtype=int)]

    arrays: dict[str, np.ndarray] = {}
    pdata = poly.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        arr = pdata.GetArray(i)
        if arr is None:
            continue
        name = str(arr.GetName() or f"array_{i}")
        try:
            vals = numpy_support.vtk_to_numpy(arr)
        except Exception:
            continue
        vals = np.asarray(vals)
        if vals.ndim > 1:
            vals = vals.reshape(vals.shape[0], -1)[:, 0]
        if vals.shape[0] == all_points.shape[0]:
            arrays[name] = vals[np.asarray(ids, dtype=int)].astype(np.float64)
    return pts, arrays


def _pick_scalar(
    arrays: dict[str, np.ndarray],
    color_by: str,
    n: int,
) -> np.ndarray:
    aliases = _COLOR_BY_ARRAYS.get(str(color_by).strip().lower(), ())
    for key in aliases:
        if key in arrays and arrays[key].shape[0] == n:
            return arrays[key]
    # Case-insensitive fallback.
    lower_map = {k.lower(): v for k, v in arrays.items()}
    for key in aliases:
        if key.lower() in lower_map and lower_map[key.lower()].shape[0] == n:
            return lower_map[key.lower()]
    return np.full(n, np.nan, dtype=np.float64)


def load_stage7_centerline_polylines(
    stage7_dir: Path,
) -> list[dict[str, Any]]:
    """Load centerline VTPs under ``stage7_dir/centerlines``."""
    cl_dir = Path(stage7_dir) / "centerlines"
    if not cl_dir.is_dir():
        raise FileNotFoundError(f"Missing centerlines directory: {cl_dir}")
    files = sorted(cl_dir.glob("*.vtp"))
    # Skip tortuosity debug dumps when present.
    files = [p for p in files if "tortuosity_debug" not in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No .vtp files in {cl_dir}")

    out: list[dict[str, Any]] = []
    for path in files:
        pts, arrays = _load_vtp_polyline_with_arrays(path)
        if pts.shape[0] < 2:
            continue
        out.append(
            {
                "name": path.stem,
                "path": path,
                "points_mm": pts,
                "arrays": arrays,
            }
        )
    return out


def _path_summary_table(stage7_dir: Path) -> pd.DataFrame:
    excel = Path(stage7_dir) / STAGE7_SKIP_MARKER
    if not excel.is_file():
        return pd.DataFrame()
    try:
        return pd.read_excel(excel, sheet_name="00_Path_Summary")
    except Exception as exc:
        log.warning("Could not read Path Summary from %s: %s", excel, exc)
        return pd.DataFrame()


def install_morphometrics_viz(
    viewer: Any,
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
    color_by: str = "radius",
    point_size: float = 2.0,
    edge_width: float = 0.35,
) -> dict[str, Any]:
    """Load stage-7 centerline VTPs into Napari and attach a summary dock."""
    stage7_dir = Path(stage7_dir)
    polylines = load_stage7_centerline_polylines(stage7_dir)
    clear_morpho_layers(viewer)

    paths: list[np.ndarray] = []
    edge_colors: list[str] = []
    sample_coords: list[np.ndarray] = []
    sample_vals: list[np.ndarray] = []
    sample_names: list[str] = []

    palette = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    )
    for i, item in enumerate(polylines):
        data_pts = _world_mm_to_data(item["points_mm"], reference_layer)
        if data_pts.shape[0] < 2:
            continue
        paths.append(data_pts.astype(np.float32))
        edge_colors.append(palette[i % len(palette)])
        vals = _pick_scalar(item["arrays"], color_by, data_pts.shape[0])
        # Subsample points for readability on dense polylines.
        step = max(1, data_pts.shape[0] // 80)
        idx = np.arange(0, data_pts.shape[0], step)
        sample_coords.append(data_pts[idx])
        sample_vals.append(vals[idx])
        sample_names.extend([str(item["name"])] * int(idx.size))

    meta = {MORPHO_OVERLAY_META: True}
    if paths:
        kwargs: dict[str, Any] = {
            "name": MORPHO_PATHS_LAYER,
            "shape_type": "path",
            "edge_color": edge_colors,
            "edge_width": float(edge_width),
            "opacity": 0.9,
            "metadata": meta,
        }
        aff = layer_affine(reference_layer) if reference_layer is not None else None
        if aff is not None:
            kwargs["affine"] = aff
        shapes = viewer.add_shapes(paths, **kwargs)
        try:
            shapes.editable = False
        except Exception:
            pass

    if sample_coords:
        coords = np.concatenate(sample_coords, axis=0)
        values = np.concatenate(sample_vals, axis=0)
        features = {
            "vessel_name": np.asarray(sample_names, dtype=object),
            "value": values.astype(np.float64),
            "color_by": np.asarray([str(color_by)] * len(sample_names), dtype=object),
        }
        finite = np.isfinite(values)
        if finite.any():
            lo = float(np.min(values[finite]))
            hi = float(np.max(values[finite]))
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        pt_kwargs: dict[str, Any] = {
            "name": MORPHO_POINTS_LAYER,
            "features": features,
            "size": float(point_size),
            "symbol": "disc",
            "face_color": "value",
            "face_colormap": "viridis",
            "face_contrast_limits": (lo, hi),
            "border_width": 0,
            "metadata": meta,
        }
        aff = layer_affine(reference_layer) if reference_layer is not None else None
        if aff is not None:
            pt_kwargs["affine"] = aff
        viewer.add_points(coords.astype(np.float64), **pt_kwargs)

    summary = _path_summary_table(stage7_dir)
    _attach_summary_dock(viewer, summary, stage7_dir=stage7_dir, color_by=color_by)

    return {
        "stage7_dir": stage7_dir,
        "n_paths": len(paths),
        "color_by": color_by,
        "n_summary_rows": int(len(summary)),
    }


def _attach_summary_dock(
    viewer: Any,
    summary: pd.DataFrame,
    *,
    stage7_dir: Path,
    color_by: str,
) -> None:
    try:
        from qtpy.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget
    except Exception:
        return

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel(f"Stage-7 morphometrics\n{stage7_dir}"))
    layout.addWidget(QLabel(f"Color by: {color_by}"))
    text = QTextEdit()
    text.setReadOnly(True)
    if summary.empty:
        text.setPlainText("No 00_Path_Summary available.")
    else:
        cols = [
            c
            for c in (
                "vessel_name",
                "full_name",
                "length_mm",
                "radius_mean_mm",
                "tortuosity_dm",
                "radius_p95_mm",
            )
            if c in summary.columns
        ]
        if not cols:
            cols = list(summary.columns[:6])
        preview = summary[cols].head(40)
        text.setPlainText(preview.to_string(index=False))
    layout.addWidget(text)
    attach_left_inspection_dock(
        viewer,
        panel,
        object_name="nvitkMorphoDock",
        title="Morphometrics",
        tabify_with=["nvitkHemoDock", "nvitkCrossSectionDock"],
        minimum_width=300,
    )


__all__ = [
    "MORPHO_OVERLAY_META",
    "MORPHO_PATHS_LAYER",
    "MORPHO_POINTS_LAYER",
    "clear_morpho_layers",
    "install_morphometrics_viz",
    "load_stage7_centerline_polylines",
]
