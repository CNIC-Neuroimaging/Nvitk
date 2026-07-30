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
    # Keep full polylines + arrays so the dock can recolor without reloading VTPs.
    stored: list[dict[str, Any]] = []

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
        stored.append(
            {
                "name": str(item["name"]),
                "points": data_pts,
                "arrays": item["arrays"],
            }
        )

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

    points_layer = _add_or_update_morpho_points(
        viewer,
        stored,
        color_by=color_by,
        point_size=point_size,
        reference_layer=reference_layer,
        metadata=meta,
    )

    summary = _path_summary_table(stage7_dir)
    _attach_summary_dock(
        viewer,
        summary,
        stage7_dir=stage7_dir,
        color_by=color_by,
        polylines=stored,
        points_layer=points_layer,
        reference_layer=reference_layer,
        point_size=point_size,
    )

    return {
        "stage7_dir": stage7_dir,
        "n_paths": len(paths),
        "color_by": color_by,
        "n_summary_rows": int(len(summary)),
    }


def _add_or_update_morpho_points(
    viewer: Any,
    polylines: list[dict[str, Any]],
    *,
    color_by: str,
    point_size: float,
    reference_layer: Any | None,
    metadata: dict[str, Any],
) -> Any | None:
    """Create/update the Morpho samples Points layer colored by *color_by*."""
    sample_coords: list[np.ndarray] = []
    sample_vals: list[np.ndarray] = []
    sample_names: list[str] = []
    for item in polylines:
        data_pts = item["points"]
        vals = _pick_scalar(item["arrays"], color_by, data_pts.shape[0])
        step = max(1, data_pts.shape[0] // 80)
        idx = np.arange(0, data_pts.shape[0], step)
        sample_coords.append(data_pts[idx])
        sample_vals.append(vals[idx])
        sample_names.extend([str(item["name"])] * int(idx.size))
    if not sample_coords:
        return None
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

    existing = None
    for lyr in viewer.layers:
        if lyr.name == MORPHO_POINTS_LAYER:
            existing = lyr
            break
    if existing is not None:
        existing.data = coords.astype(np.float64)
        existing.features = features
        existing.size = float(point_size)
        try:
            existing.face_colormap = "viridis"
            existing.face_contrast_limits = (lo, hi)
            existing.face_color = "value"
            if hasattr(existing, "face_color_mode"):
                existing.face_color_mode = "colormap"
            if hasattr(existing, "refresh_colors"):
                try:
                    existing.refresh_colors(update_color_mapping=True)
                except TypeError:
                    existing.refresh_colors()
        except Exception:
            # Fall back to explicit RGBA (NaN-safe).
            existing.face_color = _scalar_rgba(values, lo, hi)
        return existing

    pt_kwargs: dict[str, Any] = {
        "name": MORPHO_POINTS_LAYER,
        "features": features,
        "size": float(point_size),
        "symbol": "disc",
        "face_color": "value",
        "face_colormap": "viridis",
        "face_contrast_limits": (lo, hi),
        "border_width": 0,
        "metadata": metadata,
    }
    aff = layer_affine(reference_layer) if reference_layer is not None else None
    if aff is not None:
        pt_kwargs["affine"] = aff
    return viewer.add_points(coords.astype(np.float64), **pt_kwargs)


def _scalar_rgba(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import matplotlib as mpl

    try:
        cmap = mpl.colormaps["viridis"]
    except (KeyError, AttributeError):
        cmap = mpl.cm.get_cmap("viridis")
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((np.nan_to_num(values, nan=lo) - lo) / span, 0.0, 1.0)
    rgba = np.asarray(cmap(norm), dtype=np.float64)
    finite = np.isfinite(values)
    rgba[~finite] = (0.0, 0.0, 0.0, 0.0)
    return rgba


def _attach_summary_dock(
    viewer: Any,
    summary: pd.DataFrame,
    *,
    stage7_dir: Path,
    color_by: str,
    polylines: list[dict[str, Any]] | None = None,
    points_layer: Any | None = None,
    reference_layer: Any | None = None,
    point_size: float = 2.0,
) -> None:
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QComboBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except Exception:
        return

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel(f"Stage-7 morphometrics\n{stage7_dir}"))

    color_row = QHBoxLayout()
    color_row.addWidget(QLabel("Color samples by"))
    color_selector = QComboBox()
    for key in ("radius", "stenosis", "curvature"):
        color_selector.addItem(key, key)
    idx = color_selector.findData(str(color_by))
    color_selector.setCurrentIndex(idx if idx >= 0 else 0)
    color_row.addWidget(color_selector, stretch=1)
    layout.addLayout(color_row)

    table = QTableWidget(0, 0)
    table.setAlternatingRowColors(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setStyleSheet(
        "QTableWidget {"
        "  background-color: #2b2b2b; color: #e8e8e8; gridline-color: #454545;"
        "}"
        "QHeaderView::section {"
        "  background-color: #353535; color: #e8e8e8; padding: 4px;"
        "  border: 1px solid #454545;"
        "}"
    )
    preferred_cols = (
        "vessel_name",
        "full_name",
        "length_mm",
        "radius_mean_mm",
        "tortuosity_dm",
        "radius_p95_mm",
        "stenosis_percent",
        "curvature_mean",
    )
    if summary.empty:
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Info"])
        table.setRowCount(1)
        table.setItem(0, 0, QTableWidgetItem("No 00_Path_Summary available."))
    else:
        cols = [c for c in preferred_cols if c in summary.columns]
        if not cols:
            cols = list(summary.columns[:8])
        preview = summary[cols].head(80)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([str(c) for c in cols])
        table.setRowCount(len(preview))
        for r, (_, row) in enumerate(preview.iterrows()):
            for c, col in enumerate(cols):
                val = row.get(col)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    text = ""
                elif isinstance(val, float):
                    text = f"{val:.4g}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(r, c, item)
    layout.addWidget(table, stretch=1)

    state = {
        "polylines": polylines or [],
        "points_layer": points_layer,
        "reference_layer": reference_layer,
        "point_size": float(point_size),
    }

    def _on_color_changed(_index: int = 0) -> None:
        key = str(color_selector.currentData() or color_selector.currentText() or "radius")
        if not state["polylines"]:
            return
        layer = _add_or_update_morpho_points(
            viewer,
            state["polylines"],
            color_by=key,
            point_size=state["point_size"],
            reference_layer=state["reference_layer"],
            metadata={MORPHO_OVERLAY_META: True},
        )
        state["points_layer"] = layer

    color_selector.currentIndexChanged.connect(_on_color_changed)

    attach_left_inspection_dock(
        viewer,
        panel,
        object_name="nvitk_morphometrics_dock",
        title="Morphometrics",
        tabify_with=[
            "nvitk_hemodynamics_hemo_dock",
            "nvitk_hemodynamics_pitc_dock",
            "nvitk_hemodynamics_pwv_dock",
            "nvitk_vessel_cross_section_dock",
            "nvitk_qc_measurements_dock",
        ],
        minimum_width=320,
    )


__all__ = [
    "MORPHO_OVERLAY_META",
    "MORPHO_PATHS_LAYER",
    "MORPHO_POINTS_LAYER",
    "clear_morpho_layers",
    "install_morphometrics_viz",
    "load_stage7_centerline_polylines",
]
