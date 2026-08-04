"""Napari debug visualization for qvtpy stage-7 TOF morphometrics centerlines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.gui.core.spatial import layer_affine, layer_spacing
from nvitk.gui.viz.layers import install_points_style_sync
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER

log = Logger()

MORPHO_OVERLAY_META = "nvitk_morpho_overlay"
MORPHO_PATHS_LAYER = "Morpho centerlines"
MORPHO_POINTS_LAYER = "Morpho samples"
MORPHO_SURFACES_LAYER = "Morpho surfaces"

_COLOR_BY_ARRAYS: dict[str, tuple[str, ...]] = {
    "radius": ("EffectiveRadius", "CrossSectionRadius", "radius_mm"),
    "stenosis": ("StenosisPercent", "stenosis_percent_point"),
    "curvature": ("Curvature", "curvature_1_per_mm"),
    "area": ("CrossSectionArea", "cross_section_area_mm2", "EffectiveRadius", "CrossSectionRadius"),
}

# Morphometrics VTPs store points as voxel_index * spacing (origin 0), not scanner world.
# Napari's layer-controls size slider is an *integer* with minimum 1, so sub-voxel
# defaults must be driven from our dock spinbox (see _attach_summary_dock).
DEFAULT_MORPHO_POINT_SIZE = 0.35
MORPHO_POINT_CANVAS_SIZE_LIMITS = (0.5, 200.0)

# Path-summary columns preferred in the Morphometrics dock table.
_MORPHO_TABLE_COLS: tuple[str, ...] = (
    "vessel_name",
    "full_name",
    "length_mm",
    "radius_mean_mm",
    "radius_p95_mm",
    "tortuosity_dm",
    "stenosis_percent_max",
    "degree_of_stenosis_pct",
    "stenosis_length_total_mm",
    "stenosis_segments_n",
    "curvature_mean_1_per_mm",
    "n_paths",
)

# Aggregation: length-weighted means for continuous metrics; max / sum for stenosis.
_AGG_MEAN_COLS: tuple[str, ...] = (
    "radius_mean_mm",
    "radius_p95_mm",
    "tortuosity_dm",
    "curvature_mean_1_per_mm",
    "curvature_mean",
)
_AGG_MAX_COLS: tuple[str, ...] = (
    "stenosis_percent_max",
    "degree_of_stenosis_pct",
    "stenosis_percent",
)
_AGG_SUM_COLS: tuple[str, ...] = (
    "stenosis_length_total_mm",
    "stenosis_segments_n",
)


def clear_morpho_layers(viewer: Any) -> None:
    """Remove prior morphometrics overlay layers."""
    for lyr in list(viewer.layers):
        meta = getattr(lyr, "metadata", {}) or {}
        if meta.get(MORPHO_OVERLAY_META):
            viewer.layers.remove(lyr)


def _vtp_mm_to_data(points_mm: np.ndarray, reference_layer: Any | None) -> np.ndarray:
    """Map morphometrics VTP points (voxel × spacing, origin 0) → layer data indices.

    Morphometrics writes VTK points as ``ijk * spacing`` with origin ``(0,0,0)``,
    not full NIfTI world. Dividing by spacing recovers voxel indices that match
    the Labels layer when the overlay affine is set from that layer.
    """
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    xyz = pts[:, :3]
    sp = layer_spacing(reference_layer) if reference_layer is not None else None
    if sp is None or len(sp) < 3:
        # Fallback: assume unit spacing (already in voxel coords).
        return xyz
    scale = np.asarray([float(sp[0]), float(sp[1]), float(sp[2])], dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return xyz / scale


# Back-compat alias (previous incorrect name suggested scanner world).
_world_mm_to_data = _vtp_mm_to_data


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
    """Select the point-array matching *color_by* (by known aliases, case-insensitive) with length
    *n* from *arrays*; an all-NaN array if none matches."""
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
    import re

    cl_dir = Path(stage7_dir) / "centerlines"
    if not cl_dir.is_dir():
        raise FileNotFoundError(f"Missing centerlines directory: {cl_dir}")
    files = sorted(cl_dir.glob("*.vtp"))
    # Skip tortuosity debug dumps when present.
    files = [p for p in files if "tortuosity_debug" not in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No .vtp files in {cl_dir}")

    # Prefer anatomic segment VTPs (e.g. LICA_M1, LICA_M2s) over legacy
    # label_comp_arm / trunk exports that duplicate shared trunks.
    labeled_path_re = re.compile(r"^\d+_.+_comp\d{2}_(trunk|arm|tree|path)", re.I)
    anatomic = [p for p in files if not labeled_path_re.match(p.stem)]
    if anatomic:
        files = anatomic

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
    """Load and aggregate the ``00_Path_Summary`` sheet from stage-7's Excel output, one row per
    vessel; empty frame if the file is missing or unreadable."""
    excel = Path(stage7_dir) / STAGE7_SKIP_MARKER
    if not excel.is_file():
        return pd.DataFrame()
    try:
        raw = pd.read_excel(excel, sheet_name="00_Path_Summary")
    except Exception as exc:
        log.warning("Could not read Path Summary from %s: %s", excel, exc)
        return pd.DataFrame()
    return _aggregate_summary_by_vessel(raw)


def _aggregate_summary_by_vessel(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple path / component rows into one row per vessel_name.

    Morphometrics can emit several paths (tree arms) or CCs for the same vessel
    id; the GUI table should show one measure set per vessel.
    """
    if df.empty or "vessel_name" not in df.columns:
        return df
    key = "vessel_name"
    if df[key].astype(str).nunique(dropna=False) == len(df):
        return df

    rows: list[dict[str, Any]] = []
    for name, g in df.groupby(key, dropna=False, sort=False):
        row: dict[str, Any] = {key: name}
        if "full_name" in g.columns:
            fn = g["full_name"].dropna().astype(str)
            row["full_name"] = fn.iloc[0] if len(fn) else str(name)
        length = pd.to_numeric(g.get("length_mm"), errors="coerce")
        length_sum = float(length.fillna(0.0).sum())
        row["length_mm"] = length_sum
        weights = length.fillna(0.0).to_numpy(dtype=np.float64)
        w_sum = float(weights.sum())

        def _weighted_mean(col: str) -> float:
            """Length-weighted mean of *col* over the group (plain nanmean if weights sum to ~0)."""
            if col not in g.columns:
                return float("nan")
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64)
            if w_sum > 1e-9 and np.isfinite(vals).any():
                m = np.isfinite(vals) & (weights > 0)
                if m.any():
                    return float(np.average(vals[m], weights=weights[m]))
            return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

        def _max(col: str) -> float:
            """Max finite value of *col* over the group, or NaN if none are finite."""
            if col not in g.columns:
                return float("nan")
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64)
            return float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan")

        def _sum(col: str) -> float:
            """Sum of finite values of *col* over the group, or NaN if none are finite."""
            if col not in g.columns:
                return float("nan")
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64)
            return float(np.nansum(vals)) if np.isfinite(vals).any() else float("nan")

        for col in _AGG_MEAN_COLS:
            if col in g.columns:
                row[col] = _weighted_mean(col)
        for col in _AGG_MAX_COLS:
            if col in g.columns:
                row[col] = _max(col)
        for col in _AGG_SUM_COLS:
            if col in g.columns:
                row[col] = _sum(col)
        row["n_paths"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_vtp_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3, faces Mx3) from a triangle surface VTP."""
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
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    tpoly = tri.GetOutput()
    faces_vtk = tpoly.GetPolys()
    if faces_vtk is None or faces_vtk.GetNumberOfCells() == 0:
        return verts, np.empty((0, 3), dtype=np.int64)
    ids = numpy_support.vtk_to_numpy(faces_vtk.GetData()).astype(np.int64)
    # VTK cell array: [n, i, j, k, n, ...]
    faces: list[list[int]] = []
    i = 0
    while i < ids.size:
        n = int(ids[i])
        if n >= 3:
            faces.append([int(ids[i + 1]), int(ids[i + 2]), int(ids[i + 3])])
        i += n + 1
    if not faces:
        return verts, np.empty((0, 3), dtype=np.int64)
    return verts, np.asarray(faces, dtype=np.int64)


def load_stage7_surfaces(
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load vessel surface meshes under ``stage7_dir/surfaces`` in data coords."""
    surf_dir = Path(stage7_dir) / "surfaces"
    if not surf_dir.is_dir():
        return []
    files = sorted(
        p for p in surf_dir.glob("*.vtp") if "pre_refined" not in p.name.lower()
    )
    out: list[tuple[np.ndarray, np.ndarray]] = []
    vert_offset = 0
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    for path in files:
        verts_mm, faces = _load_vtp_surface(path)
        if verts_mm.shape[0] < 3 or faces.shape[0] == 0:
            continue
        verts = _vtp_mm_to_data(verts_mm, reference_layer)
        all_verts.append(verts.astype(np.float32))
        all_faces.append(faces.astype(np.int32) + int(vert_offset))
        vert_offset += int(verts.shape[0])
    if not all_verts:
        return []
    return [(np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0))]


def install_morphometrics_viz(
    viewer: Any,
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
    color_by: str = "radius",
    point_size: float = DEFAULT_MORPHO_POINT_SIZE,
    edge_width: float = 0.35,
    show_surfaces: bool = True,
) -> dict[str, Any]:
    """Load stage-7 centerline VTPs into Napari and attach a summary dock."""
    stage7_dir = Path(stage7_dir)
    if point_size == DEFAULT_MORPHO_POINT_SIZE:
        point_size = _default_morpho_point_size(reference_layer)
    try:
        polylines = load_stage7_centerline_polylines(stage7_dir)
    except FileNotFoundError as exc:
        log.warning("Morphometrics centerlines missing: %s", exc)
        polylines = []

    clear_morpho_layers(viewer)

    paths: list[np.ndarray] = []
    edge_colors: list[str] = []
    stored: list[dict[str, Any]] = []

    palette = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    )
    for i, item in enumerate(polylines):
        data_pts = _vtp_mm_to_data(item["points_mm"], reference_layer)
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
    aff = layer_affine(reference_layer) if reference_layer is not None else None
    if paths:
        kwargs: dict[str, Any] = {
            "name": MORPHO_PATHS_LAYER,
            "shape_type": "path",
            "edge_color": edge_colors,
            "edge_width": float(edge_width),
            "opacity": 0.9,
            "metadata": meta,
        }
        if aff is not None:
            kwargs["affine"] = aff
        shapes = viewer.add_shapes(paths, **kwargs)
        try:
            shapes.editable = False
        except Exception:
            pass

    n_surf = 0
    if show_surfaces:
        try:
            surfs = load_stage7_surfaces(stage7_dir, reference_layer=reference_layer)
            for verts, faces in surfs:
                if verts.shape[0] < 3 or faces.shape[0] == 0:
                    continue
                skw: dict[str, Any] = {
                    "name": MORPHO_SURFACES_LAYER,
                    "opacity": 0.35,
                    "blending": "translucent",
                    "metadata": meta,
                }
                if aff is not None:
                    skw["affine"] = aff
                viewer.add_surface((verts, faces), **skw)
                n_surf += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Morpho surfaces not loaded: %s", exc)

    points_layer = None
    try:
        points_layer = _add_or_update_morpho_points(
            viewer,
            stored,
            color_by=color_by,
            point_size=point_size,
            reference_layer=reference_layer,
            metadata=meta,
            recreate=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Morpho sample points not loaded: %s", exc)

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
        "n_surfaces": n_surf,
        "color_by": color_by,
        "n_summary_rows": int(len(summary)),
    }


def _default_morpho_point_size(reference_layer: Any | None) -> float:
    """Small data-space size; Napari's integer slider cannot go below 1."""
    _ = reference_layer
    return float(DEFAULT_MORPHO_POINT_SIZE)


def _apply_morpho_point_size(layer: Any, size: float) -> None:
    """Apply fractional point size and relax Napari canvas clamp for small markers."""
    size = float(max(0.05, size))
    try:
        layer.canvas_size_limits = MORPHO_POINT_CANVAS_SIZE_LIMITS
    except Exception:
        pass
    layer.size = size
    if hasattr(layer, "current_size"):
        layer.current_size = size


def _read_layer_point_size(layer: Any, fallback: float) -> float:
    """Recover *layer*'s current point size (correcting for Napari's integer-slider snap-to-1 quirk
    when a smaller *fallback* was requested), falling back to *fallback* if unreadable."""
    try:
        size = getattr(layer, "current_size", None)
        if size is not None:
            val = float(size)
            # Napari's integer slider often snaps fractional defaults up to 1.0;
            # treat that as "use our smaller default" when we asked for <1.
            if val >= 0.999 and float(fallback) < 0.999:
                return float(fallback)
            return val
    except Exception:
        pass
    try:
        sz = getattr(layer, "size", None)
        if isinstance(sz, (int, float)):
            return float(sz)
        arr = np.asarray(sz)
        if arr.size:
            return float(np.median(arr.astype(np.float64)))
    except Exception:
        pass
    return float(fallback)


def _add_or_update_morpho_points(
    viewer: Any,
    polylines: list[dict[str, Any]],
    *,
    color_by: str,
    point_size: float,
    reference_layer: Any | None,
    metadata: dict[str, Any],
    recreate: bool = False,
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
    # Keep points visible even when a scalar is all-NaN (e.g. missing stenosis).
    display_values = np.where(np.isfinite(values), values, 0.0).astype(np.float64)
    features = {
        "vessel_name": np.asarray(sample_names, dtype=object),
        "value": display_values,
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

    size = float(point_size)
    existing = None
    for lyr in list(viewer.layers):
        if lyr.name == MORPHO_POINTS_LAYER:
            existing = lyr
            break
    if existing is not None:
        size = _read_layer_point_size(existing, size)
        if recreate:
            prev = getattr(viewer, "_nvitk_morpho_points_style_disconnect", None)
            if callable(prev):
                try:
                    prev()
                except Exception:
                    pass
            try:
                viewer.layers.remove(existing)
            except Exception:
                pass
            existing = None

    if existing is not None:
        existing.data = coords.astype(np.float64)
        existing.features = features
        try:
            existing.face_colormap = "viridis"
            existing.face_contrast_limits = (lo, hi)
            if hasattr(existing, "face_color_mode"):
                existing.face_color_mode = "colormap"
            existing.face_color = "value"
            if hasattr(existing, "refresh_colors"):
                try:
                    existing.refresh_colors(update_color_mapping=True)
                except TypeError:
                    existing.refresh_colors()
        except Exception:
            existing.face_color = _scalar_rgba(display_values, lo, hi)
        # Size only — do not touch face_color via init (breaks colormap mode).
        _apply_morpho_point_size(existing, size)
        return existing

    pt_kwargs: dict[str, Any] = {
        "name": MORPHO_POINTS_LAYER,
        "features": features,
        "size": size,
        "symbol": "disc",
        "face_color": "value",
        "face_colormap": "viridis",
        "face_contrast_limits": (lo, hi),
        "border_width": 0,
        "border_width_is_relative": False,
        "metadata": metadata,
        "opacity": 0.95,
        "canvas_size_limits": MORPHO_POINT_CANVAS_SIZE_LIMITS,
    }
    aff = layer_affine(reference_layer) if reference_layer is not None else None
    if aff is not None:
        pt_kwargs["affine"] = aff
    try:
        layer = viewer.add_points(coords.astype(np.float64), **pt_kwargs)
    except TypeError:
        pt_kwargs.pop("border_width_is_relative", None)
        pt_kwargs.pop("canvas_size_limits", None)
        layer = viewer.add_points(coords.astype(np.float64), **pt_kwargs)

    try:
        if hasattr(layer, "face_color_mode"):
            layer.face_color_mode = "colormap"
        layer.face_color = "value"
        layer.face_colormap = "viridis"
        layer.face_contrast_limits = (lo, hi)
        if hasattr(layer, "refresh_colors"):
            try:
                layer.refresh_colors(update_color_mapping=True)
            except TypeError:
                layer.refresh_colors()
    except Exception:
        layer.face_color = _scalar_rgba(display_values, lo, hi)

    _apply_morpho_point_size(layer, size)
    if hasattr(layer, "current_symbol"):
        layer.current_symbol = "disc"

    prev = getattr(viewer, "_nvitk_morpho_points_style_disconnect", None)
    if callable(prev):
        try:
            prev()
        except Exception:
            pass
    setattr(
        viewer,
        "_nvitk_morpho_points_style_disconnect",
        install_points_style_sync(layer, sync_face_color=False),
    )
    return layer


def _scalar_rgba(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map *values* through the viridis colormap normalized to ``[lo, hi]``, making non-finite entries
    fully transparent."""
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
    point_size: float = DEFAULT_MORPHO_POINT_SIZE,
) -> None:
    """Build and dock a per-vessel morphometrics summary table with color-by/point-size controls that
    live-update the Morpho points layer; no-op if Qt bindings aren't available."""
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except Exception:
        log.warning("Morphometrics dock skipped: Qt bindings unavailable.")
        return

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel(f"Stage-7 morphometrics\n{stage7_dir}"))

    color_row = QHBoxLayout()
    color_row.addWidget(QLabel("Color samples by"))
    color_selector = QComboBox()
    for key in ("radius", "stenosis", "curvature", "area"):
        color_selector.addItem(key, key)
    idx = color_selector.findData(str(color_by))
    color_selector.setCurrentIndex(idx if idx >= 0 else 0)
    color_row.addWidget(color_selector, stretch=1)
    layout.addLayout(color_row)

    # Napari's layer-controls size slider is integer with min=1 — too coarse for
    # mouse / high-res volumes. Expose a fractional size control here instead.
    size_row = QHBoxLayout()
    size_row.addWidget(QLabel("Sample point size"))
    size_spin = QDoubleSpinBox()
    size_spin.setDecimals(2)
    size_spin.setRange(0.05, 20.0)
    size_spin.setSingleStep(0.05)
    size_spin.setValue(float(point_size))
    size_spin.setToolTip(
        "Point diameter in data coordinates. "
        "Napari's layer slider cannot go below 1 — use this control for smaller markers."
    )
    size_row.addWidget(size_spin, stretch=1)
    layout.addLayout(size_row)

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
    if summary.empty:
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Info"])
        table.setRowCount(1)
        table.setItem(0, 0, QTableWidgetItem("No 00_Path_Summary available."))
    else:
        cols = [c for c in _MORPHO_TABLE_COLS if c in summary.columns]
        if not cols:
            cols = list(summary.columns[:10])
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
        "color_by": str(color_by),
    }

    def _on_color_changed(_index: int = 0) -> None:
        """Recreate the Morpho points layer colored by the newly selected scalar array."""
        key = str(color_selector.currentData() or color_selector.currentText() or "radius")
        state["color_by"] = key
        if not state["polylines"]:
            return
        # Recreate points so Napari colormap mode stays consistent across schemes.
        layer = _add_or_update_morpho_points(
            viewer,
            state["polylines"],
            color_by=key,
            point_size=float(size_spin.value()),
            reference_layer=state["reference_layer"],
            metadata={MORPHO_OVERLAY_META: True},
            recreate=True,
        )
        state["points_layer"] = layer
        state["point_size"] = float(size_spin.value())

    def _on_size_changed(value: float) -> None:
        """Apply the newly chosen point size to the Morpho points layer, finding it by name if needed."""
        state["point_size"] = float(value)
        layer = state.get("points_layer")
        if layer is None:
            for lyr in viewer.layers:
                if lyr.name == MORPHO_POINTS_LAYER:
                    layer = lyr
                    state["points_layer"] = lyr
                    break
        if layer is None:
            return
        _apply_morpho_point_size(layer, float(value))

    color_selector.currentIndexChanged.connect(_on_color_changed)
    size_spin.valueChanged.connect(_on_size_changed)

    dock = attach_left_inspection_dock(
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
    if dock is not None:
        try:
            dock.show()
            dock.raise_()
            win = dock.parent()
            if win is not None and hasattr(win, "setActiveDockWidget"):
                win.setActiveDockWidget(dock)
        except Exception:
            pass


__all__ = [
    "DEFAULT_MORPHO_POINT_SIZE",
    "MORPHO_OVERLAY_META",
    "MORPHO_PATHS_LAYER",
    "MORPHO_POINTS_LAYER",
    "MORPHO_SURFACES_LAYER",
    "clear_morpho_layers",
    "install_morphometrics_viz",
    "load_stage7_centerline_polylines",
    "load_stage7_surfaces",
]
