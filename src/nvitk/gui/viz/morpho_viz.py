"""Napari visualization of TOF morphometrics outputs (stage 7 / the GUI *Morphometrics* tool).

Overlays: the measured centerlines (paths), samples along them coloured by a per-point array,
the vessel surfaces as one Mesh-panel surface whose vertices carry the ``label`` they belong to
(so they can be coloured per vessel, by label or by a measured value;
:mod:`nvitk.gui.viz.morpho_results`), and the detected lesions as points. Every overlay is
placed with the reference layer's full data -> world transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.gui.core.spatial import layer_spacing, layer_spatial_kwargs
from nvitk.gui.viz.layers import install_points_style_sync
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.measure.morpho.export_utils.report import aggregate_summary_by_vessel as _aggregate_summary_by_vessel
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import STAGE7_SKIP_MARKER

log = Logger()

MORPHO_OVERLAY_META = "nvitk_morpho_overlay"
MORPHO_PATHS_LAYER = "Morpho centerlines"
MORPHO_POINTS_LAYER = "Morpho samples"
MORPHO_SURFACES_LAYER = "Morpho surfaces"
MORPHO_LESIONS_LAYER = "Morpho lesions"
#: Lesion point colours by kind.
LESION_COLOURS: dict[str, tuple[float, float, float]] = {
    "stenosis": (0.93, 0.33, 0.10),
    "enlargement": (0.20, 0.50, 0.95),
    "hypoplasia": (0.62, 0.40, 0.85),  # an absent vessel: drawn only where it has a place (an A1's course)
    "fetal": (0.95, 0.75, 0.20),  # a fetal PCA: drawn on the PComm
}

_COLOR_BY_ARRAYS: dict[str, tuple[str, ...]] = {
    "radius": ("EffectiveRadius", "CrossSectionRadius", "radius_mm"),
    "stenosis": ("StenosisPercent", "stenosis_percent_point"),
    "curvature": ("Curvature", "curvature_1_per_mm"),
    "area": ("CrossSectionArea", "cross_section_area_mm2", "EffectiveRadius", "CrossSectionRadius"),
}

# Morphometrics VTPs are written in scanner RAS (field data NVITK_COORDINATE_FRAME="RAS" plus the
# NVITK_AFFINE they were mapped with); older outputs are voxel_index * spacing (origin 0). The
# loaders bring both to the latter frame (_vtp_points_in_index_mm), which _vtp_mm_to_data maps
# to layer data indices.
# Napari's layer-controls size slider is an *integer* with minimum 1, so sub-voxel
# defaults must be driven from our dock spinbox (see _attach_summary_dock).
DEFAULT_MORPHO_POINT_SIZE = 0.35
MORPHO_POINT_CANVAS_SIZE_LIMITS = (0.5, 200.0)

# Columns of the per-vessel table shown in the Morphometrics dock (the whole table, with every
# column, is in the results window: nvitk.gui.viz.morpho_results). The legacy names cover
# outputs written before the compact tables.
_MORPHO_TABLE_COLS: tuple[str, ...] = (
    "vessel_name",
    "present",
    "main_length_mm",
    "centerline_length_mm",
    "diameter_median_mm",
    "diameter_min_mm",
    "tortuosity_dm",
    "curvature_mean_1_per_mm",
    "n_stenosis",
    "n_stenosis_high_confidence",
    "stenosis_diameter_percent_max",
    "n_enlargement",
    "volume_mm3",
    "surface_area_mm2",
    "length_mm",
    "radius_mean_mm",
    "stenosis_percent_max",
    "n_paths",
)

#: Per-label volumetry columns merged in from the ``06_Volumetry`` sheet.
_VOLUMETRY_MERGE_COLS: tuple[str, ...] = (
    "n_voxels",
    "volume_mm3",
    "volume_ul",
    "mesh_volume_mm3",
    "surface_area_mm2",
    "equivalent_radius_mm",
    "input_volume_mm3",
)


def clear_morpho_layers(viewer: Any) -> None:
    """Remove prior morphometrics overlay layers."""
    for lyr in list(viewer.layers):
        meta = getattr(lyr, "metadata", {}) or {}
        if meta.get(MORPHO_OVERLAY_META):
            viewer.layers.remove(lyr)


def _vtp_mm_to_data(points_mm: np.ndarray, reference_layer: Any | None) -> np.ndarray:
    """Map morphometrics VTP points (voxel × spacing, origin 0) -> layer data indices.

    The loaders return that frame for both RAS-tagged and legacy files
    (:func:`_vtp_points_in_index_mm`). Dividing by spacing recovers voxel indices that
    match the Labels layer when the overlay affine is set from that layer.
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



def _vtp_points_in_index_mm(poly: Any, points: np.ndarray) -> np.ndarray:
    """VTP points in the ``voxel_index × spacing`` frame, whatever frame the file was written in.

    RAS-tagged files carry the voxel->RAS affine they were mapped with; it is inverted here, so
    the overlay lines up with the mask without trusting any layer metadata.
    """
    from vtk.util import numpy_support

    fd = poly.GetFieldData()
    tag = fd.GetAbstractArray("NVITK_COORDINATE_FRAME") if fd is not None else None
    aff_arr = fd.GetArray("NVITK_AFFINE") if fd is not None else None
    if tag is None or aff_arr is None or str(tag.GetValue(0)).upper() != "RAS":
        return points
    aff = numpy_support.vtk_to_numpy(aff_arr).astype(np.float64).reshape(4, 4)
    ijk = (np.linalg.inv(aff) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    return ijk * np.linalg.norm(aff[:3, :3], axis=0)


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

    all_points = _vtp_points_in_index_mm(
        poly, numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    )
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
    return _merge_volumetry(_aggregate_summary_by_vessel(raw), excel)


def _merge_volumetry(summary: pd.DataFrame, excel: Path) -> pd.DataFrame:
    """Attach per-label volumetry (voxels, mm³, surface area) from ``06_Volumetry``.

    Volumetry is measured on the segmentation mask, so it is per label rather
    than per centerline; joining on ``vessel_name`` puts it beside the
    centerline metrics without double-counting anything.
    """
    if summary.empty or "vessel_name" not in summary.columns:
        return summary
    try:
        vol = pd.read_excel(excel, sheet_name="06_Volumetry")
    except Exception:
        return summary
    if vol.empty or "vessel_name" not in vol.columns:
        return summary
    vol = vol[vol["label"].astype(str) != "TOTAL"]
    cols = ["vessel_name", *[c for c in _VOLUMETRY_MERGE_COLS if c in vol.columns]]
    keep = [c for c in summary.columns if c not in _VOLUMETRY_MERGE_COLS]
    return summary[keep].merge(vol[cols].drop_duplicates("vessel_name"), on="vessel_name", how="left")





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

    verts = _vtp_points_in_index_mm(
        poly, numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    )
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


def _surface_label(path: Path) -> int:
    """Label id of a surface VTP (``<label>_<VESSEL>_comp01.vtp``), 0 when the name has none."""
    import re

    m = re.match(r"^(\d+)_", path.stem)
    return int(m.group(1)) if m else 0


def load_stage7_surface_mesh(
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """All vessel surfaces under ``stage7_dir/surfaces`` as one mesh in the reference layer's data
    coordinates: ``(vertices, faces, label id per vertex)``; ``None`` when there are none."""
    surf_dir = Path(stage7_dir) / "surfaces"
    if not surf_dir.is_dir():
        return None
    files = sorted(p for p in surf_dir.glob("*.vtp") if "pre_refined" not in p.name.lower())
    all_verts, all_faces, all_labels = [], [], []
    offset = 0
    for path in files:
        verts_mm, faces = _load_vtp_surface(path)
        if verts_mm.shape[0] < 3 or faces.shape[0] == 0:
            continue
        verts = _vtp_mm_to_data(verts_mm, reference_layer)
        all_verts.append(verts.astype(np.float32))
        all_faces.append(faces.astype(np.int32) + int(offset))
        all_labels.append(np.full(len(verts), _surface_label(path), dtype=np.int32))
        offset += int(verts.shape[0])
    if not all_verts:
        return None
    return np.concatenate(all_verts), np.concatenate(all_faces), np.concatenate(all_labels)


def load_stage7_surfaces(
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load vessel surface meshes under ``stage7_dir/surfaces`` in data coords."""
    mesh = load_stage7_surface_mesh(stage7_dir, reference_layer=reference_layer)
    return [] if mesh is None else [(mesh[0], mesh[1])]


def _overlay_kwargs(reference_layer: Any | None) -> dict[str, Any]:
    """Placement of an overlay given in *reference_layer*'s data coordinates (scale, translate,
    affine...: every part of its transform, not just the affine)."""
    return layer_spatial_kwargs(reference_layer) if reference_layer is not None else {}


def label_colour_lut(label_ids: Any, label_layer: Any | None) -> dict[int, tuple[float, ...]]:
    """Label id -> RGBA: *label_layer*'s own colours, else a fixed palette."""
    from matplotlib import colormaps

    tab = colormaps["tab20"]
    out: dict[int, tuple[float, ...]] = {}
    for i, lid in enumerate(sorted({int(v) for v in label_ids})):
        rgba = None
        if label_layer is not None:
            try:
                from nvitk.gui.labels.visibility import get_label_color

                rgba = tuple(float(c) for c in np.asarray(get_label_color(label_layer, lid), dtype=float)[:4])
            except Exception:  # noqa: BLE001
                rgba = None
        if rgba is None or len(rgba) < 4:
            rgba = tuple(float(c) for c in tab(i % 20))
        out[lid] = rgba
    return out


def _add_morpho_surface(
    viewer: Any,
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    *,
    reference_layer: Any | None,
    label_layer: Any | None,
    metadata: dict[str, Any],
) -> Any:
    """The vessel surfaces as one Mesh-panel surface (world coordinates) with a ``label`` field,
    shown in the label colours."""
    from nvitk.gui.mesh.layers import add_field, add_mesh_layer, layer_spatial_affine
    from nvitk.types import Mesh

    aff = layer_spatial_affine(reference_layer) if reference_layer is not None else np.eye(4)
    world = np.asarray(verts, dtype=float) @ aff[:3, :3].T + aff[:3, 3]
    layer = add_mesh_layer(
        viewer, Mesh(vertices=world, faces=np.asarray(faces), metadata={"name": MORPHO_SURFACES_LAYER}),
        name=MORPHO_SURFACES_LAYER, opacity=0.35,
    )
    try:
        layer.blending = "translucent"
        layer.metadata = {**dict(layer.metadata or {}), **metadata}
    except Exception:  # noqa: BLE001
        pass
    add_field(layer, "label", np.asarray(vertex_labels, dtype=np.float32),
              categories=label_colour_lut(np.unique(vertex_labels), label_layer), show=True)
    return layer


def add_morpho_lesion_points(
    viewer: Any,
    stage7_dir: Path,
    *,
    reference_layer: Any | None,
    point_size: float = 1.5,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """The detected lesions (``lesions.csv``) as points at their midpoints: orange stenoses, blue
    enlargements, opaque in proportion to their confidence, all *point_size* across. The layer
    controls' point size resizes them all (napari's own only resizes a selection)."""
    path = Path(stage7_dir) / "lesions.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or not {"x_mm", "y_mm", "z_mm"} <= set(df.columns):
        return None
    for lyr in list(viewer.layers):
        if lyr.name == MORPHO_LESIONS_LAYER:
            viewer.layers.remove(lyr)
    xyz = np.array(df[["x_mm", "y_mm", "z_mm"]].apply(pd.to_numeric, errors="coerce"), dtype=float)  # writable
    # One point per lesions.csv row (the index is the lesion's); a hypoplasia (an absent vessel, but an A1) has
    # no position: it sits hidden on a finite placeholder so the layer's extent stays sound.
    located = np.isfinite(xyz).all(axis=1)
    if not located.all():
        xyz[~located] = xyz[located].mean(axis=0) if located.any() else 0.0
    data = _vtp_mm_to_data(xyz, reference_layer)
    conf = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    kind = df.get("kind", pd.Series(["stenosis"] * len(df))).astype(str).to_numpy()
    rgba = np.asarray([[*LESION_COLOURS.get(k, (0.6, 0.6, 0.6)), 0.35 + 0.65 * float(np.clip(c, 0, 1))]
                       for k, c in zip(kind, conf)], dtype=float)
    keep = [c for c in ("vessel_name", "kind", "segment", "length_mm", "taper_percent", "diameter_percent",
                        "area_percent", "reference", "confidence", "confidence_class", "confidence_flags")
            if c in df.columns]
    kwargs: dict[str, Any] = {
        "name": MORPHO_LESIONS_LAYER,
        "features": df[keep].reset_index(drop=True),
        "face_color": rgba,
        "border_color": "white",
        "size": float(point_size),
        "symbol": "disc",
        "out_of_slice_display": True,
        "metadata": dict(metadata or {}),
        **_overlay_kwargs(reference_layer),
    }
    try:
        layer = viewer.add_points(data.astype(np.float64), **kwargs)
    except TypeError:
        kwargs.pop("border_color", None)
        layer = viewer.add_points(data.astype(np.float64), **kwargs)
    if not located.all():
        layer.shown = located
    layer.current_size = float(point_size)

    def _resize_all(_event: Any = None) -> None:
        if not layer.selected_data and len(layer.data):
            layer.size = float(layer.current_size)

    layer.events.current_size.connect(_resize_all)
    return layer


def install_morphometrics_viz(
    viewer: Any,
    stage7_dir: Path,
    *,
    reference_layer: Any | None = None,
    color_by: str = "radius",
    point_size: float = DEFAULT_MORPHO_POINT_SIZE,
    edge_width: float = 0.35,
    show_surfaces: bool = True,
    show_lesions: bool = True,
    label_layer: Any | None = None,
) -> dict[str, Any]:
    """Load the morphometrics overlays (centerlines, samples, surfaces, lesions) into Napari and
    attach the summary dock. *label_layer* (default: *reference_layer* when it is a Labels
    layer) gives the surfaces their label colours. The returned dict names the layers made
    (``surface_layer``, ``lesion_layer``) for :mod:`nvitk.gui.viz.morpho_results`."""
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
    if label_layer is None and type(reference_layer).__name__ == "Labels":
        label_layer = reference_layer
    if paths:
        kwargs: dict[str, Any] = {
            "name": MORPHO_PATHS_LAYER,
            "shape_type": "path",
            "edge_color": edge_colors,
            "edge_width": float(edge_width),
            "opacity": 0.9,
            "metadata": meta,
            **_overlay_kwargs(reference_layer),
        }
        shapes = viewer.add_shapes(paths, **kwargs)
        try:
            shapes.editable = False
        except Exception:
            pass

    n_surf = 0
    surface_layer = None
    if show_surfaces:
        try:
            mesh = load_stage7_surface_mesh(stage7_dir, reference_layer=reference_layer)
            if mesh is not None:
                surface_layer = _add_morpho_surface(
                    viewer, *mesh, reference_layer=reference_layer, label_layer=label_layer, metadata=meta,
                )
                n_surf = int(len(np.unique(mesh[2])))
        except Exception as exc:  # noqa: BLE001
            log.warning("Morpho surfaces not loaded: %s", exc)

    lesion_layer = None
    extents_layer = None
    if show_lesions:
        try:
            # Extents first, so the points draw on top of them.
            from nvitk.gui.viz.lesion_inspector import add_lesion_extents, install_lesion_tools, load_lesion_run

            run = load_lesion_run(stage7_dir)
            if run is not None:
                extents_layer = add_lesion_extents(viewer, run, reference_layer=reference_layer, metadata=meta)
            lesion_layer = add_morpho_lesion_points(
                viewer, stage7_dir, reference_layer=reference_layer,
                point_size=max(2.0 * float(point_size), 0.5), metadata=meta,
            )
            if lesion_layer is not None:
                # Hover cards and click-to-inspect (the extents are already drawn).
                install_lesion_tools(viewer, stage7_dir, lesion_layer=lesion_layer, reference_layer=reference_layer,
                                     metadata=meta, extents=False, extents_layer=extents_layer)
        except Exception as exc:  # noqa: BLE001
            log.warning("Morpho lesions not loaded: %s", exc)

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

    from nvitk.gui.viz.morpho_results import vessel_table

    try:
        summary = vessel_table(stage7_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Morphometrics vessel table not loaded: %s", exc)
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
        "n_lesions": int(len(lesion_layer.data)) if lesion_layer is not None else 0,
        "color_by": color_by,
        "n_summary_rows": int(len(summary)),
        "surface_layer": surface_layer,
        "lesion_layer": lesion_layer,
        "lesion_extents_layer": extents_layer,
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
        # Size only: do not touch face_color via init (breaks colormap mode).
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
        **_overlay_kwargs(reference_layer),
    }
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
    layout.addWidget(QLabel(f"Morphometrics\n{stage7_dir}\n(full tables in the results window)"))

    color_row = QHBoxLayout()
    color_row.addWidget(QLabel("Color samples by"))
    color_selector = QComboBox()
    for key in ("radius", "stenosis", "curvature", "area"):
        color_selector.addItem(key, key)
    idx = color_selector.findData(str(color_by))
    color_selector.setCurrentIndex(idx if idx >= 0 else 0)
    color_row.addWidget(color_selector, stretch=1)
    layout.addLayout(color_row)

    # Napari's layer-controls size slider is integer with min=1: too coarse for
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
    if summary.empty:
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Info"])
        table.setRowCount(1)
        table.setItem(0, 0, QTableWidgetItem("No morphometrics table available."))
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
    "MORPHO_LESIONS_LAYER",
    "MORPHO_SURFACES_LAYER",
    "add_morpho_lesion_points",
    "clear_morpho_layers",
    "install_morphometrics_viz",
    "label_colour_lut",
    "load_stage7_centerline_polylines",
    "load_stage7_surface_mesh",
    "load_stage7_surfaces",
]
