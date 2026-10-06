"""Napari layers ↔ nvitk surface types, per-vertex fields, display, and mesh time series.

Coordinates
-----------
Meshes and point clouds the Mesh panel adds live in *world* coordinates (mm,
the space image layers are placed in by their affine), with an identity
transform — so they overlay every image, and measurements are in mm. Layers
from elsewhere (a Surface in voxel indices with an image's affine, Points
clicked in a 4D viewer) are read through their own data→world transform.

Fields and display
------------------
Each surface / points layer carries named per-vertex (per-point) *fields* —
curvature, a distance map, the local diameter, the label a vertex came from,
image values probed onto it — registered with :func:`add_field`. The layer's
*display* (:func:`set_display`) colours it by one field through a colormap, by
label colours, or with one solid colour. A series keeps one array per frame.

Time series
-----------
A :class:`~nvitk.types.MeshSeries` is one 3D Surface layer whose data is the
frame for the current time point. Napari's own 4D surfaces match vertices to the
slider by exact float equality and miss frames whenever the time step is not a
whole number, so the frame is swapped here instead: with a 3D+t image open the
series follows the viewer's time slider (frame *k* at time ``k · t_res``),
otherwise the Mesh panel's frame slider drives it.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Mesh, MeshSeries, PointCloud

#: Metadata key marking layers made by the Mesh panel (and how).
MESH_META_KEY = "nvitk_mesh"

#: Colormaps offered for field colouring (Napari names).
COLORMAPS: tuple[str, ...] = (
    "turbo", "viridis", "magma", "plasma", "inferno", "cividis", "bwr", "PiYG", "twilight", "hsv", "gray",
)
#: Fields every surface / points layer has, computed from the coordinates.
BUILTIN_FIELDS: tuple[str, ...] = ("x (R-L)", "y (A-P)", "z (S-I)")
#: Default solid colours.
SURFACE_COLOUR = (0.86, 0.80, 0.72, 1.0)
POINTS_COLOUR = (0.37, 0.72, 1.0, 1.0)


def layer_kind(layer: Any) -> str:
    """``"mesh"``, ``"series"``, ``"points"``, ``"image"`` or ``""`` for *layer*."""
    if layer is None:
        return ""
    kind = type(layer).__name__
    if kind == "Surface":
        return "series" if series_controller(layer) is not None else "mesh"
    if kind == "Points":
        return "points"
    if kind in ("Image", "Labels"):
        return "image"
    return ""


def _to_world(layer: Any, coords: np.ndarray) -> np.ndarray:
    """*layer*'s data coordinates → its world coordinates, last three dims kept."""
    coords = np.asarray(coords, dtype=float)
    nd = coords.shape[1]
    try:
        mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    except Exception:  # noqa: BLE001 — not a Napari layer
        mat = np.eye(nd + 1)
    if mat.shape == (nd + 1, nd + 1) and not np.allclose(mat, np.eye(nd + 1)):
        coords = coords @ mat[:nd, :nd].T + mat[:nd, nd]
    return coords[:, -3:]


def layer_spatial_affine(layer: Any) -> np.ndarray:
    """4x4 voxel→world affine of an image layer's spatial (last three) dims."""
    mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    aff = np.eye(4)
    aff[:3, :3] = mat[-4:-1, -4:-1]
    aff[:3, 3] = mat[-4:-1, -1]
    return aff


def layer_to_mesh(layer: Any) -> Mesh:
    """A Surface layer (its current frame, for a series) as a world-space :class:`Mesh`.

    Its fields come along as ``point_data``.
    """
    ctrl = series_controller(layer)
    if ctrl is not None:
        mesh = ctrl.current_mesh()
        if _mirrors(layer) or _moved(layer):
            mesh = mesh.with_vertices(_to_world(layer, mesh.vertices))
        return mesh
    data = getattr(layer, "data", None)
    if data is None or len(data) < 2:
        raise ValueError("Select a surface layer.")
    verts = np.asarray(to_numpy(data[0]), dtype=float)
    faces = np.asarray(to_numpy(data[1]), dtype=np.int64)
    if verts.shape[1] > 3:
        # A 4D (time, x, y, z) surface: the faces on screen now.
        try:
            shown = layer._slicing_state._view_faces
            if len(shown):
                faces = np.asarray(shown, dtype=np.int64)
        except Exception:  # noqa: BLE001
            pass
        used = np.unique(faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts, faces = verts[used], remap[faces]
    world = _to_world(layer, verts)
    point_data = {k: v for k, v in fields_of(layer).items() if np.ndim(v) >= 1 and len(v) == len(world)}
    meta = dict((getattr(layer, "metadata", None) or {}).get(MESH_META_KEY, {}) or {})
    meta.update({"name": str(getattr(layer, "name", "mesh")), "space": "world"})
    mesh = Mesh(vertices=world, faces=faces, metadata=meta, point_data=point_data)
    if _mirrors(layer):
        mesh.faces = mesh.faces[:, ::-1].copy()
    return mesh


def _moved(layer: Any) -> bool:
    try:
        mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
        return not np.allclose(mat, np.eye(len(mat)))
    except Exception:  # noqa: BLE001
        return False


def _mirrors(layer: Any) -> bool:
    """True when the layer's transform reflects (its winding flips in world space)."""
    try:
        mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
        return bool(np.linalg.det(mat[-4:-1, -4:-1]) < 0)
    except Exception:  # noqa: BLE001
        return False


def layer_to_point_cloud(layer: Any) -> PointCloud:
    """A Points layer (or a Surface's vertices) as a world-space :class:`PointCloud`."""
    kind = type(layer).__name__
    if kind == "Surface":
        from nvitk.meshlab.convert import mesh_to_point_cloud

        return mesh_to_point_cloud(layer_to_mesh(layer))
    if kind != "Points":
        raise ValueError("Select a points or surface layer.")
    pts = np.asarray(to_numpy(layer.data), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("The points layer needs 3D coordinates.")
    world = _to_world(layer, pts)
    data: dict[str, np.ndarray] = {}
    features = getattr(layer, "features", None)
    if features is not None and len(features) == len(world):
        for col in features.columns:
            values = np.asarray(features[col])
            if np.issubdtype(values.dtype, np.number):
                data[str(col)] = values
    if "nx" in data and "ny" in data and "nz" in data:
        data["normals"] = np.stack([data.pop("nx"), data.pop("ny"), data.pop("nz")], axis=1)
    meta = {"name": str(layer.name), "space": "world"}
    return PointCloud(points=world, metadata=meta, point_data=data)


# ──────────────────────────────────────────────────────────────────────────────
# Fields and display
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Display:
    """How a surface / points layer is coloured."""

    mode: str = "solid"  # "solid" | "field"
    field: str = ""
    colormap: str = "turbo"
    color: tuple[float, float, float, float] = SURFACE_COLOUR
    auto_range: bool = True
    limits: tuple[float, float] = (0.0, 1.0)


@dataclass
class _LayerData:
    #: name → array (static) or list of per-frame arrays (series).
    fields: dict[str, Any] = field(default_factory=dict)
    #: name → {value: rgba} for categorical fields (labels).
    categories: dict[str, dict[int, tuple[float, ...]]] = field(default_factory=dict)
    display: Display = field(default_factory=Display)


_DATA: "weakref.WeakKeyDictionary[Any, _LayerData]" = weakref.WeakKeyDictionary()


def _data(layer: Any) -> _LayerData:
    try:
        entry = _DATA.get(layer)
    except TypeError:
        return _LayerData()
    if entry is None:
        entry = _LayerData()
        if type(layer).__name__ == "Points":
            entry.display.color = POINTS_COLOUR
        _DATA[layer] = entry
    return entry


def _n_elements(layer: Any) -> int:
    kind = type(layer).__name__
    try:
        if kind == "Surface":
            return int(len(layer.data[0]))
        return int(len(layer.data))
    except Exception:  # noqa: BLE001
        return -1


def fields_of(layer: Any, frame: int | None = None) -> dict[str, np.ndarray]:
    """The layer's fields that fit its current vertices / points (one frame of a series)."""
    entry = _data(layer)
    ctrl = series_controller(layer)
    out: dict[str, np.ndarray] = {}
    n = _n_elements(layer)
    for name, val in entry.fields.items():
        if isinstance(val, list):
            k = (ctrl.frame if ctrl is not None else 0) if frame is None else frame
            if 0 <= k < len(val):
                arr = np.asarray(val[k])
            else:
                continue
        else:
            arr = np.asarray(val)
        if ctrl is None and len(arr) != n:
            continue
        out[name] = arr
    if type(layer).__name__ == "Points":
        features = getattr(layer, "features", None)
        if features is not None:
            for col in features.columns:
                vals = np.asarray(features[col])
                if str(col) not in out and len(vals) == n and np.issubdtype(vals.dtype, np.number):
                    out[str(col)] = vals
    return out


def field_names(layer: Any) -> list[str]:
    """Fields the display can colour by: the layer's own, then the coordinates."""
    names = [k for k, v in fields_of(layer).items() if np.ndim(v) == 1 and k not in ("nx", "ny", "nz")]
    return names + list(BUILTIN_FIELDS)


def _field_values(layer: Any, name: str, frame: int | None = None) -> np.ndarray | None:
    if name in BUILTIN_FIELDS:
        axis = BUILTIN_FIELDS.index(name)
        if type(layer).__name__ == "Surface":
            ctrl = series_controller(layer)
            verts = ctrl.series[ctrl.frame if frame is None else frame].vertices if ctrl else np.asarray(layer.data[0])
            return np.asarray(verts, dtype=float)[:, axis] if len(verts) else np.zeros(0)
        return np.asarray(layer.data, dtype=float)[:, -3 + axis]
    vals = fields_of(layer, frame).get(name)
    return None if vals is None else np.asarray(vals)


def add_field(
    layer: Any,
    name: str,
    values: Any,
    *,
    categories: dict[int, tuple[float, ...]] | None = None,
    show: bool = True,
    colormap: str | None = None,
) -> None:
    """Attach a per-vertex / per-point field to *layer* (a list of arrays for a series).

    *categories* maps integer values to RGBA colours (labels). With *show* the
    layer is coloured by it at once.
    """
    entry = _data(layer)
    if isinstance(values, (list, tuple)) and series_controller(layer) is not None:
        entry.fields[name] = [np.asarray(v) for v in values]
    else:
        entry.fields[name] = np.asarray(values)
        if type(layer).__name__ == "Points":
            try:
                feats = layer.features.copy()
                feats[name] = np.asarray(values)
                layer.features = feats
            except Exception:  # noqa: BLE001
                pass
    if categories:
        entry.categories[name] = {int(k): tuple(float(c) for c in v) for k, v in categories.items()}
    if show:
        set_display(layer, mode="field", field=name, auto_range=True,
                    **({"colormap": colormap} if colormap else {}))


def display_of(layer: Any) -> Display:
    return _data(layer).display


def remove_field(layer: Any, name: str) -> bool:
    """Drop field *name* from *layer* (back to a solid colour if it was shown)."""
    entry = _data(layer)
    if name not in entry.fields:
        return False
    del entry.fields[name]
    entry.categories.pop(name, None)
    if entry.display.mode == "field" and entry.display.field == name:
        entry.display.mode, entry.display.field = "solid", ""
    apply_display(layer)
    return True


def _solid_colormap(rgba: tuple[float, ...]) -> Any:
    from napari.utils.colormaps import Colormap

    c = np.asarray(rgba, dtype=float)
    return Colormap(np.stack([c, c]), name=f"nvitk_solid_{'_'.join(f'{v:.3f}' for v in c)}")


def _categorical(values: np.ndarray, lut: dict[int, tuple[float, ...]]) -> tuple[np.ndarray, Any, tuple[float, float]]:
    """Values → (rank + 0.5, zero-interpolated colormap, limits) for exact label colours."""
    from napari.utils.colormaps import Colormap

    ids = sorted(lut)
    rank = {lid: i for i, lid in enumerate(ids)}
    v = np.asarray([rank.get(int(x), 0) for x in np.asarray(values).astype(int)], dtype=np.float32) + 0.5
    colors = np.asarray([lut[i] for i in ids], dtype=float)
    if len(colors) == 1:
        colors = np.vstack([colors, colors])
        ids = ids * 2
    controls = np.linspace(0.0, 1.0, len(colors) + 1)
    cmap = Colormap(colors, controls=controls, interpolation="zero", name=f"nvitk_labels_{id(lut)}")
    return v, cmap, (0.0, float(len(colors)))


def _limits(vals: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def set_display(layer: Any, **changes: Any) -> Display:
    """Update *layer*'s colouring (``mode``, ``field``, ``colormap``, ``color``,
    ``auto_range``, ``limits``) and apply it."""
    disp = _data(layer).display
    for key, val in changes.items():
        setattr(disp, key, val)
    apply_display(layer)
    return disp


def frame_values(layer: Any, frame: int | None = None) -> tuple[np.ndarray | None, Any, tuple[float, float] | None]:
    """``(values, colormap, limits)`` the display asks for (``values`` None = solid)."""
    entry = _data(layer)
    disp = entry.display
    if disp.mode != "field" or not disp.field:
        return None, _solid_colormap(disp.color), (0.0, 1.0)
    vals = _field_values(layer, disp.field, frame)
    if vals is None or vals.ndim != 1:
        return None, _solid_colormap(disp.color), (0.0, 1.0)
    lut = entry.categories.get(disp.field)
    if lut:
        return _categorical(vals, lut)
    if disp.auto_range:
        ctrl = series_controller(layer)
        source = vals
        if ctrl is not None and isinstance(entry.fields.get(disp.field), list):
            source = np.concatenate([np.asarray(v, dtype=float).ravel() for v in entry.fields[disp.field]])
        disp.limits = _limits(source)
    return vals.astype(np.float32), disp.colormap, disp.limits


def apply_display(layer: Any) -> None:
    """Push the layer's display state to Napari."""
    kind = type(layer).__name__
    if kind == "Surface":
        ctrl = series_controller(layer)
        if ctrl is not None:
            ctrl.refresh()
            return
        verts = np.asarray(layer.data[0])
        faces = np.asarray(layer.data[1])
        vals, cmap, limits = frame_values(layer)
        if vals is None or len(vals) != len(verts):
            layer.data = (verts, faces)
        else:
            layer.data = (verts, faces, vals)
        layer.colormap = cmap
        if limits is not None:
            layer.contrast_limits = limits
        return
    if kind == "Points":
        _apply_points_display(layer)


def _apply_points_display(layer: Any) -> None:
    entry = _data(layer)
    disp = entry.display
    n = len(layer.data)
    if not n:
        return
    if disp.mode == "field" and disp.field:
        vals = _field_values(layer, disp.field)
        lut = entry.categories.get(disp.field)
        if vals is not None and lut:
            rgba = np.asarray([lut.get(int(v), (0.6, 0.6, 0.6, 1.0)) for v in vals], dtype=float)
            layer.face_color = rgba
            return
        if vals is not None and len(vals) == n:
            if disp.field not in layer.features.columns:
                feats = layer.features.copy()
                feats[disp.field] = vals
                layer.features = feats
            if disp.auto_range:
                disp.limits = _limits(vals)
            layer.face_color = disp.field
            layer.face_colormap = disp.colormap
            layer.face_contrast_limits = disp.limits
            layer.refresh_colors()
            return
    layer.face_color = np.tile(np.asarray(disp.color, dtype=float), (n, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Adding layers
# ──────────────────────────────────────────────────────────────────────────────


def _surface_kwargs(name: str, extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {MESH_META_KEY: dict(extra_meta or {}, space="world")}
    return {"name": name, "metadata": meta}


def add_mesh_layer(
    viewer: Any,
    mesh: Mesh,
    *,
    name: str | None = None,
    values: np.ndarray | None = None,
    colormap: str = "turbo",
    contrast_limits: tuple[float, float] | None = None,
    opacity: float = 1.0,
    color: tuple[float, ...] | None = None,
    field_name: str = "values",
    categories: dict[int, tuple[float, ...]] | None = None,
) -> Any:
    """Add *mesh* (world coordinates) as a Surface layer.

    Coloured by *values* (stored as the field *field_name*; *categories* for label
    colours), else in one solid *color*. The mesh's own ``point_data`` arrays
    become fields too.
    """
    name = name or mesh.name
    data = (np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64))
    layer = viewer.add_surface(data, opacity=float(opacity),
                               **_surface_kwargs(name, {"source": mesh.metadata.get("name", name)}))
    try:
        layer.shading = "smooth"
    except Exception:  # noqa: BLE001
        pass
    entry = _data(layer)
    if color is not None:
        entry.display.color = tuple(float(c) for c in color)
    for key, arr in mesh.point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == mesh.n_vertices:
            entry.fields[key] = arr
    if values is not None:
        add_field(layer, field_name, values, categories=categories, show=False)
        entry.display.mode, entry.display.field = "field", field_name
        entry.display.colormap = colormap
        if contrast_limits is not None:
            entry.display.auto_range = False
            entry.display.limits = tuple(contrast_limits)
    apply_display(layer)
    return layer


def replace_mesh_layer(layer: Any, mesh: Mesh, values: np.ndarray | None = None) -> None:
    """Write *mesh* (world coordinates) into an existing Surface layer."""
    nd = 3
    try:
        layer.affine = np.eye(nd + 1)
        layer.scale = (1.0,) * nd
        layer.translate = (0.0,) * nd
    except Exception:  # noqa: BLE001
        pass
    entry = _data(layer)
    keep = {k: v for k, v in mesh.point_data.items() if np.ndim(v) == 1 and len(v) == mesh.n_vertices}
    entry.fields = {k: v for k, v in entry.fields.items() if isinstance(v, list)}
    entry.fields.update(keep)
    layer.data = (np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64))
    if values is not None:
        add_field(layer, "values", values)
    else:
        if entry.display.mode == "field" and entry.display.field not in entry.fields \
                and entry.display.field not in BUILTIN_FIELDS:
            entry.display.mode = "solid"
        apply_display(layer)


def add_point_cloud_layer(
    viewer: Any,
    cloud: PointCloud,
    *,
    name: str | None = None,
    color_by: str | None = None,
    size: float | None = None,
    categories: dict[int, tuple[float, ...]] | None = None,
    color: tuple[float, ...] | None = None,
) -> Any:
    """Add *cloud* (world coordinates) as a Points layer; point data become features."""
    import pandas as pd

    name = name or cloud.name
    features = {}
    for key, arr in cloud.point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            features[key] = arr
        elif arr.ndim == 2 and arr.shape[1] == 3 and key == "normals":
            features["nx"], features["ny"], features["nz"] = arr[:, 0], arr[:, 1], arr[:, 2]
    if size is None:
        # About two thirds of the typical gap between points: a dense, readable cloud.
        span = float(np.ptp(cloud.points, axis=0).max()) if cloud.n_points else 1.0
        if 1 < cloud.n_points <= 2_000_000:
            from nvitk.meshlab.pointcloud import sample_spacing

            size = 0.65 * sample_spacing(np.asarray(cloud.points, dtype=float))
        else:
            size = span / 150.0
        size = float(np.clip(size, max(span / 1000.0, 1e-3), max(span / 20.0, 1e-3)))
    kwargs: dict[str, Any] = {
        "name": name,
        "size": float(size),
        "metadata": {MESH_META_KEY: {"space": "world", "kind": "points"}},
        "out_of_slice_display": True,
        "border_width": 0,
    }
    if features:
        kwargs["features"] = pd.DataFrame(features)
    layer = viewer.add_points(np.asarray(cloud.points, dtype=np.float32), **kwargs)
    try:
        layer.shading = "spherical"
    except Exception:  # noqa: BLE001
        pass
    entry = _data(layer)
    if color is not None:
        entry.display.color = tuple(float(c) for c in color)
    if categories and color_by:
        entry.categories[color_by] = {int(k): tuple(v) for k, v in categories.items()}
    if color_by and color_by in features:
        entry.display.mode, entry.display.field = "field", color_by
    apply_display(layer)
    return layer


def add_normals_layer(viewer: Any, cloud: PointCloud, *, name: str, length: float | None = None) -> Any | None:
    """Show a cloud's normals as a Vectors layer."""
    normals = cloud.normals
    if normals is None or not cloud.n_points:
        return None
    if length is None:
        span = float(np.ptp(cloud.points, axis=0).max())
        length = max(span / 60.0, 0.5)
    vec = np.stack([cloud.points, normals * float(length)], axis=1)
    return viewer.add_vectors(vec.astype(np.float32), name=name, edge_width=0.3, edge_color="#ffcc66",
                              length=1.0, out_of_slice_display=True)


def add_polylines_layer(viewer: Any, polylines: list[np.ndarray], *, name: str, color: str = "#ffd34d",
                        width: float = 0.5) -> Any | None:
    """Polylines (world mm) as a Vectors layer of segments — lines that show in 3D."""
    segs = []
    for line in polylines:
        line = np.asarray(line, dtype=float)
        if len(line) >= 2:
            segs.append(np.stack([line[:-1], line[1:] - line[:-1]], axis=1))
    if not segs:
        return None
    return viewer.add_vectors(np.concatenate(segs).astype(np.float32), name=name, edge_width=float(width),
                              edge_color=color, length=1.0, out_of_slice_display=True)


# ──────────────────────────────────────────────────────────────────────────────
# Time series
# ──────────────────────────────────────────────────────────────────────────────

_CONTROLLERS: "weakref.WeakKeyDictionary[Any, MeshSeriesController]" = weakref.WeakKeyDictionary()


def go_to_frame(viewer: Any, layer: Any, frame: int) -> None:
    """Show frame *frame* of a mesh series: through the viewer's time slider when a
    3D+t image drives it, else on the series alone."""
    ctrl = series_controller(layer)
    if ctrl is None:
        return
    frame = int(np.clip(frame, 0, len(ctrl.series) - 1))
    dim = viewer_time_dim(viewer)
    if dim is not None:
        try:
            t = float(ctrl.series.times[frame])
            step = float(viewer.dims.range[dim].step) or 1.0
            start = float(viewer.dims.range[dim].start)
            viewer.dims.set_current_step(dim, int(round((t - start) / step)))
            return
        except Exception:  # noqa: BLE001
            pass
    ctrl.set_frame(frame)


def series_controller(layer: Any) -> MeshSeriesController | None:
    """The controller animating *layer*, if it shows a mesh series."""
    if layer is None:
        return None
    try:
        return _CONTROLLERS.get(layer)
    except TypeError:
        return None


def viewer_time_dim(viewer: Any) -> int | None:
    """The viewer's time dim (the leading one when a 3D+t layer makes it 4D)."""
    dims = getattr(viewer, "dims", None)
    if dims is None or int(dims.ndim) < 4:
        return None
    return 0


class MeshSeriesController:
    """Keeps a Surface layer on the current frame of a :class:`MeshSeries`."""

    def __init__(self, viewer: Any, layer: Any, series: MeshSeries) -> None:
        self._viewer = viewer
        self._layer_ref = weakref.ref(layer)
        self.series = series
        self.frame = -1
        #: Called with the frame index whenever it changes (the panel's slider).
        self.on_frame: Callable[[int], None] | None = None
        self._follow = True
        viewer.dims.events.current_step.connect(self._on_dims)
        self.set_frame(self._frame_from_viewer() or 0)

    @property
    def layer(self) -> Any | None:
        return self._layer_ref()

    def _frame_from_viewer(self) -> int | None:
        dim = viewer_time_dim(self._viewer)
        if dim is None:
            return None
        try:
            t = float(self._viewer.dims.point[dim])
        except Exception:  # noqa: BLE001
            return None
        times = np.asarray(self.series.times, dtype=float)
        return int(np.argmin(np.abs(times - t)))

    def _on_dims(self, _event: Any = None) -> None:
        if self.layer is None:
            try:
                self._viewer.dims.events.current_step.disconnect(self._on_dims)
            except Exception:  # noqa: BLE001
                pass
            return
        frame = self._frame_from_viewer()
        if frame is not None and self._follow:
            self.set_frame(frame)

    def current_mesh(self) -> Mesh:
        frame = max(self.frame, 0)
        mesh = self.series[frame].copy()
        mesh.metadata["space"] = "world"
        layer = self.layer
        if layer is not None:
            for key, arr in fields_of(layer, frame).items():
                if np.ndim(arr) == 1 and len(arr) == mesh.n_vertices:
                    mesh.point_data[key] = np.asarray(arr)
        return mesh

    def refresh(self) -> None:
        """Re-draw the current frame (after a display change)."""
        frame, self.frame = self.frame, -1
        self.set_frame(max(frame, 0))

    def set_frame(self, frame: int) -> None:
        """Show frame *frame* (clamped to the series)."""
        layer = self.layer
        if layer is None:
            return
        frame = int(np.clip(frame, 0, len(self.series) - 1))
        if frame == self.frame:
            return
        self.frame = frame
        mesh = self.series[frame]
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vals, cmap, limits = frame_values(layer, frame)
        if not len(faces):
            # Napari cannot show an empty surface: a degenerate stand-in.
            verts = np.zeros((3, 3), dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            vals = None if vals is None else np.zeros(3, np.float32)
        if vals is not None and len(vals) != len(verts):
            vals = np.zeros(len(verts), np.float32)
        layer.data = (verts, faces) if vals is None else (verts, faces, vals)
        try:
            layer.colormap = cmap
            if limits is not None:
                layer.contrast_limits = limits
        except Exception:  # noqa: BLE001
            pass
        if self.on_frame is not None:
            self.on_frame(frame)


def add_mesh_series_layer(
    viewer: Any,
    series: MeshSeries,
    *,
    name: str | None = None,
    values: list[np.ndarray] | None = None,
    colormap: str = "turbo",
    field_name: str = "values",
    categories: dict[int, tuple[float, ...]] | None = None,
    color: tuple[float, ...] | None = None,
) -> Any:
    """Add a :class:`MeshSeries` as one animated Surface layer (fields per frame)."""
    first = series[0]
    seed = first if first.n_faces else Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    layer = viewer.add_surface(
        (np.asarray(seed.vertices, np.float32), np.asarray(seed.faces, np.int64)),
        **_surface_kwargs(name or series.name, {"kind": "series", "n_frames": len(series),
                                                "t_res": float(series.metadata.get("t_res", 1.0) or 1.0)}),
    )
    try:
        layer.shading = "smooth"
    except Exception:  # noqa: BLE001
        pass
    entry = _data(layer)
    if color is not None:
        entry.display.color = tuple(float(c) for c in color)
    for key in first.point_data:
        per_frame = [f.point_data.get(key) for f in series]
        if all(v is not None and np.ndim(v) == 1 for v in per_frame):
            entry.fields[key] = [np.asarray(v) for v in per_frame]
    if values is not None:
        entry.fields[field_name] = [np.asarray(v) for v in values]
        entry.display.mode, entry.display.field, entry.display.colormap = "field", field_name, colormap
    if categories:
        entry.categories[field_name] = {int(k): tuple(v) for k, v in categories.items()}
    _CONTROLLERS[layer] = MeshSeriesController(viewer, layer, series)
    return layer


__all__ = [
    "BUILTIN_FIELDS",
    "COLORMAPS",
    "MESH_META_KEY",
    "Display",
    "MeshSeriesController",
    "add_field",
    "add_mesh_layer",
    "add_mesh_series_layer",
    "add_normals_layer",
    "add_point_cloud_layer",
    "add_polylines_layer",
    "apply_display",
    "go_to_frame",
    "remove_field",
    "display_of",
    "field_names",
    "fields_of",
    "layer_kind",
    "layer_spatial_affine",
    "layer_to_mesh",
    "layer_to_point_cloud",
    "replace_mesh_layer",
    "series_controller",
    "set_display",
    "viewer_time_dim",
]
