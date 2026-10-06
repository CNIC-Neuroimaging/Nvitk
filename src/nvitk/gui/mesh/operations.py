"""The Mesh panel's operations: what each does, what it takes, and how it runs.

Every operation is a :class:`MeshOp` — an id, a category and label for the
pickers, the kinds of active layer it accepts, its parameters (the Tools
panel's :class:`~nvitk.gui.tools.registry.ParamSpec`), whether it needs a point
clicked in the viewer (:attr:`MeshOp.pick`), and a runner. Runners read the
active layer through :class:`OpContext` (always in world coordinates), call
:mod:`nvitk.meshlab`, and either add layers or attach per-vertex fields to the
active one (curvature, distances, diameters, image values); what they return is
the line for the log and, optionally, a table and a plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from nvitk.gui.mesh.layers import (
    _to_world,
    add_field,
    add_mesh_layer,
    add_mesh_series_layer,
    add_normals_layer,
    add_point_cloud_layer,
    add_polylines_layer,
    apply_display,
    layer_kind,
    layer_spatial_affine,
    layer_to_mesh,
    layer_to_point_cloud,
    replace_mesh_layer,
    series_controller,
)
from nvitk.gui.tools.registry import ParamSpec
from nvitk.types import Mesh, MeshSeries, PointCloud

_P = ParamSpec
_LAYER_NONE = "(none)"

#: Display order of the categories.
CATEGORIES: tuple[str, ...] = (
    "Create",
    "Edit (click on the surface)",
    "Select (on the surface)",
    "Clean & repair",
    "Smooth",
    "Remesh",
    "Transform & align",
    "Measure",
    "Vessels & tubes",
    "Compare",
    "Convert (image ↔ mesh ↔ points)",
    "Point cloud",
    "Time (3D+t)",
)

#: How "Surfaces from labels" lays its output out.
SURFACE_OUTPUTS: tuple[str, ...] = (
    "one layer per label (in a folder)",
    "one layer, coloured by label",
    "merged into one surface",
    "groups (one layer per group)",
)


@dataclass
class OpResult:
    """What a run reports: a message, optionally a table (flat, or one row per key) and a plot."""

    message: str
    table: dict[str, Any] | None = None
    title: str = ""
    plot: dict[str, Any] | None = None
    row_header: str = ""
    #: Called with a table row's key when that row is clicked (``None`` when the
    #: highlight should go): shows the branch, frame… it describes in the viewer.
    on_row: Callable[[str | None], None] | None = None


@dataclass
class OpContext:
    """The active layer and viewer an operation runs against."""

    viewer: Any
    layer: Any
    params: dict[str, Any]
    replace: bool = False
    #: The point clicked in the viewer (world x, y, z), for operations that pick one.
    point: np.ndarray | None = None
    #: The camera's view direction (world x, y, z) when the point was clicked.
    view_direction: np.ndarray | None = None
    #: Called with the layer and its previous content before an edit replaces it (undo).
    remember: Callable[[Any, Any], None] | None = None

    # -- inputs -----------------------------------------------------------------

    def layer_named(self, key: str, *, required: bool = True) -> Any | None:
        name = str(self.params.get(key) or "").strip()
        if not name or name == _LAYER_NONE:
            if required:
                raise ValueError("Pick the reference layer.")
            return None
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"Layer not found: {name}")

    def mesh(self, layer: Any | None = None) -> Mesh:
        layer = self.layer if layer is None else layer
        if type(layer).__name__ != "Surface":
            raise ValueError("Select a surface (mesh) layer.")
        return layer_to_mesh(layer)

    def cloud(self, layer: Any | None = None) -> PointCloud:
        return layer_to_point_cloud(self.layer if layer is None else layer)

    def surface(self, layer: Any | None = None) -> Mesh | PointCloud:
        layer = self.layer if layer is None else layer
        return self.mesh(layer) if type(layer).__name__ == "Surface" else self.cloud(layer)

    def series(self) -> MeshSeries:
        ctrl = series_controller(self.layer)
        if ctrl is None:
            raise ValueError("Select a mesh time series (made from a 3D+t mask, or opened from a series).")
        return ctrl.series

    def at(self) -> np.ndarray:
        """The picked point, else the viewer's cursor (world x, y, z)."""
        if self.point is not None:
            return np.asarray(self.point, dtype=float)
        pos = np.asarray(getattr(self.viewer.cursor, "position", ()) or (), dtype=float)
        if pos.size < 3:
            pos = np.asarray(self.viewer.dims.point, dtype=float)
        return pos[-3:]

    @property
    def name(self) -> str:
        return str(getattr(self.layer, "name", "mesh"))

    # -- outputs ----------------------------------------------------------------

    def put_mesh(self, mesh: Mesh, suffix: str, *, values: np.ndarray | None = None, colormap: str = "turbo",
                 force_replace: bool = False) -> Any:
        """Add *mesh* as a new layer, or write it into the active one (replace mode)."""
        replace = (self.replace or force_replace) and type(self.layer).__name__ == "Surface" \
            and series_controller(self.layer) is None
        if replace:
            if self.remember is not None:
                self.remember(self.layer, layer_to_mesh(self.layer))
            replace_mesh_layer(self.layer, mesh, values)
            return self.layer
        return add_mesh_layer(self.viewer, mesh, name=f"{self.name}_{suffix}", values=values, colormap=colormap)

    def put_cloud(self, cloud: PointCloud, suffix: str, *, color_by: str | None = None,
                  force_replace: bool = False) -> Any:
        if (self.replace or force_replace) and type(self.layer).__name__ == "Points":
            if self.remember is not None:
                self.remember(self.layer, layer_to_point_cloud(self.layer))
            set_points(self.layer, cloud)
            return self.layer
        return add_point_cloud_layer(self.viewer, cloud, name=f"{self.name}_{suffix}", color_by=color_by)

    def put_series(self, series: MeshSeries, suffix: str, *, values: list[np.ndarray] | None = None,
                   field_name: str = "values") -> Any:
        return add_mesh_series_layer(self.viewer, series, name=f"{self.name}_{suffix}", values=values,
                                     field_name=field_name)


def set_points(layer: Any, cloud: PointCloud) -> None:
    """Write *cloud* (world) into a Points layer, keeping its numeric data as features."""
    import pandas as pd

    nd = int(getattr(layer, "ndim", 3) or 3)
    try:
        layer.affine = np.eye(nd + 1)
        layer.scale = (1.0,) * nd
        layer.translate = (0.0,) * nd
    except Exception:  # noqa: BLE001
        pass
    feats = {k: np.asarray(v) for k, v in cloud.point_data.items() if np.ndim(v) == 1}
    layer.data = np.asarray(cloud.points, dtype=np.float32)
    layer.features = pd.DataFrame(feats) if feats else pd.DataFrame(index=range(cloud.n_points))
    apply_display(layer)


@dataclass(frozen=True)
class MeshOp:
    """One operation of the Mesh panel."""

    id: str
    category: str
    label: str
    inputs: tuple[str, ...]
    params: tuple[ParamSpec, ...]
    run: Callable[[OpContext], OpResult]
    description: str = ""
    #: Writes a mesh that "replace active" may put back into the same layer.
    replaceable: bool = False
    #: ``"surface"``: Run waits for a click on the active layer (or anywhere in 2D).
    pick: str = ""
    #: Online documentation (MeshLab filters).
    url: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _vec3(text: Any, default: float = 0.0) -> np.ndarray:
    """``"1, 2, 3"`` / ``"2"`` → a 3-vector."""
    parts = [p for p in str(text or "").replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    if not parts:
        return np.full(3, float(default))
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise ValueError(f"Expected one or three numbers, got {text!r}.")
    return np.asarray(vals)


_AXES = {"x (R-L)": 0, "y (A-P)": 1, "z (S-I / H-F)": 2}
_PLANES = ("vessel axis (auto)", "view direction", *_AXES)


def _plane_normal(ctx: OpContext, choice: str, mesh: Mesh | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(origin, normal)`` of the plane through the picked point."""
    origin = ctx.at()
    if choice == "view direction" and ctx.view_direction is not None:
        n = np.asarray(ctx.view_direction, dtype=float)
        return origin, n / (np.linalg.norm(n) or 1.0)
    if choice == "vessel axis (auto)" and mesh is not None:
        from nvitk.meshlab.vessels import vessel_section_at

        sec = vessel_section_at(mesh, origin)
        return np.asarray(sec["centre"]), np.asarray(sec["axis"])
    v = np.zeros(3)
    v[_AXES.get(str(choice), 2)] = 1.0
    return origin, v


def _fmt(v: Any, unit: str = "") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:,.4g}{unit}"


def _image_volumes(layer: Any) -> tuple[list[np.ndarray], int | None]:
    """A Labels/Image layer's 3D volumes (one per frame for 3D+t) and its time axis."""
    from nvitk.core.array import to_numpy
    from nvitk.gui.core.orientation import layer_is_time_leading

    data = to_numpy(layer.data)
    if data.ndim == 3:
        return [data], None
    if data.ndim == 4:
        t_ax = 0 if layer_is_time_leading(layer) else 3
        # Views by basic indexing, not np.take: that would first copy the whole
        # (Fortran-ordered, for NIfTI) 4D array once per frame.
        frames = [data[k] if t_ax == 0 else data[..., k] for k in range(data.shape[t_ax])]
        return frames, t_ax
    raise ValueError("Select a 3D or 3D+t image / labels layer.")


def _layer_time_step(layer: Any, t_ax: int | None) -> float:
    """Seconds (or units) per frame of a 3D+t layer, from its placement."""
    if t_ax is None:
        return 1.0
    try:
        mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
        return float(abs(mat[t_ax, t_ax])) or 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def _empty_mesh() -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3)))


def _mask_mesh(layer: Any, mask: np.ndarray, *, frame: int | None, t_ax: int | None, step: int,
               close: bool = True, level: float = 0.5) -> Mesh | None:
    """Marching cubes of one 3D mask (or volume at *level*) of *layer*, in world mm.

    Cropped to the structure's bounding box first (fast for small structures in
    big volumes); *close* pads it so surfaces cut by the image border are capped.
    """
    from nvitk.meshlab.cleaning import ensure_outward
    from nvitk.meshlab.marching_cubes import _measure

    above = mask if mask.dtype == bool else mask > level
    coords = np.nonzero(above)
    if not len(coords[0]):
        return None
    lo = [max(int(c.min()) - 1, 0) for c in coords]
    hi = [min(int(c.max()) + 2, n) for c, n in zip(coords, mask.shape)]
    sub = np.asarray(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    offset = np.asarray(lo, dtype=float)
    if close:
        fill = 0.0 if mask.dtype == bool or level == 0.5 else float(min(sub.min(), level - 1.0))
        sub = np.pad(sub, 1, constant_values=fill)
        offset -= 1.0
    try:
        verts, faces, _n, _v = _measure().marching_cubes(sub, level=float(level), step_size=int(step),
                                                         allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    pts = verts.astype(float) + offset
    if t_ax is not None:
        pts = np.insert(pts, t_ax, float(frame or 0), axis=1)
    world = _to_world(layer, pts)
    return ensure_outward(Mesh(vertices=world, faces=faces, metadata={"space": "world"}))


def _finish(mesh: Mesh, params: dict[str, Any]) -> Mesh:
    """Optional Taubin smoothing and decimation after marching cubes."""
    from nvitk.meshlab.remeshing import decimate
    from nvitk.meshlab.smoothing import taubin_smooth

    if not mesh.n_faces:
        return mesh
    it = int(params.get("smooth_iterations") or 0)
    if it > 0:
        mesh = taubin_smooth(mesh, iterations=it)
    red = float(params.get("decimate_reduction") or 0.0)
    if red > 0:
        mesh = decimate(mesh, red)
    return mesh


def label_info(layer: Any, ids: list[int]) -> dict[int, tuple[str, tuple[float, float, float, float]]]:
    """``{id: (name, rgba)}``: names from the layer's label vocabulary, colours as drawn."""
    from nvitk.gui.labels.catalog import get_schema, layer_schema_key
    from nvitk.gui.labels.visibility import get_label_color, supports_per_label_color

    key = None
    try:
        key = layer_schema_key(layer)
    except Exception:  # noqa: BLE001
        pass
    schema = get_schema(key) if key else None
    out = {}
    for lid in ids:
        name = (schema.name_for(lid) if schema else None) or f"label {lid}"
        rgba = (0.85, 0.75, 0.65, 1.0)
        try:
            if supports_per_label_color(layer):
                c = np.asarray(get_label_color(layer, lid), dtype=float).ravel()
                rgba = (float(c[0]), float(c[1]), float(c[2]), 1.0)
            else:
                lo, hi = (float(v) for v in layer.contrast_limits)
                t = 0.0 if hi <= lo else min(max((lid - lo) / (hi - lo), 0.0), 1.0)
                c = np.asarray(layer.colormap.map([t])[0], dtype=float)
                rgba = (float(c[0]), float(c[1]), float(c[2]), 1.0)
        except Exception:  # noqa: BLE001
            pass
        out[int(lid)] = (str(name), rgba)
    return out


def present_label_ids(layer: Any) -> list[int]:
    """The label ids an image / labels layer holds (non-zero integers)."""
    from nvitk.gui.labels.visibility import layer_label_ids

    try:
        return [int(i) for i in layer_label_ids(layer)]
    except Exception:  # noqa: BLE001
        return []


def parse_groups(text: str, available: list[int]) -> list[tuple[str, list[int]]]:
    """``"Left: 1,2 ; Right: 3-5 ; 7,8"`` → ``[("Left", [1, 2]), ("Right", [3, 4, 5]), ("group 3", [7, 8])]``."""
    groups = []
    for k, chunk in enumerate([c for c in str(text or "").replace("|", ";").split(";") if c.strip()], 1):
        name, _, spec = chunk.rpartition(":")
        name = name.strip() or f"group {k}"
        ids: list[int] = []
        for tok in spec.replace(" ", "").split(","):
            if not tok:
                continue
            if "-" in tok.lstrip("-"):
                a, b = tok.split("-", 1)
                ids.extend(range(int(a), int(b) + 1))
            else:
                ids.append(int(tok))
        ids = [i for i in ids if not available or i in available]
        if ids:
            groups.append((name, ids))
    return groups


def _metrics_table(metrics: dict[str, Any]) -> dict[str, Any]:
    """Readable keys and values for the results window."""
    out: dict[str, Any] = {}
    for key, val in metrics.items():
        label = key.replace("_", " ")
        if isinstance(val, (list, tuple, np.ndarray)):
            out[label] = ", ".join(_fmt(v) for v in val)
        elif isinstance(val, bool):
            out[label] = "yes" if val else "no"
        elif isinstance(val, float):
            out[label] = _fmt(val)
        else:
            out[label] = val
    return out


def _reference_grid(ctx: OpContext, key: str, voxel: float, bounds: np.ndarray, margin: float = 2.0):
    """``(shape, affine, description)`` of the output grid: a reference image's, or a new one."""
    ref = ctx.layer_named(key, required=False)
    if ref is not None:
        if type(ref).__name__ not in ("Image", "Labels"):
            raise ValueError("The grid comes from an image or labels layer.")
        shape = tuple(int(v) for v in np.shape(ref.data)[-3:])
        return shape, layer_spatial_affine(ref), f"grid of {ref.name}"
    vs = float(voxel) if voxel and voxel > 0 else max(float(np.ptp(bounds, axis=0).max()) / 200.0, 1e-3)
    lo = bounds[0] - margin * vs
    shape = tuple(int(np.ceil((h - l) / vs)) + 2 * int(margin) + 1 for l, h in zip(bounds[0], bounds[1]))
    aff = np.diag([vs, vs, vs, 1.0])
    aff[:3, 3] = lo
    return shape, aff, f"{vs:.3g} mm grid"


def _add_image(ctx: OpContext, data: np.ndarray, affine: np.ndarray, name: str, *, labels: bool,
               colormap: str = "gray") -> Any:
    if labels:
        lab = ctx.viewer.add_labels(np.asarray(data).astype(np.int32), name=name, affine=affine, opacity=0.6)
        lab._nvitk_label_like = True
        return lab
    return ctx.viewer.add_image(np.asarray(data, dtype=np.float32), name=name, affine=affine, colormap=colormap)


def _gather(ctx: OpContext, layers: list[Any], name: str) -> None:
    """Put several new layers in one layer-list folder."""
    folders = getattr(ctx.viewer, "_nvitk_layer_folders", None)
    if folders is not None and len(layers) > 1:
        try:
            folders.create_folder(layers, name)
        except Exception:  # noqa: BLE001 — folders are a convenience
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Create
# ──────────────────────────────────────────────────────────────────────────────


def _run_surfaces(ctx: OpContext) -> OpResult:
    from nvitk.gui.labels.visibility import is_label_like_layer

    layer = ctx.layer
    volumes, t_ax = _image_volumes(layer)
    step = int(ctx.params.get("mc_step") or 1)
    close = bool(ctx.params.get("close_borders", True))
    is_labels = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
    present = present_label_ids(layer) if is_labels else []
    chosen = [int(i) for i in (ctx.params.get("labels") or []) if int(i) in present] or present
    mode = str(ctx.params.get("surface_output") or SURFACE_OUTPUTS[0])
    use_colours = bool(ctx.params.get("label_colours", True))
    info = label_info(layer, chosen) if chosen else {}

    if not is_labels or not chosen:
        targets: list[tuple[str, list[int] | None, Any]] = [(ctx.name + "_surface", None, None)]
        mode = SURFACE_OUTPUTS[2]
    elif mode == SURFACE_OUTPUTS[3]:
        groups = parse_groups(str(ctx.params.get("label_groups") or ""), present)
        if not groups:
            raise ValueError("Write the groups, e.g.  Left: 1,2 ; Right: 3-5")
        all_info = label_info(layer, sorted({i for _n, ids in groups for i in ids}))
        targets = [(name, ids, all_info[ids[0]][1]) for name, ids in groups]
    elif mode == SURFACE_OUTPUTS[2]:
        targets = [(f"{ctx.name}_surface", chosen, info[chosen[0]][1] if len(chosen) == 1 else None)]
    else:
        targets = [(info[i][0], [i], info[i][1]) for i in chosen]

    def _select(vol: np.ndarray, ids: list[int] | None) -> np.ndarray:
        if ids is None:
            return vol != 0
        return np.isin(vol, ids) if len(ids) > 1 else vol == ids[0]

    def _build(ids: list[int] | None) -> Mesh | MeshSeries | None:
        if t_ax is None:
            m = _mask_mesh(layer, _select(volumes[0], ids), frame=None, t_ax=None, step=step, close=close)
            return _finish(m, ctx.params) if m is not None else None
        frames = []
        for k, vol in enumerate(volumes):
            m = _mask_mesh(layer, _select(vol, ids), frame=k, t_ax=t_ax, step=step, close=close)
            frames.append(_finish(m, ctx.params) if m is not None else _empty_mesh())
        if not any(f.n_faces for f in frames):
            return None
        return MeshSeries(frames=frames, metadata={"t_res": _layer_time_step(layer, t_ax)})

    added: list[Any] = []
    if mode == SURFACE_OUTPUTS[1]:
        # One layer: every label's surface, coloured by the label it came from.
        lut = {i: info[i][1] for i in chosen} if use_colours else None
        pieces = {i: _build([i]) for i in chosen}
        pieces = {i: p for i, p in pieces.items() if p is not None}
        if not pieces:
            raise ValueError("Nothing to mesh: the selected labels are empty.")

        def _combine(meshes: dict[int, Mesh]) -> tuple[Mesh, np.ndarray]:
            verts, faces, labs, off = [], [], [], 0
            for i, m in meshes.items():
                verts.append(m.vertices)
                faces.append(m.faces + off)
                labs.append(np.full(m.n_vertices, i))
                off += m.n_vertices
            if not verts:
                return _empty_mesh(), np.zeros(0)
            return Mesh(np.vstack(verts), np.vstack(faces), metadata={"space": "world"}), np.concatenate(labs)

        if t_ax is None:
            merged, labs = _combine(pieces)  # type: ignore[arg-type]
            added.append(add_mesh_layer(ctx.viewer, merged, name=f"{ctx.name}_surfaces", values=labs,
                                        field_name="label", categories=lut))
        else:
            frames, labels = [], []
            for k in range(len(volumes)):
                merged, labs = _combine({i: s[k] for i, s in pieces.items()})  # type: ignore[index]
                frames.append(merged)
                labels.append(labs)
            series = MeshSeries(frames=frames, metadata={"t_res": _layer_time_step(layer, t_ax)})
            lyr = add_mesh_series_layer(ctx.viewer, series, name=f"{ctx.name}_surfaces", values=labels,
                                        field_name="label", categories=lut)
            apply_display(lyr)
            added.append(lyr)
    else:
        for name, ids, rgba in targets:
            built = _build(ids)
            if built is None:
                continue
            colour = rgba if (use_colours and rgba is not None) else None
            if isinstance(built, MeshSeries):
                lyr = add_mesh_series_layer(ctx.viewer, built, name=name, color=colour)
                apply_display(lyr)
            else:
                lyr = add_mesh_layer(ctx.viewer, built, name=name, color=colour)
            added.append(lyr)
        if not added:
            raise ValueError("Nothing to mesh: the mask (or the selected labels) is empty.")
        _gather(ctx, added, f"{ctx.name} surfaces")
    kind = "time series" if t_ax is not None else "surface"
    return OpResult(f"Added {len(added)} {kind} layer{'s' if len(added) != 1 else ''} from {ctx.name} ({mode}).")


def _run_isosurface(ctx: OpContext) -> OpResult:
    from skimage.filters import threshold_otsu

    volumes, t_ax = _image_volumes(ctx.layer)
    level = float(ctx.params.get("iso_level") or 0.0)
    if level == 0.0:
        sample = np.asarray(volumes[0][:: max(1, volumes[0].shape[0] // 64)], dtype=float)
        level = float(threshold_otsu(sample))
    step = int(ctx.params.get("mc_step") or 1)
    if t_ax is None:
        mesh = _mask_mesh(ctx.layer, volumes[0], frame=None, t_ax=None, step=step, close=False, level=level)
        if mesh is None:
            raise ValueError(f"No surface at level {level:g}.")
        add_mesh_layer(ctx.viewer, _finish(mesh, ctx.params), name=f"{ctx.name}_iso")
    else:
        frames = []
        for k, vol in enumerate(volumes):
            m = _mask_mesh(ctx.layer, vol, frame=k, t_ax=t_ax, step=step, close=False, level=level)
            frames.append(_finish(m, ctx.params) if m is not None else _empty_mesh())
        series = MeshSeries(frames=frames, metadata={"t_res": _layer_time_step(ctx.layer, t_ax)})
        apply_display(add_mesh_series_layer(ctx.viewer, series, name=f"{ctx.name}_iso"))
    return OpResult(f"Isosurface of {ctx.name} at {level:g}.")


def _run_hull(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import volume
    from nvitk.meshlab.remeshing import convex_hull

    mesh = convex_hull(ctx.surface())
    ctx.put_mesh(mesh, "hull")
    return OpResult(f"Convex hull: {mesh.n_faces} faces, volume {_fmt(volume(mesh), ' mm³')}.")


# ──────────────────────────────────────────────────────────────────────────────
# Edit (picked point)
# ──────────────────────────────────────────────────────────────────────────────


def _run_erase(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import erase_points, erase_sphere

    r = float(ctx.params.get("edit_radius") or 5.0)
    keep = bool(ctx.params.get("edit_keep_inside"))
    if type(ctx.layer).__name__ == "Points":
        cloud = ctx.cloud()
        out = erase_points(cloud, ctx.at(), r, keep_inside=keep)
        ctx.put_cloud(out, "edited", force_replace=True)
        return OpResult(f"{cloud.n_points - out.n_points} point(s) removed within {r:g} mm.")
    mesh = ctx.mesh()
    out = erase_sphere(mesh, ctx.at(), r, keep_inside=keep)
    if not out.n_faces:
        raise ValueError("That would remove the whole surface.")
    ctx.put_mesh(out, "edited", force_replace=True)
    return OpResult(f"{mesh.n_faces - out.n_faces} face(s) removed within {r:g} mm of the click.")


def _run_piece(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import keep_piece

    mesh = ctx.mesh()
    delete = str(ctx.params.get("piece_action")) == "delete it"
    out = keep_piece(mesh, ctx.at(), delete=delete)
    if not out.n_faces:
        raise ValueError("Nothing would be left.")
    ctx.put_mesh(out, "piece", force_replace=True)
    return OpResult(("Deleted" if delete else "Kept") + f" the piece under the click ({mesh.n_faces - out.n_faces} faces removed).")


def _run_sculpt(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import sculpt

    mesh = ctx.mesh()
    r = float(ctx.params.get("edit_radius") or 5.0)
    amount = float(ctx.params.get("sculpt_amount") or 1.0)
    ctx.put_mesh(sculpt(mesh, ctx.at(), r, amount), "sculpted", force_replace=True)
    return OpResult(f"{'Pushed' if amount > 0 else 'Pulled'} the surface by {abs(amount):g} mm (radius {r:g} mm).")


def _run_smooth_spot(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import smooth_spot

    mesh = ctx.mesh()
    r = float(ctx.params.get("edit_radius") or 5.0)
    out = smooth_spot(mesh, ctx.at(), r, iterations=int(ctx.params.get("spot_iterations") or 10))
    ctx.put_mesh(out, "smoothed", force_replace=True)
    return OpResult(f"Smoothed within {r:g} mm of the click.")


# ──────────────────────────────────────────────────────────────────────────────
# Select
# ──────────────────────────────────────────────────────────────────────────────


def _selection_or_fail(ctx: OpContext) -> np.ndarray:
    from nvitk.gui.mesh.selection import selection_of

    sel = selection_of(ctx.layer)
    if not sel.any():
        raise ValueError(f"Nothing is selected on {ctx.name}: select with the brush, a box or a click first.")
    return sel


def _run_sel_interactive(ctx: OpContext) -> OpResult:
    raise ValueError("Run arms it, then select in the viewer.")


def _run_sel_piece(ctx: OpContext) -> OpResult:
    from nvitk.gui.mesh.selection import combine, selection_of, set_selection
    from nvitk.meshlab.edit import connected_selection

    picked = connected_selection(ctx.mesh(), ctx.at())
    n = set_selection(ctx.layer, combine(selection_of(ctx.layer), picked,
                                         str(ctx.params.get("sel_mode") or "add")))
    return OpResult(f"{n:,} vertices selected on {ctx.name}.")


def _run_sel_invert(ctx: OpContext) -> OpResult:
    from nvitk.gui.mesh.selection import selection_of, set_selection

    n = set_selection(ctx.layer, ~selection_of(ctx.layer))
    return OpResult(f"Selection inverted: {n:,} selected on {ctx.name}.")


def _run_sel_all(ctx: OpContext) -> OpResult:
    from nvitk.gui.mesh.selection import selection_of, set_selection

    n = set_selection(ctx.layer, np.ones(len(selection_of(ctx.layer)), dtype=bool))
    return OpResult(f"All {n:,} selected on {ctx.name}.")


def _run_sel_clear(ctx: OpContext) -> OpResult:
    from nvitk.gui.mesh.selection import clear_selection

    clear_selection(ctx.layer)
    return OpResult(f"Selection cleared on {ctx.name}.")


def _run_sel_grow(ctx: OpContext) -> OpResult:
    from nvitk.gui.mesh.selection import set_selection
    from nvitk.meshlab.edit import grow_selection

    rings = int(ctx.params.get("sel_rings") or 1)
    if str(ctx.params.get("sel_grow_mode") or "grow") == "shrink":
        rings = -rings
    n = set_selection(ctx.layer, grow_selection(ctx.mesh(), _selection_or_fail(ctx), rings))
    return OpResult(f"Selection {'grown' if rings > 0 else 'shrunk'} by {abs(rings)} ring(s): {n:,} selected.")


def _run_sel_delete(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import delete_selected

    sel = _selection_or_fail(ctx)
    if type(ctx.layer).__name__ == "Points":
        cloud = ctx.cloud()
        out = PointCloud(points=cloud.points[~sel], metadata=dict(cloud.metadata),
                         point_data={k: np.asarray(v)[~sel] for k, v in cloud.point_data.items()
                                     if len(v) == cloud.n_points})
        ctx.put_cloud(out, "edited", force_replace=True)
        return OpResult(f"{int(sel.sum()):,} point(s) deleted.")
    mesh = ctx.mesh()
    out = delete_selected(mesh, sel)
    if not out.n_faces:
        raise ValueError("That would delete the whole surface.")
    ctx.put_mesh(out, "edited", force_replace=True)
    return OpResult(f"Deleted {int(sel.sum()):,} selected vertices ({mesh.n_faces - out.n_faces:,} faces).")


def _run_sel_keep(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.edit import keep_selected

    sel = _selection_or_fail(ctx)
    copy_only = str(ctx.params.get("sel_keep_mode") or "") == "copy to a new layer"
    if type(ctx.layer).__name__ == "Points":
        cloud = ctx.cloud()
        out = PointCloud(points=cloud.points[sel], metadata=dict(cloud.metadata),
                         point_data={k: np.asarray(v)[sel] for k, v in cloud.point_data.items()
                                     if len(v) == cloud.n_points and k != "selected"})
        if copy_only:
            ctx.put_cloud(out, "selection")
        else:
            ctx.put_cloud(out, "kept", force_replace=True)
        return OpResult(f"{out.n_points:,} point(s) {'copied' if copy_only else 'kept'}.")
    out = keep_selected(ctx.mesh(), sel)
    out.point_data.pop("selected", None)
    if not out.n_faces:
        raise ValueError("No face has all its corners selected — grow the selection.")
    if copy_only:
        ctx.put_mesh(out, "selection")
    else:
        ctx.put_mesh(out, "kept", force_replace=True)
    return OpResult(f"{out.n_faces:,} faces {'copied to a new layer' if copy_only else 'kept'}.")


# ──────────────────────────────────────────────────────────────────────────────
# Clean, smooth, remesh
# ──────────────────────────────────────────────────────────────────────────────


def _run_clean(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.cleaning import clean, mesh_summary_counts

    mesh = ctx.mesh()
    before = mesh_summary_counts(mesh)
    out = clean(mesh, tolerance=float(ctx.params.get("merge_tolerance") or 1e-6))
    ctx.put_mesh(out, "clean")
    return OpResult(
        f"Cleaned {ctx.name}: {mesh.n_vertices - out.n_vertices} vertices merged/removed, "
        f"{mesh.n_faces - out.n_faces} faces removed "
        f"({before['degenerate_faces']} degenerate, {before['duplicate_faces']} duplicate)."
    )


def _run_components(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.cleaning import keep_components, split_components
    from nvitk.meshlab.topology import face_components

    mesh = ctx.mesh()
    n, _ = face_components(mesh)
    if bool(ctx.params.get("split_pieces")):
        pieces = split_components(mesh)
        limit = max(1, int(ctx.params.get("keep_largest") or len(pieces)))
        added = [add_mesh_layer(ctx.viewer, piece, name=f"{ctx.name}_piece{i + 1}")
                 for i, piece in enumerate(pieces[:limit])]
        _gather(ctx, added, f"{ctx.name} pieces")
        return OpResult(f"{n} piece(s); added the {min(limit, n)} largest as layers.")
    out = keep_components(
        mesh,
        largest=int(ctx.params.get("keep_largest") or 0),
        min_faces=int(ctx.params.get("min_faces") or 0),
        min_area=float(ctx.params.get("min_area") or 0.0),
    )
    m, _ = face_components(out)
    ctx.put_mesh(out, "pieces")
    return OpResult(f"Kept {m} of {n} piece(s).")


def _run_fill_holes(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.cleaning import fill_holes
    from nvitk.meshlab.topology import boundary_loops, is_watertight

    mesh = ctx.mesh()
    before = len(boundary_loops(mesh))
    out = fill_holes(mesh, float(ctx.params.get("max_hole") or 1e9))
    ctx.put_mesh(out, "filled")
    after = len(boundary_loops(out))
    return OpResult(f"Filled {before - after} of {before} hole(s); closed: {'yes' if is_watertight(out) else 'no'}.")


def _run_orient(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.cleaning import flip_normals, orient_faces

    mesh = ctx.mesh()
    out = flip_normals(mesh) if str(ctx.params.get("orient_mode")) == "flip" else orient_faces(mesh)
    ctx.put_mesh(out, "oriented")
    return OpResult("Faces flipped." if str(ctx.params.get("orient_mode")) == "flip" else "Faces oriented outward.")


def _run_smooth(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import volume
    from nvitk.meshlab.smoothing import laplacian_smooth, taubin_smooth

    mesh = ctx.mesh()
    it = int(ctx.params.get("smooth_iter") or 20)
    lam = float(ctx.params.get("smooth_lambda") or 0.5)
    keep = bool(ctx.params.get("smooth_keep_boundary", True))
    if str(ctx.params.get("smooth_method")) == "laplacian":
        out = laplacian_smooth(mesh, iterations=it, lam=lam, keep_boundary=keep)
    else:
        out = taubin_smooth(mesh, iterations=it, lam=lam, mu=-(lam + 0.03), keep_boundary=keep)
    ctx.put_mesh(out, "smooth")
    v0, v1 = volume(mesh), volume(out)
    change = f" (volume {100 * (v1 - v0) / v0:+.1f} %)" if v0 > 0 else ""
    return OpResult(f"Smoothed {ctx.name}: {it} iterations{change}.")


def _run_decimate(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.remeshing import decimate

    mesh = ctx.mesh()
    out = decimate(mesh, float(ctx.params.get("dec_reduction") or 0.5),
                   method=str(ctx.params.get("dec_method") or "quadric"),
                   preserve_topology=bool(ctx.params.get("dec_topology", True)))
    ctx.put_mesh(out, "decimated")
    return OpResult(f"Decimated {ctx.name}: {mesh.n_faces} → {out.n_faces} faces.")


def _run_subdivide(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.remeshing import subdivide

    mesh = ctx.mesh()
    out = subdivide(mesh, int(ctx.params.get("subdiv_iter") or 1), method=str(ctx.params.get("subdiv_method") or "loop"))
    ctx.put_mesh(out, "subdivided")
    return OpResult(f"Subdivided {ctx.name}: {mesh.n_faces} → {out.n_faces} faces.")


def _run_remesh_voxel(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.remeshing import remesh_voxel

    mesh = ctx.mesh()
    out = remesh_voxel(mesh, float(ctx.params.get("voxel_spacing") or 1.0),
                       smooth_iterations=int(ctx.params.get("smooth_iterations") or 10))
    ctx.put_mesh(out, "remeshed")
    return OpResult(f"Voxel remesh of {ctx.name}: {out.n_faces} faces, watertight.")


def _run_isotropic(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.remeshing import isotropic_remesh

    mesh = ctx.mesh()
    out = isotropic_remesh(mesh, float(ctx.params.get("iso_length") or 0.0),
                           iterations=int(ctx.params.get("iso_iterations") or 10),
                           feature_angle=float(ctx.params.get("iso_feature") or 30.0),
                           adaptive=bool(ctx.params.get("iso_adaptive")))
    ctx.put_mesh(out, "isotropic")
    edges = np.linalg.norm(out.vertices[out.faces[:, [1, 2, 0]]] - out.vertices[out.faces], axis=2)
    return OpResult(f"Isotropic remesh of {ctx.name}: {out.n_faces} faces, edge {edges.mean():.3g} ± {edges.std():.2g} mm.")


def _run_clip(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.remeshing import clip_plane

    mesh = ctx.mesh()
    origin, normal = _plane_normal(ctx, str(ctx.params.get("plane_choice")), mesh)
    out = clip_plane(mesh, origin, normal, keep=str(ctx.params.get("clip_keep") or "below"),
                     close=bool(ctx.params.get("clip_cap")))
    if not out.n_faces:
        raise ValueError("Nothing left on that side of the plane.")
    ctx.put_mesh(out, "clipped")
    return OpResult(f"Clipped {ctx.name} at {np.round(origin, 1).tolist()}: {out.n_faces} faces kept.")


# ──────────────────────────────────────────────────────────────────────────────
# Transform & align
# ──────────────────────────────────────────────────────────────────────────────


def _put_surface(ctx: OpContext, obj: Mesh | PointCloud, suffix: str) -> None:
    if isinstance(obj, Mesh):
        ctx.put_mesh(obj, suffix)
    else:
        ctx.put_cloud(obj, suffix)


def _run_transform(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.transform import transform_surface

    out = transform_surface(
        ctx.surface(),
        translate=_vec3(ctx.params.get("tr_translate"), 0.0),
        rotate_deg=_vec3(ctx.params.get("tr_rotate"), 0.0),
        scale=_vec3(ctx.params.get("tr_scale"), 1.0),
        about_centroid=bool(ctx.params.get("tr_about_centroid", True)),
    )
    _put_surface(ctx, out, "moved")
    return OpResult(f"Transformed {ctx.name}.")


def _run_icp(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import surface_distance_stats
    from nvitk.meshlab.transform import align_principal_axes, apply_affine, icp

    moving = ctx.surface()
    fixed = ctx.surface(ctx.layer_named("icp_reference"))
    initial = None
    if bool(ctx.params.get("icp_pca_start")):
        _, a_m = align_principal_axes(moving)
        _, a_f = align_principal_axes(fixed)
        initial = np.linalg.inv(a_f) @ a_m
    res = icp(moving, fixed, iterations=int(ctx.params.get("icp_iterations") or 50),
              scaling=bool(ctx.params.get("icp_scaling")), initial=initial)
    out = apply_affine(moving, res.affine)
    _put_surface(ctx, out, "aligned")
    stats = surface_distance_stats(out, fixed)
    table = {
        "RMS (fit)": _fmt(res.rms, " mm"),
        "mean surface distance": _fmt(stats["assd"], " mm"),
        "Hausdorff 95 %": _fmt(stats["hausdorff95"], " mm"),
        "iterations": res.iterations,
        "converged": "yes" if res.converged else "no (raise iterations)",
        "affine row 1": " ".join(f"{v:.4f}" for v in res.affine[0]),
        "affine row 2": " ".join(f"{v:.4f}" for v in res.affine[1]),
        "affine row 3": " ".join(f"{v:.4f}" for v in res.affine[2]),
    }
    return OpResult(f"ICP aligned {ctx.name}: RMS {res.rms:.3f} mm.", table=table, title=f"ICP: {ctx.name}")


def _run_pca_align(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import principal_axes
    from nvitk.meshlab.transform import align_principal_axes

    obj = ctx.surface()
    _, _, extents = principal_axes(obj)
    out, _aff = align_principal_axes(obj)
    _put_surface(ctx, out, "pca")
    return OpResult(f"Principal axes of {ctx.name} on x/y/z; extents {', '.join(_fmt(e) for e in extents)} mm.")


# ──────────────────────────────────────────────────────────────────────────────
# Measure / compare
# ──────────────────────────────────────────────────────────────────────────────


def _run_measure(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import mesh_metrics, point_cloud_metrics

    kind = layer_kind(ctx.layer)
    if kind == "series":
        return _run_series_measure(ctx)
    metrics = point_cloud_metrics(ctx.cloud()) if kind == "points" else mesh_metrics(ctx.mesh())
    table = _metrics_table(metrics)
    msg = f"{ctx.name}: " + ", ".join(f"{k} {table[k]}" for k in ("area", "volume", "vertices", "points") if k in table)
    return OpResult(msg, table=table, title=f"Mesh measurements: {ctx.name}")


def _per_frame(ctx: OpContext, fn: Callable[[Mesh], np.ndarray]) -> np.ndarray | list[np.ndarray]:
    """*fn* on the mesh — on every frame for a series."""
    ctrl = series_controller(ctx.layer)
    if ctrl is None:
        return fn(ctx.mesh())
    return [fn(f) if f.n_faces else np.zeros(f.n_vertices) for f in ctrl.series]


def _run_curvature(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import curvature
    from nvitk.meshlab.smoothing import smooth_point_data

    kind = str(ctx.params.get("curv_kind") or "mean")
    it = int(ctx.params.get("curv_smooth") or 0)

    def _curv(mesh: Mesh) -> np.ndarray:
        vals = curvature(mesh, kind)
        if it > 0:
            vals = smooth_point_data(mesh.with_point_data(curv=vals), "curv", iterations=it).point_data["curv"]
        return vals

    vals = _per_frame(ctx, _curv)
    name = f"curvature ({kind})"
    add_field(ctx.layer, name, vals, colormap="bwr" if kind in ("mean", "gaussian") else "turbo")
    flat = np.concatenate([np.ravel(v) for v in vals]) if isinstance(vals, list) else vals
    table = {
        "kind": kind,
        "mean": _fmt(float(np.mean(flat)), " /mm"),
        "median": _fmt(float(np.median(flat)), " /mm"),
        "5th percentile": _fmt(float(np.percentile(flat, 5)), " /mm"),
        "95th percentile": _fmt(float(np.percentile(flat, 95)), " /mm"),
    }
    return OpResult(f"{name} of {ctx.name} added as a field (Display → Colour by).", table=table,
                    title=f"Curvature: {ctx.name}")


def _run_cross_section(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import cross_section
    from nvitk.meshlab.vessels import section_of_loop, vessel_section_at

    mesh = ctx.mesh()
    choice = str(ctx.params.get("plane_choice") or _PLANES[0])
    extra: dict[str, Any] = {}
    if choice == "vessel axis (auto)":
        sec = vessel_section_at(mesh, ctx.at())
        loops = [sec["contour"]]
        origin, normal = np.asarray(sec["centre"]), np.asarray(sec["axis"])
        area, perim = sec["area"], sec["perimeter"]
        extra = {"min diameter": _fmt(sec["min_diameter"], " mm"), "max diameter": _fmt(sec["max_diameter"], " mm"),
                 "circularity": _fmt(sec["circularity"])}
    else:
        origin, normal = _plane_normal(ctx, choice, mesh)
        res = cross_section(mesh, origin, normal)
        if not res["loops"]:
            raise ValueError("The plane through that point does not cut the surface.")
        loops = res["loops"]
        area, perim = res["area"], res["perimeter"]
        if len(loops) == 1 and len(loops[0]) >= 3:
            sec = section_of_loop(loops[0], origin, normal)
            extra = {"min diameter": _fmt(sec["min_diameter"], " mm"), "max diameter": _fmt(sec["max_diameter"], " mm"),
                     "circularity": _fmt(sec["circularity"])}
    add_polylines_layer(ctx.viewer, loops, name=f"{ctx.name}_section", width=0.6)
    table = {
        "plane": choice,
        "centre": ", ".join(_fmt(v) for v in origin),
        "normal": ", ".join(f"{v:.3f}" for v in normal),
        "contours": len(loops),
        "area": _fmt(area, " mm²"),
        "perimeter": _fmt(perim, " mm"),
        "equivalent diameter": _fmt(2 * np.sqrt(area / np.pi), " mm"),
        **extra,
    }
    return OpResult(f"Cross-section of {ctx.name}: area {table['area']}, diameter {table['equivalent diameter']}.",
                    table=table, title=f"Cross-section: {ctx.name}")


def _run_distance(ctx: OpContext) -> OpResult:
    from scipy.spatial import cKDTree

    from nvitk.meshlab.measure import distance_to_surface, surface_distance_stats

    src = ctx.surface()
    ref_layer = ctx.layer_named("dist_reference")
    ref = ctx.surface(ref_layer)
    signed = bool(ctx.params.get("dist_signed"))
    pts = src.vertices if isinstance(src, Mesh) else src.points
    if isinstance(ref, Mesh):
        d = distance_to_surface(pts, ref, signed=signed)
    else:
        d, _ = cKDTree(ref.points).query(pts)
    stats = surface_distance_stats(src, ref)
    add_field(ctx.layer, f"distance to {ref_layer.name}", d, colormap="bwr" if signed else "turbo")
    table = {
        f"mean {ctx.name} → {ref_layer.name}": _fmt(stats["mean_a_to_b"], " mm"),
        f"mean {ref_layer.name} → {ctx.name}": _fmt(stats["mean_b_to_a"], " mm"),
        "mean symmetric (ASSD)": _fmt(stats["assd"], " mm"),
        "RMS": _fmt(stats["rms"], " mm"),
        "Hausdorff 95 %": _fmt(stats["hausdorff95"], " mm"),
        "Hausdorff (max)": _fmt(stats["hausdorff"], " mm"),
    }
    return OpResult(f"Distance {ctx.name} ↔ {ref_layer.name}: ASSD {table['mean symmetric (ASSD)']}.",
                    table=table, title=f"Surface distance: {ctx.name} ↔ {ref_layer.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Vessels & tubes
# ──────────────────────────────────────────────────────────────────────────────


def _run_centerlines(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.vessels import (
        bifurcations,
        branch_map,
        branch_metrics,
        branch_profiles,
        mesh_centerlines,
        radius_map,
    )

    mesh = ctx.mesh()
    branches = mesh_centerlines(
        mesh,
        voxel_size=float(ctx.params.get("cl_voxel") or 0.0),
        min_branch_length=float(ctx.params.get("cl_min_branch") or 0.0),
        smooth=float(ctx.params.get("cl_smooth") if ctx.params.get("cl_smooth") is not None else 0.5),
    )
    if not branches:
        raise ValueError("No centerline found — is the surface closed and tubular?")
    branch_profiles(mesh, branches, every=float(ctx.params.get("cl_every") or 0.0))
    # The centerline as points carrying the profile, coloured by diameter.
    keys = ("arc", "diameter", "area", "min_diameter", "max_diameter", "circularity", "radius")
    pts, feats = [], {k: [] for k in ("branch", *keys)}
    for b in branches:
        p = b.profile
        pts.append(b.points[p["index"]])
        feats["branch"].append(np.full(len(p["index"]), b.id))
        for k in keys:
            feats[k].append(p[k])
    cloud = PointCloud(points=np.vstack(pts), metadata={"name": f"{ctx.name}_centerline"},
                       point_data={k: np.concatenate(v) for k, v in feats.items()})
    span = float(np.ptp(mesh.vertices, axis=0).max())
    new_layers = [
        add_polylines_layer(ctx.viewer, [b.points for b in branches], name=f"{ctx.name}_centerline_lines",
                            width=max(span / 400.0, 0.1)),
        add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_centerline", color_by="diameter",
                              size=max(span / 120.0, 0.3)),
    ]
    if bool(ctx.params.get("cl_colour_surface", True)):
        add_field(ctx.layer, "distance to centerline", radius_map(mesh, branches, key="radius"), show=False)
        add_field(ctx.layer, "diameter (centerline)", radius_map(mesh, branches, key="diameter"))
    owner, _dist = branch_map(mesh, branches)
    add_field(ctx.layer, "branch", owner.astype(np.float32), show=False)
    bifs = bifurcations(branches)
    if bifs:
        new_layers.append(add_point_cloud_layer(
            ctx.viewer, PointCloud(points=np.asarray([b["point"] for b in bifs])),
            name=f"{ctx.name}_bifurcations", size=max(span / 50.0, 1.0), color=(1.0, 1.0, 1.0, 1.0)))
    _gather(ctx, [lyr for lyr in new_layers if lyr is not None], f"{ctx.name} centerlines")
    table: dict[Any, dict[str, Any]] = {}
    for b in branches:
        m = branch_metrics(b)
        table[f"branch {b.id}"] = {
            "length (mm)": _fmt(m["length"]),
            "tortuosity": _fmt(m["tortuosity"]),
            "mean Ø (mm)": _fmt(m.get("mean_diameter", np.nan)),
            "min Ø (mm)": _fmt(m.get("min_diameter", np.nan)),
            "min Ø at (mm)": _fmt(m.get("min_diameter_at", np.nan)),
            "max Ø (mm)": _fmt(m.get("max_diameter", np.nan)),
            "stenosis %": _fmt(m.get("stenosis_percent", np.nan)),
            "curvature (1/mm)": _fmt(m["mean_curvature"]),
        }
    for k, bf in enumerate(bifs, 1):
        angles = ", ".join(f"{pair}: {a:.0f}°" for pair, a in bf["angles"].items())
        table[f"bifurcation {k}"] = {"length (mm)": f"branches {', '.join(str(i) for i in bf['branches'])}",
                                     "tortuosity": angles}
    lines = [{"x": b.profile["arc"], "y": b.profile["diameter"], "label": f"branch {b.id}"} for b in branches[:8]]
    plot = {"lines": lines, "xlabel": "distance along the branch (mm)", "ylabel": "diameter (mm)"}
    return OpResult(
        f"{len(branches)} branch(es), {len(bifs)} bifurcation(s); longest {branches[0].length:.1f} mm.",
        table=table, title=f"Centerlines: {ctx.name}", plot=plot, row_header="Branch",
        on_row=CenterlineHighlighter(ctx.viewer, ctx.layer, new_layers[0], branches, bifs, owner),
    )


#: Highlight colours: the rest of the vessel / the branch or junction clicked in the table.
_HIGHLIGHT_COLOURS = {0: (0.62, 0.62, 0.62, 1.0), 1: (1.0, 0.72, 0.05, 1.0)}
_HIGHLIGHT_FIELD = "highlight"


class CenterlineHighlighter:
    """Shows the branch or bifurcation of a clicked results row on the surface and
    its centerline, and centres the camera on it; ``None`` puts things back."""

    def __init__(self, viewer: Any, surface: Any, lines: Any, branches: list[Any], bifs: list[dict[str, Any]],
                 owner: np.ndarray) -> None:
        import weakref

        self.viewer = viewer
        self.surface = weakref.ref(surface)
        self.lines = weakref.ref(lines) if lines is not None else (lambda: None)
        self.branches = {b.id: b for b in branches}
        self.order = [b.id for b in branches]
        self.bifs = bifs
        self.owner = np.asarray(owner)
        self._saved: Any = None
        self._line_colour: Any = None

    def _segments(self) -> dict[int, slice]:
        """Which rows of the centerline Vectors layer belong to which branch."""
        out, start = {}, 0
        for bid in self.order:
            n = max(len(self.branches[bid].points) - 1, 0)
            out[bid] = slice(start, start + n)
            start += n
        return out

    def __call__(self, key: str | None) -> None:
        import copy

        from nvitk.gui.mesh.layers import display_of

        surface = self.surface()
        if surface is None or surface not in self.viewer.layers:
            return
        if key is None:
            self._restore(surface)
            return
        text = str(key)
        mesh_v = layer_to_mesh(surface).vertices
        if len(mesh_v) != len(self.owner):
            return  # the surface has been edited since: the map no longer fits
        if text.startswith("branch "):
            bids = [int(text.split()[1])]
            b = self.branches.get(bids[0])
            if b is None:
                return
            mask = self.owner == bids[0]
            centre = np.asarray(b.points[len(b.points) // 2], dtype=float)
        elif text.startswith("bifurcation "):
            k = int(text.split()[1]) - 1
            if not 0 <= k < len(self.bifs):
                return
            bf = self.bifs[k]
            bids = [int(i) for i in bf["branches"]]
            centre = np.asarray(bf["point"], dtype=float)
            # The junction itself: vertices of the meeting branches near the point.
            reach = 2.5 * float(np.nanmedian([np.nanmedian(self.branches[i].profile.get("diameter", [np.nan]))
                                              for i in bids if i in self.branches]) or 5.0)
            mask = np.isin(self.owner, bids) & (np.linalg.norm(mesh_v - centre, axis=1) <= reach)
        else:
            return
        disp = display_of(surface)
        if self._saved is None and not (disp.mode == "field" and disp.field == _HIGHLIGHT_FIELD):
            self._saved = copy.copy(disp)
        add_field(surface, _HIGHLIGHT_FIELD, mask.astype(np.float32), categories=_HIGHLIGHT_COLOURS, show=True)
        lines = self.lines()
        if lines is not None and lines in self.viewer.layers:
            try:
                if self._line_colour is None:
                    self._line_colour = np.asarray(lines.edge_color).copy()
                colours = np.tile(np.array([[0.45, 0.45, 0.45, 1.0]]), (len(lines.data), 1))
                segs = self._segments()
                for bid in bids:
                    if bid in segs:
                        colours[segs[bid]] = (1.0, 0.25, 0.1, 1.0)
                lines.edge_color = colours
            except Exception:  # noqa: BLE001 — the line colours are a bonus
                pass
        try:
            if int(self.viewer.dims.ndisplay) == 3:
                self.viewer.camera.center = tuple(float(c) for c in centre)
        except Exception:  # noqa: BLE001
            pass

    def _restore(self, surface: Any) -> None:
        from nvitk.gui.mesh.layers import fields_of, remove_field, set_display

        remove_field(surface, _HIGHLIGHT_FIELD)
        prev, self._saved = self._saved, None
        if prev is not None and (prev.mode != "field" or prev.field in fields_of(surface)):
            set_display(surface, mode=prev.mode, field=prev.field, colormap=prev.colormap, color=prev.color,
                        auto_range=prev.auto_range, limits=prev.limits)
        lines = self.lines()
        if lines is not None and lines in self.viewer.layers and self._line_colour is not None:
            try:
                lines.edge_color = self._line_colour
            except Exception:  # noqa: BLE001
                pass
            self._line_colour = None



def _run_vessel_section(ctx: OpContext) -> OpResult:
    ctx.params = dict(ctx.params, plane_choice="vessel axis (auto)")
    return _run_cross_section(ctx)


def _run_thickness(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.vessels import thickness_map

    vals = _per_frame(ctx, lambda m: thickness_map(m))
    add_field(ctx.layer, "thickness", vals)
    flat = np.concatenate([np.ravel(v) for v in vals]) if isinstance(vals, list) else vals
    ok = flat[np.isfinite(flat)]
    table = {"median": _fmt(np.median(ok), " mm") if ok.size else "—",
             "5th percentile": _fmt(np.percentile(ok, 5), " mm") if ok.size else "—",
             "95th percentile": _fmt(np.percentile(ok, 95), " mm") if ok.size else "—"}
    return OpResult(f"Thickness (local diameter) of {ctx.name}: median {table['median']}.", table=table,
                    title=f"Thickness: {ctx.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Convert
# ──────────────────────────────────────────────────────────────────────────────


def _run_to_mask(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.voxelize import mesh_to_mask

    mesh = ctx.mesh()
    shape, aff, grid = _reference_grid(ctx, "grid_reference", float(ctx.params.get("grid_voxel") or 0.0), mesh.bounds)
    label = int(ctx.params.get("mask_label") or 1)
    data = np.asarray(mesh_to_mask(mesh, affine=aff, shape=shape, label=label).data)
    if not data.any():
        raise ValueError("The surface encloses no voxel of that grid (is it closed, and over the image?).")
    _add_image(ctx, data, aff, f"{ctx.name}_mask", labels=True)
    return OpResult(f"Filled {ctx.name} onto the {grid}: {int((data > 0).sum())} voxels.")


def _run_to_sdf(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.sampling import signed_distance_image

    mesh = ctx.mesh()
    shape, aff, grid = _reference_grid(ctx, "grid_reference", float(ctx.params.get("grid_voxel") or 0.0), mesh.bounds,
                                       margin=10)
    sdf = signed_distance_image(mesh, shape, aff)
    layer = _add_image(ctx, sdf, aff, f"{ctx.name}_distance", labels=False, colormap="bwr")
    lim = float(np.percentile(np.abs(sdf), 95)) or 1.0
    layer.contrast_limits = (-lim, lim)
    return OpResult(f"Signed distance from {ctx.name} on the {grid} (negative inside, mm).")


def _run_points_from_mask(ctx: OpContext) -> OpResult:
    from scipy import ndimage

    volumes, t_ax = _image_volumes(ctx.layer)
    frame = int(ctx.viewer.dims.current_step[0]) if t_ax is not None and int(ctx.viewer.dims.ndim) >= 4 else 0
    vol = np.asarray(volumes[min(frame, len(volumes) - 1)])
    sel = vol != 0
    if bool(ctx.params.get("boundary_only")):
        sel &= ~ndimage.binary_erosion(sel, structure=ndimage.generate_binary_structure(3, 1))
    ijk = np.argwhere(sel)
    labels = vol[sel].astype(np.int64)
    max_points = int(ctx.params.get("max_points") or 0)
    if max_points and len(ijk) > max_points:
        keep = np.random.default_rng(0).choice(len(ijk), max_points, replace=False)
        ijk, labels = ijk[keep], labels[keep]
    if not len(ijk):
        raise ValueError("The mask is empty.")
    coords = ijk.astype(float)
    if t_ax is not None:
        coords = np.insert(coords, t_ax, float(frame), axis=1)
    cloud = PointCloud(points=_to_world(ctx.layer, coords), point_data={"label": labels})
    ids = sorted(int(v) for v in np.unique(labels))
    info = label_info(ctx.layer, ids)
    add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_points", color_by="label" if len(ids) > 1 else None,
                          categories={i: info[i][1] for i in ids} if len(ids) > 1 else None,
                          color=info[ids[0]][1] if len(ids) == 1 else None)
    return OpResult(f"{cloud.n_points} {'boundary ' if ctx.params.get('boundary_only') else ''}voxel points from {ctx.name}.")


def _run_mesh_to_points(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.convert import mesh_to_point_cloud
    from nvitk.meshlab.pointcloud import sample_surface

    mesh = ctx.mesh()
    if str(ctx.params.get("mp_mode")) == "vertices":
        cloud = mesh_to_point_cloud(mesh)
    else:
        cloud = sample_surface(mesh, int(ctx.params.get("n_points") or 10000))
    add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_points")
    return OpResult(f"{cloud.n_points} points from {ctx.name}.")


def _run_points_to_mask(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.pointcloud import points_to_mask

    cloud = ctx.cloud()
    shape, aff, grid = _reference_grid(ctx, "grid_reference", float(ctx.params.get("grid_voxel") or 0.0), cloud.bounds)
    labels = cloud.point_data.get("label")
    data = points_to_mask(cloud.points, shape, aff, radius=float(ctx.params.get("pm_radius") or 0.0),
                          fill=bool(ctx.params.get("pm_fill")), labels=labels)
    if not data.any():
        raise ValueError("No point falls on that grid.")
    _add_image(ctx, data, aff, f"{ctx.name}_mask", labels=True)
    return OpResult(f"{ctx.name} onto the {grid}: {int((data > 0).sum())} voxels.")


def _run_reconstruct(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.measure import volume
    from nvitk.meshlab.pointcloud import reconstruct_surface
    from nvitk.meshlab.topology import is_watertight

    cloud = ctx.cloud()
    mesh = reconstruct_surface(
        cloud, str(ctx.params.get("recon_method") or "poisson"),
        resolution=int(ctx.params.get("recon_resolution") or 0),
        smoothing=float(ctx.params.get("recon_smoothing") if ctx.params.get("recon_smoothing") is not None else 1.0),
        trim=float(ctx.params.get("recon_trim") or 0.0),
        normals_k=int(ctx.params.get("recon_normals_k") or 16),
        sample_spacing=float(ctx.params.get("recon_spacing") or 0.0),
        alpha=float(ctx.params.get("recon_alpha") or 0.0),
        keep_largest=bool(ctx.params.get("recon_keep_largest", True)),
        smooth_iterations=int(ctx.params.get("recon_smooth") or 0),
    )
    if not mesh.n_faces:
        raise ValueError("The reconstruction came out empty.")
    add_mesh_layer(ctx.viewer, mesh, name=f"{ctx.name}_surface")
    closed = is_watertight(mesh)
    vol = f", volume {_fmt(volume(mesh), ' mm³')}" if closed else ""
    return OpResult(f"Reconstructed {mesh.n_faces} faces from {cloud.n_points} points "
                    f"({'closed' if closed else 'open'}{vol}).")


def _run_probe(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.sampling import sample_on_surface

    img_layer = ctx.layer_named("probe_image")
    if type(img_layer).__name__ not in ("Image", "Labels"):
        raise ValueError("Pick an image or labels layer to read values from.")
    volumes, t_ax = _image_volumes(img_layer)
    frame = int(ctx.viewer.dims.current_step[0]) if t_ax is not None and int(ctx.viewer.dims.ndim) >= 4 else 0
    data = np.asarray(volumes[min(frame, len(volumes) - 1)])
    aff = layer_spatial_affine(img_layer)
    is_lab = type(img_layer).__name__ == "Labels"
    order = 0 if is_lab else int(str(ctx.params.get("probe_order") or "linear") == "linear")
    vals = sample_on_surface(ctx.surface(), data, aff, order=order, depth=float(ctx.params.get("probe_depth") or 0.0),
                             mode=str(ctx.params.get("probe_mode") or "mean"))
    categories = None
    if is_lab:
        ids = sorted(int(v) for v in np.unique(vals[np.isfinite(vals)]) if v)
        if ids:
            categories = {i: c for i, (_n, c) in label_info(img_layer, ids).items()}
            categories[0] = (0.5, 0.5, 0.5, 1.0)
    add_field(ctx.layer, img_layer.name, np.nan_to_num(vals), categories=categories)
    ok = vals[np.isfinite(vals)]
    table = {"mean": _fmt(ok.mean()) if ok.size else "—", "min": _fmt(ok.min()) if ok.size else "—",
             "max": _fmt(ok.max()) if ok.size else "—", "outside the image": int((~np.isfinite(vals)).sum())}
    return OpResult(f"{img_layer.name} sampled onto {ctx.name} (Display → Colour by).", table=table,
                    title=f"{img_layer.name} on {ctx.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Point clouds
# ──────────────────────────────────────────────────────────────────────────────


def _run_pc_downsample(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.pointcloud import random_downsample, voxel_downsample

    cloud = ctx.cloud()
    if str(ctx.params.get("pc_down_method")) == "random":
        out = random_downsample(cloud, int(ctx.params.get("pc_n") or 10000))
    else:
        out = voxel_downsample(cloud, float(ctx.params.get("pc_voxel") or 2.0))
    ctx.put_cloud(out, "down")
    return OpResult(f"Downsampled {ctx.name}: {cloud.n_points} → {out.n_points} points.")


def _run_pc_outliers(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.pointcloud import radius_outliers, statistical_outliers

    cloud = ctx.cloud()
    if str(ctx.params.get("pc_out_method")) == "radius":
        flags = radius_outliers(cloud, radius=float(ctx.params.get("pc_radius") or 2.0),
                                min_neighbours=int(ctx.params.get("pc_min_nb") or 4))
    else:
        flags = statistical_outliers(cloud, k=int(ctx.params.get("pc_k") or 16),
                                     std_ratio=float(ctx.params.get("pc_std") or 2.0))
    ctx.put_cloud(cloud.subset(~flags), "inliers")
    if flags.any() and bool(ctx.params.get("pc_show_outliers", True)):
        add_point_cloud_layer(ctx.viewer, cloud.subset(flags), name=f"{ctx.name}_outliers", color=(1.0, 0.3, 0.3, 1.0))
    return OpResult(f"{int(flags.sum())} outlier(s) of {cloud.n_points} points removed.")


def _run_pc_normals(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.pointcloud import estimate_normals

    cloud = estimate_normals(ctx.cloud(), k=int(ctx.params.get("pc_k") or 16),
                             orient=str(ctx.params.get("pc_orient") or "propagate"))
    ctx.put_cloud(cloud, "normals")
    add_normals_layer(ctx.viewer, cloud, name=f"{ctx.name}_normal_vectors")
    return OpResult(f"Normals estimated for {cloud.n_points} points.")


# ──────────────────────────────────────────────────────────────────────────────
# Time (3D+t)
# ──────────────────────────────────────────────────────────────────────────────


def _run_series_measure(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.temporal import series_metrics

    series = ctx.series()
    met = series_metrics(series)
    table: dict[Any, dict[str, Any]] = {
        f"frame {k + 1}": {"time": f"{t:.4g}", "area (mm²)": _fmt(a), "volume (mm³)": _fmt(v)}
        for k, (t, a, v) in enumerate(zip(met["time"], met["area"], met["volume"]))
    }
    vol = met["volume"]
    note = ""
    if np.isfinite(vol).sum() >= 2:
        vmax, vmin = np.nanmax(vol), np.nanmin(vol)
        note = f"max {_fmt(vmax, ' mm³')} · min {_fmt(vmin, ' mm³')} · ejection fraction {100 * (vmax - vmin) / vmax:.1f} %"
        table["volume range"] = {"time": "", "area (mm²)": "", "volume (mm³)": note}
    plot = {"x": met["time"], "series": {"volume (mm³)": vol, "area (mm²)": met["area"]}, "xlabel": "time"}
    return OpResult(f"{ctx.name}: {len(series)} frames. {note}", table=table, title=f"Over time: {ctx.name}", plot=plot,
                    row_header="Frame", on_row=_frame_jumper(ctx.viewer, ctx.layer))


def _frame_jumper(viewer: Any, layer: Any) -> Callable[[str | None], None]:
    """Row callback: "frame k" shows frame k of the series."""
    import weakref

    ref = weakref.ref(layer)

    def _go(key: str | None) -> None:
        from nvitk.gui.mesh.layers import go_to_frame

        lyr = ref()
        if key is None or lyr is None or not str(key).startswith("frame "):
            return
        go_to_frame(viewer, lyr, int(str(key).split()[1]) - 1)

    return _go


def _run_series_track(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.temporal import propagate_mesh

    series = ctx.series()
    ref = int(ctx.params.get("series_reference") or 0)
    tracked = propagate_mesh(series, reference=ref, iterations=int(ctx.params.get("track_iterations") or 3),
                             smooth_iterations=int(ctx.params.get("track_smooth") or 5))
    apply_display(ctx.put_series(tracked, "tracked"))
    return OpResult(f"Tracked frame {ref}'s mesh through {len(series)} frames (shared topology).")


def _run_series_motion(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.temporal import vertex_displacement, vertex_velocity

    series = ctx.series()
    if not series.shared_topology:
        raise ValueError("Motion needs a tracked series: run 'Track surface through time' first.")
    kind = str(ctx.params.get("motion_kind") or "displacement")
    vals = vertex_velocity(series) if kind == "speed" else vertex_displacement(
        series, int(ctx.params.get("series_reference") or 0))
    add_field(ctx.layer, kind, list(vals))
    unit = " mm/s" if kind == "speed" else " mm"
    return OpResult(f"{kind.capitalize()} of {ctx.name}: max {_fmt(float(vals.max()), unit)}, "
                    f"mean {_fmt(float(vals.mean()), unit)} (Display → Colour by).")


def _run_series_smooth(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.temporal import smooth_in_time

    series = ctx.series()
    out = smooth_in_time(series, float(ctx.params.get("time_sigma") or 1.0), cyclic=bool(ctx.params.get("time_cyclic", True)))
    apply_display(ctx.put_series(out, "tsmooth"))
    return OpResult(f"Smoothed {ctx.name} along time (σ = {ctx.params.get('time_sigma')} frames).")


def _run_series_interp(ctx: OpContext) -> OpResult:
    from nvitk.meshlab.temporal import interpolate_frames

    series = ctx.series()
    out = interpolate_frames(series, factor=float(ctx.params.get("time_factor") or 2.0),
                             cyclic=bool(ctx.params.get("time_cyclic", True)))
    apply_display(ctx.put_series(out, "frames"))
    return OpResult(f"{ctx.name}: {len(series)} → {len(out)} frames.")


def _run_series_frame(ctx: OpContext) -> OpResult:
    ctrl = series_controller(ctx.layer)
    if ctrl is None:
        raise ValueError("Select a mesh time series.")
    add_mesh_layer(ctx.viewer, ctrl.current_mesh(), name=f"{ctx.name}_frame{ctrl.frame}")
    return OpResult(f"Frame {ctrl.frame} of {ctx.name} as its own mesh.")


# ──────────────────────────────────────────────────────────────────────────────
# Catalogue
# ──────────────────────────────────────────────────────────────────────────────

_SMOOTH = _P("smooth_iterations", "Smooth (Taubin iterations, 0 = off)", "int", 10, min=0, max=500)
_DECIMATE = _P("decimate_reduction", "Decimate (fraction removed, 0 = off)", "float", 0.0, min=0.0, max=0.99)
_STEP = _P("mc_step", "Marching-cubes step (voxels)", "int", 1, min=1, max=8)
_PLANE = _P("plane_choice", "Plane", "choice", _PLANES[0], choices=_PLANES)
_TIME_CYCLIC = _P("time_cyclic", "Cyclic (the cycle wraps around)", "bool", True)
_REFERENCE = _P("series_reference", "Reference frame", "int", 0, min=0, max=100000)
_RADIUS = _P("edit_radius", "Radius (mm)", "float", 5.0, min=0.01, max=10000.0)
_GRID = (
    _P("grid_reference", "Grid of image (none = new grid)", "layer", ""),
    _P("grid_voxel", "Voxel size of a new grid (mm, 0 = auto)", "float", 0.0, min=0.0, max=100.0),
)
_RECON = (
    _P("recon_method", "Method", "choice", "poisson",
       choices=("poisson", "screened_poisson", "ball_pivoting", "imls", "implicit", "alpha_shape", "convex_hull")),
    _P("recon_resolution", "Grid size, longest side (0 = from point spacing)", "int", 0, min=0, max=512),
    _P("recon_smoothing", "Smoothing, Poisson (grid steps)", "float", 1.0, min=0.0, max=10.0),
    _P("recon_trim", "Trim far from points, Poisson (× spacing, 0 = closed)", "float", 0.0, min=0.0, max=20.0),
    _P("recon_normals_k", "Neighbours for normals", "int", 16, min=4, max=200),
    _P("recon_alpha", "Alpha shape: alpha · ball pivoting: ball radius (mm, 0 = auto)", "float", 0.0, min=0.0, max=1000.0),
    _P("recon_spacing", "Grid spacing, implicit (mm, 0 = auto)", "float", 0.0, min=0.0, max=1000.0),
    _P("recon_keep_largest", "Keep only the largest piece", "bool", True),
    _P("recon_smooth", "Smooth the result (Taubin iterations)", "int", 0, min=0, max=200),
)
_RECON_DESC = (
    "A surface through a point cloud. Poisson: watertight and smooth, robust to noise and uneven "
    "sampling (normals are estimated and oriented consistently when the cloud has none). Screened "
    "Poisson and ball pivoting are MeshLab's (ball pivoting interpolates the points, keeps holes). IMLS: "
    "follows the points closely and keeps open surfaces open. Alpha shape: concave hull. "
    "Implicit: VTK's signed distance. Convex hull."
)


def _op(id_: str, category: str, label: str, inputs: tuple[str, ...], params: tuple[ParamSpec, ...],
        run: Callable[[OpContext], OpResult], description: str, replaceable: bool = False, pick: str = "") -> MeshOp:
    return MeshOp(id_, category, label, inputs, params, run, description, replaceable, pick)


(_CREATE, _EDIT, _SELECT, _CLEAN, _SMOOTHING, _REMESH, _MOVE, _MEASURE, _VESSELS, _COMPARE, _CONVERT, _POINTS,
 _TIME) = CATEGORIES

OPERATIONS: tuple[MeshOp, ...] = (
    _op("surfaces", _CREATE, "Surfaces from labels / mask", ("image",),
        (_P("labels", "Labels", "labels", None),
         _P("surface_output", "Output", "choice", SURFACE_OUTPUTS[0], choices=SURFACE_OUTPUTS),
         _P("label_groups", "Groups (name: ids ; …)", "str", ""),
         _P("label_colours", "Use the labels' colours", "bool", True),
         _P("close_borders", "Close surfaces cut by the image border", "bool", True),
         _STEP, _SMOOTH, _DECIMATE),
        _run_surfaces,
        "Marching cubes of a label map or mask, in mm. Tick the labels to mesh, then choose one layer "
        "per label (gathered in a folder, each in its label's colour), one layer coloured by label, one "
        "merged surface, or your own groups. A 3D+t mask gives time series that play with the slider."),
    _op("isosurface", _CREATE, "Isosurface of an image", ("image",),
        (_P("iso_level", "Level (0 = Otsu threshold)", "float", 0.0, min=-1e9, max=1e9), _STEP, _SMOOTH, _DECIMATE),
        _run_isosurface, "The surface where the image crosses a level (bone at ~300 HU on CT, a contrast-filled lumen…)."),
    _op("convex_hull", _CREATE, "Convex hull", ("mesh", "points"), (), _run_hull,
        "The smallest convex surface around the mesh or points.", replaceable=True),
    _op("erase", _EDIT, "Erase around a click", ("mesh", "points"),
        (_RADIUS, _P("edit_keep_inside", "Keep only what is inside instead", "bool", False)),
        _run_erase, "Run, then click on the surface: faces (or points) within the radius are removed. "
        "Undo restores the previous state.", pick="surface"),
    _op("piece", _EDIT, "Keep / delete the piece under a click", ("mesh",),
        (_P("piece_action", "Action", "choice", "keep it", choices=("keep it", "delete it")),),
        _run_piece, "Run, then click on a connected piece of the surface to keep only it — or delete it.",
        pick="surface"),
    _op("sculpt", _EDIT, "Push / pull the surface", ("mesh",),
        (_RADIUS, _P("sculpt_amount", "Amount (mm; + out, − in)", "float", 1.0, min=-1000.0, max=1000.0)),
        _run_sculpt, "Run, then click: the surface bulges out (or dents in) around the click with a smooth falloff.",
        pick="surface"),
    _op("smooth_spot", _EDIT, "Smooth around a click", ("mesh",),
        (_RADIUS, _P("spot_iterations", "Iterations", "int", 10, min=1, max=500)),
        _run_smooth_spot, "Run, then click: Laplacian smoothing confined to the radius (fades out at its rim).",
        pick="surface"),
    _op("sel_brush", _SELECT, "Brush: paint on the surface", ("mesh", "points"),
        (_P("sel_radius", "Brush radius (mm)", "float", 3.0, min=0.01, max=10000.0),
         _P("sel_mode", "Mode", "choice", "add", choices=("add", "remove"))),
        _run_sel_interactive, "Run, then drag on the surface to paint a selection (red); dragging off the "
        "surface still turns the camera. Press Done when finished.", pick="brush"),
    _op("sel_box", _SELECT, "Box: drag a rectangle on the screen", ("mesh", "points"),
        (_P("sel_mode", "Mode", "choice", "add", choices=("add", "remove", "replace")),
         _P("sel_facing", "Only the side facing you", "bool", True)),
        _run_sel_interactive, "Run, then drag a rectangle over the view: what falls inside is selected "
        "(only what faces you, or straight through). Camera rotation pauses until Done.", pick="box"),
    _op("sel_piece", _SELECT, "Piece under a click", ("mesh",),
        (_P("sel_mode", "Mode", "choice", "add", choices=("add", "remove", "replace")),),
        _run_sel_piece, "Run, then click on the surface: its whole connected piece is selected.", pick="surface"),
    _op("sel_grow", _SELECT, "Grow / shrink the selection", ("mesh",),
        (_P("sel_grow_mode", "Grow or shrink", "choice", "grow", choices=("grow", "shrink")),
         _P("sel_rings", "Rings of vertices", "int", 1, min=1, max=1000)),
        _run_sel_grow, "Add (or remove) rings of neighbouring vertices around the selection's edge."),
    _op("sel_invert", _SELECT, "Invert the selection", ("mesh", "points"), (), _run_sel_invert,
        "Select what was not selected, and the other way round."),
    _op("sel_all", _SELECT, "Select all", ("mesh", "points"), (), _run_sel_all, "Select every vertex / point."),
    _op("sel_clear", _SELECT, "Clear the selection", ("mesh", "points"), (), _run_sel_clear,
        "Drop the selection; the layer gets its previous colouring back."),
    _op("sel_delete", _SELECT, "Delete the selection", ("mesh", "points"), (), _run_sel_delete,
        "Remove the selected vertices / points and the faces touching them (Undo edit puts them back)."),
    _op("sel_keep", _SELECT, "Keep / copy the selection", ("mesh", "points"),
        (_P("sel_keep_mode", "What to do", "choice", "keep only the selection",
            choices=("keep only the selection", "copy to a new layer")),),
        _run_sel_keep, "Crop the layer to the selection (faces with every corner selected), or copy that part "
        "into a new layer."),
    _op("clean", _CLEAN, "Clean (merge duplicates, drop degenerate faces)", ("mesh",),
        (_P("merge_tolerance", "Merge vertices closer than (mm)", "float", 1e-6, min=0.0, max=10.0),),
        _run_clean, "Weld duplicate vertices (STL), drop zero-area and duplicate faces and unused vertices.",
        replaceable=True),
    _op("components", _CLEAN, "Connected pieces (keep largest / drop small)", ("mesh",),
        (_P("keep_largest", "Keep the N largest (0 = all)", "int", 1, min=0, max=100000),
         _P("min_faces", "Drop pieces under (faces)", "int", 0, min=0, max=10_000_000),
         _P("min_area", "Drop pieces under (mm²)", "float", 0.0, min=0.0, max=1e9),
         _P("split_pieces", "Split into one layer per piece", "bool", False)),
        _run_components, "Remove islands, keep the main structure, or split a mesh into its pieces (in a folder).",
        replaceable=True),
    _op("fill_holes", _CLEAN, "Fill holes", ("mesh",),
        (_P("max_hole", "Largest hole radius to fill (mm)", "float", 1e9, min=0.0, max=1e9),),
        _run_fill_holes, "Cap open borders so the surface is closed (needed for volume).", replaceable=True),
    _op("orient", _CLEAN, "Orient faces (outward / flip)", ("mesh",),
        (_P("orient_mode", "Mode", "choice", "outward", choices=("outward", "flip")),),
        _run_orient, "Consistent outward normals (volume, curvature and lighting rely on them).", replaceable=True),
    _op("smooth", _SMOOTHING, "Smooth surface", ("mesh",),
        (_P("smooth_method", "Method", "choice", "taubin", choices=("taubin", "laplacian")),
         _P("smooth_iter", "Iterations", "int", 20, min=1, max=1000),
         _P("smooth_lambda", "Step (lambda)", "float", 0.5, min=0.01, max=1.0),
         _P("smooth_keep_boundary", "Keep open borders fixed", "bool", True)),
        _run_smooth, "Taubin λ|μ removes noise and staircases with almost no shrinkage; Laplacian shrinks.",
        replaceable=True),
    _op("decimate", _REMESH, "Decimate (fewer triangles)", ("mesh",),
        (_P("dec_reduction", "Fraction of faces to remove", "float", 0.5, min=0.0, max=0.99),
         _P("dec_method", "Method", "choice", "quadric", choices=("quadric", "pro")),
         _P("dec_topology", "Preserve topology (pro)", "bool", True)),
        _run_decimate, "Quadric-error decimation keeps the shape with a fraction of the triangles.", replaceable=True),
    _op("subdivide", _REMESH, "Subdivide (more triangles)", ("mesh",),
        (_P("subdiv_iter", "Iterations (×4 faces each)", "int", 1, min=1, max=4),
         _P("subdiv_method", "Scheme", "choice", "loop", choices=("loop", "butterfly", "linear"))),
        _run_subdivide, "Split each triangle in four; Loop and butterfly smooth as they refine.", replaceable=True),
    _op("remesh_voxel", _REMESH, "Remesh through voxels (watertight)", ("mesh",),
        (_P("voxel_spacing", "Voxel size (mm)", "float", 1.0, min=0.05, max=100.0), _SMOOTH),
        _run_remesh_voxel, "Rebuild a closed mesh on a voxel grid: repairs self-intersections, uniform triangles.",
        replaceable=True),
    _op("remesh_isotropic", _REMESH, "Isotropic remesh (even triangles)", ("mesh",),
        (_P("iso_length", "Target edge length (mm, 0 = current mean)", "float", 0.0, min=0.0, max=1000.0),
         _P("iso_iterations", "Iterations", "int", 10, min=1, max=100),
         _P("iso_feature", "Keep edges sharper than (°)", "float", 30.0, min=0.0, max=180.0),
         _P("iso_adaptive", "Adapt the length to the curvature", "bool", False)),
        _run_isotropic, "MeshLab's isotropic explicit remeshing: triangles of one size and good shape, on the "
        "original surface — for simulation, fair curvature and even sampling.", replaceable=True),
    _op("clip", _REMESH, "Clip with a plane at a click", ("mesh",),
        (_PLANE, _P("clip_keep", "Keep", "choice", "below", choices=("below", "above")),
         _P("clip_cap", "Cap the cut (closed result)", "bool", False)),
        _run_clip, "Run, then click on the surface: it is cut by the plane through that point — across the vessel, "
        "facing you, or normal to an axis.", replaceable=True, pick="surface"),
    _op("transform", _MOVE, "Translate / rotate / scale (exact values)", ("mesh", "points"),
        (_P("tr_translate", "Translate x, y, z (mm)", "str", "0, 0, 0"),
         _P("tr_rotate", "Rotate about x, y, z (degrees)", "str", "0, 0, 0"),
         _P("tr_scale", "Scale (one, or x, y, z)", "str", "1"),
         _P("tr_about_centroid", "Rotate and scale about the centroid", "bool", True)),
        _run_transform, "Move, turn or resize by typed values. To move by hand, use the Move card below.",
        replaceable=True),
    _op("icp", _MOVE, "Align to another surface (ICP)", ("mesh", "points"),
        (_P("icp_reference", "Fixed surface / points", "layer", ""),
         _P("icp_iterations", "Iterations", "int", 50, min=1, max=1000),
         _P("icp_scaling", "Allow uniform scaling", "bool", False),
         _P("icp_pca_start", "Start from principal axes (large misalignment)", "bool", False)),
        _run_icp, "Iterative closest point: rigid (or similarity) alignment onto a reference surface or cloud.",
        replaceable=True),
    _op("pca_align", _MOVE, "Principal axes to x / y / z", ("mesh", "points"), (),
        _run_pca_align, "Centre at the origin and turn the longest extent onto x (shape normalisation).",
        replaceable=True),
    _op("measure", _MEASURE, "Measure (area, volume, shape, topology)", ("mesh", "points", "series"), (),
        _run_measure, "Area, enclosed volume, sphericity, extents, holes, genus, edge lengths — per frame for a series."),
    _op("curvature", _MEASURE, "Curvature map", ("mesh",),
        (_P("curv_kind", "Curvature", "choice", "mean", choices=("mean", "gaussian", "maximum", "minimum")),
         _P("curv_smooth", "Smooth the map (iterations)", "int", 0, min=0, max=200)),
        _run_curvature, "Per-vertex curvature (1/mm), added to the layer as a field and shown. "
        "Smooth marching-cubes surfaces first."),
    _op("cross_section", _MEASURE, "Cross-section at a click", ("mesh",), (_PLANE,),
        _run_cross_section, "Run, then click on the surface: the section through that point — perpendicular to "
        "the vessel there, facing you, or normal to an axis. Contour, area, perimeter, diameters.", pick="surface"),
    _op("centerlines", _VESSELS, "Centerlines & diameter profile", ("mesh",),
        (_P("cl_every", "Section every (mm, 0 = every centerline point)", "float", 1.0, min=0.0, max=100.0),
         _P("cl_voxel", "Skeleton voxel size (mm, 0 = auto)", "float", 0.0, min=0.0, max=10.0),
         _P("cl_min_branch", "Drop branches shorter than (mm, 0 = auto)", "float", 0.0, min=0.0, max=1000.0),
         _P("cl_smooth", "Centerline smoothing", "float", 0.5, min=0.0, max=10.0),
         _P("cl_colour_surface", "Colour the surface by diameter", "bool", True)),
        _run_centerlines,
        "Skeleton of the closed surface split into branches. Along each: lumen area, area-equivalent / min / max "
        "diameter, circularity and the inscribed radius; per branch length, tortuosity, curvature and stenosis "
        "(vs its median diameter); angles at bifurcations. The centerline points carry the profile."),
    _op("vessel_section", _VESSELS, "Vessel cross-section at a click", ("mesh",), (),
        _run_vessel_section, "Run, then click on the vessel wall (or inside it, in 2D): the section perpendicular "
        "to the local vessel axis — no centerline needed.", pick="surface"),
    _op("thickness", _VESSELS, "Thickness / local diameter map", ("mesh",), (),
        _run_thickness, "At every vertex, the distance across to the opposite wall (shape diameter): a tube's "
        "diameter, a wall's thickness. Added as a field."),
    _op("distance", _COMPARE, "Distance to another surface", ("mesh", "points"),
        (_P("dist_reference", "Reference surface / points", "layer", ""),
         _P("dist_signed", "Signed (negative inside the reference)", "bool", False)),
        _run_distance, "Per-vertex distance map (added as a field) and the symmetric statistics (ASSD, RMS, Hausdorff)."),
    _op("to_mask", _CONVERT, "Mesh → mask", ("mesh",),
        (*_GRID, _P("mask_label", "Label value", "int", 1, min=1, max=65535)),
        _run_to_mask, "Fill the solid a closed surface encloses — onto an image's grid (any affine) or a new grid."),
    _op("to_sdf", _CONVERT, "Mesh → distance map (image)", ("mesh",), _GRID, _run_to_sdf,
        "Signed distance to the surface on a grid (mm, negative inside)."),
    _op("mesh_to_points", _CONVERT, "Mesh → points", ("mesh",),
        (_P("mp_mode", "Points", "choice", "sampled uniformly", choices=("sampled uniformly", "vertices")),
         _P("n_points", "How many (sampled)", "int", 10000, min=10, max=5_000_000)),
        _run_mesh_to_points, "Uniform samples on the surface (with normals), or the vertices with their fields."),
    _op("points_from_mask", _CONVERT, "Mask → points", ("image",),
        (_P("boundary_only", "Boundary voxels only", "bool", True),
         _P("max_points", "At most (0 = every voxel)", "int", 50000, min=0, max=10_000_000)),
        _run_points_from_mask, "One point per voxel (or boundary voxel) of the current frame, coloured by label."),
    _op("points_to_mask", _CONVERT, "Points → mask", ("points",),
        (*_GRID, _P("pm_radius", "Grow each point by (mm)", "float", 0.0, min=0.0, max=100.0),
         _P("pm_fill", "Fill the inside (points on a closed boundary)", "bool", False)),
        _run_points_to_mask, "Mark the voxels the points fall in (grown into balls), or the solid they enclose."),
    _op("points_to_mesh", _CONVERT, "Points → surface (reconstruct)", ("points", "mesh"), _RECON,
        _run_reconstruct, _RECON_DESC),
    _op("probe", _CONVERT, "Image values → surface / points", ("mesh", "points"),
        (_P("probe_image", "Image / labels", "layer", ""),
         _P("probe_order", "Interpolation", "choice", "linear", choices=("linear", "nearest")),
         _P("probe_depth", "Average across ± depth along the normal (mm)", "float", 0.0, min=0.0, max=100.0),
         _P("probe_mode", "Combine across depth", "choice", "mean", choices=("mean", "max", "min"))),
        _run_probe, "Read an image at every vertex (point) and colour by it — wall uptake on PET, the label a "
        "surface touches. Added as a field."),
    _op("reconstruct", _POINTS, "Reconstruct a surface", ("points", "mesh"), _RECON, _run_reconstruct, _RECON_DESC),
    _op("pc_downsample", _POINTS, "Downsample", ("points",),
        (_P("pc_down_method", "Method", "choice", "voxel", choices=("voxel", "random")),
         _P("pc_voxel", "Voxel size (mm)", "float", 2.0, min=0.01, max=1000.0),
         _P("pc_n", "Points to keep (random)", "int", 10000, min=1, max=100_000_000)),
        _run_pc_downsample, "One point per voxel (even density), or a random subset.", replaceable=True),
    _op("pc_outliers", _POINTS, "Remove outliers", ("points",),
        (_P("pc_out_method", "Method", "choice", "statistical", choices=("statistical", "radius")),
         _P("pc_k", "Neighbours (statistical)", "int", 16, min=2, max=500),
         _P("pc_std", "Std ratio (statistical)", "float", 2.0, min=0.1, max=20.0),
         _P("pc_radius", "Radius (mm, radius method)", "float", 2.0, min=0.01, max=1000.0),
         _P("pc_min_nb", "Min neighbours (radius method)", "int", 4, min=1, max=1000),
         _P("pc_show_outliers", "Also add the outliers as a layer", "bool", True)),
        _run_pc_outliers, "Drop isolated points: far from their neighbours, or with few within a radius.",
        replaceable=True),
    _op("pc_normals", _POINTS, "Estimate normals", ("points",),
        (_P("pc_k", "Neighbours", "int", 16, min=3, max=500),
         _P("pc_orient", "Orientation", "choice", "propagate", choices=("propagate", "outward", "none"))),
        _run_pc_normals, "Per-point normals from the local plane, oriented consistently along the surface "
        "(propagate) or away from the centre (outward); shown as vectors."),
    _op("series_measure", _TIME, "Measure over time (area, volume)", ("series",), (),
        _run_series_measure, "Area and volume per frame, plotted; volume range and ejection fraction."),
    _op("series_track", _TIME, "Track surface through time", ("series",),
        (_REFERENCE,
         _P("track_iterations", "Projection passes per frame", "int", 3, min=1, max=20),
         _P("track_smooth", "Smoothing between passes", "int", 5, min=0, max=50)),
        _run_series_track,
        "Carry one frame's mesh through the others so vertices correspond over time (needed for motion)."),
    _op("series_motion", _TIME, "Motion map (displacement / speed)", ("series",),
        (_P("motion_kind", "Show", "choice", "displacement", choices=("displacement", "speed")), _REFERENCE),
        _run_series_motion, "Colour a tracked surface by how far each point moved, or how fast (a field per frame)."),
    _op("series_smooth", _TIME, "Smooth over time", ("series",),
        (_P("time_sigma", "Sigma (frames)", "float", 1.0, min=0.1, max=50.0), _TIME_CYCLIC),
        _run_series_smooth, "Gaussian smoothing of each vertex trajectory (tracked series)."),
    _op("series_interp", _TIME, "Interpolate frames", ("series",),
        (_P("time_factor", "Frames × factor", "float", 2.0, min=0.1, max=20.0), _TIME_CYCLIC),
        _run_series_interp, "More (or fewer) frames over the same cycle, linearly per vertex (tracked series)."),
    _op("series_frame", _TIME, "Current frame as a mesh", ("series",), (),
        _run_series_frame, "Copy the frame on screen into a static mesh layer."),
)


# ──────────────────────────────────────────────────────────────────────────────
# MeshLab filters (PyMeshLab), one operation each
# ──────────────────────────────────────────────────────────────────────────────

#: The layer-picker entry meaning "the active layer" (MeshLab filters on two meshes).
ACTIVE_LAYER = "(active layer)"
#: Prefix of the MeshLab operations' ids.
MESHLAB_PREFIX = "ml:"
#: MeshLab filters that work on the current selection (they say so when there is none).
_MESHLAB_NEEDS_SELECTION = {
    "meshing_remove_selected_vertices", "meshing_remove_selected_faces", "meshing_remove_selected_vertices_and_faces",
    "generate_from_selected_faces", "apply_selection_dilatation", "apply_selection_erosion",
    "apply_selection_by_same_connected_component", "compute_scalar_by_geodesic_distance_from_selection_per_vertex",
    "compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex", "get_area_and_perimeter_of_selection",
    "generate_polyline_from_selection_perimeter", "generate_plane_fitting_to_selection",
    "compute_matrix_by_fitting_to_plane", "compute_selection_transfer_vertex_to_face",
    "compute_selection_transfer_face_to_vertex",
}


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:32] or "meshlab"


def _meshlab_param_spec(param: Any, first_mesh: bool) -> ParamSpec:
    kind = param.kind
    if kind == "mesh":
        return _P(param.name, param.label, "layer", ACTIVE_LAYER if first_mesh else _LAYER_NONE)
    if kind in ("bool", "int"):
        return _P(param.name, param.label, kind, param.default)
    if kind == "float":
        return _P(param.name, param.label, "float", param.default,
                  min=param.min if param.min is not None else -1e9, max=param.max if param.max is not None else 1e9)
    if kind == "enum":
        return _P(param.name, param.label, "choice", param.default, choices=param.choices)
    return _P(param.name, param.label, "str", param.default)


def _meshlab_runner(info: Any) -> Callable[[OpContext], OpResult]:
    def _run(ctx: OpContext) -> OpResult:
        return _run_meshlab(ctx, info)

    return _run


def _meshlab_op(info: Any) -> MeshOp:
    seen_mesh = False
    params = []
    for p in info.params:
        if p.name == info.pick_param:
            continue
        params.append(_meshlab_param_spec(p, first_mesh=not seen_mesh))
        seen_mesh = seen_mesh or p.kind == "mesh"
    desc = info.description or f"MeshLab filter {info.name}."
    if info.name in _MESHLAB_NEEDS_SELECTION:
        desc += " Works on the selection (Select → brush, box, piece… or a MeshLab selection filter)."
    if info.pick_param:
        desc += " Run, then click on the surface."
    if not info.inputs:
        desc += " Needs no layer: it adds a new one."
    return MeshOp(f"{MESHLAB_PREFIX}{info.name}", info.group, info.label, info.inputs, tuple(params),
                  _meshlab_runner(info), desc, replaceable=bool(info.inputs), pick="surface" if info.pick_param else "",
                  url=info.url)


def _run_meshlab(ctx: OpContext, info: Any) -> OpResult:
    """Run MeshLab filter *info* on the active layer and put back whatever it produced."""
    from nvitk.gui.mesh.layers import display_of, fields_of
    from nvitk.meshlab.pymeshlab_filters import reads_quality, run_meshlab_filter

    layer = ctx.layer if info.inputs else None
    obj = ctx.surface() if layer is not None else None
    if layer is not None and info.name in _MESHLAB_NEEDS_SELECTION:
        _selection_or_fail(ctx)
    params: dict[str, Any] = {}
    for p in info.params:
        if p.name == info.pick_param:
            params[p.name] = ctx.at()
            continue
        if p.name not in ctx.params:
            continue
        value = ctx.params[p.name]
        if p.kind == "mesh":
            name = str(value or "").strip()
            if name in ("", ACTIVE_LAYER):
                params[p.name] = None
            elif name == _LAYER_NONE:
                raise ValueError(f"Pick the {p.label.lower()}.")
            else:
                params[p.name] = ctx.surface(ctx.layer_named(p.name))
            continue
        params[p.name] = value
    quality = selected = None
    if layer is not None:
        fields = fields_of(layer)
        disp = display_of(layer)
        if disp.mode == "field" and disp.field in fields and reads_quality(info.name):
            quality = fields[disp.field]
        selected = fields.get("selected")
    res = run_meshlab_filter(obj, info.name, params, quality=quality, selected=selected)

    field_name = info.label
    added: list[Any] = []
    base = ctx.name if layer is not None else _slug(info.label)
    for k, out in enumerate(res.meshes):
        suffix = _slug(info.label) + (f"_{k + 1}" if len(res.meshes) > 1 else "")
        name = f"{base}_{suffix}" if layer is not None else (suffix if len(res.meshes) == 1 else f"{base}_{k + 1}")
        q = out.point_data.pop("quality", None)
        if isinstance(out, PointCloud) and "edges" in out.metadata:
            segs = [out.points[e] for e in np.asarray(out.metadata["edges"])]
            added.append(add_polylines_layer(ctx.viewer, segs, name=name))
            continue
        if isinstance(out, PointCloud):
            if q is not None:
                out.point_data[field_name] = q
            added.append(add_point_cloud_layer(ctx.viewer, out, name=name, color_by=field_name if q is not None else None))
        else:
            added.append(add_mesh_layer(ctx.viewer, out, name=name, values=q, field_name=field_name))
    if len(added) > 1:
        _gather(ctx, added, f"{base} {info.label}")

    target = layer
    notes: list[str] = []
    if res.current is not None and layer is not None:
        cur = res.current
        if (cur.n_vertices if isinstance(cur, Mesh) else cur.n_points) == 0:
            raise ValueError(f"{info.label} would leave nothing of {ctx.name}.")
        q = cur.point_data.pop("quality", None)
        if q is None and res.quality is not None and len(res.quality) == (
                cur.n_vertices if isinstance(cur, Mesh) else cur.n_points):
            q = res.quality
        if isinstance(cur, Mesh):
            target = ctx.put_mesh(cur, _slug(info.label))
        else:
            target = ctx.put_cloud(cur, _slug(info.label))
        if q is not None:
            add_field(target, field_name, q, show=True)
        n = cur.n_faces if isinstance(cur, Mesh) else cur.n_points
        notes.append(f"{n:,} {'faces' if isinstance(cur, Mesh) else 'points'}")
    elif res.quality is not None and layer is not None:
        add_field(layer, field_name, res.quality, show=True)
        v = res.quality[np.isfinite(res.quality)]
        if v.size:
            notes.append(f"{field_name}: {v.min():.4g} … {v.max():.4g}")
    if res.selected is not None and target is not None:
        from nvitk.gui.mesh.selection import clear_selection, set_selection

        n_sel = int(np.sum(res.selected > 0.5))
        if n_sel:
            set_selection(target, res.selected > 0.5)
            notes.append(f"{n_sel:,} vertices selected")
        elif "selected" in fields_of(target):
            clear_selection(target)
            notes.append("selection cleared")
    if added:
        notes.append(f"{len(added)} new layer{'s' if len(added) > 1 else ''}")
    table = None
    if res.values:
        table = {str(k): _fmt_value(v) for k, v in res.values.items()}
    on = f" on {ctx.name}" if layer is not None else ""
    message = f"MeshLab {info.label}{on}" + (": " + ", ".join(notes) if notes else " done.")
    return OpResult(message, table=table, title=f"{info.label}: {base}")


def _fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        flat = np.asarray(value, dtype=float).ravel()
        if flat.size <= 16:
            return ", ".join(f"{v:.5g}" for v in flat)
        return f"{flat.size} values"
    if isinstance(value, np.ndarray):
        return f"array {value.shape}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


_MESHLAB_OPS: tuple[MeshOp, ...] | None = None


def meshlab_operations() -> tuple[MeshOp, ...]:
    """One operation per MeshLab filter (none without PyMeshLab)."""
    global _MESHLAB_OPS
    if _MESHLAB_OPS is None:
        try:
            from nvitk.meshlab.pymeshlab_filters import meshlab_filters

            _MESHLAB_OPS = tuple(_meshlab_op(info) for info in meshlab_filters())
        except Exception as exc:  # noqa: BLE001 — MeshLab is optional; the native tools still work
            from nvitk.gui.core.log_panel import gui_log

            gui_log(f"MeshLab filters unavailable: {exc}", error=True)
            _MESHLAB_OPS = ()
    return _MESHLAB_OPS


def all_operations() -> tuple[MeshOp, ...]:
    """The native operations, then the MeshLab filters."""
    return OPERATIONS + meshlab_operations()


def categories() -> list[str]:
    """Native categories, then MeshLab's groups."""
    extra: list[str] = []
    for op in meshlab_operations():
        if op.category not in extra:
            extra.append(op.category)
    return list(CATEGORIES) + extra


def operations_for(category: str) -> list[MeshOp]:
    return [op for op in all_operations() if op.category == category]


def operation(op_id: str) -> MeshOp | None:
    return next((op for op in all_operations() if op.id == op_id), None)


def accepts(op: MeshOp, layer: Any) -> bool:
    """True when *op* can run on *layer* (generators run with any or no layer)."""
    if not op.inputs:
        return True
    return layer_kind(layer) in op.inputs or ("mesh" in op.inputs and layer_kind(layer) == "series")


def input_hint(op: MeshOp) -> str:
    if not op.inputs:
        return "nothing (it makes a new layer)"
    names = {"mesh": "a surface", "points": "a points layer", "image": "an image / labels layer",
             "series": "a mesh time series"}
    return " or ".join(names[k] for k in op.inputs)


__all__ = ["ACTIVE_LAYER", "CATEGORIES", "MESHLAB_PREFIX", "MeshOp", "OPERATIONS", "OpContext", "OpResult",
           "SURFACE_OUTPUTS", "accepts", "all_operations", "categories", "input_hint", "label_info",
           "meshlab_operations", "operation", "operations_for", "parse_groups", "present_label_ids", "set_points"]
