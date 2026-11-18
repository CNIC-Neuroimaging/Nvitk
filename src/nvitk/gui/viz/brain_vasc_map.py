"""
The brain-vascular map in the 3D canvas: vessels, the brain, and what a clicked vessel supplies.

One scene per viewer. Every vessel label is a solid mesh in its label colour; the parenchyma is a
faint grey shell around them. A left click on a vessel (a click, not the drag that rotates the
camera) highlights it, dims the others, and draws the parcellation regions in its territory
(:mod:`nvitk.measure.vascular_territory`) in their own colours. The dock lists the regions and
their shares, and can select a vessel by name as well.

Everything is built in world coordinates (napari's, which for NIfTI files is the scanner's mm
space), so the meshes stay where they belong in either canvas resampling mode, and a click is
resolved against the vessel voxels in world space: no napari layer has to be hit by the ray.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.gui.core.layer_folders import FOLDER_KEY
from nvitk.measure.vascular_territory import TerritoryResult, downstream

log = Logger()

#: Layer-list folder of everything the scene adds.
FOLDER = "Brain-vascular map"
VESSEL_FOLDER = f"{FOLDER}/Vessels"
#: Name prefix of the scene's layers, so a rebuild removes its own and nothing else.
PREFIX = "bvm:"
SHELL_LAYER = f"{PREFIX} brain"
REGIONS_LAYER = f"{PREFIX} irrigated regions"
FOCUS_LAYER = f"{PREFIX} focused region"
#: The irrigated regions as a label volume, for the orthogonal views.
SLICES_LAYER = f"{PREFIX} irrigated regions (slices)"
PARCELLATION_LAYER = f"{PREFIX} parcellation (vessel space)"
TERRITORY_LAYER = f"{PREFIX} vascular territories"

_SCENE_ATTR = "_nvitk_brain_vasc_scene"

DEFAULT_SHELL_OPACITY = 0.08
DEFAULT_DIM_OPACITY = 0.5
DEFAULT_REGION_OPACITY = 0.55
DEFAULT_MIN_FRACTION = 0.25
DEFAULT_SLICE_OPACITY = 0.5
#: A focused region is drawn at least this opaque; the selected vessel's other regions this faint.
DEFAULT_FOCUS_OPACITY = 0.85
DEFAULT_UNFOCUSED_OPACITY = 0.08
DEFAULT_UNFOCUSED_SLICE_ALPHA = 0.2

#: Region-type filters offered by the dock: label -> categories shown (``None``: all).
#: Categories are :func:`nvitk.segmentation.brain_parcellation.region_category`'s.
REGION_TYPES: tuple[tuple[str, frozenset[str] | None], ...] = (
    ("All regions", None),
    ("Cortical", frozenset({"cortex"})),
    ("Non-cortical", frozenset({"subcortical", "white matter", "cerebellum", "brainstem", "other"})),
    ("Subcortical grey matter", frozenset({"subcortical"})),
    ("White matter", frozenset({"white matter"})),
    ("Cerebellum + brainstem", frozenset({"cerebellum", "brainstem"})),
)
#: A press that moves less than this (screen pixels) before release is a click, not a rotation.
_CLICK_SLOP_PX = 4.0
#: How far (mm) from the ray a vessel voxel may be and still count as clicked, when the ray
#: passes no voxel outright: a forgiving margin for thin vessels.
_PICK_TOLERANCE_MM = 2.5


# ──────────────────────────────────────────────────────────────────────────────
# Geometry (pure arrays: safe off the GUI thread)
# ──────────────────────────────────────────────────────────────────────────────


def _to_world(verts: np.ndarray, index_to_world: np.ndarray) -> np.ndarray:
    """Voxel-index vertices -> world (float32)."""
    m = np.asarray(index_to_world, dtype=float)[-4:, -4:]
    return (verts @ m[:3, :3].T + m[:3, 3]).astype(np.float32)


def mask_mesh(mask: np.ndarray, index_to_world: np.ndarray, *, sigma: float = 0.6,
              offset: tuple[int, int, int] = (0, 0, 0)) -> tuple[np.ndarray, np.ndarray] | None:
    """
    A smooth surface of a binary *mask* (a crop starting at voxel *offset*), in world mm.

    Smoothing the mask before contouring at 0.5 rounds the voxel staircase without moving the
    surface; *sigma* is in voxels.
    """
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes

    m = np.pad(np.asarray(mask, dtype=np.float32), 2)
    if sigma > 0:
        m = ndi.gaussian_filter(m, float(sigma))
    if not (m.max() > 0.5 > m.min()):
        return None
    try:
        verts, faces, _n, _v = marching_cubes(m, level=0.5, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    verts = verts - 2.0 + np.asarray(offset, dtype=float)
    return _to_world(verts, index_to_world), faces.astype(np.int64)


def label_meshes(labels: np.ndarray, index_to_world: np.ndarray, ids: list[int], *,
                 sigma: float = 0.6, pad: int = 3) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One world-space mesh per label id, each contoured on its own bounding box."""
    from scipy import ndimage as ndi

    lab = np.asarray(labels)
    boxes = ndi.find_objects(lab.astype(np.int32))
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lid in ids:
        if lid < 1 or lid > len(boxes) or boxes[lid - 1] is None:
            continue
        box = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(boxes[lid - 1], lab.shape))
        mesh = mask_mesh(lab[box] == lid, index_to_world, sigma=sigma, offset=tuple(s.start for s in box))
        if mesh is not None:
            out[int(lid)] = mesh
    return out


@dataclass
class SceneData:
    """Everything the scene draws, computed off the GUI thread."""

    result: TerritoryResult
    vessel_names: dict[int, str]
    vessel_colours: dict[int, tuple[float, float, float, float]]
    region_names: dict[int, str]
    vessel_meshes: dict[int, tuple[np.ndarray, np.ndarray]]
    shell: tuple[np.ndarray, np.ndarray] | None
    region_meshes: dict[int, tuple[np.ndarray, np.ndarray]]
    #: World positions and labels of every vessel voxel, for picking.
    pick_points: np.ndarray
    pick_labels: np.ndarray
    pick_radius_mm: float
    notes: list[str] = field(default_factory=list)


def build_scene_data(
    result: TerritoryResult,
    vessels: np.ndarray,
    vessel_to_world: np.ndarray,
    vessel_names: dict[int, str],
    vessel_colours: dict[int, tuple[float, float, float, float]],
    region_names: dict[int, str],
) -> SceneData:
    """Meshes and pick points for *result* (vessel meshes at the vessel grid's resolution)."""
    vessels = np.asarray(vessels)
    present = sorted(int(v) for v in np.unique(vessels) if int(v) > 0)
    m_v = np.asarray(vessel_to_world, dtype=float)[-4:, -4:]
    vessel_meshes = label_meshes(vessels, m_v, present, sigma=0.6)

    work = np.asarray(to_numpy(result.work_to_world), dtype=float)
    parenchyma = np.asarray(to_numpy(result.parenchyma), dtype=bool)
    shell = mask_mesh(parenchyma, work, sigma=1.0)
    parc = np.asarray(to_numpy(result.parcellation))
    region_meshes = label_meshes(parc, work, [int(r) for r in result.table.region_ids], sigma=0.8, pad=2)

    idx = np.argwhere(vessels > 0)
    pick_points = _to_world(idx.astype(np.float64), m_v)
    pick_labels = vessels[tuple(idx.T)].astype(np.int32)
    spacing = np.linalg.norm(m_v[:3, :3], axis=0)
    return SceneData(
        result=result,
        vessel_names=dict(vessel_names),
        vessel_colours=dict(vessel_colours),
        region_names=dict(region_names),
        vessel_meshes=vessel_meshes,
        shell=shell,
        region_meshes=region_meshes,
        pick_points=pick_points,
        pick_labels=pick_labels,
        pick_radius_mm=float(max(0.75 * float(spacing.max()), 0.3)),
    )


def pick_vessel(points: np.ndarray, labels: np.ndarray, origin: Any, direction: Any, *,
                radius_mm: float, tolerance_mm: float = _PICK_TOLERANCE_MM) -> int | None:
    """
    The vessel under a 3D click: the front-most voxel the ray passes through, else the voxel
    nearest the ray within *tolerance_mm*.

    *origin* and *direction* are the click's world position and the camera's view direction
    (pointing into the scene), as napari reports them.
    """
    if len(points) == 0:
        return None
    o = np.asarray(origin, dtype=np.float64)[-3:]
    d = np.asarray(direction, dtype=np.float64)[-3:]
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return None
    d /= norm
    rel = points.astype(np.float64) - o
    t = rel @ d
    perp2 = np.einsum("ij,ij->i", rel, rel) - t * t
    hit = perp2 <= float(radius_mm) ** 2
    if hit.any():
        candidates = np.flatnonzero(hit)
        return int(labels[candidates[np.argmin(t[candidates])]])
    near = perp2 <= float(tolerance_mm) ** 2
    if not near.any():
        return None
    candidates = np.flatnonzero(near)
    return int(labels[candidates[np.argmin(perp2[candidates])]])


# ──────────────────────────────────────────────────────────────────────────────
# The scene in the viewer
# ──────────────────────────────────────────────────────────────────────────────


def _rgba(colour: Any, fallback: tuple[float, ...] = (0.8, 0.2, 0.2, 1.0)) -> tuple[float, float, float, float]:
    try:
        c = np.asarray(colour, dtype=float).ravel()
    except (TypeError, ValueError):
        return tuple(fallback)  # type: ignore[return-value]
    if c.size < 3 or not np.all(np.isfinite(c)):
        return tuple(fallback)  # type: ignore[return-value]
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def _solid(mesh: tuple[np.ndarray, np.ndarray], rgba: tuple[float, ...]) -> dict[str, Any]:
    """``add_surface`` data + ``vertex_colors`` for *mesh* in one colour."""
    verts, faces = mesh
    return {"data": (verts, faces), "vertex_colors": np.tile(np.asarray(rgba, dtype=np.float32), (len(verts), 1))}


def clear_scene_layers(viewer: Any) -> None:
    """Remove every layer the scene added (and nothing else)."""
    for layer in list(viewer.layers):
        if str(getattr(layer, "name", "")).startswith(PREFIX):
            try:
                viewer.layers.remove(layer)
            except (ValueError, KeyError):
                pass


class BrainVascScene:
    """The drawn scene of one viewer: its layers, the selection and the click handling."""

    def __init__(self, viewer: Any, data: SceneData, *, on_select: Callable[[int | None], None] | None = None,
                 shell_opacity: float = DEFAULT_SHELL_OPACITY, dim_opacity: float = DEFAULT_DIM_OPACITY,
                 region_opacity: float = DEFAULT_REGION_OPACITY, min_fraction: float = DEFAULT_MIN_FRACTION) -> None:
        self._viewer = weakref.ref(viewer)
        self.data = data
        self.on_select = on_select
        self.shell_opacity = float(shell_opacity)
        self.dim_opacity = float(dim_opacity)
        self.region_opacity = float(region_opacity)
        self.min_fraction = float(min_fraction)
        self.selected: int | None = None
        #: Region categories shown (:data:`REGION_TYPES` values); ``None`` shows every region.
        self.region_types: frozenset[str] | None = None
        #: The region the table focused (drawn strongly, the rest faint); ``None`` for none.
        self.focused: int | None = None
        self._vessel_layers: dict[int, Any] = {}
        self._shell: Any = None
        self._regions: Any = None
        self._focus: Any = None
        self._slices: Any = None
        self._parcellation: np.ndarray | None = None
        self.region_colours: dict[int, tuple[float, float, float, float]] = {}
        self.region_categories: dict[int, str] = {}

    @property
    def viewer(self) -> Any:
        return self._viewer()

    # ---- building -------------------------------------------------------------------

    def build(self, *, extra_labels: list[tuple[np.ndarray, dict[str, Any]]] = ()) -> list[Any]:
        """Add the scene's layers (removing a previous scene's) and switch to 3D."""
        from nvitk.segmentation.brain_parcellation import region_category

        viewer = self.viewer
        clear_scene_layers(viewer)
        added: list[Any] = []
        result = self.data.result
        work = np.asarray(to_numpy(result.work_to_world), dtype=float)
        self._parcellation = np.asarray(to_numpy(result.parcellation), dtype=np.int32)

        # Label volumes first: they colour the regions, and sit under the meshes in the list.
        parc_layer = viewer.add_labels(
            self._parcellation, name=PARCELLATION_LAYER,
            affine=work, visible=False, opacity=0.5, metadata={FOLDER_KEY: FOLDER},
        )
        added.append(parc_layer)
        terr_layer = viewer.add_labels(
            np.asarray(to_numpy(result.territories), dtype=np.int32), name=TERRITORY_LAYER,
            affine=work, visible=False, opacity=0.5, metadata={FOLDER_KEY: FOLDER},
        )
        self._colour_territories(terr_layer)
        added.append(terr_layer)
        self.region_colours = {
            int(r): _rgba(parc_layer.get_color(int(r)), (0.6, 0.6, 0.6, 1.0)) for r in result.table.region_ids
        }
        self.region_categories = {
            int(r): region_category(self.region_name(int(r))) for r in result.table.region_ids
        }
        # The selected vessel's regions as a label volume, for the orthogonal views: drawn there,
        # kept off the 3D canvas (the meshes show them). Empty and hidden until a selection.
        self._slices = viewer.add_labels(
            np.zeros_like(self._parcellation), name=SLICES_LAYER, affine=work, visible=False,
            opacity=DEFAULT_SLICE_OPACITY, metadata={FOLDER_KEY: FOLDER},
        )
        added.append(self._slices)
        self._sync_slices_canvas()
        for data, kwargs in extra_labels:
            added.append(viewer.add_labels(data, **{**kwargs, "metadata": {
                **dict(kwargs.get("metadata") or {}), FOLDER_KEY: FOLDER}}))

        if self.data.shell is not None:
            self._shell = viewer.add_surface(
                **_solid(self.data.shell, (0.75, 0.75, 0.75, 1.0)), name=SHELL_LAYER,
                opacity=self.shell_opacity, shading="smooth",
                # 'translucent_no_depth': a depth-writing shell would cull the vessels inside it.
                blending="translucent_no_depth", metadata={FOLDER_KEY: FOLDER},
            )
            added.append(self._shell)

        for lid in sorted(self.data.vessel_meshes):
            layer = viewer.add_surface(
                **_solid(self.data.vessel_meshes[lid], self.data.vessel_colours.get(lid, (0.8, 0.2, 0.2, 1.0))),
                name=f"{PREFIX} {self.vessel_name(lid)}", opacity=1.0, shading="smooth",
                blending="translucent", metadata={FOLDER_KEY: VESSEL_FOLDER, "nvitk_label_id": int(lid)},
            )
            self._vessel_layers[lid] = layer
            added.append(layer)

        try:
            viewer.dims.ndisplay = 3
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not switch to 3D: %s", exc)
        try:
            viewer.dims.events.ndisplay.connect(self._sync_slices_canvas)
        except Exception as exc:  # noqa: BLE001
            log.debug("No ndisplay events: %s", exc)
        install_click_handler(viewer, self)
        return added

    def _colour_territories(self, layer: Any) -> None:
        """Territory voxels in their vessel's colour."""
        try:
            from napari.utils.colormaps import DirectLabelColormap

            colours = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
            colours.update({int(k): v for k, v in self.data.vessel_colours.items()})
            layer.colormap = DirectLabelColormap(color_dict=colours)
        except Exception as exc:  # noqa: BLE001 (cosmetic)
            log.debug("Territory colours not applied: %s", exc)

    def _sync_slices_canvas(self, _event: Any = None) -> None:
        """The slice overlay is for the orthogonal views: off the main canvas while it shows 3D
        (the region meshes are there), on it in 2D, where it is the natural slice view."""
        from nvitk.gui.viz.main_canvas import keep_off_canvas

        viewer = self.viewer
        if viewer is None or not self._alive(self._slices):
            return
        try:
            keep_off_canvas(viewer, self._slices, int(viewer.dims.ndisplay) == 3)
        except Exception as exc:  # noqa: BLE001 (cosmetic: at worst the overlay also shows in 3D)
            log.debug("Could not keep the slice overlay off the canvas: %s", exc)

    # ---- queries ----------------------------------------------------------------------

    def vessel_name(self, lid: int) -> str:
        return self.data.vessel_names.get(int(lid)) or f"Label {int(lid)}"

    def region_name(self, rid: int) -> str:
        return self.data.region_names.get(int(rid)) or f"Region {int(rid)}"

    def vessel_ids(self) -> list[int]:
        return sorted(self.data.vessel_meshes)

    def categories_present(self) -> set[str]:
        """Region categories the parcellation has (for the region-type picker)."""
        return set(self.region_categories.values())

    def irrigated(self, lid: int) -> list[tuple[int, float]]:
        """Regions *lid* supplies above the share threshold, of the shown region types."""
        rows = self.data.result.table.irrigated(int(lid), self.min_fraction)
        if self.region_types is None:
            return rows
        return [(rid, f) for rid, f in rows if self.region_categories.get(rid, "other") in self.region_types]

    def downstream(self, lid: int) -> list[int]:
        return [v for v in downstream(int(lid), self.data.result.table.tree) if v != int(lid)]

    def has_territory(self, lid: int) -> bool:
        """False for labels outside the supply tree (veins, unknown names)."""
        return int(lid) not in self.data.result.unclassified

    def pick(self, position: Any, view_direction: Any) -> int | None:
        return pick_vessel(self.data.pick_points, self.data.pick_labels, position, view_direction,
                           radius_mm=self.data.pick_radius_mm)

    # ---- selection and appearance ---------------------------------------------------

    def _alive(self, layer: Any) -> bool:
        viewer = self.viewer
        return layer is not None and viewer is not None and layer in viewer.layers

    def _notify(self) -> None:
        if self.on_select is not None:
            self.on_select(self.selected)

    def select(self, lid: int | None) -> None:
        """Highlight *lid* (``None`` clears) and draw the regions it irrigates."""
        self.selected = None if lid is None or int(lid) not in self._vessel_layers else int(lid)
        self.focused = None
        for vid, layer in self._vessel_layers.items():
            if self._alive(layer):
                layer.opacity = 1.0 if self.selected in (None, vid) else self.dim_opacity
        self._draw_regions()
        self._notify()

    def focus_region(self, rid: int | None) -> None:
        """Draw region *rid* strongly and the selected vessel's other regions faintly (``None``,
        or the focused region again, shows them all evenly)."""
        rid = None if rid is None or int(rid) == self.focused else int(rid)
        self.focused = rid if rid is not None and rid in dict(self._shown()) else None
        self._draw_regions()
        self._notify()

    def _shown(self) -> list[tuple[int, float]]:
        return self.irrigated(self.selected) if self.selected is not None else []

    def _merged_mesh(self, ids: list[int]) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
        """One mesh of the regions *ids*, each vertex in its region's colour."""
        verts, faces, colours, base = [], [], [], 0
        for rid in ids:
            if rid not in self.data.region_meshes:
                continue
            v, f = self.data.region_meshes[rid]
            verts.append(v)
            faces.append(f + base)
            colours.append(np.tile(np.asarray(self.region_colours.get(rid, (0.6, 0.6, 0.6, 1.0)), np.float32),
                                   (len(v), 1)))
            base += len(v)
        if not verts:
            return None
        return (np.concatenate(verts), np.concatenate(faces)), np.concatenate(colours)

    def _put_surface(self, layer: Any, ids: list[int], name: str, opacity: float) -> Any:
        """Show the regions *ids* in *layer* (created on first use); hide it when there are none."""
        built = self._merged_mesh(ids)
        if built is None:
            if self._alive(layer):
                layer.visible = False
            return layer
        mesh, vertex_colors = built
        if self._alive(layer):
            # Colours off first: napari checks them against the vertex count on every assignment.
            layer.vertex_colors = None
            layer.data = mesh
            layer.vertex_colors = vertex_colors
            layer.opacity = opacity
            layer.visible = True
            return layer
        return self.viewer.add_surface(
            mesh, vertex_colors=vertex_colors, name=name, opacity=opacity, shading="smooth",
            blending="translucent_no_depth", metadata={FOLDER_KEY: FOLDER},
        )

    def _draw_regions(self) -> None:
        ids = [rid for rid, _f in self._shown()]
        if self.focused not in ids:
            self.focused = None
        if self.focused is None:
            self._regions = self._put_surface(self._regions, ids, REGIONS_LAYER, self.region_opacity)
            self._focus = self._put_surface(self._focus, [], FOCUS_LAYER, self.region_opacity)
        else:
            rest = [rid for rid in ids if rid != self.focused]
            self._regions = self._put_surface(self._regions, rest, REGIONS_LAYER, DEFAULT_UNFOCUSED_OPACITY)
            self._focus = self._put_surface(self._focus, [self.focused], FOCUS_LAYER,
                                            max(self.region_opacity, DEFAULT_FOCUS_OPACITY))
        self._draw_slices(ids)

    def _draw_slices(self, ids: list[int]) -> None:
        """The shown regions as labels for the orthogonal views (the focused one emphasised)."""
        layer = self._slices
        if not self._alive(layer) or self._parcellation is None:
            return
        if not ids:
            # Emptied too, not only hidden: shown again by hand it must not bring back old regions.
            if np.any(np.asarray(layer.data)):
                layer.data = np.zeros_like(self._parcellation)
            layer.visible = False
            return
        keep = np.zeros(int(self._parcellation.max()) + 1, dtype=bool)
        keep[np.asarray(ids, dtype=np.int64)] = True
        layer.data = np.where(keep[self._parcellation], self._parcellation, 0).astype(np.int32)
        try:
            from napari.utils.colormaps import DirectLabelColormap

            colours: dict[Any, tuple[float, ...]] = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
            for rid in ids:
                r, g, b, _a = self.region_colours.get(rid, (0.6, 0.6, 0.6, 1.0))
                faint = self.focused is not None and rid != self.focused
                colours[rid] = (r, g, b, DEFAULT_UNFOCUSED_SLICE_ALPHA if faint else 1.0)
            layer.colormap = DirectLabelColormap(color_dict=colours)
        except Exception as exc:  # noqa: BLE001 (cosmetic: napari's own colours otherwise)
            log.debug("Region colours not applied to the slice overlay: %s", exc)
        layer.visible = True
        self._sync_slices_canvas()

    def set_region_types(self, types: frozenset[str] | None) -> None:
        """Show only regions of these categories (``None``: every region)."""
        self.region_types = None if types is None else frozenset(types)
        if self.selected is not None:
            self._draw_regions()
            self._notify()

    def set_min_fraction(self, value: float) -> None:
        self.min_fraction = float(value)
        if self.selected is not None:
            self._draw_regions()
            self._notify()

    def set_shell_opacity(self, value: float) -> None:
        self.shell_opacity = float(value)
        if self._alive(self._shell):
            self._shell.opacity = self.shell_opacity

    def set_dim_opacity(self, value: float) -> None:
        self.dim_opacity = float(value)
        for vid, layer in self._vessel_layers.items():
            if self._alive(layer):
                layer.opacity = 1.0 if self.selected in (None, vid) else self.dim_opacity

    def set_region_opacity(self, value: float) -> None:
        self.region_opacity = float(value)
        if self.selected is not None:
            self._draw_regions()

    def remove(self) -> None:
        viewer = self.viewer
        if viewer is not None:
            try:
                viewer.dims.events.ndisplay.disconnect(self._sync_slices_canvas)
            except Exception:  # noqa: BLE001
                pass
            clear_scene_layers(viewer)
            if getattr(viewer, _SCENE_ATTR, None) is self:
                setattr(viewer, _SCENE_ATTR, None)
        self._vessel_layers.clear()
        self._shell = self._regions = self._focus = self._slices = None


# ──────────────────────────────────────────────────────────────────────────────
# Clicks
# ──────────────────────────────────────────────────────────────────────────────


def current_scene(viewer: Any) -> BrainVascScene | None:
    return getattr(viewer, _SCENE_ATTR, None)


def _on_drag(viewer: Any, event: Any):  # generator: napari drag protocol
    """Select the vessel under a left *click* in 3D; a drag (camera rotation) selects nothing."""
    scene = current_scene(viewer)
    if scene is None or getattr(event, "button", 1) != 1 or int(viewer.dims.ndisplay) != 3:
        return
    if getattr(event, "modifiers", None):
        return  # shift / ctrl clicks belong to napari and the layer tools
    start = np.asarray(getattr(event, "pos", (0.0, 0.0)), dtype=float)
    yield
    moved = False
    while event.type == "mouse_move":
        if np.linalg.norm(np.asarray(getattr(event, "pos", start), dtype=float) - start) > _CLICK_SLOP_PX:
            moved = True
        yield
    if moved:
        return
    view_dir = getattr(event, "view_direction", None)
    if view_dir is None:
        # camera.view_direction is in *displayed*-dims order; the ray needs world order.
        try:
            view_dir = viewer.camera.calculate_nd_view_direction(viewer.dims.ndim, viewer.dims.displayed)
        except Exception:  # noqa: BLE001
            return
    if view_dir is None:
        return
    lid = scene.pick(event.position, view_dir)
    if lid is not None:
        scene.select(lid)


def install_click_handler(viewer: Any, scene: BrainVascScene) -> None:
    """Make *scene* the viewer's brain-vascular scene and route 3D clicks to it (once)."""
    setattr(viewer, _SCENE_ATTR, scene)
    if _on_drag not in viewer.mouse_drag_callbacks:
        viewer.mouse_drag_callbacks.append(_on_drag)


__all__ = [
    "REGION_TYPES",
    "BrainVascScene",
    "FOLDER",
    "PREFIX",
    "SceneData",
    "build_scene_data",
    "clear_scene_layers",
    "current_scene",
    "install_click_handler",
    "label_meshes",
    "mask_mesh",
    "pick_vessel",
]
