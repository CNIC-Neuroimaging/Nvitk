"""The views on the 3D canvas: slice planes and outlines at the crosshair, and see-inside clipping."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.gui.labels.visibility import is_label_like_layer
from nvitk.gui.viz.ortho.constants import (
    _PLANE_LAYER_SUFFIX,
    _BOX_LAYER_SUFFIX,
    BOX_AXIS_COLORS,
    _BOX_EDGE_WIDTH,
)
from nvitk.gui.viz.ortho.geometry import (
    AxisView,
    PlaneFrame,
)
from nvitk.gui.viz.ortho.sources import (
    layer_time_axis,
    _spatial_shape,
    _layer_volume,
)

# 3D planes and clipping on the Napari canvas
def _plane_layer_name(source_name: str, axis: int) -> str:
    """Name of the canvas layer carrying *axis*'s 3D slice plane."""
    return f"{source_name}{_PLANE_LAYER_SUFFIX}_{axis}"


def displayed_axes(viewer: Any, ndim: int = 3) -> list[int]:
    """Array axes in the order Napari is currently displaying them.

    ``viewer.dims.order`` is rarely the identity here (nvitk sets it so a volume
    opens axially), and a plane's geometry is expressed in *that* order, not in
    array-axis order.
    """
    try:
        order = [int(a) for a in viewer.dims.displayed]
        # With a 3D+t layer open the world has a leading time dim; the spatial
        # dims (the trailing three, where every 3D layer aligns) are what count.
        offset = int(viewer.dims.ndim) - int(ndim)
        order = [a - offset for a in order]
    except Exception:
        order = []
    if len(order) != ndim or sorted(order) != list(range(ndim)):
        return list(range(ndim))
    return order


def _plane_geometry(
    axis: int,
    position: tuple[int, int, int],
    shape: tuple[int, ...],
    displayed: list[int],
    frame: PlaneFrame | None = None,
    spacing: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """``(point, normal)`` for one cut, in Napari's displayed-dims order.

    Two frames meet here and must not be confused. ``Plane.position`` and
    ``Plane.normal`` are data (voxel-index) coordinates *permuted into displayed
    order*: Napari's own convention. ``PlaneFrame`` holds unit vectors in
    spacing-scaled millimetre space; a direction converts to voxels by dividing
    by the spacing, so the plane normal is the cross product of the two converted
    in-plane directions, permuted **after** the cross product (an odd permutation
    flips a cross product's sign).

    ``frame=None`` reproduces the axis-aligned behaviour exactly.
    """
    if frame is None:
        slot = displayed.index(int(axis))
        normal = [0.0, 0.0, 0.0]
        normal[slot] = 1.0
        point = [
            float(position[data_axis])
            if data_axis == int(axis)
            else float(shape[data_axis]) / 2.0
            for data_axis in displayed
        ]
        return point, normal

    sp = np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=float)[:3]
    row_vox = np.asarray(frame.row, dtype=float) / sp
    col_vox = np.asarray(frame.col, dtype=float) / sp
    normal_vox = np.cross(row_vox, col_vox)
    length = float(np.linalg.norm(normal_vox))
    if length <= 1e-12:
        normal_vox = np.zeros(3)
        normal_vox[int(axis)] = 1.0
    else:
        normal_vox = normal_vox / length
    # A turned plane passes through the crosshair, not the volume's middle:
    # rotating about the centre would slide the cut away from what is on screen.
    point = [float(position[data_axis]) for data_axis in displayed]
    normal = [float(normal_vox[data_axis]) for data_axis in displayed]
    return point, normal

def remove_ortho_planes(viewer: Any, source_name: str) -> int:
    """Drop every 3D plane layer nvitk added for *source_name*; returns how many."""
    removed = 0
    for layer in list(getattr(viewer, "layers", []) or []):
        name = str(getattr(layer, "name", ""))
        if name.startswith(f"{source_name}{_PLANE_LAYER_SUFFIX}"):
            try:
                viewer.layers.remove(layer)
                removed += 1
            except Exception:
                pass
    return removed


#: Display properties a slice plane copies from the layer it shows, so the cut in
#: 3D reads with the same window, colours and brightness as the 2D views.
_IMAGE_STYLE_ATTRS: tuple[str, ...] = (
    "colormap", "contrast_limits", "gamma", "interpolation2d", "interpolation3d",
)
_LABEL_STYLE_ATTRS: tuple[str, ...] = ("colormap",)


def copy_plane_style(source: Any, plane: Any) -> None:
    """Give a slice *plane* the display window and colours of the *source* layer.

    Planes used to be added with Napari's defaults (the full data range and a
    grey ramp), so a CT windowed for soft tissue appeared washed out in 3D, and
    additive blending brightened it further wherever it crossed the volume.
    Only properties that differ are written: each assignment is an event.
    """
    labels = type(source).__name__ == "Labels" or is_label_like_layer(source)
    for attr in (_LABEL_STYLE_ATTRS if labels else _IMAGE_STYLE_ATTRS):
        if not hasattr(source, attr) or not hasattr(plane, attr):
            continue
        try:
            value = getattr(source, attr)
            current = getattr(plane, attr)
            if attr == "contrast_limits":
                if np.allclose(np.asarray(current, float), np.asarray(value, float)):
                    continue
                # Widen the range first: limits outside it are clipped otherwise.
                lo, hi = (float(v) for v in value)
                rng = getattr(plane, "contrast_limits_range", None)
                if rng is not None and (lo < float(rng[0]) or hi > float(rng[1])):
                    plane.contrast_limits_range = (min(lo, float(rng[0])), max(hi, float(rng[1])))
            elif attr == "colormap":
                if getattr(current, "name", current) == getattr(value, "name", value) and not labels:
                    continue
            elif current == value:
                continue
            setattr(plane, attr, value)
        except Exception:  # noqa: BLE001 (a style that will not copy is not worth a failed sync)
            continue


def sync_ortho_planes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
    frames: Sequence[PlaneFrame] | None = None,
    spacing: Sequence[float] | None = None,
    time_index: int = 0,
    thickness: float = 1.0,
    refresh_data: bool = False,
) -> list[Any]:
    """Show *layer* as three ``depiction="plane"`` slices at *position* on the canvas.

    Napari renders a plane-depicted Image as a single slab through the volume, so
    three of them at the crosshair give the 3D view the same three cuts the 2D
    views show. Existing plane layers are moved rather than recreated: rebuilding
    them on every scroll re-uploads the volume to the GPU each time.

    The planes carry *layer*'s window, colormap and gamma (labels: its colours),
    and a 3D+t layer is shown at *time_index*; its planes are re-fed only when
    the time point changes, or with *refresh_data* (the layer was edited).

    *thickness* is the slab each plane renders: layers drawn over another give
    theirs a little more, so their slice wraps the one below and is drawn in front
    of it instead of fighting it for the same depth.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    wanted = [_plane_layer_name(source_name, a) for a in axes]
    is_4d = layer_time_axis(layer) is not None
    t_key = int(time_index) if is_4d else 0
    # Moving existing planes needs only their shape, not the volume: re-deriving
    # the array on every scroll step is what made scrolling crawl.
    if all(name in by_name for name in wanted):
        planes = [by_name[name] for name in wanted]
        stale = refresh_data or any(
            (getattr(pl, "metadata", None) or {}).get("nvitk_ortho_t", 0) != t_key for pl in planes
        )
        data = _layer_volume(layer, t_key) if stale else None
        shape = tuple(int(v) for v in planes[0].data.shape[-3:])
        displayed = displayed_axes(viewer, len(shape))
        out = []
        for i, (axis, existing) in enumerate(zip(axes, planes)):
            point, normal = _plane_geometry(
                axis, position, shape, displayed,
                frames[i] if frames is not None and i < len(frames) else None,
                spacing,
            )
            if data is not None:
                labels = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
                existing.data = np.asarray(data).astype(np.int32, copy=False) if labels else data
                existing.metadata["nvitk_ortho_t"] = t_key
            copy_plane_style(layer, existing)
            existing.plane = {"position": point, "normal": normal, "thickness": float(thickness)}
            out.append(existing)
        return out

    data = _layer_volume(layer, t_key)
    if data is None or data.ndim != 3:
        raise ValueError("3D planes need a 3D (or 3D+t) layer.")

    spatial = layer_spatial_kwargs(layer, ndim=3)
    labels = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
    # Adding a layer makes it active, which would re-bind anything watching the
    # active layer, including the panel that asked for these planes.
    try:
        previously_active = viewer.layers.selection.active
    except Exception:
        previously_active = None

    displayed = displayed_axes(viewer, data.ndim)
    out: list[Any] = []
    for i, axis in enumerate(axes):
        name = _plane_layer_name(source_name, axis)
        existing = by_name.get(name)
        point, normal = _plane_geometry(
            axis, position, data.shape, displayed,
            frames[i] if frames is not None and i < len(frames) else None,
            spacing,
        )
        if existing is None:
            if labels:
                existing = viewer.add_labels(
                    np.asarray(data).astype(np.int32, copy=False),
                    name=name, depiction="plane", opacity=1.0, **spatial,
                )
                existing._nvitk_label_like = True
            else:
                # Translucent at full opacity draws the slice exactly as the 2D
                # view does; additive (the old setting) added the volume's own
                # brightness on top wherever the plane crossed it.
                existing = viewer.add_image(
                    data, name=name, depiction="plane", rendering="mip",
                    blending="translucent", opacity=1.0, **spatial,
                )
            existing.metadata["nvitk_ortho_t"] = t_key
        copy_plane_style(layer, existing)
        existing.plane = {"position": point, "normal": normal, "thickness": float(thickness)}
        out.append(existing)

    if previously_active is not None:
        try:
            viewer.layers.selection.active = previously_active
        except Exception:
            pass
    return out


def _box_layer_name(source_name: str, axis: int) -> str:
    """Name of the bounding-box outline layer for *axis*."""
    return f"{source_name}{_BOX_LAYER_SUFFIX}_{axis}"


def _cross_layer_name(source_name: str) -> str:
    """Name of the 3D crosshair layer (under the box suffix, so selection and filters skip it)."""
    return f"{source_name}{_BOX_LAYER_SUFFIX}_cross"


def sync_ortho_crosshair(viewer: Any, layer: Any, position: Sequence[int]) -> Any:
    """Three lines through the crosshair point across the volume, one per array axis, coloured
    like the slice outlines: where the 2D views' crosshair sits, shown in 3D."""
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    shape = _spatial_shape(layer)
    if shape is None or len(shape) != 3:
        raise ValueError("The 3D crosshair needs a 3D (or 3D+t) layer.")
    p = [float(v) for v in position[:3]]
    lines = []
    for axis in range(3):
        a, b = list(p), list(p)
        a[axis], b[axis] = -0.5, float(shape[axis]) - 0.5
        lines.append(np.array([a, b], dtype=float))
    name = _cross_layer_name(source_name)
    existing = next((l for l in viewer.layers if str(getattr(l, "name", "")) == name), None)
    colours = [BOX_AXIS_COLORS[a % len(BOX_AXIS_COLORS)] for a in range(3)]
    if existing is None:
        try:
            previously_active = viewer.layers.selection.active
        except Exception:
            previously_active = None
        existing = viewer.add_shapes(
            lines, shape_type="line", name=name, edge_color=colours,
            edge_width=_BOX_EDGE_WIDTH * 1.5, **layer_spatial_kwargs(layer, ndim=3),
        )
        existing.editable = False
        if previously_active is not None:
            try:
                viewer.layers.selection.active = previously_active
            except Exception:
                pass
    else:
        current = [np.asarray(d, dtype=float) for d in existing.data]
        if len(current) != 3 or any(c.shape != l.shape or not np.allclose(c, l) for c, l in zip(current, lines)):
            existing.data = lines
            existing.edge_color = colours
    return existing


def remove_ortho_crosshair(viewer: Any, source_name: str) -> None:
    """Drop the 3D crosshair layer of *source_name*, if there is one."""
    name = _cross_layer_name(source_name)
    for layer in list(getattr(viewer, "layers", []) or []):
        if str(getattr(layer, "name", "")) == name:
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass


def remove_ortho_boxes(viewer: Any, source_name: str) -> int:
    """Drop any slice-outline layers belonging to *source_name*; returns how many."""
    wanted = {_box_layer_name(source_name, a) for a in (0, 1, 2)}
    removed = 0
    for layer in list(getattr(viewer, "layers", []) or []):
        if str(getattr(layer, "name", "")) in wanted:
            try:
                viewer.layers.remove(layer)
                removed += 1
            except Exception:
                pass
    return removed


def slice_box_corners(axis: int, index: int, shape: tuple[int, ...]) -> np.ndarray:
    """The four corners of the slice at *index* along *axis*, in voxel coordinates."""
    others = [a for a in range(3) if a != int(axis)]
    lo_a, hi_a = -0.5, float(shape[others[0]]) - 0.5
    lo_b, hi_b = -0.5, float(shape[others[1]]) - 0.5
    corners = np.zeros((4, 3), dtype=float)
    corners[:, int(axis)] = float(index)
    for k, (a, b) in enumerate(((lo_a, lo_b), (lo_a, hi_b), (hi_a, hi_b), (hi_a, lo_b))):
        corners[k, others[0]] = a
        corners[k, others[1]] = b
    return corners


def oblique_box_corners(
    center_vox: Sequence[float],
    frame: PlaneFrame,
    half_extent_mm: tuple[float, float],
    spacing: Sequence[float],
) -> np.ndarray:
    """Corners of a turned slice outline, in voxel coordinates.

    The same recipe the CPR plane marker uses: step out along the frame's two
    in-plane directions, converted from millimetres to voxels by the spacing.
    """
    sp = np.asarray(spacing, dtype=float)[:3]
    centre = np.asarray(center_vox, dtype=float)[:3]
    row = np.asarray(frame.row, dtype=float) / sp
    col = np.asarray(frame.col, dtype=float) / sp
    r, c = float(half_extent_mm[0]), float(half_extent_mm[1])
    return np.stack([
        centre - r * row - c * col,
        centre - r * row + c * col,
        centre + r * row + c * col,
        centre + r * row - c * col,
    ]).astype(float)


def _same_corners(shapes_layer: Any, corners: Any) -> bool:
    """Whether *shapes_layer* already outlines exactly *corners*."""
    try:
        current = shapes_layer.data
        if len(current) != 1:
            return False
        old = np.asarray(current[0], dtype=float)
        new = np.asarray(corners, dtype=float)
        return old.shape == new.shape and bool(np.array_equal(old, new))
    except Exception:  # noqa: BLE001 (an unreadable layer is rewritten)
        return False


def sync_ortho_boxes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
    frames: Sequence[PlaneFrame] | None = None,
    spacing: Sequence[float] | None = None,
    views: Sequence[AxisView] | None = None,
) -> list[Any]:
    """Outline each slice on the canvas instead of drawing the slice itself.

    A plane-depicted Image re-uploads the whole volume to the GPU, and three of
    them hide most of what is behind them. An outline says where the cut is
    without obscuring the anatomy or costing a texture, which is what is wanted
    when the 3D view is there to show the vessels rather than the slices.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    # Read the shape off the array, never through ``asarray``: the layer's data may
    # live on the GPU, and materialising a whole volume on the host to learn how
    # big it is costs a full copy per crosshair move.
    shape = _spatial_shape(layer)
    if shape is None or len(shape) != 3:
        raise ValueError("Slice outlines need a 3D (or 3D+t) layer.")
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    # The outlines are 3D shapes: a 3D+t layer lends them its spatial placement.
    spatial = layer_spatial_kwargs(layer, ndim=3)

    try:
        previously_active = viewer.layers.selection.active
    except Exception:
        previously_active = None

    out: list[Any] = []
    sp = tuple(float(v) for v in (spacing or (1.0, 1.0, 1.0)))[:3]
    for i, axis in enumerate(axes):
        name = _box_layer_name(source_name, axis)
        frame = frames[i] if frames is not None and i < len(frames) else None
        if frame is None:
            corners = slice_box_corners(axis, int(position[axis]), shape)
        else:
            view = views[i] if views is not None and i < len(views) else None
            rows = int(shape[view.rows]) if view is not None else int(shape[axis])
            cols = int(shape[view.cols]) if view is not None else int(shape[axis])
            # Half the field of view the 2D panel shows, so the outline marks the
            # same extent the user is looking at.
            half = (
                rows * sp[view.rows if view is not None else axis] / 2.0,
                cols * sp[view.cols if view is not None else axis] / 2.0,
            )
            corners = oblique_box_corners(position, frame, half, sp)
        existing = by_name.get(name)
        colour = BOX_AXIS_COLORS[int(axis) % len(BOX_AXIS_COLORS)]
        if existing is None:
            existing = viewer.add_shapes(
                [corners],
                shape_type="polygon",
                name=name,
                edge_color=colour,
                face_color="transparent",
                edge_width=_BOX_EDGE_WIDTH,
                **spatial,
            )
            existing.editable = False
        else:
            # Only write when the outline moved. Assigning ``data`` on a
            # Shapes layer runs Napari's whole event cascade (re-validating the
            # model, recomputing the world extent) for ~11 ms, and dragging one
            # crosshair leaves the other two planes exactly where they were.
            if not _same_corners(existing, corners):
                existing.data = [corners]
        out.append(existing)

    if previously_active is not None:
        try:
            viewer.layers.selection.active = previously_active
        except Exception:
            pass
    return out


def _scene_point(layer: Any, displayed: list[int], data_point: list[float]) -> np.ndarray:
    """A data-space point in vispy *scene* coordinates, the way Napari computes them.

    Deliberately mirrors ``VispyBaseLayer._on_matrix_change``: the layer transform
    is sliced to the displayed dims, then the axes are reversed for vispy. Deriving
    it any other way means guessing at a convention, and the two differ exactly
    when ``dims.order`` is a permutation, which is nvitk's normal case.
    """
    # A 3D+t layer's spatial dims are its trailing three; *displayed* and
    # *data_point* are spatial (0-2), so shift them onto the layer's own dims.
    lead = max(int(getattr(layer, "ndim", 3)) - 3, 0)
    transform = layer._transforms.simplified.set_slice([lead + int(d) for d in displayed])
    ordered = [float(data_point[d]) for d in displayed]
    world = np.asarray(transform(ordered), dtype=float)
    # Napari reverses the axes when handing the transform to vispy.
    return world[::-1]


def clip_geometry(
    layer: Any,
    axis: int,
    index: int,
    side: str,
    displayed: list[int] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(position, normal)`` for a cut at *index* along *axis*, in Napari's own frame.

    Napari converts a clipping plane for vispy with a plain component reversal
    (``as_array()[..., ::-1]``), but the node's transform is built from the layer
    sliced to the *displayed* dims and then reversed, so vispy's scene axes are
    ``reversed(displayed)``, not ``reversed(range(3))``. A plane built in raw axis
    order therefore cuts the wrong axis whenever ``dims.order`` is a permutation.

    Both vectors are computed in scene coordinates through Napari's own transform
    and then pre-reversed, so Napari's reversal restores them exactly.
    """
    shape = _spatial_shape(layer) or tuple(int(v) for v in layer.data.shape[-3:])
    order = list(displayed) if displayed else list(range(3))

    cut = [float(v) / 2.0 for v in shape]
    cut[int(axis)] = float(index)
    # A step towards the half that "above" keeps, so the normal comes out of the
    # geometry rather than out of an assumed sign.
    kept = list(cut)
    kept[int(axis)] += 1.0 if str(side) == "above" else -1.0

    scene_cut = _scene_point(layer, order, cut)
    scene_kept = _scene_point(layer, order, kept)
    direction = scene_kept - scene_cut
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        # A degenerate axis: fall back to a unit normal on the matching scene axis.
        direction = np.zeros(3)
        direction[len(order) - 1 - order.index(int(axis))] = 1.0
        length = 1.0
    normal = direction / length

    # The shader keeps ``dot(loc - position, normal) >= 0``: the side the normal
    # points towards. Un-reverse both, so Napari's own reversal lands them right.
    return (
        tuple(float(v) for v in scene_cut[::-1]),
        tuple(float(v) for v in normal[::-1]),
    )


def apply_clip(
    layer: Any,
    axis: int,
    index: int,
    side: str,
    displayed: list[int] | None = None,
) -> None:
    """Cut *layer* at *index* along *axis* so one side of the slice is not drawn.

    ``side="below"`` keeps the half with lower indices, ``"above"`` the higher, and
    anything else clears the cut. This is the "see inside" control: the volume is
    opened at the plane the 2D view is showing, so the interior is visible instead
    of only the outer surface.
    """
    from napari.layers.utils.plane import ClippingPlane

    if str(side) not in ("below", "above"):
        layer.experimental_clipping_planes = []
        return

    position, normal = clip_geometry(layer, axis, index, side, displayed)
    layer.experimental_clipping_planes = [
        ClippingPlane(position=position, normal=normal, enabled=True)
    ]


def clear_clip(layer: Any) -> None:
    """Remove any clipping planes nvitk put on *layer*."""
    try:
        layer.experimental_clipping_planes = []
    except Exception:
        pass
