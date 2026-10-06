"""Anatomical orientation marker on the 3D canvas: a person and R/L, A/P, H/F arrows.

The same marker the PESA-FAT QC reports draw in their 3D renders
(:mod:`nvitk.viz.orientation_marker`): three double-headed arrows lettered with
the anatomical direction of each end, and a small standing figure posed in the
same frame, so the subject's pose can be read off at a glance.

Here it lives in its own little viewport in a corner of Napari's canvas, with a
camera that copies the 3D camera's rotation (and axis flips) every frame — so it
turns with the scene but never zooms, pans or gets hidden behind a volume.

Frames
------
Napari hands vispy the displayed world dims reversed (vispy ``x`` is the last
displayed dim). nvitk layers place voxels in NIfTI world space, RAS+, so each
world dim has an anatomical direction; :func:`world_axis_codes` reads it off the
reference layer — correcting for a display-only mirror made with the
orientation tool, which moves voxels in world space without changing anatomy.
The marker is then built for the vispy frame directly
(:func:`~nvitk.viz.orientation_marker.build_orientation_marker_mesh` with the
codes of vispy x, y, z), and the camera does the rest.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: Corner viewport size and margin, in logical canvas pixels.
_MARKER_SIZE = 150
_MARKER_MARGIN = 10
#: Half-extent of the marker scene the corner camera frames (arrows + letters).
_MARKER_RADIUS = 1.45

_RAS = "RAS"
_NEG = {"R": "L", "A": "P", "S": "I"}


def _dominant_code(vector: np.ndarray) -> str:
    """Anatomical code of the RAS axis *vector* mostly points along."""
    k = int(np.argmax(np.abs(vector)))
    return _RAS[k] if vector[k] >= 0 else _NEG[_RAS[k]]


def _display_to_true_rotation(layer: Any) -> np.ndarray:
    """3x3 map from the layer's *display* world axes to true RAS world axes.

    The identity, except for a layer mirrored for viewing by negating affine
    columns (the orientation tool's "view" action): there the display world is a
    reflection of the anatomy. Compares the layer's affine with the file's own,
    taken into the layer's axis order.
    """
    from nvitk.gui.core.spatial import layer_affine, nvitk_metadata_from_layer

    eye = np.eye(3)
    try:
        disp = layer_affine(layer)
        meta = nvitk_metadata_from_layer(layer)
        src = meta.get("affine_source")
        if disp is None or src is None:
            return eye
        src = np.asarray(src, dtype=float)
        if src.shape != (4, 4):
            return eye
        from nvitk.gui.core.orientation import _axes_string_from_layer, layer_is_reordered
        from nvitk.gui.core.spatial import layer_source_axes

        if layer_is_reordered(layer):
            display = "".join(ch for ch in (_axes_string_from_layer(layer) or "").upper() if ch in "XYZ")
            source = "".join(ch for ch in (layer_source_axes(layer) or "").upper() if ch in "XYZ")
            if sorted(display) != sorted(source) or len(display) != 3:
                return eye
            src = src[:, [source.index(ch) for ch in display] + [3]]
        rot = src[:3, :3] @ np.linalg.inv(disp[:3, :3])
        # Only a signed permutation is a view mirror; anything else is a real
        # change of placement (a registration, a rotation) and keeps world = RAS.
        if not np.allclose(np.abs(rot).sum(axis=0), 1.0, atol=1e-3):
            return eye
        return np.round(rot)
    except Exception:  # noqa: BLE001 — a marker is decoration
        return eye


def world_axis_codes(viewer: Any, layer: Any | None = None) -> str | None:
    """Anatomical code of vispy ``x``, ``y``, ``z`` in the 3D view, e.g. ``"SAR"``.

    ``None`` when the view is not showing three spatial dims.
    """
    dims = getattr(viewer, "dims", None)
    if dims is None or int(getattr(dims, "ndisplay", 2)) != 3:
        return None
    displayed = [int(d) for d in dims.displayed]
    ndim = int(dims.ndim)
    if len(displayed) != 3 or ndim < 3:
        return None
    # World dims are (..., x, y, z) in RAS: the spatial three are the trailing ones.
    offset = ndim - 3
    if any(d < offset for d in displayed):
        return None
    rot = _display_to_true_rotation(layer) if layer is not None else np.eye(3)
    codes = []
    for d in reversed(displayed):  # vispy x, y, z
        codes.append(_dominant_code(rot[:, d - offset]))
    return "".join(codes)


def _marker_arrays(codes: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices, faces, face RGBA)`` of the marker built for *codes*."""
    from nvitk.viz.orientation_marker import build_orientation_marker_mesh

    mesh = build_orientation_marker_mesh(codes).triangulate()
    verts = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.uint32)
    rgb = np.asarray(mesh.cell_data["marker_rgb"], dtype=np.float32) / 255.0
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
    return verts, faces, rgba


def _clear_depth_node(parent: Any) -> Any:
    """A scene node that clears the depth buffer when drawn.

    Drawn first in the corner viewport, so the marker is depth-tested against
    itself only — never against a volume or surface the main scene left in the
    depth buffer at that corner of the screen.
    """
    from vispy import gloo
    from vispy.scene.visuals import create_visual_node
    from vispy.visuals import Visual

    class _ClearDepthVisual(Visual):
        def __init__(self) -> None:
            super().__init__(vcode="void main(){gl_Position=vec4(0.0);}", fcode="void main(){}")

        def draw(self) -> None:  # noqa: D401 — vispy hook
            gloo.clear(color=False, depth=True)

        def _prepare_transforms(self, view: Any) -> None:
            return None

    node_class = create_visual_node(_ClearDepthVisual)
    node = node_class(parent=parent)
    node.interactive = False
    node.order = -1000
    return node


class OrientationMarker:
    """The marker viewport on a vispy :class:`~vispy.scene.SceneCanvas`.

    Parameters
    ----------
    canvas
        The scene canvas to draw on (Napari's ``_scene_canvas``).
    source_camera
        Callable returning the vispy 3D camera to follow (rotation and flips),
        or ``None`` while there is none.
    light_direction
        Callable returning the light direction in the marker's (vispy) frame, or
        ``None`` for a fixed light.
    """

    def __init__(
        self,
        canvas: Any,
        source_camera: Callable[[], Any],
        *,
        light_direction: Callable[[], Any] | None = None,
        size: int = _MARKER_SIZE,
        margin: int = _MARKER_MARGIN,
    ) -> None:
        from vispy import scene

        self._canvas = canvas
        self._source_camera = source_camera
        self._light_direction = light_direction
        self._size = int(size)
        self._margin = int(margin)
        self._codes: str | None = None
        self._synced: tuple | None = None

        self._view = scene.widgets.ViewBox(parent=canvas.scene, border_width=0)
        self._view.interactive = False
        # Drawn after Napari's own view, which is the canvas' first widget.
        self._view.order = 10_000
        camera = scene.cameras.ArcballCamera(fov=0, interactive=False)
        self._view.camera = camera
        r = _MARKER_RADIUS
        camera.set_range(x=(-r, r), y=(-r, r), z=(-r, r), margin=0.0)
        # The visible extent, in scene units: the marker's full width, whatever
        # set_range derived from the bounds.
        camera.scale_factor = 2.0 * r
        camera.center = (0.0, 0.0, 0.0)
        # An orthographic camera spreads its depth buffer over ±depth_value
        # (default 1e6): the letters, a few hundredths of a unit in front of their
        # plaques, would z-fight and vanish. The marker is ~3 units deep.
        camera.depth_value = 4.0 * r
        self._camera = camera
        self._clear = _clear_depth_node(self._view.scene)
        self._mesh = scene.visuals.Mesh(parent=self._view.scene, shading="smooth")
        self._mesh.interactive = False
        self._mesh.order = 0
        self._mesh.set_gl_state("opaque", depth_test=True, cull_face=False)
        self._view.visible = False

        canvas.events.resize.connect(self._place)
        canvas.events.draw.connect(self._sync, position="first")
        self._place()

    # -- state ----------------------------------------------------------------

    @property
    def visible(self) -> bool:
        return bool(self._view.visible)

    @property
    def codes(self) -> str | None:
        """Anatomical codes of vispy x, y, z the marker is built for."""
        return self._codes

    def set_codes(self, codes: str | None) -> None:
        """Rebuild the marker for *codes* (``None`` hides it until codes are known)."""
        if codes == self._codes:
            return
        self._codes = codes
        if codes is None:
            self._view.visible = False
            return
        verts, faces, colors = _marker_arrays(codes)
        self._mesh.set_data(vertices=verts, faces=faces, face_colors=colors)
        self._synced = None
        self._canvas.update()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the marker (hidden regardless while it has no codes)."""
        self._view.visible = bool(visible) and self._codes is not None
        self._synced = None
        self._canvas.update()

    def remove(self) -> None:
        """Take the viewport off the canvas for good."""
        try:
            self._canvas.events.resize.disconnect(self._place)
            self._canvas.events.draw.disconnect(self._sync)
        except Exception:  # noqa: BLE001
            pass
        self._view.parent = None

    # -- geometry -------------------------------------------------------------

    def _place(self, _event: Any = None) -> None:
        """Keep the viewport square in the bottom-left corner."""
        width, height = (int(v) for v in self._canvas.size)
        side = max(60, min(self._size, int(min(width, height) * 0.35)))
        self._view.pos = (self._margin, max(0, height - side - self._margin))
        self._view.size = (side, side)

    def sync(self) -> None:
        """Bring the marker camera in line with the source camera now."""
        self._sync()

    def _sync(self, _event: Any = None) -> None:
        """Copy the 3D camera's rotation (and flips) before each frame is drawn.

        Only when they changed: updating the camera schedules a redraw, so an
        unconditional copy would redraw forever.
        """
        if not self._view.visible:
            return
        src = self._source_camera()
        if src is None:
            return
        quat = getattr(src, "_quaternion", None)
        flip = tuple(bool(f) for f in getattr(src, "flip", (False, False, False)))
        key = None if quat is None else (quat.w, quat.x, quat.y, quat.z, flip)
        if key == self._synced:
            return
        self._synced = key
        if quat is not None:
            self._camera._quaternion = quat.copy()
        self._camera.flip = flip
        self._camera.view_changed()
        if self._light_direction is not None and self._mesh.shading_filter is not None:
            light = self._light_direction()
            if light is not None:
                self._mesh.shading_filter.light_dir = tuple(float(v) for v in light)


# ──────────────────────────────────────────────────────────────────────────────
# Napari wiring
# ──────────────────────────────────────────────────────────────────────────────


def _napari_parts(viewer: Any) -> tuple[Any, Any] | None:
    """``(scene canvas, VispyCamera)`` of *viewer*, or ``None`` without a canvas."""
    try:
        vispy_canvas = viewer.window._qt_viewer.canvas
        return vispy_canvas._scene_canvas, vispy_canvas.camera
    except Exception:  # noqa: BLE001 — headless or internals moved
        return None


class NapariOrientationMarker:
    """:class:`OrientationMarker` kept in step with a Napari viewer.

    Follows the 3D camera, rebuilds when the displayed dims or the reference
    layer change, and hides itself in 2D.
    """

    def __init__(self, viewer: Any, reference_layer: Callable[[], Any] | None = None) -> None:
        parts = _napari_parts(viewer)
        if parts is None:
            raise RuntimeError("The Napari canvas is not available.")
        scene_canvas, vispy_camera = parts
        self._viewer = viewer
        self._reference_layer = reference_layer or (lambda: viewer.layers.selection.active)
        self._wanted = False
        self._marker = OrientationMarker(
            scene_canvas,
            lambda: getattr(vispy_camera, "_3D_camera", None),
            light_direction=self._light,
        )
        # Draw-time sync covers interactive rotation; this covers a camera set from
        # code (and screenshots, which render without a draw event).
        viewer.camera.events.angles.connect(lambda _e=None: self._marker.sync())
        viewer.dims.events.ndisplay.connect(self.refresh)
        viewer.dims.events.order.connect(self.refresh)
        viewer.layers.selection.events.active.connect(self.refresh)
        viewer.layers.events.inserted.connect(self.refresh)
        viewer.layers.events.removed.connect(self.refresh)

    def _light(self) -> np.ndarray | None:
        """Napari's own surface lighting: behind the camera, top right."""
        cam = getattr(self._viewer, "camera", None)
        try:
            up = np.asarray(cam.up_direction, dtype=float)[::-1]
            view = np.asarray(cam.view_direction, dtype=float)[::-1]
        except Exception:  # noqa: BLE001
            return None
        return up - view - np.cross(up, view)

    @property
    def marker(self) -> OrientationMarker:
        return self._marker

    def set_enabled(self, enabled: bool) -> None:
        """Show the marker whenever the canvas is in 3D (or stop showing it)."""
        self._wanted = bool(enabled)
        self.refresh()

    @property
    def enabled(self) -> bool:
        return self._wanted

    def refresh(self, _event: Any = None) -> None:
        """Rebuild for the current view and reference layer, and show or hide."""
        if not self._wanted:
            self._marker.set_visible(False)
            return
        try:
            layer = self._reference_layer()
        except Exception:  # noqa: BLE001
            layer = None
        codes = world_axis_codes(self._viewer, layer)
        self._marker.set_codes(codes)
        self._marker.set_visible(codes is not None)


def orientation_marker_for(viewer: Any, reference_layer: Callable[[], Any] | None = None) -> NapariOrientationMarker | None:
    """The viewer's marker, created on first use (``None`` without a canvas)."""
    existing = getattr(viewer, "_nvitk_orientation_marker", None)
    if existing is not None:
        return existing
    try:
        marker = NapariOrientationMarker(viewer, reference_layer)
    except Exception:  # noqa: BLE001 — headless, or vispy/pyvista missing
        return None
    viewer._nvitk_orientation_marker = marker
    return marker


__all__ = [
    "NapariOrientationMarker",
    "OrientationMarker",
    "orientation_marker_for",
    "world_axis_codes",
]
