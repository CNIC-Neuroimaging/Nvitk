"""Move and turn layers by hand: Napari's Transform mode, in the slice views and in 3D.

Napari's own Transform mode drags a box handle on a 2D canvas only; nvitk's canvas is 3D and its
slice views are not Napari canvases, so with a layer in Transform mode (its controls' transform
button) nothing happened. Here the same mode works everywhere:

* **slice views**: drag to move the layer in the view's plane, Ctrl+drag to turn it about the
  view's axis (:class:`nvitk.gui.viz.ortho.transform_drag.OrthoTransformDrag`);
* **3D view**: drag to move it in the screen plane, Ctrl+drag to turn it like a trackball,
  about its centre (:func:`install_transform_3d`).

Every selected layer in Transform mode moves together (a scan and its labels). A move is a world
transform put in front of the layer's own (its ``affine`` component), so the slice views, the
tools and saving all see it; with the canvas drawing layers on their own grids (resampling off)
the real transform is the one changed. The placement before the first move is kept, and *Reset
transform* in the layer menu puts it back.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: The layer's ``affine`` component before its first hand move (see :func:`reset_transform`).
ORIGIN_ATTR = "_nvitk_transform_origin"


def layer_mode(layer: Any) -> str:
    """The layer's interaction mode as a plain string (``"transform"``, ``"pan_zoom"``...)."""
    mode = getattr(layer, "mode", "")
    return str(getattr(mode, "value", mode))


def transform_targets(viewer: Any) -> list[Any]:
    """The layers a transform drag moves: the active one when it is in Transform mode, with
    every other selected layer in Transform mode; nothing when the active one is not."""
    layers = getattr(viewer, "layers", None)
    if layers is None:
        return []
    active = layers.selection.active
    if active is None or layer_mode(active) != "transform":
        return []
    out = [active]
    for layer in layers.selection:
        if layer is not active and layer_mode(layer) == "transform":
            out.append(layer)
    return out


def affine_matrix(layer: Any) -> np.ndarray:
    """The layer's ``affine`` component (world placement in front of scale / rotate...): the
    real one when the canvas draws the layer on its own grid."""
    from nvitk.gui.viz.main_canvas import true_transform

    parts = true_transform(layer)
    if parts is not None:
        return np.array(parts["affine"], dtype=float)
    return np.array(layer.affine.affine_matrix, dtype=float)


def set_affine_matrix(layer: Any, matrix: np.ndarray) -> None:
    """Set the layer's (real) ``affine`` component to *matrix*."""
    from nvitk.gui.core.orientation import suppress_nonorthogonal_slice_warning
    from nvitk.gui.viz.main_canvas import true_transform

    matrix = np.asarray(matrix, dtype=float)
    parts = true_transform(layer)
    if parts is not None:
        # Drawn on its own grid: the real placement changes, the canvas keeps its stand-in; the
        # event tells the slice views (and anything else placing it) to look again.
        parts["affine"] = matrix
        try:
            layer.events.affine(value=layer.affine)
        except Exception:  # noqa: BLE001
            pass
        return
    with suppress_nonorthogonal_slice_warning():
        layer.affine = matrix


def embed_spatial(world: np.ndarray, ndim: int) -> np.ndarray:
    """A spatial 4x4 world transform as a ``(ndim + 1)`` square one acting on a layer's last
    three axes (a 3D+t layer's time axis comes first and is left alone)."""
    out = np.eye(int(ndim) + 1)
    if int(ndim) < 3:
        return out
    k = int(ndim) - 3
    out[k:k + 3, k:k + 3] = world[:3, :3]
    out[k:k + 3, -1] = world[:3, 3]
    return out


def translation(delta: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = np.asarray(delta, dtype=float)[:3]
    return out


def rotation_about_point(axis: Sequence[float], angle: float, centre: Sequence[float]) -> np.ndarray:
    """4x4 rotation by *angle* (radians) about the line through *centre* along *axis*."""
    from nvitk.gui.viz.ortho.geometry import rotation_about

    rot = rotation_about(np.asarray(axis, dtype=float), float(angle))
    c = np.asarray(centre, dtype=float)[:3]
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = c - rot @ c
    return out


def _real_data_to_world(layer: Any) -> np.ndarray:
    from nvitk.gui.viz.main_canvas import real_data_to_world

    return real_data_to_world(layer)


def world_bounds(layers: Sequence[Any]) -> np.ndarray | None:
    """``(2, 3)`` world bounding box of *layers* in their real placement (spatial axes)."""
    boxes = []
    for layer in layers:
        try:
            ext = np.asarray(layer.extent.data, dtype=float)
            matrix = _real_data_to_world(layer)
        except Exception:  # noqa: BLE001
            continue
        nd = ext.shape[1]
        if nd < 3 or matrix.shape != (nd + 1, nd + 1) or not np.all(np.isfinite(ext)):
            continue
        corners = np.array(np.meshgrid(*[ext[:, d] for d in range(nd)], indexing="ij")).reshape(nd, -1).T
        world = corners @ matrix[:nd, :nd].T + matrix[:nd, -1]
        boxes.append(np.vstack([world[:, -3:].min(0), world[:, -3:].max(0)]))
    if not boxes:
        return None
    stack = np.stack(boxes)
    return np.vstack([stack[:, 0].min(0), stack[:, 1].max(0)])


def world_centre(layers: Sequence[Any]) -> np.ndarray | None:
    """Centre of *layers*' world bounding box (spatial axes)."""
    box = world_bounds(layers)
    return None if box is None else box.mean(axis=0)


class TransformSession:
    """One drag: the layers it moves and where they started; :meth:`apply` sets every layer to
    a world transform of its starting placement (absolute, so a long drag does not drift)."""

    def __init__(self, viewer: Any, layers: Sequence[Any]) -> None:
        self.viewer = viewer
        self.layers = list(layers)
        self.start = [affine_matrix(layer) for layer in self.layers]
        box = world_bounds(self.layers)
        self.centre = None if box is None else box.mean(axis=0)
        self.radius = 1.0 if box is None else max(float(np.linalg.norm(box[1] - box[0])) / 2.0, 1.0)
        self.moved = False

    def apply(self, world: np.ndarray) -> None:
        """Put the spatial world transform *world* (4x4) in front of each starting placement."""
        for layer, start in zip(self.layers, self.start):
            if not hasattr(layer, "affine"):
                continue
            nd = start.shape[0] - 1
            try:
                set_affine_matrix(layer, embed_spatial(world, nd) @ start)
            except Exception:  # noqa: BLE001 (a layer removed mid-drag)
                continue
        self.moved = True

    def cancel(self) -> None:
        for layer, start in zip(self.layers, self.start):
            try:
                set_affine_matrix(layer, start)
            except Exception:  # noqa: BLE001
                pass

    def finish(self) -> None:
        """Remember each layer's placement from before its first hand move."""
        if not self.moved:
            return
        for layer, start in zip(self.layers, self.start):
            if getattr(layer, ORIGIN_ATTR, None) is None:
                try:
                    setattr(layer, ORIGIN_ATTR, np.array(start, dtype=float))
                except Exception:  # noqa: BLE001
                    pass
        names = ", ".join(f"“{getattr(layer, 'name', '?')}”" for layer in self.layers)
        try:
            from nvitk.gui.core.log_panel import gui_log

            gui_log(f"Moved {names} by hand (layer menu › Reset transform puts it back).")
        except Exception:  # noqa: BLE001
            pass


def has_origin(layer: Any) -> bool:
    """Whether *layer* was moved by hand and can be reset."""
    return getattr(layer, ORIGIN_ATTR, None) is not None


def reset_transform(layer: Any) -> bool:
    """Put *layer* back where it was before it was moved by hand; True when it was."""
    origin = getattr(layer, ORIGIN_ATTR, None)
    if origin is None:
        return False
    set_affine_matrix(layer, origin)
    try:
        delattr(layer, ORIGIN_ATTR)
    except AttributeError:
        pass
    return True


def pin_ortho_reference(viewer: Any, layers: Sequence[Any]) -> None:
    """Keep the slice views on a grid that is not about to move: when the views are laid out
    on one of *layers*, rebind them to another visible volume (one on the same grid first).
    Otherwise dragging the bound layer would move everything else on screen instead."""
    panel = getattr(viewer, "_nvitk_ortho_panel", None)
    if panel is None:
        return
    try:
        panel.pin_reference(layers)
    except Exception:  # noqa: BLE001 (a panel being torn down)
        pass


def _transform_drag_3d(viewer: Any, event: Any) -> Any:
    """3D canvas drag in Transform mode: move in the screen plane; Ctrl: trackball turn."""
    if viewer.dims.ndisplay != 3 or int(getattr(event, "button", 1) or 1) != 1:
        return
    if getattr(viewer, "_nvitk_drag_owner", None):
        return
    targets = transform_targets(viewer)
    if not targets:
        return
    pin_ortho_reference(viewer, targets)
    session = TransformSession(viewer, targets)
    if session.centre is None:
        return
    mods = set(getattr(event, "modifiers", ()) or ())
    rotate = "Control" in mods
    start = np.asarray(event.position, dtype=float)[-3:]
    view_dir = getattr(event, "view_direction", None)
    if view_dir is None:
        view_dir = viewer.camera.view_direction
    view_dir = np.asarray(view_dir, dtype=float)[-3:]
    held = bool(viewer.camera.mouse_pan)
    viewer.camera.mouse_pan = False
    viewer._nvitk_drag_owner = "transform"
    try:
        yield
        while event.type == "mouse_move":
            # In 3D Napari puts the cursor on the plane through the camera centre, square to the
            # view: a change of position there is the screen-plane move, at the right scale.
            delta = np.asarray(event.position, dtype=float)[-3:] - start
            if rotate:
                axis = np.cross(view_dir, delta)
                length = float(np.linalg.norm(delta))
                if length > 1e-9 and float(np.linalg.norm(axis)) > 1e-12:
                    session.apply(rotation_about_point(axis, length / session.radius, session.centre))
            else:
                session.apply(translation(delta))
            yield
    finally:
        viewer.camera.mouse_pan = held
        viewer._nvitk_drag_owner = None
        session.finish()


def install_transform_3d(viewer: Any) -> None:
    """Make Transform mode drag layers on the 3D canvas (once per viewer)."""
    if getattr(viewer, "_nvitk_transform_3d", None) is not None:
        return
    viewer.mouse_drag_callbacks.insert(0, _transform_drag_3d)
    viewer._nvitk_transform_3d = _transform_drag_3d


__all__ = [
    "ORIGIN_ATTR",
    "TransformSession",
    "affine_matrix",
    "embed_spatial",
    "has_origin",
    "install_transform_3d",
    "layer_mode",
    "pin_ortho_reference",
    "reset_transform",
    "rotation_about_point",
    "set_affine_matrix",
    "transform_targets",
    "translation",
    "world_bounds",
    "world_centre",
]
