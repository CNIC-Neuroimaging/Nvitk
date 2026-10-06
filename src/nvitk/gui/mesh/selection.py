"""Selections on surfaces and point clouds, kept as a per-vertex ``selected`` field.

A selection is painted with the brush, dragged out as a box on the screen or
taken as the piece under a click (the Mesh panel's *Select* category). It is
drawn red over grey; the colouring the layer had comes back when the selection
is cleared. The selection actions (delete, keep, copy, grow, shrink, invert) and
MeshLab's selection-based filters work on it.
"""

from __future__ import annotations

import copy
import weakref
from typing import Any, Sequence

import numpy as np

from nvitk.gui.mesh.layers import (
    add_field,
    display_of,
    fields_of,
    layer_to_mesh,
    layer_to_point_cloud,
    remove_field,
    set_display,
)

#: Name of the per-vertex / per-point selection field (0 / 1).
SELECTION_FIELD = "selected"
#: Not selected / selected.
SELECTION_COLOURS = {0: (0.78, 0.78, 0.78, 1.0), 1: (0.92, 0.18, 0.15, 1.0)}
#: How a new pick combines with the current selection.
SELECTION_MODES = ("add", "remove", "replace")

_PREVIOUS_DISPLAY: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def layer_points(layer: Any) -> np.ndarray:
    """World coordinates of a Surface's vertices or a Points layer's points."""
    if type(layer).__name__ == "Surface":
        return layer_to_mesh(layer).vertices
    return layer_to_point_cloud(layer).points


def selection_of(layer: Any) -> np.ndarray:
    """The layer's selection (boolean per vertex / point); all False when none."""
    n = len(layer_points(layer))
    sel = fields_of(layer).get(SELECTION_FIELD)
    if sel is None or len(sel) != n:
        return np.zeros(n, dtype=bool)
    return np.asarray(sel) > 0.5


def combine(current: np.ndarray, picked: np.ndarray, mode: str) -> np.ndarray:
    """*picked* added to, removed from, or replacing *current*."""
    if mode == "remove":
        return current & ~picked
    if mode == "replace":
        return picked.copy()
    return current | picked


def set_selection(layer: Any, selected: np.ndarray, *, show: bool = True) -> int:
    """Store *selected* on *layer* and (with *show*) draw it; returns how many are selected."""
    sel = np.asarray(selected, dtype=bool)
    disp = display_of(layer)
    if show and not (disp.mode == "field" and disp.field == SELECTION_FIELD):
        try:
            _PREVIOUS_DISPLAY[layer] = copy.copy(disp)
        except TypeError:
            pass
    add_field(layer, SELECTION_FIELD, sel.astype(np.float32), categories=SELECTION_COLOURS, show=show)
    return int(sel.sum())


def clear_selection(layer: Any) -> None:
    """Drop the selection and give the layer back the colouring it had before."""
    remove_field(layer, SELECTION_FIELD)
    try:
        prev = _PREVIOUS_DISPLAY.pop(layer, None)
    except TypeError:
        prev = None
    if prev is not None and (prev.mode != "field" or prev.field in fields_of(layer)):
        set_display(layer, mode=prev.mode, field=prev.field, colormap=prev.colormap, color=prev.color,
                    auto_range=prev.auto_range, limits=prev.limits)


def near(points: np.ndarray, centre: Sequence[float], radius: float) -> np.ndarray:
    """Boolean per point: within *radius* (mm) of *centre*."""
    return np.linalg.norm(points - np.asarray(centre, dtype=float), axis=1) <= float(radius)


def world_to_canvas(viewer: Any, points: np.ndarray) -> np.ndarray | None:
    """Canvas pixel ``(x, y)`` of world *points* (last three dims), as Napari maps mouse events.

    ``None`` when the canvas is not reachable (no Qt viewer).
    """
    try:
        canvas = viewer.window._qt_viewer.canvas
        view = canvas.view
    except Exception:  # noqa: BLE001
        return None
    transform = view.transform * view.scene.transform
    displayed = list(viewer.dims.displayed)
    ndim = int(viewer.dims.ndim)
    pts = np.asarray(points, dtype=float)
    # World coordinates of the displayed dims, in Vispy's (reversed) order.
    full = np.tile(np.asarray(viewer.dims.point, dtype=float), (len(pts), 1))
    full[:, ndim - 3:] = pts[:, -3:] if ndim >= 3 else pts[:, -ndim:]
    shown = full[:, displayed][:, ::-1]
    homog = np.c_[shown, np.ones(len(shown))] if shown.shape[1] == 3 else np.c_[shown, np.zeros(len(shown)), np.ones(len(shown))]
    mapped = np.asarray(transform.map(homog), dtype=float)
    return mapped[:, :2] / np.where(np.abs(mapped[:, 3:4]) > 1e-12, mapped[:, 3:4], 1.0)


def in_box(viewer: Any, layer: Any, corner0: Sequence[float], corner1: Sequence[float], *,
           facing_only: bool = True, view_direction: Sequence[float] | None = None) -> np.ndarray:
    """Boolean per vertex / point: inside the screen rectangle *corner0*–*corner1*
    (canvas pixels); with *facing_only*, a surface's vertices whose normal faces away
    from the camera are left out (the side you see)."""
    pts = layer_points(layer)
    xy = world_to_canvas(viewer, pts)
    if xy is None:
        return np.zeros(len(pts), dtype=bool)
    (x0, y0), (x1, y1) = np.asarray(corner0, float)[:2], np.asarray(corner1, float)[:2]
    inside = ((xy[:, 0] >= min(x0, x1)) & (xy[:, 0] <= max(x0, x1))
              & (xy[:, 1] >= min(y0, y1)) & (xy[:, 1] <= max(y0, y1)))
    if facing_only and type(layer).__name__ == "Surface" and view_direction is not None \
            and int(viewer.dims.ndisplay) == 3:
        normals = layer_to_mesh(layer).vertex_normals
        d = np.asarray(view_direction, dtype=float)[-3:]
        inside &= (normals @ d) < 0
    return inside


__all__ = [
    "SELECTION_COLOURS",
    "SELECTION_FIELD",
    "SELECTION_MODES",
    "clear_selection",
    "combine",
    "in_box",
    "layer_points",
    "near",
    "selection_of",
    "set_selection",
    "world_to_canvas",
]
