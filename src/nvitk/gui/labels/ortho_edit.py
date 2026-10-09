"""Drawing labels in the orthogonal views (:mod:`nvitk.gui.viz.ortho_panel`).

The axial / coronal / sagittal views take the same tools as the main canvas,
read off the label layer being edited — the Labeling tab's, else the active
Labels layer:

* **Paint** / **Erase** — Napari's brush (its size, its round-in-scale shape,
  *preserve labels*, a disc in the plane or a ball through slices with the 3D
  brush); a drag is one stroke, one Ctrl+Z;
* **Fill** — Napari's bucket, in the plane clicked (or the volume with 3D fill);
* **Pick** — the label clicked becomes the active one;
* **Wand**, **Flood**, **Vessel** — the Labeling tab's region tools: a click
  grows a region from the voxel clicked, a drag adds one at every voxel reached
  (one undo step), Alt+drag tunes the tool's setting live, Shift+click takes a
  region out;
* **Tracer** — click points along a vessel, in any view and slice; double-click
  fills the tube;
* **Smart brush** — a brush painting only the voxels like its centre's;
* **Polygon** — click the corners in a view, then double-click (or click the
  first corner) to fill it with the active label in that plane; Esc cancels;
* **Right-drag** — erases with the brush whatever the tool (one stroke, one
  Ctrl+Z); the tool is back as soon as the button is released;
* **Box** (the Labeling tab's Box card, Draw box) — a drag sets the box on the
  view's two axes. The box and a traced path are drawn in every view.

With any other tool (Pan, Polygon) a click moves the crosshair, as it always
did; with a drawing tool, Ctrl+click does. Shift+drag pans and the wheel scrolls
slices either way (a Shift+click that does not move is the wand's "remove").
Every edit goes through Napari's history and the Labeling tab's editable area,
so the canvas, the views and Undo agree. The views must show the layer on its
own grid (no resampling, planes not turned): the clicked pixel is then exactly
one voxel.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

from qtpy.QtCore import Qt, QTimer

from nvitk.gui.labels import editing as E
from nvitk.gui.labels.region_tools import EXTRA_TOOLS, REGION_TOOLS, TUNED

#: Layer modes the views draw with (the wand is the Labeling tab's).
DRAW_TOOLS = ("paint", "erase", "fill", "pick")


class OrthoLabelEditor:
    """Turns clicks and drags in an :class:`~nvitk.gui.viz.ortho_panel.OrthoViewerPanel`'s
    views into edits of a label layer (see the module docstring)."""

    def __init__(self, panel: Any) -> None:
        self._panel = panel
        self._viewer = panel._viewer
        self._stroke: dict[str, Any] | None = None
        self._watched: Any = lambda: None
        self._subs: list[tuple[Any, Any]] = []
        self._labeling_hooked = False
        #: The polygon being drawn: its view, layer, corners (row-axis, col-axis voxels).
        self._poly: dict[str, Any] | None = None
        #: A region tool's stroke in progress, and — for Alt+drag — where the tuning began.
        self._region: Any | None = None
        self._tune: dict[str, float] | None = None
        #: A smart brush stroke in progress.
        self._smart: Any | None = None
        #: The box being drawn: its view, layer and first pixel.
        self._boxdrag: dict[str, Any] | None = None
        #: A traced vessel to draw: (layer, clicked voxels, path voxels).
        self._trace: tuple[Any, list[Any], list[Any]] | None = None
        # A drag paints at mouse-move rate: redraw the views once per event-loop turn.
        self._live_timer = QTimer(panel)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(0)
        self._live_timer.timeout.connect(self._live_refresh)
        try:
            self._viewer.layers.selection.events.active.connect(self.refresh)
        except Exception:  # noqa: BLE001
            pass

    # ── what is drawn on, with what ───────────────────────────────────────────

    def _labeling(self) -> Any | None:
        panel = getattr(self._viewer, "_nvitk_labeling_panel", None)
        if panel is not None and not self._labeling_hooked:
            try:
                panel.tool_changed.connect(self.refresh)
                self._labeling_hooked = True
            except Exception:  # noqa: BLE001
                pass
        return panel

    def target(self) -> Any | None:
        """The label layer the views draw on: the Labeling tab's, else the active Labels layer."""
        labeling = self._labeling()
        layer = labeling.target() if labeling is not None else None
        if layer is None or not E.is_labels(layer):
            active = self._viewer.layers.selection.active
            layer = active if E.is_labels(active) else None
        return layer

    def tool(self, layer: Any | None = None) -> str | None:
        """The drawing tool in use on *layer*, or ``None`` (the views move the crosshair)."""
        labeling = self._labeling()
        if labeling is not None and labeling.box_armed():
            return "box"  # whatever the layer: the box is drawn on the views' grid
        layer = self.target() if layer is None else layer
        if layer is None:
            return None
        if labeling is not None:
            extra = labeling.extra_tool() if labeling.target() is layer else None
            if extra in EXTRA_TOOLS:
                return extra
        mode = str(getattr(layer.mode, "value", layer.mode))
        if mode == "polygon":
            return "polygon"
        return mode if mode in DRAW_TOOLS else None

    def blocked(self, layer: Any) -> str | None:
        """Why the views cannot draw on *layer* right now (``None``: they can)."""
        from nvitk.gui.viz.ortho_panel import _same_grid, _spatial_shape

        panel = self._panel
        anchor = panel.bound_layer()
        if anchor is None or panel._data is None:
            return "Show a volume in the views first."
        if layer not in self._viewer.layers:
            return None
        if _spatial_shape(layer) is None:
            return f"“{layer.name}” is not a volume."
        if not _same_grid(layer, anchor):
            return (f"“{layer.name}” is on another grid than the views (it is drawn resampled): "
                    "draw on it in the main canvas, or show it in the views by selecting it.")
        if panel.is_oblique():
            return "The views are turned: reset the orientation to draw in them."
        return None

    # ── voxels ───────────────────────────────────────────────────────────────

    def _layer_dims(self, layer: Any) -> tuple[list[int], int | None]:
        """*layer*'s spatial axes, in the views' order, and its time axis (or ``None``)."""
        from nvitk.gui.viz.ortho_panel import layer_time_axis

        t_ax = layer_time_axis(layer)
        spatial = [d for d in range(layer.data.ndim) if d != t_ax]
        return spatial[-3:], t_ax

    def _voxel(self, layer: Any, view_index: int, row: int, col: int) -> tuple[int, ...]:
        """The voxel of *layer* under pixel (*row*, *col*) of view *view_index*."""
        panel = self._panel
        view = panel._views[view_index]
        position = list(panel._position)
        row_axis, r, col_axis, c = view.to_voxel(row, col, panel._data.shape)
        position[row_axis], position[col_axis] = r, c
        spatial, t_ax = self._layer_dims(layer)
        voxel = [0] * layer.data.ndim
        for k, d in enumerate(spatial):
            voxel[d] = int(position[k])
        if t_ax is not None:
            voxel[t_ax] = int(min(panel._time, layer.data.shape[t_ax] - 1))
        return tuple(voxel)

    def _plane_dims(self, layer: Any, view_index: int) -> list[int]:
        """*layer*'s axes drawn in view *view_index* (row axis, column axis)."""
        view = self._panel._views[view_index]
        spatial, _t = self._layer_dims(layer)
        return [spatial[view.rows], spatial[view.cols]]

    def _edit_dims(self, layer: Any, view_index: int) -> list[int]:
        """The axes a brush or bucket covers: the plane, or the volume with a 3D brush."""
        if int(getattr(layer, "n_edit_dimensions", 2)) >= 3:
            return self._layer_dims(layer)[0]
        return self._plane_dims(layer, view_index)

    # ── events from the views ───────────────────────────────────────────────

    def handle(self, view_index: int, kind: str, pixel: tuple[int, int] | None, event: Any) -> bool:
        """A mouse *event* (``press`` / ``move`` / ``release``) on view *view_index*;
        True when it was an edit (the view does nothing else with it)."""
        if kind == "move":
            return self._move(view_index, pixel, event)
        if kind == "release":
            return self._release()
        if kind == "double":
            if self._poly is not None and self._poly["view"] == int(view_index):
                self.finish_polygon()
                return True
            labeling = self._labeling()
            if labeling is not None and labeling.extra_tool() == "tracer" and labeling._trace is not None:
                labeling.tracer_fill()
                return True
            return False
        if kind == "shift-click":
            # Shift+drag pans the views; a Shift+click that did not move is the
            # region tools' "take this region out" (other tools ignore it).
            layer = self.target()
            if layer is None or pixel is None or self.tool(layer) not in REGION_TOOLS or self.blocked(layer):
                return False
            self._wand(layer, view_index, self._voxel(layer, view_index, *pixel), event, remove=True)
            return True
        # Press: Ctrl+click always moves the crosshair.
        if event is not None and bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return False
        layer = self.target()
        tool = self.tool(layer)
        if tool == "box":
            return self._box_press(view_index, pixel)
        if layer is None or tool is None:
            return False
        reason = self.blocked(layer)
        if reason is not None:
            self._say(reason, error=True)
            return False
        if pixel is None:
            return True
        right = event is not None and event.button() == Qt.MouseButton.RightButton
        alt = event is not None and bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        labeling = self._labeling()
        try:
            E.announce_history_loads(layer)
            voxel = self._voxel(layer, view_index, *pixel)
            if right:
                self._start(layer, view_index, voxel, "erase")
            elif tool == "pick":
                value = int(layer.data[voxel])
                layer.selected_label = value
                self._say(f"Picked label {value}.")
            elif tool == "fill":
                n = E.fill_at(layer, voxel, self._edit_dims(layer, view_index), int(layer.selected_label))
                self._say(f"Filled {n:,} voxel(s) with label {int(layer.selected_label)}.")
            elif tool in REGION_TOOLS and labeling is not None:
                index = labeling.region_index(layer, voxel, self._region_free(layer, view_index))
                self._region = labeling.begin_region(layer, voxel, index)
                self._tune = {"x": self._cursor_x(event)} if (alt and self._region is not None) else None
                self._region_view = int(view_index)
            elif tool == "tracer" and labeling is not None:
                labeling.tracer_add(layer, voxel)
            elif tool == "smart" and labeling is not None:
                self._smart = labeling.begin_smart(layer, self._edit_dims(layer, view_index))
                self._smart_view = int(view_index)
                if self._smart is not None:
                    self._smart.extend(voxel)
            elif tool == "polygon":
                self._polygon_click(layer, view_index, pixel)
            else:
                self._start(layer, view_index, voxel, tool)
        except Exception as exc:  # noqa: BLE001 — shown, not raised into Qt
            self._abort()
            self._say(f"Could not edit “{layer.name}”: {exc}", error=True)
        return True

    def _move(self, view_index: int, pixel: tuple[int, int] | None, event: Any) -> bool:
        if self._boxdrag is not None:
            if pixel is not None:
                self._box_move(pixel)
            return True
        if self._region is not None:
            stroke = self._region
            try:
                if self._tune is not None:
                    labeling = self._labeling()
                    value = labeling.clamp_tune(stroke.tool, stroke.options["tune"]
                                                + (self._cursor_x(event) - self._tune["x"])
                                                * labeling.tune_step(stroke.tool))
                    if abs(value - stroke.value) > 1e-9:
                        stroke.tune(value)
                        self._panel.set_draw_hint(f"✎ {TUNED[stroke.tool].capitalize()} {value:.4g}: region "
                                                  f"{stroke.size:,} voxel(s) — release to keep it")
                elif pixel is not None:
                    stroke.add(self._voxel(stroke.layer, self._region_view, *pixel))
            except Exception as exc:  # noqa: BLE001
                self._say(str(exc), error=True)
            return True
        if self._smart is not None:
            if pixel is not None:
                self._smart.extend(self._voxel(self._smart.layer, self._smart_view, *pixel))
            return True
        if self._stroke is not None and pixel is not None:
            self._extend(pixel)
        return self._stroke is not None

    def _release(self) -> bool:
        labeling = self._labeling()
        if self._boxdrag is not None:
            self._box_release()
            return True
        if self._region is not None:
            stroke, tuned = self._region, self._tune is not None
            self._region, self._tune = None, None
            if labeling is not None:
                labeling.finish_region(stroke, tuned=tuned)
            self.refresh()
            return True
        if self._smart is not None:
            stroke, self._smart = self._smart, None
            if labeling is not None:
                labeling.finish_smart(stroke)
            return True
        if self._stroke is None:
            return False
        self._finish()
        return True

    @staticmethod
    def _cursor_x(event: Any) -> float:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x())

    def _region_free(self, layer: Any, view_index: int) -> list[int]:
        """The axes a region grows over: the view's plane, or the volume (3D)."""
        labeling = self._labeling()
        if labeling is not None and labeling.wand_in_3d():
            return self._layer_dims(layer)[0]
        return self._plane_dims(layer, view_index)

    def _wand(self, layer: Any, view_index: int, voxel: tuple[int, ...], event: Any, *,
              remove: bool | None = None) -> None:
        labeling = self._labeling()
        if labeling is None:
            return
        free = set(self._region_free(layer, view_index))
        index = tuple(slice(None) if d in free else int(voxel[d]) for d in range(layer.data.ndim))
        if remove is None:
            remove = event is not None and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        tool = self.tool(layer)
        labeling.wand_from(voxel, index, remove=remove, tool=tool if tool in REGION_TOOLS else "wand")

    # ── the box ──────────────────────────────────────────────────────────────

    def _box_layer(self) -> Any | None:
        labeling = self._labeling()
        return labeling._box_grid_layer() if labeling is not None else None

    def _box_press(self, view_index: int, pixel: tuple[int, int] | None) -> bool:
        from nvitk.gui.labels.roi_box import roi_box

        layer = self._box_layer()
        if layer is None or pixel is None:
            return layer is not None
        reason = self.blocked(layer)
        if reason is not None:
            self._say(reason, error=True)
            return False
        box = roi_box(self._viewer)
        self._boxdrag = {"view": int(view_index), "layer": layer, "start": pixel,
                         "before": (box.shape, list(box.lo), list(box.hi), box.layer), "moved": False}
        return True

    def _box_ranges(self, pixel: tuple[int, int]) -> dict[int, tuple[int, int]]:
        drag = self._boxdrag
        panel = self._panel
        view = panel._views[drag["view"]]
        spatial, _t = self._layer_dims(drag["layer"])
        ra, r0, ca, c0 = view.to_voxel(*drag["start"], panel._data.shape)
        _ra, r1, _ca, c1 = view.to_voxel(*pixel, panel._data.shape)
        return {spatial[ra]: (min(r0, r1), max(r0, r1) + 1), spatial[ca]: (min(c0, c1), max(c0, c1) + 1)}

    def _box_move(self, pixel: tuple[int, int]) -> None:
        from nvitk.gui.labels.roi_box import roi_box

        drag = self._boxdrag
        if pixel != drag["start"]:
            drag["moved"] = True
        roi_box(self._viewer).set_axes(drag["layer"], self._box_ranges(pixel), emit=False)
        self.show_box()

    def _box_release(self) -> None:
        from nvitk.gui.labels.roi_box import roi_box

        drag, self._boxdrag = self._boxdrag, None
        box = roi_box(self._viewer)
        if not drag["moved"]:
            shape, lo, hi, layer = drag["before"]
            if shape is None:
                box.clear()
            elif layer is not None:
                box.set(layer, lo, hi)
            return
        box.changed.emit()
        labeling = self._labeling()
        if labeling is not None:
            labeling.box_drawn()
        self.refresh()

    def show_box(self) -> None:
        """Draw the box's rectangle in each view (faint where the slice is outside it)."""
        from nvitk.gui.labels.roi_box import spatial_dims

        panel = self._panel
        box = getattr(self._viewer, "_nvitk_roi_box", None)
        anchor = panel.bound_layer()
        views = panel._slice_views
        if (box is None or not box.active or anchor is None or panel._data is None or panel.is_oblique()
                or not box.applies(anchor) or len(spatial_dims(anchor)) != 3):
            for view in views:
                view.set_box(None)
            return
        shape = panel._data.shape
        for i, view in enumerate(panel._views):
            if i >= len(views):
                break

            def edges(axis: int, flip: bool) -> tuple[int, int]:
                lo, hi = box.lo[axis], box.hi[axis]
                return (int(shape[axis]) - hi, int(shape[axis]) - lo) if flip else (lo, hi)

            r0, r1 = edges(view.rows, view.flip_rows)
            c0, c1 = edges(view.cols, view.flip_cols)
            inside = box.lo[view.axis] <= int(panel._position[view.axis]) < box.hi[view.axis]
            views[i].set_box((r0, c0, r1, c1), inside=inside)

    # ── a traced vessel ──────────────────────────────────────────────────────

    def show_trace(self, layer: Any | None, points: list[Any], path: list[Any]) -> None:
        """Draw a vessel being traced on *layer* (``None``: nothing) in every view."""
        self._trace = (weakref.ref(layer), list(points), list(path)) if layer is not None else None
        self._draw_trace()

    def _draw_trace(self) -> None:
        panel = self._panel
        trace = self._trace
        layer = trace[0]() if trace is not None else None
        if layer is None or panel._data is None or panel.is_oblique() or self.blocked(layer) is not None:
            for view in panel._slice_views:
                view.set_trace(None)
            return
        spatial, _t = self._layer_dims(layer)
        shape = panel._data.shape

        def pixel(voxel: Any, view: Any) -> tuple[int, int]:
            position = [int(voxel[d]) for d in spatial]
            return view.to_pixel(position, shape)

        for i, view in enumerate(panel._views):
            if i >= len(panel._slice_views):
                break
            panel._slice_views[i].set_trace([pixel(v, view) for v in trace[2]], [pixel(v, view) for v in trace[1]])

    def overlays(self) -> None:
        """The box and a traced vessel, in the views as they now stand."""
        self.show_box()
        if self._trace is not None:
            self._draw_trace()

    # ── a polygon ────────────────────────────────────────────────────────────

    def _polygon_click(self, layer: Any, view_index: int, pixel: tuple[int, int]) -> None:
        """Add a corner (or close the polygon, back on its first corner)."""
        panel = self._panel
        view = panel._views[view_index]
        row_axis, r, col_axis, c = view.to_voxel(*pixel, panel._data.shape)
        plane_key = (id(layer), int(view_index), int(panel._position[view.axis]), int(panel._time))
        poly = self._poly
        if poly is not None and poly["key"] != plane_key:
            self.cancel_polygon()  # another view, slice or layer: start again there
            poly = None
        if poly is None:
            poly = self._poly = {"key": plane_key, "view": int(view_index), "layer": weakref.ref(layer),
                                 "axes": (row_axis, col_axis), "points": []}
        points = poly["points"]
        if points and (r, c) == points[-1]:
            return
        if len(points) >= 3 and abs(r - points[0][0]) <= 1 and abs(c - points[0][1]) <= 1:
            self.finish_polygon()
            return
        points.append((r, c))
        self._show_polygon()
        self._panel.set_draw_hint(f"✎ Polygon: {len(points)} corner(s) — double-click or click the first "
                                  "corner to fill · Esc cancels")

    def _show_polygon(self) -> None:
        poly = self._poly
        if poly is None:
            return
        panel = self._panel
        view = panel._views[poly["view"]]
        pixels = []
        for r, c in poly["points"] + poly["points"][:1]:
            position = list(panel._position)
            position[poly["axes"][0]], position[poly["axes"][1]] = r, c
            pixels.append(view.to_pixel(position, panel._data.shape))
        panel._slice_views[poly["view"]].set_polyline(pixels)

    def finish_polygon(self) -> None:
        """Fill the polygon with the active label in its plane (through the Labeling tab)."""
        poly, self._poly = self._poly, None
        if poly is None:
            return
        self._panel._slice_views[poly["view"]].set_polyline(None)
        layer = poly["layer"]()
        if layer is None or len(poly["points"]) < 3:
            self.refresh()
            return
        panel = self._panel
        spatial, t_ax = self._layer_dims(layer)
        dim_r, dim_c = spatial[poly["axes"][0]], spatial[poly["axes"][1]]
        voxel = [0] * layer.data.ndim
        for k, d in enumerate(spatial):
            voxel[d] = int(panel._position[k])
        if t_ax is not None:
            voxel[t_ax] = int(min(panel._time, layer.data.shape[t_ax] - 1))
        index = tuple(slice(None) if d in (dim_r, dim_c) else voxel[d] for d in range(layer.data.ndim))
        # The plane's corner lists in the order of its free axes in the array.
        first, second = sorted((dim_r, dim_c))
        coords = {dim_r: [p[0] for p in poly["points"]], dim_c: [p[1] for p in poly["points"]]}
        labeling = self._labeling()
        if labeling is not None and labeling.target() is layer:
            labeling.fill_polygon(index, coords[first], coords[second])
        else:
            E.fill_polygon(layer, index, coords[first], coords[second], int(layer.selected_label))
        self.refresh()

    def cancel(self) -> None:
        """Esc: drop the polygon or the vessel being traced, or stop drawing the box."""
        labeling = self._labeling()
        if self._poly is not None:
            self.cancel_polygon()
        elif labeling is not None and labeling._trace is not None:
            labeling.tracer_cancel()
        elif labeling is not None and labeling.box_armed():
            labeling.box_drawn()

    def cancel_polygon(self) -> None:
        """Drop the polygon being drawn."""
        poly, self._poly = self._poly, None
        if poly is not None:
            self._panel._slice_views[poly["view"]].set_polyline(None)
            self.refresh()

    def slice_moved(self) -> None:
        """The views moved to other slices: a polygon on the old one is dropped."""
        poly = self._poly
        if poly is None:
            return
        view = self._panel._views[poly["view"]]
        if int(self._panel._position[view.axis]) != poly["key"][2]:
            self.cancel_polygon()

    # ── a brush stroke ───────────────────────────────────────────────────────

    def _start(self, layer: Any, view_index: int, voxel: tuple[int, ...], tool: str) -> None:
        """Open one history item for the whole drag, as Napari's own brush does."""
        label = 0 if tool == "erase" else int(layer.selected_label)
        previous = getattr(layer, "_block_history", None)
        if previous is not None:
            layer._block_history = True
        self._stroke = {
            "layer": weakref.ref(layer),
            "view": int(view_index),
            "last": voxel,
            "label": label,
            "dims": self._edit_dims(layer, view_index),
            "previous": previous,
        }
        self._dab(voxel)

    def _extend(self, pixel: tuple[int, int]) -> None:
        stroke = self._stroke
        layer = stroke["layer"]() if stroke else None
        if layer is None:
            self._abort()
            return
        voxel = self._voxel(layer, stroke["view"], *pixel)
        last = stroke["last"]
        # Fill the gap between two mouse events: dabs at most a quarter brush apart.
        span = max(abs(a - b) for a, b in zip(voxel, last))
        step = max(float(layer.brush_size) / 4.0, 1.0)
        n = max(int(math.ceil(span / step)), 1)
        for k in range(1, n + 1):
            t = k / n
            self._dab(tuple(int(round(a + (b - a) * t)) for a, b in zip(last, voxel)))
        stroke["last"] = voxel

    def _dab(self, voxel: tuple[int, ...]) -> None:
        stroke = self._stroke
        layer = stroke["layer"]()
        indices = E.brush_indices(layer, voxel, stroke["dims"])
        if E.paint_at(layer, indices, stroke["label"], refresh=False):
            self._live_timer.start()

    def _finish(self) -> None:
        """Commit the stroke as one history item (the editable area trims it then)."""
        stroke, self._stroke = self._stroke, None
        layer = stroke["layer"]() if stroke else None
        if layer is None:
            return
        try:
            if stroke["previous"] is not None:
                layer._commit_staged_history()
                layer._block_history = stroke["previous"]
        finally:
            self._live_timer.stop()
            try:
                layer.refresh()
            except Exception:  # noqa: BLE001
                pass

    def _abort(self) -> None:
        if self._stroke is not None:
            self._finish()

    def _live_refresh(self) -> None:
        """Show the stroke so far: the views and the main canvas, without a rebuild."""
        stroke = self._stroke
        layer = stroke["layer"]() if stroke else None
        if layer is None:
            return
        try:
            if hasattr(layer, "_partial_labels_refresh"):
                layer._partial_labels_refresh()
            else:
                layer.refresh()
        except Exception:  # noqa: BLE001
            pass
        self._panel.refresh_painted(layer, (stroke["label"],))

    # ── feedback ─────────────────────────────────────────────────────────────

    def refresh(self, _event: Any = None) -> None:
        """Re-read the target and its tool: brush outlines, cursor and the hint line."""
        layer = self.target()
        self._watch(layer)
        tool = self.tool(layer)
        reason = self.blocked(layer) if (layer is not None and tool is not None) else None
        drawing = tool is not None and reason is None
        brush = None
        if drawing and tool in ("paint", "erase", "smart"):
            brush = layer
        for index, view in enumerate(self._panel._slice_views):
            radii = None
            if brush is not None and index < len(self._panel._views):
                rows, cols = self._plane_dims(brush, index)
                r = E.brush_radius_voxels(brush, [rows, cols])
                radii = (r[rows], r[cols])
            view.set_brush(radii)
            view.set_draw_cursor(drawing)
        if tool is None or layer is None:
            self._panel.set_draw_hint("")
        elif reason is not None:
            self._panel.set_draw_hint(f"✎ {reason}", error=True)
        else:
            lid = int(layer.selected_label)
            what = {
                "paint": f"Painting label {lid}",
                "erase": "Erasing",
                "fill": f"Filling with label {lid}",
                "pick": "Picking a label",
                "wand": f"Magic wand, label {lid}: click or drag (Shift+click removes, Alt+drag tunes)",
                "flood": f"Adaptive flood, label {lid}: click or drag (Alt+drag tunes k)",
                "vessel": f"Vessel flood, label {lid}: click or drag along a vessel (Alt+drag tunes)",
                "tracer": f"Vessel tracer, label {lid}: click points along the vessel, double-click to fill",
                "smart": f"Smart brush, label {lid}",
                "polygon": f"Polygon, label {lid}: click the corners, double-click to fill",
                "box": "Box: drag a rectangle (its two axes); a drag in another view sets the third",
            }[tool]
            self._panel.set_draw_hint(
                f"✎ {what} on “{layer.name}” — right-drag erases · Ctrl+click moves the crosshair · "
                "Shift+drag pans · Ctrl+Z undoes"
            )

    def _watch(self, layer: Any | None) -> None:
        """Follow *layer*'s tool, label and brush (and nothing else's)."""
        if self._watched() is layer:
            return
        for emitter, slot in self._subs:
            try:
                emitter.disconnect(slot)
            except Exception:  # noqa: BLE001
                pass
        self._subs = []
        self._watched = weakref.ref(layer) if layer is not None else (lambda: None)
        if layer is None:
            return
        for name in ("mode", "selected_label", "brush_size", "n_edit_dimensions", "scale", "affine"):
            emitter = getattr(layer.events, name, None)
            if emitter is not None:
                emitter.connect(self.refresh)
                self._subs.append((emitter, self.refresh))

    def _say(self, text: str, *, error: bool = False) -> None:
        try:
            self._viewer.status = text
        except Exception:  # noqa: BLE001
            pass
        if error:
            self._panel.set_draw_hint(f"✎ {text}", error=True)

    def undo(self) -> None:
        layer = self.target()
        if layer is not None and E.is_labels(layer):
            layer.undo()

    def redo(self) -> None:
        layer = self.target()
        if layer is not None and E.is_labels(layer):
            layer.redo()


__all__ = ["DRAW_TOOLS", "OrthoLabelEditor"]
