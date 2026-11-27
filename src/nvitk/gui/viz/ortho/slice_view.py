"""One scrollable orthogonal slice view (:class:`SliceView`) and its crosshair, brush and box overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import (
    Qt,
    Signal,
)
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    FONT_S,
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_TEXT,
    SPACE_TIGHT,
)
from nvitk.gui.viz.ortho.constants import (
    _MIN_ZOOM,
    _MAX_ZOOM,
    _ZOOM_STEP,
    _WHEEL_NOTCH,
    _HANDLE_TOL_PX,
    _HANDLE_MIN_FRACTION,
)

#: How close (canvas pixels) the cursor must be to an edge of the box to grab it.
_BOX_GRAB_PX = 6.0
from nvitk.gui.viz.ortho.geometry import AxisView
from nvitk.gui.viz.ortho.render import (
    PaintRun,
    flatten_runs,
)

class SliceView(QWidget):
    """One scrollable orthogonal slice, with crosshairs marking the other two."""

    #: (axis, new index) when the user scrolls, drags or clicks this view.
    sliceChanged = Signal(int, int)
    #: (row, column) pixel in the *rendered* slice when the user clicks in it.
    #: The panel maps it back through the view's orientation.
    pixelPicked = Signal(int, int)
    #: (line index, screen angle in radians) while a crosshair end is dragged.
    handleDragged = Signal(int, float)
    #: The title was double-clicked (the panel shows this view alone, or the grid again).
    titleDoubleClicked = Signal()
    #: The slice pixel ``(row, col)`` under the mouse, or ``None`` when it leaves the slice.
    hovered = Signal(object)

    def __init__(self, view: AxisView, parent: QWidget | None = None) -> None:
        """Build the titled image canvas and its slice slider."""
        super().__init__(parent)
        self._view = view
        self._rgb: np.ndarray | None = None
        #: The slice as interpolation runs (see :class:`PaintRun`); ``_rgb`` is
        #: their flattened native-resolution composite.
        self._runs: list[PaintRun] = []
        #: Wheel travel not yet turned into a slice step (eighths of a degree).
        self._wheel_accum = 0.0
        self._aspect = 1.0
        self._cross: tuple[int, int] | None = None
        self._count = 1
        #: Magnification over the fit-to-canvas size, and the normalised point of
        #: the slice held at the canvas centre.
        self._zoom = 1.0
        self._pan = [0.5, 0.5]
        self._drag_from: tuple[float, float] | None = None
        #: Where a pan gesture was pressed: one released without moving is a
        #: Shift+click, which the label editor may want (the wand removes with it).
        self._pan_origin: tuple[float, float] | None = None
        #: ``(d_row, d_col, colour)`` per crosshair line. Empty means the plain
        #: horizontal/vertical cross, which is what an unrotated view shows.
        self._lines: list[tuple[float, float, str]] = []
        #: Index of the line whose end is being dragged, if any.
        self._rotating: int | None = None
        #: Label editing (:class:`~nvitk.gui.labels.ortho_edit.OrthoLabelEditor`):
        #: ``handler(kind, pixel, event) -> bool`` sees presses, drags and releases
        #: first; True means it drew, and the crosshair stays put.
        self.edit_handler: Any | None = None
        #: Asked on a left press whether it pans (the Labeling tab's Pan key held).
        self.pan_check: Any | None = None
        self._editing = False
        #: Brush outline ``(radius down, radius across)`` in slice pixels, drawn at
        #: the cursor while a brush tool is on; and the cursor's canvas position.
        self._brush: tuple[float, float] | None = None
        self._hover: tuple[float, float] | None = None
        #: A polygon being drawn in this view: its corners as (row, column) slice pixels.
        self._polyline: list[tuple[float, float]] = []
        #: The Labeling box in this plane: (row0, col0, row1, col1) slice-pixel edges,
        #: and whether the slice is inside the box's depth (drawn faint when not).
        self._box: tuple[float, float, float, float] | None = None
        self._box_inside = True
        #: Dragging the box's edges: ``handler(kind, edges, (row, col), event) -> bool`` with
        #: *kind* press / move / release, *edges* a set of ``r0 c0 r1 c1`` (or ``{"move"}``)
        #: and fractional slice-pixel coordinates; True on press means it took the drag.
        self.box_handler: Any | None = None
        self._box_grab: frozenset[str] | None = None
        self._box_hover: frozenset[str] | None = None
        #: Transform mode (:class:`~nvitk.gui.viz.ortho.transform_drag.OrthoTransformDrag`):
        #: ``handler(kind, (row, col), event) -> bool`` with *kind* can / press / move / release /
        #: cancel and fractional slice-pixel edges; True on press means it took the drag.
        self.transform_handler: Any | None = None
        self._transforming = False
        self._transform_from: tuple[float, float] | None = None
        self._transform_hover = False
        #: Points and meshes drawn over the slice (:mod:`~nvitk.gui.viz.ortho.shape_overlays`).
        self._overlays: list[Any] = []
        self._draw_cursor = False
        #: A traced vessel's path and clicked points, as (row, column) slice pixels.
        self._trace: list[tuple[float, float]] = []
        self._trace_marks: list[tuple[float, float]] = []
        #: The composited slice scaled into the canvas, kept between repaints that
        #: only move overlays (the brush outline follows every mouse move).
        self._base: QPixmap | None = None
        self._base_key: tuple | None = None
        #: Bumped by every new slice image: the cache key (an ``id`` could be reused).
        self._serial = 0

        self._title = QLabel(view.title)
        self._title.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: {FONT_S}px; font-weight: bold; letter-spacing: 1px;"
        )
        self._title.setToolTip("Double-click: this view alone, and back to the four panes.\n"
                               "Drag the gaps between the panes to resize them; double-click a "
                               "gap to even them out.")
        self._title.installEventFilter(self)
        self._index_label = QLabel("—")
        self._index_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: {FONT_S}px;")
        self._index_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        head.addWidget(self._title)
        head.addStretch(1)
        head.addWidget(self._index_label)
        self._head = head

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        # Small enough for a pane to be dragged narrow; set explicitly, or the
        # label's pixmap would become its minimum and stop the dividers.
        self._canvas.setMinimumSize(60, 60)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet(
            f"background-color: #000000; border: 1px solid {COLOR_BORDER}; border-radius: 4px;"
        )
        self._canvas.installEventFilter(self)
        # Tracked always: the value readout follows the cursor, not only the brush outline.
        self._canvas.setMouseTracking(True)
        #: Whether the crosshair lines are drawn (the viewport's crosshair toggle).
        self._show_cross = True
        # A click gives the view the keyboard: the views' Undo / Esc and the Labeling
        # tab's Pan key then reach it, not a text box focused earlier.
        self._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(
            lambda value: self.sliceChanged.emit(self._view.axis, int(value))
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addLayout(head)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(self._slider)

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """Put *widget* (a pane action) at the right end of the title row."""
        self._head.addWidget(widget)
        return widget

    def title_text(self) -> str:
        """The view's current title (``"Axial · S"``...)."""
        return self._title.text()

    # rendering

    def set_slice(
        self,
        rgb: np.ndarray | None,
        *,
        index: int,
        count: int,
        aspect: float = 1.0,
        crosshair: tuple[int, int] | None = None,
    ) -> None:
        """Show *rgb* as this view's slice, at *index* of *count*.

        *rgb* is an RGB array, or a list of :class:`PaintRun`: the layer stack split
        by interpolation, each run scaled with its own filter when painted.
        """
        if isinstance(rgb, (list, tuple)):
            self._runs = list(rgb)
            self._rgb = (
                np.asarray(self._runs[0].image)[..., :3] if len(self._runs) == 1
                else flatten_runs(self._runs)
            ) if self._runs else None
        else:
            self._rgb = rgb
            self._runs = [PaintRun(np.asarray(rgb), True, "base")] if rgb is not None else []
        self._serial += 1
        self._aspect = float(aspect) if np.isfinite(aspect) and aspect > 0 else 1.0
        self._cross = crosshair
        self._count = max(int(count), 1)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(self._count - 1, 0))
        self._slider.setValue(int(np.clip(index, 0, max(self._count - 1, 0))))
        self._slider.blockSignals(False)
        self._index_label.setText(f"{int(index) + 1} / {self._count}")
        self._repaint()

    def set_crosshair(self, crosshair: tuple[int, int] | None) -> None:
        """Move the crosshair without re-uploading the slice image."""
        self._cross = crosshair
        self._repaint()

    def set_brush(self, radii: tuple[float, float] | None) -> None:
        """Draw the brush outline (radius down, radius across, in slice pixels) at the
        cursor, or stop drawing it (``None``)."""
        radii = None if radii is None else (float(radii[0]), float(radii[1]))
        if radii == self._brush:
            return
        self._brush = radii
        if radii is None:
            self._hover = None
        self._repaint()

    def set_polyline(self, points: list[tuple[float, float]] | None) -> None:
        """Draw a polygon's corners so far (slice pixels), or nothing."""
        self._polyline = list(points or [])
        self._repaint()

    def set_box(self, rect: tuple[float, float, float, float] | None, *, inside: bool = True) -> None:
        """Draw the Labeling box's rectangle (slice-pixel edges ``(r0, c0, r1, c1)``),
        faint when this slice is outside its depth; ``None``: no box."""
        rect = None if rect is None else tuple(float(v) for v in rect)
        if rect == self._box and bool(inside) == self._box_inside:
            return
        self._box, self._box_inside = rect, bool(inside)
        self._repaint()

    def set_trace(self, path: list[tuple[float, float]] | None, marks: list[tuple[float, float]] | None = None) -> None:
        """Draw a traced vessel: its path and its clicked points (slice pixels)."""
        self._trace = list(path or [])
        self._trace_marks = list(marks or [])
        self._repaint()

    def set_draw_cursor(self, drawing: bool) -> None:
        """A cross cursor while a drawing tool is on (a click edits, not moves)."""
        self._draw_cursor = bool(drawing)
        self._apply_cursor()

    def _apply_cursor(self) -> None:
        edges = self._box_grab or self._box_hover
        if self._transforming or (self._transform_hover and not edges):
            self._canvas.setCursor(Qt.ClosedHandCursor if self._transforming else Qt.SizeAllCursor)
        elif edges:
            if edges == {"move"}:
                shape = Qt.SizeAllCursor
            elif len(edges) == 2:
                diagonal = edges in ({"r0", "c0"}, {"r1", "c1"})
                shape = Qt.SizeFDiagCursor if diagonal else Qt.SizeBDiagCursor
            else:
                shape = Qt.SizeVerCursor if next(iter(edges)) in ("r0", "r1") else Qt.SizeHorCursor
            self._canvas.setCursor(shape)
        elif self._draw_cursor:
            self._canvas.setCursor(Qt.CrossCursor)
        else:
            self._canvas.unsetCursor()

    def _box_screen_rect(self) -> tuple[float, float, float, float] | None:
        """The box's rectangle on the canvas: ``(x0, y0, x1, y1)`` (``None``: no box)."""
        geometry = self._geometry()
        if geometry is None or self._box is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        r0, c0, r1, c1 = self._box
        x0, x1 = sorted((off_x + c0 / max(w, 1) * scaled_w, off_x + c1 / max(w, 1) * scaled_w))
        y0, y1 = sorted((off_y + r0 / max(h, 1) * scaled_h, off_y + r1 / max(h, 1) * scaled_h))
        return x0, y0, x1, y1

    def _box_edges_at(self, event: Any) -> frozenset[str] | None:
        """The box edges under the cursor (a corner grabs two; Alt inside the box moves it)."""
        if self.box_handler is None:
            return None
        rect = self._box_screen_rect()
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        x, y = self._cursor_xy(event)
        tol = _BOX_GRAB_PX
        if event.modifiers() & Qt.AltModifier:
            inside = x0 <= x <= x1 and y0 <= y <= y1
            return frozenset({"move"}) if inside else None
        within_y = y0 - tol <= y <= y1 + tol
        within_x = x0 - tol <= x <= x1 + tol
        edges = set()
        if within_y:
            near = [(abs(x - x0), "c0"), (abs(x - x1), "c1")]
            d, edge = min(near)
            if d <= tol:
                edges.add(edge)
        if within_x:
            near = [(abs(y - y0), "r0"), (abs(y - y1), "r1")]
            d, edge = min(near)
            if d <= tol:
                edges.add(edge)
        return frozenset(edges) or None

    def _pixel_f(self, event: Any) -> tuple[float, float] | None:
        """The cursor in fractional slice pixels (row, column), off the slice too."""
        geometry = self._geometry()
        if geometry is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        x, y = self._cursor_xy(event)
        return (y - off_y) / max(scaled_h, 1) * h, (x - off_x) / max(scaled_w, 1) * w

    def _box_call(self, kind: str, event: Any) -> bool:
        handler = self.box_handler
        if handler is None or self._box_grab is None:
            return False
        try:
            return bool(handler(kind, set(self._box_grab), self._pixel_f(event), event))
        except Exception:  # noqa: BLE001 (an editor fault must not wedge the view)
            return False

    def _transform_call(self, kind: str, event: Any) -> bool:
        handler = self.transform_handler
        if handler is None:
            return False
        try:
            return bool(handler(kind, self._pixel_f(event) if kind != "cancel" else None, event))
        except Exception:  # noqa: BLE001 (a transform fault must not wedge the view)
            return False

    def set_overlays(self, overlays: list[Any] | None) -> None:
        """Draw points / mesh cuts over the slice (:class:`~nvitk.gui.viz.ortho.shape_overlays.Overlay`)."""
        overlays = list(overlays or [])
        if not overlays and not self._overlays:
            return
        self._overlays = overlays
        self._repaint()

    def _paint_overlays(self, painter: Any, geometry: tuple[int, int, float, float]) -> None:
        """Points as discs, mesh cuts as lines, in slice-pixel coordinates mapped to the canvas."""
        from qtpy.QtCore import QLineF, QPointF
        from qtpy.QtGui import QColor, QPainter, QPen

        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        sx, sy = scaled_w / max(w, 1), scaled_h / max(h, 1)

        def qcolour(rgba: Any, opacity: float) -> QColor:
            r, g, b, a = (float(v) for v in rgba)
            return QColor.fromRgbF(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0),
                                   min(max(a * opacity, 0.0), 1.0))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for item in self._overlays:
            coords = np.asarray(item.coords, dtype=float)
            if coords.size == 0:
                continue
            if item.kind == "lines":
                x0 = off_x + (coords[:, 1] + 0.5) * sx
                y0 = off_y + (coords[:, 0] + 0.5) * sy
                x1 = off_x + (coords[:, 3] + 0.5) * sx
                y1 = off_y + (coords[:, 2] + 0.5) * sy
                # One pen per colour: a one-colour mesh is a single drawLines call.
                quant = np.round(np.asarray(item.colors, dtype=float) * 63.0).astype(np.int16)
                keys, inverse = np.unique(quant, axis=0, return_inverse=True)
                inverse = np.asarray(inverse).reshape(-1)
                for k, key in enumerate(keys):
                    pick = np.flatnonzero(inverse == k)
                    pen = QPen(qcolour(key / 63.0, item.opacity), float(getattr(item, "width", 1.5)))
                    painter.setPen(pen)
                    painter.drawLines([QLineF(x0[i], y0[i], x1[i], y1[i]) for i in pick])
                continue
            radii = np.asarray(item.radii, dtype=float) if item.radii is not None else np.ones((len(coords), 2))
            xs = off_x + (coords[:, 1] + 0.5) * sx
            ys = off_y + (coords[:, 0] + 0.5) * sy
            screen = getattr(item, "screen_size", None)
            for i in range(len(coords)):
                if screen is not None:
                    rx = ry = max(float(screen) / 2.0, 1.0)  # mesh vertices: a fixed dot size
                else:
                    rx = max(float(radii[i, 1]) * sx, 2.0)
                    ry = max(float(radii[i, 0]) * sy, 2.0)
                painter.setBrush(qcolour(item.colors[i], item.opacity))
                if item.edge_colors is not None and item.edge_width > 0:
                    painter.setPen(QPen(qcolour(item.edge_colors[i], item.opacity), float(item.edge_width)))
                else:
                    painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(float(xs[i]), float(ys[i])), rx, ry)
        painter.setBrush(Qt.NoBrush)

    def set_crosshair_visible(self, visible: bool) -> None:
        """Draw the crosshair lines, or hide them (the crosshair still moves with clicks)."""
        self._show_cross = bool(visible)
        self._repaint()

    def set_lines(self, lines: list[tuple[float, float, str]]) -> None:
        """Set the crosshair line directions, in pixel space, with their colours."""
        self._lines = list(lines or [])
        self._repaint()

    def _line_geometry(self) -> list[tuple[float, float, float, float, str]] | None:
        """Each crosshair line as ``(cx, cy, d_x, d_y, colour)`` in canvas pixels."""
        geometry = self._geometry()
        if geometry is None or self._cross is None or self._rgb is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        row, col = self._cross
        cx = off_x + (col + 0.5) / max(w, 1) * scaled_w
        cy = off_y + (row + 0.5) / max(h, 1) * scaled_h
        out = []
        entries = self._lines or [(1.0, 0.0, COLOR_ACCENT), (0.0, 1.0, COLOR_ACCENT)]
        for d_row, d_col, colour in entries:
            # Pixel directions scale with the drawn size, so a rotated line keeps
            # its angle on screen whatever the zoom or the aspect correction.
            dx = float(d_col) * scaled_w / max(w, 1)
            dy = float(d_row) * scaled_h / max(h, 1)
            norm = float(np.hypot(dx, dy))
            if norm <= 1e-9:
                continue
            out.append((cx, cy, dx / norm, dy / norm, colour))
        return out

    def _handle_at(self, event: Any) -> int | None:
        """Index of the crosshair line whose end is under the cursor, if any."""
        lines = self._line_geometry()
        if not lines:
            return None
        x, y = self._cursor_xy(event)
        reach = max(self._canvas.width(), self._canvas.height()) / 2.0
        best, best_distance = None, _HANDLE_TOL_PX
        for index, (cx, cy, dx, dy, _colour) in enumerate(lines):
            vx, vy = x - cx, y - cy
            along = vx * dx + vy * dy
            # Only the ends grab: near the middle the same drag has to keep
            # meaning "move the crosshair", which is the commoner gesture.
            if abs(along) < _HANDLE_MIN_FRACTION * reach:
                continue
            across = abs(vx * dy - vy * dx)
            if across < best_distance:
                best, best_distance = index, across
        return best

    def _screen_angle(self, event: Any) -> float:
        """Angle of the cursor about the crosshair, in the slice's pixel frame."""
        geometry = self._geometry()
        if geometry is None or self._cross is None or self._rgb is None:
            return 0.0
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        row, col = self._cross
        x, y = self._cursor_xy(event)
        # Back into the slice's own pixel units, so the angle does not depend on
        # the zoom or on the aspect correction the canvas applies.
        d_col = (x - off_x) / max(scaled_w, 1) * w - (col + 0.5)
        d_row = (y - off_y) / max(scaled_h, 1) * h - (row + 0.5)
        return float(np.arctan2(d_row, d_col))

    def _geometry(self) -> tuple[int, int, float, float] | None:
        """``(width, height, off_x, off_y)`` of the drawn slice inside the canvas.

        One place, used by the painter and by the click mapping alike: the two
        computing the transform separately is how a zoomed view ends up putting
        the crosshair somewhere other than where it was clicked.
        """
        if self._rgb is None or self._rgb.size == 0:
            return None
        h, w = self._rgb.shape[:2]
        target = self._canvas.size()
        # Physical aspect: a 3 mm slice spacing must not be drawn as if isotropic.
        fit_w = max(int(target.width()), 1)
        fit_h = max(int(fit_w * (h * self._aspect) / max(w, 1)), 1)
        if fit_h > target.height():
            fit_h = max(int(target.height()), 1)
            fit_w = max(int(fit_h * max(w, 1) / max(h * self._aspect, 1e-6)), 1)
        scaled_w = max(int(fit_w * self._zoom), 1)
        scaled_h = max(int(fit_h * self._zoom), 1)
        if self._zoom <= 1.0:
            # Fitted: centred.
            return (scaled_w, scaled_h,
                    (target.width() - scaled_w) / 2.0, (target.height() - scaled_h) / 2.0)
        # Zoomed: the panned-to point sits at the canvas centre; nothing else. It
        # is always a point of the slice (pan is 0...1), so the slice can never leave
        # the view; centring or pinning it to the edges instead is what made a zoom
        # drift away from the cursor until the slice outgrew the canvas.
        off_x = target.width() / 2.0 - self._pan[0] * scaled_w
        off_y = target.height() / 2.0 - self._pan[1] * scaled_h
        return scaled_w, scaled_h, off_x, off_y

    def set_zoom(
        self,
        zoom: float,
        *,
        about: tuple[float, float] | None = None,
        anchor: tuple[float, float] | None = None,
    ) -> None:
        """Set the magnification.

        With *anchor* (a canvas position, the cursor's), the slice point under it
        stays under it: the zoom every image viewer does, so the wheel closes in
        on what the mouse is over. *about* (a normalised slice point) is instead
        put at the canvas centre.
        """
        new_zoom = float(np.clip(zoom, _MIN_ZOOM, _MAX_ZOOM))
        geometry = self._geometry() if anchor is not None else None
        self._zoom = new_zoom
        if new_zoom <= 1.0:
            self._pan = [0.5, 0.5]
        elif geometry is not None:
            scaled_w, scaled_h, off_x, off_y = geometry
            x, y = anchor
            # The slice point under the cursor, as a fraction of the drawn slice...
            u = (x - off_x) / max(scaled_w, 1)
            v = (y - off_y) / max(scaled_h, 1)
            # ...and the pan that leaves it there at the new size (_geometry places
            # the panned-to point at the centre, then keeps the slice on the canvas).
            new_geometry = self._geometry()
            if new_geometry is not None:
                new_w, new_h = new_geometry[0], new_geometry[1]
                centre_x = self._canvas.width() / 2.0
                centre_y = self._canvas.height() / 2.0
                self._pan = [
                    float(np.clip(u - (x - centre_x) / max(new_w, 1), 0.0, 1.0)),
                    float(np.clip(v - (y - centre_y) / max(new_h, 1), 0.0, 1.0)),
                ]
        elif about is not None:
            self._pan = [float(np.clip(c, 0.0, 1.0)) for c in about]
        self._repaint()

    def zoom(self) -> float:
        """Current magnification."""
        return float(self._zoom)

    def _repaint(self) -> None:
        """Scale the current slice into the canvas and draw the crosshairs on it."""
        from qtpy.QtCore import QPointF, QRectF
        from qtpy.QtGui import QColor, QPainter, QPen

        if self._canvas.width() < 2 or self._canvas.height() < 2:
            return  # a collapsed pane: drawn again when it is opened
        geometry = self._geometry()
        if geometry is None:
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("No slice")
            return
        scaled_w, scaled_h, off_x, off_y = geometry

        # Painted into a canvas-sized pixmap rather than handed over directly, so a
        # zoomed slice is cropped by the view instead of resizing the widget. The
        # scaled slice is kept: a repaint that only moves an overlay (the brush
        # outline, at every mouse move) starts from it instead of rescaling.
        key = (self._serial, self._canvas.width(), self._canvas.height(),
               scaled_w, scaled_h, round(off_x, 2), round(off_y, 2))
        if self._base is None or self._base_key != key:
            base = QPixmap(self._canvas.size())
            base.fill(QColor("#000000"))
            painter = QPainter(base)
            # Each interpolation run is scaled with the filter its layers use in
            # napari (nearest stays blocky, linear/cubic smooth), then stacked.
            for run in self._runs or [PaintRun(np.asarray(self._rgb), True, "base")]:
                data = np.ascontiguousarray(run.image, dtype=np.uint8)
                h, w = data.shape[:2]
                if data.ndim == 3 and data.shape[2] == 4:
                    image = QImage(data.data, w, h, 4 * w, QImage.Format_RGBA8888_Premultiplied).copy()
                else:
                    rgb3 = np.ascontiguousarray(data[..., :3])
                    image = QImage(rgb3.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                filt = Qt.SmoothTransformation if run.smooth else Qt.FastTransformation
                run_map = QPixmap.fromImage(image).scaled(scaled_w, scaled_h, Qt.IgnoreAspectRatio, filt)
                painter.setCompositionMode(
                    QPainter.CompositionMode_Plus if run.mode == "plus"
                    else QPainter.CompositionMode_SourceOver
                )
                painter.drawPixmap(int(round(off_x)), int(round(off_y)), run_map)
            painter.end()
            self._base, self._base_key = base, key
        canvas = QPixmap(self._base)
        painter = QPainter(canvas)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self._overlays:
            self._paint_overlays(painter, geometry)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for cx, cy, dx, dy, colour in (self._line_geometry() or []) if self._show_cross else []:
            pen = QPen(QColor(colour))
            pen.setWidth(1)
            painter.setPen(pen)
            reach = float(canvas.width() + canvas.height())
            painter.drawLine(
                int(round(cx - dx * reach)), int(round(cy - dy * reach)),
                int(round(cx + dx * reach)), int(round(cy + dy * reach)),
            )
            # Mark the ends that can be grabbed, so the gesture is discoverable.
            handle = _HANDLE_MIN_FRACTION * max(canvas.width(), canvas.height()) / 2.0
            for sign in (-1.0, 1.0):
                painter.drawEllipse(
                    int(round(cx + sign * dx * handle)) - 3,
                    int(round(cy + sign * dy * handle)) - 3,
                    6, 6,
                )
        if self._zoom > 1.0:
            pen = QPen(QColor(COLOR_MUTED))
            painter.setPen(pen)
            painter.drawText(6, canvas.height() - 6, f"{self._zoom:.1f}x")
        if self._polyline:
            h, w = self._rgb.shape[:2]
            pts = [QPointF(off_x + (c + 0.5) / max(w, 1) * scaled_w, off_y + (r + 0.5) / max(h, 1) * scaled_h)
                   for r, c in self._polyline]
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#ffd34d"), 1.5))
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a, b)
            for pt in pts:
                painter.drawRect(int(pt.x()) - 2, int(pt.y()) - 2, 4, 4)
        if self._box is not None:
            h, w = self._rgb.shape[:2]
            r0, c0, r1, c1 = self._box
            x0, x1 = off_x + c0 / max(w, 1) * scaled_w, off_x + c1 / max(w, 1) * scaled_w
            y0, y1 = off_y + r0 / max(h, 1) * scaled_h, off_y + r1 / max(h, 1) * scaled_h
            colour = QColor("#4dd2ff")
            if not self._box_inside:
                colour.setAlpha(110)
            pen = QPen(colour, 1.5)
            pen.setStyle(Qt.PenStyle.SolidLine if self._box_inside else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1))))
        if self._trace or self._trace_marks:
            h, w = self._rgb.shape[:2]

            def point(rc: tuple[float, float]) -> Any:
                return QPointF(off_x + (rc[1] + 0.5) / max(w, 1) * scaled_w, off_y + (rc[0] + 0.5) / max(h, 1) * scaled_h)

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#ffd34d"), 1.2))
            pts = [point(rc) for rc in self._trace]
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a, b)
            painter.setBrush(QColor("#ffd34d"))
            for rc in self._trace_marks:
                painter.drawEllipse(point(rc), 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)
        if self._brush is not None and self._hover is not None:
            # The brush's footprint, round in millimetres like the slice is.
            h, w = self._rgb.shape[:2]
            rx = self._brush[1] * scaled_w / max(w, 1)
            ry = self._brush[0] * scaled_h / max(h, 1)
            hx, hy = self._hover
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(hx, hy), max(rx, 1.0), max(ry, 1.0))
        painter.end()
        self._canvas.setPixmap(canvas)

    def resizeEvent(self, event: Any) -> None:
        """Re-scale the slice when the view is resized."""
        super().resizeEvent(event)
        self._repaint()

    # interaction

    def eventFilter(self, obj: Any, event: Any) -> bool:
        """Scroll to step slices; click to move the crosshair."""
        from qtpy.QtCore import QEvent

        if obj is self._title:
            if event.type() == QEvent.MouseButtonDblClick:
                self.titleDoubleClicked.emit()
                return True
            return False
        if obj is not self._canvas:
            return False
        if event.type() == QEvent.Wheel:
            delta = float(event.angleDelta().y())
            modifiers = event.modifiers()
            # Ctrl+wheel zooms, plain wheel scrolls slices. Scrolling is the far
            # commoner gesture here, so it keeps the unmodified wheel.
            if modifiers & Qt.ControlModifier:
                # In proportion to the travel: a standard notch is one 10 % step,
                # a touchpad's stream of small deltas adds up to the same thing.
                self.set_zoom(
                    self._zoom * (_ZOOM_STEP ** (delta / _WHEEL_NOTCH)),
                    anchor=self._cursor_xy(event),
                )
                return True
            # One slice per notch, however the travel arrives.
            self._wheel_accum += delta
            steps = int(self._wheel_accum / _WHEEL_NOTCH)
            if steps:
                self._wheel_accum -= steps * _WHEEL_NOTCH
                self._slider.setValue(
                    int(np.clip(self._slider.value() + steps, 0, self._count - 1))
                )
            return True
        if event.type() == QEvent.MouseButtonDblClick:
            # The label editor's first (a polygon closes on a double-click), else a zoom reset.
            if not self._edit(event, "double"):
                self.set_zoom(1.0)
            return True
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and not (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)):
                edges = self._box_edges_at(event)
                if edges:
                    self._box_grab = edges
                    if self._box_call("press", event):
                        self._apply_cursor()
                        return True
                    self._box_grab = None
            if self._is_pan(event):
                self._drag_from = self._cursor_xy(event)
                self._pan_origin = self._drag_from
                return True
            if event.button() == Qt.LeftButton and self._transform_call("press", event):
                self._transforming = True
                self._transform_from = self._cursor_xy(event)
                self._apply_cursor()
                return True
            if self._edit(event, "press"):
                self._editing = True
                return True
            handle = self._handle_at(event)
            if handle is not None:
                self._rotating = handle
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseMove:
            if self._transforming:
                self._transform_call("move", event)
                return True
            if self._box_grab is not None:
                self._box_call("move", event)
                return True
            if not event.buttons():
                transform_hover = self.transform_handler is not None and self._transform_call("can", event)
                if transform_hover != self._transform_hover:
                    self._transform_hover = transform_hover
                    self._apply_cursor()
                hover = self._box_edges_at(event)
                if hover is not None:
                    try:
                        hover = hover if self.box_handler("can", set(hover), None, event) else None
                    except Exception:  # noqa: BLE001
                        hover = None
                if hover != self._box_hover:
                    self._box_hover = hover
                    self._apply_cursor()
            if self._brush is not None:
                self._hover = self._cursor_xy(event)
            if self._drag_from is not None and event.buttons() & (Qt.LeftButton | Qt.MiddleButton):
                # A drag that began as a pan stays one (the Pan key may come up meanwhile).
                self._pan_by(event)
                return True
            if self._editing:
                self._edit(event, "move")
                if self._brush is not None:
                    self._repaint()
                return True
            if not event.buttons():
                self.hovered.emit(self._pixel_at(event))
                if self._brush is not None:
                    self._repaint()
                return False
            if self._rotating is not None:
                self.handleDragged.emit(int(self._rotating), self._screen_angle(event))
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseButtonRelease:
            if self._transforming:
                self._transforming = False
                self._transform_call("release", event)
                self._apply_cursor()
                origin, self._transform_from = self._transform_from, None
                x, y = self._cursor_xy(event)
                if origin is not None and abs(x - origin[0]) + abs(y - origin[1]) < 3.0:
                    self._emit_click(event)  # a click, not a drag: it moves the crosshair as usual
                return True
            if self._box_grab is not None:
                self._box_call("release", event)
                self._box_grab = None
                self._apply_cursor()
                return True
            origin, self._pan_origin = self._pan_origin, None
            self._drag_from = None
            self._rotating = None
            if origin is not None and event.button() == Qt.LeftButton:
                x, y = self._cursor_xy(event)
                if abs(x - origin[0]) + abs(y - origin[1]) < 4.0:
                    # A Shift+click, not a pan: the label editor may take it.
                    self._edit(event, "shift-click")
                return True
            if self._editing:
                self._editing = False
                self._edit(event, "release")
                return True
            return False
        if event.type() == QEvent.ShortcutOverride and self._transforming and event.key() == Qt.Key_Escape:
            event.accept()  # the views' own Esc shortcut waits: this Esc cancels the drag
            return True
        if event.type() == QEvent.KeyPress and self._transforming and event.key() == Qt.Key_Escape:
            # Esc mid-drag puts the layers back where the drag started.
            self._transforming = False
            self._transform_call("cancel", event)
            self._apply_cursor()
            return True
        if event.type() == QEvent.Leave:
            self.hovered.emit(None)
        if event.type() == QEvent.Leave and self._hover is not None:
            self._hover = None
            self._repaint()
        return False

    def _edit(self, event: Any, kind: str) -> bool:
        """Offer a mouse event to the label editor; True when it took it."""
        handler = self.edit_handler
        if handler is None:
            return False
        if kind in ("shift-click", "double") and event.button() != Qt.LeftButton:
            return False
        # The right button too: a right-drag erases with the brush, whatever the tool.
        if kind == "press" and event.button() not in (Qt.LeftButton, Qt.RightButton):
            return False
        try:
            return bool(handler(kind, self._pixel_at(event), event))
        except Exception:  # noqa: BLE001 (an editor fault must not wedge the view)
            return False

    @staticmethod
    def _cursor_xy(event: Any) -> tuple[float, float]:
        """Cursor position on the canvas, across the Qt spellings."""
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x()), float(pos.y())

    def _is_pan(self, event: Any) -> bool:
        """True for the pan gesture: middle-drag, shift with the left button, or the
        left button while the Labeling tab's Pan key is held (:attr:`pan_check`)."""
        buttons = event.buttons()
        if buttons & Qt.MiddleButton:
            return True
        if not buttons & Qt.LeftButton:
            return False
        if event.modifiers() & Qt.ShiftModifier:
            return True
        return self.pan_check is not None and bool(self.pan_check())

    def _pan_by(self, event: Any) -> None:
        """Drag the zoomed slice under the canvas."""
        geometry = self._geometry()
        if geometry is None or self._drag_from is None:
            return
        scaled_w, scaled_h, _ox, _oy = geometry
        x, y = self._cursor_xy(event)
        self._pan[0] = float(np.clip(self._pan[0] - (x - self._drag_from[0]) / max(scaled_w, 1), 0.0, 1.0))
        self._pan[1] = float(np.clip(self._pan[1] - (y - self._drag_from[1]) / max(scaled_h, 1), 0.0, 1.0))
        self._drag_from = (x, y)
        self._repaint()

    def _pixel_at(self, event: Any) -> tuple[int, int] | None:
        """The (row, column) slice pixel under the cursor, ``None`` off the slice."""
        geometry = self._geometry()
        if geometry is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        x, y = self._cursor_xy(event)
        px, py = x - off_x, y - off_y
        if not (0 <= px < scaled_w and 0 <= py < scaled_h):
            return None
        col = int(np.clip(px / scaled_w * w, 0, w - 1))
        row = int(np.clip(py / scaled_h * h, 0, h - 1))
        return row, col

    def _emit_click(self, event: Any) -> None:
        """Translate a click on the canvas into a crosshair position."""
        pixel = self._pixel_at(event)
        if pixel is not None:
            self.pixelPicked.emit(*pixel)
