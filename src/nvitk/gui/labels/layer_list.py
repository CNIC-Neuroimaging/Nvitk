"""Label layers in Napari's layer list unfold into their labels.

Napari paints its layer rows with a delegate, so there is no widget per row to add
to. This subclass of Napari's own delegate draws a small ▾ chip on every label
layer's row; clicking it makes the row taller and lists the layer's labels in the
extra space, below the layer item itself. Each label line shows:

    [✓]  ●  Left ICA (3)

a show/hide box, the label's colour as drawn on the canvas, and its name in the
layer's vocabulary — the one picked for it in a label picker, else the one
guessed from its name, path and contents (:func:`~nvitk.gui.labels.catalog.layer_schema_key`).

* click a line — show or hide that label; Alt+click — show only that label;
* click the colour dot — change the label's colour (Labels layers);
* **All** / **None** in the header; **Panel** opens the full Labels panel on the
  layer (filter box, vocabulary, colormaps);
* a long list scrolls with the wheel inside it.

Everything is painted, nothing is a child widget: rows move, scroll and reorder,
and painted content cannot be left behind at a stale position. Every picker shares
the layer's live filter (:func:`~nvitk.gui.labels.selector.label_filter_hub`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QApplication, QColorDialog, QStyle, QStyleOptionViewItem, QToolTip

from nvitk.gui.labels.catalog import get_schema, layer_schema_key
from nvitk.gui.labels.panel import _known_label_like
from nvitk.gui.labels.selector import label_filter_hub, set_layer_visible_ids
from nvitk.gui.labels.visibility import (
    get_label_color,
    is_label_like_layer,
    layer_label_ids,
    layer_in_viewer,
    set_label_color,
    stored_visible_ids,
    supports_per_label_color,
)

_CHIP_W = 22
_CHIP_H = 18
_CHIP_GAP = 4
#: Geometry of the unfolded part, below the layer item.
_HEADER_H = 22
_LINE_H = 20
_FOOTER_H = 18
_PAD = 6
_INDENT = 14
#: Lines shown at once; a longer list scrolls with the wheel.
_MAX_LINES = 12
_CHIP_TOOLTIP = "Show or hide this layer's labels"


def _rgba_qcolor(rgba: Any) -> QColor:
    """A ``QColor`` from float RGB(A) in ``[0, 1]``."""
    vals = np.clip(np.asarray(rgba, dtype=float).reshape(-1), 0.0, 1.0)
    alpha = float(vals[3]) if vals.size > 3 else 1.0
    return QColor.fromRgbF(float(vals[0]), float(vals[1]), float(vals[2]), alpha)


def _crisp(rect: QRect) -> QRectF:
    """*rect* half a pixel in, so a 1px stroke lands on whole pixels."""
    return QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)


def _label_colour(layer: Any, lid: int) -> QColor:
    """*lid*'s colour as the canvas draws it (an Image mask: through its colormap)."""
    try:
        if supports_per_label_color(layer):
            return _rgba_qcolor(get_label_color(layer, lid))
        lo, hi = (float(v) for v in layer.contrast_limits)
        t = 0.0 if hi <= lo else min(max((lid - lo) / (hi - lo), 0.0), 1.0)
        return _rgba_qcolor(layer.colormap.map([t])[0])
    except Exception:  # noqa: BLE001 — a colour is decoration
        return QColor(200, 200, 200)


class _Expansion:
    """Per-layer state of an unfolded row."""

    __slots__ = ("offset",)

    def __init__(self) -> None:
        self.offset = 0


def _delegate_class() -> type:
    """Napari's layer delegate, subclassed. Built lazily: importing Napari's Qt
    internals at module import would tie every nvitk import to them."""
    from napari._qt.containers._base_item_model import ItemRole
    from napari._qt.containers._layer_delegate import LayerDelegate

    class LabelListDelegate(LayerDelegate):
        """Napari's layer delegate; label layers unfold into their labels."""

        def __init__(self, viewer: Any, parent: Any = None) -> None:
            super().__init__(parent)
            self._viewer = viewer
            self._expanded: dict[int, _Expansion] = {}
            #: Folder state of the layer list (:mod:`nvitk.gui.core.layer_folders`),
            #: once folders are installed: headers above rows, members indented.
            self._folders: Any | None = None
            label_filter_hub().changed.connect(lambda _layer: self._repaint())

        # -- folders ----------------------------------------------------------
        #
        # A row in a folder is a header band (one per folder that starts on it)
        # over the layer item, indented by its depth. Everything the label chips
        # and Napari's own delegate do happens inside the item part: they are
        # handed an option whose rect *is* that part.

        def set_folders(self, folders: Any) -> None:
            self._folders = folders
            folders.changed.connect(self._repaint)

        def _entry(self, index: Any) -> Any | None:
            if self._folders is None:
                return None
            return self._folders.entry(self._layer(index))

        @staticmethod
        def _plain(entry: Any) -> bool:
            return entry is None or (not entry.headers and entry.depth == 0 and entry.item_visible and not entry.hidden)

        def inner_rect(self, rect: QRect, index: Any) -> QRect:
            """The part of a row's *rect* the layer item is drawn in."""
            from nvitk.gui.core.layer_folders import item_rect

            entry = self._entry(index)
            return rect if self._plain(entry) else item_rect(rect, entry)

        def _inner(self, option: QStyleOptionViewItem, index: Any) -> QStyleOptionViewItem:
            entry = self._entry(index)
            if self._plain(entry):
                return option
            inner = QStyleOptionViewItem(option)
            inner.rect = self.inner_rect(option.rect, index)
            return inner

        def sizeHint(self, option: QStyleOptionViewItem, index: Any) -> QSize:
            size = self._item_size_hint(option, index)
            entry = self._entry(index)
            if self._plain(entry):
                return size
            if entry.hidden:
                return QSize(size.width(), 0)
            size.setHeight(entry.band + (size.height() if entry.item_visible else 0))
            return size

        def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: Any) -> None:
            entry = self._entry(index)
            if self._plain(entry):
                self._paint_item(painter, option, index)
                return
            if entry.hidden:
                return
            from nvitk.gui.core.layer_folders import header_rects, paint_guides, paint_header

            folders = self._folders
            for path, rect in header_rects(option.rect, entry):
                paint_header(
                    painter, option, rect, path,
                    collapsed=path in folders.collapsed,
                    count=len(folders.members(path)),
                    visibility=folders.visibility(path),
                    selected=folders.is_selected(path),
                )
            if entry.item_visible:
                paint_guides(painter, option, option.rect, entry)
                self._paint_item(painter, self._inner(option, index), index)

        def editorEvent(self, event: Any, model: Any, option: QStyleOptionViewItem, index: Any) -> bool:
            entry = self._entry(index)
            if not self._plain(entry) and hasattr(event, "pos"):
                # Header clicks are taken by the folder mouse filter before the
                # view sees them; anything that still lands in the band is not
                # the layer item's.
                if entry.hidden or event.pos().y() < option.rect.top() + entry.band or not entry.item_visible:
                    return True
            return self._item_editor_event(event, model, self._inner(option, index), index)

        def helpEvent(self, event: Any, view: Any, option: QStyleOptionViewItem, index: Any) -> bool:
            entry = self._entry(index)
            if index.isValid() and not self._plain(entry) and entry.headers:
                from nvitk.gui.core.layer_folders import folder_name, header_parts, header_rects

                pos = event.pos()
                for path, rect in header_rects(option.rect, entry):
                    if rect.top() <= pos.y() <= rect.bottom():
                        parts = header_parts(rect)
                        if parts["arrow"].contains(pos):
                            tip = "Fold / unfold the folder"
                        elif parts["box"].contains(pos):
                            tip = "Show or hide every layer in the folder"
                        else:
                            n = len(self._folders.members(path))
                            tip = (
                                f"Folder “{folder_name(path)}” — {n} layer(s). Click: select them · "
                                "double-click: rename · right-click: folder menu"
                            )
                        QToolTip.showText(event.globalPos(), tip, view)
                        return True
            return self._item_help_event(event, view, self._inner(option, index), index)

        def updateEditorGeometry(self, editor: Any, option: QStyleOptionViewItem, index: Any) -> None:
            super().updateEditorGeometry(editor, self._inner(option, index), index)

        # -- state ------------------------------------------------------------

        @staticmethod
        def _layer(index: Any) -> Any:
            return index.data(ItemRole)

        def _has_chip(self, index: Any) -> bool:
            return _known_label_like(self._layer(index))

        def is_expanded(self, layer: Any) -> bool:
            return layer is not None and id(layer) in self._expanded

        def _ids(self, layer: Any) -> list[int]:
            try:
                return layer_label_ids(layer)
            except Exception:  # noqa: BLE001 — a layer mid-removal
                return []

        def _expansion_height(self, layer: Any) -> int:
            n = len(self._ids(layer))
            shown = max(min(n, _MAX_LINES), 1)
            footer = _FOOTER_H if n > _MAX_LINES else 0
            return _HEADER_H + shown * _LINE_H + footer + 2 * _PAD

        def _item_size_hint(self, option: QStyleOptionViewItem, index: Any) -> QSize:
            size = super().sizeHint(option, index)
            layer = self._layer(index)
            if self.is_expanded(layer):
                size.setHeight(size.height() + self._expansion_height(layer))
            return size

        def _base_height(self, index: Any) -> int:
            hint = index.data(Qt.ItemDataRole.SizeHintRole)
            return int(hint.height()) if hint is not None else 34

        # -- geometry ---------------------------------------------------------

        def _chip_rect(self, rect: QRect, index: Any) -> QRect:
            base = self._base_height(index)
            return QRect(
                rect.right() - _CHIP_W - _CHIP_GAP + 1,
                rect.top() + (base - _CHIP_H) // 2,
                _CHIP_W,
                _CHIP_H,
            )

        def _panel_rect(self, rect: QRect, index: Any) -> QRect:
            """The unfolded area under the layer item."""
            top = rect.top() + self._base_height(index)
            return QRect(rect.left() + 4, top, rect.width() - 8, rect.bottom() - top - 2)

        def _layout(self, rect: QRect, index: Any) -> dict[str, Any]:
            """Hit areas of the unfolded part, shared by painting and clicking."""
            layer = self._layer(index)
            panel = self._panel_rect(rect, index)
            view = self.parent()
            fm = QFontMetrics(view.font() if view is not None else QApplication.font())
            header = QRect(panel.left() + _PAD, panel.top() + _PAD, panel.width() - 2 * _PAD, _HEADER_H)
            buttons: dict[str, QRect] = {}
            x = header.right()
            for name in ("Panel", "None", "All"):
                w = fm.horizontalAdvance(name) + 12
                buttons[name] = QRect(x - w + 1, header.top() + 2, w, _HEADER_H - 4)
                x -= w + 4
            title = QRect(header.left(), header.top(), max(x - header.left(), 0), _HEADER_H)
            ids = self._ids(layer)
            state = self._expanded.get(id(layer)) or _Expansion()
            state.offset = max(0, min(state.offset, max(len(ids) - _MAX_LINES, 0)))
            lines = []
            y = header.bottom() + 1
            for lid in ids[state.offset:state.offset + _MAX_LINES]:
                line = QRect(panel.left() + _PAD, y, panel.width() - 2 * _PAD, _LINE_H)
                check = QRect(line.left() + _INDENT - 10, line.top() + 4, 12, 12)
                swatch = QRect(check.right() + 8, line.top() + 4, 12, 12)
                text = QRect(swatch.right() + 8, line.top(), line.right() - swatch.right() - 8, _LINE_H)
                lines.append((lid, line, check, swatch, text))
                y += _LINE_H
            footer = QRect(panel.left() + _PAD, y, panel.width() - 2 * _PAD, _FOOTER_H) if len(ids) > _MAX_LINES else None
            return {
                "panel": panel, "title": title, "buttons": buttons, "lines": lines,
                "footer": footer, "ids": ids, "offset": state.offset,
            }

        # -- painting ---------------------------------------------------------

        def _paint_item(self, painter: QPainter, option: QStyleOptionViewItem, index: Any) -> None:
            if not self._has_chip(index):
                super().paint(painter, option, index)
                return
            layer = self._layer(index)
            expanded = self.is_expanded(layer)
            # The row background across the full row first, so the chip and the
            # unfolded labels sit on the same selection/hover colour as the item.
            full = QStyleOptionViewItem(option)
            self.initStyleOption(full, index)
            full.text = ""
            full.icon = QIcon()
            full.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            widget = option.widget
            style = widget.style() if widget is not None else QApplication.style()
            style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, full, painter, widget)

            # Napari paints the item itself in the top part, narrowed by the chip.
            top = QStyleOptionViewItem(option)
            top.rect = QRect(
                option.rect.left(), option.rect.top(),
                option.rect.width() - (_CHIP_W + _CHIP_GAP), self._base_height(index),
            )
            super().paint(painter, top, index)
            self._paint_chip(painter, option, index, expanded)
            if expanded:
                self._paint_labels(painter, option, index, layer)

        def _paint_chip(self, painter: QPainter, option: QStyleOptionViewItem, index: Any, expanded: bool) -> None:
            rect = self._chip_rect(option.rect, index)
            pal = option.palette
            text = pal.color(pal.ColorRole.Text)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if expanded:
                painter.setBrush(pal.color(pal.ColorRole.Highlight))
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                border = QColor(text)
                border.setAlphaF(0.45)
                painter.setPen(QPen(border, 1))
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 4, 4)
            painter.setPen(pal.color(pal.ColorRole.HighlightedText) if expanded else text)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "▴" if expanded else "▾")
            painter.restore()

        def _paint_labels(self, painter: QPainter, option: QStyleOptionViewItem, index: Any, layer: Any) -> None:
            geo = self._layout(option.rect, index)
            pal = option.palette
            text = pal.color(pal.ColorRole.Text)
            muted = QColor(text)
            muted.setAlphaF(0.6)
            accent = pal.color(pal.ColorRole.Highlight)
            fm = painter.fontMetrics()
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            bg = QColor(pal.color(pal.ColorRole.Base))
            bg.setAlphaF(0.85)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(geo["panel"], 4, 4)

            key = layer_schema_key(layer)
            schema = get_schema(key) if key else None
            ids = geo["ids"]
            painter.setPen(muted)
            title = f"{len(ids)} label(s)" + (f" — {schema.title}" if schema else " — no names known")
            painter.drawText(
                geo["title"], int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                fm.elidedText(title, Qt.TextElideMode.ElideRight, geo["title"].width()),
            )
            for name, rect in geo["buttons"].items():
                painter.setPen(QPen(muted, 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 3, 3)
                painter.setPen(text)
                painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), name)

            visible = stored_visible_ids(layer)
            for lid, line, check, swatch, text_rect in geo["lines"]:
                shown = visible is None or lid in visible
                # Show/hide box.
                painter.setPen(QPen(accent if shown else muted, 1.2))
                painter.setBrush(accent if shown else Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(_crisp(check), 2.5, 2.5)
                if shown:
                    path = QPainterPath()
                    path.moveTo(QPointF(check.left() + 2.5, check.center().y() + 0.5))
                    path.lineTo(QPointF(check.left() + 5.0, check.bottom() - 2.5))
                    path.lineTo(QPointF(check.right() - 2.0, check.top() + 3.0))
                    painter.setPen(QPen(pal.color(pal.ColorRole.HighlightedText), 1.6))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawPath(path)
                # Colour, dimmed while hidden.
                colour = _label_colour(layer, lid)
                if not shown:
                    colour.setAlphaF(0.35)
                painter.setPen(QPen(muted, 1))
                painter.setBrush(colour)
                painter.drawEllipse(swatch)
                # Name.
                name = schema.name_for(lid) if schema else None
                label = f"{name}  ({lid})" if name else f"Label {lid}"
                painter.setPen(text if shown else muted)
                painter.drawText(
                    text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    fm.elidedText(label, Qt.TextElideMode.ElideRight, text_rect.width()),
                )
            if geo["footer"] is not None:
                first = geo["offset"] + 1
                last = min(geo["offset"] + _MAX_LINES, len(ids))
                painter.setPen(muted)
                painter.drawText(
                    geo["footer"], int(Qt.AlignmentFlag.AlignCenter),
                    f"{first}–{last} of {len(ids)} · scroll for more",
                )
            painter.restore()

        # -- interaction ------------------------------------------------------

        def _item_editor_event(self, event: Any, model: Any, option: QStyleOptionViewItem, index: Any) -> bool:
            if not self._has_chip(index) or not hasattr(event, "pos"):
                return super().editorEvent(event, model, option, index)
            kind = event.type()
            mouse = (event.MouseButtonPress, event.MouseButtonRelease, event.MouseButtonDblClick)
            if kind not in mouse:
                return super().editorEvent(event, model, option, index)
            pos = event.pos()
            layer = self._layer(index)
            on_chip = self._chip_rect(option.rect, index).contains(pos)
            in_panel = self.is_expanded(layer) and self._panel_rect(option.rect, index).contains(pos)
            if not (on_chip or in_panel):
                return super().editorEvent(event, model, option, index)
            # Clicks here belong to the chip or the label list: not a selection
            # change, a drag or a rename.
            if kind == event.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if on_chip:
                    self.toggle(layer, index)
                else:
                    alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    self._click_panel(layer, option.rect, index, pos, alt=alt)
            return True

        def _click_panel(self, layer: Any, rect: QRect, index: Any, pos: QPoint, *, alt: bool) -> None:
            geo = self._layout(rect, index)
            ids = geo["ids"]
            for name, button in geo["buttons"].items():
                if button.contains(pos):
                    if name == "All":
                        set_layer_visible_ids(layer, ids, self._viewer, present=ids)
                    elif name == "None":
                        set_layer_visible_ids(layer, [], self._viewer, present=ids)
                    else:
                        self._open_panel(layer)
                    return
            for lid, line, _check, swatch, _text in geo["lines"]:
                if not line.contains(pos):
                    continue
                if swatch.contains(pos) and supports_per_label_color(layer):
                    self._edit_colour(layer, lid)
                    return
                visible = stored_visible_ids(layer)
                current = set(ids) if visible is None else set(visible)
                if alt:
                    # Like Napari's Alt+click on a layer's eye: this one alone,
                    # and again to bring the others back.
                    new = set(ids) if current == {lid} else {lid}
                else:
                    new = current ^ {lid}
                set_layer_visible_ids(layer, sorted(new), self._viewer, present=ids)
                return

        def _edit_colour(self, layer: Any, lid: int) -> None:
            current = _label_colour(layer, lid)
            try:
                options = QColorDialog.ColorDialogOption.ShowAlphaChannel
            except AttributeError:  # pragma: no cover — older Qt bindings
                options = QColorDialog.ShowAlphaChannel  # type: ignore[attr-defined]
            chosen = QColorDialog.getColor(current, self.parent(), f"Label {lid} colour", options)
            if not chosen.isValid():
                return
            rgba = [chosen.redF(), chosen.greenF(), chosen.blueF(), chosen.alphaF()]
            visible = stored_visible_ids(layer)
            set_label_color(layer, lid, rgba, selected_ids=None if visible is None else list(visible))
            label_filter_hub().changed.emit(layer)

        def _open_panel(self, layer: Any) -> None:
            """The full Labels panel, on this layer."""
            try:
                self._viewer.layers.selection.active = layer
            except Exception:  # noqa: BLE001
                pass
            manager = getattr(self._viewer, "_nvitk_panel_manager", None)
            if manager is not None:
                manager.show_labels()

        def _item_help_event(self, event: Any, view: Any, option: QStyleOptionViewItem, index: Any) -> bool:
            if index.isValid() and self._has_chip(index):
                pos = event.pos()
                tip = None
                layer = self._layer(index)
                if self._chip_rect(option.rect, index).contains(pos):
                    tip = _CHIP_TOOLTIP
                elif self.is_expanded(layer):
                    geo = self._layout(option.rect, index)
                    for name, button in geo["buttons"].items():
                        if button.contains(pos):
                            tip = {
                                "All": "Show every label",
                                "None": "Hide every label",
                                "Panel": "Open the Labels panel: filter, vocabulary, colormaps",
                            }[name]
                    for lid, line, _check, swatch, _text in geo["lines"]:
                        if line.contains(pos):
                            key = layer_schema_key(layer)
                            schema = get_schema(key) if key else None
                            name = (schema.name_for(lid) if schema else None) or f"Label {lid}"
                            tip = (
                                f"{name} ({lid}) — click to change its colour"
                                if swatch.contains(pos) and supports_per_label_color(layer)
                                else f"{name} ({lid}) — click to show/hide, Alt+click to show only this"
                            )
                if tip:
                    QToolTip.showText(event.globalPos(), tip, view)
                    return True
            return super().helpEvent(event, view, option, index)

        def toggle(self, layer: Any, index: Any = None) -> bool:
            """Unfold *layer*'s row into its labels, or fold it back; True when unfolded."""
            if layer is None:
                return False
            if id(layer) in self._expanded:
                self._expanded.pop(id(layer), None)
                expanded = False
            else:
                self._expanded[id(layer)] = _Expansion()
                expanded = True
            self._relayout(layer, index)
            return expanded

        def _relayout(self, layer: Any, index: Any = None) -> None:
            """The row changed height: have the view lay the list out again."""
            if index is None:
                index = self._index_of(layer)
            if index is not None and index.isValid():
                self.sizeHintChanged.emit(index)
            self._repaint()

        def _index_of(self, layer: Any) -> Any:
            view = self.parent()
            model = view.model() if view is not None else None
            if model is None:
                return None
            for row in range(model.rowCount()):
                idx = model.index(row, 0)
                if self._layer(idx) is layer:
                    return idx
            return None

        def scroll(self, layer: Any, steps: int) -> bool:
            """Scroll *layer*'s label list by *steps* lines; True when it moved."""
            state = self._expanded.get(id(layer))
            if state is None:
                return False
            n = len(self._ids(layer))
            new = max(0, min(state.offset + steps, max(n - _MAX_LINES, 0)))
            if new == state.offset:
                return False
            state.offset = new
            self._repaint()
            return True

        def _repaint(self) -> None:
            try:
                view = self.parent()
                if view is not None:
                    view.viewport().update()
            except RuntimeError:
                pass

        def prune(self, _event: Any = None) -> None:
            """Forget unfolded rows whose layer has left the viewer."""
            alive = {id(layer) for layer in self._viewer.layers}
            for key in [k for k in self._expanded if k not in alive]:
                self._expanded.pop(key, None)

    return LabelListDelegate


class _WheelInLabels(QObject):
    """Scrolls an unfolded label list under the mouse instead of the layer list."""

    def __init__(self, view: Any, delegate: Any) -> None:
        super().__init__(view)
        self._view = view
        self._delegate = delegate

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        index = self._view.indexAt(pos)
        if not index.isValid():
            return False
        layer = self._delegate._layer(index)
        if not self._delegate.is_expanded(layer):
            return False
        rect = self._delegate.inner_rect(self._view.visualRect(index), index)
        if not self._delegate._panel_rect(rect, index).contains(pos):
            return False
        delta = event.angleDelta().y()
        if not delta:
            return True
        # Three lines a notch; at least one for a touchpad's small deltas.
        steps = -int(delta / 40) or (-1 if delta > 0 else 1)
        # A list already at its end hands the wheel back to the layer list.
        return self._delegate.scroll(layer, steps)


def install_layer_list_label_buttons(viewer: Any) -> Any | None:
    """Swap Napari's layer-list delegate for one whose label rows unfold.

    Returns the delegate, or ``None`` when the layer list could not be reached
    (headless, or a Napari whose internals moved).
    """
    try:
        view = viewer.window._qt_viewer.layers
        delegate = _delegate_class()(viewer, view)
    except Exception:  # noqa: BLE001 — the layer list keeps Napari's delegate
        return None
    view.setItemDelegate(delegate)
    delegate.loading_frame_changed.connect(view.viewport().update)
    wheel = _WheelInLabels(view, delegate)
    view.viewport().installEventFilter(wheel)
    delegate._wheel_filter = wheel
    viewer.layers.events.removed.connect(delegate.prune)

    def _classify(layer: Any) -> None:
        """Decide once, off the insert path, whether an Image layer is a mask, so
        its row gets a chip. Labels layers need no test."""
        if layer is None or type(layer).__name__ != "Image":
            return

        def _run() -> None:
            try:
                if is_label_like_layer(layer):
                    view.viewport().update()
            except Exception:  # noqa: BLE001 — a layer mid-removal
                pass

        QTimer.singleShot(0, _run)

    def _watch(layer: Any) -> None:
        """Repaint (and re-measure, when the ids change) an unfolded row on edits."""
        events = getattr(layer, "events", None)
        if events is None:
            return

        def _on_data(_event: Any = None) -> None:
            if delegate.is_expanded(layer) and layer_in_viewer(layer, viewer):
                delegate._relayout(layer)

        def _on_look(_event: Any = None) -> None:
            if delegate.is_expanded(layer):
                delegate._repaint()

        for name, slot in (("data", _on_data), ("colormap", _on_look), ("contrast_limits", _on_look)):
            emitter = getattr(events, name, None)
            if emitter is not None:
                emitter.connect(slot)

    def _on_inserted(event: Any) -> None:
        layer = getattr(event, "value", None)
        _classify(layer)
        _watch(layer)

    viewer.layers.events.inserted.connect(_on_inserted)
    for layer in list(viewer.layers):
        _classify(layer)
        _watch(layer)
    viewer._nvitk_label_chip_delegate = delegate
    return delegate


__all__ = ["install_layer_list_label_buttons"]
