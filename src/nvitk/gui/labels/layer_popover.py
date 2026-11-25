"""Per-layer pop-over: Napari's layer controls (and, for label maps, the labels) on demand.

Opened from the layer list (right-click ▸ *Controls…*) for the layer under the cursor, instead of
a layer-controls dock open all the time. It holds:

* a header with the layer's name and type, a render-preset menu (images and labels) and a
  reorient menu (3D layers; applies to every selected layer);
* for a label map (a Labels layer, or an Image that holds integer ids): the label vocabulary
  guesses as one-click chips and the label list (show / hide / colour per label);
* Napari's own controls for the layer (opacity, blending, colormap, contrast, gamma, rendering,
  brush and modes for labels), the very widget its dock shows.

One pop-over per viewer, rebuilt for each layer; pin it to keep it open while you work.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QMenu, QToolButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.widgets import Badge, ChipFilterBar, IconButton, Popover, Section, hint_label

_ATTR = "_nvitk_layer_popover"


class LayerPopoverContent(QWidget):
    """The pop-over's body for one layer."""

    def __init__(self, viewer: Any, layer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._layer = layer
        self._controls: Any | None = None
        self.setMinimumWidth(340)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        head = QHBoxLayout()
        head.setSpacing(design.SPACE_TIGHT)
        name = QLabel(str(getattr(layer, "name", "layer")))
        name.setStyleSheet(f"font-weight: bold; font-size: {design.FONT_M}px;")
        head.addWidget(name, 1)
        head.addWidget(Badge(type(layer).__name__, "blue"))
        kind = type(layer).__name__
        if kind in ("Image", "Labels"):
            head.addWidget(self._menu_button("droplet", "Render preset (applies to the selected layers)", self._fill_render))
        from nvitk.gui.core.orientation import layer_orientation_codes

        if layer_orientation_codes(layer) is not None:
            head.addWidget(self._menu_button("move", "Reorient the selected layers", self._fill_orient))
        root.addLayout(head)

        if self._is_label_map():
            root.addWidget(self._labels_section())
        root.addWidget(self._controls_section())

    # pieces
    def _menu_button(self, icon_name: str, tooltip: str, filler: Any) -> QToolButton:
        button = IconButton(icon_name, tooltip, icon_size=14)
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        menu.aboutToShow.connect(lambda m=menu: (m.clear(), filler(m)))
        button.setMenu(menu)
        return button

    def _selected(self) -> list[Any]:
        layers = self._viewer.layers
        chosen = [l for l in layers if l in layers.selection]
        if self._layer not in chosen:
            chosen = [self._layer]
        return chosen

    def _fill_render(self, menu: QMenu) -> None:
        from nvitk.gui.viz.render_presets import build_preset_menu

        build_preset_menu(menu, self._selected)

    def _fill_orient(self, menu: QMenu) -> None:
        from nvitk.gui.core.orientation import ORIENTATION_CODES

        def _apply(code: str) -> None:
            from nvitk.gui.tools.orient_quick import reorient_selected

            selection = self._viewer.layers.selection
            if self._layer not in selection:
                selection.active = self._layer
            reorient_selected(self._viewer, code)

        for code in ORIENTATION_CODES:
            action = menu.addAction(code)
            action.triggered.connect(lambda _=False, c=code: _apply(c))

    def _is_label_map(self) -> bool:
        if type(self._layer).__name__ == "Labels":
            return True
        try:
            from nvitk.gui.labels.visibility import is_label_like_layer

            return bool(is_label_like_layer(self._layer))
        except Exception:  # noqa: BLE001
            return False

    def _labels_section(self) -> QWidget:
        from nvitk.gui.labels.catalog import get_schema, layer_schema_key, rank_schemas, remember_layer_schema
        from nvitk.gui.labels.selector import LabelSelectorWidget

        section = Section("Labels", icon_name="tag", hue="green", pref_key="layer_popover.labels")
        selector = LabelSelectorWidget(compact=True)
        selector.set_viewer(self._viewer)
        current = layer_schema_key(self._layer)
        ranked = rank_schemas(self._layer, top=3)
        chips = []
        for key, score in ranked:
            schema = get_schema(key)
            if schema is not None:
                chips.append((key, schema.title, "green", f"{schema.group}: match {score:.2f}"))
        if chips:
            section.add(hint_label("Vocabulary: the current one, or a suggestion by the ids present"))
            bar = ChipFilterBar(chips, exclusive=True, all_label=None)
            if current in {c[0] for c in chips}:
                bar.set_selected({current}, emit=False)

            def _picked(keys: frozenset) -> None:
                if keys:
                    key = next(iter(keys))
                    remember_layer_schema(self._layer, key)
                    selector.set_schema_key(key)

            bar.changed.connect(_picked)
            section.add(bar)
        if current:
            selector.set_schema_key(current, refresh=False)
        selector.refresh_from_layer(self._layer)
        section.add(selector)
        return section

    def _controls_section(self) -> QWidget:
        section = Section("Layer controls", icon_name="sliders", hue="blue", pref_key="layer_popover.controls")
        try:
            from napari._qt.layer_controls.qt_layer_controls_container import create_qt_layer_controls

            self._controls = create_qt_layer_controls(self._layer)
            section.add(self._controls)
        except Exception as exc:  # noqa: BLE001
            section.add(hint_label(f"Napari's controls are unavailable for this layer: {exc}"))
        return section

    def close_controls(self) -> None:
        """Disconnect Napari's controls from the layer (they listen to its events)."""
        if self._controls is not None:
            try:
                self._controls.close()
            except Exception:  # noqa: BLE001
                pass
            self._controls = None


def show_layer_popover(viewer: Any, layer: Any, global_pos: QPoint | None = None) -> Popover:
    """Open the pop-over for *layer* at *global_pos* (the cursor, when omitted)."""
    from qtpy.QtGui import QCursor

    popover: Popover | None = getattr(viewer, _ATTR, None)
    if popover is None:
        popover = Popover(title="Layer", parent=viewer.window._qt_window)
        try:
            setattr(viewer, _ATTR, popover)
        except Exception:  # noqa: BLE001
            pass
    old = popover.content()
    if isinstance(old, LayerPopoverContent):
        old.close_controls()
    content = LayerPopoverContent(viewer, layer)
    design.apply_theme(content)
    popover.set_content(content)
    popover.set_title(str(getattr(layer, "name", "Layer")))
    if old is not None:
        old.deleteLater()
    if not popover.isVisible() or not popover.is_pinned():
        popover.show_at_point(global_pos if global_pos is not None else QCursor.pos())
    else:
        popover.adjustSize()
    return popover


def show_inspect_popover(viewer: Any, layer: Any, kind: str, global_pos: QPoint | None = None) -> Popover:
    """A quick pop-over with *layer*'s image properties (``kind="properties"``) or DICOM tags
    (``kind="dicom"``): the Inspect panels, for one layer, without leaving the canvas.

    Pin it to keep it open; it shows the layer it was opened on, whatever becomes active.
    """
    from qtpy.QtGui import QCursor

    if kind == "dicom":
        from nvitk.gui.panels.dicom_tags import DicomTagsPanel

        content = DicomTagsPanel()
        title = f"DICOM tags · {layer.name}"
    else:
        from nvitk.gui.panels.image_properties import ImagePropertiesPanel

        content = ImagePropertiesPanel()
        content.set_viewer(viewer)
        title = f"Image properties · {layer.name}"
    content.refresh_from_layer(layer)
    content.setMinimumSize(420, 360)
    design.apply_theme(content)
    popover = Popover(content, title=title, parent=viewer.window._qt_window)
    popover.setAttribute(Qt.WA_DeleteOnClose, True)
    popover.show_at_point(global_pos if global_pos is not None else QCursor.pos())
    return popover


def hide_layer_controls_dock(viewer: Any, hidden: bool = True) -> None:
    """Hide (or show) Napari's permanent layer-controls dock; the pop-over replaces it."""
    try:
        dock = viewer.window._qt_viewer.dockLayerControls
        dock.setVisible(not hidden)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["LayerPopoverContent", "hide_layer_controls_dock", "show_inspect_popover", "show_layer_popover"]
