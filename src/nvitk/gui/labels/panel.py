"""Label selection outside the Imaging tab: a dock of its own and per-layer popups.

The Imaging tab only shows its picker for tools that consume label ids. These views
are always reachable: the **Labels** dock follows the active label layer (or one
picked from its list), and each label layer's row in the layer list unfolds into
its labels (:mod:`nvitk.gui.labels.layer_list`). Every picker shares the layer's
live filter through
:func:`nvitk.gui.labels.selector.label_filter_hub`, so they never disagree.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_MUTED, SPACE, SPACE_TIGHT
from nvitk.gui.labels.catalog import layer_schema_key
from nvitk.gui.labels.selector import LabelSelectorWidget, apply_selection_to_layer
from nvitk.gui.labels.visibility import is_label_like_layer, layer_in_viewer

#: Dock object name, the key the saved layout knows it by.
LABELS_DOCK_NAME = "nvitk:labels"


def _known_label_like(layer: Any) -> bool:
    """Label-like without computing anything new: Labels layers and Image masks
    already classified. Cheap enough for a paint loop or a combo refresh."""
    if layer is None:
        return False
    if type(layer).__name__ == "Labels":
        return True
    return getattr(layer, "_nvitk_label_like", None) is True


class _BoundLabelSelector(QWidget):
    """A :class:`LabelSelectorWidget` that filters its layer live as boxes are ticked."""

    def __init__(self, viewer: Any, *, compact: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self.selector = LabelSelectorWidget(compact=compact)
        self.selector.set_viewer(viewer)
        self.selector.set_expanded(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.selector)
        # Parented, so a pending apply dies with the panel instead of firing at
        # deleted widgets.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(120)
        self._apply_timer.timeout.connect(self._apply)
        self.selector.selection_changed.connect(self._apply_timer.start)
        self.selector._btn_refresh.clicked.connect(lambda: self.bind(self.layer(), force=True))

    def layer(self) -> Any | None:
        """The layer the picker filters (after any Image → Labels promotion)."""
        return self.selector.current_layer()

    def bind(self, layer: Any | None, *, force: bool = False) -> None:
        """Show *layer*'s labels, named with its own vocabulary (picked or guessed)."""
        if layer is self.layer() and not force:
            return
        if layer is not None:
            self.selector.set_schema_key(layer_schema_key(layer) or "generic", refresh=False)
        self.selector.refresh_from_layer(layer)

    def _apply(self) -> None:
        """Push the ticked ids to the layer."""
        layer = self.layer()
        if layer is None or not layer_in_viewer(layer, self._viewer):
            return
        apply_selection_to_layer(self.selector, self._viewer)


class LabelsPanel(QWidget):
    """The **Labels** dock: pick a label layer and choose which of its labels show."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._bound = _BoundLabelSelector(viewer, compact=False, parent=self)

        self._layer_combo = QComboBox()
        self._layer_combo.setToolTip("The label layer whose labels are listed below.")
        self._layer_combo.setMinimumContentsLength(12)
        self._follow = QCheckBox("Follow active layer")
        self._follow.setChecked(True)
        self._follow.setToolTip(
            "Switch to whichever label layer you select in the layer list. "
            "Selecting an image keeps the last label layer here."
        )
        self._empty = QLabel(
            "No label layer open. Open a segmentation, or select a layer and press "
            "Refresh if it is a mask stored as an image."
        )
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {COLOR_MUTED};")

        top = QHBoxLayout()
        top.setSpacing(SPACE_TIGHT)
        top.addWidget(QLabel("Layer:"))
        top.addWidget(self._layer_combo, stretch=1)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE, SPACE_TIGHT, SPACE)
        root.setSpacing(SPACE)
        root.addLayout(top)
        root.addWidget(self._follow)
        root.addWidget(self._empty)
        root.addWidget(self._bound, stretch=1)

        self._layer_combo.activated.connect(self._on_combo_activated)
        self._follow.toggled.connect(lambda on: on and self._follow_active())

        # Debounced: a burst of inserts (a study opening) rebuilds the list once.
        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.setInterval(0)
        self._sync_timer.timeout.connect(self._sync)
        layers = viewer.layers
        layers.events.inserted.connect(self._schedule_sync)
        layers.events.removed.connect(self._schedule_sync)
        layers.events.moved.connect(self._schedule_sync)
        layers.selection.events.active.connect(self._schedule_sync)
        self._stale = True

    def showEvent(self, event: Any) -> None:
        """Catch up on whatever changed while the panel was out of sight."""
        super().showEvent(event)
        if self._stale:
            self._sync()

    def _schedule_sync(self, _event: Any = None) -> None:
        """Rebuild the layer list and binding on the next event-loop turn."""
        try:
            self._sync_timer.start()
        except RuntimeError:  # panel already destroyed
            pass

    def _label_layers(self) -> list[Any]:
        """Label-like layers, top of the layer list first."""
        out = []
        for layer in reversed(list(self._viewer.layers)):
            if _known_label_like(layer) or is_label_like_layer(layer):
                out.append(layer)
        return out

    def _sync(self) -> None:
        """Refresh the layer combo and follow the active layer when asked to.

        A tabbed-away panel only notes that it is out of date: rebuilding a
        hundred checkbox rows on every click in the layer list is latency nobody
        sees the result of.
        """
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        layers = self._label_layers()
        current = self._bound.layer()
        if current is not None and not layer_in_viewer(current, self._viewer):
            current = None
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in layers:
            self._layer_combo.addItem(str(layer.name), id(layer))
        self._layer_combo.blockSignals(False)
        self._empty.setVisible(not layers)
        self._bound.setVisible(bool(layers))
        if self._follow.isChecked():
            self._follow_active(layers)
        elif current is None:
            self._bind(layers[0] if layers else None)
        else:
            self._select_in_combo(current)

    def _follow_active(self, layers: list[Any] | None = None) -> None:
        """Bind to the active layer when it is a label layer, else keep the current one."""
        layers = self._label_layers() if layers is None else layers
        active = self._viewer.layers.selection.active
        if active is not None and any(active is layer for layer in layers):
            self._bind(active)
            return
        current = self._bound.layer()
        if current is not None and layer_in_viewer(current, self._viewer):
            self._select_in_combo(current)
            return
        self._bind(layers[0] if layers else None)

    def _bind(self, layer: Any | None) -> None:
        """Show *layer* in the picker and in the combo."""
        self._bound.bind(layer)
        live = self._bound.layer()
        if live is not None and live is not layer:
            # Promoted from an Image mask: the combo lists the new layer.
            self._schedule_sync()
        self._select_in_combo(live)

    def _select_in_combo(self, layer: Any | None) -> None:
        """Point the combo at *layer* without re-binding."""
        idx = self._layer_combo.findData(id(layer)) if layer is not None else -1
        self._layer_combo.blockSignals(True)
        self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)

    def _on_combo_activated(self, index: int) -> None:
        """A layer picked by hand stops the panel following the selection."""
        key = self._layer_combo.itemData(index)
        for layer in self._viewer.layers:
            if id(layer) == key:
                self._follow.setChecked(False)
                self._bind(layer)
                return

    def focus_filter(self) -> None:
        """Put the cursor in the label filter, ready to type."""
        self._bound.selector._filter.setFocus(Qt.ShortcutFocusReason)
        self._bound.selector._filter.selectAll()


def build_labels_dock(viewer: Any) -> Any:
    """Create the **Labels** dock (tabbed with the other nvitk panels by the caller)."""
    from nvitk.gui.core.workspace import make_panel_dock

    panel = LabelsPanel(viewer)
    dock = make_panel_dock(viewer, panel, object_name=LABELS_DOCK_NAME, title="Labels")
    dock._nvitk_labels_panel = panel
    return dock


__all__ = ["LABELS_DOCK_NAME", "LabelsPanel", "build_labels_dock"]
