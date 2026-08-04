"""Checkbox list of labels with optional pipeline / model name mapping."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.labels.catalog import (
    all_schemas,
    get_schema,
    guess_schema_from_layer,
    schema_keys,
)
from nvitk.gui.labels.visibility import (
    ensure_labels_layer,
    get_label_color,
    is_label_like_layer,
    label_source_data,
    set_label_color,
    supports_per_label_color,
    unique_layer_labels,
)

LABEL_SELECTOR_SCROLL_MIN = 80


def _rgba_to_qcolor(rgba: np.ndarray) -> QColor:
    """Convert a 3- or 4-channel float RGBA array in ``[0, 1]`` to a Qt ``QColor``."""
    arr = np.asarray(rgba, dtype=float).reshape(-1)
    r = int(np.clip(arr[0], 0.0, 1.0) * 255)
    g = int(np.clip(arr[1], 0.0, 1.0) * 255)
    b = int(np.clip(arr[2], 0.0, 1.0) * 255)
    a = int(np.clip(arr[3] if arr.size > 3 else 1.0, 0.0, 1.0) * 255)
    return QColor(r, g, b, a)


def _qcolor_to_rgba(color: QColor) -> np.ndarray:
    """Convert a Qt ``QColor`` to a 4-channel float32 RGBA array in ``[0, 1]``."""
    return np.array(
        [
            color.red() / 255.0,
            color.green() / 255.0,
            color.blue() / 255.0,
            color.alpha() / 255.0,
        ],
        dtype=np.float32,
    )


def _swatch_stylesheet(rgba: np.ndarray) -> str:
    """Qt stylesheet giving a ``QToolButton`` a solid background swatch of color *rgba*."""
    c = _rgba_to_qcolor(rgba)
    return (
        f"QToolButton {{ background-color: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()}); "
        f"border: 1px solid #666; border-radius: 3px; }}"
    )


class LabelSelectorWidget(QGroupBox):
    """Select label ids using an optional named vocabulary (eICAB, QVTpy, TS, …)."""

    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the schema picker, All/None/Refresh buttons, and scrollable checkbox list."""
        super().__init__("Label selection", parent)
        self._checks: list[QCheckBox] = []
        self._color_buttons: dict[int, QToolButton] = {}
        self._schema_key = "generic"
        self._hint = QLabel("Choose a label mapping, then select labels below.")
        self._hint.setWordWrap(True)

        schema_row = QHBoxLayout()
        schema_row.addWidget(QLabel("Mapping:"))
        self._schema_combo = QComboBox()
        self._schema_combo.setMinimumWidth(180)
        for key in schema_keys():
            sch = all_schemas()[key]
            self._schema_combo.addItem(sch.title, key)
        schema_row.addWidget(self._schema_combo, stretch=1)
        self._btn_guess = QPushButton("Guess")
        self._btn_guess.setToolTip("Guess mapping from layer filename / metadata")
        schema_row.addWidget(self._btn_guess)

        self._show_full = QCheckBox("Show full schema")
        self._show_full.setToolTip(
            "List every id in the mapping, not only ids present in the active layer"
        )

        btn_row = QHBoxLayout()
        self._btn_all = QPushButton("All")
        self._btn_none = QPushButton("None")
        self._btn_refresh = QPushButton("Refresh")
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(80)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        self._inner.setLayout(self._inner_layout)
        self._scroll.setWidget(self._inner)

        root = QVBoxLayout()
        root.addWidget(self._hint)
        root.addLayout(schema_row)
        root.addWidget(self._show_full)
        root.addLayout(btn_row)
        root.addWidget(self._scroll, stretch=1)
        self.setLayout(root)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._btn_all.clicked.connect(self.select_all)
        self._btn_none.clicked.connect(self.select_none)
        self._schema_combo.currentIndexChanged.connect(self._on_schema_changed)
        self._show_full.toggled.connect(lambda _: self._refresh_current_layer())
        self._btn_guess.clicked.connect(self._guess_schema)

        self._layer_ref: Any | None = None
        self._viewer: Any | None = None

    def set_viewer(self, viewer: Any | None) -> None:
        """Napari viewer used to promote Image masks to Labels for color editing."""
        self._viewer = viewer

    def _emit_selection_changed(self) -> None:
        """Emit the ``selection_changed`` Qt signal."""
        self.selection_changed.emit()

    def _wire_checkbox(self, cb: QCheckBox) -> None:
        """Connect *cb*'s toggle event to emit ``selection_changed``."""
        cb.toggled.connect(lambda _checked: self._emit_selection_changed())

    def _on_schema_changed(self, _index: int) -> None:
        """Update the active schema key and re-render the checkbox list for it."""
        key = self._schema_combo.currentData()
        if key:
            self._schema_key = str(key)
        self._refresh_current_layer()

    def _guess_schema(self) -> None:
        """Attempt to auto-detect and select the label schema from the current layer's name/path."""
        if self._layer_ref is None:
            self._hint.setText("No layer to guess from.")
            return
        guessed = guess_schema_from_layer(self._layer_ref)
        if not guessed:
            self._hint.setText("Could not guess mapping from layer name/path.")
            return
        idx = self._schema_combo.findData(guessed)
        if idx >= 0:
            self._schema_combo.setCurrentIndex(idx)

    def _refresh_current_layer(self) -> None:
        """Re-render the checkbox list for whichever layer is currently bound."""
        self.refresh_from_layer(self._layer_ref)

    def set_schema_key(self, key: str) -> None:
        """Select a catalog schema by key (e.g. ``ts:total``, ``eicab``)."""
        idx = self._schema_combo.findData(key)
        if idx >= 0:
            self._schema_combo.setCurrentIndex(idx)
        else:
            self._schema_key = key
            self._refresh_current_layer()

    def schema_key(self) -> str:
        """Currently selected label schema key."""
        return self._schema_key

    def current_layer(self) -> Any | None:
        """Layer currently bound to the selector (may be Labels after Image promote)."""
        return self._layer_ref

    def set_expanded(self, expanded: bool) -> None:
        """Fill remaining dock height when visible; collapse when hidden."""
        if expanded:
            expanding = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.setSizePolicy(expanding)
            self._scroll.setSizePolicy(expanding)
            self._scroll.setMinimumHeight(LABEL_SELECTOR_SCROLL_MIN)
            self._scroll.setMaximumHeight(16777215)
        else:
            compact = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.setSizePolicy(compact)
            self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self._scroll.setMinimumHeight(0)
            self._scroll.setMaximumHeight(200)

    def _supports_color_edit(self, layer: Any) -> bool:
        """True if per-label colors on *layer* can be edited (a Labels layer, or a label-like Image
        that can be promoted to one given a bound viewer)."""
        if layer is None:
            return False
        if supports_per_label_color(layer):
            return True
        # Discrete Image masks (e.g. QC segs) can be promoted to Labels.
        return bool(self._viewer is not None and is_label_like_layer(layer))

    def _ensure_colorable_layer(self, layer: Any) -> Any:
        """Return *layer* if it already supports per-label colors, else promote it to a Labels layer
        (updating the bound layer reference); raises ``TypeError`` if no viewer is bound."""
        if supports_per_label_color(layer):
            return layer
        if self._viewer is None:
            raise TypeError("No viewer available to convert the mask to a Labels layer.")
        new_layer = ensure_labels_layer(self._viewer, layer)
        self._layer_ref = new_layer
        return new_layer

    def _make_color_button(self, layer: Any, lid: int) -> QToolButton:
        """Build a small clickable color swatch for label *lid*, opening the color editor on click."""
        btn = QToolButton()
        btn.setFixedSize(18, 18)
        btn.setToolTip(f"Change color for label {lid}")
        btn.setProperty("label_id", lid)
        rgba = get_label_color(layer, lid)
        btn.setStyleSheet(_swatch_stylesheet(rgba))
        btn.clicked.connect(lambda _checked=False, label_id=lid: self._edit_label_color(label_id))
        self._color_buttons[lid] = btn
        return btn

    def _edit_label_color(self, label_id: int) -> None:
        """Open a Qt color dialog for *label_id* and apply the chosen color to the layer and swatch."""
        layer = self._layer_ref
        if layer is None or not self._supports_color_edit(layer):
            return
        try:
            layer = self._ensure_colorable_layer(layer)
        except Exception:
            return
        current = _rgba_to_qcolor(get_label_color(layer, label_id))
        try:
            options = QColorDialog.ColorDialogOption.ShowAlphaChannel
        except AttributeError:
            options = QColorDialog.ShowAlphaChannel  # type: ignore[attr-defined]
        chosen = QColorDialog.getColor(current, self, f"Label {label_id} color", options)
        if not chosen.isValid():
            return
        rgba = _qcolor_to_rgba(chosen)
        set_label_color(layer, label_id, rgba, selected_ids=self.selected_ids())
        btn = self._color_buttons.get(int(label_id))
        if btn is not None:
            btn.setStyleSheet(_swatch_stylesheet(rgba))

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Rebuild the checkbox list (and color swatches, if supported) for *layer*'s label ids under
        the current schema, promoting a discrete Image mask to Labels first when a viewer is bound."""
        # Promote discrete Image masks once so swatches / color visibility work.
        if (
            layer is not None
            and self._viewer is not None
            and type(layer).__name__ != "Labels"
            and is_label_like_layer(layer)
        ):
            try:
                layer = ensure_labels_layer(self._viewer, layer)
            except Exception:
                pass

        self._layer_ref = layer
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checks.clear()
        self._color_buttons.clear()

        if layer is None:
            self._hint.setText("No layer selected.")
            return

        schema = get_schema(self._schema_key)
        layer_ids = unique_layer_labels(label_source_data(layer))
        if self._show_full.isChecked() and schema and schema.id_to_name:
            ids = sorted(set(schema.id_to_name.keys()) | set(layer_ids))
        else:
            ids = layer_ids

        if not ids:
            self._hint.setText(f"No labels to show in “{layer.name}”.")
            return

        mapped = sum(1 for lid in ids if schema and schema.name_for(lid))
        schema_title = schema.title if schema else "Generic"
        color_hint = (
            " — click the color square to edit"
            if self._supports_color_edit(layer)
            else ""
        )
        self._hint.setText(
            f"{len(ids)} label(s) in “{layer.name}” — {schema_title}"
            + (f" ({mapped} named)" if schema and schema.id_to_name else "")
            + color_hint
        )

        can_color = self._supports_color_edit(layer)
        for lid in ids:
            text = schema.display(lid) if schema else f"Label {lid}"
            in_layer = lid in layer_ids
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            if can_color and in_layer:
                row_layout.addWidget(self._make_color_button(layer, lid))
            elif can_color:
                spacer = QWidget()
                spacer.setFixedSize(18, 18)
                row_layout.addWidget(spacer)
            cb = QCheckBox(text)
            cb.setProperty("label_id", lid)
            cb.blockSignals(True)
            cb.setChecked(in_layer)
            cb.blockSignals(False)
            if self._show_full.isChecked() and not in_layer:
                cb.setEnabled(False)
                cb.setStyleSheet("color: gray;")
            self._wire_checkbox(cb)
            row_layout.addWidget(cb, stretch=1)
            self._inner_layout.addWidget(row)
            self._checks.append(cb)
        self._emit_selection_changed()

    def select_all(self) -> None:
        """Check every enabled label checkbox."""
        changed = False
        for cb in self._checks:
            if cb.isEnabled() and not cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                changed = True
        if changed:
            self._emit_selection_changed()

    def select_none(self) -> None:
        """Uncheck every enabled label checkbox."""
        changed = False
        for cb in self._checks:
            if cb.isEnabled() and cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                changed = True
        if changed:
            self._emit_selection_changed()

    def selected_ids(self) -> list[int]:
        """Sorted label ids whose checkbox is both checked and enabled."""
        out = []
        for cb in self._checks:
            if cb.isChecked() and cb.isEnabled():
                out.append(int(cb.property("label_id")))
        return sorted(out)

    def selected_names(self) -> list[str]:
        """Human names for checked ids (falls back to ``Label_<id>``)."""
        schema = get_schema(self._schema_key)
        names = []
        for lid in self.selected_ids():
            if schema and schema.name_for(lid):
                names.append(schema.name_for(lid))
            else:
                names.append(f"Label_{lid}")
        return names
