"""Checkbox list of labels with optional pipeline / model name mapping."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.gui.label_catalog import (
    all_schemas,
    get_schema,
    guess_schema_from_layer,
    schema_keys,
)
from nvitk.gui.label_visibility import label_source_data, unique_layer_labels

LABEL_SELECTOR_SCROLL_MIN = 80


class LabelSelectorWidget(QGroupBox):
    """Select label ids using an optional named vocabulary (eICAB, QVTpy, TS, …)."""

    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Label selection", parent)
        self._checks: list[QCheckBox] = []
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

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit()

    def _wire_checkbox(self, cb: QCheckBox) -> None:
        cb.toggled.connect(lambda _checked: self._emit_selection_changed())

    def _on_schema_changed(self, _index: int) -> None:
        key = self._schema_combo.currentData()
        if key:
            self._schema_key = str(key)
        self._refresh_current_layer()

    def _guess_schema(self) -> None:
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
        return self._schema_key

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

    def refresh_from_layer(self, layer: Any | None) -> None:
        self._layer_ref = layer
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checks.clear()

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
        self._hint.setText(
            f"{len(ids)} label(s) in “{layer.name}” — {schema_title}"
            + (f" ({mapped} named)" if schema and schema.id_to_name else "")
        )

        for lid in ids:
            text = schema.display(lid) if schema else f"Label {lid}"
            in_layer = lid in layer_ids
            cb = QCheckBox(text)
            cb.setProperty("label_id", lid)
            cb.blockSignals(True)
            cb.setChecked(in_layer)
            cb.blockSignals(False)
            if self._show_full.isChecked() and not in_layer:
                cb.setEnabled(False)
                cb.setStyleSheet("color: gray;")
            self._wire_checkbox(cb)
            self._inner_layout.addWidget(cb)
            self._checks.append(cb)
        self._emit_selection_changed()

    def select_all(self) -> None:
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
