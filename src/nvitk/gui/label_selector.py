"""Checkbox list of integer labels present in the active Napari layer."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy


def unique_layer_labels(data: np.ndarray, *, max_labels: int = 500) -> list[int]:
    flat = to_numpy(data).ravel()
    if flat.size == 0:
        return []
    if np.issubdtype(flat.dtype, np.floating):
        vals = np.unique(flat[np.isfinite(flat)])
        labels = [int(round(v)) for v in vals if v != 0]
    else:
        vals = np.unique(flat)
        labels = [int(v) for v in vals if int(v) != 0]
    labels.sort()
    if len(labels) > max_labels:
        return labels[:max_labels]
    return labels


class LabelSelectorWidget(QGroupBox):
    """Select one or more label ids from the active layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Label selection", parent)
        self._checks: list[QCheckBox] = []
        self._hint = QLabel("Select labels below (from active layer).")
        self._hint.setWordWrap(True)
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
        self._scroll.setMaximumHeight(140)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        self._inner.setLayout(self._inner_layout)
        self._scroll.setWidget(self._inner)

        root = QVBoxLayout()
        root.addWidget(self._hint)
        root.addLayout(btn_row)
        root.addWidget(self._scroll)
        self.setLayout(root)

        self._btn_all.clicked.connect(self.select_all)
        self._btn_none.clicked.connect(self.select_none)
        self._btn_refresh.clicked.connect(lambda: None)

    def refresh_from_layer(self, layer: Any | None) -> None:
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checks.clear()
        if layer is None:
            self._hint.setText("No layer selected.")
            return
        labels = unique_layer_labels(to_numpy(layer.data))
        if not labels:
            self._hint.setText(f"No non-zero labels in “{layer.name}”.")
            return
        self._hint.setText(f"{len(labels)} label(s) in “{layer.name}”.")
        for lid in labels:
            cb = QCheckBox(f"Label {lid}")
            cb.setProperty("label_id", lid)
            cb.setChecked(True)
            self._inner_layout.addWidget(cb)
            self._checks.append(cb)

    def select_all(self) -> None:
        for cb in self._checks:
            cb.setChecked(True)

    def select_none(self) -> None:
        for cb in self._checks:
            cb.setChecked(False)

    def selected_ids(self) -> list[int]:
        out: list[int] = []
        for cb in self._checks:
            if cb.isChecked():
                out.append(int(cb.property("label_id")))
        return sorted(out)
