"""Live spatial / affine properties for the active Napari layer."""

from __future__ import annotations

from typing import Any

from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.spatial import format_layer_spatial_info


class ImagePropertiesPanel(QWidget):
    """Show spacing, FOV, origin, orientation, and affine for the selected layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = QLabel("Select a layer to view spatial properties.")
        self._status.setWordWrap(True)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self._text.setFont(mono)
        self._text.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #2b2b2b;"
            "  color: #e8e8e8;"
            "  border: 1px solid #454545;"
            "}"
        )

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch(1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._status)
        root.addLayout(btn_row)
        root.addWidget(self._text, stretch=1)
        self.setLayout(root)

        self._btn_refresh.clicked.connect(self._refresh_last_layer)
        self._last_layer: Any | None = None

    def _refresh_last_layer(self) -> None:
        self.refresh_from_layer(self._last_layer)

    def refresh_from_layer(self, layer: Any | None) -> None:
        self._last_layer = layer
        if layer is None:
            self._status.setText("No layer selected.")
            self._text.setPlainText("")
            return
        try:
            text = format_layer_spatial_info(layer)
        except Exception as exc:
            name = getattr(layer, "name", "layer")
            self._status.setText(f"Could not read properties for “{name}”.")
            self._text.setPlainText(str(exc))
            return
        name = getattr(layer, "name", "layer")
        self._status.setText(f"Spatial properties for “{name}”.")
        self._text.setPlainText(text)
