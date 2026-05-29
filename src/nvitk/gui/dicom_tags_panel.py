"""DICOM tag table for the active Napari layer."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.types.image import _is_dicom_tag_key


def _nvitk_metadata(layer: Any | None) -> dict[str, Any]:
    if layer is None:
        return {}
    meta = getattr(layer, "metadata", None) or {}
    if not isinstance(meta, dict):
        return {}
    nv = meta.get("nvitk_metadata")
    return nv if isinstance(nv, dict) else {}


def dicom_tags_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Collect DICOM tag entries from nvitk layer metadata."""
    tags = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not _is_dicom_tag_key(key):
            continue
        if value is None:
            continue
        tags[key] = value
    return tags


def layer_has_dicom_tags(layer: Any | None) -> bool:
    if layer is None:
        return False
    nv = _nvitk_metadata(layer)
    if str(nv.get("source_type") or "").lower() == "dicom":
        return True
    return bool(dicom_tags_from_metadata(nv))


def _format_tag_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        parts = [_format_tag_value(v) for v in value[:32]]
        if len(value) > 32:
            parts.append("…")
        return ", ".join(parts)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)


class DicomTagsPanel(QWidget):
    """Scrollable table of DICOM tags for the selected layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._status = QLabel("Select a layer loaded from DICOM to view tags.")
        self._status.setWordWrap(True)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Tag", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(True)
        self._table.setStyleSheet(
            "QTableWidget {"
            "  background-color: #2b2b2b;"
            "  color: #e8e8e8;"
            "  gridline-color: #454545;"
            "  alternate-background-color: #2b2b2b;"
            "}"
            "QTableWidget::item {"
            "  background-color: #2b2b2b;"
            "  color: #e8e8e8;"
            "}"
            "QTableWidget::item:selected {"
            "  background-color: #3d5a80;"
            "  color: #ffffff;"
            "}"
            "QHeaderView::section {"
            "  background-color: #353535;"
            "  color: #e8e8e8;"
            "  padding: 4px;"
            "  border: 1px solid #454545;"
            "}"
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch(1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._status)
        root.addLayout(btn_row)
        root.addWidget(self._table, stretch=1)
        self.setLayout(root)

        self._btn_refresh.clicked.connect(self._refresh_last_layer)
        self._last_layer: Any | None = None

    def _refresh_last_layer(self) -> None:
        self.refresh_from_layer(self._last_layer)

    def refresh_from_layer(self, layer: Any | None) -> None:
        self._last_layer = layer
        tags = dicom_tags_from_metadata(_nvitk_metadata(layer))
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        if layer is None:
            self._status.setText("No layer selected.")
            self._table.setSortingEnabled(True)
            return

        if not tags:
            name = getattr(layer, "name", "layer")
            self._status.setText(
                f"“{name}” has no DICOM tags in metadata "
                "(open a .dcm file or DICOM folder with nvitk I/O)."
            )
            self._table.setSortingEnabled(True)
            return

        self._status.setText(f"{len(tags)} tag(s) from “{layer.name}”.")
        self._table.setRowCount(len(tags))
        for row, key in enumerate(sorted(tags.keys())):
            tag_item = QTableWidgetItem(key)
            val_item = QTableWidgetItem(_format_tag_value(tags[key]))
            tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsEditable)
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, tag_item)
            self._table.setItem(row, 1, val_item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
