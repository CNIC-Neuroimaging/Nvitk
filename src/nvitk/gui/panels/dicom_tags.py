"""DICOM tag table for the active Napari layer."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_ACCENT, COLOR_MUTED, COLOR_TEXT
from nvitk.types.image import _is_dicom_tag_key


def _nvitk_metadata(layer: Any | None) -> dict[str, Any]:
    """*layer*'s nvitk metadata dict (the ``nvitk_metadata`` sub-key), or ``{}`` if unavailable."""
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
    """True if *layer* was loaded from DICOM or carries any DICOM tag metadata."""
    if layer is None:
        return False
    nv = _nvitk_metadata(layer)
    if str(nv.get("source_type") or "").lower() == "dicom":
        return True
    return bool(dicom_tags_from_metadata(nv))


def _format_scalar(value: Any) -> str:
    """Format one DICOM value: decode bytes, else ``str()``."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)


def _format_tag_value(value: Any) -> str:
    """Format a DICOM tag *value* for table display, joining short sequences."""
    if isinstance(value, (list, tuple)):
        parts = [_format_tag_value(v) for v in value[:32]]
        if len(value) > 32:
            parts.append("…")
        return ", ".join(parts)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_format_tag_value(v)}" for k, v in list(value.items())[:8])
    return _format_scalar(value)


def expand_tag(key: str, value: Any, *, prefix: str = "") -> list[tuple[str, str, int]]:
    """Flatten a DICOM tag into ``(name, value, depth)`` rows, one per item.

    A multi-valued tag (``ImageOrientationPatient``) and a sequence tag
    (``ReferencedImageSequence``, a list of datasets) both arrive as a Python list.
    Collapsing either onto one line is what made long sequences unreadable — the
    values ran off the row and the per-item structure was lost. Each item gets its
    own indexed row, and a nested dataset's own tags are expanded beneath it.
    """
    name = f"{prefix}{key}"
    if isinstance(value, (list, tuple)):
        rows: list[tuple[str, str, int]] = [
            (name, f"[{len(value)} item(s)]", len(prefix.split("·")) - 1 if prefix else 0)
        ]
        depth = rows[0][2] + 1
        for i, item in enumerate(value):
            label = f"{name} [{i}]"
            if isinstance(item, dict):
                rows.append((label, f"[{len(item)} tag(s)]", depth))
                for sub_key in sorted(item, key=str):
                    rows.extend(
                        expand_tag(str(sub_key), item[sub_key], prefix=f"{label} · ")
                    )
            else:
                rows.append((label, _format_scalar(item), depth))
        return rows
    if isinstance(value, dict):
        rows = [(name, f"[{len(value)} tag(s)]", 0)]
        for sub_key in sorted(value, key=str):
            rows.extend(expand_tag(str(sub_key), value[sub_key], prefix=f"{name} · "))
        return rows
    return [(name, _format_scalar(value), len(prefix.split("·")) - 1 if prefix else 0)]


def tag_rows(tags: dict[str, Any]) -> list[tuple[str, str, int]]:
    """Every tag as flattened ``(name, value, depth)`` rows, in tag order."""
    rows: list[tuple[str, str, int]] = []
    for key in sorted(tags, key=str):
        rows.extend(expand_tag(str(key), tags[key]))
    return rows


class DicomTagsPanel(QWidget):
    """Scrollable table of DICOM tags for the selected layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the search box, sortable tag/value table, and refresh button."""
        super().__init__(parent)
        self._status = QLabel("Select a layer loaded from DICOM to view tags.")
        self._status.setWordWrap(True)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Tag", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        # Not sortable: the rows carry a sequence's parent/child order, and
        # sorting a column would interleave items with unrelated parents.

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search tags…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_search_filter)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch(1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._status)
        root.addWidget(self._search)
        root.addLayout(btn_row)
        root.addWidget(self._table, stretch=1)
        self.setLayout(root)

        self._btn_refresh.clicked.connect(self._refresh_last_layer)
        self._last_layer: Any | None = None

    def _apply_search_filter(self, text: str = "") -> None:
        """Hide table rows whose tag/value don't contain the (case-insensitive) search *text*."""
        query = (text or self._search.text() or "").strip().lower()
        for row in range(self._table.rowCount()):
            if not query:
                self._table.setRowHidden(row, False)
                continue
            tag_item = self._table.item(row, 0)
            val_item = self._table.item(row, 1)
            tag = (tag_item.text() if tag_item is not None else "").lower()
            val = (val_item.text() if val_item is not None else "").lower()
            self._table.setRowHidden(row, query not in tag and query not in val)

    def _refresh_last_layer(self) -> None:
        """Re-render the DICOM tag table for whichever layer was last shown."""
        self.refresh_from_layer(self._last_layer)

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Populate the tag table from *layer*'s DICOM metadata, or show a placeholder if *layer* is
        ``None`` or has no DICOM tags."""
        self._last_layer = layer
        tags = dicom_tags_from_metadata(_nvitk_metadata(layer))
        self._table.setRowCount(0)

        if layer is None:
            self._status.setText("No layer selected.")
            self._apply_search_filter()
            return

        if not tags:
            name = getattr(layer, "name", "layer")
            self._status.setText(
                f"“{name}” has no DICOM tags in metadata "
                "(open a .dcm file or DICOM folder with nvitk I/O)."
            )
            self._apply_search_filter()
            return

        rows = tag_rows(tags)
        expanded = len(rows) - len(tags)
        self._status.setText(
            f"{len(tags)} tag(s) from “{layer.name}”"
            + (f" — {len(rows)} rows with sequences expanded." if expanded > 0 else ".")
        )
        self._table.setRowCount(len(rows))
        for row, (name, value, depth) in enumerate(rows):
            # Indent nested sequence items so the structure reads at a glance.
            tag_item = QTableWidgetItem(("    " * int(depth)) + name.split(" · ")[-1])
            tag_item.setToolTip(name)
            val_item = QTableWidgetItem(value)
            val_item.setToolTip(value)
            if value.startswith("[") and value.endswith(")]"):
                # A container row: its children carry the data, so mute the header.
                tag_item.setForeground(QColor(COLOR_ACCENT))
                val_item.setForeground(QColor(COLOR_MUTED))
            elif depth:
                val_item.setForeground(QColor(COLOR_TEXT))
            tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsEditable)
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, tag_item)
            self._table.setItem(row, 1, val_item)
        self._table.resizeColumnToContents(0)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._apply_search_filter()
