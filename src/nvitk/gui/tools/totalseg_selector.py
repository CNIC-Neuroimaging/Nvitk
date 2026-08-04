"""TotalSegmentator task + ROI subset checkboxes (total / total_mr)."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS, get_class_map

_SUBSET_TASKS = frozenset({"total", "total_mr"})


class TotalSegRoiWidget(QGroupBox):
    """Checkbox list of ROI names for a TotalSegmentator subset task (``total``/``total_mr``), with
    All/None select buttons; hidden for tasks that don't support ROI subsetting."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the All/None buttons and the scrollable checkbox list container."""
        super().__init__("TotalSegmentator ROIs (subset)", parent)
        self._checks: list[QCheckBox] = []
        btn_row = QHBoxLayout()
        self._btn_all = QPushButton("All")
        self._btn_none = QPushButton("None")
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addStretch(1)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMaximumHeight(160)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        self._inner.setLayout(self._inner_layout)
        self._scroll.setWidget(self._inner)
        root = QVBoxLayout()
        root.addLayout(btn_row)
        root.addWidget(self._scroll)
        self.setLayout(root)
        self._btn_all.clicked.connect(self.select_all)
        self._btn_none.clicked.connect(self.select_none)

    @staticmethod
    def available_tasks() -> tuple[str, ...]:
        """Every registered TotalSegmentator task name."""
        return AVAILABLE_TASKS

    def set_task(self, task: str) -> None:
        """Rebuild the checkbox list for *task*'s ROI class map, hiding this widget entirely for
        tasks that don't support ROI subsetting."""
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checks.clear()
        if task not in _SUBSET_TASKS:
            self.setVisible(False)
            return
        self.setVisible(True)
        cmap = get_class_map(task)
        for _lid, name in sorted(cmap.items(), key=lambda x: x[1]):
            cb = QCheckBox(name)
            cb.setProperty("roi_name", name)
            cb.setChecked(False)
            self._inner_layout.addWidget(cb)
            self._checks.append(cb)

    def select_all(self) -> None:
        """Check every ROI checkbox."""
        for cb in self._checks:
            cb.setChecked(True)

    def select_none(self) -> None:
        """Uncheck every ROI checkbox."""
        for cb in self._checks:
            cb.setChecked(False)

    def selected_roi_names(self) -> list[str] | None:
        """Names of the checked ROIs, or ``None`` if none are checked (meaning "all ROIs")."""
        names = [str(cb.property("roi_name")) for cb in self._checks if cb.isChecked()]
        if not names:
            return None
        return names
