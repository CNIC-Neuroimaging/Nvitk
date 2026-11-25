"""Record panel (Export & record module): the journal of what was done, to replay or export.

Start recording, work as usual, and every data-changing action lands in the list: imaging tools
with their parameters, mesh operations, reorientation, ROI crops and masks, files opened and
saved. Steps can be switched off, reordered or removed; the list replays in the GUI (on the
same layers, or on the active layer for another subject), saves as JSON, and exports as a Python
script that runs the tool steps headless on files.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.journal import journal
from nvitk.gui.core.widgets import IconButton, Section, hint_label


class RecordPanel(QWidget):
    """UI over :func:`nvitk.gui.core.journal.journal`."""

    def __init__(self, viewer: Any, app_state: dict[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._state = app_state
        self._journal = journal(viewer)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        section = Section("Recording", icon_name="record", hue="red", pref_key="record.steps")
        head = QHBoxLayout()
        self._rec = IconButton("record", "Record the actions that change data", checkable=True, icon_size=18, text="Record")
        self._rec.toggled.connect(self._toggle)
        head.addWidget(self._rec)
        self._status = hint_label("")
        head.addWidget(self._status, 1)
        section.add_layout(head)

        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        from nvitk.gui.core.widgets import resizable  # drag the handle under it to resize

        section.add(resizable(self._list, height=220, minimum=80, pref_key="record.steps"))
        row = QHBoxLayout()
        for icon_name, tip, slot in (
            ("chevron_up", "Move the step up", lambda: self._move(-1)),
            ("chevron_down", "Move the step down", lambda: self._move(1)),
            ("trash", "Remove the step", self._remove),
            ("close", "Clear every step", self._journal.clear),
        ):
            button = IconButton(icon_name, tip, icon_size=14)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        section.add_layout(row)
        root.addWidget(section)

        actions = Section("Replay and export", icon_name="replay", hue="cyan", pref_key="record.actions")
        grid = QHBoxLayout()
        for text, tip, slot in (
            ("Replay", "Run the ticked steps again on the layers they used.", lambda: self._replay(False)),
            ("Replay on active", "Run the ticked steps with the active layer as the first input (another subject).", lambda: self._replay(True)),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            grid.addWidget(button)
        actions.add_layout(grid)
        grid = QHBoxLayout()
        for text, tip, slot in (
            ("Save JSON…", "Save the journal (nvitk-journal/2).", self._save),
            ("Load JSON…", "Load a journal (or an older pipeline JSON).", self._load),
            ("Export Python…", "A script running the tool steps headless on files.", self._export_py),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            grid.addWidget(button)
        actions.add_layout(grid)
        self._report = hint_label("")
        actions.add(self._report)
        root.addWidget(actions)
        root.addStretch(1)

        self._journal.changed.connect(self._refresh)
        self._refresh()

    def _toggle(self, on: bool) -> None:
        self._journal.set_recording(on)
        if self._state is not None:
            self._state["record_enabled"] = bool(on)

    def _refresh(self) -> None:
        self._rec.blockSignals(True)
        self._rec.setChecked(self._journal.recording)
        self._rec.blockSignals(False)
        self._list.blockSignals(True)
        self._list.clear()
        for step in self._journal.steps:
            item = QListWidgetItem(step.describe())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if step.enabled else Qt.Unchecked)
            item.setToolTip(f"{step.timestamp}\n{step.params}")
            self._list.addItem(item)
        self._list.blockSignals(False)
        n = len(self._journal.steps)
        state = "recording" if self._journal.recording else "not recording"
        self._status.setText(f"{n} step{'s' if n != 1 else ''}, {state}")

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        row = self._list.row(item)
        if 0 <= row < len(self._journal.steps):
            self._journal.steps[row].enabled = item.checkState() == Qt.Checked

    def _move(self, offset: int) -> None:
        row = self._list.currentRow()
        self._journal.move(row, offset)
        self._list.setCurrentRow(max(0, min(row + offset, self._list.count() - 1)))

    def _remove(self) -> None:
        self._journal.remove(self._list.currentRow())

    def _replay(self, on_active: bool) -> None:
        report = self._journal.replay(self._viewer, on_active=on_active)
        self._report.setText("\n".join(report[-8:]) or "Nothing to replay.")

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save journal", "journal.json", "JSON (*.json)")
        if path:
            self._journal.save(path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load journal", "", "JSON (*.json)")
        if path:
            self._journal.load(path)

    def _export_py(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export as Python", "nvitk_session.py", "Python (*.py)")
        if path:
            self._journal.export_python(path)


__all__ = ["RecordPanel"]
