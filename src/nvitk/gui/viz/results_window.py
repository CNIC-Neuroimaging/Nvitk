"""A window for measurement results that are text rather than a layer.

Measurements used to arrive as a Napari toast plus a line in the log panel. That
is fine for ``volume = 12.4 mm³`` and poor for anything with structure: a toast
shows only the first line, and a per-label metrics table in the log is a wall of
``name: value`` that cannot be sorted, aligned or copied as a unit.

This window takes the same payload the log line is built from and renders it as
a table — one window per viewer, reused across runs, with a history of previous
results so a second measurement does not destroy the one you were reading. The
log line is still written; this is an addition, not a replacement.

Two payload shapes are supported, both flowing through :func:`show_results`:

``{"dice": 0.91, "jaccard": 0.84}``
    a flat metric table, rendered as two columns.
``{1: {"dice": 0.91}, 2: {"dice": 0.88}}`` with ``row_header="Label"``
    a matrix, rendered with one row per key and one column per metric.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_MUTED,
    SPACE_TIGHT,
    apply_theme,
    fmt_number,
    mono_font,
)

#: Kept on the viewer so a second measurement re-uses the window.
_WINDOW_ATTR = "_nvitk_results_window"

#: How many past results stay in the picker before the oldest is dropped.
_HISTORY_LIMIT = 25


def _is_matrix(payload: Mapping[Any, Any]) -> bool:
    """Whether *payload* maps row keys to per-column dicts rather than to values."""
    return bool(payload) and all(isinstance(v, Mapping) for v in payload.values())


#: Metrics that are voxel counts rather than ratios. Formatting is decided by
#: name because the value cannot say: a Dice of 0.0 and a TP of 0 are both
#: integral floats, and rendering the Dice as "0" in a column of "0.912346"
#: breaks the decimal alignment that makes a metrics table readable.
COUNT_METRICS = frozenset({"TP", "TN", "FP", "FN", "n", "n_samples", "count", "voxels"})

#: Decimals shown for anything that is not a count.
_DIGITS = 6


def _format_value(value: Any, key: Any = None) -> str:
    """Render one cell: counts as integers, everything else to a fixed width."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if str(key) in COUNT_METRICS:
            return str(int(value)) if value.is_integer() else fmt_number(value, _DIGITS)
        # A large round number under an unrecognised name is a count too.
        if value.is_integer() and abs(value) >= 1000:
            return str(int(value))
        return f"{value:.{_DIGITS}f}"
    return str(value)


class _Result:
    """One captured result: what to show and how to lay it out."""

    def __init__(
        self,
        title: str,
        payload: Mapping[Any, Any],
        *,
        subtitle: str = "",
        row_header: str = "",
        note: str = "",
    ) -> None:
        self.title = title
        self.payload = dict(payload)
        self.subtitle = subtitle
        self.row_header = row_header
        self.note = note

    def as_text(self) -> str:
        """The result as plain text, for the clipboard and the GUI log."""
        lines = [self.title]
        if self.subtitle:
            lines.append(self.subtitle)
        if _is_matrix(self.payload):
            columns = self._columns()
            head = self.row_header or ""
            widths = [max(len(head), *(len(str(k)) for k in self.payload))]
            widths += [
                max(len(c), *(len(_format_value(r.get(c, ""), c)) for r in self.payload.values()))
                for c in columns
            ]
            lines.append("  ".join(
                [head.ljust(widths[0])] + [c.rjust(w) for c, w in zip(columns, widths[1:])]
            ))
            for key, row in self.payload.items():
                cells = [_format_value(row.get(c, ""), c) for c in columns]
                lines.append("  ".join(
                    [str(key).ljust(widths[0])]
                    + [c.rjust(w) for c, w in zip(cells, widths[1:])]
                ))
        else:
            width = max((len(str(k)) for k in self.payload), default=0)
            for key, value in self.payload.items():
                lines.append(f"{str(key).ljust(width)}  {_format_value(value, key)}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)

    def _columns(self) -> list[str]:
        """Column order for a matrix payload: first-seen order across all rows."""
        columns: list[str] = []
        for row in self.payload.values():
            for key in row:
                if key not in columns:
                    columns.append(str(key))
        return columns


class ResultsWindow(QDialog):
    """A reusable table view for measurement output."""

    def __init__(self, parent: Any = None) -> None:
        """Build the table and its toolbar."""
        super().__init__(parent)
        self.setWindowTitle("Results")
        self.setMinimumSize(560, 340)
        # Roomy enough for a per-label overlap table (a handful of rows, a dozen
        # metric columns) without the user having to resize it on every launch.
        self.resize(900, 460)
        # Modeless: results are read while carrying on with the viewer.
        self.setModal(False)

        self._results: list[_Result] = []

        self._heading = QLabel("")
        self._heading.setStyleSheet(f"color: {COLOR_ACCENT}; font-weight: 600;")
        self._subtitle = QLabel("")
        self._subtitle.setWordWrap(True)
        self._subtitle.setStyleSheet(f"color: {COLOR_MUTED};")

        self._history = QComboBox()
        self._history.setMinimumWidth(180)
        self._history.currentIndexChanged.connect(self._on_history_changed)

        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy this result to the clipboard as text")
        copy_button.clicked.connect(self._copy)

        self._table = QTableWidget(0, 0)
        self._table.setFont(mono_font(10))
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {COLOR_MUTED};")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE_TIGHT)
        toolbar.addWidget(QLabel("History"))
        toolbar.addWidget(self._history, stretch=1)
        toolbar.addWidget(copy_button)

        root = QVBoxLayout(self)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._heading)
        root.addWidget(self._subtitle)
        root.addLayout(toolbar)
        root.addWidget(self._table, stretch=1)
        root.addWidget(self._note)

    def add_result(
        self,
        title: str,
        payload: Mapping[Any, Any],
        *,
        subtitle: str = "",
        row_header: str = "",
        note: str = "",
    ) -> None:
        """Show *payload* and keep the previous results in the history picker."""
        result = _Result(title, payload, subtitle=subtitle, row_header=row_header, note=note)
        self._results.insert(0, result)
        del self._results[_HISTORY_LIMIT:]
        # Repopulating fires currentIndexChanged; suppress it so rebuilding the
        # list does not count as the user picking an entry.
        self._history.blockSignals(True)
        self._history.clear()
        self._history.addItems([r.title for r in self._results])
        self._history.setCurrentIndex(0)
        self._history.blockSignals(False)
        self._show(result)

    def current_text(self) -> str:
        """The displayed result as plain text."""
        index = max(self._history.currentIndex(), 0)
        if not self._results:
            return ""
        return self._results[min(index, len(self._results) - 1)].as_text()

    def _on_history_changed(self, index: int) -> None:
        """Show the result the user picked out of the history."""
        if 0 <= index < len(self._results):
            self._show(self._results[index])

    def _copy(self) -> None:
        """Put the displayed result on the clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.current_text())

    def _show(self, result: _Result) -> None:
        """Lay *result* out in the table."""
        self.setWindowTitle(result.title or "Results")
        self._heading.setText(result.title)
        self._subtitle.setText(result.subtitle)
        self._subtitle.setVisible(bool(result.subtitle))
        self._note.setText(result.note)
        self._note.setVisible(bool(result.note))

        # Sorting has to be off while filling, or rows move under the cursor as
        # they are inserted and the table ends up scrambled.
        self._table.setSortingEnabled(False)
        if _is_matrix(result.payload):
            columns = result._columns()
            self._table.setColumnCount(len(columns) + 1)
            self._table.setHorizontalHeaderLabels([result.row_header or ""] + columns)
            self._table.setRowCount(len(result.payload))
            for row, (key, values) in enumerate(result.payload.items()):
                self._table.setItem(row, 0, _item(str(key), align_left=True))
                for col, name in enumerate(columns, start=1):
                    self._table.setItem(row, col, _item(_format_value(values.get(name, ""), name)))
        else:
            self._table.setColumnCount(2)
            self._table.setHorizontalHeaderLabels(["Metric", "Value"])
            self._table.setRowCount(len(result.payload))
            for row, (key, value) in enumerate(result.payload.items()):
                self._table.setItem(row, 0, _item(str(key), align_left=True))
                self._table.setItem(row, 1, _item(_format_value(value, key)))
        self._table.setSortingEnabled(True)

        # Sized to content, not stretched to fill: a 13-column metrics table
        # divided equally leaves every cell too narrow and elides the digits,
        # which is the one thing the table exists to show. Wide tables scroll.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)


def _item(text: str, *, align_left: bool = False) -> QTableWidgetItem:
    """A read-only table cell, numbers right-aligned so decimal points line up."""
    item = QTableWidgetItem(text)
    item.setTextAlignment(
        (Qt.AlignLeft if align_left else Qt.AlignRight) | Qt.AlignVCenter
    )
    return item


def results_window(viewer: Any) -> ResultsWindow | None:
    """The viewer's results window, created on first use.

    ``None`` when no Qt application is running — a headless call site should get
    the log line and nothing else rather than an exception.
    """
    if QApplication.instance() is None:
        return None
    window = getattr(viewer, _WINDOW_ATTR, None)
    if window is not None:
        try:
            window.isVisible()  # a window closed by Qt leaves a dead wrapper
        except RuntimeError:
            window = None
    if window is None:
        parent = None
        try:
            parent = viewer.window._qt_window
        except Exception:  # noqa: BLE001 — a parentless dialog still works
            parent = None
        window = ResultsWindow(parent)
        apply_theme(window)
        try:
            setattr(viewer, _WINDOW_ATTR, window)
        except Exception:  # noqa: BLE001 — a viewer that rejects attributes just
            # gets a fresh window each time, which still shows the result.
            pass
    return window


def show_results(
    viewer: Any,
    title: str,
    payload: Mapping[Any, Any],
    *,
    subtitle: str = "",
    row_header: str = "",
    note: str = "",
) -> str:
    """Show *payload* in the viewer's results window; return it as text.

    The text is returned rather than logged here so the caller keeps control of
    the log — every measurement already writes one line, and this must not
    duplicate it.
    """
    text = _Result(title, payload, subtitle=subtitle, row_header=row_header, note=note).as_text()
    window = results_window(viewer)
    if window is None:
        return text
    window.add_result(title, payload, subtitle=subtitle, row_header=row_header, note=note)
    window.show()
    window.raise_()
    return text


__all__ = ["COUNT_METRICS", "ResultsWindow", "results_window", "show_results"]
