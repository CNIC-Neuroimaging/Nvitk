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

A result may come with an ``on_select`` callback: selecting a row calls it with
that row's key (a branch, a frame…) so the viewer can show what the row is
about, and ``None`` when that should go (*Clear highlight*, another result, the
window closing).

A per-label result may also name the label layer it measured and which label
each row is (``label_layer``, ``row_labels``). The window then offers *Colour
labels by*: pick a measured column and a colormap, and the layer's labels take
the colour of their value (the range automatic or typed, a legend beside it;
labels without a value turn grey), as the Meshlab panel colours a surface by a
field. *Original colours* puts the label colours back as they were. Each label
row carries a swatch of its label's colour.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Mapping, Sequence

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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

#: Colormaps offered for colouring labels by a measurement.
VALUE_COLORMAPS = ("viridis", "plasma", "magma", "inferno", "turbo", "cividis", "coolwarm",
                   "RdYlGn", "RdYlGn_r", "RdYlBu_r", "Blues", "Reds", "Greens", "jet")
#: The column choice meaning "the labels' own colours".
_OWN_COLOURS = "— label colours —"

#: The label colours a layer had before it was first coloured by a measurement.
_ORIGINAL_COLOURS: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()


def _is_matrix(payload: Mapping[Any, Any]) -> bool:
    """Whether *payload* maps row keys to per-column dicts rather than to values."""
    return bool(payload) and all(isinstance(v, Mapping) for v in payload.values())


def _int_key(key: str) -> Any:
    """*key* as the int a payload may use for it (row keys are compared as text)."""
    try:
        return int(key)
    except (TypeError, ValueError):
        return key


def _gradient(colormap: str, width: int = 140, height: int = 12) -> QPixmap:
    """A horizontal swatch of *colormap*, low on the left."""
    import numpy as np
    from matplotlib import colormaps

    ramp = (np.asarray(colormaps[colormap](np.linspace(0.0, 1.0, width)))[:, :3] * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(np.repeat(ramp[None, :, :], height, axis=0))
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def _swatch_icon(rgba: Any, size: int = 10) -> QIcon:
    pix = QPixmap(size, size)
    r, g, b = (max(0.0, min(1.0, float(c))) for c in list(rgba)[:3])
    pix.fill(QColor.fromRgbF(r, g, b, 1.0))
    return QIcon(pix)


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
        on_select: Callable[[str | None], None] | None = None,
        label_layer: Any = None,
        row_labels: Mapping[Any, int] | None = None,
    ) -> None:
        self.title = title
        self.payload = dict(payload)
        self.subtitle = subtitle
        self.row_header = row_header
        self.note = note
        self.on_select = on_select
        #: The label layer measured (weakly) and the label id of each row key.
        self.label_ref: Any = None
        if label_layer is not None:
            try:
                self.label_ref = weakref.ref(label_layer)
            except TypeError:
                self.label_ref = None
        self.row_labels = {str(k): int(v) for k, v in dict(row_labels or {}).items()}
        #: How this result last coloured its labels.
        self.colour: dict[str, Any] = {"column": "", "colormap": VALUE_COLORMAPS[0], "auto": True,
                                       "limits": None}

    def label_layer(self) -> Any:
        return self.label_ref() if self.label_ref is not None else None

    def numeric_columns(self) -> list[str]:
        """The columns with a number for at least one label row."""
        if not self.row_labels or not _is_matrix(self.payload):
            return []
        out = []
        for column in self._columns():
            for key in self.row_labels:
                value = self.payload.get(key, self.payload.get(_int_key(key), {})).get(column)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out.append(column)
                    break
        return out

    def label_values(self, column: str) -> dict[int, float]:
        """Label id → its value in *column* (the first row of a label measured twice)."""
        out: dict[int, float] = {}
        for key, lid in self.row_labels.items():
            row = self.payload.get(key, self.payload.get(_int_key(key), {}))
            value = row.get(column) if isinstance(row, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and lid not in out:
                out[lid] = float(value)
        return out

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
        self._shown: _Result | None = None
        self._filling = False

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

        self._clear_highlight = QPushButton("Clear highlight")
        self._clear_highlight.setToolTip("Stop showing the selected row in the viewer")
        self._clear_highlight.clicked.connect(self._unhighlight)
        self._clear_highlight.setVisible(False)

        self._table = QTableWidget(0, 0)
        self._table.setFont(mono_font(10))
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color: {COLOR_MUTED};")

        #: Set by :func:`results_window`: Image masks are turned into Labels to be coloured.
        self._viewer: Any = None
        self._colour_row = self._build_colour_row()

        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE_TIGHT)
        toolbar.addWidget(QLabel("History"))
        toolbar.addWidget(self._history, stretch=1)
        toolbar.addWidget(self._clear_highlight)
        toolbar.addWidget(copy_button)

        root = QVBoxLayout(self)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._heading)
        root.addWidget(self._subtitle)
        root.addLayout(toolbar)
        root.addWidget(self._colour_row)
        root.addWidget(self._table, stretch=1)
        root.addWidget(self._note)

    def _build_colour_row(self) -> QWidget:
        """*Colour labels by*: a measured column, a colormap, the range, a legend."""
        row = QWidget()
        outer = QVBoxLayout(row)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE_TIGHT)
        outer.addLayout(lay)
        self._colour_column = QComboBox()
        self._colour_column.setMinimumWidth(150)
        self._colour_column.setToolTip("Colour the measured labels in the viewer by this column "
                                       "(as the Meshlab panel colours a surface by a field).")
        self._colour_map = QComboBox()
        self._colour_map.addItems(VALUE_COLORMAPS)
        self._colour_auto = QCheckBox("auto")
        self._colour_auto.setChecked(True)
        self._colour_auto.setToolTip("Range from the lowest to the highest value; untick to type it.")
        self._colour_lo = QDoubleSpinBox()
        self._colour_hi = QDoubleSpinBox()
        for spin in (self._colour_lo, self._colour_hi):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(4)
            spin.setEnabled(False)
            spin.setMinimumWidth(120)
        self._colour_lo.setToolTip("Value at the low end of the colormap")
        self._colour_hi.setToolTip("Value at the high end of the colormap")
        self._colour_legend = QLabel()
        self._colour_legend.setToolTip("The colormap, low value on the left")
        self._colour_reset = QPushButton("Original colours")
        self._colour_reset.setToolTip("Give the labels back the colours they had before")
        self._colour_status = QLabel("")
        self._colour_status.setStyleSheet(f"color: {COLOR_MUTED};")
        lay.addWidget(QLabel("Colour labels by"))
        lay.addWidget(self._colour_column, stretch=1)
        lay.addWidget(self._colour_map)
        lay.addWidget(self._colour_legend)
        second = QHBoxLayout()
        second.setContentsMargins(0, 0, 0, 0)
        second.setSpacing(SPACE_TIGHT)
        second.addWidget(QLabel("Range"))
        second.addWidget(self._colour_auto)
        second.addWidget(self._colour_lo)
        second.addWidget(QLabel("–"))
        second.addWidget(self._colour_hi)
        second.addStretch(1)
        second.addWidget(self._colour_reset)
        outer.addLayout(second)
        outer.addWidget(self._colour_status)
        self._colour_column.currentTextChanged.connect(lambda _t: self._on_colour_changed())
        self._colour_map.currentTextChanged.connect(lambda _t: self._on_colour_changed())
        self._colour_auto.toggled.connect(self._on_colour_auto)
        self._colour_lo.editingFinished.connect(self._on_colour_changed)
        self._colour_hi.editingFinished.connect(self._on_colour_changed)
        self._colour_reset.clicked.connect(lambda: self._colour_column.setCurrentText(_OWN_COLOURS))
        row.setVisible(False)
        self._colour_guard = False
        return row

    def colour_status(self) -> str:
        return self._colour_status.text()

    def _on_colour_auto(self, on: bool) -> None:
        for spin in (self._colour_lo, self._colour_hi):
            spin.setEnabled(not on)
        self._on_colour_changed()

    def _sync_colour_row(self, result: _Result) -> None:
        """Offer *Colour labels by* for a per-label result whose layer is still there."""
        columns = result.numeric_columns()
        layer = result.label_layer()
        usable = bool(columns) and layer is not None and (
            self._viewer is None or layer in self._viewer.layers)
        self._colour_row.setVisible(usable)
        if not usable:
            return
        self._colour_guard = True
        try:
            self._colour_column.clear()
            self._colour_column.addItems([_OWN_COLOURS, *columns])
            state = result.colour
            self._colour_column.setCurrentText(state["column"] if state["column"] in columns else _OWN_COLOURS)
            self._colour_map.setCurrentText(state["colormap"])
            self._colour_auto.setChecked(bool(state["auto"]))
            for spin in (self._colour_lo, self._colour_hi):
                spin.setEnabled(not state["auto"])
            if state["limits"] is not None:
                self._colour_lo.setValue(float(state["limits"][0]))
                self._colour_hi.setValue(float(state["limits"][1]))
            self._colour_legend.setPixmap(_gradient(self._colour_map.currentText()))
            if state["column"] in columns and state["limits"] is not None:
                lo, hi = state["limits"]
                self._colour_status.setText(f"“{layer.name}” coloured by {state['column']} ({lo:.4g} – {hi:.4g}).")
            else:
                self._colour_status.setText(f"Colour the {len(result.row_labels)} measured label(s) of "
                                            f"“{layer.name}” by a column.")
        finally:
            self._colour_guard = False

    def _colour_layer(self, result: _Result) -> Any:
        """The result's label layer, as a Labels layer (an Image mask is converted)."""
        from nvitk.gui.labels.visibility import ensure_labels_layer, supports_per_label_color

        layer = result.label_layer()
        if layer is None or supports_per_label_color(layer):
            return layer
        if self._viewer is None:
            raise TypeError("Per-label colours need a Labels layer.")
        labels = ensure_labels_layer(self._viewer, layer)
        result.label_ref = weakref.ref(labels)
        return labels

    def _on_colour_changed(self) -> None:
        """Colour (or give back) the labels as the row's controls say."""
        result = self._shown
        if self._colour_guard or result is None or not self._colour_row.isVisible():
            return
        from nvitk.gui.labels.visibility import (
            apply_label_value_colors,
            label_color_state,
            restore_label_color_state,
        )

        column = self._colour_column.currentText()
        colormap = self._colour_map.currentText()
        self._colour_legend.setPixmap(_gradient(colormap))
        auto = self._colour_auto.isChecked()
        result.colour.update(column="" if column == _OWN_COLOURS else column, colormap=colormap, auto=auto)
        try:
            layer = self._colour_layer(result)
            if layer is None:
                self._colour_status.setText("The measured layer is gone.")
                return
            if column == _OWN_COLOURS:
                original = _ORIGINAL_COLOURS.pop(layer, None)
                if original is not None:
                    restore_label_color_state(layer, original)
                    self._colour_status.setText(f"“{layer.name}”: its own label colours again.")
                self._row_swatches(result)
                return
            values = result.label_values(column)
            if layer not in _ORIGINAL_COLOURS:
                _ORIGINAL_COLOURS[layer] = label_color_state(layer)
            limits = None if auto else (self._colour_lo.value(), self._colour_hi.value())
            lo, hi = apply_label_value_colors(layer, values, colormap, limits=limits)
        except Exception as exc:  # noqa: BLE001 — a colouring must never break the window
            self._colour_status.setText(f"Could not colour the labels: {exc}")
            return
        result.colour["limits"] = (lo, hi)
        if auto:
            self._colour_guard = True
            self._colour_lo.setValue(lo)
            self._colour_hi.setValue(hi)
            self._colour_guard = False
        from nvitk.gui.labels.visibility import layer_label_ids

        missing = len(set(layer_label_ids(layer)) - set(values))
        self._colour_status.setText(
            f"{len(values)} label(s) of “{layer.name}” coloured by {column} ({lo:.4g} – {hi:.4g})"
            + (f"; {missing} without a value in grey." if missing else ".")
        )
        self._row_swatches(result)

    def _row_swatches(self, result: _Result) -> None:
        """A swatch of each label row's current label colour, beside its key."""
        layer = result.label_layer()
        if layer is None or not result.row_labels or result is not self._shown:
            return
        from nvitk.gui.labels.visibility import get_label_color

        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None or item.text() not in result.row_labels:
                continue
            try:
                item.setIcon(_swatch_icon(get_label_color(layer, result.row_labels[item.text()])))
            except Exception:  # noqa: BLE001
                continue

    def add_result(
        self,
        title: str,
        payload: Mapping[Any, Any],
        *,
        subtitle: str = "",
        row_header: str = "",
        note: str = "",
        on_select: Callable[[str | None], None] | None = None,
        label_layer: Any = None,
        row_labels: Mapping[Any, int] | None = None,
    ) -> None:
        """Show *payload* and keep the previous results in the history picker.

        With *label_layer* and *row_labels* (row key → label id), the labels can be
        coloured by any measured column."""
        if on_select is not None and not note:
            note = "Click a row to show it in the viewer."
        result = _Result(title, payload, subtitle=subtitle, row_header=row_header, note=note, on_select=on_select,
                         label_layer=label_layer, row_labels=row_labels)
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

    def _notify_row(self, result: _Result | None, key: str | None) -> None:
        """Tell *result*'s owner which row is selected (``None``: none)."""
        if result is None or result.on_select is None:
            return
        try:
            result.on_select(key)
        except Exception:  # noqa: BLE001 — a highlight must never break the window
            pass

    def _on_row_selected(self) -> None:
        if self._filling or self._shown is None or self._shown.on_select is None:
            return
        rows = {i.row() for i in self._table.selectedItems()}
        if not rows:
            self._notify_row(self._shown, None)
            return
        item = self._table.item(min(rows), 0)
        self._notify_row(self._shown, item.text() if item is not None else None)

    def _unhighlight(self) -> None:
        self._filling = True
        self._table.clearSelection()
        self._filling = False
        self._notify_row(self._shown, None)

    def hideEvent(self, event: Any) -> None:  # noqa: N802 — Qt naming
        """A closed window leaves nothing highlighted."""
        self._notify_row(self._shown, None)
        super().hideEvent(event)

    def _show(self, result: _Result) -> None:
        """Lay *result* out in the table."""
        if self._shown is not None and self._shown is not result:
            self._notify_row(self._shown, None)
        self._shown = result
        self._clear_highlight.setVisible(result.on_select is not None)
        self._filling = True
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
        self._table.clearSelection()
        self._filling = False
        self._sync_colour_row(result)
        self._row_swatches(result)


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
        window._viewer = viewer
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
    on_select: Callable[[str | None], None] | None = None,
    label_layer: Any = None,
    row_labels: Mapping[Any, int] | None = None,
) -> str:
    """Show *payload* in the viewer's results window; return it as text.

    With *on_select*, selecting a row calls it with the row's key (and ``None``
    when the highlight should go). With *label_layer* and *row_labels* (row key →
    label id), the window offers to colour those labels by a measured column.

    The text is returned rather than logged here so the caller keeps control of
    the log — every measurement already writes one line, and this must not
    duplicate it.
    """
    text = _Result(title, payload, subtitle=subtitle, row_header=row_header, note=note).as_text()
    window = results_window(viewer)
    if window is None:
        return text
    window.add_result(title, payload, subtitle=subtitle, row_header=row_header, note=note, on_select=on_select,
                      label_layer=label_layer, row_labels=row_labels)
    window.show()
    window.raise_()
    return text


__all__ = ["COUNT_METRICS", "VALUE_COLORMAPS", "ResultsWindow", "results_window", "show_results"]
