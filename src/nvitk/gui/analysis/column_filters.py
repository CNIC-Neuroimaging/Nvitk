"""Spreadsheet-style column filters for a pandas-backed table view.

Right-click a column header for its menu:

* sort ascending / descending;
* **values**: the column's distinct values (among the rows the *other* filters keep) with their
  counts, a search box, All / None, and Apply: tick the ones to keep (categorical columns, and
  numeric ones with few distinct values);
* **range**: minimum / maximum for a numeric column;
* contains-text for a text column with many values;
* clear this column's filter, clear every filter, hide the column, show every column.

Right-click a cell for *Only this value* / *Exclude this value* / *Filter by this value's range*.

:class:`ColumnFilterProxy` holds the filters (one per column, by name, so they survive a new
frame with the same columns) and applies them as one pandas mask, so 100 000 rows filter at once;
:meth:`ColumnFilterProxy.filter_frame` applies the same filters to any frame (the full table
behind a capped display). :class:`FilterChips` lists the active filters, each with a ✕.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QModelIndex, QObject, QSortFilterProxyModel, Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from nvitk.db.frame_filters import EMPTY, _key_series, _numeric, describe_filter, filter_mask
from nvitk.gui.core import design

#: Columns with at most this many distinct values get the value checklist.
MAX_LIST_VALUES = 2000
#: A numeric column with at most this many distinct values also gets the checklist.
NUMERIC_LIST_MAX = 30



class ColumnFilterProxy(QSortFilterProxyModel):
    """A sort/filter proxy over a model with a ``frame()`` method, with per-column filters."""

    filters_changed = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._filters: dict[str, dict[str, Any]] = {}
        self._mask: np.ndarray | None = None

    # filters
    def filters(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._filters.items()}

    def set_column_filter(self, column: str, spec: dict[str, Any] | None) -> None:
        if spec is None:
            self._filters.pop(column, None)
        else:
            self._filters[column] = dict(spec)
        self._refilter()

    def clear_filters(self) -> None:
        if self._filters:
            self._filters.clear()
            self._refilter()

    def set_filters(self, filters: dict[str, dict[str, Any]]) -> None:
        self._filters = {k: dict(v) for k, v in filters.items()}
        self._refilter()

    def prune(self, columns: Any) -> None:
        """Drop the filters on columns not in *columns* (a new table), quietly."""
        keep = set(map(str, columns))
        self._filters = {c: f for c, f in self._filters.items() if c in keep}
        self._mask = None

    def frame(self) -> pd.DataFrame:
        source = self.sourceModel()
        return source.frame() if source is not None and hasattr(source, "frame") else pd.DataFrame()

    def filter_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """*frame* with the column filters applied (not the quick text filter)."""
        if not self._filters:
            return frame
        return frame[filter_mask(frame, self._filters)]

    def _refilter(self) -> None:
        self._mask = None
        self.invalidateFilter() if hasattr(self, "invalidateFilter") else self.invalidate()
        self.headerDataChanged.emit(Qt.Horizontal, 0, max(self.columnCount() - 1, 0))
        self.filters_changed.emit()

    def setSourceModel(self, model: Any) -> None:  # noqa: N802
        super().setSourceModel(model)
        model.modelReset.connect(self._source_reset)
        model.dataChanged.connect(lambda *_a: setattr(self, "_mask", None))

    def _source_reset(self) -> None:
        self._mask = None

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802
        if self._filters:
            if self._mask is None or len(self._mask) != self.sourceModel().rowCount():
                self._mask = filter_mask(self.frame(), self._filters)
            if row < len(self._mask) and not self._mask[row]:
                return False
        return super().filterAcceptsRow(row, parent)

    def headerData(self, section: int, orientation: Any, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if orientation == Qt.Horizontal and role in (Qt.DecorationRole, Qt.ToolTipRole):
            frame = self.frame()
            if 0 <= section < len(frame.columns):
                column = str(frame.columns[section])
                spec = self._filters.get(column)
                if spec is not None:
                    if role == Qt.DecorationRole:
                        from nvitk.gui.core.icons import icon

                        return icon("filter", design.COLOR_ACCENT)
                    return "Filtered: " + describe_filter(column, spec) + "\nRight-click for the column's filter."
                if role == Qt.ToolTipRole:
                    return f"{column}\nRight-click to filter, sort or hide."
        return super().headerData(section, orientation, role)


class _ValueList(QWidget):
    """The value checklist inside the header menu."""

    def __init__(self, counts: pd.Series, checked: set[str] | None, on_apply: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_apply = on_apply
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)
        self._search = QLineEdit()
        self._search.setPlaceholderText(f"Search {len(counts):,} values…")
        self._search.setClearButtonEnabled(True)
        lay.addWidget(self._search)
        self._list = QListWidget()
        self._list.setMinimumSize(240, 180)
        self._list.setMaximumHeight(320)
        for value, n in counts.items():
            item = QListWidgetItem(f"{value}   ({n:,})")
            item.setData(Qt.UserRole, str(value))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if checked is None or str(value) in checked else Qt.Unchecked)
            self._list.addItem(item)
        lay.addWidget(self._list)
        row = QHBoxLayout()
        for text, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, s=state: self._set_all(s))
            row.addWidget(button)
        only = QPushButton("Only shown")
        only.setToolTip("Keep just the values the search shows")
        only.clicked.connect(self._only_shown)
        row.addWidget(only)
        row.addStretch(1)
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self._apply)
        row.addWidget(apply)
        lay.addLayout(row)
        self._search.textChanged.connect(self._filter)
        self._search.returnPressed.connect(self._apply)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(needle) and needle not in str(item.data(Qt.UserRole)).lower())

    def _set_all(self, state: Any) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(state)

    def _only_shown(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Unchecked if item.isHidden() else Qt.Checked)
        self._apply()

    def checked(self) -> set[str]:
        return {str(self._list.item(i).data(Qt.UserRole)) for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked}

    def _apply(self) -> None:
        values = self.checked()
        self._on_apply(None if len(values) == self._list.count() else values)


class _RangeBox(QWidget):
    def __init__(self, lo: float, hi: float, current: dict[str, Any] | None, on_apply: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self._lo = QLineEdit("" if not current or current.get("lo") is None else f"{current['lo']:g}")
        self._hi = QLineEdit("" if not current or current.get("hi") is None else f"{current['hi']:g}")
        self._lo.setPlaceholderText(f"min {lo:.4g}")
        self._hi.setPlaceholderText(f"max {hi:.4g}")
        lay.addWidget(QLabel("from"))
        lay.addWidget(self._lo)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self._hi)
        apply = QPushButton("Apply")
        lay.addWidget(apply)

        def _go() -> None:
            def num(edit: QLineEdit) -> float | None:
                try:
                    return float(edit.text().replace(",", ".")) if edit.text().strip() else None
                except ValueError:
                    return None

            a, b = num(self._lo), num(self._hi)
            on_apply(None if a is None and b is None else {"kind": "range", "lo": a, "hi": b})

        apply.clicked.connect(_go)
        self._lo.returnPressed.connect(_go)
        self._hi.returnPressed.connect(_go)


class FilterChips(QWidget):
    """The active filters as removable chips, with *Clear all*."""

    def __init__(self, proxy: ColumnFilterProxy, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from nvitk.gui.core.flow_layout import FlowRow

        self._proxy = proxy
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._row = FlowRow(self)
        lay.addWidget(self._row)
        proxy.filters_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        layout = self._row.flow()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        filters = self._proxy.filters()
        self.setVisible(bool(filters))
        for column, spec in filters.items():
            chip = QToolButton()
            chip.setText(describe_filter(column, spec) + "  ✕")
            chip.setToolTip("Remove this filter")
            chip.setStyleSheet(
                f"QToolButton {{ border: 1px solid {design.COLOR_ACCENT}; border-radius: 9px; padding: 1px 8px;"
                f" font-size: {design.FONT_S}px; }}"
            )
            chip.clicked.connect(lambda _=False, c=column: self._proxy.set_column_filter(c, None))
            self._row.add(chip)
        if len(filters) > 1:
            clear = QToolButton()
            clear.setText("Clear all")
            clear.setAutoRaise(True)
            clear.clicked.connect(self._proxy.clear_filters)
            self._row.add(clear)


class _HeaderFilterMenus(QObject):
    """The header and cell menus of one table view.

    The slots are methods of this QObject (a child of the view), not closures: a closure that
    captures the view or its header forms a reference cycle with the signal connection that
    Python's garbage collector breaks, clearing the closure while PyQt still calls it, and the next
    right-click then crashed the application (a segmentation fault).
    """

    def __init__(self, view: QTableView, proxy: ColumnFilterProxy, extra_menu: Any, cell_menu: bool) -> None:
        super().__init__(view)
        self._view = view
        self._proxy = proxy
        self._extra_menu = extra_menu
        header = view.horizontalHeader()
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.setSectionsClickable(True)
        header.customContextMenuRequested.connect(self.open_header_menu)
        if cell_menu:
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(self.open_cell_menu)

    def _column_at(self, section: int) -> str | None:
        frame = self._proxy.frame()
        return str(frame.columns[section]) if 0 <= section < len(frame.columns) else None

    def open_header_menu(self, pos: Any) -> None:
        header = self._view.horizontalHeader()
        section = header.logicalIndexAt(pos)
        column = self._column_at(section)
        if column is None:
            return
        menu = QMenu(self._view)
        build_column_menu(menu, self._view, self._proxy, section, column)
        if self._extra_menu is not None:
            menu.addSeparator()
            self._extra_menu(menu, column)
        menu.exec(header.mapToGlobal(pos))
        menu.deleteLater()

    def open_cell_menu(self, pos: Any) -> None:
        view, proxy = self._view, self._proxy
        index = view.indexAt(pos)
        if not index.isValid():
            return
        source = proxy.mapToSource(index)
        frame = proxy.frame()
        if source.column() >= len(frame.columns) or source.row() >= len(frame):
            return
        column = str(frame.columns[source.column()])
        value = frame.iat[source.row(), source.column()]
        key = _key_series(pd.Series([value], dtype=frame[column].dtype))[0]
        current = proxy.filters().get(column)
        menu = QMenu(view)
        title = menu.addAction(f"{column} = {key}")
        title.setEnabled(False)
        menu.addAction("Only this value", lambda: proxy.set_column_filter(column, {"kind": "values", "values": [key]}))

        def _exclude() -> None:
            if current and current.get("kind") == "exclude":
                values = set(current.get("values") or ()) | {key}
            else:
                values = {key}
            proxy.set_column_filter(column, {"kind": "exclude", "values": sorted(values)})

        menu.addAction("Exclude this value", _exclude)
        if _numeric(frame[column]) is not None and not pd.isna(value):
            number = float(value)
            menu.addAction(f"{column} ≥ {key}", lambda: proxy.set_column_filter(column, {"kind": "range", "lo": number, "hi": None}))
            menu.addAction(f"{column} ≤ {key}", lambda: proxy.set_column_filter(column, {"kind": "range", "lo": None, "hi": number}))
        if current is not None:
            menu.addAction(f"Clear the filter on {column}", lambda: proxy.set_column_filter(column, None))
        if proxy.filters():
            menu.addAction("Clear all filters", proxy.clear_filters)
        menu.addSeparator()
        text = str(value)
        menu.addAction("Copy value", lambda: _copy(text))
        menu.exec(view.viewport().mapToGlobal(pos))
        menu.deleteLater()


def install_header_filters(view: QTableView, proxy: ColumnFilterProxy, *, extra_menu: Any = None,
                           cell_menu: bool = True) -> None:
    """Give *view*'s header (and, with *cell_menu*, its cells) the filter menus.

    *extra_menu(menu, column_name)*, when given, adds caller entries to the header menu (the import
    editor's rename / role / variable entries).
    """
    # Kept on the view as well as parented to it: it lives exactly as long as the table.
    view._nvitk_filter_menus = _HeaderFilterMenus(view, proxy, extra_menu, cell_menu)


def _copy(text: str) -> None:
    from qtpy.QtWidgets import QApplication

    QApplication.clipboard().setText(text)


def build_column_menu(menu: QMenu, view: QTableView, proxy: ColumnFilterProxy, section: int, column: str) -> None:
    """Fill *menu* with the sort / filter / hide entries for *column*."""
    frame = proxy.frame()
    series = frame[column]
    title = menu.addAction(f"Column: {column}")
    title.setEnabled(False)
    menu.addAction("Sort ascending", lambda: view.sortByColumn(section, Qt.AscendingOrder))
    menu.addAction("Sort descending", lambda: view.sortByColumn(section, Qt.DescendingOrder))
    menu.addSeparator()
    current = proxy.filters().get(column)
    # The values the other filters leave, as Excel does.
    others = filter_mask(frame, proxy.filters(), skip=column)
    visible = series[others]
    keys = _key_series(visible)
    counts = keys.value_counts()
    numeric = _numeric(series)
    few = numeric is None or counts.size <= NUMERIC_LIST_MAX

    def _set(spec: dict[str, Any] | None) -> None:
        proxy.set_column_filter(column, spec)
        menu.close()

    if few and counts.size <= MAX_LIST_VALUES:
        if numeric is not None:
            counts = counts.reindex(sorted(counts.index, key=lambda v: (v == EMPTY, float(v) if v != EMPTY else 0.0)))
        checked = None
        if current is not None and current.get("kind") == "values":
            checked = set(current.get("values") or ())
        elif current is not None and current.get("kind") == "exclude":
            checked = set(counts.index) - set(current.get("values") or ())
        box = _ValueList(counts, checked, lambda values: _set(None if values is None else {"kind": "values", "values": sorted(values)}))
        action = QWidgetAction(menu)
        action.setDefaultWidget(box)
        menu.addAction(action)
    if numeric is not None:
        finite = pd.to_numeric(visible, errors="coerce").dropna()
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 0.0)
        label = menu.addAction(f"Range ({lo:.4g} – {hi:.4g})")
        label.setEnabled(False)
        box = _RangeBox(lo, hi, current if current and current.get("kind") == "range" else None, _set)
        action = QWidgetAction(menu)
        action.setDefaultWidget(box)
        menu.addAction(action)
    elif counts.size > MAX_LIST_VALUES:
        label = menu.addAction(f"{counts.size:,} distinct values: filter by text")
        label.setEnabled(False)
        edit = QLineEdit(str(current.get("text") or "") if current and current.get("kind") == "text" else "")
        edit.setPlaceholderText("contains…")
        edit.returnPressed.connect(lambda: _set({"kind": "text", "text": edit.text()} if edit.text().strip() else None))
        action = QWidgetAction(menu)
        action.setDefaultWidget(edit)
        menu.addAction(action)
    menu.addSeparator()
    clear = menu.addAction(f"Clear the filter on {column}", lambda: proxy.set_column_filter(column, None))
    clear.setEnabled(current is not None)
    clear_all = menu.addAction("Clear all filters", proxy.clear_filters)
    clear_all.setEnabled(bool(proxy.filters()))
    menu.addSeparator()
    menu.addAction("Hide this column", lambda: view.setColumnHidden(section, True))
    hidden = [i for i in range(len(frame.columns)) if view.isColumnHidden(i)]
    show = menu.addAction(f"Show all columns ({len(hidden)} hidden)",
                          lambda: [view.setColumnHidden(i, False) for i in hidden])
    show.setEnabled(bool(hidden))


__all__ = [
    "ColumnFilterProxy",
    "FilterChips",
    "build_column_menu",
    "describe_filter",
    "filter_mask",
    "install_header_filters",
]
