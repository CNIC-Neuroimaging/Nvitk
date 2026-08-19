"""
Analysis-dataframe table with column-anchored filtering.

Description
-----------
The analysis frame is where filtering belongs: you look at a ``territory`` column, see ``RPComm``
and ``LPComm``, and want them gone. This module makes that a two-click operation — right-click the
column header, uncheck the levels — instead of typing a column name, an operator and a value into a
detached form.

Widgets
-------
``PandasFrameModel``   read-only Qt model over a DataFrame, with a numeric sort role so ``-1.2e-05``
                       sorts as a number rather than as text.
``AnalysisFrameView``  the table plus its header context menu (filter, sort, transform, plot-x).
``ColumnFilterDialog`` per-column filter editor: level checklist, numeric range/comparison, IQR.
``FilterChipBar``      the active rules, each removable, with the running row count.

The rules themselves live in :mod:`nvitk.stats.frame_ops` so notebooks can apply the same filter set.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nvitk.stats.frame_ops import (
    COLUMN_TYPES,
    DEFAULT_IQR_K,
    FILTER_OPS,
    TRANSFORM_LABELS,
    FilterRule,
)

from nvitk.stats.interactive import COLUMN_PLOT_KINDS
from nvitk.stats.qc_filters import available_presets

from .constants import MAX_CATEGORICAL_LEVELS, TABLE_ROW_CAP
from .theme import COLOR_ACCENT, COLOR_MUTED, muted_label_style


# ──────────────────────────────────────────────────────────────────────────────
# Table model
# ──────────────────────────────────────────────────────────────────────────────
class PandasFrameModel(QAbstractTableModel):
    """
    Read-only Qt model over a pandas frame.

    Numeric cells are formatted with ``%.6g`` for display but expose their raw float under
    :data:`SORT_ROLE`, so a :class:`~qtpy.QtCore.QSortFilterProxyModel` configured with that role
    orders them numerically. Sorting on the display strings would put ``-1.2e-05`` next to ``-1.2``.
    """

    SORT_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        """Start empty; call :meth:`set_frame` to populate."""
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._filtered: set[str] = set()

    # ---- population -----------------------------------------------------------
    def set_frame(self, df: pd.DataFrame | None, *, filtered_columns: set[str] = frozenset()) -> None:
        """Show *df* (capped at :data:`TABLE_ROW_CAP` rows) and badge *filtered_columns* in the header."""
        self.beginResetModel()
        self._df = pd.DataFrame() if df is None else df.head(TABLE_ROW_CAP)
        self._filtered = set(filtered_columns)
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        """The (capped) frame currently displayed."""
        return self._df

    # ---- QAbstractTableModel --------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Number of displayed rows (0 while a parent index is given, as Qt requires for tables)."""
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Number of columns."""
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """Display text, right-alignment for numerics, and the raw value under :data:`SORT_ROLE`."""
        if not index.isValid():
            return None
        value = self._df.iloc[index.row(), index.column()]

        if role == self.SORT_ROLE:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return float(value)
            return str(value)
        if role == Qt.TextAlignmentRole:
            numeric = pd.api.types.is_numeric_dtype(self._df.dtypes.iloc[index.column()])
            return int(Qt.AlignRight | Qt.AlignVCenter) if numeric else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ForegroundRole and pd.isna(value):
            return QColor(COLOR_MUTED)
        if role != Qt.DisplayRole:
            return None

        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def headerData(self, section: int, orientation: int, role: int = Qt.DisplayRole) -> Any:
        """Column names (marked with a funnel when filtered) and 1-based row numbers."""
        if orientation == Qt.Horizontal:
            if section >= len(self._df.columns):
                return None
            name = str(self._df.columns[section])
            if role == Qt.DisplayRole:
                return f"⧩ {name}" if name in self._filtered else name
            if role == Qt.ToolTipRole:
                dtype = self._df.dtypes.iloc[section]
                suffix = "  ·  filtered" if name in self._filtered else ""
                return f"{name}  ({dtype}){suffix}"
            if role == Qt.ForegroundRole and name in self._filtered:
                return QColor(COLOR_ACCENT)
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-column filter dialog
# ──────────────────────────────────────────────────────────────────────────────
class ColumnFilterDialog(QDialog):
    """
    Build the filter rules for one column.

    Three tabs, each producing at most one rule: **Levels** (checkable list, keep or exclude — the
    vessel-exclusion case), **Numeric** (range and/or an operator comparison), and **IQR** (Tukey
    fences, optionally computed within each level of a scope column).

    The dialog is seeded from the rules already active on the column, so reopening it edits rather
    than stacking duplicates.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        column: str,
        series: pd.Series,
        scope_columns: Sequence[str],
        existing: Sequence[FilterRule] = (),
    ) -> None:
        """Populate the tabs from *series* and pre-fill them from *existing*."""
        super().__init__(parent)
        self.setWindowTitle(f"Filter — {column}")
        self.resize(420, 520)
        self._column = column
        self._series = series
        self._numeric = pd.to_numeric(series, errors="coerce")
        self._is_numeric = bool(self._numeric.notna().any())

        lay = QVBoxLayout(self)
        summary = QLabel(self._summary_text())
        summary.setWordWrap(True)
        summary.setStyleSheet(muted_label_style())
        lay.addWidget(summary)

        self._tabs = QTabWidget()
        lay.addWidget(self._tabs, stretch=1)
        self._build_levels_tab()
        self._build_numeric_tab(scope_columns)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._apply_existing(existing)

    def _summary_text(self) -> str:
        """One-line description of the column's content, to orient the choice of filter."""
        n_unique = int(self._series.nunique(dropna=True))
        n_missing = int(self._series.isna().sum())
        parts = [f"{len(self._series)} rows", f"{n_unique} distinct"]
        if n_missing:
            parts.append(f"{n_missing} missing")
        if self._is_numeric:
            parts.append(f"range {self._numeric.min():.6g} … {self._numeric.max():.6g}")
        return "  ·  ".join(parts)

    # ---- tabs -----------------------------------------------------------------
    def _build_levels_tab(self) -> None:
        """Checkable list of the column's distinct values, with a keep/exclude switch."""
        page = QWidget()
        lay = QVBoxLayout(page)
        levels = sorted(str(v) for v in self._series.dropna().unique())

        self._levels_enabled = QCheckBox("Filter by level")
        lay.addWidget(self._levels_enabled)

        if len(levels) > MAX_CATEGORICAL_LEVELS:
            note = QLabel(
                f"{len(levels)} distinct values — too many to pick from a list. "
                "Use the Numeric tab, or a derived column."
            )
            note.setWordWrap(True)
            note.setStyleSheet(muted_label_style())
            lay.addWidget(note)
            self._levels_enabled.setEnabled(False)
            self._level_list = None
            self._levels_mode = None
            self._tabs.addTab(page, "Levels")
            return

        self._levels_mode = QComboBox()
        self._levels_mode.addItem("Keep only the checked levels", False)
        self._levels_mode.addItem("Exclude the checked levels", True)
        lay.addWidget(self._levels_mode)

        search = QLineEdit()
        search.setPlaceholderText("Search levels…")
        lay.addWidget(search)

        self._level_list = QListWidget()
        self._level_list.setSelectionMode(QAbstractItemView.NoSelection)
        for level in levels:
            item = QListWidgetItem(level)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self._level_list.addItem(item)
        lay.addWidget(self._level_list, stretch=1)
        search.textChanged.connect(self._on_level_search)

        row = QHBoxLayout()
        for label, action in (("All", "all"), ("None", "none"), ("Invert", "invert")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, a=action: self._set_all_levels(a))
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        # Checking anything is a clear signal the user wants this rule active.
        self._level_list.itemChanged.connect(
            lambda *_: self._levels_enabled.setChecked(True)
        )
        self._tabs.addTab(page, "Levels")

    def _build_numeric_tab(self, scope_columns: Sequence[str]) -> None:
        """Range bounds, an operator comparison, and the IQR outlier filter."""
        page = QWidget()
        lay = QVBoxLayout(page)

        # ---- range ------------------------------------------------------------
        self._range_enabled = QCheckBox("Keep values in a range")
        lay.addWidget(self._range_enabled)
        range_form = QFormLayout()
        self._range_low = QLineEdit()
        self._range_high = QLineEdit()
        if self._is_numeric:
            self._range_low.setPlaceholderText(f"{self._numeric.min():.6g}")
            self._range_high.setPlaceholderText(f"{self._numeric.max():.6g}")
        self._range_low.setToolTip("Leave blank for no lower bound.")
        self._range_high.setToolTip("Leave blank for no upper bound.")
        range_form.addRow("min", self._range_low)
        range_form.addRow("max", self._range_high)
        lay.addLayout(range_form)

        # ---- comparison -------------------------------------------------------
        self._compare_enabled = QCheckBox("Compare against a value")
        lay.addWidget(self._compare_enabled)
        compare_row = QHBoxLayout()
        self._compare_op = QComboBox()
        for op in FILTER_OPS:
            self._compare_op.addItem(op)
        self._compare_value = QLineEdit()
        compare_row.addWidget(self._compare_op)
        compare_row.addWidget(self._compare_value, stretch=1)
        lay.addLayout(compare_row)

        # ---- IQR --------------------------------------------------------------
        self._iqr_enabled = QCheckBox("Drop IQR outliers")
        self._iqr_enabled.setToolTip(
            "Drop values outside [Q1 - k·IQR, Q3 + k·IQR].\n"
            "A per-level scope computes the fences within each level of the scope column — the "
            "right choice when levels differ in magnitude (e.g. LMCA ~200 vs LPCOMM ~20 mL/min), "
            "since a global fence is set by the spread between levels and misses outliers within "
            "them."
        )
        lay.addWidget(self._iqr_enabled)
        iqr_form = QFormLayout()
        self._iqr_k = QDoubleSpinBox()
        self._iqr_k.setRange(0.1, 10.0)
        self._iqr_k.setSingleStep(0.25)
        self._iqr_k.setValue(DEFAULT_IQR_K)
        self._iqr_scope = QComboBox()
        self._iqr_scope.addItem("global", None)
        for name in scope_columns:
            if name != self._column:
                self._iqr_scope.addItem(f"per {name}", name)
        iqr_form.addRow("k", self._iqr_k)
        iqr_form.addRow("scope", self._iqr_scope)
        lay.addLayout(iqr_form)

        lay.addStretch(1)
        if not self._is_numeric:
            note = QLabel("This column is not numeric — range and IQR will match nothing.")
            note.setWordWrap(True)
            note.setStyleSheet(muted_label_style())
            lay.addWidget(note)
        self._tabs.addTab(page, "Numeric")

    # ---- level list helpers ---------------------------------------------------
    def _on_level_search(self, text: str) -> None:
        """Hide level items that do not contain *text*."""
        if self._level_list is None:
            return
        needle = str(text or "").strip().lower()
        for i in range(self._level_list.count()):
            item = self._level_list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _set_all_levels(self, action: str) -> None:
        """Bulk check/uncheck/invert the *visible* level items."""
        if self._level_list is None:
            return
        for i in range(self._level_list.count()):
            item = self._level_list.item(i)
            if item.isHidden():
                continue
            if action == "all":
                item.setCheckState(Qt.Checked)
            elif action == "none":
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(
                    Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                )

    def _apply_existing(self, existing: Sequence[FilterRule]) -> None:
        """Pre-fill the tabs from the rules currently active on this column."""
        for rule in existing:
            if rule.kind == "values" and self._level_list is not None:
                self._levels_enabled.setChecked(True)
                self._levels_mode.setCurrentIndex(1 if rule.exclude else 0)
                wanted = {str(v) for v in rule.values}
                for i in range(self._level_list.count()):
                    item = self._level_list.item(i)
                    item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
                self._tabs.setCurrentIndex(0)
            elif rule.kind == "range":
                self._range_enabled.setChecked(True)
                self._range_low.setText("" if rule.low is None else f"{rule.low:g}")
                self._range_high.setText("" if rule.high is None else f"{rule.high:g}")
                self._tabs.setCurrentIndex(1)
            elif rule.kind == "compare":
                self._compare_enabled.setChecked(True)
                idx = self._compare_op.findText(rule.op)
                if idx >= 0:
                    self._compare_op.setCurrentIndex(idx)
                self._compare_value.setText(rule.value)
                self._tabs.setCurrentIndex(1)
            elif rule.kind == "iqr":
                self._iqr_enabled.setChecked(True)
                self._iqr_k.setValue(float(rule.k))
                scope_idx = self._iqr_scope.findData(rule.by)
                if scope_idx >= 0:
                    self._iqr_scope.setCurrentIndex(scope_idx)
                self._tabs.setCurrentIndex(1)

    # ---- result ---------------------------------------------------------------
    def rules(self) -> list[FilterRule]:
        """The rules the user configured, in application order (levels → range → compare → IQR)."""
        out: list[FilterRule] = []

        if self._level_list is not None and self._levels_enabled.isChecked():
            checked = tuple(
                self._level_list.item(i).text()
                for i in range(self._level_list.count())
                if self._level_list.item(i).checkState() == Qt.Checked
            )
            if checked:
                out.append(
                    FilterRule(
                        column=self._column,
                        kind="values",
                        values=checked,
                        exclude=bool(self._levels_mode.currentData()),
                    )
                )

        if self._range_enabled.isChecked():
            low = _parse_float(self._range_low.text())
            high = _parse_float(self._range_high.text())
            if low is not None or high is not None:
                out.append(FilterRule(column=self._column, kind="range", low=low, high=high))

        if self._compare_enabled.isChecked() and self._compare_value.text().strip():
            out.append(
                FilterRule(
                    column=self._column,
                    kind="compare",
                    op=self._compare_op.currentText(),
                    value=self._compare_value.text().strip(),
                )
            )

        if self._iqr_enabled.isChecked():
            out.append(
                FilterRule(
                    column=self._column,
                    kind="iqr",
                    k=float(self._iqr_k.value()),
                    by=self._iqr_scope.currentData(),
                )
            )
        return out


def _parse_float(text: str) -> float | None:
    """Parse *text* as a float, or ``None`` when blank/unparseable."""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Active-filter chips
# ──────────────────────────────────────────────────────────────────────────────
class FilterChipBar(QWidget):
    """
    Horizontal strip of the active filter rules, each removable, plus the running row count.

    Rules that could not be applied (their column left the frame after a reload) are shown greyed
    with the reason, rather than dropped — a filter you cannot see is a filter you cannot fix.
    """

    rulesChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the scrollable chip strip and its clear-all button."""
        super().__init__(parent)
        self._rules: list[FilterRule] = []

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFixedHeight(34)
        self._host = QWidget()
        self._host_layout = QHBoxLayout(self._host)
        self._host_layout.setContentsMargins(0, 0, 0, 0)
        self._host_layout.setSpacing(4)
        self._scroll.setWidget(self._host)
        lay.addWidget(self._scroll, stretch=1)

        self._count = QLabel("")
        self._count.setStyleSheet(muted_label_style())
        lay.addWidget(self._count)

        self._btn_clear = QPushButton("Clear all")
        self._btn_clear.clicked.connect(self._on_clear)
        lay.addWidget(self._btn_clear)
        self._sync_clear_enabled()

    def rules(self) -> list[FilterRule]:
        """Currently held rules."""
        return list(self._rules)

    def set_rules(self, rules: Sequence[FilterRule], report: Sequence[dict[str, Any]] = ()) -> None:
        """Replace the rules and redraw the chips, annotating each with its effect from *report*."""
        self._rules = list(rules)
        by_index = {i: entry for i, entry in enumerate(report)}

        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._rules:
            hint = QLabel("No filters — right-click a column header to add one.")
            hint.setStyleSheet(muted_label_style())
            self._host_layout.addWidget(hint)
        for i, rule in enumerate(self._rules):
            self._host_layout.addWidget(self._make_chip(i, rule, by_index.get(i)))
        self._host_layout.addStretch(1)
        self._sync_clear_enabled()

    def set_counts(self, n_filtered: int, n_total: int) -> None:
        """Show ``m of n rows`` next to the chips."""
        if n_filtered == n_total:
            self._count.setText(f"{n_total} rows")
        else:
            self._count.setText(f"{n_filtered} of {n_total} rows")

    def _make_chip(self, index: int, rule: FilterRule, entry: dict[str, Any] | None) -> QWidget:
        """One chip: the rule label, its row effect, and a remove button."""
        chip = QWidget()
        lay = QHBoxLayout(chip)
        lay.setContentsMargins(8, 2, 4, 2)
        lay.setSpacing(4)

        skipped = bool(entry and entry.get("skipped"))
        removed = int(entry.get("removed", 0)) if entry else 0
        label = QLabel(rule.label())
        if skipped:
            label.setToolTip(f"Not applied: {entry.get('reason', '')}")
        elif entry:
            label.setToolTip(f"Removed {removed} row(s).")
        lay.addWidget(label)

        if entry and not skipped and removed:
            effect = QLabel(f"−{removed}")
            effect.setStyleSheet(muted_label_style())
            lay.addWidget(effect)

        remove = QPushButton("✕")
        remove.setFixedWidth(22)
        remove.setToolTip("Remove this filter")
        remove.clicked.connect(lambda _=False, i=index: self._on_remove(i))
        lay.addWidget(remove)

        border = "#666" if skipped else COLOR_ACCENT
        color = COLOR_MUTED if skipped else "#e0e0e0"
        chip.setStyleSheet(
            f"QWidget {{ border: 1px solid {border}; border-radius: 10px; }} "
            f"QLabel {{ border: none; color: {color}; font-weight: normal; }}"
        )
        return chip

    def _on_remove(self, index: int) -> None:
        """Drop the rule at *index* and notify."""
        if 0 <= index < len(self._rules):
            del self._rules[index]
            self.rulesChanged.emit()

    def _on_clear(self) -> None:
        """Drop every rule and notify."""
        if self._rules:
            self._rules = []
            self.rulesChanged.emit()

    def _sync_clear_enabled(self) -> None:
        """Enable the clear-all button only when there is something to clear."""
        self._btn_clear.setEnabled(bool(self._rules))


# ──────────────────────────────────────────────────────────────────────────────
# Table view
# ──────────────────────────────────────────────────────────────────────────────
class AnalysisFrameView(QWidget):
    """
    The analysis-dataframe table and its column header menu.

    Signals
    -------
    filtersRequested(column)         open the filter dialog for a column
    clearFiltersRequested(column)    drop every rule anchored to a column
    transformRequested(column, key)  add a derived column applying a canned transform
    binsRequested(column)            open the derived-columns editor to bin a column into groups
    plotXRequested(column)           use a column as the plot's x axis
    publishRequested(column)         write a derived column back into the dataset
    columnPlotRequested(column, kind) open the distribution viewer for a column
    qcFilterRequested(column, key)   apply a ready-made quality-control filter
    summaryRequested(column, by)    export per-group descriptive statistics for a column
    dropRequested(column)            remove a column from the analysis frame
    restoreRequested()               put every dropped column back
    typeChangeRequested(column, kind) recast a column (numeric / factor / …)
    referenceRequested(column, level) make a level the model's reference
    subjectPlotRequested(subject_uid) open the per-subject viewer for one subject
    """

    filtersRequested = Signal(str)
    clearFiltersRequested = Signal(str)
    transformRequested = Signal(str, str)
    binsRequested = Signal(str)
    plotXRequested = Signal(str)
    publishRequested = Signal(str)
    columnPlotRequested = Signal(str, str)
    qcFilterRequested = Signal(str, str)
    summaryRequested = Signal(str, str)
    dropRequested = Signal(str)
    restoreRequested = Signal()
    typeChangeRequested = Signal(str, str)
    referenceRequested = Signal(str, str)
    subjectPlotRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the table view over a sorting proxy and wire the header context menu."""
        super().__init__(parent)
        self._dropped: set[str] = set()
        self._column_types: dict[str, str] = {}
        self._references: dict[str, str] = {}
        self._model = PandasFrameModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(PandasFrameModel.SORT_ROLE)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.horizontalHeader().customContextMenuRequested.connect(self._on_header_menu)
        # A cell menu as well as a header one: the header answers "what about this variable", the
        # cell answers "what about this row" — and for the subject column that is a different and
        # frequently-wanted question.
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_cell_menu)
        self._table.verticalHeader().setDefaultSectionSize(22)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._table)

        self._filtered: set[str] = set()
        self._derived: set[str] = set()

    def set_frame(
        self,
        df: pd.DataFrame | None,
        *,
        filtered_columns: set[str] = frozenset(),
        derived_columns: set[str] = frozenset(),
    ) -> None:
        """Display *df*, badging *filtered_columns*, and size the columns to their content once."""
        self._filtered = set(filtered_columns)
        self._derived = set(derived_columns)
        self._model.set_frame(df, filtered_columns=self._filtered)
        self._table.resizeColumnsToContents()

    def columns(self) -> list[str]:
        """Column names of the displayed frame."""
        return [str(c) for c in self._model.frame().columns]

    def _column_at(self, section: int) -> str:
        """Name of the column at header *section*, or ``""`` when out of range."""
        columns = self.columns()
        return columns[section] if 0 <= section < len(columns) else ""

    def _grouping_columns(self, *, exclude: str = "", max_levels: int = 60) -> list[str]:
        """
        Columns worth grouping a summary by: discrete, and not one row per level.

        A continuous measurement has as many levels as rows, and ``subject_uid`` nearly so — a
        summary keyed on either is the raw data with extra columns.
        """
        frame = self._model.frame()
        out: list[str] = []
        for name in frame.columns:
            if str(name) == str(exclude):
                continue
            series = frame[name]
            if pd.api.types.is_numeric_dtype(series) and not isinstance(
                series.dtype, pd.CategoricalDtype
            ):
                continue
            if 2 <= int(series.nunique(dropna=True)) <= int(max_levels):
                out.append(str(name))

        # Frame order puts identifier columns first, which is the wrong end of the menu: a summary
        # by territory is the common case, one per subject is the rare one.
        def rank(name: str) -> tuple[int, str]:
            """Sort key placing the usual groupings before the incidental ones."""
            preferred = ("territory", "group_key", "sex", "tacsctot_group", "visit_id")
            return (preferred.index(name), "") if name in preferred else (len(preferred), name)

        return sorted(out, key=rank)

    def _on_cell_menu(self, point) -> None:
        """
        Per-row menu, offered only where a row identifies something worth viewing on its own.

        Today that is ``subject_uid``: one row of the analysis frame is one subject × region, and
        the interesting question about a subject is what *all* their regions look like together —
        which is a picture of anatomy, not a cell value. Right-clicking anywhere else falls through
        to the header menu's territory and does nothing here, rather than offering an action that
        would not mean anything for that column.
        """
        index = self._table.indexAt(point)
        if not index.isValid():
            return
        # The view sorts through a proxy, so the visible row is not the frame's row. Map back before
        # reading anything, or a sorted table reports the wrong subject.
        source = self._proxy.mapToSource(index)
        column = self._column_at(source.column())
        if column != "subject_uid":
            return

        frame = self._model.frame()
        if source.row() >= len(frame):
            return
        subject = str(frame.iloc[source.row()][column])
        if not subject or subject.lower() == "nan":
            return

        menu = QMenu(self)
        menu.addAction(
            f"Visualize subject “{subject}”…",
            lambda: self.subjectPlotRequested.emit(subject),
        )
        menu.exec(self._table.viewport().mapToGlobal(point))

    def _on_header_menu(self, point) -> None:
        """Build and run the per-column header menu."""
        header = self._table.horizontalHeader()
        column = self._column_at(header.logicalIndexAt(point))
        if not column:
            return

        menu = QMenu(self)
        menu.addAction(f"Filter “{column}”…", lambda: self.filtersRequested.emit(column))
        clear = menu.addAction("Clear filters on this column", lambda: self.clearFiltersRequested.emit(column))
        clear.setEnabled(column in self._filtered)
        menu.addSeparator()

        frame = self._model.frame()
        numeric = column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
        transform_menu = menu.addMenu("Add derived column")
        transform_menu.setEnabled(numeric)
        for key, label in TRANSFORM_LABELS.items():
            transform_menu.addAction(
                label, lambda _=False, k=key: self.transformRequested.emit(column, k)
            )
        if not numeric:
            transform_menu.setToolTip("Transforms apply to numeric columns only.")

        bins = menu.addAction(
            "Group into bins…", lambda: self.binsRequested.emit(column)
        )
        bins.setEnabled(numeric)
        bins.setToolTip(
            "Cut this column into labelled groups (cp0, cp1, …), like tacsctot → tacsctot_group."
        )

        menu.addSeparator()
        # Only derived columns: everything else is already in the dataset, and re-publishing a
        # column read straight out of it would write a copy under a second name.
        publish = menu.addAction(
            "Save to database…", lambda: self.publishRequested.emit(column)
        )
        publish.setEnabled(column in self._derived)
        publish.setToolTip(
            "Write this derived column into the dataset as a variable, so later sessions and other "
            "tools can use it."
            if column in self._derived
            else f"“{column}” is not a derived column — it already comes from the dataset."
        )

        # Descriptive statistics of this column, one row per level of the chosen grouping.
        numeric_column = column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
        summary_menu = menu.addMenu(f"Export summary of “{column}” by")
        summary_menu.setEnabled(numeric_column)
        if numeric_column:
            groupings = self._grouping_columns(exclude=column)
            for name in groupings:
                summary_menu.addAction(
                    name, lambda _=False, g=name: self.summaryRequested.emit(column, g)
                )
            if groupings:
                summary_menu.addSeparator()
            summary_menu.addAction(
                "(whole cohort, no grouping)", lambda: self.summaryRequested.emit(column, "")
            )
        else:
            summary_menu.setToolTip("Summaries apply to numeric measurement columns only.")
        menu.addSeparator()

        qc_menu = menu.addMenu("Quality-control filter")
        presets = available_presets(self.columns(), column=column)
        for preset, state in presets:
            action = qc_menu.addAction(
                preset.label,
                lambda _=False, k=preset.key: self.qcFilterRequested.emit(column, k),
            )
            action.setEnabled(state == "ready")
            action.setToolTip(
                preset.description if state == "ready"
                else f"Needs {', '.join(preset.requires)}, which this dataset does not carry. "
                     f"Run the qvtpy stage 9 (autoqc) to publish the QC metrics, then reload."
            )
        qc_menu.setEnabled(bool(presets))
        menu.addSeparator()

        plot_menu = menu.addMenu(f"Plot “{column}”")
        for key, description in COLUMN_PLOT_KINDS.items():
            plot_menu.addAction(
                description.split("—")[0].strip(),
                lambda _=False, k=key: self.columnPlotRequested.emit(column, k),
            )

        menu.addSeparator()
        menu.addAction("Set as plot x", lambda: self.plotXRequested.emit(column))
        type_menu = menu.addMenu("Column type")
        current_kind = self._column_types.get(column, "auto")
        for key, (label, effect) in COLUMN_TYPES.items():
            action = type_menu.addAction(
                f"{label}{'  ✓' if key == current_kind else ''}",
                lambda _=False, k=key: self.typeChangeRequested.emit(column, k),
            )
            action.setToolTip(effect)
        if column in frame.columns:
            type_menu.setToolTip(f"Currently {frame[column].dtype}")

        # Reference level: only meaningful for a column with a manageable number of levels, and
        # only for the categorical ones — a continuous predictor has a slope, not contrasts.
        ref_menu = menu.addMenu("Reference level")
        levels = self._column_levels(column)
        ref_menu.setEnabled(bool(levels))
        if levels:
            current = self._references.get(column, "")
            clear = ref_menu.addAction(
                f"Default (first level){'  ✓' if not current else ''}",
                lambda: self.referenceRequested.emit(column, ""),
            )
            clear.setToolTip("Let the frame's own level order decide.")
            ref_menu.addSeparator()
            for level in levels:
                ref_menu.addAction(
                    f"{level}{'  ✓' if level == current else ''}",
                    lambda _=False, lv=level: self.referenceRequested.emit(column, lv),
                )
        else:
            ref_menu.setToolTip(
                "Only for categorical columns with a workable number of levels — set the type to "
                "Factor first if this should be one."
            )

        menu.addAction("Copy column name", lambda: QApplication.clipboard().setText(column))

        menu.addSeparator()
        drop = menu.addAction(
            f"Drop “{column}” from the frame", lambda: self.dropRequested.emit(column)
        )
        drop.setToolTip(
            "Hide this column from the analysis frame. The dataset is untouched and the column "
            "can be restored — dropping only changes what this session works with."
        )
        restore = menu.addAction(
            f"Restore {len(self._dropped)} dropped column(s)",
            lambda: self.restoreRequested.emit(),
        )
        restore.setEnabled(bool(self._dropped))
        if self._dropped:
            restore.setToolTip("Dropped: " + ", ".join(sorted(self._dropped)))
        menu.exec(header.mapToGlobal(point))

    def _column_levels(self, column: str, *, cap: int = MAX_CATEGORICAL_LEVELS) -> list[str]:
        """Distinct levels of *column*, or ``[]`` when it is not usable as a factor."""
        frame = self._model.frame()
        if column not in frame.columns:
            return []
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            return []
        levels = [str(v) for v in pd.unique(series.dropna())]
        return sorted(levels) if 1 < len(levels) <= cap else []

    def set_references(self, references: Mapping[str, str]) -> None:
        """Record the active reference levels, so the menu can tick the current one."""
        self._references = {str(k): str(v) for k, v in dict(references).items()}

    def set_column_types(self, types: Mapping[str, str]) -> None:
        """Record the active per-column type overrides, so the menu can tick the current one."""
        self._column_types = {str(k): str(v) for k, v in dict(types).items()}

    def set_dropped(self, columns: Iterable[str]) -> None:
        """Record which columns are currently dropped, for the restore entry."""
        self._dropped = {str(c) for c in columns}


__all__ = [
    "AnalysisFrameView",
    "ColumnFilterDialog",
    "FilterChipBar",
    "PandasFrameModel",
]
