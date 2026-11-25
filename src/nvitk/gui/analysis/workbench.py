"""Analysis module: explore any table (database, Results, a file), edit it, plot it, model it.

* **Source**: a dataset table (``subjects``, ``clinical_measurements``, ``image_measurements``...,
  wide-pivoted on request), the table shown in Results, or a CSV / Excel / Parquet file.
* **Table**: sortable, with a quick text filter and a pandas query; cells are editable, and an
  edited dataset table can be written back (a confirmation shows how many rows change).
* **Plot**: the :class:`~nvitk.gui.analysis.plot_builder.PlotBuilder` on the filtered rows.
* **Statistical models**: the Statmodels launcher (mixed models, OLS, GLM, mediation...), as
  before; its standalone window is kept while its tools move here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.analysis.plot_builder import PlotBuilder
from nvitk.gui.core import design
from nvitk.gui.core.widgets import Section, hint_label


#: Rows the table shows (a query narrows the whole table first; the plot samples beyond its cap).
DISPLAY_LIMIT = 20_000
PLOT_LIMIT = 200_000


class FrameModel(QAbstractTableModel):
    """A pandas DataFrame as an (optionally editable) Qt table model."""

    def __init__(self, frame: pd.DataFrame | None = None, *, editable: bool = True, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._editable = editable
        self.edited: set[int] = set()

    def frame(self) -> pd.DataFrame:
        return self._frame

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame.reset_index(drop=True)
        self.edited = set()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if pd.isna(value) if not isinstance(value, (list, dict)) else False:
                return ""
            if isinstance(value, float):
                return f"{value:.6g}" if role == Qt.DisplayRole else value
            return str(value) if role == Qt.DisplayRole else value
        if role == Qt.UserRole:  # sort key
            return value if isinstance(value, (int, float)) and not pd.isna(value) else str(value)
        if role == Qt.BackgroundRole and index.row() in self.edited:
            from qtpy.QtGui import QColor

            return QColor(design.STATUS_COLORS["warn"])
        return None

    def headerData(self, section: int, orientation: Any, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        return str(self._frame.columns[section]) if orientation == Qt.Horizontal else str(section + 1)

    def flags(self, index: QModelIndex) -> Any:
        base = super().flags(index)
        return base | Qt.ItemIsEditable if self._editable else base

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False
        column = self._frame.columns[index.column()]
        old = self._frame.iat[index.row(), index.column()]
        if pd.api.types.is_numeric_dtype(self._frame[column]):
            try:
                value = float(value) if str(value).strip() else float("nan")
            except ValueError:
                return False
        if (pd.isna(old) and pd.isna(value)) if not isinstance(value, str) else str(old) == value:
            return False
        self._frame.iat[index.row(), index.column()] = value
        self.edited.add(index.row())
        self.dataChanged.emit(index, index)
        return True


class AnalysisWorkbench(QWidget):
    """The Analysis module's main panel."""

    def __init__(self, viewer: Any, statmodels: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._table_name: str | None = None
        self._repo: Any = None
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        self._statmodels = statmodels
        if statmodels is not None:
            models = Section("Statistical models", icon_name="chart", hue="red", pref_key="analysis.models", expanded=False)
            models.add(statmodels)
            if hasattr(statmodels, "open_with_frame"):
                send = QPushButton("Model the table below in Statmodels")
                send.setToolTip("Open a new Statmodels session whose analysis frame is the table shown here "
                                "(query and quick filter applied): mixed models, GLMs, mediation, plots.")
                send.clicked.connect(self._to_statmodels)
                models.add(send)
            root.addWidget(models)

        source = Section("Table", icon_name="table", hue="blue", pref_key="analysis.source")
        row = QHBoxLayout()
        self._db_tables = QComboBox()
        self._db_tables.setToolTip("Dataset tables (the configured nvitk dataset)")
        self._wide = QCheckBox("wide")
        self._wide.setToolTip("Pivot long measurement tables to one column per variable")
        load_db = QPushButton("Load")
        load_db.clicked.connect(self._load_db)
        refresh = QPushButton("↻")
        refresh.setToolTip("List the dataset's tables again")
        refresh.clicked.connect(self._list_db_tables)
        row.addWidget(hint_label("Dataset"))
        row.addWidget(self._db_tables, 1)
        row.addWidget(self._wide)
        row.addWidget(load_db)
        row.addWidget(refresh)
        source.add_layout(row)
        row = QHBoxLayout()
        for text, slot in (("From Results", self._from_results), ("Open file…", self._open_file), ("Save as…", self._save_file)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        self._write_back = QPushButton("Write edits to dataset")
        self._write_back.setEnabled(False)
        self._write_back.clicked.connect(self._write_db)
        row.addWidget(self._write_back)
        source.add_layout(row)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Quick filter (any column contains)…")
        self._query = QLineEdit()
        self._query.setPlaceholderText("Pandas query, e.g.  variable_id == 'cbf' and value_num > 40")
        self._query.editingFinished.connect(self._apply_query)
        source.add(self._filter)
        source.add(self._query)
        self._summary = hint_label("No table loaded.")
        source.add(self._summary)
        root.addWidget(source)

        self._model = FrameModel(parent=self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.UserRole)
        self._proxy.setFilterKeyColumn(-1)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setDynamicSortFilter(False)
        self._filter.textChanged.connect(self._proxy.setFilterFixedString)
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._plot = PlotBuilder()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self._table)
        split.addWidget(self._plot)
        split.setSizes([300, 400])
        root.addWidget(split, 1)
        self._full = pd.DataFrame()
        self._truncated = False
        self._model.dataChanged.connect(
            lambda *_a: self._write_back.setEnabled(bool(self._table_name and self._model.edited and not self._truncated))
        )

    # sources
    def _repo_or_none(self) -> Any:
        if self._repo is None:
            try:
                from nvitk.db.repo import get_repo

                self._repo = get_repo()
            except Exception as exc:  # noqa: BLE001
                self._summary.setText(f"No dataset: {exc}")
                return None
        return self._repo

    def _list_db_tables(self) -> None:
        repo = self._repo_or_none()
        self._db_tables.clear()
        if repo is not None:
            try:
                self._db_tables.addItems(sorted(repo.list_tables()))
            except Exception as exc:  # noqa: BLE001
                self._summary.setText(f"Could not list tables: {exc}")

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self._db_tables.count() == 0:
            self._list_db_tables()

    def _load_db(self) -> None:
        repo = self._repo_or_none()
        name = self._db_tables.currentText()
        if repo is None or not name:
            return
        try:
            frame = repo.get(name, wide=self._wide.isChecked())
        except Exception as exc:  # noqa: BLE001
            self._summary.setText(f"Could not load {name}: {exc}")
            return
        self.set_frame(frame, table=None if self._wide.isChecked() else name, label=f"dataset · {name}")

    def _from_results(self) -> None:
        from nvitk.gui.viz.results_window import results_window

        window = results_window(self._viewer)
        if window is None or getattr(window, "_shown", None) is None:
            self._summary.setText("Results is empty.")
            return
        self.set_frame(window.table_frame(), label=f"Results · {window._shown.title}")

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open table", "", "Tables (*.csv *.tsv *.xlsx *.xls *.parquet *.json)")
        if not path:
            return
        from nvitk.db.importers import read_tabular_source

        try:
            self.set_frame(read_tabular_source(path), label=path)
        except Exception as exc:  # noqa: BLE001
            self._summary.setText(f"Could not read {path}: {exc}")

    def _save_file(self) -> None:
        frame = self.current_frame()
        path, _ = QFileDialog.getSaveFileName(self, "Save table", "table.csv", "CSV (*.csv);;Excel (*.xlsx);;Parquet (*.parquet)")
        if not path:
            return
        if path.endswith(".xlsx"):
            frame.to_excel(path, index=False)
        elif path.endswith(".parquet"):
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    # table
    def set_frame(self, frame: pd.DataFrame, *, table: str | None = None, label: str = "") -> None:
        """Show *frame* (a dataset table *table* can be written back after edits)."""
        self._full = frame.reset_index(drop=True)
        self._table_name = table
        self._label = label
        self._apply_query()

    def _apply_query(self) -> None:
        frame = self._full
        text = self._query.text().strip()
        if text:
            try:
                frame = frame.query(text, engine="python")
            except Exception as exc:  # noqa: BLE001
                self._summary.setText(f"Query error: {exc}")
                return
        self._truncated = len(frame) > DISPLAY_LIMIT
        self._table.setSortingEnabled(False)
        self._model.set_frame(frame.head(DISPLAY_LIMIT).copy())
        self._table.setSortingEnabled(True)
        self._write_back.setEnabled(False)
        shown = f" (showing the first {DISPLAY_LIMIT:,}; narrow with a query to edit)" if self._truncated else ""
        self._summary.setText(f"{getattr(self, '_label', '')}: {len(frame):,} of {len(self._full):,} rows, {len(frame.columns)} columns{shown}")
        plot_frame = frame.sample(PLOT_LIMIT, random_state=0) if len(frame) > PLOT_LIMIT else frame
        self._plot.set_frame(plot_frame)

    def current_frame(self) -> pd.DataFrame:
        """The rows shown (query and quick filter applied), in the table's order."""
        rows = [self._proxy.mapToSource(self._proxy.index(r, 0)).row() for r in range(self._proxy.rowCount())]
        return self._model.frame().iloc[rows].reset_index(drop=True)

    def model_frame(self) -> pd.DataFrame:
        """The rows to model: those shown, or every row the query keeps when the display is capped."""
        if not self._truncated:
            return self.current_frame()
        text = self._query.text().strip()
        return self._full.query(text, engine="python") if text else self._full

    def _to_statmodels(self) -> None:
        frame = self.model_frame()
        if frame.empty:
            self._summary.setText("No rows to model: load a table first.")
            return
        label = str(getattr(self, "_label", "") or "Analysis table")
        try:
            self._statmodels.open_with_frame(frame, title=label.split(" · ")[-1][:40])
            self._summary.setText(f"Sent {len(frame):,} rows to a new Statmodels session.")
        except Exception as exc:  # noqa: BLE001
            self._summary.setText(f"Statmodels could not take the table: {exc}")

    def _write_db(self) -> None:
        repo = self._repo_or_none()
        if repo is None or not self._table_name or not self._model.edited:
            return
        n = len(self._model.edited)
        answer = QMessageBox.question(
            self, "Write to the dataset",
            f"Write {n} edited row(s) back to the dataset table “{self._table_name}”?\n"
            "The table file is rewritten (the previous version is kept by the dataset's versioning only).",
        )
        if answer != QMessageBox.Yes:
            return
        edited = self._model.frame()
        merged = self._full.copy()
        if self._query.text().strip():
            # Rows were filtered: write the edited rows back by their original position.
            original = self._full.query(self._query.text().strip(), engine="python").index
            merged.loc[original, :] = edited.values
        else:
            merged = edited
        try:
            repo.write_table(self._table_name, merged, provenance={"source": "nvitk GUI edit", "rows_edited": n},
                             build_sqlite_index=True)
            self._full = merged
            self._model.edited = set()
            self._write_back.setEnabled(False)
            self._summary.setText(f"Wrote {n} edited row(s) to {self._table_name}.")
        except Exception as exc:  # noqa: BLE001
            self._summary.setText(f"Could not write {self._table_name}: {exc}")


__all__ = ["AnalysisWorkbench", "FrameModel"]
