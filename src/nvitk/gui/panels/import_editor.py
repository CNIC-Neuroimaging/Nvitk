"""Spreadsheet import editor: load a table, shape it, map its columns, check, import.

Opened from *Data ▸ Dataset & XNAT setup ▸ Import a spreadsheet* (or Analysis ▸ *Import into the
dataset…* with the table shown there). The library side is :mod:`nvitk.db.staged_import`.

* **Source**: file, sheet and header row; a saved *recipe* (the edits and column mapping of an
  earlier import of the same layout) can be applied in one click.
* **Settings**: kind (clinical / cognitive / image), for images the modality and the pipeline
  version (a new id is registered as a new version, optionally the default), for clinical data the
  visit given to rows without one, the batch id, and the write mode (replace matching values, only
  add missing ones, or keep both copies by source file).
* **Table** (left): the data, editable. Each header shows where the column goes
  (``→ freq · tot``). Right-click a header to rename, drop, set its role / variable / region /
  unit, convert units, find & replace, cast, or filter its values (as in Analysis); right-click a
  cell to filter on its value. The toolbar keeps the filtered rows, drops rows without a subject,
  adds a computed column, splits names like ``FreqTot`` into variable ``freq`` + region ``tot``,
  and undoes / redoes any edit.
* **Columns** (right): one row per column: use, role, variable (existing ones offered), region and
  its label (images; the regions this variable already has are offered), unit, value kind, how
  many values parse as numbers, and whether it is a new variable, a new region or an update.
* **Check** shows what the import would write (new rows, replaced values, unchanged, new
  variables / regions / pipeline version, unknown subjects); **Import** asks once more, writes in
  the background, saves the recipe if asked and rebuilds the query index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from qtpy.QtCore import QModelIndex, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.db import staged_import as si
from nvitk.gui.analysis.column_filters import ColumnFilterProxy, FilterChips, install_header_filters
from nvitk.gui.analysis.workbench import FrameModel
from nvitk.gui.core import design
from nvitk.gui.core.thread_worker import ThreadWorker
from nvitk.gui.core.widgets import IconButton, hint_label

#: Plan table columns.
P_COLUMN, P_ROLE, P_VARIABLE, P_REGION, P_REGION_LABEL, P_UNIT, P_KIND, P_VALUES, P_STATUS = range(9)
PLAN_HEADERS = ("Column", "Role", "Variable", "Region", "Region label", "Unit", "Kind", "Values", "Status")
ROLE_LABELS = {
    "value": "value", "subject": "subject id", "session": "session", "visit": "visit", "date": "date", "skip": "skip",
}
ROLE_COLORS = {"subject": "blue", "session": "cyan", "visit": "cyan", "date": "cyan", "skip": None, "value": "green"}


def _hue(name: str | None) -> QColor | None:
    if not name:
        return None
    from nvitk.gui.core.widgets import _hue_color

    color = QColor(_hue_color(name))
    color.setAlpha(60)
    return color


# --------------------------------------------------------------------------- workers


class _CatalogWorker(ThreadWorker):
    """What the dataset already has: variables, regions, pipelines, visits."""

    done = Signal(dict)
    failed = Signal(str)

    def run(self) -> None:
        try:
            from nvitk.db.repo import get_repo

            repo = get_repo()
            out: dict[str, Any] = {"repo": repo, "root": str(repo.root)}
            out["variables"] = list(repo.catalog.variable_entries())
            out["pipelines"] = list(repo.catalog.list_pipelines())
            out["kinds"] = [k for k, t in si.KIND_TABLES.items() if repo.catalog.table_exists(t)] or ["clinical", "image"]
            try:
                out["regions"] = si.known_regions(repo)
            except Exception:  # noqa: BLE001
                out["regions"] = pd.DataFrame(columns=["modality", "variable_id", "region_id", "region_label", "pipeline_id"])
            visits: list[str] = []
            for table in ("clinical_measurements", "cognitive_measurements"):
                if repo.catalog.table_exists(table):
                    try:
                        got = repo.get(table, columns=["visit_id"], cohort_id=False)["visit_id"].dropna().astype(str)
                        visits += [v for v in got.unique() if v not in visits]
                    except Exception:  # noqa: BLE001
                        pass
            out["visits"] = sorted(visits)
            self.done.emit(out)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _CheckWorker(ThreadWorker):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, repo: Any, frame: pd.DataFrame, plan: si.ImportPlan) -> None:
        super().__init__()
        self._repo, self._frame, self._plan = repo, frame, plan

    def run(self) -> None:
        try:
            self.done.emit(si.check_import(self._repo, self._frame, self._plan))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ImportWorker(ThreadWorker):
    progress = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, repo: Any, frame: pd.DataFrame, plan: si.ImportPlan, ops: list[dict[str, Any]],
                 recipe: str, header: int, rebuild: bool) -> None:
        super().__init__()
        self._args = (repo, frame, plan, ops, recipe, header, rebuild)

    def run(self) -> None:
        repo, frame, plan, ops, recipe, header, rebuild = self._args
        try:
            result = si.apply_import(repo, frame, plan, ops=ops, rebuild_index=rebuild, progress=self.progress.emit)
            if recipe:
                result["recipe"] = str(si.save_recipe(repo, recipe, plan, ops, header=header))
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- table model


class _StagedModel(FrameModel):
    """The staged table; each header says where its column goes."""

    cell_edited = Signal(int, str, object)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(editable=True, parent=parent)
        self._plans: dict[str, si.ColumnPlan] = {}
        self._kind = "clinical"

    def set_plans(self, plans: list[si.ColumnPlan], kind: str) -> None:
        self._plans = {p.column: p for p in plans}
        self._kind = kind
        if self.columnCount():
            self.headerDataChanged.emit(Qt.Horizontal, 0, self.columnCount() - 1)

    def mapping(self, column: str) -> str:
        plan = self._plans.get(column)
        if plan is None:
            return ""
        if plan.role == "skip":
            return "— skipped"
        if plan.role != "value":
            return f"◆ {ROLE_LABELS[plan.role]}"
        if not plan.variable_id:
            return "→ (no variable)"
        text = f"→ {plan.variable_id}"
        if self._kind == "image" and plan.region_id:
            text += f" · {plan.region_id}"
        if plan.unit:
            text += f" [{plan.unit}]"
        return text

    def headerData(self, section: int, orientation: Any, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if orientation == Qt.Horizontal and 0 <= section < len(self._frame.columns):
            column = str(self._frame.columns[section])
            plan = self._plans.get(column)
            if role == Qt.DisplayRole:
                return f"{column}\n{self.mapping(column)}"
            if role == Qt.BackgroundRole and plan is not None:
                return _hue(ROLE_COLORS.get(plan.role))
            if role == Qt.ForegroundRole and plan is not None and plan.role == "skip":
                return QColor(design.COLOR_MUTED)
            if role == Qt.ToolTipRole:
                return f"{column}: {self.mapping(column)}\nRight-click for the column's options."
        return super().headerData(section, orientation, role)

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:  # noqa: N802
        changed = super().setData(index, value, role)
        if changed:
            column = str(self._frame.columns[index.column()])
            self.cell_edited.emit(index.row(), column, self._frame.iat[index.row(), index.column()])
        return changed


# --------------------------------------------------------------------------- editor


class ImportEditor(QWidget):
    """The import editor window."""

    imported = Signal(dict)

    def __init__(self, parent: QWidget | None = None, *, path: str | None = None, frame: pd.DataFrame | None = None,
                 source_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowTitle("Import a spreadsheet into the dataset")
        self.setAcceptDrops(True)
        self.resize(1400, 860)
        self._raw = pd.DataFrame()
        self._frame = pd.DataFrame()
        self._ops: list[dict[str, Any]] = []
        #: Undo history: (op, plans before, plans after).
        self._history: list[tuple[dict[str, Any], list[si.ColumnPlan], list[si.ColumnPlan]]] = []
        self._redo: list[tuple[dict[str, Any], list[si.ColumnPlan], list[si.ColumnPlan]]] = []
        self._plans: list[si.ColumnPlan] = []
        self._path = ""
        self._source_name = source_name
        self._catalog: dict[str, Any] = {}
        self._repo: Any = None
        self._check: si.ImportCheck | None = None
        self._workers: list[ThreadWorker] = []
        self._building = False

        root = QVBoxLayout(self)
        root.setContentsMargins(design.PAD, design.PAD, design.PAD, design.PAD)
        root.setSpacing(design.SPACE_TIGHT)
        root.addLayout(self._build_source_row())
        root.addWidget(self._build_settings())
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_table_side())
        split.addWidget(self._build_plan_side())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 560])
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(split)
        vsplit.addWidget(self._build_bottom())
        vsplit.setStretchFactor(0, 1)
        vsplit.setSizes([640, 170])
        root.addWidget(vsplit, 1)
        design.apply_theme(self)

        from nvitk.gui.core.dataset_root import dataset_bus

        dataset_bus().changed.connect(lambda _r: self._load_catalog())
        self._load_catalog()
        self._sync_kind()
        if frame is not None:
            self.set_raw(frame, source_name=source_name or "table")
        elif path:
            self.open_path(path)

    # ------------------------------------------------------------------ layout
    def _build_source_row(self) -> QHBoxLayout:
        from nvitk.gui.core.dataset_root import DatasetRootButton

        row = QHBoxLayout()
        row.setSpacing(design.SPACE_TIGHT)
        self._file = QLineEdit()
        self._file.setReadOnly(True)
        self._file.setPlaceholderText("Open a CSV / Excel / Parquet file, or drop one here")
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._pick_file)
        self._sheet = QComboBox()
        self._sheet.setMinimumContentsLength(10)
        self._sheet.activated.connect(lambda _i: self._reload_file())
        self._header = QSpinBox()
        self._header.setRange(0, 50)
        self._header.setToolTip("Row holding the column names (0 = the first row)")
        self._header.valueChanged.connect(lambda _v: self._reload_file())
        row.addWidget(QLabel("File"))
        row.addWidget(self._file, 3)
        row.addWidget(open_btn)
        row.addWidget(QLabel("Sheet"))
        row.addWidget(self._sheet, 1)
        row.addWidget(QLabel("Header row"))
        row.addWidget(self._header)
        row.addSpacing(design.PAD)
        row.addWidget(QLabel("Recipe"))
        self._recipes = QComboBox()
        self._recipes.setMinimumContentsLength(12)
        self._recipes.setToolTip("Saved imports of this dataset: their table edits and column mapping")
        row.addWidget(self._recipes, 1)
        apply_recipe = QPushButton("Apply")
        apply_recipe.setToolTip("Replay the recipe's edits on this table and take its column mapping and settings")
        apply_recipe.clicked.connect(self._apply_recipe)
        row.addWidget(apply_recipe)
        save_recipe = QPushButton("Save…")
        save_recipe.setToolTip("Save the edits and mapping as a recipe for the next file of this layout")
        save_recipe.clicked.connect(self._save_recipe)
        row.addWidget(save_recipe)
        self._dataset_button = DatasetRootButton()
        row.addWidget(self._dataset_button)
        return row

    def _build_settings(self) -> QWidget:
        box = QWidget()
        grid = QHBoxLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(design.PAD)
        left = QFormLayout()
        left.setContentsMargins(0, 0, 0, 0)
        self._kind = QComboBox()
        for kind in si.KIND_TABLES:
            self._kind.addItem(kind, kind)
        self._kind.setToolTip("clinical / cognitive: one value per subject and visit; image: per subject, region "
                              "(territory) and pipeline version")
        self._kind.currentIndexChanged.connect(lambda _i: self._sync_kind())
        left.addRow("Kind", self._kind)
        self._mode = QComboBox()
        for mode in si.MODES:
            self._mode.addItem(si.MODE_LABELS[mode], mode)
        self._mode.setToolTip(
            "Replace: a value for the same subject, visit (images: region and pipeline version) and variable replaces "
            "the dataset's, whatever file it came from.\nOnly add missing: values the dataset has are left alone.\n"
            "Keep both: rows are told apart by source file / sheet / column (the old importer's behaviour).")
        self._mode.currentIndexChanged.connect(lambda _i: self._invalidate_check())
        left.addRow("Write", self._mode)
        grid.addLayout(left, 1)

        mid = QFormLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        self._modality = QComboBox()
        self._modality.setEditable(True)
        self._modality.lineEdit().setPlaceholderText("flair, t1, asl, 4dflow…")
        self._modality.currentTextChanged.connect(lambda _t: self._modality_changed())
        self._modality_label = QLabel("Modality")
        mid.addRow(self._modality_label, self._modality)
        prow = QHBoxLayout()
        self._pipeline = QComboBox()
        self._pipeline.setEditable(True)
        self._pipeline.lineEdit().setPlaceholderText("pipeline version id, e.g. wmh_v1 (new id: new version)")
        self._pipeline.currentTextChanged.connect(lambda _t: self._pipeline_changed())
        self._pipeline_default = QCheckBox("default")
        self._pipeline_default.setToolTip("Register this version as the modality's default (what 'latest' resolves to)")
        self._pipeline_default.toggled.connect(lambda _c: self._invalidate_check())
        prow.addWidget(self._pipeline, 1)
        prow.addWidget(self._pipeline_default)
        self._pipeline_box = QWidget()
        self._pipeline_box.setLayout(prow)
        prow.setContentsMargins(0, 0, 0, 0)
        self._pipeline_label = QLabel("Pipeline")
        mid.addRow(self._pipeline_label, self._pipeline_box)
        self._visit = QComboBox()
        self._visit.setEditable(True)
        self._visit.lineEdit().setPlaceholderText("visit for rows without a visit column, e.g. 4")
        self._visit.currentTextChanged.connect(lambda _t: self._invalidate_check())
        self._visit_label = QLabel("Default visit")
        mid.addRow(self._visit_label, self._visit)
        grid.addLayout(mid, 2)

        right = QFormLayout()
        right.setContentsMargins(0, 0, 0, 0)
        self._batch = QLineEdit()
        self._batch.setPlaceholderText("batch id (version tag of this import), e.g. wmh_freqtot_2026-11")
        self._batch.textChanged.connect(lambda _t: self._invalidate_check())
        right.addRow("Batch id", self._batch)
        self._source_edit = QLineEdit()
        self._source_edit.setToolTip("Recorded as the rows' source file (keep-both mode tells copies apart by it)")
        self._source_edit.textChanged.connect(lambda _t: self._invalidate_check())
        right.addRow("Source name", self._source_edit)
        grid.addLayout(right, 2)
        return box

    def _build_table_side(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(design.SPACE_TIGHT)
        tools = QHBoxLayout()
        tools.setSpacing(2)
        self._undo = IconButton("undo", "Undo the last table edit (Ctrl+Z)")
        self._undo.clicked.connect(self.undo)
        self._redo_btn = IconButton("redo", "Redo (Ctrl+Shift+Z)")
        self._redo_btn.clicked.connect(self.redo)
        tools.addWidget(self._undo)
        tools.addWidget(self._redo_btn)
        for text, tip, slot in (
            ("Keep filtered rows", "Make the header filters an edit: drop the rows they hide", self._keep_filtered),
            ("Drop rows without subject", "Drop rows whose subject id is empty", self._drop_no_subject),
            ("Delete selected rows", "Delete the selected rows (Del)", self._delete_rows),
            ("New column…", "Add a column computed from others, e.g.  FreqTot * 100", self._compute),
            ("Split names…", "Split column names into variable + region, e.g. FreqTot → freq · tot", self._split_names),
            ("Drop duplicates", "Drop repeated rows (the last copy is kept)", self._dedupe),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            tools.addWidget(button)
        tools.addStretch(1)
        lay.addLayout(tools)
        self._model = _StagedModel(self)
        self._model.cell_edited.connect(self._cell_edited)
        self._proxy = ColumnFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.UserRole)
        self._proxy.setDynamicSortFilter(False)
        self._table = QTableView()
        self._table.setModel(self._proxy)
        # File order until a header is clicked (Qt would sort by the first column, descending).
        self._table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.horizontalHeader().setMinimumHeight(40)
        install_header_filters(self._table, self._proxy, extra_menu=self._column_menu)
        self._chips = FilterChips(self._proxy)
        lay.addWidget(self._chips)
        lay.addWidget(self._table, 1)
        self._history_label = hint_label("No edits.")
        self._history_label.setWordWrap(True)
        lay.addWidget(self._history_label)
        from qtpy.QtGui import QKeySequence, QShortcut

        QShortcut(QKeySequence.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.Redo, self, activated=self.redo)
        QShortcut(QKeySequence.Delete, self._table, activated=self._delete_rows)
        return side

    def _build_plan_side(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(design.SPACE_TIGHT)
        head = QHBoxLayout()
        head.addWidget(QLabel("Columns"), 1)
        reguess = QPushButton("Re-guess")
        reguess.setToolTip("Guess every column's role and variable again from its name and the dataset's catalog")
        reguess.clicked.connect(self._reguess)
        head.addWidget(reguess)
        bulk = QToolButton()
        bulk.setText("Selected rows")
        bulk.setPopupMode(QToolButton.InstantPopup)
        bulk_menu = QMenu(bulk)
        bulk_menu.aboutToShow.connect(lambda m=bulk_menu: self._fill_bulk_menu(m))
        bulk.setMenu(bulk_menu)
        head.addWidget(bulk)
        lay.addLayout(head)
        self._plan_table = QTableWidget(0, len(PLAN_HEADERS))
        self._plan_table.setHorizontalHeaderLabels(PLAN_HEADERS)
        self._plan_table.verticalHeader().setVisible(False)
        self._plan_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._plan_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._plan_table.horizontalHeader().setStretchLastSection(True)
        self._plan_table.itemChanged.connect(self._plan_item_changed)
        self._plan_table.currentCellChanged.connect(lambda r, _c, _pr, _pc: self._focus_column(r))
        lay.addWidget(self._plan_table, 1)
        self._plan_hint = hint_label("")
        lay.addWidget(self._plan_hint)
        return side

    def _build_bottom(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self._check_btn = QPushButton("Check")
        self._check_btn.setToolTip("Compare with the dataset: what would be new, replaced or unchanged (writes nothing)")
        self._check_btn.clicked.connect(self._run_check)
        self._import_btn = QPushButton("Import into the dataset")
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._run_import)
        self._save_recipe_on_import = QCheckBox("Save as recipe")
        self._save_recipe_on_import.setChecked(True)
        self._recipe_name = QLineEdit()
        self._recipe_name.setPlaceholderText("recipe name (default: the file name)")
        self._rebuild = QCheckBox("Rebuild the query index")
        self._rebuild.setChecked(True)
        self._rebuild.setToolTip("Analysis and Statmodels read the index: without it, new rows show only after the next rebuild")
        row.addWidget(self._check_btn)
        row.addWidget(self._import_btn)
        row.addSpacing(design.PAD)
        row.addWidget(self._save_recipe_on_import)
        row.addWidget(self._recipe_name, 1)
        row.addWidget(self._rebuild)
        lay.addLayout(row)
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setPlaceholderText("Open a file, map its columns, then Check.")
        lay.addWidget(self._summary, 1)
        return box

    # ------------------------------------------------------------------ catalog
    def _load_catalog(self) -> None:
        worker = _CatalogWorker()
        worker.setParent(self)
        worker.done.connect(self._catalog_loaded)
        worker.failed.connect(lambda m: self._summary.setPlainText(f"Could not read the dataset: {m}"))
        self._workers.append(worker)
        self._plan_hint.setText("Reading the dataset's variables and regions…")
        worker.start()

    def _catalog_loaded(self, catalog: dict) -> None:
        self._catalog = catalog
        self._repo = catalog.get("repo")
        kinds = catalog.get("kinds") or list(si.KIND_TABLES)
        current = self._kind.currentData()
        self._kind.blockSignals(True)
        self._kind.clear()
        for kind in kinds:
            self._kind.addItem(kind, kind)
        index = self._kind.findData(current)
        self._kind.setCurrentIndex(max(index, 0))
        self._kind.blockSignals(False)
        modalities = sorted({str(p.get("modality") or "").lower() for p in catalog.get("pipelines") or [] if p.get("modality")}
                            | set(catalog.get("regions", pd.DataFrame()).get("modality", pd.Series(dtype=str)).dropna().astype(str).str.lower()))
        text = self._modality.currentText()
        self._modality.blockSignals(True)
        self._modality.clear()
        self._modality.addItems(modalities)
        self._modality.setCurrentText(text)
        self._modality.blockSignals(False)
        visit = self._visit.currentText()
        self._visit.clear()
        self._visit.addItems(catalog.get("visits") or [])
        self._visit.setCurrentText(visit or (self._default_visit() or ""))
        self._recipes.clear()
        try:
            self._recipes.addItems(si.list_recipes(self._repo))
        except Exception:  # noqa: BLE001
            pass
        self._plan_hint.setText(f"Dataset: {catalog.get('root')}")
        self._modality_changed()
        self._sync_kind()

    def _default_visit(self) -> str | None:
        try:
            from nvitk.db.config.dataset_config import load_dataset_config

            return str(load_dataset_config().default_visit_label or "") or None
        except Exception:  # noqa: BLE001
            return None

    def _variables(self) -> list[dict[str, Any]]:
        kind = self.kind()
        modality = self._modality.currentText().strip().lower()
        out = [e for e in self._catalog.get("variables") or [] if e.get("domain") == kind]
        if kind == "image" and modality:
            mine = [e for e in out if str(e.get("modality") or "").lower() == modality]
            out = mine + [e for e in out if e not in mine]
        return out

    def _regions(self, variable: str) -> list[tuple[str, str]]:
        regions = self._catalog.get("regions")
        if regions is None or regions.empty or not variable:
            return []
        modality = self._modality.currentText().strip().lower()
        sub = regions[regions["variable_id"].astype(str) == variable]
        if modality:
            sub = sub[sub["modality"].astype(str).str.lower() == modality]
        pairs = sub[["region_id", "region_label"]].dropna(subset=["region_id"]).drop_duplicates()
        return sorted((str(r), str(l) if pd.notna(l) else "") for r, l in pairs.itertuples(index=False))

    # ------------------------------------------------------------------ settings
    def kind(self) -> str:
        return str(self._kind.currentData() or "clinical")

    def _sync_kind(self) -> None:
        image = self.kind() == "image"
        for widget in (self._modality, self._modality_label, self._pipeline_box, self._pipeline_label):
            widget.setVisible(image)
        for widget in (self._visit, self._visit_label):
            widget.setVisible(not image)
        for col in (P_REGION, P_REGION_LABEL):
            self._plan_table.setColumnHidden(col, not image)
        self._model.set_plans(self._plans, self.kind())
        self._refresh_plan_table()
        self._invalidate_check()

    def _modality_changed(self) -> None:
        modality = self._modality.currentText().strip().lower()
        current = self._pipeline.currentText()
        self._pipeline.blockSignals(True)
        self._pipeline.clear()
        for entry in self._catalog.get("pipelines") or []:
            if str(entry.get("modality") or "").lower() == modality:
                pid = str(entry.get("pipeline_id"))
                default = " (default)" if entry.get("is_default") else ""
                self._pipeline.addItem(f"{pid}{default}", pid)
        if current:
            self._pipeline.setCurrentText(current)
        elif self._pipeline.count():
            defaults = [i for i in range(self._pipeline.count()) if "(default)" in self._pipeline.itemText(i)]
            self._pipeline.setCurrentIndex(defaults[0] if defaults else 0)
        self._pipeline.blockSignals(False)
        self._pipeline_changed()

    def _pipeline_id(self) -> str:
        text = self._pipeline.currentText().strip()
        return text.replace(" (default)", "").strip()

    def _pipeline_changed(self) -> None:
        pid = self._pipeline_id()
        known = {str(p.get("pipeline_id")) for p in self._catalog.get("pipelines") or []}
        new = bool(pid) and pid not in known
        self._pipeline.setToolTip(f"New version {pid}: registered on import" if new else
                                  "Existing pipeline version" if pid else "No pipeline version")
        self._pipeline_label.setText("Pipeline (new)" if new else "Pipeline")
        self._refresh_plan_table()
        self._invalidate_check()

    def plan(self) -> si.ImportPlan:
        return si.ImportPlan(
            kind=self.kind(), source_name=self._source_edit.text().strip() or self._source_name or "spreadsheet",
            sheet=self._sheet.currentText() or "0", batch_id=self._batch.text().strip(),
            modality=self._modality.currentText().strip().lower() if self.kind() == "image" else "",
            pipeline_id=self._pipeline_id() if self.kind() == "image" else "",
            make_default_pipeline=self._pipeline_default.isChecked(),
            visit_id=self._visit.currentText().strip() if self.kind() != "image" else "",
            mode=str(self._mode.currentData() or "replace"),
            columns=[si.ColumnPlan(**p.to_dict()) for p in self._plans],
        )

    # ------------------------------------------------------------------ loading
    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Spreadsheet", str(Path(self._path).parent) if self._path else "",
                                              "Tables (*.xlsx *.xls *.csv *.tsv *.parquet)")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        from nvitk.db.importers import tabular_sheet_names

        self._path = str(path)
        self._file.setText(self._path)
        self._sheet.blockSignals(True)
        self._sheet.clear()
        try:
            self._sheet.addItems([str(s) for s in tabular_sheet_names(path)] or ["0"])
        except Exception:  # noqa: BLE001
            self._sheet.addItem("0")
        self._sheet.blockSignals(False)
        self._source_name = Path(path).name
        self._source_edit.setText(self._source_name)
        self._recipe_name.setPlaceholderText(f"recipe name (default: {Path(path).stem})")
        self._reload_file()

    def _reload_file(self) -> None:
        if not self._path:
            return
        from nvitk.db.importers import read_tabular_source

        sheet: Any = self._sheet.currentText() or 0
        try:
            frame = read_tabular_source(self._path, sheet_name=sheet, header=int(self._header.value()))
        except Exception as exc:  # noqa: BLE001
            self._summary.setPlainText(f"Could not read {self._path}: {exc}")
            return
        self.set_raw(frame, source_name=Path(self._path).name)

    def set_raw(self, frame: pd.DataFrame, *, source_name: str = "") -> None:
        """Start over from *frame* (as read): no edits, a guessed column plan."""
        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        self._raw = frame.reset_index(drop=True)
        self._ops, self._history, self._redo = [], [], []
        if source_name:
            self._source_name = source_name
            if not self._source_edit.text():
                self._source_edit.setText(source_name)
        self._plans = si.guess_plan(self._raw, self.kind(), repo=self._repo, source_name=self._source_name)
        self._proxy.clear_filters()
        self._set_frame(self._raw)

    def _set_frame(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self._reconcile_plans()
        self._table.setSortingEnabled(False)
        self._model.set_frame(frame.copy())
        self._model.set_plans(self._plans, self.kind())
        self._table.setSortingEnabled(True)
        self._refresh_plan_table()
        self._history_label.setText(
            f"{len(frame):,} rows × {len(frame.columns)} columns. Edits: "
            + ("; ".join(si.describe_op(op) for op in self._ops[-6:]) if self._ops else "none")
            + (" (…)" if len(self._ops) > 6 else ""))
        self._undo.setEnabled(bool(self._history))
        self._redo_btn.setEnabled(bool(self._redo))
        self._invalidate_check()

    def _reconcile_plans(self) -> None:
        """One plan per column of the current frame, in its order (new columns guessed)."""
        by_name = {p.column: p for p in self._plans}
        columns = [str(c) for c in self._frame.columns]
        missing = [c for c in columns if c not in by_name]
        if missing:
            guessed = si.guess_plan(self._frame[missing], self.kind(), repo=self._repo, source_name=self._source_name)
            for plan in guessed:
                if plan.role in ("subject", "session", "visit", "date") and any(
                        p.role == plan.role for p in self._plans if p.column in columns):
                    plan.role = "value"
                    plan.variable_id = plan.variable_id or plan.column.lower()
                by_name[plan.column] = plan
        self._plans = [by_name[c] for c in columns]

    # ------------------------------------------------------------------ edits
    def _snapshot(self) -> list[si.ColumnPlan]:
        return [si.ColumnPlan(**p.to_dict()) for p in self._plans]

    def push_op(self, op: dict[str, Any], *, plans_after: list[si.ColumnPlan] | None = None) -> bool:
        """Apply one table edit (undoable)."""
        before = self._snapshot()
        try:
            frame = si.apply_op(self._frame, op)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Edit failed", f"{si.describe_op(op)}:\n{exc}")
            return False
        if op.get("op") == "rename":
            for plan in self._plans:
                if plan.column == op["from"]:
                    plan.column = op["to"]
        if plans_after is not None:
            self._plans = plans_after
        self._ops.append(op)
        self._set_frame(frame)
        self._history.append((op, before, self._snapshot()))
        self._redo.clear()
        self._undo.setEnabled(True)
        self._redo_btn.setEnabled(False)
        return True

    def undo(self) -> None:
        if not self._history:
            return
        op, before, after = self._history.pop()
        self._ops.pop()
        self._redo.append((op, before, after))
        self._plans = before
        self._set_frame(si.apply_ops(self._raw, self._ops))

    def redo(self) -> None:
        if not self._redo:
            return
        op, before, after = self._redo.pop()
        self._ops.append(op)
        self._history.append((op, before, after))
        self._plans = after
        self._set_frame(si.apply_ops(self._raw, self._ops))

    def _cell_edited(self, row: int, column: str, value: Any) -> None:
        # The model holds the edit already; record it so undo / replays see it.
        before = self._snapshot()
        op = {"op": "set", "row": int(row), "column": column, "value": None if pd.isna(value) else value}
        self._ops.append(op)
        self._frame = self._model.frame().copy()
        self._history.append((op, before, self._snapshot()))
        self._redo.clear()
        self._undo.setEnabled(True)
        self._invalidate_check()

    def _selected_columns(self) -> list[str]:
        cols = sorted({self._proxy.mapToSource(i).column() for i in self._table.selectionModel().selectedIndexes()})
        return [str(self._frame.columns[c]) for c in cols if c < len(self._frame.columns)]

    def _column_menu(self, menu: QMenu, column: str) -> None:
        """The import entries of a header's menu."""
        plan = self._plan(column)
        menu.addAction("Rename…", lambda: self._rename(column))
        roles = menu.addMenu("Role")
        for role in si.ROLES:
            action = roles.addAction(ROLE_LABELS[role])
            action.setCheckable(True)
            action.setChecked(plan is not None and plan.role == role)
            action.triggered.connect(lambda _=False, r=role: self._set_fields([column], role=r))
        menu.addAction("Variable…", lambda: self._ask_field([column], "variable_id"))
        if self.kind() == "image":
            menu.addAction("Region…", lambda: self._ask_field([column], "region_id"))
            menu.addAction("Region label…", lambda: self._ask_field([column], "region_label"))
        menu.addAction("Unit…", lambda: self._ask_field([column], "unit"))
        kinds = menu.addMenu("Value kind")
        for kind in si.VALUE_KINDS:
            action = kinds.addAction(kind)
            action.setCheckable(True)
            action.setChecked(plan is not None and plan.value_kind == kind)
            action.triggered.connect(lambda _=False, k=kind: self._set_fields([column], value_kind=k))
        menu.addSeparator()
        selected = self._selected_columns()
        menu.addAction("Drop column", lambda: self.push_op({"op": "drop", "columns": [column]}))
        if len(selected) > 1:
            menu.addAction(f"Drop the {len(selected)} selected columns", lambda: self.push_op({"op": "drop", "columns": selected}))
            menu.addAction(f"Keep only the {len(selected)} selected columns",
                           lambda: self.push_op({"op": "keep", "columns": selected}))
        menu.addAction("Copy column…", lambda: self._copy_column(column))
        menu.addAction("Scale / convert units…", lambda: self._scale(column))
        menu.addAction("Find && replace…", lambda: self._replace(column))
        menu.addAction("Trim text / strip prefix…", lambda: self._strip(column))
        cast = menu.addMenu("Convert")
        cast.addAction("To numbers (decimal commas too)", lambda: self.push_op({"op": "cast", "column": column, "to": "numeric"}))
        cast.addAction("To text", lambda: self.push_op({"op": "cast", "column": column, "to": "text"}))

    def _plan(self, column: str) -> si.ColumnPlan | None:
        return next((p for p in self._plans if p.column == column), None)

    def _rename(self, column: str) -> None:
        name, ok = QInputDialog.getText(self, "Rename column", f"New name for {column}:", text=column)
        name = str(name).strip()
        if ok and name and name != column:
            if name in self._frame.columns:
                QMessageBox.warning(self, "Rename", f"There is already a column {name}.")
                return
            self.push_op({"op": "rename", "from": column, "to": name})

    def _copy_column(self, column: str) -> None:
        name, ok = QInputDialog.getText(self, "Copy column", f"Name of the copy of {column}:", text=f"{column}_copy")
        if ok and str(name).strip():
            self.push_op({"op": "duplicate", "column": column, "to": str(name).strip()})

    def _scale(self, column: str) -> None:
        text, ok = QInputDialog.getText(
            self, "Scale / convert units",
            f"{column} × factor + offset (e.g. 100 for ratio → %, 0.001 for mm³ → mL, '1, -273.15'):", text="1")
        if not ok:
            return
        parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
        try:
            factor = float(parts[0]) if parts else 1.0
            offset = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            return
        self.push_op({"op": "scale", "column": column, "factor": factor, "offset": offset})

    def _replace(self, column: str) -> None:
        find, ok = QInputDialog.getText(self, "Find & replace", f"In {column}, replace the value (whole cell):")
        if not ok:
            return
        repl, ok = QInputDialog.getText(self, "Find & replace", f"…“{find}” with (empty: missing):")
        if ok:
            self.push_op({"op": "replace", "column": column, "find": find, "replace": repl})

    def _strip(self, column: str) -> None:
        prefix, ok = QInputDialog.getText(self, "Trim", f"Trim {column}; also remove this prefix (optional):")
        if ok:
            self.push_op({"op": "strip", "column": column, "prefix": prefix})

    def _keep_filtered(self) -> None:
        filters = self._proxy.filters()
        if not filters:
            self._summary.setPlainText("No header filters: right-click a column header to filter its values first.")
            return
        if self.push_op({"op": "filter", "filters": filters}):
            self._proxy.clear_filters()

    def _drop_no_subject(self) -> None:
        subject = next((p.column for p in self._plans if p.role == "subject"), None)
        if subject is None:
            self._summary.setPlainText("Mark the subject id column first (its header's Role menu).")
            return
        self.push_op({"op": "dropna", "columns": [subject]})

    def _delete_rows(self) -> None:
        rows = sorted({self._proxy.mapToSource(i).row() for i in self._table.selectionModel().selectedIndexes()})
        if rows:
            self.push_op({"op": "delete_rows", "rows": rows})

    def _compute(self) -> None:
        name, ok = QInputDialog.getText(self, "New column", "Name of the new column:")
        if not ok or not str(name).strip():
            return
        cols = ", ".join(map(str, list(self._frame.columns)[:8]))
        expr, ok = QInputDialog.getText(self, "New column",
                                        f"Expression over the columns (pandas eval; `back quotes` for names with spaces):\n{cols}…")
        if ok and str(expr).strip():
            self.push_op({"op": "compute", "column": str(name).strip(), "expr": str(expr).strip()})

    def _dedupe(self) -> None:
        self.push_op({"op": "dedupe"})

    def _split_names(self) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Split column names into variable + region")
        form = QFormLayout(dialog)
        presets = QComboBox()
        presets.addItems(list(si.NAME_PATTERNS))
        pattern = QLineEdit(si.NAME_PATTERNS[presets.currentText()])
        presets.currentTextChanged.connect(lambda key: pattern.setText(si.NAME_PATTERNS[key]))
        scope = QComboBox()
        selected = self._selected_columns()
        scope.addItem("every value column")
        if selected:
            scope.addItem(f"the {len(selected)} selected columns")
        labels = QLineEdit()
        labels.setPlaceholderText("region labels, e.g.  tot=Total brain; fl=Frontal left")
        preview = QLabel()
        preview.setWordWrap(True)

        def _preview() -> None:
            trial = [si.ColumnPlan(**p.to_dict()) for p in self._plans]
            try:
                changed = si.split_names(trial, pattern.text(), columns=selected if scope.currentIndex() == 1 else None)
            except Exception as exc:  # noqa: BLE001
                preview.setText(f"Pattern error: {exc}")
                return
            shown = [f"{p.column} → {p.variable_id} · {p.region_id}" for p in trial if p.column in changed][:8]
            preview.setText(f"{len(changed)} column(s): " + ("; ".join(shown) or "none match") + ("…" if len(changed) > 8 else ""))

        for widget in (pattern,):
            widget.textChanged.connect(lambda _t: _preview())
        scope.currentIndexChanged.connect(lambda _i: _preview())
        form.addRow("Name layout", presets)
        form.addRow("Pattern", pattern)
        form.addRow("Apply to", scope)
        form.addRow("Region labels", labels)
        form.addRow(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        _preview()
        if dialog.exec() != QDialog.Accepted:
            return
        mapping = {}
        for part in labels.text().split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                mapping[key.strip().lower()] = value.strip()
        try:
            si.split_names(self._plans, pattern.text(), columns=selected if scope.currentIndex() == 1 else None,
                           region_labels=mapping)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Split names", str(exc))
            return
        self._plans_changed()

    # ------------------------------------------------------------------ plan table
    def _set_fields(self, columns: list[str], **fields: Any) -> None:
        for plan in self._plans:
            if plan.column in columns:
                for key, value in fields.items():
                    setattr(plan, key, value)
                if fields.get("role") == "value" and not plan.variable_id:
                    from nvitk.db.storage import normalize_variable_id

                    plan.variable_id = normalize_variable_id(plan.column)
        if fields.get("role") in ("subject", "session", "visit", "date"):
            # One column per id role.
            for plan in self._plans:
                if plan.column not in columns and plan.role == fields["role"]:
                    plan.role = "skip"
        self._plans_changed()

    def _ask_field(self, columns: list[str], field: str) -> None:
        plan = self._plan(columns[0])
        current = getattr(plan, field, "") if plan is not None else ""
        titles = {"variable_id": "Variable", "region_id": "Region (territory) id", "region_label": "Region label", "unit": "Unit"}
        if field == "variable_id":
            items = [str(e["variable_id"]) for e in self._variables()]
        elif field == "region_id":
            items = sorted({r for v in {p.variable_id for p in self._plans if p.column in columns} for r, _l in self._regions(v)})
        else:
            items = []
        if items:
            value, ok = QInputDialog.getItem(self, titles[field], f"{titles[field]} for {', '.join(columns[:4])}:",
                                             [current] + [i for i in items if i != current] if current else items, 0, True)
        else:
            value, ok = QInputDialog.getText(self, titles[field], f"{titles[field]} for {', '.join(columns[:4])}:", text=current)
        if ok:
            fields: dict[str, Any] = {field: str(value).strip()}
            if field in ("variable_id", "region_id") and len(columns) == 1 and plan is not None and plan.role != "value":
                fields["role"] = "value"
            if field == "region_id":
                label = dict(self._regions(plan.variable_id if plan else "")).get(str(value).strip())
                if label:
                    fields["region_label"] = label
            self._set_fields(columns, **fields)

    def _fill_bulk_menu(self, menu: QMenu) -> None:
        menu.clear()
        rows = sorted({i.row() for i in self._plan_table.selectionModel().selectedRows()})
        columns = [self._plans[r].column for r in rows if r < len(self._plans)]
        if not columns:
            menu.addAction("Select rows in the column list first").setEnabled(False)
            return
        roles = menu.addMenu(f"Role of {len(columns)} column(s)")
        for role in ("value", "skip"):
            roles.addAction(ROLE_LABELS[role], lambda r=role: self._set_fields(columns, role=r))
        menu.addAction("Variable…", lambda: self._ask_field(columns, "variable_id"))
        if self.kind() == "image":
            menu.addAction("Region…", lambda: self._ask_field(columns, "region_id"))
            menu.addAction("Region label…", lambda: self._ask_field(columns, "region_label"))
        menu.addAction("Unit…", lambda: self._ask_field(columns, "unit"))
        menu.addAction("Region = column name", lambda: self._region_from_name(columns))
        menu.addSeparator()
        menu.addAction("Drop these columns from the table", lambda: self.push_op({"op": "drop", "columns": columns}))

    def _region_from_name(self, columns: list[str]) -> None:
        si.split_names(self._plans, si.NAME_PATTERNS["Whole name is the region"], columns=columns)
        self._plans_changed()

    def _plans_changed(self) -> None:
        self._model.set_plans(self._plans, self.kind())
        self._schedule_plan_refresh()
        self._invalidate_check()

    def _schedule_plan_refresh(self) -> None:
        """Rebuild the column list after the current event: an edit arrives from one of the list's
        own widgets, which the rebuild replaces (deleting a widget inside its own signal crashes)."""
        if getattr(self, "_refresh_pending", False):
            return
        self._refresh_pending = True
        from qtpy.QtCore import QTimer

        def run() -> None:
            self._refresh_pending = False
            self._refresh_plan_table()

        QTimer.singleShot(0, run)

    def _status(self, plan: si.ColumnPlan) -> tuple[str, str | None]:
        if plan.role == "skip":
            return "not imported", None
        if plan.role != "value":
            return f"{ROLE_LABELS[plan.role]} column", "blue"
        if not plan.variable_id:
            return "no variable: not imported", "orange"
        from nvitk.db.storage import normalize_variable_id

        vid = normalize_variable_id(plan.variable_id)
        known = {str(e["variable_id"]) for e in self._catalog.get("variables") or [] if e.get("domain") == self.kind()}
        if vid not in known:
            return "new variable", "purple"
        if self.kind() == "image":
            regions = {r for r, _l in self._regions(vid)}
            region = normalize_variable_id(plan.region_id) if plan.region_id else ""
            pid = self._pipeline_id()
            known_pids = {str(p.get("pipeline_id")) for p in self._catalog.get("pipelines") or []}
            if pid and pid not in known_pids:
                return "existing variable · new pipeline version", "cyan"
            if region and region not in regions:
                return f"existing variable · new region {region}", "cyan"
            if not region and regions:
                return "existing variable · no region (the others have one)", "orange"
        return "existing variable: values update", "green"

    def _refresh_plan_table(self) -> None:
        if not hasattr(self, "_plan_table"):
            return
        self._building = True
        table = self._plan_table
        # Old row widgets go now (hidden at once): Qt only schedules their deletion, and until the
        # event loop gets to it they would still paint over the new rows.
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                old = table.cellWidget(r, c)
                if old is not None:
                    old.hide()
                    table.removeCellWidget(r, c)
        table.setRowCount(len(self._plans))
        frame = self._frame
        from qtpy.QtCore import QStringListModel

        # One list model shared by every row's variable combo (a wide sheet has hundreds of rows).
        self._variable_model = QStringListModel([str(e["variable_id"]) for e in self._variables()], self)
        for row, plan in enumerate(self._plans):
            item = QTableWidgetItem(plan.column)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if plan.role == "skip" else Qt.Checked)
            item.setToolTip("Untick to skip the column")
            table.setItem(row, P_COLUMN, item)
            role = QComboBox()
            for key in si.ROLES:
                role.addItem(ROLE_LABELS[key], key)
            role.setCurrentIndex(si.ROLES.index(plan.role))
            role.currentIndexChanged.connect(lambda _i, c=plan.column, w=role: self._set_fields([c], role=w.currentData()))
            table.setCellWidget(row, P_ROLE, role)
            is_value = plan.role == "value"
            var = QComboBox()
            var.setEditable(True)
            var.setModel(self._variable_model)
            var.setCurrentText(plan.variable_id)
            var.setEnabled(is_value)
            var.lineEdit().editingFinished.connect(
                lambda c=plan.column, w=var: self._edit_field(c, "variable_id", w.currentText()))
            var.activated.connect(lambda _i, c=plan.column, w=var: self._edit_field(c, "variable_id", w.currentText()))
            table.setCellWidget(row, P_VARIABLE, var)
            reg = QComboBox()
            reg.setEditable(True)
            for rid, rlabel in self._regions(plan.variable_id):
                reg.addItem(rid, rid)
                reg.setItemData(reg.count() - 1, rlabel, Qt.ToolTipRole)
            reg.setCurrentText(plan.region_id)
            reg.setEnabled(is_value)
            reg.lineEdit().editingFinished.connect(
                lambda c=plan.column, w=reg: self._edit_field(c, "region_id", w.currentText()))
            reg.activated.connect(lambda _i, c=plan.column, w=reg: self._edit_field(c, "region_id", w.currentText()))
            table.setCellWidget(row, P_REGION, reg)
            for col, field in ((P_REGION_LABEL, "region_label"), (P_UNIT, "unit")):
                edit = QLineEdit(getattr(plan, field))
                edit.setEnabled(is_value)
                edit.editingFinished.connect(lambda c=plan.column, f=field, w=edit: self._edit_field(c, f, w.text()))
                table.setCellWidget(row, col, edit)
            kind = QComboBox()
            kind.addItems(list(si.VALUE_KINDS))
            kind.setCurrentText(plan.value_kind)
            kind.setEnabled(is_value)
            kind.currentTextChanged.connect(lambda t, c=plan.column: self._edit_field(c, "value_kind", t))
            table.setCellWidget(row, P_KIND, kind)
            if plan.column in frame.columns:
                series = frame[plan.column]
                n = int(series.notna().sum())
                numeric = pd.to_numeric(series, errors="coerce").notna().sum()
                text = f"{n:,} · {100.0 * numeric / n:.0f}% numeric" if n else "empty"
            else:
                text = "—"
            values = QTableWidgetItem(text)
            values.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, P_VALUES, values)
            status_text, hue = self._status(plan)
            status = QTableWidgetItem(status_text)
            status.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            color = _hue(hue)
            if color is not None:
                status.setBackground(color)
            table.setItem(row, P_STATUS, status)
        table.resizeColumnToContents(P_COLUMN)
        table.resizeColumnToContents(P_ROLE)
        n_values = len([p for p in self._plans if p.role == "value" and p.variable_id])
        subject = next((p.column for p in self._plans if p.role == "subject"), None)
        self._plan_hint.setText(
            f"{n_values} value column(s); subject id column: {subject or 'none (mark one)'}. "
            f"Dataset: {self._catalog.get('root', '…')}")
        self._building = False

    def _edit_field(self, column: str, field: str, value: str) -> None:
        if self._building:
            return
        plan = self._plan(column)
        if plan is None or getattr(plan, field) == str(value).strip():
            return
        fields: dict[str, Any] = {field: str(value).strip()}
        if field == "region_id":
            label = dict(self._regions(plan.variable_id)).get(str(value).strip())
            if label and not plan.region_label:
                fields["region_label"] = label
        self._set_fields([column], **fields)

    def _plan_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building or item.column() != P_COLUMN:
            return
        plan = self._plans[item.row()] if item.row() < len(self._plans) else None
        if plan is None:
            return
        use = item.checkState() == Qt.Checked
        if use and plan.role == "skip":
            self._set_fields([plan.column], role="value")
        elif not use and plan.role != "skip":
            self._set_fields([plan.column], role="skip")

    def _focus_column(self, row: int) -> None:
        if 0 <= row < len(self._plans) and self._plans[row].column in self._frame.columns:
            col = list(self._frame.columns).index(self._plans[row].column)
            self._table.scrollTo(self._proxy.index(0, col))
            self._table.selectColumn(col)

    def _reguess(self) -> None:
        self._plans = si.guess_plan(self._frame, self.kind(), repo=self._repo, source_name=self._source_name)
        self._plans_changed()

    # ------------------------------------------------------------------ recipes
    def _save_recipe(self) -> None:
        if self._repo is None:
            return
        default = self._recipe_name.text().strip() or Path(self._source_name or "import").stem
        name, ok = QInputDialog.getText(self, "Save recipe", "Recipe name:", text=default)
        if ok and str(name).strip():
            path = si.save_recipe(self._repo, str(name).strip(), self.plan(), self._ops, header=int(self._header.value()))
            self._summary.setPlainText(f"Recipe saved: {path}")
            self._recipes.clear()
            self._recipes.addItems(si.list_recipes(self._repo))
            self._recipes.setCurrentText(str(name).strip())

    def _apply_recipe(self) -> None:
        name = self._recipes.currentText().strip()
        if not name or self._repo is None:
            return
        try:
            recipe = si.load_recipe(self._repo, name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Recipe", f"Could not read {name}: {exc}")
            return
        plan: si.ImportPlan = recipe["plan"]
        if int(recipe.get("header") or 0) != int(self._header.value()):
            self._header.setValue(int(recipe.get("header") or 0))
        frame, failed = self._raw, []
        ops = []
        for op in recipe.get("ops") or []:
            try:
                frame = si.apply_op(frame, op)
                ops.append(op)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{si.describe_op(op)}: {exc}")
        self._ops = ops
        self._history = [(op, self._snapshot(), self._snapshot()) for op in ops]
        self._redo = []
        index = self._kind.findData(plan.kind)
        if index >= 0:
            self._kind.setCurrentIndex(index)
        self._modality.setCurrentText(plan.modality)
        if plan.pipeline_id:
            self._pipeline.setCurrentText(plan.pipeline_id)
        self._pipeline_default.setChecked(plan.make_default_pipeline)
        self._visit.setCurrentText(plan.visit_id)
        self._mode.setCurrentIndex(max(self._mode.findData(plan.mode), 0))
        self._batch.clear()
        self._batch.setPlaceholderText(f"new batch id (the recipe's last: {plan.batch_id})")
        self._recipe_name.setText(name)
        by_name = {c.column: c for c in plan.columns}
        self._plans = [by_name[c] for c in map(str, frame.columns) if c in by_name]
        self._set_frame(frame)
        unmapped = [c for c in map(str, frame.columns) if c not in by_name]
        note = f"Applied recipe {name}: {len(ops)} edit(s)."
        if unmapped:
            note += f"\nColumns the recipe does not know (guessed): {', '.join(unmapped[:10])}"
        if failed:
            note += "\nEdits that did not apply:\n  " + "\n  ".join(failed)
        self._summary.setPlainText(note + "\nGive this import a batch id, then Check.")

    # ------------------------------------------------------------------ check / import
    def _invalidate_check(self) -> None:
        self._check = None
        if hasattr(self, "_import_btn"):
            self._import_btn.setEnabled(False)

    def _run_check(self) -> None:
        if self._repo is None:
            self._summary.setPlainText("The dataset is still being read; try again in a moment.")
            return
        if self._frame.empty:
            self._summary.setPlainText("Open a file first.")
            return
        plan = self.plan()
        worker = _CheckWorker(self._repo, self._frame.copy(), plan)
        worker.setParent(self)
        worker.done.connect(lambda check, p=plan: self._checked(check, p))
        worker.failed.connect(lambda m: self._summary.setPlainText(f"Check failed: {m}"))
        worker.finished.connect(lambda: self._check_btn.setEnabled(True))
        self._workers.append(worker)
        self._check_btn.setEnabled(False)
        self._summary.setPlainText("Comparing with the dataset…")
        worker.start()

    def _checked(self, check: si.ImportCheck, plan: si.ImportPlan) -> None:
        self._check = check
        lines = [check.summary(plan)]
        if not check.per_variable.empty:
            lines.append("")
            lines.append(check.per_variable.to_string(index=False, max_rows=40))
        self._summary.setPlainText("\n".join(lines))
        self._import_btn.setEnabled(check.ok)

    def _run_import(self) -> None:
        if self._check is None or not self._check.ok or self._repo is None:
            return
        plan = self.plan()
        answer = QMessageBox.question(
            self, "Import into the dataset",
            f"Write into the dataset at\n{self._repo.root}?\n\n{self._check.summary(plan)}",
        )
        if answer != QMessageBox.Yes:
            return
        recipe = ""
        if self._save_recipe_on_import.isChecked():
            recipe = self._recipe_name.text().strip() or Path(self._source_name or "import").stem
        worker = _ImportWorker(self._repo, self._frame.copy(), plan, list(self._ops), recipe,
                               int(self._header.value()), self._rebuild.isChecked())
        worker.setParent(self)
        worker.progress.connect(self._summary.appendPlainText)
        worker.done.connect(self._imported)
        worker.failed.connect(lambda m: self._summary.appendPlainText(f"Import failed: {m}"))
        worker.finished.connect(lambda: (self._check_btn.setEnabled(True)))
        self._workers.append(worker)
        self._import_btn.setEnabled(False)
        self._check_btn.setEnabled(False)
        self._summary.setPlainText("Importing…")
        worker.start()

    def _imported(self, result: dict) -> None:
        parts = [f"Imported into {result.get('table')}: {result.get('written', result.get('rows'))} row(s) written"]
        if result.get("replaced"):
            parts.append(f"{result['replaced']} replaced")
        if result.get("pipeline_registered"):
            parts.append("pipeline version registered")
        if result.get("recipe"):
            parts.append(f"recipe saved ({result['recipe']})")
        self._summary.appendPlainText("; ".join(parts) + ".")
        self._check = None
        self.imported.emit(result)
        self._load_catalog()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        busy = [w for w in self._workers if isinstance(w, _ImportWorker) and w.isRunning()]
        if busy:
            QMessageBox.information(self, "Import running", "The import is still writing; close when it finishes.")
            event.ignore()
            return
        super().closeEvent(event)


def open_import_editor(parent: QWidget | None = None, *, path: str | None = None, frame: pd.DataFrame | None = None,
                       source_name: str = "") -> ImportEditor:
    """Open an import editor window (on *path*, or on *frame*)."""
    editor = ImportEditor(parent, path=path, frame=frame, source_name=source_name)
    editor.setAttribute(Qt.WA_DeleteOnClose, True)
    editor.show()
    editor.raise_()
    return editor


__all__ = ["ImportEditor", "open_import_editor"]
