"""Statmodels explorer: interactive MixedLM formula builder over the NVITK DB."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.gui.tools.runner import notify
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.stats import (
    aggregate_territory_measurements,
    build_analysis_df_from_repo_frames,
    fit_or_load_mixedlm,
    melt_imaging_territories,
    plot_mixedlm_params,
    print_mixedlm_info,
)

log = Logger()

PIPELINE_KIND_QVTPY = "qvtpy"
PIPELINE_KIND_ASL = "asl"

_MODALITY_BY_KIND = {
    PIPELINE_KIND_QVTPY: "4dflow",
    PIPELINE_KIND_ASL: "asl",
}

_GROUPING_MODES = (
    ("Territory (melted)", "territory"),
    ("Region L/R (region_id)", "region_lr"),
    ("Region merged L/R", "region_merged_lr"),
)

_ASL_ATLASES = (
    ("vascular-0", "vascular-0"),
    ("vascular-8", "vascular-8"),
    ("vascular-12", "vascular-12"),
    ("desikan", "desikan"),
)

_FILTER_OPS = (">", ">=", "<", "<=", "==", "!=", "contains", "equals")

_FEATURE_ALIASES: dict[str, str] = {
    "flow": "flow_mean",
    "pwv": "pwv",
    "pi": "pi",
    "ri": "ri",
    "pitc": "pitc_slope",
    "mean_cbf": "mean_cbf",
    "att_mean": "att_mean",
    "att_median": "att_median",
    "cov_cbf": "cov_cbf",
    "att_cov": "att_cov",
}

_DEFAULT_FORMULA = (
    "flow ~ C(tacsctot_group, Treatment('None')) "
    "* C(group_key, Treatment('MCA')) + age_c + sex + Hematocrit"
)
_DEFAULT_GROUPS = "group_key"
_DEFAULT_RE = "0"
_DEFAULT_VC = '{"patient": "0 + C(subject_uid)"}'

_TABLE_ROW_CAP = 500

_LR_PREFIX_RE = re.compile(r"^(left_|right_|l_|r_)", re.IGNORECASE)

_DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLineEdit, QPlainTextEdit, QComboBox, QListWidget, QTableWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #555;
    selection-background-color: #3d6ea5;
}
QPushButton {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #666;
    padding: 4px 10px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #555;
}
QHeaderView::section {
    background-color: #333;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px;
}
QSplitter::handle {
    background-color: #444;
}
QScrollArea {
    border: none;
}
"""


def _repo() -> DataRepo:
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def _statmodels_root(repo: DataRepo) -> Path:
    root = Path(repo.root) / "nvitk-statmodels"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _apply_dark_theme(widget: QWidget) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(58, 58, 58))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight, QColor(61, 110, 165))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    widget.setPalette(palette)
    widget.setStyleSheet(_DARK_STYLESHEET)


def _parse_vc_formula(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"vc_formula must be a Python dict literal: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError('vc_formula must be a dict, e.g. {"patient": "0 + C(subject_uid)"}')
    return {str(k): str(v) for k, v in value.items()}


def _strip_lr_prefix(region_id: str) -> str:
    return _LR_PREFIX_RE.sub("", str(region_id))


def _resolve_feature_id(feature: str) -> str:
    text = str(feature).strip() or "flow_mean"
    return _FEATURE_ALIASES.get(text, text)


def _attach_group_key(territory_df: pd.DataFrame, grouping: str) -> pd.DataFrame:
    out = territory_df.copy()
    if grouping == "region_lr":
        out["group_key"] = out["region_id"].astype(str)
    elif grouping == "region_merged_lr":
        out["group_key"] = out["region_id"].astype(str).map(_strip_lr_prefix)
    else:
        out["group_key"] = out["territory"].astype(str)
    return out


def _merge_covariate_frames(
    repo: DataRepo,
    clinical_vars: list[str],
    cognitive_vars: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if clinical_vars:
        try:
            clinical = repo.clinical(variables=clinical_vars, wide=True)
            if clinical is not None and not clinical.empty:
                frames.append(clinical)
        except Exception as exc:
            log.warning("Could not load clinical covariates: %s", exc)
    if cognitive_vars:
        try:
            cognitive = repo.cognitive(variables=cognitive_vars, wide=True)
            if cognitive is not None and not cognitive.empty:
                frames.append(cognitive)
        except Exception as exc:
            log.warning("Could not load cognitive covariates: %s", exc)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    out = frames[0]
    for frame in frames[1:]:
        if "subject_uid" not in frame.columns or "subject_uid" not in out.columns:
            continue
        overlap = set(out.columns) & set(frame.columns) - {"subject_uid"}
        frame_use = frame.drop(columns=[c for c in overlap if c in frame.columns], errors="ignore")
        out = out.merge(frame_use, on="subject_uid", how="outer")
    return out


def _postprocess_analysis_df(
    analysis: pd.DataFrame,
    *,
    feature: str,
    variable_id: str,
) -> pd.DataFrame:
    rename: dict[str, str] = {}
    if "territory_base" in analysis.columns and "territory" not in analysis.columns:
        rename["territory_base"] = "territory"
    if "Hematocrit" not in analysis.columns and "hematocrit" in analysis.columns:
        rename["hematocrit"] = "Hematocrit"
    if rename:
        analysis = analysis.rename(columns=rename)

    if "patient_id" not in analysis.columns and "subject_uid" in analysis.columns:
        analysis["patient_id"] = analysis["subject_uid"]

    if "age" in analysis.columns and "age_c" not in analysis.columns:
        age = pd.to_numeric(analysis["age"], errors="coerce")
        analysis["age_c"] = age - age.mean(skipna=True)
    elif "age_at_mri" in analysis.columns and "age_c" not in analysis.columns:
        age = pd.to_numeric(analysis["age_at_mri"], errors="coerce")
        analysis["age_c"] = age - age.mean(skipna=True)

    if variable_id in analysis.columns and feature not in analysis.columns:
        if feature in _FEATURE_ALIASES or variable_id == "flow_mean":
            analysis[feature] = analysis[variable_id]
    return analysis


def _load_analysis_frame(
    repo: DataRepo,
    *,
    pipeline_kind: str,
    pipeline: str,
    feature: str,
    atlas: str | None,
    grouping: str,
    clinical_vars: list[str],
    cognitive_vars: list[str],
) -> pd.DataFrame:
    """Build a long analysis frame for image measurements + clinical/cognitive covariates."""
    modality = _MODALITY_BY_KIND.get(pipeline_kind, "4dflow")
    variable_id = _resolve_feature_id(feature)
    pipeline_sel = (pipeline or "latest").strip() or "latest"

    image_kwargs: dict[str, Any] = {
        "modality": modality,
        "pipeline": pipeline_sel,
        "variables": [variable_id],
        "wide": True,
    }
    if pipeline_kind == PIPELINE_KIND_ASL and atlas:
        image_kwargs["atlas"] = str(atlas).strip()

    image = repo.image(**image_kwargs)
    if image.empty:
        raise ValueError(
            f"No {modality!r} image measurements for variable={variable_id!r} "
            f"(pipeline={pipeline_sel!r})."
        )

    if modality == "4dflow":
        melt_kwargs: dict[str, Any] = {"flow_vars": [variable_id], "asl_vars": []}
    else:
        melt_kwargs = {"flow_vars": [], "asl_vars": [variable_id]}

    territory_df = melt_imaging_territories(image, **melt_kwargs)
    if territory_df.empty:
        raise ValueError("melt_imaging_territories returned no rows.")

    territory_df = _attach_group_key(territory_df, grouping)
    if grouping != "territory":
        territory_df = territory_df.copy()
        territory_df["territory"] = territory_df["group_key"]

    covariate_vars = list(dict.fromkeys([*clinical_vars, *cognitive_vars]))
    covariates = _merge_covariate_frames(repo, clinical_vars, cognitive_vars)
    present = [c for c in covariate_vars if c in covariates.columns] if not covariates.empty else []

    if covariates.empty or not present:
        wide = aggregate_territory_measurements(territory_df, [variable_id])
    else:
        wide = build_analysis_df_from_repo_frames(
            territory_df,
            covariates,
            imaging_variable_ids=[variable_id],
            covariate_cols=present,
        )
    wide["group_key"] = wide["territory"]

    return _postprocess_analysis_df(wide, feature=feature, variable_id=variable_id)


def _apply_row_filter(df: pd.DataFrame, column: str, op: str, value: str) -> pd.DataFrame:
    if df.empty or not column or column not in df.columns:
        return df
    series = df[column]
    op = (op or "==").strip()
    raw = value.strip()

    if op in {">", ">=", "<", "<="}:
        num = pd.to_numeric(series, errors="coerce")
        try:
            threshold = float(raw)
        except ValueError:
            return df.iloc[0:0]
        if op == ">":
            mask = num > threshold
        elif op == ">=":
            mask = num >= threshold
        elif op == "<":
            mask = num < threshold
        else:
            mask = num <= threshold
        return df.loc[mask]

    if op == "contains":
        return df.loc[series.astype(str).str.contains(raw, case=False, na=False)]
    if op == "equals":
        return df.loc[series.astype(str).str.lower() == raw.lower()]

    if op == "!=":
        return df.loc[series.astype(str) != raw]
    return df.loc[series.astype(str) == raw]


def _catalog_image_features(repo: DataRepo) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for alias in _FEATURE_ALIASES:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)
    for entry in repo.catalog.variable_entries(domain="image"):
        vid = str(entry.get("variable_id", "")).strip()
        if vid and vid not in seen:
            seen.add(vid)
            out.append(vid)
        for alias in entry.get("aliases") or []:
            text = str(alias).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _populate_checklist(widget: QListWidget, entries: list[dict[str, Any]]) -> None:
    widget.clear()
    for entry in entries:
        vid = str(entry.get("variable_id", "")).strip()
        if not vid:
            continue
        label = str(entry.get("label") or vid)
        item = QListWidgetItem(f"{label} ({vid})")
        item.setData(Qt.UserRole, vid)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        widget.addItem(item)


def _checked_variable_ids(widget: QListWidget) -> list[str]:
    out: list[str] = []
    for i in range(widget.count()):
        item = widget.item(i)
        if item.checkState() == Qt.Checked:
            vid = item.data(Qt.UserRole)
            if vid:
                out.append(str(vid))
    return out


def _set_checked_variable_ids(widget: QListWidget, ids: list[str]) -> None:
    want = {str(v).strip() for v in ids if str(v).strip()}
    for i in range(widget.count()):
        item = widget.item(i)
        vid = str(item.data(Qt.UserRole) or "")
        item.setCheckState(Qt.Checked if vid in want else Qt.Unchecked)


class StatmodelsWindow(QMainWindow):
    """Floating / maximizable MixedLM explorer."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_pipeline_kind: str = PIPELINE_KIND_QVTPY,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("nvitk Statmodels")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1400, 900)
        _apply_dark_theme(self)

        self._repo = _repo()
        self._initial_pipeline_kind = initial_pipeline_kind
        self._last_result = None
        self._last_df: pd.DataFrame | None = None
        self._last_meta: dict[str, Any] | None = None
        self._analysis_df: pd.DataFrame | None = None
        self._filtered_df: pd.DataFrame | None = None
        self._plot_canvas = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_data_controls())
        root.addWidget(self._build_filter_row())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

        self._status = QLabel(
            f"Dataset: {self._repo.root}  |  qvtpy pipeline: {QVTPY_PIPELINE_ID}"
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._pipeline_kind.currentIndexChanged.connect(self._on_pipeline_kind_changed)
        self._btn_reload.clicked.connect(self._on_reload)
        self._btn_apply_filter.clicked.connect(self._on_apply_filter)
        self._btn_clear_filter.clicked.connect(self._on_clear_filter)
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_plot.clicked.connect(self._on_plot)
        self._plot_mode.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_x.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._include_points.stateChanged.connect(lambda *_: self._on_plot())

        self._on_pipeline_kind_changed()
        self._populate_feature_combo()

    def _build_data_controls(self) -> QGroupBox:
        box = QGroupBox("Data selection")
        lay = QFormLayout(box)

        self._pipeline_kind = QComboBox()
        self._pipeline_kind.addItem("qvtpy (4D flow hemodynamics)", PIPELINE_KIND_QVTPY)
        self._pipeline_kind.addItem("ASL", PIPELINE_KIND_ASL)
        idx = self._pipeline_kind.findData(self._initial_pipeline_kind)
        if idx >= 0:
            self._pipeline_kind.setCurrentIndex(idx)

        self._pipeline = QComboBox()
        self._grouping = QComboBox()
        for label, key in _GROUPING_MODES:
            self._grouping.addItem(label, key)

        self._atlas = QComboBox()
        for label, key in _ASL_ATLASES:
            self._atlas.addItem(label, key)
        atlas_idx = self._atlas.findData("vascular-8")
        if atlas_idx >= 0:
            self._atlas.setCurrentIndex(atlas_idx)

        self._feature = QComboBox()
        self._feature.setEditable(True)

        self._btn_reload = QPushButton("Reload data")

        lay.addRow("Pipeline kind", self._pipeline_kind)
        lay.addRow("Measurement pipeline", self._pipeline)
        lay.addRow("Vessel / territory grouping", self._grouping)
        lay.addRow("ASL atlas", self._atlas)
        lay.addRow("Image feature", self._feature)
        lay.addRow("", self._btn_reload)
        return box

    def _build_filter_row(self) -> QGroupBox:
        box = QGroupBox("Pre-fit filter")
        row = QHBoxLayout(box)
        self._filter_col = QComboBox()
        self._filter_op = QComboBox()
        for op in _FILTER_OPS:
            self._filter_op.addItem(op)
        self._filter_val = QLineEdit()
        self._btn_apply_filter = QPushButton("Apply")
        self._btn_clear_filter = QPushButton("Clear")
        row.addWidget(QLabel("Column"))
        row.addWidget(self._filter_col, stretch=2)
        row.addWidget(QLabel("Op"))
        row.addWidget(self._filter_op)
        row.addWidget(QLabel("Value"))
        row.addWidget(self._filter_val, stretch=2)
        row.addWidget(self._btn_apply_filter)
        row.addWidget(self._btn_clear_filter)
        return box

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        cov_split = QSplitter(Qt.Vertical)

        clinical_box = QGroupBox("Clinical covariates")
        clinical_lay = QVBoxLayout(clinical_box)
        self._clinical_list = QListWidget()
        clinical_lay.addWidget(self._clinical_list)

        cognitive_box = QGroupBox("Cognitive covariates")
        cognitive_lay = QVBoxLayout(cognitive_box)
        self._cognitive_list = QListWidget()
        cognitive_lay.addWidget(self._cognitive_list)

        cov_split.addWidget(clinical_box)
        cov_split.addWidget(cognitive_box)
        cov_split.setStretchFactor(0, 1)
        cov_split.setStretchFactor(1, 1)
        lay.addWidget(cov_split)

        table_box = QGroupBox("Analysis dataframe (preview)")
        table_lay = QVBoxLayout(table_box)
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table_lay.addWidget(self._table)
        lay.addWidget(table_box, stretch=2)

        _populate_checklist(
            self._clinical_list, self._repo.catalog.variable_entries(domain="clinical")
        )
        _populate_checklist(
            self._cognitive_list, self._repo.catalog.variable_entries(domain="cognitive")
        )
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        form = QFormLayout()
        self._formula = QPlainTextEdit(_DEFAULT_FORMULA)
        self._formula.setFixedHeight(80)
        self._groups = QLineEdit(_DEFAULT_GROUPS)
        self._re_formula = QLineEdit(_DEFAULT_RE)
        self._vc_formula = QLineEdit(_DEFAULT_VC)
        self._model_name = QLineEdit("mixedlm_model")

        form.addRow("mm_formula", self._formula)
        form.addRow("groups", self._groups)
        form.addRow("re_formula", self._re_formula)
        form.addRow("vc_formula", self._vc_formula)
        form.addRow("Model name (save)", self._model_name)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit model")
        self._btn_save = QPushButton("Save model")
        self._btn_load = QPushButton("Load model…")
        self._btn_plot = QPushButton("Refresh plot")
        btn_row.addWidget(self._btn_fit)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_plot)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        plot_row = QHBoxLayout()
        self._plot_mode = QComboBox()
        self._plot_mode.addItem("auto", "auto")
        self._plot_mode.addItem("continuous (scatter + regression)", "continuous")
        self._plot_mode.addItem("categorical (marginal means)", "categorical")
        self._plot_x = QComboBox()
        self._include_points = QCheckBox("Include points")
        self._include_points.setChecked(True)
        plot_row.addWidget(QLabel("Plot mode"))
        plot_row.addWidget(self._plot_mode)
        plot_row.addWidget(QLabel("x"))
        plot_row.addWidget(self._plot_x, stretch=1)
        plot_row.addWidget(self._include_points)
        lay.addLayout(plot_row)

        splitter = QSplitter(Qt.Vertical)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText("MixedLM summary will appear here.")
        splitter.addWidget(self._report)

        self._plot_host = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_hint = QLabel("Parameter / EMM plots appear here after fitting.")
        self._plot_hint.setWordWrap(True)
        self._plot_layout.addWidget(self._plot_hint)
        splitter.addWidget(self._plot_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        lay.addWidget(splitter, stretch=1)
        return panel

    def show_maximized_floating(self) -> None:
        self.show()
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def _current_pipeline_kind(self) -> str:
        return str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)

    def _current_modality(self) -> str:
        return _MODALITY_BY_KIND.get(self._current_pipeline_kind(), "4dflow")

    def _populate_pipeline_combo(self) -> None:
        current = str(self._pipeline.currentData() or "latest")
        self._pipeline.blockSignals(True)
        self._pipeline.clear()
        self._pipeline.addItem("latest (catalog default)", "latest")
        for entry in self._repo.catalog.list_pipelines(modality=self._current_modality()):
            pid = str(entry.get("pipeline_id", "")).strip()
            if not pid:
                continue
            name = str(entry.get("pipeline_name") or pid)
            self._pipeline.addItem(f"{name} ({pid})", pid)
        idx = self._pipeline.findData(current)
        self._pipeline.setCurrentIndex(idx if idx >= 0 else 0)
        self._pipeline.blockSignals(False)

    def _populate_feature_combo(self) -> None:
        current = self._feature.currentText().strip() or "flow"
        self._feature.blockSignals(True)
        self._feature.clear()
        for feat in _catalog_image_features(self._repo):
            self._feature.addItem(feat)
        self._feature.setCurrentText(current)
        self._feature.blockSignals(False)

    def _on_pipeline_kind_changed(self) -> None:
        is_asl = self._current_pipeline_kind() == PIPELINE_KIND_ASL
        self._atlas.setEnabled(is_asl)
        self._populate_pipeline_combo()

    def _clinical_vars(self) -> list[str]:
        return _checked_variable_ids(self._clinical_list)

    def _cognitive_vars(self) -> list[str]:
        return _checked_variable_ids(self._cognitive_list)

    def _current_feature(self) -> str:
        return self._feature.currentText().strip() or "flow"

    def _working_df(self) -> pd.DataFrame | None:
        if self._filtered_df is not None:
            return self._filtered_df
        return self._analysis_df

    def _refresh_table(self, df: pd.DataFrame | None) -> None:
        self._table.clear()
        if df is None or df.empty:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._filter_col.clear()
            self._plot_x.clear()
            return

        preview = df.head(_TABLE_ROW_CAP)
        self._table.setRowCount(len(preview))
        self._table.setColumnCount(len(preview.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                val = preview.iloc[r, c]
                text = "" if pd.isna(val) else str(val)
                self._table.setItem(r, c, QTableWidgetItem(text))

        prev_col = self._filter_col.currentText()
        prev_x = self._plot_x.currentText()
        self._filter_col.blockSignals(True)
        self._plot_x.blockSignals(True)
        self._filter_col.clear()
        self._plot_x.clear()
        for col in preview.columns:
            self._filter_col.addItem(str(col))
            self._plot_x.addItem(str(col))
        idx = self._filter_col.findText(prev_col)
        if idx >= 0:
            self._filter_col.setCurrentIndex(idx)
        for cand in (prev_x, "tacsctot_group", "age_c", "group_key", "territory"):
            xi = self._plot_x.findText(cand)
            if xi >= 0:
                self._plot_x.setCurrentIndex(xi)
                break
        self._filter_col.blockSignals(False)
        self._plot_x.blockSignals(False)

    def _load_data(self) -> pd.DataFrame:
        atlas = None
        if self._current_pipeline_kind() == PIPELINE_KIND_ASL:
            atlas = str(self._atlas.currentData() or "vascular-8")
        return _load_analysis_frame(
            self._repo,
            pipeline_kind=self._current_pipeline_kind(),
            pipeline=str(self._pipeline.currentData() or "latest"),
            feature=self._current_feature(),
            atlas=atlas,
            grouping=str(self._grouping.currentData() or "territory"),
            clinical_vars=self._clinical_vars(),
            cognitive_vars=self._cognitive_vars(),
        )

    def _on_reload(self) -> None:
        try:
            df = self._load_data()
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))
            notify(f"Statmodels reload failed: {exc}", error=True)
            return
        self._analysis_df = df
        self._filtered_df = None
        self._refresh_table(df)
        self._status.setText(
            f"Loaded n={len(df)} rows  |  pipeline={self._pipeline.currentText()}  |  "
            f"dataset={self._repo.root}"
        )
        notify(f"Reloaded analysis frame ({len(df)} rows).")

    def _on_apply_filter(self) -> None:
        if self._analysis_df is None:
            notify("Reload data before applying a filter.", error=True)
            return
        col = self._filter_col.currentText()
        op = self._filter_op.currentText()
        val = self._filter_val.text()
        self._filtered_df = _apply_row_filter(self._analysis_df, col, op, val)
        self._refresh_table(self._filtered_df)
        notify(f"Filter applied: {len(self._filtered_df)} rows.")

    def _on_clear_filter(self) -> None:
        self._filtered_df = None
        self._filter_val.clear()
        self._refresh_table(self._analysis_df)
        notify("Filter cleared.")

    def _on_fit(self) -> None:
        feature = self._current_feature()
        formula = self._formula.toPlainText().strip()
        groups = self._groups.text().strip() or "group_key"
        re_formula = self._re_formula.text().strip() or "0"
        lhs = formula.split("~", 1)[0].strip()
        try:
            vc = _parse_vc_formula(self._vc_formula.text())
            if self._working_df() is None:
                df = self._load_data()
                self._analysis_df = df
                self._filtered_df = None
                self._refresh_table(df)
            else:
                df = self._working_df().copy()
            if lhs and lhs not in df.columns and feature in df.columns:
                df = df.rename(columns={feature: lhs})
            result, model_df, meta = fit_or_load_mixedlm(
                data=df,
                formula=formula,
                groups=groups,
                re_formula=re_formula,
                vc_formula=vc,
                overwrite=True,
                dropna_columns=None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            notify(f"Statmodels fit failed: {exc}", error=True)
            return

        self._last_result = result
        self._last_df = model_df
        self._last_meta = meta
        report = print_mixedlm_info(result, outcome_name=lhs or feature, group_name=groups)
        self._report.setPlainText(report)
        self._refresh_table(model_df)
        self._status.setText(
            f"Fitted n={meta.get('n_rows')}  |  groups={groups}  |  "
            f"re={re_formula!r}  |  dataset={self._repo.root}"
        )
        self._on_plot()
        notify("MixedLM fit complete.")

    def _clear_plot(self) -> None:
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._plot_canvas = None

    def _on_plot(self) -> None:
        if self._last_result is None or self._last_df is None:
            return
        self._clear_plot()
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            plt.style.use("dark_background")

            feature = self._current_feature()
            y = feature if feature in self._last_df.columns else None
            if y is None:
                for cand in ("flow", "flow_mean", "pi", "pwv", "pitc_slope", "mean_cbf"):
                    if cand in self._last_df.columns:
                        y = cand
                        break
            x = self._plot_x.currentText().strip()
            if not x or x not in self._last_df.columns:
                for cand in ("tacsctot_group", "age_c", "group_key"):
                    if cand in self._last_df.columns:
                        x = cand
                        break
            group = self._groups.text().strip() or "group_key"
            mode = str(self._plot_mode.currentData() or "auto")
            if not x or group not in self._last_df.columns or y is None or y not in self._last_df.columns:
                raise ValueError(
                    f"Cannot plot: need x/y/group columns (have {list(self._last_df.columns)})"
                )

            plot_mixedlm_params(
                result=self._last_result,
                df_fit=self._last_df,
                x=x,
                y=y,
                group=group,
                mode=mode,
                include_points=self._include_points.isChecked(),
                title=f"MixedLM: {y} ~ {x} | {group}",
            )
            fig = plt.gcf()
            fig.patch.set_facecolor("#2b2b2b")
            canvas = FigureCanvasQTAgg(fig)
            self._plot_layout.addWidget(canvas)
            self._plot_canvas = canvas
            canvas.draw_idle()
        except Exception as exc:
            err = QLabel(f"Plot unavailable: {exc}")
            err.setWordWrap(True)
            self._plot_layout.addWidget(err)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "pipeline_kind": self._current_pipeline_kind(),
            "pipeline": str(self._pipeline.currentData() or "latest"),
            "grouping": str(self._grouping.currentData() or "territory"),
            "atlas": str(self._atlas.currentData() or ""),
            "feature": self._current_feature(),
            "clinical": self._clinical_vars(),
            "cognitive": self._cognitive_vars(),
            "mm_formula": self._formula.toPlainText().strip(),
            "groups": self._groups.text().strip(),
            "re_formula": self._re_formula.text().strip(),
            "vc_formula": self._vc_formula.text().strip(),
            "model_name": self._model_name.text().strip(),
            "pipeline_id": QVTPY_PIPELINE_ID,
            "plot_mode": str(self._plot_mode.currentData() or "auto"),
            "plot_x": self._plot_x.currentText().strip(),
            "include_points": self._include_points.isChecked(),
        }

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        kind = str(cfg.get("pipeline_kind") or PIPELINE_KIND_QVTPY)
        idx = self._pipeline_kind.findData(kind)
        if idx >= 0:
            self._pipeline_kind.setCurrentIndex(idx)
        self._on_pipeline_kind_changed()

        pipeline = str(cfg.get("pipeline") or "latest")
        pidx = self._pipeline.findData(pipeline)
        if pidx >= 0:
            self._pipeline.setCurrentIndex(pidx)

        grouping = str(cfg.get("grouping") or "territory")
        gidx = self._grouping.findData(grouping)
        if gidx >= 0:
            self._grouping.setCurrentIndex(gidx)

        if "feature" in cfg:
            self._feature.setCurrentText(str(cfg["feature"]))

        atlas = str(cfg.get("atlas") or "")
        if atlas:
            aidx = self._atlas.findData(atlas)
            if aidx >= 0:
                self._atlas.setCurrentIndex(aidx)

        clinical = cfg.get("clinical")
        if isinstance(clinical, str):
            clinical = [c.strip() for c in clinical.split(",") if c.strip()]
        if isinstance(clinical, list):
            _set_checked_variable_ids(self._clinical_list, clinical)

        cognitive = cfg.get("cognitive")
        if isinstance(cognitive, list):
            _set_checked_variable_ids(self._cognitive_list, cognitive)

        if "mm_formula" in cfg:
            self._formula.setPlainText(str(cfg["mm_formula"]))
        if "groups" in cfg:
            self._groups.setText(str(cfg["groups"]))
        if "re_formula" in cfg:
            self._re_formula.setText(str(cfg["re_formula"]))
        if "vc_formula" in cfg:
            self._vc_formula.setText(str(cfg["vc_formula"]))
        if "model_name" in cfg:
            self._model_name.setText(str(cfg["model_name"]))

        plot_mode = str(cfg.get("plot_mode") or "")
        if plot_mode:
            midx = self._plot_mode.findData(plot_mode)
            if midx >= 0:
                self._plot_mode.setCurrentIndex(midx)
        if cfg.get("plot_x"):
            self._plot_x.setCurrentText(str(cfg["plot_x"]))
        if "include_points" in cfg:
            self._include_points.setChecked(bool(cfg["include_points"]))

    def _on_save(self) -> None:
        if self._last_result is None:
            notify("Fit a model before saving.", error=True)
            return
        name = self._model_name.text().strip() or "model"
        out_dir = _statmodels_root(self._repo) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl = out_dir / "model.pkl"
        cfg_path = out_dir / "config.json"
        info_path = out_dir / "info.txt"
        try:
            self._last_result.save(str(pkl))
            cfg_path.write_text(json.dumps(self._config_dict(), indent=2), encoding="utf-8")
            info_path.write_text(self._report.toPlainText(), encoding="utf-8")
        except Exception as exc:
            notify(f"Save failed: {exc}", error=True)
            return
        notify(f"Saved model → {out_dir}")
        self._status.setText(f"Saved {out_dir}")

    def _on_load(self) -> None:
        start = str(_statmodels_root(self._repo))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Statmodels config or pickle",
            start,
            "Config/Model (config.json model.pkl);;All (*)",
        )
        if not path:
            return
        p = Path(path)
        model_dir = p.parent if p.name in {"config.json", "model.pkl", "info.txt"} else p
        cfg_path = model_dir / "config.json"
        pkl = model_dir / "model.pkl"
        try:
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self._apply_config(cfg)
            if pkl.is_file():
                from statsmodels.regression.mixed_linear_model import MixedLMResults

                self._last_result = MixedLMResults.load(str(pkl))
                report = print_mixedlm_info(self._last_result)
                self._report.setPlainText(report)
                try:
                    df = self._load_data()
                    self._analysis_df = df
                    self._filtered_df = None
                    self._last_df = df
                    self._refresh_table(df)
                    self._on_plot()
                except Exception:
                    pass
            notify(f"Loaded model from {model_dir}")
        except Exception as exc:
            notify(f"Load failed: {exc}", error=True)


class StatmodelsPanel(QWidget):
    """Right-tab launcher for the floating Statmodels window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._window: StatmodelsWindow | None = None

        self._pipeline_kind = QComboBox()
        self._pipeline_kind.addItem("qvtpy (4D flow hemodynamics)", PIPELINE_KIND_QVTPY)
        self._pipeline_kind.addItem("ASL", PIPELINE_KIND_ASL)

        self._btn = QPushButton("Open Statmodels window")
        self._btn.clicked.connect(self._open_window)

        hint = QLabel(
            "Explore MixedLM formulas over image / clinical / cognitive measurements "
            "from the dataset catalog. Models are saved under "
            "<dataset>/nvitk-statmodels/."
        )
        hint.setWordWrap(True)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Pipeline kind"))
        lay.addWidget(self._pipeline_kind)
        lay.addWidget(self._btn)
        lay.addWidget(hint)
        lay.addStretch(1)
        self.setLayout(lay)

    def _open_window(self) -> None:
        kind = str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)
        if self._window is None:
            self._window = StatmodelsWindow(initial_pipeline_kind=kind)
        else:
            idx = self._window._pipeline_kind.findData(kind)
            if idx >= 0:
                self._window._pipeline_kind.setCurrentIndex(idx)
        self._window.show_maximized_floating()
