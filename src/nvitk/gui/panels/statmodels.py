"""Statmodels explorer: interactive MixedLM formula builder over the NVITK DB."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QSlider,
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
    build_long_analysis_frame,
    features_for_kind,
    fit_or_load_mixedlm,
    grouping_choices_for,
    plot_mixedlm_params,
    print_mixedlm_info,
    resolve_feature_id,
    subject_attribute_entries,
)

log = Logger()

PIPELINE_KIND_QVTPY = "qvtpy"
PIPELINE_KIND_ASL = "asl"
PIPELINE_KIND_T1 = "t1"
PIPELINE_KIND_FLAIR = "flair"
PIPELINE_KIND_TOF = "tof"

_PIPELINE_KIND_ITEMS = (
    ("qvtpy — 4D flow hemodynamics", PIPELINE_KIND_QVTPY),
    ("ASL — perfusion (CBF / ATT)", PIPELINE_KIND_ASL),
    ("T1 — volumetry", PIPELINE_KIND_T1),
    ("FLAIR — WMH", PIPELINE_KIND_FLAIR),
    ("TOF — morphometrics (eICAB)", PIPELINE_KIND_TOF),
)

_ASL_ATLASES = (
    ("Desikan (cortical parcels)", "desikan"),
    ("Vascular atlas · smooth 0", "vascular-0"),
    ("Vascular atlas · smooth 8", "vascular-8"),
    ("Vascular atlas · smooth 12", "vascular-12"),
)

_FILTER_OPS = (">", ">=", "<", "<=", "==", "!=", "contains", "equals")

_DEFAULT_FORMULA = (
    "flow_mean ~ C(tacsctot_group, Treatment('None')) "
    "* C(group_key) + age_c + sex + Hematocrit"
)
_DEFAULT_GROUPS = "group_key"
_DEFAULT_RE = "0"
_DEFAULT_VC = '{"patient": "0 + C(subject_uid)"}'

_TABLE_ROW_CAP = 500

# Tukey fence multiplier for the optional IQR outlier filter.
_DEFAULT_IQR_K = 1.5
# Slider resolution for the axis-limit controls, and how far past the plotted data they may reach.
_AXIS_SLIDER_STEPS = 1000
_AXIS_SLIDER_MARGIN = 0.25

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
    """Open the configured dataset repo, unwrapping the ``(repo, ...)`` tuple form if returned."""
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def _statmodels_root(repo: DataRepo) -> Path:
    """Ensure and return the ``nvitk-statmodels`` scratch directory under the dataset root, for cached
    fits and saved configs."""
    root = Path(repo.root) / "nvitk-statmodels"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _apply_dark_theme(widget: QWidget) -> None:
    """Apply the explorer's dark Qt palette and stylesheet to *widget*."""
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
    """Parse the variance-components formula field (a Python dict literal, e.g.
    ``{"patient": "0 + C(subject_uid)"}``) into a ``{group: formula}`` dict, or ``None`` if blank;
    raises ``ValueError`` if it isn't a dict literal."""
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
    return build_long_analysis_frame(
        repo,
        pipeline_kind=pipeline_kind,
        pipeline=pipeline,
        feature=feature,
        grouping=grouping,
        atlas=atlas,
        clinical_vars=clinical_vars,
        cognitive_vars=cognitive_vars,
    )


def _apply_row_filter(df: pd.DataFrame, column: str, op: str, value: str) -> pd.DataFrame:
    """Filter *df* to rows where *column* satisfies *op* against *value* (numeric comparisons,
    substring/exact string match, or equality/inequality); returns *df* unchanged if *column* is
    missing or blank."""
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


def _whiten_figure(fig: Any) -> None:
    """Force a white figure/axes background with dark text, so plots stay readable against the
    explorer's dark chrome and export cleanly."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.tick_params(colors="#222222", which="both")
        ax.xaxis.label.set_color("#111111")
        ax.yaxis.label.set_color("#111111")
        ax.title.set_color("#111111")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("#999999")
            for text in legend.get_texts():
                text.set_color("#111111")


def _apply_iqr_filter(
    df: pd.DataFrame,
    column: str,
    *,
    k: float = _DEFAULT_IQR_K,
    by: str | None = None,
) -> pd.DataFrame:
    """
    Drop rows whose *column* value falls outside the Tukey fences ``[Q1 - k·IQR, Q3 + k·IQR]``.

    With *by* set, the fences are computed within each level of that column. That is the meaningful
    scope for image measurements, whose magnitude depends on the region: a global fence is driven by
    the spread *between* regions rather than within them, so it misses real outliers inside each
    region and — when one region sits far from the rest — can discard that region wholesale. Rows
    with a missing value are kept; dropping them is the fit's job, not the filter's.
    """
    if df.empty or column not in df.columns:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    if not values.notna().any():
        return df

    def fence_mask(series: pd.Series) -> pd.Series:
        """Boolean mask of rows inside the Tukey fences of *series* (NaN kept)."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            return pd.Series(True, index=series.index)
        iqr = q3 - q1
        if iqr <= 0:  # degenerate spread: nothing is an outlier by this rule
            return pd.Series(True, index=series.index)
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return series.isna() | series.between(lo, hi)

    if by and by in df.columns:
        mask = pd.Series(True, index=df.index)
        for _, idx in df.groupby(by, sort=False).groups.items():
            mask.loc[idx] = fence_mask(values.loc[idx])
    else:
        mask = fence_mask(values)
    return df.loc[mask]


def _dropped_rows_note(meta: dict[str, Any]) -> str:
    """Human-readable note about rows dropped for missing values during a fit, or ``""`` if none were."""
    dropped = int(meta.get("n_rows_dropped") or 0)
    if dropped <= 0:
        return ""
    by_col = dict(meta.get("dropped_by_column") or {})
    detail = ", ".join(f"{col} ({n})" for col, n in sorted(by_col.items(), key=lambda kv: -kv[1]))
    return (
        f"NOTE: dropped {dropped} of {meta.get('n_rows_input')} rows with missing values "
        f"before fitting (n={meta.get('n_rows')})."
        + (f" Missing per column: {detail}." if detail else "")
    )


def _populate_checklist(widget: QListWidget, entries: list[dict[str, Any]]) -> None:
    """Fill *widget* with one unchecked, checkable item per variable *entries*, labeled
    ``"<label> (<variable_id>)"``."""
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
    """Variable ids of every checked item in *widget*."""
    out: list[str] = []
    for i in range(widget.count()):
        item = widget.item(i)
        if item.checkState() == Qt.Checked:
            vid = item.data(Qt.UserRole)
            if vid:
                out.append(str(vid))
    return out


def _set_checked_variable_ids(widget: QListWidget, ids: list[str]) -> None:
    """Check the items in *widget* whose variable id is in *ids*, uncheck the rest."""
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
        """Build the three-column control row (covariates | data selection + filters | model
        formulation), the plot/report row beneath it, and wire up all signal handlers."""
        super().__init__(parent)
        self.setWindowTitle("nvitk Statmodels")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1600, 950)
        _apply_dark_theme(self)

        self._repo = _repo()
        self._initial_pipeline_kind = initial_pipeline_kind
        self._last_result = None
        self._last_df: pd.DataFrame | None = None
        self._last_meta: dict[str, Any] | None = None
        self._analysis_df: pd.DataFrame | None = None
        self._filtered_df: pd.DataFrame | None = None
        self._plot_canvas = None
        self._plot_fig = None
        self._plot_axes = None
        self._axis_base: dict[str, tuple[float, float]] = {}
        self._axis_span: dict[str, tuple[float, float]] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Controls on top (three columns), plots + model info underneath, so the figure gets the
        # full window width and most of its height.
        controls = QSplitter(Qt.Horizontal)
        controls.addWidget(self._build_left_panel())
        controls.addWidget(self._build_center_panel())
        controls.addWidget(self._build_formula_panel())
        controls.setSizes([440, 560, 520])

        output = QSplitter(Qt.Horizontal)
        output.addWidget(self._build_plot_panel())
        output.addWidget(self._build_report_panel())
        output.setStretchFactor(0, 3)
        output.setStretchFactor(1, 1)
        output.setSizes([1200, 400])

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(controls)
        main_split.addWidget(output)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([340, 610])
        root.addWidget(main_split, stretch=1)

        self._status = QLabel(
            f"Dataset: {self._repo.root}  |  qvtpy pipeline: {QVTPY_PIPELINE_ID}"
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._pipeline_kind.currentIndexChanged.connect(self._on_pipeline_kind_changed)
        self._feature.currentIndexChanged.connect(self._on_feature_changed)
        self._feature.editTextChanged.connect(self._on_feature_changed)
        self._btn_reload.clicked.connect(self._on_reload)
        self._btn_apply_filter.clicked.connect(self._on_apply_filter)
        self._btn_clear_filter.clicked.connect(self._on_clear_filter)
        self._btn_iqr.toggled.connect(self._on_iqr_toggled)
        self._iqr_k.valueChanged.connect(self._on_iqr_param_changed)
        self._iqr_scope.currentIndexChanged.connect(self._on_iqr_param_changed)
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_plot.clicked.connect(self._on_plot)
        self._plot_mode.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_x.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._include_points.stateChanged.connect(lambda *_: self._on_plot())
        for slider in self._axis_sliders.values():
            slider.valueChanged.connect(self._on_axis_slider_changed)
        self._btn_reset_axes.clicked.connect(self._on_reset_axes)
        self._show_legend.stateChanged.connect(self._on_legend_toggled)

        self._on_pipeline_kind_changed()

    def _build_data_controls(self) -> QGroupBox:
        """Build the pipeline-kind/pipeline/feature/atlas/grouping selector form."""
        box = QGroupBox("Data selection")
        lay = QFormLayout(box)

        self._pipeline_kind = QComboBox()
        for label, key in _PIPELINE_KIND_ITEMS:
            self._pipeline_kind.addItem(label, key)
        idx = self._pipeline_kind.findData(self._initial_pipeline_kind)
        if idx >= 0:
            self._pipeline_kind.setCurrentIndex(idx)

        self._pipeline = QComboBox()
        self._feature = QComboBox()
        self._feature.setEditable(True)
        self._grouping = QComboBox()
        self._atlas = QComboBox()
        for label, key in _ASL_ATLASES:
            self._atlas.addItem(label, key)
        atlas_idx = self._atlas.findData("vascular-8")
        if atlas_idx >= 0:
            self._atlas.setCurrentIndex(atlas_idx)

        self._atlas_label = QLabel("ASL atlas / smoothing")
        self._grouping_label = QLabel("Grouping")
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #a0a0a0; font-weight: normal;")

        self._btn_reload = QPushButton("Reload data")

        lay.addRow("Pipeline kind", self._pipeline_kind)
        lay.addRow("Measurement pipeline", self._pipeline)
        lay.addRow("Image feature", self._feature)
        lay.addRow(self._atlas_label, self._atlas)
        lay.addRow(self._grouping_label, self._grouping)
        lay.addRow("", self._hint)
        lay.addRow("", self._btn_reload)
        return box

    def _build_filter_row(self) -> QGroupBox:
        """Build the column/operator/value row plus the optional IQR outlier filter for the image
        measurements, both applied to the analysis frame before fitting."""
        box = QGroupBox("Pre-fit filter")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
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
        lay.addLayout(row)

        iqr_row = QHBoxLayout()
        self._btn_iqr = QPushButton("IQR filter")
        self._btn_iqr.setCheckable(True)
        self._btn_iqr.setToolTip(
            "Drop image measurements outside [Q1 - k·IQR, Q3 + k·IQR].\n"
            "'per group_key' computes the fences within each region — the right scope when regions "
            "differ in magnitude (e.g. LMCA ~200 vs LPCOMM ~20 mL/min), since a global fence is set "
            "by the spread between regions and misses outliers within them."
        )
        self._iqr_k = QDoubleSpinBox()
        self._iqr_k.setRange(0.1, 10.0)
        self._iqr_k.setSingleStep(0.25)
        self._iqr_k.setValue(_DEFAULT_IQR_K)
        self._iqr_scope = QComboBox()
        self._iqr_scope.addItem("per group_key", "group")
        self._iqr_scope.addItem("global", "global")
        self._iqr_status = QLabel("")
        self._iqr_status.setStyleSheet("color: #a0a0a0; font-weight: normal;")

        iqr_row.addWidget(self._btn_iqr)
        iqr_row.addWidget(QLabel("k"))
        iqr_row.addWidget(self._iqr_k)
        iqr_row.addWidget(QLabel("scope"))
        iqr_row.addWidget(self._iqr_scope)
        iqr_row.addWidget(self._iqr_status, stretch=1)
        lay.addLayout(iqr_row)
        return box

    def _build_center_panel(self) -> QWidget:
        """Middle column: data selection above the pre-fit filter."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._build_data_controls())
        lay.addWidget(self._build_filter_row())
        lay.addStretch(1)
        return panel

    def _build_left_panel(self) -> QWidget:
        """Build the clinical/cognitive covariate checklists and the analysis-dataframe preview table."""
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

        # Clinical catalog variables, plus any leftover ``subjects``-table attributes that are not
        # already available as clinical measurements (see :func:`subject_attribute_entries`).
        self._refresh_covariate_lists()
        return panel

    def _refresh_covariate_lists(self) -> None:
        """Re-read the catalog and rebuild the clinical/cognitive checklists, preserving checks.

        Call this on Reload so newly imported variables (e.g. ``sex`` from ``import_sex.py``) appear
        and so a stale in-memory catalog cannot keep offering the sparse ``subjects.sex`` column.
        """
        try:
            self._repo.catalog.refresh()
        except Exception:
            pass
        clinical_checked = _checked_variable_ids(self._clinical_list)
        cognitive_checked = _checked_variable_ids(self._cognitive_list)
        _populate_checklist(
            self._clinical_list,
            [
                *subject_attribute_entries(self._repo),
                *self._repo.catalog.variable_entries(domain="clinical"),
            ],
        )
        _populate_checklist(
            self._cognitive_list, self._repo.catalog.variable_entries(domain="cognitive")
        )
        _set_checked_variable_ids(self._clinical_list, clinical_checked)
        _set_checked_variable_ids(self._cognitive_list, cognitive_checked)

    def _build_formula_panel(self) -> QWidget:
        """Right column: model formulation fields, fit/save/load buttons, and plot options."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Model formulation")
        box_lay = QVBoxLayout(box)

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
        box_lay.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit model")
        self._btn_save = QPushButton("Save model")
        self._btn_load = QPushButton("Load model…")
        self._btn_plot = QPushButton("Refresh plot")
        btn_row.addWidget(self._btn_fit)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_plot)
        box_lay.addLayout(btn_row)

        plot_form = QFormLayout()
        self._plot_mode = QComboBox()
        self._plot_mode.addItem("auto", "auto")
        self._plot_mode.addItem("continuous (scatter + regression)", "continuous")
        self._plot_mode.addItem("categorical (marginal means)", "categorical")
        self._plot_x = QComboBox()
        self._include_points = QCheckBox("Include points")
        self._include_points.setChecked(True)
        plot_form.addRow("Plot mode", self._plot_mode)
        plot_form.addRow("Plot x", self._plot_x)
        plot_form.addRow("", self._include_points)
        box_lay.addLayout(plot_form)

        lay.addWidget(box)
        lay.addStretch(1)
        return panel

    def _build_plot_panel(self) -> QWidget:
        """Bottom-left pane: the Matplotlib canvas plus the axis-limit sliders."""
        box = QGroupBox("Plots")
        lay = QVBoxLayout(box)

        self._plot_host = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_hint = QLabel("Parameter / EMM plots appear here after fitting.")
        self._plot_hint.setWordWrap(True)
        self._plot_layout.addWidget(self._plot_hint)
        lay.addWidget(self._plot_host, stretch=1)
        lay.addWidget(self._build_axis_controls())
        return box

    def _build_axis_controls(self) -> QWidget:
        """Legend toggle plus sliders that rescale the current figure's axes without recomputing."""
        box = QGroupBox("Axis limits")
        lay = QVBoxLayout(box)

        legend_row = QHBoxLayout()
        self._show_legend = QCheckBox("Show legend")
        self._show_legend.setChecked(True)
        self._show_legend.setToolTip("Show or hide the plot legend (model curves only; raw points are never listed).")
        legend_row.addWidget(self._show_legend)
        legend_row.addStretch(1)
        lay.addLayout(legend_row)

        grid = QHBoxLayout()
        self._axis_sliders: dict[str, QSlider] = {}
        self._axis_value_labels: dict[str, QLabel] = {}
        for axis, bound, text in (
            ("x", "min", "x min"),
            ("x", "max", "x max"),
            ("y", "min", "y min"),
            ("y", "max", "y max"),
        ):
            key = f"{axis}{bound}"
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, _AXIS_SLIDER_STEPS)
            slider.setEnabled(False)
            value_label = QLabel("—")
            value_label.setMinimumWidth(64)
            value_label.setStyleSheet("color: #a0a0a0; font-weight: normal;")
            self._axis_sliders[key] = slider
            self._axis_value_labels[key] = value_label
            grid.addWidget(QLabel(text))
            grid.addWidget(slider, stretch=1)
            grid.addWidget(value_label)

        self._btn_reset_axes = QPushButton("Reset")
        self._btn_reset_axes.setEnabled(False)
        grid.addWidget(self._btn_reset_axes)
        lay.addLayout(grid)
        return box

    def _build_report_panel(self) -> QWidget:
        """Bottom-right pane: the MixedLM summary text."""
        box = QGroupBox("Model info")
        lay = QVBoxLayout(box)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText("MixedLM summary will appear here.")
        lay.addWidget(self._report)
        return box

    def show_maximized_floating(self) -> None:
        """Show, maximize, raise, and focus this window."""
        self.show()
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def _current_pipeline_kind(self) -> str:
        """The currently selected pipeline kind (``"qvtpy"``, ``"asl"``, ``"t1"``, etc.)."""
        return str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)

    def _current_modality(self) -> str:
        """Imaging modality string corresponding to the currently selected pipeline kind."""
        from nvitk.stats import KIND_MODALITY

        return KIND_MODALITY.get(self._current_pipeline_kind(), "4dflow")

    def _populate_pipeline_combo(self) -> None:
        """Repopulate the measurement-pipeline combo with entries registered for the current modality,
        preserving the current selection if still valid."""
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
        """Repopulate the image-feature combo with features for the current pipeline kind, preserving
        the current text if still valid."""
        kind = self._current_pipeline_kind()
        feats = features_for_kind(kind)
        current = self._feature.currentText().strip()
        self._feature.blockSignals(True)
        self._feature.clear()
        for feat in feats:
            self._feature.addItem(feat)
        if current and current in feats:
            self._feature.setCurrentText(current)
        elif feats:
            self._feature.setCurrentIndex(0)
        self._feature.blockSignals(False)

    def _populate_grouping_combo(self) -> None:
        """Repopulate the grouping combo with choices valid for the current pipeline kind/feature."""
        kind = self._current_pipeline_kind()
        feature = self._current_feature()
        choices = grouping_choices_for(kind, feature)
        current = str(self._grouping.currentData() or "")
        self._grouping.blockSignals(True)
        self._grouping.clear()
        for label, key in choices:
            self._grouping.addItem(label, key)
        idx = self._grouping.findData(current)
        self._grouping.setCurrentIndex(idx if idx >= 0 else 0)
        self._grouping.blockSignals(False)

    def _sync_atlas_visibility(self) -> None:
        """Show the ASL-atlas picker only when the current pipeline kind is ASL."""
        is_asl = self._current_pipeline_kind() == PIPELINE_KIND_ASL
        self._atlas.setEnabled(is_asl)
        self._atlas_label.setEnabled(is_asl)
        self._atlas.setVisible(is_asl)
        self._atlas_label.setVisible(is_asl)

    def _sync_hint(self) -> None:
        """Update the explanatory hint text for the currently selected pipeline kind/feature."""
        kind = self._current_pipeline_kind()
        vid = resolve_feature_id(self._current_feature())
        if kind == PIPELINE_KIND_QVTPY:
            if vid in {"pwv", "pwv_fielding_xcor", "pitc_slope", "pitc_intercept"}:
                self._hint.setText(
                    "Tree metrics: one value per arterial root (L_ICA / R_ICA / Basilar). "
                    "Hemisphere grouping averages L/R ICA and keeps Basilar."
                )
            else:
                self._hint.setText(
                    "Vessel-wise LOC metrics (flow_mean / pi / ri). "
                    "Hemisphere grouping averages left/right pairs (e.g. LMCA+RMCA → MCA)."
                )
        elif kind == PIPELINE_KIND_ASL:
            self._hint.setText(
                "Pick Desikan or one vascular-atlas smoothing (0 / 8 / 12). "
                "Only that atlas’s regions are loaded."
            )
        elif kind == PIPELINE_KIND_T1:
            self._hint.setText(
                "T1 cortical vs subcortical volume — regions come from the matching atlas."
            )
        elif kind == PIPELINE_KIND_FLAIR:
            self._hint.setText("FLAIR WMH metrics by published region_id.")
        elif kind == PIPELINE_KIND_TOF:
            self._hint.setText(
                "TOF morphometrics by eICAB vessel. "
                "Hemisphere grouping averages L/R pairs (e.g. LICA+RICA → ICA)."
            )
        else:
            self._hint.setText("")

    def _on_pipeline_kind_changed(self) -> None:
        """Repopulate every dependent combo (pipeline/feature/grouping) and sync visibility/hints for
        the newly selected pipeline kind."""
        self._populate_pipeline_combo()
        self._populate_feature_combo()
        self._populate_grouping_combo()
        self._sync_atlas_visibility()
        self._sync_hint()

    def _on_feature_changed(self, *_args: Any) -> None:
        """Repopulate the grouping combo and refresh the hint for the newly selected feature."""
        self._populate_grouping_combo()
        self._sync_hint()

    def _clinical_vars(self) -> list[str]:
        """Checked clinical covariate variable ids."""
        return _checked_variable_ids(self._clinical_list)

    def _cognitive_vars(self) -> list[str]:
        """Checked cognitive covariate variable ids."""
        return _checked_variable_ids(self._cognitive_list)

    def _current_feature(self) -> str:
        """Currently entered/selected image feature name, falling back to the first available feature
        for the current pipeline kind."""
        text = self._feature.currentText().strip()
        if text:
            return text
        feats = features_for_kind(self._current_pipeline_kind())
        return feats[0] if feats else "flow_mean"

    def _working_df(self) -> pd.DataFrame | None:
        """The filtered analysis frame if a filter is active, else the raw loaded frame."""
        if self._filtered_df is not None:
            return self._filtered_df
        return self._analysis_df

    def _refresh_table(self, df: pd.DataFrame | None) -> None:
        """Repopulate the preview table (capped at ``_TABLE_ROW_CAP`` rows) and the filter/plot-x
        column combos from *df*'s columns."""
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
        """Build the long analysis frame for the currently selected pipeline/feature/grouping/atlas
        and checked covariates."""
        atlas = None
        if self._current_pipeline_kind() == PIPELINE_KIND_ASL:
            atlas = str(self._atlas.currentData() or "vascular-8")
        choices = grouping_choices_for(
            self._current_pipeline_kind(), self._current_feature()
        )
        default_grouping = choices[0][1] if choices else "vessel"
        return _load_analysis_frame(
            self._repo,
            pipeline_kind=self._current_pipeline_kind(),
            pipeline=str(self._pipeline.currentData() or "latest"),
            feature=self._current_feature(),
            atlas=atlas,
            grouping=str(self._grouping.currentData() or default_grouping),
            clinical_vars=self._clinical_vars(),
            cognitive_vars=self._cognitive_vars(),
        )

    def _on_reload(self) -> None:
        """Reload the analysis frame, clear any active filter, and refresh the preview table/status."""
        self._refresh_covariate_lists()
        try:
            df = self._load_data()
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))
            notify(f"Statmodels reload failed: {exc}", error=True)
            return
        self._analysis_df = df
        self._filtered_df = None
        self._refresh_table(df)
        # Keep an active row/IQR filter applied across reloads.
        self._recompute_filters(announce=False)
        working = self._working_df()
        self._status.setText(
            f"Loaded n={len(df)} rows"
            + (f" (filtered to {len(working)})" if working is not None and len(working) != len(df) else "")
            + f"  |  pipeline={self._pipeline.currentText()}  |  dataset={self._repo.root}"
        )
        notify(f"Reloaded analysis frame ({len(df)} rows).")

    def _feature_column(self, df: pd.DataFrame) -> str | None:
        """The analysis-frame column holding the current image measurements, if present."""
        for cand in (resolve_feature_id(self._current_feature()), self._current_feature()):
            if cand and cand in df.columns:
                return cand
        return None

    def _recompute_filters(self, *, announce: bool = True) -> None:
        """
        Re-derive the working frame from the loaded analysis frame by applying the row filter and
        then the optional IQR filter, and refresh the preview table.

        Both filters are recomputed from the untouched ``_analysis_df`` every time, so toggling one
        never compounds on a previously filtered frame.
        """
        base = self._analysis_df
        if base is None:
            return

        out = base
        row_filter_active = bool(self._filter_val.text().strip())
        if row_filter_active:
            out = _apply_row_filter(
                out,
                self._filter_col.currentText(),
                self._filter_op.currentText(),
                self._filter_val.text(),
            )

        iqr_note = ""
        if self._btn_iqr.isChecked():
            column = self._feature_column(out)
            if column is None:
                iqr_note = "IQR: feature column not in frame"
            else:
                scope = str(self._iqr_scope.currentData() or "group")
                by = "group_key" if scope == "group" and "group_key" in out.columns else None
                before = len(out)
                out = _apply_iqr_filter(out, column, k=float(self._iqr_k.value()), by=by)
                iqr_note = (
                    f"IQR on {column} (k={self._iqr_k.value():g}, "
                    f"{'per group_key' if by else 'global'}): removed {before - len(out)}"
                )
        self._iqr_status.setText(iqr_note)

        self._filtered_df = out if (row_filter_active or self._btn_iqr.isChecked()) else None
        self._refresh_table(self._working_df())
        if announce:
            notify(f"Filters applied: {len(out)} of {len(base)} rows.")

    def _on_apply_filter(self) -> None:
        """Apply the filter row's column/op/value (plus any IQR filter) and refresh the preview."""
        if self._analysis_df is None:
            notify("Reload data before applying a filter.", error=True)
            return
        self._recompute_filters()

    def _on_clear_filter(self) -> None:
        """Clear both the row filter and the IQR filter, restoring the full analysis frame."""
        self._filter_val.clear()
        self._btn_iqr.blockSignals(True)
        self._btn_iqr.setChecked(False)
        self._btn_iqr.blockSignals(False)
        self._filtered_df = None
        self._iqr_status.setText("")
        self._refresh_table(self._analysis_df)
        notify("Filters cleared.")

    def _on_iqr_toggled(self, checked: bool) -> None:
        """Apply or lift the IQR outlier filter."""
        if self._analysis_df is None:
            if checked:
                notify("Reload data before applying the IQR filter.", error=True)
                self._btn_iqr.blockSignals(True)
                self._btn_iqr.setChecked(False)
                self._btn_iqr.blockSignals(False)
            return
        self._recompute_filters()

    def _on_iqr_param_changed(self, *_args: Any) -> None:
        """Re-apply the IQR filter when its multiplier or scope changes (no-op when it is off)."""
        if self._btn_iqr.isChecked() and self._analysis_df is not None:
            self._recompute_filters(announce=False)

    def _on_fit(self) -> None:
        """Fit (or reload a cached) MixedLM model from the formula/groups/random-effects fields on the
        current working frame, then display the summary and refresh the plot."""
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
        dropped_note = _dropped_rows_note(meta)
        if dropped_note:
            report = f"{dropped_note}\n\n{report}"
        self._report.setPlainText(report)
        self._refresh_table(model_df)
        self._status.setText(
            f"Fitted n={meta.get('n_rows')}"
            + (f" (dropped {meta.get('n_rows_dropped')} incomplete)" if meta.get("n_rows_dropped") else "")
            + f"  |  groups={groups}  |  re={re_formula!r}  |  dataset={self._repo.root}"
        )
        self._on_plot()
        notify("MixedLM fit complete.")

    def _clear_plot(self) -> None:
        """Remove any existing plot widget from the plot host and release its figure."""
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self._plot_fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._plot_fig)
            except Exception:
                pass
        self._plot_canvas = None
        self._plot_fig = None
        self._plot_axes = None
        self._disable_axis_sliders()

    def _on_plot(self) -> None:
        """Redraw the parameter/EMM plot for the last fitted model using the current x/y/group/mode
        selections, embedding it as a Matplotlib canvas (or an error label on failure)."""
        if self._last_result is None or self._last_df is None:
            return
        self._clear_plot()
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

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

            # Draw on the light default style regardless of what the app (or a previous call to
            # plt.style.use) left in the global rcParams, then force the figure/axes patches white.
            with plt.style.context("default"):
                fig = plot_mixedlm_params(
                    result=self._last_result,
                    df_fit=self._last_df,
                    x=x,
                    y=y,
                    group=group,
                    mode=mode,
                    include_points=self._include_points.isChecked(),
                    title=f"MixedLM: {y} ~ {x} | {group}",
                )
            if fig is None:
                fig = plt.gcf()
            _whiten_figure(fig)

            canvas = FigureCanvasQTAgg(fig)
            self._plot_layout.addWidget(canvas)
            self._plot_canvas = canvas
            self._plot_fig = fig
            self._plot_axes = fig.axes[0] if fig.axes else None
            fig.tight_layout()
            canvas.draw_idle()
            self._sync_axis_sliders()
            self._apply_legend_visibility()
        except Exception as exc:
            err = QLabel(f"Plot unavailable: {exc}")
            err.setWordWrap(True)
            self._plot_layout.addWidget(err)
            self._disable_axis_sliders()

    def _apply_legend_visibility(self) -> None:
        """Show or hide the current axes legend according to the Show-legend toggle."""
        ax = self._plot_axes
        if ax is None or self._plot_canvas is None:
            return
        legend = ax.get_legend()
        if legend is None:
            return
        legend.set_visible(self._show_legend.isChecked())
        self._plot_canvas.draw_idle()

    def _on_legend_toggled(self, *_args: Any) -> None:
        """Toggle legend visibility without recomputing the plot."""
        self._apply_legend_visibility()

    def _disable_axis_sliders(self) -> None:
        """Grey out the axis-limit controls (no figure to rescale)."""
        for key, slider in self._axis_sliders.items():
            slider.blockSignals(True)
            slider.setEnabled(False)
            slider.blockSignals(False)
            self._axis_value_labels[key].setText("—")
        self._btn_reset_axes.setEnabled(False)
        self._axis_base = {}
        self._axis_span = {}

    def _sync_axis_sliders(self) -> None:
        """
        Capture the freshly drawn axes limits as the slider baseline and move the handles to match.

        Each slider spans the plotted range padded by ``_AXIS_SLIDER_MARGIN`` on both sides, so the
        handles start inset and the user can zoom out as well as in.
        """
        ax = self._plot_axes
        if ax is None:
            self._disable_axis_sliders()
            return

        self._axis_base = {"x": tuple(ax.get_xlim()), "y": tuple(ax.get_ylim())}
        self._axis_span = {}
        for axis, (lo, hi) in self._axis_base.items():
            span = float(hi) - float(lo)
            pad = (abs(span) * _AXIS_SLIDER_MARGIN) if span else 1.0
            self._axis_span[axis] = (float(lo) - pad, float(hi) + pad)

        for key, slider in self._axis_sliders.items():
            axis, bound = key[0], key[1:]
            value = self._axis_base[axis][0 if bound == "min" else 1]
            slider.blockSignals(True)
            slider.setEnabled(True)
            slider.setValue(self._axis_value_to_slider(axis, value))
            slider.blockSignals(False)
            self._axis_value_labels[key].setText(f"{value:.4g}")
        self._btn_reset_axes.setEnabled(True)

    def _axis_slider_to_value(self, axis: str, position: int) -> float:
        """Map a slider position to a data coordinate on *axis*."""
        lo, hi = self._axis_span[axis]
        return lo + (hi - lo) * (position / _AXIS_SLIDER_STEPS)

    def _axis_value_to_slider(self, axis: str, value: float) -> int:
        """Map a data coordinate on *axis* back to a slider position."""
        lo, hi = self._axis_span[axis]
        if hi == lo:
            return 0
        frac = (float(value) - lo) / (hi - lo)
        return int(round(min(max(frac, 0.0), 1.0) * _AXIS_SLIDER_STEPS))

    def _on_axis_slider_changed(self, *_args: Any) -> None:
        """Rescale the current figure's axes to the slider positions and redraw the canvas."""
        ax = self._plot_axes
        if ax is None or not self._axis_span or self._plot_canvas is None:
            return
        for axis, setter in (("x", ax.set_xlim), ("y", ax.set_ylim)):
            lo = self._axis_slider_to_value(axis, self._axis_sliders[f"{axis}min"].value())
            hi = self._axis_slider_to_value(axis, self._axis_sliders[f"{axis}max"].value())
            self._axis_value_labels[f"{axis}min"].setText(f"{lo:.4g}")
            self._axis_value_labels[f"{axis}max"].setText(f"{hi:.4g}")
            if hi <= lo:  # crossed handles would raise; keep a hair of range instead
                span = self._axis_span[axis]
                hi = lo + abs(span[1] - span[0]) / _AXIS_SLIDER_STEPS or lo + 1e-9
            setter(lo, hi)
        self._plot_canvas.draw_idle()

    def _on_reset_axes(self) -> None:
        """Restore the autoscaled limits captured when the plot was drawn."""
        ax = self._plot_axes
        if ax is None or not self._axis_base:
            return
        ax.set_xlim(*self._axis_base["x"])
        ax.set_ylim(*self._axis_base["y"])
        for key, slider in self._axis_sliders.items():
            axis, bound = key[0], key[1:]
            value = self._axis_base[axis][0 if bound == "min" else 1]
            slider.blockSignals(True)
            slider.setValue(self._axis_value_to_slider(axis, value))
            slider.blockSignals(False)
            self._axis_value_labels[key].setText(f"{value:.4g}")
        if self._plot_canvas is not None:
            self._plot_canvas.draw_idle()

    def _config_dict(self) -> dict[str, Any]:
        """Serialize the current data-selection, covariate, formula, and plot settings to a plain dict
        (for saving alongside a fitted model)."""
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
            "show_legend": self._show_legend.isChecked(),
            "iqr_enabled": self._btn_iqr.isChecked(),
            "iqr_k": float(self._iqr_k.value()),
            "iqr_scope": str(self._iqr_scope.currentData() or "group"),
        }

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        """Restore the panel's controls (pipeline kind/pipeline/feature/atlas/grouping, covariates,
        formula fields, plot options) from a previously saved config dict."""
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
        if "show_legend" in cfg:
            self._show_legend.setChecked(bool(cfg["show_legend"]))

        if "iqr_k" in cfg:
            self._iqr_k.setValue(float(cfg["iqr_k"]))
        iqr_scope = str(cfg.get("iqr_scope") or "")
        if iqr_scope:
            sidx = self._iqr_scope.findData(iqr_scope)
            if sidx >= 0:
                self._iqr_scope.setCurrentIndex(sidx)
        if "iqr_enabled" in cfg:
            self._btn_iqr.blockSignals(True)
            self._btn_iqr.setChecked(bool(cfg["iqr_enabled"]))
            self._btn_iqr.blockSignals(False)

    def _on_save(self) -> None:
        """Save the last fitted model (pickle), its config, and its summary text under
        ``nvitk-statmodels/<model_name>/``."""
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
        """Prompt for a saved model directory and restore its config/fitted model/summary, reloading
        the analysis frame and refreshing the plot."""
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
                    # Re-apply any IQR / row filter restored from the saved config.
                    self._recompute_filters(announce=False)
                    working = self._working_df()
                    if working is not None:
                        self._last_df = working
                    self._on_plot()
                except Exception:
                    pass
            notify(f"Loaded model from {model_dir}")
        except Exception as exc:
            notify(f"Load failed: {exc}", error=True)


class StatmodelsPanel(QWidget):
    """Right-tab launcher for the floating Statmodels window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the pipeline-kind selector and the button that opens the floating explorer window."""
        super().__init__(parent)
        self._window: StatmodelsWindow | None = None

        self._pipeline_kind = QComboBox()
        for label, key in _PIPELINE_KIND_ITEMS:
            self._pipeline_kind.addItem(label, key)

        self._btn = QPushButton("Open Statmodels window")
        self._btn.clicked.connect(self._open_window)

        hint = QLabel(
            "Explore MixedLM formulas over 4D-flow, ASL, T1, FLAIR WMH, or TOF "
            "morphometrics plus clinical / cognitive covariates from the dataset "
            "catalog. Models are saved under <dataset>/nvitk-statmodels/."
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
        """Open (creating once, then reusing) the floating :class:`StatmodelsWindow`, selecting the
        chosen pipeline kind."""
        kind = str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)
        if self._window is None:
            self._window = StatmodelsWindow(initial_pipeline_kind=kind)
        else:
            idx = self._window._pipeline_kind.findData(kind)
            if idx >= 0:
                self._window._pipeline_kind.setCurrentIndex(idx)
        self._window.show_maximized_floating()
