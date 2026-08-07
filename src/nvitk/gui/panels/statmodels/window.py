"""
The Statmodels explorer window.

Description
-----------
A floating, maximizable window laid out in three draggable rows:

* **top** — what data to load (measurements) and what to do with it (MixedLM formula, or mediation)
* **middle** — the plot and the model report, which get most of the height
* **bottom** — clinical / cognitive covariate pickers and the analysis dataframe

The frame flows one way: measurements → ``_analysis_df`` (raw, never mutated) → derived columns →
filter rules → ``_working_df`` (what gets fitted). Both derivation and filtering are recomputed from
the raw frame every time, so toggling one never compounds on the other's output, and a reload keeps
the whole set applied.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.tools.runner import notify
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.stats.mixedlm import formula_columns as _formula_columns
from nvitk.stats.interactive import forest_plot, matrix_plot, network_plot
from nvitk.stats.interactive_adapters import (
    mmrm_geometry,
    r_model_geometry,
    render,
    statsmodels_geometry,
)
from nvitk.stats.r_gam import (
    GAM_FAMILIES,
    GAM_METHODS,
    fit_mrf,
    gam_backend_status,
    mrf_field_frame,
    plot_mrf_field,
    plot_mrf_graph,
)
from nvitk.stats.qc_filters import (
    KEEP_COLUMN,
    QC_FILTER_BY_KEY,
    SUBJECT_AWARE_KEYS,
    flow_qc_keep,
    preset_rules,
)
from nvitk.stats.summaries import OVERALL_LABEL, summarize_by_group, summary_provenance
from nvitk.stats.region_algebra import RegionCombination, apply_region_combinations
from nvitk.stats.sem import (
    SemSpec,
    fit_sem,
    path_effects,
    plot_sem_network,
    plot_sem_paths,
    sem_backend_status,
    sem_paths_frame,
)
from nvitk.stats.vessel_network import VESSEL_NODES, canonical_node, sem_model_syntax
from nvitk.stats import (
    GLM_FAMILIES,
    NONLINEAR_MODELS,
    MeasurementSpec,
    COVARIANCE_STRUCTURES,
    DF_METHODS,
    fit_glm,
    fit_lme4,
    fit_mmrm,
    fit_nonlinear,
    LMROB_ESTIMATORS,
    LMROB_PSI,
    LMROB_SETTINGS,
    fit_lmrob,
    fit_ols,
    fit_or_load_mixedlm,
    mixedlm_info_dict,
    model_info_dict,
    mixedlm_to_lme4_formula,
    mmrm_backend_status,
    robust_backend_status,
    parse_mmrm_covariance,
    mmrm_emmeans,
    lmrob_weights_frame,
    plot_lme4_params,
    plot_lmrob_params,
    plot_lmrob_weights,
    plot_mmrm_correlation,
    mmrm_correlation_matrix,
    plot_mixedlm_params,
    plot_nonlinear_fit,
    r_backend_status,
    validate_mmrm_data,
    render_mixedlm_info,
    subject_attribute_entries,
)
from nvitk.stats.frame_ops import (
    DerivedColumn,
    FilterRule,
    apply_derived_columns,
    apply_filter_rules,
    default_derived_name,
    filtered_columns,
)
from nvitk.stats.mediation import (
    MediationSpec,
    plot_indirect_bootstrap,
    plot_indirect_by_level,
    plot_mediation_forest,
    plot_partial_paths_mediation,
    render_mediation_info,
)

from .constants import (
    ANALYSIS_FORMULA_KINDS,
    ANALYSIS_PANEL_KINDS,
    ANALYSIS_GLM,
    ANALYSIS_HINTS,
    ANALYSIS_ITEMS,
    ANALYSIS_LME4,
    ANALYSIS_LMROB,
    ANALYSIS_MRF,
    ANALYSIS_SEM,
    ANALYSIS_MEDIATION,
    ANALYSIS_MMRM,
    ANALYSIS_MIXEDLM,
    ANALYSIS_NONLINEAR,
    ANALYSIS_OLS,
    ANALYSIS_R_KINDS,
    CONFIG_VERSION,
    LME4_FAMILY_ITEMS,
    ROBUST_COV_ITEMS,
    DEFAULT_FORMULA,
    DEFAULT_GROUPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RE,
    DEFAULT_VC,
    PIPELINE_KIND_QVTPY,
)
from .column_plot_dialog import ColumnPlotDialog
from .combinations_dialog import RegionCombinationsDialog
from .db_publish import resolve_publish_target
from .derived import CovarianceTermDialog, DerivedColumnsDialog, SplineTermDialog
from .publish_dialog import PublishDerivedDialog
from .export import build_provenance_frame, export_analysis_frame, export_group_summary
from .frame_table import AnalysisFrameView, ColumnFilterDialog, FilterChipBar
from .helpers import (
    checked_variable_ids,
    dropped_rows_note,
    filter_list_widget,
    open_repo,
    parse_vc_formula,
    populate_checklist,
    resolve_outcome_column,
    set_checked_variable_ids,
    statmodels_root,
)
from .measurements import FrameLoadWorker, MeasurementForm, MeasurementsWidget
from .mediation_panel import MediationFormPanel, MediationWorker
from .plot_view import PlotPanel
from .report import ModelReportPanel
from .theme import apply_dark_theme, muted_label_style

log = Logger()

# Mediation figures the plot pane can draw, keyed by the engines that can produce them.
def _scrollable(widget: QWidget) -> QScrollArea:
    """
    Wrap *widget* in a vertical scroll area so it stops dictating the window's proportions.

    A control column's ``minimumSizeHint`` is the sum of everything in it — and a
    :class:`QStackedWidget` reports the *largest* of its pages, so the tall mediation form was
    forcing ~485 px of minimum height on the controls row even while the shorter MixedLM form was
    showing. A splitter cannot shrink a child below its minimum, so the figure could never get the
    space it was allotted. Scrolling makes the minimum small and the requested sizes achievable.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    # Enough to show a couple of rows; the splitter is free to give it more.
    area.setMinimumHeight(90)
    return area


_MMRM_PLOTS: tuple[tuple[str, str], ...] = (
    ("Least-squares means", "emmeans"),
    ("Correlation between levels", "correlation"),
)

_SEM_PLOTS: tuple[tuple[str, str], ...] = (
    ("Path coefficients", "paths"),
    ("Network diagram", "network"),
)

_MRF_PLOTS: tuple[tuple[str, str], ...] = (
    ("Smoothed field", "field"),
    ("Adjacency graph", "graph"),
)

_LMROB_PLOTS: tuple[tuple[str, str], ...] = (
    ("Robust fit", "fit"),
    ("Robustness weights", "weights"),
)

_MEDIATION_PLOTS: tuple[tuple[str, str], ...] = (
    ("Path forest", "forest"),
    ("Indirect bootstrap", "bootstrap"),
    ("Indirect by level", "by_level"),
    ("Partial paths", "partial"),
)


class StatmodelsWindow(QMainWindow):
    """Floating / maximizable MixedLM and mediation explorer."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_pipeline_kind: str = PIPELINE_KIND_QVTPY,
    ) -> None:
        """Build the three-row layout and wire every signal handler."""
        super().__init__(parent)
        self.setWindowTitle("nvitk Statmodels")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1700, 1000)
        apply_dark_theme(self)

        self._repo = open_repo()

        # ---- analysis state ---------------------------------------------------
        self._analysis_df: pd.DataFrame | None = None   # as loaded, never mutated
        self._working_df: pd.DataFrame | None = None    # + derived columns, − filtered rows
        self._load_meta: dict[str, Any] | None = None
        self._derived: list[DerivedColumn] = []
        self._combinations: list[RegionCombination] = []
        self._filter_report: list[dict[str, Any]] = []

        # ---- result state -----------------------------------------------------
        self._last_result = None
        self._last_model_df: pd.DataFrame | None = None
        self._last_fit_meta: dict[str, Any] | None = None
        self._last_outcome: str | None = None
        self._mediation_bundle: dict[str, Any] | None = None
        # Group selection restored from a config, applied once the levels are known after a fit.
        self._pending_plot_groups: list[str] | None = None
        # Probing R shells out to Rscript; do it once, lazily.
        self._r_status = None
        self._mmrm_status = None
        self._robust_status = None

        # ---- workers ----------------------------------------------------------
        self._load_worker: FrameLoadWorker | None = None
        self._mediation_worker: MediationWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self._top_split = QSplitter(Qt.Horizontal)
        self._top_split.addWidget(self._build_data_panel(initial_pipeline_kind))
        self._top_split.addWidget(_scrollable(self._build_model_panel()))
        self._top_split.setSizes([760, 840])

        self._output_split = QSplitter(Qt.Horizontal)
        self._plot = PlotPanel()
        self._plot.set_options_widget(self._build_plot_options())
        self._mediation_plot = self._plot.kind_combo()
        self._report = ModelReportPanel()
        self._output_split.addWidget(self._plot)
        self._output_split.addWidget(self._report)
        self._output_split.setStretchFactor(0, 3)
        self._output_split.setStretchFactor(1, 1)
        self._output_split.setSizes([1150, 450])

        self._bottom_split = QSplitter(Qt.Horizontal)
        self._bottom_split.addWidget(self._build_covariate_box("Clinical covariates", "clinical"))
        self._bottom_split.addWidget(self._build_covariate_box("Cognitive covariates", "cognitive"))
        self._bottom_split.addWidget(self._build_frame_box())
        self._bottom_split.setStretchFactor(0, 1)
        self._bottom_split.setStretchFactor(1, 1)
        self._bottom_split.setStretchFactor(2, 3)
        self._bottom_split.setSizes([320, 320, 1000])

        self._main_split = QSplitter(Qt.Vertical)
        for pane, stretch in (
            (self._top_split, 0),
            (self._output_split, 1),
            (self._bottom_split, 0),
        ):
            self._main_split.addWidget(pane)
            self._main_split.setStretchFactor(self._main_split.count() - 1, stretch)
        # The figure is what the user reads; the controls rows get only what they need.
        self._main_split.setSizes([230, 820, 210])
        root.addWidget(self._main_split, stretch=1)

        status_row = QHBoxLayout()
        self._status = QLabel(f"Dataset: {self._repo.root}  |  qvtpy pipeline: {QVTPY_PIPELINE_ID}")
        self._status.setWordWrap(True)
        status_row.addWidget(self._status, stretch=1)

        # Collapse the controls and the dataframe when iterating on a plot, so the figure gets the
        # whole window without having to drag three splitters back and forth.
        self._btn_focus = QPushButton("Focus plot")
        self._btn_focus.setCheckable(True)
        self._btn_focus.setToolTip(
            "Collapse the controls and dataframe rows to give the plot the full window."
        )
        self._btn_focus.toggled.connect(self._on_focus_plot)
        status_row.addWidget(self._btn_focus)
        root.addLayout(status_row)

        self._connect_signals()
        self._refresh_covariate_lists()
        self._measurements.set_specs([self._data_form.spec()])
        self._sync_analysis_type()

    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────
    def _build_data_panel(self, initial_pipeline_kind: str) -> QWidget:
        """
        Top-left: the primary measurement selector beside the measurement list.

        Side by side rather than stacked — stacking them made the controls row about twice as tall
        as it needed to be, and every pixel there comes straight out of the figure.
        """
        split = QSplitter(Qt.Horizontal)

        box = QGroupBox("Data selection")
        box_lay = QVBoxLayout(box)
        box_lay.setContentsMargins(6, 4, 6, 4)
        self._data_form = MeasurementForm(self._repo)
        self._data_form.apply_spec(MeasurementSpec(pipeline_kind=initial_pipeline_kind))
        box_lay.addWidget(self._data_form)
        self._btn_reload = QPushButton("Reload data")
        box_lay.addWidget(self._btn_reload)
        box_lay.addStretch(1)
        split.addWidget(_scrollable(box))

        self._measurements = MeasurementsWidget(self._repo)
        split.addWidget(self._measurements)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 380])
        return split

    def _build_model_panel(self) -> QWidget:
        """Top-right: the analysis-type switch over the MixedLM / mediation forms."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Analysis type"))
        self._analysis_type = QComboBox()
        for label, key in ANALYSIS_ITEMS:
            self._analysis_type.addItem(label, key)
        type_row.addWidget(self._analysis_type, stretch=1)
        lay.addLayout(type_row)

        self._analysis_hint = QLabel("")
        self._analysis_hint.setWordWrap(True)
        self._analysis_hint.setStyleSheet(muted_label_style())
        lay.addWidget(self._analysis_hint)

        self._model_stack = QStackedWidget()
        self._model_stack.addWidget(self._build_formula_box())      # MixedLM / OLS / GLM
        self._model_stack.addWidget(self._build_nonlinear_box())
        self._mediation_form = MediationFormPanel()
        self._model_stack.addWidget(self._mediation_form)
        lay.addWidget(self._model_stack, stretch=1)
        return panel

    def _build_nonlinear_box(self) -> QWidget:
        """Non-linear curve fit: one predictor, one response, an explicit parametric shape."""
        box = QGroupBox("Non-linear curve fit")
        box_lay = QVBoxLayout(box)

        form = QFormLayout()
        self._nl_model = QComboBox()
        for key, spec in NONLINEAR_MODELS.items():
            self._nl_model.addItem(f"{spec.label} — {spec.expression}", key)
        self._nl_model.currentIndexChanged.connect(self._on_nonlinear_model_changed)
        self._nl_x = QComboBox()
        self._nl_y = QComboBox()
        self._nl_p0 = QLineEdit()
        self._nl_p0.setPlaceholderText("(optional) starting values, comma separated")
        form.addRow("Curve", self._nl_model)
        form.addRow("x (predictor)", self._nl_x)
        form.addRow("y (response)", self._nl_y)
        form.addRow("Start values", self._nl_p0)
        box_lay.addLayout(form)

        self._nl_hint = QLabel("")
        self._nl_hint.setWordWrap(True)
        self._nl_hint.setStyleSheet(muted_label_style())
        box_lay.addWidget(self._nl_hint)

        row = QHBoxLayout()
        self._btn_nl_fit = QPushButton("Fit curve")
        self._btn_nl_fit.clicked.connect(self._on_fit)
        row.addWidget(self._btn_nl_fit)
        row.addStretch(1)
        box_lay.addLayout(row)
        box_lay.addStretch(1)
        self._on_nonlinear_model_changed()
        return box

    def _on_nonlinear_model_changed(self) -> None:
        """Describe the selected curve and the parameters it will estimate."""
        spec = NONLINEAR_MODELS.get(str(self._nl_model.currentData() or ""))
        if spec is None:
            self._nl_hint.setText("")
            return
        self._nl_hint.setText(
            f"{spec.description}  Parameters: {', '.join(spec.params)}. "
            "Leave the start values blank to let them be estimated from the data."
        )
        self._nl_p0.setPlaceholderText(
            f"(optional) {len(spec.params)} values: {', '.join(spec.params)}"
        )

    def _build_formula_box(self) -> QWidget:
        """The MixedLM formulation fields, action buttons and plot options."""
        box = QGroupBox("Model formulation")
        box_lay = QVBoxLayout(box)

        form = QFormLayout()
        self._formula = QPlainTextEdit(DEFAULT_FORMULA)
        self._formula.setFixedHeight(80)
        self._formula.setToolTip(
            "Patsy formula. The left-hand side should be a column of the analysis frame — add a "
            "derived column (e.g. log_pi) rather than writing log(pi) here, so the transformed "
            "outcome can also be plotted and filtered."
        )
        self._groups = QLineEdit(DEFAULT_GROUPS)
        self._re_formula = QLineEdit(DEFAULT_RE)
        self._vc_formula = QLineEdit(DEFAULT_VC)
        self._model_name = QLineEdit(DEFAULT_MODEL_NAME)

        # GLM-only: the link is where a GLM's non-linearity lives.
        self._glm_family = QComboBox()
        for key, spec in GLM_FAMILIES.items():
            self._glm_family.addItem(spec.label, key)
        self._glm_family.currentIndexChanged.connect(self._on_glm_family_changed)
        self._glm_link = QComboBox()
        self._glm_family_label = QLabel("Family")
        self._glm_link_label = QLabel("Link")

        # OLS-only.
        self._robust = QComboBox()
        for label, key in ROBUST_COV_ITEMS:
            self._robust.addItem(label, key)
        self._robust_label = QLabel("Std. errors")

        self._groups_label = QLabel("groups")
        self._re_label = QLabel("re_formula")
        self._vc_label = QLabel("vc_formula")

        # R / lme4 only.
        self._lme4_family = QComboBox()
        for label, key in LME4_FAMILY_ITEMS:
            self._lme4_family.addItem(label, key)
        self._lme4_family_label = QLabel("Family")
        self._lme4_reml = QCheckBox("REML")
        self._lme4_reml.setChecked(True)
        self._lme4_reml.setToolTip(
            "Restricted maximum likelihood — leave on for variance estimates. Turn it off to "
            "compare models differing in their fixed effects by likelihood, where REML fits are "
            "not comparable.\n\n"
            "Note: pymer4 0.9 does not expose this — lmerTest's default (REML on) is always used. "
            "Use the statsmodels MixedLM engine if you need ML estimation."
        )
        self._lme4_reml_label = QLabel("Estimation")

        # R / mmrm only. The covariance structure is written into the formula the way it is in R —
        # ``+ us(territory | subject_uid)`` — so there are no separate controls for it. Only the
        # arguments that are genuinely not part of the formula get their own widgets.
        self._mmrm_method = QComboBox()
        for key, spec in DF_METHODS.items():
            self._mmrm_method.addItem(spec.label, key)
        self._mmrm_method.setToolTip(
            "\n".join(f"{s.label}: {s.description}" for s in DF_METHODS.values())
        )
        self._mmrm_method_label = QLabel("Denominator df")

        # R / robustbase only. lmrob takes a plain formula — the estimator and the loss function are
        # what shape the fit, so they are the only extra controls.
        self._lmrob_method = QComboBox()
        for key, spec in LMROB_ESTIMATORS.items():
            self._lmrob_method.addItem(spec.label, key)
        self._lmrob_method.setToolTip(
            "\n\n".join(f"{s.label}: {s.description}" for s in LMROB_ESTIMATORS.values())
        )
        self._lmrob_method_label = QLabel("Estimator")
        self._lmrob_psi = QComboBox()
        for key, description in LMROB_PSI.items():
            self._lmrob_psi.addItem(description, key)
        self._lmrob_psi.setToolTip(
            "How fast a residual's influence is cut off as it grows. Redescending functions give "
            "extreme observations zero weight; Huber only bounds their influence."
        )
        self._lmrob_psi_label = QLabel("Loss (psi)")
        self._lmrob_setting = QComboBox()
        for key, description in LMROB_SETTINGS.items():
            self._lmrob_setting.addItem(description, key)
        self._lmrob_setting.setToolTip(
            "A published control preset. Choosing one overrides the estimator and loss above."
        )
        self._lmrob_setting_label = QLabel("Preset")

        form.addRow("Formula", self._formula)
        form.addRow(self._groups_label, self._groups)
        form.addRow(self._re_label, self._re_formula)
        form.addRow(self._vc_label, self._vc_formula)
        form.addRow(self._glm_family_label, self._glm_family)
        form.addRow(self._glm_link_label, self._glm_link)
        form.addRow(self._lme4_family_label, self._lme4_family)
        form.addRow(self._lme4_reml_label, self._lme4_reml)
        form.addRow(self._mmrm_method_label, self._mmrm_method)
        # SEM and MRF only.
        self._sem_backend = QComboBox()
        self._sem_backend.addItem("auto — semopy if present, else lavaan", "")
        self._sem_backend.addItem("semopy (Python)", "semopy")
        self._sem_backend.addItem("lavaan (R)", "lavaan")
        self._sem_backend_label = QLabel("SEM backend")
        self._sem_standardize = QCheckBox("Standardize variables")
        self._sem_standardize.setChecked(True)
        self._sem_standardize.setToolTip(
            "Flows in mL/min and ages in years differ by two orders of magnitude, which makes the "
            "raw path coefficients incomparable and the optimiser badly conditioned. Standardizing "
            "puts every path on the same SD-per-SD scale."
        )
        self._sem_standardize_label = QLabel("Scaling")
        self._mrf_family = QComboBox()
        for key, description in GAM_FAMILIES.items():
            self._mrf_family.addItem(description, key)
        self._mrf_family_label = QLabel("Family")
        self._mrf_method = QComboBox()
        for key, description in GAM_METHODS.items():
            self._mrf_method.addItem(description, key)
        self._mrf_method_label = QLabel("Smoothing")

        form.addRow(self._sem_backend_label, self._sem_backend)
        form.addRow(self._sem_standardize_label, self._sem_standardize)
        form.addRow(self._mrf_family_label, self._mrf_family)
        form.addRow(self._mrf_method_label, self._mrf_method)
        form.addRow(self._lmrob_method_label, self._lmrob_method)
        form.addRow(self._lmrob_psi_label, self._lmrob_psi)
        form.addRow(self._lmrob_setting_label, self._lmrob_setting)
        form.addRow(self._robust_label, self._robust)
        form.addRow("Model name (save)", self._model_name)
        box_lay.addLayout(form)

        term_row = QHBoxLayout()
        self._btn_network_syntax = QPushButton("Insert network syntax")
        self._btn_network_syntax.setToolTip(
            "Write the vascular path model into the formula box — one equation per junction, built "
            "from the vessels actually present, with the current covariates added to each."
        )
        self._btn_network_syntax.clicked.connect(self._on_insert_network_syntax)
        term_row.addWidget(self._btn_network_syntax)
        self._btn_lme4_convert = QPushButton("Convert MixedLM → lme4")
        self._btn_lme4_convert.setToolTip(
            "Rewrite the current groups / re_formula / vc_formula as one lme4 formula, so an "
            "existing MixedLM specification can be moved across without retyping it."
        )
        self._btn_lme4_convert.clicked.connect(self._on_convert_to_lme4)
        term_row.addWidget(self._btn_lme4_convert)
        self._btn_insert_covariance = QPushButton("Insert covariance term…")
        self._btn_insert_covariance.setToolTip(
            "Build an mmrm covariance term — us(territory | subject_uid) and the other eight "
            "structures — and insert it into the formula at the cursor."
        )
        self._btn_insert_covariance.clicked.connect(self._on_insert_covariance_term)
        term_row.addWidget(self._btn_insert_covariance)
        self._btn_insert_term = QPushButton("Insert curved term…")
        self._btn_insert_term.setToolTip(
            "Add a spline or polynomial term to the formula, for a predictor whose effect is not a "
            "straight line. The model stays linear in its parameters, so the coefficient table and "
            "the marginal-means plot work exactly as before."
        )
        self._btn_insert_term.clicked.connect(self._on_insert_term)
        term_row.addWidget(self._btn_insert_term)
        term_row.addStretch(1)
        box_lay.addLayout(term_row)

        btn_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit model")
        self._btn_save = QPushButton("Save model")
        self._btn_load = QPushButton("Load model…")
        self._btn_plot = QPushButton("Refresh plot")
        for btn in (self._btn_fit, self._btn_save, self._btn_load, self._btn_plot):
            btn_row.addWidget(btn)
        box_lay.addLayout(btn_row)

        box_lay.addStretch(1)
        return box

    def _build_plot_options(self) -> QWidget:
        """Plot display / QC gate / mode / x / points, laid out for the plot pane's own top row."""
        widget = QWidget()
        lay = QHBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Show the rows the analysis dataframe's filters removed, in grey, so a filter's effect is
        # visible on the plot. A plain toggle rather than a metric picker: the filters already live
        # on the frame, and asking the plot to re-derive a second, different exclusion was confusing
        # and let the two disagree.
        self._show_filtered = QCheckBox("Show filtered")
        self._show_filtered.setToolTip(
            "Draw the observations removed by the active dataframe filters in grey instead of "
            "hiding them, so you can see whether a filter took out a coherent cluster or scattered "
            "noise.\nDisabled when no filter is active — there is nothing to show."
        )
        self._show_filtered.stateChanged.connect(lambda *_: self._on_plot())
        lay.addWidget(self._show_filtered)

        # Matplotlib by default; the interactive backend is opt-in per plot.
        self._interactive_plot = QCheckBox("Interactive")
        self._interactive_plot.setToolTip(
            "Render with Plotly instead of Matplotlib: hover a point for its subject and values, "
            "drag to zoom, click a legend entry to hide a series.\n"
            "Matplotlib is the default — it is what the axis-limit sliders act on and what exports "
            "as a publication figure."
        )
        self._interactive_plot.stateChanged.connect(lambda *_: self._on_plot())
        lay.addWidget(self._interactive_plot)

        self._plot_display = QComboBox()
        self._plot_display.addItem("Overview", "overview")
        self._plot_display.addItem("Grouped", "grouped")
        self._display_tooltip = (
            "Overview draws every group on one pair of axes.\n"
            "Grouped splits them into a grid of anatomical panels — carotids / anterior / "
            "posterior / venous for vessels, lobes for cortical parcels — each autoscaled to its "
            "own range, so a venous measurement no longer flattens an arterial one.\n"
            "This is a view of the same fit: the population estimate is identical in every panel."
        )
        self._plot_display.setToolTip(self._display_tooltip)

        self._plot_mode = QComboBox()
        self._plot_mode.addItem("auto", "auto")
        self._plot_mode.addItem("continuous (scatter + regression)", "continuous")
        self._plot_mode.addItem("categorical (marginal means)", "categorical")
        self._plot_x = QComboBox()
        self._plot_x.setMinimumWidth(140)
        self._include_points = QCheckBox("Points")
        self._include_points.setChecked(True)
        self._include_points.setToolTip(
            "Overlay the observed data, in a lighter tone of each group's colour.\n"
            "Continuous plots show individual observations; categorical plots show the mean of "
            "each cell — unadjusted, so the gap to the model curve is the covariate adjustment."
        )
        self._show_ci = QCheckBox("95% CI")
        self._show_ci.setToolTip(
            "Show the confidence interval of the model predictions: whiskers on each marginal "
            "mean, or a band around the fixed-effect line.\n"
            "Per-group lines in continuous mode get no band — their random-effect deviation has no "
            "standard error in the fit, so any interval drawn there would be too narrow."
        )

        self._plot_display_label = QLabel("Display")
        lay.addWidget(self._plot_display_label)
        lay.addWidget(self._plot_display)
        lay.addWidget(QLabel("Mode"))
        lay.addWidget(self._plot_mode)
        lay.addWidget(QLabel("x"))
        lay.addWidget(self._plot_x)
        lay.addWidget(self._include_points)
        lay.addWidget(self._show_ci)
        lay.addStretch(1)
        return widget

    def _build_covariate_box(self, title: str, domain: str) -> QWidget:
        """Bottom row: a searchable, checkable covariate list."""
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        search = QLineEdit()
        search.setPlaceholderText("Search…")
        lay.addWidget(search)
        widget = QListWidget()
        widget.setMinimumHeight(60)
        lay.addWidget(widget, stretch=1)
        search.textChanged.connect(lambda text, w=widget: filter_list_widget(w, text))
        if domain == "clinical":
            self._clinical_list = widget
        else:
            self._cognitive_list = widget
        return box

    def _build_frame_box(self) -> QWidget:
        """Bottom row: filter chips over the analysis dataframe, plus the derived-columns entry."""
        box = QGroupBox("Analysis dataframe")
        lay = QVBoxLayout(box)

        self._chips = FilterChipBar()
        lay.addWidget(self._chips)

        self._frame_view = AnalysisFrameView()
        lay.addWidget(self._frame_view, stretch=1)

        row = QHBoxLayout()
        self._btn_derived = QPushButton("Derived columns…")
        self._btn_derived.setToolTip(
            "Add transformed measurements (log, z-score, ratios, grouped bins) as real columns, "
            "usable as the model outcome, as predictors, as plot axes and as filter targets."
        )
        row.addWidget(self._btn_derived)
        self._btn_combinations = QPushButton("Region combinations…")
        self._btn_combinations.setToolTip(
            "Combine a measurement across regions — TCBF = RICA + LICA + BASI, or a mass-balance "
            "residual. Derived columns work within a row; these work across a subject's rows."
        )
        self._btn_combinations.clicked.connect(lambda: self._on_edit_combinations())
        row.addWidget(self._btn_combinations)
        self._btn_export = QPushButton("Export table…")
        self._btn_export.setToolTip(
            "Save exactly what this table shows — measurements joined, derived columns computed, "
            "filters applied. An .xlsx export adds a provenance sheet recording how the frame was "
            "built, so the numbers stay traceable."
        )
        row.addWidget(self._btn_export)
        self._derived_label = QLabel("")
        self._derived_label.setStyleSheet(muted_label_style())
        row.addWidget(self._derived_label, stretch=1)
        lay.addLayout(row)
        return box

    def _connect_signals(self) -> None:
        """Wire every widget signal to its handler."""
        self._data_form.changed.connect(self._on_primary_measurement_changed)
        self._measurements.changed.connect(self._on_measurements_changed)
        self._btn_reload.clicked.connect(self._on_reload)

        self._analysis_type.currentIndexChanged.connect(self._sync_analysis_type)
        self._formula.textChanged.connect(self._on_formula_changed)
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_plot.clicked.connect(self._on_plot)
        self._plot.exportRequested.connect(self._on_export_plot)
        self._plot_display.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_mode.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_x.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._include_points.stateChanged.connect(lambda *_: self._on_plot())
        self._show_ci.stateChanged.connect(lambda *_: self._on_plot())
        self._plot.optionsChanged.connect(self._on_plot)
        self._mediation_plot.currentIndexChanged.connect(lambda *_: self._sync_display_enabled())
        self._mediation_plot.currentIndexChanged.connect(lambda *_: self._on_plot())

        self._chips.rulesChanged.connect(lambda: self._recompute_frame())
        self._frame_view.filtersRequested.connect(self._on_edit_column_filter)
        self._frame_view.clearFiltersRequested.connect(self._on_clear_column_filter)
        self._frame_view.transformRequested.connect(self._on_quick_transform)
        self._frame_view.binsRequested.connect(self._on_bin_column)
        self._frame_view.publishRequested.connect(self._on_publish_column)
        self._frame_view.columnPlotRequested.connect(self._on_column_plot)
        self._frame_view.qcFilterRequested.connect(self._on_qc_filter)
        self._frame_view.summaryRequested.connect(self._on_export_summary)
        self._frame_view.plotXRequested.connect(self._on_set_plot_x)
        self._btn_derived.clicked.connect(lambda *_: self._on_edit_derived())
        self._btn_export.clicked.connect(self._on_export_frame)

        self._mediation_form.runRequested.connect(self._on_run_mediation)
        self._mediation_form.cancelRequested.connect(self._on_cancel_mediation)

    def show_maximized_floating(self) -> None:
        """Show, maximize, raise, and focus this window."""
        self.show()
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def _on_focus_plot(self, focused: bool) -> None:
        """Collapse (or restore) the controls and dataframe rows around the plot."""
        if focused:
            self._restore_sizes = self._main_split.sizes()
            total = sum(self._restore_sizes) or self.height()
            self._main_split.setSizes([0, total, 0])
            self._btn_focus.setText("Show controls")
        else:
            sizes = getattr(self, "_restore_sizes", None)
            self._main_split.setSizes(sizes or [230, 820, 210])
            self._btn_focus.setText("Focus plot")

    def closeEvent(self, event: Any) -> None:
        """Stop and join any running worker before the window (and its signal targets) go away."""
        if self._mediation_worker is not None and self._mediation_worker.isRunning():
            self._mediation_worker.cancel()
            self._mediation_worker.wait(5000)
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.wait(5000)
        super().closeEvent(event)

    def set_pipeline_kind(self, kind: str) -> None:
        """Point the primary measurement at *kind*, keeping the rest of its settings."""
        spec = self._data_form.spec()
        if spec.pipeline_kind != kind:
            self._data_form.apply_spec(MeasurementSpec(pipeline_kind=kind))

    # ──────────────────────────────────────────────────────────────────────────
    # Covariates and measurements
    # ──────────────────────────────────────────────────────────────────────────
    def _refresh_covariate_lists(self) -> None:
        """Re-read the catalog and rebuild the clinical/cognitive checklists, preserving checks.

        Call this on Reload so newly imported variables (e.g. ``sex`` from ``import_sex.py``) appear
        and so a stale in-memory catalog cannot keep offering the sparse ``subjects.sex`` column.
        """
        try:
            self._repo.catalog.refresh()
        except Exception as exc:
            log.debug("Catalog refresh failed: %s", exc)
        clinical_checked = checked_variable_ids(self._clinical_list)
        cognitive_checked = checked_variable_ids(self._cognitive_list)
        populate_checklist(
            self._clinical_list,
            [
                *subject_attribute_entries(self._repo),
                *self._repo.catalog.variable_entries(domain="clinical"),
            ],
        )
        populate_checklist(
            self._cognitive_list, self._repo.catalog.variable_entries(domain="cognitive")
        )
        set_checked_variable_ids(self._clinical_list, clinical_checked)
        set_checked_variable_ids(self._cognitive_list, cognitive_checked)

    def _clinical_vars(self) -> list[str]:
        """Checked clinical covariate variable ids."""
        return checked_variable_ids(self._clinical_list)

    def _cognitive_vars(self) -> list[str]:
        """Checked cognitive covariate variable ids."""
        return checked_variable_ids(self._cognitive_list)

    def _on_primary_measurement_changed(self) -> None:
        """Mirror the inline Data-selection form into measurement 0."""
        self._measurements.set_primary(self._data_form.spec())

    def _primary_column(self) -> str:
        """Output column of the primary measurement."""
        specs = self._measurements.specs()
        return specs[0].column() if specs else "flow_mean"

    def _on_measurements_changed(self) -> None:
        """Flag that the loaded frame no longer matches the selection."""
        if self._analysis_df is not None:
            self._btn_reload.setText("Reload data  •")
            self._btn_reload.setToolTip(
                "The measurement selection changed — reload to rebuild the analysis frame."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Loading, deriving, filtering
    # ──────────────────────────────────────────────────────────────────────────
    def _on_reload(self) -> None:
        """Rebuild the analysis frame from the current measurements, off the UI thread."""
        if self._load_worker is not None and self._load_worker.isRunning():
            notify("A reload is already running.", error=True)
            return
        self._refresh_covariate_lists()
        specs = self._measurements.specs()
        self._btn_reload.setEnabled(False)
        self._status.setText(f"Loading {len(specs)} measurement(s)…")

        worker = FrameLoadWorker(
            self._repo,
            measurements=specs,
            clinical_vars=self._clinical_vars(),
            cognitive_vars=self._cognitive_vars(),
            join=self._measurements.join(),
        )
        worker.finished_ok.connect(self._on_frame_loaded)
        worker.failed.connect(self._on_frame_load_failed)
        worker.progress.connect(self._status.setText)
        worker.finished.connect(lambda: self._btn_reload.setEnabled(True))
        self._load_worker = worker
        worker.start()

    def _on_frame_loaded(self, payload: tuple[pd.DataFrame, dict[str, Any]]) -> None:
        """Adopt a freshly loaded frame and re-apply derived columns and filters to it."""
        frame, meta = payload
        self._analysis_df = frame
        self._load_meta = meta
        self._btn_reload.setText("Reload data")
        self._btn_reload.setToolTip("")
        self._measurements.set_diagnostics(meta)
        self._recompute_frame(announce=False)

        working = self._working_df
        n_working = 0 if working is None else len(working)
        self._status.setText(
            f"Loaded n={len(frame)} rows"
            + (f" (filtered to {n_working})" if n_working != len(frame) else "")
            + f"  |  measurements={', '.join(self._measurements.columns())}"
            + f"  |  dataset={self._repo.root}"
        )
        if frame.empty:
            QMessageBox.warning(
                self,
                "Empty analysis frame",
                "\n\n".join(meta.get("warnings") or ["The selected measurements produced no rows."]),
            )
        notify(f"Reloaded analysis frame ({len(frame)} rows).")

    def _on_frame_load_failed(self, message: str) -> None:
        """Report a failed reload."""
        QMessageBox.critical(self, "Reload failed", message)
        self._status.setText(f"Reload failed: {message}")
        notify(f"Statmodels reload failed: {message}", error=True)

    def _recompute_frame(self, *, announce: bool = True) -> None:
        """
        Rebuild the working frame: derived columns first, then the filter rules.

        Both stages always start from the untouched ``_analysis_df``, so removing a filter or editing
        a derived column can never compound on a previously filtered frame. Derived columns come
        first because filters must be able to target them (``log_pi > 0``).
        """
        base = self._analysis_df
        if base is None:
            self._frame_view.set_frame(None)
            self._chips.set_rules(self._chips.rules(), [])
            return

        # Region algebra first: a derived column may well be built from a combination
        # (``log(TCBF)``), and the reverse never happens — a combination reads one measurement
        # across rows, which a within-row transform cannot produce.
        combined, combo_errors, _reports = apply_region_combinations(base, self._combinations)
        derived_frame, derived_errors = apply_derived_columns(combined, self._derived)
        derived_errors = [*combo_errors, *derived_errors]
        if derived_errors:
            self._derived_label.setText("⚠ " + "; ".join(derived_errors))
        else:
            self._derived_label.setText(
                f"{len(self._derived)} derived column(s)" if self._derived else ""
            )

        rules = self._chips.rules()
        working, report = apply_filter_rules(derived_frame, rules)
        self._working_df = working
        self._filter_report = report

        self._chips.set_rules(rules, report)
        self._chips.set_counts(len(working), len(derived_frame))
        self._sync_filter_toggle()
        self._frame_view.set_frame(
            working,
            filtered_columns=filtered_columns(rules),
            derived_columns={d.name for d in self._derived},
        )
        self._sync_column_combos(working)
        self._sync_nonlinear_columns(working)
        self._mediation_form.set_columns(working)

        if announce:
            notify(f"Filters applied: {len(working)} of {len(derived_frame)} rows.")

    def _sync_column_combos(self, df: pd.DataFrame | None) -> None:
        """
        Repopulate the plot-x combo from *df*'s columns.

        Kept explicit rather than folded into the table refresh: when it rode along with the old
        table population it silently stopped updating whenever the table was drawn from a different
        frame.
        """
        previous = self._plot_x.currentText()
        self._plot_x.blockSignals(True)
        self._plot_x.clear()
        if df is not None and not df.empty:
            for col in df.columns:
                self._plot_x.addItem(str(col))
            for candidate in (previous, "tacsctot_group", "age_c", "group_key", "territory"):
                idx = self._plot_x.findText(candidate)
                if idx >= 0:
                    self._plot_x.setCurrentIndex(idx)
                    break
        self._plot_x.blockSignals(False)

    # ---- filter actions -------------------------------------------------------
    def _on_edit_column_filter(self, column: str) -> None:
        """Open the per-column filter dialog and merge its rules into the active set."""
        working = self._working_df
        base = self._analysis_df
        if base is None or working is None or column not in working.columns:
            return
        # Offer the levels of the *unfiltered* frame, so a level filtered out earlier can be
        # re-included without clearing everything first.
        combined, _, _ = apply_region_combinations(base, self._combinations)
        derived_frame, _ = apply_derived_columns(combined, self._derived)
        series = derived_frame[column] if column in derived_frame.columns else working[column]

        existing = [r for r in self._chips.rules() if r.column == column]
        dialog = ColumnFilterDialog(
            self,
            column=column,
            series=series,
            scope_columns=[c for c in derived_frame.columns if not pd.api.types.is_float_dtype(derived_frame[c])],
            existing=existing,
        )
        if not dialog.exec():
            return
        others = [r for r in self._chips.rules() if r.column != column]
        self._chips.set_rules([*others, *dialog.rules()], [])
        self._recompute_frame()

    def _on_clear_column_filter(self, column: str) -> None:
        """Drop every rule anchored to *column*."""
        remaining = [r for r in self._chips.rules() if r.column != column]
        if len(remaining) != len(self._chips.rules()):
            self._chips.set_rules(remaining, [])
            self._recompute_frame()

    # ---- derived columns ------------------------------------------------------
    def _on_quick_transform(self, column: str, transform: str) -> None:
        """Add a canned transform of *column* straight from the header menu."""
        if self._analysis_df is None:
            return
        name = default_derived_name(column, transform)
        if any(d.name == name for d in self._derived):
            notify(f"{name} already exists.", error=True)
            return
        self._derived.append(
            DerivedColumn(name=name, kind="transform", source=column, transform=transform)
        )
        self._recompute_frame(announce=False)
        notify(f"Added derived column {name}.")


    def _on_edit_combinations(self) -> None:
        """Open the region-combinations editor and re-apply the result."""
        if self._analysis_df is None:
            notify("Reload data before adding region combinations.", error=True)
            return
        dialog = RegionCombinationsDialog(
            self,
            frame=self._analysis_df,
            combinations=self._combinations,
            region_column="territory" if "territory" in self._analysis_df.columns else "group_key",
        )
        if dialog.exec():
            self._combinations = dialog.combinations()
            self._recompute_frame(announce=False)
            if self._combinations:
                notify(f"{len(self._combinations)} region combination(s) applied.")

    def _on_edit_derived(self, *, bin_column: str | None = None) -> None:
        """Open the derived-columns editor, optionally on the bins page for *bin_column*."""
        if self._analysis_df is None:
            notify("Reload data before adding derived columns.", error=True)
            return
        dialog = DerivedColumnsDialog(
            self, frame=self._analysis_df, columns=self._derived, bin_column=bin_column
        )
        if dialog.exec():
            self._derived = dialog.columns()
            self._recompute_frame(announce=False)

    def _on_bin_column(self, column: str) -> None:
        """Open the derived-columns editor to cut *column* into labelled groups."""
        self._on_edit_derived(bin_column=column)



    def _hover_columns(self, df: pd.DataFrame) -> list[str]:
        """Identifier and QC columns worth attaching to every point's hover box."""
        from nvitk.pipes.qvtpy.stage9_autoqc import QC_LABELS

        base = [c for c in ("subject_uid", "territory", "group_key", "visit_id") if c in df.columns]
        qc = [str(c) for c in df.columns if str(c) in QC_LABELS]
        return base + qc



    @staticmethod
    def _panel_note(fig: Any) -> str:
        """
        How many anatomical panels a grouped figure ended up with.

        The two backends record it differently: Matplotlib figures carry ``linked_axes``, Plotly
        subplot figures carry a summary in ``layout.meta``. Reading only one gave "0 panels" on the
        other.
        """
        meta = getattr(getattr(fig, "layout", None), "meta", None) or {}
        summary = meta.get("panels") if isinstance(meta, dict) else ""
        if summary:
            return str(summary)
        return f"{len(getattr(fig, 'linked_axes', []) or [])} anatomical panels."

    def _excluded_points(self, x: str, y: str, group: str) -> pd.DataFrame | None:
        """
        The observations the active filters removed, in the columns the plot uses.

        These are **not** in the model frame — the fit ran on the filtered data — so there is nothing
        inside it to grey out. They have to come from the unfiltered analysis frame and be drawn as
        a separate series, which is what makes it visible whether a filter took out a coherent
        cluster or scattered noise.

        ``None`` when nothing was filtered, or when the plot's columns are not all present in the
        analysis frame (a fit can rename them), since guessing a mapping would mislabel points.
        """
        base, working = self._analysis_df, self._working_df
        if base is None or working is None or len(working) >= len(base):
            return None
        derived, _ = apply_derived_columns(base, self._derived)
        wanted = [c for c in (x, y, group) if c]
        if any(c not in derived.columns for c in wanted):
            return None
        dropped = derived.loc[~derived.index.isin(set(working.index))]
        return dropped if not dropped.empty else None

    def _sync_filter_toggle(self) -> None:
        """Enable the grey-out toggle only when a filter has actually removed something."""
        base, working = self._analysis_df, self._working_df
        removed = len(base) - len(working) if base is not None and working is not None else 0
        self._show_filtered.setEnabled(removed > 0)
        self._show_filtered.setText(
            f"Show filtered ({removed})" if removed > 0 else "Show filtered"
        )

    def _region_column_name(self) -> str:
        """Whichever column carries the vessel identity in this frame."""
        frame = self._analysis_df
        if frame is None:
            return "territory"
        return next(
            (c for c in ("territory", "group_key", "region_id") if c in frame.columns), "territory"
        )

    def _on_export_summary(self, column: str, by: str) -> None:
        """
        Export per-group descriptive statistics for one measurement.

        Summarises the **working** frame — derived columns computed, filters applied — so the table
        describes the rows a model would be fitted on rather than the raw load. The export carries a
        provenance sheet and the underlying values, because a sheet of means with no record of which
        column, which grouping and which confidence level is unreadable a month later.
        """
        frame = self._working_df
        if frame is None or frame.empty:
            notify("Reload the data before exporting a summary.", error=True)
            return

        try:
            summary = summarize_by_group(frame, column, by=by or "")
        except ValueError as exc:
            notify(str(exc), error=True)
            self._status.setText(str(exc))
            return

        stem = f"{column}_by_{by}" if by else f"{column}_summary"
        suggested = statmodels_root(self._repo) / f"{stem}.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export summary of {column}", str(suggested),
            "Excel (*.xlsx);;CSV (*.csv);;TSV (*.tsv);;All (*)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".xlsx")

        keep = [
            c for c in dict.fromkeys([by, column, "subject_uid", "territory"])
            if c and c in frame.columns
        ]
        try:
            written = export_group_summary(
                target,
                summary,
                provenance=summary_provenance(
                    summary, dataset=str(statmodels_root(self._repo)), n_rows_source=len(frame)
                ),
                values=frame.loc[:, keep] if keep else None,
            )
        except Exception as exc:
            log.debug("Summary export failed: %s", exc, exc_info=True)
            notify(f"Export failed: {exc}", error=True)
            self._status.setText(f"Export failed: {exc}")
            return

        n_groups = int((summary["group"] != OVERALL_LABEL).sum())
        notify(f"Summary of {column} \u2192 {written.name}")
        self._status.setText(
            f"Exported {column} summarised by {by or 'the whole cohort'} \u2014 {n_groups} group(s) "
            f"over {len(frame)} row(s) \u2192 {written}"
        )

    def _on_qc_filter(self, column: str, key: str) -> None:
        """
        Apply a ready-made quality-control filter to the analysis dataframe.

        These *drop* rows, so they change the fit — unlike the plot's QC gate, which only greys
        observations out. Existing rules on the same columns are replaced rather than stacked, so
        re-picking a preset re-applies it instead of intersecting two copies of itself.
        """
        if self._analysis_df is None or self._analysis_df.empty:
            notify("Reload the data before applying a QC filter.", error=True)
            return

        decision: dict[str, object] = {}
        try:
            preset = QC_FILTER_BY_KEY[key]
            missing = [c for c in preset.requires if c not in self._analysis_df.columns]
            if missing:
                # Read from the dataset only. Scoring here would give this session's filter a
                # different answer from the published metrics and from every other session; the QC
                # has to mean the same thing wherever it is read.
                raise ValueError(
                    f"{preset.label!r} needs {', '.join(missing)}, which this dataset does not "
                    f"carry. Run 'nvitk-qvtpy-autoqc' to publish the QC metrics, then reload."
                )
            if key in SUBJECT_AWARE_KEYS:
                # The flow presets are not a column threshold: exempt vessels are kept whatever
                # their (absent) score, and a failing carotid or basilar takes its whole subject.
                # The decision is materialised as a column so the chip stays removable.
                keep, decision = flow_qc_keep(
                    self._analysis_df, metric=SUBJECT_AWARE_KEYS[key],
                    region_column=self._region_column_name(),
                )
                self._analysis_df = self._analysis_df.assign(
                    **{KEEP_COLUMN: keep.astype(float)}
                )
            rules = preset_rules(key, self._analysis_df)
        except (KeyError, ValueError) as exc:
            notify(str(exc), error=True)
            self._status.setText(str(exc))
            return

        before = len(self._working_df) if self._working_df is not None else 0
        touched = {r.column for r in rules}
        kept = [r for r in self._chips.rules() if r.column not in touched]
        self._chips.set_rules([*kept, *rules], report=[])
        self._recompute_frame(announce=False)
        after = len(self._working_df) if self._working_df is not None else 0

        removed = max(before - after, 0)
        notify(f"{preset.label}: {removed} row(s) removed.")
        if decision:
            note_parts = [
                f"{decision['n_failing']} vessel(s) failed",
                f"{decision['n_exempt_kept']} exempt row(s) kept (communicating / venous)",
            ]
            if decision["subjects_dropped"]:
                note_parts.append(
                    f"{decision['subjects_dropped']} subject(s) dropped entirely because "
                    f"{', '.join(decision['critical_failures'])} failed"
                )
            log.info("QC filter: %s.", "; ".join(note_parts))
        note = (
            f"{preset.label} — {removed} of {before} row(s) removed, {after} remain. "
            f"This filters the model, not just the plot; remove the chip to undo."
        )
        if decision and decision.get("subjects_dropped"):
            note += (
                f"  {decision['subjects_dropped']} subject(s) removed entirely — "
                f"{', '.join(decision['critical_failures'])} implausible, which makes that "
                f"subject's other flows unreliable too."
            )
        if decision:
            note += f"  {decision['n_exempt_kept']} communicating/venous row(s) kept."
        self._status.setText(note)

    def _on_column_plot(self, column: str, kind: str) -> None:
        """
        Open the distribution viewer for one column.

        Draws from the **unfiltered** frame with the filtered-out rows marked, so the plot shows
        what the filters removed rather than hiding it — which is the only way to tell a filter that
        took out a coherent cluster from one that took out scattered noise.
        """
        base = self._analysis_df if self._analysis_df is not None else self._working_df
        if base is None or base.empty:
            notify("Reload the data before plotting a column.", error=True)
            return

        frame, excluded = self._frame_with_exclusions(base)
        if column not in frame.columns:
            notify(f"“{column}” is not in the analysis dataframe.", error=True)
            return
        dialog = ColumnPlotDialog(
            self, frame=frame, column=column, kind=kind, excluded_mask=excluded,
            default_directory=statmodels_root(self._repo),
        )
        dialog.show()

    def _frame_with_exclusions(self, base: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
        """
        The derived frame plus a boolean mask of the rows the active filters removed.

        The mask is built by comparing the filtered frame's index against the unfiltered one, so it
        follows whatever rules are active without this needing to re-implement them.
        """
        derived, _ = apply_derived_columns(base, self._derived)
        working = self._working_df
        if working is None or len(working) >= len(derived):
            return derived, None
        kept = set(working.index)
        return derived, ~derived.index.isin(kept) if kept else None

    def _on_publish_column(self, column: str) -> None:
        """
        Write a derived column into the dataset as a variable.

        The target table follows the column's *source*: a transform of an image measurement becomes
        an image measurement keyed on subject × region, a transform of a clinical variable becomes a
        clinical one keyed on subject. The dialog states which and why, and nothing is written until
        it is confirmed — this is the one action here that changes a shared dataset.
        """
        frame = self._working_df
        spec = next((d for d in self._derived if d.name == column), None)
        if frame is None or frame.empty:
            notify("Reload the data before publishing a column.", error=True)
            return
        if spec is None:
            notify(f"“{column}” is not a derived column.", error=True)
            return

        target = resolve_publish_target(
            column,
            derived=self._derived,
            measurement_columns={s.column(): s for s in self._measurements.specs()},
            clinical_columns=checked_variable_ids(self._clinical_list),
            cognitive_columns=checked_variable_ids(self._cognitive_list),
            frame_columns=list(frame.columns),
        )
        definition = (
            f"{spec.kind}: {spec.expression}" if spec.kind == "expression"
            else f"{spec.kind}({spec.source}"
            + (f", {spec.transform}" if spec.transform else "")
            + ")"
        )
        region_columns = [c for c in ("territory", "group_key", "region_id") if c in frame.columns]

        dialog = PublishDerivedDialog(
            self,
            repo=self._repo,
            frame=frame,
            column=column,
            target=target,
            definition=definition,
            region_columns=region_columns or ["territory"],
        )
        if not dialog.exec():
            return
        summary = dialog.summary()
        request = dialog.request()
        notify(
            f"Published {request.variable_id} → {summary.get('table')} "
            f"({summary.get('n_rows', 0)} rows)."
        )
        self._status.setText(
            f"Published “{column}” as {request.variable_id} in {summary.get('table')} — "
            f"{summary.get('n_rows', 0)} rows over {summary.get('n_subjects', 0)} subjects. "
            f"Reload to see it in the covariate lists."
        )

    def _on_export_frame(self) -> None:
        """Save the active (derived + filtered) frame to a spreadsheet."""
        frame = self._working_df
        if frame is None or frame.empty:
            notify("Nothing to export — reload the data first.", error=True)
            return

        suggested = statmodels_root(self._repo) / f"{self._model_name.text().strip() or 'analysis'}.xlsx"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export analysis dataframe",
            str(suggested),
            "Excel workbook (*.xlsx);;CSV (*.csv);;Tab-separated (*.tsv)",
        )
        if not path:
            return
        # A user who typed a bare name gets the extension of the filter they picked.
        out = Path(path)
        if not out.suffix:
            out = out.with_suffix({"CSV (*.csv)": ".csv", "Tab-separated (*.tsv)": ".tsv"}.get(selected, ".xlsx"))

        provenance = build_provenance_frame(
            frame=frame,
            source_rows=len(self._analysis_df) if self._analysis_df is not None else len(frame),
            measurements=self._measurements.specs(),
            join=self._measurements.join(),
            covariates=(self._load_meta or {}).get("covariates", []),
            visit_provenance=(self._load_meta or {}).get("visit_provenance", {}),
            derived=self._derived,
            filters=self._chips.rules(),
            filter_report=self._filter_report,
            dataset=str(self._repo.root),
        )
        try:
            written = export_analysis_frame(out, frame, provenance=provenance)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            notify(f"Export failed: {exc}", error=True)
            return
        self._status.setText(f"Exported {len(frame)} rows → {written}")
        notify(f"Exported analysis dataframe → {written}")

    def _on_set_plot_x(self, column: str) -> None:
        """Use *column* as the plot's x axis."""
        idx = self._plot_x.findText(column)
        if idx >= 0:
            self._plot_x.setCurrentIndex(idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Analysis type
    # ──────────────────────────────────────────────────────────────────────────
    def _analysis_kind(self) -> str:
        """The selected analysis type."""
        return str(self._analysis_type.currentData() or ANALYSIS_MIXEDLM)

    def _on_glm_family_changed(self) -> None:
        """Offer only the links the selected GLM family supports, and describe it."""
        spec = GLM_FAMILIES.get(str(self._glm_family.currentData() or "gaussian"))
        if spec is None:
            return
        current = str(self._glm_link.currentData() or "")
        self._glm_link.blockSignals(True)
        self._glm_link.clear()
        for link in spec.links:
            self._glm_link.addItem(link, link)
        idx = self._glm_link.findData(current if current in spec.links else spec.default_link)
        self._glm_link.setCurrentIndex(max(idx, 0))
        self._glm_link.blockSignals(False)
        self._glm_family.setToolTip(spec.description)
        if self._analysis_kind() == ANALYSIS_GLM:
            self._analysis_hint.setText(spec.description)

    def _r_backend_status(self):
        """Probe the R backend once per window — the Rscript call takes a moment."""
        if self._r_status is None:
            self._r_status = r_backend_status()
            log.info("R/lme4 backend: %s", self._r_status.summary())
        return self._r_status

    def _on_formula_changed(self) -> None:
        """Keep the MMRM covariance controls in step with an inline term as it is typed."""
        if self._analysis_kind() == ANALYSIS_MMRM:
            self._sync_mmrm_hint()

    def _mmrm_backend_status(self):
        """Probe the mmrm backend once per window."""
        if self._mmrm_status is None:
            self._mmrm_status = mmrm_backend_status()
            log.info("R/mmrm backend: %s", self._mmrm_status.summary())
        return self._mmrm_status

    def _robust_backend_status(self):
        """Probe the robustbase backend once per window."""
        if self._robust_status is None:
            self._robust_status = robust_backend_status()
            log.info("R/robustbase backend: %s", self._robust_status.summary())
        return self._robust_status

    def _suggested_mmrm_term(self) -> str:
        """A covariance term that would suit the loaded frame, offered when the formula lacks one."""
        df = self._working_df
        columns = set(df.columns) if df is not None else set()
        visit = next((c for c in ("territory", "group_key") if c in columns), "territory")
        subject = next((c for c in ("subject_uid", "patient_id") if c in columns), "subject_uid")
        return f"us({visit} | {subject})"

    def _sync_mmrm_hint(self) -> None:
        """
        Describe the covariance term written in the formula, and pre-flight it against the frame.

        The term is part of the formula, so it is read from there rather than assembled from
        controls — but it still drives the checks that R's own error messages do not make obvious:
        which columns it needs, whether any subject × level cell is duplicated, and whether the
        structure has more parameters than the data can support.
        """
        formula = self._formula.toPlainText().strip()
        _fixed, term = parse_mmrm_covariance(formula)

        if term is None:
            self._analysis_hint.setText(
                "Write the covariance structure into the formula, as in R — for example:\n"
                f"    … + {self._suggested_mmrm_term()}\n"
                "Structures: "
                + ", ".join(f"{k} ({v.label.lower()})" for k, v in COVARIANCE_STRUCTURES.items())
            )
            return

        spec = COVARIANCE_STRUCTURES.get(term.structure)
        lines = [f"{term.text} — {spec.label}: {spec.description}" if spec else term.text]

        df = self._working_df
        if df is not None:
            missing = [c for c in term.columns() if c not in df.columns]
            if missing:
                lines.append(f"⚠ Not a column of the analysis dataframe: {', '.join(missing)}.")
            else:
                lines.extend(
                    f"⚠ {p}"
                    for p in validate_mmrm_data(
                        df, visit=term.visit, subject=term.subject, structure=term.structure
                    )
                )
        self._analysis_hint.setText("\n".join(lines))

    def _on_convert_to_lme4(self) -> None:
        """Rewrite the MixedLM fields as a single lme4 formula, in place."""
        try:
            converted = mixedlm_to_lme4_formula(
                self._formula.toPlainText().strip(),
                groups=self._groups.text().strip() or "group_key",
                re_formula=self._re_formula.text().strip() or "1",
                vc_formula=parse_vc_formula(self._vc_formula.text()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Cannot convert", str(exc))
            return
        self._formula.setPlainText(converted)
        notify("Converted the MixedLM specification to an lme4 formula.")

    def _on_insert_term(self) -> None:
        """Insert a spline / polynomial term into the formula at the cursor."""
        df = self._working_df
        if df is None or df.empty:
            notify("Reload data before building a term.", error=True)
            return
        numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            notify("No numeric column to build a term from.", error=True)
            return
        dialog = SplineTermDialog(self, columns=numeric)
        if not dialog.exec():
            return
        cursor = self._formula.textCursor()
        cursor.insertText(dialog.term())

    def _on_insert_covariance_term(self) -> None:
        """Insert an mmrm covariance term into the formula at the cursor."""
        df = self._working_df
        if df is None or df.empty:
            notify("Reload data before building a covariance term.", error=True)
            return
        # A covariance term indexes levels, so float columns are never candidates.
        candidates = [str(c) for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
        if len(candidates) < 2:
            notify("Need at least two non-numeric columns to build a covariance term.", error=True)
            return

        _fixed, existing = parse_mmrm_covariance(self._formula.toPlainText())
        dialog = CovarianceTermDialog(
            self,
            columns=candidates,
            visit=existing.visit if existing else "",
            subject=existing.subject if existing else "",
        )
        if not dialog.exec():
            return
        if existing is not None:
            # Replacing beats appending: two covariance terms is not a valid mmrm formula, and the
            # parser would silently take the first.
            self._formula.setPlainText(
                self._formula.toPlainText().replace(existing.text, dialog.term(), 1)
            )
            return
        cursor = self._formula.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(f" + {dialog.term()}")

    def _sync_analysis_type(self) -> None:
        """Swap the formulation panel and hide the controls the selected engine does not use."""
        kind = self._analysis_kind()
        page = {ANALYSIS_NONLINEAR: 1, ANALYSIS_MEDIATION: 2}.get(kind, 0)
        self._model_stack.setCurrentIndex(page)
        self._analysis_hint.setText(ANALYSIS_HINTS.get(kind, ""))

        # A random structure only exists for MixedLM; a family/link only for GLM; robust standard
        # errors only for OLS. Showing them all at once invites fitting something you did not mean.
        is_mixed = kind == ANALYSIS_MIXEDLM
        for widget in (self._groups, self._groups_label, self._re_formula, self._re_label,
                       self._vc_formula, self._vc_label):
            widget.setVisible(is_mixed)
        is_glm = kind == ANALYSIS_GLM
        for widget in (self._glm_family, self._glm_family_label, self._glm_link, self._glm_link_label):
            widget.setVisible(is_glm)
        is_ols = kind == ANALYSIS_OLS
        for widget in (self._robust, self._robust_label):
            widget.setVisible(is_ols)
        is_lme4 = kind == ANALYSIS_LME4
        for widget in (self._lme4_family, self._lme4_family_label,
                       self._lme4_reml, self._lme4_reml_label, self._btn_lme4_convert):
            widget.setVisible(is_lme4)
        is_mmrm = kind == ANALYSIS_MMRM
        for widget in (self._mmrm_method, self._mmrm_method_label):
            widget.setVisible(is_mmrm)
        is_sem = kind == ANALYSIS_SEM
        for widget in (self._sem_backend, self._sem_backend_label,
                       self._sem_standardize, self._sem_standardize_label):
            widget.setVisible(is_sem)
        is_mrf = kind == ANALYSIS_MRF
        for widget in (self._mrf_family, self._mrf_family_label,
                       self._mrf_method, self._mrf_method_label):
            widget.setVisible(is_mrf)
        is_lmrob = kind == ANALYSIS_LMROB
        for widget in (self._lmrob_method, self._lmrob_method_label,
                       self._lmrob_psi, self._lmrob_psi_label,
                       self._lmrob_setting, self._lmrob_setting_label):
            widget.setVisible(is_lmrob)
        if is_lmrob:
            # A preset fixes both the estimation chain and the loss, so leaving them editable would
            # show a choice that has no effect.
            preset = bool(self._lmrob_setting.currentData())
            self._lmrob_method.setEnabled(not preset)
            self._lmrob_psi.setEnabled(not preset)
        self._btn_insert_covariance.setVisible(is_mmrm)
        # The network-syntax builder writes the whole path model, which only SEM consumes.
        self._btn_network_syntax.setVisible(is_sem)
        # The curved-term builder emits patsy syntax (bs(...), I(x ** 2)); the R engines parse their
        # formulas in R, where those are wrong. Offer it only where it applies.
        self._btn_insert_term.setVisible(kind not in ANALYSIS_R_KINDS)
        # An MMRM's covariance is estimated by REML in the same sense, so the toggle applies here too.
        for widget in (self._lme4_reml, self._lme4_reml_label):
            widget.setVisible(is_lme4 or is_mmrm)

        if is_lme4 or is_mmrm or is_lmrob or is_sem or is_mrf:
            status = (
                self._r_backend_status() if is_lme4
                else self._mmrm_backend_status() if is_mmrm
                else self._robust_backend_status() if is_lmrob
                else sem_backend_status() if is_sem
                else gam_backend_status()
            )
            if not status.available:
                self._analysis_hint.setText(f"⚠ {status.reason}\n{status.install_hint()}")
            else:
                self._analysis_hint.setText(f"{ANALYSIS_HINTS[kind]}\nUsing {status.summary()}.")
        if is_mmrm:
            self._sync_mmrm_hint()

        # The figure picker serves mediation and MMRM, which each produce several plots; the group
        # checklist only applies to the grouped model plots.
        self._plot.set_kind_row_visible(
            kind in {ANALYSIS_MEDIATION, ANALYSIS_MMRM, ANALYSIS_LMROB,
                     ANALYSIS_SEM, ANALYSIS_MRF}
        )
        self._plot.set_groups_visible(kind in ANALYSIS_FORMULA_KINDS)
        # Panelling needs a family of curves over anatomical levels to split up.
        for widget in (self._plot_display, self._plot_display_label):
            widget.setVisible(kind in ANALYSIS_PANEL_KINDS)
        self._sync_display_enabled()
        if is_mmrm:
            self._set_plot_kinds(_MMRM_PLOTS)
        elif is_lmrob:
            self._set_plot_kinds(_LMROB_PLOTS)
        elif is_sem:
            self._set_plot_kinds(_SEM_PLOTS)
        elif is_mrf:
            self._set_plot_kinds(_MRF_PLOTS)
        elif kind == ANALYSIS_MEDIATION and self._mediation_bundle is None:
            self._set_plot_kinds((("Run a mediation first", ""),))

        if kind == ANALYSIS_MEDIATION:
            self._mediation_form.set_columns(self._working_df)
            if self._mediation_bundle is None:
                self._report.set_message("Configure the mediation and press Run.")
                self._plot.clear()
                self._plot.set_levels("", [])
        else:
            if kind == ANALYSIS_NONLINEAR:
                self._sync_nonlinear_columns(self._working_df)
            if self._last_result is None:
                self._report.set_message("Fit a model to see its summary here.")
                self._plot.clear()

    def _sync_nonlinear_columns(self, df: pd.DataFrame | None) -> None:
        """Offer the frame's numeric columns as the curve's x and y."""
        numeric = (
            [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if df is not None and not df.empty
            else []
        )
        for combo, preferred in ((self._nl_x, ("age_c", "age_at_mri")), (self._nl_y, tuple(self._measurements.columns()))):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for name in numeric:
                combo.addItem(name)
            idx = combo.findText(current)
            if idx < 0:
                for candidate in preferred:
                    idx = combo.findText(candidate)
                    if idx >= 0:
                        break
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────────
    # MixedLM
    # ──────────────────────────────────────────────────────────────────────────
    def _on_fit(self) -> None:
        """Fit the selected model on the working frame, then show its report and plot."""
        kind = self._analysis_kind()
        try:
            if self._working_df is None:
                raise ValueError("Reload the data before fitting.")
            if kind == ANALYSIS_NONLINEAR:
                result, model_df, meta, outcome, groups = self._fit_nonlinear()
            else:
                result, model_df, meta, outcome, groups = self._fit_formula_model(kind)
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            notify(f"Statmodels fit failed: {exc}", error=True)
            return

        self._last_result = result
        self._last_model_df = model_df
        self._last_fit_meta = meta
        self._last_outcome = outcome
        self._mediation_bundle = None

        info = model_info_dict(
            result, outcome_name=outcome or self._primary_column(), group_name=groups, meta=meta
        )
        if kind == ANALYSIS_LMROB:
            # The weights need the fitted frame to name each row's subject and territory, which the
            # info builder never sees — it only gets the fit.
            try:
                info["robust_weights"] = lmrob_weights_frame(result, model_df)
            except Exception as exc:
                log.debug("Could not read the robustness weights: %s", exc, exc_info=True)
        note = dropped_rows_note(meta)
        raw = render_mixedlm_info(info)
        self._report.set_mixedlm(info, raw_text=f"{note}\n\n{raw}" if note else raw, note=note)

        label = dict(ANALYSIS_ITEMS).get(kind, kind)
        detail = ""
        if kind == ANALYSIS_MIXEDLM:
            detail = f"  |  groups={groups}  |  re={self._re_formula.text().strip()!r}"
        elif kind == ANALYSIS_GLM:
            detail = f"  |  {meta.get('family_label')} / {meta.get('link_label')} link"
        elif kind == ANALYSIS_OLS and meta.get("robust"):
            detail = f"  |  {meta['robust']} standard errors"
        self._status.setText(
            f"Fitted {label}: n={meta.get('n_rows')}"
            + (f" (dropped {meta.get('n_rows_dropped')} incomplete)" if meta.get("n_rows_dropped") else "")
            + detail
            + f"  |  dataset={self._repo.root}"
        )
        self._sync_plot_levels(model_df, groups)
        self._on_plot()
        notify(f"{label} fit complete.")

    def _fit_formula_model(self, kind: str):
        """Fit MixedLM / OLS / GLM from the shared formulation panel."""
        formula = self._formula.toPlainText().strip()
        df, outcome = resolve_outcome_column(
            self._working_df.copy(), formula, self._measurements.columns()
        )
        groups = self._groups.text().strip() or "group_key"

        if kind == ANALYSIS_SEM:
            spec = SemSpec(
                syntax=formula,
                backend=str(self._sem_backend.currentData() or ""),
                standardize=self._sem_standardize.isChecked(),
            )
            result, model_df, meta = fit_sem(data=df, spec=spec)
            return result, model_df, meta, outcome, groups
        if kind == ANALYSIS_MRF:
            result, model_df, meta = fit_mrf(
                data=df,
                formula=formula if "s(" in formula else "",
                outcome=outcome or formula.split("~")[0].strip(),
                region_column=groups,
                covariates=[
                    c for c in _formula_columns(df.columns, formula.split("~", 1)[-1])
                    if c != groups
                ],
                family=str(self._mrf_family.currentData() or "gaussian"),
                method=str(self._mrf_method.currentData() or "REML"),
            )
            return result, model_df, meta, outcome, meta["region_column"]
        if kind == ANALYSIS_MMRM:
            # The covariance term comes from the formula; only the arguments that are not part of it
            # are passed separately.
            result, model_df, meta = fit_mmrm(
                data=df,
                formula=formula,
                method=str(self._mmrm_method.currentData() or "Satterthwaite"),
                reml=self._lme4_reml.isChecked(),
            )
            # The repeated dimension is what the plot splits by.
            return result, model_df, meta, outcome, meta["visit"]
        if kind == ANALYSIS_LME4:
            result, model_df, meta = fit_lme4(
                data=df,
                formula=formula,
                family=str(self._lme4_family.currentData() or "gaussian"),
                reml=self._lme4_reml.isChecked(),
            )
            # lme4 names its grouping factors in the formula; use the first for the plot's groups.
            factors = meta.get("grouping_factors") or []
            return result, model_df, meta, outcome, (factors[0] if factors else groups)
        if kind == ANALYSIS_LMROB:
            result, model_df, meta = fit_lmrob(
                data=df,
                formula=formula,
                method=str(self._lmrob_method.currentData() or "MM"),
                psi=str(self._lmrob_psi.currentData() or "bisquare"),
                setting=str(self._lmrob_setting.currentData() or ""),
            )
        elif kind == ANALYSIS_OLS:
            result, model_df, meta = fit_ols(
                data=df, formula=formula, robust=self._robust.currentData()
            )
        elif kind == ANALYSIS_GLM:
            result, model_df, meta = fit_glm(
                data=df,
                formula=formula,
                family=str(self._glm_family.currentData() or "gaussian"),
                link=str(self._glm_link.currentData() or "") or None,
            )
        else:
            result, model_df, meta = fit_or_load_mixedlm(
                data=df,
                formula=formula,
                groups=groups,
                re_formula=self._re_formula.text().strip() or "0",
                vc_formula=parse_vc_formula(self._vc_formula.text()),
                overwrite=True,
                dropna_columns=None,
            )
        # OLS and GLM have no grouping of their own, but the plot still splits by ``groups`` when the
        # formula contains that factor — so it is returned for every engine.
        return result, model_df, meta, outcome, groups

    def _fit_nonlinear(self):
        """Fit the selected parametric curve of y against x."""
        p0_text = self._nl_p0.text().strip()
        p0 = None
        if p0_text:
            try:
                p0 = [float(t) for t in p0_text.replace(";", ",").split(",") if t.strip()]
            except ValueError as exc:
                raise ValueError(f"Start values must be numbers: {exc}") from exc
        result, model_df, meta = fit_nonlinear(
            data=self._working_df,
            x=self._nl_x.currentText(),
            y=self._nl_y.currentText(),
            model=str(self._nl_model.currentData() or "exp_decay"),
            p0=p0,
        )
        return result, model_df, meta, result["y"], ""

    def _sync_plot_levels(self, df: pd.DataFrame, column: str) -> None:
        """Rebuild the plot's group checklist from *column*'s levels in *df*."""
        if column in df.columns:
            levels = sorted(str(v) for v in df[column].dropna().unique())
        else:
            levels = []
        self._plot.set_levels(column, levels)
        # A restored config named its groups before any fit told us what the levels are.
        if self._pending_plot_groups is not None:
            self._plot.set_checked_levels(self._pending_plot_groups)
            self._pending_plot_groups = None

    def _covariate_reference_values(
        self, df: pd.DataFrame, formula: str, exclude: set[str]
    ) -> dict[str, Any]:
        """
        Reference value for every formula term the EMM grid does not already vary.

        The grid patsy evaluates only carries the x axis and the facet column. Any *other* term in
        the formula — a second measurement, a covariate, another categorical factor — is simply
        absent, and patsy raises ``NameError`` for it, which used to leave the plot showing raw
        points and no model curves at all.

        Numeric terms are pinned at their mean, categorical ones at their most frequent level. The
        marginal means are then read "at average age, for the modal group", which is the usual EMM
        convention.
        """
        from nvitk.stats import formula_columns

        refs: dict[str, Any] = {}
        for name in formula_columns(df.columns, formula):
            if name in exclude or name not in df.columns:
                continue
            series = df[name]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                if values.notna().any():
                    refs[name] = float(values.mean())
            else:
                modes = series.dropna().mode()
                if not modes.empty:
                    refs[name] = modes.iloc[0]
        return refs

    def _on_plot(self) -> None:
        """Redraw the current analysis's plot with the active display options."""
        kind = self._analysis_kind()
        if kind == ANALYSIS_MEDIATION:
            self._plot_mediation()
            return
        if self._last_result is None or self._last_model_df is None:
            return
        if kind == ANALYSIS_NONLINEAR:
            self._plot_nonlinear()
            return
        if kind == ANALYSIS_LME4:
            self._plot_lme4()
            return
        if kind == ANALYSIS_MMRM:
            self._plot_mmrm()
            return
        if kind == ANALYSIS_LMROB:
            self._plot_lmrob()
            return
        if kind == ANALYSIS_SEM:
            self._plot_sem()
            return
        if kind == ANALYSIS_MRF:
            self._plot_mrf()
            return

        try:
            import matplotlib.pyplot as plt

            df = self._last_model_df
            y = self._last_outcome if (self._last_outcome and self._last_outcome in df.columns) else None
            if y is None:
                for candidate in [*self._measurements.columns(), "flow", "flow_mean", "pi", "mean_cbf"]:
                    if candidate in df.columns:
                        y = candidate
                        break
            x = self._plot_x.currentText().strip()
            if not x or x not in df.columns:
                for candidate in ("tacsctot_group", "age_c", "group_key"):
                    if candidate in df.columns:
                        x = candidate
                        break
            group = self._groups.text().strip() or "group_key"
            if not x or y is None or y not in df.columns or group not in df.columns:
                raise ValueError(
                    f"Cannot plot: need x/y/group columns (have {list(df.columns)})"
                )

            selected = self._plot.checked_levels()
            all_levels = sorted(str(v) for v in df[group].dropna().unique())
            subset = bool(selected) and len(selected) < len(all_levels)
            if not selected:
                raise ValueError("No groups selected — tick at least one in the Groups list.")

            formula = self._formula.toPlainText().strip()
            # Only x and the outcome are excluded: the grouping column has to be pinned too, or the
            # continuous CI band cannot build its design matrix when the formula contains it. The
            # plotter drops any reference that collides with the axis or the facet it is varying.
            refs = self._covariate_reference_values(df, formula, exclude={x, y})

            display = str(self._plot_display.currentData() or "overview")
            display_note = ""
            dropped = (
                self._excluded_points(x, y, group)
                if self._show_filtered.isChecked() and self._show_filtered.isEnabled() else None
            )
            geometry_error = ""

            with plt.style.context("default"):
                if self._interactive_plot.isChecked():
                    geometry = statsmodels_geometry(
                        self._last_result, df, x=x, y=y, group=group,
                        mode=str(self._plot_mode.currentData() or "auto"),
                        group_order=selected, covariate_refs=refs,
                        errorbar=self._show_ci.isChecked(),
                    )
                    if not self._include_points.isChecked():
                        geometry.points = None
                    elif dropped is not None:
                        # Append them so one mask can mark which of the plotted rows were filtered.
                        geometry.points = pd.concat(
                            [geometry.points, dropped.loc[:, geometry.points.columns]],
                            ignore_index=True,
                        )
                    excluded = (
                        np.r_[np.zeros(len(df), bool), np.ones(len(dropped), bool)]
                        if dropped is not None and self._include_points.isChecked() else None
                    )
                    render_kwargs = dict(
                        x=x, y=y, group=group, hover_columns=self._hover_columns(df),
                        excluded_mask=excluded,
                        show_excluded=True,
                        errorbar=self._show_ci.isChecked(),
                        title=f"{y} ~ {x} | {group}", x_label=x, y_label=y,
                    )
                    try:
                        fig = render(geometry, display=display, **render_kwargs)
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                        fig = render(geometry, display="overview", **render_kwargs)
                    geometry_error = geometry.error
                else:
                    plot_kwargs = dict(
                        result=self._last_result, df_fit=df, x=x, y=y, group=group,
                        mode=str(self._plot_mode.currentData() or "auto"),
                        include_points=self._include_points.isChecked(),
                        errorbar=self._show_ci.isChecked(),
                        group_order=selected, restrict_to_orders=subset,
                        covariate_refs=refs, excluded_points=dropped,
                        title=f"MixedLM: {y} ~ {x} | {group}",
                    )
                    try:
                        fig = plot_mixedlm_params(display=display, **plot_kwargs)
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                        fig = plot_mixedlm_params(display="overview", **plot_kwargs)
                    geometry_error = getattr(fig, "emm_error", "") or getattr(fig, "ci_error", "")
            if fig is None:
                fig = plt.gcf()
            self._plot.show_figure(fig)

            notes = []
            if display_note:
                notes.append(display_note)
            elif display == "grouped":
                notes.append(self._panel_note(fig))
            if subset:
                notes.append(f"Showing {len(selected)} of {len(all_levels)} {group} levels.")
            if dropped is not None:
                notes.append(f"{len(dropped)} filtered observation(s) shown in grey.")
            if geometry_error:
                notes.append(f"\u26a0 {geometry_error}")
            self._plot.set_status("  ".join(notes))
        except Exception as exc:
            log.debug("Plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _plot_mmrm(self) -> None:
        """
        Least-squares means by the repeated dimension, or the estimated correlation matrix.

        The correlation heatmap is the one worth looking at for an unstructured fit: it shows
        whether the equal-correlation assumption a random-intercept model makes is anywhere near
        the truth.
        """
        try:
            import matplotlib.pyplot as plt

            meta = self._last_fit_meta or {}
            visit = meta.get("visit", "")
            kind = str(self._mediation_plot.currentData() or "emmeans")

            # The Groups checklist filters both figures: which levels appear on the axis, and which
            # rows/columns of the correlation matrix are shown.
            selected = self._plot.checked_levels()
            all_levels = list(self._plot._group_boxes)
            subset = bool(selected) and len(selected) < len(all_levels)
            if not selected and all_levels:
                raise ValueError("No groups selected — tick at least one in the Groups list.")

            display = str(self._plot_display.currentData() or "overview")
            with plt.style.context("default"):
                if kind == "correlation":
                    # A correlation matrix is one object, not a family of curves; there is nothing
                    # to split into panels.
                    correlation = mmrm_correlation_matrix(self._last_result)
                    if selected:
                        keep = [str(v) for v in selected if str(v) in correlation.index]
                        if keep:
                            correlation = correlation.loc[keep, keep]
                    fig = matrix_plot(
                        correlation,
                        title=f"MMRM {meta.get('structure_label', '')}: correlation between "
                        f"{visit} levels",
                        value_label="correlation",
                    )
                    note = ""
                else:
                    x = self._plot_x.currentText().strip() or visit
                    hue = visit if x != visit else None
                    specs = f"~ {x}" + (f" | {hue}" if hue else "")
                    frame = mmrm_emmeans(self._last_result, specs)
                    # Marginal means are estimated over every level; hiding one only removes it from
                    # the display, exactly as for the other engines.
                    if selected and visit in frame.columns:
                        frame = frame.loc[frame[visit].astype(str).isin(set(selected))]
                    geometry = mmrm_geometry(frame, x=x, hue=hue or "")
                    note = f"Least-squares means from emmeans, at {meta.get('method')} df."
                    panel_group = hue or x
                    try:
                        fig = render(
                            geometry, x=x, y=self._last_outcome or "estimate",
                            group=panel_group, display=display,
                            title=f"MMRM least-squares means: {self._last_outcome or ''} by {x}",
                            y_label=self._last_outcome or "Estimated marginal mean",
                        )
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        note = f"⚠ Grouped display unavailable — {exc}"
                        fig = render(
                            geometry, x=x, y=self._last_outcome or "estimate",
                            group=panel_group, display="overview",
                            title=f"MMRM least-squares means: {self._last_outcome or ''} by {x}",
                            y_label=self._last_outcome or "Estimated marginal mean",
                        )
                    else:
                        if display == "grouped":
                            note += f"  {(fig.layout.meta or {}).get('panels', '')}"
            self._plot.show_figure(fig)
            if subset:
                note = (note + "  " if note else "") + (
                    f"Showing {len(selected)} of {len(all_levels)} {visit} levels."
                )
            self._plot.set_status(note)
        except Exception as exc:
            log.debug("MMRM plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _plot_lme4(self) -> None:
        """Population and per-group curves predicted through pymer4."""
        try:
            import matplotlib.pyplot as plt

            df = self._last_model_df
            group = (self._last_fit_meta or {}).get("grouping_factors") or []
            group_col = group[0] if group else ""
            y = self._last_outcome or self._primary_column()
            x = self._plot_x.currentText().strip()
            if not x or x not in df.columns:
                x = next((c for c in ("age_c", "tacsctot_group") if c in df.columns), "")
            if not x:
                raise ValueError("Choose a plot x column.")

            selected = self._plot.checked_levels()
            all_levels = (
                sorted(str(v) for v in df[group_col].dropna().unique()) if group_col in df.columns else []
            )
            subset = bool(selected) and len(selected) < len(all_levels)
            display = str(self._plot_display.currentData() or "overview")
            display_note = ""
            fixed_formula = (self._last_fit_meta or {}).get("fixed_formula", "")
            dropped = (
                self._excluded_points(x, y, group_col)
                if self._show_filtered.isChecked() and self._show_filtered.isEnabled() else None
            )

            with plt.style.context("default"):
                if self._interactive_plot.isChecked():
                    from nvitk.stats.r_mixedlm import _emmeans_band, lme4_predict

                    geometry = r_model_geometry(
                        self._last_result, df, x=x, y=y, group=group_col,
                        mode=str(self._plot_mode.currentData() or "auto"),
                        group_order=selected or None,
                        predict_fn=lme4_predict, band_fn=_emmeans_band,
                        fixed_formula=fixed_formula,
                        errorbar=self._show_ci.isChecked(),
                    )
                    if not self._include_points.isChecked():
                        geometry.points = None
                    elif dropped is not None:
                        geometry.points = pd.concat(
                            [geometry.points, dropped.loc[:, geometry.points.columns]],
                            ignore_index=True,
                        )
                    excluded = (
                        np.r_[np.zeros(len(df), bool), np.ones(len(dropped), bool)]
                        if dropped is not None and self._include_points.isChecked() else None
                    )
                    render_kwargs = dict(
                        x=x, y=y, group=group_col, hover_columns=self._hover_columns(df),
                        excluded_mask=excluded, show_excluded=True,
                        errorbar=self._show_ci.isChecked(),
                        title=f"lme4: {y} ~ {x} | {group_col}", x_label=x, y_label=y,
                    )
                    try:
                        fig = render(geometry, display=display, **render_kwargs)
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                        fig = render(geometry, display="overview", **render_kwargs)
                    ci_error = geometry.error
                else:
                    kwargs = dict(
                        model=self._last_result, df_fit=df, x=x, y=y, group=group_col,
                        mode=str(self._plot_mode.currentData() or "auto"),
                        include_points=self._include_points.isChecked(),
                        errorbar=self._show_ci.isChecked(),
                        # Whether the grouping factor has per-level marginal means depends on it
                        # being in the fixed part, which only the formula knows.
                        fixed_formula=fixed_formula,
                        group_order=selected or None, restrict_to_orders=subset,
                        excluded_points=dropped,
                        title=f"lme4: {y} ~ {x} | {group_col}",
                    )
                    try:
                        fig = plot_lme4_params(display=display, **kwargs)
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                        fig = plot_lme4_params(display="overview", **kwargs)
                    ci_error = getattr(fig, "ci_error", "")
            self._plot.show_figure(fig)
            notes = []
            if display_note:
                notes.append(display_note)
            elif display == "grouped":
                notes.append(self._panel_note(fig))
            if subset:
                notes.append(f"Showing {len(selected)} of {len(all_levels)} {group_col} levels.")
            if dropped is not None:
                notes.append(f"{len(dropped)} filtered observation(s) shown in grey.")
            if self._show_ci.isChecked() and ci_error:
                notes.append(f"⚠ {ci_error}")
            self._plot.set_status("  ".join(notes))
        except Exception as exc:
            log.debug("lme4 plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _plot_lmrob(self) -> None:
        """
        Robust fit curves, or the robustness weights.

        The weights figure is the one specific to this engine: it shows which observations the
        estimator discounted, which is a QC shortlist rather than a result.
        """
        try:
            import matplotlib.pyplot as plt

            df = self._last_model_df
            group = self._groups.text().strip() or "group_key"
            kind = str(self._mediation_plot.currentData() or "fit")

            if kind == "weights":
                weights = lmrob_weights_frame(self._last_result, df)
                label = next(
                    (c for c in ("territory", "group_key", "subject_uid") if c in weights.columns), ""
                )
                with plt.style.context("default"):
                    fig = plot_lmrob_weights(
                        weights,
                        label_column=label,
                        title=f"Robustness weights — {self._last_outcome or ''}",
                    )
                self._plot.show_figure(fig)
                rejected = int(weights["rejected"].sum())
                self._plot.set_status(
                    f"{int((weights['weight'] < 0.5).sum())} of {len(weights)} observations carry a "
                    f"weight below 0.5 ({rejected} rejected outright). Low-weight rows are "
                    f"candidates for QC, not findings."
                )
                return

            y = self._last_outcome or self._primary_column()
            x = self._plot_x.currentText().strip()
            if not x or x not in df.columns:
                x = next((c for c in ("age_c", "tacsctot_group") if c in df.columns), "")
            if not x:
                raise ValueError("Choose a plot x column.")

            selected = self._plot.checked_levels()
            all_levels = (
                sorted(str(v) for v in df[group].dropna().unique()) if group in df.columns else []
            )
            subset = bool(selected) and len(selected) < len(all_levels)
            display = str(self._plot_display.currentData() or "overview")
            display_note = ""
            kwargs = dict(
                fit=self._last_result,
                df_fit=df,
                x=x,
                y=y,
                group=group if group in df.columns else "",
                mode=str(self._plot_mode.currentData() or "auto"),
                include_points=self._include_points.isChecked(),
                errorbar=self._show_ci.isChecked(),
                fixed_formula=(self._last_fit_meta or {}).get("formula", ""),
                group_order=selected or None,
                restrict_to_orders=subset,
                title=f"lmrob: {y} ~ {x}",
            )
            with plt.style.context("default"):
                try:
                    fig = plot_lmrob_params(display=display, **kwargs)
                except ValueError as exc:
                    if display != "grouped":
                        raise
                    display_note = f"⚠ Grouped display unavailable — {exc}"
                    fig = plot_lmrob_params(display="overview", **kwargs)
            self._plot.show_figure(fig)

            notes = []
            if display_note:
                notes.append(display_note)
            elif display == "grouped":
                notes.append(self._panel_note(fig))
            if subset:
                notes.append(f"Showing {len(selected)} of {len(all_levels)} {group} levels.")
            if self._show_ci.isChecked() and getattr(fig, "ci_error", ""):
                notes.append(f"⚠ {fig.ci_error}")
            self._plot.set_status("  ".join(notes))
        except Exception as exc:
            log.debug("lmrob plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")


    def _on_insert_network_syntax(self) -> None:
        """Write the vascular path model into the formula box, built from the vessels present."""
        df = self._working_df
        if df is None or df.empty:
            notify("Reload the data before building the network syntax.", error=True)
            return
        column = "territory" if "territory" in df.columns else "group_key"
        if column not in df.columns:
            notify("No territory column to build a network from.", error=True)
            return

        nodes = sorted({n for n in (canonical_node(v) for v in df[column].dropna().unique()) if n})
        covariates = [c for c in ("age_c", "sex") if c in df.columns]
        try:
            syntax = sem_model_syntax(nodes=nodes, covariates=covariates)
        except ValueError as exc:
            notify(str(exc), error=True)
            return
        self._formula.setPlainText(syntax)
        self._status.setText(
            f"Network syntax for {len(nodes)} vessel(s). It needs one column per vessel — pivot the "
            f"frame with a region combination, or fit on a wide frame."
        )

    def _plot_sem(self) -> None:
        """Path coefficients as a forest, or the fitted network as a diagram."""
        try:
            import matplotlib.pyplot as plt

            kind = str(self._mediation_plot.currentData() or "paths")
            paths = sem_paths_frame(
                self._last_result, backend=(self._last_fit_meta or {}).get("backend", "")
            )
            regressions = paths.loc[paths["op"] == "~"] if "op" in paths.columns else paths
            with plt.style.context("default"):
                if kind == "network":
                    fig = network_plot(
                        regressions, node_labels=VESSEL_NODES, title="Fitted vascular network"
                    )
                    note = (
                        "Edge width is the coefficient's magnitude, colour its sign, dotted means "
                        "the interval covers zero. Hover an edge for the exact value."
                    )
                else:
                    fig = forest_plot(
                        regressions, label="parameter", title="Path coefficients",
                        x_label="Path coefficient",
                    )
                    note = (
                        "Standardized paths, strongest first. Blue = interval excludes zero."
                        if (self._last_fit_meta or {}).get("standardized")
                        else "Unstandardized paths — magnitudes are not comparable between edges."
                    )
            self._plot.show_figure(fig)
            self._plot.set_status(note)
        except Exception as exc:
            log.debug("SEM plot failed: %s", exc, exc_info=True)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _plot_mrf(self) -> None:
        """The smoothed field over the vessels, or the adjacency graph it was smoothed on."""
        try:
            import matplotlib.pyplot as plt

            meta = self._last_fit_meta or {}
            kind = str(self._mediation_plot.currentData() or "field")
            field = mrf_field_frame(self._last_result, self._last_model_df, meta)
            with plt.style.context("default"):
                if kind == "graph":
                    fig = plot_mrf_graph(
                        field, meta.get("neighbours", {}), node_labels=VESSEL_NODES
                    )
                    note = (
                        "Node colour is the fitted effect. A sharp jump between two adjacent "
                        "vessels is where the data overruled the smoothing penalty."
                    )
                else:
                    labelled = field.assign(
                        vessel=[VESSEL_NODES.get(str(v), str(v)) for v in field["level"]]
                    )
                    fig = forest_plot(
                        labelled, label="vessel", estimate="effect",
                        title="Smoothed field over the vessel graph",
                        x_label="Deviation from the overall level",
                    )
                    note = (
                        f"Each vessel's departure from the overall level, shrunk toward its "
                        f"neighbours. {len(meta.get('levels', []))} vessels, "
                        f"{len(meta.get('isolated') or [])} isolated."
                    )
            self._plot.show_figure(fig)
            self._plot.set_status(note)
        except Exception as exc:
            log.debug("MRF plot failed: %s", exc, exc_info=True)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _plot_nonlinear(self) -> None:
        """Scatter plus the fitted curve, coloured by the grouping column when it is present."""
        try:
            import matplotlib.pyplot as plt

            result = self._last_result
            data = self._working_df
            group = self._groups.text().strip()
            if data is None or group not in (data.columns if data is not None else []):
                group = None
            display = str(self._plot_display.currentData() or "overview")
            display_note = ""
            args = (result, data if data is not None else self._last_model_df)
            kwargs = dict(
                errorbar=self._show_ci.isChecked(),
                include_points=self._include_points.isChecked(),
                group=group,
            )
            with plt.style.context("default"):
                try:
                    fig = plot_nonlinear_fit(*args, display=display, **kwargs)
                except ValueError as exc:
                    if display != "grouped":
                        raise
                    display_note = f"⚠ Grouped display unavailable — {exc}"
                    fig = plot_nonlinear_fit(*args, display="overview", **kwargs)
            self._plot.show_figure(fig)
            notes = []
            if display_note:
                notes.append(display_note)
            elif display == "grouped":
                notes.append(self._panel_note(fig))
            if group:
                notes.append(
                    "One curve is fitted over all rows; the grouping column only colours the points."
                )
            self._plot.set_status("  ".join(notes))
        except Exception as exc:
            log.debug("Non-linear plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Mediation
    # ──────────────────────────────────────────────────────────────────────────
    def _on_run_mediation(self, spec: MediationSpec) -> None:
        """Start a mediation run on the working frame."""
        if self._working_df is None or self._working_df.empty:
            self._mediation_form.set_error("Reload the data before running a mediation.")
            return
        problem = spec.validate(self._working_df)
        if problem:
            self._mediation_form.set_error(problem)
            return
        if self._mediation_worker is not None and self._mediation_worker.isRunning():
            self._mediation_form.set_error("A mediation run is already in progress.")
            return

        self._mediation_form.set_error("")
        self._mediation_form.set_running(True)
        self._report.set_message("Running mediation…")
        self._status.setText(f"Running {spec.engine} mediation ({spec.n_boot} draws)…")

        worker = MediationWorker(self._working_df, spec)
        worker.progress.connect(self._mediation_form.set_progress)
        worker.finished_ok.connect(self._on_mediation_done)
        worker.failed.connect(self._on_mediation_failed)
        worker.finished.connect(lambda: self._mediation_form.set_running(False))
        self._mediation_worker = worker
        worker.start()

    def _on_cancel_mediation(self) -> None:
        """Ask the running mediation to stop."""
        if self._mediation_worker is not None and self._mediation_worker.isRunning():
            self._mediation_worker.cancel()
            self._status.setText("Cancelling mediation — finishing the current draw…")

    def _on_mediation_done(self, bundle: dict[str, Any]) -> None:
        """Show the mediation report and its default plot."""
        self._mediation_bundle = bundle
        self._last_result = None
        self._report.set_mediation(bundle, raw_text=render_mediation_info(bundle))
        self._sync_mediation_plot_choices(bundle)
        self._plot_mediation()
        spec: MediationSpec = bundle["spec"]
        self._status.setText(
            f"Mediation complete ({spec.engine})  |  {spec.x} → {spec.m} → {spec.y}"
            + (f"  |  {bundle['note']}" if bundle.get("note") else "")
        )
        notify("Mediation analysis complete.")

    def _on_mediation_failed(self, message: str) -> None:
        """Report a failed mediation run."""
        self._mediation_form.set_error(message)
        self._report.set_message(f"Mediation failed: {message}")
        self._status.setText(f"Mediation failed: {message}")
        notify(f"Mediation failed: {message}", error=True)

    def _set_plot_kinds(self, choices: Sequence[tuple[str, str]]) -> None:
        """Repopulate the figure picker, preserving the current pick when it survives."""
        current = str(self._mediation_plot.currentData() or "")
        self._mediation_plot.blockSignals(True)
        self._mediation_plot.clear()
        for label, key in choices:
            self._mediation_plot.addItem(label, key)
        idx = self._mediation_plot.findData(current)
        self._mediation_plot.setCurrentIndex(idx if idx >= 0 else 0)
        self._mediation_plot.blockSignals(False)

    def _sync_mediation_plot_choices(self, bundle: dict[str, Any]) -> None:
        """Offer only the figures this engine's result can actually produce."""
        available = {"forest", "partial"}
        if isinstance(bundle.get("raw"), dict) and "dist" in bundle["raw"]:
            available.add("bootstrap")
        summary = bundle.get("summary")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            available.add("by_level")

        current = str(self._mediation_plot.currentData() or "forest")
        self._mediation_plot.blockSignals(True)
        self._mediation_plot.clear()
        for label, key in _MEDIATION_PLOTS:
            if key in available:
                self._mediation_plot.addItem(label, key)
        idx = self._mediation_plot.findData(current)
        self._mediation_plot.setCurrentIndex(idx if idx >= 0 else 0)
        self._mediation_plot.blockSignals(False)

    def _plot_mediation(self) -> None:
        """Draw the selected mediation figure."""
        bundle = self._mediation_bundle
        if bundle is None:
            return
        spec: MediationSpec = bundle["spec"]
        kind = str(self._mediation_plot.currentData() or "forest")
        try:
            import matplotlib.pyplot as plt

            with plt.style.context("default"):
                if kind == "bootstrap":
                    raw = bundle["raw"]
                    summary = raw["bootstrap"]["indirect"]
                    fig = plot_indirect_bootstrap(
                        raw["dist"]["indirect"],
                        ci_low=summary["ci_low"],
                        ci_high=summary["ci_high"],
                    )
                elif kind == "by_level":
                    fig = plot_indirect_by_level(bundle["summary"])
                elif kind == "partial":
                    fig = plot_partial_paths_mediation(
                        self._working_df,
                        x=spec.x,
                        m=spec.m,
                        y=spec.y,
                        covars=spec.covariates,
                    )
                else:
                    fig = plot_mediation_forest(
                        bundle["paths"],
                        title=f"Mediation: {spec.x} → {spec.m} → {spec.y}",
                    )
            self._plot.show_figure(fig)
            self._plot.set_status(str(bundle.get("note") or ""))
        except Exception as exc:
            log.debug("Mediation plot failed: %s", exc)
            self._plot.show_error(f"Plot unavailable: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Config persistence
    # ──────────────────────────────────────────────────────────────────────────
    def _config_dict(self) -> dict[str, Any]:
        """Serialize every panel setting to a plain dict (saved alongside a fitted model)."""
        return {
            "version": CONFIG_VERSION,
            "measurements": [spec.to_dict() for spec in self._measurements.specs()],
            "join": self._measurements.join(),
            "clinical": self._clinical_vars(),
            "cognitive": self._cognitive_vars(),
            "filters": [rule.to_dict() for rule in self._chips.rules()],
            "derived": [column.to_dict() for column in self._derived],
            "combinations": [c.to_dict() for c in self._combinations],
            "analysis_type": self._analysis_kind(),
            "mm_formula": self._formula.toPlainText().strip(),
            "groups": self._groups.text().strip(),
            "re_formula": self._re_formula.text().strip(),
            "vc_formula": self._vc_formula.text().strip(),
            "model_name": self._model_name.text().strip(),
            "glm_family": str(self._glm_family.currentData() or "gaussian"),
            "glm_link": str(self._glm_link.currentData() or ""),
            "robust": self._robust.currentData(),
            "lme4_family": str(self._lme4_family.currentData() or "gaussian"),
            "lme4_reml": self._lme4_reml.isChecked(),
            "mmrm": {"method": str(self._mmrm_method.currentData() or "Satterthwaite")},
            "nonlinear": {
                "model": str(self._nl_model.currentData() or ""),
                "x": self._nl_x.currentText(),
                "y": self._nl_y.currentText(),
                "p0": self._nl_p0.text().strip(),
            },
            "mediation": self._mediation_form.spec().to_dict(),
            "pipeline_id": QVTPY_PIPELINE_ID,
            "plot_display": str(self._plot_display.currentData() or "overview"),
            "plot_mode": str(self._plot_mode.currentData() or "auto"),
            "plot_x": self._plot_x.currentText().strip(),
            "include_points": self._include_points.isChecked(),
            "show_ci": self._show_ci.isChecked(),
            "show_legend": self._plot.show_legend(),
            "plot_groups": self._plot.checked_levels(),
            "splitters": {
                "main": self._main_split.sizes(),
                "top": self._top_split.sizes(),
                "output": self._output_split.sizes(),
                "bottom": self._bottom_split.sizes(),
            },
        }

    def _apply_config(self, cfg: dict[str, Any], *, allow_expressions: bool = True) -> None:
        """
        Restore the panel from a saved config, migrating older schema versions first.

        Parameters
        ----------
        allow_expressions : bool
            When ``False``, expression-kind derived columns are dropped instead of restored.
            Expressions are evaluated code, and a config file can come from anywhere the user
            browsed to — :meth:`_on_load` decides whether the source is trusted.
        """
        cfg = _migrate_config(cfg)

        specs = [MeasurementSpec.from_dict(entry) for entry in cfg.get("measurements") or []]
        if specs:
            self._measurements.set_specs(specs)
            self._data_form.apply_spec(specs[0])
        if cfg.get("join"):
            self._measurements.set_join(str(cfg["join"]))

        clinical = cfg.get("clinical")
        if isinstance(clinical, str):
            clinical = [c.strip() for c in clinical.split(",") if c.strip()]
        if isinstance(clinical, list):
            set_checked_variable_ids(self._clinical_list, clinical)
        if isinstance(cfg.get("cognitive"), list):
            set_checked_variable_ids(self._cognitive_list, cfg["cognitive"])

        self._chips.set_rules([FilterRule.from_dict(r) for r in cfg.get("filters") or []], [])
        self._combinations = [
            RegionCombination.from_dict(c) for c in cfg.get("combinations") or []
        ]
        derived = [DerivedColumn.from_dict(d) for d in cfg.get("derived") or []]
        if not allow_expressions:
            dropped = [d.name for d in derived if d.kind == "expression"]
            derived = [d for d in derived if d.kind != "expression"]
            if dropped:
                notify(f"Skipped untrusted expression column(s): {', '.join(dropped)}", error=True)
        self._derived = derived

        for key, widget in (
            ("mm_formula", self._formula),
            ("groups", self._groups),
            ("re_formula", self._re_formula),
            ("vc_formula", self._vc_formula),
            ("model_name", self._model_name),
        ):
            if key in cfg:
                if isinstance(widget, QPlainTextEdit):
                    widget.setPlainText(str(cfg[key]))
                else:
                    widget.setText(str(cfg[key]))

        family = str(cfg.get("glm_family") or "")
        if family:
            fidx = self._glm_family.findData(family)
            if fidx >= 0:
                self._glm_family.setCurrentIndex(fidx)
                self._on_glm_family_changed()
        link = str(cfg.get("glm_link") or "")
        if link:
            lidx = self._glm_link.findData(link)
            if lidx >= 0:
                self._glm_link.setCurrentIndex(lidx)
        ridx = self._robust.findData(cfg.get("robust"))
        if ridx >= 0:
            self._robust.setCurrentIndex(ridx)
        lidx = self._lme4_family.findData(str(cfg.get("lme4_family") or ""))
        if lidx >= 0:
            self._lme4_family.setCurrentIndex(lidx)
        if "lme4_reml" in cfg:
            self._lme4_reml.setChecked(bool(cfg["lme4_reml"]))
        mmrm_cfg = cfg.get("mmrm")
        if isinstance(mmrm_cfg, dict):
            midx = self._mmrm_method.findData(str(mmrm_cfg.get("method") or ""))
            if midx >= 0:
                self._mmrm_method.setCurrentIndex(midx)

        nonlinear = cfg.get("nonlinear")
        if isinstance(nonlinear, dict):
            midx = self._nl_model.findData(str(nonlinear.get("model") or ""))
            if midx >= 0:
                self._nl_model.setCurrentIndex(midx)
            for combo, key in ((self._nl_x, "x"), (self._nl_y, "y")):
                pos = combo.findText(str(nonlinear.get(key) or ""))
                if pos >= 0:
                    combo.setCurrentIndex(pos)
            self._nl_p0.setText(str(nonlinear.get("p0") or ""))

        analysis = str(cfg.get("analysis_type") or ANALYSIS_MIXEDLM)
        aidx = self._analysis_type.findData(analysis)
        if aidx >= 0:
            self._analysis_type.setCurrentIndex(aidx)
        if isinstance(cfg.get("mediation"), dict):
            self._mediation_form.apply_spec(MediationSpec.from_dict(cfg["mediation"]))

        plot_display = str(cfg.get("plot_display") or "")
        if plot_display:
            didx = self._plot_display.findData(plot_display)
            if didx >= 0:
                self._plot_display.setCurrentIndex(didx)
        plot_mode = str(cfg.get("plot_mode") or "")
        if plot_mode:
            midx = self._plot_mode.findData(plot_mode)
            if midx >= 0:
                self._plot_mode.setCurrentIndex(midx)
        if cfg.get("plot_x"):
            self._plot_x.setCurrentText(str(cfg["plot_x"]))
        if "include_points" in cfg:
            self._include_points.setChecked(bool(cfg["include_points"]))
        if "show_ci" in cfg:
            self._show_ci.setChecked(bool(cfg["show_ci"]))
        if "show_legend" in cfg:
            self._plot.set_show_legend(bool(cfg["show_legend"]))
        self._pending_plot_groups = cfg.get("plot_groups")

        sizes = cfg.get("splitters") or {}
        for key, splitter in (
            ("main", self._main_split),
            ("top", self._top_split),
            ("output", self._output_split),
            ("bottom", self._bottom_split),
        ):
            if isinstance(sizes.get(key), list) and sizes[key]:
                splitter.setSizes([int(v) for v in sizes[key]])

    def _on_save(self) -> None:
        """
        Save the fitted model, its configuration and its report under ``nvitk-statmodels/<name>/``.

        The engines serialize very differently — a statsmodels result pickles, an lme4 or MMRM fit
        lives in R, a non-linear fit is a dict holding a closure — so the model artifact is written
        best-effort and *last*. The configuration, the report and the coefficient table are written
        regardless: losing the record of a fit because its object would not pickle is the worse
        outcome, and that is exactly what used to happen.
        """
        if self._last_result is None and self._mediation_bundle is None:
            notify("Fit a model or run a mediation before saving.", error=True)
            return

        name = self._model_name.text().strip() or "model"
        out_dir = statmodels_root(self._repo) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        problems: list[str] = []

        def attempt(label: str, action) -> None:
            """Run one save step, recording success or the reason it failed."""
            try:
                action()
            except Exception as exc:
                problems.append(f"{label}: {exc}")
                log.debug("Save step %r failed", label, exc_info=True)
            else:
                written.append(label)

        attempt("config.json", lambda: (out_dir / "config.json").write_text(
            json.dumps(self._config_dict(), indent=2), encoding="utf-8"))

        # The figure as it currently looks, so the saved model carries the picture that was being
        # looked at when it was saved. Skipped rather than reported when nothing is displayed —
        # a config-only save is a legitimate thing to do.
        if self._plot.has_figure():
            attempt("plot.png", lambda: self._plot.save_figure(out_dir / "plot.png"))

        if self._last_result is not None:
            info = model_info_dict(
                self._last_result,
                outcome_name=self._last_outcome or self._primary_column(),
                group_name=self._groups.text().strip() or "group_key",
                meta=self._last_fit_meta or {},
            )
            attempt("info.txt", lambda: (out_dir / "info.txt").write_text(
                render_mixedlm_info(info), encoding="utf-8"))
            # Engine-independent and the thing most likely to be wanted later.
            attempt("coefficients.csv", lambda: info["fixed_effects"].to_csv(
                out_dir / "coefficients.csv", index=False))
            random_effects = info.get("random_effects")
            if isinstance(random_effects, pd.DataFrame) and not random_effects.empty:
                attempt("random_effects.csv", lambda: random_effects.to_csv(
                    out_dir / "random_effects.csv", index=False))
            group_effects = info.get("group_effects")
            if isinstance(group_effects, pd.DataFrame) and not group_effects.empty:
                attempt("group_coefficients.csv", lambda: group_effects.to_csv(
                    out_dir / "group_coefficients.csv", index=False))
            try:
                written.append(self._save_model_artifact(out_dir))
            except Exception as exc:
                problems.append(f"model object: {exc}")
                log.debug("Could not serialize the model object", exc_info=True)

        if self._mediation_bundle is not None:
            attempt("mediation.txt", lambda: (out_dir / "mediation.txt").write_text(
                render_mediation_info(self._mediation_bundle), encoding="utf-8"))
            attempt("mediation_paths.csv", lambda: self._mediation_bundle["paths"].to_csv(
                out_dir / "mediation_paths.csv", index=False))
            summary = self._mediation_bundle.get("summary")
            if isinstance(summary, pd.DataFrame) and not summary.empty:
                attempt("mediation_by_level.csv", lambda: summary.to_csv(
                    out_dir / "mediation_by_level.csv", index=False))

        written = [w for w in written if w]
        if problems:
            for problem in problems:
                log.warning("Save: %s", problem)
            notify(f"Saved {len(written)} file(s) to {out_dir}; {len(problems)} step(s) failed.",
                   error=True)
            self._status.setText(
                f"Saved {', '.join(written)} → {out_dir}   |   not saved: "
                + "; ".join(problems)
            )
            return
        notify(f"Saved → {out_dir}")
        self._status.setText(f"Saved {', '.join(written)} → {out_dir}")

    def _sync_display_enabled(self) -> None:
        """
        Grey the Display picker out for figures that are not a family of curves.

        MMRM offers both least-squares means, which panel, and a correlation heatmap, which is a
        single matrix with nothing to split — so the picker follows the figure choice there.
        """
        applicable = True
        if self._analysis_kind() == ANALYSIS_MMRM:
            applicable = str(self._mediation_plot.currentData() or "emmeans") != "correlation"
        self._plot_display.setEnabled(applicable)
        self._plot_display.setToolTip(
            self._display_tooltip
            if applicable
            else "The correlation heatmap is a single matrix — there is nothing to split into panels."
        )

    def _on_export_plot(self) -> None:
        """Prompt for a location and write the displayed figure there as a PNG."""
        if not self._plot.has_figure():
            notify("There is no plot to export — fit a model first.", error=True)
            return

        name = self._model_name.text().strip() or "model"
        suggested = statmodels_root(self._repo) / name / f"{name}_plot.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot as PNG", str(suggested), "PNG image (*.png);;All (*)"
        )
        if not path:
            return
        # A user who typed a bare name still means a PNG; savefig would otherwise infer the format
        # from an extension that is not there and fail.
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")

        try:
            written = self._plot.save_figure(target)
        except Exception as exc:
            log.debug("Plot export failed: %s", exc, exc_info=True)
            notify(f"Export failed: {exc}", error=True)
            return
        notify(f"Plot exported → {written}")
        self._status.setText(f"Plot exported → {written}")

    def _save_model_artifact(self, out_dir: Path) -> str:
        """
        Serialize the fitted model object itself, however this engine allows.

        statsmodels results pickle. An lme4 or MMRM fit is an R object, so it goes to ``.rds`` —
        readable back in R with ``readRDS``, which is more useful than a Python pickle would be. A
        non-linear fit holds a closure, so its parameters are written instead of the object.

        Returns the file name written, and raises if the engine offers no route at all.
        """
        result = self._last_result
        engine = str((self._last_fit_meta or {}).get("engine") or "")

        if hasattr(result, "save"):  # statsmodels: MixedLM / OLS / GLM
            result.save(str(out_dir / "model.pkl"))
            return "model.pkl"

        if engine == ANALYSIS_NONLINEAR and isinstance(result, dict):
            # ``predict`` is a closure and ``spec`` holds the model function; neither pickles, and
            # neither is needed to reconstruct the fit from its parameters.
            result["params"].to_csv(out_dir / "nonlinear_parameters.csv", index=False)
            return "nonlinear_parameters.csv"

        r_object = getattr(result, "r_model", None) or (
            result if engine in {ANALYSIS_MMRM, ANALYSIS_LMROB} else None
        )
        if r_object is not None:
            from rpy2.robjects import r as R_

            R_["saveRDS"](r_object, str(out_dir / "model.rds"))
            return "model.rds"

        raise TypeError(
            f"a {type(result).__name__} cannot be serialized by this engine — "
            "the configuration and report were still written, so the fit is reproducible."
        )

    def _on_load(self) -> None:
        """Prompt for a saved model directory and restore its config, model and summary."""
        start = str(statmodels_root(self._repo))
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
                self._apply_config(
                    cfg, allow_expressions=self._trusts_expressions(model_dir, cfg)
                )
            if pkl.is_file():
                from statsmodels.regression.mixed_linear_model import MixedLMResults

                self._last_result = MixedLMResults.load(str(pkl))
                self._last_fit_meta = {"engine": ANALYSIS_MIXEDLM}
                self._mediation_bundle = None
                info = mixedlm_info_dict(
                    self._last_result, group_name=self._groups.text().strip() or "group_key"
                )
                self._report.set_mixedlm(info, raw_text=render_mixedlm_info(info))
            elif (model_dir / "info.txt").is_file():
                # An R or non-linear fit was saved as .rds / .csv rather than a Python object:
                # its settings are restored and the report is shown, but reproducing the fitted
                # object means re-running it.
                self._report.set_message(
                    (model_dir / "info.txt").read_text(encoding="utf-8")
                    + "\n\nThis engine's model object is not restorable in Python — the settings "
                    "have been loaded, so press Reload data and Fit to reproduce it."
                )
            notify(f"Loaded model from {model_dir}")
            self._status.setText(f"Loaded {model_dir} — press Reload data to rebuild the frame.")
        except Exception as exc:
            notify(f"Load failed: {exc}", error=True)
            QMessageBox.critical(self, "Load failed", str(exc))

    def _trusts_expressions(self, model_dir: Path, cfg: dict[str, Any]) -> bool:
        """
        Decide whether to evaluate expression-kind derived columns from a loaded config.

        Expressions are code. A config saved by this tool under the dataset's own
        ``nvitk-statmodels/`` directory is the user's own work; one browsed to from anywhere else
        might not be, so ask before evaluating it.
        """
        expressions = [
            d for d in (cfg.get("derived") or []) if str(d.get("kind")) == "expression"
        ]
        if not expressions:
            return True
        try:
            model_dir.resolve().relative_to(statmodels_root(self._repo).resolve())
            return True
        except ValueError:
            pass
        listing = "\n".join(f"  {d.get('name')} = {d.get('expression')}" for d in expressions)
        answer = QMessageBox.question(
            self,
            "Evaluate derived expressions?",
            f"This config comes from outside the dataset's nvitk-statmodels directory and defines "
            f"{len(expressions)} derived column(s) as expressions, which will be evaluated:\n\n"
            f"{listing}\n\nEvaluate them?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes


# ──────────────────────────────────────────────────────────────────────────────
# Config migration
# ──────────────────────────────────────────────────────────────────────────────
def _migrate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade a saved config to :data:`CONFIG_VERSION`.

    v1 held a single measurement as flat ``pipeline_kind`` / ``pipeline`` / ``feature`` /
    ``grouping`` / ``atlas`` keys and expressed its only filter as ``iqr_enabled`` / ``iqr_k`` /
    ``iqr_scope``. Both become their v2 equivalents; every other key already carries over, and each
    read is ``cfg.get``-guarded so a newer file degrades to best-effort rather than failing.
    """
    version = int(cfg.get("version") or 1)
    if version >= CONFIG_VERSION:
        if version > CONFIG_VERSION:
            log.warning(
                "Statmodels config version %d is newer than this build (%d); applying best-effort.",
                version,
                CONFIG_VERSION,
            )
        return cfg

    out = dict(cfg)
    out["version"] = CONFIG_VERSION

    if not out.get("measurements"):
        spec = MeasurementSpec(
            pipeline_kind=str(cfg.get("pipeline_kind") or PIPELINE_KIND_QVTPY),
            pipeline=str(cfg.get("pipeline") or "latest"),
            feature=str(cfg.get("feature") or "flow_mean"),
            grouping=str(cfg.get("grouping") or "vessel"),
            atlas=(str(cfg["atlas"]) if cfg.get("atlas") else None),
        )
        out["measurements"] = [spec.to_dict()]
        primary_column = spec.column()
    else:
        primary_column = MeasurementSpec.from_dict(out["measurements"][0]).column()

    if not out.get("filters") and cfg.get("iqr_enabled"):
        scope = str(cfg.get("iqr_scope") or "group")
        out["filters"] = [
            FilterRule(
                column=primary_column,
                kind="iqr",
                k=float(cfg.get("iqr_k", 1.5)),
                by="group_key" if scope == "group" else None,
            ).to_dict()
        ]
    out.setdefault("join", "inner")
    out.setdefault("derived", [])
    out.setdefault("analysis_type", ANALYSIS_MIXEDLM)
    return out


__all__ = ["StatmodelsWindow"]
