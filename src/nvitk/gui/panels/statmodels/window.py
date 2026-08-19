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
    QInputDialog,
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
    nonlinear_geometry,
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
    SEM_ESTIMATORS,
    SemSpec,
    fit_sem,
    path_effects,
    plot_sem_network,
    plot_sem_paths,
    resolve_network_syntax,
    sem_backend_status,
    sem_paths_frame,
)
from nvitk.stats.vessel_network import (
    VESSEL_NODES,
    canonical_node,
    network_frame,
    sem_model_syntax,
)
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
    plot_mmrm_emmeans,
    plot_lmrob_weights,
    plot_mmrm_correlation,
    mmrm_correlation_matrix,
    plot_mixedlm_params,
    plot_nonlinear_fit,
    r_backend_status,
    validate_mmrm_data,
    render_mixedlm_info,
    subject_attribute_entries,
    subject_image_annotation_entries,
)
from nvitk.stats.frame_ops import (
    COLUMN_TYPES,
    DerivedColumn,
    FilterRule,
    apply_column_types,
    apply_derived_columns,
    apply_reference_levels,
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

from nvitk.gui.core.flow_layout import FlowRow
from nvitk.gui.core.geometry import cap_minimum_size, fit_to_screen
from nvitk.stats.brain_map import BRAIN_ATLASES, BRAIN_SURFACES, BRAIN_VIEWS

from .constants import (
    MAX_CATEGORICAL_LEVELS,
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
from .subject_plot_dialog import SubjectPlotDialog
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
    ("Factor loadings", "loadings"),
    ("Modification indices", "modindices"),
)

#: Sub-views of the vascular map: what colour encodes. Kept separate from the estimate/p-value
#: split because "which vessel carries the most flow" and "where is the evidence" are different
#: questions and a single control conflating them would make the figure ambiguous to read.
#: Colormaps offered for the vascular map. Diverging ones are listed first because the default
#: view is an effect, where a meaningful midpoint matters more than perceptual uniformity.
_VASCULAR_COLORMAPS: tuple[tuple[str, str], ...] = (
    ("auto (diverging for effects)", ""),
    ("RdBu (diverging)", "RdBu_r"),
    ("coolwarm (diverging)", "coolwarm"),
    ("PuOr (diverging)", "PuOr_r"),
    ("viridis", "viridis"),
    ("magma", "magma"),
    ("cividis", "cividis"),
    ("plasma", "plasma"),
)

_VASCULAR_PLOTS: tuple[tuple[str, str], ...] = (
    # First, because it is the only coefficient view that gives *every* vessel a value: treatment
    # coding leaves the reference territory without a term, and a map with one artery missing reads
    # as a measurement failure rather than as a contrast baseline.
    ("Marginal mean per vessel", "emmeans"),
    ("Model estimate (grey = n.s.)", "estimate"),
    ("Model estimate (all vessels)", "estimate_all"),
    ("p-value (significant only)", "pvalue"),
    ("p-value (all vessels)", "pvalue_all"),
    ("Observed mean per vessel", "means"),
)

#: Sub-views of the brain map. Deliberately the same keys as :data:`_VASCULAR_PLOTS` so the two
#: displays are interchangeable from the window's point of view — switching Display from one to the
#: other keeps the view you were looking at instead of resetting the picker.
_BRAIN_PLOTS: tuple[tuple[str, str], ...] = (
    ("Marginal mean per parcel", "emmeans"),
    ("Model estimate (grey = n.s.)", "estimate"),
    ("Model estimate (all parcels)", "estimate_all"),
    ("p-value (significant only)", "pvalue"),
    ("p-value (all parcels)", "pvalue_all"),
    ("Observed mean per parcel", "means"),
)

#: Display keys that draw a *map of anatomy* rather than a family of curves. They share every
#: control the plot pane gates on — the figure picker, the group checklist, the colormap — so the
#: gating asks "is this a map" once instead of naming each one at every branch.
_MAP_DISPLAYS: frozenset[str] = frozenset({"vascular", "brain"})

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
        # Preferred size, clamped to whatever screen this actually opens on — 1700×1000 runs
        # off a laptop display, taking the plot pane's right edge with it.
        fit_to_screen(self, 1700, 1000)
        apply_dark_theme(self)

        self._repo = open_repo()

        # ---- analysis state ---------------------------------------------------
        self._analysis_df: pd.DataFrame | None = None   # as loaded, never mutated
        self._working_df: pd.DataFrame | None = None    # + derived columns, − filtered rows
        self._load_meta: dict[str, Any] | None = None
        self._derived: list[DerivedColumn] = []
        self._combinations: list[RegionCombination] = []
        self._filter_report: list[dict[str, Any]] = []
        # Wide = one row per subject, one column per vessel. Applied *after* derived columns and
        # filters, so a vessel-wise QC filter still gets to act on the rows it was written for.
        self._wide_mode: bool = False
        self._wide_coverage: pd.DataFrame | None = None
        self._wide_complete: int = 0
        # Measurement family melted back to long after the wide pivot, or "".
        self._melt_family: str = ""
        # Columns hidden from the working frame. Held as names rather than by dropping them from
        # ``_analysis_df``, so a reload or a restore brings them back without re-querying.
        self._dropped_columns: set[str] = set()
        # Per-column dtype overrides. A column's type decides whether a model term is a slope or a
        # set of contrasts, so it is part of the frame recipe rather than a display preference.
        self._column_types: dict[str, str] = {}
        # Treatment reference per factor. Both patsy and R contrast against a factor's first level,
        # so this is applied by reordering categories rather than by rewriting the formula.
        self._reference_levels: dict[str, str] = {}

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
        self._plot.set_map_options_widget(self._build_map_options())
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

        self._sem_estimator = QComboBox()
        for key, description in SEM_ESTIMATORS.items():
            self._sem_estimator.addItem(f"{key} — {description.split('—')[-1].strip()}", key)
        self._sem_estimator.setToolTip(
            "How the model is estimated.\n\n"
            + "\n".join(f"{k}: {v}" for k, v in SEM_ESTIMATORS.items())
            + "\n\nMLR is the safe choice when the residuals are skewed — the estimates are the "
            "same as ML, only the standard errors change. semopy accepts ML, GLS and WLS; it has "
            "no MLR and falls back to ML."
        )
        self._sem_estimator_label = QLabel("Estimator")

        self._sem_missing = QComboBox()
        self._sem_missing.addItem("listwise — drop incomplete rows", "listwise")
        self._sem_missing.addItem("FIML — use every observed value", "fiml")
        self._sem_missing.setToolTip(
            "What to do with missing values.\n\n"
            "Listwise deletion keeps only subjects measured on *every* modelled variable, so the "
            "sparsest term decides the sample size — a model with one 40%-covered vessel is fitted "
            "on 40% of the cohort.\n"
            "FIML uses each subject's observed values instead and is the better default whenever "
            "the missingness is unrelated to what is missing."
        )
        self._sem_missing_label = QLabel("Missing data")

        self._sem_group = QComboBox()
        self._sem_group.setToolTip(
            "Fit the same model separately in each level of this column, so the paths themselves "
            "are free to differ between groups.\n\n"
            "Different from adding the column as a covariate, which only shifts the means and "
            "assumes every path is shared. lavaan only — semopy has no multi-group support."
        )
        self._sem_group_label = QLabel("Multi-group")

        self._mrf_family = QComboBox()
        for key, description in GAM_FAMILIES.items():
            self._mrf_family.addItem(description, key)
        self._mrf_family_label = QLabel("Family")
        self._mrf_method = QComboBox()
        for key, description in GAM_METHODS.items():
            self._mrf_method.addItem(description, key)
        self._mrf_method_label = QLabel("Smoothing")

        form.addRow(self._sem_backend_label, self._sem_backend)
        form.addRow(self._sem_estimator_label, self._sem_estimator)
        form.addRow(self._sem_missing_label, self._sem_missing)
        form.addRow(self._sem_group_label, self._sem_group)
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
        # Flow, not horizontal: this row grew to a dozen-plus controls, and a QHBoxLayout's minimum
        # width is their sum — which Qt enforces as the *window's* minimum, locking it wider than
        # the screen. Wrapping keeps the minimum at the widest single control.
        widget = FlowRow()
        lay = widget.flow()

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
        self._interactive_plot.stateChanged.connect(
            lambda *_: (self._sync_analysis_type(), self._on_plot())
        )
        lay.addWidget(self._interactive_plot)

        self._plot_display = QComboBox()
        self._plot_display.addItem("Overview", "overview")
        self._plot_display.addItem("Grouped", "grouped")
        self._plot_display.addItem("Vascular map", "vascular")
        self._plot_display.addItem("Brain map", "brain")
        self._display_tooltip = (
            "Overview draws every group on one pair of axes.\n"
            "Grouped splits them into a grid of anatomical panels — carotids / anterior / "
            "posterior / venous for vessels, lobes for cortical parcels — each autoscaled to its "
            "own range, so a venous measurement no longer flattens an arterial one.\n"
            "Vascular map draws each vessel's estimate on a schematic of the circle of Willis and "
            "the dural sinuses, so a pattern across the anatomy — both carotids together, the "
            "posterior circulation alone — is visible instead of being spread down a forest plot.\n"
            "Brain map does the same for the parenchymal measurements — ASL perfusion, T1 "
            "volumetry — on the cortical surface, since those are parcellated by Desikan rather "
            "than by vessel. Grey means measured but not significant; an unpainted parcel means "
            "the model has no estimate for it, which is not the same thing.\n"
            "This is a view of the same fit: the population estimate is identical in every panel."
        )
        self._plot_display.setToolTip(self._display_tooltip)

        self._plot_mode = QComboBox()
        self._plot_mode.addItem("auto", "auto")
        self._plot_mode.addItem("continuous (scatter + regression)", "continuous")
        self._plot_mode.addItem("categorical (marginal means)", "categorical")
        self._plot_x = QComboBox()
        self._plot_x.setMinimumWidth(140)
        # Which factor the curves are coloured by. For lme4 this used to be the random grouping
        # factor and nothing else, which draws 510 subject curves for a model whose interesting
        # structure is a 17-level fixed effect.
        self._plot_group = QComboBox()
        self._plot_group.setMinimumWidth(120)
        self._plot_group.setToolTip(
            "Factor the curves are drawn per level of.\n\n"
            "Offers the model's random grouping factors and any categorical fixed effect. A "
            "categorical fixed effect is preferred by default: a per-subject curve set is rarely "
            "readable, and the fixed term is what the coefficient table is about."
        )
        self._plot_group_label = QLabel("colour by")

        # ---- Vascular-map-only controls ------------------------------------------
        self._vasc_cmap = QComboBox()
        for label, key in _VASCULAR_COLORMAPS:
            self._vasc_cmap.addItem(label, key)
        self._vasc_cmap.setToolTip(
            "Colormap for the vascular map.\n\n"
            "Diverging maps (RdBu, coolwarm) centre on zero and are the right choice for an effect "
            "— the midpoint is 'no change'. Sequential maps (viridis, magma) have no meaningful "
            "midpoint and suit a magnitude such as a mean or a p-value."
        )
        self._vasc_cmap.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._vasc_cmap_label = QLabel("cmap")

        self._vasc_contrast = QComboBox()
        self._vasc_contrast.setMinimumWidth(120)
        self._vasc_contrast.setToolTip(
            "Which side of an interaction to draw.\n\n"
            "With a term like 'territory * sex' the model estimates a different vessel profile per "
            "group. The main effect is the profile at the interacting factor's reference level; "
            "picking a contrast adds the interaction, giving that group's own profile.\n"
            "The p-value shown then belongs to the interaction — it tests whether the vessel's "
            "effect differs between groups."
        )
        self._vasc_contrast.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._vasc_contrast_label = QLabel("contrast")

        # ---- Brain-map-only controls ---------------------------------------------
        self._brain_atlas = QComboBox()
        # "auto" first and default: the right atlas is the one the measurement was made under, and
        # the measurement form already knows it. Making the user restate it invites a mismatch that
        # draws a blank brain and looks like a modelling failure.
        self._brain_atlas.addItem("auto (from the measurement)", "")
        for label, key in BRAIN_ATLASES:
            self._brain_atlas.addItem(label, key)
        self._brain_atlas.setToolTip(
            "Parcellation the values are painted on.\n\n"
            "Desikan is what the ASL cortical tables and the T1 volumetry are reported against. "
            "The vascular atlas is the arterial-territory / watershed parcellation ASL uses by "
            "default — coarser, and the one that makes a perfusion result comparable to a 4D-flow "
            "one.\n"
            "Pick the atlas the measurement was actually made under: a value has no meaning on a "
            "parcellation it was not averaged over."
        )
        self._brain_atlas.currentIndexChanged.connect(
            lambda *_: (self._sync_map_contrasts(self._map_display()), self._on_plot())
        )
        self._brain_atlas_label = QLabel("atlas")

        self._brain_hemi = QComboBox()
        for label, key in (("both hemispheres", "both"), ("left", "left"), ("right", "right")):
            self._brain_hemi.addItem(label, key)
        self._brain_hemi.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._brain_hemi_label = QLabel("hemi")

        self._brain_views = QComboBox()
        for label, key in BRAIN_VIEWS:
            self._brain_views.addItem(label, key)
        self._brain_views.setToolTip(
            "Which surface views to draw.\n\n"
            "Lateral alone hides the medial wall, which is where the cingulate, the precuneus and "
            "the medial orbitofrontal cortex live — a third of the Desikan parcels."
        )
        self._brain_views.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._brain_views_label = QLabel("views")

        self._brain_surface = QComboBox()
        for label, key in BRAIN_SURFACES:
            self._brain_surface.addItem(label, key)
        self._brain_surface.setToolTip(
            "Which fsaverage surface the parcels are painted on.\n\n"
            "Inflated exposes the parcels buried in sulci — about two thirds of the cortex — which "
            "a folded pial surface simply does not show.\n"
            "Pial is the real geometry and reads as a brain.\n"
            "Flat puts the whole cortex in one panel per hemisphere: no view to choose, no hidden "
            "parcels, and by far the fastest to draw."
        )
        # Also re-syncs visibility: a flat map has no view angle, so the views picker goes away.
        self._brain_surface.currentIndexChanged.connect(
            lambda *_: (self._sync_analysis_type(), self._on_plot())
        )
        self._brain_surface_label = QLabel("surface")

        self._brain_shading = QCheckBox("Shading")
        self._brain_shading.setChecked(True)
        self._brain_shading.setToolTip(
            "Shade the unpainted surface with its own curvature, so gyri and sulci are visible "
            "under the parcels.\nOff gives a flat silhouette — cleaner for a figure, and it makes "
            "the parcel boundaries the only structure on the page."
        )
        self._brain_shading.stateChanged.connect(
            lambda *_: (self._sync_analysis_type(), self._on_plot())
        )

        self._brain_threshold = QDoubleSpinBox()
        self._brain_threshold.setRange(0.0, 1e9)
        self._brain_threshold.setDecimals(3)
        self._brain_threshold.setSingleStep(0.1)
        self._brain_threshold.setSpecialValueText("off")
        self._brain_threshold.setValue(0.0)
        self._brain_threshold.setMaximumWidth(90)
        self._brain_threshold.setToolTip(
            "Leave parcels whose |value| is below this unpainted, the way a stat map is "
            "thresholded.\n\n"
            "Different from the significance mask: this is about effect *size*, not evidence. A "
            "parcel hidden here is counted in the caption so it is never read as one the model has "
            "no estimate for.\n0 turns it off."
        )
        self._brain_threshold.valueChanged.connect(lambda *_: self._on_plot())
        self._brain_threshold_label = QLabel("|min|")

        self._brain_blend = QCheckBox("Blend")
        self._brain_blend.setChecked(True)
        self._brain_blend.setToolTip(
            "Shade the *painted* parcels with the surface curvature too, not just the bare "
            "surface.\n\n"
            "The sulcal pattern shows through the colours, which keeps the folding legible where a "
            "parcel covers a whole gyrus. It darkens the colours unevenly, so a value read off the "
            "colourbar is no longer exact — a reading aid, not a setting to publish at.\n"
            "Needs Shading on: there is nothing to blend without a curvature map."
        )
        self._brain_blend.stateChanged.connect(lambda *_: self._on_plot())

        self._brain_opacity = QDoubleSpinBox()
        self._brain_opacity.setRange(0.05, 1.0)
        self._brain_opacity.setSingleStep(0.05)
        self._brain_opacity.setDecimals(2)
        self._brain_opacity.setValue(1.0)
        self._brain_opacity.setMaximumWidth(80)
        self._brain_opacity.setToolTip(
            "Opacity of the painted parcels.\n\n"
            "Below 1 the surface shows through — the other way to keep the folding visible under a "
            "dense parcellation. Like Blend it shifts the rendered colour away from the colourbar, "
            "so treat it as a reading aid."
        )
        self._brain_opacity.valueChanged.connect(lambda *_: self._on_plot())
        self._brain_opacity_label = QLabel("opacity")

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
        lay.addWidget(self._plot_group_label)
        lay.addWidget(self._plot_group)
        lay.addWidget(self._include_points)
        lay.addWidget(self._show_ci)
        return widget

    def _build_map_options(self) -> QWidget:
        """
        Second options row: the controls that belong to the anatomical maps.

        Split off the general row because there are now a dozen of them, and mixing the two meant
        the map controls reflowed into whatever gap the general ones left — different position every
        time the analysis type changed. Its own row keeps them together and hides them as a group.
        """
        widget = FlowRow()
        lay = widget.flow()
        lay.addWidget(self._vasc_cmap_label)
        lay.addWidget(self._vasc_cmap)
        lay.addWidget(self._vasc_contrast_label)
        lay.addWidget(self._vasc_contrast)
        lay.addWidget(self._brain_atlas_label)
        lay.addWidget(self._brain_atlas)
        lay.addWidget(self._brain_hemi_label)
        lay.addWidget(self._brain_hemi)
        lay.addWidget(self._brain_views_label)
        lay.addWidget(self._brain_views)
        lay.addWidget(self._brain_surface_label)
        lay.addWidget(self._brain_surface)
        lay.addWidget(self._brain_threshold_label)
        lay.addWidget(self._brain_threshold)
        lay.addWidget(self._brain_opacity_label)
        lay.addWidget(self._brain_opacity)
        lay.addWidget(self._brain_shading)
        lay.addWidget(self._brain_blend)
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
        self._btn_reshape = QPushButton("Reshape → wide")
        self._btn_reshape.setToolTip(
            "Switch between one row per subject × region (long) and one row per subject with a "
            "column per vessel (wide). Path models need the wide shape — an equation like "
            "'lmca ~ lica' compares two columns of the same row. Every other engine wants long."
        )
        self._btn_reshape.clicked.connect(self._on_reshape_frame)
        row.addWidget(self._btn_reshape)
        self._btn_melt = QPushButton("Melt by…")
        self._btn_melt.setToolTip(
            "Melt one measurement's per-region columns of a wide frame back into a 'territory' "
            "column, keeping every other column repeated down the rows.\n\n"
            "This is what lets a region be a model *term* in a cross-modality frame: melt "
            "flow_mean and you can write 'flow_mean ~ psqeduca * territory + t1_volume_mm3', with "
            "the eTIV and the covariates still there as predictors."
        )
        self._btn_melt.clicked.connect(self._on_melt_frame)
        self._btn_melt.setEnabled(False)
        row.addWidget(self._btn_melt)
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
        self._plot_display.currentIndexChanged.connect(lambda *_: self._sync_analysis_type())
        self._plot_display.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_mode.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_x.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_group.currentIndexChanged.connect(lambda *_: self._on_plot())
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
        self._frame_view.subjectPlotRequested.connect(self._on_subject_plot)
        self._frame_view.qcFilterRequested.connect(self._on_qc_filter)
        self._frame_view.summaryRequested.connect(self._on_export_summary)
        self._frame_view.dropRequested.connect(self._on_drop_column)
        self._frame_view.restoreRequested.connect(self._on_restore_columns)
        self._frame_view.typeChangeRequested.connect(self._on_change_column_type)
        self._frame_view.referenceRequested.connect(self._on_set_reference_level)
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
                # Manual anatomy annotations (cow_config / venous_config): image variables, but one
                # value per subject, so they belong with the covariates rather than the measurements.
                *subject_image_annotation_entries(self._repo),
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
            grain=self._measurements.grain(),
            attach_qc=self._measurements.attach_qc(),
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

        derived_frame, type_notes = apply_column_types(derived_frame, self._column_types)
        # References after casts: casting to Factor is what makes a column eligible for one, and a
        # cast rebuilds the categories, discarding any order set before it.
        derived_frame, ref_notes = apply_reference_levels(derived_frame, self._reference_levels)
        if type_notes or ref_notes:
            log.info("Column types: %s", " ".join([*type_notes, *ref_notes]))

        rules = self._chips.rules()
        working, report = apply_filter_rules(derived_frame, rules)

        # Reshape after the long-shape filters: derived columns and most rules are written against
        # long rows, so pivoting first would put their targets out of reach.
        long_rows = len(working)
        reshaped = False
        if self._wide_mode:
            working = self._to_wide(working)
            reshaped = True
        # Melt independently of the reshape button: the frame can already be wide because the
        # measurements were loaded on the subject grain, in which case the button was never pressed
        # and _wide_mode is False. The frame's own columns decide, not how it got that way.
        if self._melt_family:
            working = self._melt_working(working)
            reshaped = True

        if reshaped:
            # A rule can name a column that only exists *after* the reshape — an IQR fence on
            # 'flow_mean__TCBF', say. Those were skipped above with "column not in frame", so they
            # get a second pass here, in the shape they were written for. Rules already applied are
            # not re-run: apply_filter_rules skips what it cannot find, and a rule's column exists
            # in exactly one of the two shapes.
            deferred = [
                entry["rule"] for entry in report
                if entry.get("skipped") and entry.get("reason") == "column not in frame"
            ]
            if deferred:
                working, late = apply_filter_rules(working, deferred)
                by_rule = {id(entry["rule"]): entry for entry in late}
                report = [by_rule.get(id(entry["rule"]), entry) for entry in report]

            # Casts and reference levels get the same second pass, and for the same reason: a melt
            # *creates* the 'territory' column, so a reference set on it could not have been applied
            # on the first pass — it was dropped as "not a level of this column", and the model then
            # used whichever level sorted first. Re-applying is idempotent for columns already done.
            working, late_types = apply_column_types(working, self._column_types)
            working, late_refs = apply_reference_levels(working, self._reference_levels)
            if late_types or late_refs:
                log.info("Post-reshape recode: %s", " ".join([*late_types, *late_refs]))

        # Dropped columns go last of all, so a rule or a derived column defined on one still
        # evaluated above — dropping hides a column from the model, it does not undo the frame.
        drop = [c for c in self._dropped_columns if c in working.columns]
        if drop:
            working = working.drop(columns=drop)
        self._working_df = working
        self._filter_report = report

        self._chips.set_rules(rules, report)
        self._chips.set_counts(long_rows, len(derived_frame))
        self._sync_filter_toggle()
        self._sync_reshape_buttons(working)
        self._frame_view.set_dropped(self._dropped_columns)
        self._frame_view.set_column_types(self._column_types)
        self._frame_view.set_references(self._reference_levels)
        self._frame_view.set_frame(
            working,
            filtered_columns=filtered_columns(rules),
            derived_columns={d.name for d in self._derived},
        )
        self._sync_column_combos(working)
        self._sync_nonlinear_columns(working)
        self._mediation_form.set_columns(working)

        if announce:
            shape = f", reshaped to {len(working)} subject rows" if self._wide_mode else ""
            notify(f"Filters applied: {long_rows} of {len(derived_frame)} rows{shape}.")

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
        self._sync_group_combo(df)

    def _sync_group_combo(self, df: pd.DataFrame | None) -> None:
        """
        Repopulate the colour-by combo: random grouping factors, then categorical fixed effects.

        A column is offered when it has at least two levels and few enough to draw — a 510-level
        subject factor is still listed (it is the model's own grouping) but never preselected.
        """
        previous = self._plot_group.currentText()
        meta = self._last_fit_meta or {}
        factors = [str(g) for g in (meta.get("grouping_factors") or [])]
        groups_field = self._groups.text().strip()
        if groups_field and groups_field not in factors:
            factors.append(groups_field)

        candidates: list[str] = []
        if df is not None and not df.empty:
            for column in df.columns:
                name = str(column)
                if name in candidates:
                    continue
                series = df[name]
                if pd.api.types.is_float_dtype(series):
                    continue
                levels = series.dropna().nunique()
                if 1 < levels <= MAX_CATEGORICAL_LEVELS or name in factors:
                    candidates.append(name)

        ordered = [c for c in candidates if c not in factors] + [c for c in factors if c in candidates]
        self._plot_group.blockSignals(True)
        self._plot_group.clear()
        for name in ordered:
            self._plot_group.addItem(name)
        # Prefer what the user already had, then a vessel-like factor, then anything small.
        for candidate in (previous, self._region_column_name(), "territory", "group_key"):
            idx = self._plot_group.findText(candidate)
            if idx >= 0:
                self._plot_group.setCurrentIndex(idx)
                break
        self._plot_group.blockSignals(False)

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

        # Region combinations aggregate one measurement *across rows of a region column*, so they
        # need a frame that has one. A subject-grain load has none — every region is already its
        # own column — and offering those columns as if they were regions is what made the dialog
        # list 'flow_mean__LICA' while the table showed melted rows. Prefer the working frame when
        # it carries a usable region column, and fall back to the base otherwise.
        base = self._analysis_df
        working = self._working_df
        frame = base
        for candidate in (working, base):
            if candidate is None or candidate.empty:
                continue
            column = next(
                (c for c in ("territory", "group_key", "region_id") if c in candidate.columns), ""
            )
            if column and candidate[column].nunique() > 1:
                frame, region_column = candidate, column
                break
        else:
            region_column = "territory" if "territory" in base.columns else "group_key"

        if region_column not in frame.columns or frame[region_column].nunique() <= 1:
            notify(
                "This frame has no region column to combine across — it is one row per subject. "
                "Melt a measurement back to long first, or load on the territory grain.",
                error=True,
            )
            return

        dialog = RegionCombinationsDialog(
            self,
            frame=frame,
            combinations=self._combinations,
            region_column=region_column,
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

    def _to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot a long frame to one row per subject and one column per vessel.

        The cell value is the primary measurement; subject-level covariates ride along, so
        ``age_c`` and ``sex`` stay available to every structural equation. Per-vessel coverage is
        recorded in ``self._wide_coverage`` — with a dozen vessels in one model, listwise deletion
        is decided by whichever vessel is measured least often, and that is the number worth
        knowing before blaming the model.

        Returns the frame unchanged, with a warning, when it cannot be pivoted: a failed reshape
        should not empty the table the user is looking at.
        """
        if df is None or df.empty:
            return df

        measurements = self._measurements.columns()
        value_column = next((c for c in measurements if c in df.columns), "")
        region_column = self._region_column_name()
        if not value_column or region_column not in df.columns or "subject_uid" not in df.columns:
            notify(
                "Cannot reshape: this needs 'subject_uid', a region column and a measurement.",
                error=True,
            )
            return df

        try:
            wide = network_frame(
                df,
                value_column=value_column,
                region_column=region_column,
                subject_column="subject_uid",
            )
        except ValueError as exc:
            notify(f"Cannot reshape: {exc}", error=True)
            return df

        vessels = list(wide.attrs.get("vessels", []))
        n_subjects = len(wide)
        present = wide.loc[:, vessels].notna() if vessels else pd.DataFrame(index=wide.index)
        self._wide_coverage = pd.DataFrame({
            "vessel": vessels,
            "n_measured": [int(present[v].sum()) for v in vessels],
            "pct": [100.0 * float(present[v].mean()) for v in vessels],
        }).sort_values("n_measured") if vessels else None
        self._wide_complete = int(present.all(axis=1).sum()) if vessels else 0

        wide = wide.reset_index()
        wide.attrs["vessels"] = vessels
        wide.attrs["value_column"] = value_column
        log.info(
            "Wide frame: %d subject(s) × %d vessel(s); %d complete across every vessel.",
            n_subjects, len(vessels), self._wide_complete,
        )
        return wide

    def _sync_reshape_buttons(self, df: pd.DataFrame | None) -> None:
        """
        Enable the reshape controls from the *frame's* shape, not from which button was last pressed.

        A frame loaded on the subject grain is already wide without the reshape button having been
        touched, so gating on ``_wide_mode`` left Melt disabled on exactly the frames it exists for.
        """
        from nvitk.stats._statmodels_frames import subject_measurement_families

        families = subject_measurement_families(df) if df is not None and not df.empty else {}
        # A single-region measurement has no '__' columns and nothing to melt; a family needs at
        # least one region column to spread down the rows.
        self._btn_melt.setEnabled(bool(families) or bool(self._melt_family))
        self._btn_melt.setText(f"Melt: {self._melt_family}" if self._melt_family else "Melt by…")

        already_wide = bool(families) or (
            df is not None and "territory" not in df.columns and not self._melt_family
        )
        if already_wide and not self._wide_mode:
            self._btn_reshape.setEnabled(False)
            self._btn_reshape.setToolTip(
                "This frame is already one row per subject — it was loaded on the subject grain. "
                "Change the grain in the Measurements panel to load it long, or use 'Melt by…' to "
                "spread one measurement's regions back down the rows."
            )
        else:
            self._btn_reshape.setEnabled(True)
            self._btn_reshape.setToolTip(
                "Switch between one row per subject × region (long) and one row per subject with a "
                "column per vessel (wide). Path models need the wide shape — an equation like "
                "'lmca ~ lica' compares two columns of the same row. Every other engine wants long."
            )

    def _melt_working(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melt ``self._melt_family``'s per-region columns back to long, or return *df* unchanged."""
        from nvitk.stats._statmodels_frames import melt_subject_frame

        if df is None or df.empty or not self._melt_family:
            return df
        try:
            return melt_subject_frame(df, family=self._melt_family)
        except ValueError as exc:
            notify(f"Cannot melt: {exc}", error=True)
            self._melt_family = ""
            return df

    def _on_melt_frame(self) -> None:
        """Choose a measurement family to melt back into a territory column."""
        from nvitk.stats._statmodels_frames import subject_measurement_families

        # Offer the families of the *un-melted* frame: once one is melted its columns are gone, so
        # asking the current frame would hide the option that is already active. This deliberately
        # does not consult _wide_mode — a frame loaded on the subject grain is already wide.
        probe = self._melt_family
        self._melt_family = ""
        self._recompute_frame(announce=False)
        # Explicit None test: a DataFrame in a boolean context raises rather than being falsy.
        probe_df = self._working_df
        families = (
            subject_measurement_families(probe_df) if probe_df is not None else {}
        )
        self._melt_family = probe

        if not families:
            self._recompute_frame(announce=False)
            notify(
                "Nothing to melt: no column is named '<measurement>__<region>'. That naming comes "
                "from a subject-grain load or the reshape button, and only a measurement with more "
                "than one region produces it.",
                error=True,
            )
            return

        options = ["(none — keep it wide)"] + [
            f"{name}  ({len(regions)} regions: {', '.join(regions[:4])}"
            f"{'…' if len(regions) > 4 else ''})"
            for name, regions in families.items()
        ]
        keys = ["", *families]
        current = keys.index(self._melt_family) if self._melt_family in keys else 0
        choice, ok = QInputDialog.getItem(
            self, "Melt by region", "Measurement to spread down the rows:", options, current, False
        )
        if not ok:
            self._recompute_frame(announce=False)
            return

        self._melt_family = keys[options.index(choice)]
        self._recompute_frame(announce=False)
        frame = self._working_df

        if not self._melt_family:
            self._btn_melt.setText("Melt by…")
            self._status.setText("Analysis dataframe: wide, not melted.")
            notify("Kept wide.")
            return

        self._btn_melt.setText(f"Melt: {self._melt_family}")
        levels = frame["territory"].nunique() if frame is not None and "territory" in frame else 0
        self._status.setText(
            f"Melted {self._melt_family} into {len(frame)} row(s) over {levels} territory level(s). "
            f"'territory' is a model term again; every other column repeats down the rows."
        )
        notify(f"Melted {self._melt_family} — {levels} territory levels.")

    def _coverage_note(self) -> str:
        """One line naming the vessels that decide listwise deletion, or ``""``."""
        coverage = self._wide_coverage
        if coverage is None or coverage.empty:
            return ""
        sparse = coverage.head(3)
        return "  |  sparsest: " + ", ".join(
            f"{r.vessel} {r.pct:.0f}%" for r in sparse.itertuples()
        )

    def _on_reshape_frame(self) -> None:
        """Toggle the analysis dataframe between long and wide (one column per vessel)."""
        if self._analysis_df is None:
            notify("Load the data first.", error=True)
            return

        self._wide_mode = not self._wide_mode
        self._recompute_frame(announce=False)

        if not self._wide_mode:
            self._wide_coverage = None
            self._melt_family = ""
            self._btn_melt.setText("Melt by…")
            self._btn_reshape.setText("Reshape → wide")
            self._status.setText("Analysis dataframe: long (one row per subject × region).")
            notify("Reshaped to long.")
            return

        self._btn_reshape.setText("Reshape → long")
        frame = self._working_df
        vessels = list(frame.attrs.get("vessels", [])) if frame is not None else []
        if not vessels:
            # The pivot bailed out and returned the long frame — keep the button honest.
            self._wide_mode = False
            self._btn_reshape.setText("Reshape → wide")
            return

        complete = self._wide_complete
        self._status.setText(
            f"Analysis dataframe: wide — {len(frame)} subject(s) × {len(vessels)} vessel column(s) "
            f"of {frame.attrs.get('value_column')}; {complete} complete across every vessel."
            + self._coverage_note()
        )
        if complete == 0:
            notify(
                "Reshaped to wide, but no subject has every vessel measured — a path model over "
                "all of them would have nothing to fit. Narrow the model to the well-covered "
                "vessels.",
                error=True,
            )
        else:
            notify(f"Reshaped to wide: {len(vessels)} vessel columns, {complete} complete rows.")

    def _structural_columns(self) -> set[str]:
        """
        Columns the frame is built on, which dropping would break rather than simplify.

        The subject key, the region key and the primary measurement are what every later stage
        addresses: filters join on them, the reshape pivots on them, and the measurement is the
        cell value. Removing one does not give a smaller frame, it gives a frame nothing can be
        rebuilt from.
        """
        structural = {"subject_uid", self._region_column_name()}
        measurements = self._measurements.columns()
        if measurements:
            structural.add(measurements[0])
        return structural

    def _on_drop_column(self, column: str) -> None:
        """Hide a column from the working frame, leaving the dataset untouched."""
        if not column:
            return
        if column in self._structural_columns():
            notify(
                f"“{column}” is a key the frame is built on (subject, region or the primary "
                f"measurement) — dropping it would leave nothing to rebuild from.",
                error=True,
            )
            return

        self._dropped_columns.add(column)
        used_by = [d.name for d in self._derived if column in getattr(d, "expression", "")]
        self._recompute_frame(announce=False)
        note = f" It is still read by {', '.join(used_by)}." if used_by else ""
        self._status.setText(
            f"Dropped “{column}” — {len(self._dropped_columns)} column(s) hidden. "
            f"The dataset is unchanged; restore from any column's right-click menu.{note}"
        )
        notify(f"Dropped {column}.{note}")

    def _on_change_column_type(self, column: str, kind: str) -> None:
        """Recast one column, or clear its override with ``auto``."""
        if not column:
            return
        if kind == "auto":
            self._column_types.pop(column, None)
        else:
            self._column_types[column] = kind

        self._recompute_frame(announce=False)
        frame = self._working_df
        dtype = frame[column].dtype if frame is not None and column in frame.columns else "?"
        label = COLUMN_TYPES.get(kind, (kind, ""))[0]

        # A cast that silently drops values is the one worth naming: it shrinks every model built
        # on the column, and the row count alone will not say why.
        note = ""
        if frame is not None and column in frame.columns:
            base = self._analysis_df
            if base is not None and column in base.columns:
                lost = int(base[column].notna().sum()) - int(frame[column].notna().sum())
                if lost > 0:
                    note = f"  |  {lost} value(s) did not convert and are now missing"
        self._status.setText(f"{column} → {label} (dtype {dtype}){note}")
        notify(f"{column} is now {label}.{note}")

    def _on_set_reference_level(self, column: str, level: str) -> None:
        """Make *level* the treatment reference for *column*, or clear the override."""
        if not column:
            return
        # The region is carried twice — 'territory' and 'group_key' hold the same labels — and a
        # formula may name either. Setting the reference on only the clicked one left the model
        # using the untouched copy, where R falls back to the alphabetically first level.
        from nvitk.stats._statmodels_frames import region_alias_columns

        targets = region_alias_columns(self._analysis_df, column)
        if not level:
            for target in targets:
                self._reference_levels.pop(target, None)
        else:
            for target in targets:
                self._reference_levels[target] = level

        self._recompute_frame(announce=False)
        frame = self._working_df
        if not level:
            self._status.setText(f"{column}: reference cleared — the frame's own order decides.")
            notify(f"{column}: default reference.")
            return

        # Confirm against the rebuilt frame rather than the request: an unusable level is dropped
        # by apply_reference_levels with a note, and the status must not claim it was applied.
        applied = (
            frame is not None
            and column in frame.columns
            and list(getattr(frame[column].dtype, "categories", [])[:1]) == [level]
        )
        if applied:
            also = [t for t in targets if t != column]
            mirror = f" (also applied to {', '.join(also)})" if also else ""
            self._status.setText(
                f"{column}: reference is now {level!r}{mirror} — every other level is contrasted "
                f"against it, and {level!r} itself drops out of the coefficient table."
            )
            notify(f"{column} reference: {level}.")
        else:
            self._reference_levels.pop(column, None)
            notify(f"Could not set {level!r} as the reference for {column}.", error=True)

    def _on_restore_columns(self) -> None:
        """Put every dropped column back."""
        if not self._dropped_columns:
            return
        restored = sorted(self._dropped_columns)
        self._dropped_columns.clear()
        self._recompute_frame(announce=False)
        self._status.setText(f"Restored {len(restored)} column(s): {', '.join(restored)}.")
        notify(f"Restored {len(restored)} column(s).")

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

    def _on_subject_plot(self, subject: str) -> None:
        """
        Open the anatomical viewer for one subject.

        Drawn from the **working** frame, unlike the column viewer: this window is about what a
        subject's anatomy looks like, and painting a vessel the filters have already rejected would
        show a value the analysis is not using.
        """
        frame = self._working_df if self._working_df is not None else self._analysis_df
        if frame is None or frame.empty:
            notify("Reload the data before opening a subject.", error=True)
            return
        if "subject_uid" not in frame.columns:
            notify("The analysis dataframe has no subject_uid column.", error=True)
            return

        dialog = SubjectPlotDialog(
            self,
            frame=frame,
            subject=str(subject),
            region_column=self._region_column_name(),
            atlas=self._brain_atlas_key(),
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
                       self._sem_estimator, self._sem_estimator_label,
                       self._sem_missing, self._sem_missing_label,
                       self._sem_group, self._sem_group_label,
                       self._sem_standardize, self._sem_standardize_label):
            widget.setVisible(is_sem)
        if is_sem:
            self._sync_sem_groups()
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
        display = self._map_display()
        self._plot.set_kind_row_visible(
            kind in {ANALYSIS_MEDIATION, ANALYSIS_MMRM, ANALYSIS_LMROB,
                     ANALYSIS_SEM, ANALYSIS_MRF}
            or bool(display)
        )
        # The checklist drives the anatomical maps too, so it stays visible there whatever
        # engine produced the fit.
        self._plot.set_groups_visible(kind in ANALYSIS_FORMULA_KINDS or bool(display))
        # Panelling needs a family of curves over anatomical levels to split up.
        for widget in (self._plot_display, self._plot_display_label):
            widget.setVisible(kind in ANALYSIS_PANEL_KINDS)
        # The colormap applies to both maps; the contrast picker to both; atlas / hemisphere / views
        # only to the cortical one, which is the only display with a choice of geometry.
        # The whole second row belongs to the maps, so it appears and disappears with them.
        self._plot.set_map_options_visible(bool(display))
        for widget in (self._vasc_cmap, self._vasc_cmap_label):
            widget.setVisible(bool(display))
        for widget in (self._brain_atlas, self._brain_atlas_label,
                       self._brain_hemi, self._brain_hemi_label,
                       self._brain_views, self._brain_views_label,
                       self._brain_surface, self._brain_surface_label,
                       self._brain_threshold, self._brain_threshold_label,
                       self._brain_opacity, self._brain_opacity_label,
                       self._brain_shading, self._brain_blend):
            widget.setVisible(display == "brain")
        self._brain_blend.setEnabled(self._brain_shading.isChecked())
        # No view angle to pick when a flat map shows the whole cortex at once, nor in 3-D where
        # the view is whatever you rotate the brain to.
        fixed_view = display == "brain" and (
            str(self._brain_surface.currentData() or "") == "flat"
            or self._interactive_plot.isChecked()
        )
        for widget in (self._brain_views, self._brain_views_label):
            widget.setVisible(display == "brain" and not fixed_view)
        self._sync_map_contrasts(display)
        self._sync_display_enabled()
        # A map display is checked first because it *replaces* the engine's own figure set: left
        # after the engine branches, an MMRM or lmrob fit kept offering only its own plots and the
        # map's estimate/p-value views were unreachable.
        if display == "vascular":
            self._set_plot_kinds(self._vascular_plot_kinds())
        elif display == "brain":
            self._set_plot_kinds(self._brain_plot_kinds())
        elif is_mmrm:
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
        elif kind == ANALYSIS_SEM:
            detail = f"  |  wide frame on {meta.get('value_column')}" + self._coverage_note()
            if meta.get("resolved_names"):
                detail += f"  |  resolved {meta['resolved_names']}"
        self._status.setText(
            f"Fitted {label}: n={meta.get('n_rows')}"
            + (f" (dropped {meta.get('n_rows_dropped')} incomplete)" if meta.get("n_rows_dropped") else "")
            + detail
            + f"  |  dataset={self._repo.root}"
        )
        self._sync_plot_levels(model_df, groups)
        # The contrast list is a property of *this* fit — without a refresh here a new model keeps
        # the previous one's interaction terms, or shows none because the previous one had none.
        self._sync_map_contrasts(self._map_display())
        self._on_plot()
        notify(f"{label} fit complete.")

    def _fit_formula_model(self, kind: str):
        """Fit MixedLM / OLS / GLM from the shared formulation panel."""
        formula = self._formula.toPlainText().strip()
        groups = self._groups.text().strip() or "group_key"

        # A path model's left-hand sides are vessels, not the outcome column, so the usual
        # LHS-to-measurement resolution would rename the very column the model reads.
        if kind == ANALYSIS_SEM:
            wide = self._working_df
            if wide is None or wide.empty:
                raise ValueError("Load the data before fitting a path model.")
            # Published labels come in several spellings for one artery; map them onto the
            # columns the reshape produced so a hand-typed model still fits.
            syntax, renames = resolve_network_syntax(formula, wide.columns)
            spec = SemSpec(
                syntax=syntax,
                backend=str(self._sem_backend.currentData() or ""),
                estimator=str(self._sem_estimator.currentData() or "ML"),
                listwise=str(self._sem_missing.currentData() or "listwise") == "listwise",
                group=str(self._sem_group.currentData() or ""),
                standardize=self._sem_standardize.isChecked(),
            )
            # A *long* frame is only a problem for a model whose terms are vessels — those need one
            # column each, and a long frame has one row each. A CFA over subject-level indicators is
            # perfectly fittable on whatever grain the frame happens to be, so the refusal is
            # conditioned on the syntax rather than applied to every SEM.
            long_frame = "territory" in wide.columns or (
                "group_key" in wide.columns and wide["group_key"].nunique() > 1
            )
            missing = [v for v in spec.observed() if v not in wide.columns]
            if long_frame and missing:
                raise ValueError(
                    f"A path model needs one column per vessel, and this frame is long (one row "
                    f"per subject × region) — {', '.join(missing[:6])} "
                    f"{'is' if len(missing) == 1 else 'are'} not a column. Press 'Reshape → wide' "
                    f"on the analysis dataframe, or load the measurements on the subject grain."
                )
            result, model_df, meta = fit_sem(data=wide, spec=spec)
            if renames:
                meta["resolved_names"] = ", ".join(f"{k}→{v}" for k, v in sorted(renames.items()))
            meta["value_column"] = wide.attrs.get("value_column", "")
            return result, model_df, meta, meta["value_column"], groups

        df, outcome = resolve_outcome_column(
            self._working_df.copy(), formula, self._measurements.columns()
        )
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
        # Before the per-engine branches: the anatomical maps read the fitted parameter table, which
        # every engine produces, so an engine's own plotter must not shadow them.
        display = self._map_display()
        if display == "vascular":
            self._plot_vascular()
            return
        if display == "brain":
            self._plot_brain()
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
                    title = (
                        f"MMRM {meta.get('structure_label', '')}: correlation between "
                        f"{visit} levels"
                    )
                    fig = (
                        matrix_plot(correlation, title=title, value_label="correlation")
                        if self._interactive_plot.isChecked()
                        else plot_mmrm_correlation(
                            self._last_result, levels=selected or None, title=title
                        )
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
                    note = f"Least-squares means from emmeans, at {meta.get('method')} df."
                    title = f"MMRM least-squares means: {self._last_outcome or ''} by {x}"
                    y_label = self._last_outcome or "Estimated marginal mean"

                    if self._interactive_plot.isChecked():
                        geometry = mmrm_geometry(frame, x=x, hue=hue or "")
                        kwargs = dict(
                            x=x, y=self._last_outcome or "estimate", group=hue or x,
                            errorbar=self._show_ci.isChecked(),
                            title=title, y_label=y_label,
                        )
                    else:
                        kwargs = dict(
                            x=x, hue=hue, errorbar=self._show_ci.isChecked(),
                            title=title, y_label=y_label,
                        )

                    def draw(display_mode: str):
                        """One MMRM figure on whichever backend is selected."""
                        if self._interactive_plot.isChecked():
                            return render(geometry, display=display_mode, **kwargs)
                        return plot_mmrm_emmeans(frame, display=display_mode, **kwargs)

                    try:
                        fig = draw(display)
                    except ValueError as exc:
                        if display != "grouped":
                            raise
                        note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                        fig = draw("overview")
                    else:
                        if display == "grouped":
                            note += f"  {self._panel_note(fig)}"
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
            factors = (self._last_fit_meta or {}).get("grouping_factors") or []
            chosen = self._plot_group.currentText().strip()
            group_col = (
                chosen if chosen and chosen in (df.columns if df is not None else [])
                else (factors[0] if factors else "")
            )
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
                if self._interactive_plot.isChecked():
                    from nvitk.stats.r_robust import _lmrob_band, lmrob_predict

                    geometry = r_model_geometry(
                        self._last_result, df, x=x, y=y,
                        group=group if group in df.columns else "",
                        mode=str(self._plot_mode.currentData() or "auto"),
                        group_order=selected or None,
                        predict_fn=lmrob_predict, band_fn=_lmrob_band,
                        fixed_formula=(self._last_fit_meta or {}).get("formula", ""),
                        errorbar=self._show_ci.isChecked(),
                    )
                    if not self._include_points.isChecked():
                        geometry.points = None
                    render_kwargs = dict(
                        x=x, y=y, group=group if group in df.columns else "",
                        hover_columns=self._hover_columns(df),
                        errorbar=self._show_ci.isChecked(),
                        title=f"lmrob: {y} ~ {x}", x_label=x, y_label=y,
                    )
                    draw = lambda mode: render(geometry, display=mode, **render_kwargs)
                else:
                    draw = lambda mode: plot_lmrob_params(display=mode, **kwargs)
                try:
                    fig = draw(display)
                except ValueError as exc:
                    if display != "grouped":
                        raise
                    display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                    fig = draw("overview")
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
        measurement = (self._measurements.columns() or ["the primary measurement"])[0]
        self._status.setText(
            f"Network syntax for {len(nodes)} vessel(s). Fitting pivots the frame to one row per "
            f"subject with {measurement} spread across the vessel columns."
        )

    def _sync_sem_groups(self) -> None:
        """
        Offer the frame's low-cardinality factors as a multi-group split.

        Continuous columns and identifiers are excluded: a multi-group fit estimates the *whole
        model* once per level, so grouping on ``subject_uid`` asks for one SEM per subject, each on
        a single row.
        """
        previous = str(self._sem_group.currentData() or "")
        self._sem_group.blockSignals(True)
        self._sem_group.clear()
        self._sem_group.addItem("(none — one model for everyone)", "")

        df = self._working_df
        if df is not None and not df.empty:
            for name in df.columns:
                series = df[name]
                if pd.api.types.is_numeric_dtype(series) and not isinstance(
                    series.dtype, pd.CategoricalDtype
                ):
                    continue
                levels = int(series.nunique(dropna=True))
                # Two is the minimum for a comparison; the cap keeps a per-subject split out of a
                # menu it would otherwise dominate.
                if 2 <= levels <= 6:
                    self._sem_group.addItem(f"{name} ({levels} groups)", str(name))

        index = self._sem_group.findData(previous)
        self._sem_group.setCurrentIndex(index if index >= 0 else 0)
        self._sem_group.blockSignals(False)

    def _plot_sem(self) -> None:
        """Path coefficients, the fitted diagram, the measurement model, or the misfit diagnostics."""
        try:
            import matplotlib.pyplot as plt

            from nvitk.stats.sem import (
                plot_sem_modification_indices,
                plot_sem_network,
                sem_modification_indices,
            )

            meta = self._last_fit_meta or {}
            kind = str(self._mediation_plot.currentData() or "paths")
            # ``latent`` is what lets a semopy fit report its loadings at all — that backend writes a
            # measurement line as an ordinary regression and only the spec knows which names are
            # factors.
            paths = sem_paths_frame(
                self._last_result,
                backend=meta.get("backend", ""),
                latent=meta.get("latent", ()),
            )
            regressions = paths.loc[paths["op"] == "~"] if "op" in paths.columns else paths
            loadings = paths.loc[paths["op"] == "=~"] if "op" in paths.columns else paths.iloc[0:0]

            with plt.style.context("default"):
                if kind == "modindices":
                    indices = sem_modification_indices(
                        self._last_result, backend=meta.get("backend", "")
                    )
                    fig = plot_sem_modification_indices(indices)
                    note = (
                        "Parameters the model fixes at zero, ranked by the χ² drop freeing each "
                        "would give. Read them as hypotheses about *where* the model misfits — a "
                        "model rebuilt by chasing this list is fitted to its own residuals and its "
                        "p-values no longer mean anything."
                    )
                elif kind == "loadings":
                    fig = forest_plot(
                        loadings, label="parameter", title="Factor loadings",
                        x_label="Loading",
                    )
                    note = (
                        f"How strongly each indicator loads on its factor "
                        f"({len(meta.get('latent', ())) or 'no'} latent variable(s)). A factor "
                        f"whose loadings are all small is one its indicators do not actually "
                        f"share, whatever the global fit indices say."
                    )
                elif kind == "network":
                    # The matplotlib plotter is used when the model has latents: it draws them as
                    # ellipses and orients each loading factor → indicator, which the vessel-only
                    # interactive diagram has no notion of.
                    if len(loadings):
                        fig = plot_sem_network(paths, title="Fitted structural model")
                        note = (
                            "Rounded blue nodes are latent variables, boxes are measured ones. "
                            "Edge width is the coefficient's magnitude, colour its sign, dashed "
                            "means the interval covers zero."
                        )
                    else:
                        fig = network_plot(
                            regressions, node_labels=VESSEL_NODES,
                            title="Fitted vascular network",
                        )
                        note = (
                            "Edge width is the coefficient's magnitude, colour its sign, dotted "
                            "means the interval covers zero. Hover an edge for the exact value."
                        )
                else:
                    # A pure CFA has no structural paths; leading with an empty forest would read as
                    # a failed fit rather than as a model that estimates loadings only.
                    table = regressions if len(regressions) else loadings
                    fig = forest_plot(
                        table, label="parameter",
                        title="Path coefficients" if len(regressions) else "Factor loadings",
                        x_label="Path coefficient" if len(regressions) else "Loading",
                    )
                    note = (
                        "Standardized paths, strongest first. Blue = interval excludes zero."
                        if meta.get("standardized")
                        else "Unstandardized paths — magnitudes are not comparable between edges."
                    )
                    if not len(regressions):
                        note = "This model estimates a measurement model only. " + note
            self._plot.show_figure(fig)
            self._plot.set_status(note)
        except Exception as exc:
            log.debug("SEM plot failed: %s", exc, exc_info=True)
            self._plot.show_error(f"Plot unavailable: {exc}")

    def _map_display(self) -> str:
        """
        Which anatomical map is selected — ``"vascular"``, ``"brain"``, or ``""`` for neither.

        Every place that used to compare the Display combo against the literal ``"vascular"`` asks
        this instead, so adding a third map is one entry in :data:`_MAP_DISPLAYS` rather than a
        hunt through the gating.
        """
        key = str(self._plot_display.currentData() or "")
        return key if key in _MAP_DISPLAYS else ""

    def _sync_map_contrasts(self, display: str) -> None:
        """
        Populate the contrast picker from the fit's interaction terms, or hide it.

        Discovery is anatomy-specific: an interaction term only counts if one of its factors names
        a level *this map can draw*. So the vascular map asks the vessel table and the brain map
        asks the atlas — using the vessel resolver for a cortical model finds no interaction at all
        and silently hides the picker, which is what made the brain map's contrast views
        unreachable.
        """
        contrasts: list[str] = []
        if display and self._last_result is not None:
            groups = self._plot_group.currentText().strip() or self._region_column_name()
            try:
                if display == "brain":
                    from nvitk.stats.brain_map import brain_interaction_contrasts

                    contrasts = brain_interaction_contrasts(
                        self._last_result, group_column=groups, data=self._last_model_df,
                        atlas=self._brain_atlas_key(),
                    )
                else:
                    from nvitk.stats.vascular_map import interaction_contrasts

                    contrasts = interaction_contrasts(
                        self._last_result, group_column=groups, data=self._last_model_df
                    )
            except Exception as exc:
                log.debug("Contrast discovery failed: %s", exc, exc_info=True)

        # Hidden rather than disabled when the model has no interaction: an empty picker invites
        # the reader to look for a setting that does not apply to this fit.
        show = bool(display) and bool(contrasts)
        for widget in (self._vasc_contrast, self._vasc_contrast_label):
            widget.setVisible(show)
        if not show:
            return

        previous = str(self._vasc_contrast.currentData() or "")
        self._vasc_contrast.blockSignals(True)
        self._vasc_contrast.clear()
        self._vasc_contrast.addItem("main effect (reference level)", "")
        for contrast in contrasts:
            self._vasc_contrast.addItem(contrast, contrast)
        index = self._vasc_contrast.findData(previous)
        self._vasc_contrast.setCurrentIndex(index if index >= 0 else 0)
        self._vasc_contrast.blockSignals(False)

    def _vascular_plot_kinds(self) -> tuple[tuple[str, str], ...]:
        """
        Views the vascular map can offer for the current fit.

        A model whose vessel information lives in a random term — ``(1 + age_c | territory)`` — has
        nothing about individual vessels in its fixed effects, so the coefficient view would fall
        back to observed means without saying why. When per-group terms exist they are offered
        first, one entry each, because they are what that model actually estimated per vessel.
        """
        from nvitk.stats.vascular_map import group_coefficient_terms

        groups = self._groups.text().strip() or self._region_column_name()
        terms = (
            group_coefficient_terms(self._last_result, group_column=groups)
            if self._last_result is not None else []
        )
        if not terms:
            return _VASCULAR_PLOTS

        offered: list[tuple[str, str]] = []
        for term in terms:
            pretty = "intercept" if term.strip("()").lower() == "intercept" else f"{term} slope"
            offered.append((f"Per-vessel {pretty}", f"group:{term}"))
        # The fixed-effect views stay available: a model can have both a territory term and a
        # random slope over it, and they answer different questions.
        return (*offered, *_VASCULAR_PLOTS)

    def _brain_plot_kinds(self) -> tuple[tuple[str, str], ...]:
        """
        Views the brain map can offer for the current fit.

        Same rule as :meth:`_vascular_plot_kinds`: a model whose parcel information lives in a
        random term has nothing about individual parcels in its fixed effects, so those terms are
        offered first rather than letting the coefficient view fall back to observed means without
        saying why.
        """
        from nvitk.stats.brain_map import brain_group_coefficient_terms

        groups = self._groups.text().strip() or self._region_column_name()
        terms = (
            brain_group_coefficient_terms(self._last_result, group_column=groups)
            if self._last_result is not None else []
        )
        if not terms:
            return _BRAIN_PLOTS
        offered = [
            (
                "Per-parcel "
                + ("intercept" if term.strip("()").lower() == "intercept" else f"{term} slope"),
                f"group:{term}",
            )
            for term in terms
        ]
        return (*offered, *_BRAIN_PLOTS)

    def _brain_atlas_key(self) -> str:
        """
        The atlas to draw on — the picker's choice, or the one the plotted measurement was made
        under.

        A measurement loaded under the vascular parcellation carries no Desikan parcels at all, so
        silently defaulting to Desikan there draws nothing and makes a configuration mismatch look
        like a modelling failure.
        """
        chosen = str(self._brain_atlas.currentData() or "")
        if chosen:
            return chosen

        specs = list(self._measurements.specs())
        outcome = self._last_outcome or self._primary_column()
        spec = next((s for s in specs if s.column() == outcome), specs[0] if specs else None)
        atlas = str(getattr(spec, "atlas", "") or "")
        return "vascular" if atlas.startswith("vascular") else "desikan"

    def _plot_brain(self) -> None:
        """Draw the fit's per-parcel numbers on the cortical surface."""
        try:
            import matplotlib.pyplot as plt

            from nvitk.stats.brain_map import (
                parcel_resolver,
                parcel_values_from_result,
                plot_brain_map,
                regions_without_geometry,
            )

            df = self._last_model_df
            groups = (
                self._plot_group.currentText().strip()
                or self._groups.text().strip()
                or self._region_column_name()
            )
            if df is None or groups not in df.columns:
                self._plot.show_error(
                    f"The brain map needs a per-parcel term: {groups!r} is not in the model frame. "
                    f"Group the measurement by region and put it in the formula."
                )
                return

            atlas = self._brain_atlas_key()
            # Say *why* a map would be blank before drawing one. A FLAIR frame parcellates into
            # white-matter zones, not cortical parcels, so none of its levels resolve — reporting
            # that is far more useful than an empty brain the reader has to diagnose.
            levels = [str(v) for v in pd.unique(df[groups].astype(str))]
            undrawable = regions_without_geometry(levels, atlas=atlas)
            if len(undrawable) == len(levels):
                self._plot.show_error(
                    f"None of the {len(levels)} level(s) of {groups!r} are parcels of the "
                    f"{atlas!r} atlas — e.g. {', '.join(undrawable[:4])}. This measurement is not "
                    f"parcellated by it; FLAIR white-matter zones, whole-head scalars and 4D-flow "
                    f"vessels have no cortical geometry. Use the vascular map for vessels, or "
                    f"group the measurement by a region the atlas carries."
                )
                return

            view = str(self._mediation_plot.currentData() or "estimate")
            contrast = str(self._vasc_contrast.currentData() or "")
            if view in {"means", "emmeans"} or view.startswith("group:"):
                contrast = ""
            if view.startswith("group:"):
                source = view
            elif view == "means":
                source = "mean"
            elif view == "emmeans":
                source = "emmeans"
            else:
                source = "coefficient"

            outcome = self._last_outcome or self._primary_column()
            try:
                values, pvalues, note = parcel_values_from_result(
                    self._last_result, group_column=groups, source=source, data=df,
                    outcome=outcome, contrast=contrast, atlas=atlas,
                )
            except ValueError:
                if source == "coefficient":
                    values, pvalues, note = parcel_values_from_result(
                        self._last_result, group_column=groups, source="mean", data=df,
                        outcome=outcome, atlas=atlas,
                    )
                else:
                    raise

            # The Groups checklist drives this map the way it drives every other plot. Both sides go
            # through the atlas resolver, so an aggregate level such as a lobe keeps every parcel it
            # stands for instead of hiding all of them.
            resolve, _ = parcel_resolver(atlas)
            checked = self._plot.checked_levels()
            hide: list[int] = []
            if checked:
                keep = {index for level in checked for index in resolve(level)}
                hide = [index for index in values if index not in keep]

            mode = "pvalue" if view.startswith("pvalue") else "estimate"
            mask = bool(pvalues) and view in {"estimate", "pvalue"}
            with plt.style.context("default"):
                fig = plot_brain_map(
                    values,
                    pvalues=pvalues,
                    mode=mode,
                    mask_nonsignificant=mask,
                    hide=hide,
                    cmap=str(self._vasc_cmap.currentData() or "") or None,
                    atlas=atlas,
                    hemisphere=str(self._brain_hemi.currentData() or "both"),
                    views=str(self._brain_views.currentData() or "lateral,medial"),
                    surface=str(self._brain_surface.currentData() or "pial"),
                    # The pane's existing Interactive toggle: a Plotly figure goes to the web view
                    # and a Matplotlib one to the canvas, decided by the figure's own type.
                    interactive=self._interactive_plot.isChecked(),
                    shading=self._brain_shading.isChecked(),
                    blend=self._brain_blend.isChecked(),
                    opacity=float(self._brain_opacity.value()),
                    threshold=(self._brain_threshold.value() or None),
                    title=f"{outcome} — {note}",
                    label="p" if mode == "pvalue" else outcome,
                )
            self._plot.show_figure(fig)

            if self._interactive_plot.isChecked():
                self._plot.set_status(
                    f"{len(values) - len(hide)} parcel(s) on a rotatable 3-D surface — drag to "
                    f"turn it, scroll to zoom. Same values and colour scale as the static view; "
                    f"grey is non-significant and unpainted is no estimate."
                )
                return

            skipped = len(undrawable)
            self._plot.set_status(
                f"{len(values) - len(hide)} parcel(s) drawn from {note} on the {atlas} atlas"
                + (f", {len(hide)} hidden by the group checklist" if hide else "")
                + (f", {skipped} level(s) with no geometry in this atlas" if skipped else "")
                + ". "
                + (
                    "Colour is −log₁₀(p): brighter means stronger evidence, and nothing about "
                    "direction or size."
                    if mode == "pvalue"
                    else (
                        "Grey parcels did not reach significance; an unpainted parcel means the "
                        "model has no estimate for it."
                        if mask
                        else "Significance is not shown — every parcel with an estimate is "
                             "coloured. An unpainted parcel has no estimate."
                    )
                )
            )
        except Exception as exc:
            log.debug("Brain map failed: %s", exc, exc_info=True)
            self._plot.show_error(f"Brain map unavailable: {exc}")

    def _plot_vascular(self) -> None:
        """Draw the fit's per-vessel numbers on the cerebral circulation schematic."""
        try:
            import matplotlib.pyplot as plt

            from nvitk.stats.vascular_map import (
                plot_vascular_map,
                unmapped_vessel_labels,
                vascular_values_from_result,
            )

            df = self._last_model_df
            groups = (
                self._plot_group.currentText().strip()
                or self._groups.text().strip()
                or self._region_column_name()
            )
            if df is None or groups not in df.columns:
                self._plot.show_error(
                    f"The vascular map needs a per-vessel term: {groups!r} is not in the model "
                    f"frame. Group the measurement vessel-wise and put it in the formula."
                )
                return

            # Say why before drawing a partial figure. ASL perfusion territories are the case that
            # matters: ``left_mca_8`` shares its name with the MCA, so it *looks* drawable, and
            # painting perfusion along an artery is a category error rather than a missing value.
            levels = [str(v) for v in pd.unique(df[groups].astype(str))]
            unmapped = unmapped_vessel_labels(levels)
            if len(unmapped) == len(levels):
                from nvitk.stats.vascular_map import is_perfusion_territory

                perfusion = [u for u in unmapped if is_perfusion_territory(u)]
                if perfusion:
                    self._plot.show_error(
                        f"These are ASL perfusion territories, not vessels — e.g. "
                        f"{', '.join(perfusion[:4])}. They measure the parenchyma an artery "
                        f"supplies (mL/100 g/min), so they do not belong on a schematic of the "
                        f"arteries themselves (mL/min).\n"
                        f"Switch Display to “Brain map” with the vascular atlas, which is that "
                        f"territory parcellation."
                    )
                else:
                    self._plot.show_error(
                        f"None of the {len(levels)} level(s) of {groups!r} are vessels this "
                        f"schematic draws — e.g. {', '.join(unmapped[:4])}. Cortical parcels and "
                        f"whole-head scalars belong on the brain map instead."
                    )
                return

            view = str(self._mediation_plot.currentData() or "estimate")
            # A contrast only applies to a coefficient view; an observed mean has no interaction.
            contrast = str(self._vasc_contrast.currentData() or "")
            if view in {"means", "emmeans"} or view.startswith("group:"):
                contrast = ""
            if view.startswith("group:"):
                source = view
            elif view == "means":
                source = "mean"
            elif view == "emmeans":
                source = "emmeans"
            else:
                source = "coefficient"
            try:
                values, pvalues, note = vascular_values_from_result(
                    self._last_result, group_column=groups, source=source,
                    data=df, outcome=self._last_outcome or self._primary_column(),
                    contrast=contrast,
                )
            except ValueError:
                if source == "coefficient":
                    values, pvalues, note = vascular_values_from_result(
                        self._last_result, group_column=groups, source="mean",
                        data=df, outcome=self._last_outcome or self._primary_column(),
                    )
                else:
                    raise

            # The Groups checklist drives the map the same way it drives every other plot. Its
            # entries are published labels and the map is keyed by drawn node, so both sides go
            # through ``nodes_for_label`` — which, unlike ``canonical_node``, expands a
            # hemisphere-melted level such as 'ICA' to *both* carotids. Using the narrower resolver
            # here meant a hemisphere-grouped fit had its values mirrored correctly and then hid
            # every mirrored vessel, leaving only the three midline ones on the figure.
            from nvitk.stats.vascular_map import nodes_for_label

            checked = self._plot.checked_levels()
            hide: list[str] = []
            if checked:
                keep = {node for level in checked for node in nodes_for_label(level)}
                hide = [node for node in values if node not in keep]

            mode = "pvalue" if view.startswith("pvalue") else "estimate"
            # "all vessels" and the observed means both drop the mask — the first by request, the
            # second because a mean has no p-value and greying every vessel would say the opposite
            # of what is true.
            # The two "_all" views are the unmasked halves of their pairs; the observed mean has
            # no p-values at all, so masking it would grey every vessel and say the opposite of
            # what is true.
            mask = bool(pvalues) and view in {"estimate", "pvalue"}
            outcome = self._last_outcome or self._primary_column()
            with plt.style.context("default"):
                fig = plot_vascular_map(
                    values,
                    pvalues=pvalues,
                    mode=mode,
                    mask_nonsignificant=mask,
                    hide=hide,
                    cmap=str(self._vasc_cmap.currentData() or "") or None,
                    title=f"{outcome} — {note}",
                    label="p" if mode == "pvalue" else outcome,
                )
            self._plot.show_figure(fig)
            self._plot.set_status(
                f"{len(values) - len(hide)} vessel(s) drawn from {note}"
                + (f", {len(hide)} hidden by the group checklist" if hide else "")
                + ". "
                + (
                    "Colour is −log₁₀(p): brighter means stronger evidence, and nothing about "
                    "direction or size."
                    if mode == "pvalue"
                    else (
                        "Grey vessels did not reach significance; a dashed outline means the "
                        "model has no estimate for that vessel."
                        if mask
                        else "Significance is not shown — every vessel with an estimate is "
                             "coloured. A dashed outline means the model has no estimate."
                    )
                )
            )
        except Exception as exc:
            log.debug("Vascular map failed: %s", exc, exc_info=True)
            self._plot.show_error(f"Vascular map unavailable: {exc}")

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
                if self._interactive_plot.isChecked():
                    geometry = nonlinear_geometry(
                        result, args[1], group=group or "",
                        errorbar=self._show_ci.isChecked(),
                    )
                    if not self._include_points.isChecked():
                        geometry.points = None
                    render_kwargs = dict(
                        x=result["x"], y=result["y"], group=group or "",
                        hover_columns=self._hover_columns(args[1]),
                        errorbar=self._show_ci.isChecked(),
                        title=f"{result['y']} ~ f({result['x']})",
                    )
                    draw = lambda mode: render(geometry, display=mode, **render_kwargs)
                else:
                    draw = lambda mode: plot_nonlinear_fit(*args, display=mode, **kwargs)
                try:
                    fig = draw(display)
                except ValueError as exc:
                    if display != "grouped":
                        raise
                    display_note = f"\u26a0 Grouped display unavailable \u2014 {exc}"
                    fig = draw("overview")
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
                    summary = bundle["summary"]
                    if self._interactive_plot.isChecked():
                        fig = forest_plot(
                            summary.rename(columns={
                                "indirect": "coef", "indirect_lo": "ci_low",
                                "indirect_hi": "ci_high",
                            }),
                            label="level", title="Indirect effect by level",
                            x_label="Indirect effect (a·b)",
                        )
                    else:
                        fig = plot_indirect_by_level(summary)
                elif kind == "partial":
                    fig = plot_partial_paths_mediation(
                        self._working_df,
                        x=spec.x,
                        m=spec.m,
                        y=spec.y,
                        covars=spec.covariates,
                    )
                else:
                    title = f"Mediation: {spec.x} → {spec.m} → {spec.y}"
                    fig = (
                        forest_plot(
                            bundle["paths"].rename(columns={"pval": "p_value"}),
                            label="path", title=title, x_label="Effect",
                        )
                        if self._interactive_plot.isChecked()
                        else plot_mediation_forest(bundle["paths"], title=title)
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
            "grain": self._measurements.grain(),
            "attach_qc": self._measurements.attach_qc(),
            "clinical": self._clinical_vars(),
            "cognitive": self._cognitive_vars(),
            "filters": [rule.to_dict() for rule in self._chips.rules()],
            "derived": [column.to_dict() for column in self._derived],
            "combinations": [c.to_dict() for c in self._combinations],
            # Part of the frame recipe, not a view preference: a model fitted on a wide frame with
            # columns dropped cannot be reproduced from a long frame that still has them.
            "dropped_columns": sorted(self._dropped_columns),
            "column_types": dict(self._column_types),
            "reference_levels": dict(self._reference_levels),
            "wide": self._wide_mode,
            "melt_family": self._melt_family,
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
            "sem": {
                "backend": str(self._sem_backend.currentData() or ""),
                "estimator": str(self._sem_estimator.currentData() or "ML"),
                "missing": str(self._sem_missing.currentData() or "listwise"),
                "group": str(self._sem_group.currentData() or ""),
                "standardize": self._sem_standardize.isChecked(),
            },
            "nonlinear": {
                "model": str(self._nl_model.currentData() or ""),
                "x": self._nl_x.currentText(),
                "y": self._nl_y.currentText(),
                "p0": self._nl_p0.text().strip(),
            },
            "mediation": self._mediation_form.spec().to_dict(),
            "pipeline_id": QVTPY_PIPELINE_ID,
            "plot_display": str(self._plot_display.currentData() or "overview"),
            "brain_atlas": str(self._brain_atlas.currentData() or ""),
            "brain_hemisphere": str(self._brain_hemi.currentData() or "both"),
            "brain_views": str(self._brain_views.currentData() or "lateral,medial"),
            "brain_surface": str(self._brain_surface.currentData() or "pial"),
            "brain_shading": self._brain_shading.isChecked(),
            "brain_blend": self._brain_blend.isChecked(),
            "brain_opacity": float(self._brain_opacity.value()),
            "brain_threshold": float(self._brain_threshold.value()),
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
        if cfg.get("grain"):
            self._measurements.set_grain(str(cfg["grain"]))
        self._measurements.set_attach_qc(bool(cfg.get("attach_qc", False)))

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
        self._dropped_columns = {str(c) for c in cfg.get("dropped_columns") or []}
        self._column_types = {
            str(k): str(v) for k, v in (cfg.get("column_types") or {}).items()
        }
        self._reference_levels = {
            str(k): str(v) for k, v in (cfg.get("reference_levels") or {}).items()
        }
        self._wide_mode = bool(cfg.get("wide", False))
        self._melt_family = str(cfg.get("melt_family") or "")
        self._btn_reshape.setText("Reshape → long" if self._wide_mode else "Reshape → wide")
        self._btn_melt.setEnabled(self._wide_mode)
        self._btn_melt.setText(f"Melt: {self._melt_family}" if self._melt_family else "Melt by…")

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

        sem_cfg = cfg.get("sem")
        if isinstance(sem_cfg, dict):
            # The group list is rebuilt from whatever frame is loaded, so a saved choice only
            # restores when that column is still there — which is the correct behaviour, not a
            # silent fallback to a different grouping.
            self._sync_sem_groups()
            for key, combo in (
                ("backend", self._sem_backend),
                ("estimator", self._sem_estimator),
                ("missing", self._sem_missing),
                ("group", self._sem_group),
            ):
                if key in sem_cfg:
                    index = combo.findData(str(sem_cfg[key] or ""))
                    if index >= 0:
                        combo.setCurrentIndex(index)
            if "standardize" in sem_cfg:
                self._sem_standardize.setChecked(bool(sem_cfg["standardize"]))

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
        # "" is a real choice for the atlas (auto), so an absent key and an empty one differ.
        for key, combo in (
            ("brain_atlas", self._brain_atlas),
            ("brain_hemisphere", self._brain_hemi),
            ("brain_views", self._brain_views),
            ("brain_surface", self._brain_surface),
        ):
            if key in cfg:
                index = combo.findData(str(cfg[key] or ""))
                if index >= 0:
                    combo.setCurrentIndex(index)
        if "brain_shading" in cfg:
            self._brain_shading.setChecked(bool(cfg["brain_shading"]))
        if "brain_blend" in cfg:
            self._brain_blend.setChecked(bool(cfg["brain_blend"]))
        if "brain_opacity" in cfg:
            self._brain_opacity.setValue(float(cfg["brain_opacity"] or 1.0))
        if "brain_threshold" in cfg:
            self._brain_threshold.setValue(float(cfg["brain_threshold"] or 0.0))
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
