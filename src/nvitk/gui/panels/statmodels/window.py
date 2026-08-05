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
from typing import Any

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from nvitk.stats import (
    MeasurementSpec,
    fit_or_load_mixedlm,
    mixedlm_info_dict,
    plot_mixedlm_params,
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
    ANALYSIS_ITEMS,
    ANALYSIS_MEDIATION,
    ANALYSIS_MIXEDLM,
    CONFIG_VERSION,
    DEFAULT_FORMULA,
    DEFAULT_GROUPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RE,
    DEFAULT_VC,
    PIPELINE_KIND_QVTPY,
)
from .derived import DerivedColumnsDialog
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
        self._filter_report: list[dict[str, Any]] = []

        # ---- result state -----------------------------------------------------
        self._last_result = None
        self._last_model_df: pd.DataFrame | None = None
        self._last_fit_meta: dict[str, Any] | None = None
        self._last_outcome: str | None = None
        self._mediation_bundle: dict[str, Any] | None = None
        # Group selection restored from a config, applied once the levels are known after a fit.
        self._pending_plot_groups: list[str] | None = None

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

        self._model_stack = QStackedWidget()
        self._model_stack.addWidget(self._build_formula_box())
        self._mediation_form = MediationFormPanel()
        self._model_stack.addWidget(self._mediation_form)
        lay.addWidget(self._model_stack, stretch=1)
        return panel

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
        for btn in (self._btn_fit, self._btn_save, self._btn_load, self._btn_plot):
            btn_row.addWidget(btn)
        box_lay.addLayout(btn_row)

        box_lay.addStretch(1)
        return box

    def _build_plot_options(self) -> QWidget:
        """Plot mode / x / points, laid out for the plot pane's own top row."""
        widget = QWidget()
        lay = QHBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

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
            "Add transformed measurements (log, z-score, ratios) as real columns, usable as the "
            "model outcome, as predictors, as plot axes and as filter targets."
        )
        row.addWidget(self._btn_derived)
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
        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_plot.clicked.connect(self._on_plot)
        self._plot_mode.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._plot_x.currentIndexChanged.connect(lambda *_: self._on_plot())
        self._include_points.stateChanged.connect(lambda *_: self._on_plot())
        self._show_ci.stateChanged.connect(lambda *_: self._on_plot())
        self._plot.optionsChanged.connect(self._on_plot)
        self._mediation_plot.currentIndexChanged.connect(lambda *_: self._plot_mediation())

        self._chips.rulesChanged.connect(lambda: self._recompute_frame())
        self._frame_view.filtersRequested.connect(self._on_edit_column_filter)
        self._frame_view.clearFiltersRequested.connect(self._on_clear_column_filter)
        self._frame_view.transformRequested.connect(self._on_quick_transform)
        self._frame_view.binsRequested.connect(self._on_bin_column)
        self._frame_view.plotXRequested.connect(self._on_set_plot_x)
        self._btn_derived.clicked.connect(lambda *_: self._on_edit_derived())

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

        derived_frame, derived_errors = apply_derived_columns(base, self._derived)
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
        self._frame_view.set_frame(working, filtered_columns=filtered_columns(rules))
        self._sync_column_combos(working)
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
        derived_frame, _ = apply_derived_columns(base, self._derived)
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

    def _sync_analysis_type(self) -> None:
        """Swap the formulation panel and clear results that no longer apply."""
        is_mediation = self._analysis_kind() == ANALYSIS_MEDIATION
        self._model_stack.setCurrentIndex(1 if is_mediation else 0)
        # The figure picker only applies to mediation; the group checklist only to MixedLM.
        self._plot.set_kind_row_visible(is_mediation)
        self._plot.set_groups_visible(not is_mediation)
        if is_mediation:
            self._mediation_form.set_columns(self._working_df)
            if self._mediation_bundle is None:
                self._report.set_message("Configure the mediation and press Run.")
                self._plot.clear()
                self._plot.set_levels("", [])
        elif self._last_result is None:
            self._report.set_message("Fit a model to see its summary here.")
            self._plot.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # MixedLM
    # ──────────────────────────────────────────────────────────────────────────
    def _on_fit(self) -> None:
        """Fit a MixedLM on the working frame, then show its report and plot."""
        formula = self._formula.toPlainText().strip()
        groups = self._groups.text().strip() or "group_key"
        re_formula = self._re_formula.text().strip() or "0"
        try:
            vc = parse_vc_formula(self._vc_formula.text())
            if self._working_df is None:
                raise ValueError("Reload the data before fitting.")
            df, outcome = resolve_outcome_column(
                self._working_df.copy(), formula, self._measurements.columns()
            )
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
        self._last_model_df = model_df
        self._last_fit_meta = meta
        self._last_outcome = outcome
        self._mediation_bundle = None

        info = mixedlm_info_dict(
            result, outcome_name=outcome or self._primary_column(), group_name=groups
        )
        note = dropped_rows_note(meta)
        raw = render_mixedlm_info(info)
        self._report.set_mixedlm(info, raw_text=f"{note}\n\n{raw}" if note else raw, note=note)

        self._status.setText(
            f"Fitted n={meta.get('n_rows')}"
            + (f" (dropped {meta.get('n_rows_dropped')} incomplete)" if meta.get("n_rows_dropped") else "")
            + f"  |  groups={groups}  |  re={re_formula!r}  |  dataset={self._repo.root}"
        )
        self._sync_plot_levels(model_df, groups)
        self._on_plot()
        notify("MixedLM fit complete.")

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
        if self._analysis_kind() == ANALYSIS_MEDIATION:
            self._plot_mediation()
            return
        if self._last_result is None or self._last_model_df is None:
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

            # Draw on the light default style regardless of what the app (or a previous call to
            # plt.style.use) left in the global rcParams; the canvas is whitened on embed.
            with plt.style.context("default"):
                fig = plot_mixedlm_params(
                    result=self._last_result,
                    df_fit=df,
                    x=x,
                    y=y,
                    group=group,
                    mode=str(self._plot_mode.currentData() or "auto"),
                    include_points=self._include_points.isChecked(),
                    errorbar=self._show_ci.isChecked(),
                    group_order=selected,
                    restrict_to_orders=subset,
                    covariate_refs=refs,
                    title=f"MixedLM: {y} ~ {x} | {group}",
                )
            if fig is None:
                fig = plt.gcf()
            self._plot.show_figure(fig)

            notes = []
            if subset:
                notes.append(f"Showing {len(selected)} of {len(all_levels)} {group} levels.")
            emm_error = getattr(fig, "emm_error", "")
            if emm_error:
                notes.append(
                    f"⚠ Model curves unavailable — the marginal-means grid could not be built "
                    f"({emm_error}). The raw observations are still shown."
                )
            self._plot.set_status("  ".join(notes))
        except Exception as exc:
            log.debug("Plot failed: %s", exc)
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
            "analysis_type": self._analysis_kind(),
            "mm_formula": self._formula.toPlainText().strip(),
            "groups": self._groups.text().strip(),
            "re_formula": self._re_formula.text().strip(),
            "vc_formula": self._vc_formula.text().strip(),
            "model_name": self._model_name.text().strip(),
            "mediation": self._mediation_form.spec().to_dict(),
            "pipeline_id": QVTPY_PIPELINE_ID,
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

        analysis = str(cfg.get("analysis_type") or ANALYSIS_MIXEDLM)
        aidx = self._analysis_type.findData(analysis)
        if aidx >= 0:
            self._analysis_type.setCurrentIndex(aidx)
        if isinstance(cfg.get("mediation"), dict):
            self._mediation_form.apply_spec(MediationSpec.from_dict(cfg["mediation"]))

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
        """Save the last fitted model (pickle), its config, and its summary text under
        ``nvitk-statmodels/<model_name>/``."""
        if self._last_result is None and self._mediation_bundle is None:
            notify("Fit a model or run a mediation before saving.", error=True)
            return
        name = self._model_name.text().strip() or "model"
        out_dir = statmodels_root(self._repo) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            (out_dir / "config.json").write_text(
                json.dumps(self._config_dict(), indent=2), encoding="utf-8"
            )
            if self._last_result is not None:
                self._last_result.save(str(out_dir / "model.pkl"))
                (out_dir / "info.txt").write_text(
                    render_mixedlm_info(
                        mixedlm_info_dict(
                            self._last_result,
                            outcome_name=self._last_outcome or self._primary_column(),
                            group_name=self._groups.text().strip() or "group_key",
                        )
                    ),
                    encoding="utf-8",
                )
            if self._mediation_bundle is not None:
                (out_dir / "mediation.txt").write_text(
                    render_mediation_info(self._mediation_bundle), encoding="utf-8"
                )
                self._mediation_bundle["paths"].to_csv(out_dir / "mediation_paths.csv", index=False)
                summary = self._mediation_bundle.get("summary")
                if isinstance(summary, pd.DataFrame) and not summary.empty:
                    summary.to_csv(out_dir / "mediation_by_level.csv", index=False)
        except Exception as exc:
            notify(f"Save failed: {exc}", error=True)
            return
        notify(f"Saved → {out_dir}")
        self._status.setText(f"Saved {out_dir}")

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
                self._mediation_bundle = None
                info = mixedlm_info_dict(
                    self._last_result, group_name=self._groups.text().strip() or "group_key"
                )
                self._report.set_mixedlm(info, raw_text=render_mixedlm_info(info))
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
