"""
Mediation formulation panel and its background worker.

Description
-----------
Mediation asks a different question from a MixedLM — "how much of X's effect on Y goes through M?" —
so it gets its own form rather than a free-text formula: pick X, M and Y, tick the covariates, choose
the engine, and run.

Three engines are offered (see :mod:`nvitk.stats.mediation`). The MixedLM cluster bootstrap is the
one that respects the subject × territory nesting, and it is also the slow one: ``2 × n_boot`` model
fits. It therefore runs on :class:`MediationWorker` with a progress bar, a live ETA and a working
Cancel — a cancelled run still summarizes the draws that completed.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import threading
import time

import pandas as pd
from qtpy.QtCore import QThread, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.stats.mediation import ENGINE_LABELS, MEDIATION_ENGINES, MediationSpec, run_mediation

from .theme import COLOR_ERROR, muted_label_style

log = Logger()

ENGINE_HINTS: dict[str, str] = {
    "mixedlm_bootstrap": (
        "Fits the mediator and outcome models as MixedLMs and resamples whole subjects with "
        "replacement. Respects the subject × territory nesting. Cost: 2 × draws model fits — start "
        "with a few hundred."
    ),
    "pingouin_by_level": (
        "Runs pingouin's mediation separately within each level of the grouping column, giving a "
        "per-territory picture. Fast, but each fit is an OLS that ignores the nesting."
    ),
    "statsmodels_parametric": (
        "statsmodels Mediation with OLS mediator/outcome models over the pooled frame. Fast and "
        "standard, but pooling repeated rows treats them as independent."
    ),
}


class MediationWorker(QThread):
    """
    Background thread running :func:`~nvitk.stats.mediation.run_mediation`.

    Emits ``progress(done, total)`` as draws complete and ``finished_ok(bundle)`` or
    ``failed(message)`` at the end. :meth:`cancel` sets an event the engines poll every draw, so the
    run stops within one model fit rather than at the next natural boundary.
    """

    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, df: pd.DataFrame, spec: MediationSpec) -> None:
        """Copy the frame so the worker never reads one the GUI is mutating."""
        super().__init__()
        self._df = df.copy()
        self._spec = spec
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop at the next draw boundary."""
        self._cancel.set()

    def run(self) -> None:
        """Run the configured engine and emit the result bundle."""
        try:
            bundle = run_mediation(
                self._df,
                self._spec,
                progress=lambda done, total: self.progress.emit(int(done), int(total)),
                should_cancel=self._cancel.is_set,
            )
        except Exception as exc:
            log.exception("Mediation analysis failed.")
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(bundle)


class MediationFormPanel(QGroupBox):
    """
    X / M / Y selectors, covariate checklist, engine choice and run controls.

    Signals
    -------
    runRequested(MediationSpec)
        The user pressed Run and the spec validated.
    cancelRequested
        The user pressed Cancel during a run.
    """

    runRequested = Signal(object)
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the form, the covariate list and the progress row."""
        super().__init__("Mediation analysis", parent)
        self._columns: list[str] = []
        self._numeric_columns: list[str] = []
        self._started_at: float | None = None

        lay = QVBoxLayout(self)

        form = QFormLayout()
        self._engine = QComboBox()
        for key in MEDIATION_ENGINES:
            self._engine.addItem(ENGINE_LABELS[key], key)
        self._engine.currentIndexChanged.connect(self._on_engine_changed)

        self._x = QComboBox()
        self._m = QComboBox()
        self._y = QComboBox()
        self._x.setToolTip("Exposure — the variable whose effect is being decomposed.")
        self._m.setToolTip("Mediator — the intermediate variable the indirect path runs through.")
        self._y.setToolTip("Outcome.")

        self._group_col = QComboBox()
        self._group_col_label = QLabel("Grouping level")
        self._subject_col = QComboBox()
        self._subject_col_label = QLabel("Subject (cluster)")

        self._n_boot = QSpinBox()
        self._n_boot.setRange(10, 100000)
        self._n_boot.setSingleStep(100)
        self._n_boot.setValue(500)
        self._n_boot_label = QLabel("Bootstrap draws")
        self._seed = QSpinBox()
        self._seed.setRange(0, 10**6)
        self._seed.setValue(42)
        self._ci = QDoubleSpinBox()
        self._ci.setRange(0.50, 0.999)
        self._ci.setSingleStep(0.01)
        self._ci.setDecimals(3)
        self._ci.setValue(0.95)

        form.addRow("Engine", self._engine)
        form.addRow("X (exposure)", self._x)
        form.addRow("M (mediator)", self._m)
        form.addRow("Y (outcome)", self._y)
        form.addRow(self._group_col_label, self._group_col)
        form.addRow(self._subject_col_label, self._subject_col)
        form.addRow(self._n_boot_label, self._n_boot)
        form.addRow("Seed", self._seed)
        form.addRow("CI level", self._ci)
        lay.addLayout(form)

        lay.addWidget(QLabel("Covariates (adjusted for in both models)"))
        self._covariates = QListWidget()
        self._covariates.setMaximumHeight(110)
        lay.addWidget(self._covariates)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(muted_label_style())
        lay.addWidget(self._hint)

        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: normal;")
        self._error.setVisible(False)
        lay.addWidget(self._error)

        run_row = QHBoxLayout()
        self._btn_run = QPushButton("Run mediation")
        self._btn_run.clicked.connect(self._on_run)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancelRequested.emit)
        run_row.addWidget(self._btn_run)
        run_row.addWidget(self._btn_cancel)
        run_row.addStretch(1)
        lay.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        lay.addStretch(1)
        self._on_engine_changed()

    # ---- columns --------------------------------------------------------------
    def set_columns(self, df: pd.DataFrame | None) -> None:
        """
        Repopulate every selector from *df*, preserving choices that are still valid.

        X / M / Y offer numeric columns only — the engines read coefficients back by name, which a
        categorical expansion (``C(x)[T.b]``) makes impossible. Transform a categorical into a
        derived numeric column first if it really is the exposure.
        """
        if df is None or df.empty:
            self._columns = []
            self._numeric_columns = []
        else:
            self._columns = [str(c) for c in df.columns]
            self._numeric_columns = [
                str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]

        for combo, options in (
            (self._x, self._numeric_columns),
            (self._m, self._numeric_columns),
            (self._y, self._numeric_columns),
            (self._group_col, self._columns),
            (self._subject_col, self._columns),
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for name in options:
                combo.addItem(name)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        # Sensible defaults on first population.
        for combo, preferred in (
            (self._group_col, ("territory", "group_key")),
            (self._subject_col, ("subject_uid", "patient_id")),
        ):
            if combo.currentText() not in preferred:
                for candidate in preferred:
                    idx = combo.findText(candidate)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        break

        checked = {
            self._covariates.item(i).text()
            for i in range(self._covariates.count())
            if self._covariates.item(i).checkState() == Qt.Checked
        }
        self._covariates.clear()
        for name in self._numeric_columns:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            self._covariates.addItem(item)

    def _checked_covariates(self) -> tuple[str, ...]:
        """Covariate names currently ticked, excluding whatever plays X/M/Y."""
        roles = {self._x.currentText(), self._m.currentText(), self._y.currentText()}
        return tuple(
            self._covariates.item(i).text()
            for i in range(self._covariates.count())
            if self._covariates.item(i).checkState() == Qt.Checked
            and self._covariates.item(i).text() not in roles
        )

    # ---- engine-dependent fields ----------------------------------------------
    def _on_engine_changed(self) -> None:
        """Enable only the fields the selected engine uses, and update the hint."""
        engine = str(self._engine.currentData() or MEDIATION_ENGINES[0])
        uses_group = engine in {"mixedlm_bootstrap", "pingouin_by_level"}
        uses_subject = engine == "mixedlm_bootstrap"

        for widget in (self._group_col, self._group_col_label):
            widget.setEnabled(uses_group)
        for widget in (self._subject_col, self._subject_col_label):
            widget.setEnabled(uses_subject)

        if engine == "statsmodels_parametric":
            self._n_boot_label.setText("Replications")
            self._n_boot.setValue(max(self._n_boot.value(), 1000))
        elif engine == "pingouin_by_level":
            self._n_boot_label.setText("Bootstrap draws")
            self._n_boot.setValue(max(self._n_boot.value(), 1000))
        else:
            self._n_boot_label.setText("Bootstrap draws")

        self._hint.setText(ENGINE_HINTS.get(engine, ""))

    # ---- spec round-trip ------------------------------------------------------
    def spec(self) -> MediationSpec:
        """The mediation currently described by the form."""
        return MediationSpec(
            x=self._x.currentText(),
            m=self._m.currentText(),
            y=self._y.currentText(),
            covariates=self._checked_covariates(),
            group_col=self._group_col.currentText(),
            subject_col=self._subject_col.currentText(),
            engine=str(self._engine.currentData() or MEDIATION_ENGINES[0]),
            n_boot=int(self._n_boot.value()),
            seed=int(self._seed.value()),
            ci=float(self._ci.value()),
        )

    def apply_spec(self, spec: MediationSpec) -> None:
        """Load *spec* into the form, ignoring columns that are not currently available."""
        idx = self._engine.findData(spec.engine)
        if idx >= 0:
            self._engine.setCurrentIndex(idx)
        for combo, value in (
            (self._x, spec.x),
            (self._m, spec.m),
            (self._y, spec.y),
            (self._group_col, spec.group_col),
            (self._subject_col, spec.subject_col),
        ):
            pos = combo.findText(value)
            if pos >= 0:
                combo.setCurrentIndex(pos)
        self._n_boot.setValue(int(spec.n_boot))
        self._seed.setValue(int(spec.seed))
        self._ci.setValue(float(spec.ci))
        wanted = {str(c) for c in spec.covariates}
        for i in range(self._covariates.count()):
            item = self._covariates.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)

    # ---- run lifecycle --------------------------------------------------------
    def _on_run(self) -> None:
        """Validate and emit the run request."""
        spec = self.spec()
        problem = spec.validate()
        if problem:
            self.set_error(problem)
            return
        self.set_error("")
        self.runRequested.emit(spec)

    def set_error(self, message: str) -> None:
        """Show a validation/failure message under the form."""
        self._error.setText(message)
        self._error.setVisible(bool(message))

    def set_running(self, running: bool) -> None:
        """Swap the panel between idle and running states."""
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._progress.setVisible(running)
        if running:
            self._started_at = time.monotonic()
            self._progress.setRange(0, 0)  # indeterminate until the first progress tick
            self._progress.setFormat("starting…")
        else:
            self._started_at = None
            self._progress.setRange(0, 100)
            self._progress.setValue(0)

    def set_progress(self, done: int, total: int) -> None:
        """Advance the bar and, after enough draws to be meaningful, show an ETA."""
        if total <= 0:
            return
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        text = f"{done} / {total}"
        # 10 draws is enough for the per-draw cost to stabilize; before that the estimate is noise.
        if self._started_at is not None and done >= 10:
            elapsed = time.monotonic() - self._started_at
            remaining = elapsed / done * (total - done)
            text = f"{text}  ·  ~{_format_duration(remaining)} left"
        self._progress.setFormat(text)


def _format_duration(seconds: float) -> str:
    """Compact duration string, e.g. ``45s`` / ``3m 20s`` / ``1h 05m``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


__all__ = ["ENGINE_HINTS", "MediationFormPanel", "MediationWorker"]
