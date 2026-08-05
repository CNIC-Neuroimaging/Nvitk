"""
Derived-column editor.

Description
-----------
Lets the analysis frame carry transformed measurements — ``log_pi``, a z-scored covariate, a ratio
between two pipelines' measurements — as **real columns**. That matters because a formula-level
transform (``log(pi) ~ …``) only exists inside patsy: it cannot be plotted, filtered on, or used as
a mediation variable. A derived column can do all three.

Three kinds:

``transform``   a canned function (:data:`~nvitk.stats.frame_ops.TRANSFORMS`) applied to one column
``expression``  a free-form expression over the frame's columns, e.g. ``pi / flow_mean``
``bins``        a continuous column cut into labelled groups — the ``tacsctot`` → ``tacsctot_group``
                pattern, e.g. left carotid plaque volume into ``cp0`` (= 0), ``cp1`` (0–25],
                ``cp2`` (25–100], ``cp3`` (> 100)

Names are constrained to Python identifiers because
:func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm` sanitizes column names before fitting — a column
called ``log(pi)`` would silently become ``log_pi_`` in the model frame and stop matching.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nvitk.stats.frame_ops import (
    TRANSFORM_LABELS,
    DerivedColumn,
    apply_derived_columns,
    bin_counts,
    bin_interval_labels,
    default_bin_name,
    default_derived_name,
    parse_cut_points,
    suggest_cut_points,
)

from .theme import COLOR_ERROR, muted_label_style

EXPRESSION_HELP = (
    "Write an expression over the frame's columns, e.g. <code>pi / flow_mean</code> or "
    "<code>log(att_mean) - log(pi)</code>. Available functions: log, log1p, log10, exp, sqrt, abs, "
    "sign, clip, where, minimum, maximum, zscore, rank, mean, std, median. Earlier derived columns "
    "can be referenced by later ones."
)


class DerivedColumnsDialog(QDialog):
    """
    Add, edit, reorder and remove the frame's derived columns.

    The preview pane evaluates the whole list against the current frame every time it changes, so a
    broken expression or an out-of-domain transform is visible before the dialog is accepted.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        frame: pd.DataFrame,
        columns: Sequence[DerivedColumn] = (),
        bin_column: str | None = None,
    ) -> None:
        """
        Build the editor over *frame*'s columns, seeded with the existing *columns*.

        Parameters
        ----------
        bin_column : str, optional
            Open on the bins page with this column preselected, for the table header's
            "Group into bins…" action.
        """
        super().__init__(parent)
        self.setWindowTitle("Derived columns")
        self.resize(680, 520)
        self._frame = frame
        self._columns: list[DerivedColumn] = list(columns)
        # Columns that exist before any derived one — what a new name must not collide with.
        self._base_columns = [str(c) for c in frame.columns]
        # The frame with the current definitions applied, so a bin can be cut from a log or a ratio
        # defined earlier in the list. Rebuilt whenever the list changes.
        self._preview_frame = frame

        lay = QHBoxLayout(self)
        lay.addWidget(self._build_list_side(), stretch=1)
        lay.addWidget(self._build_editor_side(), stretch=1)

        self._refresh_list()
        self._refresh_preview()

        if bin_column:
            kidx = self._kind.findData("bins")
            if kidx >= 0:
                self._kind.setCurrentIndex(kidx)
            sidx = self._bin_source.findText(bin_column)
            if sidx >= 0:
                self._bin_source.setCurrentIndex(sidx)
            self._on_bin_source_changed()

    # ---- construction ---------------------------------------------------------
    def _build_list_side(self) -> QWidget:
        """Left column: the ordered list of derived columns plus its reorder/remove controls."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("Derived columns (applied in order)"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        lay.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        for label, slot in (
            ("Remove", self._on_remove),
            ("↑", lambda: self._on_move(-1)),
            ("↓", lambda: self._on_move(1)),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(muted_label_style())
        lay.addWidget(self._preview)
        return panel

    def _build_editor_side(self) -> QWidget:
        """Right column: the definition form for a single derived column."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self._kind = QComboBox()
        self._kind.addItem("Transform of a column", "transform")
        self._kind.addItem("Expression", "expression")
        self._kind.addItem("Grouped bins (categorical)", "bins")
        self._kind.currentIndexChanged.connect(self._on_kind_changed)
        self._name = QLineEdit()
        self._name.setPlaceholderText("log_pi")
        form.addRow("Kind", self._kind)
        form.addRow("Name", self._name)
        lay.addLayout(form)

        self._stack = QStackedWidget()

        transform_page = QWidget()
        transform_form = QFormLayout(transform_page)
        self._source = QComboBox()
        self._transform = QComboBox()
        for key, label in TRANSFORM_LABELS.items():
            self._transform.addItem(label, key)
        transform_form.addRow("Source", self._source)
        transform_form.addRow("Transform", self._transform)
        self._source.currentIndexChanged.connect(self._suggest_name)
        self._transform.currentIndexChanged.connect(self._suggest_name)
        self._stack.addWidget(transform_page)

        expression_page = QWidget()
        expression_lay = QVBoxLayout(expression_page)
        self._expression = QLineEdit()
        self._expression.setPlaceholderText("pi / flow_mean")
        expression_lay.addWidget(self._expression)
        help_label = QLabel(EXPRESSION_HELP)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.RichText)
        help_label.setStyleSheet(muted_label_style())
        expression_lay.addWidget(help_label)
        expression_lay.addStretch(1)
        self._stack.addWidget(expression_page)

        self._stack.addWidget(self._build_bins_page())

        lay.addWidget(self._stack)

        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: normal;")
        lay.addWidget(self._error)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add")
        self._btn_add.clicked.connect(self._on_add)
        self._btn_update = QPushButton("Update selected")
        self._btn_update.clicked.connect(self._on_update)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_update)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._refresh_sources()
        return panel

    def _build_bins_page(self) -> QWidget:
        """
        Cut a continuous column into labelled groups.

        The cut points are stored as explicit numbers rather than as a rule (“quartiles”): a rule
        would be recomputed against whatever rows are loaded, so the same label would mean a
        different range after a filter changed. The suggest buttons write numbers into the field,
        which you can then edit.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self._bin_source = QComboBox()
        self._bin_source.currentIndexChanged.connect(self._on_bin_source_changed)
        self._bin_cuts = QLineEdit()
        self._bin_cuts.setPlaceholderText("0, 25, 100")
        self._bin_cuts.setToolTip(
            "Interior cut points, lowest first. Bins run from −∞ to +∞ around them, so "
            "'0, 25, 100' makes four groups: ≤ 0, (0, 25], (25, 100], > 100."
        )
        self._bin_cuts.textChanged.connect(self._on_bins_changed)
        self._bin_prefix = QLineEdit("g")
        self._bin_prefix.setToolTip("Labels become prefix0, prefix1, … unless you set them explicitly.")
        self._bin_prefix.textChanged.connect(self._on_bins_changed)
        self._bin_labels = QLineEdit()
        self._bin_labels.setPlaceholderText("(optional) cp0, cp1, cp2, cp3")
        self._bin_labels.textChanged.connect(self._on_bins_changed)
        self._bin_right = QCheckBox("Cut points belong to the lower group")
        self._bin_right.setChecked(True)
        self._bin_right.setToolTip(
            "Checked: bins are (lo, hi], so a value exactly equal to a cut point falls below it — "
            "that is what puts 'exactly 0' in its own group when 0 is the first cut point."
        )
        self._bin_right.stateChanged.connect(self._on_bins_changed)

        form.addRow("Source", self._bin_source)
        form.addRow("Cut points", self._bin_cuts)
        form.addRow("Label prefix", self._bin_prefix)
        form.addRow("Labels", self._bin_labels)
        form.addRow("", self._bin_right)
        lay.addLayout(form)

        suggest_row = QHBoxLayout()
        suggest_row.addWidget(QLabel("Suggest:"))
        for text, method, n_bins in (
            ("Tertiles", "quantile", 3),
            ("Quartiles", "quantile", 4),
            ("Quintiles", "quantile", 5),
            ("Equal width ×4", "equal_width", 4),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, m=method, n=n_bins: self._on_suggest_cuts(m, n))
            suggest_row.addWidget(btn)
        suggest_row.addStretch(1)
        lay.addLayout(suggest_row)

        self._bin_preview = QLabel("")
        self._bin_preview.setWordWrap(True)
        self._bin_preview.setStyleSheet(muted_label_style())
        lay.addWidget(self._bin_preview)
        lay.addStretch(1)
        return page

    def _numeric_columns(self) -> list[str]:
        """Columns that can be binned — base columns plus any numeric derived ones."""
        frame = self._preview_frame
        return [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

    def _on_bin_source_changed(self) -> None:
        """Suggest a name from the source, mirroring ``tacsctot`` → ``tacsctot_group``."""
        source = self._bin_source.currentText()
        if not source:
            return
        current = self._name.text().strip()
        known = {default_bin_name(c) for c in self._numeric_columns()}
        if not current or current in known:
            self._name.setText(default_bin_name(source))
        self._on_bins_changed()

    def _on_suggest_cuts(self, method: str, n_bins: int) -> None:
        """Fill the cut-point field with values derived from the source column's distribution."""
        source = self._bin_source.currentText()
        if not source or source not in self._preview_frame.columns:
            return
        try:
            points = suggest_cut_points(
                self._preview_frame[source], method=method, n_bins=n_bins
            )
        except Exception as exc:
            self._error.setText(str(exc))
            return
        self._error.setText("")
        self._bin_cuts.setText(", ".join(f"{p:g}" for p in points))

    def _on_bins_changed(self, *_args) -> None:
        """Refresh the interval/count preview for the bin definition being edited."""
        source = self._bin_source.currentText()
        if not source or source not in self._preview_frame.columns:
            self._bin_preview.setText("")
            return
        try:
            spec = self._current_spec()
        except ValueError as exc:
            self._bin_preview.setText(str(exc))
            return
        problem = spec.validate()
        if problem and "not a valid column name" not in problem:
            self._bin_preview.setText(problem)
            return

        intervals = bin_interval_labels(spec.cut_points, right=spec.right)
        try:
            counts = bin_counts(self._preview_frame[source], spec)
        except Exception as exc:
            self._bin_preview.setText(str(exc))
            return
        total = int(counts.sum())
        missing = int(pd.to_numeric(self._preview_frame[source], errors="coerce").isna().sum())
        lines = [
            f"{label}:  {interval}  —  {int(counts.get(label, 0))} rows"
            for label, interval in zip(spec.bin_labels(), intervals)
        ]
        if missing:
            lines.append(f"(no group): {missing} rows with no value")
        lines.append(f"{total} of {len(self._preview_frame)} rows grouped.")
        self._bin_preview.setText("\n".join(lines))

    # ---- state ----------------------------------------------------------------
    def columns(self) -> list[DerivedColumn]:
        """The configured derived columns, in application order."""
        return list(self._columns)

    def _available_columns(self, up_to: int | None = None) -> list[str]:
        """Base columns plus every derived column defined before index *up_to*."""
        derived = [c.name for c in self._columns[: up_to if up_to is not None else len(self._columns)]]
        return [*self._base_columns, *derived]

    def _refresh_sources(self) -> None:
        """Repopulate the transform source combo, preserving the current pick."""
        current = self._source.currentText()
        self._source.blockSignals(True)
        self._source.clear()
        for name in self._available_columns():
            self._source.addItem(name)
        idx = self._source.findText(current)
        if idx >= 0:
            self._source.setCurrentIndex(idx)
        self._source.blockSignals(False)

        # Bins only make sense over a continuous column.
        current_bin = self._bin_source.currentText()
        self._bin_source.blockSignals(True)
        self._bin_source.clear()
        for name in self._numeric_columns():
            self._bin_source.addItem(name)
        bidx = self._bin_source.findText(current_bin)
        if bidx >= 0:
            self._bin_source.setCurrentIndex(bidx)
        self._bin_source.blockSignals(False)

    def _refresh_list(self) -> None:
        """Redraw the list of derived columns."""
        current = self._list.currentRow()
        self._list.clear()
        for spec in self._columns:
            self._list.addItem(QListWidgetItem(spec.label()))
        if 0 <= current < self._list.count():
            self._list.setCurrentRow(current)
        self._refresh_sources()

    def _refresh_preview(self) -> None:
        """Evaluate the whole list against the frame and report the outcome."""
        out, errors = apply_derived_columns(self._frame, self._columns)
        self._preview_frame = out
        self._refresh_sources()
        if not self._columns:
            self._preview.setText("No derived columns.")
            self._error.setText("")
            return
        added = [c.name for c in self._columns if c.name in out.columns]
        lines = [f"{len(added)} of {len(self._columns)} column(s) evaluate."]
        for name in added:
            series = out[name]
            n_missing = int(series.isna().sum())
            suffix = f"  ({n_missing} missing)" if n_missing else ""
            if not series.notna().any():
                lines.append(f"{name}: all missing")
            elif isinstance(series.dtype, pd.CategoricalDtype):
                # A binned column has labels, not a range.
                counts = series.value_counts()
                shown = ", ".join(f"{lvl}={int(counts.get(lvl, 0))}" for lvl in series.dtype.categories)
                lines.append(f"{name}: {shown}{suffix}")
            else:
                lines.append(f"{name}: {series.min():.4g} … {series.max():.4g}{suffix}")
        self._preview.setText("\n".join(lines))
        self._error.setText("\n".join(errors))

    # ---- form -----------------------------------------------------------------
    def _current_spec(self) -> DerivedColumn:
        """
        Build a :class:`DerivedColumn` from the editor form.

        Raises
        ------
        ValueError
            If the bin cut points cannot be parsed.
        """
        kind = str(self._kind.currentData())
        cut_points: tuple[float, ...] = ()
        labels: tuple[str, ...] = ()
        if kind == "bins":
            cut_points = parse_cut_points(self._bin_cuts.text())
            raw_labels = [t.strip() for t in self._bin_labels.text().split(",") if t.strip()]
            labels = tuple(raw_labels)
        return DerivedColumn(
            name=self._name.text().strip(),
            kind=kind,
            source=(
                self._source.currentText() if kind == "transform"
                else self._bin_source.currentText() if kind == "bins"
                else ""
            ),
            transform=str(self._transform.currentData() or "") if kind == "transform" else "",
            expression=self._expression.text().strip() if kind == "expression" else "",
            cut_points=cut_points,
            labels=labels,
            label_prefix=self._bin_prefix.text().strip() or "g",
            right=self._bin_right.isChecked(),
        )

    def _on_kind_changed(self) -> None:
        """Swap the definition page and refresh the suggested name."""
        self._stack.setCurrentIndex(self._kind.currentIndex())
        if self._kind.currentData() == "bins":
            self._on_bin_source_changed()
        else:
            self._suggest_name()

    def _suggest_name(self) -> None:
        """Fill the name field with the conventional name while the user has not typed one."""
        if self._kind.currentData() != "transform":
            return
        source = self._source.currentText()
        transform = str(self._transform.currentData() or "")
        if not (source and transform):
            return
        suggested = default_derived_name(source, transform)
        current = self._name.text().strip()
        known = {default_derived_name(source, key) for key in TRANSFORM_LABELS}
        if not current or current in known:
            self._name.setText(suggested)

    def _validate(self, spec: DerivedColumn, *, ignore_index: int | None = None) -> str:
        """Return why *spec* cannot be added, or ``""``."""
        problem = spec.validate()
        if problem:
            return problem
        taken = {c.name for i, c in enumerate(self._columns) if i != ignore_index}
        if spec.name in taken:
            return f"A derived column named {spec.name!r} already exists."
        if spec.name in self._base_columns:
            return f"{spec.name!r} is already a column of the analysis frame."
        return ""

    def _on_add(self) -> None:
        """Append the form's column to the list."""
        try:
            spec = self._current_spec()
        except ValueError as exc:
            self._error.setText(str(exc))
            return
        problem = self._validate(spec)
        if problem:
            self._error.setText(problem)
            return
        self._columns.append(spec)
        self._refresh_list()
        self._list.setCurrentRow(len(self._columns) - 1)
        self._refresh_preview()

    def _on_update(self) -> None:
        """Replace the selected column with the form's definition."""
        row = self._list.currentRow()
        if not (0 <= row < len(self._columns)):
            return
        try:
            spec = self._current_spec()
        except ValueError as exc:
            self._error.setText(str(exc))
            return
        problem = self._validate(spec, ignore_index=row)
        if problem:
            self._error.setText(problem)
            return
        self._columns[row] = spec
        self._refresh_list()
        self._refresh_preview()

    def _on_remove(self) -> None:
        """Drop the selected column."""
        row = self._list.currentRow()
        if 0 <= row < len(self._columns):
            del self._columns[row]
            self._refresh_list()
            self._refresh_preview()

    def _on_move(self, delta: int) -> None:
        """Move the selected column by *delta* positions (order matters: later ones may use it)."""
        row = self._list.currentRow()
        target = row + delta
        if 0 <= row < len(self._columns) and 0 <= target < len(self._columns):
            self._columns[row], self._columns[target] = self._columns[target], self._columns[row]
            self._refresh_list()
            self._list.setCurrentRow(target)
            self._refresh_preview()

    def _on_select(self, row: int) -> None:
        """Load the selected column back into the editor form."""
        if not (0 <= row < len(self._columns)):
            return
        spec = self._columns[row]
        kidx = self._kind.findData(spec.kind)
        self._kind.blockSignals(True)
        self._kind.setCurrentIndex(kidx if kidx >= 0 else 0)
        self._kind.blockSignals(False)
        self._stack.setCurrentIndex(self._kind.currentIndex())
        self._name.setText(spec.name)

        if spec.kind == "transform":
            idx = self._source.findText(spec.source)
            if idx >= 0:
                self._source.setCurrentIndex(idx)
            tidx = self._transform.findData(spec.transform)
            if tidx >= 0:
                self._transform.setCurrentIndex(tidx)
        elif spec.kind == "bins":
            for widget in (self._bin_source, self._bin_cuts, self._bin_prefix,
                           self._bin_labels, self._bin_right):
                widget.blockSignals(True)
            try:
                sidx = self._bin_source.findText(spec.source)
                if sidx >= 0:
                    self._bin_source.setCurrentIndex(sidx)
                self._bin_cuts.setText(", ".join(f"{c:g}" for c in spec.cut_points))
                self._bin_prefix.setText(spec.label_prefix)
                self._bin_labels.setText(", ".join(spec.labels))
                self._bin_right.setChecked(spec.right)
            finally:
                for widget in (self._bin_source, self._bin_cuts, self._bin_prefix,
                               self._bin_labels, self._bin_right):
                    widget.blockSignals(False)
            self._on_bins_changed()
        else:
            self._expression.setText(spec.expression)
        self._error.setText("")


__all__ = ["DerivedColumnsDialog"]
