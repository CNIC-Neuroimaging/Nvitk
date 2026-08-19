"""
Editor for region combinations — arithmetic across the rows of one measurement.

Description
-----------
The derived-columns editor works *within* a row: ``log(pi)``, ``pi / flow_mean``. The quantities a
vascular analysis actually needs are sums *over* rows of one subject — ``TCBF = RICA + LICA + BASI``,
or the mass-balance residual ``LVA + RVA − BASI`` — because the analysis frame is long, one row per
subject × territory.

This dialog builds those. Pick a measurement, tick the regions, give each a coefficient, and choose
whether the result becomes a **column** (the subject's value on every one of their rows, usable as a
covariate) or a **row** (a synthetic territory modelled alongside the real ones).

Two prefill buttons cover the cases worth having ready: the standard composites (TCBF and friends)
and the conservation balances from :mod:`~nvitk.stats.vessel_network`, which is what makes
"conservation residuals as derived columns" a two-click operation.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.geometry import fit_dialog
from nvitk.core.logger import Logger
from nvitk.stats.region_algebra import (
    COMBINE_MODES,
    COMBINE_OPS,
    RegionCombination,
    composite_combinations,
    conservation_combinations,
    evaluate_region_combination,
)

from .theme import COLOR_ERROR, COLOR_MUTED, muted_label_style

log = Logger()


class RegionCombinationsDialog(QDialog):
    """
    Add, edit and remove the region combinations applied to the analysis frame.

    The list on the left holds the definitions; the form on the right edits the selected one. A live
    preview evaluates it against the real frame, so a combination that resolves to nothing says so
    before it reaches a model.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        frame: pd.DataFrame,
        combinations: Sequence[RegionCombination] = (),
        region_column: str = "territory",
    ) -> None:
        """Build the editor, seeded from *combinations*."""
        super().__init__(parent)
        self.setWindowTitle("Region combinations")
        fit_dialog(self, 880, 620)
        self._frame = frame
        self._combinations = [c for c in combinations]
        self._region_column = region_column if region_column in frame.columns else "territory"

        root = QVBoxLayout(self)
        intro = QLabel(
            "Combine a measurement <b>across regions</b> — TCBF = RICA + LICA + BASI, or a "
            "mass-balance residual with −1 coefficients. Within-row arithmetic belongs in "
            "<i>Derived columns</i> instead."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        body = QHBoxLayout()
        body.addWidget(self._build_list(), stretch=1)
        body.addWidget(self._build_form(), stretch=2)
        root.addLayout(body, stretch=1)

        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(muted_label_style())
        root.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_list()
        if not self._combinations:
            # Open on a live, editable definition rather than a disabled form. An empty list greys
            # out every control, which reads as a broken window — the user has no way to know the
            # form only wakes up after pressing Add.
            self._on_add()
        self._list.setCurrentRow(0)
        self._sync_form(self._current())

    # ---- construction ---------------------------------------------------------
    def _build_list(self) -> QWidget:
        """The definitions list plus its add/remove/prefill buttons."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_selected)
        lay.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        for label, slot in (("Add", self._on_add), ("Remove", self._on_remove)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        lay.addLayout(row)

        prefill = QHBoxLayout()
        composites = QPushButton("Add composites")
        composites.setToolTip(
            "TCBF, carotid inflow and the posterior share — the whole-brain quantities."
        )
        composites.clicked.connect(lambda: self._prefill(composite_combinations))
        prefill.addWidget(composites)
        balances = QPushButton("Add conservation")
        balances.setToolTip(
            "One residual per mass-balance rule: vertebrals → basilar, each carotid → its "
            "branches, and the global inflow check. Each should sit near zero."
        )
        balances.clicked.connect(lambda: self._prefill(conservation_combinations))
        prefill.addWidget(balances)
        lay.addLayout(prefill)
        return panel

    def _build_form(self) -> QWidget:
        """The editor for the selected combination."""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self._name = QLineEdit()
        self._name.setPlaceholderText("TCBF")
        self._name.textChanged.connect(self._commit)
        form.addRow("Name", self._name)

        self._value_column = QComboBox()
        for column in self._numeric_columns():
            self._value_column.addItem(column)
        self._value_column.currentIndexChanged.connect(self._commit)
        form.addRow("Measurement", self._value_column)

        self._op = QComboBox()
        for key, description in COMBINE_OPS.items():
            self._op.addItem(description, key)
        self._op.currentIndexChanged.connect(self._commit)
        form.addRow("Operation", self._op)

        self._mode = QComboBox()
        for key, description in COMBINE_MODES.items():
            self._mode.addItem(description, key)
        self._mode.currentIndexChanged.connect(self._commit)
        form.addRow("Result", self._mode)

        self._require_all = QCheckBox("Require every region")
        self._require_all.setChecked(True)
        self._require_all.setToolTip(
            "Refuse rather than compute a partial result. A balance missing one outflow term "
            "returns that term's whole magnitude dressed up as a residual."
        )
        self._require_all.stateChanged.connect(self._commit)
        form.addRow("", self._require_all)
        lay.addLayout(form)

        lay.addWidget(QLabel("Regions and coefficients"))
        self._regions = QListWidget()
        self._regions.setToolTip(
            "Tick a region to include it. A coefficient of −1 subtracts it, which is how a balance "
            "is written."
        )
        lay.addWidget(self._regions, stretch=1)
        self._populate_regions()
        return panel

    def _numeric_columns(self) -> list[str]:
        """Measurement columns a combination can operate on."""
        return [
            str(c) for c in self._frame.columns
            if pd.api.types.is_numeric_dtype(self._frame[c])
        ]

    def _populate_regions(self) -> None:
        """One checkable row per region level, each with a coefficient spinner."""
        self._regions.clear()
        self._spinners: dict[str, QDoubleSpinBox] = {}
        if self._region_column not in self._frame.columns:
            return
        for level in sorted({str(v) for v in self._frame[self._region_column].dropna()}):
            item = QListWidgetItem(self._regions)
            widget = QWidget()
            row = QHBoxLayout(widget)
            row.setContentsMargins(4, 1, 4, 1)
            check = QCheckBox(level)
            check.stateChanged.connect(self._commit)
            spin = QDoubleSpinBox()
            spin.setRange(-99.0, 99.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(1.0)
            spin.setMaximumWidth(80)
            spin.valueChanged.connect(self._commit)
            row.addWidget(check, stretch=1)
            row.addWidget(spin)
            item.setSizeHint(widget.sizeHint())
            self._regions.setItemWidget(item, widget)
            self._spinners[level] = spin
            check.setProperty("level", level)

    # ---- state ----------------------------------------------------------------
    def _checks(self) -> dict[str, QCheckBox]:
        """Region name → its checkbox, read back out of the list's item widgets."""
        out: dict[str, QCheckBox] = {}
        for i in range(self._regions.count()):
            widget = self._regions.itemWidget(self._regions.item(i))
            if widget is None:
                continue
            box = widget.findChild(QCheckBox)
            if box is not None:
                out[str(box.property("level"))] = box
        return out

    def _current(self) -> RegionCombination | None:
        """The selected combination, or ``None`` when the list is empty."""
        row = self._list.currentRow()
        return self._combinations[row] if 0 <= row < len(self._combinations) else None

    def _on_selected(self, _row: int) -> None:
        """Load the selected definition into the form."""
        self._sync_form(self._current())

    def _sync_form(self, combo: RegionCombination | None) -> None:
        """Show *combo* in the form, or clear and disable it when there is none."""
        self._suspend = True
        try:
            enabled = combo is not None
            for widget in (self._name, self._value_column, self._op, self._mode,
                           self._require_all, self._regions):
                widget.setEnabled(enabled)
            if combo is None:
                self._name.setText("")
                for box in self._checks().values():
                    box.setChecked(False)
                self._preview.setStyleSheet(f"color: {COLOR_MUTED};")
                self._preview.setText(
                    "No definition selected — press Add to create one, or Add composites / "
                    "Add conservation for a ready-made family."
                )
                return
            self._name.setText(combo.name)
            index = self._value_column.findText(combo.value_column)
            if index >= 0:
                self._value_column.setCurrentIndex(index)
            self._op.setCurrentIndex(max(self._op.findData(combo.op), 0))
            self._mode.setCurrentIndex(max(self._mode.findData(combo.mode), 0))
            self._require_all.setChecked(combo.require_all)

            wanted = {str(k): float(v) for k, v in combo.terms.items()}
            from nvitk.stats.vessel_network import canonical_node

            canonical = {canonical_node(k) or k: v for k, v in wanted.items()}
            for level, box in self._checks().items():
                node = canonical_node(level)
                coefficient = wanted.get(level, canonical.get(node) if node else None)
                box.setChecked(coefficient is not None)
                if coefficient is not None:
                    self._spinners[level].setValue(float(coefficient))
        finally:
            self._suspend = False
        self._update_preview()

    def _commit(self, *_args: Any) -> None:
        """Write the form back onto the selected combination and re-preview."""
        if getattr(self, "_suspend", False):
            return
        row = self._list.currentRow()
        if not (0 <= row < len(self._combinations)):
            return
        terms = {
            level: float(self._spinners[level].value())
            for level, box in self._checks().items() if box.isChecked()
        }
        self._combinations[row] = RegionCombination(
            name=self._name.text().strip(),
            value_column=self._value_column.currentText().strip(),
            terms=terms,
            op=str(self._op.currentData() or "sum"),
            mode=str(self._mode.currentData() or "column"),
            region_column=self._region_column,
            require_all=self._require_all.isChecked(),
        )
        self._refresh_list(keep=row)
        self._update_preview()

    def _refresh_list(self, *, keep: int | None = None) -> None:
        """Redraw the definitions list, optionally keeping the selection."""
        self._list.blockSignals(True)
        self._list.clear()
        for combo in self._combinations:
            self._list.addItem(combo.name or "(unnamed)")
        if keep is not None and 0 <= keep < self._list.count():
            self._list.setCurrentRow(keep)
        self._list.blockSignals(False)

    def _update_preview(self) -> None:
        """Evaluate the selected combination against the real frame and report the outcome."""
        combo = self._current()
        if combo is None:
            self._preview.setText("")
            return
        try:
            values, report = evaluate_region_combination(self._frame, combo)
        except Exception as exc:
            self._preview.setStyleSheet(f"color: {COLOR_ERROR};")
            self._preview.setText(f"{combo.name or '(unnamed)'}: {exc}")
            return
        self._preview.setStyleSheet(f"color: {COLOR_MUTED};")
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        summary = (
            f"mean {numeric.mean():.4g} · SD {numeric.std():.4g} · "
            f"range {numeric.min():.4g}–{numeric.max():.4g}"
            if not numeric.empty else "no values"
        )
        self._preview.setText(
            f"{report['expression']}   →   {report['n_subjects']} subject(s), {summary}"
        )

    # ---- actions --------------------------------------------------------------
    def _on_add(self) -> None:
        """Append an empty definition and select it."""
        columns = self._numeric_columns()
        self._combinations.append(RegionCombination(
            name=f"combo{len(self._combinations) + 1}",
            value_column=columns[0] if columns else "",
            terms={}, region_column=self._region_column,
        ))
        self._refresh_list(keep=len(self._combinations) - 1)
        self._sync_form(self._current())

    def _on_remove(self) -> None:
        """Delete the selected definition."""
        row = self._list.currentRow()
        if 0 <= row < len(self._combinations):
            self._combinations.pop(row)
            self._refresh_list(keep=min(row, len(self._combinations) - 1))
            self._sync_form(self._current())

    def _prefill(self, factory: Any) -> None:
        """
        Add a ready-made family, skipping any whose regions this frame does not carry.

        Silently adding a balance that cannot be evaluated would put a permanently-failing entry in
        the list; the ones that do not apply are reported instead.
        """
        column = self._value_column.currentText().strip() or (
            self._numeric_columns()[0] if self._numeric_columns() else ""
        )
        existing = {c.name for c in self._combinations}
        added, skipped = 0, 0
        for combo in factory(column, region_column=self._region_column):
            if combo.name in existing:
                continue
            try:
                evaluate_region_combination(self._frame, combo)
            except Exception:
                skipped += 1
                continue
            self._combinations.append(combo)
            added += 1
        self._refresh_list(keep=len(self._combinations) - 1 if self._combinations else None)
        self._sync_form(self._current())
        if skipped:
            self._preview.setStyleSheet(f"color: {COLOR_MUTED};")
            self._preview.setText(
                f"Added {added}; skipped {skipped} whose vessels are not in this frame."
            )

    def combinations(self) -> list[RegionCombination]:
        """The edited definitions, dropping any that are not usable."""
        return [c for c in self._combinations if not c.validate()]


__all__ = ["RegionCombinationsDialog"]
