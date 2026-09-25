"""
Distribution viewer for one column of the analysis dataframe.

Description
-----------
Before fitting anything you want to know what a column looks like: is it bimodal because two
territories are pooled, is the tail a real effect or three bad segmentations, does the binning make
sense. That is a different question from the model plot, so it gets its own non-modal window — you
can leave it open beside the main one and keep changing the plot type or the split.

Interactive throughout: hovering a point names the subject and its territory, which is what turns
"there is an outlier" into "sub-0142's left MCA".
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.flow_layout import FlowRow
from nvitk.gui.core.geometry import fit_dialog
from nvitk.stats.distribution_plots import column_panels_static, column_plot_static
from nvitk.stats.group_counts import counts_note, displayed_counts, level_strings
from nvitk.stats.interactive import (
    COLUMN_FACET_MODES,
    COLUMN_PLOT_KINDS,
    column_panel_figure,
    column_plot,
)

from .figure_host import FigureHostMixin
from .helpers import grouping_columns
from .theme import muted_label_style

log = Logger()

#: Columns offered as a split, in preference order — the ones a distribution usually needs.
_PREFERRED_SPLITS: tuple[str, ...] = ("territory", "group_key", "sex", "tacsctot_group")

#: Most levels a column may have and still be offered, per grouping mode. Overlaid violins become
#: slivers well before panels do, so the caps differ — and both must clear a full qvtpy vessel set
#: (13–17 levels), which an earlier cap of 12 did not, leaving the picker empty on real data.
_MAX_SPLIT_LEVELS: int = 24
_MAX_PANEL_LEVELS: int = 60


class LevelOrderDialog(QDialog):
    """
    Reorder the levels a distribution is split or panelled by.

    The default order is a natural sort, which is right for ``g0 … g3`` and wrong for anything
    whose meaning is not alphabetical — a severity scale, a vessel sequence following the
    circulation, a control group that belongs first. Presets cover the orders worth computing;
    the arrows cover the rest.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        column: str,
        levels: Sequence[str],
        frame: pd.DataFrame | None = None,
        value_column: str = "",
    ) -> None:
        """Build the list over *levels*, with presets computed from *value_column* where given."""
        super().__init__(parent)
        self.setWindowTitle(f"Order — {column}")
        fit_dialog(self, 340, 420, minimum=(300, 300))
        self._frame = frame
        self._column = column
        self._value = value_column

        lay = QVBoxLayout(self)
        hint = QLabel(f"Drag or use the arrows to order {column}'s levels.")
        hint.setWordWrap(True)
        hint.setStyleSheet(muted_label_style())
        lay.addWidget(hint)

        self._list = QListWidget()
        # Dragging is the gesture people try first; the arrows stay for keyboard use and for
        # the single-step nudge a drag makes fiddly.
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        for level in levels:
            self._list.addItem(str(level))
        lay.addWidget(self._list, stretch=1)

        arrows = QHBoxLayout()
        for label, delta in (("↑", -1), ("↓", 1)):
            button = QPushButton(label)
            button.setFixedWidth(40)
            button.clicked.connect(lambda _checked=False, d=delta: self._move(d))
            arrows.addWidget(button)
        arrows.addStretch(1)
        lay.addLayout(arrows)

        presets = FlowRow()
        presets_lay = presets.flow()
        for label, key, tip in (
            ("Natural", "natural", "g2 before g10 — the default."),
            ("A–Z", "alpha", "Plain alphabetical."),
            ("By median", "median", "Ascending median of the plotted column."),
            ("By count", "count", "Most observations first."),
            ("Reverse", "reverse", "Flip the current order."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, k=key: self._apply_preset(k))
            presets_lay.addWidget(button)
        lay.addWidget(presets)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def levels(self) -> list[str]:
        """The order the user settled on."""
        return [self._list.item(i).text() for i in range(self._list.count())]

    def _move(self, delta: int) -> None:
        """Shift the selected level one place, keeping it selected."""
        row = self._list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self._list.count():
            return
        self._list.insertItem(target, self._list.takeItem(row))
        self._list.setCurrentRow(target)

    def _apply_preset(self, key: str) -> None:
        """Reorder by one of the computed orders, leaving the list selected where it was."""
        from nvitk.stats.region_groups import natural_level_key

        current = self.levels()
        if key == "reverse":
            ordered = list(reversed(current))
        elif key == "alpha":
            ordered = sorted(current)
        elif key in {"median", "count"} and self._frame is not None and self._value:
            ordered = self._by_statistic(current, key)
        else:
            ordered = sorted(current, key=natural_level_key)

        self._list.clear()
        for level in ordered:
            self._list.addItem(level)

    def _by_statistic(self, levels: Sequence[str], key: str) -> list[str]:
        """Levels ranked by the median of the plotted column, or by how many rows they have."""
        frame = self._frame
        if frame is None or self._column not in frame.columns:
            return list(levels)
        keys = level_strings(frame[self._column])
        values = pd.to_numeric(frame.get(self._value), errors="coerce")

        def rank(level: str) -> float:
            rows = keys == level
            if key == "count":
                # Negated so the biggest group sorts first, which is what "By count" means.
                return -float(rows.sum())
            inside = values.loc[rows].dropna() if values is not None else pd.Series(dtype=float)
            # A level with nothing to take a median of goes last rather than to the front, which
            # is where NaN would sort it.
            return float(inside.median()) if not inside.empty else float("inf")

        return sorted(levels, key=rank)


class ColumnPlotDialog(FigureHostMixin, QDialog):
    """
    Interactive distribution of one column, optionally split by another.

    Non-modal on purpose: it is an exploration window, not a step in a workflow.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        frame: pd.DataFrame,
        column: str,
        kind: str = "violin",
        excluded_mask: Any = None,
        default_directory: Path | None = None,
    ) -> None:
        """Build the controls and draw the first figure."""
        super().__init__(parent)
        self.setWindowTitle(f"Distribution — {column}")
        fit_dialog(self, 900, 640)
        # Non-modal, and it owns its own lifetime so closing it does not disturb the main window.
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._frame = frame
        self._column = column
        self._excluded = excluded_mask
        self._directory = default_directory
        # ``{column: [level, …]}``. Kept per column so switching the split and coming back does
        # not lose the order that was chosen for it.
        self._orders: dict[str, list[str]] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Flow, not horizontal: a QHBoxLayout's minimum width is the sum of its children, which
        # Qt then enforces as the *dialog's* — so the Order and Legend controls pushed the row
        # past the window and clipped the end of it. Wrapping keeps the minimum at the widest
        # single control. Same remedy as the plot pane's own options row.
        controls_row = FlowRow()
        controls = controls_row.flow()

        controls.addWidget(QLabel("Column"))
        self._column_box = QComboBox()
        for name in self._plottable_columns():
            self._column_box.addItem(name)
        index = self._column_box.findText(column)
        if index >= 0:
            self._column_box.setCurrentIndex(index)
        self._column_box.currentIndexChanged.connect(self._redraw)
        controls.addWidget(self._column_box)

        controls.addWidget(QLabel("Plot"))
        self._kind = QComboBox()
        for key, description in COLUMN_PLOT_KINDS.items():
            self._kind.addItem(description, key)
        kind_index = self._kind.findData(kind)
        if kind_index >= 0:
            self._kind.setCurrentIndex(kind_index)
        self._kind.currentIndexChanged.connect(self._redraw)
        self._kind.setMinimumWidth(200)
        controls.addWidget(self._kind)

        controls.addWidget(QLabel("Group"))
        self._facet_mode = QComboBox()
        for key, description in COLUMN_FACET_MODES.items():
            self._facet_mode.addItem(description.split("—")[0].strip(), key)
        self._facet_mode.setToolTip(
            "\n".join(f"{k or 'all'}: {v}" for k, v in COLUMN_FACET_MODES.items())
        )
        self._facet_mode.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._facet_mode)

        controls.addWidget(QLabel("by"))
        self._split = QComboBox()
        self._split.addItem("(none)", "")
        self._split.currentIndexChanged.connect(self._redraw)
        controls.addWidget(self._split)

        # Only meaningful once the levels are on separate axes: inside a panel you can still split
        # into coloured series by a second column, which is how "per territory, split by sex" works.
        self._sub_split = QComboBox()
        self._sub_split.addItem("(no sub-split)", "")
        self._sub_split.setToolTip(
            "Within each panel, split into one coloured series per level of this column."
        )
        self._sub_split.currentIndexChanged.connect(self._redraw)
        self._sub_split.setVisible(False)
        controls.addWidget(self._sub_split)

        self._btn_order = QPushButton("Order…")
        self._btn_order.setToolTip(
            "Reorder the levels on the axis — or the panels, when panelled.\n\n"
            "The default is a natural sort, which is right for g0 … g3 and wrong for anything "
            "whose meaning is not alphabetical."
        )
        self._btn_order.clicked.connect(self._on_edit_order)
        controls.addWidget(self._btn_order)

        self._show_legend = QCheckBox("Legend")
        self._show_legend.setChecked(True)
        self._show_legend.setToolTip(
            "Show or hide the plot legend.\n\n"
            "The violin, box and strip views name their levels on the axis instead, so there is "
            "no legend to hide there."
        )
        self._show_legend.stateChanged.connect(self._apply_legend_visibility)
        controls.addWidget(self._show_legend)

        self._show_excluded = QCheckBox("Grey excluded")
        self._show_excluded.setChecked(True)
        self._show_excluded.setToolTip(
            "Draw rows removed by the active filters in grey instead of hiding them, so you can "
            "see whether a filter took out a coherent cluster or scattered noise."
        )
        self._show_excluded.setEnabled(excluded_mask is not None)
        self._show_excluded.stateChanged.connect(self._redraw)
        controls.addWidget(self._show_excluded)

        # Matplotlib by default; Plotly is opt-in, as on the model plots.
        self._interactive = QCheckBox("Interactive")
        self._interactive.setToolTip(
            "Render with Plotly instead of Matplotlib: hover a point for its subject and "
            "territory, drag to zoom.\nMatplotlib is the default and exports as a publication "
            "figure."
        )
        self._interactive.stateChanged.connect(self._redraw)
        controls.addWidget(self._interactive)

        self._btn_export = QPushButton("Export PNG…")
        self._btn_export.clicked.connect(self._on_export)
        controls.addWidget(self._btn_export)
        lay.addWidget(controls_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(muted_label_style())
        lay.addWidget(self._status)

        # The static backend gets its own canvas host, swapped in and out beside the web view.
        self._build_figure_host(lay)
        self._sync_split_choices()
        preferred_mode = self._facet_mode.findData("split")
        if preferred_mode >= 0 and self._split.count() > 1:
            self._facet_mode.blockSignals(True)
            self._facet_mode.setCurrentIndex(preferred_mode)
            self._facet_mode.blockSignals(False)
        self._on_mode_changed()


    # ---- columns --------------------------------------------------------------
    def _plottable_columns(self) -> list[str]:
        """Every column worth a distribution — numeric first, then the categoricals."""
        numeric = [c for c in self._frame.columns if pd.api.types.is_numeric_dtype(self._frame[c])]
        other = [c for c in self._frame.columns if c not in numeric]
        return [str(c) for c in (*numeric, *other)]

    def _split_columns(self, max_levels: int = _MAX_SPLIT_LEVELS) -> list[str]:
        """
        Columns usable as a grouping: discrete, and few enough levels to stay readable.

        How many distinct values there are, not what dtype holds them — a factor is a factor
        whether it is spelled ``"M"``/``"F"`` or ``0``/``1``, and requiring a Categorical kept
        every numerically coded one out of this picker. ``subject_uid`` is excluded by the level
        cap, which is what excludes a continuous measurement too.
        """
        return grouping_columns(self._frame, cap=max_levels)

    def _sync_split_choices(self) -> None:
        """
        Repopulate the grouping pickers for the current mode, keeping the selection where possible.

        Panels tolerate far more levels than overlaid violins, so the list genuinely differs between
        modes rather than being one compromise cap.
        """
        mode = str(self._facet_mode.currentData() or "")
        cap = _MAX_PANEL_LEVELS if mode in {"panels", "anatomical"} else _MAX_SPLIT_LEVELS
        columns = self._split_columns(cap)

        for box, placeholder in ((self._split, "(none)"), (self._sub_split, "(no sub-split)")):
            current = str(box.currentData() or "")
            box.blockSignals(True)
            box.clear()
            box.addItem(placeholder, "")
            for name in columns:
                box.addItem(name, name)
            index = box.findData(current) if current else -1
            if index < 0 and box is self._split:
                # No prior choice: land on the column people group by most, so the window opens on
                # something useful rather than on the placeholder.
                index = next(
                    (box.findData(c) for c in _PREFERRED_SPLITS if box.findData(c) >= 0), -1
                )
            box.setCurrentIndex(index if index >= 0 else 0)
            box.blockSignals(False)

        if not columns:
            self._split.setToolTip(
                "No column in this frame has between 2 and "
                f"{cap} distinct values, so there is nothing to group by."
            )

    def _hover_columns(self) -> list[str]:
        """Identifier columns worth attaching to every point's hover box."""
        return [
            c for c in ("subject_uid", "territory", "group_key", "visit_id")
            if c in self._frame.columns
        ]

    # ---- drawing --------------------------------------------------------------
    def _on_mode_changed(self, *_args: Any) -> None:
        """Show only the controls the chosen grouping actually uses, then redraw."""
        self._sync_split_choices()
        mode = str(self._facet_mode.currentData() or "")
        self._split.setVisible(mode != "")
        self._sub_split.setVisible(mode in {"panels", "anatomical"})
        self._redraw()

    # ---- level order ----------------------------------------------------------
    def _is_categorical(self, column: str) -> bool:
        """Whether *column* is drawn as counts of its own levels rather than as a distribution."""
        if not column or column not in self._frame.columns:
            return False
        series = self._frame[column]
        return isinstance(series.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(
            series
        )

    def _ordered_by(self) -> str:
        """
        The column whose levels the Order button acts on.

        The split when there is one — its levels are the axis groups, or the panels. With no
        split, a categorical column is still drawn as counts of its own levels, and those are the
        axis groups; ordering them is the same request.
        """
        mode = str(self._facet_mode.currentData() or "")
        split = str(self._split.currentData() or "") if mode else ""
        if split:
            return split
        column = self._column_box.currentText().strip()
        return column if self._is_categorical(column) else ""

    def _present_levels(self, column: str) -> list[str]:
        """*column*'s levels in the order the plot would draw them without an override."""
        from nvitk.stats.distribution_plots import base_levels
        from nvitk.stats.group_counts import ordered_levels
        from nvitk.stats.region_groups import natural_level_key

        if not column or column not in self._frame.columns:
            return []
        # Through ``base_levels``, so a binned column opens on the order it was cut in rather
        # than on the order its rows happen to arrive in — which is also what the axis draws.
        seen = base_levels(self._frame[column])
        # Panels sort naturally; a split keeps first-appearance order. Show whichever the plot
        # will use, so the editor opens on what is on screen rather than on a third order.
        if str(self._facet_mode.currentData() or "") in {"panels", "anatomical"}:
            seen = sorted(seen, key=natural_level_key)
        return ordered_levels(seen, self._orders.get(column))

    def _sync_order_button(self) -> None:
        """Enable the Order button only when there is a grouping whose levels can be moved."""
        column = self._ordered_by()
        enabled = bool(column) and len(self._present_levels(column)) > 1
        self._btn_order.setEnabled(enabled)
        if enabled and self._orders.get(column):
            self._btn_order.setText("Order ✓")
        else:
            self._btn_order.setText("Order…")

    def _on_edit_order(self) -> None:
        """Open the reorder editor for the current grouping."""
        column = self._ordered_by()
        levels = self._present_levels(column)
        if len(levels) < 2:
            return
        editor = LevelOrderDialog(
            self, column=column, levels=levels,
            frame=self._frame, value_column=self._column_box.currentText().strip(),
        )
        if not editor.exec():
            return
        chosen = editor.levels()
        # A run that ends up back at the default is stored as no override, so the button stops
        # claiming an order is in force and a later reload picks up any new level naturally.
        if chosen == self._present_levels_default(column):
            self._orders.pop(column, None)
        else:
            self._orders[column] = chosen
        self._redraw()

    def _present_levels_default(self, column: str) -> list[str]:
        """*column*'s levels with no override applied — what "no order" looks like."""
        saved = self._orders.pop(column, None)
        try:
            return self._present_levels(column)
        finally:
            if saved is not None:
                self._orders[column] = saved

    # ---- legend ---------------------------------------------------------------
    def _apply_legend_visibility(self, *_args: Any) -> None:
        """Show or hide the legend on whichever backend is live, without redrawing the data."""
        visible = self._show_legend.isChecked()
        if self._static_figure is not None:
            for ax in self._static_figure.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_visible(visible)
            canvas = getattr(self, "_static_canvas", None)
            if canvas is not None:
                canvas.draw_idle()
            return
        try:
            self._view.set_legend_visible(visible)
        except Exception as exc:
            log.debug("Could not toggle the Plotly legend: %s", exc)

    def _counts_line(self, column: str, split: str, sub: str) -> str:
        """The N of everything on screen, as text.

        The figure carries each level's count on its own tick or legend entry, which is where it
        is wanted while reading a shape. This is the same set spelled out in one line: it survives
        a figure too crowded to label, it says what the *whole* display is drawn from, and it can
        be copied into a methods section. Faceting by one column and splitting inside the panels
        by another are reported together, since both decide what a single violin contains.
        """
        if column not in self._frame.columns:
            return ""
        # The mask always goes in; ``show_excluded`` is what decides whether those rows are drawn
        # greyed and counted apart, or not drawn and not counted. Withholding the mask instead
        # would count rows the figure is not showing — the two have to be told the same thing.
        shown = self._show_excluded.isChecked()
        try:
            overall = displayed_counts(
                self._frame, column, excluded=self._excluded, show_excluded=shown
            )
            total = overall[0] if overall else None
            lines = []
            for by in dict.fromkeys(b for b in (split, sub) if b):
                lines.append(
                    counts_note(
                        displayed_counts(
                            self._frame, column, group=by,
                            excluded=self._excluded, show_excluded=shown,
                            # In the order the axis draws them: a status line that lists the
                            # levels left to right is read against the figure, and one that
                            # lists them in a third order is read twice.
                            levels=self._present_levels(by) or None,
                        ),
                        # The whole is the same whichever way it is cut, so it leads the first
                        # line only rather than being repeated under every grouping.
                        total=total if not lines else None,
                        group=by,
                    )
                )
            if not lines:
                lines = [counts_note([], total=total)]
        except Exception as exc:
            log.debug("Could not count %s: %s", column, exc, exc_info=True)
            return ""
        return "\n".join(line for line in lines if line)

    def _redraw(self, *_args: Any) -> None:
        """Rebuild the figure from the current control state."""
        column = self._column_box.currentText().strip()
        kind = str(self._kind.currentData() or "violin")
        mode = str(self._facet_mode.currentData() or "")
        split = str(self._split.currentData() or "") if mode else ""
        sub = str(self._sub_split.currentData() or "") if mode in {"panels", "anatomical"} else ""
        interactive = self._interactive.isChecked()
        panelled = mode in {"panels", "anatomical"} and bool(split)
        # The Order button acts on the split column, which is the panel axis when panelled and
        # the x axis otherwise — so the same saved order feeds whichever parameter applies.
        chosen_order = self._orders.get(split) if split else None
        panel_order = chosen_order if panelled else None
        level_order = None if panelled else chosen_order
        sub_order = self._orders.get(sub) if sub else None
        # With no split, a categorical column is drawn as counts of its own levels — so the order
        # saved against *it* is the one the axis needs. Without this the Order button was enabled,
        # the editor saved, and the bars did not move.
        if not split:
            level_order = self._orders.get(column)
        try:
            if not interactive:
                figure = (
                    column_panels_static(
                        self._frame, column, facet_by=split, kind=kind, group=sub,
                        excluded_mask=self._excluded,
                        show_excluded=self._show_excluded.isChecked(),
                        panel_order=panel_order, level_order=sub_order,
                        anatomical=mode == "anatomical",
                        title=f"{column} by {split}" + (f", split by {sub}" if sub else ""),
                    )
                    if panelled else
                    column_plot_static(
                        self._frame, column, kind=kind, group=split,
                        excluded_mask=self._excluded,
                        show_excluded=self._show_excluded.isChecked(),
                        level_order=level_order,
                        title=f"{column}" + (f" by {split}" if split else ""),
                    )
                )
                self._show_static(figure)
                self._apply_legend_visibility()
                self._sync_order_button()
                self.setWindowTitle(f"Distribution — {column}")
                self._set_status(
                    column, split, sub,
                    "Matplotlib rendering. Tick Interactive for hover, zoom and per-point identity.",
                )
                return
            if panelled:
                figure = column_panel_figure(
                    self._frame, column, facet_by=split, kind=kind, group=sub,
                    hover_columns=self._hover_columns(), excluded_mask=self._excluded,
                    show_excluded=self._show_excluded.isChecked(),
                    panel_order=panel_order, level_order=sub_order,
                    anatomical=mode == "anatomical",
                    title=f"{column} by {split}" + (f", split by {sub}" if sub else ""),
                )
            else:
                figure = column_plot(
                    self._frame,
                    column,
                    kind=kind,
                    group=split,
                    hover_columns=self._hover_columns(),
                    excluded_mask=self._excluded,
                    show_excluded=self._show_excluded.isChecked(),
                    level_order=level_order,
                    title=f"{column}" + (f" by {split}" if split else ""),
                )
        except Exception as exc:
            log.debug("Column plot failed: %s", exc, exc_info=True)
            self._clear_static()
            self._view.setVisible(True)
            self._view.show_error(f"Cannot plot {column!r}: {exc}")
            self._status.setText("")
            return
        self._show_interactive(figure)
        self._apply_legend_visibility()
        self._sync_order_button()
        self.setWindowTitle(f"Distribution — {column}")
        self._set_status(
            column, split, sub,
            "Hover a point for its subject and territory. Drag to zoom, double-click to reset.",
        )

    def _set_status(self, column: str, split: str, sub: str, hint: str) -> None:
        """Put the counts first and the backend hint after — the N is what gets read."""
        counts = self._counts_line(column, split, sub)
        self._status.setText(f"{counts}\n{hint}" if counts else hint)

    def _on_export(self) -> None:
        """Write the current figure to a PNG the user chooses."""
        if not (self._view.has_figure() or self._static_figure is not None):
            return
        suggested = (self._directory or Path.home()) / f"{self._column_box.currentText()}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export distribution as PNG", str(suggested), "PNG image (*.png);;All (*)"
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        try:
            if self._static_figure is not None:
                self._static_figure.savefig(
                    target, dpi=200, bbox_inches="tight",
                    facecolor=self._static_figure.get_facecolor(),
                )
                written = target
            else:
                written = self._view.save_figure(target)
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
            return
        self._status.setText(f"Exported → {written}")


__all__ = ["ColumnPlotDialog"]
