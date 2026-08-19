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
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.geometry import fit_dialog
from nvitk.stats.distribution_plots import column_panels_static, column_plot_static
from nvitk.stats.interactive import (
    COLUMN_FACET_MODES,
    COLUMN_PLOT_KINDS,
    column_panel_figure,
    column_plot,
)

from .figure_host import FigureHostMixin
from .theme import muted_label_style

log = Logger()

#: Columns offered as a split, in preference order — the ones a distribution usually needs.
_PREFERRED_SPLITS: tuple[str, ...] = ("territory", "group_key", "sex", "tacsctot_group")

#: Most levels a column may have and still be offered, per grouping mode. Overlaid violins become
#: slivers well before panels do, so the caps differ — and both must clear a full qvtpy vessel set
#: (13–17 levels), which an earlier cap of 12 did not, leaving the picker empty on real data.
_MAX_SPLIT_LEVELS: int = 24
_MAX_PANEL_LEVELS: int = 60


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

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)

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
        controls.addWidget(self._kind, stretch=1)

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
        lay.addLayout(controls)

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

        A numeric column is only offered when it is a Categorical — a continuous measurement has as
        many "levels" as rows and grouping by it is meaningless. ``subject_uid`` is excluded the same
        way, by the level cap.
        """
        out: list[str] = []
        for column in self._frame.columns:
            series = self._frame[column]
            if pd.api.types.is_numeric_dtype(series) and not isinstance(
                series.dtype, pd.CategoricalDtype
            ):
                continue
            if 2 <= int(series.nunique(dropna=True)) <= int(max_levels):
                out.append(str(column))
        return out

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

    def _redraw(self, *_args: Any) -> None:
        """Rebuild the figure from the current control state."""
        column = self._column_box.currentText().strip()
        kind = str(self._kind.currentData() or "violin")
        mode = str(self._facet_mode.currentData() or "")
        split = str(self._split.currentData() or "") if mode else ""
        sub = str(self._sub_split.currentData() or "") if mode in {"panels", "anatomical"} else ""
        interactive = self._interactive.isChecked()
        panelled = mode in {"panels", "anatomical"} and bool(split)
        try:
            if not interactive:
                figure = (
                    column_panels_static(
                        self._frame, column, facet_by=split, kind=kind, group=sub,
                        excluded_mask=self._excluded,
                        show_excluded=self._show_excluded.isChecked(),
                        anatomical=mode == "anatomical",
                        title=f"{column} by {split}" + (f", split by {sub}" if sub else ""),
                    )
                    if panelled else
                    column_plot_static(
                        self._frame, column, kind=kind, group=split,
                        excluded_mask=self._excluded,
                        show_excluded=self._show_excluded.isChecked(),
                        title=f"{column}" + (f" by {split}" if split else ""),
                    )
                )
                self._show_static(figure)
                self.setWindowTitle(f"Distribution — {column}")
                self._status.setText(
                    "Matplotlib rendering. Tick Interactive for hover, zoom and per-point identity."
                )
                return
            if panelled:
                figure = column_panel_figure(
                    self._frame, column, facet_by=split, kind=kind, group=sub,
                    hover_columns=self._hover_columns(), excluded_mask=self._excluded,
                    show_excluded=self._show_excluded.isChecked(),
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
        self.setWindowTitle(f"Distribution — {column}")
        self._status.setText(
            "Hover a point for its subject and territory. Drag to zoom, double-click to reset."
        )

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
