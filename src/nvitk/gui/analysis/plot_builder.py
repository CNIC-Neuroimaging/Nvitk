"""Plot builder: choose a plot kind and columns, filter, facet, style; export the figure.

Bound to a DataFrame (a Results table, a database table, a Statmodels frame), it keeps one
:class:`~nvitk.gui.analysis.plot_spec.PlotSpec` and redraws whenever a choice changes. Styling
(palette, log axes, bins, title) sits in a pop-over to keep the bar compact.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from qtpy.QtCore import QSize, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.analysis.plot_spec import AGGREGATIONS, PALETTES, PLOT_KINDS, PlotSpec
from nvitk.gui.core import design
from nvitk.gui.core.widgets import IconButton, hint_label, popover_button

_NONE = "—"


class PlotBuilder(QWidget):
    """Interactive plot over a DataFrame."""

    def __init__(self, frame: pd.DataFrame | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._frame = pd.DataFrame()
        self.spec = PlotSpec()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        bar = QHBoxLayout()
        bar.setSpacing(design.SPACE_TIGHT)
        self._kind = QComboBox()
        for key, (label, _y, _n) in PLOT_KINDS.items():
            self._kind.addItem(label, key)
        self._x, self._y, self._hue, self._col = (QComboBox() for _ in range(4))
        self._agg = QComboBox()
        self._agg.addItems(AGGREGATIONS)
        for label, widget in (("", self._kind), ("x", self._x), ("y", self._y), ("colour", self._hue), ("facet", self._col), ("agg", self._agg)):
            if label:
                bar.addWidget(hint_label(label))
            bar.addWidget(widget, 1 if widget is not self._agg else 0)
            widget.currentIndexChanged.connect(self._changed)
        bar.addWidget(popover_button("sliders", "Style: palette, log axes, bins, title", self._style_panel, title="Plot style"))
        save = IconButton("save", "Save the figure (PNG, SVG, PDF)")
        save.clicked.connect(self._save)
        bar.addWidget(save)
        root.addLayout(bar)

        self._query = QLineEdit()
        self._query.setPlaceholderText("Filter rows (pandas query), e.g.  age > 50 and sex == 'F'")
        self._query.editingFinished.connect(self._changed)
        root.addWidget(self._query)

        self._figure = Figure(figsize=(7, 5), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(260)
        toolbar = NavigationToolbar2QT(self._canvas, self)
        toolbar.setIconSize(QSize(16, 16))
        root.addWidget(toolbar)
        root.addWidget(self._canvas, 1)
        self._message = hint_label("")
        root.addWidget(self._message)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.redraw)
        self._palette = "tab10"
        self._logx = self._logy = False
        self._bins = 30
        self._title = ""
        if frame is not None:
            self.set_frame(frame)

    # data
    def set_frame(self, frame: pd.DataFrame) -> None:
        """Plot *frame* from now on (column choices are kept where they still exist)."""
        from nvitk.gui.analysis.render_mpl import numeric_like

        frame = frame if frame is not None else pd.DataFrame()
        # Numbers stored as text (a spreadsheet column with a stray "n/a", values read from
        # value_text) are numbers here, so they are offered as y and plotted on a numeric axis.
        converted = {c: s for c in frame.columns if (s := numeric_like(frame[c])) is not None}
        self._frame = frame.assign(**{str(c): s for c, s in converted.items()}) if converted else frame
        columns = [str(c) for c in self._frame.columns]
        numeric = [str(c) for c in self._frame.select_dtypes("number").columns]
        categorical = [c for c in columns if c not in numeric]
        # Numeric codes with a few levels (sex 0/1, visit, a group id) can split facets too.
        few_levels = [c for c in numeric if isinstance(self._frame[c], pd.Series) and self._frame[c].nunique(dropna=True) <= 12]
        keep = {w: w.currentText() for w in (self._x, self._y, self._hue, self._col)}
        for combo, options, optional in (
            (self._x, columns, False), (self._y, numeric or columns, True),
            (self._hue, categorical + numeric, True), (self._col, categorical + few_levels, True),
        ):
            combo.blockSignals(True)
            combo.clear()
            if optional:
                combo.addItem(_NONE)
            combo.addItems(options)
            if keep[combo] and combo.findText(keep[combo]) >= 0:
                combo.setCurrentText(keep[combo])
            combo.blockSignals(False)
        survived = bool(keep[self._x]) and self._x.currentText() == keep[self._x]
        if not survived:
            # A sensible first plot: first categorical column against the first numeric one.
            if categorical and numeric:
                self._x.setCurrentText(categorical[0])
                self._y.setCurrentText(numeric[0])
                self._kind.setCurrentIndex(self._kind.findData("box"))
            elif len(numeric) >= 2:
                self._x.setCurrentText(numeric[0])
                self._y.setCurrentText(numeric[1])
                self._kind.setCurrentIndex(self._kind.findData("scatter"))
            elif numeric:
                self._x.setCurrentText(numeric[0])
                self._kind.setCurrentIndex(self._kind.findData("hist"))
        self._changed()

    def frame(self) -> pd.DataFrame:
        return self._frame

    # spec
    def _value(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == _NONE else text

    def _changed(self, *_args: Any) -> None:
        self.spec = PlotSpec(
            kind=str(self._kind.currentData()), x=self._value(self._x), y=self._value(self._y),
            hue=self._value(self._hue), col=self._value(self._col), agg=self._agg.currentText(),
            query=self._query.text(), logx=self._logx, logy=self._logy, palette=self._palette,
            bins=int(self._bins), title=self._title,
        )
        needs_y = PLOT_KINDS.get(self.spec.kind, ("", True, True))[1]
        self._y.setEnabled(needs_y or self.spec.kind == "heatmap")
        self._timer.start()

    def set_spec(self, spec: PlotSpec) -> None:
        """Show *spec* (columns that do not exist are left unset)."""
        self._kind.setCurrentIndex(max(self._kind.findData(spec.kind), 0))
        for combo, value in ((self._x, spec.x), (self._y, spec.y), (self._hue, spec.hue), (self._col, spec.col)):
            combo.setCurrentText(value or _NONE)
        self._agg.setCurrentText(spec.agg)
        self._query.setText(spec.query)
        self._palette, self._logx, self._logy, self._bins, self._title = spec.palette, spec.logx, spec.logy, spec.bins, spec.title
        self._changed()

    def redraw(self) -> None:
        """Draw the spec now."""
        from nvitk.gui.analysis.render_mpl import render

        if self._frame.empty:
            self._figure.clear()
            self._canvas.draw_idle()
            self._message.setText("No rows to plot.")
            return
        try:
            render(self.spec, self._frame, self._figure)
            self._message.setText("; ".join(getattr(self._figure, "nvitk_notes", [])))
        except Exception as exc:  # noqa: BLE001 (a half-chosen plot is not an error)
            self._figure.clear()
            self._message.setText(f"Cannot draw: {exc}")
        self._canvas.draw_idle()

    # style
    def _style_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        palette = QComboBox()
        palette.addItems(PALETTES)
        palette.setCurrentText(self._palette)
        palette.currentTextChanged.connect(lambda v: (setattr(self, "_palette", v), self._changed()))
        logx, logy = QCheckBox("log x"), QCheckBox("log y")
        logx.setChecked(self._logx)
        logy.setChecked(self._logy)
        logx.toggled.connect(lambda v: (setattr(self, "_logx", v), self._changed()))
        logy.toggled.connect(lambda v: (setattr(self, "_logy", v), self._changed()))
        bins = QSpinBox()
        bins.setRange(2, 500)
        bins.setValue(self._bins)
        bins.valueChanged.connect(lambda v: (setattr(self, "_bins", v), self._changed()))
        title = QLineEdit(self._title)
        title.editingFinished.connect(lambda: (setattr(self, "_title", title.text()), self._changed()))
        form.addRow("Palette", palette)
        row = QHBoxLayout()
        row.addWidget(logx)
        row.addWidget(logy)
        form.addRow("Axes", row)
        form.addRow("Bins", bins)
        form.addRow("Title", title)
        design.apply_theme(panel)
        return panel

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", "plot.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            self._figure.savefig(path, dpi=200)


__all__ = ["PlotBuilder"]
