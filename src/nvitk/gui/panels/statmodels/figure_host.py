"""
A pane that shows either a Matplotlib figure or a Plotly one, and swaps cleanly between them.

Description
-----------
Every exploration dialog in this panel needs the same three things: hand a Plotly figure to a web
view, hand a Matplotlib figure to a canvas, and tear down whichever was there before without leaking
the previous figure. The logic is short but easy to get subtly wrong — a canvas removed from a layout
without ``deleteLater`` keeps its parent alive, and a figure dropped without ``plt.close`` leaks a
Matplotlib manager on every redraw — so it lives in one place rather than being re-typed per dialog.

Mixed in rather than wrapped in a widget: the host layout differs between dialogs (a controls row
above, a status label between, a splitter beside), and a widget that owned the layout would have to
grow a parameter for each of those.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any

from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from nvitk.core.logger import Logger

from .plotly_view import PlotlyView

log = Logger()


def _let_canvas_shrink(canvas: Any) -> None:
    """
    Allow a Matplotlib canvas to be smaller than the figure it was built at.

    A canvas reports its figure's inch size as its size hint. Left alone in a layout that is smaller
    than that, the hint wins and the pane grows past the window — which is how a 4-panel brain map
    pushes its own colourbar off the screen. Making the canvas expanding with a token minimum lets
    the layout drive the size, and Matplotlib then resizes the *figure* to match.
    """
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    canvas.setMinimumSize(120, 120)


class FigureHostMixin:
    """
    Matplotlib / Plotly figure hosting for a dialog.

    Call :meth:`_build_figure_host` once while laying the dialog out, then :meth:`_show_static` or
    :meth:`_show_interactive` per redraw. The two are mutually exclusive: showing one hides the
    other, so a dialog can offer an "Interactive" toggle without tracking which backend is live.
    """

    _view: PlotlyView
    _static_host: QWidget
    _static_layout: QVBoxLayout
    _static_canvas: Any
    _static_figure: Any

    def _build_figure_host(self, layout: QVBoxLayout, *, stretch: int = 1) -> None:
        """Add the web view and the canvas host to *layout*, with the canvas hidden initially."""
        self._view = PlotlyView()
        layout.addWidget(self._view, stretch=stretch)

        self._static_host = QWidget()
        self._static_layout = QVBoxLayout(self._static_host)
        self._static_layout.setContentsMargins(0, 0, 0, 0)
        self._static_canvas = None
        self._static_figure = None
        layout.addWidget(self._static_host, stretch=stretch)
        self._static_host.setVisible(False)

    # ---- backends -------------------------------------------------------------
    def _show_interactive(self, figure: Any) -> None:
        """Hand a Plotly figure to the web view and hide the static canvas."""
        self._clear_static()
        self._static_host.setVisible(False)
        self._view.setVisible(True)
        self._view.show_figure(figure)

    def _show_static(self, figure: Any) -> None:
        """Embed a Matplotlib figure and hide the web view."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from .theme import whiten_figure

        self._clear_static()
        whiten_figure(figure)
        canvas = FigureCanvasQTAgg(figure)
        _let_canvas_shrink(canvas)
        self._static_layout.addWidget(canvas)
        self._static_canvas, self._static_figure = canvas, figure
        canvas.draw_idle()
        self._view.setVisible(False)
        self._static_host.setVisible(True)

    def _clear_static(self) -> None:
        """Drop the current Matplotlib canvas and release its figure."""
        while self._static_layout.count():
            item = self._static_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self._static_figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._static_figure)
            except Exception as exc:
                log.debug("Could not close the previous figure: %s", exc)
        self._static_canvas = self._static_figure = None


__all__ = ["FigureHostMixin"]
