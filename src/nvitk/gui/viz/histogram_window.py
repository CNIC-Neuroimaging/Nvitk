"""A read-only intensity histogram for the active layer.

Its own window rather than a dock: a histogram is something you glance at while
deciding a threshold or a window, then close. Docking it would cost a panel slot
permanently for something consulted in bursts.
"""

from __future__ import annotations

from typing import Any, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_MUTED,
    SPACE_TIGHT,
    apply_theme,
    style_image_figure,
)

#: Kept on the viewer so a second request re-uses the window instead of stacking
#: one per invocation.
_WINDOW_ATTR = "_nvitk_histogram_window"


class HistogramWindow(QDialog):
    """A histogram plot with the layer's display window marked on it."""

    def __init__(self, parent: Any = None) -> None:
        """Build the canvas."""
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.setMinimumSize(520, 320)
        # Modeless: the point is to read it *while* adjusting something else.
        self.setModal(False)

        self._caption = QLabel("")
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(f"color: {COLOR_MUTED};")

        root = QVBoxLayout(self)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._caption)

        self._fig = self._ax = self._canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5.2, 3.0), dpi=96, layout="constrained")
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            root.addWidget(self._canvas, stretch=1)
        except Exception as exc:  # noqa: BLE001
            root.addWidget(QLabel(f"Matplotlib unavailable: {exc}"))

    def show_counts(
        self,
        counts: Sequence[float],
        edges: Sequence[float],
        *,
        title: str = "",
        limits: Sequence[float] = (),
        log_y: bool = True,
    ) -> None:
        """Draw *counts* over the bin *edges*, marking the display *limits*."""
        self.setWindowTitle(title or "Histogram")
        if self._ax is None or self._canvas is None:
            return
        ax = self._ax
        ax.clear()
        centres = [(float(edges[i]) + float(edges[i + 1])) / 2.0 for i in range(len(counts))]
        width = (float(edges[-1]) - float(edges[0])) / max(len(counts), 1)
        ax.bar(centres, list(counts), width=width, color=COLOR_MUTED, linewidth=0)
        if len(limits) == 2:
            for value in limits:
                ax.axvline(float(value), color=COLOR_ACCENT, lw=1.2)
        if log_y:
            # Background dominates a medical histogram by orders of magnitude; on a
            # linear axis everything else is a flat line along the bottom.
            ax.set_yscale("log")
        ax.set_xlabel("intensity", fontsize=8)
        ax.set_ylabel("voxels", fontsize=8)
        ax.tick_params(labelsize=7)
        style_image_figure(self._fig)
        self._canvas.draw_idle()

    def set_caption(self, text: str) -> None:
        """Set the line above the plot."""
        self._caption.setText(text)


def show_histogram_window(
    viewer: Any,
    counts: Sequence[float],
    edges: Sequence[float],
    *,
    title: str = "",
    limits: Sequence[float] = (),
) -> HistogramWindow:
    """Show *counts* in the viewer's histogram window, creating it if needed."""
    window = getattr(viewer, _WINDOW_ATTR, None)
    if window is None:
        parent = None
        try:
            parent = viewer.window._qt_window
        except Exception:  # noqa: BLE001 — a parentless dialog still works
            parent = None
        window = HistogramWindow(parent)
        apply_theme(window)
        try:
            setattr(viewer, _WINDOW_ATTR, window)
        except Exception:  # noqa: BLE001
            pass
    window.show_counts(counts, edges, title=title, limits=limits)
    window.set_caption(title)
    window.show()
    window.raise_()
    return window


__all__ = ["HistogramWindow", "show_histogram_window"]
