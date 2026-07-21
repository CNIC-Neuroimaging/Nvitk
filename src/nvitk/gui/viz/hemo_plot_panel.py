"""Docked display of the paper-style PITC and PWV diagnostic plots."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QDockWidget,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

DOCK_OBJECT_NAME = "nvitk_hemodynamics_plot_dock"


class HemodynamicsPlotPanel(QWidget):
    """Scrollable image panel for a generated PITC or PWV figure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scroll = QScrollArea()
        scroll.setWidget(self._image)
        scroll.setWidgetResizable(False)
        layout = QVBoxLayout()
        layout.addWidget(self._title)
        layout.addWidget(scroll, stretch=1)
        self.setLayout(layout)

    def show_plot(
        self,
        region_plot_data: dict[str, dict[str, Any]],
        *,
        mode: str,
    ) -> None:
        """Generate the existing measurement figure and display it."""
        if mode == "pitc":
            from nvitk.pipes.qvtpy.util.measure_plots import plot_pitc_figure

            plot_fn = plot_pitc_figure
            label = "PITC: quality and pulsatility index vs distance"
        else:
            from nvitk.pipes.qvtpy.util.measure_plots import plot_pwv_figure

            plot_fn = plot_pwv_figure
            label = "PWV: XCor delay, time-to-upstroke, and weights vs distance"
        self._title.setText(label)
        with tempfile.TemporaryDirectory(prefix="nvitk_hemo_plot_") as tmp:
            out = Path(tmp) / f"{mode}.png"
            generated = plot_fn(region_plot_data, out)
            if generated is None:
                self._image.clear()
                self._image.setText("No plot data available.")
                return
            pixmap = QPixmap(str(generated))
        if pixmap.isNull():
            self._image.clear()
            self._image.setText("Could not render the generated plot.")
            return
        self._image.setPixmap(pixmap)
        self._image.resize(pixmap.size())


def attach_hemodynamics_plot_dock(
    viewer: Any,
    panel: HemodynamicsPlotPanel,
    *,
    mode: str,
) -> Any:
    """Attach or replace the hemodynamics plot dock."""
    try:
        win = viewer.window._qt_window
    except Exception:
        try:
            win = viewer.window.qt_viewer.parent()
        except Exception:
            return None
    for child in win.findChildren(QDockWidget):
        if child.objectName() == DOCK_OBJECT_NAME:
            child.setWindowTitle(f"{mode.upper()} diagnostics")
            child.setWidget(panel)
            child.show()
            child.raise_()
            return child
    dock = QDockWidget(f"{mode.upper()} diagnostics", win)
    dock.setObjectName(DOCK_OBJECT_NAME)
    dock.setWidget(panel)
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    panel.setMinimumSize(360, 300)
    dock.setMinimumWidth(380)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    dock.show()
    dock.raise_()
    return dock


def show_hemodynamics_plot(
    viewer: Any,
    region_plot_data: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> Any:
    """Generate and show a PITC/PWV diagnostics dock."""
    panel = HemodynamicsPlotPanel()
    panel.show_plot(region_plot_data, mode=mode)
    return attach_hemodynamics_plot_dock(viewer, panel, mode=mode)


__all__ = [
    "DOCK_OBJECT_NAME",
    "HemodynamicsPlotPanel",
    "attach_hemodynamics_plot_dock",
    "show_hemodynamics_plot",
]
