"""Matplotlib dock for oblique cross-section inspection."""

from __future__ import annotations

from typing import Any

import numpy as np

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

LAYER_NAME = "Cross-section (2D dock)"
DOCK_OBJECT_NAME = "nvitk_vessel_cross_section_dock"


class CrossSectionPanel(QWidget):
    """Dock widget showing intensity + mask overlay for one oblique plane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel("Click a centerline in 3D to view cross-section")
        self._title.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(self._title)
        self._canvas = None
        self._fig = None
        self._ax = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(3.2, 3.2), dpi=96)
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas, stretch=1)
        except Exception as exc:
            err = QLabel(f"Matplotlib unavailable: {exc}")
            err.setWordWrap(True)
            layout.addWidget(err)
        self.setLayout(layout)

    def clear(self, message: str = "") -> None:
        if message:
            self._title.setText(message)
        if self._ax is None or self._canvas is None:
            return
        self._ax.clear()
        self._ax.set_axis_off()
        self._canvas.draw_idle()

    def show_slice(
        self,
        intensity: np.ndarray,
        mask: np.ndarray | None,
        *,
        title: str,
    ) -> None:
        self._title.setText(title)
        if self._ax is None or self._canvas is None:
            return
        self._ax.clear()
        sl = np.asarray(intensity, dtype=np.float64)
        vmin = float(np.min(sl))
        vmax = float(np.max(sl))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._ax.imshow(
            sl,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if np.any(m):
                rgba = np.zeros((*m.shape, 4), dtype=np.float32)
                rgba[m, 0] = 1.0
                rgba[m, 3] = 0.35
                self._ax.imshow(rgba, origin="lower", interpolation="nearest")
        self._ax.set_axis_off()
        self._fig.tight_layout(pad=0.2)
        self._canvas.draw_idle()


def _napari_left_layer_docks(viewer: Any) -> tuple[Any | None, Any | None]:
    """Return (layer controls dock, layer list dock) on Napari's left edge."""
    try:
        qt_viewer = viewer.window._qt_viewer
    except Exception:
        return None, None
    controls = getattr(qt_viewer, "dockLayerControls", None)
    layer_list = getattr(qt_viewer, "dockLayerList", None)
    return controls, layer_list


def attach_cross_section_dock(viewer: Any, panel: CrossSectionPanel) -> Any:
    """
    Attach *panel* on the left: below Napari layer controls, above the layer list.
    """
    try:
        win = viewer.window._qt_window
    except Exception:
        try:
            win = viewer.window.qt_viewer.parent()
        except Exception:
            return None
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget, QSizePolicy

        for child in win.findChildren(QDockWidget):
            if child.objectName() == DOCK_OBJECT_NAME:
                child.setWidget(panel)
                child.show()
                child.raise_()
                return child

        dock = QDockWidget("Vessel cross-section", win)
        dock.setObjectName(DOCK_OBJECT_NAME)
        dock.setWidget(panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        panel.setMinimumSize(240, 220)
        dock.setMinimumWidth(240)
        dock.setMaximumWidth(340)

        controls, layer_list = _napari_left_layer_docks(viewer)
        win.addDockWidget(Qt.LeftDockWidgetArea, dock)
        if controls is not None and layer_list is not None:
            win.splitDockWidget(controls, dock, Qt.Vertical)
            win.splitDockWidget(dock, layer_list, Qt.Vertical)
        elif layer_list is not None:
            win.splitDockWidget(dock, layer_list, Qt.Vertical)
        elif controls is not None:
            win.splitDockWidget(controls, dock, Qt.Vertical)

        dock.show()
        dock.raise_()
        return dock
    except Exception:
        return None


__all__ = [
    "CrossSectionPanel",
    "DOCK_OBJECT_NAME",
    "attach_cross_section_dock",
]
