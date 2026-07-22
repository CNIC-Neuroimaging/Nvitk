"""Matplotlib dock for oblique cross-section inspection."""

from __future__ import annotations

from typing import Any

import numpy as np

from qtpy.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from nvitk.gui.viz.left_dock import attach_left_inspection_dock

LAYER_NAME = "Cross-section (2D dock)"
DOCK_OBJECT_NAME = "nvitk_vessel_cross_section_dock"


class CrossSectionPanel(QWidget):
    """Dock widget showing intensity + mask overlay and optional flow waveforms."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel("Click a centerline in 3D to view cross-section")
        self._title.setWordWrap(True)
        layout = QVBoxLayout()
        self._pick_toggle = QCheckBox("Pick cross-section on click")
        self._pick_toggle.setChecked(True)
        self._pick_toggle.setToolTip(
            "Uncheck to freely rotate / pan / zoom the 3D view without triggering "
            "a cross-section pick on left-click."
        )
        layout.addWidget(self._pick_toggle)
        layout.addWidget(self._title)
        self._slice_canvas = None
        self._slice_fig = None
        self._slice_ax = None
        self._wave_canvas = None
        self._wave_fig = None
        self._wave_ax = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._slice_fig = Figure(figsize=(3.2, 3.0), dpi=96)
            self._slice_ax = self._slice_fig.add_subplot(111)
            self._slice_canvas = FigureCanvasQTAgg(self._slice_fig)
            layout.addWidget(self._slice_canvas, stretch=2)

            self._wave_fig = Figure(figsize=(3.2, 2.0), dpi=96)
            self._wave_ax = self._wave_fig.add_subplot(111)
            self._wave_canvas = FigureCanvasQTAgg(self._wave_fig)
            layout.addWidget(self._wave_canvas, stretch=1)
        except Exception as exc:
            err = QLabel(f"Matplotlib unavailable: {exc}")
            err.setWordWrap(True)
            layout.addWidget(err)
        self.setLayout(layout)

    def is_picking_enabled(self) -> bool:
        """True when left-click should trigger a cross-section pick."""
        return bool(self._pick_toggle.isChecked())

    def set_picking_enabled(self, enabled: bool) -> None:
        self._pick_toggle.setChecked(bool(enabled))

    def clear(self, message: str = "") -> None:
        if message:
            self._title.setText(message)
        if self._slice_ax is not None and self._slice_canvas is not None:
            self._slice_ax.clear()
            self._slice_ax.set_axis_off()
            self._slice_canvas.draw_idle()
        if self._wave_ax is not None and self._wave_canvas is not None:
            self._wave_ax.clear()
            self._wave_ax.set_axis_off()
            self._wave_canvas.draw_idle()

    def show_slice(
        self,
        intensity: np.ndarray,
        mask: np.ndarray | None,
        *,
        title: str,
        waveforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self._title.setText(title)
        if self._slice_ax is None or self._slice_canvas is None:
            return
        self._slice_ax.clear()
        sl = np.asarray(intensity, dtype=np.float64)
        vmin = float(np.min(sl))
        vmax = float(np.max(sl))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._slice_ax.imshow(
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
                self._slice_ax.imshow(rgba, origin="lower", interpolation="nearest")
        self._slice_ax.set_axis_off()
        self._slice_fig.tight_layout(pad=0.2)
        self._slice_canvas.draw_idle()
        self._draw_waveforms(waveforms)

    def _draw_waveforms(self, waveforms: list[dict[str, Any]] | None) -> None:
        if self._wave_ax is None or self._wave_canvas is None:
            return
        self._wave_ax.clear()
        if not waveforms:
            self._wave_ax.set_axis_off()
            self._wave_canvas.draw_idle()
            return
        phases = None
        for item in waveforms:
            flow = np.asarray(item.get("flow_ml_s", []), dtype=np.float64).reshape(-1)
            if flow.size == 0:
                continue
            if phases is None:
                phases = np.arange(flow.size, dtype=np.float64)
            offset = int(item.get("offset", 0))
            index = int(item.get("index", 0))
            if offset == 0:
                color = "#1f3b73"
                label = f"selected (idx {index})"
                lw = 2.2
            elif offset < 0:
                color = "#d62728"
                label = f"{offset:+d} (idx {index})"
                lw = 1.4
            else:
                color = "#2ca02c"
                label = f"{offset:+d} (idx {index})"
                lw = 1.4
            self._wave_ax.plot(phases, flow, color=color, lw=lw, label=label)
        self._wave_ax.set_xlabel("cardiac phase")
        self._wave_ax.set_ylabel("Q (ml/s)")
        self._wave_ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        self._wave_ax.grid(True, alpha=0.25)
        self._wave_fig.tight_layout(pad=0.3)
        self._wave_canvas.draw_idle()


def attach_cross_section_dock(viewer: Any, panel: CrossSectionPanel) -> Any:
    """Attach *panel* on Napari's left edge."""
    return attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="Vessel cross-section",
        minimum_width=280,
    )


__all__ = [
    "CrossSectionPanel",
    "DOCK_OBJECT_NAME",
    "attach_cross_section_dock",
]
