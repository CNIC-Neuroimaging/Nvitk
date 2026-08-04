"""Docked cohort violin hemodynamics plots with selected-subject highlight."""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from nvitk.gui.viz.cross_section_panel import DOCK_OBJECT_NAME as XS_DOCK_OBJECT_NAME
from nvitk.gui.viz.hemo_plot_panel import DOCK_OBJECT_NAME as HEMO_DOCK_OBJECT_NAME
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.stats.violin_hemodynamics import (
    METRICS,
    PITC_PWV_SPECS,
    VESSEL_SPECS,
    draw_violin_figure,
    load_long_measurements,
    prepare_plot_frame,
)

DOCK_OBJECT_NAME = "nvitk_cohort_violin_dock"


class CohortViolinPanel(QWidget):
    """Interactive cohort violin plot with IQR rem + selected-subject highlight."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the metric selector, status label, and Matplotlib canvas/toolbar for the panel."""
        super().__init__(parent)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._canvas = None
        self._toolbar = None
        self._fig = None
        self._long_df = None
        self._highlight_subject: str | None = None
        self._outlier_rem = True

        layout = QVBoxLayout()
        layout.addWidget(self._title)

        self._metric_selector = QComboBox()
        for meta in METRICS:
            self._metric_selector.addItem(str(meta["title"]), meta["key"])
        self._metric_selector.currentIndexChanged.connect(self._render_current)
        layout.addWidget(self._metric_selector)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
            from matplotlib.figure import Figure
        except Exception:
            from matplotlib.backends.backend_qt5agg import (  # type: ignore
                FigureCanvasQTAgg,
                NavigationToolbar2QT,
            )
            from matplotlib.figure import Figure

        self._fig = Figure(figsize=(7.5, 5.5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)
        self.setLayout(layout)

    def set_data(
        self,
        *,
        highlight_subject: str,
        long_df: Any | None = None,
        pipeline_id: str = QVTPY_PIPELINE_ID,
        outlier_rem: bool = True,
    ) -> None:
        """Load cohort measurements and render the current metric."""
        self._highlight_subject = str(highlight_subject)
        self._outlier_rem = bool(outlier_rem)
        if long_df is not None:
            self._long_df = long_df
        else:
            variable_ids = sorted({m["variable_id"] for m in METRICS})
            try:
                self._long_df = load_long_measurements(
                    pipeline_id=pipeline_id,
                    variable_ids=variable_ids,
                )
            except Exception as exc:  # noqa: BLE001
                self._long_df = None
                self._status.setText(f"Failed to load cohort measurements: {exc}")
                self._title.setText("Cohort hemodynamics")
                if self._fig is not None and self._canvas is not None:
                    self._fig.clear()
                    self._canvas.draw_idle()
                return
        n = 0 if self._long_df is None else int(len(self._long_df))
        self._status.setText(
            f"Subject {self._highlight_subject} highlighted "
            f"(IQR outliers removed; subject points kept). "
            f"Cohort rows={n}."
        )
        self._render_current()

    def _render_current(self, _index: int | None = None) -> None:
        """Redraw the violin plot for the currently selected metric, or a placeholder message if no
        cohort data is loaded."""
        if self._fig is None or self._canvas is None:
            return
        key = self._metric_selector.currentData() or "flow"
        meta = next((m for m in METRICS if m["key"] == key), METRICS[0])
        self._title.setText(str(meta["title"]))
        if self._long_df is None or getattr(self._long_df, "empty", True):
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No cohort measurements available.", ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw_idle()
            return

        vessel_specs = VESSEL_SPECS if meta["kind"] == "loc" else PITC_PWV_SPECS
        plot_df = prepare_plot_frame(
            self._long_df,
            variable_id=str(meta["variable_id"]),
            specs=vessel_specs,
            derive_tcbf=bool(meta.get("derive_tcbf")),
        )
        fig = draw_violin_figure(
            plot_df,
            title=str(meta["title"]),
            ylabel=str(meta["ylabel"]),
            panel=str(meta["panel"]),
            outlier_rem=self._outlier_rem,
            highlight_subject=self._highlight_subject,
            fig=self._fig,
            figsize=(7.5, 5.5),
        )
        if fig is None:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data for this metric.", ha="center", va="center")
            ax.set_axis_off()
        if self._toolbar is not None:
            self._toolbar.update()
        self._canvas.draw_idle()


def attach_cohort_violin_dock(viewer: Any, panel: CohortViolinPanel) -> Any:
    """Attach or replace the cohort violin dock on the left."""
    return attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="Cohort flow violins",
        tabify_with=[
            XS_DOCK_OBJECT_NAME,
            HEMO_DOCK_OBJECT_NAME,
            "nvitk_morphometrics_dock",
            "nvitk_qc_measurements_dock",
        ],
        minimum_width=380,
    )


def show_cohort_violin(
    viewer: Any,
    *,
    highlight_subject: str,
    pipeline_id: str = QVTPY_PIPELINE_ID,
    outlier_rem: bool = True,
) -> CohortViolinPanel:
    """Create and show the cohort violin dock for the selected QC subject."""
    panel = CohortViolinPanel()
    panel.set_data(
        highlight_subject=highlight_subject,
        pipeline_id=pipeline_id,
        outlier_rem=outlier_rem,
    )
    attach_cohort_violin_dock(viewer, panel)
    return panel


__all__ = [
    "CohortViolinPanel",
    "DOCK_OBJECT_NAME",
    "attach_cohort_violin_dock",
    "show_cohort_violin",
]
