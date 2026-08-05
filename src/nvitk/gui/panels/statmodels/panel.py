"""Right-tab launcher for the floating Statmodels window."""

from __future__ import annotations

from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .constants import PIPELINE_KIND_ITEMS, PIPELINE_KIND_QVTPY
from .window import StatmodelsWindow


class StatmodelsPanel(QWidget):
    """Right-tab launcher for the floating Statmodels window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the pipeline-kind selector and the button that opens the floating explorer window."""
        super().__init__(parent)
        self._window: StatmodelsWindow | None = None

        self._pipeline_kind = QComboBox()
        for label, key in PIPELINE_KIND_ITEMS:
            self._pipeline_kind.addItem(label, key)

        self._btn = QPushButton("Open Statmodels window")
        self._btn.clicked.connect(self._open_window)

        hint = QLabel(
            "Explore mixed-effects models and mediation over 4D-flow, ASL, T1, FLAIR WMH or TOF "
            "morphometrics — several measurements at once — plus clinical / cognitive covariates "
            "from the dataset catalog. Models are saved under <dataset>/nvitk-statmodels/."
        )
        hint.setWordWrap(True)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Pipeline kind"))
        lay.addWidget(self._pipeline_kind)
        lay.addWidget(self._btn)
        lay.addWidget(hint)
        lay.addStretch(1)
        self.setLayout(lay)

    def _open_window(self) -> None:
        """Open (creating once, then reusing) the floating :class:`StatmodelsWindow`, selecting the
        chosen pipeline kind."""
        kind = str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)
        if self._window is None:
            self._window = StatmodelsWindow(initial_pipeline_kind=kind)
        else:
            self._window.set_pipeline_kind(kind)
        self._window.show_maximized_floating()


__all__ = ["StatmodelsPanel"]
