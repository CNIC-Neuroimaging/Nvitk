"""Right-tab launcher for the floating Statmodels window."""

from __future__ import annotations

from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .constants import PIPELINE_KIND_ITEMS, PIPELINE_KIND_QVTPY
from .sessions import StatmodelsShell


class StatmodelsPanel(QWidget):
    """Right-tab launcher for the floating Statmodels window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the pipeline-kind selector and the button that opens the floating explorer window."""
        super().__init__(parent)
        self._shell: StatmodelsShell | None = None

        self._pipeline_kind = QComboBox()
        for label, key in PIPELINE_KIND_ITEMS:
            self._pipeline_kind.addItem(label, key)

        self._btn = QPushButton("Open Statmodels window")
        self._btn.clicked.connect(self._open_window)

        hint = QLabel(
            "Explore mixed-effects models and mediation over 4D-flow, ASL, T1, FLAIR WMH or TOF "
            "morphometrics — several measurements at once — plus clinical / cognitive covariates "
            "from the dataset catalog. Each tab is an independent session with its own dataframe "
            "and model; right-click a session's tab to pull its dataframe into the current one. "
            "Models are saved under db.statmodels_root."
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
        """Open (creating once, then reusing) the floating :class:`StatmodelsShell`, pointing the
        current session at the chosen pipeline kind."""
        kind = str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)
        if self._shell is None:
            self._shell = StatmodelsShell(initial_pipeline_kind=kind)
        else:
            self._shell.set_pipeline_kind(kind)
        self._shell.show_maximized_floating()


__all__ = ["StatmodelsPanel"]
