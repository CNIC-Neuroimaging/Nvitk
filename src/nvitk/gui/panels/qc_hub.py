"""The QC tab: one place to review QVTPy, eICAB and PESA-Health results, subject by subject.

One pipeline choice at the top (QVTPy · 4D flow, eICAB · TOF, PESA-Health · CT-PET / Dixon) drives
two sections, laid out like the PESA-Health review:

* **Subject**: where the subjects come from and which one to load. QVTPy and eICAB read the XNAT
  catalog or local NIfTI / results roots (:class:`~nvitk.gui.panels.qc.QcPanel`); PESA-Health reads
  the pipeline's data preset (cohort folders) (:class:`~nvitk.gui.panels.pesa_health_qc.PesaHealthQcPanel`).
  Subjects already reviewed are tagged ✓ (all decided) or ◑ (some).
* **Review**: the loaded subject's measurements with a status and a comment per row, saved to
  the dataset:

  - QVTPy: per vessel / territory flow metrics, autoQC colouring, *Mark as revised*
    (:class:`~nvitk.gui.viz.qc_review_panel.QcMeasurementsPanel`), into
    ``image_measurements.qc_status`` (pipeline ``4dflow_v4``);
  - eICAB: per vessel geometry / lesion / variant groups of the TOF morphometrics
    (:class:`~nvitk.gui.viz.tof_qc_review.TofReviewPanel`), into ``image_measurements.qc_status``
    (pipeline ``tof_morpho_v2``);
  - PESA-Health: per structure segmentation / measurement, tinted by the expected ranges, into
    ``pesa_health_qc_reviews``.

The two sections share a splitter, so the subject list and the review table can be given the
room each needs.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QStackedWidget, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.widgets import Section, SegmentedButtons

#: (key, label, icon) of the pipelines the hub reviews.
PIPELINES: tuple[tuple[str, str, str], ...] = (
    ("qvtpy", "QVTPy · 4D flow", "wave"),
    ("eicab", "eICAB · TOF", "vessel"),
    ("pesa_health", "PESA-Health", "droplet"),
)
PREF_KEY = "qc.pipeline"


def _stored_pipeline() -> str:
    try:
        from nvitk.gui.core.prefs import load_prefs

        value = str(load_prefs().get(PREF_KEY) or "qvtpy")
    except Exception:  # noqa: BLE001
        value = "qvtpy"
    return value if value in {k for k, _l, _i in PIPELINES} else "qvtpy"


class QcHub(QWidget):
    """The unified QC panel (see the module docstring)."""

    def __init__(self, viewer: Any, app_state: dict[str, Any], *, on_inputs_opened: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from nvitk.gui.panels.pesa_health_qc import PesaHealthQcPanel
        from nvitk.gui.panels.qc import QcPanel
        from nvitk.gui.viz.qc_review_panel import QcMeasurementsPanel
        from nvitk.gui.viz.tof_qc_review import TofReviewPanel

        self._viewer = viewer
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        self._choice = SegmentedButtons([(k, label, icon, label) for k, label, icon in PIPELINES],
                                        current=_stored_pipeline())
        self._choice.setToolTip("The pipeline whose results are reviewed")
        root.addWidget(self._choice)

        # Reviews first: the subject panel is told where to put them.
        self.qvtpy_review = QcMeasurementsPanel()
        self.tof_review = TofReviewPanel()
        self.subjects = QcPanel(viewer, app_state, on_inputs_opened=on_inputs_opened, embedded=True,
                                review_panel=self.qvtpy_review, tof_review=self.tof_review)
        self.pesa_health = PesaHealthQcPanel(viewer, embedded=True)
        self.pesa_health.setParent(self)
        self.pesa_health.hide()

        self._pickers = QStackedWidget()
        self._pickers.addWidget(self.subjects)
        self._pickers.addWidget(self.pesa_health.picker)
        self._reviews = QStackedWidget()
        for widget in (self.qvtpy_review, self.tof_review, self.pesa_health.review):
            self._reviews.addWidget(widget)

        subject = Section("Subject", icon_name="database", hue="blue", pref_key="qc.subject")
        # The dataset the subjects are read from and the reviews saved to: switchable for the
        # whole GUI, as on the Data tab.
        from nvitk.gui.core.dataset_root import DatasetRootButton

        self._dataset_button = subject.add_header_widget(DatasetRootButton())
        subject.add(self._pickers, 1)
        review = Section("Review", icon_name="qc", hue="green", pref_key="qc.review")
        review.add(self._reviews, 1)
        self._split = QSplitter(Qt.Vertical)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(subject)
        self._split.addWidget(review)
        self._split.setStretchFactor(0, 2)
        self._split.setStretchFactor(1, 3)
        root.addWidget(self._split, 1)

        self._choice.changed.connect(self.set_pipeline)
        self.set_pipeline(self._choice.current(), remember=False)

    def pipeline(self) -> str:
        """Key of the pipeline being reviewed."""
        return self._choice.current()

    def set_pipeline(self, key: str, *, remember: bool = True) -> None:
        """Review pipeline *key* (``qvtpy``, ``eicab`` or ``pesa_health``)."""
        if self._choice.current() != key:
            self._choice.set_current(key)
        if key == "pesa_health":
            self._pickers.setCurrentIndex(1)
            self._reviews.setCurrentIndex(2)
            if self.isVisible():
                self.pesa_health.ensure_loaded()
        else:
            self._pickers.setCurrentIndex(0)
            self._reviews.setCurrentIndex(0 if key == "qvtpy" else 1)
            self.subjects.set_pipeline(key)
        if remember:
            try:
                from nvitk.gui.core.prefs import save_prefs

                save_prefs({PREF_KEY: key})
            except Exception:  # noqa: BLE001
                pass

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self.pipeline() == "pesa_health":
            self.pesa_health.ensure_loaded()


__all__ = ["PIPELINES", "QcHub"]
