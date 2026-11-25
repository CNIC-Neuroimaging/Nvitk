"""eICAB / TOF morphometrics review: OK / FAIL per vessel and measurement group, saved to the DB.

The counterpart of the QVTPy measurement review (:mod:`nvitk.gui.viz.qc_review_panel`) for the
TOF pipeline: one row per vessel (``region_id``) and measurement group, with the subject's values
from ``image_measurements`` (pipeline ``tof_morpho_v2``), a status and a comment. Saving writes
``qc_status`` (and ``qc_comment`` where the table has it) onto every variable of the group for
that vessel, through :func:`nvitk.db.qvtpy_qc.publish_qvtpy_qc_reviews`, the same writer the
QVTPy review uses; :func:`subject_review_status` gives the subject list its ✓ / ◑ tags.
"""

from __future__ import annotations

import getpass
from typing import Any, Callable

import pandas as pd
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.widgets import hint_label

#: Measurement pipeline reviewed here.
TOF_PIPELINE_ID = "tof_morpho_v2"

#: Review group -> the variables one OK / FAIL covers (those a subject has are used).
TOF_QC_GROUPS: dict[str, tuple[str, ...]] = {
    "Geometry": ("volume_mm3", "centerline_length_mm", "radius_mean_mm", "diameter_mean_mm",
                 "diameter_median_mm", "tortuosity_dm", "surface_area_mm2"),
    "Lesions": ("has_stenosis", "has_enlargement", "stenosis_diameter_percent_max",
                "enlargement_diameter_percent_max", "stenosis_percent_max"),
    "Variants": ("has_hypoplasia", "has_fetal_pca"),
}
#: Values shown in the table per group (the first few that exist).
_SHOWN: dict[str, tuple[str, ...]] = {
    "Geometry": ("volume_mm3", "centerline_length_mm", "diameter_mean_mm", "tortuosity_dm"),
    "Lesions": ("has_stenosis", "stenosis_diameter_percent_max", "has_enlargement", "enlargement_diameter_percent_max"),
    "Variants": ("has_hypoplasia", "has_fetal_pca"),
}
_SHORT = {
    "volume_mm3": "vol", "centerline_length_mm": "len", "diameter_mean_mm": "Ø", "tortuosity_dm": "DM",
    "has_stenosis": "sten", "stenosis_diameter_percent_max": "sten %", "has_enlargement": "enl",
    "enlargement_diameter_percent_max": "enl %", "has_hypoplasia": "hypopl", "has_fetal_pca": "fetal PCA",
}
STATUSES = ("PENDING", "OK", "FAIL")


def review_variables() -> list[str]:
    """Every variable a TOF review can set."""
    return sorted({v for vs in TOF_QC_GROUPS.values() for v in vs})


def subject_review_status(repo: Any = None) -> dict[str, str]:
    """``{subject_uid: revised | partial | pending}`` for the TOF review."""
    from nvitk.db.qvtpy_qc import subject_qc_status_summary

    return subject_qc_status_summary(repo, pipeline_id=TOF_PIPELINE_ID, variables=review_variables())


def subject_rows(subject_uid: str, repo: Any = None) -> pd.DataFrame:
    """The subject's reviewed TOF measurement rows (``region_id``, ``variable_id``, values, status)."""
    from nvitk.db.repo import get_repo

    repo = repo or get_repo()
    frame = repo.get(
        "image_measurements",
        cohort_id=False,
        filters={"subject_uid": str(subject_uid), "pipeline_id": TOF_PIPELINE_ID, "variable_id": review_variables()},
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["region_id", "variable_id", "value_num", "value_text", "qc_status"])
    return frame[frame["subject_uid"].astype(str) == str(subject_uid)]


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


class TofReviewPanel(QWidget):
    """Review table for one subject's TOF morphometrics."""

    def __init__(self, parent: QWidget | None = None, *, on_revised: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.on_revised = on_revised
        self._subject = ""
        self._rows: pd.DataFrame = pd.DataFrame()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)
        self._status = hint_label("Load an eICAB subject to review its vessel measurements.")
        root.addWidget(self._status)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Vessel", "Group", "Values", "Status", "Comment"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self._table.horizontalHeader()
        for c in (0, 1, 3):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setMinimumHeight(120)
        root.addWidget(self._table, 1)
        row = QHBoxLayout()
        self._all_ok = QPushButton("Mark pending as OK")
        self._all_ok.clicked.connect(self.mark_pending_ok)
        self._save = QPushButton("Save review")
        self._save.clicked.connect(self.save)
        row.addWidget(self._all_ok)
        row.addStretch(1)
        row.addWidget(self._save)
        root.addLayout(row)
        self._set_enabled(False)

    def _set_enabled(self, on: bool) -> None:
        self._all_ok.setEnabled(on)
        self._save.setEnabled(on)

    def clear(self, message: str = "") -> None:
        self._table.setRowCount(0)
        self._subject = ""
        self._set_enabled(False)
        if message:
            self._status.setText(message)

    def load_subject(self, subject_uid: str, repo: Any = None) -> int:
        """Fill the table for *subject_uid*; returns the number of rows."""
        self._subject = str(subject_uid)
        try:
            self._rows = subject_rows(subject_uid, repo)
        except Exception as exc:  # noqa: BLE001 (no dataset, no measurements table)
            self.clear(f"No TOF measurements for {subject_uid}: {exc}")
            return 0
        rows = self._rows
        self._table.setRowCount(0)
        if rows.empty:
            self.clear(f"No {TOF_PIPELINE_ID} measurements for {subject_uid} in the dataset.")
            return 0
        by_region = {str(r): g for r, g in rows.groupby(rows["region_id"].astype(str), sort=False)}
        entries = []
        for region in sorted(by_region, key=lambda r: (r.startswith("label_"), r)):
            group_rows = by_region[region]
            present = set(group_rows["variable_id"].astype(str))
            for group, variables in TOF_QC_GROUPS.items():
                if present & set(variables):
                    entries.append((region, group, group_rows[group_rows["variable_id"].astype(str).isin(variables)]))
        self._table.setRowCount(len(entries))
        for r, (region, group, part) in enumerate(entries):
            values = {str(v): (n if n == n and n is not None else t) for v, n, t in zip(
                part["variable_id"], part.get("value_num", pd.Series([None] * len(part))),
                part.get("value_text", pd.Series([None] * len(part))))}
            text = "  ".join(f"{_SHORT.get(k, k)} {_fmt(values[k])}" for k in _SHOWN[group] if k in values)
            statuses = {str(s).strip().upper() for s in part.get("qc_status", pd.Series(dtype=str)).dropna()}
            status = "FAIL" if "FAIL" in statuses else "OK" if statuses == {"OK"} else "PENDING"
            comments = [str(c) for c in part.get("qc_comment", pd.Series(dtype=str)).dropna() if str(c).strip()]
            self._table.setItem(r, 0, QTableWidgetItem(region))
            self._table.setItem(r, 1, QTableWidgetItem(group))
            item = QTableWidgetItem(text)
            item.setToolTip("\n".join(f"{k} = {_fmt(v)}" for k, v in values.items()))
            self._table.setItem(r, 2, item)
            combo = QComboBox()
            combo.addItems(STATUSES)
            combo.setCurrentText(status)
            self._table.setCellWidget(r, 3, combo)
            self._table.setItem(r, 4, QTableWidgetItem(comments[0] if comments else ""))
        self._set_enabled(True)
        self._status.setText(f"{subject_uid}: {len(entries)} vessel group(s) from {TOF_PIPELINE_ID}.")
        return len(entries)

    def mark_pending_ok(self) -> None:
        for r in range(self._table.rowCount()):
            combo = self._table.cellWidget(r, 3)
            if combo is not None and combo.currentText() == "PENDING":
                combo.setCurrentText("OK")

    def decisions(self) -> list[Any]:
        """One decision per variable of every reviewed (vessel, group)."""
        from nvitk.db.qvtpy_qc import QcReviewDecision

        out = []
        present = {(str(r), str(v)) for r, v in zip(self._rows.get("region_id", []), self._rows.get("variable_id", []))}
        for r in range(self._table.rowCount()):
            combo = self._table.cellWidget(r, 3)
            status = combo.currentText() if combo is not None else "PENDING"
            comment = self._table.item(r, 4).text() if self._table.item(r, 4) else ""
            if status == "PENDING" and not comment:
                continue
            region = self._table.item(r, 0).text()
            for variable in TOF_QC_GROUPS[self._table.item(r, 1).text()]:
                if (region, variable) in present:
                    out.append(QcReviewDecision(variable_id=variable, region_id=region, qc_status=status, comment=comment))
        return out

    def save(self) -> dict[str, int] | None:
        """Write the review to ``image_measurements``; returns the writer's counts."""
        from nvitk.db.qvtpy_qc import publish_qvtpy_qc_reviews
        from nvitk.gui.core.activity import activity

        if not self._subject:
            return None
        decisions = self.decisions()
        if not decisions:
            self._status.setText("Nothing to save: every row is still PENDING.")
            return None
        try:
            with activity(f"Saving the TOF review of {self._subject}"):
                result = publish_qvtpy_qc_reviews(subject_uid=self._subject, decisions=decisions,
                                                  reviewer=getpass.getuser(), pipeline_id=TOF_PIPELINE_ID,
                                                  build_sqlite_index=True)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not save the review: {exc}")
            return None
        self._status.setText(f"Saved: {result.get('updated', 0)} measurement row(s) of {self._subject} updated.")
        if self.on_revised is not None:
            self.on_revised()
        return result


__all__ = ["TOF_PIPELINE_ID", "TOF_QC_GROUPS", "TofReviewPanel", "review_variables", "subject_review_status", "subject_rows"]
