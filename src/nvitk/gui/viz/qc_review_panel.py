"""QC measurement review dock: OK/FAIL per metric with optional comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.db.qvtpy_qc import QcReviewDecision, publish_qvtpy_qc_reviews
from nvitk.gui.tools.runner import notify
from nvitk.gui.viz.left_dock import attach_left_inspection_dock

DOCK_OBJECT_NAME = "nvitk_qc_measurements_dock"

# UI metric groups shown as review rows.
_LOC_METRICS: tuple[tuple[str, str, str], ...] = (
    ("flow", "flow_mean", "Flow (time-avg)"),
    ("flow_tseries", "flow_tseries", "Flow (timeseries)"),
    ("pi", "pi", "PI"),
    ("ri", "ri", "RI"),
)
_HEMO_METRICS: tuple[tuple[str, str, str], ...] = (
    ("pitc", "pitc_slope", "PITC slope"),
    ("pitc_intercept", "pitc_intercept", "PITC intercept"),
    ("pwv", "pwv", "PWV (Bjornfoot)"),
    ("pwv_fielding", "pwv_fielding_xcor", "PWV (Fielding)"),
)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def load_qc_measurement_rows(stage6_dir: Path) -> list[dict[str, Any]]:
    """Build review-table rows from stage6 CSV resource files."""
    rows: list[dict[str, Any]] = []
    loc_csv = Path(stage6_dir) / "loc_measurements.csv"
    hemo_csv = Path(stage6_dir) / "vessel_hemodynamics.csv"

    if loc_csv.is_file():
        df = pd.read_csv(loc_csv)
        flow_cols = sorted(
            (c for c in df.columns if str(c).startswith("loc_flow_ml_s_t")),
            key=lambda c: int(str(c).rsplit("_t", 1)[1]),
        )
        for _, row in df.iterrows():
            region = str(row.get("vessel_name") or "").strip()
            if not region:
                continue
            mean_flow = row.get("loc_mean_flow_ml_s")
            mean_flow_ml_min = (
                float(mean_flow) * 60.0 if pd.notna(mean_flow) else None
            )
            rows.append(
                {
                    "metric_key": "flow",
                    "variable_id": "flow_mean",
                    "metric_label": "Flow (time-avg)",
                    "region_id": region,
                    "value": mean_flow_ml_min,
                    "unit": "mL/min",
                }
            )
            if flow_cols:
                series = [
                    float(row[c]) * 60.0
                    for c in flow_cols
                    if pd.notna(row.get(c))
                ]
                summary = (
                    f"n={len(series)} mean={sum(series)/len(series):.3g}"
                    if series
                    else ""
                )
                rows.append(
                    {
                        "metric_key": "flow_tseries",
                        "variable_id": "flow_tseries",
                        "metric_label": "Flow (timeseries)",
                        "region_id": region,
                        "value": summary,
                        "unit": "mL/min",
                    }
                )
            for key, var, label in _LOC_METRICS[2:]:
                col = f"loc_{var}" if var in ("pi", "ri") else var
                val = row.get(col)
                rows.append(
                    {
                        "metric_key": key,
                        "variable_id": var,
                        "metric_label": label,
                        "region_id": region,
                        "value": float(val) if pd.notna(val) else None,
                        "unit": "dimensionless",
                    }
                )

    if hemo_csv.is_file():
        df = pd.read_csv(hemo_csv)
        col_map = {
            "pitc_slope": ("pitc_slope", "PITC slope", "1/mm"),
            "pitc_intercept": ("pitc_intercept", "PITC intercept", "dimensionless"),
            "pwv_bjornfoot_m_s": ("pwv", "PWV (Bjornfoot)", "m/s"),
            "pwv_fielding_m_s": ("pwv_fielding_xcor", "PWV (Fielding)", "m/s"),
        }
        for _, row in df.iterrows():
            region = str(row.get("region_id") or "").strip()
            if not region:
                continue
            for col, (var, label, unit) in col_map.items():
                if col not in df.columns:
                    continue
                val = row.get(col)
                rows.append(
                    {
                        "metric_key": var,
                        "variable_id": var,
                        "metric_label": label,
                        "region_id": region,
                        "value": float(val) if pd.notna(val) else None,
                        "unit": unit,
                    }
                )
    return rows


class QcMeasurementsPanel(QWidget):
    """Table of per-vessel metrics with OK/FAIL + comments."""

    COL_REGION = 0
    COL_METRIC = 1
    COL_VALUE = 2
    COL_STATUS = 3
    COL_COMMENT = 4

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        on_revised: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_revised = on_revised
        self._subject_uid = ""
        self._rows: list[dict[str, Any]] = []
        self._status = QLabel("Load a qvtpy subject to review measurements.")
        self._status.setWordWrap(True)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Region / vessel", "Metric", "Value", "OK / FAIL", "Comment"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)

        self._btn_revise = QPushButton("Mark as revised")
        self._btn_revise.setEnabled(False)
        self._btn_revise.clicked.connect(self._on_mark_revised)

        root = QVBoxLayout()
        root.addWidget(self._status)
        root.addWidget(self._table, stretch=1)
        root.addWidget(self._btn_revise)
        self.setLayout(root)

    def clear(self) -> None:
        self._subject_uid = ""
        self._rows = []
        self._table.setRowCount(0)
        self._btn_revise.setEnabled(False)
        self._status.setText("Load a qvtpy subject to review measurements.")

    def load_from_stage6(self, subject_uid: str, stage6_dir: Path) -> int:
        self._subject_uid = str(subject_uid).strip()
        self._rows = load_qc_measurement_rows(stage6_dir)
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            region_item = QTableWidgetItem(str(row.get("region_id") or ""))
            region_item.setFlags(region_item.flags() & ~Qt.ItemIsEditable)
            metric_item = QTableWidgetItem(str(row.get("metric_label") or ""))
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemIsEditable)
            value_item = QTableWidgetItem(_fmt(row.get("value")))
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)

            status = QComboBox()
            status.addItem("—", "")
            status.addItem("OK", "OK")
            status.addItem("FAIL", "FAIL")
            status.currentIndexChanged.connect(self._refresh_revise_enabled)

            comment = QLineEdit()
            comment.setPlaceholderText("optional comment")

            self._table.setItem(i, self.COL_REGION, region_item)
            self._table.setItem(i, self.COL_METRIC, metric_item)
            self._table.setItem(i, self.COL_VALUE, value_item)
            self._table.setCellWidget(i, self.COL_STATUS, status)
            self._table.setCellWidget(i, self.COL_COMMENT, comment)

        self._status.setText(
            f"{len(self._rows)} measurement(s) for {self._subject_uid}. "
            "Mark every row OK/FAIL, then revise."
        )
        self._refresh_revise_enabled()
        return len(self._rows)

    def _all_marked(self) -> bool:
        if self._table.rowCount() == 0:
            return False
        for i in range(self._table.rowCount()):
            w = self._table.cellWidget(i, self.COL_STATUS)
            if not isinstance(w, QComboBox):
                return False
            if not str(w.currentData() or "").strip():
                return False
        return True

    def _refresh_revise_enabled(self, *_args: Any) -> None:
        self._btn_revise.setEnabled(self._all_marked())

    def _collect_decisions(self) -> list[QcReviewDecision]:
        decisions: list[QcReviewDecision] = []
        for i, row in enumerate(self._rows):
            status_w = self._table.cellWidget(i, self.COL_STATUS)
            comment_w = self._table.cellWidget(i, self.COL_COMMENT)
            status = (
                str(status_w.currentData() or "").strip()
                if isinstance(status_w, QComboBox)
                else ""
            )
            comment = (
                comment_w.text().strip() if isinstance(comment_w, QLineEdit) else ""
            )
            if not status:
                continue
            decisions.append(
                QcReviewDecision(
                    variable_id=str(row.get("variable_id") or ""),
                    region_id=str(row.get("region_id") or ""),
                    qc_status=status,
                    comment=comment,
                )
            )
        return decisions

    def _on_mark_revised(self) -> None:
        if not self._all_marked():
            notify("Mark every measurement as OK or FAIL first.", error=True)
            return
        if not self._subject_uid:
            notify("No subject loaded.", error=True)
            return
        try:
            result = publish_qvtpy_qc_reviews(
                subject_uid=self._subject_uid,
                decisions=self._collect_decisions(),
                build_sqlite_index=True,
            )
        except Exception as exc:
            notify(f"Failed to update qc_status: {exc}", error=True)
            return
        notify(
            f"Revised {self._subject_uid}: "
            f"updated {result.get('updated', 0)} image_measurements row(s)."
        )
        if self._on_revised is not None:
            self._on_revised()


def attach_qc_measurements_dock(viewer: Any, panel: QcMeasurementsPanel) -> Any:
    return attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="QC measurements",
        tabify_with=["nvitk_vessel_cross_section_dock"],
        minimum_width=360,
    )


def show_qc_measurements(
    viewer: Any,
    *,
    subject_uid: str,
    stage6_dir: Path,
) -> QcMeasurementsPanel:
    panel = QcMeasurementsPanel()
    attach_qc_measurements_dock(viewer, panel)
    n = panel.load_from_stage6(subject_uid, stage6_dir)
    if n == 0:
        notify(
            f"No measurement CSVs found under {stage6_dir} "
            "(expected loc_measurements.csv / vessel_hemodynamics.csv).",
            error=True,
        )
    return panel
