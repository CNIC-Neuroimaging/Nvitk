"""QC measurement review dock: OK/FAIL per metric with optional comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.db.qvtpy_qc import (
    QC_METRIC_VARIABLES,
    QcReviewDecision,
    publish_qvtpy_qc_reviews,
)
from nvitk.gui.tools.runner import notify
from nvitk.gui.viz.left_dock import attach_left_inspection_dock

DOCK_OBJECT_NAME = "nvitk_qc_measurements_dock"

# Territory roots in vessel_hemodynamics.csv (row_kind == "root").
_HEMO_TERRITORIES: tuple[str, ...] = ("L_ICA", "R_ICA", "Basilar")
_HEMO_TERRITORY_LABEL: dict[str, str] = {
    "L_ICA": "LICA",
    "R_ICA": "RICA",
    "Basilar": "BASILAR",
}


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _fmt_parts(parts: list[str]) -> str:
    return "; ".join(p for p in parts if p)


def load_qc_measurement_rows(stage6_dir: Path) -> list[dict[str, Any]]:
    """Build grouped QC review rows from stage-6 CSVs.

    Grouping
    --------
    1. **LOC / vessel** — one OK/FAIL covering flow (time-avg + timeseries), PI, RI.
    2. **PITC / territory** — one OK/FAIL per LICA / RICA / BASILAR covering slope +
       intercept (all stations in that territory).
    3. **PWV / territory** — one OK/FAIL per territory covering Bjornfoot + Fielding.

    Branch-level PITC/PWV rows are omitted; only territory (root) values are shown.
    """
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
            value_parts: list[str] = []
            if mean_flow_ml_min is not None:
                value_parts.append(f"flow_tavg={_fmt(mean_flow_ml_min)} mL/min")
            if flow_cols:
                series = [
                    float(row[c]) * 60.0
                    for c in flow_cols
                    if pd.notna(row.get(c))
                ]
                if series:
                    value_parts.append(
                        f"flow_tseries n={len(series)} "
                        f"mean={sum(series) / len(series):.3g} mL/min"
                    )
            pi = row.get("loc_pi")
            ri = row.get("loc_ri")
            if pd.notna(pi):
                value_parts.append(f"PI={_fmt(float(pi))}")
            if pd.notna(ri):
                value_parts.append(f"RI={_fmt(float(ri))}")
            rows.append(
                {
                    "metric_key": "loc",
                    "variable_ids": list(QC_METRIC_VARIABLES["loc"]),
                    "metric_label": "Flow / PI / RI",
                    "region_id": region,
                    "region_label": region,
                    "value": _fmt_parts(value_parts),
                    "unit": "",
                }
            )

    if hemo_csv.is_file():
        df = pd.read_csv(hemo_csv)
        # Territory roots only — skip named-branch damping / empty PITC-PWV rows.
        if "row_kind" in df.columns:
            roots = df[df["row_kind"].astype(str).str.lower() == "root"]
        else:
            roots = df[df["region_id"].astype(str).isin(_HEMO_TERRITORIES)]
        for territory in _HEMO_TERRITORIES:
            hit = roots[roots["region_id"].astype(str) == territory]
            if hit.empty:
                continue
            row = hit.iloc[0]
            label = _HEMO_TERRITORY_LABEL.get(territory, territory)

            pitc_parts: list[str] = []
            slope = row.get("pitc_slope")
            intercept = row.get("pitc_intercept")
            r2 = row.get("pitc_r2")
            n_fit = row.get("pitc_n")
            if pd.notna(slope):
                pitc_parts.append(f"slope={_fmt(float(slope))} 1/mm")
            if pd.notna(intercept):
                pitc_parts.append(f"intercept={_fmt(float(intercept))}")
            if pd.notna(r2):
                pitc_parts.append(f"r²={_fmt(float(r2))}")
            if pd.notna(n_fit):
                pitc_parts.append(f"n={int(float(n_fit))}")
            if pitc_parts:
                rows.append(
                    {
                        "metric_key": "pitc",
                        "variable_ids": list(QC_METRIC_VARIABLES["pitc"]),
                        "metric_label": "PITC (slope + intercept)",
                        "region_id": territory,
                        "region_label": label,
                        "value": _fmt_parts(pitc_parts),
                        "unit": "",
                    }
                )

            pwv_parts: list[str] = []
            bj = row.get("pwv_bjornfoot_m_s")
            fi = row.get("pwv_fielding_m_s")
            if pd.notna(bj):
                pwv_parts.append(f"Bjornfoot={_fmt(float(bj))} m/s")
            if pd.notna(fi):
                pwv_parts.append(f"Fielding={_fmt(float(fi))} m/s")
            if pwv_parts:
                rows.append(
                    {
                        "metric_key": "pwv",
                        "variable_ids": list(QC_METRIC_VARIABLES["pwv"]),
                        "metric_label": "PWV (Bjornfoot + Fielding)",
                        "region_id": territory,
                        "region_label": label,
                        "value": _fmt_parts(pwv_parts),
                        "unit": "",
                    }
                )
    return rows


class QcMeasurementsPanel(QWidget):
    """Table of grouped per-vessel / per-territory metrics with OK/FAIL + comments."""

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
        # Uniform row color — alternating rows hide light text on dark Napari UI.
        self._table.setAlternatingRowColors(False)
        self._table.setStyleSheet(
            "QTableWidget {"
            "  background-color: #2b2b2b;"
            "  color: #e8e8e8;"
            "  gridline-color: #454545;"
            "  alternate-background-color: #2b2b2b;"
            "}"
            "QTableWidget::item {"
            "  background-color: #2b2b2b;"
            "  color: #e8e8e8;"
            "}"
            "QTableWidget::item:selected {"
            "  background-color: #3d5a80;"
            "  color: #ffffff;"
            "}"
            "QHeaderView::section {"
            "  background-color: #353535;"
            "  color: #e8e8e8;"
            "  padding: 4px;"
            "  border: 1px solid #454545;"
            "}"
            "QComboBox, QLineEdit {"
            "  background-color: #1e1e1e;"
            "  color: #e8e8e8;"
            "  border: 1px solid #555;"
            "}"
        )

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
            region_item = QTableWidgetItem(
                str(row.get("region_label") or row.get("region_id") or "")
            )
            region_item.setFlags(region_item.flags() & ~Qt.ItemIsEditable)
            metric_item = QTableWidgetItem(str(row.get("metric_label") or ""))
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemIsEditable)
            value_item = QTableWidgetItem(str(row.get("value") or ""))
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            value_item.setToolTip(str(row.get("value") or ""))

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

        n_loc = sum(1 for r in self._rows if r.get("metric_key") == "loc")
        n_pitc = sum(1 for r in self._rows if r.get("metric_key") == "pitc")
        n_pwv = sum(1 for r in self._rows if r.get("metric_key") == "pwv")
        self._status.setText(
            f"{len(self._rows)} check(s) for {self._subject_uid} "
            f"(LOC vessels={n_loc}, PITC territories={n_pitc}, PWV territories={n_pwv}). "
            "One OK/FAIL covers the grouped metrics; then revise."
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
        """Expand each grouped UI row into one decision per underlying variable_id."""
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
            region = str(row.get("region_id") or "")
            vars_ = list(row.get("variable_ids") or ())
            if not vars_:
                key = str(row.get("metric_key") or "")
                vars_ = list(QC_METRIC_VARIABLES.get(key, ()))
            for var in vars_:
                decisions.append(
                    QcReviewDecision(
                        variable_id=str(var),
                        region_id=region,
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
    on_revised: Callable[[], None] | None = None,
) -> QcMeasurementsPanel:
    panel = QcMeasurementsPanel(on_revised=on_revised)
    attach_qc_measurements_dock(viewer, panel)
    n = panel.load_from_stage6(subject_uid, stage6_dir)
    if n == 0:
        notify(
            f"No measurement CSVs found under {stage6_dir} "
            "(expected loc_measurements.csv / vessel_hemodynamics.csv).",
            error=True,
        )
    return panel
