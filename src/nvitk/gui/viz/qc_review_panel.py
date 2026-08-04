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
    """Format *value* for table display: 4 significant figures for numeric values, ``""`` for
    ``None``/NaN, else ``str()``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _fmt_parts(parts: list[str]) -> str:
    """Join non-empty *parts* with ``"; "``, for compact multi-value table cells."""
    return "; ".join(p for p in parts if p)


def load_qc_measurement_rows(
    stage6_dir: Path,
    *,
    stage7_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Build grouped QC review rows from stage-6 CSVs (+ optional stage-7 stenosis).

    Grouping
    --------
    1. **LOC / vessel** — one OK/FAIL covering flow (time-avg + timeseries), PI, RI.
    2. **PITC / territory** — one OK/FAIL per LICA / RICA / BASILAR covering slope +
       intercept (all stations in that territory).
    3. **PWV / territory** — one OK/FAIL per territory covering Bjornfoot + Fielding.
    4. **Stenosis / vessel** — informational morphometrics rows from stage-7 Path Summary
       (max stenosis %, length, segment count). Empty ``variable_ids`` → display only.

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

    # Stage-7 morphometrics stenosis (optional; display-only unless DB variables exist).
    s7 = Path(stage7_dir) if stage7_dir is not None else Path(stage6_dir).parent / "stage7_morphometrics"
    excel = s7 / "case_metrics_donut_tree.xlsx"
    if excel.is_file():
        try:
            from nvitk.gui.viz.morpho_viz import _aggregate_summary_by_vessel

            raw = pd.read_excel(excel, sheet_name="00_Path_Summary")
            summary = _aggregate_summary_by_vessel(raw)
            for _, row in summary.iterrows():
                vessel = str(row.get("vessel_name") or "").strip()
                if not vessel:
                    continue
                parts: list[str] = []
                sten_max = row.get("stenosis_percent_max", row.get("degree_of_stenosis_pct"))
                if pd.notna(sten_max):
                    parts.append(f"stenosis_max={_fmt(float(sten_max))} %")
                sten_len = row.get("stenosis_length_total_mm")
                if pd.notna(sten_len):
                    parts.append(f"stenosis_len={_fmt(float(sten_len))} mm")
                sten_n = row.get("stenosis_segments_n")
                if pd.notna(sten_n):
                    parts.append(f"segments={int(float(sten_n))}")
                r_mean = row.get("radius_mean_mm")
                if pd.notna(r_mean):
                    parts.append(f"radius_mean={_fmt(float(r_mean))} mm")
                if not parts:
                    continue
                rows.append(
                    {
                        "metric_key": "stenosis",
                        "variable_ids": [],
                        "metric_label": "Stenosis / caliber",
                        "region_id": vessel,
                        "region_label": vessel,
                        "value": _fmt_parts(parts),
                        "unit": "",
                        "review_optional": True,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            # Keep hemodynamics QC usable even if morphometrics Excel is incomplete.
            from nvitk.core.logger import Logger

            Logger().warning("QC stenosis rows skipped: %s", exc)

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
        """Build the QC review table (region/metric/value/status/comment columns) and the
        mark-as-revised button."""
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
        """Reset the panel to its empty state (no subject, empty table, disabled revise button)."""
        self._subject_uid = ""
        self._rows = []
        self._table.setRowCount(0)
        self._btn_revise.setEnabled(False)
        self._status.setText("Load a qvtpy subject to review measurements.")

    def load_from_stage6(self, subject_uid: str, stage6_dir: Path) -> int:
        """Load and populate the review table for *subject_uid* from *stage6_dir* (and its sibling
        stage-7 morphometrics directory). Returns the number of rows loaded."""
        self._subject_uid = str(subject_uid).strip()
        stage7_dir = Path(stage6_dir).parent / "stage7_morphometrics"
        self._rows = load_qc_measurement_rows(stage6_dir, stage7_dir=stage7_dir)
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
        n_sten = sum(1 for r in self._rows if r.get("metric_key") == "stenosis")
        self._status.setText(
            f"{len(self._rows)} check(s) for {self._subject_uid} "
            f"(LOC={n_loc}, PITC={n_pitc}, PWV={n_pwv}, stenosis={n_sten}). "
            "One OK/FAIL covers the grouped metrics; stenosis rows are optional."
        )
        self._refresh_revise_enabled()
        return len(self._rows)

    def _all_marked(self) -> bool:
        """True if every non-optional row has an OK/FAIL status selected (and there's at least one)."""
        if self._table.rowCount() == 0:
            return False
        required = False
        for i in range(self._table.rowCount()):
            row = self._rows[i] if i < len(self._rows) else {}
            if row.get("review_optional"):
                continue
            required = True
            w = self._table.cellWidget(i, self.COL_STATUS)
            if not isinstance(w, QComboBox):
                return False
            if not str(w.currentData() or "").strip():
                return False
        return required

    def _refresh_revise_enabled(self, *_args: Any) -> None:
        """Enable the mark-as-revised button only once every required row is marked."""
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
        """Publish the collected OK/FAIL decisions for the loaded subject, notifying on success/failure."""
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
    """Dock *panel* on Napari's left edge, tabbed with the vessel cross-section dock."""
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
    """Create, dock, and populate a :class:`QcMeasurementsPanel` for *subject_uid*'s stage-6 results."""
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
