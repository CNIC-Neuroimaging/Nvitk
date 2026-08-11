"""QC measurement review dock: OK/FAIL per metric with optional comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.db.qvtpy_anatomy import (
    ANATOMY_CONFIG_VARIABLES,
    load_anatomy_configs,
    publish_anatomy_configs,
)
from nvitk.db.qvtpy_qc import (
    QC_METRIC_VARIABLES,
    QcReviewDecision,
    publish_qvtpy_qc_reviews,
)
from nvitk.gui.tools.runner import notify
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.core.logger import Logger

log = Logger()

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


def anatomy_config_rows() -> list[dict[str, Any]]:
    """
    The manual anatomy rows that head the review table — one per configuration variable.

    These carry no measured value: their ``Value`` cell is a dropdown the reviewer fills in, and
    they are ``review_optional`` because a config is an annotation, not an OK/FAIL verdict on a
    number. Marking the subject as revised publishes whichever ones were set.
    """
    return [
        {
            "metric_key": "anatomy",
            "variable_ids": [],
            "anatomy_variable": var.variable_id,
            "metric_label": var.label,
            "region_id": "",
            "region_label": var.region_label,
            "value": "",
            "unit": "",
            "review_optional": True,
        }
        for var in ANATOMY_CONFIG_VARIABLES.values()
    ]


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
            log.warning("QC stenosis rows skipped: %s", exc)

    return rows




#: Labels for the automatic metrics, mirroring ``stage9_autoqc.QC_LABELS`` without importing the
#: pipeline into the viewer — the GUI must open on a machine that has no qvtpy stage installed.
QC_VESSEL_METRIC_LABELS: dict[str, str] = {
    "qc_flow_plausible": "Flow plausibility (literature band)",
    "qc_hypoplastic": "Plausibly hypoplastic (<0.8 mm)",
    "qc_conservation": "Mass-conservation residual",
    "qc_segment_cv": "Along-segment flow CV",
    "qc_score": "Combined QC score",
    "qc_flag": "QC flag (review)",
}

#: Subject-level metrics live in ``clinical_measurements`` (one value per subject).
QC_SUBJECT_METRIC_LABELS: dict[str, str] = {
    "qc_ap_share": "Anterior share of cerebral inflow (%)",
    "qc_ap_flag": "Anterior/posterior split flag",
    "qc_subject_flag": "Subject QC flag",
}

#: Union used by colouring helpers that accept either vessel- or subject-level ids.
QC_METRIC_LABELS: dict[str, str] = {
    **QC_VESSEL_METRIC_LABELS,
    **QC_SUBJECT_METRIC_LABELS,
}

# ──────────────────────────────────────────────────────────────────────────────
# Automatic-QC colouring
# ──────────────────────────────────────────────────────────────────────────────
#: Row tints for the automatic metrics, dark enough to keep the light table text readable.
_QC_GOOD = QColor("#1e4620")
_QC_WARN = QColor("#5a4a1e")
_QC_BAD = QColor("#5a2424")
_QC_NEUTRAL = QColor("#2b2b2b")
_QC_UNKNOWN = QColor("#333333")


def qc_metric_colour(metric: str, value: Any) -> QColor:
    """
    Row tint for one automatic-QC value.

    Each metric reads differently, so the mapping is per metric rather than one shared scale:
    a plausibility of 0 is bad while a conservation residual of 0 is perfect, and a hypoplasia
    flag is anatomy rather than a verdict. A missing value is tinted as *unknown* rather than as
    passing — never having been checked is not the same as having passed.
    """
    if value is None:
        return _QC_UNKNOWN
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _QC_UNKNOWN
    if number != number:  # NaN
        return _QC_UNKNOWN

    if metric in {"qc_flow_plausible", "qc_score"}:
        return _QC_GOOD if number >= 0.75 else _QC_WARN if number >= 0.5 else _QC_BAD
    if metric == "qc_conservation":
        magnitude = abs(number)
        return _QC_GOOD if magnitude <= 0.10 else _QC_WARN if magnitude <= 0.15 else _QC_BAD
    if metric == "qc_segment_cv":
        return _QC_GOOD if number <= 0.05 else _QC_WARN if number <= 0.15 else _QC_BAD
    if metric == "qc_hypoplastic":
        # Not a failure: a hypoplastic vessel is normal anatomy that merely excuses the other
        # checks, so it is marked as notable rather than bad.
        return _QC_WARN if number >= 0.5 else _QC_NEUTRAL
    if metric in {"qc_flag", "qc_ap_flag", "qc_subject_flag"}:
        return _QC_BAD if number >= 0.5 else _QC_GOOD
    if metric == "qc_ap_share":
        return _QC_GOOD if abs(number - 72.0) <= 10.0 else _QC_BAD
    return _QC_NEUTRAL



def _numeric_or_none(value: Any) -> float | None:
    """Coerce *value* to float, returning ``None`` for missing / non-finite entries."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def load_autoqc_for_subject(
    subject_uid: str, *, repo: Any = None, pipeline: str = ""
) -> dict[str, dict[str, float]]:
    """
    Per-vessel automatic QC metrics for one subject, keyed ``{metric: {region: value}}``.

    Read from ``image_measurements`` — the qvtpy stage 9 writes them there as ordinary rows — so the
    panel colours by whatever the dataset actually holds rather than recomputing anything.

    Regions are keyed **both** by their published spelling and by their canonical vessel, because a
    dataset can carry two importers' conventions (``LICA`` and ``left_ica``) while the review table
    shows only one of them. Matching on either is what makes the colouring land on the right row.

    Never raises: no dataset, no stage-9 output, or an unreadable table all mean "no colouring",
    which is a normal state and must not take the QC panel down with it.
    """
    metrics: dict[str, dict[str, float]] = {}
    subject = str(subject_uid).strip()
    if not subject:
        return metrics
    try:
        if repo is None:
            from nvitk.db.repo import get_repo_from_settings

            repo = get_repo_from_settings()
        rows = repo.get("image_measurements", cohort_id=False)
    except Exception as exc:
        log.debug("Automatic QC unavailable (%s).", exc)
        return metrics
    if rows is None or rows.empty or "variable_id" not in rows.columns:
        return metrics

    wanted = set(QC_VESSEL_METRIC_LABELS)
    mine = rows.loc[
        (rows["subject_uid"].astype(str) == subject)
        & (rows["variable_id"].astype(str).isin(wanted))
    ]
    if pipeline and "pipeline_id" in mine.columns:
        mine = mine.loc[mine["pipeline_id"].astype(str) == str(pipeline)]
    if mine.empty:
        return metrics

    try:
        from nvitk.stats.vessel_network import canonical_node
    except Exception:  # pragma: no cover - the viewer must open without nvitk.stats
        canonical_node = lambda value: None  # noqa: E731

    for metric, group in mine.groupby(mine["variable_id"].astype(str)):
        by_region: dict[str, float] = {}
        for region, value in zip(group["region_id"].astype(str), group["value_num"]):
            number = _numeric_or_none(value)
            if number is None:
                continue
            by_region[region] = number
            by_region[region.upper()] = number
            node = canonical_node(region)
            if node:
                by_region.setdefault(node, number)
        if by_region:
            metrics[str(metric)] = by_region
    log.info(
        "Automatic QC for %s: %d vessel metric(s) over %d region(s).",
        subject, len(metrics), len({r for m in metrics.values() for r in m}),
    )
    return metrics


def load_subject_autoqc(
    subject_uid: str, *, repo: Any = None
) -> dict[str, float]:
    """
    Subject-level automatic QC metrics for one subject, keyed ``{metric: value}``.

    Stage 9 publishes these to ``clinical_measurements`` (``qc_ap_share``, ``qc_ap_flag``,
    ``qc_subject_flag``). Never raises — missing data simply yields an empty dict.
    """
    metrics: dict[str, float] = {}
    subject = str(subject_uid).strip()
    if not subject:
        return metrics
    try:
        if repo is None:
            from nvitk.db.repo import get_repo_from_settings

            repo = get_repo_from_settings()
        rows = repo.get("clinical_measurements", cohort_id=False)
    except Exception as exc:
        log.debug("Subject automatic QC unavailable (%s).", exc)
        return metrics
    if rows is None or rows.empty or "variable_id" not in rows.columns:
        return metrics

    wanted = set(QC_SUBJECT_METRIC_LABELS)
    mine = rows.loc[
        (rows["subject_uid"].astype(str) == subject)
        & (rows["variable_id"].astype(str).isin(wanted))
    ]
    if mine.empty:
        return metrics

    for metric, group in mine.groupby(mine["variable_id"].astype(str)):
        # One value per subject; if duplicates somehow exist, take the first finite one.
        for value in group["value_num"]:
            number = _numeric_or_none(value)
            if number is not None:
                metrics[str(metric)] = number
                break
    log.info(
        "Subject automatic QC for %s: %s.",
        subject,
        ", ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items())) or "none",
    )
    return metrics


def _fmt_subject_qc_value(metric: str, value: float | None) -> str:
    """Compact display string for one subject-level autoQC metric."""
    if value is None:
        return "—"
    if metric in {"qc_ap_flag", "qc_subject_flag"}:
        return "FAIL" if value >= 0.5 else "OK"
    if metric == "qc_ap_share":
        return f"{value:.1f}%"
    return f"{value:.4g}"


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
        self._autoqc: dict[str, Mapping[str, float]] = {}
        self._subject_autoqc: dict[str, float] = {}
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
            # No ``background-color`` here on purpose. A stylesheet rule on ``::item`` overrides
            # whatever ``QTableWidgetItem.setBackground`` sets, which silently disabled the
            # automatic-QC row tinting — the values were applied, the paint just never used them.
            # The widget-level rule above still gives the dark viewport.
            "QTableWidget::item {"
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

        # Subject-level autoQC summary (clinical_measurements). Shown above the vessel table so
        # anterior/posterior split and the subject flag are visible without being vessel rows.
        self._subject_qc_row = QWidget()
        subject_lay = QHBoxLayout(self._subject_qc_row)
        subject_lay.setContentsMargins(0, 0, 0, 0)
        subject_lay.setSpacing(8)
        subject_lay.addWidget(QLabel("Subject autoQC"))
        self._subject_qc_chips: dict[str, QLabel] = {}
        for metric, label in QC_SUBJECT_METRIC_LABELS.items():
            chip = QLabel("—")
            chip.setToolTip(label)
            chip.setAlignment(Qt.AlignCenter)
            chip.setMinimumWidth(72)
            chip.setStyleSheet(
                "QLabel {"
                "  background-color: #333333;"
                "  color: #e8e8e8;"
                "  border: 1px solid #555;"
                "  border-radius: 4px;"
                "  padding: 3px 8px;"
                "}"
            )
            self._subject_qc_chips[metric] = chip
            # Short heading above each chip via a tiny stacked widget.
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(1)
            heading = QLabel(
                {
                    "qc_ap_share": "A/P share",
                    "qc_ap_flag": "A/P flag",
                    "qc_subject_flag": "Subject flag",
                }.get(metric, metric)
            )
            heading.setStyleSheet("color: #9a9a9a; font-size: 11px;")
            heading.setAlignment(Qt.AlignCenter)
            cell_lay.addWidget(heading)
            cell_lay.addWidget(chip)
            subject_lay.addWidget(cell)
        subject_lay.addStretch(1)
        self._subject_qc_row.setVisible(False)

        # Colour-by picker for the automatic QC metrics published by stage 9. Populated on load
        # from whatever that stage actually wrote, so a dataset without it simply shows "none".
        self._colour_by = QComboBox()
        self._colour_by.addItem("No colouring", "")
        self._colour_by.currentIndexChanged.connect(self._apply_colouring)
        self._colour_row = QWidget()
        colour_lay = QHBoxLayout(self._colour_row)
        colour_lay.setContentsMargins(0, 0, 0, 0)
        colour_lay.addWidget(QLabel("Colour by"))
        colour_lay.addWidget(self._colour_by, stretch=1)
        self._colour_legend = QLabel("")
        self._colour_legend.setStyleSheet("color: #9a9a9a;")
        colour_lay.addWidget(self._colour_legend, stretch=2)

        self._btn_revise = QPushButton("Mark as revised")
        self._btn_revise.setEnabled(False)
        self._btn_revise.clicked.connect(self._on_mark_revised)

        root = QVBoxLayout()
        root.addWidget(self._status)
        root.addWidget(self._subject_qc_row)
        root.addWidget(self._colour_row)
        root.addWidget(self._table, stretch=1)
        root.addWidget(self._btn_revise)
        self.setLayout(root)

    # ---- automatic-QC colouring ----------------------------------------------
    def set_autoqc(
        self,
        values: Mapping[str, Mapping[str, float]] | None,
        *,
        subject_values: Mapping[str, float] | None = None,
    ) -> None:
        """
        Attach automatic QC metrics for this subject.

        Parameters
        ----------
        values :
            Per-vessel metrics keyed ``{metric: {region_id: value}}``.
        subject_values :
            Subject-level metrics keyed ``{metric: value}`` (from ``clinical_measurements``).

        Populates the colour picker with whichever metrics are present, refreshes the subject
        summary chips, and applies colouring. Passing empty/``None`` clears both.
        """
        self._autoqc = dict(values or {})
        self._subject_autoqc = {
            str(k): float(v)
            for k, v in dict(subject_values or {}).items()
            if _numeric_or_none(v) is not None
        }
        self._refresh_subject_qc_chips()

        current = str(self._colour_by.currentData() or "")
        self._colour_by.blockSignals(True)
        self._colour_by.clear()
        self._colour_by.addItem("No colouring", "")
        for metric in sorted(self._autoqc):
            self._colour_by.addItem(
                f"Vessel · {QC_METRIC_LABELS.get(metric, metric)}", metric
            )
        for metric in sorted(self._subject_autoqc):
            self._colour_by.addItem(
                f"Subject · {QC_METRIC_LABELS.get(metric, metric)}", metric
            )
        index = self._colour_by.findData(current)
        self._colour_by.setCurrentIndex(index if index >= 0 else 0)
        self._colour_by.blockSignals(False)
        self._colour_row.setVisible(bool(self._autoqc) or bool(self._subject_autoqc))
        self._apply_colouring()

    def _refresh_subject_qc_chips(self) -> None:
        """Update the subject-level autoQC chips from ``self._subject_autoqc``."""
        any_present = bool(self._subject_autoqc)
        self._subject_qc_row.setVisible(any_present)
        for metric, chip in self._subject_qc_chips.items():
            value = self._subject_autoqc.get(metric)
            text = _fmt_subject_qc_value(metric, value)
            colour = qc_metric_colour(metric, value) if value is not None else _QC_UNKNOWN
            chip.setText(text)
            chip.setToolTip(
                f"{QC_SUBJECT_METRIC_LABELS.get(metric, metric)}: "
                + ("not computed" if value is None else f"{float(value):.4g}")
            )
            chip.setStyleSheet(
                "QLabel {"
                f"  background-color: {colour.name()};"
                "  color: #e8e8e8;"
                "  border: 1px solid #555;"
                "  border-radius: 4px;"
                "  padding: 3px 8px;"
                "  font-weight: 600;"
                "}"
            )

    @staticmethod
    def _qc_value(by_region: Mapping[str, float], row: Mapping[str, Any]) -> float | None:
        """This row's value for the selected metric, trying each spelling the table might use."""
        for key in ("region_id", "region_label"):
            name = str(row.get(key) or "").strip()
            if not name:
                continue
            for candidate in (name, name.upper(), name.lower()):
                if candidate in by_region:
                    return by_region[candidate]
        return None

    def _apply_colouring(self, *_args: Any) -> None:
        """Tint each row by the selected metric (per-vessel lookup, or whole-subject value)."""
        metric = str(self._colour_by.currentData() or "")
        subject_value = self._subject_autoqc.get(metric) if metric else None
        by_region = self._autoqc.get(metric, {}) if metric else {}
        is_subject_metric = metric in QC_SUBJECT_METRIC_LABELS

        for i, row in enumerate(self._rows):
            if not metric:
                value = None
            elif is_subject_metric:
                value = subject_value
            else:
                value = self._qc_value(by_region, row)
            colour = qc_metric_colour(metric, value) if metric else QColor("#2b2b2b")
            for column in (self.COL_REGION, self.COL_METRIC, self.COL_VALUE):
                item = self._table.item(i, column)
                if item is not None:
                    item.setBackground(colour)
                    if metric:
                        item.setToolTip(
                            f"{QC_METRIC_LABELS.get(metric, metric)}: "
                            + ("not computed" if value is None else f"{float(value):.4g}")
                        )

        if not metric:
            self._colour_legend.setText("")
            return
        scope = "all rows" if is_subject_metric else "per vessel"
        self._colour_legend.setText(
            f"{scope}  ·  green = passes  ·  amber = borderline  ·  "
            "red = review  ·  grey = not computed"
        )

    def clear(self) -> None:
        """Reset the panel to its empty state (no subject, empty table, disabled revise button)."""
        self._subject_uid = ""
        self._rows = []
        self._autoqc = {}
        self._subject_autoqc = {}
        self._table.setRowCount(0)
        self._btn_revise.setEnabled(False)
        self._subject_qc_row.setVisible(False)
        self._colour_row.setVisible(False)
        self._status.setText("Load a qvtpy subject to review measurements.")

    def load_from_stage6(self, subject_uid: str, stage6_dir: Path) -> int:
        """
        Load and populate the review table for *subject_uid* from *stage6_dir* (and its sibling
        stage-7 morphometrics directory).

        Returns the number of **measured** rows, excluding the two manual anatomy rows — those are
        always present, so counting them would hide the "no measurement CSVs found" case from the
        caller.
        """
        self._subject_uid = str(subject_uid).strip()
        stage7_dir = Path(stage6_dir).parent / "stage7_morphometrics"
        measured = load_qc_measurement_rows(stage6_dir, stage7_dir=stage7_dir)
        # Anatomy first: it is the one thing the reviewer must supply, and burying it under a
        # hundred vessel rows is how it ends up never filled in.
        self._rows = [*anatomy_config_rows(), *measured]
        saved_configs = load_anatomy_configs(self._subject_uid)
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            region_item = QTableWidgetItem(
                str(row.get("region_label") or row.get("region_id") or "")
            )
            region_item.setFlags(region_item.flags() & ~Qt.ItemIsEditable)
            metric_item = QTableWidgetItem(str(row.get("metric_label") or ""))
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemIsEditable)

            anatomy_variable = str(row.get("anatomy_variable") or "")
            if anatomy_variable:
                self._fill_anatomy_row(i, anatomy_variable, saved_configs, metric_item)
            else:
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

                self._table.setItem(i, self.COL_VALUE, value_item)
                self._table.setCellWidget(i, self.COL_STATUS, status)
                self._table.setCellWidget(i, self.COL_COMMENT, comment)

            self._table.setItem(i, self.COL_REGION, region_item)
            self._table.setItem(i, self.COL_METRIC, metric_item)

        # Vessel- and subject-level autoQC from the dataset, if stage 9 has run.
        self.set_autoqc(
            load_autoqc_for_subject(self._subject_uid),
            subject_values=load_subject_autoqc(self._subject_uid),
        )

        n_loc = sum(1 for r in self._rows if r.get("metric_key") == "loc")
        n_pitc = sum(1 for r in self._rows if r.get("metric_key") == "pitc")
        n_pwv = sum(1 for r in self._rows if r.get("metric_key") == "pwv")
        n_sten = sum(1 for r in self._rows if r.get("metric_key") == "stenosis")
        stored = (
            "; ".join(f"{k}={v}" for k, v in sorted(saved_configs.items()))
            if saved_configs
            else "not set yet"
        )
        self._status.setText(
            f"{len(measured)} check(s) for {self._subject_uid} "
            f"(LOC={n_loc}, PITC={n_pitc}, PWV={n_pwv}, stenosis={n_sten}). "
            "One OK/FAIL covers the grouped metrics; stenosis rows are optional. "
            f"Anatomy configs: {stored}."
        )
        self._refresh_revise_enabled()
        return len(measured)

    def _fill_anatomy_row(
        self,
        index: int,
        variable_id: str,
        saved: Mapping[str, str],
        metric_item: QTableWidgetItem,
    ) -> None:
        """
        Render one manual-anatomy row: a vocabulary dropdown in ``Value``, nothing else editable.

        The OK/FAIL and comment cells are placeholders — an anatomy config is not a verdict on a
        number, and a comment typed here would have nowhere to go in ``image_measurements``. The
        variable's description becomes the row tooltip so the vocabulary is explained in place.
        """
        var = ANATOMY_CONFIG_VARIABLES[variable_id]
        metric_item.setToolTip(var.description)

        choice = QComboBox()
        choice.addItem("— select —", "")
        for code, label in var.choices:
            choice.addItem(f"{label}  ({code})", code)
        stored = str(saved.get(variable_id) or "")
        if stored:
            found = choice.findData(stored)
            if found >= 0:
                choice.setCurrentIndex(found)
        choice.setToolTip(var.description)
        choice.currentIndexChanged.connect(self._refresh_revise_enabled)
        self._table.setCellWidget(index, self.COL_VALUE, choice)

        for column, text, tip in (
            (self.COL_STATUS, "manual", "Anatomy configs are annotations, not OK/FAIL checks."),
            (self.COL_COMMENT, "—", "Notes are not stored for anatomy rows."),
        ):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip(tip)
            self._table.setItem(index, column, item)

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

    def _has_required_rows(self) -> bool:
        """True if the table holds at least one row that needs an OK/FAIL."""
        return any(not row.get("review_optional") for row in self._rows)

    def _revise_ready(self) -> bool:
        """
        True when there is something to publish.

        Normally that means every required row is marked. A subject whose stage-6 CSVs are missing
        has no required rows at all, and there the anatomy configs are the only thing to save — so
        a chosen config is enough to enable the button rather than leaving the reviewer stuck.
        """
        if self._all_marked():
            return True
        return not self._has_required_rows() and bool(self._collect_anatomy_values())

    def _refresh_revise_enabled(self, *_args: Any) -> None:
        """Enable the mark-as-revised button once the review has something publishable."""
        self._btn_revise.setEnabled(self._revise_ready())

    def _collect_anatomy_values(self) -> dict[str, str]:
        """``{variable_id: code}`` for the anatomy rows the reviewer actually filled in."""
        out: dict[str, str] = {}
        for i, row in enumerate(self._rows):
            variable_id = str(row.get("anatomy_variable") or "")
            if not variable_id:
                continue
            widget = self._table.cellWidget(i, self.COL_VALUE)
            if not isinstance(widget, QComboBox):
                continue
            code = str(widget.currentData() or "").strip()
            if code:
                out[variable_id] = code
        return out

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
        """
        Publish the review for the loaded subject: OK/FAIL decisions first, then the manual anatomy
        configs. Notifies on success and on either failure.

        The QC pass runs first because it rewrites the whole ``image_measurements`` table from what
        it read; the anatomy upsert always reads Parquet directly, so it is the safe one to run
        against a freshly rebuilt index.
        """
        if not self._revise_ready():
            notify("Mark every measurement as OK or FAIL first.", error=True)
            return
        if not self._subject_uid:
            notify("No subject loaded.", error=True)
            return
        decisions = self._collect_decisions()
        updated = 0
        if decisions:
            try:
                result = publish_qvtpy_qc_reviews(
                    subject_uid=self._subject_uid,
                    decisions=decisions,
                    build_sqlite_index=True,
                )
            except Exception as exc:
                notify(f"Failed to update qc_status: {exc}", error=True)
                return
            updated = int(result.get("updated", 0))

        parts = [f"updated {updated} image_measurements row(s)"]
        chosen = self._collect_anatomy_values()
        if chosen:
            try:
                published = publish_anatomy_configs(
                    subject_uid=self._subject_uid,
                    values=chosen,
                    build_sqlite_index=True,
                )
            except Exception as exc:
                # The QC decisions are already saved — report the anatomy failure on its own rather
                # than making the whole revise look like it did not happen.
                notify(f"QC saved, but anatomy configs failed: {exc}", error=True)
                if self._on_revised is not None:
                    self._on_revised()
                return
            parts.append(
                "anatomy " + ", ".join(f"{k}={v}" for k, v in sorted(published.items()))
            )
        missing = [v for v in ANATOMY_CONFIG_VARIABLES if v not in chosen]
        if missing:
            parts.append(f"not set: {', '.join(missing)}")
        notify(f"Revised {self._subject_uid}: " + "; ".join(parts) + ".")
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
