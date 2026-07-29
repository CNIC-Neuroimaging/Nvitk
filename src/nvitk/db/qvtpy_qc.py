"""Write QVTPY QC review status onto ``image_measurements.qc_status``.

Each review decision updates matching rows for the latest qvtpy pipeline
(``4dflow_v3`` / aliases ``latest`` / ``qvtpy``) keyed by
``subject_uid`` + ``pipeline_id`` + ``variable_id`` (+ optional ``region_id``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo
from nvitk.db.storage import utc_now_iso
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID, resolve_repo

log = Logger()

QC_STATUS_VALUES = frozenset({"PENDING", "OK", "FAIL"})

# Metrics shown in the QC review table (grouped UI key -> DB variable_id(s)).
# One OK/FAIL in the UI expands to all listed variables for that region.
QC_METRIC_VARIABLES: dict[str, tuple[str, ...]] = {
    "loc": ("flow_mean", "flow_tseries", "pi", "ri"),
    "pitc": ("pitc_slope", "pitc_intercept"),
    "pwv": ("pwv", "pwv_fielding_xcor"),
}


class TableMissingError(RuntimeError):
    """Raised when a required catalog table is absent."""


@dataclass(frozen=True)
class QcReviewDecision:
    """One OK/FAIL decision for a subject measurement."""

    variable_id: str
    region_id: str
    qc_status: str
    comment: str = ""


def normalize_qc_status(value: str) -> str:
    status = str(value or "").strip().upper()
    if status not in QC_STATUS_VALUES:
        raise ValueError(f"qc_status must be one of {sorted(QC_STATUS_VALUES)}, got {value!r}")
    return status


def publish_qvtpy_qc_reviews(
    *,
    subject_uid: str,
    decisions: Sequence[QcReviewDecision] | Iterable[dict[str, Any]],
    repo: DataRepo | None = None,
    reviewer: str = "",
    build_sqlite_index: bool = False,
    pipeline_id: str = QVTPY_PIPELINE_ID,
) -> dict[str, int]:
    """Set ``qc_status`` on matching ``image_measurements`` rows.

    Parameters
    ----------
    subject_uid
        Subject to update.
    decisions
        Iterable of :class:`QcReviewDecision` or dicts with keys
        ``variable_id``, ``region_id``, ``qc_status``, optional ``comment``.
    """
    repo = resolve_repo(repo)
    subject = str(subject_uid).strip()
    if not subject:
        raise ValueError("subject_uid is required")
    if not repo.catalog.table_exists("image_measurements"):
        raise TableMissingError("image_measurements")

    parsed: list[QcReviewDecision] = []
    for item in decisions:
        if isinstance(item, QcReviewDecision):
            parsed.append(item)
            continue
        parsed.append(
            QcReviewDecision(
                variable_id=str(item.get("variable_id") or "").strip(),
                region_id=str(item.get("region_id") or "").strip(),
                qc_status=normalize_qc_status(str(item.get("qc_status") or "")),
                comment=str(item.get("comment") or "").strip(),
            )
        )
    if not parsed:
        return {"updated": 0, "decisions": 0}

    all_df = repo.get("image_measurements", cohort_id=False)
    if all_df.empty:
        log.warning("QC review: image_measurements is empty")
        return {"updated": 0, "decisions": len(parsed)}

    df = all_df.copy()
    now = utc_now_iso()
    updated = 0
    for decision in parsed:
        status = normalize_qc_status(decision.qc_status)
        var = str(decision.variable_id).strip()
        region = str(decision.region_id).strip()
        if not var:
            continue
        mask = (
            (df["subject_uid"].astype("string").fillna("") == subject)
            & (df["pipeline_id"].astype("string").fillna("") == str(pipeline_id))
            & (df["variable_id"].astype("string").fillna("") == var)
        )
        if region and "region_id" in df.columns:
            mask = mask & (df["region_id"].astype("string").fillna("") == region)
        if not mask.any():
            log.warning(
                "QC review: no rows for subject=%s pipeline=%s variable=%s region=%s",
                subject,
                pipeline_id,
                var,
                region or "*",
            )
            continue
        df.loc[mask, "qc_status"] = status
        if "updated_at" in df.columns:
            df.loc[mask, "updated_at"] = now
        # Store optional comment in value_text only when empty / unused.
        if decision.comment and "value_text" in df.columns:
            # Prefer a dedicated comment column if present; else leave value_text alone.
            if "qc_comment" in df.columns:
                df.loc[mask, "qc_comment"] = decision.comment
        updated += int(mask.sum())

    if updated:
        provenance: dict[str, Any] = {
            "importer": "qvtpy_qc_review",
            "subject_uid": subject,
            "pipeline_id": str(pipeline_id),
            "reviewer": str(reviewer).strip(),
            "n_decisions": len(parsed),
            "n_updated": updated,
        }
        repo.write_table(
            "image_measurements",
            df,
            provenance=provenance,
            build_sqlite_index=build_sqlite_index,
        )
    return {"updated": updated, "decisions": len(parsed)}


def try_publish_qvtpy_qc_reviews(**kwargs: Any) -> dict[str, int] | None:
    """Best-effort wrapper around :func:`publish_qvtpy_qc_reviews`."""
    try:
        return publish_qvtpy_qc_reviews(**kwargs)
    except Exception as exc:
        log.warning("QC review publish failed: %s", exc)
        return None


def subject_qc_status_summary(
    repo: DataRepo | None = None,
    *,
    pipeline_id: str = QVTPY_PIPELINE_ID,
) -> dict[str, str]:
    """Per-subject revision status from ``image_measurements.qc_status``.

    Returns a dict mapping ``subject_uid`` to one of:
    - ``"revised"``  — every QC-relevant row for the subject has OK or FAIL.
    - ``"partial"``  — some rows are reviewed, others still PENDING / blank.
    - ``"pending"``  — no rows have been reviewed yet (all PENDING or no rows).
    """
    repo = resolve_repo(repo)
    if not repo.catalog.table_exists("image_measurements"):
        return {}
    try:
        df = repo.get(
            "image_measurements",
            cohort_id=False,
            columns=["subject_uid", "pipeline_id", "variable_id", "qc_status"],
        )
    except Exception:
        return {}
    if df.empty:
        return {}
    df = df[df["pipeline_id"].astype(str) == str(pipeline_id)]
    qc_vars = set()
    for vars_ in QC_METRIC_VARIABLES.values():
        qc_vars.update(vars_)
    if qc_vars:
        df = df[df["variable_id"].astype(str).isin(qc_vars)]
    if df.empty:
        return {}
    out: dict[str, str] = {}
    for subj, grp in df.groupby("subject_uid"):
        statuses = grp["qc_status"].astype(str).str.strip().str.upper()
        reviewed = statuses.isin({"OK", "FAIL"})
        if reviewed.all():
            out[str(subj)] = "revised"
        elif reviewed.any():
            out[str(subj)] = "partial"
        else:
            out[str(subj)] = "pending"
    return out


__all__ = [
    "QC_METRIC_VARIABLES",
    "QC_STATUS_VALUES",
    "QcReviewDecision",
    "TableMissingError",
    "normalize_qc_status",
    "publish_qvtpy_qc_reviews",
    "subject_qc_status_summary",
    "try_publish_qvtpy_qc_reviews",
]
