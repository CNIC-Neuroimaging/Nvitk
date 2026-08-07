"""
Per-group descriptive statistics for one measurement.

Description
-----------
The question "what is the mean flow per territory, and how variable is it?" comes up constantly, and
answering it currently means exporting the whole frame and pivoting it in a spreadsheet. This module
is that table: one row per group, the statistics a methods section actually reports, computed once
and identically whether they are read on screen or written to Excel.

What is reported, and why
-------------------------
=================  ============================================================================
Statistic          Why it is here
=================  ============================================================================
``n`` / ``n_missing``  A mean over 8 subjects and a mean over 300 are different claims.
``mean`` / ``sd``  What gets reported, and what a normal-ish distribution needs.
``sem`` / ``ci_low`` / ``ci_high``  Precision of the mean, which the SD alone does not give.
``median`` / ``q1`` / ``q3`` / ``iqr``  What to report instead when the distribution is skewed.
``min`` / ``max``  Where to look for the segmentation failures.
``cv``             Scale-free spread, so territories at very different flows stay comparable.
``skew``           Whether the mean or the median is the honest summary.
=================  ============================================================================

Both the parametric and the robust summaries are always produced rather than one being chosen for
you: 4D-flow measurements are frequently skewed by a handful of bad segmentations, and a table that
silently reported only the mean would hide exactly that.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Column order of the emitted summary, so every export looks the same.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "group", "n", "n_missing", "mean", "sd", "sem", "ci_low", "ci_high",
    "median", "q1", "q3", "iqr", "min", "max", "cv", "skew",
)

#: Label of the row summarising every group together.
OVERALL_LABEL = "ALL"


def _t_critical(n: int, ci: float) -> float:
    """
    Two-sided critical value for a mean's confidence interval.

    Student's *t* rather than the normal quantile: territories can end up with a handful of subjects
    after filtering, where 1.96 understates the interval noticeably.
    """
    if n < 2:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(0.5 + ci / 2.0, df=n - 1))
    except Exception:
        # SciPy is optional here; the normal approximation is close enough above ~30 and the
        # interval is labelled with the level either way.
        return 1.959963984540054 if abs(ci - 0.95) < 1e-9 else float("nan")


def _describe(values: pd.Series, *, label: str, ci: float) -> dict[str, Any]:
    """One row of statistics for a single group's values."""
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.dropna()
    n = int(len(present))

    row: dict[str, Any] = {
        "group": label,
        "n": n,
        "n_missing": int(len(numeric) - n),
    }
    if n == 0:
        return {**row, **{k: np.nan for k in SUMMARY_COLUMNS if k not in row}}

    mean = float(present.mean())
    # ddof=1: these are samples of a population, not the population.
    sd = float(present.std(ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    critical = _t_critical(n, ci)
    q1, q3 = (float(present.quantile(0.25)), float(present.quantile(0.75)))

    row.update({
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - critical * sem if np.isfinite(critical) and np.isfinite(sem) else np.nan,
        "ci_high": mean + critical * sem if np.isfinite(critical) and np.isfinite(sem) else np.nan,
        "median": float(present.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(present.min()),
        "max": float(present.max()),
        # A CV around a mean near zero is meaningless rather than merely large.
        "cv": (sd / abs(mean)) if (np.isfinite(sd) and abs(mean) > 1e-12) else np.nan,
        "skew": float(present.skew()) if n > 2 else np.nan,
    })
    return row


def summarize_by_group(
    frame: pd.DataFrame,
    column: str,
    *,
    by: str | Sequence[str] = "",
    ci: float = 0.95,
    include_overall: bool = True,
    sort_by: str = "group",
) -> pd.DataFrame:
    """
    Descriptive statistics of *column*, one row per group of *by*.

    Parameters
    ----------
    by : str or sequence of str
        Grouping column(s). Several are combined into one label (``Left_ICA | male``) rather than a
        MultiIndex, so the sheet stays flat and readable. Empty summarises the column as a whole.
    include_overall : bool
        Append an :data:`OVERALL_LABEL` row over every group. Worth having in the same table — a
        per-territory mean is usually read against the whole-cohort one.
    sort_by : {"group", "mean", "median", "n"}
        Row order. ``"group"`` sorts naturally, so ``g2`` precedes ``g10``.

    Returns
    -------
    pandas.DataFrame
        Columns as in :data:`SUMMARY_COLUMNS`. ``frame.attrs`` carries ``column``, ``by`` and
        ``ci`` so an exporter can label the sheet without being told again.

    Raises
    ------
    ValueError
        When *column* or a grouping column is absent, or *column* holds nothing numeric.
    """
    from .region_groups import natural_level_key

    if column not in frame.columns:
        raise ValueError(f"Column {column!r} is not in the frame.")
    keys = [str(by)] if isinstance(by, str) and by else [str(b) for b in (by or [])]
    missing = [k for k in keys if k not in frame.columns]
    if missing:
        raise ValueError(f"Grouping column(s) {', '.join(missing)} are not in the frame.")

    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        raise ValueError(
            f"{column!r} has no numeric values to summarise. Pick a measurement column, or use "
            f"the distribution plot for a categorical one."
        )

    rows: list[dict[str, Any]] = []
    if keys:
        labels = frame[keys[0]].astype(str)
        for extra in keys[1:]:
            labels = labels.str.cat(frame[extra].astype(str), sep=" | ")
        for label, index in labels.groupby(labels).groups.items():
            rows.append(_describe(values.loc[index], label=str(label), ci=ci))
    if include_overall or not keys:
        rows.append(_describe(values, label=OVERALL_LABEL, ci=ci))

    out = pd.DataFrame(rows).reindex(columns=list(SUMMARY_COLUMNS))
    # The overall row is a footer, not a group — keep it last whatever the sort.
    overall = out.loc[out["group"] == OVERALL_LABEL]
    groups = out.loc[out["group"] != OVERALL_LABEL]
    if sort_by == "group":
        groups = groups.reindex(
            groups["group"].map(natural_level_key).sort_values(kind="stable").index
        )
    elif sort_by in groups.columns:
        groups = groups.sort_values(sort_by, ascending=False, kind="stable")

    out = pd.concat([groups, overall], ignore_index=True)
    out.attrs.update({"column": column, "by": keys, "ci": float(ci)})
    return out


def summary_provenance(
    summary: pd.DataFrame, *, dataset: str = "", n_rows_source: int | None = None
) -> pd.DataFrame:
    """
    Two-column record of what a summary describes, for the export's second sheet.

    A table of means with no note of which column, which grouping, which confidence level or how
    many rows it came from is unreconstructable a month later — including by its author.
    """
    from datetime import datetime, timezone

    keys = summary.attrs.get("by") or []
    ci = float(summary.attrs.get("ci", 0.95))
    rows: list[tuple[str, str]] = [
        ("Exported", datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")),
        ("Dataset", str(dataset)),
        ("Measurement", str(summary.attrs.get("column", ""))),
        ("Grouped by", ", ".join(keys) if keys else "(whole cohort)"),
        ("Groups", str(int((summary["group"] != OVERALL_LABEL).sum()))),
        ("Confidence level", f"{ci:.0%}"),
        ("Interval", "Student's t on the mean" ),
        ("SD convention", "sample (ddof=1)"),
    ]
    if n_rows_source is not None:
        rows.append(("Rows summarised", str(int(n_rows_source))))
    rows.append((
        "Note",
        "Both parametric (mean/SD/CI) and robust (median/IQR) summaries are given. Prefer the "
        "robust ones when |skew| is large — 4D-flow measurements are frequently skewed by a few "
        "bad segmentations.",
    ))
    return pd.DataFrame(rows, columns=["item", "detail"])


__all__ = [
    "OVERALL_LABEL",
    "SUMMARY_COLUMNS",
    "summarize_by_group",
    "summary_provenance",
]
