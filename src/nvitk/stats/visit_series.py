"""
Variables across visits — per-visit columns, longitudinal derivatives, long reshape.

Description
-----------
A cohort measures the same variable at several visits: carotid plaque volume at visits 1–4,
blood pressure at every visit, a questionnaire at two. The statmodels frame is one row per
subject, so a variable normally arrives *collapsed* to one visit (latest, or a pinned one). This
module keeps every visit instead:

- :func:`pivot_visits` turns the long ``(subject, visit, value)`` table into one column per visit,
  ``total_carotid_plaque_vol_v1 … _v4`` — a **visit family**;
- :func:`visit_series_values` computes a subject-level longitudinal summary from a family —
  change between two visits (absolute, percent, annualised), area under the curve, time-averaged
  level, slope, mean / max / min / SD, first / last value, number of visits, progression and
  new-onset flags;
- :func:`melt_visit_families` reshapes families to one row per subject × visit, with a time
  column, for trajectory models (``plaque ~ years + (years | subject)``).

Time axis
---------
Change rates, AUC and slopes need *when* each visit happened. Three choices, per call:

- ``""`` — the visit number itself (visit ``"3"`` → 3.0);
- a **date family** (e.g. ``peqdate_v1 … _v4``, the physical-exam date) → years since the
  subject's first visit in the selection, so unequal visit spacing is respected;
- a **numeric family** (e.g. age at each visit) → used as is.

Column naming
-------------
A visit family is ``<variable>_v<visit>`` with at least two visits. :func:`visit_families` finds
them from the column names alone, so a family survives a saved dataset or an imported session.

I/O: pandas frames in and out; NumPy on the host (small per-subject arithmetic).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Separator between a variable and its visit in a per-visit column name.
VISIT_SUFFIX = "_v"

#: ``plaque_vol_v3`` → (``plaque_vol``, ``3``). Visit ids are short tokens (digits/letters).
_VISIT_COLUMN = re.compile(r"^(?P<base>.+)_v(?P<visit>[0-9]+[A-Za-z]?|[A-Za-z][0-9]*)$")

#: Name of the visit / time columns a long reshape adds.
VISIT_COLUMN_NAME = "visit"
VISIT_TIME_COLUMN = "visit_time"

#: operation key → (label, needs two visits, uses time axis)
VISIT_OPERATIONS: dict[str, tuple[str, bool, bool]] = {
    "delta": ("Difference (to − from)", True, False),
    "pct_change": ("Percent change (to − from) / from × 100", True, False),
    "annualized": ("Change per time unit (to − from) / Δt", True, True),
    "auc": ("Area under the curve (trapezoid)", False, True),
    "auc_mean": ("Time-averaged level (AUC / time span)", False, True),
    "slope": ("Slope over time (least squares)", False, True),
    "mean": ("Mean over visits", False, False),
    "max": ("Maximum over visits", False, False),
    "min": ("Minimum over visits", False, False),
    "sd": ("Standard deviation over visits", False, False),
    "first": ("First available value", False, False),
    "last": ("Last available value", False, False),
    "n_visits": ("Number of visits with a value", False, False),
    "progressed": ("Progressed: to > from (0/1)", True, False),
    "new_onset": ("New onset: from ≤ threshold, any later > threshold (0/1)", False, False),
}


# ---------------------------------------------------------------------------
# Naming and discovery
# ---------------------------------------------------------------------------


def visit_label(visit: Any) -> str:
    """A visit id as text, with ``4.0`` → ``"4"`` (ids come back from parquet as floats)."""
    text = str(visit).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def visit_sort_key(visit: Any) -> tuple:
    """Order visits numerically when they look like numbers (``"10"`` after ``"9"``)."""
    text = visit_label(visit)
    match = re.match(r"^(\d+)(.*)$", text)
    return (0, int(match.group(1)), match.group(2)) if match else (1, 0, text)


def visit_column(variable: str, visit: Any) -> str:
    """``plaque_vol`` at visit 3 → ``plaque_vol_v3``."""
    return f"{variable}{VISIT_SUFFIX}{visit_label(visit)}"


def visit_families(columns: Sequence[str] | pd.DataFrame, *, min_visits: int = 2) -> dict[str, dict[str, str]]:
    """
    ``{variable: {visit: column}}`` for every per-visit family among *columns*.

    Visits are ordered numerically. A base with fewer than *min_visits* columns is not a family
    (``systolic_v2`` alone is just a column whose name ends in ``_v2``).
    """
    names = list(columns.columns) if isinstance(columns, pd.DataFrame) else list(columns)
    found: dict[str, dict[str, str]] = {}
    for name in names:
        match = _VISIT_COLUMN.match(str(name))
        if match:
            found.setdefault(match.group("base"), {})[match.group("visit")] = str(name)
    return {
        base: dict(sorted(visits.items(), key=lambda kv: visit_sort_key(kv[0])))
        for base, visits in sorted(found.items())
        if len(visits) >= int(min_visits)
    }


# ---------------------------------------------------------------------------
# Pivot: long → one column per visit
# ---------------------------------------------------------------------------


def pivot_visits(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    visits: Mapping[str, Sequence[str] | None] | Sequence[str] | None = None,
    subject_key: str = "subject_uid",
    visit_key: str = "visit_id",
) -> pd.DataFrame:
    """
    One row per subject, one column per (variable, visit): ``<variable>_v<visit>``.

    Parameters
    ----------
    frame
        Per-visit frame with *subject_key*, *visit_key* and the variables as columns (what
        ``repo.clinical(..., wide=True)`` returns).
    variables
        Which columns to spread.
    visits
        Restrict the visits: one list for every variable, or ``{variable: [visit, …]}``;
        ``None`` keeps every visit present.
    """
    if frame.empty or subject_key not in frame.columns or visit_key not in frame.columns:
        return pd.DataFrame(columns=[subject_key])
    present = [v for v in variables if v in frame.columns]
    if not present:
        return pd.DataFrame(columns=[subject_key])
    work = frame[[subject_key, visit_key, *present]].copy()
    work[visit_key] = work[visit_key].map(visit_label)
    out = pd.DataFrame({subject_key: pd.unique(work[subject_key].dropna())})
    for variable in present:
        if isinstance(visits, Mapping):
            allowed = visits.get(variable)
        else:
            allowed = visits
        sub = work[[subject_key, visit_key, variable]].dropna(subset=[variable])
        if allowed:
            keep = {visit_label(v) for v in allowed}
            sub = sub[sub[visit_key].isin(keep)]
        if sub.empty:
            continue
        # Duplicate (subject, visit) rows would be a data error; keep the last one deterministically.
        sub = sub.drop_duplicates(subset=[subject_key, visit_key], keep="last")
        wide = sub.pivot(index=subject_key, columns=visit_key, values=variable)
        wide = wide[sorted(wide.columns, key=visit_sort_key)]
        wide.columns = [visit_column(variable, v) for v in wide.columns]
        out = out.merge(wide.reset_index(), on=subject_key, how="left")
    return out


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------


def _family_matrix(df: pd.DataFrame, family: Mapping[str, str], visits: Sequence[str]) -> np.ndarray:
    """``(n_rows, n_visits)`` float matrix of a family's values (NaN where missing)."""
    cols = []
    for visit in visits:
        column = family.get(visit)
        cols.append(
            pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            if column in df.columns else np.full(len(df), np.nan)
        )
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def visit_times(
    df: pd.DataFrame,
    visits: Sequence[str],
    *,
    time_family: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, str]:
    """
    ``(times, unit)`` — an ``(n_rows, n_visits)`` matrix placing each visit in time.

    Without *time_family*: the visit numbers (``unit = "visit"``). With a family of dates: years
    since each row's earliest dated visit in *visits* (``unit = "years"``). With a numeric family
    (age, months since baseline…): its values as they are (``unit = "time"``).
    """
    if not time_family:
        numbers = []
        for visit in visits:
            try:
                numbers.append(float(visit_label(visit)))
            except ValueError:
                numbers.append(float(len(numbers)))
        return np.tile(np.asarray(numbers, dtype=float), (len(df), 1)), "visit"

    raw = [df[time_family[v]] if time_family.get(v) in df.columns else pd.Series(np.nan, index=df.index)
           for v in visits]
    numeric = [pd.to_numeric(r, errors="coerce") for r in raw]
    n_numeric = sum(int(n.notna().sum()) for n in numeric)
    n_values = sum(int(r.notna().sum()) for r in raw)
    if n_values and n_numeric >= 0.9 * n_values:
        return np.column_stack([n.to_numpy(dtype=float) for n in numeric]), "time"
    dates = [pd.to_datetime(r, errors="coerce") for r in raw]
    days = np.column_stack([
        (d - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        for d in dates
    ])
    origin = np.nanmin(np.where(np.isfinite(days), days, np.inf), axis=1, keepdims=True)
    origin = np.where(np.isfinite(origin), origin, np.nan)
    return (days - origin) / 365.25, "years"


# ---------------------------------------------------------------------------
# Longitudinal summaries
# ---------------------------------------------------------------------------


def _pick_visit(visits: Sequence[str], wanted: str, default_index: int) -> int:
    """Index of *wanted* in *visits*, or *default_index* when it is blank."""
    if wanted:
        label = visit_label(wanted)
        if label not in visits:
            raise ValueError(f"visit {wanted!r} is not among {list(visits)}")
        return list(visits).index(label)
    return default_index


def visit_series_values(
    df: pd.DataFrame,
    family: str,
    operation: str,
    *,
    visits: Sequence[str] = (),
    time_family: str = "",
    visit_a: str = "",
    visit_b: str = "",
    threshold: float = 0.0,
    min_visits: int = 2,
) -> pd.Series:
    """
    A subject-level longitudinal summary of the visit family *family* (one value per row).

    Parameters
    ----------
    df
        Frame holding ``<family>_v<visit>`` columns (and the time family's, when used).
    operation
        One of :data:`VISIT_OPERATIONS`.
    visits
        Visits to use (default: every visit of the family). Order follows the visit numbers.
    time_family
        Base name of a date or numeric visit family for the time axis; ``""`` = visit number.
    visit_a, visit_b
        "From" and "to" visits for the two-visit operations; default first and last selected.
    threshold
        For ``new_onset``: values at or below it count as "absent".
    min_visits
        Rows with fewer non-missing values among *visits* get NaN for AUC, slope and the
        summaries — a slope through one point is not a slope.

    Returns
    -------
    pandas.Series
        Float series aligned to *df*'s index.
    """
    if operation not in VISIT_OPERATIONS:
        raise ValueError(f"unknown visit operation {operation!r}")
    families = visit_families(df)
    if family not in families:
        raise ValueError(f"no visit family {family!r} (columns {family}_v1, {family}_v2, …) in the frame")
    fam = families[family]
    chosen = [visit_label(v) for v in visits] if visits else list(fam)
    chosen = [v for v in chosen if v in fam]
    if not chosen:
        raise ValueError(f"none of the visits {list(visits)} exist for {family}")
    values = _family_matrix(df, fam, chosen)
    n_present = np.sum(np.isfinite(values), axis=1)
    _label, two_visit, uses_time = VISIT_OPERATIONS[operation]

    if two_visit:
        ia = _pick_visit(chosen, visit_a, 0)
        ib = _pick_visit(chosen, visit_b, len(chosen) - 1)
        if ia == ib:
            raise ValueError("'from' and 'to' must be different visits")
        a, b = values[:, ia], values[:, ib]
        with np.errstate(divide="ignore", invalid="ignore"):
            if operation == "delta":
                out = b - a
            elif operation == "pct_change":
                out = np.where(a != 0, (b - a) / np.abs(a) * 100.0, np.nan)
            elif operation == "progressed":
                out = np.where(np.isfinite(a) & np.isfinite(b), (b > a).astype(float), np.nan)
            else:  # annualized
                times, _unit = visit_times(df, chosen, time_family=families.get(time_family) if time_family else None)
                dt = times[:, ib] - times[:, ia]
                out = np.where(np.isfinite(dt) & (dt != 0), (b - a) / dt, np.nan)
        return pd.Series(out, index=df.index, dtype=float)

    enough = n_present >= max(int(min_visits), 1)
    if operation == "n_visits":
        return pd.Series(n_present.astype(float), index=df.index)
    if operation == "new_onset":
        base = values[:, 0]
        later = values[:, 1:]
        with np.errstate(invalid="ignore"):
            any_later = np.nanmax(np.where(np.isfinite(later), later, -np.inf), axis=1) > threshold
        out = np.where(np.isfinite(base) & (base <= threshold), any_later.astype(float),
                       np.where(np.isfinite(base), 0.0, np.nan))
        return pd.Series(out, index=df.index, dtype=float)

    out = np.full(len(df), np.nan)
    if operation in ("mean", "max", "min", "sd", "first", "last"):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # All-NaN rows (a subject never measured) warn; they become NaN below anyway.
            warnings.simplefilter("ignore", RuntimeWarning)
            if operation == "mean":
                out = np.nanmean(values, axis=1)
            elif operation == "max":
                out = np.nanmax(values, axis=1)
            elif operation == "min":
                out = np.nanmin(values, axis=1)
            elif operation == "sd":
                out = np.nanstd(values, axis=1, ddof=1)
        if operation in ("first", "last"):
            for i, row in enumerate(values):
                finite = np.flatnonzero(np.isfinite(row))
                if finite.size:
                    out[i] = row[finite[0] if operation == "first" else finite[-1]]
        return pd.Series(np.where(enough, out, np.nan), index=df.index, dtype=float)

    # ---- time-based: auc, auc_mean, slope -------------------------------------
    times, _unit = visit_times(df, chosen, time_family=families.get(time_family) if time_family else None)
    for i in range(len(df)):
        if not enough[i]:
            continue
        mask = np.isfinite(values[i]) & np.isfinite(times[i])
        if mask.sum() < max(int(min_visits), 2):
            continue
        t, y = times[i][mask], values[i][mask]
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
        if operation == "slope":
            if np.ptp(t) == 0:
                continue
            out[i] = np.polyfit(t, y, 1)[0]
        else:
            area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))
            if operation == "auc":
                out[i] = area
            else:
                span = float(t[-1] - t[0])
                out[i] = area / span if span > 0 else np.nan
    return pd.Series(out, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Long reshape
# ---------------------------------------------------------------------------


def melt_visit_families(
    df: pd.DataFrame,
    families: Sequence[str],
    *,
    time_family: str = "",
    visit_col: str = VISIT_COLUMN_NAME,
    time_col: str = VISIT_TIME_COLUMN,
) -> pd.DataFrame:
    """
    Spread visit families down the rows: one row per (original row, visit).

    Each melted family becomes one column named after its variable (``plaque_vol``); every other
    column repeats down the rows. *visit_col* holds the visit id and *time_col* its position in
    time (see :func:`visit_times`: visit number, years since the subject's first visit from a date
    family, or a numeric family's value). A visit present for one family but not another leaves the
    other missing on that row. Rows where every melted family is missing are dropped.

    Raises
    ------
    ValueError
        When none of *families* is a visit family of *df*.
    """
    all_families = visit_families(df)
    picked = [f for f in families if f in all_families]
    if not picked:
        raise ValueError(f"no visit family among {list(families)}")
    visits = sorted({v for f in picked for v in all_families[f]}, key=visit_sort_key)
    melted_cols = {c for f in picked for c in all_families[f].values()}
    time_fam = all_families.get(time_family) if time_family else None
    if time_fam:
        melted_cols |= set(time_fam.values())
    keep = [c for c in df.columns if c not in melted_cols]
    times, unit = visit_times(df, visits, time_family=time_fam)

    pieces = []
    for j, visit in enumerate(visits):
        part = df[keep].copy()
        part[visit_col] = visit
        part[time_col] = times[:, j]
        for fam in picked:
            column = all_families[fam].get(visit)
            part[fam] = pd.to_numeric(df[column], errors="coerce") if column else np.nan
        pieces.append(part)
    long = pd.concat(pieces, ignore_index=True)
    long = long.dropna(subset=picked, how="all")
    sort_keys = [c for c in ("subject_uid", "territory") if c in long.columns]
    long = long.sort_values([*sort_keys, time_col, visit_col], kind="stable").reset_index(drop=True)
    long.attrs["visit_time_unit"] = unit
    log.info(
        "Melted %s over %d visit(s) → %d rows (time: %s).",
        ", ".join(picked), len(visits), len(long), unit if time_fam else "visit number",
    )
    return long


__all__ = [
    "VISIT_COLUMN_NAME",
    "VISIT_OPERATIONS",
    "VISIT_SUFFIX",
    "VISIT_TIME_COLUMN",
    "melt_visit_families",
    "pivot_visits",
    "visit_column",
    "visit_families",
    "visit_label",
    "visit_series_values",
    "visit_sort_key",
    "visit_times",
]
