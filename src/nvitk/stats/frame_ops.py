"""
Analysis-frame operations — declarative row filters and derived columns.

Description
-----------
Pure-pandas building blocks shared by the Statmodels GUI and by notebooks. Two families:

``FilterRule`` / :func:`apply_filter_rules`
    A declarative, serializable description of a row filter (keep/exclude a set of levels, compare
    against a threshold, keep a numeric range, or drop Tukey-fence outliers). Rules compose: they are
    applied in order against the *same* frame, and each one reports how many rows it removed. A rule
    whose column is absent from the frame is **skipped and reported**, never treated as "drop
    everything" — that is what lets a saved filter set survive a reload that changes the column set.

``DerivedColumn`` / :func:`apply_derived_columns`
    Named columns computed from existing ones, either by a canned transform (``log``, ``zscore``, …)
    or by a free-form expression over the frame. They are materialized as real columns so they work
    everywhere a measurement does: as a formula LHS/RHS, as a plot axis, and as a filter target.

Both dataclasses round-trip through ``to_dict`` / ``from_dict`` so a GUI session can be saved to JSON.

Numerical conventions
---------------------
Out-of-domain inputs to ``log`` / ``sqrt`` / ``inverse`` become ``NaN``, never ``±inf``. Infinities
survive :func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm`'s ``dropna`` and turn a whole fit into NaN
parameters with no error; ``NaN`` is dropped and counted in the fit metadata instead.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

# Comparison operators accepted by ``FilterRule(kind="compare")``.
FILTER_OPS = (">", ">=", "<", "<=", "==", "!=", "contains", "equals")

# Tukey fence multiplier used by the IQR outlier filter.
DEFAULT_IQR_K = 1.5

# Derived-column and measurement-alias names must be valid Python identifiers: ``fit_or_load_mixedlm``
# sanitizes every column through ``_safe_col``, so ``log(pi)`` would silently become ``log_pi_`` in the
# model frame and stop matching the analysis frame.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ──────────────────────────────────────────────────────────────────────────────
# Row filters
# ──────────────────────────────────────────────────────────────────────────────
def apply_row_filter(df: pd.DataFrame, column: str, op: str, value: str) -> pd.DataFrame:
    """Filter *df* to rows where *column* satisfies *op* against *value* (numeric comparisons,
    substring/exact string match, or equality/inequality); returns *df* unchanged if *column* is
    missing or blank."""
    if df.empty or not column or column not in df.columns:
        return df
    series = df[column]
    op = (op or "==").strip()
    raw = value.strip()

    if op in {">", ">=", "<", "<="}:
        num = pd.to_numeric(series, errors="coerce")
        try:
            threshold = float(raw)
        except ValueError:
            return df.iloc[0:0]
        if op == ">":
            mask = num > threshold
        elif op == ">=":
            mask = num >= threshold
        elif op == "<":
            mask = num < threshold
        else:
            mask = num <= threshold
        return df.loc[mask]

    if op == "contains":
        return df.loc[series.astype(str).str.contains(raw, case=False, na=False)]
    if op == "equals":
        return df.loc[series.astype(str).str.lower() == raw.lower()]

    if op == "!=":
        return df.loc[series.astype(str) != raw]
    return df.loc[series.astype(str) == raw]


def apply_iqr_filter(
    df: pd.DataFrame,
    column: str,
    *,
    k: float = DEFAULT_IQR_K,
    by: str | None = None,
) -> pd.DataFrame:
    """
    Drop rows whose *column* value falls outside the Tukey fences ``[Q1 - k·IQR, Q3 + k·IQR]``.

    With *by* set, the fences are computed within each level of that column. That is the meaningful
    scope for image measurements, whose magnitude depends on the region: a global fence is driven by
    the spread *between* regions rather than within them, so it misses real outliers inside each
    region and — when one region sits far from the rest — can discard that region wholesale. Rows
    with a missing value are kept; dropping them is the fit's job, not the filter's.
    """
    if df.empty or column not in df.columns:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    if not values.notna().any():
        return df

    def fence_mask(series: pd.Series) -> pd.Series:
        """Boolean mask of rows inside the Tukey fences of *series* (NaN kept)."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            return pd.Series(True, index=series.index)
        iqr = q3 - q1
        if iqr <= 0:  # degenerate spread: nothing is an outlier by this rule
            return pd.Series(True, index=series.index)
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return series.isna() | series.between(lo, hi)

    if by and by in df.columns:
        mask = pd.Series(True, index=df.index)
        for _, idx in df.groupby(by, sort=False).groups.items():
            mask.loc[idx] = fence_mask(values.loc[idx])
    else:
        mask = fence_mask(values)
    return df.loc[mask]


def _apply_values_rule(df: pd.DataFrame, column: str, values: Sequence[str], exclude: bool) -> pd.DataFrame:
    """Keep (or, with *exclude*, drop) rows whose *column* level is listed in *values*."""
    wanted = {str(v) for v in values}
    if not wanted:
        return df
    hit = df[column].astype(str).isin(wanted)
    return df.loc[~hit] if exclude else df.loc[hit]


def _apply_range_rule(
    df: pd.DataFrame, column: str, low: float | None, high: float | None
) -> pd.DataFrame:
    """Keep rows whose numeric *column* value lies in ``[low, high]`` (either bound may be open)."""
    num = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if low is not None:
        mask &= num >= float(low)
    if high is not None:
        mask &= num <= float(high)
    return df.loc[mask.fillna(False)]


@dataclass(frozen=True)
class FilterRule:
    """
    One declarative row filter over a single column.

    Parameters
    ----------
    column : str
        Column the rule is anchored to.
    kind : {"values", "compare", "range", "iqr"}
        ``values``  keep/exclude an explicit set of levels (the vessel-exclusion case);
        ``compare`` apply *op* / *value* through :func:`apply_row_filter`;
        ``range``   keep rows inside ``[low, high]``;
        ``iqr``     drop Tukey-fence outliers, optionally per level of *by*.
    enabled : bool
        Disabled rules are kept (so the UI can toggle them) but contribute no filtering.
    """

    column: str
    kind: str
    # ---- kind="compare" --------------------------------------------------------
    op: str = "=="
    value: str = ""
    # ---- kind="values" ---------------------------------------------------------
    values: tuple[str, ...] = ()
    exclude: bool = False
    # ---- kind="range" ----------------------------------------------------------
    low: float | None = None
    high: float | None = None
    # ---- kind="iqr" ------------------------------------------------------------
    k: float = DEFAULT_IQR_K
    by: str | None = None
    enabled: bool = True

    def label(self) -> str:
        """Short human-readable description, used as the chip text in the GUI."""
        if self.kind == "values":
            shown = list(self.values)[:4]
            more = len(self.values) - len(shown)
            body = ", ".join(shown) + (f", +{more}" if more > 0 else "")
            sign = "∉" if self.exclude else "∈"
            return f"{self.column} {sign} {{{body}}}"
        if self.kind == "compare":
            return f"{self.column} {self.op} {self.value}"
        if self.kind == "range":
            lo = "−∞" if self.low is None else f"{float(self.low):.4g}"
            hi = "+∞" if self.high is None else f"{float(self.high):.4g}"
            return f"{lo} ≤ {self.column} ≤ {hi}"
        if self.kind == "iqr":
            scope = f"per {self.by}" if self.by else "global"
            return f"IQR {self.column} (k={float(self.k):g}, {scope})"
        return f"{self.column} [{self.kind}]"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (tuples become lists)."""
        return {
            "column": self.column,
            "kind": self.kind,
            "op": self.op,
            "value": self.value,
            "values": list(self.values),
            "exclude": bool(self.exclude),
            "low": self.low,
            "high": self.high,
            "k": float(self.k),
            "by": self.by,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FilterRule":
        """Rebuild a rule from :meth:`to_dict` output, tolerating missing keys."""
        return cls(
            column=str(data.get("column") or ""),
            kind=str(data.get("kind") or "compare"),
            op=str(data.get("op") or "=="),
            value=str(data.get("value") or ""),
            values=tuple(str(v) for v in (data.get("values") or ())),
            exclude=bool(data.get("exclude", False)),
            low=None if data.get("low") is None else float(data["low"]),
            high=None if data.get("high") is None else float(data["high"]),
            k=float(data.get("k", DEFAULT_IQR_K)),
            by=(str(data["by"]) if data.get("by") else None),
            enabled=bool(data.get("enabled", True)),
        )


def apply_filter_rules(
    df: pd.DataFrame,
    rules: Sequence[FilterRule],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply *rules* to *df* in order, reporting the effect of each.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to filter. Never mutated.
    rules : sequence of FilterRule
        Applied left to right; each sees the output of the previous one.

    Returns
    -------
    (filtered, report)
        *report* has one entry per rule: ``{"rule", "n_before", "n_after", "removed", "skipped",
        "reason"}``. A rule referencing a column that is not in the frame is skipped with
        ``reason="column not in frame"`` and removes nothing — a saved filter set must not silently
        empty a frame whose columns changed.
    """
    out = df
    report: list[dict[str, Any]] = []
    for rule in rules:
        entry: dict[str, Any] = {
            "rule": rule,
            "n_before": int(len(out)),
            "n_after": int(len(out)),
            "removed": 0,
            "skipped": False,
            "reason": "",
        }
        if not rule.enabled:
            entry.update(skipped=True, reason="disabled")
            report.append(entry)
            continue
        if not rule.column or rule.column not in out.columns:
            entry.update(skipped=True, reason="column not in frame")
            report.append(entry)
            log.debug("Filter skipped: column %r not in frame.", rule.column)
            continue

        if rule.kind == "values":
            out = _apply_values_rule(out, rule.column, rule.values, rule.exclude)
        elif rule.kind == "compare":
            out = apply_row_filter(out, rule.column, rule.op, rule.value)
        elif rule.kind == "range":
            out = _apply_range_rule(out, rule.column, rule.low, rule.high)
        elif rule.kind == "iqr":
            by = rule.by if (rule.by and rule.by in out.columns) else None
            out = apply_iqr_filter(out, rule.column, k=float(rule.k), by=by)
        else:
            entry.update(skipped=True, reason=f"unknown kind {rule.kind!r}")
            report.append(entry)
            continue

        entry["n_after"] = int(len(out))
        entry["removed"] = entry["n_before"] - entry["n_after"]
        report.append(entry)
    return out, report


def filtered_columns(rules: Sequence[FilterRule]) -> set[str]:
    """Columns touched by at least one enabled rule (used to badge table headers)."""
    return {r.column for r in rules if r.enabled and r.column}


# ──────────────────────────────────────────────────────────────────────────────
# Derived columns
# ──────────────────────────────────────────────────────────────────────────────
def _zscore(series: pd.Series) -> pd.Series:
    """Standardize to zero mean / unit SD, ignoring NaN; constant input yields NaN."""
    num = pd.to_numeric(series, errors="coerce")
    sd = num.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (num - num.mean(skipna=True)) / sd


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce to float, turning non-numeric entries into NaN."""
    return pd.to_numeric(series, errors="coerce")


# Out-of-domain inputs are masked to NaN *before* the transform, so no entry can produce ±inf.
TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "log": lambda s: np.log(_numeric(s).where(_numeric(s) > 0)),
    "log1p": lambda s: np.log1p(_numeric(s).where(_numeric(s) > -1)),
    "sqrt": lambda s: np.sqrt(_numeric(s).where(_numeric(s) >= 0)),
    "zscore": _zscore,
    "inverse": lambda s: 1.0 / _numeric(s).where(_numeric(s) != 0),
    "rank": lambda s: _numeric(s).rank(method="average"),
}

TRANSFORM_LABELS: dict[str, str] = {
    "log": "log (natural)",
    "log1p": "log1p",
    "sqrt": "square root",
    "zscore": "z-score",
    "inverse": "inverse (1/x)",
    "rank": "rank",
}

# Names an expression may reference beyond the frame's own columns. Everything else — including
# builtins — is unavailable; see :func:`evaluate_expression`.
_SAFE_NAMESPACE: dict[str, Any] = {
    "log": np.log,
    "log1p": np.log1p,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "clip": np.clip,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "mean": np.nanmean,
    "std": np.nanstd,
    "median": np.nanmedian,
    "pi": np.pi,
    "e": np.e,
    "zscore": _zscore,
    "rank": lambda s: _numeric(pd.Series(s)).rank(method="average"),
}


def default_derived_name(source: str, transform: str) -> str:
    """Conventional name for a canned transform, e.g. ``("pi", "log") -> "log_pi"``."""
    return f"{transform}_{source}"


# ──────────────────────────────────────────────────────────────────────────────
# Binned categorical columns
# ──────────────────────────────────────────────────────────────────────────────
def parse_cut_points(text: str) -> tuple[float, ...]:
    """
    Parse a comma/space separated list of interior cut points, e.g. ``"0, 25, 100"``.

    Raises
    ------
    ValueError
        If a token is not a number, or the points are not strictly increasing — overlapping or
        repeated edges would make the bin a value falls into ambiguous.
    """
    tokens = [t for t in re.split(r"[,;\s]+", str(text or "").strip()) if t]
    if not tokens:
        raise ValueError("Enter at least one cut point.")
    points: list[float] = []
    for token in tokens:
        try:
            points.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{token!r} is not a number.") from exc
    if any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError("Cut points must be strictly increasing.")
    return tuple(points)


def default_bin_labels(n_bins: int, prefix: str) -> tuple[str, ...]:
    """``("cp0", "cp1", …)`` for *n_bins* bins — the ``tacsctot_group`` convention."""
    return tuple(f"{prefix}{i}" for i in range(int(n_bins)))


def default_bin_name(source: str) -> str:
    """Conventional name for a binned column, mirroring ``tacsctot`` → ``tacsctot_group``."""
    return f"{source}_group"


def bin_interval_labels(cut_points: Sequence[float], *, right: bool = True) -> list[str]:
    """
    Human-readable interval for each bin, e.g. ``["≤ 0", "(0, 25]", "(25, 100]", "> 100"]``.

    The bins are always bracketed by ±∞, so every finite value lands in exactly one of them and
    nothing is silently dropped for being off the end of the scale.
    """
    points = list(cut_points)
    if not points:
        return []
    out: list[str] = []
    lo_op, hi_op = ("≤", ">") if right else ("<", "≥")
    out.append(f"{lo_op} {points[0]:g}")
    for lo, hi in zip(points, points[1:]):
        out.append(f"({lo:g}, {hi:g}]" if right else f"[{lo:g}, {hi:g})")
    out.append(f"{hi_op} {points[-1]:g}")
    return out


def suggest_cut_points(series: pd.Series, *, method: str, n_bins: int = 4) -> tuple[float, ...]:
    """
    Propose interior cut points for *series*.

    The result is meant to be written into the definition as explicit numbers, not recomputed on
    every load: cut points derived live from the data would shift whenever a filter changed, so the
    same named group would mean different things between two fits.

    Parameters
    ----------
    method : {"quantile", "equal_width"}
    n_bins : int
        Number of bins wanted; ``n_bins - 1`` interior cut points are returned.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("The column has no numeric values to derive cut points from.")
    n_bins = max(2, int(n_bins))

    if method == "quantile":
        quantiles = [i / n_bins for i in range(1, n_bins)]
        points = [float(values.quantile(q)) for q in quantiles]
    elif method == "equal_width":
        lo, hi = float(values.min()), float(values.max())
        step = (hi - lo) / n_bins
        points = [lo + step * i for i in range(1, n_bins)]
    else:
        raise ValueError(f"Unknown cut-point method {method!r}.")

    # Round to something a person would type, then drop duplicates a skewed column can produce.
    rounded = sorted({float(f"{p:.4g}") for p in points})
    if len(rounded) < len(points):
        log.warning(
            "Cut points collapsed from %d to %d — the column is too concentrated for %d bins.",
            len(points),
            len(rounded),
            n_bins,
        )
    if not rounded:
        raise ValueError("Could not derive distinct cut points from this column.")
    return tuple(rounded)


def cut_into_bins(
    series: pd.Series,
    cut_points: Sequence[float],
    labels: Sequence[str],
    *,
    right: bool = True,
) -> pd.Series:
    """
    Bin a continuous *series* into an **ordered** categorical.

    The bins are ``(-inf, c0]``, ``(c0, c1]``, …, ``(cn, +inf)`` when *right* is true, so a value
    equal to a cut point falls in the lower bin — which is what makes "exactly 0" its own group when
    ``0`` is the first cut point.

    Ordered rather than plain categorical: patsy then treats the first label as the reference level
    and plots keep the bins in their natural sequence rather than alphabetically.
    """
    values = pd.to_numeric(series, errors="coerce")
    edges = [-np.inf, *[float(c) for c in cut_points], np.inf]
    binned = pd.cut(values, bins=edges, labels=list(labels), right=bool(right), ordered=True)
    return pd.Series(binned, index=series.index)


def bin_counts(series: pd.Series, spec: "DerivedColumn") -> pd.Series:
    """Rows falling in each bin of *spec*, in bin order — the preview that makes cut points checkable."""
    binned = cut_into_bins(series, spec.cut_points, spec.bin_labels(), right=spec.right)
    return binned.value_counts().reindex(spec.bin_labels()).fillna(0).astype(int)


def evaluate_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """
    Evaluate *expression* against *df*'s columns and return the resulting series.

    The expression is compiled and evaluated with ``__builtins__`` stripped and a namespace holding
    only the frame's columns plus :data:`_SAFE_NAMESPACE`. That blocks the obvious mistakes
    (``__import__``, ``open``) but is **not** a sandbox — an expression is code the user typed, and
    should be treated with the same trust as the rest of the session. Expressions restored from a
    config file that the user did not author must be confirmed before evaluation.

    Raises
    ------
    ValueError
        If the expression is blank, fails to compile/evaluate, or does not produce a value that can
        be aligned to *df*'s index.
    """
    text = str(expression or "").strip()
    if not text:
        raise ValueError("Expression is empty.")

    namespace: dict[str, Any] = dict(_SAFE_NAMESPACE)
    namespace.update({str(c): df[c] for c in df.columns})
    try:
        value = eval(compile(text, "<derived>", "eval"), {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc

    if isinstance(value, pd.Series):
        return value.reindex(df.index)
    if np.isscalar(value):
        return pd.Series(value, index=df.index)
    arr = np.asarray(value)
    if arr.ndim == 1 and len(arr) == len(df):
        return pd.Series(arr, index=df.index)
    raise ValueError(
        f"Expression produced a {type(value).__name__} that does not align with the frame "
        f"({len(df)} rows)."
    )


@dataclass(frozen=True)
class DerivedColumn:
    """
    A column computed from the analysis frame.

    Parameters
    ----------
    name : str
        Output column name. Must be a valid identifier — see :data:`IDENTIFIER_RE`.
    kind : {"transform", "expression", "bins"}
        ``transform`` applies :data:`TRANSFORMS`\\ ``[transform]`` to *source*;
        ``expression`` evaluates *expression* via :func:`evaluate_expression`;
        ``bins`` cuts *source* into an ordered categorical at *cut_points*, the
        ``tacsctot`` → ``tacsctot_group`` pattern.
    cut_points : tuple of float
        Interior cut points for ``kind="bins"``. The bins are bracketed by ±∞, so ``(0, 25, 100)``
        gives four groups: ``≤ 0``, ``(0, 25]``, ``(25, 100]``, ``> 100``.
    labels : tuple of str
        Explicit bin labels. When empty, ``label_prefix`` + index is used.
    """

    name: str
    kind: str = "transform"
    source: str = ""
    transform: str = ""
    expression: str = ""
    # ---- kind="bins" -----------------------------------------------------------
    cut_points: tuple[float, ...] = ()
    labels: tuple[str, ...] = ()
    label_prefix: str = "g"
    right: bool = True

    def n_bins(self) -> int:
        """Number of bins this definition produces (one more than the interior cut points)."""
        return len(self.cut_points) + 1

    def bin_labels(self) -> tuple[str, ...]:
        """Labels for each bin — the explicit ones when given, else ``prefix0``, ``prefix1``, …"""
        if self.labels:
            return tuple(self.labels)
        return default_bin_labels(self.n_bins(), self.label_prefix or "g")

    def label(self) -> str:
        """Human-readable definition, e.g. ``"cp_group = bins(lcp: 0, 25, 100)"``."""
        if self.kind == "transform":
            return f"{self.name} = {self.transform}({self.source})"
        if self.kind == "bins":
            points = ", ".join(f"{c:g}" for c in self.cut_points)
            return f"{self.name} = bins({self.source}: {points}) → {', '.join(self.bin_labels())}"
        return f"{self.name} = {self.expression}"

    def validate(self) -> str:
        """Return an error message describing why this column is unusable, or ``""`` if it is fine."""
        if not IDENTIFIER_RE.match(self.name or ""):
            return (
                f"{self.name!r} is not a valid column name — use letters, digits and underscores, "
                "starting with a letter or underscore."
            )
        if self.kind == "transform":
            if not self.source:
                return "No source column selected."
            if self.transform not in TRANSFORMS:
                return f"Unknown transform {self.transform!r}."
        elif self.kind == "expression":
            if not str(self.expression or "").strip():
                return "Expression is empty."
        elif self.kind == "bins":
            if not self.source:
                return "No source column selected."
            if not self.cut_points:
                return "Enter at least one cut point."
            if any(b <= a for a, b in zip(self.cut_points, self.cut_points[1:])):
                return "Cut points must be strictly increasing."
            labels = self.bin_labels()
            if len(labels) != self.n_bins():
                return (
                    f"{len(self.cut_points)} cut point(s) make {self.n_bins()} groups, "
                    f"but {len(labels)} label(s) were given."
                )
            if len(set(labels)) != len(labels):
                return "Bin labels must be unique."
        else:
            return f"Unknown derived-column kind {self.kind!r}."
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "transform": self.transform,
            "expression": self.expression,
            "cut_points": [float(c) for c in self.cut_points],
            "labels": list(self.labels),
            "label_prefix": self.label_prefix,
            "right": bool(self.right),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DerivedColumn":
        """Rebuild from :meth:`to_dict` output, tolerating missing keys."""
        return cls(
            name=str(data.get("name") or ""),
            kind=str(data.get("kind") or "transform"),
            source=str(data.get("source") or ""),
            transform=str(data.get("transform") or ""),
            expression=str(data.get("expression") or ""),
            cut_points=tuple(float(c) for c in (data.get("cut_points") or ())),
            labels=tuple(str(v) for v in (data.get("labels") or ())),
            label_prefix=str(data.get("label_prefix") or "g"),
            right=bool(data.get("right", True)),
        )


def apply_derived_columns(
    df: pd.DataFrame,
    columns: Sequence[DerivedColumn],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Append each derived column to a copy of *df*, in list order.

    Later entries may reference earlier ones, so ``log_pi`` then ``z_log_pi`` works. A column that
    fails (bad name, missing source, broken expression) is skipped and its error collected; the rest
    still land, so one typo does not cost the whole set.

    Returns
    -------
    (frame, errors)
    """
    if not columns:
        return df, []

    out = df.copy()
    errors: list[str] = []
    for spec in columns:
        problem = spec.validate()
        if problem:
            errors.append(f"{spec.name or '(unnamed)'}: {problem}")
            continue
        try:
            if spec.kind == "transform":
                if spec.source not in out.columns:
                    raise ValueError(f"source column {spec.source!r} is not in the frame")
                series = TRANSFORMS[spec.transform](out[spec.source])
            elif spec.kind == "bins":
                if spec.source not in out.columns:
                    raise ValueError(f"source column {spec.source!r} is not in the frame")
                # Categorical result: keep it as-is, numeric coercion would destroy the labels.
                out[spec.name] = cut_into_bins(
                    out[spec.source], spec.cut_points, spec.bin_labels(), right=spec.right
                )
                continue
            else:
                series = evaluate_expression(out, spec.expression)
        except Exception as exc:
            errors.append(f"{spec.name}: {exc}")
            log.debug("Derived column %r failed: %s", spec.name, exc)
            continue
        # Guard against a transform or expression that still produced infinities (e.g. ``a / b``
        # in a free-form expression): they would survive the fit's dropna and poison every estimate.
        out[spec.name] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out, errors


__all__ = [
    "DEFAULT_IQR_K",
    "FILTER_OPS",
    "IDENTIFIER_RE",
    "TRANSFORMS",
    "TRANSFORM_LABELS",
    "DerivedColumn",
    "FilterRule",
    "apply_derived_columns",
    "apply_filter_rules",
    "apply_iqr_filter",
    "apply_row_filter",
    "bin_counts",
    "bin_interval_labels",
    "cut_into_bins",
    "default_bin_labels",
    "default_bin_name",
    "default_derived_name",
    "evaluate_expression",
    "filtered_columns",
    "parse_cut_points",
    "suggest_cut_points",
]
