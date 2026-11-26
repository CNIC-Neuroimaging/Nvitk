"""Column filters on a DataFrame, as data: what the GUI's header filters build and recipes replay.

A filter is ``{"kind": "values" | "exclude" | "range" | "text", ...}`` per column name:

* ``values`` / ``exclude``: keep (drop) the rows whose value, as shown, is one of ``values``
  (missing values show as ``(empty)``, floats with 6 significant digits);
* ``range``: ``lo`` <= value <= ``hi`` (either may be ``None``);
* ``text``: the value contains ``text`` (case-insensitive).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EMPTY = "(empty)"


def _key_series(series: pd.Series) -> pd.Series:
    """Values as the strings the checklist shows (``(empty)`` for missing)."""
    if pd.api.types.is_float_dtype(series):
        text = series.map(lambda v: EMPTY if pd.isna(v) else f"{v:.6g}")
    else:
        text = series.astype("string").fillna(EMPTY).map(lambda v: EMPTY if str(v).strip() == "" else str(v))
    return text.astype(str)


def _numeric(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(series):
        return None
    return None


def filter_mask(frame: pd.DataFrame, filters: dict[str, dict[str, Any]], *, skip: str | None = None) -> np.ndarray:
    """Rows of *frame* every filter (but *skip*'s) keeps."""
    mask = np.ones(len(frame), dtype=bool)
    for column, spec in filters.items():
        if column == skip or column not in frame.columns:
            continue
        series = frame[column]
        kind = spec.get("kind")
        if kind == "values":
            keep = set(spec.get("values") or ())
            mask &= _key_series(series).isin(keep).to_numpy()
        elif kind == "exclude":
            drop = set(spec.get("values") or ())
            mask &= ~_key_series(series).isin(drop).to_numpy()
        elif kind == "range":
            numeric = pd.to_numeric(series, errors="coerce")
            ok = numeric.notna()
            if spec.get("lo") is not None:
                ok &= numeric >= float(spec["lo"])
            if spec.get("hi") is not None:
                ok &= numeric <= float(spec["hi"])
            mask &= ok.fillna(False).to_numpy()
        elif kind == "text":
            needle = str(spec.get("text") or "")
            if needle:
                mask &= series.astype("string").str.contains(needle, case=False, regex=False).fillna(False).to_numpy()
    return mask


def describe_filter(column: str, spec: dict[str, Any]) -> str:
    """``territory ∈ {ACA, MCA}``-style summary of one filter."""
    kind = spec.get("kind")
    if kind in ("values", "exclude"):
        values = sorted(spec.get("values") or ())
        shown = ", ".join(values[:3]) + (f" +{len(values) - 3}" if len(values) > 3 else "")
        return f"{column} {'∈' if kind == 'values' else '∉'} {{{shown}}}"
    if kind == "range":
        lo, hi = spec.get("lo"), spec.get("hi")
        if lo is not None and hi is not None:
            return f"{lo:g} ≤ {column} ≤ {hi:g}"
        return f"{column} ≥ {lo:g}" if lo is not None else f"{column} ≤ {hi:g}"
    if kind == "text":
        return f"{column} contains “{spec.get('text')}”"
    return column


__all__ = ["EMPTY", "describe_filter", "filter_mask"]
