"""
How many observations a distribution plot is actually showing.

Description
-----------
``n = 126`` over a pooled violin is the number people read, and it is rarely the number they need.
Split by territory, the question is how many observations each territory contributed — a panel
drawn from eleven of them should not look like one drawn from ninety, and a level that quietly
lost half its rows to a filter should say so where the figure is read rather than in a log line.

Both distribution backends count through here, on the same rules they draw on:

* the column is coerced to numeric when it is numeric, so a text cell in a float column is
  missing rather than mysterious;
* rows with no value for the column are not counted, because nothing is drawn for them;
* rows an active filter removed are counted separately — greyed on the figure, and reported as
  such — or not at all when the figure is not drawing them.

Counting in one place is what keeps the Matplotlib figure and the Plotly one from disagreeing
about what they are showing.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


def ordered_levels(
    present: Sequence[str], order: Sequence[str] | None = None
) -> list[str]:
    """
    *present* arranged by *order*, with anything *order* does not name kept after it.

    A saved order outlives the frame it was chosen on — a filter can remove a level, a reload can
    add one — so it selects and ranks rather than dictating: levels it names appear first in its
    sequence, and a level that arrived since is appended rather than silently dropped.
    """
    if not order:
        return [str(level) for level in present]
    seen = [str(level) for level in present]
    known = set(seen)
    named = [str(level) for level in order if str(level) in known]
    taken = set(named)
    return named + [level for level in seen if level not in taken]


def level_strings(series: pd.Series) -> pd.Series:
    """
    *series* as the labels a plot groups by, with missing values left missing.

    Two things a bare ``astype(str)`` gets wrong on a numerically coded factor:

    * ``NaN`` becomes the string ``"nan"``, which then draws as a level of its own — a violin of
      the rows that have no value for the column;
    * a whole float prints as ``1.0``, so ``sex`` reads as ``0.0`` / ``1.0``. That spelling is
      the *common* case rather than an odd one: a single missing value upcasts an integer-coded
      factor to float64.

    Categoricals and strings are returned unchanged apart from the cast, so the levels a factor
    already has keep their spelling.
    """
    values = series.dropna()
    if (
        pd.api.types.is_float_dtype(series)
        and not isinstance(series.dtype, pd.CategoricalDtype)
        and not values.empty
        and bool(np.isfinite(values.to_numpy(dtype=float)).all())
        and bool((values.to_numpy(dtype=float) % 1 == 0).all())
    ):
        text = series.astype("Int64").astype(str)
    else:
        text = series.astype(str)
    # ``where`` keeps the label where there was a value and restores NA everywhere else.
    return text.where(series.notna())


@dataclass(frozen=True)
class GroupCount:
    """What one level of a grouping contributes to the figure."""

    #: The level's name, or ``""`` for an ungrouped whole.
    level: str
    #: Observations drawn and kept by the active filters.
    n: int
    #: Observations drawn greyed out, having been removed by a filter.
    excluded: int = 0

    @property
    def total(self) -> int:
        """Everything drawn for this level, kept or greyed."""
        return self.n + self.excluded

    def suffix(self) -> str:
        """``n=42``, or ``n=42 +3 excl`` when filtered rows are being drawn alongside."""
        return f"n={self.n}" if not self.excluded else f"n={self.n} +{self.excluded} excl"

    def label(self, *, separator: str = "\n") -> str:
        """The level's name with its count under it — an axis tick, or a trace name.

        A newline by default: on a categorical axis the count belongs under the label rather than
        doubling its width, which is what turns thirteen vessel names into an unreadable row.
        """
        return f"{self.level}{separator}({self.suffix()})" if self.level else f"({self.suffix()})"


def displayed_counts(
    frame: pd.DataFrame,
    column: str,
    *,
    group: str = "",
    excluded: Any = None,
    show_excluded: bool = True,
    levels: Sequence[Any] | None = None,
) -> list[GroupCount]:
    """
    How many observations of *column* each level of *group* puts on the figure.

    Returns one entry per level, in *levels* order when given — pass the order the plot drew in,
    so a level that had nothing to draw is not labelled as though it were there. Without *group*,
    returns a single ungrouped entry.

    Parameters
    ----------
    excluded : array-like of bool, optional
        Rows an active filter removed, aligned with *frame*.
    show_excluded : bool
        Whether the figure is drawing those rows greyed out. When ``False`` they are not counted
        at all, which is what the figure is doing with them.
    """
    if column not in frame.columns or frame.empty:
        return []

    values = frame[column]
    if pd.api.types.is_numeric_dtype(values) and not isinstance(values.dtype, pd.CategoricalDtype):
        values = pd.to_numeric(values, errors="coerce")
    present = values.notna()

    dropped = (
        pd.Series(np.asarray(excluded, dtype=bool), index=frame.index)
        if excluded is not None else pd.Series(False, index=frame.index)
    )
    if not show_excluded:
        present = present & ~dropped
        dropped = pd.Series(False, index=frame.index)

    def _at(level: str, rows: Any) -> GroupCount:
        """One level's kept / greyed split."""
        shown = present & rows
        greyed = int((shown & dropped).sum())
        return GroupCount(level=level, n=int(shown.sum()) - greyed, excluded=greyed)

    if not group or group not in frame.columns:
        return [_at("", pd.Series(True, index=frame.index))]

    # Through :func:`level_strings`, not a bare cast: the figure labels its levels that way, and
    # a status line reading "1.0" beside a tick reading "1" is two answers to one question.
    keys = level_strings(frame[group])
    order = (
        [str(level) for level in levels] if levels is not None
        else [str(v) for v in pd.unique(keys.dropna())]
    )
    return [_at(level, keys == level) for level in order]


def counts_note(
    counts: Sequence[GroupCount],
    *,
    total: GroupCount | None = None,
    group: str = "",
    limit: int = 14,
) -> str:
    """
    One line naming every level's N — the breakdown a crowded figure cannot spell out.

    Truncated past *limit* levels rather than run off the edge of the window; the figure's own
    tick labels still carry the rest.
    """
    if not counts and total is None:
        return ""
    head = f"n = {total.suffix().removeprefix('n=')}" if total is not None else ""
    if not group or (len(counts) == 1 and not counts[0].level):
        return head

    shown = [f"{c.level} {c.suffix()}" for c in counts[:limit]]
    if len(counts) > limit:
        shown.append(f"+{len(counts) - limit} more")
    body = f"{group}:  " + "  ·  ".join(shown)
    return f"{head}   |   {body}" if head else body


__all__ = [
    "GroupCount",
    "counts_note",
    "displayed_counts",
    "level_strings",
    "ordered_levels",
]
