"""
Static (Matplotlib) distribution plots, mirroring the interactive ones.

Description
-----------
:mod:`~nvitk.stats.interactive` renders distributions with Plotly, which is what you want when the
question is "which subject is that outlier". This module draws the same eight views with Matplotlib,
which is what you want when the answer goes into a paper: vector output, a familiar style, and no
browser engine between the figure and the page.

The two share an API — same *kind* keys, same grouping, same excluded-mask handling — so the GUI
toggles between them without knowing which is which.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

from .group_counts import displayed_counts, level_strings

log = Logger()

#: Grey for observations a filter excluded — present but visibly set aside.
GREYED = "#B0B0B0"



def _fit(figure: Any) -> None:
    """``tight_layout``, unless the figure already has a layout engine that would fight it."""
    try:
        if figure.get_layout_engine() is not None:
            return
    except AttributeError:  # matplotlib < 3.6
        pass
    figure.tight_layout()


def _palette(levels: Sequence[str]) -> dict[str, Any]:
    """One colour per level, from the same qualitative palette the interactive plots use."""
    import seaborn as sns

    colours = sns.color_palette("tab10", n_colors=max(len(levels), 3))
    return {str(level): colours[i % len(colours)] for i, level in enumerate(levels)}


def column_plot_static(
    frame: pd.DataFrame,
    column: str,
    *,
    kind: str = "violin",
    group: str = "",
    excluded_mask: Any = None,
    show_excluded: bool = True,
    title: str = "",
    ax: Any = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> Any:
    """
    Distribution of one column, drawn with Matplotlib.

    Same *kind* vocabulary as :func:`~nvitk.stats.interactive.column_plot`: ``violin``,
    ``violin_points``, ``box``, ``box_points``, ``strip``, ``histogram``, ``density``, ``ecdf``. A
    categorical column falls back to counts, since a violin of labels is not defined.

    Raises
    ------
    ValueError
        When *column* is absent or holds nothing plottable.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if column not in frame.columns:
        raise ValueError(f"Column {column!r} is not in the frame.")

    work = frame.copy()
    excluded = (
        pd.Series(np.asarray(excluded_mask, dtype=bool), index=work.index)
        if excluded_mask is not None else pd.Series(False, index=work.index)
    )
    if not show_excluded:
        work = work.loc[~excluded]
        excluded = excluded.loc[work.index]

    figure, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
    categorical = isinstance(work[column].dtype, pd.CategoricalDtype) or not (
        pd.api.types.is_numeric_dtype(work[column])
    )

    if categorical:
        _counts(ax, work, column, group=group, excluded=excluded, show_excluded=show_excluded)
        ax.set_title(title or f"{column} — counts", pad=22)
        _fit(figure)
        figure.linked_axes = [ax]
        return figure

    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    excluded = excluded.reindex(work.index).fillna(False)
    if work.empty:
        raise ValueError(f"{column!r} has no numeric values to plot.")

    grouped = bool(group) and group in work.columns
    if grouped:
        # The level order, the palette keys and the counts are all built as strings, but seaborn
        # keys its palette on the column's own values — so a numerically coded factor raised
        # "The palette dictionary is missing keys: {0.0, 1.0}". Casting once here is what keeps
        # every one of them talking about the same levels.
        work[group] = level_strings(work[group])
    order = (
        [str(v) for v in pd.unique(work[group].dropna().astype(str))] if grouped else [column]
    )
    colours = _palette(order)
    hue = group if grouped else None
    # Counted from ``work``, which is already coerced, dropped and filtered — so these are the
    # observations the axes below will actually carry, not the rows the frame started with.
    counted = {
        c.level: c
        for c in displayed_counts(
            work, column, group=group if grouped else "", excluded=excluded, levels=order,
        )
    } if grouped else {}

    if kind in {"violin", "violin_points"}:
        sns.violinplot(
            data=work, x=hue, y=column, hue=hue, order=order if grouped else None,
            palette=colours if grouped else None, legend=False, ax=ax,
            inner="box", cut=0, density_norm="width",
        )
    elif kind in {"box", "box_points"}:
        sns.boxplot(
            data=work, x=hue, y=column, hue=hue, order=order if grouped else None,
            palette=colours if grouped else None, legend=False, ax=ax,
            showfliers=kind == "box",
        )
    elif kind == "strip":
        sns.stripplot(
            data=work, x=hue, y=column, hue=hue, order=order if grouped else None,
            palette=colours if grouped else None, legend=False, ax=ax,
            jitter=0.28, alpha=0.6, size=4,
        )
    elif kind == "histogram":
        sns.histplot(
            data=work, x=column, hue=hue, palette=colours if grouped else None,
            ax=ax, element="step", alpha=0.45, common_norm=False,
        )
    elif kind == "density":
        sns.kdeplot(
            data=work, x=column, hue=hue, palette=colours if grouped else None,
            ax=ax, fill=True, alpha=0.35, common_norm=False, warn_singular=False,
        )
    elif kind == "ecdf":
        sns.ecdfplot(
            data=work, x=column, hue=hue, palette=colours if grouped else None, ax=ax,
        )
    else:
        raise ValueError(f"Unknown plot kind {kind!r}.")

    # Points on top of a violin or box, and the greyed-out exclusions beside them.
    if kind in {"violin_points", "box_points"}:
        sns.stripplot(
            data=work.loc[~excluded], x=hue, y=column, order=order if grouped else None,
            color="#333333", alpha=0.35, size=3, jitter=0.25, ax=ax, legend=False,
        )
    if show_excluded and bool(excluded.any()) and kind in {"violin_points", "box_points", "strip"}:
        sns.stripplot(
            data=work.loc[excluded], x=hue, y=column, order=order if grouped else None,
            color=GREYED, alpha=0.75, size=4, jitter=0.25, ax=ax, legend=False,
        )

    # Padded: the descriptive line below is annotated just above the axes, and a default-placed
    # title lands on top of it.
    ax.set_title(title or column, pad=22)
    if kind in {"histogram", "density", "ecdf"}:
        ax.set_xlabel(column)
    else:
        ax.set_xlabel(group if grouped else "")
        ax.set_ylabel(column)
    ax.grid(True, axis="y", alpha=0.25)

    if grouped:
        # On the categorical kinds the level *is* an x tick, so the count goes under it; on the
        # overlaid ones the level is only a legend entry, so it goes there. Either way it is on
        # the figure rather than in a tooltip, and exports with it.
        if kind in {"histogram", "density", "ecdf"}:
            legend = ax.get_legend()
            for text in legend.get_texts() if legend is not None else []:
                size = counted.get(text.get_text())
                if size is not None:
                    text.set_text(size.label(separator=" "))
        else:
            ax.set_xticks(list(range(len(order))))
            ax.set_xticklabels(
                [counted[level].label() if level in counted else level for level in order]
            )

    if grouped and len(order) > 6:
        # Long vessel names overlap badly once there are more than a handful.
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    summary = _summary_text(work[column], excluded)
    if summary:
        ax.annotate(
            summary, xy=(0.0, 1.02), xycoords="axes fraction", fontsize=8, color="#666666",
        )
    _fit(figure)
    figure.linked_axes = [ax]
    return figure


def column_panels_static(
    frame: pd.DataFrame,
    column: str,
    *,
    facet_by: str,
    kind: str = "violin",
    group: str = "",
    excluded_mask: Any = None,
    show_excluded: bool = True,
    anatomical: bool = False,
    title: str = "",
    n_cols: int = 2,
) -> Any:
    """
    One static distribution panel per level of *facet_by*, each autoscaled to its own range.

    Mirrors :func:`~nvitk.stats.interactive.column_panel_figure`, including the anatomical grouping.
    """
    from .region_groups import natural_level_key, panel_grid, resolve_panels

    if facet_by not in frame.columns:
        raise ValueError(f"Cannot facet by {facet_by!r}: it is not in the frame.")

    levels = [str(v) for v in pd.unique(frame[facet_by].dropna().astype(str))]
    groups = (
        resolve_panels(levels, column=facet_by) if anatomical
        else {level: [level] for level in sorted(levels, key=natural_level_key)}
    )
    mask = (
        pd.Series(np.asarray(excluded_mask, dtype=bool), index=frame.index)
        if excluded_mask is not None else None
    )

    panels = {
        name: frame.loc[frame[facet_by].astype(str).isin({str(m) for m in members})]
        for name, members in groups.items()
    }
    panels = {name: sub for name, sub in panels.items() if not sub.empty}
    if not panels:
        raise ValueError(f"No rows in any {facet_by!r} panel.")

    figure, axes = panel_grid(
        len(panels), n_cols=n_cols, panel_size=(6.5, 4.2),
        title=title or f"{column} by {facet_by}",
    )
    drawn: list[Any] = []
    for ax, (name, sub) in zip(axes, panels.items()):
        panel_mask = mask.loc[sub.index].to_numpy() if mask is not None else None
        # Each panel autoscales to its own range, so one drawn from eleven observations looks as
        # solid as one drawn from ninety until its heading says which it is.
        size = displayed_counts(sub, column, excluded=panel_mask, show_excluded=show_excluded)
        column_plot_static(
            sub, column, kind=kind, group=group, excluded_mask=panel_mask,
            show_excluded=show_excluded, ax=ax,
            title=f"{name}  ({size[0].suffix()})" if size else name,
        )
        drawn.append(ax)
    figure.linked_axes = drawn
    return figure


def _counts(
    ax: Any,
    frame: pd.DataFrame,
    column: str,
    *,
    group: str,
    excluded: Any = None,
    show_excluded: bool = True,
) -> None:
    """Bar counts per level, for a column a distribution is not defined on."""
    import seaborn as sns

    hue = group if group and group in frame.columns else None
    sns.countplot(
        data=frame, x=column, hue=hue, ax=ax,
        palette=_palette(sorted({str(v) for v in frame[hue].dropna()})) if hue else None,
        legend=bool(hue),
    )
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    if hue:
        # The bars are counts of *column*; the legend says how many rows each series has in total,
        # which is the denominator those bars have to be read against.
        sizes = {
            c.level: c
            for c in displayed_counts(
                frame, column, group=hue, excluded=excluded, show_excluded=show_excluded
            )
        }
        legend = ax.get_legend()
        for text in legend.get_texts() if legend is not None else []:
            size = sizes.get(text.get_text())
            if size is not None:
                text.set_text(size.label(separator=" "))


def _summary_text(values: pd.Series, excluded: pd.Series | None) -> str:
    """
    The one-line descriptive summary shown above a distribution plot.

    Reported over the rows a filter *kept*, so that ``n`` here is the sum of the per-level ``n``
    on the axis below it, and so that the mean and SD are not the ones the filtering was there to
    get rid of. Rows drawn greyed alongside are counted separately, in the same wording the level
    labels use.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    dropped = (
        excluded.reindex(numeric.index).fillna(False) if excluded is not None
        else pd.Series(False, index=numeric.index)
    )
    kept = numeric.loc[~dropped]
    if kept.empty:
        kept = numeric
    parts = [
        f"n = {len(kept)}" + (f" +{int(dropped.sum())} excl" if bool(dropped.any()) else ""),
        f"mean {kept.mean():.4g}",
        f"SD {kept.std():.4g}",
        f"median {kept.median():.4g}",
    ]
    return "   ·   ".join(parts)


__all__ = ["GREYED", "column_panels_static", "column_plot_static"]
