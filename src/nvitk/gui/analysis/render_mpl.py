"""Draw a :class:`~nvitk.gui.analysis.plot_spec.PlotSpec` with seaborn on a Matplotlib figure."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from nvitk.gui.analysis.plot_spec import PlotSpec


def filtered(frame: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """*frame* through the spec's pandas query (``age > 50 and sex == 'F'``)."""
    if spec.query.strip():
        return frame.query(spec.query, engine="python")
    return frame


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.DataFrame:
    if column and column in frame and not pd.api.types.is_numeric_dtype(frame[column]):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().mean() > 0.8:
            frame = frame.assign(**{column: converted})
    return frame


#: Kinds whose x axis holds categories (one box / violin / bar per value).
CATEGORY_X_KINDS = frozenset({"bar", "box", "violin", "strip"})
#: A numeric column with more distinct values than this is continuous: used as a category it is
#: grouped into quantile bins instead of giving one box / violin / facet per value.
MAX_LEVELS = 12
#: Number of quantile bins a continuous column is grouped into.
GROUP_BINS = 5


def numeric_like(series: pd.Series) -> pd.Series | None:
    """*series* as numbers when it is text holding numbers (``"12.5"``), else ``None``."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    present = series.dropna()
    if present.empty:
        return None
    converted = pd.to_numeric(series, errors="coerce")
    if converted[series.notna()].notna().mean() >= 0.9:
        return converted
    return None


def is_continuous(series: pd.Series) -> bool:
    """Numeric with more than :data:`MAX_LEVELS` distinct values."""
    return (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and series.nunique(dropna=True) > MAX_LEVELS)


def as_groups(data: pd.DataFrame, column: str, bins: int = GROUP_BINS) -> tuple[pd.DataFrame, str]:
    """*column* grouped into *bins* quantile ranges (ordered categories), and a note saying so."""
    values = data[column]
    try:
        grouped = pd.qcut(values, q=bins, duplicates="drop")
    except ValueError:
        grouped = pd.cut(values, bins=bins)
    labels = {iv: f"{iv.left:.3g}–{iv.right:.3g}" for iv in grouped.cat.categories}
    grouped = grouped.cat.rename_categories(labels)
    return data.assign(**{column: grouped}), f"{column} is continuous: grouped into {len(labels)} quantile bins"


def render(spec: PlotSpec, frame: pd.DataFrame, figure: Any | None = None) -> Any:
    """A figure showing *spec* drawn from *frame* (on *figure* when given, cleared first)."""
    import matplotlib

    matplotlib.use("QtAgg", force=False)
    import seaborn as sns
    from matplotlib.figure import Figure

    problems = spec.problems(list(frame.columns))
    if problems:
        raise ValueError("; ".join(problems))
    data = filtered(frame, spec)
    for column in (spec.x, spec.y):
        data = _numeric(data, column)
    notes: list[str] = []
    # Columns used as categories (the x of a box / violin / bar, the facet; the colour where
    # the plot draws one group per colour) must hold a few levels, not every distinct number.
    as_category = [spec.col]
    if spec.kind in CATEGORY_X_KINDS:
        as_category += [spec.x, spec.hue]
    for column in dict.fromkeys(c for c in as_category if c and c in data):
        if is_continuous(data[column]):
            data, note = as_groups(data, column)
            notes.append(note)
    fig = figure if figure is not None else Figure(figsize=(7, 5), constrained_layout=True)
    fig.clear()
    fig.nvitk_notes = notes
    if spec.kind == "pair":
        numeric = data.select_dtypes("number").columns[:6]
        axes = fig.subplots(len(numeric), len(numeric), squeeze=False)
        for i, a in enumerate(numeric):
            for j, b in enumerate(numeric):
                ax = axes[i][j]
                if i == j:
                    sns.histplot(data=data, x=a, hue=spec.hue, ax=ax, palette=spec.palette if spec.hue else None, legend=False)
                else:
                    sns.scatterplot(data=data, x=b, y=a, hue=spec.hue, ax=ax, palette=spec.palette if spec.hue else None, legend=False, s=12)
                ax.set_xlabel(b if i == len(numeric) - 1 else "")
                ax.set_ylabel(a if j == 0 else "")
        _finish(fig, spec)
        return fig
    facets = [None]
    if spec.col and spec.col in data:
        column = data[spec.col]
        if isinstance(column.dtype, pd.CategoricalDtype):
            facets = [c for c in column.cat.categories if (column == c).any()][:12]
        else:
            facets = sorted(pd.unique(column.dropna()), key=lambda v: (str(type(v)), v))[:12]
    n = len(facets)
    cols = min(n, 3)
    rows = int(math.ceil(n / cols))
    axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False).ravel())
    for ax, value in zip(axes, facets):
        part = data if value is None else data[data[spec.col] == value]
        _draw(sns, ax, spec, part)
        if value is not None:
            ax.set_title(f"{spec.col} = {value}")
        if spec.logx:
            ax.set_xscale("log")
        if spec.logy:
            ax.set_yscale("log")
    for ax in axes[len(facets):]:
        ax.set_visible(False)
    _finish(fig, spec)
    return fig


def _draw(sns: Any, ax: Any, spec: PlotSpec, data: pd.DataFrame) -> None:
    hue = spec.hue if spec.hue else None
    palette = spec.palette if hue else None
    kind = spec.kind
    if kind == "scatter":
        sns.scatterplot(data=data, x=spec.x, y=spec.y, hue=hue, palette=palette, ax=ax, s=24)
    elif kind == "regression":
        if hue:
            for i, (level, part) in enumerate(data.groupby(hue)):
                sns.regplot(data=part, x=spec.x, y=spec.y, ax=ax, label=str(level), scatter_kws={"s": 16},
                            color=sns.color_palette(spec.palette)[i % 10])
            ax.legend(title=hue)
        else:
            sns.regplot(data=data, x=spec.x, y=spec.y, ax=ax, scatter_kws={"s": 16})
    elif kind == "line":
        sns.lineplot(data=data, x=spec.x, y=spec.y, hue=hue, palette=palette, ax=ax, estimator=spec.agg if spec.agg != "count" else "count")
    elif kind == "bar":
        if spec.agg == "count":
            sns.countplot(data=data, x=spec.x, hue=hue, palette=palette, ax=ax)
        else:
            sns.barplot(data=data, x=spec.x, y=spec.y, hue=hue, palette=palette, ax=ax, estimator=spec.agg)
    elif kind in ("box", "violin", "strip"):
        # Coloured by category even without a colour column: the x column is the hue then
        # (seaborn >= 0.13 wants that spelled out rather than a bare palette).
        colour = {"hue": hue} if hue else {"hue": spec.x, "legend": False}
        extra = {"box": {}, "violin": {"cut": 0}, "strip": {"dodge": bool(hue), "size": 4}}[kind]
        fn = {"box": sns.boxplot, "violin": sns.violinplot, "strip": sns.stripplot}[kind]
        fn(data=data, x=spec.x, y=spec.y, palette=palette or spec.palette, ax=ax, orient="x", **colour, **extra)
    elif kind == "hist":
        sns.histplot(data=data, x=spec.x, hue=hue, palette=palette, ax=ax, bins=int(spec.bins))
    elif kind == "kde":
        sns.kdeplot(data=data, x=spec.x, hue=hue, palette=palette, ax=ax, fill=True, common_norm=False)
    elif kind == "ecdf":
        sns.ecdfplot(data=data, x=spec.x, hue=hue, palette=palette, ax=ax)
    elif kind == "heatmap":
        if spec.x and spec.y and spec.hue:
            table = data.pivot_table(index=spec.y, columns=spec.x, values=spec.hue, aggfunc=spec.agg)
        else:
            table = data.select_dtypes("number").corr()
        sns.heatmap(table, ax=ax, cmap="coolwarm" if table.min().min() < 0 else "viridis", annot=table.size <= 100, fmt=".2g")
    if spec.x and kind in ("bar", "box", "violin", "strip"):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")


def _finish(fig: Any, spec: PlotSpec) -> None:
    from nvitk.gui.core.design import style_figure

    if spec.title:
        fig.suptitle(spec.title)
    style_figure(fig)


__all__ = ["CATEGORY_X_KINDS", "MAX_LEVELS", "as_groups", "filtered", "is_continuous", "numeric_like", "render"]
