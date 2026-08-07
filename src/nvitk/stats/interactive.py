"""
Interactive Plotly renderings of the model plots, with the numbers on hover.

Description
-----------
The matplotlib versions in :mod:`~nvitk.stats.mixedlm` and friends draw the same geometry, but
reading a value off them means squinting at an axis. These carry the numbers in the hover box,
which is what the plots are actually consulted for:

=====================  ==========================================================================
Element                Hover shows
=====================  ==========================================================================
observation            subject id, group level, x and y, plus any extra identifier column
model line             the group, its slope and intercept, and the value under the cursor
fixed-effect line      the population slope and intercept
confidence band        the interval's bounds at that x
marginal mean (EMM)    the estimate, its standard error and the interval
observed cell mean     the unadjusted mean and how many observations went into it
=====================  ==========================================================================

Design
------
These functions take **already-computed** geometry rather than a fitted model: a frame of points, a
frame of lines, a frame of marginal means. The statistics stay where they are — this module only
renders — which is what keeps one plotting path for every engine instead of one per backend.

:func:`model_plot` is the entry point the GUI uses; :func:`column_plot` covers the distribution
views (violin, box, strip, histogram, ECDF) that hang off the dataframe's column menu.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Qualitative palette, matching the matplotlib ``tab10`` the static plots use so a figure looks
#: the same whichever backend drew it.
PALETTE: tuple[str, ...] = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)

#: Colour for observations excluded by a QC filter — present but visibly set aside.
GREYED = "#B0B0B0"

#: Distribution views available from a column's context menu.
COLUMN_PLOT_KINDS: dict[str, str] = {
    "violin": "Violin — the full distribution's shape",
    "violin_points": "Violin + points — the shape with every observation on top",
    "box": "Box — quartiles and outliers",
    "box_points": "Box + points",
    "strip": "Points only — one dot per observation, jittered",
    "histogram": "Histogram — binned counts",
    "density": "Density (KDE) — a smoothed probability density",
    "ecdf": "ECDF — the empirical cumulative distribution",
}



#: How a distribution is broken up. *Split* overlays the levels on one axes with a colour each;
#: *panels* gives each level its own axes, which is what a per-territory violin figure needs when
#: the levels differ in scale enough that overlaying them flattens the small ones.
COLUMN_FACET_MODES: dict[str, str] = {
    "": "All at once — one distribution over every row",
    "split": "Split — one coloured series per level, on shared axes",
    "panels": "Panels — one plot per level, each autoscaled",
    "anatomical": "Anatomical panels — grouped by vascular territory, one plot each",
}


def column_panel_figure(
    frame: pd.DataFrame,
    column: str,
    *,
    facet_by: str,
    kind: str = "violin",
    group: str = "",
    hover_columns: Sequence[str] = (),
    excluded_mask: Any = None,
    show_excluded: bool = True,
    anatomical: bool = False,
    title: str = "",
    n_cols: int = 2,
) -> Any:
    """
    One distribution panel per level of *facet_by*, laid out as a grid.

    This is the shape a per-territory hemodynamics figure wants: overlaying a sagittal sinus around
    −1.0 with a carotid around −0.2 on shared axes flattens both, and the per-panel autoscale is the
    whole point. Within a panel, *group* still splits into coloured series, so you can facet by
    territory **and** split by sex at the same time.

    Parameters
    ----------
    anatomical : bool
        Group the levels of *facet_by* into vascular panels (carotids / anterior / posterior /
        venous) instead of giving each level its own, using
        :func:`~nvitk.stats.region_groups.resolve_panels`.

    Raises
    ------
    ValueError
        When *facet_by* is absent, or nothing survives in any panel.
    """
    from .region_groups import resolve_panels

    if facet_by not in frame.columns:
        raise ValueError(f"Cannot facet by {facet_by!r}: it is not in the frame.")

    levels = [str(v) for v in pd.unique(frame[facet_by].dropna().astype(str))]
    if anatomical:
        groups = resolve_panels(levels, column=facet_by)
    else:
        from .region_groups import natural_level_key

        groups = {level: [level] for level in sorted(levels, key=natural_level_key)}

    mask = (
        pd.Series(np.asarray(excluded_mask, dtype=bool), index=frame.index)
        if excluded_mask is not None else None
    )

    panels: dict[str, Any] = {}
    for name, members in groups.items():
        subset = frame.loc[frame[facet_by].astype(str).isin({str(m) for m in members})]
        if subset.empty:
            continue
        panels[name] = column_plot(
            subset, column, kind=kind, group=group, hover_columns=hover_columns,
            excluded_mask=(mask.loc[subset.index].to_numpy() if mask is not None else None),
            show_excluded=show_excluded, title="",
        )
    if not panels:
        raise ValueError(f"No rows in any {facet_by!r} panel.")

    from .interactive import panel_grid_figure  # local: same module, keeps the import graph flat

    figure = panel_grid_figure(
        panels, title=title or f"{column} by {facet_by}", n_cols=n_cols, height_per_row=380
    )
    figure.update_layout(violinmode="group", boxmode="group", barmode="overlay")
    return figure

def _lighten(colour: str, amount: float = 0.55) -> str:
    """Mix *colour* toward white, for the unadjusted-observation series."""
    colour = colour.lstrip("#")
    rgb = [int(colour[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [int(round(c + (255 - c) * amount)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def palette_for(levels: Sequence[Any]) -> dict[str, str]:
    """Stable colour per level, cycling the palette."""
    return {str(level): PALETTE[i % len(PALETTE)] for i, level in enumerate(levels)}


def _customdata(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Column block for a trace's ``customdata``, as strings so mixed dtypes survive."""
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    return np.column_stack([frame[c].astype(str).to_numpy() for c in columns])


# ---------------------------------------------------------------------------
# Model plots
# ---------------------------------------------------------------------------
def model_plot(
    *,
    points: pd.DataFrame | None = None,
    lines: pd.DataFrame | None = None,
    bands: pd.DataFrame | None = None,
    marginal_means: pd.DataFrame | None = None,
    x: str,
    y: str,
    group: str = "",
    mode: str = "continuous",
    categorical_order: Sequence[str] | None = None,
    hover_columns: Sequence[str] = (),
    excluded_mask: Any = None,
    show_excluded: bool = True,
    errorbar: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = 620,
) -> Any:
    """
    One interactive model panel.

    Parameters
    ----------
    points : pandas.DataFrame, optional
        The observations: *x*, *y*, and *group* when there is one. Every column named in
        *hover_columns* is attached to the hover box, which is how ``subject_uid`` gets there.
    lines : pandas.DataFrame, optional
        Fitted curves, long: ``group``, *x*, *y*, and optionally ``slope`` / ``intercept``, which
        are shown in the hover so a line's parameters can be read without the coefficient table.
        A group of ``""`` is drawn as the dashed population line.
    bands : pandas.DataFrame, optional
        Confidence ribbons: ``group``, *x*, ``lower``, ``upper``.
    marginal_means : pandas.DataFrame, optional
        Categorical estimates: ``group``, *x*, ``estimate``, and optionally ``se`` / ``lower`` /
        ``upper``. Drawn as markers with error bars.
    excluded_mask : array-like of bool, optional
        Rows of *points* that a filter excluded. With *show_excluded* they are drawn in grey and
        kept out of the legend; without it they are dropped. Greying rather than dropping is what
        lets you see whether a filter removed a coherent cluster or scattered noise.
    errorbar : bool
        Draw the confidence intervals. The marginal-means frame carries ``lower``/``upper`` whether
        or not they were asked for — they fall out of the same delta-method computation — so this
        has to gate the *drawing*, not the data.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    figure = go.Figure()
    levels = _levels_of(points, lines, marginal_means, group)
    colours = palette_for(levels)
    hover_columns = [c for c in hover_columns if points is not None and c in points.columns]

    x_positions: dict[str, int] | None = None
    if mode == "categorical":
        order = list(categorical_order) if categorical_order is not None else _natural_levels(
            points, marginal_means, x
        )
        x_positions = {str(v): i for i, v in enumerate(order)}

    if points is not None and not points.empty:
        _add_points(
            figure, points, x=x, y=y, group=group, colours=colours,
            hover_columns=hover_columns, excluded_mask=excluded_mask,
            show_excluded=show_excluded, mode=mode, x_positions=x_positions,
        )
    if errorbar and bands is not None and not bands.empty:
        _add_bands(figure, bands, x=x, group=group, colours=colours, x_positions=x_positions)
    if lines is not None and not lines.empty:
        _add_lines(figure, lines, x=x, y=y, group=group, colours=colours)
    if marginal_means is not None and not marginal_means.empty:
        _add_marginal_means(
            figure, marginal_means, x=x, group=group, colours=colours,
            x_positions=x_positions, errorbar=errorbar,
        )

    layout: dict[str, Any] = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {"title": x_label or x, "showgrid": True, "gridcolor": "#E6E6E6", "zeroline": False},
        "yaxis": {"title": y_label or y, "showgrid": True, "gridcolor": "#E6E6E6", "zeroline": False},
        "hovermode": "closest",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "height": height,
        "margin": {"l": 70, "r": 30, "t": 60, "b": 60},
        "legend": {"bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#CCCCCC", "borderwidth": 1},
    }
    if x_positions is not None:
        layout["xaxis"].update({
            "tickmode": "array",
            "tickvals": list(x_positions.values()),
            "ticktext": list(x_positions),
        })
    figure.update_layout(**layout)
    return figure


def _levels_of(
    points: pd.DataFrame | None,
    lines: pd.DataFrame | None,
    means: pd.DataFrame | None,
    group: str,
) -> list[str]:
    """Group levels across every supplied frame, in first-seen order."""
    if not group:
        return []
    seen: list[str] = []
    for frame in (lines, means, points):
        if frame is None or frame.empty or group not in frame.columns:
            continue
        for value in frame[group].astype(str):
            if value and value not in seen:
                seen.append(value)
    return seen


def _natural_levels(
    points: pd.DataFrame | None, means: pd.DataFrame | None, x: str
) -> list[str]:
    """Categorical x order, digit-aware so ``g2`` precedes ``g10``."""
    from .region_groups import natural_level_key

    values: set[str] = set()
    for frame in (means, points):
        if frame is not None and not frame.empty and x in frame.columns:
            values |= {str(v) for v in frame[x].dropna()}
    return sorted(values, key=natural_level_key)


def _add_points(
    figure: Any,
    points: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    colours: Mapping[str, str],
    hover_columns: Sequence[str],
    excluded_mask: Any,
    show_excluded: bool,
    mode: str,
    x_positions: Mapping[str, int] | None,
) -> None:
    """Observations, split into kept and excluded so a filter is visible rather than silent."""
    import plotly.graph_objects as go

    frame = points.copy()
    if excluded_mask is not None:
        excluded = pd.Series(np.asarray(excluded_mask, dtype=bool), index=frame.index)
    else:
        excluded = pd.Series(False, index=frame.index)
    if not show_excluded:
        frame = frame.loc[~excluded]
        excluded = excluded.loc[frame.index]

    extra = list(hover_columns)
    hover_lines = [f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(extra)]

    def positioned(sub: pd.DataFrame) -> Any:
        """x values, mapped to tick positions with jitter in categorical mode."""
        if x_positions is None:
            return sub[x]
        base = sub[x].astype(str).map(x_positions).astype(float)
        # Deterministic jitter: a seeded generator keyed on length, so the cloud does not
        # reshuffle on every redraw and points stay where the user last saw them.
        rng = np.random.default_rng(len(sub))
        return base + rng.uniform(-0.16, 0.16, len(sub))

    if excluded.any() and show_excluded:
        sub = frame.loc[excluded]
        figure.add_trace(go.Scattergl(
            x=positioned(sub), y=sub[y], mode="markers", name="excluded by filter",
            marker={"size": 6, "color": GREYED, "opacity": 0.45,
                    "line": {"width": 0}},
            customdata=_customdata(sub, extra),
            hovertemplate="<b>excluded by filter</b><br>"
                          + f"{x}: %{{x}}<br>{y}: %{{y:.4g}}<br>"
                          + "<br>".join(hover_lines) + "<extra></extra>",
            legendgroup="excluded", showlegend=True,
        ))

    kept = frame.loc[~excluded]
    if kept.empty:
        return
    if group and group in kept.columns:
        for level, sub in kept.groupby(kept[group].astype(str), sort=False):
            colour = colours.get(str(level), PALETTE[0])
            figure.add_trace(go.Scattergl(
                x=positioned(sub), y=sub[y], mode="markers", name=str(level),
                marker={"size": 7, "color": _lighten(colour, 0.35), "opacity": 0.65,
                        "line": {"width": 0.5, "color": colour}},
                customdata=_customdata(sub, extra),
                hovertemplate=f"<b>{group}</b>: {level}<br>{x}: %{{x}}<br>{y}: %{{y:.4g}}<br>"
                              + "<br>".join(hover_lines) + "<extra></extra>",
                legendgroup=str(level), showlegend=False,
            ))
    else:
        figure.add_trace(go.Scattergl(
            x=positioned(kept), y=kept[y], mode="markers", name="observations",
            marker={"size": 7, "color": _lighten(PALETTE[0], 0.3), "opacity": 0.65},
            customdata=_customdata(kept, extra),
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y:.4g}}<br>"
                          + "<br>".join(hover_lines) + "<extra></extra>",
        ))


def _add_lines(
    figure: Any,
    lines: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    colours: Mapping[str, str],
) -> None:
    """Fitted curves; a blank group is the dashed population line."""
    import plotly.graph_objects as go

    key = group if group and group in lines.columns else None
    grouped = lines.groupby(lines[key].astype(str), sort=False) if key else [("", lines)]

    for level, sub in grouped:
        sub = sub.sort_values(x)
        population = not str(level).strip()
        colour = "#111111" if population else colours.get(str(level), PALETTE[0])
        name = "Population (fixed effects)" if population else str(level)

        detail = ""
        for column, label in (("slope", "slope"), ("intercept", "intercept")):
            if column in sub.columns and pd.notna(sub[column].iloc[0]):
                detail += f"<br><b>{label}</b>: {float(sub[column].iloc[0]):.4g}"
        figure.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="lines", name=name,
            line={"color": colour, "width": 3.2 if population else 2.4,
                  "dash": "dash" if population else "solid"},
            hovertemplate=f"<b>{name}</b><br>{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}{detail}<extra></extra>",
            legendgroup=str(level) or "population",
        ))


def _add_bands(
    figure: Any,
    bands: pd.DataFrame,
    *,
    x: str,
    group: str,
    colours: Mapping[str, str],
    x_positions: Mapping[str, int] | None,
) -> None:
    """Confidence ribbons, drawn under the lines and excluded from the legend."""
    import plotly.graph_objects as go

    key = group if group and group in bands.columns else None
    grouped = bands.groupby(bands[key].astype(str), sort=False) if key else [("", bands)]

    for level, sub in grouped:
        sub = sub.sort_values(x)
        population = not str(level).strip()
        colour = "#111111" if population else colours.get(str(level), PALETTE[0])
        rgb = tuple(int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)"
        xs = sub[x] if x_positions is None else sub[x].astype(str).map(x_positions)
        # One closed polygon rather than two traces: a single fill has one hover surface, so the
        # cursor reports the interval instead of whichever edge it happened to land on.
        figure.add_trace(go.Scatter(
            x=list(xs) + list(xs)[::-1],
            y=list(sub["upper"]) + list(sub["lower"])[::-1],
            fill="toself", fillcolor=fill, line={"width": 0},
            name=f"CI {level}" if level else "CI", showlegend=False, hoverinfo="skip",
            legendgroup=str(level) or "population",
        ))


def _add_marginal_means(
    figure: Any,
    means: pd.DataFrame,
    *,
    x: str,
    group: str,
    colours: Mapping[str, str],
    x_positions: Mapping[str, int] | None,
    errorbar: bool = True,
) -> None:
    """Estimated marginal means with their intervals — the estimate, SE and bounds on hover."""
    import plotly.graph_objects as go

    key = group if group and group in means.columns else None
    grouped = means.groupby(means[key].astype(str), sort=False) if key else [("", means)]

    for level, sub in grouped:
        xs = sub[x].astype(str).map(x_positions) if x_positions is not None else sub[x]
        sub = sub.assign(_x=xs).sort_values("_x")
        colour = colours.get(str(level), PALETTE[0]) if level else "#111111"

        error: dict[str, Any] | None = None
        if errorbar and {"lower", "upper"} <= set(sub.columns):
            error = {
                "type": "data", "symmetric": False,
                "array": (sub["upper"] - sub["estimate"]).to_numpy(),
                "arrayminus": (sub["estimate"] - sub["lower"]).to_numpy(),
                "color": colour, "thickness": 1.5, "width": 6,
            }

        detail = ""
        block: list[str] = []
        for column, label in (("se", "SE"), ("lower", "CI low"), ("upper", "CI high")):
            if column in sub.columns:
                block.append(column)
                detail += f"<br><b>{label}</b>: %{{customdata[{len(block) - 1}]:.4g}}"
        figure.add_trace(go.Scatter(
            x=sub["_x"], y=sub["estimate"], mode="lines+markers",
            name=str(level) if level else "Marginal mean",
            line={"color": colour, "width": 2.6},
            marker={"size": 9, "color": colour},
            error_y=error,
            customdata=sub[block].to_numpy() if block else None,
            hovertemplate=(f"<b>{level or 'Marginal mean'}</b><br>{x}: %{{text}}<br>"
                           f"estimate: %{{y:.4g}}{detail}<extra></extra>"),
            text=sub[x].astype(str),
            legendgroup=str(level) or "emm",
        ))


def panel_grid_figure(
    panels: Mapping[str, Any],
    *,
    title: str = "",
    n_cols: int = 2,
    height_per_row: int = 420,
) -> Any:
    """
    Combine per-panel figures into one subplot grid, preserving each panel's traces.

    The anatomical grouped display builds one figure per panel and hands them here, so the panel
    logic stays in :mod:`~nvitk.stats.region_groups` and this only does layout. Legends are shown
    for the first panel only — every panel repeats the same series, and eight copies of one legend
    is most of the figure.
    """
    from plotly.subplots import make_subplots

    names = list(panels)
    if not names:
        raise ValueError("No panels to lay out.")
    n_cols = max(1, min(int(n_cols), len(names)))
    n_rows = -(-len(names) // n_cols)

    grid = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=names,
        horizontal_spacing=0.09, vertical_spacing=max(0.06, 0.30 / n_rows),
    )
    for index, name in enumerate(names):
        row, col = index // n_cols + 1, index % n_cols + 1
        source = panels[name]
        for trace in source.data:
            trace.showlegend = bool(getattr(trace, "showlegend", False)) and index == 0
            grid.add_trace(trace, row=row, col=col)
        # Carry each panel's own axis titles and any categorical ticks across.
        grid.update_xaxes(source.layout.xaxis, row=row, col=col)
        grid.update_yaxes(source.layout.yaxis, row=row, col=col)

    grid.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=height_per_row * n_rows + 120,
        plot_bgcolor="white", paper_bgcolor="white", hovermode="closest",
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    return grid


# ---------------------------------------------------------------------------
# Column distribution plots
# ---------------------------------------------------------------------------
def column_plot(
    frame: pd.DataFrame,
    column: str,
    *,
    kind: str = "violin",
    group: str = "",
    hover_columns: Sequence[str] = (),
    excluded_mask: Any = None,
    show_excluded: bool = True,
    title: str = "",
    height: int = 560,
) -> Any:
    """
    Distribution of one column — violin, box, points, histogram, density or ECDF.

    Splitting by *group* puts one violin (or box, or histogram trace) per level, which is usually
    the point: a flow distribution pooled over every territory is bimodal for anatomical reasons
    and tells you nothing.

    Categorical columns are shown as counts whatever *kind* asks for, since a violin of labels is
    not defined — the request is honoured in spirit rather than refused.

    Raises
    ------
    ValueError
        When *column* is absent, *kind* is unknown, or nothing numeric survives.
    """
    import plotly.graph_objects as go

    if column not in frame.columns:
        raise ValueError(f"Column {column!r} is not in the frame.")
    if kind not in COLUMN_PLOT_KINDS:
        raise ValueError(f"Unknown plot kind {kind!r}. Available: {', '.join(COLUMN_PLOT_KINDS)}.")

    work = frame.copy()
    if excluded_mask is not None:
        excluded = pd.Series(np.asarray(excluded_mask, dtype=bool), index=work.index)
    else:
        excluded = pd.Series(False, index=work.index)
    if not show_excluded:
        work = work.loc[~excluded]
        excluded = excluded.loc[work.index]

    categorical = isinstance(work[column].dtype, pd.CategoricalDtype) or not (
        pd.api.types.is_numeric_dtype(work[column])
    )
    if categorical:
        return _category_counts(work, column, group=group, title=title, height=height)

    values = pd.to_numeric(work[column], errors="coerce")
    work = work.assign(**{column: values}).dropna(subset=[column])
    excluded = excluded.reindex(work.index).fillna(False)
    if work.empty:
        raise ValueError(f"{column!r} has no numeric values to plot.")

    levels = (
        [str(v) for v in pd.unique(work[group].dropna().astype(str))]
        if group and group in work.columns else [""]
    )
    colours = palette_for(levels)
    extra = [c for c in hover_columns if c in work.columns]
    figure = go.Figure()

    for level in levels:
        sub = work if not level else work.loc[work[group].astype(str) == level]
        if sub.empty:
            continue
        sub_excluded = excluded.reindex(sub.index).fillna(False)
        colour = colours.get(level, PALETTE[0])
        _add_distribution(
            figure, sub, column, kind=kind, name=level or column, colour=colour,
            hover_columns=extra, excluded=sub_excluded, show_excluded=show_excluded,
        )

    summary = _summary_text(work[column], excluded)
    figure.update_layout(
        title={"text": title or f"{column} — {COLUMN_PLOT_KINDS[kind].split('—')[0].strip()}",
               "x": 0.5, "xanchor": "center"},
        yaxis={"title": column if kind not in {"histogram", "density", "ecdf"} else
               ("count" if kind == "histogram" else "density" if kind == "density" else "cumulative"),
               "gridcolor": "#E6E6E6"},
        xaxis={"title": (group or "") if kind not in {"histogram", "density", "ecdf"} else column,
               "gridcolor": "#E6E6E6"},
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
        violinmode="group", boxmode="group", barmode="overlay",
        annotations=[{
            "text": summary, "showarrow": False, "xref": "paper", "yref": "paper",
            "x": 0.0, "y": 1.06, "xanchor": "left", "font": {"size": 11, "color": "#666666"},
        }],
    )
    return figure


def _add_distribution(
    figure: Any,
    sub: pd.DataFrame,
    column: str,
    *,
    kind: str,
    name: str,
    colour: str,
    hover_columns: Sequence[str],
    excluded: pd.Series,
    show_excluded: bool,
) -> None:
    """One level's trace(s), whichever distribution view was asked for."""
    import plotly.graph_objects as go

    values = sub[column]
    custom = _customdata(sub, hover_columns)
    hover_lines = [f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(hover_columns)]
    point_hover = f"{column}: %{{y:.4g}}<br>" + "<br>".join(hover_lines) + "<extra></extra>"

    if kind in {"violin", "violin_points"}:
        figure.add_trace(go.Violin(
            y=values, name=name, line_color=colour, fillcolor=_lighten(colour, 0.65),
            box_visible=True, meanline_visible=True, opacity=0.85,
            points="all" if kind == "violin_points" else False,
            pointpos=0, jitter=0.35, marker={"size": 5, "opacity": 0.55, "color": colour},
            customdata=custom if kind == "violin_points" else None,
            hovertemplate=point_hover if kind == "violin_points" else None,
        ))
    elif kind in {"box", "box_points"}:
        figure.add_trace(go.Box(
            y=values, name=name, line_color=colour, fillcolor=_lighten(colour, 0.65),
            boxpoints="all" if kind == "box_points" else "outliers",
            jitter=0.35, pointpos=0, marker={"size": 5, "opacity": 0.55, "color": colour},
            customdata=custom if kind == "box_points" else None,
            hovertemplate=point_hover if kind == "box_points" else None,
        ))
    elif kind == "strip":
        # An invisible violin rather than a scatter. A violin places itself on the *categorical*
        # axis keyed by its name and jitters its own points; a scatter would need numeric x, and
        # mixing numeric x with the categorical violins of the other kinds makes Plotly treat every
        # jitter value as its own category — which is what turned the x axis into a wall of floats.
        figure.add_trace(go.Violin(
            y=values, name=name, points="all", pointpos=0, jitter=0.5,
            box_visible=False, meanline_visible=False, width=0.9,
            line_color="rgba(0,0,0,0)", fillcolor="rgba(0,0,0,0)",
            marker={"size": 7, "color": colour, "opacity": 0.6},
            customdata=custom, hovertemplate=point_hover,
        ))
    elif kind == "histogram":
        figure.add_trace(go.Histogram(
            x=values, name=name, marker_color=colour, opacity=0.6,
            hovertemplate=f"{column}: %{{x}}<br>count: %{{y}}<extra></extra>",
        ))
    elif kind == "density":
        grid, density = _kde(values)
        if grid is not None:
            figure.add_trace(go.Scatter(
                x=grid, y=density, mode="lines", name=name, fill="tozeroy",
                line={"color": colour, "width": 2.2},
                fillcolor=_lighten(colour, 0.75),
                hovertemplate=f"{column}: %{{x:.4g}}<br>density: %{{y:.4g}}<extra></extra>",
            ))
    else:  # ecdf
        ordered = np.sort(values.to_numpy(dtype=float))
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        figure.add_trace(go.Scatter(
            x=ordered, y=cumulative, mode="lines", name=name,
            line={"color": colour, "width": 2.2, "shape": "hv"},
            hovertemplate=f"{column}: %{{x:.4g}}<br>≤ this value: %{{y:.1%}}<extra></extra>",
        ))

    if show_excluded and bool(excluded.any()) and kind in {"violin_points", "box_points", "strip"}:
        dropped = sub.loc[excluded]
        # Same trick as the strip: the greyed points must land on the same categorical slot as the
        # series they were excluded from, not on a numeric axis of their own.
        figure.add_trace(go.Violin(
            y=dropped[column], name=name, points="all", pointpos=0, jitter=0.5,
            box_visible=False, meanline_visible=False, width=0.9,
            line_color="rgba(0,0,0,0)", fillcolor="rgba(0,0,0,0)",
            marker={"size": 6, "color": GREYED, "opacity": 0.55},
            customdata=_customdata(dropped, hover_columns),
            hovertemplate="<b>excluded by filter</b><br>" + point_hover,
            showlegend=False, scalegroup=f"{name}-excluded",
        ))


def _kde(values: pd.Series, *, points: int = 200) -> tuple[Any, Any]:
    """Gaussian KDE over the observed range, or ``(None, None)`` when it cannot be estimated."""
    data = values.to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 3 or float(np.std(data)) == 0.0:
        return None, None
    try:
        from scipy.stats import gaussian_kde

        kernel = gaussian_kde(data)
    except Exception as exc:
        log.debug("KDE unavailable (%s); falling back to a histogram-shaped density.", exc)
        return None, None
    grid = np.linspace(data.min(), data.max(), points)
    return grid, kernel(grid)


def _summary_text(values: pd.Series, excluded: pd.Series) -> str:
    """The one-line descriptive summary shown above a distribution plot."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    parts = [
        f"n = {len(numeric)}",
        f"mean {numeric.mean():.4g}",
        f"SD {numeric.std():.4g}",
        f"median {numeric.median():.4g}",
        f"IQR {numeric.quantile(0.25):.4g}–{numeric.quantile(0.75):.4g}",
    ]
    n_excluded = int(excluded.sum()) if excluded is not None else 0
    if n_excluded:
        parts.append(f"{n_excluded} excluded by filter")
    return "   ·   ".join(parts)


def _category_counts(
    frame: pd.DataFrame, column: str, *, group: str, title: str, height: int
) -> Any:
    """Counts per level, for a column a distribution plot is not defined on."""
    import plotly.graph_objects as go

    figure = go.Figure()
    if group and group in frame.columns:
        levels = [str(v) for v in pd.unique(frame[group].dropna().astype(str))]
        colours = palette_for(levels)
        for level in levels:
            counts = frame.loc[frame[group].astype(str) == level, column].astype(str).value_counts()
            figure.add_trace(go.Bar(
                x=counts.index.tolist(), y=counts.to_numpy(), name=level,
                marker_color=colours[level], opacity=0.85,
                hovertemplate=f"<b>{level}</b><br>{column}: %{{x}}<br>count: %{{y}}<extra></extra>",
            ))
    else:
        counts = frame[column].astype(str).value_counts()
        figure.add_trace(go.Bar(
            x=counts.index.tolist(), y=counts.to_numpy(), name=column,
            marker_color=PALETTE[0], opacity=0.85,
            hovertemplate=f"{column}: %{{x}}<br>count: %{{y}}<extra></extra>",
        ))
    figure.update_layout(
        title={"text": title or f"{column} — counts", "x": 0.5, "xanchor": "center"},
        xaxis={"title": column, "gridcolor": "#E6E6E6"},
        yaxis={"title": "count", "gridcolor": "#E6E6E6"},
        barmode="group", plot_bgcolor="white", paper_bgcolor="white", height=height,
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
    )
    return figure


# ---------------------------------------------------------------------------
# Forests, matrices and networks
# ---------------------------------------------------------------------------
def forest_plot(
    frame: pd.DataFrame,
    *,
    label: str,
    estimate: str = "coef",
    lower: str = "ci_low",
    upper: str = "ci_high",
    p_value: str = "p_value",
    max_rows: int = 40,
    reference: float = 0.0,
    title: str = "",
    x_label: str = "Estimate",
    height_per_row: int = 26,
) -> Any:
    """
    Interactive forest — one row per estimate, with its interval and the exact numbers on hover.

    Used for path coefficients, mediation paths and the MRF field alike: they differ in what the
    estimate *means*, not in how it is read. Rows whose interval excludes *reference* are drawn in
    the accent colour, the rest in grey, so significance is visible without reading every interval.
    """
    import plotly.graph_objects as go

    if frame is None or frame.empty:
        raise ValueError("Nothing to plot.")
    work = frame.copy()
    for column in (estimate, lower, upper, p_value):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[estimate])
    if len(work) > max_rows:
        # Keep the largest magnitudes: a forest of 500 rows is unreadable, and the ones nearest
        # zero are the ones a reader would skip anyway.
        work = work.reindex(work[estimate].abs().sort_values(ascending=False).index).head(max_rows)
    work = work.iloc[::-1]

    has_ci = lower in work.columns and upper in work.columns and work[lower].notna().any()
    if has_ci:
        excludes = (work[lower] > reference) | (work[upper] < reference)
    elif p_value in work.columns:
        excludes = work[p_value] < 0.05
    else:
        excludes = pd.Series(True, index=work.index)

    detail_columns = [c for c in (lower, upper, p_value) if c in work.columns]
    detail = "".join(
        f"<br><b>{c}</b>: %{{customdata[{i}]:.4g}}" for i, c in enumerate(detail_columns)
    )
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=work[estimate], y=work[label].astype(str), mode="markers",
        marker={"size": 9, "color": np.where(excludes, PALETTE[0], "#999999")},
        error_x=(
            {"type": "data", "symmetric": False,
             "array": (work[upper] - work[estimate]).to_numpy(),
             "arrayminus": (work[estimate] - work[lower]).to_numpy(),
             "color": "#777777", "thickness": 1.5, "width": 5}
            if has_ci else None
        ),
        customdata=work[detail_columns].to_numpy() if detail_columns else None,
        hovertemplate=f"<b>%{{y}}</b><br>{x_label}: %{{x:.4g}}{detail}<extra></extra>",
        showlegend=False,
    ))
    figure.add_vline(x=reference, line={"color": "#333333", "dash": "dash", "width": 1.2})
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis={"title": x_label, "gridcolor": "#E6E6E6", "zeroline": False},
        yaxis={"automargin": True, "gridcolor": "#F2F2F2"},
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(360, height_per_row * len(work) + 140),
        margin={"l": 40, "r": 30, "t": 70, "b": 60},
    )
    return figure


def matrix_plot(
    matrix: pd.DataFrame,
    *,
    title: str = "",
    colorscale: str = "RdBu_r",
    zmid: float | None = 0.0,
    value_label: str = "value",
) -> Any:
    """
    Interactive heatmap of a square matrix — the MMRM correlation between levels, say.

    Hover names both levels and the exact value, which is the thing a static heatmap makes you
    estimate from a colour bar. Long level names get automatic margins rather than being clipped.
    """
    import plotly.graph_objects as go

    if matrix is None or matrix.empty:
        raise ValueError("Nothing to plot.")
    labels = [str(i) for i in matrix.index]
    figure = go.Figure(go.Heatmap(
        z=matrix.to_numpy(dtype=float),
        x=[str(c) for c in matrix.columns], y=labels,
        colorscale=colorscale, zmid=zmid,
        hovertemplate=f"%{{y}} ↔ %{{x}}<br>{value_label}: %{{z:.4g}}<extra></extra>",
        colorbar={"title": value_label},
    ))
    # A square matrix should look square, and long labels need room rather than truncation.
    side = max(520, 34 * len(labels) + 220)
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=side, width=None,
        xaxis={"automargin": True, "tickangle": -45, "constrain": "domain"},
        yaxis={"automargin": True, "autorange": "reversed", "scaleanchor": "x"},
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    return figure


def network_plot(
    edges: pd.DataFrame,
    *,
    source: str = "rhs",
    target: str = "lhs",
    weight: str = "coef",
    lower: str = "ci_low",
    upper: str = "ci_high",
    node_labels: Mapping[str, str] | None = None,
    title: str = "",
) -> Any:
    """
    The fitted network as a layered diagram, edges weighted by their coefficient.

    Nodes are laid out by depth, so upstream vessels sit left of the ones they feed. Edge width is
    the coefficient's magnitude and colour its sign; an edge whose interval covers zero is drawn
    faint. Hovering an edge gives the exact coefficient rather than leaving it to the line width.
    """
    import plotly.graph_objects as go

    if edges is None or edges.empty:
        raise ValueError("No edges to draw.")
    work = edges.loc[edges[source].astype(str) != edges[target].astype(str)].copy()
    work[weight] = pd.to_numeric(work[weight], errors="coerce")
    work = work.dropna(subset=[weight])
    if work.empty:
        raise ValueError("No edges to draw.")

    nodes = sorted(set(work[source].astype(str)) | set(work[target].astype(str)))
    incoming = {n: set(work.loc[work[target].astype(str) == n, source].astype(str)) for n in nodes}

    depth: dict[str, int] = {}

    def depth_of(node: str, seen: frozenset[str] = frozenset()) -> int:
        """Longest path back to a node with no predecessors."""
        if node in depth:
            return depth[node]
        if node in seen or not incoming.get(node):
            return 0
        value = 1 + max(depth_of(p, seen | {node}) for p in incoming[node])
        depth[node] = value
        return value

    for node in nodes:
        depth[node] = depth_of(node)
    layers: dict[int, list[str]] = {}
    for node in nodes:
        layers.setdefault(depth[node], []).append(node)
    positions = {
        node: (float(level), float(i - (len(members) - 1) / 2))
        for level, members in layers.items()
        for i, node in enumerate(sorted(members))
    }

    scale = float(work[weight].abs().max() or 1.0)
    figure = go.Figure()
    for _, row in work.iterrows():
        a, b = str(row[source]), str(row[target])
        if a not in positions or b not in positions:
            continue
        coefficient = float(row[weight])
        low = pd.to_numeric(pd.Series([row.get(lower)]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([row.get(upper)]), errors="coerce").iloc[0]
        faint = bool(pd.notna(low) and pd.notna(high) and low <= 0 <= high)
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        figure.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line={"width": 1.0 + 5.0 * abs(coefficient) / scale,
                  "color": "#C44E52" if coefficient < 0 else "#4C72B0",
                  "dash": "dot" if faint else "solid"},
            opacity=0.35 if faint else 0.85,
            hovertemplate=f"<b>{a} → {b}</b><br>{weight}: {coefficient:.4g}<extra></extra>",
            showlegend=False,
        ))

    labels = dict(node_labels or {})
    figure.add_trace(go.Scatter(
        x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
        mode="markers+text", text=[labels.get(n, n) for n in nodes],
        textposition="middle center",
        marker={"size": 42, "color": "white", "line": {"width": 1.6, "color": "#555555"}},
        hovertemplate="<b>%{text}</b><extra></extra>", showlegend=False,
    ))
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis={"visible": False}, yaxis={"visible": False},
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(480, 130 * max(len(m) for m in layers.values()) + 160),
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
    )
    return figure


__all__ = [
    "COLUMN_FACET_MODES",
    "COLUMN_PLOT_KINDS",
    "GREYED",
    "PALETTE",
    "column_panel_figure",
    "column_plot",
    "forest_plot",
    "matrix_plot",
    "network_plot",
    "model_plot",
    "palette_for",
    "panel_grid_figure",
]
