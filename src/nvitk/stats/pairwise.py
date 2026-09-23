"""
Pairwise contrasts between the levels of a factor, for the significance brackets on a plot.

Description
-----------
"Is this territory different from that one" is the question a grouped plot is actually read for,
and it is not the question a coefficient table answers: those are contrasts against whichever
level happened to sort first, so ``LICA`` vs ``RICA`` is nowhere in them. This module computes
every level-versus-level comparison instead.

How
---
A pairwise contrast is a linear combination of the fitted coefficients. Build a prediction grid
with one row per level, holding the covariates at their reference values; the difference between
two rows of the resulting design matrix is the contrast vector ``c``, and then

.. code-block:: text

    estimate = c' β            se = sqrt(c' V c)            t = estimate / se

which needs nothing from the model but its coefficients and their covariance — so one
implementation serves MixedLM, OLS and GLM alike. The R engines do not expose a patsy design
matrix, so they go through ``emmeans::pairs`` instead, which computes the same quantity in R.

Multiplicity
------------
Thirteen vessels are seventy-eight comparisons, and at α = 0.05 four of them are expected to be
"significant" with no effect present at all. Holm's step-down correction is applied across every
comparison computed — not only the ones that fit on the figure — and both backends are corrected
the same way in Python rather than each engine's own default, so a bracket means the same thing
whichever engine drew it. Holm rather than Tukey because it needs no distributional assumption
about the set, and it is valid for the unbalanced, covariate-adjusted comparisons these are.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Columns every backend returns, in order.
CONTRAST_COLUMNS: tuple[str, ...] = (
    "by", "a", "b", "estimate", "se", "statistic", "df", "p_value", "p_adj", "stars",
)


@dataclass(frozen=True)
class Contrast:
    """One level-versus-level comparison, optionally within one level of a second factor."""

    by: str
    a: str
    b: str
    estimate: float
    se: float
    p_adj: float
    stars: str

    def label(self) -> str:
        """What goes over the bracket."""
        return self.stars or "NA"


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """
    Holm step-down adjusted p-values, NaNs passed through.

    Uniformly more powerful than Bonferroni and valid under any dependence, which is what these
    comparisons have: every contrast shares the same coefficient vector.
    """
    raw = np.asarray(list(pvalues), dtype=float)
    out = np.full(raw.shape, np.nan)
    finite = np.isfinite(raw)
    if not finite.any():
        return out

    values = raw[finite]
    order = np.argsort(values)
    n = values.size
    # Step down, keeping the sequence monotone: an adjusted p may never fall below one that
    # preceded it, or a later comparison could be "more significant" than a smaller raw p.
    running = 0.0
    adjusted = np.empty(n)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(1.0, running)
    out[finite] = adjusted
    return out


def _stars(pvalues: Sequence[float]) -> list[str]:
    """Conventional markers, with ``NA`` where there is no p-value to mark."""
    from .mixedlm import significance_stars

    return [
        significance_stars(float(p)) if np.isfinite(p) else "NA" for p in pvalues
    ]


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble, Holm-adjust and star a backend's raw comparisons."""
    if not rows:
        return pd.DataFrame(columns=list(CONTRAST_COLUMNS))
    out = pd.DataFrame(rows)
    if "by" not in out.columns:
        out["by"] = ""
    out["by"] = out["by"].astype(str)
    out["p_adj"] = holm(out["p_value"])
    out["stars"] = _stars(out["p_adj"])
    for column in CONTRAST_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out.loc[:, list(CONTRAST_COLUMNS)].sort_values(
        "p_adj", na_position="last", kind="stable"
    ).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# statsmodels: MixedLM / OLS / GLM
# ──────────────────────────────────────────────────────────────────────────────
def _statsmodels_contrasts(
    result: Any,
    df: pd.DataFrame,
    *,
    factor: str,
    levels: Sequence[str],
    covariate_refs: Mapping[str, Any],
    by: str = "",
    by_levels: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Every pair of *levels*, as differences of design-matrix rows at *covariate_refs*.

    With *by*, the grid is the factorial ``factor x by`` and the pairs are formed **within** each
    ``by`` level — the simple effect there rather than an effect averaged over it. On a model with
    an interaction those are different numbers, and the one a per-series bracket is claiming is
    the simple effect.
    """
    import patsy

    from .mixedlm import _match_grid_columns_to_df_dtypes, model_params

    # Every column the design info evaluates has to be in the grid, not only the ones the caller
    # named: patsy raises a bare NameError for a missing one, which surfaces three frames up as
    # "no contrasts available" rather than as the omission it is. The caller's references win;
    # anything it left out is held at its own mean or modal level, the same rule the marginal
    # means use, so the comparison is adjusted rather than evaluated at an arbitrary row.
    from .interactive_adapters import _reference_row

    held = [factor] + ([by] if by else [])
    refs = {**_reference_row(df, exclude=held), **dict(covariate_refs)}
    cells = (
        [(level, group) for group in by_levels for level in levels] if by
        else [(level, "") for level in levels]
    )
    grid = pd.DataFrame({factor: [str(level) for level, _group in cells]})
    if by:
        grid[by] = [str(group) for _level, group in cells]
    for name, value in refs.items():
        if name not in set(held):
            grid[name] = value
    grid = _match_grid_columns_to_df_dtypes(
        grid, df, [c for c in grid.columns if c in df.columns]
    )
    design = patsy.build_design_matrices(
        [result.model.data.design_info], grid, return_type="dataframe"
    )[0]

    fixed = model_params(result)
    # MixedLM's cov_params also covers the variance components, which are not part of a contrast
    # over fixed effects; the same subsetting the marginal-means grid does.
    covariance = result.cov_params().loc[fixed.index, fixed.index]
    design = design.reindex(columns=fixed.index, fill_value=0.0)

    beta = fixed.to_numpy(dtype=float)
    vcov = covariance.to_numpy(dtype=float)
    rows_design = design.to_numpy(dtype=float)
    dof = float(getattr(result, "df_resid", np.nan) or np.nan)

    rows: list[dict[str, Any]] = []
    groups = [str(g) for g in by_levels] if by else [""]
    width = len(levels)
    for block, group in enumerate(groups):
        offset = block * width
        for i in range(width):
            for j in range(i + 1, width):
                contrast = rows_design[offset + i] - rows_design[offset + j]
                estimate = float(contrast @ beta)
                variance = float(contrast @ vcov @ contrast)
                se = float(np.sqrt(variance)) if variance > 0 else np.nan
                statistic = estimate / se if np.isfinite(se) and se > 0 else np.nan
                rows.append({
                    "by": group, "a": str(levels[i]), "b": str(levels[j]),
                    "estimate": estimate, "se": se, "statistic": statistic,
                    "df": dof, "p_value": _two_sided(statistic, dof),
                })
    return _frame(rows)


def _two_sided(statistic: float, dof: float) -> float:
    """Two-sided p for *statistic*, from t when a residual df is known and z otherwise."""
    if not np.isfinite(statistic):
        return float("nan")
    from scipy import stats

    if np.isfinite(dof) and dof > 0:
        return float(2.0 * stats.t.sf(abs(statistic), dof))
    return float(2.0 * stats.norm.sf(abs(statistic)))


# ──────────────────────────────────────────────────────────────────────────────
# R engines: lme4, lmrob, MMRM
# ──────────────────────────────────────────────────────────────────────────────
#: ``contrast`` rather than ``pairs``: ``pairs`` is an S3 method registered on ``emmGrid``, not an
#: exported object, so ``emmeans::pairs`` raises "not an exported object from namespace:emmeans".
#: ``contrast(em, "pairwise")`` is the exported spelling of the same thing.
#:
#: No adjustment of its own — Holm is applied in Python so a bracket means the same thing
#: whichever engine produced it.
_R_PAIRS_HELPER = """
.nvitk_pairs <- function(model, factor_name, by_name) {
  suppressMessages(try(emmeans::emm_options(lmer.df = "satterthwaite"), silent = TRUE))
  spec <- stats::as.formula(
    if (nzchar(by_name)) paste("~", factor_name, "|", by_name) else paste("~", factor_name)
  )
  em <- emmeans::emmeans(model, specs = spec)
  out <- as.data.frame(summary(emmeans::contrast(em, method = "pairwise", adjust = "none")))
  # The by column comes back under its own name; rename it so one parser reads every shape.
  if (nzchar(by_name) && by_name %in% names(out)) {
    names(out)[match(by_name, names(out))] <- ".by"
  }
  out
}
"""

_PAIRS_HELPER_LOADED = False


def _r_contrasts(
    model: Any, *, factor: str, levels: Sequence[str], by: str = ""
) -> pd.DataFrame:
    """Every pair of *levels* from ``emmeans``, restricted to the levels being drawn.

    With *by*, the comparisons are made within each of its levels rather than averaged over it.
    """
    global _PAIRS_HELPER_LOADED
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects import r as R_
    from rpy2.robjects.conversion import localconverter

    if not _PAIRS_HELPER_LOADED:
        R_(_R_PAIRS_HELPER)
        _PAIRS_HELPER_LOADED = True

    try:
        with localconverter(default_converter + pandas2ri.converter):
            table = pd.DataFrame(globalenv[".nvitk_pairs"](model, str(factor), str(by)))
    except Exception as exc:
        # emmeans has no ``emm_basis`` for every model this toolkit fits — ``lmrob`` is the one
        # that matters here — and says so by raising. The contrasts are still computable from the
        # fit's own coefficients and covariance, which is what this falls back to.
        log.debug("emmeans declined %s; using the design-matrix basis.", type(model), exc_info=True)
        log.info("emmeans could not contrast this model (%s); using its design matrix instead.", exc)
        from .r_basis import linear_pairs

        table = linear_pairs(model, factor, by=by)

    # emmeans names the comparison column "contrast" and spells a pair "LICA - RICA"; levels
    # holding a "-" are re-split on the separator emmeans actually uses, " - ".
    wanted = {str(level) for level in levels}
    rows: list[dict[str, Any]] = []
    for _index, row in table.iterrows():
        parts = str(row.get("contrast", "")).split(" - ")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a not in wanted or b not in wanted:
            continue
        statistic = _first(row, ("t.ratio", "z.ratio", "statistic"))
        rows.append({
            "by": str(row.get(".by", "") if pd.notna(row.get(".by", "")) else ""),
            "a": a, "b": b,
            "estimate": _first(row, ("estimate",)),
            "se": _first(row, ("SE", "std.error")),
            "statistic": statistic,
            "df": _first(row, ("df",)),
            "p_value": _first(row, ("p.value", "p_value")),
        })
    return _frame(rows)


def _first(row: Any, names: Sequence[str]) -> float:
    """The first of *names* present in *row*, as a float, or NaN."""
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def pairwise_contrasts(
    result: Any,
    df: pd.DataFrame,
    *,
    factor: str,
    levels: Sequence[str] | None = None,
    covariate_refs: Mapping[str, Any] | None = None,
    by: str = "",
    by_levels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Every level-versus-level comparison of *factor*, Holm-adjusted and starred.

    Parameters
    ----------
    factor : str
        The column whose levels are being compared — the plot's categorical axis.
    levels : sequence of str, optional
        Restrict to these levels, in this order. Defaults to every level present in *df*.
        Passing the levels actually drawn matters: the correction is over the comparisons
        computed, so hiding half the vessels should not leave the rest carrying their penalty.
    covariate_refs : mapping, optional
        Where to hold the other covariates. Ignored by the R path, which uses ``emmeans``' own
        reference grid.
    by : str, optional
        Compare *factor*'s levels **within** each level of this column instead of averaging over
        it. On a model with ``factor * by`` in it those are different questions: a plot drawing
        one series per ``by`` level is showing the simple effects, and brackets over a series
        have to be the comparison inside that series.
    by_levels : sequence of str, optional
        Restrict *by* to these levels. Defaults to every level present in *df*.

    Returns
    -------
    pandas.DataFrame
        ``by, a, b, estimate, se, statistic, df, p_value, p_adj, stars``, most significant first.
        ``by`` is ``""`` when none was asked for. Empty when the model exposes no way to compute
        them.

    Raises
    ------
    ValueError
        When *factor* is absent, or has fewer than two levels to compare.
    """
    if factor not in df.columns:
        raise ValueError(f"{factor!r} is not in the fitted frame.")
    present = [str(v) for v in pd.unique(df[factor].dropna().astype(str))]
    order = [str(level) for level in levels] if levels is not None else sorted(present)
    order = [level for level in order if level in set(present)]
    if len(order) < 2:
        raise ValueError(f"{factor!r} has fewer than two levels to compare.")

    groups: list[str] = []
    if by:
        if by not in df.columns:
            raise ValueError(f"{by!r} is not in the fitted frame.")
        seen = [str(v) for v in pd.unique(df[by].dropna().astype(str))]
        groups = (
            [str(level) for level in by_levels if str(level) in set(seen)]
            if by_levels is not None else sorted(seen)
        )
        if not groups:
            raise ValueError(f"{by!r} has no levels to compare within.")

    if hasattr(result, "cov_params") and hasattr(getattr(result, "model", None), "data"):
        return _statsmodels_contrasts(
            result, df, factor=factor, levels=order,
            covariate_refs=covariate_refs or {}, by=by, by_levels=groups,
        )
    out = _r_contrasts(_r_object(result), factor=factor, levels=order, by=by)
    if by and groups:
        out = out.loc[out["by"].isin(set(groups))].reset_index(drop=True)
    return out


def _r_object(result: Any) -> Any:
    """The underlying R model, whichever wrapper this engine hands back.

    An ``lmrob`` or ``mmrm`` fit *is* the R object. A pymer4 ``lmer`` holds one, and where it
    holds it moved between versions — the same two names the rest of this package reads, in the
    same order. Guessing ``.model`` instead got the pymer4 object itself handed to rpy2, which
    fails with "Conversion 'py2rpy' not defined for objects of type ... lmer".
    """
    for name in ("r_model", "model_obj"):
        inner = getattr(result, name, None)
        if inner is not None:
            return inner
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Brackets
# ──────────────────────────────────────────────────────────────────────────────
#: Show every comparison, or only the ones that survived the correction.
MODE_ALL = "all"
MODE_SIGNIFICANT = "significant"

#: Adjusted p below which a comparison is drawn in ``MODE_SIGNIFICANT``.
ALPHA = 0.05

#: How many brackets one axes will carry. Thirteen vessels are seventy-eight comparisons; past a
#: dozen the brackets are taller than the figure and hide the data they are about. The cap keeps
#: the most significant ones, because the frame arrives sorted by adjusted p.
MAX_BRACKETS = 12


def format_p(value: float) -> str:
    """The p-value as it goes under a bracket — short enough not to widen it."""
    if not np.isfinite(value):
        return "p n/a"
    # One threshold, not two: ``0.0004`` is easier to read than ``4.0e-04``, and below 1e-4 the
    # exact value is not what anyone is reading off a bracket.
    if value < 1e-4:
        return "p<1e-4"
    return f"p={value:.3g}"


@dataclass(frozen=True)
class Bracket:
    """One bracket to draw: which two positions it spans, how high it stacks, what it says."""

    left: float
    right: float
    row: int
    stars: str
    p_adj: float
    #: Level of the second factor this comparison was made within, or ``""``.
    by: str = ""

    def label(self, *, show_p: bool = True) -> str:
        """Stars over the number, so the marker reads at a glance and the value is there to cite."""
        marker = self.stars or "NA"
        return f"{marker}\n{format_p(self.p_adj)}" if show_p else marker


def level_positions(labels: Sequence[Any], ticks: Sequence[float]) -> dict[str, float]:
    """
    Map each categorical tick's level name to its x position.

    Tick text is not always the bare level — the distribution plots put the group's N on a second
    line — so only the first line is matched. That is the level; everything after it is decoration.
    """
    out: dict[str, float] = {}
    for label, position in zip(labels, ticks):
        text = label if isinstance(label, str) else label.get_text()
        name = str(text).split("\n")[0].strip()
        if name:
            out.setdefault(name, float(position))
    return out


def bracket_layout(
    contrasts: pd.DataFrame,
    positions: Mapping[str, float],
    *,
    mode: str = MODE_ALL,
    max_brackets: int = MAX_BRACKETS,
) -> tuple[list[Bracket], int]:
    """
    Choose and stack the brackets to draw, backend-independently.

    Returns ``(brackets, omitted)`` — the second is how many comparisons were left off, which the
    caller has to report rather than let a capped figure imply that is all there was.

    Brackets are stacked greedily into rows: a bracket goes in the lowest row whose spans it does
    not overlap, so a short comparison sits under a long one instead of on top of it.
    """
    if contrasts is None or contrasts.empty:
        return [], 0

    usable = eligible(contrasts, mode)
    usable = usable.loc[
        usable["a"].astype(str).isin(positions) & usable["b"].astype(str).isin(positions)
    ]
    total = len(usable)
    if not total:
        return [], 0

    # Sorted by adjusted p on the way in, so the head of the frame is what survives the cap.
    chosen = usable.head(max(int(max_brackets), 0))

    brackets: list[Bracket] = []
    base_row = 0
    # One contiguous band per by-level, in the order they are drawn, rather than every series'
    # comparisons interleaved up the axis: a band is what makes "these are RICA's" readable at a
    # glance, which the colour alone does not do once three series overlap.
    for group in dict.fromkeys(str(g) for g in chosen.get("by", pd.Series(dtype=str))) or [""]:
        block = chosen if not group else chosen.loc[chosen["by"].astype(str) == group]
        rows: list[list[tuple[float, float]]] = []
        # Narrow spans first: a comparison between neighbours belongs below one that reaches
        # across the axis, and that order is what lets the greedy search find a low row.
        ordered = sorted(
            block.itertuples(index=False),
            key=lambda c: abs(positions[str(c.b)] - positions[str(c.a)]),
        )
        for contrast in ordered:
            left, right = sorted((positions[str(contrast.a)], positions[str(contrast.b)]))
            row = next(
                (
                    index for index, occupied in enumerate(rows)
                    if all(right < start or left > end for start, end in occupied)
                ),
                len(rows),
            )
            if row == len(rows):
                rows.append([])
            rows[row].append((left, right))
            brackets.append(
                Bracket(
                    left=left, right=right, row=base_row + row,
                    stars=str(contrast.stars) or "NA",
                    p_adj=float(contrast.p_adj),
                    by=group,
                )
            )
        base_row += len(rows)
    return brackets, total - len(brackets)


def series_colours(ax: Any) -> dict[str, Any]:
    """
    Each legend entry's level name and the colour drawn for it.

    Used to paint a series' brackets in that series' colour, and — on a grouped display, where
    each panel carries only some of the levels — to tell which comparisons belong on this panel
    at all. Legend labels are matched after the last ``=``, since the plotters spell them both
    ``RICA`` and ``territory=RICA``.
    """
    try:
        handles, labels = ax.get_legend_handles_labels()
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        name = str(label).rsplit("=", 1)[-1].strip()
        if not name:
            continue
        colour = (
            getattr(handle, "get_color", None) and handle.get_color()
        ) or getattr(handle, "get_facecolor", lambda: None)()
        out.setdefault(name, colour)
    return out


def draw_brackets(
    ax: Any,
    brackets: Sequence[Bracket],
    *,
    colour: str = "#444444",
    colours: Mapping[str, Any] | None = None,
    fontsize: int = 8,
    show_p: bool = True,
) -> None:
    """Draw *brackets* above the data on a Matplotlib axes, making room for them.

    ``colours`` paints each ``by`` level's band in its own series colour, so a bracket belongs to
    a curve by sight rather than by counting bands.
    """
    if not brackets:
        return

    bottom, top = ax.get_ylim()
    span = float(top - bottom) or 1.0
    # Taller rows when the p-value rides under the stars, or the two lines collide with the
    # bracket above them.
    step = span * (0.115 if show_p else 0.075)
    base = top + step * 0.3
    drop = step * 0.13
    palette = dict(colours or {})

    for bracket in brackets:
        y = base + bracket.row * step
        pen = palette.get(bracket.by, colour) if bracket.by else colour
        ax.plot(
            [bracket.left, bracket.left, bracket.right, bracket.right],
            [y - drop, y, y, y - drop],
            lw=1.0, color=pen, clip_on=False, solid_joinstyle="miter",
        )
        ax.text(
            (bracket.left + bracket.right) / 2.0, y + drop * 0.3,
            bracket.label(show_p=show_p),
            ha="center", va="bottom", fontsize=fontsize, color=pen, clip_on=False,
            linespacing=0.95,
        )
    highest = base + max(b.row for b in brackets) * step
    ax.set_ylim(bottom, highest + step * 1.1)


def annotate_axes(
    ax: Any,
    contrasts: pd.DataFrame,
    *,
    mode: str = MODE_ALL,
    max_brackets: int = MAX_BRACKETS,
    show_p: bool = True,
) -> tuple[int, int]:
    """
    Put significance brackets over the categorical levels of a finished axes.

    Works from the axes' own tick labels and legend rather than from the plotting call, so one
    implementation serves every engine's plotter — they all end up with the levels on the
    categorical axis and the series in the legend, and none of them agree on much else.

    That is also what makes the grouped display work without being told about it: a panel holding
    three of thirteen territories has three legend entries, so only those three series' brackets
    are drawn on it, in their own colours.

    Returns ``(drawn, omitted)``.
    """
    positions = level_positions(ax.get_xticklabels(), ax.get_xticks())
    colours = series_colours(ax)

    scoped = contrasts
    if scoped is not None and not scoped.empty and str(scoped["by"].iloc[0] or ""):
        present = set(colours)
        if present:
            scoped = scoped.loc[scoped["by"].astype(str).isin(present)]

    brackets, omitted = bracket_layout(
        scoped, positions, mode=mode, max_brackets=max_brackets
    )
    draw_brackets(ax, brackets, colours=colours, show_p=show_p)
    return len(brackets), omitted


def _plotly_extent(figure: Any) -> tuple[float, float]:
    """``(top, span)`` of whatever the traces actually reach, for placing brackets above them."""
    lo, hi = np.inf, -np.inf
    for trace in figure.data:
        values = getattr(trace, "y", None)
        if values is None:
            continue
        numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
        if numeric.empty:
            continue
        lo, hi = min(lo, float(numeric.min())), max(hi, float(numeric.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    return hi, (hi - lo) or abs(hi) or 1.0


def annotate_plotly(
    figure: Any,
    contrasts: pd.DataFrame,
    order: Sequence[str],
    *,
    mode: str = MODE_ALL,
    max_brackets: int = MAX_BRACKETS,
    colour: str = "#555555",
) -> tuple[int, int]:
    """
    The same brackets on a Plotly figure, over a categorical x axis.

    Positions are the category *indices*: a Plotly categorical axis accepts a number as a position
    along it, which is what lets one layout serve both backends instead of two that drift.

    Returns ``(drawn, omitted)``.
    """
    positions = {str(level): float(index) for index, level in enumerate(order)}
    brackets, omitted = bracket_layout(
        contrasts, positions, mode=mode, max_brackets=max_brackets
    )
    if not brackets:
        return 0, omitted

    top, span = _plotly_extent(figure)
    step = span * 0.085
    base = top + step * 0.4
    drop = step * 0.18
    for bracket in brackets:
        y = base + bracket.row * step
        figure.add_shape(
            type="path",
            path=(
                f"M {bracket.left},{y - drop} L {bracket.left},{y} "
                f"L {bracket.right},{y} L {bracket.right},{y - drop}"
            ),
            line={"color": colour, "width": 1},
            xref="x", yref="y", layer="above",
        )
        figure.add_annotation(
            x=(bracket.left + bracket.right) / 2.0, y=y,
            text=bracket.label(show_p=True).replace("\n", "<br>"),
            showarrow=False, yshift=9, font={"size": 10, "color": colour},
            xref="x", yref="y",
        )
    highest = base + max(b.row for b in brackets) * step
    figure.update_yaxes(range=[top - span * 1.08, highest + step * 0.9])
    return len(brackets), omitted


def eligible(contrasts: pd.DataFrame, mode: str) -> pd.DataFrame:
    """The comparisons *mode* asks to show, most significant first."""
    if contrasts is None or contrasts.empty:
        return pd.DataFrame(columns=list(CONTRAST_COLUMNS))
    if str(mode) == MODE_SIGNIFICANT:
        return contrasts.loc[contrasts["p_adj"].astype(float) < ALPHA]
    return contrasts


def annotate_corner(
    ax: Any,
    contrasts: pd.DataFrame,
    *,
    mode: str = MODE_ALL,
    max_rows: int = MAX_BRACKETS,
    colour: str = "#444444",
) -> tuple[int, int]:
    """
    List the comparisons in a box on the axes, for a plot with no categorical axis to span.

    A continuous plot draws one fitted line per level, so "different from which" is still the
    question — but there are no ticks to bracket between, and a bracket across a numeric axis
    would claim a range of x it does not mean. A list says the same thing without the lie.

    Returns ``(drawn, omitted)``.
    """
    usable = eligible(contrasts, mode)
    total = len(usable)
    if not total:
        return 0, 0
    shown = usable.head(max(int(max_rows), 0))
    lines = [
        (f"{row.by}:  " if str(getattr(row, "by", "") or "") else "")
        + f"{row.a} − {row.b}   {row.stars or 'NA'}  {format_p(float(row.p_adj))}"
        for row in shown.itertuples(index=False)
    ]
    if total > len(shown):
        lines.append(f"(+{total - len(shown)} more)")
    ax.text(
        0.995, 0.995, "\n".join(lines),
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=colour,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.78,
              "edgecolor": "#CCCCCC"},
    )
    return len(shown), total - len(shown)


def contrast_note(drawn: int, omitted: int, mode: str) -> str:
    """What the figure has to say about brackets it did not draw."""
    if not drawn and not omitted:
        return (
            "No pairwise comparison reached p < 0.05." if str(mode) == MODE_SIGNIFICANT
            else "No pairwise comparisons available for this plot."
        )
    note = f"{drawn} pairwise comparison(s), Holm-adjusted"
    if omitted:
        note += f"; {omitted} more not drawn"
    return note + "."


__all__ = [
    "ALPHA",
    "CONTRAST_COLUMNS",
    "MAX_BRACKETS",
    "MODE_ALL",
    "MODE_SIGNIFICANT",
    "Bracket",
    "Contrast",
    "annotate_axes",
    "annotate_corner",
    "annotate_plotly",
    "bracket_layout",
    "contrast_note",
    "draw_brackets",
    "eligible",
    "format_p",
    "holm",
    "level_positions",
    "pairwise_contrasts",
    "series_colours",
]
