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

What counts as "the set" is the family the figure is read in. With a ``by`` — one series or one
panel per level of a second factor — each level is corrected on its own, which is also
``emmeans``' convention for a ``by`` specification. Pooling them instead makes seventeen
territories a single family of 102 tests, and a real effect at raw p = 0.003 comes out at 0.30:
every bracket reads NS and nothing is drawn, which says far more about the denominator than about
the data.
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

#: Whether the p-values a bracket shows are corrected for multiplicity.
ADJUST_HOLM = "holm"
ADJUST_NONE = "none"

#: What a bracket between two levels is testing — the simple effect inside a series, or the
#: difference-in-differences against the reference series, which is what an interaction term is.
BASIS_WITHIN = "within"
#: One bracket per interaction row of the coefficient table — pairs against the factor's own
#: reference, which is what an interaction *term* contrasts.
BASIS_INTERACTION = "interaction"
#: The same difference-in-differences over *every* pair of levels. A g1-versus-g2 bracket is then
#: the difference of two interaction coefficients: a real contrast, and one no table row carries.
BASIS_INTERACTION_ALL = "interaction_all"

#: Both bases that measure against the reference series.
INTERACTION_BASES = frozenset({BASIS_INTERACTION, BASIS_INTERACTION_ALL})

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
    # Corrected within each ``by`` level, not across the whole set. That is what ``emmeans`` does
    # with a ``by`` specification, and it is how the figure is read: a series' brackets are a
    # family of their own, drawn over its own curve. Across the set instead, seventeen
    # territories make one family of 102 tests — where a real effect at raw p = 0.003 adjusts to
    # 0.30, every bracket reads NS, and "significant only" draws nothing at all.
    grouped = out["by"].astype(bool).any()
    out["p_adj"] = (
        out.groupby("by", sort=False)["p_value"].transform(lambda s: holm(s.to_numpy()))
        if grouped else holm(out["p_value"])
    )
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
    basis: str = BASIS_WITHIN,
    reference: str = "",
    factor_reference: str = "",
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
    # The reference block's rows, for the difference-in-differences: an interaction contrast is
    # this series' (a − b) minus the reference series' (a − b), which is exactly what the
    # coefficient table's interaction row reports.
    baseline = (
        groups.index(reference) * width
        if basis in INTERACTION_BASES and reference in groups else None
    )
    for block, group in enumerate(groups):
        offset = block * width
        for i in range(width):
            for j in range(i + 1, width):
                if factor_reference and factor_reference not in {
                    str(levels[i]), str(levels[j])
                }:
                    # An interaction *term* contrasts a level against the factor's reference.
                    # Any other pair is the difference of two of them — a real contrast, but not
                    # one the coefficient table prints, so not one this basis claims to show.
                    continue
                contrast = rows_design[offset + i] - rows_design[offset + j]
                if baseline is not None:
                    contrast = contrast - (rows_design[baseline + i] - rows_design[baseline + j])
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

#: The difference-in-differences form. ``interaction = "pairwise"`` gives every pair of *factor*
#: crossed with every pair of *by*; the rows wanted are the ones whose ``by`` pair involves the
#: reference level, which is what the coefficient table's interaction rows report.
_R_PAIRS_INTERACTION_HELPER = """
.nvitk_pairs_interaction <- function(model, factor_name, by_name) {
  suppressMessages(try(emmeans::emm_options(lmer.df = "satterthwaite"), silent = TRUE))
  spec <- stats::as.formula(paste("~", factor_name, "*", by_name))
  em <- emmeans::emmeans(model, specs = spec)
  out <- as.data.frame(summary(
    emmeans::contrast(em, interaction = c("pairwise", "pairwise"), adjust = "none")))
  nm <- names(out)
  names(out)[nm == paste0(factor_name, "_pairwise")] <- ".contrast"
  names(out)[nm == paste0(by_name, "_pairwise")] <- ".by_contrast"
  out
}
"""

_PAIRS_HELPER_LOADED = False


def _r_contrasts(
    model: Any,
    *,
    factor: str,
    levels: Sequence[str],
    by: str = "",
    basis: str = BASIS_WITHIN,
    reference: str = "",
    factor_reference: str = "",
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
        R_(_R_PAIRS_INTERACTION_HELPER)
        _PAIRS_HELPER_LOADED = True

    if basis in INTERACTION_BASES:
        return _r_interaction_contrasts(
            model, factor=factor, levels=levels, by=by, reference=reference,
            factor_reference=factor_reference,
        )

    try:
        with localconverter(default_converter + pandas2ri.converter):
            table = pd.DataFrame(globalenv[".nvitk_pairs"](model, str(factor), str(by)))
    except Exception as exc:
        # emmeans has no ``emm_basis`` for every model this toolkit fits — ``lmrob`` is the one
        # that matters here — and says so by raising. The contrasts are still computable from the
        # fit's own coefficients and covariance, which is what this falls back to.
        # Debug, not info: for an lmrob fit this fires on every plot, the fallback reproduces
        # emmeans to 1e-9, and an R traceback in the log on each redraw reads like a failure.
        log.debug(
            "emmeans declined %s (%s); using the design-matrix basis.",
            type(model).__name__, str(exc).strip().splitlines()[0] if str(exc).strip() else exc,
            exc_info=True,
        )
        from .r_basis import linear_pairs

        table = linear_pairs(model, factor, by=by, reference="", factor_reference="")

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


def _r_interaction_contrasts(
    model: Any,
    *,
    factor: str,
    levels: Sequence[str],
    by: str,
    reference: str,
    factor_reference: str = "",
) -> pd.DataFrame:
    """
    Difference-in-differences against *reference*, from ``emmeans``' interaction contrasts.

    ``emmeans`` returns every pair of *factor* crossed with every pair of *by*; only the rows
    whose ``by`` pair involves the reference level are an interaction *coefficient*. Which side
    of that pair the reference falls on decides the sign — ``BASILAR - RICA`` is the negative of
    the row the table prints for RICA.
    """
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter

    try:
        with localconverter(default_converter + pandas2ri.converter):
            table = pd.DataFrame(
                globalenv[".nvitk_pairs_interaction"](model, str(factor), str(by))
            )
    except Exception as exc:
        log.debug(
            "emmeans declined the interaction contrast (%s); using the design-matrix basis.",
            str(exc).strip().splitlines()[0] if str(exc).strip() else exc,
            exc_info=True,
        )
        from .r_basis import linear_pairs

        table = linear_pairs(
            model, factor, by=by, reference=reference, factor_reference=factor_reference
        )

    wanted = {str(level) for level in levels}
    rows: list[dict[str, Any]] = []
    for _index, row in table.iterrows():
        pair = str(row.get(".contrast", "")).split(" - ")
        groups = str(row.get(".by_contrast", "")).split(" - ")
        if len(pair) != 2 or len(groups) != 2:
            continue
        a, b = pair[0].strip(), pair[1].strip()
        left, right = groups[0].strip(), groups[1].strip()
        if a not in wanted or b not in wanted or reference not in {left, right}:
            continue
        if factor_reference and factor_reference not in {a, b}:
            continue
        group = right if left == reference else left
        # Oriented so the estimate is this series minus the reference series, matching the sign
        # the coefficient table prints.
        sign = -1.0 if left == reference else 1.0
        statistic = _first(row, ("t.ratio", "z.ratio", "statistic"))
        rows.append({
            "by": group, "a": a, "b": b,
            "estimate": sign * _first(row, ("estimate",)),
            "se": _first(row, ("SE", "std.error")),
            "statistic": sign * statistic,
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


def reference_level(result: Any, df: pd.DataFrame, column: str) -> str:
    """
    The level of *column* the model contrasts everything else against.

    Found by elimination rather than by guessing at the sort order: every level with a
    coefficient of its own is not the reference, so the one left over is. That is exactly the
    level the coefficient table is silent about, which is what makes an interaction contrast
    computed here line up with the interaction row the table shows.
    """
    from ._model_values import term_parts

    if column not in df.columns:
        return ""
    levels = [str(v) for v in pd.unique(df[column].dropna().astype(str))]
    named: set[str] = set()
    for term in fixed_effect_terms(result):
        for part in term_parts(term):
            name = str(part)
            if name.startswith(f"{column}[") :
                inner = name[name.index("[") + 1: -1]
                named.add(inner[2:] if inner.startswith("T.") else inner)
            elif name.startswith(column) and name[len(column):] in set(levels):
                named.add(name[len(column):])
    missing = [level for level in levels if level not in named]
    return missing[0] if len(missing) == 1 else ""


# ──────────────────────────────────────────────────────────────────────────────
# What the model can actually be asked
# ──────────────────────────────────────────────────────────────────────────────
def fixed_effect_terms(result: Any) -> list[str]:
    """Names of the model's fixed-effect parameters, whichever engine produced it.

    Empty when no normalizer recognizes the object, which callers read as "unknown" rather than
    as "none": refusing a comparison on the strength of a check that did not run is worse than
    letting the engine answer for itself.
    """
    from ._model_values import coefficient_series

    try:
        params, _pvalues = coefficient_series(result)
        return [str(term) for term in params.index]
    except Exception as exc:
        log.debug("Could not read the fixed effects: %s", exc)
        return []


def has_fixed_effect(result: Any, df: pd.DataFrame, column: str) -> bool:
    """
    Whether *column* has a fixed-effect term — the only kind a contrast can be formed from.

    A factor that appears only as a random-effects *grouping* — ``(1 + plaque | territory)`` —
    has no fixed-effect parameters, so there is nothing to contrast between its levels. What it
    has instead are shrunk per-level predictions, which are not tested parameters: a BLUP carries
    no null hypothesis. Asking ``emmeans`` anyway gets *No variable named territory in the
    reference grid*, three frames below where the mistake was made.

    Matching is by coefficient name, which each engine spells differently — patsy's
    ``territory[T.LICA]``, R's ``territoryLICA`` — and the R form is only accepted when what
    follows the column's name is one of its actual levels, so ``age`` does not match ``age_c``.
    """
    terms = fixed_effect_terms(result)
    if not terms:
        return True

    levels = (
        {str(v) for v in df[column].dropna().astype(str)} if column in df.columns else set()
    )
    from ._model_values import term_parts

    for term in terms:
        for part in term_parts(term):
            name = str(part)
            if name == column or name.startswith(f"{column}["):
                return True
            if name.startswith(column) and name[len(column):] in levels:
                return True
    return False


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
    basis: str = BASIS_WITHIN,
    adjust: str = ADJUST_HOLM,
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
    adjust : {"holm", "none"}
        Whether to correct for multiplicity. ``holm`` is the default and the responsible one.
        ``none`` reports the raw p of each comparison, which is what a coefficient *table* shows
        — a regression table corrects nothing — so it is the setting that makes a figure and a
        table agree. An uncorrected bracket is a real result about one comparison; it is only
        misleading if read as though the whole figure had been tested at that level.
    basis : {"within", "interaction", "interaction_all"}
        What a comparison between two levels is testing.

        ``within``
            The simple effect inside each ``by`` level — "does g1 differ from g0 *in RICA*",
            over every pair of levels.
        ``interaction``
            The difference-in-differences against the reference ``by`` level — "does the g0→g1
            effect differ *between RICA and the reference territory*", which is the quantity the
            coefficient table's ``plaque[g1]:territory[RICA]`` row reports. Restricted to pairs
            against the *factor's* own reference, so there is one comparison per interaction row
            of the table and no more.
        ``interaction_all``
            The same quantity over every pair. A g1-versus-g2 comparison is then the difference
            of two interaction coefficients — a real contrast, but one no table row carries, so
            it cannot be checked against the table.

        The last two need a ``by``, and the reference level itself gets no comparisons: it is the
        baseline they are measured from.

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
    if not has_fixed_effect(result, df, factor):
        raise ValueError(
            f"{factor!r} has no fixed-effect term in this model — it appears only in the random "
            "structure, or not at all — so there is nothing to contrast between its levels. "
            "Add it as a fixed effect to compare them."
        )
    present = [str(v) for v in pd.unique(df[factor].dropna().astype(str))]
    order = [str(level) for level in levels] if levels is not None else sorted(present)
    order = [level for level in order if level in set(present)]
    if len(order) < 2:
        raise ValueError(f"{factor!r} has fewer than two levels to compare.")

    # A ``by`` the model has no fixed effect for is dropped rather than raised on: the comparison
    # it was qualifying is still perfectly answerable, just not split per series. Refusing the
    # whole annotation because the colouring column happens not to be in the model would take
    # away brackets that are correct.
    dropped_by = ""
    if by and (by not in df.columns or not has_fixed_effect(result, df, by)):
        dropped_by, by, by_levels = by, "", None

    groups: list[str] = []
    if by:
        seen = [str(v) for v in pd.unique(df[by].dropna().astype(str))]
        groups = (
            [str(level) for level in by_levels if str(level) in set(seen)]
            if by_levels is not None else sorted(seen)
        )
        if not groups:
            raise ValueError(f"{by!r} has no levels to compare within.")

    basis = str(basis or BASIS_WITHIN)
    reference = ""
    factor_reference = ""
    if basis in INTERACTION_BASES:
        if not by:
            raise ValueError(
                "An interaction contrast needs a second factor to compare against — this plot "
                "has only one grouping, so there is no interaction to test."
            )
        reference = reference_level(result, df, by)
        # The factor's own reference too: an interaction term is a level against *that*, so this
        # is what makes the brackets one-for-one with the table's interaction rows. Left empty
        # for the all-pairs basis, which deliberately goes beyond them.
        factor_reference = (
            reference_level(result, df, factor) if basis == BASIS_INTERACTION else ""
        )
        if not reference:
            raise ValueError(
                f"Could not identify {by!r}'s reference level, so there is nothing to measure an "
                "interaction contrast against."
            )
        # Every level is contrasted against the reference, so the reference's own comparisons are
        # identically zero. Dropped rather than drawn as a row of NS brackets that mean nothing.
        groups = [g for g in groups if g != reference]
        if not groups:
            raise ValueError(
                f"{by!r} has only its reference level {reference!r} selected — an interaction "
                "contrast needs at least one level to compare against it."
            )

    if hasattr(result, "cov_params") and hasattr(getattr(result, "model", None), "data"):
        out = _statsmodels_contrasts(
            result, df, factor=factor, levels=order,
            covariate_refs=covariate_refs or {},
            by=by, by_levels=([reference] + groups if reference else groups),
            basis=basis, reference=reference, factor_reference=factor_reference,
        )
        if reference:
            out = out.loc[out["by"] != reference].reset_index(drop=True)
            out = _frame(out.to_dict("records"))
    else:
        out = _r_contrasts(
            _r_object(result), factor=factor, levels=order, by=by,
            basis=basis, reference=reference, factor_reference=factor_reference,
        )
        if by and groups:
            out = out.loc[out["by"].isin(set(groups))].reset_index(drop=True)
    # Carried on the frame so the caller can say why the brackets are not per-series without
    # having to re-derive the reason.
    if str(adjust) == ADJUST_NONE:
        # Re-stated rather than re-derived: the backends always Holm-adjust on the way out, so
        # this puts the raw p back in the column everything downstream reads — the brackets, the
        # cap's ordering and the "significant only" filter all key on ``p_adj``.
        out["p_adj"] = out["p_value"]
        out["stars"] = _stars(out["p_adj"])
        out = out.sort_values("p_adj", na_position="last", kind="stable").reset_index(drop=True)

    out.attrs["dropped_by"] = dropped_by
    # The baseline the interaction contrasts were measured from, for the figure to name.
    out.attrs["reference"] = reference
    out.attrs["adjust"] = str(adjust)
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
    "ADJUST_HOLM",
    "ADJUST_NONE",
    "ALPHA",
    "BASIS_INTERACTION",
    "BASIS_INTERACTION_ALL",
    "BASIS_WITHIN",
    "INTERACTION_BASES",
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
    "fixed_effect_terms",
    "format_p",
    "has_fixed_effect",
    "holm",
    "level_positions",
    "pairwise_contrasts",
    "reference_level",
    "series_colours",
]
