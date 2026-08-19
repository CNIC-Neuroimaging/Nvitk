"""
Per-region numbers pulled off a fitted model, for any anatomical map.

Description
-----------
:mod:`nvitk.stats.vascular_map` and :mod:`nvitk.stats.brain_map` draw completely different pictures —
a schematic circle of Willis, a cortical surface — but they ask the model exactly the same question:
*what number does this fit give for this anatomical level, and how sure is it?* Answering it means
knowing how patsy spells a categorical contrast, how each R engine wraps its coefficient table, what
a marginal mean is on the outcome's own scale, and which half of an interaction a contrast view
means. None of that is about anatomy.

So it lives here once, parameterised by a **resolver**: a callable turning a published level label
(``LICA``, ``ctx-lh-precuneus``, ``ctx-Left-Frontal-Lobe``) into the drawing keys it refers to — zero
when the level is not on this map, one for a plain level, several when it is an aggregate that
belongs on every member it was averaged from.

Engines
-------
statsmodels exposes ``params``/``pvalues`` directly; every R engine wraps an object that does not,
and each wraps it differently — an ``lmrob`` fit *is* the R object, a pymer4 model holds one, MMRM
another. :data:`_COEF_NORMALIZERS` is tried in turn so one code path serves every engine instead of
the maps silently working only for OLS and MixedLM.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Callable, Hashable, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: ``label -> drawing keys``. Several keys for an aggregate level, none when the level is not drawn.
Resolver = Callable[[Any], "list[Hashable]"]

#: Per-engine normalizers to a ``parameter / coef / p_value`` table, tried in turn.
_COEF_NORMALIZERS: tuple[tuple[str, str], ...] = (
    ("nvitk.stats.r_mixedlm", "lme4_coef_frame"),
    ("nvitk.stats.r_robust", "lmrob_coef_frame"),
    ("nvitk.stats.r_mmrm", "mmrm_coef_frame"),
    ("nvitk.stats.r_gam", "mrf_coef_frame"),
)


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------
def coefficient_series(result: Any) -> tuple[pd.Series, pd.Series | None]:
    """
    ``(params, pvalues)`` as pandas Series, whichever engine produced *result*.

    Raises
    ------
    ValueError
        When no normalizer recognizes the object, naming what was tried — a silent empty result
        here becomes "the map shows observed means" three frames up, which looks like data rather
        than a failure.
    """
    from importlib import import_module

    from nvitk.stats.mixedlm import model_params

    params = model_params(result)
    pvals = getattr(result, "pvalues", None)
    if isinstance(params, pd.Series) and not params.empty:
        return params, pvals if isinstance(pvals, pd.Series) else None

    tried: list[str] = []
    for module_name, function_name in _COEF_NORMALIZERS:
        try:
            table = getattr(import_module(module_name), function_name)(result)
        except Exception as exc:
            tried.append(f"{function_name} ({type(exc).__name__})")
            continue
        if table is None or table.empty or "parameter" not in table.columns:
            tried.append(f"{function_name} (empty)")
            continue
        index = table["parameter"].astype(str)
        return (
            pd.Series(pd.to_numeric(table["coef"], errors="coerce").to_numpy(), index=index),
            pd.Series(pd.to_numeric(table["p_value"], errors="coerce").to_numpy(), index=index),
        )
    raise ValueError(
        "No coefficient table could be read from this fit. Tried: " + "; ".join(tried) + "."
    )


def term_parts(name: str) -> list[str]:
    """Split a patsy/R interaction term into its factors (``a[T.x]:b[T.y]`` → two parts)."""
    return [part for part in str(name).split(":") if part]


def level_of(part: str, prefixes: Sequence[str]) -> str:
    """The factor level a single (non-interaction) term names."""
    name = str(part)
    if "[" in name and name.endswith("]"):
        level = name[name.index("[") + 1: -1]
        return level[2:] if level.startswith("T.") else level
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return name


def term_prefixes(group_column: str, data: pd.DataFrame | None) -> list[str]:
    """
    Candidate factor names a term may be prefixed with, longest first.

    Longest first so ``territory_id`` is tried before ``territory`` and cannot mis-split a term.
    Every column of the model frame is a candidate, not just *group_column*: the plot's grouping is
    often ``group_key`` while the model term is ``territory``, and matching only the former left the
    R engines resolving nothing.
    """
    return sorted(
        {str(c) for c in (data.columns if data is not None else [])} | {str(group_column)},
        key=len, reverse=True,
    )


def interaction_contrasts(
    result: Any,
    *,
    resolver: Resolver,
    group_column: str = "territory",
    data: pd.DataFrame | None = None,
) -> list[str]:
    """
    Levels of the *other* factor in interactions with the anatomical term, e.g. ``["sex[T.M]"]``.

    A model with ``territory * sex`` estimates a different profile for each sex, and the coefficient
    table holds them as ``territory[T.LICA]:sex[T.M]``. Without naming which side of the interaction
    to draw, a map would show only the main effect — the profile at the *other* factor's reference
    level — and quietly present it as the whole story.

    Empty when the model has no interaction on that term, which is the signal a caller needs to hide
    the selector rather than offer one with nothing in it.
    """
    try:
        params = coefficient_series(result)[0]
    except Exception:
        return []
    prefixes = term_prefixes(group_column, data)
    found: list[str] = []
    for term in params.index:
        parts = term_parts(str(term))
        if len(parts) < 2:
            continue
        anatomical = [p for p in parts if resolver(level_of(p, prefixes))]
        if not anatomical:
            continue
        for other in (p for p in parts if p not in anatomical):
            if other not in found:
                found.append(other)
    return found


def group_coefficient_terms(result: Any, *, group_column: str = "territory") -> list[str]:
    """
    Per-group terms this fit estimates for *group_column*, e.g. ``["(Intercept)", "age_c"]``.

    Empty when the model has no random structure over that factor — which is the signal a caller
    needs to fall back to the fixed effects instead of offering a menu with nothing in it.
    """
    try:
        from nvitk.stats.r_mixedlm import lme4_group_coefficients

        table = lme4_group_coefficients(result)
    except Exception:
        return []
    if table is None or table.empty or "factor" not in table.columns:
        return []
    rows = table.loc[table["factor"].astype(str) == str(group_column)]
    if rows.empty:
        return []
    return [
        c for c in rows.columns
        if c not in {"factor", "level"}
        and not str(c).endswith("_dev")
        and rows[c].notna().any()
    ]


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------
def values_from_frame(
    frame: pd.DataFrame,
    *,
    resolver: Resolver,
    key_column: str,
    value_column: str,
    pvalue_column: str | None = None,
) -> tuple[dict[Hashable, float], dict[Hashable, float]]:
    """
    Map a per-region result table onto drawing keys.

    Accepts whatever spelling the frame uses — the *resolver* absorbs the differences. Rows that do
    not resolve (an intercept term, a whole-head scalar, a region from a different modality) are
    dropped silently: a mixed result table is the normal case, and only the rows this map can draw
    belong on it.

    Returns
    -------
    (values, pvalues)
        Both keyed by drawing key; *pvalues* is empty when *pvalue_column* is ``None``.
    """
    values: dict[Hashable, float] = {}
    pvalues: dict[Hashable, float] = {}
    if frame is None or frame.empty or key_column not in frame.columns:
        return values, pvalues
    if value_column not in frame.columns:
        raise ValueError(
            f"{value_column!r} is not in the frame. Columns: {', '.join(map(str, frame.columns))}."
        )

    for _, row in frame.iterrows():
        keys = resolver(row[key_column])
        if not keys:
            continue
        value = pd.to_numeric(row[value_column], errors="coerce")
        if pd.isna(value):
            continue
        for key in keys:
            values[key] = float(value)
        if pvalue_column and pvalue_column in frame.columns:
            p = pd.to_numeric(row.get(pvalue_column), errors="coerce")
            if not pd.isna(p):
                for key in keys:
                    pvalues[key] = float(p)
    return values, pvalues


def values_from_result(
    result: Any,
    *,
    resolver: Resolver,
    group_column: str = "territory",
    source: str = "coefficient",
    data: pd.DataFrame | None = None,
    outcome: str = "",
    contrast: str = "",
    unit: str = "region",
) -> tuple[dict[Hashable, float], dict[Hashable, float], str]:
    """
    Pull per-region numbers off a fitted model.

    Sources, because they answer different questions and are read differently:

    ``coefficient``
        The fixed effect of each level, read from the parameter table — patsy names them
        ``territory[T.LICA]``, so the level is recovered from the term. These are contrasts
        *against the reference level*, which is therefore absent from the map by construction: it
        is the zero everything else is measured from, not a region with no effect.

    ``emmeans``
        The model-predicted mean per region on the outcome's own scale, **including** the reference
        level. See :func:`values_from_marginal_means`.

    ``mean``
        The observed mean of *outcome* within each region, from *data*. Not a model estimate at all
        — no covariate adjustment, no p-values — but it is what "estimated marginal means" reduces
        to for a model whose only term is the region, and it is the honest fallback when the fit has
        no per-region parameter to read.

    ``group:<term>``
        A per-region value from an ``lme4`` random-effects structure. See
        :func:`values_from_group_coefficients`.

    contrast : str
        A level of an interacting factor, as :func:`interaction_contrasts` names it (``sex[T.M]``).
        The region's value becomes its **simple effect at that level** — main effect plus the
        interaction — which is the profile actually seen in that group. Left empty, the map shows
        the main effect, i.e. the profile at the interacting factor's *reference* level; on a model
        with an interaction that is one group's answer presented as if it were everyone's.

        The p-value reported alongside stays the **interaction** term's: it tests whether this
        region's effect differs between the groups, which is the question a contrast view raises.
        A simple effect's own test needs a linear combination the coefficient table does not carry.

    Returns
    -------
    (values, pvalues, note)
        *note* describes what was extracted, for the figure subtitle.
    """
    source = str(source).strip()
    values: dict[Hashable, float] = {}
    pvalues: dict[Hashable, float] = {}

    if source.startswith("group:"):
        term = source.split(":", 1)[1] or "(Intercept)"
        return values_from_group_coefficients(
            result, resolver=resolver, group_column=group_column, term=term, unit=unit
        )

    if source.lower() in {"emmeans", "emm"}:
        return values_from_marginal_means(
            result, resolver=resolver, group_column=group_column, data=data, unit=unit
        )

    if source.lower() == "mean":
        if data is None or not outcome or outcome not in data.columns:
            raise ValueError(
                "source='mean' needs the model frame and an outcome column to average."
            )
        if group_column not in data.columns:
            raise ValueError(f"{group_column!r} is not in the model frame.")
        grouped = data.groupby(data[group_column].astype(str), observed=True)[outcome].mean()
        for level, value in grouped.items():
            if not np.isfinite(value):
                continue
            for key in resolver(level):
                values[key] = float(value)
        return values, pvalues, f"observed mean {outcome} per {unit}"

    params, pvals = coefficient_series(result)
    prefixes = term_prefixes(group_column, data)

    reference: str | None = None
    mirrored = False
    for term, coef in params.items():
        # The bracketed level is taken from *any* term rather than only from ones naming
        # ``group_column``: the plot's grouping column is often 'group_key' while the model names
        # the term 'territory', and requiring them to match meant no parameter was ever recognized.
        level = level_of(str(term), prefixes)
        keys = resolver(level)
        if not keys or not np.isfinite(coef):
            continue
        mirrored |= len(keys) > 1
        for key in keys:
            values[key] = float(coef)
        if pvals is not None and term in getattr(pvals, "index", []):
            p = pvals[term]
            if np.isfinite(p):
                for key in keys:
                    pvalues[key] = float(p)

    # ---- Fold in the interaction, when one was asked for ---------------------------------------
    if contrast:
        matched = 0
        for term, coef in params.items():
            parts = term_parts(str(term))
            if len(parts) < 2 or contrast not in parts:
                continue
            region_part = next((p for p in parts if resolver(level_of(p, prefixes))), "")
            keys = resolver(level_of(region_part, prefixes)) if region_part else []
            keys = [k for k in keys if k in values]
            if not keys or not np.isfinite(coef):
                continue
            for key in keys:
                values[key] += float(coef)
            matched += 1
            # The interaction's own p-value replaces the main effect's: it is the one that tests
            # the difference this view is about.
            if pvals is not None and term in getattr(pvals, "index", []):
                p = pvals[term]
                if np.isfinite(p):
                    for key in keys:
                        pvalues[key] = float(p)
        if not matched:
            available = interaction_contrasts(
                result, resolver=resolver, group_column=group_column, data=data
            )
            raise ValueError(
                f"No interaction term pairs a {unit} with {contrast!r}. Available contrasts: "
                f"{', '.join(available) or 'none'}."
            )

    if not values:
        raise ValueError(
            f"No parameter of this model names a drawable level of {group_column!r}. Fit a model "
            f"with {group_column} as a term, or switch the source to the observed mean."
        )

    # Name the reference level so the reader knows why one region is blank.
    if data is not None and group_column in data.columns:
        levels = {k for v in data[group_column].astype(str).unique() for k in resolver(v)}
        missing = sorted(levels - set(values), key=str)
        reference = str(missing[0]) if len(missing) == 1 else None

    note = f"{group_column} coefficients"
    if mirrored:
        note += ", mirrored across members"
    if contrast:
        note += f" at {contrast}"
    if reference:
        note += f" (reference: {reference})"
    return values, pvalues, note


def values_from_marginal_means(
    result: Any,
    *,
    resolver: Resolver,
    group_column: str,
    data: pd.DataFrame | None,
    unit: str = "region",
) -> tuple[dict[Hashable, float], dict[Hashable, float], str]:
    """
    Model-predicted mean per region, on the outcome's own scale — **including the reference level**.

    Treatment coding gives the reference level no coefficient: it is absorbed into the intercept,
    which is why a ``~ territory + …`` fit shows every region but one. That is correct for a
    contrast table and wrong for a map, where the reader is looking at anatomy and one structure is
    simply missing.

    The marginal mean rebuilds it: ``intercept + β_region`` (with ``β = 0`` for the reference), plus
    each numeric covariate held at its mean. Categorical covariates stay at their own reference
    level, so the result is "predicted value in an average subject", comparable across regions and
    interpretable on the outcome's unit rather than as a difference from whichever region sorted
    first.

    No p-values come back. A coefficient's p-value tests the *contrast against the reference*, not
    whether a marginal mean differs from zero, and carrying it here would attach a test to a number
    it does not describe. Use the coefficient view when significance is the question.
    """
    params, _ = coefficient_series(result)
    prefixes = term_prefixes(group_column, data)

    intercept = 0.0
    for name in ("Intercept", "(Intercept)", "const"):
        if name in params.index:
            intercept = float(params[name])
            break

    # Covariate offset: numeric terms at their mean, everything else at its reference. Shared by
    # every region, so it shifts the map without changing the ordering — but it is what puts the
    # numbers on the outcome's scale instead of an arbitrary one.
    offset = 0.0
    if data is not None:
        for term, coef in params.items():
            name = str(term)
            if name in {"Intercept", "(Intercept)", "const"} or ":" in name:
                continue
            if resolver(level_of(name, prefixes)):
                continue
            if name in data.columns and pd.api.types.is_numeric_dtype(data[name]):
                column = pd.to_numeric(data[name], errors="coerce")
                if column.notna().any() and np.isfinite(coef):
                    offset += float(coef) * float(column.mean())

    values: dict[Hashable, float] = {}
    for term, coef in params.items():
        keys = resolver(level_of(str(term), prefixes))
        if keys and np.isfinite(coef) and ":" not in str(term):
            for key in keys:
                values[key] = intercept + offset + float(coef)

    # The reference level has no term of its own; its marginal mean is the intercept.
    reference = None
    if data is not None and group_column in data.columns:
        levels = {
            k for v in data[group_column].astype(str).dropna().unique() for k in resolver(v)
        }
        missing = sorted(levels - set(values), key=str)
        for key in missing:
            values[key] = intercept + offset
        reference = str(missing[0]) if len(missing) == 1 else None

    if not values:
        raise ValueError(
            f"No parameter of this model names a drawable level of {group_column!r}, so there are "
            f"no marginal means to draw."
        )
    note = f"marginal mean per {unit}"
    if reference:
        note += f" ({reference} recovered from the intercept)"
    return values, {}, note


def values_from_group_coefficients(
    result: Any,
    *,
    resolver: Resolver,
    group_column: str,
    term: str,
    unit: str = "region",
) -> tuple[dict[Hashable, float], dict[Hashable, float], str]:
    """
    Per-region values from an ``lme4`` random-effects structure.

    A model like ``flow_mean ~ age_c + (1 + age_c | territory)`` puts nothing about individual
    regions in its *fixed* effects — the region information is the random structure, one intercept
    and one ``age_c`` slope per level. ``coef()`` totals are used rather than ``ranef()`` deviations
    because a total is the quantity with a physical reading: the age slope *in that region*, not its
    departure from the average slope.

    Random effects are shrunk point predictions, not tested parameters, so no p-values come back.
    Greying by significance is therefore unavailable here, which is correct — a BLUP has no null
    hypothesis attached to it.
    """
    from nvitk.stats.r_mixedlm import lme4_group_coefficients

    table = lme4_group_coefficients(result)
    if table is None or table.empty:
        raise ValueError("This model exposes no per-group coefficients.")
    rows = table.loc[table["factor"].astype(str) == str(group_column)]
    if rows.empty:
        factors = sorted({str(f) for f in table["factor"]})
        raise ValueError(
            f"No random effects over {group_column!r}. Grouping factors in this fit: "
            f"{', '.join(factors) or 'none'}."
        )
    if term not in rows.columns or not rows[term].notna().any():
        available = [c for c in rows.columns if c not in {"factor", "level"}
                     and not str(c).endswith("_dev") and rows[c].notna().any()]
        raise ValueError(
            f"{term!r} is not a per-group term of {group_column!r}. Available: "
            f"{', '.join(available) or 'none'}."
        )

    values: dict[Hashable, float] = {}
    for level, value in zip(rows["level"], rows[term]):
        if not np.isfinite(value):
            continue
        for key in resolver(level):
            values[key] = float(value)
    if not values:
        raise ValueError(f"None of {group_column!r}'s levels resolve to a drawn {unit}.")
    label = "intercept" if term.strip("()").lower() == "intercept" else f"{term} slope"
    return values, {}, f"per-{unit} {label} ({group_column} random effects)"


__all__ = [
    "Resolver",
    "coefficient_series",
    "group_coefficient_terms",
    "interaction_contrasts",
    "level_of",
    "term_parts",
    "term_prefixes",
    "values_from_frame",
    "values_from_group_coefficients",
    "values_from_marginal_means",
    "values_from_result",
]
