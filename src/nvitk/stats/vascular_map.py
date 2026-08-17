"""
Model estimates drawn on a schematic of the cerebral circulation.

Description
-----------
A forest plot of nineteen vessel coefficients is a list. It answers "which vessel has the largest
effect" and nothing else — the reader cannot see that the two carotids moved together, that the
effect is confined to the posterior circulation, or that the venous side is untouched, because a
forest plot has no notion of which vessel feeds which.

This module draws the same numbers on a top-down schematic of the circle of Willis and the dural
sinuses, so those patterns are visible at a glance. It is a *view of a fit*, not a new analysis:
the numbers are whatever the caller passes in — estimated marginal means, model coefficients,
conservation residuals, QC scores.

Geometry
--------
The layout is schematic, not registered to any atlas: vessels are drawn where a reader expects them
in a top-down projection (anterior at the top), with the anastomotic ring closed by the anterior and
posterior communicating arteries. Coordinates live in a fixed ``[0, 1]²`` space in
:data:`VESSEL_PATHS`, keyed by the same canonical node names as
:mod:`nvitk.stats.vessel_network`, so anything that speaks that vocabulary can be mapped without a
translation table.

Significance
------------
Two ways to show it, and they answer different questions. Colouring by **estimate** with the
non-significant vessels greyed asks "where is the effect, and is it credible" — the default,
because a coloured map that ignores uncertainty invites reading noise as signal. Colouring by
**p-value** asks only "where is the evidence", and deliberately says nothing about direction or
size; it is the right view when comparing coverage across models, and the wrong one for reporting
an effect.

A vessel with no value is drawn in outline. That is different from a vessel that was measured and
found non-significant, and the two must not look alike.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Colour for a vessel the model has no estimate for — drawn as an empty outline, never filled.
COLOR_ABSENT: str = "#d9d9d9"
#: Colour for a vessel whose estimate did not reach *alpha*. Distinct from absent on purpose.
COLOR_NONSIGNIFICANT: str = "#9e9e9e"
#: Skull / dural outline, drawn behind everything as an anatomical frame.
COLOR_OUTLINE: str = "#b9c4d0"

DEFAULT_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
#: ``node -> (control points, stroke width, label anchor, label text)``.
#:
#: Control points are smoothed into a curve (see :func:`_smooth_path`), so three or four per vessel
#: are enough. Widths are relative and roughly follow calibre — an ICA is drawn heavier than a
#: communicating artery — which keeps the schematic legible when many vessels are grey.
VESSEL_PATHS: dict[str, tuple[tuple[tuple[float, float], ...], float, tuple[float, float], str]] = {
    # ---- Anterior circulation ------------------------------------------------
    # The carotid siphon enters from below, turns up, and bifurcates into ACA and MCA at ~y=0.60.
    "lica": (((0.320, 0.330), (0.332, 0.470), (0.362, 0.600)), 9.0, (0.248, 0.470), "ICA"),
    "rica": (((0.680, 0.330), (0.668, 0.470), (0.638, 0.600)), 9.0, (0.752, 0.470), "ICA"),
    "lmca": (((0.362, 0.600), (0.300, 0.638), (0.246, 0.694), (0.294, 0.730)), 7.5,
             (0.286, 0.788), "MCA"),
    "rmca": (((0.638, 0.600), (0.700, 0.638), (0.754, 0.694), (0.706, 0.730)), 7.5,
             (0.788, 0.735), "MCA"),
    "laca": (((0.362, 0.600), (0.392, 0.700), (0.432, 0.798), (0.464, 0.856)), 7.0,
             (0.372, 0.792), "ACA"),
    "raca": (((0.638, 0.600), (0.608, 0.700), (0.568, 0.798), (0.536, 0.856)), 7.0,
             (0.628, 0.792), "ACA"),
    "acomm": (((0.464, 0.856), (0.500, 0.874), (0.536, 0.856)), 6.0, (0.582, 0.898), "AComA"),
    # ---- Posterior circulation -----------------------------------------------
    "basi": (((0.500, 0.330), (0.500, 0.400), (0.500, 0.468)), 8.0, (0.586, 0.352), "Basilar"),
    # The vertebrals converge on the basilar from below. Kept tight to the midline so the straight
    # sinus can pass outside them without the two systems crossing.
    # Kept short: the torcula sits directly below, and a long vertebral leaves the straight sinus
    # no corridor to reach it without crossing one of them.
    "lva": (((0.462, 0.268), (0.480, 0.300), (0.500, 0.330)), 6.5, (0.428, 0.244), "VA"),
    "rva": (((0.538, 0.268), (0.520, 0.300), (0.500, 0.330)), 6.5, (0.572, 0.244), "VA"),
    # Carried past the PComm junction: the posterior cerebral does not end where the communicating
    # artery joins it, and stopping there drew P1 alone — the shortest vessel on the figure for one
    # of the larger ones.
    # Carried past the PComm junction, posterolaterally around the midbrain. Extending it *up* and
    # out instead would have it cross the carotid — true of the real anatomy, which passes behind
    # the ICA, but unreadable in a flat schematic with no depth to show the crossing.
    "lpca": (((0.500, 0.468), (0.456, 0.492), (0.412, 0.500), (0.386, 0.470),
              (0.376, 0.420)), 7.0, (0.430, 0.404), "PCA"),
    "rpca": (((0.500, 0.468), (0.544, 0.492), (0.588, 0.500), (0.614, 0.470),
              (0.624, 0.420)), 7.0, (0.570, 0.404), "PCA"),
    "lpcomm": (((0.362, 0.600), (0.378, 0.552), (0.412, 0.500)), 5.0, (0.240, 0.560), "PComA"),
    "rpcomm": (((0.638, 0.600), (0.622, 0.552), (0.588, 0.500)), 5.0, (0.790, 0.560), "PComA"),
    # ---- Venous drainage -----------------------------------------------------
    # Routed outside the vertebral confluence: the previous path crossed both of them on its way
    # to the torcula, which read as an anastomosis that does not exist.
    "strs": (((0.336, 0.300), (0.362, 0.226), (0.416, 0.168), (0.500, 0.140)), 7.0,
             (0.240, 0.410), "Straight Sinus"),
    "lts": (((0.500, 0.140), (0.368, 0.092), (0.196, 0.140), (0.104, 0.300)), 8.0,
            (0.196, 0.062), "Left Transverse Sinus"),
    "rts": (((0.500, 0.140), (0.648, 0.100), (0.812, 0.166), (0.896, 0.330)), 8.0,
            (0.816, 0.076), "Right Transverse Sinus"),
    # The sagittal sinus is strictly midline, and drawing it that way bisected the circle of Willis
    # — a line through the middle of the arterial anatomy that belongs to a different system and
    # reads as though it connected to it. It is swept laterally instead, following the falx from
    # the vertex back to the torcula the way an oblique view shows it, so it sits *outside* the
    # arterial circle and *between* the straight sinus and the left transverse. That is a
    # legibility choice, not anatomy: this schematic is not registered to anything, and the
    # midline position carried no information the label does not already give.
    "sss": (((0.500, 0.958), (0.345, 0.912), (0.205, 0.790), (0.140, 0.610),
             (0.132, 0.430), (0.208, 0.252), (0.330, 0.168), (0.500, 0.140)), 10.5,
            (0.196, 0.716), "Superior Sagittal Sinus"),
}

#: Venous nodes are painted first so the arterial circle reads on top of the midline sinus.
_VENOUS_NODES: frozenset[str] = frozenset({"strs", "lts", "rts", "sss"})

#: Drawn behind the vessels as an anatomical frame; carries no value and is never coloured.
_SKULL_OUTLINE: tuple[tuple[float, float], ...] = (
    (0.500, 0.985), (0.760, 0.940), (0.930, 0.760), (0.968, 0.520),
    (0.930, 0.280), (0.780, 0.098), (0.500, 0.048), (0.220, 0.098),
    (0.070, 0.280), (0.032, 0.520), (0.070, 0.760), (0.240, 0.940), (0.500, 0.985),
)

#: Where the confluence of sinuses sits, annotated so the venous panel reads as a system.
_CONFLUENCE: tuple[float, float] = (0.500, 0.140)


#: Hemisphere-melted keys and the pair of drawn vessels each stands for. A frame grouped by
#: hemisphere reports ``ICA`` rather than ``LICA``/``RICA``, and that key canonicalizes to nothing —
#: which left the map able to draw only the three midline vessels. One averaged value belongs on
#: *both* sides, since that is what it was averaged from.
BILATERAL_KEYS: dict[str, tuple[str, str]] = {
    "ica": ("lica", "rica"),
    "mca": ("lmca", "rmca"),
    "aca": ("laca", "raca"),
    "pca": ("lpca", "rpca"),
    "va": ("lva", "rva"),
    "vertebral": ("lva", "rva"),
    "pcomm": ("lpcomm", "rpcomm"),
    "pcoma": ("lpcomm", "rpcomm"),
    "pcom": ("lpcomm", "rpcomm"),
    "ts": ("lts", "rts"),
    "tsv": ("lts", "rts"),
    "transverse": ("lts", "rts"),
    "transversesinus": ("lts", "rts"),
}


def nodes_for_label(label: Any) -> list[str]:
    """
    Drawn vessels a grouping level refers to — one, or **two** for a hemisphere-melted key.

    A vessel-wise frame gives ``LICA`` and the answer is one node. A hemisphere-wise frame gives
    ``ICA``: a single number averaged over the two carotids, which belongs on both of them. Painting
    it on neither — the previous behaviour, because ``ICA`` canonicalizes to nothing — left a
    hemisphere-grouped model with three drawable vessels out of seventeen.

    Returns ``[]`` for anything that is not a vessel at all.

    Examples
    --------
    >>> nodes_for_label("LICA"), nodes_for_label("ICA"), nodes_for_label("Basilar")
    (['lica'], ['lica', 'rica'], ['basi'])
    """
    from nvitk.stats.vessel_network import canonical_node

    node = canonical_node(label)
    if node in VESSEL_PATHS:
        return [node]
    key = re.sub(r"[^0-9a-z]+", "", str(label or "").strip().lower())
    pair = BILATERAL_KEYS.get(key)
    return [n for n in pair if n in VESSEL_PATHS] if pair else []


def _smooth_path(points: Sequence[tuple[float, float]]) -> Any:
    """
    A smooth :class:`~matplotlib.path.Path` through *points*.

    Quadratic Béziers anchored on the midpoints of successive control points — the standard way to
    get a C¹ curve through a polyline without fitting a spline. Two points degrade to a straight
    line, which is what a two-point vessel should be.
    """
    from matplotlib.path import Path

    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return Path(pts, [Path.MOVETO, *[Path.LINETO] * (len(pts) - 1)])

    verts: list[tuple[float, float]] = [pts[0]]
    codes: list[int] = [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        mid = ((pts[i][0] + pts[i + 1][0]) / 2.0, (pts[i][1] + pts[i + 1][1]) / 2.0)
        verts += [pts[i], mid]
        codes += [Path.CURVE3, Path.CURVE3]
    verts += [pts[-1], pts[-1]]
    codes += [Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------
def vascular_values_from_frame(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    pvalue_column: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Map a per-region result table onto canonical vessel nodes.

    Accepts whatever spelling the frame uses — ``LICA``, ``left_ica``, ``Left-ICA`` all resolve —
    through :func:`~nvitk.stats.vessel_network.canonical_node`. Rows that are not vessels (a lobe,
    a cortical parcel, an intercept term) are dropped silently: a mixed result table is the normal
    case, and only its vessel rows belong on this figure.

    Returns
    -------
    (values, pvalues)
        Both keyed by canonical node; *pvalues* is empty when *pvalue_column* is ``None``.
    """
    from nvitk.stats.vessel_network import canonical_node

    values: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    if frame is None or frame.empty or key_column not in frame.columns:
        return values, pvalues
    if value_column not in frame.columns:
        raise ValueError(
            f"{value_column!r} is not in the frame. Columns: {', '.join(map(str, frame.columns))}."
        )

    for _, row in frame.iterrows():
        nodes = nodes_for_label(row[key_column])
        if not nodes:
            continue
        value = pd.to_numeric(row[value_column], errors="coerce")
        if pd.isna(value):
            continue
        for node in nodes:
            values[node] = float(value)
        if pvalue_column and pvalue_column in frame.index.union(frame.columns):
            p = pd.to_numeric(row.get(pvalue_column), errors="coerce")
            if not pd.isna(p):
                for node in nodes:
                    pvalues[node] = float(p)
    return values, pvalues


def vascular_values_from_result(
    result: Any,
    *,
    group_column: str = "territory",
    source: str = "coefficient",
    data: pd.DataFrame | None = None,
    outcome: str = "",
    contrast: str = "",
) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Pull per-vessel numbers off a fitted model.

    Two sources, because they answer different questions and are read differently:

    ``coefficient``
        The fixed effect of each vessel level, read from the parameter table — patsy names them
        ``territory[T.LICA]``, so the level is recovered from the term. These are contrasts
        *against the reference level*, which is therefore absent from the map by construction: it
        is the zero everything else is measured from, not a vessel with no effect.

    ``mean``
        The observed mean of *outcome* within each vessel, from *data*. Not a model estimate at
        all — no covariate adjustment, no p-values — but it is what "estimated marginal means"
        reduces to for a model whose only term is the vessel, and it is the honest fallback when
        the fit has no per-vessel parameter to read.

    contrast : str
        A level of an interacting factor, as :func:`interaction_contrasts` names it (``sex[T.M]``).
        The vessel's value becomes its **simple effect at that level** — main effect plus the
        interaction — which is the profile actually seen in that group. Left empty, the map shows
        the main effect, i.e. the profile at the interacting factor's *reference* level; on a model
        with an interaction that is one group's answer presented as if it were everyone's.

        The p-value reported alongside stays the **interaction** term's: it tests whether this
        vessel's effect differs between the groups, which is the question a contrast view raises.
        A simple effect's own test needs a linear combination the coefficient table does not carry.

    Returns
    -------
    (values, pvalues, note)
        *note* describes what was extracted, for the figure subtitle.
    """

    source = str(source).strip()
    values: dict[str, float] = {}
    pvalues: dict[str, float] = {}

    if source.startswith("group:"):
        term = source.split(":", 1)[1] or "(Intercept)"
        return _values_from_group_coefficients(result, group_column=group_column, term=term)

    if source.lower() in {"emmeans", "emm"}:
        return _values_from_marginal_means(
            result, group_column=group_column, data=data
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
            for node in nodes_for_label(level):
                values[node] = float(value)
        return values, pvalues, f"observed mean {outcome} per vessel"

    # statsmodels exposes params/pvalues directly; the R engines (lme4, mmrm, lmrob) wrap an R
    # object whose table has to be normalized first. Trying the normalizer means one code path
    # serves every engine instead of the map silently working only for OLS and MixedLM.
    params, pvals = _coefficient_series(result)

    # Longest first so 'territory_id' is tried before 'territory' and cannot mis-split a term.
    prefixes = sorted(
        {str(c) for c in (data.columns if data is not None else [])} | {str(group_column)},
        key=len, reverse=True,
    )
    reference: str | None = None
    mirrored = False
    for term, coef in params.items():
        name = str(term)
        # patsy spells a categorical contrast 'col[T.level]', a no-intercept fit 'col[level]', and
        # lme4 concatenates without brackets ('territoryLICA'). The bracketed level is taken from
        # *any* term rather than only from ones naming ``group_column``: the plot's grouping column
        # is often 'group_key' while the model names the term 'territory', and requiring them to
        # match meant no parameter was ever recognized — every view silently fell back to the
        # observed means, which is why each one showed the same numbers.
        level = name
        if "[" in name and name.endswith("]"):
            level = name[name.index("[") + 1: -1]
            if level.startswith("T."):
                level = level[2:]
        else:
            # R concatenates factor and level without punctuation ('territoryLICA'), so the prefix
            # has to be stripped by name. Every column of the model frame is a candidate, not just
            # ``group_column``: the plot's grouping is often 'group_key' while the model term is
            # 'territory', and matching only the former left lmrob and lme4 resolving nothing.
            for prefix in prefixes:
                if name.startswith(prefix) and len(name) > len(prefix):
                    level = name[len(prefix):]
                    break
        nodes = nodes_for_label(level)
        if not nodes or not np.isfinite(coef):
            continue
        mirrored |= len(nodes) > 1
        for node in nodes:
            values[node] = float(coef)
        if pvals is not None and term in getattr(pvals, "index", []):
            p = pvals[term]
            if np.isfinite(p):
                for node in nodes:
                    pvalues[node] = float(p)

    # ---- Fold in the interaction, when one was asked for -------------------------------------
    if contrast:
        matched = 0
        for term, coef in params.items():
            parts = _term_parts(str(term))
            if len(parts) < 2 or contrast not in parts:
                continue
            vessel_part = next(
                (p for p in parts if nodes_for_label(_level_of(p, prefixes))), ""
            )
            nodes = nodes_for_label(_level_of(vessel_part, prefixes)) if vessel_part else []
            nodes = [n for n in nodes if n in values]
            if not nodes or not np.isfinite(coef):
                continue
            for node in nodes:
                values[node] += float(coef)
            matched += 1
            # The interaction's own p-value replaces the main effect's: it is the one that tests
            # the difference this view is about.
            if pvals is not None and term in getattr(pvals, "index", []):
                p = pvals[term]
                if np.isfinite(p):
                    for node in nodes:
                        pvalues[node] = float(p)
        if not matched:
            raise ValueError(
                f"No interaction term pairs a vessel with {contrast!r}. Available contrasts: "
                f"{', '.join(interaction_contrasts(result, group_column=group_column, data=data)) or 'none'}."
            )

    if not values:
        raise ValueError(
            f"No parameter of this model names a vessel level of {group_column!r}. Fit a model "
            f"with {group_column} as a term, or switch the source to the observed mean."
        )

    # Name the reference level so the reader knows why one vessel is blank.
    if data is not None and group_column in data.columns:
        levels = {n for v in data[group_column].astype(str).unique() for n in nodes_for_label(v)}
        missing = sorted(levels - set(values))
        reference = missing[0] if len(missing) == 1 else None

    note = f"{group_column} coefficients"
    if mirrored:
        note += ", mirrored across hemispheres"
    if contrast:
        note += f" at {contrast}"
    if reference:
        note += f" (reference: {reference})"
    return values, pvalues, note


def _term_parts(name: str) -> list[str]:
    """Split a patsy/R interaction term into its factors (``a[T.x]:b[T.y]`` → two parts)."""
    return [part for part in str(name).split(":") if part]


def interaction_contrasts(
    result: Any, *, group_column: str = "territory", data: pd.DataFrame | None = None
) -> list[str]:
    """
    Levels of the *other* factor in interactions with the vessel term, e.g. ``["sex[T.M]"]``.

    A model with ``territory * sex`` estimates a different vessel profile for each sex, and the
    coefficient table holds them as ``territory[T.LICA]:sex[T.M]``. Without naming which side of
    the interaction to draw, the map would show only the main effect — the profile at the *other*
    factor's reference level — and quietly present it as the whole story.

    Empty when the model has no interaction on the vessel term, which is the signal a caller needs
    to hide the selector rather than offer one with nothing in it.
    """

    try:
        params = _coefficient_series(result)[0]
    except Exception:
        return []
    prefixes = sorted(
        {str(c) for c in (data.columns if data is not None else [])} | {str(group_column)},
        key=len, reverse=True,
    )
    found: list[str] = []
    for term in params.index:
        parts = _term_parts(str(term))
        if len(parts) < 2:
            continue
        vessel_parts = [p for p in parts if nodes_for_label(_level_of(p, prefixes))]
        if not vessel_parts:
            continue
        for other in (p for p in parts if p not in vessel_parts):
            if other not in found:
                found.append(other)
    return found


def _level_of(part: str, prefixes: Sequence[str]) -> str:
    """The factor level a single (non-interaction) term names."""
    name = str(part)
    if "[" in name and name.endswith("]"):
        level = name[name.index("[") + 1: -1]
        return level[2:] if level.startswith("T.") else level
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return name


#: Per-engine normalizers to a ``parameter / coef / p_value`` table, tried in turn. statsmodels
#: exposes ``params``/``pvalues`` directly; every R engine wraps an object that does not, and each
#: wraps it differently — an ``lmrob`` fit *is* the R object, a pymer4 model holds one, MMRM another.
#: Trying only one of them is what left lmrob and MMRM silently falling back to observed means.
_COEF_NORMALIZERS: tuple[tuple[str, str], ...] = (
    ("nvitk.stats.r_mixedlm", "lme4_coef_frame"),
    ("nvitk.stats.r_robust", "lmrob_coef_frame"),
    ("nvitk.stats.r_mmrm", "mmrm_coef_frame"),
    ("nvitk.stats.r_gam", "mrf_coef_frame"),
)


def _coefficient_series(result: Any) -> tuple[pd.Series, pd.Series | None]:
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


def _values_from_marginal_means(
    result: Any, *, group_column: str, data: pd.DataFrame | None
) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Model-predicted mean per vessel, on the outcome's own scale — **including the reference level**.

    Treatment coding gives the reference territory no coefficient: it is absorbed into the
    intercept, which is why a ``~ territory + …`` fit shows sixteen vessels and leaves the basilar
    blank. That is correct for a contrast table and wrong for a map, where the reader is looking at
    anatomy and one artery is simply missing.

    The marginal mean rebuilds it: ``intercept + β_vessel`` (with ``β = 0`` for the reference),
    plus each numeric covariate held at its mean. Categorical covariates stay at their own reference
    level, so the result is "predicted flow in an average subject", comparable across vessels and
    interpretable in mL/min rather than as a difference from whichever vessel sorted first.

    No p-values come back. A coefficient's p-value tests the *contrast against the reference*, not
    whether a marginal mean differs from zero, and carrying it here would attach a test to a number
    it does not describe. Use the coefficient view when significance is the question.
    """

    params, _ = _coefficient_series(result)
    prefixes = sorted(
        {str(c) for c in (data.columns if data is not None else [])} | {str(group_column)},
        key=len, reverse=True,
    )

    intercept = 0.0
    for name in ("Intercept", "(Intercept)", "const"):
        if name in params.index:
            intercept = float(params[name])
            break

    # Covariate offset: numeric terms at their mean, everything else at its reference. Shared by
    # every vessel, so it shifts the map without changing the ordering — but it is what puts the
    # numbers on the outcome's scale instead of an arbitrary one.
    offset = 0.0
    if data is not None:
        for term, coef in params.items():
            name = str(term)
            if name in {"Intercept", "(Intercept)", "const"} or ":" in name:
                continue
            if nodes_for_label(_level_of(name, prefixes)):
                continue
            if name in data.columns and pd.api.types.is_numeric_dtype(data[name]):
                column = pd.to_numeric(data[name], errors="coerce")
                if column.notna().any() and np.isfinite(coef):
                    offset += float(coef) * float(column.mean())

    values: dict[str, float] = {}
    for term, coef in params.items():
        nodes = nodes_for_label(_level_of(str(term), prefixes))
        if nodes and np.isfinite(coef) and ":" not in str(term):
            for node in nodes:
                values[node] = intercept + offset + float(coef)

    # The reference level has no term of its own; its marginal mean is the intercept.
    reference = None
    if data is not None and group_column in data.columns:
        levels = {
            n for v in data[group_column].astype(str).dropna().unique()
            for n in nodes_for_label(v)
        }
        missing = sorted(levels - set(values))
        for node in missing:
            values[node] = intercept + offset
        reference = missing[0] if len(missing) == 1 else None

    if not values:
        raise ValueError(
            f"No parameter of this model names a vessel level of {group_column!r}, so there are no "
            f"marginal means to draw."
        )
    note = "marginal mean per vessel"
    if reference:
        note += f" ({reference} recovered from the intercept)"
    return values, {}, note


def _values_from_group_coefficients(
    result: Any, *, group_column: str, term: str
) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Per-vessel values from an ``lme4`` random-effects structure.

    A model like ``flow_mean ~ age_c + (1 + age_c | territory)`` puts nothing about individual
    vessels in its *fixed* effects — the vessel information is the random structure, one intercept
    and one ``age_c`` slope per territory. ``coef()`` totals are used rather than ``ranef()``
    deviations because a total is the quantity with a physical reading: the age slope *in that
    vessel*, not its departure from the average slope.

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

    values: dict[str, float] = {}
    for level, value in zip(rows["level"], rows[term]):
        if not np.isfinite(value):
            continue
        for node in nodes_for_label(level):
            values[node] = float(value)
    if not values:
        raise ValueError(
            f"None of {group_column!r}'s levels resolve to a drawn vessel."
        )
    label = "intercept" if term.strip("()").lower() == "intercept" else f"{term} slope"
    return values, {}, f"per-vessel {label} ({group_column} random effects)"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_vascular_map(
    values: Mapping[str, float],
    *,
    pvalues: Mapping[str, float] | None = None,
    mode: str = "estimate",
    alpha: float = DEFAULT_ALPHA,
    mask_nonsignificant: bool = True,
    cmap: str | None = None,
    center: float | None = None,
    label: str = "",
    title: str = "",
    annotate: bool = True,
    hide: Sequence[str] = (),
    ax: Any = None,
):
    """
    Draw *values* on the cerebral vasculature.

    Parameters
    ----------
    values : mapping
        ``{canonical node: value}``. Nodes absent from the mapping are drawn as empty outlines.
    pvalues : mapping, optional
        ``{canonical node: p}``. Required for *mode* ``"pvalue"`` and for significance masking.
    mode : {"estimate", "pvalue"}
        What colour encodes. ``"estimate"`` maps the value through a diverging or sequential
        colormap; ``"pvalue"`` maps −log₁₀(p), which spreads the small p-values that matter instead
        of compressing them all against zero.
    mask_nonsignificant : bool
        In ``"estimate"`` mode, draw vessels with ``p >= alpha`` in grey. Has no effect without
        *pvalues* — and is skipped with a warning rather than silently colouring everything, since
        an unmasked map looks identical to one where everything is significant.
    center : float, optional
        Value the diverging colormap centres on. Defaults to 0 when the values straddle it (an
        effect), and to no centring when they do not (a mean).
    annotate : bool
        Print each vessel's value beside it.
    hide : sequence of str
        Vessels to leave off the figure entirely — not drawn, not labelled, and **not counted in
        the colour scale**. That last part is the point: hiding the one vessel that dominates the
        range is how you see the structure among the rest, which a figure that merely blanked it
        would not give you. Distinct from a vessel with no estimate, which is drawn as an outline
        because "not measured" is information.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import PathPatch

    mode = str(mode).strip().lower()
    if mode not in {"estimate", "pvalue"}:
        raise ValueError(f"mode must be 'estimate' or 'pvalue', not {mode!r}.")

    hidden = {str(h) for h in hide}
    values = {
        str(k): float(v) for k, v in dict(values or {}).items()
        if np.isfinite(v) and str(k) not in hidden
    }
    pvalues = {
        str(k): float(v) for k, v in dict(pvalues or {}).items()
        if np.isfinite(v) and str(k) not in hidden
    }
    if mode == "pvalue" and not pvalues:
        raise ValueError("mode='pvalue' needs a pvalues mapping.")
    if mask_nonsignificant and not pvalues:
        log.warning(
            "mask_nonsignificant was requested without p-values — every vessel will be drawn as "
            "though it were significant."
        )
        mask_nonsignificant = False

    unknown = sorted(set(values) - set(VESSEL_PATHS))
    if unknown:
        log.info("Vascular map: %s have no drawn geometry and are omitted.", ", ".join(unknown))

    # ---- Colour scale ---------------------------------------------------------
    if mode == "pvalue":
        scalars = {k: -np.log10(max(p, 1e-300)) for k, p in pvalues.items() if k in VESSEL_PATHS}
        colormap = plt.get_cmap(cmap or "viridis")
        finite = list(scalars.values())
        # With the mask on, nothing below the threshold is drawn, so starting the scale at zero
        # would spend most of the colormap on vessels that are grey. Anchoring vmin at the
        # threshold gives the drawn vessels the whole range.
        floor = -np.log10(alpha) if mask_nonsignificant else 0.0
        norm = Normalize(vmin=floor, vmax=max([*finite, floor + 1.0]))
        bar_label = label or "−log₁₀(p)"
    else:
        scalars = {k: v for k, v in values.items() if k in VESSEL_PATHS}
        finite = list(scalars.values())
        if not finite:
            raise ValueError("No value maps onto a drawn vessel — nothing to plot.")
        lo, hi = min(finite), max(finite)
        diverging = center is not None or (lo < 0.0 < hi)
        if diverging:
            pivot = 0.0 if center is None else float(center)
            span = max(abs(lo - pivot), abs(hi - pivot)) or 1.0
            norm = TwoSlopeNorm(vmin=pivot - span, vcenter=pivot, vmax=pivot + span)
            colormap = plt.get_cmap(cmap or "RdBu_r")
        else:
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
            colormap = plt.get_cmap(cmap or "viridis")
        bar_label = label or "estimate"

    # ---- Canvas ---------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 8.0))
    else:
        fig = ax.figure

    outline = _smooth_path(_SKULL_OUTLINE)
    ax.add_patch(PathPatch(outline, facecolor="none", edgecolor=COLOR_OUTLINE, lw=10.0,
                           capstyle="round", joinstyle="round", zorder=1))

    # ---- Vessels --------------------------------------------------------------
    ordered = [n for n in VESSEL_PATHS if n in _VENOUS_NODES]
    ordered += [n for n in VESSEL_PATHS if n not in _VENOUS_NODES]
    for depth, node in enumerate(ordered):
        if node in hidden:
            continue
        points, width, anchor, name = VESSEL_PATHS[node]
        base = 2 + depth * 2
        path = _smooth_path(points)
        has_value = node in scalars
        significant = (not mask_nonsignificant) or (pvalues.get(node, np.inf) < alpha)

        if not has_value:
            colour, edge = "none", COLOR_ABSENT
        elif not significant:
            # Applies in both modes. Greying the non-significant vessels of a p-value map leaves
            # the colour reading only.
            colour, edge = COLOR_NONSIGNIFICANT, COLOR_NONSIGNIFICANT
        else:
            colour = colormap(norm(scalars[node]))
            edge = colour

        # Stroke twice: a dark casing under the coloured core, so adjacent vessels stay separable
        # when their values are close.
        # The casing is dark under a valued vessel, to separate neighbours whose colours are
        # close, and light under an absent one — a dark casing shows through the dashes and turns
        # "no estimate" into a striped bar that reads as emphasis rather than absence.
        casing = "#2b2b2b" if has_value else "#e2e2e2"
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=casing,
                               lw=width + 2.2, capstyle="round", joinstyle="round", zorder=base))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=edge, lw=width,
                               capstyle="round", joinstyle="round", zorder=base + 1,
                               linestyle="-" if has_value else (0, (2, 2))))

        ax.annotate(
            name, xy=anchor, ha="center", va="center", fontsize=7.5,
            color="#1a1a1a", zorder=90,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
        )
        if annotate and has_value:
            text = f"{scalars[node]:.3g}" if mode == "estimate" else f"p={pvalues[node]:.3g}"
            ax.annotate(
                text, xy=(anchor[0], anchor[1] - 0.032), ha="center", va="center",
                fontsize=6.8, color="#333333", zorder=90,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.62),
            )

    # The confluence is where the sinuses meet, so it only means anything while one is drawn.
    if not _VENOUS_NODES <= hidden:
        ax.annotate("Confluence", xy=(_CONFLUENCE[0], _CONFLUENCE[1] - 0.062), ha="center",
                    va="center", fontsize=7.5, color="#1a1a1a", zorder=90)

    # ---- Frame ----------------------------------------------------------------
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    bar = fig.colorbar(mappable, ax=ax, fraction=0.040, pad=0.02)
    bar.set_label(bar_label, fontsize=9)
    if mode == "pvalue":
        line = -np.log10(alpha)
        bar.ax.axhline(line, color="#c44e52", lw=1.4)
        bar.ax.annotate(f"p={alpha:g}", xy=(1.6, line), xycoords=("axes fraction", "data"),
                        fontsize=7, color="#c44e52", va="center")

    notes = []
    if mask_nonsignificant:
        notes.append(f"grey = p ≥ {alpha:g}")
    if hidden:
        notes.append(f"{len(hidden)} vessel(s) hidden")
    if len(scalars) + len(hidden) < len(VESSEL_PATHS):
        notes.append("dashed outline = no estimate")
    if notes:
        ax.annotate(
            "  ·  ".join(notes), xy=(0.5, -0.01), xycoords="axes fraction",
            ha="center", va="top", fontsize=8, color="#555555",
        )

    fig.tight_layout()
    return fig


__all__ = [
    "COLOR_ABSENT",
    "COLOR_NONSIGNIFICANT",
    "DEFAULT_ALPHA",
    "VESSEL_PATHS",
    "plot_vascular_map",
    "vascular_values_from_frame",
    "vascular_values_from_result",
    "group_coefficient_terms",
    "interaction_contrasts",
]
