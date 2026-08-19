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
from nvitk.stats import _model_values

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


#: ASL parcels carry the smoothing kernel as a trailing ``_0`` / ``_8`` / ``_12``. Their presence is
#: what distinguishes a *perfusion territory* from the artery that feeds it.
_ASL_TERRITORY_RE = re.compile(r"_(?:0|8|12)$")


def is_perfusion_territory(label: Any) -> bool:
    """
    Whether *label* names an ASL perfusion territory rather than a vessel.

    ``left_mca_8`` is the parenchyma the left MCA supplies, measured in mL/100 g/min. ``LMCA`` is
    the artery itself, measured in mL/min. They share a name and are not the same structure, and the
    trailing smoothing kernel is the only thing in the published id that tells them apart.
    """
    return bool(_ASL_TERRITORY_RE.search(re.sub(r"[^0-9a-z]+", "_", str(label or "").strip().lower())))


def unmapped_vessel_labels(labels: Sequence[Any]) -> list[str]:
    """
    Which of *labels* this schematic cannot draw — for telling the user *why* it would be empty.

    Mirrors :func:`nvitk.stats.brain_map.regions_without_geometry`, and exists for the same reason:
    an empty figure is a worse answer than a sentence naming the levels that did not resolve.
    """
    return [str(label) for label in labels if not nodes_for_label(label)]


def nodes_for_label(label: Any) -> list[str]:
    """
    Drawn vessels a grouping level refers to — one, or **two** for a hemisphere-melted key.

    A vessel-wise frame gives ``LICA`` and the answer is one node. A hemisphere-wise frame gives
    ``ICA``: a single number averaged over the two carotids, which belongs on both of them. Painting
    it on neither — the previous behaviour, because ``ICA`` canonicalizes to nothing — left a
    hemisphere-grouped model with three drawable vessels out of seventeen.

    Returns ``[]`` for anything that is not a vessel at all — **including an ASL perfusion
    territory**. ``left_mca_8`` canonicalizes to ``lmca`` because the two share an anatomical name,
    but a perfusion value belongs on the parenchyma that artery supplies, not on the artery: drawing
    mL/100 g/min along a conduit measured in mL/min states something the data does not say. Those
    levels belong on the brain map's ``vascular`` atlas, which is that territory parcellation.

    Examples
    --------
    >>> nodes_for_label("LICA"), nodes_for_label("ICA"), nodes_for_label("Basilar")
    (['lica'], ['lica', 'rica'], ['basi'])
    >>> nodes_for_label("left_mca_8")
    []
    """
    from nvitk.stats.vessel_network import canonical_node

    if is_perfusion_territory(label):
        return []
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
    through :func:`nodes_for_label`. Rows that are not vessels (a lobe, a cortical parcel, an
    intercept term) are dropped silently: a mixed result table is the normal case, and only its
    vessel rows belong on this figure.

    Returns
    -------
    (values, pvalues)
        Both keyed by canonical node; *pvalues* is empty when *pvalue_column* is ``None``.
    """
    values, pvalues = _model_values.values_from_frame(
        frame,
        resolver=nodes_for_label,
        key_column=key_column,
        value_column=value_column,
        pvalue_column=pvalue_column,
    )
    return {str(k): v for k, v in values.items()}, {str(k): v for k, v in pvalues.items()}


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

    values, pvalues, note = _model_values.values_from_result(
        result,
        resolver=nodes_for_label,
        group_column=group_column,
        source=source,
        data=data,
        outcome=outcome,
        contrast=contrast,
        unit="vessel",
    )
    # ``mirrored across members`` is the generic phrasing; here the members are the two sides.
    note = note.replace("mirrored across members", "mirrored across hemispheres")
    return (
        {str(k): v for k, v in values.items()},
        {str(k): v for k, v in pvalues.items()},
        note,
    )


def _term_parts(name: str) -> list[str]:
    """Split a patsy/R interaction term into its factors (``a[T.x]:b[T.y]`` → two parts)."""
    return _model_values.term_parts(name)


def _level_of(part: str, prefixes: Sequence[str]) -> str:
    """The factor level a single (non-interaction) term names."""
    return _model_values.level_of(part, prefixes)


def _coefficient_series(result: Any) -> tuple[pd.Series, pd.Series | None]:
    """``(params, pvalues)`` as pandas Series, whichever engine produced *result*."""
    return _model_values.coefficient_series(result)


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
    return _model_values.interaction_contrasts(
        result, resolver=nodes_for_label, group_column=group_column, data=data
    )


def group_coefficient_terms(result: Any, *, group_column: str = "territory") -> list[str]:
    """
    Per-group terms this fit estimates for *group_column*, e.g. ``["(Intercept)", "age_c"]``.

    Empty when the model has no random structure over that factor — which is the signal a caller
    needs to fall back to the fixed effects instead of offering a menu with nothing in it.
    """
    return _model_values.group_coefficient_terms(result, group_column=group_column)


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
    vmin: float | None = None,
    vmax: float | None = None,
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
    vmin, vmax : float, optional
        Pin the colour scale instead of taking it from these values. Needed whenever several
        figures have to be *compared* — the frames of a cardiac animation above all, where a scale
        refitted per frame animates the scale rather than the flow, and systole and diastole come
        out looking identical because each saturates its own range.
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
        lo = min(finite) if vmin is None else float(vmin)
        hi = max(finite) if vmax is None else float(vmax)
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
    "is_perfusion_territory",
    "plot_vascular_map",
    "unmapped_vessel_labels",
    "vascular_values_from_frame",
    "vascular_values_from_result",
    "group_coefficient_terms",
    "interaction_contrasts",
]
