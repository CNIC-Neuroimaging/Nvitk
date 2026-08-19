"""
Model estimates painted on the cortical surface.

Description
-----------
The parenchymal counterpart of :mod:`nvitk.stats.vascular_map`. That module answers "which vessel"
on a schematic of the circulation; this one answers "where in the cortex" on an actual
parcellation — ASL perfusion (CBF / ATT) and T1 volumetry are measured per Desikan parcel, and a
forest plot of sixty-eight coefficients is a list that hides every spatial pattern in it.

It is a *view of a fit*, not a new analysis: the numbers are whatever the caller passes in —
estimated marginal means, model coefficients, per-subject measurements, QC scores.

Atlases
-------
``desikan``
    The Desikan–Killiany cortical parcellation the ASL and T1 pipelines report against. Located
    rather than fetched — see :mod:`nvitk.viz.atlas_sources`.
``vascular``
    The arterial-territory / watershed atlas the ASL pipeline uses by default. Same figure, coarser
    parcels, and the one that makes an ASL result comparable to a 4D-flow one.

Aggregate levels
----------------
A frame grouped by hemisphere reports ``precuneus``, a lobe-grouped one ``ctx-Left-Frontal-Lobe``:
one number averaged over several parcels. It is painted on **every parcel it was averaged from**,
because that is what it describes. Painting it on none — which an exact-match lookup does — leaves a
lobe-grouped fit with an empty brain. Same rule, and the same reason, as
:func:`nvitk.stats.vascular_map.nodes_for_label` mirroring a hemisphere-melted ``ICA`` onto both
carotids.

Significance
------------
Three states, and they must not look alike:

============================  =====================================================
Coloured by the colormap      an estimate that reached *alpha*
Flat grey                     measured, but did not reach *alpha*
Not painted (background)      no estimate for that parcel at all
============================  =====================================================

The last row is the one that is easy to get wrong. A parcel the model has nothing to say about is
not the same as one it says nothing significant about, and a figure that draws them identically
invites the reader to count the second as evidence of the first.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from functools import lru_cache
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.stats import _model_values
from nvitk.stats.vascular_map import (
    COLOR_ABSENT,
    COLOR_NONSIGNIFICANT,
    DEFAULT_ALPHA,
)

log = Logger()

#: Atlases the brain map can draw on, as ``(label, key)`` for a GUI picker.
BRAIN_ATLASES: tuple[tuple[str, str], ...] = (
    ("Desikan (cortical parcels)", "desikan"),
    ("Desikan lobes", "desikan_lobes"),
    ("Vascular territories / watershed", "vascular"),
)

#: Surface the parcels are painted on. Inflated shows every sulcal parcel at once, which is why it
#: is the default; pial is the real geometry and reads as a brain but hides whatever is in a sulcus;
#: flat shows the whole cortex in one panel with no view to choose.
#: Pial first because it is the default: it is the real cortical geometry, and with the curvature
#: blended through the parcel colours the folding stays readable, which is what made the inflated
#: surface worth defaulting to in the first place.
BRAIN_SURFACES: tuple[tuple[str, str], ...] = (
    ("pial (folded)", "pial"),
    ("inflated", "infl"),
    ("flat", "flat"),
)

#: Surface views, as ``(label, key)``. Lateral and medial together show the whole cortex; a single
#: view hides half of it, and the medial half is where the cingulate and precuneus live.
BRAIN_VIEWS: tuple[tuple[str, str], ...] = (
    ("lateral + medial", "lateral,medial"),
    ("lateral", "lateral"),
    ("medial", "medial"),
    ("dorsal", "dorsal"),
    ("ventral", "ventral"),
    ("anterior", "anterior"),
    ("posterior", "posterior"),
    ("all four", "lateral,medial,dorsal,ventral"),
)

#: How much of the colour range below *vmin* the non-significant sentinel sits at. Any strictly
#: negative offset triggers the colormap's ``set_under``; a full range keeps it clear of rounding.
_UNDER_MARGIN: float = 1.0


# ---------------------------------------------------------------------------
# Atlas resolution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _resolved_atlas(atlas: str, atlas_kwargs: tuple[tuple[str, Any], ...]) -> Any:
    """
    Cached :class:`~nvitk.viz.brainshow.ResolvedAtlas` for *atlas*.

    Cached because resolution reads files off disk (two ``.annot``\\ s, or a NIfTI), and the GUI
    re-resolves on every view change — flipping from lateral to medial should not re-read the atlas.
    """
    from nvitk.viz.brainshow import resolve_atlas

    return resolve_atlas(atlas, dict(atlas_kwargs))


def parcel_resolver(
    atlas: str = "desikan", atlas_kwargs: Mapping[str, Any] | None = None
) -> tuple[Any, Any]:
    """
    ``(resolver, resolved_atlas)`` for *atlas*.

    The resolver maps a published region label onto the atlas indices it refers to — see
    :func:`nvitk.viz.brainshow.atlas_indices_for_region`. It is the only thing
    :mod:`nvitk.stats._model_values` needs to know about anatomy.
    """
    return _resolver_and_atlas(str(atlas), tuple(sorted(dict(atlas_kwargs or {}).items())))


@lru_cache(maxsize=8)
def _resolver_and_atlas(atlas: str, atlas_kwargs: tuple[tuple[str, Any], ...]) -> tuple[Any, Any]:
    """
    Cached ``(resolver, resolved atlas)`` for one atlas spec.

    Both halves are worth caching and both are keyed on the *spec* rather than on the resolved
    atlas: resolution reads files off disk, the resolver's inverse index is otherwise rebuilt once
    per model term, and ``ResolvedAtlas`` carries a dict and two arrays so it cannot be a cache key
    itself.
    """
    from nvitk.viz.brainshow import region_index_resolver

    resolved = _resolved_atlas(atlas, atlas_kwargs)
    return region_index_resolver(resolved.index_to_label), resolved


def regions_without_geometry(
    labels: Sequence[Any], *, atlas: str = "desikan", atlas_kwargs: Mapping[str, Any] | None = None
) -> list[str]:
    """
    Which of *labels* this atlas cannot draw — for telling the user *why* the map would be empty.

    A FLAIR frame is the motivating case: its regions are white-matter zones (``frperiv``,
    ``juxta``, ``bgit``), not cortical parcels, so none of them resolve and the honest answer is
    "this measurement is not parcellated by this atlas", not a blank brain.
    """
    resolve, _ = parcel_resolver(atlas, atlas_kwargs)
    return [str(label) for label in labels if not resolve(label)]


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------
def parcel_values_from_frame(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    pvalue_column: str | None = None,
    atlas: str = "desikan",
    atlas_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Map a per-region table onto atlas indices.

    Accepts whatever spelling the frame uses (``ctx-lh-precuneus``, ``ctx_lh_precuneus``,
    ``left_precuneus``). Rows that are not parcels of this atlas — a vessel, a whole-head scalar
    such as ``etiv``, an intercept term — are dropped silently: a mixed result table is the normal
    case, and only the rows this figure can draw belong on it.
    """
    resolve, _ = parcel_resolver(atlas, atlas_kwargs)
    values, pvalues = _model_values.values_from_frame(
        frame,
        resolver=resolve,
        key_column=key_column,
        value_column=value_column,
        pvalue_column=pvalue_column,
    )
    return {int(k): v for k, v in values.items()}, {int(k): v for k, v in pvalues.items()}


def parcel_values_from_result(
    result: Any,
    *,
    group_column: str = "territory",
    source: str = "coefficient",
    data: pd.DataFrame | None = None,
    outcome: str = "",
    contrast: str = "",
    atlas: str = "desikan",
    atlas_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[int, float], dict[int, float], str]:
    """
    Pull per-parcel numbers off a fitted model.

    Sources and their meanings are shared with the vascular map — see
    :func:`nvitk.stats._model_values.values_from_result` for what ``coefficient``, ``emmeans``,
    ``mean`` and ``group:<term>`` each report and why they are read differently.

    Returns
    -------
    (values, pvalues, note)
        Keyed by atlas index; *note* describes what was extracted, for the figure subtitle.
    """
    resolve, resolved = parcel_resolver(atlas, atlas_kwargs)
    values, pvalues, note = _model_values.values_from_result(
        result,
        resolver=resolve,
        group_column=group_column,
        source=source,
        data=data,
        outcome=outcome,
        contrast=contrast,
        unit="parcel",
    )
    note = note.replace("mirrored across members", "spread over the parcels it averages")
    return (
        {int(k): v for k, v in values.items()},
        {int(k): v for k, v in pvalues.items()},
        _label_indices_in_note(note, resolved.index_to_label),
    )


def _label_indices_in_note(note: str, index_to_label: Mapping[int, str]) -> str:
    """
    Replace bare atlas indices in a note with the parcel names they stand for.

    The shared extractor names the reference level by whatever it keys values on, which here is an
    integer — ``(reference: 1025)`` says nothing to a reader looking at a cortex. This is cosmetic
    and deliberately narrow: only the two phrases the extractor emits are rewritten.
    """
    import re

    def _swap(match: "re.Match[str]") -> str:
        label = index_to_label.get(int(match.group(2)))
        return f"{match.group(1)}{label}" if label else match.group(0)

    note = re.sub(r"(reference: )(\d+)", _swap, note)
    return re.sub(r"()(\d+)(?= recovered from the intercept)", _swap, note)


def brain_interaction_contrasts(
    result: Any,
    *,
    group_column: str = "territory",
    data: pd.DataFrame | None = None,
    atlas: str = "desikan",
    atlas_kwargs: Mapping[str, Any] | None = None,
) -> list[str]:
    """Levels of the *other* factor in interactions with the parcel term, e.g. ``["sex[T.M]"]``."""
    try:
        resolve, _ = parcel_resolver(atlas, atlas_kwargs)
    except Exception as exc:
        log.debug("Contrast discovery: atlas %r unavailable (%s).", atlas, exc)
        return []
    return _model_values.interaction_contrasts(
        result, resolver=resolve, group_column=group_column, data=data
    )


def brain_group_coefficient_terms(result: Any, *, group_column: str = "territory") -> list[str]:
    """Per-group terms this fit estimates for *group_column* (an ``lme4`` random structure)."""
    return _model_values.group_coefficient_terms(result, group_column=group_column)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _vertex_textures(
    resolved: Any, index_to_scalar: Mapping[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(left, right)`` per-vertex textures, ``NaN`` wherever no value was supplied.

    Volumetric atlases are painted into a stat image and projected with nearest-neighbour
    interpolation, so a parcel's exact value survives the projection rather than being blended with
    its neighbours' — a mean of two coefficients is not a coefficient.
    """
    from nilearn import surface

    from nvitk.viz.brainshow import build_volume_stat_image

    if resolved.flavor == "surface_fs_vertex":
        return (
            _paint_vertices(resolved.map_left, index_to_scalar),
            _paint_vertices(resolved.map_right, index_to_scalar),
        )

    stat_img = build_volume_stat_image(resolved.atlas_img, index_to_scalar, hemisphere="both")
    meshes = fsaverage_meshes(resolved.surf_mesh)
    return tuple(
        np.asarray(
            surface.vol_to_surf(stat_img, meshes[f"pial_{side}"], interpolation="nearest", radius=2)
        )
        for side in ("left", "right")
    )


@lru_cache(maxsize=4)
def fsaverage_meshes(mesh: str = "fsaverage5"):
    """
    Cached ``fetch_surf_fsaverage`` result.

    The fetch re-checks the download cache on every call, and the brain map asks for it once per
    draw *and* once per texture build — several times a second while a user drags a control.
    """
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage(mesh=str(mesh))


def _paint_vertices(map_arr: np.ndarray, index_to_scalar: Mapping[int, float]) -> np.ndarray:
    """
    Per-vertex texture from a parcel map, via a lookup table rather than a scan per region.

    The obvious loop — ``tex[map_arr == index] = value`` — walks all 10k vertices once *per region*,
    so a 68-parcel atlas reads the map 68 times to write it once. Indexing a table built over the
    label range does it in a single pass.
    """
    source = np.asarray(map_arr)
    if source.size == 0:
        return np.full(source.shape, np.nan, dtype=float)
    top = int(source.max(initial=0))
    table = np.full(top + 2, np.nan, dtype=float)
    for index, value in index_to_scalar.items():
        if 0 <= int(index) <= top:
            table[int(index)] = float(value)
    # ``-1`` marks "no parcel" in an annot, and negatives must not wrap around the table.
    safe = np.where(source >= 0, source, top + 1).astype(np.int64)
    return table[np.minimum(safe, top + 1)]


def _view_keys(views: Any) -> list[str]:
    """Normalize the *views* argument into a list of nilearn view names."""
    if isinstance(views, str):
        return [v.strip() for v in views.split(",") if v.strip()]
    return [str(v).strip() for v in (views or ()) if str(v).strip()] or ["lateral"]



def _scale_notes(
    mask_nonsignificant: bool,
    alpha: float,
    n_scalars: int,
    n_significant: int,
    hidden: set,
    threshold: Any,
    n_below: int,
    blended: bool,
    opacity: float,
    n_known: int,
    n_drawn: int,
    n_hidden_known: int,
) -> list[str]:
    """
    The caption fragments describing what the colours do and do not mean.

    Shared by the static and interactive renderers so the two never disagree about how many parcels
    were greyed, hidden or left unpainted — a figure and its rotatable twin telling different
    stories about the same fit would be worse than either alone.
    """
    notes: list[str] = []
    if mask_nonsignificant:
        notes.append(f"grey = p ≥ {alpha:g} ({n_scalars - n_significant} parcel(s))")
    if hidden:
        notes.append(f"{len(hidden)} parcel(s) hidden")
    if n_below:
        notes.append(f"|value| < {float(threshold):g} hidden ({n_below} parcel(s))")
    # Both of these shift the rendered colour away from the colourbar, so the figure has to say so.
    if blended:
        notes.append("colours blended with curvature")
    if float(opacity) < 1.0:
        notes.append(f"opacity {float(opacity):.0%}")
    undrawn = n_known - n_drawn - n_hidden_known
    if undrawn > 0:
        notes.append(f"unpainted = no estimate ({undrawn} parcel(s))")
    return notes


def _interactive_surface(
    meshes: Any,
    sides: Sequence[str],
    *,
    surface: str,
    left: np.ndarray,
    right: np.ndarray,
    colormap: Any,
    norm: Any,
    shading: bool,
    blend: bool,
    opacity: float,
    title: str,
    label: str,
    notes: Sequence[str],
) -> Any:
    """
    Both hemispheres as one rotatable Plotly scene.

    The vertex colours are computed here rather than by nilearn's Plotly engine, and that is not a
    stylistic choice: that engine **clips** the texture into ``[vmin, vmax]`` before applying the
    colormap, so the below-range sentinel that marks a non-significant parcel comes out as the
    colormap's most extreme colour. A parcel the model found *no evidence for* would render as the
    strongest negative result on the map — the exact misreading this module's three-state encoding
    exists to prevent.

    Mapping ``colormap(norm(texture))`` directly keeps ``set_under`` (non-significant grey) and
    ``set_bad`` (no estimate) intact, because that is the same call the static figure makes.

    fsaverage's two hemispheres already share a coordinate space, so the meshes are concatenated
    with no transform. A flat surface is promoted to pial: a flat map is a 2-D projection with
    nothing to rotate.
    """
    import plotly.graph_objects as go
    from matplotlib.colors import to_hex
    from nilearn import surface as nl_surface

    if surface == "flat":
        log.info("A flat map has no 3-D structure to rotate — using the pial surface instead.")
        surface = "pial"

    traces: list[Any] = []
    for side in sides:
        coords, faces = nl_surface.load_surf_mesh(meshes[f"{surface}_{side}"])
        texture = np.asarray(left if side == "left" else right, dtype=float)
        rgba = colormap(norm(np.ma.masked_invalid(texture)))
        rgba = np.array(rgba, dtype=float, copy=True)

        if shading:
            background = meshes.get(f"sulc_{side}") or meshes.get(f"curv_{side}")
            curvature = np.asarray(nl_surface.load_surf_data(background), dtype=float)
            # Curvature sign is all a reader needs: two tones for gyrus and sulcus. Scaling by its
            # magnitude makes the shading fight the data colours for attention.
            tone = np.where(curvature > 0, 0.72, 0.94)
            painted = np.isfinite(texture)
            # Unpainted vertices always take the shading — that *is* the background. Painted ones
            # only when blending was asked for.
            shade = tone if blend else np.where(painted, 1.0, tone)
            rgba[:, :3] *= shade[:, None]

        rgba[:, 3] = float(np.clip(opacity, 0.05, 1.0))
        traces.append(
            go.Mesh3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                vertexcolor=[to_hex(c) for c in rgba[:, :3]],
                opacity=float(np.clip(opacity, 0.05, 1.0)),
                hoverinfo="skip",
                name=side,
                lighting=dict(ambient=0.75, diffuse=0.55, specular=0.05),
                flatshading=False,
            )
        )

    scene = go.Figure(data=traces)
    # A vertex-coloured mesh carries no scale for Plotly to build a colourbar from, so an invisible
    # marker supplies one. Without it the figure has colours and no way to read a value off them.
    scene.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(
                size=0.1, color=[float(norm.vmin)], colorscale=_plotly_colorscale(colormap),
                cmin=float(norm.vmin), cmax=float(norm.vmax),
                colorbar=dict(title=label or "estimate", thickness=14, len=0.7),
                showscale=True,
            ),
        )
    )
    subtitle = "  ·  ".join(notes)
    scene.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>" if subtitle else title, x=0.5),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=-1.9, z=0.25)),
        ),
    )
    return scene


def _plotly_colorscale(colormap: Any, samples: int = 64) -> list:
    """Sample a Matplotlib colormap into the ``[[position, "rgb(...)"], …]`` Plotly wants."""
    from matplotlib.colors import to_hex

    return [
        [i / (samples - 1), to_hex(colormap(i / (samples - 1)))]
        for i in range(samples)
    ]


def plot_brain_map(
    values: Mapping[int, float],
    *,
    pvalues: Mapping[int, float] | None = None,
    mode: str = "estimate",
    alpha: float = DEFAULT_ALPHA,
    mask_nonsignificant: bool = True,
    cmap: str | None = None,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    atlas: str = "desikan",
    atlas_kwargs: Mapping[str, Any] | None = None,
    hemisphere: str = "both",
    views: Any = ("lateral", "medial"),
    surface: str = "pial",
    interactive: bool = False,
    shading: bool = True,
    blend: bool = True,
    opacity: float = 1.0,
    threshold: float | None = None,
    label: str = "",
    title: str = "",
    hide: Sequence[int] = (),
):
    """
    Draw *values* on the cortical surface.

    Parameters
    ----------
    values : mapping
        ``{atlas index: value}``, as :func:`parcel_values_from_result` returns. Indices absent from
        the mapping are left unpainted.
    pvalues : mapping, optional
        ``{atlas index: p}``. Required for *mode* ``"pvalue"`` and for significance masking.
    mode : {"estimate", "pvalue"}
        What colour encodes. ``"estimate"`` maps the value through a diverging or sequential
        colormap; ``"pvalue"`` maps −log₁₀(p), which spreads the small p-values that matter instead
        of compressing them all against zero.
    mask_nonsignificant : bool
        Draw parcels with ``p >= alpha`` in flat grey. Has no effect without *pvalues* — and is
        skipped with a warning rather than silently colouring everything, since an unmasked map
        looks identical to one where everything is significant.
    center : float, optional
        Value the diverging colormap centres on. Defaults to 0 when the values straddle it (an
        effect), and to no centring when they do not (a mean).
    vmin, vmax : float, optional
        Pin the colour scale instead of taking it from these values, so a series of figures can be
        compared frame to frame.
    hemisphere : {"both", "left", "right"}
    views : sequence of str or comma-separated str
        nilearn surface views — ``lateral``, ``medial``, ``dorsal``, ``ventral``, …
    interactive : bool
        Return a rotatable 3-D Plotly figure instead of a static Matplotlib one, with both
        hemispheres in a single scene. Same values, same colour scale and the same three-state
        encoding — nilearn's Plotly engine computes per-vertex colours through the colormap, so
        ``set_under`` and ``set_bad`` survive and non-significant still reads differently from
        no-estimate. There are no separate view panels because the view is whatever you rotate it
        to, and no *flat* surface because a flat map has no 3-D structure to explore.
    surface : {"infl", "pial", "flat"}
        Which fsaverage surface to paint on. Inflated exposes the parcels buried in sulci, which on
        a folded pial surface are simply not visible — roughly two thirds of the cortex.
    shading : bool
        Shade the unpainted surface with its own curvature. Off gives a flat silhouette, which is
        cleaner for a figure and makes the parcel boundaries the only structure on the page.
    blend : bool
        Shade the *painted* parcels with the curvature too, rather than only the bare surface. The
        sulcal pattern then shows through the colours, which keeps the anatomy legible where a
        parcel covers a whole gyrus — at the cost of darkening the colours unevenly, so a value
        read off the colourbar is no longer exact.
    opacity : float
        Opacity of the painted parcels, ``0``–``1``. Below 1 the surface shows through, which is
        the other way to keep the folding visible under a dense parcellation. Like *blend*, it
        shifts the rendered colour away from the colourbar, so it is a reading aid rather than a
        setting to publish at.

        Named *opacity* rather than *alpha* on purpose: in this module ``alpha`` already means the
        significance level, and one symbol for "how transparent" and "what counts as significant"
        would be a genuinely dangerous overload.
    threshold : float, optional
        Leave parcels whose \|value\| is below this unpainted, the way a stat map is thresholded.
        Distinct from the significance mask: this one is about effect *size*, and a parcel hidden
        by it is reported in the caption so it is never confused with one that has no estimate.
    hide : sequence of int
        Atlas indices to leave off entirely — not painted and **not counted in the colour scale**.
        That last part is the point: hiding the one parcel that dominates the range is how you see
        the structure among the rest. Distinct from a parcel with no estimate, which is unpainted
        because "not measured" is information.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from nilearn import plotting

    mode = str(mode).strip().lower()
    if mode not in {"estimate", "pvalue"}:
        raise ValueError(f"mode must be 'estimate' or 'pvalue', not {mode!r}.")
    if hemisphere not in {"both", "left", "right"}:
        raise ValueError("hemisphere must be 'both', 'left', or 'right'.")

    hidden = {int(h) for h in hide}
    values = {
        int(k): float(v) for k, v in dict(values or {}).items()
        if np.isfinite(v) and int(k) not in hidden
    }
    pvalues = {
        int(k): float(v) for k, v in dict(pvalues or {}).items()
        if np.isfinite(v) and int(k) not in hidden
    }
    if mode == "pvalue" and not pvalues:
        raise ValueError("mode='pvalue' needs a pvalues mapping.")
    if mask_nonsignificant and not pvalues:
        log.warning(
            "mask_nonsignificant was requested without p-values — every parcel will be drawn as "
            "though it were significant."
        )
        mask_nonsignificant = False

    _, resolved = parcel_resolver(atlas, atlas_kwargs)
    known = set(resolved.index_to_label)
    unknown = sorted(set(values) - known)
    if unknown:
        log.info(
            "Brain map: %d value(s) reference indices this atlas does not carry and are omitted.",
            len(unknown),
        )
        values = {k: v for k, v in values.items() if k in known}

    # ---- Colour scale ---------------------------------------------------------
    if mode == "pvalue":
        scalars = {k: -np.log10(max(p, 1e-300)) for k, p in pvalues.items() if k in known}
        colormap = plt.get_cmap(cmap or "viridis").copy()
        finite = list(scalars.values())
        # With the mask on, nothing below the threshold is coloured, so starting the scale at zero
        # would spend most of the colormap on parcels that are grey.
        floor = -np.log10(alpha) if mask_nonsignificant else 0.0
        norm = Normalize(vmin=floor, vmax=max([*finite, floor + 1.0]))
        bar_label = label or "−log₁₀(p)"
    else:
        scalars = dict(values)
        finite = list(scalars.values())
        if not finite:
            raise ValueError("No value maps onto a parcel of this atlas — nothing to plot.")
        lo = min(finite) if vmin is None else float(vmin)
        hi = max(finite) if vmax is None else float(vmax)
        diverging = center is not None or (lo < 0.0 < hi)
        if diverging:
            pivot = 0.0 if center is None else float(center)
            span = max(abs(lo - pivot), abs(hi - pivot)) or 1.0
            norm = TwoSlopeNorm(vmin=pivot - span, vcenter=pivot, vmax=pivot + span)
            colormap = plt.get_cmap(cmap or "RdBu_r").copy()
        else:
            norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
            colormap = plt.get_cmap(cmap or "viridis").copy()
        bar_label = label or "estimate"

    # Three states in one texture: a value inside the range, a sentinel below it for the
    # non-significant parcels, and NaN for the ones with no estimate. One texture and one render
    # pass — overlaying a second surface on the same faces z-fights.
    #
    # ``set_under`` is what produces the flat non-significant grey. ``set_bad`` is defensive only:
    # nilearn short-circuits NaN to the background map before the colormap ever sees it, and that
    # background renders around #e3e3e3 — already clearly lighter than COLOR_NONSIGNIFICANT's
    # #9e9e9e, which is the separation this figure needs. Setting it anyway means the two states
    # stay distinct if a future nilearn routes NaN through the colormap instead.
    colormap.set_under(COLOR_NONSIGNIFICANT)
    colormap.set_bad(COLOR_ABSENT)
    span = float(norm.vmax) - float(norm.vmin) or 1.0
    sentinel = float(norm.vmin) - _UNDER_MARGIN * span

    texture_values: dict[int, float] = {}
    n_significant = 0
    n_below = 0
    limit = None if threshold in (None, "") else abs(float(threshold))
    for index, scalar in scalars.items():
        # Thresholding drops the parcel entirely (NaN → unpainted), rather than greying it: a
        # small effect is not the same claim as a non-significant one, and the caption counts each.
        if limit is not None and abs(float(scalar)) < limit:
            n_below += 1
            continue
        significant = (not mask_nonsignificant) or (pvalues.get(index, np.inf) < alpha)
        texture_values[index] = float(scalar) if significant else sentinel
        n_significant += int(bool(significant))

    left, right = _vertex_textures(resolved, texture_values)

    # ---- Canvas ---------------------------------------------------------------
    view_keys = _view_keys(views)
    sides = ["left", "right"] if hemisphere == "both" else [hemisphere]
    meshes = fsaverage_meshes(resolved.surf_mesh)
    surface = str(surface or "infl").strip().lower()
    surface = {"inflated": "infl", "folded": "pial"}.get(surface, surface)
    if f"{surface}_left" not in meshes:
        log.warning("No %r surface in %s — falling back to pial.", surface, resolved.surf_mesh)
        surface = "pial"
    # A flat map already shows the whole cortex, so a view angle would only rotate a plane. Forcing
    # one panel per hemisphere avoids drawing the same picture two or four times.
    if surface == "flat":
        view_keys = ["dorsal"]

    # Rows are *views*, columns are *hemispheres*. The two hemispheres of one view then sit side by
    # side, which is the comparison a reader actually makes — left lateral against right lateral —
    # rather than one hemisphere's two views being adjacent and the homologous pair split apart.
    if interactive:
        return _interactive_surface(
            meshes, sides, surface=surface, left=left, right=right, colormap=colormap, norm=norm,
            shading=shading, blend=blend, opacity=opacity, title=title, label=bar_label,
            notes=_scale_notes(
                mask_nonsignificant, alpha, len(scalars), n_significant, hidden, threshold,
                n_below, blend and shading, opacity, len(known), len(scalars),
                len(hidden & known),
            ),
        )

    fig, axes = plt.subplots(
        len(view_keys), len(sides),
        figsize=(4.2 * len(sides), 3.6 * len(view_keys)),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for col, side in enumerate(sides):
        texture = left if side == "left" else right
        for row, view in enumerate(view_keys):
            ax = axes[row][col]
            plotting.plot_surf_stat_map(
                meshes[f"{surface}_{side}"],
                texture,
                hemi=side,
                view=view,
                cmap=colormap,
                vmin=float(norm.vmin),
                vmax=float(norm.vmax),
                symmetric_cbar=False,
                colorbar=False,
                bg_map=(
                    meshes.get(f"sulc_{side}") or meshes.get(f"curv_{side}")
                ) if shading else None,
                # ``bg_on_data`` needs a background to blend, so it follows *shading*: asking for a
                # blend with no curvature map would silently do nothing.
                bg_on_data=bool(blend and shading),
                alpha=float(np.clip(opacity, 0.05, 1.0)),
                axes=ax,
                figure=fig,
            )
            ax.set_title(f"{side} · {view}" if surface != "flat" else f"{side} · flat", fontsize=9)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    bar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.024, pad=0.02)
    bar.set_label(bar_label, fontsize=9)
    if mode == "pvalue":
        line = -np.log10(alpha)
        if norm.vmin <= line <= norm.vmax:
            bar.ax.axhline(line, color="#c44e52", lw=1.4)

    if title:
        fig.suptitle(title, fontsize=11)

    notes = _scale_notes(
        mask_nonsignificant, alpha, len(scalars), n_significant, hidden, threshold, n_below,
        blend and shading, opacity, len(known), len(scalars), len(hidden & known),
    )
    if notes:
        fig.text(0.5, 0.015, "  ·  ".join(notes), ha="center", va="bottom",
                 fontsize=8, color="#555555")

    # ``linked_axes`` is what the Statmodels plot pane's axis sliders act on. A 3-D surface has no
    # meaningful x/y limits to drag, so it is left empty on purpose rather than wired to something
    # that would distort the projection.
    fig.linked_axes = []
    return fig


__all__ = [
    "BRAIN_ATLASES",
    "BRAIN_SURFACES",
    "BRAIN_VIEWS",
    "COLOR_ABSENT",
    "COLOR_NONSIGNIFICANT",
    "DEFAULT_ALPHA",
    "brain_group_coefficient_terms",
    "brain_interaction_contrasts",
    "fsaverage_meshes",
    "parcel_resolver",
    "parcel_values_from_frame",
    "parcel_values_from_result",
    "plot_brain_map",
    "regions_without_geometry",
]
