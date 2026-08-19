"""
Anatomical panel groups — partition plot levels into small multiples.

Description
-----------
A model over 13 vessels draws 13 curves on one pair of axes, where a Left_ICA line at ``log(PI) ≈
-0.2`` and a Sagital_Sinus line at ``≈ -1.0`` share a y range that flattens both. Splitting them
into anatomically coherent panels — carotids, anterior, posterior, venous — lets each panel scale to
its own range while keeping the population line comparable across all of them.

This module answers one question: *which panel does this level belong to?* It works on whatever the
grouping column happens to contain, because the analysis frame's ``group_key`` takes several shapes
depending on the ``grouping`` used to build it:

===================================  ==========================================================
Level shape                          Example
===================================  ==========================================================
qvtpy vessel ids                     ``Left_ICA``, ``RMCA``, ``Sagital_Sinus``, ``LPCOMM``
ASL vascular-8 parcels               ``left_mca_8``, ``right_pca_8``, ``watershed_0``
ASL / T1 Desikan parcels             ``ctx-lh-superiorfrontal``, ``rh_precuneus``
FreeSurfer subcortical structures    ``Left-Hippocampus``, ``Brain-Stem``
hemisphere keys (melted L/R)         ``ICA``, ``MCA``, ``PCA``, ``Basilar``, ``Watershed``
territory keys (already melted)      ``Anterior Circulation``, ``Venous Drainage``
===================================  ==========================================================

Vascular levels reuse :data:`~nvitk.stats._vessel_territory_map.REGION_TO_TERRITORY_FLOW` so the
panels agree with the ``territory`` grouping — a level that melts into "Posterior Circulation" also
*panels* into it. Cortical parcels have no vascular territory, so they group by lobe instead.

Levels already equal to a panel name pass through unchanged: a frame grouped by territory gets one
panel per territory, which is the small-multiples view of the same data.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from typing import Any, Iterable, Mapping

from nvitk.stats._vessel_territory_map import REGION_TO_TERRITORY_FLOW

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
PANEL_OTHER = "Other"

#: Vascular panels, in the order they should be laid out (feeding arteries → outflow).
VASCULAR_PANELS: tuple[str, ...] = (
    "Internal Carotid Arteries",
    "Anterior Circulation",
    "Posterior Circulation",
    "Communicating",
    "Watershed",
    "Venous Drainage",
)

#: Parenchymal panels for atlases that parcellate tissue rather than vessels (Desikan, aseg).
LOBE_PANELS: tuple[str, ...] = (
    "Frontal",
    "Parietal",
    "Temporal",
    "Occipital",
    "Cingulate",
    "Insula",
    "Subcortical",
    "Cerebellum & Brainstem",
    "White Matter",
    "Ventricles & CSF",
    "Whole Brain",
)

#: Canonical display order; anything unrecognized sorts after these, with ``Other`` always last.
PANEL_ORDER: tuple[str, ...] = (*VASCULAR_PANELS, *LOBE_PANELS)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
# Hemisphere-melted keys (region_to_hemisphere_pair_key output) have no side prefix, so they miss
# the vessel table's "lmca"/"left_mca" entries and need their own row.
_HEMISPHERE_PANEL: dict[str, str] = {
    "ica": "Internal Carotid Arteries",
    "cca": "Internal Carotid Arteries",
    "mca": "Anterior Circulation",
    "aca": "Anterior Circulation",
    "pca": "Posterior Circulation",
    "basi": "Posterior Circulation",
    "basilar": "Posterior Circulation",
    "va": "Posterior Circulation",
    "vertebral": "Posterior Circulation",
    "pcomm": "Communicating",
    "acomm": "Communicating",
    "comm": "Communicating",
    "communicating": "Communicating",
    "watershed": "Watershed",
    "sss": "Venous Drainage",
    "sssv": "Venous Drainage",
    "strv": "Venous Drainage",
    "transverse": "Venous Drainage",
    "sigmoid": "Venous Drainage",
    "sagital_sinus": "Venous Drainage",
    "sagittal_sinus": "Venous Drainage",
    "straight_sinus": "Venous Drainage",
    "superior_sagittal_sinus": "Venous Drainage",
}

#: Desikan–Killiany cortical parcels by lobe. ``paracentral`` is conventionally counted frontal.
DESIKAN_LOBES: dict[str, tuple[str, ...]] = {
    "Frontal": (
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parstriangularis",
        "parsorbitalis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "precentral",
        "paracentral",
        "frontalpole",
    ),
    "Parietal": (
        "superiorparietal",
        "inferiorparietal",
        "supramarginal",
        "postcentral",
        "precuneus",
    ),
    "Temporal": (
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "fusiform",
        "transversetemporal",
        "entorhinal",
        "parahippocampal",
        "temporalpole",
    ),
    "Occipital": ("lateraloccipital", "lingual", "cuneus", "pericalcarine"),
    "Cingulate": (
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
        "posteriorcingulate",
        "isthmuscingulate",
    ),
    "Insula": ("insula",),
    "Subcortical": (
        "thalamus",
        "thalamus_proper",
        "caudate",
        "putamen",
        "pallidum",
        "hippocampus",
        "amygdala",
        "accumbens",
        "accumbens_area",
        "ventraldc",
        "basal_forebrain",
        "choroid_plexus",
    ),
    "Cerebellum & Brainstem": (
        "cerebellum",
        "cerebellum_cortex",
        "cerebellum_white_matter",
        "brain_stem",
        "brainstem",
        "pons",
        "midbrain",
        "medulla",
    ),
    # The corpus callosum and the WM hypointensity classes are white matter, not grey structures,
    # and pooling them with the basal ganglia would mix tissue types inside one panel.
    "White Matter": (
        "cc_anterior",
        "cc_central",
        "cc_mid_anterior",
        "cc_mid_posterior",
        "cc_posterior",
        "cerebralwhitemattervol",
        "lhcerebralwhitemattervol",
        "rhcerebralwhitemattervol",
        "wm_hypointensities",
        "non_wm_hypointensities",
        "optic_chiasm",
        "vessel",
    ),
    "Ventricles & CSF": (
        "3rd_ventricle",
        "4th_ventricle",
        "5th_ventricle",
        "lateral_ventricle",
        "inf_lat_vent",
        "csf",
        "ventriclechoroidvol",
    ),
    # FreeSurfer's whole-head summaries. They are not regions at all — every parcel is measured
    # *against* them — so they get their own panel rather than diluting an anatomical one.
    "Whole Brain": (
        "brainsegvol",
        "brainsegvolnotvent",
        "brainsegvolnotventsurf",
        "cortexvol",
        "lhcortexvol",
        "rhcortexvol",
        "subcortgrayvol",
        "totalgrayvol",
        "supratentorialvol",
        "supratentorialvolnotvent",
        "maskvol",
        "etiv",
        "meanthickness",
        "whitesurfarea",
        "numvert",
        "surfaceholes",
        "brainsegvol_to_etiv",
        "maskvol_to_etiv",
    ),
}

_PARCEL_PANEL: dict[str, str] = {
    parcel: panel for panel, parcels in DESIKAN_LOBES.items() for parcel in parcels
}

#: Aggregate levels the ASL / T1 pipelines publish **alongside** the parcels they pool:
#: ``ctx-Left-Frontal-Lobe``, ``ctx-left-hemisphere``, ``ctx-whole-brain``. They are the pipeline's
#: own summaries, not extra anatomy, so they map to the panel they summarize rather than falling
#: through to ``Other``.
_AGGREGATE_PANEL: dict[str, str] = {
    "frontal_lobe": "Frontal",
    "parietal_lobe": "Parietal",
    "temporal_lobe": "Temporal",
    "occipital_lobe": "Occipital",
    "cingulate_lobe": "Cingulate",
    "insula_lobe": "Insula",
    "anterior_cingulate": "Cingulate",
    "posterior_cingulate": "Cingulate",
    "whole_brain": "Whole Brain",
    "hemisphere": "Whole Brain",
    "brain": "Whole Brain",
}

# ---------------------------------------------------------------------------
# Granularity
# ---------------------------------------------------------------------------
#: How coarse a published region id is. The ASL and T1 tables publish several of these *in the same
#: column*, which is the whole reason this exists: ``ctx-lh-precuneus``, ``ctx-Left-Frontal-Lobe``
#: and ``ctx-whole-brain`` are all valid ``region_id`` values, and a model that treats them as
#: sibling levels of one factor is fitting a parcel against a sum that already contains it.
GRANULARITY_PARCEL = "parcel"
GRANULARITY_LOBE = "lobe"
GRANULARITY_HEMISPHERE = "hemisphere"
GRANULARITY_WHOLE = "whole"

#: Coarse → fine, so a caller can ask for "no coarser than X".
GRANULARITY_ORDER: tuple[str, ...] = (
    GRANULARITY_WHOLE, GRANULARITY_HEMISPHERE, GRANULARITY_LOBE, GRANULARITY_PARCEL,
)

_WHOLE_TOKENS: frozenset[str] = frozenset({"whole_brain", "brain", "wholebrain"})
_HEMISPHERE_TOKENS: frozenset[str] = frozenset({"hemisphere", "hemi"})


def region_granularity(level: object) -> str:
    """
    How coarse *level* is — ``parcel`` / ``lobe`` / ``hemisphere`` / ``whole``.

    Only the aggregates are detected by name; everything else is a parcel, which is the right
    default because an unrecognised id is far more likely to be a parcellation this function has
    never heard of than a summary of one.

    Examples
    --------
    >>> region_granularity("ctx-lh-precuneus")
    'parcel'
    >>> region_granularity("ctx-Left-Frontal-Lobe")
    'lobe'
    >>> region_granularity("ctx_left_hemisphere"), region_granularity("ctx-whole-brain")
    ('hemisphere', 'whole')
    """
    token = _normalize(level)
    if not token:
        return GRANULARITY_PARCEL
    stem = _SIDE_SUFFIX_RE.sub("", _SIDE_PREFIX_RE.sub("", token))
    # ``ctx_whole_brain`` keeps a bare ``ctx_`` that the side regex leaves behind.
    stem = re.sub(r"^ctx_", "", stem)
    if stem in _WHOLE_TOKENS:
        return GRANULARITY_WHOLE
    if stem in _HEMISPHERE_TOKENS:
        return GRANULARITY_HEMISPHERE
    if stem.endswith("_lobe") or stem in _AGGREGATE_PANEL:
        # ``anterior_cingulate`` pools two parcels the way a lobe does, so it is a lobe-level
        # summary even though its published name does not say "lobe".
        return GRANULARITY_LOBE
    return GRANULARITY_PARCEL


#: Side token → the word used in a lateralized lobe label.
_SIDE_WORD: dict[str, str] = {
    "lh": "Left", "left": "Left", "l": "Left",
    "rh": "Right", "right": "Right", "r": "Right",
}


def region_side(level: object) -> str:
    """
    ``"Left"`` / ``"Right"`` / ``""`` for a published region id.

    Reads the side token wherever it sits — ``ctx-lh-precuneus``, ``left_precuneus``,
    ``ctx-Left-Frontal-Lobe`` and ``precuneus_lh`` all answer the same. ``""`` for anything
    bilateral (``ctx_bh_insula``) or midline, which is the honest answer: those genuinely have no
    side, and inventing one would put a bilateral mean on half a brain.
    """
    token = _normalize(level)
    if not token:
        return ""
    match = _SIDE_PREFIX_RE.match(token)
    if match:
        return _SIDE_WORD.get(match.group(0).replace("ctx_", "").strip("_"), "")
    match = _SIDE_SUFFIX_RE.search(token)
    if match:
        return _SIDE_WORD.get(match.group(0).strip("_"), "")
    return ""


def region_lobe_key(level: object) -> str:
    """
    Lobe a region belongs to, **keeping its side** — ``Left Frontal``, ``Right Parietal``.

    Different from :func:`region_display_panel`, which deliberately drops the side: that one groups
    levels into *panels* for small multiples, where the left and right curves of one lobe belong on
    the same axes. As a grouping key the side has to survive, because a lobe analysis that averages
    the two hemispheres together cannot answer a lateralized question at all — and hemispheric
    asymmetry is most of what a lobe-level perfusion or volumetry analysis is looking for.

    Falls back to the bare panel name when the region carries no side, so bilateral and midline
    structures keep one level rather than being duplicated onto both.

    Examples
    --------
    >>> region_lobe_key("ctx-lh-superiorfrontal"), region_lobe_key("ctx-Right-Parietal-Lobe")
    ('Left Frontal', 'Right Parietal')
    >>> region_lobe_key("ctx_bh_insula")
    'Insula'
    """
    panel = region_display_panel(level)
    if not panel:
        return PANEL_OTHER
    side = region_side(level)
    return f"{side} {panel}" if side else panel


def split_by_granularity(levels: "Iterable[object]") -> dict[str, list[str]]:
    """Group *levels* by :func:`region_granularity`, preserving order within each class."""
    out: dict[str, list[str]] = {g: [] for g in GRANULARITY_ORDER}
    for level in levels:
        out.setdefault(region_granularity(level), []).append(str(level))
    return {k: v for k, v in out.items() if v}

# ASL parcels carry the smoothing kernel as a trailing "_0"/"_8"; T1/Desikan parcels carry a
# hemisphere prefix. Neither is anatomy, so both are stripped before any lookup.
_KERNEL_SUFFIX_RE = re.compile(r"_(?:0|8)$")
# ``bh`` is ASL's bilateral parcel prefix (``ctx_bh_insula``) — a side token like any other,
# and without it a bilateral parcel resolves to no lobe at all.
_SIDE_PREFIX_RE = re.compile(r"^(?:ctx_)?(?:lh|rh|bh|left|right|l|r)_")
_SIDE_SUFFIX_RE = re.compile(r"_(?:lh|rh|bh|left|right)$")
_NUMERIC_CHUNK_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def _normalize(level: str) -> str:
    """Lowercase, underscore-separated form of a level label (``Left-ICA`` → ``left_ica``)."""
    text = str(level).strip().lower()
    text = re.sub(r"[\s\-.]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def region_display_panel(level: str) -> str | None:
    """
    Anatomical panel a grouping level belongs to, or ``None`` when it is not recognized.

    Resolution order — most specific first, so a vessel never falls through to a lobe:

    1. the level is already a panel name (a territory-grouped frame);
    2. watershed parcels, which have no feeding vessel of their own;
    3. the 4D-flow vessel table, side prefixes included (``left_ica`` → carotids);
    4. hemisphere-melted keys, which carry no side (``MCA`` → anterior circulation);
    5. published aggregates (``ctx-Left-Frontal-Lobe``, ``ctx-whole-brain``);
    6. Desikan / aseg parcels by lobe, after stripping the hemisphere prefix.

    Examples
    --------
    >>> region_display_panel("Left_ICA")
    'Internal Carotid Arteries'
    >>> region_display_panel("right_pca_8")
    'Posterior Circulation'
    >>> region_display_panel("ctx-lh-superiorfrontal")
    'Frontal'
    >>> region_display_panel("mystery_region") is None
    True
    """
    raw = str(level).strip()
    if not raw:
        return None

    # ---- 1. Already a panel name ------------------------------------------------
    for panel in PANEL_ORDER:
        if raw.casefold() == panel.casefold():
            return panel

    token = _normalize(raw)
    if not token:
        return None
    # "Desikan/Cortical" and friends: the territory mapper's catch-all for cortical parcels.
    if token.startswith("desikan"):
        return None

    # ---- 2. Watershed: not fed by a single vessel, so it is its own panel --------
    if "watershed" in token:
        return "Watershed"

    stem = _KERNEL_SUFFIX_RE.sub("", token)

    # ---- 3. Vessel table — shares its mapping with the ``territory`` grouping ----
    for candidate in (token, stem):
        panel = REGION_TO_TERRITORY_FLOW.get(candidate)
        if panel:
            return panel

    # ---- 4. Hemisphere-melted keys (no side prefix) ------------------------------
    sideless = _SIDE_SUFFIX_RE.sub("", _SIDE_PREFIX_RE.sub("", stem))
    for candidate in (stem, sideless):
        panel = _HEMISPHERE_PANEL.get(candidate)
        if panel:
            return panel
        panel = REGION_TO_TERRITORY_FLOW.get(candidate)
        if panel:
            return panel

    # ---- 5. Published aggregates (``ctx-Left-Frontal-Lobe``, ``ctx-whole-brain``) -
    # Before the parcel table, because an aggregate's name contains no parcel to match on and
    # would otherwise fall through to ``Other`` — putting a pipeline's own lobe summary in the
    # panel that is not the lobe it summarizes.
    bare = re.sub(r"^ctx_", "", sideless)
    for candidate in (bare, sideless):
        panel = _AGGREGATE_PANEL.get(candidate)
        if panel:
            return panel

    # ---- 6. Cortical / subcortical parcels by lobe -------------------------------
    for candidate in (sideless, sideless.replace("_", "")):
        panel = _PARCEL_PANEL.get(candidate)
        if panel:
            return panel
    # aseg names like "left_cerebellum_cortex" survive step 5's exact match, but compound
    # structures ("cerebellum_white_matter") are safer matched by their leading structure word.
    for parcel, panel in _PARCEL_PANEL.items():
        if sideless.startswith(parcel):
            return panel
    return None


def group_levels_into_panels(
    levels: Iterable[str],
    *,
    other_label: str = PANEL_OTHER,
    include_other: bool = True,
) -> dict[str, list[str]]:
    """
    Partition grouping levels into anatomical panels, in display order.

    Parameters
    ----------
    levels : iterable of str
        Levels of the grouping column, in the order the caller wants them within a panel.
    other_label : str
        Panel collecting levels :func:`region_display_panel` does not recognize.
    include_other : bool
        Drop unrecognized levels entirely when ``False``, instead of collecting them.

    Returns
    -------
    dict
        ``{panel: [level, ...]}`` — only non-empty panels, ordered by :data:`PANEL_ORDER` with
        unknown panels alphabetically after them and ``other_label`` always last. Empty when no
        level was recognized, which the caller should treat as "grouped display not applicable".
    """
    buckets: dict[str, list[str]] = {}
    recognized = 0
    for level in levels:
        panel = region_display_panel(level)
        if panel is None:
            if not include_other:
                continue
            panel = other_label
        else:
            recognized += 1
        buckets.setdefault(panel, []).append(str(level))

    if not recognized:
        return {}

    def sort_key(panel: str) -> tuple[int, str]:
        if panel == other_label:
            return (2, "")
        if panel in PANEL_ORDER:
            return (0, f"{PANEL_ORDER.index(panel):03d}")
        return (1, panel.casefold())

    return {panel: buckets[panel] for panel in sorted(buckets, key=sort_key)}


def resolve_panels(levels: Iterable[str], *, column: str = "group") -> dict[str, list[str]]:
    """
    :func:`group_levels_into_panels`, raising a caller-ready message when nothing maps.

    Every grouped plot needs the same refusal — panelling levels that carry no anatomy would produce
    one "Other" panel identical to the overview — so the wording lives here rather than in each
    plotting function.

    Raises
    ------
    ValueError
        When no level resolves to a panel.
    """
    levels = list(levels)
    panels = group_levels_into_panels(levels)
    if not panels:
        raise ValueError(
            f"None of the {len(levels)} {column!r} levels map to a known anatomical region, so "
            f"there are no groups to split into. Use the Overview display for this model."
        )
    return panels


def panel_grid(
    n_panels: int,
    *,
    panel_size: tuple[float, float] = (8.5, 4.8),
    n_cols: int | None = None,
    title: str = "",
) -> tuple[Any, list[Any]]:
    """
    A figure holding one visible axes per panel, two per row.

    Constrained layout rather than ``tight_layout``: a grid with a suptitle needs to re-flow on every
    draw as the canvas is resized, and it also signals the plot pane not to re-fit on top of it.

    Returns
    -------
    (figure, axes)
        Exactly *n_panels* axes in reading order; any spare cell in the last row is hidden.
    """
    import matplotlib.pyplot as plt

    n_panels = max(int(n_panels), 1)
    n_cols = n_cols or (1 if n_panels == 1 else 2)
    n_rows = -(-n_panels // n_cols)  # ceiling division
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows), squeeze=False
    )
    flat = [ax for row in grid for ax in row]
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.set_layout_engine("constrained")
    return fig, flat[:n_panels]


def natural_level_key(value: object) -> tuple:
    """Sort key placing ``region_2`` before ``region_10`` — digit runs compare numerically."""
    parts = _NUMERIC_CHUNK_RE.split(str(value))
    return tuple(int(p) if p.isdigit() else p.casefold() for p in parts)


def panel_summary(panels: Mapping[str, list[str]]) -> str:
    """One-line description of a partition, for a status bar (``ICA (2) · Anterior (4)``)."""
    return " · ".join(f"{panel} ({len(levels)})" for panel, levels in panels.items())


__all__ = [
    "DESIKAN_LOBES",
    "GRANULARITY_HEMISPHERE",
    "GRANULARITY_LOBE",
    "GRANULARITY_ORDER",
    "GRANULARITY_PARCEL",
    "GRANULARITY_WHOLE",
    "LOBE_PANELS",
    "PANEL_ORDER",
    "PANEL_OTHER",
    "VASCULAR_PANELS",
    "group_levels_into_panels",
    "natural_level_key",
    "panel_grid",
    "panel_summary",
    "region_display_panel",
    "region_granularity",
    "region_lobe_key",
    "region_side",
    "resolve_panels",
    "split_by_granularity",
]
