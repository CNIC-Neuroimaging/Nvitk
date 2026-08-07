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
}

_PARCEL_PANEL: dict[str, str] = {
    parcel: panel for panel, parcels in DESIKAN_LOBES.items() for parcel in parcels
}

# ASL parcels carry the smoothing kernel as a trailing "_0"/"_8"; T1/Desikan parcels carry a
# hemisphere prefix. Neither is anatomy, so both are stripped before any lookup.
_KERNEL_SUFFIX_RE = re.compile(r"_(?:0|8)$")
_SIDE_PREFIX_RE = re.compile(r"^(?:ctx_)?(?:lh|rh|left|right|l|r)_")
_SIDE_SUFFIX_RE = re.compile(r"_(?:lh|rh|left|right)$")
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
    5. Desikan / aseg parcels by lobe, after stripping the hemisphere prefix.

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

    # ---- 5. Cortical / subcortical parcels by lobe -------------------------------
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
    "LOBE_PANELS",
    "PANEL_ORDER",
    "PANEL_OTHER",
    "VASCULAR_PANELS",
    "group_levels_into_panels",
    "natural_level_key",
    "panel_grid",
    "panel_summary",
    "region_display_panel",
    "resolve_panels",
]
