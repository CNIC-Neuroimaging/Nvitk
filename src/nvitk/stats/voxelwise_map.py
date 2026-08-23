"""
Figures for a voxelwise ``randomise`` result.

The napari tool and the Statmodels display both render the same maps, so the thresholds, the
colour meaning and the caption live here rather than in either of them. Two viewers that disagree
about whether 0.95 means "p < 0.05" or "an effect of 0.95" is not a cosmetic inconsistency: it is
two different claims about the same file.

What the colours mean
---------------------
``randomise`` writes corrected p-values as **1 − p**, so a bright voxel in a ``*_corrp_*`` map is
*strong evidence*, not a large effect. Nothing in these figures encodes effect size unless the
``tstat`` map is what was asked for. Every builder here returns its caption alongside the figure
for exactly that reason — a p-map drawn without saying so reads as an effect map.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

log = Logger()

#: p-value thresholds offered by the viewers, as α. 0.05 is the default everywhere.
ALPHA_CHOICES: tuple[float, ...] = (0.10, 0.05, 0.01, 0.001)

#: How a result should be drawn. ``(label, key)`` in the order a picker should offer them — the
#: single source both viewers populate from, so they cannot label the same view differently.
VIEW_KINDS: tuple[tuple[str, str], ...] = (
    ("Cortical surface", "surface"),
    ("Glass brain (whole brain)", "glass"),
    ("Orthogonal slices", "slices"),
)

#: Sequential colormaps offered for a 1−p map. Deliberately not the diverging list the vascular and
#: brain maps use: 1−p is one-sided and has no meaningful midpoint, so a diverging map would invite
#: centring a quantity that cannot be centred.
VOXELWISE_COLORMAPS: tuple[tuple[str, str], ...] = (
    ("hot", "hot"),
    ("viridis", "viridis"),
    ("magma", "magma"),
    ("inferno", "inferno"),
    ("plasma", "plasma"),
    ("cividis", "cividis"),
    ("YlOrRd", "YlOrRd"),
)

#: ``display_mode`` values nilearn accepts, per builder — the registries genuinely differ, and
#: offering a projector-only mode to the slice view fails at draw time.
#: From ``nilearn.plotting.displays._slicers.SLICERS`` / ``._projectors.PROJECTORS`` (0.13).
SLICE_DISPLAY_MODES: tuple[str, ...] = (
    "ortho", "tiled", "mosaic", "x", "y", "z", "xz", "yx", "yz",
)
GLASS_DISPLAY_MODES: tuple[str, ...] = (
    "ortho", "l", "r", "lr", "lyr", "lzr", "lyrz", "lzry", "x", "y", "z", "xz", "yx", "yz",
)

#: How many cut coordinates each ``display_mode`` consumes. ``1`` means the mode also accepts an
#: int (a montage of N cuts) in place of a coordinate; ``0`` means it takes none.
DISPLAY_MODE_N_CUTS: dict[str, int] = {
    "ortho": 3, "tiled": 3, "mosaic": 0,
    "x": 1, "y": 1, "z": 1,
    "xz": 2, "yx": 2, "yz": 2,
    "l": 1, "r": 1, "lr": 2, "lyr": 3, "lzr": 3, "lyrz": 4, "lzry": 4,
}


def _accepted(fn: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep only the *kwargs* that *fn* actually declares.

    nilearn's plotting signatures move between releases and this repo is run against several at
    once — 0.11 in the GUI environment, 0.13/0.14 elsewhere. An unknown keyword does not raise at
    the nilearn call: it is swallowed into ``**kwargs`` and forwarded to matplotlib, which fails
    much later with ``AxesImage.set() got an unexpected keyword argument``. Filtering here turns a
    confusing crash into a silently unavailable option.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def supports_cut_coords(view: str) -> bool:
    """Whether the installed nilearn lets *view* take cut coordinates.

    ``plot_glass_brain`` gained ``cut_coords`` after 0.11, so on an older install the slice
    sliders have nothing to drive and the caller should say so rather than move a control that
    does nothing.
    """
    import inspect

    from nilearn import plotting

    fn = plotting.plot_glass_brain if str(view).lower() == "glass" else plotting.plot_stat_map
    try:
        return "cut_coords" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def supports_opacity(view: str) -> bool:
    """Whether the installed nilearn exposes an opacity control for *view*."""
    import inspect

    from nilearn import plotting

    key = str(view).lower()
    if key == "surface":
        return False
    fn = plotting.plot_glass_brain if key == "glass" else plotting.plot_stat_map
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return ("alpha" if key == "glass" else "transparency") in params


def display_modes_for(view: str) -> tuple[str, ...]:
    """The ``display_mode`` values valid for *view* (empty for the surface, which has none)."""
    key = str(view or "").strip().lower()
    if key == "glass":
        return GLASS_DISPLAY_MODES
    if key == "slices":
        return SLICE_DISPLAY_MODES
    return ()


def cut_axes_for(display_mode: str) -> tuple[str, ...]:
    """Which of ``x``/``y``/``z`` a *display_mode* takes a coordinate for, in nilearn's own order.

    ``ortho``/``tiled`` take all three. A two-letter mode takes its two letters. A single-axis mode
    takes one — and there, nilearn also accepts an int meaning *number of cuts*, which is how a
    montage is requested. Projector-only modes (``l``, ``lyrz``, …) fix their own angles, so they
    take no coordinate from us.
    """
    mode = str(display_mode or "").strip().lower()
    if mode in ("ortho", "tiled"):
        return ("x", "y", "z")
    if mode in ("x", "y", "z"):
        return (mode,)
    if mode in ("xz", "yx", "yz"):
        return tuple(mode)
    return ()


def world_bounds(image: Any) -> dict[str, tuple[float, float]]:
    """World-space (mm) min/max per axis for *image*, from its own affine and shape.

    Computed rather than hardcoded to MNI's bounding box: a result on a 4 mm grid, a cropped FOV or
    a non-MNI template all have different extents, and a slider clamped to the wrong range silently
    refuses to reach half the brain.
    """
    shape = np.asarray(image.shape[:3], dtype=float)
    affine = np.asarray(image.affine, dtype=float)
    corners = np.array(
        [[i, j, k, 1.0]
         for i in (0.0, shape[0] - 1)
         for j in (0.0, shape[1] - 1)
         for k in (0.0, shape[2] - 1)]
    )
    world = corners @ affine.T
    return {
        axis: (float(world[:, i].min()), float(world[:, i].max()))
        for i, axis in enumerate("xyz")
    }


def alpha_to_map_threshold(alpha: float) -> float:
    """α → the value a ``*_corrp_*`` map must exceed. ``0.05`` → ``0.95``."""
    return 1.0 - float(alpha)


def is_corrp_kind(kind: str) -> bool:
    """True when *kind* names a 1−p map rather than a statistic."""
    from nvitk.measure.voxelwise import CORRP_KINDS

    return str(kind) in CORRP_KINDS or str(kind).endswith("p_tstat")


def load_map(path: str | Path) -> Any:
    """Read one statistical map as a nibabel image."""
    import nibabel as nib

    return nib.load(str(Path(path).expanduser().resolve()))


def resolve_band(
    *, lo: float | None = None, hi: float | None = None, alpha: float | None = None
) -> tuple[float, float]:
    """Normalise a 1−p window to ``(lo, hi)``.

    ``alpha`` is the older, coarser spelling of the same thing — ``alpha=0.05`` is the band
    ``0.95 … 1.0``. It stays supported so existing callers keep working; an explicit *lo* wins.
    """
    if lo is None:
        lo = alpha_to_map_threshold(alpha if alpha is not None else 0.05)
    hi = 1.0 if hi is None else float(hi)
    lo = float(lo)
    if hi <= lo:
        raise ValueError(f"Threshold band is empty: lo={lo:g} must be below hi={hi:g}.")
    return lo, hi


def threshold_image(
    image: Any,
    *,
    kind: str,
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[Any, float, float]:
    """Return *image* with out-of-band voxels zeroed, plus the ``(lo, hi)`` band used.

    For a 1−p map everything outside the band is *not the result being looked at*, so it is zeroed
    rather than merely hidden — a colour scale stretched over the whole 0…1 range would give
    p = 0.6 a visible colour. An upper bound below 1.0 is how you isolate a marginal shell
    (0.95–0.99) from the voxels that pass overwhelmingly.

    For a statistic map nothing is zeroed here; the caller passes the threshold to nilearn, which
    handles a signed statistic symmetrically.
    """
    import nibabel as nib

    band_lo, band_hi = resolve_band(lo=lo, hi=hi, alpha=alpha)
    if not is_corrp_kind(kind):
        return image, 0.0, 0.0
    data = np.asarray(image.dataobj, dtype=float)
    data[(data <= band_lo) | (data > band_hi)] = 0.0
    return nib.Nifti1Image(data, image.affine, image.header), band_lo, band_hi


def significant_voxels(
    image: Any,
    *,
    kind: str,
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> int:
    """How many voxels fall in the band — the number a caption has to be able to quote."""
    data = np.asarray(image.dataobj, dtype=float)
    if is_corrp_kind(kind):
        band_lo, band_hi = resolve_band(lo=lo, hi=hi, alpha=alpha)
        return int(np.count_nonzero((data > band_lo) & (data <= band_hi)))
    return int(np.count_nonzero(np.isfinite(data) & (data != 0)))


def map_caption(
    *,
    kind: str,
    contrast: str,
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    n_significant: int | None = None,
    n_subjects: int | None = None,
    n_perm: int | None = None,
    evs: Sequence[str] = (),
    interactive: bool = False,
) -> str:
    """The sentence under the figure. Says what the colour is before it says anything else."""
    from nvitk.measure.voxelwise import STAT_KINDS

    band_lo, band_hi = resolve_band(lo=lo, hi=hi, alpha=alpha)
    bounded = band_hi < 1.0

    if is_corrp_kind(kind):
        corrected = "FWE-corrected" if "corrp" in kind else "uncorrected"
        if bounded:
            # A bounded window is not a significance threshold and must not read as one: it hides
            # the *strongest* voxels as well as the weakest.
            lead = (
                f"Colour is 1 − p ({corrected}), windowed to {band_lo:.3g}–{band_hi:.3g} — that is "
                f"p between {1.0 - band_hi:.3g} and {1.0 - band_lo:.3g}. Voxels outside the window, "
                f"including more significant ones, are hidden. It is evidence, not effect size."
            )
        else:
            lead = (
                f"Colour is 1 − p ({corrected}), so {band_lo:.3g} means p < {1.0 - band_lo:.3g}. "
                "It is evidence, not effect size."
            )
    elif kind == "tstat":
        lead = "Colour is the t-statistic — signed effect size over its standard error, unpermuted."
    else:
        lead = f"Colour is {STAT_KINDS.get(kind, kind)}."

    parts = [lead, f"Contrast: {contrast}."]
    if n_significant is not None:
        if not is_corrp_kind(kind):
            parts.append(f"{n_significant} voxel(s) with an estimate.")
        elif bounded:
            parts.append(f"{n_significant} voxel(s) in the window.")
        else:
            parts.append(f"{n_significant} voxel(s) pass p < {1.0 - band_lo:.3g}.")
    provenance: list[str] = []
    if n_subjects:
        provenance.append(f"{n_subjects} subject(s)")
    if n_perm:
        provenance.append(f"{n_perm} permutation(s)")
    if evs:
        provenance.append("EVs " + ", ".join(str(e) for e in evs))
    if provenance:
        parts.append(" · ".join(provenance) + ".")
    if interactive:
        parts.append("Drag to rotate, scroll to zoom.")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Figure builders
# ──────────────────────────────────────────────────────────────────────────────
def _unlink_axes(figure: Any) -> Any:
    """Mark *figure* as having no axes the pane's x/y sliders should rescale.

    A slice montage, a glass-brain projection and a 3-D surface panel all have axes in units that
    are not the data's, so dragging an axis-limit slider would crop the picture rather than rescale
    it. Same opt-out the brain map takes (``brain_map.py``).
    """
    figure.linked_axes = []
    return figure


def plot_voxelwise_surface(
    image: Any,
    *,
    kind: str,
    contrast: str = "",
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    interactive: bool = False,
    cmap: str = "hot",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    surface: str = "pial",
    surf_mesh: str = "fsaverage5",
    title: str = "",
) -> Any:
    """Project the map onto the cortical surface.

    ``interactive`` picks the renderer the way the Statmodels pane's Interactive checkbox does
    everywhere else: a rotatable 3-D view (:func:`nilearn.plotting.view_img_on_surf`) or a static
    Matplotlib panel grid (:func:`nilearn.plotting.plot_img_on_surf`). Same map, same threshold,
    same colours — only the medium changes.

    ``surface`` selects the geometry: ``"pial"`` folded, ``"infl"`` inflated (nilearn's
    ``inflate=True``), ``"flat"`` — which has no view angle to pick, so the caller should hide the
    views control for it.
    """
    from nilearn import plotting

    thresholded, band_lo, band_hi = threshold_image(image, kind=kind, alpha=alpha, lo=lo, hi=hi)
    label = title or (f"{contrast} — {kind}" if contrast else kind)
    # A 1−p map is one-sided and lives in [0, 1]; a t-map is signed and must keep a symmetric scale.
    corrp = is_corrp_kind(kind)
    vmin = band_lo if corrp else None
    vmax = band_hi if corrp else None

    if interactive:
        return plotting.view_img_on_surf(
            thresholded,
            surf_mesh=surf_mesh,
            threshold=band_lo if corrp else None,
            cmap=cmap if corrp else "RdBu_r",
            symmetric_cmap=not corrp,
            vmin=vmin,
            vmax=vmax,
            title=label,
        )
    figure = plotting.plot_img_on_surf(
        thresholded,
        surf_mesh=surf_mesh,
        views=list(views),
        hemispheres=list(hemispheres),
        cmap=cmap if corrp else "RdBu_r",
        threshold=band_lo if corrp else None,
        symmetric_cbar=not corrp,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        cbar_tick_format="%.2g",
        title=label,
        inflate=str(surface).strip().lower() in ("infl", "inflated"),
    )[0]
    return _unlink_axes(figure)


def plot_voxelwise_glass(
    image: Any,
    *,
    kind: str,
    contrast: str = "",
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    cmap: str = "hot",
    display_mode: str = "ortho",
    cut_coords: Any = None,
    opacity: float = 0.7,
    title: str = "",
) -> Any:
    """Whole-brain maximum-intensity projection.

    The surface view can only show cortex. A perfusion or lesion effect in the thalamus, the
    cerebellum or deep white matter is invisible there and present here, which is why this is an
    alternative rather than a style option.
    """
    import matplotlib.pyplot as plt
    from nilearn import plotting

    thresholded, band_lo, band_hi = threshold_image(image, kind=kind, alpha=alpha, lo=lo, hi=hi)
    corrp = is_corrp_kind(kind)
    fig = plt.figure(figsize=(9.0, 3.2))
    # Glass brain spells opacity 'alpha' and only gained 'cut_coords' after nilearn 0.11, so both
    # go through the signature filter rather than being assumed present.
    optional = _accepted(
        plotting.plot_glass_brain,
        cut_coords=cut_coords,
        alpha=float(np.clip(opacity, 0.05, 1.0)),
    )
    plotting.plot_glass_brain(
        thresholded,
        display_mode=str(display_mode or "ortho"),
        threshold=band_lo if corrp else "auto",
        cmap=cmap if corrp else "RdBu_r",
        colorbar=True,
        # A 1−p map has no sign, so taking |value| would be a lie about a one-sided quantity.
        plot_abs=not corrp,
        symmetric_cbar=not corrp,
        vmin=band_lo if corrp else None,
        vmax=band_hi if corrp else None,
        title=title or (f"{contrast} — {kind}" if contrast else kind),
        figure=fig,
        **optional,
    )
    return _unlink_axes(fig)


def plot_voxelwise_slices(
    image: Any,
    *,
    kind: str,
    contrast: str = "",
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    cmap: str = "hot",
    display_mode: str = "ortho",
    cut_coords: Any = None,
    opacity: float = 1.0,
    title: str = "",
) -> Any:
    """Orthogonal slices over the MNI152 template — the view that gives a cluster coordinates."""
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from nilearn.datasets import load_mni152_template

    thresholded, band_lo, band_hi = threshold_image(image, kind=kind, alpha=alpha, lo=lo, hi=hi)
    corrp = is_corrp_kind(kind)
    fig = plt.figure(figsize=(9.0, 3.2))
    # 'transparency' arrived in nilearn 0.12. On 0.11 an unknown keyword is not rejected here — it
    # is forwarded to matplotlib and blows up inside AxesImage.set(), which is what this filter
    # exists to prevent.
    transparency = float(np.clip(opacity, 0.05, 1.0))
    kwargs: dict[str, Any] = (
        _accepted(plotting.plot_stat_map, transparency=transparency) if transparency < 1.0 else {}
    )
    plotting.plot_stat_map(
        thresholded,
        bg_img=load_mni152_template(),
        display_mode=str(display_mode or "ortho"),
        cut_coords=cut_coords,
        threshold=band_lo if corrp else 1e-6,
        cmap=cmap if corrp else "RdBu_r",
        colorbar=True,
        symmetric_cbar=not corrp,
        vmin=band_lo if corrp else None,
        vmax=band_hi if corrp else None,
        title=title or (f"{contrast} — {kind}" if contrast else kind),
        figure=fig,
        **kwargs,
    )
    return _unlink_axes(fig)


def plot_voxelwise_result(
    result: Any,
    *,
    contrast: str,
    kind: str = "",
    view: str = "surface",
    alpha: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    interactive: bool = False,
    cmap: str = "hot",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    surface: str = "pial",
    display_mode: str = "ortho",
    cut_coords: Any = None,
    opacity: float = 1.0,
    title: str = "",
) -> tuple[Any, str]:
    """``(figure, caption)`` for one contrast of a :class:`~nvitk.measure.voxelwise.VoxelwiseResult`.

    The single entry point both viewers call, so neither can pick a threshold, a colormap or a
    window the other does not.
    """
    kind = str(kind or result.primary_kind())
    path = result.map_path(kind, contrast)
    image = load_map(path)
    band_lo, band_hi = resolve_band(lo=lo, hi=hi, alpha=alpha)
    n_significant = significant_voxels(image, kind=kind, lo=band_lo, hi=band_hi)
    manifest = getattr(result, "manifest", {}) or {}

    view_key = str(view or "surface").strip().lower()
    if view_key == "glass":
        figure = plot_voxelwise_glass(
            image, kind=kind, contrast=contrast, lo=band_lo, hi=band_hi, cmap=cmap,
            display_mode=display_mode, cut_coords=cut_coords,
            # The glass brain's own default is 0.7; a caller leaving opacity alone should get that
            # rather than a fully opaque projection that hides the brain outline.
            opacity=0.7 if opacity >= 1.0 else opacity,
            title=title,
        )
        rotatable = False
    elif view_key == "slices":
        figure = plot_voxelwise_slices(
            image, kind=kind, contrast=contrast, lo=band_lo, hi=band_hi, cmap=cmap,
            display_mode=display_mode, cut_coords=cut_coords, opacity=opacity, title=title,
        )
        rotatable = False
    else:
        figure = plot_voxelwise_surface(
            image, kind=kind, contrast=contrast, lo=band_lo, hi=band_hi,
            interactive=interactive, cmap=cmap, views=views, hemispheres=hemispheres,
            surface=surface, title=title,
        )
        rotatable = bool(interactive)

    caption = map_caption(
        kind=kind,
        contrast=contrast,
        lo=band_lo,
        hi=band_hi,
        n_significant=n_significant,
        n_subjects=manifest.get("n_subjects"),
        n_perm=manifest.get("n_perm"),
        evs=manifest.get("evs", ()),
        interactive=rotatable,
    )
    if n_significant == 0 and is_corrp_kind(kind):
        caption += (
            " Nothing survives correction — that is a result, not a rendering failure."
            if band_hi >= 1.0
            else " No voxel falls in this window — widen it, or check a lower threshold."
        )
    return figure, caption


__all__ = [
    "ALPHA_CHOICES",
    "DISPLAY_MODE_N_CUTS",
    "GLASS_DISPLAY_MODES",
    "SLICE_DISPLAY_MODES",
    "VIEW_KINDS",
    "VOXELWISE_COLORMAPS",
    "cut_axes_for",
    "display_modes_for",
    "resolve_band",
    "supports_cut_coords",
    "supports_opacity",
    "world_bounds",
    "alpha_to_map_threshold",
    "is_corrp_kind",
    "load_map",
    "map_caption",
    "plot_voxelwise_glass",
    "plot_voxelwise_result",
    "plot_voxelwise_slices",
    "plot_voxelwise_surface",
    "significant_voxels",
    "threshold_image",
]
