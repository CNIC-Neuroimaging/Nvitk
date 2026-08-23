"""
A 3-D napari scene for a voxelwise result: suprathreshold voxels inside a translucent brain.

The flat Image layers the results loader adds answer "is there an effect"; they do not answer
*where*, which is the question a voxelwise analysis exists to ask. Scrolling a 2-D slice stack to
find out which clusters are cortical, which are deep, and how they relate to one another is the
wrong instrument for that.

This puts the whole result in one view: a semi-transparent shell of the brain, with the
suprathreshold voxels rendered inside it — as solid iso-surfaces (where are the clusters) or as
points coloured by value (how strong is each one). The shell is drawn in **both** modes; without
it the clusters float in a void with nothing to locate them against.

Any map, not just 1−p
---------------------
``randomise`` writes several maps per contrast and they are not interchangeable. A ``*_corrp_*``
map holds 1−p in ``[0, 1]`` and is thresholded from below; a ``tstat`` map holds a *signed*
statistic and is thresholded on magnitude, two-sided. :func:`suggest_band` picks a defensible
default per kind from the data itself, and :func:`scene_mask` applies the right comparison — so
switching the map kind does not silently keep a window that means nothing for it.

Space
-----
Everything is built in **voxel-index** space and placed by the reference layer's affine, the
convention both existing ``add_surface`` call sites in this repo follow (:mod:`nvitk.gui.app`,
:mod:`nvitk.gui.viz.morpho_viz`). Shell and clusters therefore co-register by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

log = Logger()

SHELL_LAYER = "brain shell"

#: Layer-name prefix for everything this module adds, so a redraw clears its own output without
#: touching the template or anything the user opened alongside.
CLUSTER_PREFIX = "voxelwise:"

#: How the suprathreshold voxels are drawn.
SCENE_MODES: tuple[tuple[str, str], ...] = (
    ("Iso-surfaces (cluster shapes)", "surface"),
    ("Points (coloured by value)", "points"),
)

#: Sequential colormaps offered for the clusters.
SCENE_COLORMAPS: tuple[str, ...] = (
    "hot", "viridis", "magma", "inferno", "plasma", "turbo", "cividis",
)

#: Points beyond this are subsampled by descending magnitude. A whole-brain result at 2 mm can
#: pass 200 000 voxels, and napari redraws every one on every camera move.
MAX_POINTS = 30000


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SceneSpec:
    """Everything the 3-D scene needs, in one object.

    A frozen spec rather than a dozen keyword arguments so the configuration window and the
    builder cannot drift: the dialog produces one of these, the builder consumes it, and adding a
    control is one field in one place.
    """

    out_dir: Path
    kind: str = ""
    contrasts: tuple[str, ...] = ()
    lo: float = 0.95
    hi: float = 1.0
    mode: str = "surface"
    colormap: str = "hot"
    point_size: float = 2.5
    show_shell: bool = True
    shell_opacity: float = 0.22
    shell_step: int = 2
    cluster_opacity: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)


def suggest_band(data: np.ndarray, kind: str) -> tuple[float, float]:
    """A defensible default window for *kind*, read from the data rather than assumed.

    For a 1−p map that is the conventional ``0.95 … 1.0``. For a signed statistic there is no
    conventional cut at all, so this offers the 99th percentile of ``|value|`` up to its maximum —
    a starting point that shows something on any map, and which the caller is expected to change.
    """
    from nvitk.stats.voxelwise_map import is_corrp_kind

    if is_corrp_kind(kind):
        return 0.95, 1.0
    finite = np.abs(data[np.isfinite(data) & (data != 0)])
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 99.0)), float(finite.max())


def value_range(data: np.ndarray, kind: str) -> tuple[float, float]:
    """The full range a threshold control should span for *kind*."""
    from nvitk.stats.voxelwise_map import is_corrp_kind

    if is_corrp_kind(kind):
        return 0.0, 1.0
    finite = np.abs(data[np.isfinite(data)])
    return (0.0, float(finite.max()) if finite.size else 1.0)


def scene_mask(data: np.ndarray, kind: str, lo: float, hi: float) -> np.ndarray:
    """Which voxels the scene draws.

    A 1−p map is one-sided and lives in ``[0, 1]``, so the window is on the value itself. A
    ``tstat`` map is signed, so the window is on ``|value|`` and keeps both tails — a negative
    effect is as real as a positive one, and thresholding the raw value would silently drop half
    the result.
    """
    from nvitk.stats.voxelwise_map import is_corrp_kind

    if is_corrp_kind(kind):
        return (data > float(lo)) & (data <= float(hi))
    magnitude = np.abs(data)
    return np.isfinite(data) & (magnitude >= float(lo)) & (magnitude <= float(hi))


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────
def _mesh_from_mask(mask: np.ndarray, *, step_size: int = 1) -> tuple[np.ndarray, np.ndarray] | None:
    """Marching-cubes *mask* in voxel-index space, or ``None`` when it encloses nothing.

    ``world_space=False`` is deliberate — see the module docstring. A mask with no interior (a
    single voxel, or an empty band) makes skimage raise, which this turns into ``None``.
    """
    from nvitk.meshlab.marching_cubes import marching_cubes_binary

    if not np.any(mask):
        return None
    try:
        mesh = marching_cubes_binary(
            mask.astype(np.uint8), level=0.5, step_size=int(step_size), world_space=False
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Marching cubes failed: %s", exc)
        return None
    if mesh is None:
        return None
    surface = mesh.to_napari_surface()
    verts = np.asarray(surface["vertices"])
    faces = np.asarray(surface["faces"])
    if verts.size == 0 or faces.size == 0:
        return None
    return verts, faces


def resolve_shell_mask(result: Any, reference_path: Path) -> np.ndarray | None:
    """The brain mask to build the shell from, whatever the analysis left behind.

    Three sources, in order, because the obvious one is often absent: ``run_voxelwise`` only writes
    ``mask.nii.gz`` into the results folder when it *derived* the mask itself. Pass ``--mask`` and
    the file stays wherever it was, so a folder from a real run frequently has no mask in it at
    all — which is why the shell silently never appeared.

    1. ``mask.nii.gz`` beside the maps.
    2. The path recorded in ``manifest.json``, if it is still readable from here.
    3. Nilearn's MNI152 brain mask resampled onto the map's own grid — always available, and on an
       MNI-normalised result it is what the analysis mask approximates anyway.
    """
    import nibabel as nib

    out_dir = Path(result.out_dir)

    local = out_dir / "mask.nii.gz"
    if local.is_file():
        log.info(f"Shell from {local.name}")
        return np.asarray(nib.load(str(local)).dataobj) > 0

    recorded = str((getattr(result, "manifest", {}) or {}).get("mask") or "").strip()
    if recorded:
        candidate = Path(recorded)
        if candidate.is_file():
            log.info(f"Shell from the manifest's mask: {candidate}")
            return np.asarray(nib.load(str(candidate)).dataobj) > 0
        log.info(f"Manifest mask {candidate} is not readable here; deriving one instead.")

    try:
        from nilearn.datasets import load_mni152_brain_mask
        from nilearn.image import resample_to_img

        reference = nib.load(str(reference_path))
        resampled = resample_to_img(
            load_mni152_brain_mask(), reference,
            interpolation="nearest", force_resample=True, copy_header=True,
        )
        log.info("Shell from the MNI152 brain mask, resampled onto the result's grid")
        return np.asarray(resampled.dataobj) > 0
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not derive a brain mask for the shell: {exc}")
        return None


def _subsample(coords: np.ndarray, values: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the *limit* strongest voxels by magnitude.

    Dropping the weakest is the honest way to thin a point cloud — a random sample would make a
    dense cluster look sparse.
    """
    if len(coords) <= limit:
        return coords, values
    keep = np.argsort(np.abs(values))[::-1][:limit]
    log.info(f"Subsampled {len(coords)} → {limit} point(s) by descending magnitude")
    return coords[keep], values[keep]


# ──────────────────────────────────────────────────────────────────────────────
# Scene
# ──────────────────────────────────────────────────────────────────────────────
def clear_voxelwise_layers(viewer: Any) -> None:
    """Remove the shell and every cluster layer this module added, and nothing else."""
    for layer in list(viewer.layers):
        name = str(getattr(layer, "name", ""))
        if name == SHELL_LAYER or name.startswith(CLUSTER_PREFIX):
            viewer.layers.remove(layer)


def build_scene(viewer: Any, spec: SceneSpec) -> tuple[list[Any], str]:
    """Draw *spec* into *viewer*; return ``(layers, caption)``.

    One call does the whole thing — load, threshold, shell, clusters, 3-D view — so there is no
    state to get out of step between a "load" step and a "show" step.
    """
    import nibabel as nib

    from nvitk.gui.core.spatial import layer_spatial_kwargs
    from nvitk.gui.io.napari_io import open_paths_with_nvitk
    from nvitk.measure.voxelwise import load_voxelwise_result
    from nvitk.stats.voxelwise_map import is_corrp_kind, map_caption

    result = load_voxelwise_result(spec.out_dir)
    kind = str(spec.kind or result.primary_kind())
    names = list(spec.contrasts) or list(result.contrast_names)
    if not names:
        raise ValueError(f"{result.out_root.name} has no contrasts to draw.")
    unknown = [n for n in names if n not in result.contrast_names]
    if unknown:
        raise ValueError(f"No contrast named {unknown!r}. Available: {result.contrast_names}")

    clear_voxelwise_layers(viewer)
    added: list[Any] = []

    # The affine has to come from one of *these* maps — the MNI template sits on its own grid.
    # Loading the first contrast doubles as the reference; it stays hidden because the volume is
    # not part of the 3-D picture, only its geometry is.
    first_path = result.map_path(kind, names[0])
    reference = None
    for layer in open_paths_with_nvitk(viewer, first_path):
        layer.name = f"{CLUSTER_PREFIX}{names[0]} ({kind})"
        layer.visible = False
        reference = layer
        added.append(layer)
    if reference is None:
        raise RuntimeError(f"Could not load {first_path} as a layer.")
    spatial = layer_spatial_kwargs(reference)

    # ---- shell, drawn in every mode -----------------------------------------
    if spec.show_shell:
        mask = resolve_shell_mask(result, first_path)
        shell = _mesh_from_mask(mask, step_size=spec.shell_step) if mask is not None else None
        if shell is not None:
            verts, faces = shell
            added.append(
                viewer.add_surface(
                    (verts, faces),
                    name=SHELL_LAYER,
                    opacity=float(np.clip(spec.shell_opacity, 0.02, 1.0)),
                    # 'translucent_no_depth' rather than 'translucent': the latter writes to the
                    # depth buffer, so geometry *inside* the shell is culled — which would hide
                    # exactly what this scene exists to show.
                    blending="translucent_no_depth",
                    colormap="gray",
                    shading="smooth",
                    **spatial,
                )
            )
        else:
            log.warning("No brain mask available; drawing clusters without a shell.")

    # ---- clusters ------------------------------------------------------------
    total = 0
    for name in names:
        data = np.asarray(nib.load(str(result.map_path(kind, name))).dataobj, dtype=float)
        band = scene_mask(data, kind, spec.lo, spec.hi)
        n = int(band.sum())
        total += n
        if n == 0:
            log.info(f"Contrast {name!r}: nothing in {spec.lo:g}–{spec.hi:g}; nothing to draw.")
            continue

        if str(spec.mode).lower() == "points":
            coords = np.argwhere(band).astype(float)
            values = data[band]
            coords, values = _subsample(coords, values, MAX_POINTS)
            layer = viewer.add_points(
                coords,
                name=f"{CLUSTER_PREFIX}{name} points",
                size=float(spec.point_size),
                features={"value": values},
                face_color="value",
                face_colormap=spec.colormap,
                face_contrast_limits=(float(values.min()), float(values.max())),
                border_width=0.0,
                opacity=float(np.clip(spec.cluster_opacity, 0.05, 1.0)),
                blending="translucent",
                **spatial,
            )
            try:
                from nvitk.gui.viz.layers import install_points_style_sync

                install_points_style_sync(layer)
            except Exception as exc:  # noqa: BLE001
                log.debug("Points style sync unavailable: %s", exc)
        else:
            built = _mesh_from_mask(band)
            if built is None:
                log.info(
                    f"Contrast {name!r}: {n} voxel(s) but no closed surface — the clusters are "
                    "too thin to contour. Use the points mode to see them."
                )
                continue
            verts, faces = built
            layer = viewer.add_surface(
                (verts, faces),
                name=f"{CLUSTER_PREFIX}{name} clusters",
                opacity=float(np.clip(spec.cluster_opacity, 0.05, 1.0)),
                blending="translucent",
                colormap=spec.colormap,
                shading="smooth",
                **spatial,
            )
        added.append(layer)
        log.info(f"Contrast {name!r}: {n} voxel(s) in {spec.lo:g}–{spec.hi:g} as {spec.mode}")

    set_3d_view(viewer)

    manifest = getattr(result, "manifest", {}) or {}
    if is_corrp_kind(kind):
        caption = map_caption(
            kind=kind, contrast=", ".join(names), lo=spec.lo, hi=spec.hi,
            n_significant=total, n_subjects=manifest.get("n_subjects"),
            n_perm=manifest.get("n_perm"), evs=manifest.get("evs", ()),
        )
    else:
        # map_caption's band wording is written for 1−p; a signed statistic needs its own sentence
        # or the figure would claim a p-value it does not carry.
        caption = (
            f"Colour is the {kind} value, shown where |value| is between {spec.lo:g} and "
            f"{spec.hi:g} — both tails. {total} voxel(s). Contrast: {', '.join(names)}."
        )
    if total == 0:
        caption += " Nothing falls in this window — widen it."
    return added, caption


def set_3d_view(viewer: Any) -> int | None:
    """Switch the viewer to 3-D, returning the previous ``ndisplay`` so a caller can restore it."""
    try:
        previous = int(viewer.dims.ndisplay)
        viewer.dims.ndisplay = 3
        return previous
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not switch to the 3-D view: %s", exc)
        return None


def map_data(out_dir: str | Path, kind: str = "", contrast: str = "") -> tuple[np.ndarray, str, list[str]]:
    """``(data, kind, contrast_names)`` for one map — what a dialog needs to pick sensible bounds."""
    import nibabel as nib

    from nvitk.measure.voxelwise import load_voxelwise_result

    result = load_voxelwise_result(out_dir)
    kind = str(kind or result.primary_kind())
    names = list(result.contrast_names)
    name = contrast if contrast in names else (names[0] if names else "")
    data = np.asarray(nib.load(str(result.map_path(kind, name))).dataobj, dtype=float)
    return data, kind, names


__all__ = [
    "CLUSTER_PREFIX",
    "MAX_POINTS",
    "SCENE_COLORMAPS",
    "SCENE_MODES",
    "SHELL_LAYER",
    "SceneSpec",
    "build_scene",
    "clear_voxelwise_layers",
    "map_data",
    "resolve_shell_mask",
    "scene_mask",
    "set_3d_view",
    "suggest_band",
    "value_range",
]
