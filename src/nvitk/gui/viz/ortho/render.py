"""Turning slices into pixels: contrast, colormaps, label colours, and compositing the layer stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nvitk.gui.labels.visibility import (
    get_label_color,
    is_label_like_layer,
    stored_visible_ids,
)
from nvitk.gui.viz.ortho.geometry import AxisView
from nvitk.gui.viz.ortho.sources import (
    _slice_of,
    SliceCache,
)

def volume_contrast(data: np.ndarray, *, sample: int = 400_000) -> tuple[float, float]:
    """Robust display range for a volume, from a subsample of its finite voxels.

    Computed once for the volume rather than per slice: per-slice percentiles are
    both slower and *wrong*; the window would shift as you scroll, so the same
    tissue changes brightness from one slice to the next.
    """
    arr = np.asarray(data)
    flat = arr.reshape(-1)
    if flat.size > sample:
        flat = flat[:: max(int(flat.size // sample), 1)]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, (1.0, 99.0)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def layer_contrast(layer: Any, data: np.ndarray) -> tuple[float, float] | None:
    """The window to draw *layer* with: its own, when it has one.

    Following the layer means the panels show what the Napari canvas shows: adjust
    brightness on the layer controls and these views track it, instead of staying
    on a percentile window computed once and disagreeing with the canvas from then
    on. ``None`` for label layers, which are drawn through their colour table.
    """
    if is_label_like_layer(layer):
        return None
    limits = getattr(layer, "contrast_limits", None)
    try:
        lo, hi = float(limits[0]), float(limits[1])
    except (TypeError, ValueError, IndexError):
        return volume_contrast(data)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return (lo, hi)
    return volume_contrast(data)


def _grayscale_rgb(plane: np.ndarray, contrast: tuple[float, float] | None = None) -> np.ndarray:
    """Window a 2D intensity slice to RGB uint8 using *contrast* (or its own range)."""
    arr = np.asarray(plane, dtype=np.float32)
    lo, hi = contrast if contrast is not None else volume_contrast(arr)
    if hi <= lo:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    norm = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def label_lut(layer: Any, label_ids: list[int]) -> dict[int, np.ndarray]:
    """RGB uint8 per label id, read once from the layer's own colours."""
    lut: dict[int, np.ndarray] = {}
    for lid in label_ids:
        rgba = get_label_color(layer, int(lid))
        lut[int(lid)] = (np.clip(np.asarray(rgba[:3], dtype=float), 0, 1) * 255).astype(np.uint8)
    return lut


def _label_rgb(plane: np.ndarray, layer: Any, lut: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Colour a 2D label slice, mapping ids through *lut* in one pass."""
    arr = np.rint(np.asarray(plane, dtype=np.float64)).astype(np.int64, copy=False)
    present = [int(v) for v in np.unique(arr) if int(v) != 0]
    if not present:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    colors = lut if lut is not None else label_lut(layer, present)
    # One table indexed by id beats one boolean mask per id.
    table = np.zeros((max(present) + 1, 3), dtype=np.uint8)
    for lid in present:
        table[lid] = colors.get(lid, np.array([255, 255, 255], dtype=np.uint8))
    return table[np.clip(arr, 0, table.shape[0] - 1)]


# RGBA pipeline
#: Entries in a tabulated colormap. Matches the size of the texture Napari's own
#: shader samples, so the two agree to within a quantisation step.
_COLORMAP_STEPS = 256


def layer_colormap(layer: Any) -> Any | None:
    """*layer*'s colormap, or ``None`` for a label layer or one without.

    Label colormaps map *integers* and raise on float input, so they are
    deliberately excluded here: labels go through :func:`label_rgba_lut`.
    """
    if is_label_like_layer(layer):
        return None
    return getattr(layer, "colormap", None)


def layer_gamma(layer: Any) -> float:
    """*layer*'s display gamma, defaulting to 1."""
    try:
        return float(getattr(layer, "gamma", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def colormap_table(colormap: Any, gamma: float = 1.0, size: int = _COLORMAP_STEPS) -> np.ndarray:
    """*colormap* sampled into a ``(size, 4)`` uint8 table, with *gamma* folded in.

    Tabulated once rather than mapped per pixel: ``Colormap.map`` over a 250k-voxel
    slice costs milliseconds per redraw, and the GPU does exactly this (samples a
    256-texel texture), so the result matches what the canvas shows.
    """
    ramp = np.linspace(0.0, 1.0, int(size), dtype=np.float32) ** max(float(gamma), 1e-6)
    try:
        rgba = np.asarray(colormap.map(ramp), dtype=np.float32)
    except Exception:  # noqa: BLE001 (an unusable colormap falls back to grey)
        rgba = np.repeat(ramp[:, None], 4, axis=1)
        rgba[:, 3] = 1.0
    if rgba.ndim != 2 or rgba.shape[1] < 3:
        rgba = np.repeat(ramp[:, None], 4, axis=1)
        rgba[:, 3] = 1.0
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1), dtype=np.float32)], axis=1)
    return (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)


def image_rgba(
    plane: np.ndarray, contrast: tuple[float, float] | None, table: np.ndarray
) -> np.ndarray:
    """Window a 2D intensity slice and colour it through *table*; ``(H, W, 4)`` uint8."""
    arr = np.asarray(plane, dtype=np.float32)
    lo, hi = contrast if contrast is not None else volume_contrast(arr)
    if hi <= lo:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    # Windowed in place. Each of ``nan_to_num``, the subtraction, the division
    # and the clip used to allocate its own copy of the slice; on a 512x512
    # float64 CT that is five megabytes of churn per view per redraw.
    scale = float(table.shape[0] - 1) / (hi - lo)
    norm = np.subtract(arr, lo, dtype=np.float32)
    np.nan_to_num(norm, copy=False, nan=0.0, posinf=hi - lo, neginf=0.0)
    np.multiply(norm, scale, out=norm)
    np.clip(norm, 0.0, float(table.shape[0] - 1), out=norm)
    return table[norm.astype(np.uint16, copy=False)]


def label_rgba_lut(layer: Any, label_ids: list[int]) -> dict[int, np.ndarray]:
    """RGBA uint8 per label id, read from the layer's own colours.

    Alpha is kept, unlike :func:`label_lut`, and a label hidden in the label list
    (nvitk's visibility filter) comes out fully transparent, as on the canvas.
    :func:`get_label_color` returns a hidden label's *own* colour (the one it
    gets back when shown), so the filter is read separately.
    """
    visible = stored_visible_ids(layer)
    shown = None if visible is None else {int(v) for v in visible}
    lut: dict[int, np.ndarray] = {}
    for lid in label_ids:
        rgba = np.clip(np.asarray(get_label_color(layer, int(lid)), dtype=float), 0, 1)
        if shown is not None and int(lid) not in shown:
            rgba = np.array([rgba[0], rgba[1], rgba[2], 0.0])
        lut[int(lid)] = (rgba * 255).astype(np.uint8)
    return lut


def _label_rgba(
    plane: np.ndarray,
    layer: Any,
    lut: dict[int, np.ndarray] | None = None,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """Colour a 2D label slice to ``(H, W, 4)`` uint8; background transparent.

    A Labels layer drawn as outlines (its ``contour``) is outlined here too:
    Napari's own contour, so the views match the canvas, and whatever the canvas
    is doing: in a 3D canvas Napari cannot draw contours, the views still can.
    """
    raw = np.asarray(plane)
    # An integer label slice is already what the lookup wants. Rounding it
    # through float64 and back cost two full-size temporaries per view per
    # redraw for a value that could not have changed.
    if np.issubdtype(raw.dtype, np.integer):
        arr = raw
    else:
        arr = np.rint(np.asarray(raw, dtype=np.float64)).astype(np.int64, copy=False)
    contour = int(getattr(layer, "contour", 0) or 0)
    if contour > 0 and arr.ndim == 2 and arr.size:
        from napari.layers.labels._labels_utils import get_contours

        arr = get_contours(arr, contour, 0)
    if table is None:
        present = [int(v) for v in np.unique(arr) if int(v) != 0]
        if not present:
            return np.zeros((*arr.shape, 4), dtype=np.uint8)
        table = label_color_table(present, lut if lut is not None else label_rgba_lut(layer, present))
    if table.shape[0] <= 1:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    idx = np.clip(arr, 0, table.shape[0] - 1)
    return table[idx]


def label_color_table(label_ids: Sequence[int], lut: dict[int, np.ndarray]) -> np.ndarray:
    """A ``(max_id + 1, 4)`` uint8 lookup table for *label_ids*.

    Built once per layer rather than per slice: it depends only on the ids and
    their colours, neither of which changes while scrolling.
    """
    ids = [int(v) for v in label_ids if int(v) != 0]
    if not ids:
        return np.zeros((1, 4), dtype=np.uint8)
    table = np.zeros((max(ids) + 1, 4), dtype=np.uint8)
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    for lid in ids:
        table[lid] = lut.get(lid, white)
    return table


def slice_to_rgba(
    layer: Any,
    data: np.ndarray,
    axis: int,
    index: int,
    view: AxisView | None = None,
    *,
    contrast: tuple[float, float] | None = None,
    table: np.ndarray | None = None,
    lut: dict[int, np.ndarray] | None = None,
    cache: SliceCache | None = None,
    label_table: np.ndarray | None = None,
) -> np.ndarray:
    """One orthogonal slice as ``(H, W, 4)`` uint8, ready to composite."""
    raw = cache.slice(axis, index) if cache is not None else _slice_of(data, axis, index)
    plane = _oriented(raw, axis, view)
    if is_label_like_layer(layer):
        return _label_rgba(plane, layer, lut, label_table)
    if table is None:
        table = colormap_table(layer_colormap(layer), layer_gamma(layer))
    return image_rgba(plane, contrast, table)


def composite(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
) -> np.ndarray:
    """Blend RGBA *planes* bottom-to-top into one RGB uint8 image.

    Walks in layer-list order, the way Napari draws, and implements its blending
    modes (``napari.layers.base._base_constants``). This reproduces the 2D canvas;
    the depth-test difference between ``translucent`` and ``translucent_no_depth``
    only shows in 3D.
    """
    if not len(planes):
        return np.zeros((1, 1, 3), dtype=np.uint8)
    shape = np.asarray(planes[0]).shape[:2]
    out = np.zeros((*shape, 3), dtype=np.float32)
    # Written in place throughout. Every arithmetic step here used to allocate a
    # fresh (H, W, 3) float array, so a two-layer composite of a 512x512 slice
    # churned tens of megabytes per view per redraw for a result of 768 KB.
    scratch = np.empty((*shape, 3), dtype=np.float32)
    alpha = np.empty((*shape, 1), dtype=np.float32)
    for plane, blending, opacity in zip(planes, blendings, opacities):
        src = np.asarray(plane)
        if src.shape[:2] != shape:
            continue
        rgb = src[..., :3]
        np.multiply(src[..., 3:4], float(np.clip(opacity, 0.0, 1.0)) / 255.0, out=alpha,
                    dtype=np.float32, casting="unsafe")
        mode = str(blending or "translucent")
        if mode == "opaque":
            np.copyto(out, rgb, where=alpha > 0, casting="unsafe")
        elif mode == "additive":
            np.multiply(rgb, alpha, out=scratch, dtype=np.float32, casting="unsafe")
            np.add(out, scratch, out=out)
        elif mode == "minimum":
            np.copyto(scratch, out)
            np.copyto(scratch, rgb, where=alpha > 0, casting="unsafe")
            np.minimum(out, scratch, out=out)
        elif mode == "multiplicative":
            np.multiply(rgb, 1.0 / 255.0, out=scratch, dtype=np.float32, casting="unsafe")
            np.multiply(out, scratch, out=out)
        else:  # translucent, translucent_no_depth
            np.multiply(rgb, alpha, out=scratch, dtype=np.float32, casting="unsafe")
            np.subtract(1.0, alpha, out=alpha)
            np.multiply(out, alpha, out=out)
            np.add(out, scratch, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8, copy=False)


def _oriented(plane: np.ndarray, axis: int, view: AxisView | None) -> np.ndarray:
    """Transpose / flip a raw slice into *view*'s anatomical orientation."""
    if view is None:
        return plane
    rest = [i for i in range(3) if i != axis]
    # ``np.take`` leaves the surviving axes in array order; put the view's row
    # axis first when it is the other one.
    if view.rows != rest[0]:
        plane = plane.T
    if view.flip_rows:
        plane = plane[::-1, :]
    if view.flip_cols:
        plane = plane[:, ::-1]
    return plane


def blend_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """Composite *overlay* over *base*, leaving *base* showing where overlay is black.

    A label overlay is black exactly where it has no label, so treating black as
    transparent is what makes "raw image plus segmentation" read correctly.
    """
    if overlay.shape != base.shape:
        return base
    mask = overlay.any(axis=2)
    if not mask.any():
        return base
    out = base.astype(np.float32)
    a = float(np.clip(opacity, 0.0, 1.0))
    out[mask] = out[mask] * (1.0 - a) + overlay[mask].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def slice_to_rgb(
    layer: Any,
    data: np.ndarray,
    axis: int,
    index: int,
    view: AxisView | None = None,
    *,
    contrast: tuple[float, float] | None = None,
    lut: dict[int, np.ndarray] | None = None,
    cache: SliceCache | None = None,
) -> np.ndarray:
    """Render one orthogonal slice of *layer* as an RGB uint8 image.

    With a *view*, the slice is transposed and flipped into that view's anatomical
    orientation; without one it is drawn in raw array-axis order.
    """
    raw = cache.slice(axis, index) if cache is not None else _slice_of(data, axis, index)
    plane = _oriented(raw, axis, view)
    if is_label_like_layer(layer):
        return _label_rgb(plane, layer, lut)
    return _grayscale_rgb(plane, contrast)


# interpolation


def layer_interpolation_smooth(layer: Any) -> bool:
    """Whether *layer* is drawn smoothed when magnified (its napari 2D interpolation).

    Labels never are (napari draws them nearest-neighbour, whatever the image
    layers under them do), and an Image layer follows ``interpolation2d``
    (napari's own default is ``nearest``).
    """
    if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
        return False
    return str(getattr(layer, "interpolation2d", "nearest") or "nearest").lower() != "nearest"


def layer_interpolation_order(layer: Any) -> int:
    """Spline order matching *layer*'s interpolation: 0 nearest, 1 linear, 3 the cubic family."""
    if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
        return 0
    mode = str(getattr(layer, "interpolation2d", "nearest") or "nearest").lower()
    if mode == "nearest":
        return 0
    if mode in ("linear", "bilinear"):
        return 1
    return 3


@dataclass(frozen=True)
class PaintRun:
    """A run of consecutive layers drawn with one interpolation, ready to paint.

    *mode* says how it lands on what is under it: ``"base"`` is opaque RGB (the
    bottom run), ``"over"`` premultiplied RGBA painted source-over, ``"plus"`` RGB
    added on top (additive layers).
    """

    image: np.ndarray
    smooth: bool
    mode: str = "base"


def composite_over_rgba(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
) -> np.ndarray:
    """Translucent/opaque *planes* over transparency -> premultiplied RGBA uint8."""
    shape = np.asarray(planes[0]).shape[:2]
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    acc = np.zeros((*shape, 1), dtype=np.float32)
    for plane, blending, opacity in zip(planes, blendings, opacities):
        src = np.asarray(plane)
        if src.shape[:2] != shape:
            continue
        a = src[..., 3:4].astype(np.float32) * (float(np.clip(opacity, 0.0, 1.0)) / 255.0)
        colour = src[..., :3].astype(np.float32)
        if str(blending) == "opaque":
            hit = a > 0
            np.copyto(rgb, colour, where=hit)
            np.copyto(acc, 1.0, where=hit)
            continue
        rgb = colour * a + rgb * (1.0 - a)
        acc = a + acc * (1.0 - a)
    out = np.empty((*shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3:] = np.clip(acc * 255.0, 0, 255).astype(np.uint8)
    return out


def composite_runs(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
    smooth: Sequence[bool],
) -> list[PaintRun]:
    """Split the layer stack into runs that share an interpolation, each composited.

    The bottom run is composited exactly as :func:`composite` does; each later run
    becomes premultiplied RGBA (translucent/opaque layers) or an additive RGB, so
    Qt can scale every run with its own filter and stack them in order. A later
    run mixing other blend modes (minimum, multiplicative) cannot be expressed
    that way, so the whole stack then falls back to one run, smoothed if any layer
    in it is.
    """
    if not len(planes):
        return [PaintRun(np.zeros((1, 1, 3), dtype=np.uint8), False)]
    groups: list[list[int]] = []
    for i, flag in enumerate(smooth):
        if groups and bool(smooth[groups[-1][-1]]) == bool(flag):
            groups[-1].append(i)
        else:
            groups.append([i])

    def _pick(indices: list[int]) -> tuple[list, list, list]:
        return ([planes[i] for i in indices], [blendings[i] for i in indices],
                [opacities[i] for i in indices])

    runs = [PaintRun(composite(*_pick(groups[0])), bool(smooth[groups[0][0]]), "base")]
    for group in groups[1:]:
        p, b, o = _pick(group)
        modes = {str(m or "translucent") for m in b}
        if modes <= {"translucent", "translucent_no_depth", "opaque"}:
            runs.append(PaintRun(composite_over_rgba(p, b, o), bool(smooth[group[0]]), "over"))
        elif modes == {"additive"}:
            runs.append(PaintRun(composite(p, b, o), bool(smooth[group[0]]), "plus"))
        else:
            return [PaintRun(composite(planes, blendings, opacities), any(smooth), "base")]
    return runs


def flatten_runs(runs: Sequence[PaintRun]) -> np.ndarray:
    """The runs merged at native resolution: what the view shows, before scaling."""
    out = np.asarray(runs[0].image, dtype=np.float32)[..., :3].copy()
    for run in runs[1:]:
        img = np.asarray(run.image, dtype=np.float32)
        if run.mode == "over":
            a = img[..., 3:4] / 255.0
            out = img[..., :3] + out * (1.0 - a)
        elif run.mode == "plus":
            out = out + img[..., :3]
    return np.clip(out, 0, 255).astype(np.uint8)
