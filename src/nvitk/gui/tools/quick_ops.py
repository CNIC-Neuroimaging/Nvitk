"""Quick image operations for the command palette — the ImageJ-style basics.

Adjusting a window, thresholding by eye, projecting, rotating, cropping to a
mask: things you do constantly while looking at an image, which do not warrant
walking through the tool form each time. Each one acts on the active layer and
either adjusts it in place or adds one new layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.core.spatial import layer_spatial_kwargs


@dataclass(frozen=True)
class OpParam:
    """One option a quick operation asks for before it runs."""

    name: str
    label: str
    #: ``"float"``, ``"int"`` or ``"choice"``.
    kind: str
    default: Any
    minimum: float = 0.0
    maximum: float = 1.0
    decimals: int = 4
    choices: tuple[tuple[str, Any], ...] = ()
    hint: str = ""


def _intensity_range(data: np.ndarray) -> tuple[float, float]:
    """Robust low/high of *data*, for a slider that lands on useful values."""
    flat = data.reshape(-1)
    if flat.size > 400_000:
        flat = flat[:: max(int(flat.size // 400_000), 1)]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def threshold_params(viewer: Any) -> tuple[OpParam, ...]:
    """A threshold slider spanning the active layer's intensity range."""
    layer, data = _require_array_layer(viewer)
    lo, hi = _intensity_range(data)
    limits = getattr(layer, "contrast_limits", None)
    # Start where the display already is: the window the user has been looking
    # through is usually close to the cut they want.
    start = float(limits[0]) if limits else (lo + hi) / 2.0
    return (
        OpParam(
            name="value",
            label="Threshold",
            kind="float",
            default=float(np.clip(start, lo, hi)),
            minimum=lo,
            maximum=hi,
            hint=f"Keeps voxels ≥ the threshold.  Range {lo:.4g} … {hi:.4g}",
        ),
    )


def gaussian_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Sigma for the Gaussian blur."""
    return (
        OpParam("sigma", "Sigma (voxels)", "float", 1.0, minimum=0.1, maximum=20.0,
                decimals=2, hint="Larger sigma, smoother result."),
    )


def median_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Kernel size for the median filter."""
    return (
        OpParam("size", "Kernel size", "int", 3, minimum=2, maximum=15,
                hint="Odd sizes keep the image centred."),
    )


def projection_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Projection kind and the axis to collapse."""
    return (
        OpParam("how", "Projection", "choice", "max",
                choices=(("Maximum", "max"), ("Mean", "mean"),
                         ("Minimum", "min"), ("Sum", "sum"))),
        OpParam("axis", "Along axis", "choice", 0,
                choices=(("0", 0), ("1", 1), ("2", 2)),
                hint="The axis that is collapsed away."),
    )


def rotate_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Rotation plane and how many quarter turns."""
    return (
        OpParam("axis", "Perpendicular to axis", "choice", 0,
                choices=(("0", 0), ("1", 1), ("2", 2))),
        OpParam("times", "Quarter turns", "choice", 1,
                choices=(("90°", 1), ("180°", 2), ("270°", 3))),
    )


def dtype_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Target data type."""
    return (
        OpParam("dtype", "Convert to", "choice", "float32",
                choices=tuple((name, name) for name in sorted(_DTYPES)),
                hint="Integer targets rescale into range rather than truncating."),
    )


def crop_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Padding to leave around the content bounding box."""
    return (
        OpParam("pad", "Padding (voxels)", "int", 0, minimum=0, maximum=50),
    )


def contrast_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Percentile window for the auto-contrast."""
    return (
        OpParam("low", "Low percentile", "float", 1.0, minimum=0.0, maximum=49.0, decimals=2),
        OpParam("high", "High percentile", "float", 99.0, minimum=51.0, maximum=100.0, decimals=2),
    )


def _active(viewer: Any) -> Any:
    """The active layer, or ``None`` when the viewer is empty."""
    if not getattr(viewer, "layers", None):
        return None
    return viewer.layers.selection.active or viewer.layers[-1]


def _require_array_layer(viewer: Any) -> tuple[Any, np.ndarray]:
    """The active layer and its data, raising if it is not an array layer."""
    layer = _active(viewer)
    data = getattr(layer, "data", None)
    if layer is None or data is None:
        raise ValueError("Select an image or labels layer first.")
    return layer, to_numpy(data)


def _add(viewer: Any, layer: Any, data: np.ndarray, suffix: str, *, labels: bool = False) -> Any:
    """Add *data* beside *layer*, carrying its spatial metadata."""
    kwargs = {"name": f"{getattr(layer, 'name', 'layer')}_{suffix}", **layer_spatial_kwargs(layer)}
    if labels:
        return viewer.add_labels(np.asarray(data).astype(np.int32, copy=False), **kwargs)
    return viewer.add_image(data, **kwargs)


# ── contrast ──────────────────────────────────────────────────────────────────
def auto_contrast(viewer: Any, *, low: float = 1.0, high: float = 99.0) -> str:
    """Set the active layer's contrast limits to a robust percentile window.

    The equivalent of ImageJ's *Auto* in Brightness/Contrast: it moves the display
    window only, never the data, so it is always safe and always reversible.
    """
    layer, _data = _require_array_layer(viewer)
    lo, hi = contrast_window(layer, low, high)
    layer.contrast_limits = (lo, hi)
    return f"Contrast set to [{lo:.4g}, {hi:.4g}] on “{layer.name}”."


def contrast_window(layer: Any, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    """The percentile window :func:`auto_contrast` would apply to *layer*."""
    data = layer_data(layer)
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.reshape(-1)
    if finite.size == 0:
        raise ValueError("Layer has no finite voxels to window.")
    if finite.size > 400_000:
        finite = finite[:: max(int(finite.size // 400_000), 1)]
    lo, hi = (float(v) for v in np.percentile(finite, (low, high)))
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        raise ValueError("Layer is constant; there is no window to set.")
    return lo, hi


def ct_window_params(viewer: Any) -> tuple[OpParam, ...]:
    """Preset picker for :func:`ct_window`, seeded from the active layer."""
    from nvitk.viz.ct_windows import (
        DEFAULT_WINDOW_KEY,
        get_window,
        suggest_window,
        window_keys,
    )

    keys = tuple(window_keys())
    default = DEFAULT_WINDOW_KEY
    try:
        layer = _active(viewer)
        data = layer_data(layer)
        suggested = suggest_window(
            str(getattr(layer, "metadata", {}).get("modality", "") or ""),
            float(np.nanmin(data)),
            float(np.nanmax(data)),
        )
        if suggested:
            default = suggested
    except Exception:  # noqa: BLE001 — a suggestion is a convenience, not a gate
        pass
    return (
        OpParam(
            "preset",
            "Window",
            "choice",
            default,
            choices=tuple((get_window(k).label, k) for k in keys),
            hint="Hounsfield presets. Changes the display window only, never the data.",
        ),
        OpParam("apply_all", "Apply to every image layer", "choice", "no",
                choices=(("no", "no"), ("yes", "yes"))),
    )


def ct_window_limits(preset: str) -> tuple[float, float]:
    """The contrast limits *preset* maps to, for the live preview."""
    from nvitk.viz.ct_windows import limits_for

    return tuple(float(v) for v in limits_for(str(preset)))  # type: ignore[return-value]


def ct_window(viewer: Any, *, preset: str = "brain", apply_all: str = "no") -> str:
    """Set the display window to a named CT preset.

    The same presets the Layers tab's *CT display window* panel offers, reachable
    from the search bar. Like every windowing operation this moves the display
    range only — the voxels are untouched, so it is always reversible.
    """
    from nvitk.viz.ct_windows import get_window

    layer, _data = _require_array_layer(viewer)
    lo, hi = ct_window_limits(preset)
    targets = (
        [l for l in viewer.layers if l.__class__.__name__ == "Image"]
        if str(apply_all) == "yes"
        else [layer]
    )
    for target in targets:
        try:
            target.contrast_limits = (lo, hi)
        except Exception:  # noqa: BLE001 — a layer whose range excludes the window
            continue
    where = f"{len(targets)} image layer(s)" if len(targets) > 1 else f"“{layer.name}”"
    return f"{get_window(preset).label} applied to {where}."


def reset_contrast(viewer: Any) -> str:
    """Restore the active layer's contrast limits to its full data range."""
    layer, data = _require_array_layer(viewer)
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    if hi <= lo:
        raise ValueError("Layer is constant; there is no range to reset to.")
    layer.contrast_limits = (lo, hi)
    return f"Contrast reset to the full range [{lo:.4g}, {hi:.4g}]."


# ── thresholding ──────────────────────────────────────────────────────────────
def layer_data(layer: Any) -> np.ndarray:
    """*layer*'s data as a host array, raising if it has none."""
    data = getattr(layer, "data", None)
    if layer is None or data is None:
        raise ValueError("Select an image or labels layer first.")
    return to_numpy(data)


def threshold_mask(layer: Any, value: float) -> np.ndarray:
    """The mask *layer* produces at *value* — used for the live preview.

    Takes the layer rather than the viewer on purpose: adding the preview layer
    makes *it* the active one, so resolving the source by "whatever is active"
    would threshold the preview on the next slider step.
    """
    return (layer_data(layer) >= float(value)).astype(np.int32)


def threshold_at_display(viewer: Any, value: float | None = None) -> str:
    """Binarise the active layer at *value*, defaulting to its lower contrast limit.

    With the options popup this is a live slider: the preview updates as it moves,
    so the cut is chosen by eye instead of typed blind.
    """
    layer, data = _require_array_layer(viewer)
    if value is None:
        limits = getattr(layer, "contrast_limits", None)
        if not limits:
            raise ValueError("This layer has no contrast limits to threshold at.")
        value = float(limits[0])
    mask = (data >= float(value)).astype(np.int32)
    kept = int(mask.sum())
    _add(viewer, layer, mask, "mask", labels=True)
    return f"Thresholded at {float(value):.6g} — {kept:,} voxels kept."


def threshold_otsu(viewer: Any) -> str:
    """Binarise the active layer at an automatically chosen (Otsu) threshold."""
    from skimage.filters import threshold_otsu as _otsu

    layer, data = _require_array_layer(viewer)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Layer has no finite voxels to threshold.")
    cut = float(_otsu(finite))
    mask = (data >= cut).astype(np.int32)
    _add(viewer, layer, mask, "otsu", labels=True)
    return f"Otsu threshold {cut:.6g} — {int(mask.sum()):,} voxels kept."


# ── filters ───────────────────────────────────────────────────────────────────
def gaussian(viewer: Any, sigma: float = 1.0) -> str:
    """Gaussian-blur the active layer into a new layer."""
    from scipy.ndimage import gaussian_filter

    layer, data = _require_array_layer(viewer)
    out = gaussian_filter(data.astype(np.float32, copy=False), sigma=float(sigma))
    _add(viewer, layer, out, f"gauss{sigma:g}")
    return f"Gaussian blur (sigma {sigma:g}) on “{layer.name}”."


def median(viewer: Any, size: int = 3) -> str:
    """Median-filter the active layer into a new layer."""
    from scipy.ndimage import median_filter

    layer, data = _require_array_layer(viewer)
    out = median_filter(data, size=int(size))
    _add(viewer, layer, out, f"median{int(size)}")
    return f"Median filter (size {int(size)}) on “{layer.name}”."


def invert(viewer: Any) -> str:
    """Invert the active layer's intensities into a new layer."""
    layer, data = _require_array_layer(viewer)
    out = float(np.nanmax(data)) + float(np.nanmin(data)) - data
    _add(viewer, layer, out, "inverted")
    return f"Inverted “{layer.name}”."


# ── geometry ──────────────────────────────────────────────────────────────────
def project(viewer: Any, how: str = "max", axis: int = 0) -> str:
    """Project the active layer along *axis* (``max``, ``mean``, ``min``, ``sum``)."""
    layer, data = _require_array_layer(viewer)
    if data.ndim < 3:
        raise ValueError("Projection needs a 3D layer.")
    fns = {"max": np.nanmax, "mean": np.nanmean, "min": np.nanmin, "sum": np.nansum}
    if how not in fns:
        raise ValueError(f"Unknown projection {how!r}. Use one of {sorted(fns)}.")
    out = fns[how](data, axis=int(axis))
    # A projection loses an axis, so the source's 3D affine no longer applies.
    viewer.add_image(out, name=f"{getattr(layer, 'name', 'layer')}_{how}proj")
    return f"{how.title()} projection along axis {axis} of “{layer.name}”."


def rotate90(viewer: Any, axis: int = 0, times: int = 1) -> str:
    """Rotate the active layer 90° *times* within the plane perpendicular to *axis*."""
    layer, data = _require_array_layer(viewer)
    if data.ndim < 3:
        raise ValueError("Rotation needs a 3D layer.")
    plane = tuple(i for i in range(3) if i != int(axis))
    out = np.rot90(data, k=int(times), axes=plane)
    _add(viewer, layer, out, f"rot{90 * int(times)}", labels=_is_labels(layer))
    return f"Rotated “{layer.name}” by {90 * int(times)}° in plane {plane}."


def crop_bounds(data: np.ndarray, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` voxel bounds of *data*'s non-zero content, grown by *pad*."""
    nonzero = np.argwhere(data != 0)
    if nonzero.size == 0:
        raise ValueError("Layer is empty; there is nothing to crop to.")
    lo = np.maximum(nonzero.min(axis=0) - int(pad), 0)
    hi = np.minimum(nonzero.max(axis=0) + 1 + int(pad), np.asarray(data.shape))
    return lo, hi


def cropped_spatial(layer: Any, lo: np.ndarray) -> dict[str, Any]:
    """Spatial kwargs placing a crop starting at voxel *lo* back where it came from.

    Cropping moves the origin, so the source affine alone would put the result in
    the wrong place. Composing it with the offset keeps the crop registered to the
    volume it came out of — which is what lets the preview be shown in place.
    """
    spatial = dict(layer_spatial_kwargs(layer))
    affine = spatial.get("affine")
    if affine is None:
        return spatial
    offset = np.eye(4, dtype=float)
    offset[:3, 3] = np.asarray(lo, dtype=float)[:3]
    spatial["affine"] = np.asarray(affine, dtype=float) @ offset
    return spatial


def crop_to_content(viewer: Any, pad: int = 0) -> str:
    """Crop the active layer to the bounding box of its non-zero voxels, in place."""
    layer, data = _require_array_layer(viewer)
    lo, hi = crop_bounds(data, pad)
    out = data[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    kwargs = {"name": f"{getattr(layer, 'name', 'layer')}_crop", **cropped_spatial(layer, lo)}
    if _is_labels(layer):
        viewer.add_labels(out.astype(np.int32, copy=False), **kwargs)
    else:
        viewer.add_image(out, **kwargs)
    return f"Cropped “{layer.name}” to {tuple(int(v) for v in out.shape)}."


# ── dtype ─────────────────────────────────────────────────────────────────────
_DTYPES: dict[str, Any] = {
    "uint8": np.uint8, "uint16": np.uint16, "int16": np.int16,
    "int32": np.int32, "float32": np.float32, "float64": np.float64,
}


def convert_dtype(viewer: Any, dtype: str = "float32") -> str:
    """Convert the active layer to *dtype*, rescaling into range for integer targets."""
    layer, data = _require_array_layer(viewer)
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown type {dtype!r}. Use one of {sorted(_DTYPES)}.")
    target = _DTYPES[dtype]
    if np.issubdtype(target, np.integer):
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        info = np.iinfo(target)
        if hi > lo:
            # Rescale rather than truncate: a plain cast of a float image to uint8
            # would clip almost everything to 0 or 255.
            scaled = (data - lo) / (hi - lo) * (info.max - info.min) + info.min
        else:
            scaled = np.zeros_like(data)
        out = np.clip(scaled, info.min, info.max).astype(target)
    else:
        out = data.astype(target, copy=False)
    _add(viewer, layer, out, dtype, labels=np.issubdtype(target, np.integer) and _is_labels(layer))
    return f"Converted “{layer.name}” to {dtype}."


# ── exposure ──────────────────────────────────────────────────────────────────
#: How the intensity operations reach scikit-image. ndimage is preferred across
#: nvitk, but it has no exposure module — no gamma, log, sigmoid or histogram
#: equalisation — so these are skimage's, moved to host at the boundary.
def _as_float01(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """*data* scaled into ``[0, 1]`` with the window needed to undo it.

    skimage's exposure functions are defined on that range; feeding raw Hounsfield
    units to ``adjust_gamma`` raises on the negatives and silently distorts
    everything else.
    """
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Layer has no finite voxels.")
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        raise ValueError("Layer is constant; there is nothing to adjust.")
    return np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0), lo, hi


def adjust_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`adjust_intensity`."""
    return (
        OpParam("method", "Curve", "choice", "gamma",
                choices=(("Gamma", "gamma"), ("Logarithmic", "log"), ("Sigmoid", "sigmoid"))),
        OpParam("strength", "Gamma / gain", "float", 1.0, minimum=0.05, maximum=5.0,
                decimals=2,
                hint="Gamma: <1 brightens, >1 darkens. Gain for log and sigmoid."),
        OpParam("cutoff", "Sigmoid cutoff", "float", 0.5, minimum=0.0, maximum=1.0,
                decimals=2, hint="Where the sigmoid crosses its midpoint. Ignored otherwise."),
    )


def adjusted_intensity(
    layer: Any, method: str = "gamma", strength: float = 1.0, cutoff: float = 0.5
) -> np.ndarray:
    """The adjusted image :func:`adjust_intensity` would produce — used for preview."""
    from skimage import exposure

    scaled, lo, hi = _as_float01(layer_data(layer))
    key = str(method).lower()
    if key == "log":
        out = exposure.adjust_log(scaled, gain=float(strength))
    elif key == "sigmoid":
        out = exposure.adjust_sigmoid(scaled, cutoff=float(cutoff), gain=float(strength) * 10.0)
    else:
        out = exposure.adjust_gamma(scaled, gamma=max(float(strength), 1e-3))
    # Back to the source's own range, so the result is comparable with it. Clipped
    # first: adjust_log with a gain above 1 runs past 1.0 by design, which would
    # otherwise push the result outside the range it is meant to be read against.
    clipped = np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)
    return (clipped * (hi - lo) + lo).astype(np.float32)


def adjust_intensity(
    viewer: Any, *, method: str = "gamma", strength: float = 1.0, cutoff: float = 0.5
) -> str:
    """Apply a gamma, logarithmic or sigmoid intensity curve."""
    from nvitk.core.backend import using

    layer, _data = _require_array_layer(viewer)
    with using("numpy"):
        out = adjusted_intensity(layer, method, strength, cutoff)
    _add(viewer, layer, out, str(method))
    return f"{str(method).capitalize()} curve applied to “{layer.name}”."


def equalize_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`equalize_histogram`."""
    return (
        OpParam("method", "Method", "choice", "clahe",
                choices=(("CLAHE (adaptive)", "clahe"), ("Global", "global"))),
        OpParam("clip_limit", "CLAHE clip limit", "float", 0.01, minimum=0.001,
                maximum=0.2, decimals=3,
                hint="Higher lifts more contrast, and more noise with it."),
    )


def equalized_histogram(layer: Any, method: str = "clahe", clip_limit: float = 0.01) -> np.ndarray:
    """The equalised image :func:`equalize_histogram` would produce."""
    from skimage import exposure

    scaled, lo, hi = _as_float01(layer_data(layer))
    if str(method).lower() == "global":
        out = exposure.equalize_hist(scaled)
    else:
        out = exposure.equalize_adapthist(scaled, clip_limit=float(clip_limit))
    return (np.asarray(out, dtype=np.float32) * (hi - lo) + lo).astype(np.float32)


def equalize_histogram(viewer: Any, *, method: str = "clahe", clip_limit: float = 0.01) -> str:
    """Equalise the intensity histogram, globally or adaptively (CLAHE)."""
    from nvitk.core.backend import using

    layer, _data = _require_array_layer(viewer)
    with using("numpy"):
        out = equalized_histogram(layer, method, clip_limit)
    name = "clahe" if str(method).lower() != "global" else "equalized"
    _add(viewer, layer, out, name)
    return f"Histogram equalised ({method}) on “{layer.name}”."


def rescale_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`rescale_intensity`."""
    return (
        OpParam("low", "Low percentile", "float", 1.0, minimum=0.0, maximum=49.0, decimals=2),
        OpParam("high", "High percentile", "float", 99.0, minimum=51.0, maximum=100.0,
                decimals=2),
        OpParam("out_max", "Output maximum", "float", 1.0, minimum=1.0, maximum=65535.0,
                decimals=0, hint="The rescaled data spans 0 to this."),
    )


def rescaled_intensity(
    layer: Any, low: float = 1.0, high: float = 99.0, out_max: float = 1.0
) -> np.ndarray:
    """The rescaled image :func:`rescale_intensity` would produce."""
    from skimage import exposure

    data = np.asarray(layer_data(layer), dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Layer has no finite voxels.")
    lo, hi = (float(v) for v in np.percentile(finite, (float(low), float(high))))
    if hi <= lo:
        raise ValueError("That percentile window is empty.")
    return np.asarray(
        exposure.rescale_intensity(data, in_range=(lo, hi), out_range=(0.0, float(out_max))),
        dtype=np.float32,
    )


def rescale_intensity(
    viewer: Any, *, low: float = 1.0, high: float = 99.0, out_max: float = 1.0
) -> str:
    """Stretch a percentile window of the data onto a fixed output range."""
    from nvitk.core.backend import using

    layer, _data = _require_array_layer(viewer)
    with using("numpy"):
        out = rescaled_intensity(layer, low, high, out_max)
    _add(viewer, layer, out, "rescaled")
    return f"Rescaled “{layer.name}” to [0, {float(out_max):g}]."


def show_histogram(viewer: Any, *, bins: int = 256) -> str:
    """Open a window showing the active layer's intensity histogram.

    Read-only: it adds no layer and changes nothing, which is why it is the one
    quick operation that reports what it drew rather than what it made.
    """
    layer, data = _require_array_layer(viewer)
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Layer has no finite voxels to histogram.")
    counts, edges = np.histogram(finite, bins=max(int(bins), 8))
    from nvitk.gui.viz.histogram_window import show_histogram_window

    show_histogram_window(
        viewer, counts, edges, title=f"{getattr(layer, 'name', 'layer')} — histogram",
        limits=tuple(getattr(layer, "contrast_limits", ()) or ()),
    )
    return (
        f"Histogram of “{layer.name}”: {finite.size:,} voxels, "
        f"range [{finite.min():.4g}, {finite.max():.4g}]."
    )


def histogram_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`show_histogram`."""
    return (
        OpParam("bins", "Bins", "int", 256, minimum=8, maximum=4096),
    )


# ── segmentation helpers ──────────────────────────────────────────────────────
def contour_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`find_contours`."""
    return (
        OpParam("level", "Level (0 = midpoint)", "float", 0.0, minimum=-1e6, maximum=1e6,
                decimals=4, hint="Intensity the contour follows. 0 uses the data's midpoint."),
        OpParam("axis", "Slice axis", "int", 0, minimum=0, maximum=2),
    )


def find_contours(viewer: Any, *, level: float = 0.0, axis: int = 0) -> str:
    """Trace iso-intensity contours slice by slice into a Shapes layer.

    skimage's marching squares is 2D, so a volume is traced one slice at a time
    along *axis* — which is also how the result is read back, as planar outlines.
    """
    from nvitk.core.backend import using
    from skimage import measure

    layer, data = _require_array_layer(viewer)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError("Contours need a 2D or 3D layer.")
    value = float(level)
    if value == 0.0:
        finite = arr[np.isfinite(arr)]
        value = float((finite.min() + finite.max()) / 2.0) if finite.size else 0.0

    paths: list[np.ndarray] = []
    with using("numpy"):
        if arr.ndim == 2:
            paths.extend(measure.find_contours(arr, value))
        else:
            ax = int(np.clip(axis, 0, 2))
            for index in range(arr.shape[ax]):
                plane = np.take(arr, index, axis=ax)
                for contour in measure.find_contours(plane, value):
                    # Marching squares returns 2D rows/cols; put the slice back.
                    full = np.insert(contour, ax, float(index), axis=1)
                    paths.append(full)
    if not paths:
        raise ValueError(f"No contour at level {value:.4g}.")
    viewer.add_shapes(
        paths, shape_type="path", name=f"{layer.name}_contours",
        edge_color="#ffa400", edge_width=0.5, **layer_spatial_kwargs(layer),
    )
    return f"{len(paths)} contour(s) at {value:.4g} from “{layer.name}”."


def flood_params(viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`flood_fill_from_cursor`, scaled to the layer's range."""
    _layer, data = _require_array_layer(viewer)
    lo, hi = _intensity_range(data)
    span = max(hi - lo, 1e-6)
    return (
        OpParam("tolerance", "Tolerance", "float", span * 0.05, minimum=0.0, maximum=span,
                decimals=4,
                hint="How far from the seed's intensity the region may stray. "
                     "The seed is the viewer's current cursor position."),
    )


def flood_fill_from_cursor(viewer: Any, *, tolerance: float = 0.0) -> str:
    """Flood the region connected to the cursor, as a mask."""
    from nvitk.core.backend import using
    from skimage import segmentation

    layer, data = _require_array_layer(viewer)
    arr = np.asarray(data)
    seed = _cursor_index(viewer, layer, arr.shape)
    with using("numpy"):
        mask = segmentation.flood(arr, seed, tolerance=float(tolerance))
    count = int(mask.sum())
    if count == 0:
        raise ValueError("The flood filled nothing; try a larger tolerance.")
    _add(viewer, layer, mask.astype(np.int32), "flood", labels=True)
    return f"Flood from {seed} filled {count:,} voxel(s) of “{layer.name}”."


def _cursor_index(viewer: Any, layer: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The viewer's cursor as a voxel index into *layer*, clipped to *shape*."""
    try:
        position = layer.world_to_data(viewer.cursor.position)
    except Exception:  # noqa: BLE001 — fall back to the middle of the volume
        position = [s / 2.0 for s in shape]
    index = [int(round(float(v))) for v in list(position)[-len(shape):]]
    return tuple(int(np.clip(v, 0, n - 1)) for v, n in zip(index, shape))


def watershed_params(_viewer: Any) -> tuple[OpParam, ...]:
    """Options for :func:`watershed_split`."""
    return (
        OpParam("footprint", "Marker separation (voxels)", "int", 3, minimum=1, maximum=32,
                hint="Local maxima closer than this merge into one basin."),
        OpParam("use_gradient", "Flood", "choice", "distance",
                choices=(("Distance transform", "distance"), ("Intensity", "intensity"))),
    )


def watershed_split(viewer: Any, *, footprint: int = 3, use_gradient: str = "distance") -> str:
    """Split touching objects with a watershed.

    On a mask the distance transform is the surface to flood — the standard recipe
    for separating objects that touch. On an intensity image the image itself is,
    which is what you want for basins already visible in the data.
    """
    from nvitk.core.backend import using
    from scipy import ndimage as host_ndi
    from skimage import feature, segmentation

    layer, data = _require_array_layer(viewer)
    arr = np.asarray(data)
    with using("numpy"):
        binary = arr > 0
        if not binary.any():
            raise ValueError("Nothing to split: the layer is empty.")
        if str(use_gradient) == "intensity":
            surface = np.asarray(arr, dtype=np.float32)
        else:
            surface = -host_ndi.distance_transform_edt(binary)
        size = max(int(footprint), 1)
        peaks = feature.peak_local_max(
            -surface, footprint=np.ones((size,) * arr.ndim), labels=binary
        )
        markers = np.zeros(arr.shape, dtype=np.int32)
        for i, peak in enumerate(peaks, start=1):
            markers[tuple(peak)] = i
        if markers.max() == 0:
            raise ValueError("No markers found; try a smaller separation.")
        out = segmentation.watershed(surface, markers, mask=binary)
    _add(viewer, layer, out.astype(np.int32), "watershed", labels=True)
    return f"Watershed split “{layer.name}” into {int(out.max())} region(s)."


def _is_labels(layer: Any) -> bool:
    """True for a Napari ``Labels`` layer."""
    return type(layer).__name__ == "Labels"


__all__ = [
    "OpParam",
    "auto_contrast",
    "adjust_intensity",
    "adjust_params",
    "adjusted_intensity",
    "contour_params",
    "equalize_histogram",
    "equalize_params",
    "equalized_histogram",
    "find_contours",
    "flood_fill_from_cursor",
    "flood_params",
    "histogram_params",
    "rescale_intensity",
    "rescale_params",
    "rescaled_intensity",
    "show_histogram",
    "watershed_params",
    "watershed_split",
    "convert_dtype",
    "ct_window",
    "ct_window_limits",
    "ct_window_params",
    "contrast_window",
    "crop_bounds",
    "cropped_spatial",
    "crop_to_content",
    "gaussian",
    "invert",
    "median",
    "project",
    "reset_contrast",
    "rotate90",
    "threshold_at_display",
    "layer_data",
    "threshold_mask",
    "threshold_otsu",
    "threshold_params",
]
