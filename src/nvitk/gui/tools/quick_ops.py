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


def _is_labels(layer: Any) -> bool:
    """True for a Napari ``Labels`` layer."""
    return type(layer).__name__ == "Labels"


__all__ = [
    "OpParam",
    "auto_contrast",
    "convert_dtype",
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
