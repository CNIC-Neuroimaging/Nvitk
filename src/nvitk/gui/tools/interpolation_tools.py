"""Interpolation tools of the Tools panel: resampling, block downsampling, slice filling.

The array work lives in :mod:`nvitk.transform.interpolation`; this module turns a
Napari layer into an :class:`~nvitk.types.Image`, runs it, and adds the result.

Placement
---------
A resampled array is on a new grid, so it cannot borrow the source layer's
placement the way same-grid tool outputs do. Its data→world transform is the
source layer's own — every part of it, read from Napari (``affine``, ``scale``,
``translate``) — composed with :func:`~nvitk.transform.interpolation.resample_index_map`,
so the result overlays its source exactly whatever kind of layer that was: a
plain volume, a world-ordered sagittal one, a time-first 3D+t series.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.core.spatial import layer_spatial_kwargs, layer_to_image
from nvitk.gui.labels.visibility import copy_layer_metadata_for_output, is_label_like_layer
from nvitk.transform.interpolation import (
    INTERPOLATION_ORDERS,
    block_reduce_axes,
    fill_missing_slices,
    interpolate_mask_slices,
    parse_axes,
    resample_axes,
    resample_index_map,
)

#: Tool ids handled here.
INTERPOLATION_TOOL_IDS: frozenset[str] = frozenset({
    "interp_resample_axes",
    "interp_block_reduce",
    "interp_mask_slices",
    "interp_fill_slices",
    "time_interpolate_frames",
})

#: nvitk metadata keys describing the *source file's* layout, which a resampled
#: layer no longer has: it is written back in its own (display) order.
_FILE_LAYOUT_KEYS = ("affine_source", "display_reordered", "source", "source_type", "frame_times_s")


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


def _is_labels(layer: Any) -> bool:
    """A label map: a Labels layer, or an Image layer holding one."""
    return type(layer).__name__ == "Labels" or is_label_like_layer(layer)


def layer_data_to_world(layer: Any) -> np.ndarray:
    """The layer's full homogeneous data→world matrix (affine, scale, translate…)."""
    try:
        return np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    except Exception:  # noqa: BLE001 — a non-Napari stand-in: rebuild from its parts
        nd = int(np.ndim(layer.data))
        mat = np.eye(nd + 1)
        scale = getattr(layer, "scale", None)
        if scale is not None:
            mat[:nd, :nd] = np.diag([float(s) for s in scale][:nd])
        translate = getattr(layer, "translate", None)
        if translate is not None:
            mat[:nd, nd] = [float(t) for t in translate][:nd]
        aff = getattr(layer, "affine", None)
        aff = None if aff is None else np.asarray(getattr(aff, "affine_matrix", aff), dtype=float)
        if aff is not None and aff.shape == (nd + 1, nd + 1):
            mat = aff @ mat
        return mat


def _axes_letters(layer: Any, ndim: int) -> str:
    """The layer's axis letters in its own (display) order, ``""`` when unknown."""
    from nvitk.gui.core.orientation import _axes_string_from_layer

    letters = (_axes_string_from_layer(layer) or "").upper()
    if len(letters) == ndim:
        return letters
    # An unlabelled 2D / 3D layer (added from code, not a file) reads as XY / XYZ;
    # a 4D one is ambiguous (time first or last) and gets no letters.
    return {2: "XY", 3: "XYZ"}.get(ndim, "")


def _regridded_metadata(layer: Any, new_shape: tuple[int, ...], world: np.ndarray) -> dict[str, Any]:
    """The source layer's metadata, updated for the new grid placed by *world*."""
    from nvitk.gui.core.orientation import DISPLAY_REORDERED_KEY, SOURCE_AXES_KEY, TIME_LEADING_KEY

    meta = copy_layer_metadata_for_output(getattr(layer, "metadata", None))
    nested = dict(meta.get("nvitk_metadata") or {})
    ndim = len(new_shape)
    letters = _axes_letters(layer, ndim)
    time_ax = next((i for i, ch in enumerate(letters) if ch in ("T", "C")), None)
    spatial = [i for i in range(ndim) if i != time_ax][-3:]
    steps = [float(np.linalg.norm(world[:ndim, k])) for k in range(ndim)]
    patch: dict[str, Any] = {"shape": tuple(int(v) for v in new_shape)}
    if len(spatial) == 3:
        sp = tuple(steps[k] for k in spatial)
        patch["spacing"] = sp
        patch["x_res"], patch["y_res"], patch["z_res"] = sp
        aff4 = np.eye(4)
        for row_i, row in enumerate(spatial):
            for col_i, col in enumerate(spatial):
                aff4[row_i, col_i] = world[row, col]
            aff4[row_i, 3] = world[row, ndim]
        patch["affine"] = aff4
    if time_ax is not None:
        patch["t_res"] = steps[time_ax]
    for key in _FILE_LAYOUT_KEYS:
        nested.pop(key, None)
        meta.pop(key, None)
    if nested.get(TIME_LEADING_KEY) and letters:
        # Still exported with time last, but in the displayed spatial order.
        nested[SOURCE_AXES_KEY] = "".join(ch for ch in letters if ch not in ("T", "C")) + letters[time_ax]
    else:
        nested.pop(SOURCE_AXES_KEY, None)
    nested.pop(DISPLAY_REORDERED_KEY, None)
    if letters:
        nested["axes"] = letters
        meta["axes"] = letters
    nested.update(patch)
    meta["nvitk_metadata"] = nested
    return meta


def _style_kwargs(layer: Any) -> dict[str, Any]:
    """Display settings worth carrying to a derived Image layer."""
    out: dict[str, Any] = {}
    for key in ("colormap", "gamma", "blending", "opacity"):
        val = getattr(layer, key, None)
        if val is not None:
            out[key] = getattr(val, "name", val) if key == "colormap" else val
    return out


def add_regridded_layer(
    viewer: Any,
    layer: Any,
    data: np.ndarray,
    *,
    name: str,
    labels: bool,
) -> Any:
    """Add *data* — *layer* resampled onto a new grid of the same extent — as a layer."""
    data = to_numpy(data)
    old_shape = tuple(int(v) for v in np.shape(layer.data))
    world = layer_data_to_world(layer) @ resample_index_map(old_shape, data.shape)
    kwargs: dict[str, Any] = {
        "name": name,
        "affine": world,
        "metadata": _regridded_metadata(layer, tuple(data.shape), world),
    }
    axis_labels = getattr(layer, "axis_labels", None)
    if axis_labels is not None and len(tuple(axis_labels)) == data.ndim:
        kwargs["axis_labels"] = tuple(axis_labels)
    if labels:
        out = viewer.add_labels(np.asarray(data).astype(np.int32, copy=False), opacity=0.7, **kwargs)
        out._nvitk_label_like = True
        return out
    return viewer.add_image(data, **kwargs, **_style_kwargs(layer))


def add_same_grid_layer(
    viewer: Any,
    layer: Any,
    data: np.ndarray,
    *,
    name: str,
    labels: bool,
    replace: bool = False,
) -> Any:
    """Add *data* on *layer*'s own grid, or write it into *layer* when *replace*."""
    data = to_numpy(data)
    if replace and tuple(data.shape) == tuple(np.shape(layer.data)):
        layer.data = data.astype(np.asarray(layer.data).dtype, copy=False) if labels else data
        return layer
    kwargs: dict[str, Any] = {"name": name, **layer_spatial_kwargs(layer)}
    meta = copy_layer_metadata_for_output(getattr(layer, "metadata", None))
    if meta:
        kwargs["metadata"] = meta
    if labels:
        out = viewer.add_labels(np.asarray(data).astype(np.int32, copy=False), opacity=0.7, **kwargs)
        out._nvitk_label_like = True
        return out
    return viewer.add_image(data, **kwargs, **_style_kwargs(layer))


def _values(text: Any) -> list[float]:
    """``"2"`` / ``"0.5, 0.5, 1"`` → floats."""
    parts = [p for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Enter a factor, spacing or size (one value, or one per axis).")
    return [float(p) for p in parts]


def _layer_image(layer: Any) -> Any:
    """*layer* as an :class:`Image` in its own axis order, axes labelled when known."""
    img = layer_to_image(layer)
    letters = _axes_letters(layer, img.ndim)
    if letters and str(img.axes or "").upper() != letters:
        img = img.with_data(img.data, axes=letters)
    return img


def _target_shape(layer: Any, axes: tuple[int, ...], mode: str, values: list[float]) -> tuple[int, ...]:
    """Output shape, with spacing read from the layer's own placement (any layer kind)."""
    shape = list(int(v) for v in np.shape(layer.data))
    if len(values) == 1:
        values = values * len(axes)
    if len(values) != len(axes):
        raise ValueError(f"Give one value, or one per selected axis ({len(axes)}).")
    world = layer_data_to_world(layer)
    nd = len(shape)
    for ax, val in zip(axes, values):
        if val <= 0:
            raise ValueError("Resampling values must be positive.")
        if mode == "factor":
            new = shape[ax] * val
        elif mode == "spacing":
            step = float(np.linalg.norm(world[:nd, ax])) or 1.0
            new = shape[ax] * step / val
        else:
            new = val
        shape[ax] = max(1, int(round(new)))
    return tuple(shape)


def _describe_axes(layer: Any, axes: tuple[int, ...]) -> str:
    letters = _axes_letters(layer, int(np.ndim(layer.data)))
    return ", ".join(f"{a}" + (f" ({letters[a]})" if letters else "") for a in axes)


def run_interpolation_tool(
    tool_id: str,
    viewer: Any,
    layer: Any,
    params: dict[str, Any],
    label_ids: list[int] | None = None,
) -> None:
    """Run one of :data:`INTERPOLATION_TOOL_IDS` on *layer* and add its result."""
    if layer is None or getattr(layer, "data", None) is None:
        raise ValueError("Select an image or labels layer.")
    ndim = int(np.ndim(layer.data))
    letters = _axes_letters(layer, ndim)
    labels = _is_labels(layer)
    replace = str(params.get("overlay_mode") or "") == "replace_active"

    if tool_id in ("interp_resample_axes", "time_interpolate_frames"):
        if tool_id == "time_interpolate_frames":
            time_ax = next((i for i, ch in enumerate(letters) if ch in ("T", "C")), None)
            if time_ax is None:
                if ndim != 4:
                    raise ValueError("Interpolating frames needs a 3D+t layer.")
                from nvitk.gui.core.orientation import layer_is_time_leading

                time_ax = 0 if layer_is_time_leading(layer) else 3
            axes: tuple[int, ...] = (time_ax,)
            mode = str(params.get("interp_time_mode") or "factor")
            values = [float(params.get("interp_time_value") or 2.0)]
        else:
            axes = parse_axes(params.get("interp_axes") or "all", ndim, letters)
            mode = str(params.get("interp_mode") or "factor")
            values = _values(params.get("interp_values") or "2")
        new_shape = _target_shape(layer, axes, mode, values)
        if new_shape == tuple(np.shape(layer.data)):
            _notify("Nothing to do: the layer already has that size on those axes.")
            return None
        order = INTERPOLATION_ORDERS.get(str(params.get("interp_order") or "linear"), 1)
        img = _layer_image(layer)
        out = resample_axes(
            img,
            shape=new_shape,
            order=0 if labels else order,
            antialias=bool(params.get("interp_antialias", True)),
            labels=labels,
            label_method=str(params.get("interp_label_method") or "nearest"),
        )
        suffix = "frames" if tool_id == "time_interpolate_frames" else "resampled"
        add_regridded_layer(viewer, layer, out.data, name=f"{layer.name}_{suffix}", labels=labels)
        _notify(
            f"Resampled axes {_describe_axes(layer, axes)}: "
            f"{tuple(np.shape(layer.data))} → {tuple(out.shape)}"
            + (" (labels)" if labels else "")
        )
        return None

    if tool_id == "interp_block_reduce":
        axes = parse_axes(params.get("interp_axes") or "all", ndim, letters)
        factors = [int(round(v)) for v in _values(params.get("interp_factors") or "2")]
        method = str(params.get("interp_reduce") or "mean")
        if labels and method not in ("mode", "max", "min"):
            # A mean of label ids is not a label: vote instead.
            method = "mode"
        out = block_reduce_axes(_layer_image(layer), axes=axes, factors=factors, method=method)
        # Block reduction keeps whole blocks only: place it from the kept extent.
        facs = factors if len(factors) == len(axes) else factors[:1] * len(axes)
        block = [1] * ndim
        for ax, fac in zip(axes, facs):
            block[ax] = fac
        kept = tuple(int(n) * b for n, b in zip(out.shape, block))
        world = layer_data_to_world(layer) @ resample_index_map(kept, out.shape)
        kwargs: dict[str, Any] = {
            "name": f"{layer.name}_blocks_{method}",
            "affine": world,
            "metadata": _regridded_metadata(layer, tuple(out.shape), world),
        }
        if labels:
            lab = viewer.add_labels(np.asarray(out.data).astype(np.int32, copy=False), opacity=0.7, **kwargs)
            lab._nvitk_label_like = True
        else:
            viewer.add_image(out.data, **kwargs, **_style_kwargs(layer))
        _notify(f"Block {method} over axes {_describe_axes(layer, axes)}: {tuple(np.shape(layer.data))} → {tuple(out.shape)}")
        return None

    if tool_id == "interp_mask_slices":
        if not labels:
            raise ValueError("Interpolating between slices needs a mask or labels layer.")
        spec = str(params.get("interp_slice_axes") or "auto").strip() or "auto"
        img = _layer_image(layer)
        if spec.lower() != "auto":
            spec = ",".join(str(a) for a in parse_axes(spec, ndim, letters))
        out = interpolate_mask_slices(
            img,
            axes=spec,
            method=str(params.get("interp_slice_method") or "shape"),
            label_ids=label_ids or None,
            max_gap=int(params.get("interp_max_gap") or 0),
            overwrite=bool(params.get("interp_overwrite")),
        )
        before = np.count_nonzero(to_numpy(img.data))
        after = np.count_nonzero(to_numpy(out.data))
        add_same_grid_layer(
            viewer, layer, out.data, name=f"{layer.name}_interpolated", labels=True, replace=replace,
        )
        _notify(f"Interpolated between slices: {after - before} voxel(s) filled.")
        return None

    if tool_id == "interp_fill_slices":
        axis = parse_axes(params.get("interp_axis") or "2", ndim, letters)
        if len(axis) != 1:
            raise ValueError("Fill slices along one axis at a time.")
        slices = str(params.get("interp_slices") or "").strip() or None
        order = INTERPOLATION_ORDERS.get(str(params.get("interp_order") or "linear"), 1)
        img = _layer_image(layer)
        from nvitk.transform.interpolation import _parse_indices, missing_slices

        data = to_numpy(img.data)
        todo = _parse_indices(slices, data.shape[axis[0]]) if slices else missing_slices(data, axis[0])
        if not todo:
            _notify("No missing slices found along that axis (none empty or NaN between data).")
            return None
        out = fill_missing_slices(img, axis=axis[0], slices=todo, order=0 if labels else min(order, 1))
        add_same_grid_layer(
            viewer, layer, out.data, name=f"{layer.name}_filled", labels=labels, replace=replace,
        )
        shown = ", ".join(str(i) for i in todo[:12]) + ("…" if len(todo) > 12 else "")
        _notify(f"Rebuilt {len(todo)} slice(s) along axis {_describe_axes(layer, axis)}: {shown}")
        return None

    raise NotImplementedError(f"Interpolation tool {tool_id!r} is not implemented.")


__all__ = [
    "INTERPOLATION_TOOL_IDS",
    "add_regridded_layer",
    "add_same_grid_layer",
    "layer_data_to_world",
    "run_interpolation_tool",
]
