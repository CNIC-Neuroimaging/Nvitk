"""Reconstruction tools of the Tools panel: corrections applied to reconstructed images.

*Background phase correction* is the spatial polynomial background qvtpy takes
off the 4D-flow complex difference
(:func:`nvitk.io.conversors._phase2volume_bg.subtract_polynomial_background`),
run on the active layer: a 3D image, or a 3D+t series frame by frame (or with
one background fitted on the temporal mean). The array work runs on the current
compute backend; the result is added on the layer's own grid, a 3D+t result
time-first like its source.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.backend import get_global_backend, using

#: Tool ids handled here.
RECONSTRUCTION_TOOL_IDS: frozenset[str] = frozenset({"recon_background_correction"})


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


def _fit_mask(viewer: Any, params: dict[str, Any], spatial_shape: tuple[int, ...]) -> Any:
    """The optional mask the background is fitted inside, on the image's spatial grid."""
    from nvitk.gui.tools.runner import _layer_param, _resolve_layer

    name = _layer_param(params, "bgc_mask_layer")
    if not name:
        return None
    mask_layer = _resolve_layer(viewer, name)
    mask = mask_layer.data
    if tuple(np.shape(mask)) != tuple(spatial_shape):
        raise ValueError(
            f"The mask “{name}” is {' × '.join(map(str, np.shape(mask)))}; it must be on the image's "
            f"spatial grid ({' × '.join(map(str, spatial_shape))})."
        )
    return mask


def run_background_correction(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Take the spatial polynomial background off the active 3D or 3D+t layer."""
    from nvitk.gui.core.spatial import _time_axis_index
    from nvitk.gui.viz.time_frequency import _add_like
    from nvitk.io.conversors._phase2volume_bg import subtract_polynomial_background

    if layer is None or getattr(layer, "data", None) is None:
        raise ValueError("Select the image to correct (e.g. the complex difference).")
    data = layer.data
    ndim = int(np.ndim(data))
    if ndim not in (3, 4):
        raise ValueError(f"“{layer.name}” is {ndim}D; the correction needs a 3D or 3D+t image.")
    time_axis = _time_axis_index(layer) if ndim == 4 else None
    spatial_shape = tuple(int(n) for k, n in enumerate(np.shape(data)) if k != time_axis)
    order = int(str(params.get("bgc_order") or "2"))
    percentile = float(params.get("bgc_percentile") or 25.0)
    per_frame = str(params.get("bgc_time_mode") or "per frame") != "time mean"
    signed = bool(params.get("bgc_signed", False))
    add_background = bool(params.get("bgc_add_background", False))
    mask = _fit_mask(viewer, params, spatial_shape)

    with using(get_global_backend()):
        result = subtract_polynomial_background(
            data,
            time_axis=time_axis,
            per_frame=per_frame,
            order=order,
            static_percentile=percentile,
            signed=signed,
            mask=mask,
            return_background=add_background,
        )
    corrected, background = result if add_background else (result, None)
    _add_like(viewer, layer, corrected, name=f"{layer.name}_bgc")
    if background is not None:
        _add_like(viewer, layer, background, name=f"{layer.name}_background")
    how = ("" if time_axis is None
           else ", per frame" if per_frame else ", fitted on the temporal mean")
    _notify(
        f"Background correction (order {order}, fitted on the lowest {percentile:g} %"
        f"{' of |values|' if signed else ''}{how}{', inside the mask' if mask is not None else ''}) "
        f"on “{layer.name}” → “{layer.name}_bgc”."
    )


def run_reconstruction_tool(tool_id: str, viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run one of :data:`RECONSTRUCTION_TOOL_IDS` on *layer* and add its result."""
    if tool_id == "recon_background_correction":
        run_background_correction(viewer, layer, params)
        return
    raise ValueError(f"Unknown reconstruction tool: {tool_id}")


__all__ = ["RECONSTRUCTION_TOOL_IDS", "run_background_correction", "run_reconstruction_tool"]
