"""Run catalog tools on Napari image layers inside nvitk-gui."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from nvitk.core.backend import using
from nvitk.types import Image


def _as_image(data: np.ndarray, layer: Any) -> Image:
    meta = getattr(layer, "metadata", None) or {}
    axes = meta.get("axes", "XYZ" if data.ndim == 3 else "YX")
    return Image(data=data, metadata=dict(meta), axes=axes)


def _unique_labels(data: np.ndarray) -> list[int]:
    flat = np.asarray(data).ravel()
    if flat.size == 0:
        return []
    labels = np.unique(flat)
    labels = labels[labels != 0]
    return [int(x) for x in labels]


def prepare_layer_data(
    data: np.ndarray,
    *,
    target_mode: str,
    label_ids: list[int] | None,
) -> tuple[np.ndarray, str]:
    """
    Select processing volume from a Napari layer.

    target_mode: raw | binary_mask | label | all_labels
    """
    mode = target_mode.strip().lower()
    arr = np.asarray(data)

    if mode == "raw":
        return arr, "raw"

    if mode == "binary_mask":
        if arr.dtype == bool:
            return arr.astype(np.uint8), "binary_mask"
        return (arr != 0).astype(np.uint8), "binary_mask"

    if mode == "label":
        if not label_ids:
            raise ValueError("Label mode requires at least one label id.")
        mask = np.isin(arr, label_ids)
        if not mask.any():
            raise ValueError(f"No voxels found for label id(s): {label_ids}")
        return mask.astype(np.uint8), f"label_{'_'.join(str(i) for i in label_ids)}"

    if mode == "all_labels":
        labels = _unique_labels(arr)
        if not labels:
            raise ValueError("No non-zero labels found in the active layer.")
        return (arr != 0).astype(np.uint8), "all_labels"

    raise ValueError(f"Unknown target mode: {target_mode}")


def coerce_tool_output(out: Any) -> np.ndarray:
    """Normalize library return values to a single ndarray for Napari layers."""
    if isinstance(out, Image):
        return np.asarray(out.data)
    if isinstance(out, tuple):
        if len(out) == 0:
            raise ValueError("Tool returned an empty tuple.")
        out = out[0]
    arr = np.asarray(out)
    if arr.dtype == object:
        raise ValueError(
            "Tool returned an inhomogeneous object array. "
            "Check that the selected tool returns a single volume."
        )
    return arr


def _run_with_backend(fn: Callable[..., Any], img: Image, *, use_gpu: bool) -> np.ndarray:
    bk = "cupy" if use_gpu else "numpy"
    with using(bk):
        out = fn(img)
    return coerce_tool_output(out)


def _match_tool(tool_name: str, *keywords: str) -> bool:
    t = tool_name.lower()
    return any(k in t for k in keywords)


def _run_sliding_threshold(img: Image) -> np.ndarray:
    from nvitk.filters.sliding_threshold import (
        binary_mask_sliding_threshold_2d,
        binary_mask_sliding_threshold_3d,
    )

    data = np.asarray(img.data, dtype=np.float64)
    if data.ndim == 2:
        return binary_mask_sliding_threshold_2d(data)
    if data.ndim == 3:
        mask, _thresh = binary_mask_sliding_threshold_3d(data)
        return mask
    raise ValueError(f"Sliding threshold expects 2D or 3D data, got ndim={data.ndim}")


def _run_centerline(img: Image) -> np.ndarray:
    from nvitk.morphology.centerline import compute_centerlines, skeletonize_binary

    data = np.asarray(img.data)
    if data.ndim != 3:
        raise ValueError("Centerline / skeletonize requires a 3D volume.")

    labels = _unique_labels(data)
    if len(labels) > 1:
        paths = compute_centerlines(data, labels=labels, min_points=5)
        out = np.zeros(data.shape, dtype=np.uint8)
        for lid, pts in paths.items():
            pts_i = np.round(np.asarray(pts)).astype(int)
            for x, y, z in pts_i:
                if (
                    0 <= x < out.shape[0]
                    and 0 <= y < out.shape[1]
                    and 0 <= z < out.shape[2]
                ):
                    out[x, y, z] = int(lid)
        if not out.any():
            raise ValueError("No centerline points found for the selected labels.")
        return out

    sk = skeletonize_binary(data > 0)
    return np.asarray(sk, dtype=np.uint8)


def run_tool_on_layer(
    tool_name: str,
    layer: Any,
    *,
    use_gpu: bool,
    target_mode: str,
    label_ids: list[int] | None,
) -> np.ndarray:
    data = np.asarray(layer.data)
    proc_data, _tag = prepare_layer_data(
        data,
        target_mode=target_mode,
        label_ids=label_ids,
    )
    img = _as_image(proc_data, layer)

    if _match_tool(tool_name, "bilateral"):
        from nvitk.restoration import bilateral

        return _run_with_backend(bilateral, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "dilate"):
        from nvitk.morphology.binary import dilate

        return _run_with_backend(dilate, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "erode"):
        from nvitk.morphology.binary import erode

        return _run_with_backend(erode, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "open") and "close" not in tool_name.lower():
        from nvitk.morphology.binary import open as morph_open

        return _run_with_backend(morph_open, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "close"):
        from nvitk.morphology.binary import close

        return _run_with_backend(close, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "fill", "hole"):
        from nvitk.morphology.binary import fill_holes

        return _run_with_backend(fill_holes, img, use_gpu=use_gpu)

    if _match_tool(tool_name, "sliding", "threshold"):
        bk = "cupy" if use_gpu else "numpy"
        with using(bk):
            return _run_sliding_threshold(img)

    if _match_tool(tool_name, "centerline", "skeleton"):
        bk = "cupy" if use_gpu else "numpy"
        with using(bk):
            return _run_centerline(img)

    raise NotImplementedError(
        f"Tool '{tool_name}' is listed in the catalog but not wired in the GUI yet. "
        "Use the matching nvitk-* CLI or extend gui/tool_runner.py."
    )


def parse_label_ids(text: str) -> list[int]:
    """Parse '1,2,3' or '1 2' into label ids."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[,;\s]+", str(text).strip())
    return [int(p) for p in parts if p]


def notify(message: str, *, error: bool = False) -> None:
    try:
        if error:
            from napari.utils.notifications import show_error

            show_error(message)
        else:
            from napari.utils.notifications import show_info

            show_info(message)
    except Exception:
        print(("ERROR: " if error else "INFO: ") + message, flush=True)
