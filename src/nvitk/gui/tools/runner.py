"""Run GUI catalog tools on Napari image layers."""

from __future__ import annotations

import csv
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import get_global_backend, using, setup
from nvitk.gui.core.backend import (
    gpu_enabled,
    layer_data_for_tool,
    napari_array,
    napari_label_array,
    run_with_backend,
)
from nvitk.gui.core.log_panel import gui_log, run_subprocess_logged
from nvitk.gui.core.spatial import (
    align_mask_to_reference_layer,
    layer_to_image,
    nvitk_metadata_from_layer,
)
from nvitk.gui.tools.registry import tool_by_id
from nvitk.types import Image

setup(globals())

_MEASURE_NOTIFY = frozenset({
    "volume_mm3",
    "volume_cc",
    "masked_stats",
    "integrated_intensity",
    "suv_stats",
    "intensity_similarity",
    "dice",
    "jaccard",
    "voxel_metrics",
    "surface_metrics",
})


def _reference_and_mask_images(
    viewer: Any,
    mask_layer: Any,
    reference_layer_name: str,
    mask_data = None,
) -> tuple[Image, Image]:
    """Reference grid + mask (affine-resampled when Napari shows them aligned)."""
    ref_layer = _resolve_layer(viewer, reference_layer_name)
    ref_img, mask_img, resampled = align_mask_to_reference_layer(
        mask_layer, ref_layer, mask_data, order=0
    )
    if resampled:
        gui_log(
            f"Resampled mask '{getattr(mask_layer, 'name', 'layer')}' onto reference "
            f"'{ref_layer.name}' grid {tuple(ref_img.data.shape)}."
        )
    return ref_img, mask_img


def _resolve_layer(viewer: Any, name: str) -> Any:
    """Look up the layer named *name* in *viewer*; raises ``ValueError`` if blank or not found."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("Select a reference layer.")
    for lyr in viewer.layers:
        if lyr.name == name:
            return lyr
    raise ValueError(f"Layer not found: {name}")


def _layer_param(params: dict[str, Any], key: str) -> str:
    """Layer dropdown value; treats ``(none)`` and empty as unset."""
    raw = str(params.get(key) or "").strip()
    if raw in ("", "(none)"):
        return ""
    return raw


def _format_metrics(metrics: dict[str, Any]) -> str:
    """Render a ``{name: value}`` metrics dict as ``"name: value"`` lines for a notification."""
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    return "\n".join(lines)


def _suv_stats_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Shared SUV conversion options for stats and volume tools."""
    return {
        "kinds": (str(params.get("suv_kind") or "bw"),),
        "philips": bool(params.get("philips_factor", True)),
        "revert_scaling": bool(params.get("revert_scaling", False)),
    }


def _layer_source_path(layer: Any) -> Path | None:
    """*layer*'s original source file path (from nvitk metadata), or ``None`` if unrecorded or the
    file no longer exists."""
    meta = nvitk_metadata_from_layer(layer)
    src = meta.get("source")
    if not src:
        return None
    path = Path(str(src))
    return path if path.is_file() else None


def _ensure_nifti_path(layer: Any, *, prefix: str) -> Path:
    """Return on-disk NIfTI for *layer*, exporting a temp file when needed."""
    src = _layer_source_path(layer)
    if src is not None:
        name = src.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return src
    from nvitk.io import imsave

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"nvitk_{prefix}_"))
    path = tmp_dir / "volume.nii.gz"
    imsave(path, layer_to_image(layer))
    return path


def _export_layer_nifti(layer: Any, *, prefix: str, filename: str = "volume.nii.gz") -> Path:
    """Always write *layer*'s current data + affine to a temp NIfTI (no source short-circuit)."""
    from nvitk.io import imsave

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"nvitk_{prefix}_"))
    path = tmp_dir / filename
    imsave(path, layer_to_image(layer))
    return path


def _bridge_same_label_components(seg: Any, *, max_gap: int = 24) -> Any:
    """Connect nearby disconnected CCs that share the same label id (MST tubes)."""
    import numpy as numpy
    from nvitk.morphology.mst_bridge import bridge_binary_components_mst
    from scipy import ndimage as ndi

    # Force host arrays: scipy.ndimage / MST bridge are CPU-only.
    # Use stock NumPy — module ``np`` may be a CuPy proxy via ``setup(globals())``.
    out = numpy.asarray(to_numpy(seg), dtype=numpy.int32).copy()
    labels = [int(x) for x in numpy.unique(out) if int(x) != 0]
    for lab in labels:
        mask = out == lab
        n_cc = int(ndi.label(mask)[1])
        if n_cc <= 1:
            continue
        bridged = bridge_binary_components_mst(mask, max_gap=max_gap, tube_radius=1)
        out[numpy.asarray(to_numpy(bridged), dtype=bool)] = lab
    return out


def _run_pipeline_cli(spec: Any, params: dict[str, Any]) -> None:
    """Run a registered pipeline's CLI command as a subprocess (streaming output to the log panel),
    building argv from ``params["pipeline_argv"]`` if given, else a raw ``extra_args`` string."""
    cmd = str(spec.cli_command or "").strip()
    if not cmd:
        notify("Pipeline command not configured.", error=True)
        return
    exe = cmd.split()[0]
    if shutil.which(exe) is None:
        notify(
            f"Command {exe!r} not found on PATH. "
            f"Install the pipeline extra or run: {cmd} --help",
            error=True,
        )
        return
    wd = str(params.get("working_dir") or ".").strip() or "."
    argv = params.get("pipeline_argv")
    if not argv:
        extra = str(params.get("extra_args") or "--help").strip()
        argv = [exe, *shlex.split(extra)]
    notify(f"Running pipeline in {wd}")
    code = run_subprocess_logged(list(argv), cwd=wd)
    if code == 0:
        notify(f"Pipeline finished: {cmd}")
    else:
        notify(f"Pipeline exited with code {code}", error=True)


def _unique_labels(data: np.ndarray) -> list[int]:
    """Sorted distinct non-zero integer values in *data* (backend array, so CuPy inputs stay on-device)."""
    flat = as_backend_array(data).ravel()
    if flat.size == 0:
        return []
    labels = np.unique(flat)
    return [int(x) for x in labels if int(x) != 0]


def _label_ids_array(label_ids: list[int]) -> np.ndarray:
    """Backend array of label ids for :func:`np.isin` (CuPy rejects plain lists)."""
    return to_numpy(label_ids)


def _multilabel_input_and_labels(
    layer: Any,
    *,
    target_mode: str,
    label_ids: list[int] | None,
    what: str = "this tool",
) -> tuple[Any, list[int]]:
    """
    The selected labels as a **multilabel** array plus their ids, for label-preserving tools.

    Deliberately bypasses :func:`prepare_layer_data`, which reduces a selection to its binary union
    and so destroys exactly the information a per-label tool needs.

    Reads through :func:`~nvitk.gui.labels.visibility.label_source_data` rather than ``layer.data``:
    the live label-visibility filter zeroes unselected labels *in the layer's own array* for Image
    masks, so a tool reading ``layer.data`` would silently operate on whatever the user last chose
    to look at.
    """
    from nvitk.gui.labels.visibility import label_source_data

    src = as_backend_array(label_source_data(layer))
    all_labels = _unique_labels(src)
    if not all_labels:
        raise ValueError("No non-zero labels in the active layer.")

    mode = target_mode.strip().lower()
    if mode == "label":
        if not label_ids:
            raise ValueError(f"Select at least one label for {what}.")
        labels = [int(x) for x in label_ids]
        src_np = to_numpy(src)
        work = np.where(np.isin(src_np, _label_ids_array(labels)), src_np, 0)
        return as_backend_array(work), labels

    return src, all_labels


def _skeletonize_input_and_labels(
    layer: Any,
    *,
    target_mode: str,
    label_ids: list[int] | None,
) -> tuple[Any, list[int]]:
    """Multilabel mask + label ids for skeletonize (preserve per-vessel ids in output)."""
    return _multilabel_input_and_labels(
        layer, target_mode=target_mode, label_ids=label_ids, what="skeletonize"
    )


def multilabel_selection(
    layer: Any, *, spec: Any, target_mode: str, label_ids: list[int] | None
) -> list[int]:
    """
    Label ids a per-label run should iterate, or ``[]`` when the ordinary binary path applies.

    Per-label routing is only *different* from the ordinary path when there is more than one label
    in play — with a single label the two agree exactly, and taking the plain path there keeps the
    long-standing behaviour of every binary tool byte-identical.

    Shared with :mod:`nvitk.gui.sge.submit`, which stages the same decision for the cluster.
    """
    if spec is None or not getattr(spec, "multilabel", False):
        return []
    mode = str(target_mode).strip().lower()
    if mode == "label":
        ids = [int(x) for x in (label_ids or [])]
    elif mode == "all_labels":
        from nvitk.gui.labels.visibility import label_source_data

        ids = [int(x) for x in (label_ids or [])] or _unique_labels(
            as_backend_array(label_source_data(layer))
        )
    else:
        return []
    return ids if len(ids) > 1 else []


def _run_multilabel(
    tool_id: str,
    layer: Any,
    viewer: Any,
    *,
    label_ids: list[int],
    params: dict[str, Any],
) -> np.ndarray:
    """
    Run *tool_id* once per label and recombine, keeping the original ids.

    The per-label run is a plain single-label invocation of the same dispatcher, so the operation a
    label sees here is *exactly* the one it would see if it were the only label selected — there is
    no second implementation of any tool to keep in step with the first.

    Ordering: earlier ids in *label_ids* win where two results overlap, matching
    :func:`~nvitk.segmentation.labels.apply_per_label`'s ``overlap="first"``. Inputs are disjoint by
    construction; outputs need not be — a dilation of two adjacent labels overlaps by design.
    """
    from nvitk.gui.labels.visibility import label_source_data

    source = to_numpy(as_backend_array(label_source_data(layer)))
    out = np.zeros(source.shape, dtype=_multilabel_dtype(source, label_ids))
    claimed = np.zeros(source.shape, dtype=bool)

    for label in label_ids:
        result = run_gui_tool(
            tool_id, layer, viewer, target_mode="label", label_ids=[int(label)], params=params
        )
        if result is None:
            continue
        kept = to_numpy(result) != 0
        if kept.shape != source.shape:
            raise ValueError(
                f"{tool_id} returned shape {tuple(kept.shape)} for label {label}, but the layer is "
                f"{tuple(source.shape)}. Per-label runs need a shape-preserving tool."
            )
        kept &= ~claimed
        out[kept] = int(label)
        claimed |= kept

    gui_log(
        f"{tool_id}: ran per label over {len(label_ids)} label(s) "
        f"({', '.join(str(i) for i in label_ids)}); output keeps the input's ids."
    )
    return out


def _multilabel_dtype(source: np.ndarray, label_ids: list[int]) -> Any:
    """Integer dtype for a recombined per-label output — the input's own, else the smallest that fits."""
    import numpy as numpy_host  # module ``np`` may be a CuPy proxy via ``setup(globals())``

    if numpy_host.issubdtype(source.dtype, numpy_host.integer):
        return source.dtype
    top = max((abs(int(v)) for v in label_ids), default=0)
    if top <= 255:
        return numpy_host.uint8
    return numpy_host.uint16 if top <= 65535 else numpy_host.int32


def prepare_layer_data(
    data: np.ndarray,
    *,
    target_mode: str,
    label_ids: list[int] | None,
) -> tuple[np.ndarray, str]:
    """Reduce layer *data* to the array a tool should actually operate on given *target_mode*
    (``"raw"``, ``"binary_mask"``, ``"label"``, or ``"all_labels"``), returning ``(array, mode_tag)``;
    raises ``ValueError`` for missing label ids or an unknown mode."""
    mode = target_mode.strip().lower()
    arr = as_backend_array(data)

    if mode == "raw":
        return arr, "raw"

    if mode == "binary_mask":
        if arr.dtype == bool:
            return arr.astype(np.uint8), "binary_mask"
        return (arr != 0).astype(np.uint8), "binary_mask"

    if mode == "label":
        if not label_ids:
            raise ValueError("Label mode requires one or more label ids.")
        mask = np.isin(arr, _label_ids_array(label_ids))
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
    """Napari-safe NumPy output (never implicit CuPy → NumPy conversion)."""
    return napari_array(out)


def coerce_label_output(out: Any) -> np.ndarray:
    """Napari-safe output for a **label map** — keeps the integer dtype the ids live in."""
    return napari_label_array(out)


def _morph_common(img: Image, op: str, params: dict[str, Any]) -> np.ndarray:
    """Apply the base morphology op named *op* (dilate/erode/open/close/fill_holes) to *img* with GUI
    parameters (footprint, iterations, mode, connectivity)."""
    from nvitk.morphology.binary import close, dilate, erode, fill_holes, open as morph_open

    fn = {
        "dilate": dilate,
        "erode": erode,
        "open": morph_open,
        "close": close,
        "fill_holes": fill_holes,
    }[op]
    kw = {
        "footprint": int(params.get("footprint") or 1) or None,
        "iterations": int(params.get("iterations") or 1),
        "mode": str(params.get("mode") or "binary"),
        "connectivity": int(params.get("connectivity") or 2),
    }
    if op == "fill_holes":
        # nvitk.morphology.binary.fill_holes does not accept connectivity; it uses scipy.ndimage.binary_fill_holes.
        kw = {"mode": kw["mode"]}
    with run_with_backend():
        return coerce_tool_output(fn(img, **kw))


def run_gui_tool(
    tool_id: str,
    layer: Any,
    viewer: Any,
    *,
    target_mode: str,
    label_ids: list[int] | None,
    params = None,
) -> np.ndarray | None:
    """Central GUI tool dispatcher: resolve *tool_id* in the registry and route to its implementation
    (pipeline CLI subprocess, a dedicated viz/measure handler, or a generic array-in/array-out tool
    call on *layer*'s prepared data). Returns the new layer data, or ``None`` for tools that manage
    their own output (pipelines, viewer overlays, notifications)."""
    spec = tool_by_id(tool_id)
    if spec is None:
        raise ValueError(f"Unknown tool id: {tool_id}")

    params = dict(params or {})

    if spec.run_mode == "pipeline":
        _run_pipeline_cli(spec, params)
        return None

    if tool_id == "centerline_detect_junctions":
        _run_centerline_detect_junctions(viewer, layer, label_ids, params)
        return None

    if tool_id == "centerline_to_polyline":
        _run_centerline_to_polyline(viewer, layer, label_ids, params)
        return None

    if tool_id == "viz_vessel_cross_sections":
        _run_viz_vessel_cross_sections(viewer, layer, params)
        return None

    if tool_id == "measure_voxelwise":
        _run_measure_voxelwise(viewer)
        return None

    if tool_id == "viz_voxelwise_3d":
        _run_viz_voxelwise_3d(viewer)
        return None

    if tool_id == "lab_mouse_tof_cow":
        from nvitk.gui.lab.mouse_tof_cow import start_mouse_tof_cow

        start_mouse_tof_cow(viewer, layer)
        return None

    if tool_id == "measure_morphometrics":
        _run_measure_morphometrics(viewer, layer, params)
        return None

    if tool_id == "viz_flowshow":
        _run_viz_flowshow_napari(viewer, layer, params, label_ids=label_ids)
        return None

    if tool_id == "viz_flow_streamlines":
        _run_viz_flow_streamlines_napari(viewer, layer, params, label_ids=label_ids)
        return None

    if tool_id in {"viz_vessel_hemo", "viz_pitc", "viz_pwv"}:
        _run_viz_vessel_hemo(viewer, layer, params)
        return None

    if tool_id == "viz_tof_morphometrics":
        _run_viz_tof_morphometrics(viewer, layer, params)
        return None

    if tool_id == "measure_generate_suv":
        from nvitk.measure.suv import suv_image

        pet = layer_to_image(layer)
        if pet.ndim != 3:
            raise ValueError("Generate SUV requires a 3D PET image layer.")
        suv_out = suv_image(
            pet,
            kind=str(params.get("suv_kind") or "bw"),
            philips=bool(params.get("philips_factor", True)),
            revert_scaling=bool(params.get("revert_scaling", False)),
        )
        notify(
            f"SUV ({params.get('suv_kind') or 'bw'}) volume ready "
            f"(shape {tuple(suv_out.data.shape)})."
        )
        return to_numpy(suv_out.data)

    if tool_id == "qvtpy_locs":
        _run_qvtpy_locs(viewer, layer, params)
        return None

    if tool_id.startswith("qvtpy_stage"):
        _run_qvtpy_stage(tool_id, params)
        return None

    if tool_id == "measure_centerline_arc_length":
        _run_measure_centerline_arc_length(viewer, layer, label_ids, params)
        return None

    if tool_id == "measure_loc_hemodynamics":
        _run_measure_loc_hemodynamics(viewer, layer, params)
        return None

    if tool_id == "export_view_png":
        from nvitk.gui.viz.view_capture import export_view_png

        out = str(params.get("output_path") or "view.png").strip()
        if not out:
            raise ValueError("Set an output PNG path.")
        export_view_png(
            viewer,
            out,
            canvas_only=bool(params.get("canvas_only", True)),
        )
        notify(f"Saved 3D view → {out}")
        return None

    if tool_id == "export_view_gif":
        from nvitk.gui.viz.view_capture import export_view_gif

        out = str(params.get("output_path") or "view.gif").strip()
        if not out:
            raise ValueError("Set an output GIF path.")
        t_axis = int(params.get("time_axis", -1))
        n = export_view_gif(
            viewer,
            out,
            fps=float(params.get("gif_fps") or 8.0),
            time_axis=t_axis if t_axis >= 0 else None,
            canvas_only=bool(params.get("canvas_only", True)),
            layer=layer,
        )
        notify(f"Saved {n}-frame GIF → {out}")
        return None

    if tool_id == "orient_volume":
        from nvitk.gui.core.spatial import orientation_text
        from nvitk.gui.core.orientation import configure_viewer_for_layer
        from nvitk.io._common import orientation_codes_from_affine

        raw = layer_data_for_tool(layer.data)
        src = layer_to_image(layer, raw)
        if src.affine is None:
            raise ValueError("Active layer has no affine; orientation requires spatial metadata.")
        if src.ndim != 3:
            raise ValueError("View / reorient orientation requires a 3D layer.")
        current = orientation_codes_from_affine(src.affine) or "unknown"
        target = str(params.get("target_orientation") or "RAS").upper()
        mode = str(params.get("orient_mode") or "view").lower()
        if mode == "view":
            detail = orientation_text(layer, viewer)
            notify(
                f"Current orientation: {current}\n"
                f"Target (reorient): {target}\n\n{detail}"
            )
            return None
        if current == target:
            notify(f"Layer is already {target}.")
            return None
        from nvitk.gui.core.orientation import reorient_layer_for_view

        previous, new_axes = reorient_layer_for_view(layer, target)
        new_aff = np.asarray(to_numpy(layer.affine), dtype=float)
        meta = dict(getattr(layer, "metadata", None) or {})
        if new_axes:
            meta["axes"] = new_axes
        nv = dict(meta.get("nvitk_metadata") or {})
        nv["affine"] = new_aff
        nv["orientation"] = target
        for i, key in enumerate(("x_res", "y_res", "z_res")):
            nv[key] = float(np.linalg.norm(new_aff[:3, i]))
        spacing = (nv["x_res"], nv["y_res"], nv["z_res"])
        nv["spacing"] = spacing
        meta["nvitk_metadata"] = nv
        meta["orientation"] = target
        layer.metadata = meta
        if new_axes and len(new_axes) == int(layer.data.ndim):
            layer.axis_labels = tuple(new_axes)
        configure_viewer_for_layer(viewer, layer, configure_dims=True)
        try:
            viewer.reset_view()
        except Exception:
            pass
        notify(f"Reoriented {previous} → {target}.")
        return None

    if tool_id == "reorient_volume":
        from nvitk.gui.core.orientation import configure_viewer_for_layer
        from nvitk.io._common import orientation_codes_from_affine
        from nvitk.transform.reorient import reorient_volume

        raw = layer_data_for_tool(layer.data)
        src = layer_to_image(layer, raw)
        if src.ndim != 3:
            raise ValueError("Reorient requires a 3D layer.")

        mode = str(params.get("reorient_mode") or "mouse").strip().lower()
        ref_img = None
        ref_name = str(params.get("reference_layer") or "").strip()
        if mode in ("reference", "ref") or ref_name:
            if not ref_name or ref_name.lower() in ("(none)", "none", ""):
                if mode in ("reference", "ref"):
                    raise ValueError("Reference mode requires a reference layer.")
            else:
                ref_layer = _resolve_layer(viewer, ref_name)
                ref_img = layer_to_image(ref_layer)

        flip_axes = (
            bool(params.get("flip_x")),
            bool(params.get("flip_y")),
            bool(params.get("flip_z")),
        )
        with using(get_global_backend()):
            out = reorient_volume(
                src,
                mode=mode,
                reference=ref_img,
                target_orientation=str(params.get("target_orientation") or "LAS"),
                permute_order=str(params.get("permute_order") or "0,1,2"),
                flip_axes=flip_axes,
                reset_affine=bool(params.get("reset_affine")),
            )

        arr = to_numpy(out.data)
        new_aff = out.affine
        if new_aff is None:
            raise ValueError("Reorient produced no affine.")
        new_aff = np.asarray(to_numpy(new_aff), dtype=float)
        codes = orientation_codes_from_affine(new_aff) or (
            out.orientation or str(params.get("target_orientation") or "")
        )

        name = f"{layer.name}_reorient"
        meta = dict(getattr(layer, "metadata", None) or {})
        nv = dict(nvitk_metadata_from_layer(layer))
        nv["affine"] = new_aff
        nv["orientation"] = codes
        for i, key in enumerate(("x_res", "y_res", "z_res")):
            nv[key] = float(np.linalg.norm(new_aff[:3, i]))
        nv["spacing"] = (nv["x_res"], nv["y_res"], nv["z_res"])
        nv["shape"] = tuple(int(s) for s in arr.shape)
        meta["nvitk_metadata"] = nv
        meta["orientation"] = codes
        if out.axes:
            meta["axes"] = out.axes

        new_layer = viewer.add_image(arr, name=name, affine=new_aff, metadata=meta)
        if out.axes and len(out.axes) == int(arr.ndim):
            try:
                new_layer.axis_labels = tuple(out.axes)
            except Exception:
                pass
        configure_viewer_for_layer(viewer, new_layer, configure_dims=True)
        try:
            viewer.reset_view()
        except Exception:
            pass
        notify(
            f"Reoriented → {name} ({codes}, shape={tuple(arr.shape)}, "
            f"spacing={[round(nv[k], 4) for k in ('x_res', 'y_res', 'z_res')]})"
        )
        return None

    bk = get_global_backend()
    data = layer_data_for_tool(layer.data)
    if not gpu_enabled():
        data = as_backend_array(to_numpy(data))

    if tool_id == "skeletonize":
        from nvitk.morphology.centerline import skeletonize_labeled

        work, labels = _skeletonize_input_and_labels(
            layer, target_mode=target_mode, label_ids=label_ids
        )
        if not gpu_enabled():
            work = as_backend_array(to_numpy(work))
        with using(bk):
            # ``skeletonize_labeled`` already paints the ids back, so this is a label map and must
            # keep its integer dtype rather than being cast to float like an intensity result.
            return coerce_label_output(
                skeletonize_labeled(work, labels=labels, min_points=1)
            )

    # ---- Per-label routing --------------------------------------------------------------------
    # A multi-label selection on a per-label tool runs the tool once on each label and recombines,
    # so the output carries the same parcellation the input did. Placed here so it is decided before
    # ``prepare_layer_data`` collapses the selection into its binary union.
    per_label = multilabel_selection(
        layer, spec=spec, target_mode=target_mode, label_ids=label_ids
    )
    if per_label:
        return _run_multilabel(tool_id, layer, viewer, label_ids=per_label, params=params)

    if tool_id in _MEASURE_NOTIFY:
        per_label_ids = label_ids
        if target_mode == "all_labels":
            per_label_ids = label_ids or _unique_labels(data)
            if not per_label_ids:
                raise ValueError("No non-zero labels in the active layer.")
            _run_measure_per_label(
                tool_id, layer, viewer, data, per_label_ids, params=params
            )
            return None
        if target_mode == "label" and label_ids and len(label_ids) > 1:
            _run_measure_per_label(
                tool_id, layer, viewer, data, label_ids, params=params
            )
            return None

    proc_data, _tag = prepare_layer_data(data, target_mode=target_mode, label_ids=label_ids)
    if not gpu_enabled():
        proc_data = as_backend_array(to_numpy(proc_data))
    img = layer_to_image(layer, proc_data)

    if spec.needs_3d and img.ndim != 3:
        raise ValueError(f"{spec.label} requires a 3D layer.")

    if tool_id == "bilateral":
        from nvitk.restoration import bilateral

        kw = {
            "do_3d": bool(params.get("do_3d")),
            "axis": int(params.get("axis") or 0),
            "backend": bk,
        }
        ss = float(params.get("sigma_spatial") or 0)
        sc = float(params.get("sigma_color") or 0)
        if ss > 0:
            kw["sigma_spatial"] = ss
        if sc > 0:
            kw["sigma_color"] = sc
        with using(bk):
            return coerce_tool_output(bilateral(img, **kw))

    if tool_id == "n4_bias":
        from nvitk.restoration import n4_bias_field_correction

        mask_img = None
        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name and ref_name not in ("", "(none)"):
            mask_layer = _resolve_layer(viewer, ref_name)
            _, mask_img, _ = align_mask_to_reference_layer(mask_layer, layer, order=0)
        shrink = int(params.get("shrink_factor") or 4)
        spline = float(params.get("spline_param") or 0.0)
        with using("numpy"):
            out = n4_bias_field_correction(
                img,
                mask=mask_img,
                shrink_factor=shrink,
                spline_param=spline if spline > 0 else None,
                rescale_intensities=bool(params.get("rescale_intensities")),
            )
            return coerce_tool_output(out)

    if tool_id == "mri_super_resolution":
        from nvitk.restoration import mri_super_resolution

        text = str(params.get("expansion_factor") or "1,1,2").replace(";", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 1:
            v = int(round(float(parts[0])))
            expansion = (v, v, v)
        elif len(parts) == 3:
            expansion = tuple(int(round(float(p))) for p in parts)
        else:
            raise ValueError("expansion_factor must be one value or three comma-separated integers")
        with using("numpy"):
            out = mri_super_resolution(
                img,
                expansion_factor=expansion,
                feature=str(params.get("sr_feature") or "vgg"),
            )
            return coerce_tool_output(out)

    if tool_id == "sliding_threshold":
        from nvitk.filters.sliding_threshold import (
            binary_mask_sliding_threshold_2d,
            binary_mask_sliding_threshold_3d,
        )

        arr = layer_data_for_tool(img.data)
        step = float(params.get("step") or 0.001)
        up = float(params.get("up_thresh") or 0.8)
        smf = int(params.get("smf") or 10)
        shift_hm = bool(params.get("shift_hm", True))
        with run_with_backend():
            if arr.ndim == 2:
                out = binary_mask_sliding_threshold_2d(
                    arr,
                    step=step,
                    up_thresh=up,
                    smf=smf,
                    shift_hm_flag=shift_hm,
                )
            elif arr.ndim == 3:
                out, _ = binary_mask_sliding_threshold_3d(
                    arr,
                    step=step,
                    up_thresh=up,
                    smf=smf,
                    shift_hm_flag=shift_hm,
                )
            else:
                raise ValueError(f"Sliding threshold expects 2D or 3D data, got {arr.ndim}D")
            return coerce_tool_output(out)

    if tool_id == "hessian_filter":
        from nvitk.filters.hessian import hessian_filter, parse_sigmas

        with using("numpy"):
            out = hessian_filter(
                img,
                sigmas=parse_sigmas(str(params.get("hessian_sigmas") or "")),
                black_ridges=bool(params.get("black_ridges", False)),
                alpha=float(params.get("hessian_alpha") or 0.5),
                beta=float(params.get("hessian_beta") or 0.5),
                gamma=float(params.get("hessian_gamma") or 15.0),
            )
            return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "jerman_filter":
        from nvitk.filters.jerman import jerman_filter, parse_sigmas

        with using("numpy"):
            out = jerman_filter(
                img,
                sigmas=parse_sigmas(str(params.get("jerman_sigmas") or "")),
                tau=float(params.get("jerman_tau") or 0.5),
                black_ridges=bool(params.get("black_ridges", False)),
            )
            return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "snakes_filter":
        from nvitk.gui.labels.visibility import is_label_like_layer, label_source_data
        from nvitk.filters.snakes import snakes_filter

        mask_name = _layer_param(params, "reference_layer")
        if not mask_name:
            raise ValueError("Select an init contour mask layer in the reference dropdown.")
        mask_layer = _resolve_layer(viewer, mask_name)
        mask_src = label_source_data(mask_layer) if is_label_like_layer(mask_layer) else None
        _, mask_img, resampled = align_mask_to_reference_layer(
            mask_layer, layer, mask_src, order=0
        )
        if resampled:
            gui_log(
                f"Resampled init mask '{mask_layer.name}' onto image '{layer.name}' "
                f"grid {tuple(img.data.shape)}."
            )
        with using("numpy"):
            def _f(key: str, default: float) -> float:
                """Read ``params[key]`` as a float, falling back to *default* when unset."""
                v = params.get(key)
                return float(default if v is None else v)

            def _i(key: str, default: int) -> int:
                """Read ``params[key]`` as an int, falling back to *default* when unset."""
                v = params.get(key)
                return int(default if v is None else v)

            out = snakes_filter(
                img,
                mask_img,
                alpha=_f("snakes_alpha", 0.01),
                beta=_f("snakes_beta", 0.1),
                w_line=_f("snakes_w_line", 0.0),
                w_edge=_f("snakes_w_edge", 1.0),
                gamma=_f("snakes_gamma", 0.01),
                max_num_iter=_i("snakes_max_iter", 2500),
                gaussian_sigma=_f("snakes_sigma", 1.0),
                n_points=_i("snakes_n_points", 400),
                axis=_i("snakes_axis", 0),
            )
            return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id in ("img_mask_keep_inside", "img_mask_keep_outside"):
        from nvitk.gui.labels.visibility import is_label_like_layer, label_source_data
        from nvitk.segmentation.mask_ops import apply_mask_to_image

        mask_name = _layer_param(params, "reference_layer")
        if not mask_name:
            raise ValueError("Select a mask / segmentation layer in the reference dropdown.")
        mask_layer = _resolve_layer(viewer, mask_name)
        # Prefer unfiltered label source so hidden labels still contribute if listed.
        mask_src = label_source_data(mask_layer) if is_label_like_layer(mask_layer) else None
        _, mask_img, resampled = align_mask_to_reference_layer(
            mask_layer, layer, mask_src, order=0
        )
        if resampled:
            gui_log(
                f"Resampled mask '{mask_layer.name}' onto image '{layer.name}' "
                f"grid {tuple(img.data.shape)}."
            )
        lids = parse_label_ids(str(params.get("mask_label_ids") or ""))
        mode = "keep_inside" if tool_id == "img_mask_keep_inside" else "keep_outside"
        with using(bk):
            out = apply_mask_to_image(
                img,
                mask_img,
                mode=mode,
                fill_value=float(params.get("fill_value") or 0.0),
                label_ids=lids or None,
            )
            gui_log(
                f"Mask apply ({mode}): fill={params.get('fill_value') or 0.0}"
                + (f", labels={lids}" if lids else ", labels=all nonzero")
            )
            return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id in ("dilate", "erode", "open", "close", "fill_holes"):
        return _morph_common(img, tool_id, params)

    if tool_id == "label_cc":
        from nvitk.morphology.components import label_connected

        with using(bk):
            labeled, _n = label_connected(
                img, connectivity=int(params.get("connectivity") or 1)
            )
            return coerce_tool_output(labeled)

    if tool_id == "remove_small_components":
        from nvitk.morphology.components import remove_small_components

        with using(bk):
            return coerce_tool_output(
                remove_small_components(
                    img,
                    min_size=int(params.get("min_size") or 64),
                    connectivity=int(params.get("connectivity") or 1),
                )
            )

    if tool_id == "volume_projection":
        from nvitk.gui.core.orientation import _axes_string_from_layer
        from nvitk.transform.projection import project_volume

        axis = int(params.get("projection_axis", -1))
        if axis < 0:
            axes_str = _axes_string_from_layer(layer)
            if axes_str and "Z" in axes_str.upper():
                axis = axes_str.upper().index("Z")
            else:
                axis = min(2, int(img.ndim) - 1)
        method = str(params.get("projection_method") or "max")
        if int(img.ndim) < 3:
            raise ValueError("Volume projection requires at least 3 dimensions.")
        if axis < 0 or axis >= int(img.ndim):
            raise ValueError(f"Projection axis {axis} is invalid for ndim={img.ndim}.")
        with using(bk):
            out_img = project_volume(img, axis=axis, method=method)
        notify(
            f"Projection ({method}) along axis {axis}: "
            f"{tuple(img.data.shape)} → {tuple(out_img.data.shape)}"
        )
        return coerce_tool_output(out_img.data)

    if tool_id == "isotropy":
        from nvitk.transform.isotropy import isotropy

        axis = int(params.get("axis", -1))
        factor = float(params.get("factor") or 0)
        order = int(params.get("order") or 1)
        kw = {"order": order}
        if axis >= 0:
            kw["axis"] = axis
        if factor > 0:
            kw["factor"] = factor
        with using(bk):
            return coerce_tool_output(isotropy(img, **kw))

    if tool_id == "rotate_volume":
        from nvitk.transform.rotate import rotate_volume

        with using(bk):
            out = rotate_volume(
                img,
                float(params.get("angle_degrees") if params.get("angle_degrees") is not None else 90.0),
                axis=int(params.get("axis") if params.get("axis") is not None else 2),
                order=int(params.get("order") if params.get("order") is not None else 1),
                reshape=bool(params.get("reshape")),
            )
            return coerce_tool_output(getattr(out, "data", out))

    if tool_id == "swap_axes":
        from nvitk.transform.swap_axes import swap_axes

        a0 = int(params.get("swap_axis0") if params.get("swap_axis0") is not None else 0)
        a1 = int(params.get("swap_axis1") if params.get("swap_axis1") is not None else 1)
        with using(bk):
            out = swap_axes(img, a0, a1)
            return coerce_tool_output(getattr(out, "data", out))

    if tool_id == "resample_to":
        from nvitk.transform.resampling import resample_to

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        target = layer_to_image(ref_layer)
        with using(bk):
            return coerce_tool_output(
                resample_to(img, target, order=int(params.get("order") or 0))
            )

    if tool_id == "oblique_slice":
        from nvitk.transform.oblique import oblique_slice

        arr = as_backend_array(img.data).astype(np.float64)
        cx, cy, cz = [s / 2.0 for s in arr.shape]
        center = (cx, cy, cz)
        u = (0.0, 1.0, 0.0)
        v = (0.0, 0.0, 1.0)
        with using(bk):
            slc = oblique_slice(
                arr,
                center_xyz=center,
                u_xyz=u,
                v_xyz=v,
                radius_vox=float(params.get("radius_vox") or 40),
                res=int(params.get("res") or 256),
                order=int(params.get("order") or 1),
            )
            return coerce_tool_output(slc)

    if tool_id == "volume_mm3":
        from nvitk.measure.volume import volume_mm3

        vol = volume_mm3(img)
        notify(f"Volume: {vol:.3f} mm³")
        return None

    if tool_id == "volume_cc":
        from nvitk.measure.volume import volume_cc

        vol = volume_cc(img)
        notify(f"Volume: {vol:.6f} cc ({vol * 1000:.3f} mm³)")
        return None

    if tool_id == "masked_stats":
        from nvitk.measure.intensity import masked_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        intensity, mask_img = _reference_and_mask_images(
            viewer, layer, ref_name, proc_data
        )
        stats = masked_stats(intensity, mask_img)
        notify(_format_metrics({k: round(float(v), 6) for k, v in stats.items()}))
        return None

    if tool_id == "integrated_intensity":
        from nvitk.measure.radiomics import integrated_intensity

        ref_name = str(params.get("reference_layer") or "").strip()
        intensity, mask_img = _reference_and_mask_images(
            viewer, layer, ref_name, proc_data
        )
        val = integrated_intensity(intensity, mask_img)
        notify(f"Integrated intensity: {val:.6g}")
        return None

    if tool_id == "suv_stats":
        from nvitk.measure.suv import suv_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name:
            pet, mask_img = _reference_and_mask_images(viewer, layer, ref_name, proc_data)
        else:
            pet = img
            mask_img = img
        stats = suv_stats(pet, mask_img, **_suv_stats_kwargs(params))
        notify(_format_metrics({k: round(float(v), 6) for k, v in stats.items()}))
        return None

    if tool_id == "intensity_similarity":
        _run_intensity_similarity(viewer, layer, params)
        return None

    if tool_id == "dice":
        from nvitk.measure.voxel import dice

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, proc_data)
        val = dice(ref_img, mask_img)
        notify(f"Dice: {val:.6f}")
        return None

    if tool_id == "jaccard":
        from nvitk.measure.voxel import jaccard

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, proc_data)
        val = jaccard(ref_img, mask_img)
        notify(f"Jaccard: {val:.6f}")
        return None

    if tool_id == "voxel_metrics":
        from nvitk.measure.voxel import voxel_metrics

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, proc_data)
        metrics = voxel_metrics(ref_img, mask_img)
        notify(_format_metrics({k: round(float(v), 6) for k, v in metrics.items()}))
        return None

    if tool_id == "surface_metrics":
        from nvitk.measure.surface import surface_metrics

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, proc_data)
        metrics = surface_metrics(ref_img, mask_img)
        notify(_format_metrics({k: round(float(v), 6) for k, v in metrics.items()}))
        return None

    if tool_id == "measure_mask_hemodynamics":
        mask_vol = data if target_mode in ("label", "all_labels", "raw") else proc_data
        lids = params.get("selected_label_ids") or label_ids
        if target_mode == "all_labels" and not lids:
            lids = _unique_labels(mask_vol)
        _run_measure_mask_hemodynamics(viewer, layer, mask_vol, params, label_ids=lids)
        return None

    if tool_id == "viz_pet_hotspots":
        # prepare_layer_data already merged selected labels into a binary ROI mask.
        hotspot_label_ids = (
            None if target_mode in ("label", "binary_mask", "all_labels") else label_ids
        )
        _run_viz_pet_hotspots(
            viewer, layer, proc_data, params, label_ids=hotspot_label_ids
        )
        return None

    if tool_id == "centerline_cut_junctions":
        return _run_centerline_cut_junctions(viewer, layer, label_ids, params)

    if tool_id == "siphon_correct":
        from nvitk.morphology.centerline_siphon import correct_siphon_centerlines

        tof_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        tof_img = layer_to_image(tof_layer)
        ids = params.get("selected_label_ids") or parse_label_ids(
            str(params.get("correction_ids") or "1,2")
        )
        if not ids:
            ids = [1, 2]
        out_dir = str(params.get("output_dir") or "").strip() or None
        notify("Running ICA siphon correction (CPU, may take a while)…")
        result = correct_siphon_centerlines(
            tof_img,
            img,
            correction_ids=ids,
            out_dir=out_dir,
            save_qc=bool(out_dir),
        )
        n_cl = len(result.get("centerlines", {}))
        notify(f"Siphon correction done. Centerline sets: {n_cl}. Outputs: {out_dir or '(none)'}")
        return None

    if tool_id == "mask_genus":
        from nvitk.morphology.centerline_siphon import compute_mask_genus

        report = compute_mask_genus(proc_data)
        notify(
            f"Genus: β₀={report.beta0} β₁={report.beta1} "
            f"(raw β₁={report.beta1_raw}, max cycle={report.max_cycle_len})"
        )
        return None

    if tool_id == "seg_get_label":
        from nvitk.segmentation.labels import get_label

        lid = int(params.get("label_id") or 1)
        return coerce_tool_output(get_label(img, lid))

    if tool_id == "seg_combine_labels":
        from nvitk.segmentation.labels import combine_labels

        ids = parse_label_ids(str(params.get("label_ids") or ""))
        new_id = int(params.get("new_id") or 1)
        return coerce_tool_output(combine_labels(img, ids, new_id=new_id))

    if tool_id == "seg_remove_labels":
        from nvitk.segmentation.labels import remove_labels

        ids = parse_label_ids(str(params.get("label_ids") or ""))
        return coerce_tool_output(remove_labels(img, ids))

    if tool_id == "seg_pet_ureter":
        return _run_seg_pet_ureter(viewer, layer, params)

    if tool_id == "seg_convex_hull_slice":
        from nvitk.segmentation.hull_edt import convex_hull_slicewise

        out = convex_hull_slicewise(img, axis=int(params.get("hull_axis") or 2))
        return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "seg_convex_hull_3d":
        from nvitk.segmentation.hull_edt import convex_hull_3d

        out = convex_hull_3d(img)
        return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "seg_distance_transform":
        from nvitk.gui.core.spatial import layer_spacing
        from nvitk.segmentation.hull_edt import distance_transform

        sp = layer_spacing(layer) if bool(params.get("edt_use_spacing", True)) else None
        spacing = None
        if sp is not None and len(sp) >= 3:
            spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
        rad = float(params.get("radius_mm") or 0.0)
        out = distance_transform(
            img,
            spacing=spacing,
            radius_mm=rad if rad > 0 else None,
        )
        return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "seg_mask_union":
        return _run_mask_binary_op(viewer, layer, proc_data, params, "mask_union")

    if tool_id == "seg_mask_intersection":
        return _run_mask_binary_op(viewer, layer, proc_data, params, "mask_intersection")

    if tool_id == "seg_mask_subtract":
        return _run_mask_binary_op(viewer, layer, proc_data, params, "mask_subtract")

    if tool_id == "seg_mask_xor":
        return _run_mask_binary_op(viewer, layer, proc_data, params, "mask_xor")

    if tool_id == "seg_mask_complement":
        from nvitk.segmentation.mask_ops import mask_complement

        a_img = layer_to_image(layer, proc_data)
        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name:
            ref_layer = _resolve_layer(viewer, ref_name)
            _, within_img, _ = align_mask_to_reference_layer(
                ref_layer, layer, proc_data, order=0
            )
            out = mask_complement(a_img, within_img)
        else:
            out = mask_complement(a_img)
        return coerce_tool_output(out.data if hasattr(out, "data") else out)

    if tool_id == "seg_biggest_cc":
        from nvitk.segmentation.labels import biggest_cc

        n_keep = max(1, int(params.get("n_largest") or 1))
        return coerce_tool_output(biggest_cc(img, n=n_keep))

    if tool_id == "morph_biggest_cc":
        from nvitk.morphology import make_ball_footprint
        from nvitk.segmentation.labels import biggest_cc

        conn = int(params.get("connectivity") or 1)
        n_keep = max(1, int(params.get("n_largest") or 1))
        rank = min(3, int(proc_data.ndim))
        # Base-tool structuring element (equals generate_binary_structure at r=1).
        structure = make_ball_footprint(rank, connectivity=conn)
        return coerce_tool_output(biggest_cc(img, structure=structure, n=n_keep))

    if tool_id == "seg_split_lr_cc":
        from nvitk.segmentation.hemisphere import split_lr_by_cc

        left, right = split_lr_by_cc(img)
        notify("Split L/R: added left and right layers (see new images).")
        viewer.add_image(coerce_tool_output(left), **_layer_kwargs_from(layer, "split_L"))
        viewer.add_image(coerce_tool_output(right), **_layer_kwargs_from(layer, "split_R"))
        return None

    if tool_id == "seg_split_lr_midline":
        from nvitk.segmentation.hemisphere import split_lr_by_midline

        px = int(params.get("plane_x") or 0)
        left, right = split_lr_by_midline(img, plane_x=px if px > 0 else None)
        viewer.add_image(coerce_tool_output(left), **_layer_kwargs_from(layer, "midline_L"))
        viewer.add_image(coerce_tool_output(right), **_layer_kwargs_from(layer, "midline_R"))
        notify("Midline L/R split added as new layers.")
        return None

    if tool_id == "seg_region_grow":
        from nvitk.segmentation.region_growing import (
            forbidden_from_label_mask,
            forbidden_other_labels,
            merge_forbidden,
            region_grow_binary_mask,
        )

        intensity_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        ref = layer_to_image(intensity_layer)
        _, mask_img, mask_resampled = align_mask_to_reference_layer(
            layer, intensity_layer, proc_data, order=0
        )
        if mask_resampled:
            gui_log(
                f"Resampled mask '{layer.name}' onto intensity '{intensity_layer.name}' "
                f"grid {tuple(ref.data.shape)}."
            )
        mask = as_backend_array(mask_img.data).astype(bool).copy()
        intensity = as_backend_array(ref.data).astype(np.float64)
        seed_ids = list(params.get("selected_label_ids") or label_ids or [])
        if bool(params.get("seed_from_label")) and seed_ids:
            from nvitk.gui.tools.presets import label_centroid_voxel

            sz, sy, sx = label_centroid_voxel(mask, int(seed_ids[0]))
        else:
            sz = int(params.get("seed_z") or 0)
            sy = int(params.get("seed_y") or 0)
            sx = int(params.get("seed_x") or 0)
        mask[sz, sy, sx] = True
        thresh = float(params.get("threshold") or 0.0)
        mask_rad = int(
            params.get("mask_barrier_dilation_vox")
            or params.get("barrier_radius_vox")
            or 0
        )
        cl_rad = int(params.get("centerline_barrier_dilation_vox") or 0)
        forbidden = None
        exclude = [int(x) for x in seed_ids]

        def _layer_name(raw: str) -> str:
            """Normalize a layer-dropdown value, treating ``"(none)"`` and blank as unset."""
            name = str(raw or "").strip()
            return "" if name in ("", "(none)") else name

        bar_name = _layer_name(params.get("barrier_layer"))
        if bar_name:
            bar_layer = _resolve_layer(viewer, bar_name)
            _, bar_img, _ = align_mask_to_reference_layer(
                bar_layer, intensity_layer, order=0
            )
            bar_vol = as_backend_array(bar_img.data)
            if len(np.unique(to_numpy(bar_vol))) > 2:
                forb = forbidden_other_labels(bar_vol, exclude, radius_vox=mask_rad)
            else:
                forb = forbidden_from_label_mask(
                    bar_vol, radius_vox=mask_rad, exclude_label_ids=exclude
                )
            forbidden = merge_forbidden(forbidden, forb)

        cl_bar_name = _layer_name(params.get("centerline_barrier_layer"))
        if cl_bar_name:
            cl_layer = _resolve_layer(viewer, cl_bar_name)
            _, cl_img, _ = align_mask_to_reference_layer(
                cl_layer, intensity_layer, order=0
            )
            cl_vol = as_backend_array(cl_img.data)
            if len(np.unique(to_numpy(cl_vol))) > 2:
                cl_forb = forbidden_other_labels(cl_vol, exclude, radius_vox=cl_rad)
            else:
                cl_forb = forbidden_from_label_mask(
                    cl_vol, radius_vox=cl_rad, exclude_label_ids=exclude
                )
            forbidden = merge_forbidden(forbidden, cl_forb)

        if bool(params.get("barrier_other_labels")):
            _, label_img, _ = align_mask_to_reference_layer(
                layer, intensity_layer, data, order=0
            )
            label_vol = as_backend_array(label_img.data)
            if not exclude:
                growing = as_backend_array(mask).astype(bool)
                for lid in np.unique(to_numpy(label_vol)):
                    lid = int(lid)
                    if lid != 0 and np.any(label_vol[growing] == lid):
                        exclude.append(lid)
            if not exclude:
                raise ValueError(
                    "Barrier other labels: select label id(s) or seed a labeled region."
                )
            other_forb = forbidden_other_labels(label_vol, exclude, radius_vox=mask_rad)
            forbidden = merge_forbidden(forbidden, other_forb)
        region_grow_binary_mask(
            mask,
            intensity,
            intensity_frac=thresh,
            abs_floor=None,
            forbidden=forbidden,
            polarity="hyperintense",
        )
        return mask.astype(np.uint8)

    if tool_id == "seg_blood_flood":
        from nvitk.segmentation.blood_flood import blood_flood, blood_flood_from_scratch

        mode = str(params.get("blood_flood_mode") or "expand").strip().lower()
        sigmas_raw = str(params.get("frangi_sigmas") or "").strip()
        if sigmas_raw:
            frangi_sigmas = tuple(
                float(x) for x in sigmas_raw.replace(";", ",").split(",") if x.strip()
            )
        else:
            frangi_sigmas = None
        thin_pct = float(
            params.get("thin_vesselness_percentile")
            if params.get("thin_vesselness_percentile") is not None
            else 55.0
        )
        thin = None if thin_pct < 0 else thin_pct
        common_kw = {
            "frangi_sigmas": frangi_sigmas,
            "hyst_low_factor": float(params.get("hyst_low_factor") or 3.0),
            "hyst_high_factor": float(params.get("hyst_high_factor") or 0.5),
            "thicken_iter": int(params.get("thicken_iter") or 0),
            "thin_vesselness_percentile": thin,
            "connectivity": int(params.get("connectivity") or 3),
        }

        barrier_arr = None
        bar_name = _layer_param(params, "barrier_layer")

        if mode in ("from_scratch", "scratch", "de_novo", "segment"):
            # Active layer = intensity; optional mask_layer = ROI / brain mask.
            intensity = img
            roi_mask = None
            mask_name = _layer_param(params, "mask_layer")
            if mask_name:
                roi_layer = _resolve_layer(viewer, mask_name)
                _, roi_img, _ = align_mask_to_reference_layer(roi_layer, layer, order=0)
                roi_mask = to_numpy(roi_img.data)
            if bar_name:
                bar_layer = _resolve_layer(viewer, bar_name)
                _, bar_img, _ = align_mask_to_reference_layer(bar_layer, layer, order=0)
                barrier_arr = to_numpy(bar_img.data)
            with using(bk):
                result = blood_flood_from_scratch(
                    to_numpy(intensity.data),
                    mask=roi_mask,
                    barrier=barrier_arr,
                    min_cc_voxels=int(params.get("min_cc_voxels") or 5),
                    **common_kw,
                )
                gui_log(
                    f"Blood flood (from_scratch): vesselness={result.vesselness_mode}, "
                    f"tree_voxels={int(np.count_nonzero(result.tree))}, "
                    f"components={result.info.get('n_components')}, "
                    f"labeled={int(np.count_nonzero(result.labels))}"
                )
                return coerce_tool_output(result.labels)

        # expand: active = markers; reference_layer = intensity (required).
        # Keep distinct seed label ids (do not binarize) so watershed returns
        # a multilabel segmentation; non-selected labels act as hard barriers.
        intensity_name = _layer_param(params, "reference_layer")
        if not intensity_name:
            raise ValueError(
                "expand mode requires an intensity image layer "
                "(CD/TOF) in the reference layer dropdown."
            )
        intensity_layer = _resolve_layer(viewer, intensity_name)
        ref = layer_to_image(intensity_layer)

        from nvitk.gui.labels.visibility import label_source_data, unique_layer_labels

        src_markers = to_numpy(label_source_data(layer))
        sel_ids = [int(x) for x in (label_ids or [])]
        if not sel_ids:
            sel_ids = unique_layer_labels(src_markers)
        if not sel_ids:
            raise ValueError("expand mode needs seed label(s) on the active layer.")
        sel_arr = np.asarray(sel_ids, dtype=src_markers.dtype)
        markers_ml = np.where(
            np.isin(src_markers, sel_arr), src_markers, 0
        ).astype(np.int32, copy=False)
        other_labels = (src_markers != 0) & ~np.isin(src_markers, sel_arr)

        _, markers_img, markers_resampled = align_mask_to_reference_layer(
            layer, intensity_layer, markers_ml, order=0
        )
        if markers_resampled:
            gui_log(
                f"Resampled markers '{layer.name}' onto intensity '{intensity_layer.name}' "
                f"grid {tuple(ref.data.shape)}."
            )
        markers_np = to_numpy(markers_img.data).astype(np.int32, copy=False)

        if bar_name:
            bar_layer = _resolve_layer(viewer, bar_name)
            _, bar_img, _ = align_mask_to_reference_layer(bar_layer, intensity_layer, order=0)
            barrier_arr = to_numpy(bar_img.data).astype(bool, copy=False)
        if np.any(other_labels):
            _, other_img, _ = align_mask_to_reference_layer(
                layer, intensity_layer, other_labels.astype(np.uint8), order=0
            )
            other_np = to_numpy(other_img.data).astype(bool, copy=False)
            barrier_arr = other_np if barrier_arr is None else (barrier_arr | other_np)

        with using(bk):
            result = blood_flood(
                to_numpy(ref.data),
                markers_np,
                barrier=barrier_arr,
                **common_kw,
            )
            n_lab = int(len(np.unique(to_numpy(result.labels))) - (
                1 if np.any(to_numpy(result.labels) == 0) else 0
            ))
            gui_log(
                f"Blood flood (expand): vesselness={result.vesselness_mode}, "
                f"seeds={sel_ids}, tree_voxels={int(np.count_nonzero(result.tree))}, "
                f"output_labels={n_lab}, labeled={int(np.count_nonzero(result.labels))}"
            )
            return coerce_tool_output(result.labels)

    if tool_id == "seg_mouse_brain":
        from nvitk.segmentation.mouse_brain import mouse_brain_segmentation

        mask_img = None
        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name and ref_name not in ("", "(none)"):
            mask_layer = _resolve_layer(viewer, ref_name)
            _, mask_img, _ = align_mask_to_reference_layer(mask_layer, layer, order=0)
        gui_log(
            "Mouse brain: running ANTsPyNet "
            f"(mode={params.get('mouse_brain_mode')!r}, "
            f"modality={params.get('mouse_modality')!r}, "
            f"parc={params.get('which_parcellation')!r}, "
            f"n4={bool(params.get('do_n4', True))})"
        )
        with using("numpy"):
            out = mouse_brain_segmentation(
                img,
                mode=str(params.get("mouse_brain_mode") or "extraction"),
                modality=str(params.get("mouse_modality") or "t2"),
                which_parcellation=str(params.get("which_parcellation") or "nick"),
                mask=mask_img,
                do_n4=bool(params.get("do_n4", True)),
                binarize=bool(params.get("binarize", True)),
                return_isotropic_output=bool(params.get("return_isotropic_output", False)),
                fix_spacing=bool(params.get("fix_spacing", True)),
                verbose=True,
            )
            gui_log(
                f"Mouse brain: finished → shape={getattr(out, 'shape', None)} "
                f"dtype={getattr(out, 'dtype', None)}"
            )
            return coerce_tool_output(out)

    if tool_id == "seg_brain_extraction":
        from nvitk.segmentation.brain_extraction import brain_extraction

        modality = str(params.get("brain_modality") or "t1").lower()
        image2_name = str(params.get("image2_layer") or "").strip()
        ants_in: Any = img
        if image2_name and image2_name not in ("", "(none)"):
            layer2 = _resolve_layer(viewer, image2_name)
            img2 = layer_to_image(layer2)
            ants_in = [img, img2]
        with using("numpy"):
            out = brain_extraction(ants_in, modality=modality)
            return coerce_tool_output(out)

    if tool_id == "seg_mra_vessel":
        from nvitk.segmentation.mra_vessel import mra_vessel_segmentation

        mask_img = None
        mask_name = _layer_param(params, "mask_layer")
        if not mask_name:
            # Back-compat if an older panel still sends reference_layer.
            mask_name = _layer_param(params, "reference_layer")
        if mask_name:
            mask_layer = _resolve_layer(viewer, mask_name)
            _, mask_img, _ = align_mask_to_reference_layer(mask_layer, layer, order=0)
        with using("numpy"):
            out = mra_vessel_segmentation(
                img,
                mask=mask_img,
                prediction_batch_size=int(params.get("prediction_batch_size") or 2),
                patch_stride_length=int(params.get("patch_stride_length") or 32),
            )
            return coerce_tool_output(out)

    if tool_id == "seg_dkt":
        from nvitk.segmentation.dkt import desikan_killiany_tourville_labeling

        with using("numpy"):
            out = desikan_killiany_tourville_labeling(
                img,
                do_preprocessing=bool(params.get("dkt_preprocessing", True)),
                do_lobar_parcellation=bool(params.get("dkt_lobar", False)),
                do_denoising=bool(params.get("dkt_denoising", True)),
                version=int(params.get("dkt_version") or 0),
            )
            return coerce_tool_output(out)

    if tool_id == "seg_adjust_masks":
        from nvitk.segmentation.labels import adjust_masks

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        mask2 = layer_to_image(ref_layer)
        m1, m2 = adjust_masks(img, mask2)
        viewer.add_image(coerce_tool_output(m1), **_layer_kwargs_from(layer, "adjusted"))
        viewer.add_image(coerce_tool_output(m2), **_layer_kwargs_from(ref_layer, "adjusted"))
        notify("Adjusted masks: overlapping slices kept on both layers.")
        return None

    if tool_id == "seg_totalsegmentator":
        from nvitk.io import imsave
        from nvitk.segmentation.total_segmentator import run_totalsegmentator

        out_dir = Path(str(params.get("output_dir") or "").strip() or tempfile.mkdtemp(prefix="nvitk_ts_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        inp = out_dir / "input.nii.gz"
        imsave(inp, img)
        task = str(params.get("task") or "total")
        roi = params.get("roi_subset")
        notify(f"Running TotalSegmentator ({task})…")
        if roi:
            gui_log(f"ROI subset ({len(roi)}): {', '.join(roi[:8])}{'…' if len(roi) > 8 else ''}")
        proc = run_totalsegmentator(
            inp, out_dir, task, roi_subset=roi, capture_output=True, check=False
        )
        if proc.stdout:
            for line in str(proc.stdout).splitlines():
                gui_log(line)
        if proc.stderr:
            for line in str(proc.stderr).splitlines():
                gui_log(line, error=True)
        if proc.returncode != 0:
            notify(f"TotalSegmentator failed (code {proc.returncode})", error=True)
            return None
        notify(f"TotalSegmentator finished. Output: {out_dir}")
        return None

    if tool_id == "seg_eicab":
        notify(
            "eICAB is cluster-oriented. Use: nvitk-eicab --help "
            "or run the QVTpy / PESA-FAT pipeline stages with eICAB enabled."
        )
        return None

    if tool_id == "reg_flirt_rigid":
        from nvitk.registration.fsl.flirt import flirt_register_rigid

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        moving_path = _ensure_nifti_path(layer, prefix="flirt_moving")
        fixed_path = _ensure_nifti_path(ref_layer, prefix="flirt_fixed")
        out_dir = Path(
            str(params.get("output_dir") or "").strip()
            or tempfile.mkdtemp(prefix="nvitk_flirt_reg_")
        )
        dof = int(params.get("dof") or 6)
        cost = str(params.get("cost") or "corratio").strip() or "corratio"
        searchr = float(params.get("searchr_x") or 0)
        searchr_x = searchr if searchr > 0 else None
        warped_name = str(params.get("warped_name") or "moving_warped.nii.gz").strip()
        matrix_name = str(params.get("matrix_name") or "affine.mat").strip()
        notify(f"FLIRT rigid: moving→fixed, output {out_dir}")
        res = flirt_register_rigid(
            moving_path,
            fixed_path,
            out_dir,
            dof=dof,
            cost=cost,
            warped_name=warped_name,
            matrix_name=matrix_name,
            searchr_x=searchr_x,
        )
        notify(f"FLIRT matrix: {res.matrix_path}")
        if res.warped_path is not None:
            notify(f"FLIRT warped: {res.warped_path}")
            from nvitk.io import imread

            warped = imread(res.warped_path)
            viewer.add_image(
                coerce_tool_output(warped),
                **_layer_kwargs_from(ref_layer, "flirt_warped"),
            )
        return None

    if tool_id == "reg_flirt_apply":
        from nvitk.registration.fsl.flirt import flirt_apply_rigid

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        mat_path = Path(str(params.get("mat_path") or "").strip())
        if not mat_path.is_file():
            raise ValueError("Set a valid FLIRT matrix path (.mat).")
        out_s = str(params.get("out_path") or "").strip()
        if out_s:
            out_path = Path(out_s)
        else:
            out_path = Path(tempfile.mkdtemp(prefix="nvitk_flirt_apply_")) / "resampled.nii.gz"
        in_path = _ensure_nifti_path(layer, prefix="flirt_in")
        ref_path = _ensure_nifti_path(ref_layer, prefix="flirt_ref")
        interp = str(params.get("interp") or "trilinear").strip() or "trilinear"
        notify(f"FLIRT apply: {in_path.name} → {out_path}")
        flirt_apply_rigid(in_path, ref_path, mat_path, out_path, interp=interp)
        from nvitk.io import imread

        out_img = imread(out_path)
        viewer.add_image(
            coerce_tool_output(out_img),
            **_layer_kwargs_from(ref_layer, "flirt_applied"),
        )
        notify(f"FLIRT output layer from {out_path}")
        return None

    if tool_id == "reg_ants_register":
        from nvitk.registration.ants import ants_register

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        moving_path = _ensure_nifti_path(layer, prefix="ants_moving")
        fixed_path = _ensure_nifti_path(ref_layer, prefix="ants_fixed")
        out_dir = Path(str(params.get("output_dir") or "").strip() or tempfile.mkdtemp(prefix="nvitk_ants_reg_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        tot = str(params.get("type_of_transform") or "SyN").strip() or "SyN"
        write_comp = bool(params.get("write_composite_transform") or False)
        verbose = bool(params.get("verbose") or False)
        notify(f"ANTsPy register: moving→fixed ({tot}), output {out_dir}")
        res = ants_register(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_dir=out_dir,
            type_of_transform=tot,
            write_composite_transform=write_comp,
            verbose=verbose,
        )
        notify(f"ANTs warped: {res.warped_moving_path}")
        try:
            from nvitk.io import imread

            warped = imread(res.warped_moving_path)
            viewer.add_image(
                coerce_tool_output(warped),
                **_layer_kwargs_from(ref_layer, "ants_warped"),
            )
        except Exception:
            pass
        return None

    if tool_id == "reg_ants_apply":
        from nvitk.registration.ants import ants_apply

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        fixed_path = _ensure_nifti_path(ref_layer, prefix="ants_fixed")
        moving_path = _ensure_nifti_path(layer, prefix="ants_moving")
        tr = str(params.get("transform_paths") or "").strip()
        if not tr:
            raise ValueError("Set transform_paths (comma-separated).")
        transform_paths = [Path(p.strip()) for p in tr.split(",") if p.strip()]
        for p in transform_paths:
            if not p.is_file():
                raise ValueError(f"Transform not found: {p}")
        out_s = str(params.get("out_path") or "").strip()
        out_path = Path(out_s) if out_s else (Path(tempfile.mkdtemp(prefix="nvitk_ants_apply_")) / "resampled.nii.gz")
        interpolator = str(params.get("interpolator") or "linear").strip() or "linear"
        verbose = bool(params.get("verbose") or False)
        notify(f"ANTs apply: {moving_path.name} → {out_path}")
        ants_apply(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_path=out_path,
            transforms=transform_paths,
            interpolator=interpolator,
            verbose=verbose,
        )
        from nvitk.io import imread

        out_img = imread(out_path)
        viewer.add_image(
            coerce_tool_output(out_img),
            **_layer_kwargs_from(ref_layer, "ants_applied"),
        )
        notify(f"ANTs output layer from {out_path}")
        return None

    if tool_id == "reg_fireants_register":
        from nvitk.registration.fireants import fireants_register

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        moving_path = _ensure_nifti_path(layer, prefix="fireants_moving")
        fixed_path = _ensure_nifti_path(ref_layer, prefix="fireants_fixed")
        out_dir = Path(str(params.get("output_dir") or "").strip() or tempfile.mkdtemp(prefix="nvitk_fireants_reg_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        device = str(params.get("device") or "cuda:0").strip() or "cuda:0"
        verbose = bool(params.get("verbose") or False)
        notify(f"FireANTs register: moving→fixed (device={device}), output {out_dir}")
        res = fireants_register(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_dir=out_dir,
            device=device,
            verbose=verbose,
        )
        notify(f"FireANTs warped: {res.warped_moving_path}")
        try:
            from nvitk.io import imread

            warped = imread(res.warped_moving_path)
            viewer.add_image(
                coerce_tool_output(warped),
                **_layer_kwargs_from(ref_layer, "fireants_warped"),
            )
        except Exception:
            pass
        return None

    if tool_id == "reg_fireants_apply":
        from nvitk.registration.fireants import fireants_apply

        ref_layer = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        fixed_path = _ensure_nifti_path(ref_layer, prefix="fireants_fixed")
        moving_path = _ensure_nifti_path(layer, prefix="fireants_moving")
        tr = str(params.get("transform_paths") or "").strip()
        if not tr:
            raise ValueError("Set transform_paths (comma-separated).")
        transform_paths = [Path(p.strip()) for p in tr.split(",") if p.strip()]
        for p in transform_paths:
            if not p.is_file():
                raise ValueError(f"Transform not found: {p}")
        out_s = str(params.get("out_path") or "").strip()
        out_path = Path(out_s) if out_s else (Path(tempfile.mkdtemp(prefix="nvitk_fireants_apply_")) / "resampled.nii.gz")
        notify(f"FireANTs apply: {moving_path.name} → {out_path}")
        fireants_apply(
            fixed_path=fixed_path,
            moving_path=moving_path,
            out_path=out_path,
            transforms=transform_paths,
        )
        from nvitk.io import imread

        out_img = imread(out_path)
        viewer.add_image(
            coerce_tool_output(out_img),
            **_layer_kwargs_from(ref_layer, "fireants_applied"),
        )
        notify(f"FireANTs output layer from {out_path}")
        return None

    raise NotImplementedError(f"Tool '{tool_id}' is not implemented in the GUI runner.")


def _image_to_mock_layer(img: Image, name: str) -> Any:
    """Minimal Napari-like layer for headless :func:`run_gui_tool`."""
    from types import SimpleNamespace

    data = to_numpy(img.data)
    meta = dict(img.metadata or {})
    scale = meta.get("spacing")
    if scale is not None:
        scale = tuple(float(x) for x in scale[: max(3, data.ndim)])
    else:
        scale = tuple(1.0 for _ in range(min(3, data.ndim)))
    aff = meta.get("affine")
    return SimpleNamespace(
        name=name,
        data=data,
        metadata=meta,
        scale=scale,
        affine=aff,
    )


def _build_headless_viewer(primary_layer: Any, aux_layers: list[Any]) -> Any:
    """Build a minimal object mimicking a Napari viewer's ``.layers`` API (iteration + active-layer
    selection), for running GUI tools outside Napari on a cluster worker."""
    from types import SimpleNamespace

    layers_list = [primary_layer, *aux_layers]

    class _Layers:
        """Minimal stand-in for Napari's ``LayerList``: iterable with an active-selection namespace."""

        def __init__(self, items: list[Any]) -> None:
            """Store *items* and select the first one (if any) as active."""
            self._items = items
            self.selection = SimpleNamespace(active=items[0] if items else None)

        def __iter__(self):
            """Iterate over the mock layers."""
            return iter(self._items)

    return SimpleNamespace(layers=_Layers(layers_list))


def run_gui_tool_headless(
    tool_id: str,
    *,
    primary: Image,
    aux: dict[str, Image],
    target_mode: str,
    label_ids: list[int] | None,
    params = None,
) -> np.ndarray | None:
    """Run a layer tool without Napari (cluster worker entry point)."""
    primary_name = str(getattr(primary, "name", None) or "input")
    primary_layer = _image_to_mock_layer(primary, primary_name)
    aux_layers = [_image_to_mock_layer(img, name) for name, img in (aux or {}).items()]
    viewer = _build_headless_viewer(primary_layer, aux_layers)
    with using("numpy"):
        result = run_gui_tool(
            tool_id,
            primary_layer,
            viewer,
            target_mode=target_mode,
            label_ids=label_ids,
            params=dict(params or {}),
        )
    if result is None:
        return None
    return to_numpy(result)


def _layer_spacing(layer: Any) -> tuple[float, float, float]:
    """*layer*'s (x, y, z) voxel spacing, or ``(1.0, 1.0, 1.0)`` if unavailable."""
    from nvitk.gui.core.spatial import layer_spacing

    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)


def _phase_arrays_from_layers_or_disk(
    viewer: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load AP/RL/FH phase volumes and voxel spacing, from the three named layers if all are set, else
    by discovering and reading them from ``nifti_root/subject``; raises ``ValueError`` if neither
    source is fully specified."""
    ap_name = str(params.get("ap_layer") or "").strip()
    rl_name = str(params.get("rl_layer") or "").strip()
    fh_name = str(params.get("fh_layer") or "").strip()
    if ap_name and rl_name and fh_name:
        ap = as_backend_array(_resolve_layer(viewer, ap_name).data).astype(np.float64)
        rl = as_backend_array(_resolve_layer(viewer, rl_name).data).astype(np.float64)
        fh = as_backend_array(_resolve_layer(viewer, fh_name).data).astype(np.float64)
        return ap, rl, fh, _layer_spacing(_resolve_layer(viewer, ap_name))

    subject = str(params.get("subject") or "").strip()
    nifti_root = str(params.get("nifti_root") or "").strip()
    if not subject or not nifti_root:
        raise ValueError("Provide AP/RL/FH layers or subject + nifti_root.")
    from nvitk.io.conversors.phase2volume import discover_phase_inputs
    from nvitk.io.imageio import imread

    root = Path(nifti_root)
    inputs = discover_phase_inputs(root / subject)
    ap_img = imread(inputs.ap_phase_path)
    rl_img = imread(inputs.rl_phase_path)
    fh_img = imread(inputs.fh_phase_path)
    ap = as_backend_array(ap_img.data).astype(np.float64)
    rl = as_backend_array(rl_img.data).astype(np.float64)
    fh = as_backend_array(fh_img.data).astype(np.float64)
    sp = ap_img.spacing
    if sp is not None and len(sp) >= 3:
        voxel_spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
    else:
        voxel_spacing = (1.0, 1.0, 1.0)
    return ap, rl, fh, voxel_spacing


def _load_contrast_volumes(nifti_root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load *subject*'s 4D-flow contrast volumes from disk: ``(magnitude, complex_difference,
    velocity_magnitude)``, falling back to derived values (CD as magnitude, ``abs(CD)`` as velocity)
    when the dedicated Angiography/VelocityMagnitude files aren't present."""
    from nvitk.io.imageio import imread

    sub = nifti_root / subject / "4DFlow"
    cd_p = sub / "ComplexDifference_3D.nii.gz"
    if not cd_p.is_file():
        cd_p = sub / "ComplexDifference_3D.nii"
    cd = as_backend_array(imread(cd_p).data).astype(np.float64)
    mag = cd
    angio = sub / "Angiography_3D.nii.gz"
    if angio.is_file():
        mag = as_backend_array(imread(angio).data).astype(np.float64)
    vel = np.abs(cd)
    vmag = sub / "VelocityMagnitude_3D.nii.gz"
    if vmag.is_file():
        vel = as_backend_array(imread(vmag).data).astype(np.float64)
    return mag, cd, vel


def _run_mask_binary_op(
    viewer: Any,
    layer: Any,
    mask_data: Any,
    params: dict[str, Any],
    op_name: str,
) -> np.ndarray:
    """Apply a binary mask op (``and``/``or``/``subtract``/etc., named by *op_name*) between *layer*'s
    label-derived mask and *mask_data*'s mask (both converted from label ids if needed)."""
    from nvitk.segmentation import mask_ops

    def _as_bool_mask_from_labels(data: np.ndarray, ids: list[int] | None) -> np.ndarray:
        """Coerce label/mask *data* to a boolean array, restricted to *ids* if given (else non-zero)."""
        arr = np.asarray(data)
        if not ids:
            return (arr != 0)
        return np.isin(arr, _label_ids_array(ids))

    ref_name = str(params.get("reference_layer") or "").strip()
    if not ref_name:
        raise ValueError("Select the second mask layer (reference).")
    ref_layer = _resolve_layer(viewer, ref_name)

    # A: active layer selected labels (preferred) or fallback to precomputed mask_data.
    a_ids = list(params.get("selected_label_ids") or [])
    if not a_ids:
        # label_ids already passed in by run_gui_tool via prepare_layer_data; keep compatibility.
        a_img = layer_to_image(layer, mask_data)
    else:
        a_img_raw = layer_to_image(layer)
        a_mask = _as_bool_mask_from_labels(to_numpy(a_img_raw.data), a_ids).astype(np.uint8)
        a_img = a_img_raw.with_data(as_backend_array(a_mask))

    # B: reference layer label selection (optional).
    ref_ids = parse_label_ids(str(params.get("reference_label_ids") or ""))
    b_mask_data = to_numpy(layer_to_image(ref_layer).data)
    b_mask = _as_bool_mask_from_labels(b_mask_data, ref_ids).astype(np.uint8)
    b_img_raw = layer_to_image(ref_layer).with_data(as_backend_array(b_mask))

    _, b_img, resampled = align_mask_to_reference_layer(ref_layer, layer, b_img_raw.data, order=0)
    if resampled:
        gui_log(f"Resampled '{ref_layer.name}' onto active layer grid.")
    fn = getattr(mask_ops, op_name)
    out = fn(a_img, b_img)
    data = out.data if hasattr(out, "data") else out
    return coerce_tool_output(data)


def _run_seg_pet_ureter(viewer: Any, layer: Any, params: dict[str, Any]) -> np.ndarray:
    """Run MCP-path PET ureter segmentation from the PET/organ/body layers named in *params*, using
    the kidney/bladder label ids to seed the search."""
    from nvitk.measure.suv import suv_image
    from nvitk.segmentation.labels import get_label
    from nvitk.segmentation.pet.ureter_segmentation import segment_ureter

    pet_name = str(params.get("reference_layer") or "").strip()
    if not pet_name:
        raise ValueError("Select PET image layer.")
    pet_layer = _resolve_layer(viewer, pet_name)
    pet_img = layer_to_image(pet_layer)

    organ_name = str(params.get("organ_layer") or "").strip()
    organ_layer = _resolve_layer(viewer, organ_name) if organ_name else layer

    body_name = str(params.get("body_layer") or "").strip()
    if not body_name:
        raise ValueError("Select body mask layer.")
    body_layer = _resolve_layer(viewer, body_name)

    _, organ_img, _ = align_mask_to_reference_layer(organ_layer, pet_layer, order=0)
    _, body_img, _ = align_mask_to_reference_layer(body_layer, pet_layer, order=0)

    kr_id = int(params.get("kidney_r_id") or 2)
    kl_id = int(params.get("kidney_l_id") or 3)
    bl_id = int(params.get("bladder_id") or 21)

    kidney_r = get_label(organ_img, kr_id, missing="empty")
    kidney_l = get_label(organ_img, kl_id, missing="empty")
    bladder = get_label(organ_img, bl_id, missing="empty")

    suv = suv_image(
        pet_img,
        kind=str(params.get("suv_kind") or "bw"),
        philips=bool(params.get("philips_factor", True)),
        revert_scaling=bool(params.get("revert_scaling", False)),
    )
    notify("Running PET ureter segmentation (MCP paths)…")
    mask_ureter, _paths, _paths_sp = segment_ureter(
        suv,
        kidney_r,
        kidney_l,
        bladder,
        body_img,
        radius_mm=float(params.get("radius_mm") or 6.0),
        w_pet=float(params.get("w_pet") or 5.0),
    )
    notify("Ureter segmentation finished.")
    return coerce_tool_output(mask_ureter.data)


def _run_centerline_detect_junctions(
    viewer: Any,
    layer: Any,
    label_ids: list[int] | None,
    params: dict[str, Any],
) -> None:
    """Detect skeleton junctions in *layer*'s centerline mask and add them as a Points overlay,
    notifying with the count found."""
    from nvitk.gui.viz.centerline import (
        JUNCTION_POINTS_LAYER,
        add_junction_points_layer,
        detect_junctions_from_layer,
    )

    lid = label_ids[0] if label_ids else None
    junctions = detect_junctions_from_layer(
        layer,
        label_id=lid,
        min_degree=int(params.get("min_junction_degree") or 3),
        reskeletonize=bool(params.get("reskeletonize", False)),
    )
    add_junction_points_layer(
        viewer,
        junctions,
        reference_layer=layer,
        source_layer_name=str(getattr(layer, "name", "")),
        min_degree=int(params.get("min_junction_degree") or 3),
    )
    if junctions.shape[0] == 0:
        notify("No skeleton junctions found (try lowering min degree).", error=True)
        return
    notify(f"Marked {junctions.shape[0]} junction(s) on '{JUNCTION_POINTS_LAYER}'.")


def _run_centerline_to_polyline(
    viewer: Any,
    layer: Any,
    label_ids: list[int] | None,
    params: dict[str, Any],
) -> None:
    """Extract polyline (main + branch) representations from *layer*'s 3D centerline mask for the
    given/all label ids and add them as a Shapes overlay, notifying with a summary count."""
    from nvitk.gui.viz.centerline import (
        DEFAULT_POLYLINE_LAYER,
        add_centerline_polylines_shapes,
        centerline_mask_to_polylines,
    )

    arr = to_numpy(layer.data)
    if arr.ndim != 3:
        raise ValueError("To polyline expects a 3D centerline mask layer.")
    if not bool((arr > 0).any()):
        raise ValueError("Centerline layer has no foreground voxels.")

    labs = [int(x) for x in (label_ids or []) if int(x) != 0]
    if not labs:
        labs = sorted(int(v) for v in np.unique(arr) if int(v) != 0)
    if not labs:
        raise ValueError("No non-zero labels in the centerline mask.")

    raw_min = params.get("min_branch_points", 0)
    try:
        min_bp = int(raw_min) if raw_min is not None else 0
    except (TypeError, ValueError):
        min_bp = 0
    # 0 / negative → keep all edges (None semantics).
    min_branch_points = None if min_bp <= 0 else min_bp

    polylines = centerline_mask_to_polylines(
        arr,
        labels=labs,
        min_branch_points=min_branch_points,
        reskeletonize=bool(params.get("reskeletonize", False)),
        smooth=True,
    )
    if not polylines:
        notify("No centerline polylines extracted (empty or too short).", error=True)
        return

    src = str(getattr(layer, "name", "centerline") or "centerline")
    layer_name = f"{src} polylines"
    shapes = add_centerline_polylines_shapes(
        viewer,
        polylines,
        reference_layer=layer,
        layer_name=layer_name or DEFAULT_POLYLINE_LAYER,
        edge_width=float(params.get("edge_width") or 0.35),
    )
    n_labels = len({int(p["label"]) for p in polylines})
    n_main = sum(1 for p in polylines if p.get("role") == "main")
    n_branch = len(polylines) - n_main
    notify(
        f"To polyline: {len(polylines)} path(s) from {n_labels} label(s) "
        f"({n_main} main, {n_branch} branch) → '{getattr(shapes, 'name', layer_name)}'."
    )


def _resolve_centerline_mask_layer(viewer: Any) -> Any:
    """3D centerline mask recorded when junctions were detected."""
    from nvitk.gui.viz.centerline import JUNCTION_META_KEY, JUNCTION_POINTS_LAYER

    for lyr in viewer.layers:
        if lyr.name != JUNCTION_POINTS_LAYER:
            continue
        meta = getattr(lyr, "metadata", None) or {}
        jmeta = meta.get(JUNCTION_META_KEY) if isinstance(meta, dict) else None
        if isinstance(jmeta, dict):
            src_name = str(jmeta.get("source_layer") or "").strip()
            if src_name:
                for ref in viewer.layers:
                    if ref.name == src_name:
                        return ref
    raise ValueError(
        f"No '{JUNCTION_POINTS_LAYER}' layer with source metadata. "
        "Run Detect skeleton junctions on the centerline mask first."
    )


def _run_centerline_cut_junctions(
    viewer: Any,
    layer: Any,
    label_ids: list[int] | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Split the selected label at previously detected skeleton junctions, returning the updated
    label volume with new label ids for each severed branch."""
    from nvitk.gui.viz.centerline import (
        centerline_polyline_for_label,
        read_junction_coords,
        split_label_at_junctions,
    )

    arr = to_numpy(layer.data)
    if arr.ndim != 3:
        raise ValueError("Junction cut requires a 3D labels/image layer.")

    if not label_ids:
        raise ValueError("Select one label id to split at junctions.")
    src_lid = int(label_ids[0])
    junctions = read_junction_coords(viewer)
    centerline_layer = _resolve_centerline_mask_layer(viewer)
    poly = centerline_polyline_for_label(
        centerline_layer,
        src_lid,
        reskeletonize=bool(params.get("reskeletonize", False)),
    )
    start_raw = int(params.get("new_label_start") or 0)
    new_start = None if start_raw <= 0 else start_raw
    vol, new_ids = split_label_at_junctions(
        arr,
        src_lid,
        poly,
        junctions,
        new_label_start=new_start,
    )
    notify(
        f"Split label {src_lid} at {junctions.shape[0]} junction(s) → new labels {new_ids}."
    )
    return vol


def _run_qvtpy_locs(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Load LOC points into the viewer, either from an existing CSV or by first running QVTpy stage-5
    LOC generation as a subprocess and loading its output."""
    from nvitk.gui.viz.loc_points import add_locs_layer, load_locs_csv
    from nvitk.pipes.qvtpy import config as qcfg

    mode = str(params.get("loc_mode") or "load_csv").strip()
    if mode == "generate":
        subject = str(params.get("subject") or "").strip()
        nifti_root = str(params.get("nifti_root") or "").strip()
        output_root = str(params.get("output_root") or "").strip()
        if not subject or not nifti_root or not output_root:
            raise ValueError("Generate mode requires subject, nifti_root, and output_root.")
        argv = [
            sys.executable,
            "-m",
            "nvitk.pipes.qvtpy.stage5_loc_generation",
            "--subject",
            subject,
            "--nifti-root",
            nifti_root,
            "--output-root",
            output_root,
            "--loc-arterial-strategy",
            str(params.get("loc_arterial_strategy") or "qvtpy"),
            "--cross-section-radius-vox",
            str(float(params.get("cross_section_radius_vox") or 10.0)),
        ]
        notify(f"Running stage5 LOC generation: {' '.join(argv)}")
        code = run_subprocess_logged(argv)
        if code != 0:
            notify(f"Stage5 failed (code {code})", error=True)
            return
        csv_path = (
            Path(output_root) / subject / qcfg.QVT_SUBDIR / qcfg.STAGE5_LOC_DIR / "locs.csv"
        )
    else:
        csv_path = Path(str(params.get("locs_csv") or "").strip())
    rows = load_locs_csv(csv_path)
    ref = layer
    add_locs_layer(viewer, rows, reference_layer=ref)
    notify(
        f"Loaded {len(rows)} LOCs from {csv_path} "
        "(adjust size/symbol/color in the LOCs layer panel)."
    )


def _run_qvtpy_stage(tool_id: str, params: dict[str, Any]) -> None:
    """Run a single QVTpy pipeline stage (identified by *tool_id*) as a subprocess, notifying on
    success/failure."""
    from nvitk.gui.pipeline.qvtpy_stages import build_qvtpy_stage_argv

    argv = build_qvtpy_stage_argv(tool_id, params)
    notify(f"Running: {' '.join(argv)}")
    code = run_subprocess_logged(argv)
    if code == 0:
        notify(f"Stage finished: {tool_id}")
    else:
        notify(f"Stage failed (exit {code}): {tool_id}", error=True)


def _run_measure_centerline_arc_length(
    viewer: Any,
    layer: Any,
    label_ids: list[int] | None,
    params: dict[str, Any],
) -> None:
    """Compute and notify per-branch/per-label centerline arc lengths (voxel and mm units) for *layer*'s
    3D centerline mask, either for the selected labels or every connected-component branch (label 0)."""
    from nvitk.gui.viz.centerline import centerline_polyline_for_label
    from nvitk.morphology.polyline_graph import extract_polylines_from_centerline
    from nvitk.pipes.qvtpy.util.loc.loc_selection import polyline_cumulative_arc_length

    arr = to_numpy(layer.data)
    if arr.ndim != 3:
        raise ValueError("Centerline arc length requires a 3D layer.")

    scale = getattr(layer, "scale", None)
    if scale is not None and len(scale) >= 3:
        sx, sy, sz = float(scale[2]), float(scale[1]), float(scale[0])
    else:
        sx = sy = sz = 1.0

    reskel = bool(params.get("reskeletonize", False))
    lid_raw = int(params.get("label_id") or 0)
    targets = [int(x) for x in (label_ids or [])]
    if lid_raw > 0:
        targets = [lid_raw]
    elif not targets:
        targets = [0]

    lines = []
    if targets == [0]:
        polys = extract_polylines_from_centerline(
            arr,
            mode="junction_split",
            reskeletonize=reskel,
            per_connected_component=True,
        )
        for i, poly in enumerate(polys):
            cum = polyline_cumulative_arc_length(poly)
            vox_len = float(cum[-1]) if cum.size else 0.0
            seg = np.diff(poly.astype(np.float64), axis=0)
            mm_len = float(
                np.sum(np.linalg.norm(seg * np.array([sx, sy, sz], dtype=np.float64), axis=1))
            ) if seg.shape[0] else 0.0
            lines.append(f"branch[{i}]: {vox_len:.2f} vox, {mm_len:.2f} mm, {poly.shape[0]} points")
    else:
        for lid in targets:
            try:
                poly = centerline_polyline_for_label(
                    layer, lid, reskeletonize=reskel
                )
            except ValueError:
                polys = extract_polylines_from_centerline(
                    arr,
                    label_id=lid,
                    mode="junction_split",
                    reskeletonize=reskel,
                )
                if not polys:
                    lines.append(f"label {lid}: no polyline")
                    continue
                poly = max(polys, key=lambda p: p.shape[0])
            cum = polyline_cumulative_arc_length(poly)
            vox_len = float(cum[-1]) if cum.size else 0.0
            seg = np.diff(poly.astype(np.float64), axis=0)
            mm_len = float(
                np.sum(np.linalg.norm(seg * np.array([sx, sy, sz], dtype=np.float64), axis=1))
            ) if seg.shape[0] else 0.0
            lines.append(
                f"label {lid}: {vox_len:.2f} vox, {mm_len:.2f} mm, {poly.shape[0]} points"
            )

    notify("Centerline arc length:\n" + "\n".join(lines))


def _run_measure_loc_hemodynamics(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Compute per-LOC hemodynamic indices (PI/RI/flow) from AP/RL/FH phase volumes and a LOC CSV,
    writing ``loc_measurements.csv`` next to the input and notifying with a summary."""
    from nvitk.gui.viz.loc_points import load_locs_csv
    from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
    from nvitk.pipes.qvtpy import config as qcfg
    from nvitk.pipes.qvtpy.util.loc.loc_measure import run_loc_measurements

    csv_s = str(params.get("locs_csv") or "").strip()
    subject = str(params.get("subject") or "").strip()
    output_root = str(params.get("output_root") or "").strip()
    if not csv_s and subject and output_root:
        csv_path = (
            Path(output_root) / subject / qcfg.QVT_SUBDIR / qcfg.STAGE5_LOC_DIR / "locs.csv"
        )
    else:
        csv_path = Path(csv_s)
    loc_rows = load_locs_csv(csv_path)

    ap, rl, fh, voxel_spacing = _phase_arrays_from_layers_or_disk(viewer, params)
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)

    nifti_root = str(params.get("nifti_root") or "").strip()
    volume_seg = None
    measure_resegment = bool(params.get("measure_resegment", False))
    if not measure_resegment and subject and nifti_root:
        seg_p = (
            Path(output_root or ".")
            / subject
            / qcfg.QVT_SUBDIR
            / qcfg.STAGE4_SEG_DIR
            / "seg_4dflow.nii.gz"
        )
        if seg_p.is_file():
            from nvitk.io.imageio import imread

            volume_seg = as_backend_array(imread(seg_p).data).astype(np.int32)

    if subject and nifti_root:
        mag, cd, vel_mag = _load_contrast_volumes(Path(nifti_root), subject)
    else:
        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name:
            ref_data = as_backend_array(_resolve_layer(viewer, ref_name).data).astype(np.float64)
            mag = cd = vel_mag = ref_data
        else:
            vel_mag = np.sqrt(vx[..., 0] ** 2 + vy[..., 0] ** 2 + vz[..., 0] ** 2)
            mag = cd = vel_mag

    rows_out = run_loc_measurements(
        loc_rows,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        vx=vx,
        vy=vy,
        vz=vz,
        voxel_spacing=voxel_spacing,
        cross_section_radius_vox=float(params.get("cross_section_radius_vox") or 10.0),
        measure_resegment=measure_resegment,
        cs_supersampling=bool(params.get("cs_supersampling", True)),
        volume_seg=volume_seg,
    )
    out_csv = csv_path.parent / "loc_measurements.csv"
    if rows_out:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    lines = [
        f"{r['vessel_name']}: PI={r['loc_pi']:.3f} RI={r['loc_ri']:.3f} flow={r['loc_mean_flow_ml_s']:.2f} ml/s"
        for r in rows_out[:12]
    ]
    if len(rows_out) > 12:
        lines.append(f"... ({len(rows_out)} vessels total)")
    notify(f"Wrote {out_csv}\n" + "\n".join(lines))


def _run_measure_mask_hemodynamics(
    viewer: Any,
    layer: Any,
    mask_data: np.ndarray,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    """Compute PI/RI (and flow, per *method*) hemodynamic indices for each selected label id from the
    active mask and AP/RL/FH phase volumes, notifying with the per-label results."""
    from nvitk.measure.mask_hemodynamics import measure_mask_hemodynamics

    lids = list(label_ids or []) or [int(params.get("label_id") or 1)]
    ap_name = str(params.get("ap_layer") or "").strip()
    if ap_name:
        ap_layer = _resolve_layer(viewer, ap_name)
        _, mask_img, resampled = align_mask_to_reference_layer(
            layer, ap_layer, mask_data, order=0
        )
        if resampled:
            gui_log(
                f"Resampled mask '{layer.name}' onto phase grid '{ap_layer.name}' "
                f"{tuple(mask_img.data.shape)}."
            )
        mask_data = mask_img.data
    ap, rl, fh, voxel_spacing = _phase_arrays_from_layers_or_disk(viewer, params)
    ref_name = str(params.get("reference_layer") or "").strip()
    mag = cd = vel_mag = None
    if ref_name:
        ref_layer = _resolve_layer(viewer, ref_name)
        if ap_name:
            _, ref_on_ap, _ = align_mask_to_reference_layer(ref_layer, ap_layer, order=1)
            mag = cd = vel_mag = as_backend_array(ref_on_ap.data).astype(np.float64)
        else:
            mag = cd = vel_mag = as_backend_array(ref_layer.data).astype(np.float64)
    method = str(params.get("hemo_method") or "both")
    lines = []
    for lid in lids:
        results = measure_mask_hemodynamics(
            mask_data,
            ap,
            rl,
            fh,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            label_id=int(lid),
            method=method,
            voxel_spacing=voxel_spacing,
            radius_vox=float(params.get("cross_section_radius_vox") or 10.0),
            measure_resegment=bool(params.get("measure_resegment", False)),
            volume_seg=mask_data if not bool(params.get("measure_resegment", False)) else None,
        )
        for res in results:
            extra = ""
            if res.mean_flow_ml_s is not None:
                extra = f" flow={res.mean_flow_ml_s:.2f} ml/s"
            lines.append(
                f"Label {lid} [{res.method}]: PI={res.pi:.3f} RI={res.ri:.3f}{extra} — {res.note}"
            )
    notify("\n".join(lines))


def _prepare_vessel_hemo_for_viz(
    viewer: Any,
    layer: Any,
    params: dict[str, Any],
):
    """Gather everything needed to visualize per-vessel hemodynamics (velocity field, resampled
    segmentation, magnitude/CD reference, temporal resolution) from the AP/RL/FH/reference layers
    named in *params*, ready to hand off to :func:`~nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics.compute_vessel_hemodynamics`."""
    from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
    from nvitk.pipes.qvtpy.stage6_measure import _cardiac_frame_duration_s
    from nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics import compute_vessel_hemodynamics

    ap_name = _layer_param(params, "ap_layer")
    rl_name = _layer_param(params, "rl_layer")
    fh_name = _layer_param(params, "fh_layer")
    ref_name = _layer_param(params, "reference_layer")
    if not ap_name or not rl_name or not fh_name:
        raise ValueError("Select AP, RL, and FH phase layers.")
    if not ref_name:
        raise ValueError("Select an angio / complex-difference reference layer.")

    ap_layer = _resolve_layer(viewer, ap_name)
    segmentation = to_numpy(layer_to_image(layer).data)
    if segmentation.ndim != 3:
        raise ValueError("Active layer must be a 3D stage-4 multilabel segmentation.")

    _, seg_img, resampled = align_mask_to_reference_layer(
        layer, ap_layer, segmentation, order=0
    )
    if resampled:
        gui_log(
            f"Resampled segmentation '{getattr(layer, 'name', 'layer')}' onto "
            f"phase grid '{ap_layer.name}' {tuple(seg_img.data.shape)}."
        )
    segmentation = to_numpy(seg_img.data).astype(np.int32, copy=False)

    ap, rl, fh, voxel_spacing = _phase_arrays_from_layers_or_disk(viewer, params)
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)

    reference_layer = _resolve_layer(viewer, ref_name)
    _, ref_on_ap, _ = align_mask_to_reference_layer(reference_layer, ap_layer, order=1)
    mag = cd = vel_mag = as_backend_array(ref_on_ap.data).astype(np.float64)

    ap_data = to_numpy(ap_layer.data)
    n_t = int(ap_data.shape[3]) if getattr(ap_data, "ndim", 0) >= 4 else None
    temporal_resolution = None
    tr_source = "none"
    # Prefer an explicit frame duration (e.g. stage-6 measure_meta) when provided.
    tr_override = params.get("temporal_resolution_s")
    if tr_override is not None and str(tr_override).strip() != "":
        try:
            temporal_resolution = float(tr_override)
            tr_source = "params"
            gui_log(
                f"Cardiac frame duration {temporal_resolution * 1e3:.3f} ms "
                f"(source=params, n_t={n_t})"
            )
        except (TypeError, ValueError):
            temporal_resolution = None
    json_path = str(params.get("heart_rate_json") or "").strip()
    if temporal_resolution is None and json_path:
        from pathlib import Path
        import json as _json

        path = Path(json_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Cardiac metadata JSON not found: {path}")
        try:
            data_js = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read cardiac metadata JSON {path}: {exc}") from exc
        if not isinstance(data_js, dict):
            raise ValueError(f"Cardiac metadata JSON must be an object: {path}")
        temporal_resolution, tr_source = _cardiac_frame_duration_s(
            data_js, n_timepoints=n_t
        )
        if temporal_resolution is not None:
            gui_log(
                f"Cardiac frame duration {temporal_resolution * 1e3:.3f} ms "
                f"(source={tr_source} via {path.name}, n_t={n_t})"
            )
        else:
            gui_log(
                f"No usable HeartRate/NominalInterval/FrameTime in {path.name}; "
                "falling back to phase-layer metadata."
            )
    if temporal_resolution is None:
        meta = dict(getattr(ap_layer, "metadata", None) or {})
        temporal_resolution, tr_source = _cardiac_frame_duration_s(meta, n_timepoints=n_t)
        if temporal_resolution is not None:
            gui_log(
                f"Cardiac frame duration {temporal_resolution * 1e3:.3f} ms "
                f"(source={tr_source} from AP layer metadata, n_t={n_t})"
            )

    prefer_polylines = params.get("prefer_polylines")
    if prefer_polylines is not None and not isinstance(prefer_polylines, dict):
        prefer_polylines = None
    waveform_centerlines = params.get("waveform_centerlines")
    if waveform_centerlines is not None and not isinstance(waveform_centerlines, dict):
        waveform_centerlines = None

    hemo = compute_vessel_hemodynamics(
        waveform_centerlines,
        volume_seg=segmentation,
        prefer_polylines=prefer_polylines,
        cd=cd,
        mag=mag,
        vel_mag=vel_mag,
        vx=vx,
        vy=vy,
        vz=vz,
        voxel_spacing=voxel_spacing,
        temporal_resolution_s=temporal_resolution,
        stride=int(params.get("stride") or 1),
        radius_vox=float(params.get("cross_section_radius_vox") or 10.0),
        quality_thresh=float(params.get("quality_thresh") or 2.5),
        quality_metric=str(params.get("quality_metric") or "stdv_from_mean"),
        measure_resegment=bool(params.get("measure_resegment", False)),
        label_constrain=bool(params.get("label_constrain", True)),
        thr_algorithm=str(params.get("thr_algorithm") or "lsthr"),
        cross_section_res=int(params.get("cross_section_res") or 0),
        plane_interp_order=int(params.get("cross_section_plane_interp") or 1),
        cs_supersampling=bool(params.get("cs_supersampling", True)),
        collect_plot_data=True,
    )
    root_region = str(params.get("root_region") or "All").strip()
    regions = list(hemo.geometry_by_region.values())
    if root_region and root_region != "All":
        regions = [r for r in regions if r.region_id == root_region]
    if not regions:
        raise ValueError(f"No hemodynamics geometry available for root region {root_region!r}.")
    return hemo, regions, reference_layer


def _run_viz_voxelwise_3d(viewer: Any) -> None:
    """Open the 3-D voxelwise window.

    Reachable on its own: viewing a finished result should not require opening the analysis dialog
    that would have run it.
    """
    from nvitk.gui.viz.voxelwise_3d_panel import start_voxelwise_3d

    start_voxelwise_3d(viewer)
    notify("Voxelwise 3-D window open. Choose a results folder to draw it.")


def _run_measure_voxelwise(viewer: Any) -> None:
    """Open the voxelwise configure/run/load window.

    Takes no layer: the analysis reads a whole cohort from a directory, and the design matrix comes
    from the database, so there is nothing about the active layer that would inform it.
    """
    from nvitk.gui.viz.voxelwise_panel import start_voxelwise

    start_voxelwise(viewer)
    notify(
        "Voxelwise analysis window open. Configure a run, or load an existing results folder."
    )


def _run_viz_vessel_hemo(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run PITC + PWV together and open the shared diagnostics dock."""
    from nvitk.gui.viz.hemo_geometry import add_hemo_geometry_layers
    from nvitk.gui.viz.hemo_plot_panel import show_hemodynamics_plot

    hemo, regions, reference_layer = _prepare_vessel_hemo_for_viz(viewer, layer, params)
    add_hemo_geometry_layers(
        viewer,
        regions,
        reference_layer=reference_layer,
        mode="hemo",
        face_key="quality",
        point_size=float(params.get("station_point_size") or 2.5),
    )
    selected_ids = {region.region_id for region in regions}
    show_hemodynamics_plot(
        viewer,
        {key: value for key, value in hemo.region_plot_data.items() if key in selected_ids},
        mode="hemo",
        initial_plot="pitc",
    )
    lines = ["PITC / PWV geometry: init = min-Z endpoint of each root polyline."]
    for region in regions:
        bj = region.pwv_bjornfoot_m_s if region.pwv_bjornfoot_m_s != "" else "n/a"
        fi = region.pwv_fielding_m_s if region.pwv_fielding_m_s != "" else "n/a"
        init_xyz = ", ".join(f"{v:.1f}" for v in to_numpy(region.root_init_xyz))
        lines.append(
            f"{region.region_id}: PITC slope={region.pitc_slope:.4g} r2={region.pitc_r2:.3f} "
            f"n={region.pitc_n} | Bjornfoot={bj} Fielding={fi} "
            f"(n_pwv={region.pwv_n_stations}) | init=({init_xyz})"
        )
    notify("\n".join(lines))


# Backwards-compatible aliases (QC loader / older call sites).
_run_viz_pitc = _run_viz_vessel_hemo
_run_viz_pwv = _run_viz_vessel_hemo


def _run_viz_tof_morphometrics(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Load stage-7 morphometrics centerline VTPs for debugging."""
    from nvitk.gui.viz.morpho_viz import install_morphometrics_viz

    stage7 = str(params.get("stage7_dir") or "").strip()
    if not stage7:
        raise ValueError(
            "Set stage7_dir to the subject stage-7 folder "
            "(…/<subject>/qvtpy/stage7_morphometrics)."
        )
    stage7_path = Path(stage7)
    if not stage7_path.is_dir():
        raise FileNotFoundError(f"stage7_dir not found: {stage7_path}")

    ref_name = str(params.get("reference_layer") or "").strip()
    ref_layer = _resolve_layer(viewer, ref_name) if ref_name else layer
    info = install_morphometrics_viz(
        viewer,
        stage7_path,
        reference_layer=ref_layer,
        color_by=str(params.get("color_by") or "radius"),
        point_size=float(params.get("point_size") or 2.0),
        edge_width=float(params.get("edge_width") or 0.35),
    )
    notify(
        f"TOF morphometrics: loaded {info['n_paths']} centerline path(s) "
        f"(color_by={info['color_by']}) from {stage7_path}."
    )


def _run_measure_morphometrics(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run morphometrics on the selected multilabel Labels / Image layer.

    Empty ``output_dir``: write to a temp folder (not kept as a user result) and
    display centerlines / surfaces via the morphometrics viz dock.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs
    from nvitk.gui.viz.morpho_viz import DEFAULT_MORPHO_POINT_SIZE, install_morphometrics_viz
    from nvitk.io import imsave
    from nvitk.measure.morphometrics import run_morphometrics_case
    from nvitk.measure.morpho.anatomy_axes import SPECIES_AUTO
    from nvitk.measure.morpho.topology_io import TOPOLOGY_NONE

    if layer is None:
        raise ValueError("Select a multilabel segmentation layer first.")

    out_s = str(params.get("output_dir") or "").strip()
    persist = bool(out_s)
    if persist:
        out_dir = Path(out_s).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="nvitk_morphometrics_"))

    # Always export the *current* Labels data with the layer affine (never reuse a
    # stale intensity source path that would scramble spacing / orientation).
    import numpy as numpy

    seg = numpy.asarray(to_numpy(layer.data), dtype=numpy.int32)
    bridged = numpy.asarray(_bridge_same_label_components(seg, max_gap=24), dtype=numpy.int32)
    n_bridged = int(numpy.count_nonzero(bridged != seg))
    if n_bridged:
        notify(f"Bridged {n_bridged} voxel(s) to reconnect same-label vessel fragments.")

    seg_path = Path(tempfile.mkdtemp(prefix="nvitk_morphometrics_seg_")) / "seg.nii.gz"
    imsave(seg_path, layer_to_image(layer, data=bridged))

    topology = str(params.get("topology") or TOPOLOGY_NONE).strip() or TOPOLOGY_NONE
    species = str(params.get("species") or SPECIES_AUTO).strip() or SPECIES_AUTO
    n_workers = int(params.get("n_workers") or 1)
    skip_existing = bool(params.get("skip_existing")) and persist
    already_smoothed = bool(params.get("input_already_smoothed"))

    if persist:
        notify(f"Morphometrics running (topology={topology!r}, species={species!r}) → {out_dir} …")
    else:
        notify(
            f"Morphometrics running (topology={topology!r}, species={species!r}); "
            "no output directory — results displayed in GUI only…"
        )
    excel = run_morphometrics_case(
        seg_path,
        out_dir,
        mapping_json=topology,
        case_out_dir_override=out_dir,
        n_workers=n_workers,
        input_already_smoothed=already_smoothed,
        skip_if_excel_exists=skip_existing,
        species=species,
    )

    # Show the (bridged) vessel mask used for computation as a Labels layer.
    spatial = layer_spatial_kwargs(layer)
    mask_name = f"{getattr(layer, 'name', 'seg')}_morpho_input"
    try:
        mask_layer = viewer.add_labels(
            bridged.astype(numpy.int32, copy=False),
            name=mask_name,
            opacity=0.45,
            **spatial,
        )
        try:
            mask_layer._nvitk_label_like = True
        except Exception:
            pass
    except Exception as mask_exc:  # noqa: BLE001
        notify(f"Could not add morpho input mask layer: {mask_exc}", error=True)

    try:
        info = install_morphometrics_viz(
            viewer,
            out_dir,
            reference_layer=layer,
            color_by="radius",
            point_size=DEFAULT_MORPHO_POINT_SIZE,
            show_surfaces=True,
        )
        n_paths = int(info.get("n_paths") or 0)
        n_surf = int(info.get("n_surfaces") or 0)
    except Exception as viz_exc:  # noqa: BLE001
        n_paths = 0
        n_surf = 0
        notify(f"Morphometrics viz skipped: {viz_exc}", error=True)

    if persist:
        notify(
            f"Morphometrics done: {excel} "
            f"({n_paths} path(s), {n_surf} surface layer(s) shown)."
        )
    else:
        notify(
            f"Morphometrics done ({n_paths} path(s), {n_surf} surface(s) in GUI; "
            "nothing saved — set Output directory to persist)."
        )


def _run_intensity_similarity(
    viewer: Any,
    primary_layer: Any,
    params: dict[str, Any],
) -> None:
    """Pearson/Spearman/MAE/RMSE between two intensity images (no mask)."""
    from nvitk.measure.compare import correlation_stats

    ref_name = str(params.get("reference_layer") or "").strip()
    if not ref_name:
        raise ValueError("Select a second image layer to compare.")
    other_layer = _resolve_layer(viewer, ref_name)
    if other_layer is primary_layer:
        raise ValueError("Choose a different layer than the active image.")

    primary_img, other_on_primary, resampled = align_mask_to_reference_layer(
        other_layer, primary_layer, order=1
    )
    if resampled:
        gui_log(
            f"Resampled '{getattr(other_layer, 'name', 'layer')}' onto "
            f"'{getattr(primary_layer, 'name', 'layer')}' for comparison."
        )

    a = layer_data_for_tool(primary_img.data)
    b = layer_data_for_tool(other_on_primary.data)
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"Shape mismatch after alignment: {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    stats = correlation_stats(a, b)
    display = {
        k: round(float(v), 6) if k != "n_samples" else int(v)
        for k, v in stats.items()
    }
    notify(_format_metrics(display))


def _run_viz_pet_hotspots(
    viewer: Any,
    layer: Any,
    mask_data: np.ndarray,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    """Find SUV hotspots in the selected ROI of the reference PET layer and add them as a color-coded
    Points overlay."""
    from nvitk.gui.viz.layers import add_hotspot_points_layer, hotspot_points_from_volumes
    from nvitk.viz.pet_hotspots import HotspotMode

    ref_name = str(params.get("reference_layer") or "").strip()
    ref_layer = _resolve_layer(viewer, ref_name)
    ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, mask_data)
    mode = str(params.get("hotspot") or "top_percent")
    coords, _vals, features = hotspot_points_from_volumes(
        to_numpy(ref_img.data),
        to_numpy(mask_img.data),
        label_ids=label_ids,
        hotspot=mode,
        top_percent=float(params.get("top_percent") or 0.1),
        max_points=int(params.get("max_points") or 20000),
    )
    if coords.shape[0] == 0:
        notify("No hotspots found in the selected ROI.", error=True)
        return
    add_hotspot_points_layer(
        viewer,
        coords,
        features,
        reference_layer=ref_layer,
        point_size=float(params.get("point_size") or 6.0),
        colormap=str(params.get("cmap") or params.get("colormap") or "viridis"),
    )
    notify(
        f"Added {coords.shape[0]} SUV hotspot point(s) in Napari "
        f"(colormap={params.get('cmap') or 'viridis'}; adjust size/symbol in the layer panel)."
    )


def _run_viz_vessel_cross_sections(
    viewer: Any,
    centerline_layer: Any,
    params: dict[str, Any],
) -> None:
    """Set up the interactive vessel cross-section tool: resolve/align the CD, segmentation, and phase
    layers, load named arterial/venous branch geometry from nearby pipeline output if available, and
    install the pick-and-inspect session via
    :func:`~nvitk.gui.viz.vessel_cross_sections.install_vessel_cross_sections`."""
    from nvitk.gui.viz.vessel_cross_sections import install_vessel_cross_sections
    from nvitk.pipes.qvtpy.util.centerline.centerline_io import (
        CENTERLINE_SEG_BRANCHES_JSON,
        load_arterial_branches,
        load_centerline_meta,
        load_venous_centerlines,
    )
    from nvitk.pipes.qvtpy import config as qvt_cfg

    cd_name = _layer_param(params, "cd_layer")
    if not cd_name:
        raise ValueError(
            "Select a complex difference image layer in the tool parameters "
            "(Complex difference layer dropdown)."
        )
    intensity_layer = _resolve_layer(viewer, cd_name)
    if to_numpy(intensity_layer.data).ndim != 3:
        raise ValueError("Complex difference layer must be 3D.")

    cl_data = to_numpy(centerline_layer.data)
    if cl_data.ndim != 3:
        raise ValueError(
            "Select the centerline mask as the active layer in the layer list "
            "(3D multilabel skeleton mask), then run the tool."
        )
    cl_layer = centerline_layer
    _, cl_img, cl_rs = align_mask_to_reference_layer(
        cl_layer, intensity_layer, to_numpy(cl_layer.data), order=0
    )
    if cl_rs:
        gui_log(
            f"Resampled centerline '{cl_layer.name}' onto '{intensity_layer.name}'."
        )
    centerline_mask = to_numpy(cl_img.data)

    seg_arr = None
    seg_name = _layer_param(params, "segmentation_layer")
    if seg_name:
        seg_layer = _resolve_layer(viewer, seg_name)
        _, seg_img, seg_rs = align_mask_to_reference_layer(
            seg_layer, intensity_layer, to_numpy(seg_layer.data), order=0
        )
        if seg_rs:
            gui_log(
                f"Resampled segmentation '{seg_layer.name}' onto '{intensity_layer.name}'."
            )
        seg_arr = to_numpy(seg_img.data).astype(np.int32, copy=False)

    if not bool(params.get("measure_resegment", False)) and seg_arr is None:
        raise ValueError(
            "Disable re-segmentation only when a segmentation layer is provided."
        )

    ap_name = _layer_param(params, "ap_layer")
    rl_name = _layer_param(params, "rl_layer")
    fh_name = _layer_param(params, "fh_layer")
    vx = vy = vz = None
    if ap_name and rl_name and fh_name:
        from nvitk.measure.hemodynamics import velocity_mm_s_from_phases

        ap, rl, fh, _ = _phase_arrays_from_layers_or_disk(viewer, params)
        vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)
    elif ap_name or rl_name or fh_name:
        raise ValueError("Select AP, RL, and FH phase layers together for flow waveforms.")

    # Prefer stage-4 named branches (same geometry as stage 6) when the
    # centerline mask lives next to centerlines_seg_branches.json.
    # Venous polylines come from stage-3 (not present in seg_4dflow).
    arterial_br = None
    venous_cls = None
    venous_labels = None
    cl_src = _layer_source_path(cl_layer)
    stage_dirs: list[Path] = []
    if cl_src is not None:
        stage_dirs.append(cl_src.parent)
    # Also try seg layer path / common QC layout.
    if seg_name:
        try:
            seg_src = _layer_source_path(_resolve_layer(viewer, seg_name))
            if seg_src is not None and seg_src.parent not in stage_dirs:
                stage_dirs.append(seg_src.parent)
        except Exception:
            pass
    for stage_dir in stage_dirs:
        if (stage_dir / CENTERLINE_SEG_BRANCHES_JSON).is_file():
            try:
                arterial_br = load_arterial_branches(
                    stage_dir, min_points=3, from_segmentation=True
                )
                gui_log(
                    f"Vessel XS: using stage-4/6 named branches from {stage_dir} "
                    f"({sum(len(v) for v in arterial_br.values())} polylines)."
                )
                break
            except Exception as exc:
                gui_log(f"Vessel XS: could not load branches from {stage_dir}: {exc}")
    for stage_dir in stage_dirs:
        candidates = [
            stage_dir,
            stage_dir.parent / qvt_cfg.STAGE3_CENTERLINE_DIR,
        ]
        for s3 in candidates:
            try:
                if not (s3 / "centerline_meta.json").is_file():
                    continue
                meta3 = load_centerline_meta(s3)
                venous_cls = load_venous_centerlines(s3, min_points=3, meta=meta3)
                venous_labels = {
                    str(k): int(v)
                    for k, v in (meta3.get("venous_label_by_name") or {}).items()
                }
                if venous_cls:
                    gui_log(
                        f"Vessel XS: including {len(venous_cls)} venous "
                        f"centerlines from {s3}."
                    )
                break
            except Exception as exc:
                gui_log(f"Vessel XS: could not load venous from {s3}: {exc}")
        if venous_cls:
            break

    app_state = getattr(viewer, "_nvitk_app_state", None)
    if not isinstance(app_state, dict):
        app_state = {}

    install_vessel_cross_sections(
        viewer,
        app_state,
        intensity_layer=intensity_layer,
        centerline_mask=centerline_mask,
        segmentation=seg_arr,
        params=params,
        vx=vx,
        vy=vy,
        vz=vz,
        arterial_branches=arterial_br,
        venous_centerlines=venous_cls,
        venous_label_by_name=venous_labels,
    )
    if arterial_br:
        branch_note = " (stage-4/6 named branches"
        if venous_cls:
            branch_note += f" + {len(venous_cls)} venous"
        branch_note += ")"
    else:
        branch_note = " (re-extracted from mask"
        if venous_cls:
            branch_note += f" + {len(venous_cls)} venous"
        branch_note += ")"
    notify(
        f"Vessel cross-sections active (centerline: {cl_layer.name}, CD: {intensity_layer.name})"
        f"{branch_note}.\n"
        "Uncheck 'Pick cross-section on click' in the dock (or hide 'Vessel centerlines (xs)') "
        "to freely rotate/pan the 3D view. Left-click near a centerline to inspect. "
        "Normal arrow follows centerline order."
    )


def _run_viz_flowshow_napari(
    viewer: Any,
    mask_layer: Any,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    """Add an animated, dims-synced flow-vector overlay for the mask/label ids over all cardiac phases,
    computed from the AP/RL/FH phase layers."""
    from nvitk.gui.viz.layers import add_animated_flow_vectors_layer, flow_vectors_all_times

    ap_name = str(params.get("ap_layer") or "").strip()
    rl_name = str(params.get("rl_layer") or "").strip()
    fh_name = str(params.get("fh_layer") or "").strip()
    if not ap_name or not rl_name or not fh_name:
        raise ValueError("Select AP, RL, and FH phase layers.")

    ap_layer = _resolve_layer(viewer, ap_name)
    rl_layer = _resolve_layer(viewer, rl_name)
    fh_layer = _resolve_layer(viewer, fh_name)
    ref_name = str(params.get("reference_layer") or "").strip()
    if ref_name:
        ref_layer = _resolve_layer(viewer, ref_name)
        # Align 3D mask to spatial grid of reference (supports 4D phase volumes).
        _, mask_on_ref, _ = align_mask_to_reference_layer(
            mask_layer, ref_layer, to_numpy(mask_layer.data), order=0
        )
        mask_arr = to_numpy(mask_on_ref.data)
        spatial_ref = ref_layer if int(getattr(ref_layer.data, "ndim", 0)) <= 3 else ap_layer
    else:
        ref_layer = ap_layer
        _, mask_on_ref, _ = align_mask_to_reference_layer(
            mask_layer, ap_layer, to_numpy(mask_layer.data), order=0
        )
        mask_arr = to_numpy(mask_on_ref.data)
        spatial_ref = ap_layer

    max_arrow_voxels = float(params.get("length_scale") or 5.0)
    cache = flow_vectors_all_times(
        ap_layer.data,
        rl_layer.data,
        fh_layer.data,
        mask_arr,
        phase_layer=ap_layer,
        label_ids=label_ids,
        max_points=int(params.get("max_points") or 4000),
        max_arrow_voxels=max_arrow_voxels,
    )
    if cache.positions.shape[0] == 0:
        notify("No voxels in mask for flow vectors.", error=True)
        return

    t0 = int(params.get("time_index") or 0)
    sync_dims = bool(params.get("sync_dims", True))
    colormap = str(params.get("cmap") or params.get("colormap") or "turbo")

    add_animated_flow_vectors_layer(
        viewer,
        cache,
        phase_layer=ap_layer,
        spatial_reference_layer=spatial_ref,
        initial_time=t0,
        sync_dims=sync_dims,
        colormap=colormap,
    )
    lo, hi = float(np.min(cache.magnitudes)), float(np.max(cache.magnitudes))
    notify(
        f"Added {cache.positions.shape[0]} flow vector(s) × {cache.n_time} phases "
        f"(speed {lo:.2f}–{hi:.2f} mm/s; arrow length capped). "
        "Scrub or play cardiac phase with Napari's dims slider."
    )


def _run_viz_flow_streamlines_napari(
    viewer: Any,
    mask_layer: Any,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    """Add an animated, dims-synced streamline/pathline overlay for the mask/label ids, precomputing
    every cardiac phase up front so scrubbing is instant."""
    from nvitk.gui.viz.flow_streamlines import (
        add_animated_flow_streamlines_layer,
        build_flow_streamline_cache,
        precompute_flow_streamline_frames,
    )

    mask_data = to_numpy(mask_layer.data)
    if mask_data.ndim != 3:
        raise ValueError(
            "Active layer must be a 3D vessel segmentation mask. "
            "Select the segmentation labels layer, then run the tool."
        )
    if not bool((mask_data > 0).any()):
        raise ValueError("Active segmentation mask has no foreground labels.")

    ap_name = str(params.get("ap_layer") or "").strip()
    rl_name = str(params.get("rl_layer") or "").strip()
    fh_name = str(params.get("fh_layer") or "").strip()
    if not ap_name or not rl_name or not fh_name:
        raise ValueError("Select AP, RL, and FH phase layers.")

    ap_layer = _resolve_layer(viewer, ap_name)
    rl_layer = _resolve_layer(viewer, rl_name)
    fh_layer = _resolve_layer(viewer, fh_name)
    ref_name = str(params.get("reference_layer") or "").strip()
    if ref_name:
        ref_layer = _resolve_layer(viewer, ref_name)
        _, mask_on_ref, _ = align_mask_to_reference_layer(
            mask_layer, ref_layer, to_numpy(mask_layer.data), order=0
        )
        mask_arr = to_numpy(mask_on_ref.data)
        spatial_ref = ref_layer if int(getattr(ref_layer.data, "ndim", 0)) <= 3 else ap_layer
    else:
        ref_layer = ap_layer
        _, mask_on_ref, _ = align_mask_to_reference_layer(
            mask_layer, ap_layer, to_numpy(mask_layer.data), order=0
        )
        mask_arr = to_numpy(mask_on_ref.data)
        spatial_ref = ap_layer

    if not bool((mask_arr > 0).any()):
        notify("No voxels in mask for flow streamlines.", error=True)
        return

    notify(
        f"Precomputing streamlines for {int(np.unique(mask_arr[mask_arr > 0]).size)} "
        f"label(s) across cardiac phases…"
    )
    cache = build_flow_streamline_cache(
        ap_layer.data,
        rl_layer.data,
        fh_layer.data,
        mask_arr,
        phase_layer=ap_layer,
        label_ids=label_ids,
        trace_mode=str(params.get("trace_mode") or "streamlines"),
        n_seeds=int(params.get("n_seeds") or 64),
        max_length=float(params.get("max_length") or 35.0),
        stream_seed=int(params.get("stream_seed") or 42),
        integration_direction=str(params.get("integration_direction") or "forward"),
        seed_mode=str(params.get("seed_mode") or "planar"),
        seed_plane_axis=int(params.get("seed_plane_axis") or 2),
        seed_plane_side=str(params.get("seed_plane_side") or "min"),
        dt_seconds=float(params.get("dt_seconds") or 1.0),
        resample_paths=bool(params.get("resample_paths", False)),
        resample_spacing_vox=float(params.get("resample_spacing_vox") or 0.5),
        edge_width=float(params.get("edge_width") or 0.25),
        opacity=float(params.get("opacity") or 0.55),
        colormap=str(params.get("cmap") or params.get("colormap") or "turbo"),
        color_metric=str(params.get("color_metric") or "speed"),
        per_vertex_color=bool(params.get("per_vertex_color", True)),
    )
    precompute_flow_streamline_frames(cache)

    t0 = int(params.get("time_index") or 0)
    sync_dims = bool(params.get("sync_dims", True))
    colormap = str(params.get("cmap") or params.get("colormap") or "turbo")

    playback = add_animated_flow_streamlines_layer(
        viewer,
        cache,
        phase_layer=ap_layer,
        spatial_reference_layer=spatial_ref,
        initial_time=t0,
        sync_dims=sync_dims,
    )
    layer_name = str(getattr(playback.layer, "name", "Flow streamlines"))
    notify(
        f"Added layer {layer_name!r}: {cache.params.trace_mode} "
        f"({cache.params.n_seeds} seeds × {cache.n_time} phases). "
        f"Color: {cache.color_metric} / {colormap!r}"
        + (
            f" ({cache.speed_lo:.2f}–{cache.speed_hi:.2f} mm/s for speed)"
            if cache.color_metric == "speed"
            else ""
        )
        + f". Active mask: {mask_layer.name}. Scrub cardiac phase with Napari's dims slider."
    )


def _layer_kwargs_from(layer: Any, name: str) -> dict[str, Any]:
    """Build ``add_*`` kwargs (name suffixed with *name*, plus spatial metadata) derived from *layer*."""
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    return {"name": f"{layer.name}_{name}", **layer_spatial_kwargs(layer)}


def _run_measure_per_label(
    tool_id: str,
    layer: Any,
    viewer: Any,
    data: np.ndarray,
    label_ids: list[int],
    *,
    params: dict[str, Any],
) -> None:
    """Run a measure tool once per label id and log each result."""
    if not gpu_enabled():
        data = as_backend_array(to_numpy(data))
    lines = []
    for lid in label_ids:
        proc, _ = prepare_layer_data(data, target_mode="label", label_ids=[lid])
        img = layer_to_image(layer, proc)
        line = _measure_line(tool_id, img, layer, viewer, params)
        lines.append(f"Label {lid}: {line}")
    notify("\n".join(lines))


def _measure_line(
    tool_id: str,
    img: Image,
    layer: Any,
    viewer: Any,
    params: dict[str, Any],
) -> str:
    """Run the measurement tool named *tool_id* on *img* and format its result as a single summary
    line for notification (volume, masked stats, similarity metrics, etc.)."""
    if tool_id == "volume_mm3":
        from nvitk.measure.volume import volume_mm3

        return f"volume = {volume_mm3(img):.3f} mm³"
    if tool_id == "volume_cc":
        from nvitk.measure.volume import volume_cc

        v = volume_cc(img)
        return f"volume = {v:.6f} cc ({v * 1000:.3f} mm³)"
    if tool_id == "masked_stats":
        from nvitk.measure.intensity import masked_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        intensity, mask_img = _reference_and_mask_images(
            viewer, layer, ref_name, img.data
        )
        stats = masked_stats(intensity, mask_img)
        return _format_metrics({k: round(float(v), 6) for k, v in stats.items()})
    if tool_id == "integrated_intensity":
        from nvitk.measure.radiomics import integrated_intensity

        ref_name = str(params.get("reference_layer") or "").strip()
        intensity, mask_img = _reference_and_mask_images(
            viewer, layer, ref_name, img.data
        )
        return f"integrated = {integrated_intensity(intensity, mask_img):.6g}"
    if tool_id == "suv_stats":
        from nvitk.measure.suv import suv_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        kw = _suv_stats_kwargs(params)
        if ref_name:
            pet, mask_img = _reference_and_mask_images(viewer, layer, ref_name, img.data)
            stats = suv_stats(pet, mask_img, **kw)
        else:
            stats = suv_stats(img, img, **kw)
        return _format_metrics({k: round(float(v), 6) for k, v in stats.items()})
    if tool_id == "dice":
        from nvitk.measure.voxel import dice

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, img.data)
        return f"dice = {dice(ref_img, mask_img):.6f}"
    if tool_id == "jaccard":
        from nvitk.measure.voxel import jaccard

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, img.data)
        return f"jaccard = {jaccard(ref_img, mask_img):.6f}"
    if tool_id == "voxel_metrics":
        from nvitk.measure.voxel import voxel_metrics

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, img.data)
        return _format_metrics(
            {k: round(float(v), 6) for k, v in voxel_metrics(ref_img, mask_img).items()}
        )
    if tool_id == "surface_metrics":
        from nvitk.measure.surface import surface_metrics

        ref_name = str(params.get("reference_layer") or "").strip()
        ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, img.data)
        return _format_metrics(
            {k: round(float(v), 6) for k, v in surface_metrics(ref_img, mask_img).items()}
        )
    raise ValueError(f"Per-label measure not supported for {tool_id}")


def parse_label_ids(text: str) -> list[int]:
    """Parse a comma/semicolon/whitespace-separated label-id string into a list of ints."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[,;\s]+", str(text).strip())
    return [int(p) for p in parts if p]


def log_tool_failure(exc: BaseException) -> None:
    """Log exception message and full traceback to the GUI log panel."""
    import traceback

    gui_log(f"Tool failed: {exc}", error=True)
    gui_log(traceback.format_exc(), error=True)


def notify(message: str, *, error: bool = False) -> None:
    """Log *message* to the GUI log panel and show it as a Napari notification (error or info),
    silently ignoring failures if Napari's notification API is unavailable."""
    gui_log(message, error=error)
    try:
        if error:
            from napari.utils.notifications import show_error

            show_error(message)
        else:
            from napari.utils.notifications import show_info

            show_info(message.splitlines()[0] if "\n" in message else message)
    except Exception:
        pass
