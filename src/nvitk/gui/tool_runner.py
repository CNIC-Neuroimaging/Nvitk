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
from nvitk.gui.gui_backend import gpu_enabled, layer_data_for_tool, napari_array, run_with_backend
from nvitk.gui.log_panel import gui_log, run_subprocess_logged
from nvitk.gui.spatial import (
    align_mask_to_reference_layer,
    layer_to_image,
    nvitk_metadata_from_layer,
)
from nvitk.gui.tools_registry import tool_by_id
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
    mask_data: np.ndarray | None = None,
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
    name = str(name or "").strip()
    if not name:
        raise ValueError("Select a reference layer.")
    for lyr in viewer.layers:
        if lyr.name == name:
            return lyr
    raise ValueError(f"Layer not found: {name}")


def _format_metrics(metrics: dict[str, Any]) -> str:
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


def _run_pipeline_cli(spec: Any, params: dict[str, Any]) -> None:
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
    flat = as_backend_array(data).ravel()
    if flat.size == 0:
        return []
    labels = np.unique(flat)
    return [int(x) for x in labels if int(x) != 0]


def _label_ids_array(label_ids: list[int]) -> np.ndarray:
    """Backend array of label ids for :func:`np.isin` (CuPy rejects plain lists)."""
    return to_numpy(label_ids)


def prepare_layer_data(
    data: np.ndarray,
    *,
    target_mode: str,
    label_ids: list[int] | None,
) -> tuple[np.ndarray, str]:
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


def _morph_common(img: Image, op: str, params: dict[str, Any]) -> np.ndarray:
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
    params: dict[str, Any] | None = None,
) -> np.ndarray | None:
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

    if tool_id == "viz_flowshow":
        _run_viz_flowshow_napari(viewer, layer, params, label_ids=label_ids)
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

    bk = get_global_backend()
    data = layer_data_for_tool(layer.data)
    if not gpu_enabled():
        data = as_backend_array(to_numpy(data))

    if tool_id in _MEASURE_NOTIFY:
        per_label_ids: list[int] | None = label_ids
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

        kw: dict[str, Any] = {
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

    if tool_id == "skeletonize":
        from nvitk.morphology.centerline import compute_centerlines, skeletonize_binary

        arr = as_backend_array(img.data)
        labels = _unique_labels(arr)
        with using(bk):
            if len(labels) > 1:
                paths = compute_centerlines(arr, labels=labels, min_points=5)
                out = np.zeros(arr.shape, dtype=np.uint8)
                for lid, pts in paths.items():
                    pts_i = np.round(as_backend_array(pts)).astype(int)
                    for x, y, z in pts_i:
                        if (
                            0 <= x < out.shape[0]
                            and 0 <= y < out.shape[1]
                            and 0 <= z < out.shape[2]
                        ):
                            out[x, y, z] = int(lid)
                if not out.any():
                    raise ValueError("No centerline points found.")
                return out
            sk = skeletonize_binary(arr > 0)
            return as_backend_array(sk).astype(np.uint8)

    if tool_id == "isotropy":
        from nvitk.transform.isotropy import isotropy

        axis = int(params.get("axis", -1))
        factor = float(params.get("factor") or 0)
        order = int(params.get("order") or 1)
        kw: dict[str, Any] = {"order": order}
        if axis >= 0:
            kw["axis"] = axis
        if factor > 0:
            kw["factor"] = factor
        with using(bk):
            return coerce_tool_output(isotropy(img, **kw))

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
        from nvitk.gui.spatial import layer_spacing
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

        return coerce_tool_output(biggest_cc(img))

    if tool_id == "morph_biggest_cc":
        from scipy.ndimage import generate_binary_structure

        from nvitk.segmentation.labels import biggest_cc

        conn = int(params.get("connectivity") or 1)
        rank = min(3, int(proc_data.ndim))
        structure = generate_binary_structure(rank, conn)
        return coerce_tool_output(biggest_cc(img, structure=structure))

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
            from nvitk.gui.tool_presets import label_centroid_voxel

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
    from types import SimpleNamespace

    layers_list = [primary_layer, *aux_layers]

    class _Layers:
        def __init__(self, items: list[Any]) -> None:
            self._items = items
            self.selection = SimpleNamespace(active=items[0] if items else None)

        def __iter__(self):
            return iter(self._items)

    return SimpleNamespace(layers=_Layers(layers_list))


def run_gui_tool_headless(
    tool_id: str,
    *,
    primary: Image,
    aux: dict[str, Image],
    target_mode: str,
    label_ids: list[int] | None,
    params: dict[str, Any] | None = None,
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
    from nvitk.gui.spatial import layer_spacing

    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)


def _phase_arrays_from_layers_or_disk(
    viewer: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
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
    from nvitk.segmentation import mask_ops

    def _as_bool_mask_from_labels(data: np.ndarray, ids: list[int] | None) -> np.ndarray:
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
    from nvitk.gui.centerline_flexion import (
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


def _resolve_centerline_mask_layer(viewer: Any) -> Any:
    """3D centerline mask recorded when junctions were detected."""
    from nvitk.gui.centerline_flexion import JUNCTION_META_KEY, JUNCTION_POINTS_LAYER

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
    from nvitk.gui.centerline_flexion import (
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
    from nvitk.gui.loc_points import add_locs_layer, load_locs_csv
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
    notify(f"Loaded {len(rows)} LOCs from {csv_path}")


def _run_qvtpy_stage(tool_id: str, params: dict[str, Any]) -> None:
    from nvitk.gui.qvtpy_stages import build_qvtpy_stage_argv

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
    from nvitk.gui.centerline_flexion import centerline_polyline_for_label
    from nvitk.morphology.polyline_graph import extract_polylines_from_centerline
    from nvitk.pipes.qvtpy.util.loc_selection import polyline_cumulative_arc_length

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

    lines: list[str] = []
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
    from nvitk.gui.loc_points import load_locs_csv
    from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
    from nvitk.pipes.qvtpy import config as qcfg
    from nvitk.pipes.qvtpy.util.loc_measure import run_loc_measurements

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
    measure_resegment = bool(params.get("measure_resegment", True))
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
    lines: list[str] = []
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
            method=method,  # type: ignore[arg-type]
            voxel_spacing=voxel_spacing,
            radius_vox=float(params.get("cross_section_radius_vox") or 10.0),
            measure_resegment=bool(params.get("measure_resegment", True)),
            volume_seg=mask_data if not bool(params.get("measure_resegment", True)) else None,
        )
        for res in results:
            extra = ""
            if res.mean_flow_ml_s is not None:
                extra = f" flow={res.mean_flow_ml_s:.2f} ml/s"
            lines.append(
                f"Label {lid} [{res.method}]: PI={res.pi:.3f} RI={res.ri:.3f}{extra} — {res.note}"
            )
    notify("\n".join(lines))


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
    from nvitk.gui.napari_viz import add_hotspot_points_layer, hotspot_points_from_volumes
    from nvitk.viz.pet_hotspots import HotspotMode

    ref_name = str(params.get("reference_layer") or "").strip()
    ref_layer = _resolve_layer(viewer, ref_name)
    ref_img, mask_img = _reference_and_mask_images(viewer, layer, ref_name, mask_data)
    mode = str(params.get("hotspot") or "top_percent")
    coords, _vals, features = hotspot_points_from_volumes(
        to_numpy(ref_img.data),
        to_numpy(mask_img.data),
        label_ids=label_ids,
        hotspot=mode,  # type: ignore[arg-type]
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


def _run_viz_flowshow_napari(
    viewer: Any,
    mask_layer: Any,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    from nvitk.gui.napari_viz import add_animated_flow_vectors_layer, flow_vectors_all_times

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


def _layer_kwargs_from(layer: Any, name: str) -> dict[str, Any]:
    from nvitk.gui.spatial import layer_spatial_kwargs

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
    lines: list[str] = []
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
