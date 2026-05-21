"""Run GUI catalog tools on Napari image layers."""

from __future__ import annotations

import csv
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy as array_to_numpy
from nvitk.core.backend import get_global_backend, using, setup
from nvitk.gui.gui_backend import gpu_enabled, layer_data_for_tool, napari_array, run_with_backend
from nvitk.gui.log_panel import gui_log, run_subprocess_logged
from nvitk.gui.spatial import nvitk_metadata_from_layer
from nvitk.gui.spatial import layer_to_image
from nvitk.gui.tools_registry import tool_by_id
from nvitk.types import Image

setup(globals())

_MEASURE_NOTIFY = frozenset({
    "volume_mm3",
    "volume_cc",
    "masked_stats",
    "integrated_intensity",
    "suv_stats",
    "dice",
    "jaccard",
    "voxel_metrics",
    "surface_metrics",
})


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
    return array_to_numpy(label_ids)


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
        kw = {"connectivity": kw["connectivity"]}
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

    if tool_id == "viz_flowshow":
        _run_flowshow_tool(params)
        return None

    if tool_id == "qvtpy_locs":
        _run_qvtpy_locs(viewer, layer, params)
        return None

    if tool_id == "measure_loc_hemodynamics":
        _run_measure_loc_hemodynamics(viewer, layer, params)
        return None

    bk = get_global_backend()
    data = layer_data_for_tool(layer.data)
    if not gpu_enabled():
        data = as_backend_array(array_to_numpy(data))

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
        proc_data = as_backend_array(array_to_numpy(proc_data))
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
        with run_with_backend():
            if arr.ndim == 2:
                out = binary_mask_sliding_threshold_2d(
                    arr, step=step, up_thresh=up, smf=smf
                )
            elif arr.ndim == 3:
                out, _ = binary_mask_sliding_threshold_3d(
                    arr, step=step, up_thresh=up, smf=smf
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

        ref = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        intensity = layer_to_image(ref)
        stats = masked_stats(intensity, img)
        notify(_format_metrics({k: round(float(v), 6) for k, v in stats.items()}))
        return None

    if tool_id == "integrated_intensity":
        from nvitk.measure.radiomics import integrated_intensity

        ref = _resolve_layer(viewer, str(params.get("reference_layer") or ""))
        intensity = layer_to_image(ref)
        val = integrated_intensity(intensity, img)
        notify(f"Integrated intensity: {val:.6g}")
        return None

    if tool_id == "suv_stats":
        from nvitk.measure.suv import suv_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name:
            pet = layer_to_image(_resolve_layer(viewer, ref_name))
            mask = img
        else:
            pet = img
            mask = img
        stats = suv_stats(pet, mask)
        notify(_format_metrics({k: round(float(v), 6) for k, v in stats.items()}))
        return None

    if tool_id == "dice":
        from nvitk.measure.voxel import dice

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        val = dice(ref, img)
        notify(f"Dice: {val:.6f}")
        return None

    if tool_id == "jaccard":
        from nvitk.measure.voxel import jaccard

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        val = jaccard(ref, img)
        notify(f"Jaccard: {val:.6f}")
        return None

    if tool_id == "voxel_metrics":
        from nvitk.measure.voxel import voxel_metrics

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        metrics = voxel_metrics(ref, img)
        notify(_format_metrics({k: round(float(v), 6) for k, v in metrics.items()}))
        return None

    if tool_id == "surface_metrics":
        from nvitk.measure.surface import surface_metrics

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        metrics = surface_metrics(ref, img)
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
        _run_viz_pet_hotspots(viewer, layer, proc_data, params, label_ids=label_ids)
        return None

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

    if tool_id == "seg_biggest_cc":
        from nvitk.segmentation.labels import biggest_cc

        return coerce_tool_output(biggest_cc(img))

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

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        mask = as_backend_array(proc_data).astype(bool).copy()
        intensity = as_backend_array(ref.data).astype(np.float64)
        if mask.shape != intensity.shape:
            raise ValueError("Mask and intensity layers must have the same shape.")
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
        bar_rad = int(params.get("barrier_radius_vox") or 0)
        forbidden = None
        bar_name = str(params.get("barrier_layer") or "").strip()
        if bar_name:
            bar_layer = _resolve_layer(viewer, bar_name)
            bar_vol = as_backend_array(layer_data_for_tool(bar_layer.data))
            if bar_vol.shape != mask.shape:
                raise ValueError("Barrier layer must match the active mask shape.")
            forbidden = forbidden_from_label_mask(bar_vol, radius_vox=bar_rad)
        if bool(params.get("barrier_other_labels")):
            label_vol = as_backend_array(data)
            if label_vol.shape != mask.shape:
                raise ValueError("Active layer shape mismatch for label barriers.")
            exclude = [int(x) for x in seed_ids]
            if not exclude:
                growing = np.asarray(mask, dtype=bool)
                for lid in np.unique(to_numpy(label_vol)):
                    lid = int(lid)
                    if lid != 0 and np.any(label_vol[growing] == lid):
                        exclude.append(lid)
            if not exclude:
                raise ValueError(
                    "Barrier other labels: select label id(s) or seed a labeled region."
                )
            other_forb = forbidden_other_labels(label_vol, exclude, radius_vox=bar_rad)
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

    raise NotImplementedError(f"Tool '{tool_id}' is not implemented in the GUI runner.")


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


def _run_flowshow_tool(params: dict[str, Any]) -> None:
    argv = ["nvitk-qvtpy-flowshow"]
    subject = str(params.get("subject") or "").strip()
    if subject:
        argv.extend(["--subject", subject])
    nifti_root = str(params.get("nifti_root") or "").strip()
    if nifti_root:
        argv.extend(["--nifti-root", nifti_root])
    batch = str(params.get("batch") or "").strip()
    if batch:
        argv.extend(["--batch", batch])
    out_root = str(params.get("pipeline_output_root") or "").strip()
    if out_root:
        argv.extend(["--pipeline-output-root", out_root])
    vm = str(params.get("vessel_mask") or "").strip()
    if vm:
        argv.extend(["--vessel-mask", vm])
    if bool(params.get("notebook")):
        argv.append("--notebook")
    notify(f"Launching FlowShow: {' '.join(argv)}")
    code = run_subprocess_logged(argv)
    if code == 0:
        notify("FlowShow finished.")
    else:
        notify(f"FlowShow exited with code {code}", error=True)


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
            "qvtpy-stage5-loc",
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
    ap, rl, fh, voxel_spacing = _phase_arrays_from_layers_or_disk(viewer, params)
    ref_name = str(params.get("reference_layer") or "").strip()
    mag = cd = vel_mag = None
    if ref_name:
        ref_data = as_backend_array(_resolve_layer(viewer, ref_name).data).astype(np.float64)
        mag = cd = vel_mag = ref_data
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


def _run_viz_pet_hotspots(
    viewer: Any,
    layer: Any,
    mask_data: np.ndarray,
    params: dict[str, Any],
    *,
    label_ids: list[int] | None,
) -> None:
    ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
    suv = as_backend_array(ref.data)
    mask = mask_data
    lids = label_ids

    def _show() -> None:
        from nvitk.viz.pet_hotspots import show_hotspots

        show_hotspots(
            suv,
            mask,
            label_ids=lids,
            hotspot=str(params.get("hotspot") or "top_percent"),
            top_percent=float(params.get("top_percent") or 0.1),
            max_points=int(params.get("max_points") or 20000),
            cmap=str(params.get("cmap") or "turbo"),
            show=True,
        )

    threading.Thread(target=_show, daemon=True).start()
    notify("Opened PyVista hotspots window (background thread).")


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
        data = as_backend_array(array_to_numpy(data))
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

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        stats = masked_stats(ref, img)
        return _format_metrics({k: round(float(v), 6) for k, v in stats.items()})
    if tool_id == "integrated_intensity":
        from nvitk.measure.radiomics import integrated_intensity

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        return f"integrated = {integrated_intensity(ref, img):.6g}"
    if tool_id == "suv_stats":
        from nvitk.measure.suv import suv_stats

        ref_name = str(params.get("reference_layer") or "").strip()
        if ref_name:
            pet = layer_to_image(_resolve_layer(viewer, ref_name))
            stats = suv_stats(pet, img)
        else:
            stats = suv_stats(img, img)
        return _format_metrics({k: round(float(v), 6) for k, v in stats.items()})
    if tool_id == "dice":
        from nvitk.measure.voxel import dice

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        return f"dice = {dice(ref, img):.6f}"
    if tool_id == "jaccard":
        from nvitk.measure.voxel import jaccard

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        return f"jaccard = {jaccard(ref, img):.6f}"
    if tool_id == "voxel_metrics":
        from nvitk.measure.voxel import voxel_metrics

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        return _format_metrics({k: round(float(v), 6) for k, v in voxel_metrics(ref, img).items()})
    if tool_id == "surface_metrics":
        from nvitk.measure.surface import surface_metrics

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        return _format_metrics(
            {k: round(float(v), 6) for k, v in surface_metrics(ref, img).items()}
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
