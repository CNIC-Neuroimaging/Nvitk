"""Run GUI catalog tools on Napari image layers."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
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
        from nvitk.segmentation.region_growing import region_grow_binary_mask

        ref = layer_to_image(_resolve_layer(viewer, str(params.get("reference_layer") or "")))
        mask = as_backend_array(proc_data).astype(bool).copy()
        intensity = as_backend_array(ref.data).astype(np.float64)
        if mask.shape != intensity.shape:
            raise ValueError("Mask and intensity layers must have the same shape.")
        sz = int(params.get("seed_z") or 0)
        sy = int(params.get("seed_y") or 0)
        sx = int(params.get("seed_x") or 0)
        mask[sz, sy, sx] = True
        thresh = float(params.get("threshold") or 0.0)
        region_grow_binary_mask(
            mask,
            intensity,
            intensity_frac=thresh,
            abs_floor=None,
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
