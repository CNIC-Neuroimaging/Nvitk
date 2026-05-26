"""Magicgui Tools tab with category/operation pickers and per-tool parameters."""

from __future__ import annotations

from typing import Any, Callable

from magicgui import magicgui

from nvitk.gui.tool_presets import apply_preset_to_panel, preset_key_from_title
from nvitk.gui.tool_runner import log_tool_failure, notify, parse_label_ids, run_gui_tool
from nvitk.gui.tools_registry import (
    categories,
    default_category,
    default_operation,
    operation_help_text,
    operations_for_category,
    params_for_tool,
    tool_by_id,
    tool_id_from_label,
)
from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS


def _collect_params(widget: Any, tool_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pspec in params_for_tool(tool_id):
        w = getattr(widget, pspec.name, None)
        if w is None:
            continue
        val = w.value
        if pspec.kind == "layer":
            out[pspec.name] = str(val or "")
        else:
            out[pspec.name] = val
    return out


def _set_param_visibility(widget: Any, tool_id: str) -> None:
    visible = {p.name for p in params_for_tool(tool_id)}
    all_names = (
        "footprint",
        "iterations",
        "mode",
        "connectivity",
        "sigma_spatial",
        "sigma_color",
        "do_3d",
        "axis",
        "step",
        "up_thresh",
        "smf",
        "shift_hm",
        "min_size",
        "reference_layer",
        "barrier_layer",
        "centerline_barrier_layer",
        "barrier_other_labels",
        "mask_barrier_dilation_vox",
        "centerline_barrier_dilation_vox",
        "barrier_radius_vox",
        "factor",
        "order",
        "radius_vox",
        "res",
        "label_id",
        "label_ids",
        "new_id",
        "output_dir",
        "working_dir",
        "task",
        "correction_ids",
        "plane_x",
        "seed_z",
        "seed_y",
        "seed_x",
        "threshold",
        "pipeline_preset",
        "seed_from_label",
        "loc_mode",
        "locs_csv",
        "subject",
        "nifti_root",
        "output_root",
        "batch",
        "pipeline_output_root",
        "vessel_mask",
        "notebook",
        "hotspot",
        "top_percent",
        "max_points",
        "point_size",
        "cmap",
        "ap_layer",
        "rl_layer",
        "fh_layer",
        "loc_arterial_strategy",
        "cross_section_radius_vox",
        "measure_resegment",
        "hemo_method",
        "min_turn_angle_deg",
        "min_separation_points",
        "smooth_window",
        "use_flexion_layer",
        "new_label_start",
        "polyline_mode",
        "min_branch_points",
        "min_junction_degree",
        "branch_id",
        "reskeletonize",
        "suv_kind",
        "philips_factor",
        "revert_scaling",
        "time_index",
        "organ_layer",
        "body_layer",
        "kidney_r_id",
        "kidney_l_id",
        "bladder_id",
        "radius_mm",
        "w_pet",
        "hull_axis",
        "edt_use_spacing",
        "dicom_root",
        "skip_existing",
        "compute_phase_derived",
        "phase_background_correction",
        "no_cd_4d_background_correction",
        "eicab_mask",
        "length_scale",
        "sync_dims",
    )
    for name in all_names:
        sub = getattr(widget, name, None)
        if sub is not None:
            sub.visible = name in visible

    if tool_id.startswith("qvtpy_stage"):
        for name in all_names:
            sub = getattr(widget, name, None)
            if sub is not None:
                sub.visible = False
        for name in ("subject", "skip_existing"):
            sub = getattr(widget, name, None)
            if sub is not None:
                sub.visible = True
        if tool_id == "qvtpy_stage0_download":
            for name in ("dicom_root",):
                sub = getattr(widget, name, None)
                if sub is not None:
                    sub.visible = True
        else:
            for name in ("nifti_root", "output_root", "dicom_root"):
                sub = getattr(widget, name, None)
                if sub is not None:
                    sub.visible = True
        if tool_id == "qvtpy_stage0_convert":
            for name in (
                "compute_phase_derived",
                "phase_background_correction",
                "no_cd_4d_background_correction",
            ):
                sub = getattr(widget, name, None)
                if sub is not None:
                    sub.visible = True
        if tool_id == "qvtpy_stage3_centerline":
            sub = getattr(widget, "eicab_mask", None)
            if sub is not None:
                sub.visible = True


_LAYER_NONE = "(none)"


def _update_reference_layers(widget: Any, viewer: Any) -> None:
    names = [lyr.name for lyr in viewer.layers]
    optional_choices = [_LAYER_NONE, *names]
    for attr in (
        "reference_layer",
        "barrier_layer",
        "centerline_barrier_layer",
        "ap_layer",
        "rl_layer",
        "fh_layer",
        "organ_layer",
        "body_layer",
    ):
        ref = getattr(widget, attr, None)
        if ref is None:
            continue
        if attr in ("barrier_layer", "centerline_barrier_layer"):
            ref.choices = optional_choices
            if ref.value not in optional_choices:
                ref.value = _LAYER_NONE
        else:
            ref.choices = names
            if names and ref.value not in names:
                ref.value = names[0] if attr == "reference_layer" else ""


def _update_phase_layers(widget: Any, viewer: Any) -> None:
    _update_reference_layers(widget, viewer)


def build_tool_panel(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    layer_display_kwargs: Callable[..., dict[str, Any]],
    on_layers_changed: Callable[[], None],
    record_step: Callable[[dict[str, Any]], None] | None = None,
    get_label_ids: Callable[[], list[int]] | None = None,
    get_pipeline_argv_builder: Callable[[], Any] | None = None,
    get_totalseg_roi: Callable[[], list[str] | None] | None = None,
    label_selector: Any | None = None,
) -> Any:
    """Return magicgui FunctionGui for the Tools tab."""
    _ = label_selector

    @magicgui(
        category={"choices": categories(), "label": "Category", "value": default_category()},
        operation={
            "choices": operations_for_category(default_category()),
            "label": "Operation",
        },
        operation_help={
            "label": "",
            "widget_type": "TextEdit",
            "value": operation_help_text(
                tool_id_from_label(default_category(), default_operation(default_category()))
            ),
            "enabled": False,
        },
        target_mode={
            "choices": ["raw", "binary_mask", "label", "all_labels"],
            "label": "Process target",
            "value": "raw",
        },
        label_ids={"label": "Label id(s) (comma-separated)", "value": ""},
        overlay_mode={"choices": ["add_layer", "replace_active"], "label": "Output mode"},
        footprint={"label": "Footprint (radius)", "min": 1, "max": 32, "value": 1},
        iterations={"label": "Iterations", "min": 1, "max": 20, "value": 1},
        mode={"choices": ["binary", "gray"], "label": "Morph mode", "value": "binary"},
        connectivity={"label": "Connectivity", "min": 1, "max": 3, "value": 2},
        sigma_spatial={"label": "Sigma spatial (0=auto)", "value": 0.0},
        sigma_color={"label": "Sigma color (0=auto)", "value": 0.0},
        do_3d={"label": "Bilateral 3D", "value": False},
        axis={"label": "Axis / slice (-1=auto isotropy)", "min": -1, "max": 2, "value": 0},
        step={
            "label": "Sliding step",
            "value": 0.001,
            "min": 0.0001,
            "max": 0.1,
            "step": 0.0001,
        },
        up_thresh={"label": "Sliding up_thresh", "value": 0.8, "min": 0.0, "max": 1.0},
        smf={"label": "Sliding smooth win", "min": 1, "max": 200, "value": 10},
        shift_hm={"label": "Shift HM (half-max curvature)", "value": True},
        min_size={"label": "Min component size", "min": 1, "value": 64},
        reference_layer={
            "label": "Reference layer",
            "widget_type": "ComboBox",
            "choices": [""],
            "value": "",
        },
        barrier_layer={
            "label": "Barrier mask layer",
            "widget_type": "ComboBox",
            "choices": [_LAYER_NONE],
            "value": _LAYER_NONE,
        },
        centerline_barrier_layer={
            "label": "Barrier centerline layer",
            "widget_type": "ComboBox",
            "choices": [_LAYER_NONE],
            "value": _LAYER_NONE,
        },
        barrier_other_labels={"label": "Barrier: other labels on active mask", "value": False},
        mask_barrier_dilation_vox={
            "label": "Mask barrier dilation (vox)",
            "min": 0,
            "max": 32,
            "value": 1,
        },
        centerline_barrier_dilation_vox={
            "label": "Centerline barrier dilation (vox)",
            "min": 0,
            "max": 32,
            "value": 3,
        },
        barrier_radius_vox={"label": "Barrier dilation (vox)", "min": 0, "max": 32, "value": 3},
        factor={"label": "Isotropy factor (0=auto)", "value": 0.0, "min": 0.0},
        order={"label": "Interpolation order", "min": 0, "max": 5, "value": 1},
        radius_vox={"label": "Oblique half-size", "value": 40.0, "min": 1.0},
        res={"label": "Oblique resolution", "min": 16, "max": 1024, "value": 256},
        label_id={"label": "Label id", "min": 0, "max": 9999, "value": 1},
        new_id={"label": "Output label id", "min": 0, "max": 9999, "value": 1},
        output_dir={"label": "Output directory", "value": ""},
        working_dir={"label": "Working directory", "value": "."},
        task={
            "choices": list(AVAILABLE_TASKS),
            "label": "TotalSegmentator task",
            "value": "total",
        },
        correction_ids={"label": "ICA label ids (e.g. 1,2)", "value": "1,2"},
        plane_x={"label": "Midline X (voxel, 0=auto)", "min": 0, "value": 0},
        seed_z={"label": "Seed Z", "min": 0, "value": 0},
        seed_y={"label": "Seed Y", "min": 0, "value": 0},
        seed_x={"label": "Seed X", "min": 0, "value": 0},
        threshold={"label": "Intensity fraction", "value": 0.0, "min": 0.0, "max": 1.0},
        pipeline_preset={
            "choices": [
                "Custom",
                "QVTpy default (frac=0.45)",
                "QVTpy explore (frac=0.25)",
                "QVTpy ICA test (frac=0.45)",
            ],
            "label": "Pipeline preset",
            "value": "Custom",
        },
        seed_from_label={"label": "Seed from label centroid", "value": False},
        loc_mode={"choices": ["load_csv", "generate"], "label": "LOC mode", "value": "load_csv"},
        locs_csv={"label": "LOCs CSV path", "value": ""},
        subject={"label": "Subject id", "value": ""},
        nifti_root={"label": "NIfTI root", "value": ""},
        output_root={"label": "Output root", "value": ""},
        batch={"label": "Batch folder", "value": ""},
        pipeline_output_root={"label": "Pipeline output root", "value": ""},
        vessel_mask={"label": "Vessel mask path", "value": ""},
        notebook={"label": "FlowShow notebook mode", "value": False},
        hotspot={
            "choices": ["top_percent", "top_k", "threshold"],
            "label": "Hotspot mode",
            "value": "top_percent",
        },
        top_percent={"label": "Top percent", "value": 0.1, "min": 0.01, "max": 100.0},
        max_points={"label": "Max hotspot points", "min": 100, "max": 500000, "value": 20000},
        point_size={"label": "Point size", "value": 6.0, "min": 0.1, "max": 100.0},
        cmap={
            "label": "SUV colormap",
            "choices": ["viridis", "turbo", "magma", "inferno", "plasma", "cividis", "hot", "coolwarm"],
            "value": "viridis",
        },
        ap_layer={"label": "AP phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        rl_layer={"label": "RL phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        fh_layer={"label": "FH phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        loc_arterial_strategy={
            "choices": ["qvtpy", "midpoint"],
            "label": "Arterial LOC strategy",
            "value": "qvtpy",
        },
        cross_section_radius_vox={"label": "Cross-section radius (vox)", "value": 10.0, "min": 1.0},
        measure_resegment={"label": "Resegment in-plane", "value": True},
        hemo_method={
            "choices": ["pseudo_loc", "voxel_avg", "both"],
            "label": "Mask hemo method",
            "value": "both",
        },
        min_turn_angle_deg={"label": "Min turn angle (deg)", "value": 45.0, "min": 5.0, "max": 180.0},
        min_separation_points={"label": "Min separation (points)", "min": 1, "max": 500, "value": 8},
        smooth_window={"label": "Tangent smooth window", "min": 1, "max": 20, "value": 3},
        use_flexion_layer={"label": "Use Flexion points layer", "value": True},
        new_label_start={"label": "First new label id (0=auto)", "min": 0, "max": 9999, "value": 0},
        polyline_mode={
            "choices": ["junction_split", "longest_path"],
            "label": "Polyline extraction mode",
            "value": "junction_split",
        },
        min_branch_points={"label": "Min points per branch", "min": 2, "max": 5000, "value": 5},
        min_junction_degree={"label": "Min skeleton degree", "min": 2, "max": 26, "value": 3},
        branch_id={"label": "Branch id (-1 = auto)", "min": -1, "max": 99, "value": -1},
        reskeletonize={"label": "Re-skeletonize centerline mask", "value": False},
        suv_kind={"choices": ["bw", "lbm", "bsa", "ibw"], "label": "SUV kind", "value": "bw"},
        philips_factor={"label": "Use Philips SUV factor", "value": True},
        revert_scaling={"label": "Revert per-slice rescale", "value": False},
        time_index={"label": "Initial cardiac phase", "min": 0, "max": 64, "value": 0},
        length_scale={
            "label": "Max arrow length (vox @ 95th %ile speed)",
            "value": 5.0,
            "min": 0.5,
            "max": 50.0,
        },
        sync_dims={"label": "Sync vectors to dims slider", "value": True},
        organ_layer={"label": "Organ labels layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        body_layer={"label": "Body mask layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        kidney_r_id={"label": "Kidney right label id", "min": 0, "max": 9999, "value": 2},
        kidney_l_id={"label": "Kidney left label id", "min": 0, "max": 9999, "value": 3},
        bladder_id={"label": "Bladder label id", "min": 0, "max": 9999, "value": 21},
        radius_mm={"label": "Radius (mm)", "value": 6.0, "min": 0.0, "max": 100.0},
        w_pet={"label": "PET cost weight", "value": 5.0, "min": 0.1, "max": 50.0},
        hull_axis={"label": "Hull slice axis", "min": 0, "max": 2, "value": 2},
        edt_use_spacing={"label": "EDT distance in mm", "value": True},
        dicom_root={"label": "DICOM root", "value": ""},
        skip_existing={"label": "Skip existing outputs", "value": False},
        compute_phase_derived={"label": "Compute phase derivatives (stage 0)", "value": True},
        phase_background_correction={
            "label": "Phase background correction (stage 0)",
            "value": True,
        },
        no_cd_4d_background_correction={
            "label": "Disable CD 4D background correction",
            "value": False,
        },
        eicab_mask={"choices": ["cw", "wb"], "label": "eICAB mask (stage 3)", "value": "cw"},
        call_button="Run tool",
    )
    def tool_panel(
        category: str,
        operation: str,
        operation_help: str,
        target_mode: str,
        label_ids: str,
        overlay_mode: str,
        footprint: int,
        iterations: int,
        mode: str,
        connectivity: int,
        sigma_spatial: float,
        sigma_color: float,
        do_3d: bool,
        axis: int,
        step: float,
        up_thresh: float,
        smf: int,
        shift_hm: bool,
        min_size: int,
        reference_layer: str,
        barrier_layer: str,
        centerline_barrier_layer: str,
        barrier_other_labels: bool,
        mask_barrier_dilation_vox: int,
        centerline_barrier_dilation_vox: int,
        barrier_radius_vox: int,
        factor: float,
        order: int,
        radius_vox: float,
        res: int,
        label_id: int,
        new_id: int,
        output_dir: str,
        working_dir: str,
        task: str,
        correction_ids: str,
        plane_x: int,
        seed_z: int,
        seed_y: int,
        seed_x: int,
        threshold: float,
        pipeline_preset: str,
        seed_from_label: bool,
        loc_mode: str,
        locs_csv: str,
        subject: str,
        nifti_root: str,
        output_root: str,
        batch: str,
        pipeline_output_root: str,
        vessel_mask: str,
        notebook: bool,
        hotspot: str,
        top_percent: float,
        max_points: int,
        point_size: float,
        cmap: str,
        ap_layer: str,
        rl_layer: str,
        fh_layer: str,
        loc_arterial_strategy: str,
        cross_section_radius_vox: float,
        measure_resegment: bool,
        hemo_method: str,
        min_turn_angle_deg: float,
        min_separation_points: int,
        smooth_window: int,
        use_flexion_layer: bool,
        new_label_start: int,
        polyline_mode: str,
        min_branch_points: int,
        min_junction_degree: int,
        branch_id: int,
        reskeletonize: bool,
        suv_kind: str,
        philips_factor: bool,
        revert_scaling: bool,
        time_index: int,
        length_scale: float,
        sync_dims: bool,
        organ_layer: str,
        body_layer: str,
        kidney_r_id: int,
        kidney_l_id: int,
        bladder_id: int,
        radius_mm: float,
        w_pet: float,
        hull_axis: int,
        edt_use_spacing: bool,
        dicom_root: str,
        skip_existing: bool,
        compute_phase_derived: bool,
        phase_background_correction: bool,
        no_cd_4d_background_correction: bool,
        eicab_mask: str,
    ) -> None:
        if not viewer.layers:
            notify("No layers loaded. Open an image first (Ctrl+O or drag-and-drop).", error=True)
            return
        tool_id = tool_id_from_label(category, operation)
        if not tool_id:
            notify("Select a valid operation.", error=True)
            return
        spec = tool_by_id(tool_id)
        run_mode = spec.run_mode if spec else "layer"
        layer = viewer.layers.selection.active or viewer.layers[-1]

        ids: list[int] = []
        if get_label_ids is not None:
            ids = list(get_label_ids())
        if not ids:
            ids = parse_label_ids(label_ids)
        if tool_id == "siphon_correct" and ids:
            correction_ids = ",".join(str(i) for i in ids)

        if run_mode == "layer" and target_mode == "label" and not ids:
            notify("Select at least one label (checkbox list or id field).", error=True)
            return

        _update_reference_layers(tool_panel, viewer)
        _update_phase_layers(tool_panel, viewer)
        if tool_id == "seg_region_grow":
            key = preset_key_from_title(tool_id, pipeline_preset)
            apply_preset_to_panel(tool_panel, tool_id, key)
        params = _collect_params(tool_panel, tool_id)
        if correction_ids and tool_id == "siphon_correct":
            params["correction_ids"] = correction_ids
        params["selected_label_ids"] = ids
        if ids and tool_id in ("seg_combine_labels", "seg_remove_labels"):
            params["label_ids"] = ",".join(str(i) for i in ids)

        if run_mode == "pipeline" and get_pipeline_argv_builder is not None:
            form = get_pipeline_argv_builder()
            exe = str(spec.cli_command if spec else "").strip()
            if exe:
                try:
                    params["pipeline_argv"] = form.build_argv(exe)
                except ValueError as exc:
                    notify(str(exc), error=True)
                    return

        if tool_id == "seg_totalsegmentator" and get_totalseg_roi is not None:
            params["roi_subset"] = get_totalseg_roi()

        try:
            result = run_gui_tool(
                tool_id,
                layer,
                viewer,
                target_mode=target_mode,
                label_ids=ids or None,
                params=params,
            )
        except NotImplementedError as exc:
            log_tool_failure(exc)
            notify(str(exc), error=True)
            return
        except Exception as exc:
            log_tool_failure(exc)
            notify(f"Tool failed: {exc}", error=True)
            return

        if result is None:
            on_layers_changed()
            return

        name = f"{layer.name}_{tool_id}"
        out_kwargs = layer_display_kwargs(layer, name=name)
        can_replace = (
            overlay_mode == "replace_active"
            and tuple(result.shape) == tuple(layer.data.shape)
        )
        if can_replace:
            try:
                layer.data = result
                layer.name = name
            except Exception:
                viewer.add_image(result, **out_kwargs)
        else:
            viewer.add_image(result, **out_kwargs)

        app_state["outputs"].append({"name": name, "shape": tuple(result.shape)})
        if record_step is not None:
            record_step(
                {
                    "type": "tool",
                    "tool": tool_id,
                    "category": category,
                    "operation": operation,
                    "source_layer": layer.name,
                    "output_layer": name,
                    "params": params,
                }
            )
        notify(f"Applied {operation} → {name}")
        on_layers_changed()

    def _signal_value(event: Any) -> Any:
        return event.value if hasattr(event, "value") else event

    def _sync_operation_help() -> None:
        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op)
        tool_panel.operation_help.value = operation_help_text(tid)

    @tool_panel.category.changed.connect
    def _on_category_changed(event) -> None:
        cat = _signal_value(event)
        ops = operations_for_category(cat)
        tool_panel.operation.choices = ops
        if ops:
            tool_panel.operation.value = ops[0]
        tid = tool_id_from_label(cat, tool_panel.operation.value)
        if tid:
            _set_param_visibility(tool_panel, tid)
        _sync_operation_help()

    @tool_panel.operation.changed.connect
    def _on_operation_changed(event) -> None:
        tid = tool_id_from_label(tool_panel.category.value, _signal_value(event))
        if tid:
            _set_param_visibility(tool_panel, tid)
        _update_reference_layers(tool_panel, viewer)
        _sync_operation_help()

    tid0 = tool_id_from_label(default_category(), default_operation(default_category()))
    if tid0:
        _set_param_visibility(tool_panel, tid0)
        _update_reference_layers(tool_panel, viewer)
        _update_phase_layers(tool_panel, viewer)
    _sync_operation_help()

    @tool_panel.pipeline_preset.changed.connect
    def _on_preset_changed(event) -> None:
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value)
        if tid != "seg_region_grow":
            return
        title = _signal_value(event)
        key = preset_key_from_title(tid, title)
        apply_preset_to_panel(tool_panel, tid, key)

    return tool_panel
