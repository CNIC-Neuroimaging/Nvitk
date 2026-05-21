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
        "min_size",
        "reference_layer",
        "barrier_layer",
        "barrier_other_labels",
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
        "cmap",
        "ap_layer",
        "rl_layer",
        "fh_layer",
        "loc_arterial_strategy",
        "cross_section_radius_vox",
        "measure_resegment",
        "hemo_method",
    )
    for name in all_names:
        sub = getattr(widget, name, None)
        if sub is not None:
            sub.visible = name in visible


def _update_reference_layers(widget: Any, viewer: Any) -> None:
    names = [lyr.name for lyr in viewer.layers]
    for attr in ("reference_layer", "barrier_layer", "ap_layer", "rl_layer", "fh_layer"):
        ref = getattr(widget, attr, None)
        if ref is None:
            continue
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
        step={"label": "Sliding step", "value": 0.001},
        up_thresh={"label": "Sliding up_thresh", "value": 0.8, "min": 0.0, "max": 1.0},
        smf={"label": "Sliding smooth win", "min": 1, "max": 200, "value": 10},
        min_size={"label": "Min component size", "min": 1, "value": 64},
        reference_layer={
            "label": "Reference layer",
            "widget_type": "ComboBox",
            "choices": [""],
            "value": "",
        },
        barrier_layer={
            "label": "Barrier mask layer (optional)",
            "widget_type": "ComboBox",
            "choices": [""],
            "value": "",
        },
        barrier_other_labels={"label": "Barrier: other labels on active layer", "value": False},
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
                "QVTpy explore (frac=0.35)",
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
        cmap={"label": "Colormap", "value": "turbo"},
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
        call_button="Run tool",
    )
    def tool_panel(
        category: str,
        operation: str,
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
        min_size: int,
        reference_layer: str,
        barrier_layer: str,
        barrier_other_labels: bool,
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
        cmap: str,
        ap_layer: str,
        rl_layer: str,
        fh_layer: str,
        loc_arterial_strategy: str,
        cross_section_radius_vox: float,
        measure_resegment: bool,
        hemo_method: str,
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

    @tool_panel.operation.changed.connect
    def _on_operation_changed(event) -> None:
        tid = tool_id_from_label(tool_panel.category.value, _signal_value(event))
        if tid:
            _set_param_visibility(tool_panel, tid)
        _update_reference_layers(tool_panel, viewer)

    tid0 = tool_id_from_label(default_category(), default_operation(default_category()))
    if tid0:
        _set_param_visibility(tool_panel, tid0)
        _update_reference_layers(tool_panel, viewer)
        _update_phase_layers(tool_panel, viewer)

    @tool_panel.pipeline_preset.changed.connect
    def _on_preset_changed(event) -> None:
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value)
        if tid != "seg_region_grow":
            return
        title = _signal_value(event)
        key = preset_key_from_title(tid, title)
        apply_preset_to_panel(tool_panel, tid, key)

    return tool_panel
