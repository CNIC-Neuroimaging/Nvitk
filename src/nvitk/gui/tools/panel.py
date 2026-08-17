"""Magicgui Tools tab with category/operation pickers and per-tool parameters."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from magicgui import magicgui

from nvitk.gui.labels.visibility import infer_target_mode
from nvitk.gui.tools.presets import apply_preset_to_panel, preset_key_from_title
from nvitk.gui.tools.runner import log_tool_failure, notify, parse_label_ids, run_gui_tool
from nvitk.gui.tools.registry import (
    categories,
    default_category,
    default_operation,
    operation_help_text,
    operations_for_category,
    params_for_tool,
    tool_by_id,
    tool_id_from_label,
)
from nvitk.measure.morpho.anatomy_axes import SPECIES_AUTO, SPECIES_CHOICES
from nvitk.measure.morpho.topology_io import topology_choices as _morpho_topology_choices
from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS


def _collect_params(widget: Any, tool_id: str) -> dict[str, Any]:
    """Read the current values of *tool_id*'s parameter widgets off *widget* into a plain dict
    (layer-picker params are coerced to their layer-name string)."""
    out = {}
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
    """Show only the parameter widgets on *widget* that *tool_id* actually uses; hide the rest of the
    shared parameter pool."""
    visible = {p.name for p in params_for_tool(tool_id)}
    all_names = (
        "footprint",
        "iterations",
        "mode",
        "connectivity",
        "n_largest",
        "sigma_spatial",
        "sigma_color",
        "do_3d",
        "axis",
        "step",
        "up_thresh",
        "smf",
        "shift_hm",
        "hessian_sigmas",
        "black_ridges",
        "hessian_alpha",
        "hessian_beta",
        "hessian_gamma",
        "jerman_sigmas",
        "jerman_tau",
        "snakes_alpha",
        "snakes_beta",
        "snakes_w_line",
        "snakes_w_edge",
        "snakes_gamma",
        "snakes_max_iter",
        "snakes_sigma",
        "snakes_n_points",
        "snakes_axis",
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
        "topology",
        "species",
        "n_workers",
        "input_already_smoothed",
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
        "cd_layer",
        "segmentation_layer",
        "cross_section_res",
        "interpolate_plane",
        "interp_vals",
        "thr_algorithm",
        "centerline_window",
        "show_segmentation_3d",
        "loc_arterial_strategy",
        "cross_section_radius_vox",
        "measure_resegment",
        "cs_supersampling",
        "label_constrain",
        "quality_metric",
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
        "n_seeds",
        "max_length",
        "stream_seed",
        "edge_width",
        "opacity",
        "trace_mode",
        "integration_direction",
        "seed_mode",
        "seed_plane_axis",
        "seed_plane_side",
        "dt_seconds",
        "color_metric",
        "per_vertex_color",
        "resample_paths",
        "resample_spacing_vox",
        "orient_mode",
        "target_orientation",
        "reorient_mode",
        "flip_x",
        "flip_y",
        "flip_z",
        "permute_order",
        "reset_affine",
        "output_path",
        "canvas_only",
        "gif_fps",
        "time_axis",
        "projection_axis",
        "projection_method",
        "quality_thresh",
        "stride",
        "root_region",
        "temporal_resolution_s",
        "pwv_method",
        "heart_rate_json",
        "station_color_feature",
        "station_point_size",
        "shrink_factor",
        "spline_param",
        "rescale_intensities",
        "hyst_low_factor",
        "hyst_high_factor",
        "thicken_iter",
        "thin_vesselness_percentile",
        "frangi_sigmas",
        "blood_flood_mode",
        "min_cc_voxels",
        "mouse_brain_mode",
        "mouse_modality",
        "which_parcellation",
        "do_n4",
        "fix_spacing",
        "binarize",
        "return_isotropic_output",
        "brain_modality",
        "image2_layer",
        "mask_layer",
        "fill_value",
        "mask_label_ids",
        "prediction_batch_size",
        "patch_stride_length",
        "dkt_preprocessing",
        "dkt_lobar",
        "dkt_denoising",
        "dkt_version",
        "expansion_factor",
        "sr_feature",
        "angle_degrees",
        "reshape",
        "swap_axis0",
        "swap_axis1",
    )
    for name in all_names:
        sub = getattr(widget, name, None)
        if sub is not None:
            sub.visible = name in visible


_LAYER_NONE = "(none)"


def _update_reference_layers(widget: Any, viewer: Any) -> None:
    """Refresh every reference/barrier/mask layer-picker widget on *widget* with the viewer's current
    layer names, adding a "(none)" option for the optional pickers and re-validating each widget's
    current selection against the new choice list."""
    names = [lyr.name for lyr in viewer.layers]
    optional_choices = [_LAYER_NONE, *names]
    _optional_layer_attrs = (
        "barrier_layer",
        "centerline_barrier_layer",
        "segmentation_layer",
        "image2_layer",
        "mask_layer",
    )
    for attr in (
        "reference_layer",
        "barrier_layer",
        "centerline_barrier_layer",
        "ap_layer",
        "rl_layer",
        "fh_layer",
        "cd_layer",
        "segmentation_layer",
        "organ_layer",
        "body_layer",
        "image2_layer",
        "mask_layer",
    ):
        ref = getattr(widget, attr, None)
        if ref is None:
            continue
        if attr in _optional_layer_attrs:
            ref.choices = optional_choices
            if ref.value not in optional_choices:
                ref.value = _LAYER_NONE
        else:
            ref.choices = names
            if names and ref.value not in names:
                ref.value = names[0] if attr == "reference_layer" else ""


def _prefill_vessel_cross_section_layers(widget: Any, viewer: Any) -> None:
    """Default CD layer dropdown from the active layer (centerline = active layer)."""
    if not viewer.layers:
        return
    active = viewer.layers.selection.active or viewer.layers[-1]
    aname = active.name
    names = [lyr.name for lyr in viewer.layers]
    cd = getattr(widget, "cd_layer", None)
    if cd is not None and aname in names and not str(cd.value or "").strip():
        cd.value = aname


def _update_phase_layers(widget: Any, viewer: Any) -> None:
    """Alias for :func:`_update_reference_layers`, kept for call sites conceptually refreshing
    per-phase layer pickers."""
    _update_reference_layers(widget, viewer)


def build_tool_panel(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    layer_display_kwargs: Callable[..., dict[str, Any]],
    on_layers_changed: Callable[[], None],
    record_step = None,
    get_label_ids = None,
    get_pipeline_argv_builder = None,
    get_totalseg_roi = None,
    label_selector = None,
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
        label_ids={"label": "Label id(s) (comma-separated)", "value": ""},
        overlay_mode={"choices": ["add_layer", "replace_active"], "label": "Output mode"},
        footprint={"label": "Footprint (radius)", "min": 1, "max": 32, "value": 1},
        iterations={"label": "Iterations", "min": 1, "max": 20, "value": 1},
        mode={"choices": ["binary", "gray"], "label": "Morph mode", "value": "binary"},
        connectivity={"label": "Connectivity", "min": 1, "max": 3, "value": 2},
        n_largest={"label": "Keep N largest CCs", "min": 1, "max": 1000, "value": 1},
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
        hessian_sigmas={"label": "Hessian sigmas (comma-separated)", "value": "1,3,5,7,9"},
        black_ridges={"label": "Black ridges (else bright)", "value": False},
        hessian_alpha={"label": "Hessian alpha", "value": 0.5, "min": 0.01, "max": 10.0},
        hessian_beta={"label": "Hessian beta", "value": 0.5, "min": 0.01, "max": 10.0},
        hessian_gamma={"label": "Hessian gamma", "value": 15.0, "min": 0.1, "max": 100.0},
        jerman_sigmas={"label": "Jerman sigmas (comma-separated)", "value": "1,3,5,7,9"},
        jerman_tau={"label": "Jerman tau", "value": 0.5, "min": 0.5, "max": 1.0},
        snakes_alpha={"label": "Snakes alpha (tension)", "value": 0.01, "min": 0.0, "max": 5.0},
        snakes_beta={"label": "Snakes beta (rigidity)", "value": 0.1, "min": 0.0, "max": 50.0},
        snakes_w_line={"label": "Snakes w_line", "value": 0.0, "min": -10.0, "max": 10.0},
        snakes_w_edge={"label": "Snakes w_edge", "value": 1.0, "min": -10.0, "max": 10.0},
        snakes_gamma={"label": "Snakes gamma", "value": 0.01, "min": 1e-5, "max": 1.0},
        snakes_max_iter={"label": "Snakes max iterations", "min": 10, "max": 20000, "value": 2500},
        snakes_sigma={"label": "Snakes Gaussian sigma", "value": 1.0, "min": 0.0, "max": 20.0},
        snakes_n_points={"label": "Snakes control points", "min": 16, "max": 4000, "value": 400},
        snakes_axis={"label": "Snakes 3D slice axis", "min": 0, "max": 2, "value": 0},
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
        topology={
            "choices": list(_morpho_topology_choices()),
            "label": "Topology JSON",
            "value": "none",
        },
        species={
            "choices": list(SPECIES_CHOICES),
            "label": "Species",
            "value": SPECIES_AUTO,
        },
        n_workers={"label": "Workers", "min": 1, "max": 64, "value": 1},
        input_already_smoothed={"label": "Input already Taubin-smoothed", "value": False},
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
        station_point_size={"label": "Station point size", "value": 2.5, "min": 0.1, "max": 100.0},
        station_color_feature={
            "choices": [
                "distance_mm",
                "pi",
                "quality",
                "area_mm2",
                "pwv_weight_area",
                "pwv_weight_quality",
                "pwv_xcor_time_s",
                "pwv_time_to_upstroke_s",
                "pwv_bjornfoot_weighted_rms",
                "pwv_bjornfoot_delay_residual_s",
                "pwv_bjornfoot_waveform_corr",
            ],
            "label": "Color stations by",
            "value": "quality",
        },
        heart_rate_json={
            "label": "Cardiac metadata JSON (HeartRate)",
            "value": "",
        },
        cmap={
            "label": "SUV colormap",
            "choices": ["viridis", "turbo", "magma", "inferno", "plasma", "cividis", "hot", "coolwarm"],
            "value": "viridis",
        },
        ap_layer={"label": "AP phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        rl_layer={"label": "RL phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        fh_layer={"label": "FH phase layer", "widget_type": "ComboBox", "choices": [""], "value": ""},
        cd_layer={
            "label": "Complex difference layer",
            "widget_type": "ComboBox",
            "choices": [""],
            "value": "",
        },
        segmentation_layer={
            "label": "Segmentation mask (optional)",
            "widget_type": "ComboBox",
            "choices": [_LAYER_NONE],
            "value": _LAYER_NONE,
        },
        cross_section_res={
            "label": "Plane resolution (0=auto)",
            "min": 0,
            "max": 1024,
            "value": 0,
        },
        interpolate_plane={"label": "Interpolate plane sampling", "value": True},
        interp_vals={"label": "Samples per voxel (auto res)", "min": 1, "max": 16, "value": 4},
        thr_algorithm={
            "choices": ["otsu", "lsthr", "lthr"],
            "label": "2D threshold method",
            "value": "lsthr",
        },
        centerline_window={
            "choices": ["3", "5"],
            "label": "Tangent window",
            "value": "5",
        },
        show_segmentation_3d={"label": "Show segmentation in 3D", "value": True},
        loc_arterial_strategy={
            "choices": ["qvtpy", "midpoint"],
            "label": "Arterial LOC strategy",
            "value": "qvtpy",
        },
        cross_section_radius_vox={"label": "Cross-section radius (vox)", "value": 10.0, "min": 1.0},
        measure_resegment={"label": "Resegment in-plane", "value": False},
        cs_supersampling={"label": "Supersample plane (~4×)", "value": True},
        label_constrain={"label": "Constrain to vessel label", "value": True},
        quality_metric={
            "choices": ["stdv_from_mean", "waveform"],
            "label": "Quality metric",
            "value": "stdv_from_mean",
        },
        quality_thresh={"label": "Quality threshold", "value": 2.5, "min": 0.0, "max": 4.0},
        stride={"label": "Station stride", "min": 1, "max": 50, "value": 1},
        root_region={
            "choices": ["All", "L_ICA", "R_ICA", "Basilar"],
            "label": "Root region",
            "value": "All",
        },
        temporal_resolution_s={"label": "Temporal resolution (s)", "value": 0.04, "min": 0.0001, "max": 10.0},
        pwv_method={
            "choices": ["bjornfoot", "fielding", "both"],
            "label": "PWV overlay mode",
            "value": "both",
        },
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
        min_branch_points={
            "label": "Min branch points (0 = keep all)",
            "min": 0,
            "max": 5000,
            "value": 0,
        },
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
        trace_mode={
            "choices": ["streamlines", "pathlines"],
            "label": "Trace mode",
            "value": "streamlines",
        },
        n_seeds={"label": "Seed count", "min": 1, "max": 10000, "value": 64},
        max_length={
            "label": "Max length (vox; pathline horizon s if pathlines)",
            "value": 35.0,
            "min": 1.0,
            "max": 500.0,
        },
        stream_seed={"label": "Random seed", "min": 0, "max": 999999, "value": 42},
        integration_direction={
            "choices": ["forward", "backward", "both"],
            "label": "Integration direction (streamlines)",
            "value": "forward",
        },
        seed_mode={
            "choices": ["planar", "volume"],
            "label": "Seed placement",
            "value": "planar",
        },
        seed_plane_axis={"label": "Planar seed axis (0/1/2)", "min": 0, "max": 2, "value": 2},
        seed_plane_side={
            "choices": ["min", "max"],
            "label": "Planar seed side",
            "value": "min",
        },
        dt_seconds={"label": "Pathline dt (seconds)", "value": 1.0, "min": 0.01, "max": 60.0},
        color_metric={
            "choices": ["speed", "integration_time", "arc_length", "fixed"],
            "label": "Color by",
            "value": "speed",
        },
        per_vertex_color={"label": "Per-vertex color gradient", "value": True},
        resample_paths={"label": "Resample paths uniformly", "value": False},
        resample_spacing_vox={
            "label": "Resample spacing (vox)",
            "value": 0.5,
            "min": 0.1,
            "max": 5.0,
        },
        edge_width={"label": "Path line width (vox)", "value": 0.25, "min": 0.05, "max": 10.0},
        opacity={"label": "Trace opacity", "value": 0.55, "min": 0.05, "max": 1.0},
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
        shrink_factor={"label": "N4 shrink factor", "min": 1, "max": 16, "value": 4},
        spline_param={"label": "N4 spline param (0=default)", "value": 0.0, "min": 0.0},
        rescale_intensities={"label": "N4 rescale intensities", "value": False},
        hyst_low_factor={"label": "Hysteresis low factor", "value": 3.0, "min": 0.1, "max": 20.0},
        hyst_high_factor={"label": "Hysteresis high factor", "value": 0.5, "min": 0.01, "max": 5.0},
        thicken_iter={"label": "Thicken iterations", "min": 0, "max": 20, "value": 0},
        thin_vesselness_percentile={
            "label": "Thin vesselness percentile (<0 off)",
            "value": 55.0,
            "min": -1.0,
            "max": 100.0,
        },
        frangi_sigmas={"label": "Frangi sigmas (comma-separated)", "value": "0.5,1.0,1.5,2.0,2.5"},
        blood_flood_mode={
            "choices": ["expand", "from_scratch"],
            "label": "Blood flood mode",
            "value": "expand",
        },
        min_cc_voxels={"label": "Min tree CC size (from-scratch)", "min": 1, "max": 100000, "value": 5},
        mouse_brain_mode={
            "choices": ["extraction", "parcellation"],
            "label": "Mouse brain mode",
            "value": "extraction",
        },
        mouse_modality={
            "choices": ["t2", "ex5coronal", "ex5sagittal"],
            "label": "Mouse modality (no T1)",
            "value": "t2",
        },
        which_parcellation={
            "choices": ["nick", "tct", "jay"],
            "label": "Parcellation scheme",
            "value": "nick",
        },
        do_n4={"label": "N4 bias correction first", "value": True},
        fix_spacing={"label": "Auto-fix unit spacing (~20 mm FOV)", "value": True},
        binarize={"label": "Binarize extraction mask", "value": True},
        return_isotropic_output={"label": "Isotropic output resampling", "value": False},
        brain_modality={
            "choices": [
                "t1",
                "t1nobrainer",
                "t1combined",
                "t1threetissue",
                "t1hemi",
                "t1lobes",
                "flair",
                "t2",
                "t2star",
                "bold",
                "fa",
                "mra",
                "t1t2infant",
                "t1infant",
                "t2infant",
            ],
            "label": "Brain extraction modality",
            "value": "t1",
        },
        image2_layer={
            "label": "Second modality layer (optional)",
            "widget_type": "ComboBox",
            "choices": [_LAYER_NONE],
            "value": _LAYER_NONE,
        },
        mask_layer={
            "label": "Brain mask layer (optional)",
            "widget_type": "ComboBox",
            "choices": [_LAYER_NONE],
            "value": _LAYER_NONE,
        },
        fill_value={"label": "Fill value (masked voxels)", "value": 0.0},
        mask_label_ids={
            "label": "Mask label id(s) (empty = all nonzero)",
            "value": "",
        },
        prediction_batch_size={"label": "MRA prediction batch size", "min": 1, "max": 64, "value": 2},
        patch_stride_length={"label": "MRA patch stride", "min": 8, "max": 128, "value": 32},
        dkt_preprocessing={"label": "DKT preprocessing", "value": True},
        dkt_lobar={"label": "DKT lobar parcellation", "value": False},
        dkt_denoising={"label": "DKT denoising", "value": True},
        dkt_version={"label": "DKT model version", "min": 0, "max": 2, "value": 0},
        expansion_factor={"label": "SR expansion factor (ints)", "value": "1,1,2"},
        sr_feature={"choices": ["vgg", "grader"], "label": "SR feature backbone", "value": "vgg"},
        angle_degrees={"label": "Rotate angle (degrees, CCW)", "value": 90.0, "min": -360.0, "max": 360.0},
        reshape={"label": "Expand canvas to fit rotation", "value": False},
        swap_axis0={"label": "Swap axis A", "min": 0, "max": 3, "value": 0},
        swap_axis1={"label": "Swap axis B", "min": 0, "max": 3, "value": 1},
        orient_mode={"choices": ["view", "reorient"], "label": "Action", "value": "view"},
        target_orientation={
            "choices": ["RAS", "LPS", "LAS", "RPS", "RSA", "LPI", "LSA", "RPI", "LIA", "RIA"],
            "label": "Target orientation",
            "value": "RAS",
        },
        reorient_mode={
            "choices": ["mouse", "reference", "manual"],
            "label": "Reorient mode",
            "value": "mouse",
        },
        permute_order={"label": "Permute order (e.g. 0,2,1)", "value": "0,1,2"},
        flip_x={"label": "Flip axis 0 (X)", "value": False},
        flip_y={"label": "Flip axis 1 (Y)", "value": False},
        flip_z={"label": "Flip axis 2 (Z)", "value": False},
        reset_affine={
            "label": "Reset affine to target codes (ignore wrong header)",
            "value": False,
        },
        call_button="Run tool",
    )
    def tool_panel(
        category: str,
        operation: str,
        operation_help: str,
        label_ids: str,
        overlay_mode: str,
        footprint: int,
        iterations: int,
        mode: str,
        connectivity: int,
        n_largest: int,
        sigma_spatial: float,
        sigma_color: float,
        do_3d: bool,
        axis: int,
        step: float,
        up_thresh: float,
        smf: int,
        shift_hm: bool,
        hessian_sigmas: str,
        black_ridges: bool,
        hessian_alpha: float,
        hessian_beta: float,
        hessian_gamma: float,
        jerman_sigmas: str,
        jerman_tau: float,
        snakes_alpha: float,
        snakes_beta: float,
        snakes_w_line: float,
        snakes_w_edge: float,
        snakes_gamma: float,
        snakes_max_iter: int,
        snakes_sigma: float,
        snakes_n_points: int,
        snakes_axis: int,
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
        topology: str,
        species: str,
        n_workers: int,
        input_already_smoothed: bool,
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
        cd_layer: str,
        segmentation_layer: str,
        cross_section_res: int,
        interpolate_plane: bool,
        interp_vals: int,
        thr_algorithm: str,
        centerline_window: str,
        show_segmentation_3d: bool,
        loc_arterial_strategy: str,
        cross_section_radius_vox: float,
        measure_resegment: bool,
        cs_supersampling: bool,
        label_constrain: bool,
        quality_metric: str,
        quality_thresh: float,
        stride: int,
        root_region: str,
        temporal_resolution_s: float,
        pwv_method: str,
        station_color_feature: str,
        station_point_size: float,
        heart_rate_json: str,
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
        trace_mode: str,
        n_seeds: int,
        max_length: float,
        stream_seed: int,
        integration_direction: str,
        seed_mode: str,
        seed_plane_axis: int,
        seed_plane_side: str,
        dt_seconds: float,
        color_metric: str,
        per_vertex_color: bool,
        resample_paths: bool,
        resample_spacing_vox: float,
        edge_width: float,
        opacity: float,
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
        shrink_factor: int,
        spline_param: float,
        rescale_intensities: bool,
        hyst_low_factor: float,
        hyst_high_factor: float,
        thicken_iter: int,
        thin_vesselness_percentile: float,
        frangi_sigmas: str,
        blood_flood_mode: str,
        min_cc_voxels: int,
        mouse_brain_mode: str,
        mouse_modality: str,
        which_parcellation: str,
        do_n4: bool,
        fix_spacing: bool,
        binarize: bool,
        return_isotropic_output: bool,
        brain_modality: str,
        image2_layer: str,
        mask_layer: str,
        fill_value: float,
        mask_label_ids: str,
        prediction_batch_size: int,
        patch_stride_length: int,
        dkt_preprocessing: bool,
        dkt_lobar: bool,
        dkt_denoising: bool,
        dkt_version: int,
        expansion_factor: str,
        sr_feature: str,
        angle_degrees: float,
        reshape: bool,
        swap_axis0: int,
        swap_axis1: int,
        orient_mode: str,
        target_orientation: str,
        reorient_mode: str,
        permute_order: str,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        reset_affine: bool,
    ) -> None:
        """Run the tool selected by ``category``/``operation`` on the active (or last) layer with the
        collected parameters, then add or replace a layer with the result and record the step."""
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

        ids = []
        if get_label_ids is not None:
            ids = list(get_label_ids())
        if not ids:
            ids = parse_label_ids(label_ids)
        if tool_id == "siphon_correct" and ids:
            correction_ids = ",".join(str(i) for i in ids)

        target_mode = infer_target_mode(layer, label_ids=ids or None)
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
                    params["pipeline_argv"] = form.build_argv(exe, active_layer=layer)
                    params["pipeline_layer_bindings"] = form.build_layer_bindings(
                        active_layer=layer
                    )
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
        result_arr = np.asarray(result)
        as_labels = tool_id == "seg_blood_flood" and (
            np.issubdtype(result_arr.dtype, np.integer)
            and int(result_arr.max(initial=0)) > 1
        )
        if can_replace and not as_labels:
            try:
                layer.data = result
                layer.name = name
            except Exception:
                viewer.add_image(result, **out_kwargs)
        elif as_labels:
            from nvitk.gui.core.spatial import layer_spatial_kwargs

            spatial_src = layer
            ref_name = str(params.get("reference_layer") or "").strip()
            if ref_name and ref_name not in ("", "(none)"):
                try:
                    spatial_src = next(ly for ly in viewer.layers if ly.name == ref_name)
                except StopIteration:
                    pass
            spatial = layer_spatial_kwargs(spatial_src)
            try:
                lab = viewer.add_labels(
                    result_arr.astype(np.int32, copy=False),
                    name=name,
                    opacity=0.7,
                    **spatial,
                )
                lab._nvitk_label_like = True
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
        """Extract the new value from a magicgui change *event* (or pass through a raw value)."""
        return event.value if hasattr(event, "value") else event

    def _sync_operation_help() -> None:
        """Refresh the read-only operation-help text box for the current category/operation selection."""
        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op)
        tool_panel.operation_help.value = operation_help_text(tid)

    @tool_panel.category.changed.connect
    def _on_category_changed(event) -> None:
        """Repopulate the operation dropdown for the newly selected category and resync parameter
        visibility and help text."""
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
        """Resync parameter visibility, reference-layer choices, and help text for the newly selected
        operation."""
        tid = tool_id_from_label(tool_panel.category.value, _signal_value(event))
        if tid:
            _set_param_visibility(tool_panel, tid)
        _update_reference_layers(tool_panel, viewer)
        if tid == "viz_vessel_cross_sections":
            _prefill_vessel_cross_section_layers(tool_panel, viewer)
        _sync_operation_help()

    tid0 = tool_id_from_label(default_category(), default_operation(default_category()))
    if tid0:
        _set_param_visibility(tool_panel, tid0)
        _update_reference_layers(tool_panel, viewer)
        _update_phase_layers(tool_panel, viewer)
    _sync_operation_help()

    @tool_panel.pipeline_preset.changed.connect
    def _on_preset_changed(event) -> None:
        """Apply the newly selected pipeline preset's parameter values (region-grow tool only)."""
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value)
        if tid != "seg_region_grow":
            return
        title = _signal_value(event)
        key = preset_key_from_title(tid, title)
        apply_preset_to_panel(tool_panel, tid, key)

    return tool_panel
