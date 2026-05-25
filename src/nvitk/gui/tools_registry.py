"""GUI tool catalog: categories, operations, and parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nvitk.gui.pipes_catalog import PIPELINE_TOOLS

ParamKind = Literal[
    "int",
    "float",
    "bool",
    "choice",
    "str",
    "layer",
]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: ParamKind = "float"
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()


RunMode = Literal["layer", "notify", "pipeline"]


@dataclass(frozen=True)
class GuiToolSpec:
    id: str
    category: str
    label: str
    params: tuple[ParamSpec, ...] = ()
    needs_reference_layer: bool = False
    needs_3d: bool = False
    run_mode: RunMode = "layer"
    cli_command: str = ""
    description: str = ""


TOOL_IDS_USING_LABEL_PICKER: frozenset[str] = frozenset({
    "siphon_correct",
    "label_cc",
    "remove_small_components",
    "seg_region_grow",
    "seg_split_lr_cc",
    "seg_combine_labels",
    "seg_remove_labels",
    "centerline_detect_junctions",
    "centerline_cut_junctions",
    "viz_pet_hotspots",
    "viz_flowshow",
})

_CATEGORY_ORDER = (
    "Restoration",
    "Filters",
    "Morphology",
    "Centerline",
    "Segmentation",
    "Registration",
    "Visualization",
    "Transform",
    "Measure",
    "Pipelines",
)

_LABEL_ID = ParamSpec("label_id", "Label id", "int", 1, min=0, max=9999)
_LABEL_IDS = ParamSpec("label_ids", "Label id(s) comma-separated", "str", "1")
_NEW_ID = ParamSpec("new_id", "Output label id", "int", 1, min=0, max=9999)
_OUTPUT_DIR = ParamSpec("output_dir", "Output directory", "str", "")
_WORKING_DIR = ParamSpec("working_dir", "Working directory", "str", "")
def _totalseg_task_choices() -> tuple[str, ...]:
    try:
        from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS

        return AVAILABLE_TASKS
    except Exception:
        return ("total", "total_mr", "brain_structures", "body")


_TASK = ParamSpec(
    "task",
    "TotalSegmentator task",
    "choice",
    "total",
    choices=_totalseg_task_choices(),
)
_CORRECTION_IDS = ParamSpec("correction_ids", "ICA label ids (e.g. 1,2)", "str", "1,2")

_MORPH_PARAMS = (
    ParamSpec("footprint", "Footprint (radius voxels)", "int", 1, min=1, max=32),
    ParamSpec("iterations", "Iterations", "int", 1, min=1, max=20),
    ParamSpec("mode", "Mode", "choice", "binary", choices=("binary", "gray")),
    ParamSpec("connectivity", "Connectivity", "int", 2, min=1, max=3),
)

_TOOLS: tuple[GuiToolSpec, ...] = (
    GuiToolSpec(
        "bilateral",
        "Restoration",
        "Bilateral filter",
        (
            ParamSpec("sigma_spatial", "Sigma spatial (0=auto)", "float", 0.0, min=0),
            ParamSpec("sigma_color", "Sigma color (0=auto)", "float", 0.0, min=0),
            ParamSpec("do_3d", "3D filter (else slice-wise)", "bool", False),
            ParamSpec("axis", "Slice axis (2D-on-3D)", "int", 0, min=0, max=2),
        ),
    ),
    GuiToolSpec(
        "sliding_threshold",
        "Filters",
        "Sliding threshold",
        (
            ParamSpec("step", "Step", "float", 0.001, min=0.0001, max=0.1),
            ParamSpec("up_thresh", "Upper threshold frac", "float", 0.8, min=0.1, max=1.0),
            ParamSpec("smf", "Smoothing window", "int", 10, min=1, max=200),
            ParamSpec("shift_hm", "Shift HM (half-max on curvature)", "bool", True),
        ),
    ),
    GuiToolSpec("dilate", "Morphology", "Dilate", _MORPH_PARAMS),
    GuiToolSpec("erode", "Morphology", "Erode", _MORPH_PARAMS),
    GuiToolSpec("open", "Morphology", "Open", _MORPH_PARAMS),
    GuiToolSpec("close", "Morphology", "Close", _MORPH_PARAMS),
    GuiToolSpec(
        "fill_holes",
        "Morphology",
        "Fill holes",
        (ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),),
    ),
    GuiToolSpec(
        "label_cc",
        "Morphology",
        "Label connected components",
        (ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),),
    ),
    GuiToolSpec(
        "remove_small_components",
        "Morphology",
        "Remove small components",
        (
            ParamSpec("min_size", "Min size (voxels)", "int", 64, min=1, max=1_000_000),
            ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),
        ),
    ),
    GuiToolSpec(
        "morph_biggest_cc",
        "Morphology",
        "Largest connected component",
        (ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),),
        needs_3d=True,
    ),
    GuiToolSpec(
        "skeletonize",
        "Morphology",
        "Skeletonize / centerline",
        needs_3d=True,
    ),
    GuiToolSpec(
        "centerline_detect_junctions",
        "Centerline",
        "Detect skeleton junctions",
        (
            ParamSpec("min_junction_degree", "Min skeleton degree", "int", 3, min=2, max=26),
            ParamSpec("reskeletonize", "Re-skeletonize mask (thick masks only)", "bool", False),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Mark skeleton branch points (degree ≥ N) on a 3D centerline mask."
        ),
    ),
    GuiToolSpec(
        "centerline_cut_junctions",
        "Centerline",
        "Cut centerline at junction points",
        (
            ParamSpec("new_label_start", "First new label id (0=auto)", "int", 0, min=0, max=9999),
            ParamSpec("reskeletonize", "Re-skeletonize mask (thick masks only)", "bool", False),
        ),
        needs_3d=True,
        run_mode="layer",
        description=(
            "Split a label at junction markers from Detect skeleton junctions."
        ),
    ),
    GuiToolSpec(
        "siphon_correct",
        "Morphology",
        "ICA siphon centerline correction",
        (
            ParamSpec("reference_layer", "TOF / MRA image layer", "layer", ""),
            _CORRECTION_IDS,
            _OUTPUT_DIR,
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "mask_genus",
        "Morphology",
        "Mask genus (topology β₁)",
        (),
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "seg_get_label",
        "Segmentation",
        "Extract label → binary mask",
        (_LABEL_ID,),
    ),
    GuiToolSpec(
        "seg_combine_labels",
        "Segmentation",
        "Combine labels → mask",
        (_LABEL_IDS, _NEW_ID),
    ),
    GuiToolSpec(
        "seg_remove_labels",
        "Segmentation",
        "Remove label ids",
        (_LABEL_IDS,),
    ),
    GuiToolSpec(
        "seg_pet_ureter",
        "Segmentation",
        "PET ureter segmentation",
        (
            ParamSpec("reference_layer", "PET image layer", "layer", ""),
            ParamSpec("organ_layer", "Organ labels layer (optional)", "layer", ""),
            ParamSpec("body_layer", "Body mask layer", "layer", ""),
            ParamSpec("kidney_r_id", "Kidney right label id", "int", 2, min=0, max=9999),
            ParamSpec("kidney_l_id", "Kidney left label id", "int", 3, min=0, max=9999),
            ParamSpec("bladder_id", "Bladder label id", "int", 21, min=0, max=9999),
            ParamSpec("radius_mm", "Ureter tube radius (mm)", "float", 6.0, min=0.5, max=30.0),
            ParamSpec("w_pet", "PET cost weight", "float", 5.0, min=0.1, max=50.0),
            ParamSpec("suv_kind", "SUV kind", "choice", "bw", choices=("bw", "lbm", "bsa", "ibw")),
            ParamSpec("philips_factor", "Use Philips SUV factor tag", "bool", True),
            ParamSpec("revert_scaling", "Revert per-slice rescale", "bool", False),
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="layer",
    ),
    GuiToolSpec(
        "seg_convex_hull_slice",
        "Segmentation",
        "Convex hull (slice-wise)",
        (ParamSpec("hull_axis", "Slice axis (0/1/2)", "int", 2, min=0, max=2),),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_convex_hull_3d",
        "Segmentation",
        "Convex hull (3D)",
        (),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_distance_transform",
        "Segmentation",
        "Distance transform (EDT)",
        (
            ParamSpec("radius_mm", "Tube radius mm (0=full map)", "float", 0.0, min=0.0, max=100.0),
            ParamSpec("edt_use_spacing", "Distance in mm (layer spacing)", "bool", True),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_union",
        "Segmentation",
        "Mask union (OR)",
        (ParamSpec("reference_layer", "Second mask layer", "layer", ""),),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_intersection",
        "Segmentation",
        "Mask intersection (AND)",
        (ParamSpec("reference_layer", "Second mask layer", "layer", ""),),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_subtract",
        "Segmentation",
        "Mask subtract (A \\ B)",
        (ParamSpec("reference_layer", "Subtract mask B layer", "layer", ""),),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_xor",
        "Segmentation",
        "Mask XOR",
        (ParamSpec("reference_layer", "Second mask layer", "layer", ""),),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_complement",
        "Segmentation",
        "Mask complement (NOT)",
        (
            ParamSpec("reference_layer", "Limit to ROI layer (optional)", "layer", ""),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_biggest_cc",
        "Segmentation",
        "Largest connected component",
        (),
    ),
    GuiToolSpec(
        "seg_split_lr_cc",
        "Segmentation",
        "Split L/R (connected components)",
        (),
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "seg_split_lr_midline",
        "Segmentation",
        "Split L/R (midline plane)",
        (ParamSpec("plane_x", "Midline X (voxel, 0=auto)", "int", 0, min=0),),
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "seg_region_grow",
        "Segmentation",
        "Region growing (intensity)",
        (
            ParamSpec(
                "pipeline_preset",
                "Pipeline preset",
                "choice",
                "Custom",
                choices=("Custom", "QVTpy default (frac=0.45)", "QVTpy explore (frac=0.35)", "QVTpy ICA test (frac=0.45)"),
            ),
            ParamSpec("reference_layer", "Intensity image layer", "layer", ""),
            ParamSpec("barrier_layer", "Barrier mask layer (optional)", "layer", ""),
            ParamSpec("barrier_other_labels", "Barrier: other labels on active layer", "bool", False),
            ParamSpec("barrier_radius_vox", "Barrier dilation (vox)", "int", 3, min=0, max=32),
            ParamSpec("seed_from_label", "Seed from label centroid", "bool", False),
            ParamSpec("seed_z", "Seed Z", "int", 0, min=0),
            ParamSpec("seed_y", "Seed Y", "int", 0, min=0),
            ParamSpec("seed_x", "Seed X", "int", 0, min=0),
            ParamSpec("threshold", "Intensity fraction", "float", 0.0, min=0.0, max=1.0),
        ),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_adjust_masks",
        "Segmentation",
        "Adjust masks (overlap slices only)",
        (ParamSpec("reference_layer", "Second mask layer", "layer", ""),),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_totalsegmentator",
        "Segmentation",
        "TotalSegmentator (CLI)",
        (_OUTPUT_DIR, _TASK),
        run_mode="notify",
    ),
    GuiToolSpec(
        "seg_eicab",
        "Segmentation",
        "eICAB cluster (CLI)",
        (_OUTPUT_DIR,),
        run_mode="notify",
    ),
    GuiToolSpec(
        "viz_pet_hotspots",
        "Visualization",
        "PET / SUV hotspots (Napari)",
        (
            ParamSpec("reference_layer", "Intensity / SUV layer", "layer", ""),
            ParamSpec("hotspot", "Hotspot mode", "choice", "top_percent", choices=("top_percent", "top_k", "threshold")),
            ParamSpec("top_percent", "Top percent", "float", 0.1, min=0.01, max=100.0),
            ParamSpec("max_points", "Max points", "int", 20000, min=100, max=500000),
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "viz_flowshow",
        "Visualization",
        "4D flow vectors (Napari)",
        (
            ParamSpec("ap_layer", "AP phase layer", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer", "layer", ""),
            ParamSpec("reference_layer", "Vessel mask layer", "layer", ""),
            ParamSpec("time_index", "Cardiac phase index", "int", 0, min=0, max=64),
            ParamSpec("max_points", "Max vector glyphs", "int", 4000, min=100, max=100000),
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "reg_flirt_rigid",
        "Registration",
        "FLIRT rigid register",
        (
            ParamSpec("reference_layer", "Fixed / reference layer", "layer", ""),
            _OUTPUT_DIR,
            ParamSpec("dof", "Degrees of freedom", "int", 6, min=6, max=12),
            ParamSpec("cost", "FLIRT cost", "str", "corratio"),
            ParamSpec(
                "searchr_x",
                "Search range X (deg, 0=default)",
                "float",
                0.0,
                min=0.0,
                max=180.0,
            ),
            ParamSpec("warped_name", "Warped output filename", "str", "moving_warped.nii.gz"),
            ParamSpec("matrix_name", "Matrix output filename", "str", "affine.mat"),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "reg_flirt_apply",
        "Registration",
        "FLIRT apply transform",
        (
            ParamSpec("reference_layer", "Reference space layer", "layer", ""),
            ParamSpec("mat_path", "FLIRT matrix (.mat)", "str", ""),
            ParamSpec("out_path", "Output NIfTI path (empty=temp)", "str", ""),
            ParamSpec("interp", "Interpolation", "str", "trilinear"),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "isotropy",
        "Transform",
        "Isotropy (resample anisotropic axis)",
        (
            ParamSpec("axis", "Axis (-1=auto)", "int", -1, min=-1, max=2),
            ParamSpec("factor", "Factor (0=auto)", "float", 0.0, min=0),
            ParamSpec("order", "Interpolation order", "int", 1, min=0, max=5),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "resample_to",
        "Transform",
        "Resample to reference layer grid",
        (
            ParamSpec("reference_layer", "Reference layer", "layer", ""),
            ParamSpec("order", "Interpolation order", "int", 0, min=0, max=5),
        ),
        needs_reference_layer=True,
    ),
    GuiToolSpec(
        "oblique_slice",
        "Transform",
        "Oblique slice (mid-volume plane)",
        (
            ParamSpec("radius_vox", "Half-size (voxels)", "float", 40.0, min=1),
            ParamSpec("res", "Output resolution", "int", 256, min=16, max=1024),
            ParamSpec("order", "Interpolation order", "int", 1, min=0, max=5),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "qvtpy_locs",
        "Measure",
        "QVTpy LOCs (generate / load CSV)",
        (
            ParamSpec("loc_mode", "LOC mode", "choice", "load_csv", choices=("load_csv", "generate")),
            ParamSpec("locs_csv", "LOCs CSV path", "str", ""),
            ParamSpec("subject", "Subject id (generate)", "str", ""),
            ParamSpec("nifti_root", "NIfTI root (generate)", "str", ""),
            ParamSpec("output_root", "Output root (generate)", "str", ""),
            ParamSpec("loc_arterial_strategy", "Arterial strategy", "choice", "qvtpy", choices=("qvtpy", "midpoint")),
            ParamSpec("cross_section_radius_vox", "Cross-section radius (vox)", "float", 10.0, min=1.0),
        ),
        run_mode="notify",
        cli_command="qvtpy-stage5-loc",
    ),
    GuiToolSpec(
        "measure_loc_hemodynamics",
        "Measure",
        "LOC hemodynamics (PI / RI)",
        (
            ParamSpec("locs_csv", "LOCs CSV path", "str", ""),
            ParamSpec("subject", "Subject id (disk phases)", "str", ""),
            ParamSpec("nifti_root", "NIfTI root", "str", ""),
            ParamSpec("output_root", "Pipeline output root", "str", ""),
            ParamSpec("ap_layer", "AP phase layer (optional)", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer (optional)", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer (optional)", "layer", ""),
            ParamSpec("cross_section_radius_vox", "Cross-section radius (vox)", "float", 10.0, min=1.0),
            ParamSpec("measure_resegment", "Resegment in-plane", "bool", True),
        ),
        run_mode="notify",
    ),
    GuiToolSpec(
        "measure_mask_hemodynamics",
        "Measure",
        "Mask hemodynamics (pseudo-LOC / voxel avg)",
        (
            ParamSpec("ap_layer", "AP phase layer", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer", "layer", ""),
            ParamSpec("reference_layer", "Angio / CD layer (optional)", "layer", ""),
            ParamSpec(
                "hemo_method",
                "Method",
                "choice",
                "both",
                choices=("pseudo_loc", "voxel_avg", "both"),
            ),
            ParamSpec("cross_section_radius_vox", "Pseudo-LOC radius (vox)", "float", 10.0, min=1.0),
            ParamSpec("measure_resegment", "Resegment in-plane", "bool", True),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec("volume_mm3", "Measure", "Volume (mm³)", (), run_mode="notify"),
    GuiToolSpec("volume_cc", "Measure", "Volume (cc)", (), run_mode="notify"),
    GuiToolSpec(
        "masked_stats",
        "Measure",
        "Masked intensity stats",
        (ParamSpec("reference_layer", "Intensity image layer", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "integrated_intensity",
        "Measure",
        "Integrated intensity",
        (ParamSpec("reference_layer", "Intensity image layer", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "suv_stats",
        "Measure",
        "SUV statistics",
        (
            ParamSpec("reference_layer", "PET image layer (optional)", "layer", ""),
            ParamSpec("suv_kind", "SUV kind", "choice", "bw", choices=("bw", "lbm", "bsa", "ibw")),
            ParamSpec("philips_factor", "Use Philips SUV factor tag", "bool", True),
            ParamSpec("revert_scaling", "Revert per-slice rescale", "bool", False),
        ),
        run_mode="notify",
    ),
    GuiToolSpec(
        "measure_generate_suv",
        "Measure",
        "Generate SUV volume from PET",
        (
            ParamSpec("suv_kind", "SUV kind", "choice", "bw", choices=("bw", "lbm", "bsa", "ibw")),
            ParamSpec("philips_factor", "Use Philips SUV factor tag", "bool", True),
            ParamSpec("revert_scaling", "Revert per-slice rescale", "bool", False),
        ),
        needs_3d=True,
        run_mode="layer",
    ),
    GuiToolSpec(
        "dice",
        "Measure",
        "Dice vs reference layer",
        (ParamSpec("reference_layer", "Reference mask/labels", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "jaccard",
        "Measure",
        "Jaccard vs reference layer",
        (ParamSpec("reference_layer", "Reference mask/labels", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "voxel_metrics",
        "Measure",
        "Voxel overlap metrics",
        (ParamSpec("reference_layer", "Reference mask/labels", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "surface_metrics",
        "Measure",
        "Surface distance metrics",
        (ParamSpec("reference_layer", "Reference mask/labels", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
    ),
)

_PIPELINE_GUI: tuple[GuiToolSpec, ...] = tuple(
    GuiToolSpec(
        p.id,
        "Pipelines",
        p.label,
        (_WORKING_DIR,),
        run_mode="pipeline",
        cli_command=p.cli_command,
        description=p.description,
    )
    for p in PIPELINE_TOOLS
)

_TOOLS = _TOOLS + _PIPELINE_GUI


def categories() -> list[str]:
    return list(_CATEGORY_ORDER)


def tools_for_category(category: str) -> list[GuiToolSpec]:
    return [t for t in _TOOLS if t.category == category]


def tool_by_id(tool_id: str) -> GuiToolSpec | None:
    for t in _TOOLS:
        if t.id == tool_id:
            return t
    return None


def operations_for_category(category: str) -> list[str]:
    return [t.label for t in tools_for_category(category)]


def tool_id_from_label(category: str, label: str) -> str | None:
    for t in tools_for_category(category):
        if t.label == label:
            return t.id
    return None


def default_category() -> str:
    return _CATEGORY_ORDER[0]


def default_operation(category: str) -> str:
    ops = operations_for_category(category)
    return ops[0] if ops else ""


def params_for_tool(tool_id: str) -> tuple[ParamSpec, ...]:
    t = tool_by_id(tool_id)
    return t.params if t else ()


def operation_help_text(tool_id: str | None) -> str:
    """Brief description for the selected tool (Tools panel)."""
    from nvitk.gui.tool_descriptions import tool_description_text

    if not tool_id:
        return tool_description_text("", fallback_label="Select a category and operation.")
    spec = tool_by_id(tool_id)
    fallback = spec.label if spec else ""
    if spec and spec.description.strip():
        return spec.description.strip()
    return tool_description_text(tool_id, fallback_label=fallback)
