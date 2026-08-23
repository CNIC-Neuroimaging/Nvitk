"""GUI tool catalog: categories, operations, and parameter schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nvitk.gui.pipeline.catalog import PIPELINE_TOOLS
from nvitk.measure.morpho.anatomy_axes import SPECIES_AUTO, SPECIES_CHOICES
from nvitk.measure.morpho.topology_io import topology_choices as _morpho_topology_choices

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
    """One tool parameter's GUI widget spec: name, label, kind, default, and (for numeric/choice
    kinds) bounds or allowed values."""

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
    """Registry entry for one GUI tool: id, category, label, its parameters, layer/3D requirements,
    and how running it should behave (``run_mode``)."""

    id: str
    category: str
    label: str
    params: tuple[ParamSpec, ...] = ()
    needs_reference_layer: bool = False
    needs_3d: bool = False
    run_mode: RunMode = "layer"
    cli_command: str = ""
    description: str = ""
    #: False only for a tool that has nothing to do with the active layer at all — it reads a
    #: directory, a database, or a results folder, and manages its own output (napari layers, a
    #: dialog). Defaults to True because almost every tool here operates on the active layer's
    #: data; the Tools panel's "no layers loaded" gate consults this before running anything.
    requires_layer: bool = True
    #: The operation is meaningful **per label**, so a multi-label selection runs it once on each
    #: label and recombines the results with the original ids, instead of on their binary union.
    #:
    #: Set this only where per-label and per-union genuinely differ and the per-label reading is the
    #: one a user means. Connected components qualifies: on a union, two touching labels report one
    #: component and a disjoint smaller label is dropped outright. The mask set-ops (``seg_mask_*``)
    #: do not — a union is exactly what they are for — and neither does ``label_cc``, which
    #: *assigns* new ids and would need a documented offset scheme per label rather than a flag.
    multilabel: bool = False


TOOL_IDS_USING_LABEL_PICKER: frozenset[str] = frozenset({
    "siphon_correct",
    "label_cc",
    "remove_small_components",
    "seg_region_grow",
    "seg_blood_flood",
    "seg_split_lr_cc",
    "seg_combine_labels",
    "seg_remove_labels",
    "centerline_detect_junctions",
    "centerline_cut_junctions",
    "centerline_to_polyline",
    "viz_pet_hotspots",
    "viz_flowshow",
    "viz_flow_streamlines",
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
    "Lab",
    "Pipelines",
)

_LABEL_ID = ParamSpec("label_id", "Label id", "int", 1, min=0, max=9999)
_LABEL_IDS = ParamSpec("label_ids", "Label id(s) comma-separated", "str", "1")
_NEW_ID = ParamSpec("new_id", "Output label id", "int", 1, min=0, max=9999)
_OUTPUT_DIR = ParamSpec("output_dir", "Output directory", "str", "")
_WORKING_DIR = ParamSpec("working_dir", "Working directory", "str", "")


def _totalseg_task_choices() -> tuple[str, ...]:
    """Available TotalSegmentator task names, falling back to a small hardcoded set if the class-map
    registry can't be imported."""
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
        "n4_bias",
        "Restoration",
        "N4 bias field correction (ANTs)",
        (
            ParamSpec("reference_layer", "Mask layer (optional)", "layer", ""),
            ParamSpec("shrink_factor", "Shrink factor", "int", 4, min=1, max=16),
            ParamSpec("spline_param", "Spline param (0=ANTs default)", "float", 0.0, min=0),
            ParamSpec("rescale_intensities", "Rescale intensities", "bool", False),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "mri_super_resolution",
        "Restoration",
        "MRI super-resolution (ANTsPyNet)",
        (
            ParamSpec("expansion_factor", "Expansion factor (1,1,2 | 1,1,3 | 1,1,4 | 1,1,6 | 2,2,2 | 2,2,4)", "str", "1,1,2"),
            ParamSpec(
                "sr_feature",
                "Feature backbone",
                "choice",
                "vgg",
                choices=("vgg", "grader"),
            ),
        ),
        needs_3d=True,
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
    GuiToolSpec(
        "hessian_filter",
        "Filters",
        "Hessian (skimage)",
        (
            ParamSpec(
                "hessian_sigmas",
                "Sigmas (comma-separated)",
                "str",
                "1,3,5,7,9",
            ),
            ParamSpec("black_ridges", "Black ridges (else bright)", "bool", False),
            ParamSpec("hessian_alpha", "Alpha", "float", 0.5, min=0.01, max=10.0),
            ParamSpec("hessian_beta", "Beta", "float", 0.5, min=0.01, max=10.0),
            ParamSpec("hessian_gamma", "Gamma", "float", 15.0, min=0.1, max=100.0),
        ),
    ),
    GuiToolSpec(
        "jerman_filter",
        "Filters",
        "Jerman vesselness",
        (
            ParamSpec(
                "jerman_sigmas",
                "Sigmas (comma-separated)",
                "str",
                "1,3,5,7,9",
            ),
            ParamSpec("black_ridges", "Black ridges (else bright)", "bool", False),
            ParamSpec("jerman_tau", "Tau", "float", 0.5, min=0.5, max=1.0),
        ),
    ),
    GuiToolSpec(
        "snakes_filter",
        "Filters",
        "Snakes (active contours)",
        (
            ParamSpec("reference_layer", "Init contour mask layer", "layer", ""),
            ParamSpec("snakes_alpha", "Alpha (tension)", "float", 0.01, min=0.0, max=5.0),
            ParamSpec("snakes_beta", "Beta (rigidity)", "float", 0.1, min=0.0, max=50.0),
            ParamSpec("snakes_w_line", "w_line (intensity)", "float", 0.0, min=-10.0, max=10.0),
            ParamSpec("snakes_w_edge", "w_edge (edges)", "float", 1.0, min=-10.0, max=10.0),
            ParamSpec("snakes_gamma", "Gamma (time step)", "float", 0.01, min=1e-5, max=1.0),
            ParamSpec("snakes_max_iter", "Max iterations", "int", 2500, min=10, max=20000),
            ParamSpec("snakes_sigma", "Gaussian sigma", "float", 1.0, min=0.0, max=20.0),
            ParamSpec("snakes_n_points", "Control points", "int", 400, min=16, max=4000),
            ParamSpec("snakes_axis", "3D slice axis", "int", 0, min=0, max=2),
        ),
        needs_reference_layer=True,
    ),
    GuiToolSpec(
        "img_mask_keep_inside",
        "Filters",
        "Mask: keep inside",

        (
            ParamSpec("reference_layer", "Mask / segmentation layer", "layer", ""),
            ParamSpec("fill_value", "Fill value (outside mask)", "float", 0.0),
            ParamSpec(
                "mask_label_ids",
                "Mask label id(s) (empty = all nonzero)",
                "str",
                "",
            ),
        ),
    ),
    GuiToolSpec(
        "img_mask_keep_outside",
        "Filters",
        "Mask: keep outside",
        (
            ParamSpec("reference_layer", "Mask / segmentation layer", "layer", ""),
            ParamSpec("fill_value", "Fill value (inside mask)", "float", 0.0),
            ParamSpec(
                "mask_label_ids",
                "Mask label id(s) (empty = all nonzero)",
                "str",
                "",
            ),
        ),
    ),
    GuiToolSpec("dilate", "Morphology", "Dilate", _MORPH_PARAMS, multilabel=True),
    GuiToolSpec("erode", "Morphology", "Erode", _MORPH_PARAMS, multilabel=True),
    GuiToolSpec("open", "Morphology", "Open", _MORPH_PARAMS, multilabel=True),
    GuiToolSpec("close", "Morphology", "Close", _MORPH_PARAMS, multilabel=True),
    GuiToolSpec(
        "fill_holes",
        "Morphology",
        "Fill holes",
        (ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),),
        multilabel=True,
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
        multilabel=True,
    ),
    GuiToolSpec(
        "morph_biggest_cc",
        "Morphology",
        "Connected components",
        (
            ParamSpec("n_largest", "Keep N largest", "int", 1, min=1, max=1000),
            ParamSpec("connectivity", "Connectivity", "int", 1, min=1, max=3),
        ),
        needs_3d=True,
        multilabel=True,
    ),
    GuiToolSpec(
        "skeletonize",
        "Morphology",
        "Skeletonize / centerline",
        needs_3d=True,
        multilabel=True,
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
        "centerline_to_polyline",
        "Centerline",
        "To polyline",
        (
            ParamSpec(
                "min_branch_points",
                "Min branch points (0 = keep all)",
                "int",
                0,
                min=0,
                max=5000,
            ),
            ParamSpec("reskeletonize", "Re-skeletonize mask (thick masks only)", "bool", False),
            ParamSpec("edge_width", "Path line width", "float", 0.35, min=0.05, max=5.0),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Convert a complete centerline mask into smoothed Napari path shapes. "
            "Per label: longest main path plus every unique branch edge through "
            "bifurcations (no dropped corridors). Optional min branch points is "
            "the only length prune (0 = keep all)."
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
        multilabel=True,
    ),
    GuiToolSpec(
        "seg_convex_hull_3d",
        "Segmentation",
        "Convex hull (3D)",
        (),
        needs_3d=True,
        multilabel=True,
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
        (
            ParamSpec("reference_layer", "Second mask layer", "layer", ""),
            ParamSpec("reference_label_ids", "Reference label id(s) (comma-separated)", "str", ""),
        ),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_intersection",
        "Segmentation",
        "Mask intersection (AND)",
        (
            ParamSpec("reference_layer", "Second mask layer", "layer", ""),
            ParamSpec("reference_label_ids", "Reference label id(s) (comma-separated)", "str", ""),
        ),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_subtract",
        "Segmentation",
        "Mask subtract (A \\ B)",
        (
            ParamSpec("reference_layer", "Subtract mask B layer", "layer", ""),
            ParamSpec("reference_label_ids", "Reference label id(s) (comma-separated)", "str", ""),
        ),
        needs_reference_layer=True,
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mask_xor",
        "Segmentation",
        "Mask XOR",
        (
            ParamSpec("reference_layer", "Second mask layer", "layer", ""),
            ParamSpec("reference_label_ids", "Reference label id(s) (comma-separated)", "str", ""),
        ),
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
        "Connected components",
        (ParamSpec("n_largest", "Keep N largest", "int", 1, min=1, max=1000),),
        multilabel=True,
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
                choices=(
                    "Custom",
                    "QVTpy default (frac=0.45)",
                    "QVTpy explore (frac=0.25)",
                    "QVTpy ICA test (frac=0.45)",
                ),
            ),
            ParamSpec("reference_layer", "Intensity image layer", "layer", ""),
            ParamSpec("barrier_layer", "Barrier mask layer", "layer", ""),
            ParamSpec("centerline_barrier_layer", "Barrier centerline layer", "layer", ""),
            ParamSpec("barrier_other_labels", "Barrier: other labels on active mask", "bool", False),
            ParamSpec("mask_barrier_dilation_vox", "Mask barrier dilation (vox)", "int", 1, min=0, max=32),
            ParamSpec("centerline_barrier_dilation_vox", "Centerline barrier dilation (vox)", "int", 3, min=0, max=32),
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
        "seg_blood_flood",
        "Segmentation",
        "Blood flood (vessel tree)",
        (
            ParamSpec(
                "blood_flood_mode",
                "Mode",
                "choice",
                "expand",
                choices=("expand", "from_scratch"),
            ),
            ParamSpec(
                "reference_layer",
                "Intensity image (expand mode)",
                "layer",
                "",
            ),
            ParamSpec(
                "mask_layer",
                "ROI / brain mask (from-scratch, optional)",
                "layer",
                "",
            ),
            ParamSpec("barrier_layer", "Barrier mask layer (optional)", "layer", ""),
            ParamSpec("hyst_low_factor", "Hysteresis low factor", "float", 3.0, min=0.1, max=20.0),
            ParamSpec("hyst_high_factor", "Hysteresis high factor", "float", 0.5, min=0.01, max=5.0),
            ParamSpec("thicken_iter", "Thicken iterations", "int", 0, min=0, max=20),
            ParamSpec(
                "thin_vesselness_percentile",
                "Thin vesselness percentile (<0 off)",
                "float",
                55.0,
                min=-1.0,
                max=100.0,
            ),
            ParamSpec(
                "frangi_sigmas",
                "Frangi sigmas (comma-separated)",
                "str",
                "0.5,1.0,1.5,2.0,2.5",
            ),
            ParamSpec("min_cc_voxels", "Min tree CC size (from-scratch)", "int", 5, min=1, max=100000),
            ParamSpec("connectivity", "Connectivity", "int", 3, min=1, max=3),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mouse_brain",
        "Segmentation",
        "Mouse brain (ANTsPyNet)",
        (
            ParamSpec(
                "mouse_brain_mode",
                "Mode",
                "choice",
                "extraction",
                choices=("extraction", "parcellation"),
            ),
            ParamSpec(
                "mouse_modality",
                "Modality (extraction)",
                "choice",
                "t2",
                choices=("t2", "ex5coronal", "ex5sagittal"),
            ),
            ParamSpec(
                "which_parcellation",
                "Parcellation scheme",
                "choice",
                "nick",
                choices=("nick", "tct", "jay"),
            ),
            ParamSpec("do_n4", "N4 bias correction first", "bool", True),
            ParamSpec("fix_spacing", "Auto-fix unit spacing → ~20 mm FOV", "bool", True),
            ParamSpec("binarize", "Binarize extraction (thresh 0.5)", "bool", True),
            ParamSpec("return_isotropic_output", "Isotropic output resampling", "bool", False),
            ParamSpec("reference_layer", "Brain mask layer (parcellation, optional)", "layer", ""),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_brain_extraction",
        "Segmentation",
        "Brain extraction (ANTsPyNet)",
        (
            ParamSpec(
                "brain_modality",
                "Modality",
                "choice",
                "t1",
                choices=(
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
                ),
            ),
            ParamSpec("image2_layer", "Second modality layer (optional)", "layer", ""),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_mra_vessel",
        "Segmentation",
        "MRA-TOF vessel segmentation (ANTsPyNet)",
        (
            ParamSpec("mask_layer", "Brain mask layer (optional; auto if none)", "layer", ""),
            ParamSpec("prediction_batch_size", "Prediction batch size", "int", 2, min=1, max=64),
            ParamSpec("patch_stride_length", "Patch stride length", "int", 32, min=8, max=128),
        ),
        needs_3d=True,
    ),
    GuiToolSpec(
        "seg_dkt",
        "Segmentation",
        "DKT parcellation (ANTsPyNet)",
        (
            ParamSpec("dkt_preprocessing", "Preprocessing", "bool", True),
            ParamSpec("dkt_lobar", "Lobar parcellation", "bool", False),
            ParamSpec("dkt_denoising", "Denoising", "bool", True),
            ParamSpec("dkt_version", "Model version", "int", 0, min=0, max=2),
        ),
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
            ParamSpec("point_size", "Point size", "float", 6.0, min=0.1, max=100.0),
            ParamSpec(
                "cmap",
                "SUV colormap",
                "choice",
                "viridis",
                choices=("viridis", "turbo", "magma", "inferno", "plasma", "cividis", "hot", "coolwarm"),
            ),
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
            ParamSpec("time_index", "Initial cardiac phase", "int", 0, min=0, max=64),
            ParamSpec("max_points", "Max vector glyphs", "int", 4000, min=100, max=100000),
            ParamSpec(
                "length_scale",
                "Max arrow length in voxels (at 95th %ile speed)",
                "float",
                5.0,
                min=0.5,
                max=50.0,
            ),
            ParamSpec("cmap", "Speed colormap", "str", "turbo"),
            ParamSpec(
                "sync_dims",
                "Update vectors when dims slider moves",
                "bool",
                True,
            ),
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "viz_flow_streamlines",
        "Visualization",
        "4DFlow Streamlines",
        (
            ParamSpec("ap_layer", "AP phase layer", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer", "layer", ""),
            ParamSpec(
                "reference_layer",
                "Spatial reference layer (optional)",
                "layer",
                "",
            ),
            ParamSpec("time_index", "Initial cardiac phase", "int", 0, min=0, max=64),
            ParamSpec(
                "trace_mode",
                "Trace mode",
                "choice",
                "streamlines",
                choices=("streamlines", "pathlines"),
            ),
            ParamSpec("n_seeds", "Seed count", "int", 64, min=1, max=10000),
            ParamSpec(
                "max_length",
                "Max length (vox; pathline horizon in seconds if pathlines)",
                "float",
                35.0,
                min=1.0,
                max=500.0,
            ),
            ParamSpec("stream_seed", "Random seed", "int", 42, min=0, max=999999),
            ParamSpec(
                "integration_direction",
                "Integration direction (streamlines)",
                "choice",
                "forward",
                choices=("forward", "backward", "both"),
            ),
            ParamSpec(
                "seed_mode",
                "Seed placement",
                "choice",
                "planar",
                choices=("planar", "volume"),
            ),
            ParamSpec("seed_plane_axis", "Planar seed axis (0/1/2)", "int", 2, min=0, max=2),
            ParamSpec(
                "seed_plane_side",
                "Planar seed side",
                "choice",
                "min",
                choices=("min", "max"),
            ),
            ParamSpec("dt_seconds", "Pathline dt (seconds)", "float", 1.0, min=0.01, max=60.0),
            ParamSpec(
                "color_metric",
                "Color by",
                "choice",
                "speed",
                choices=("speed", "integration_time", "arc_length", "fixed"),
            ),
            ParamSpec(
                "cmap",
                "Colormap",
                "choice",
                "turbo",
                choices=("turbo", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "hsv"),
            ),
            ParamSpec("per_vertex_color", "Per-vertex color gradient", "bool", True),
            ParamSpec("resample_paths", "Resample paths uniformly", "bool", False),
            ParamSpec("resample_spacing_vox", "Resample spacing (vox)", "float", 0.5, min=0.1, max=5.0),
            ParamSpec("edge_width", "Path line width (vox)", "float", 0.25, min=0.05, max=10.0),
            ParamSpec("opacity", "Trace opacity", "float", 0.55, min=0.05, max=1.0),
            ParamSpec(
                "sync_dims",
                "Update traces when dims slider moves",
                "bool",
                True,
            ),
        ),
        needs_reference_layer=True,
        needs_3d=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "viz_vessel_cross_sections",
        "Visualization",
        "Vessel cross-sections",
        (
            ParamSpec("cd_layer", "Complex difference image layer", "layer", ""),
            ParamSpec("ap_layer", "AP phase layer", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer", "layer", ""),
            ParamSpec("segmentation_layer", "Segmentation mask (optional)", "layer", ""),
            ParamSpec("cross_section_radius_vox", "Plane half-size (vox)", "float", 12.0, min=1.0),
            ParamSpec("cross_section_res", "Plane resolution (0=auto)", "int", 0, min=0, max=1024),
            ParamSpec("interpolate_plane", "Interpolate plane sampling", "bool", True),
            ParamSpec("interp_vals", "Samples per voxel (auto res)", "int", 4, min=1, max=16),
            ParamSpec("measure_resegment", "Resegment in cross-section plane", "bool", False),
            ParamSpec(
                "cs_supersampling",
                "Supersample plane (~4×)",
                "bool",
                True,
            ),
            ParamSpec(
                "thr_algorithm",
                "2D threshold method",
                "choice",
                "lsthr",
                choices=("otsu", "lsthr", "lthr"),
            ),
            ParamSpec(
                "centerline_window",
                "Tangent window",
                "choice",
                "5",
                choices=("3", "5"),
            ),
            ParamSpec("show_segmentation_3d", "Show segmentation in 3D", "bool", True),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Active layer = centerline mask; pick CD, phases, and optional segmentation. "
            "Click in 3D for oblique cross-sections; optional supersampling (~4×) for "
            "finer in-plane resegmentation or mask upsampling."
        ),
    ),
    GuiToolSpec(
        "viz_vessel_hemo",
        "Visualization",
        "PITC / PWV hemodynamics",
        (
            ParamSpec("ap_layer", "AP phase layer", "layer", ""),
            ParamSpec("rl_layer", "RL phase layer", "layer", ""),
            ParamSpec("fh_layer", "FH phase layer", "layer", ""),
            ParamSpec("reference_layer", "Angio / CD layer", "layer", ""),
            ParamSpec(
                "heart_rate_json",
                "Cardiac metadata JSON (HeartRate)",
                "str",
                "",
            ),
            ParamSpec(
                "root_region",
                "Root region",
                "choice",
                "All",
                choices=("All", "L_ICA", "R_ICA", "Basilar"),
            ),
            ParamSpec("quality_thresh", "Quality threshold", "float", 2.5, min=0.0, max=4.0),
            ParamSpec(
                "quality_metric",
                "Quality metric",
                "choice",
                "stdv_from_mean",
                choices=("stdv_from_mean", "waveform"),
            ),
            ParamSpec("stride", "Station stride", "int", 1, min=1, max=50),
            ParamSpec(
                "cross_section_radius_vox",
                "Cross-section radius (vox)",
                "float",
                10.0,
                min=1.0,
            ),
            ParamSpec("measure_resegment", "Resegment in-plane", "bool", False),
            ParamSpec(
                "cs_supersampling",
                "Supersample plane (~4×)",
                "bool",
                True,
            ),
            ParamSpec("label_constrain", "Constrain to vessel label", "bool", True),
            ParamSpec(
                "station_point_size",
                "Station point size",
                "float",
                2.5,
                min=0.1,
                max=100.0,
            ),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Active layer = stage-4 multilabel segmentation. Requires AP/RL/FH and "
            "angio/CD layers. Runs PITC and PWV together; switch plots and station "
            "coloring in the diagnostics dock."
        ),
    ),
    GuiToolSpec(
        "viz_tof_morphometrics",
        "Visualization",
        "TOF morphometrics (stage7)",
        (
            ParamSpec(
                "stage7_dir",
                "Stage-7 morphometrics directory",
                "str",
                "",
            ),
            ParamSpec(
                "reference_layer",
                "Reference image layer (affine)",
                "layer",
                "",
            ),
            ParamSpec(
                "color_by",
                "Color samples by",
                "choice",
                "radius",
                choices=("radius", "stenosis", "curvature"),
            ),
            ParamSpec("point_size", "Sample point size", "float", 2.0, min=0.1, max=50.0),
            ParamSpec("edge_width", "Centerline width", "float", 0.35, min=0.05, max=5.0),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Load stage-7 centerline VTPs from qvtpy/stage7_morphometrics/centerlines/. "
            "Paths are drawn as polylines; samples are colored by radius, stenosis, or "
            "curvature. Set stage7_dir to the subject stage7 folder."
        ),
    ),
    GuiToolSpec(
        "export_view_png",
        "Visualization",
        "Export 3D view (PNG)",
        (
            ParamSpec("output_path", "Output PNG path", "str", "view.png"),
            ParamSpec("canvas_only", "Canvas only (no window chrome)", "bool", True),
        ),
        run_mode="notify",
        description=(
            "Save a PNG of the Napari 3D canvas (camera, orientation, colormap) "
            "matching the current viewer."
        ),
    ),
    GuiToolSpec(
        "export_view_gif",
        "Visualization",
        "Export 3D view (GIF, 4D)",
        (
            ParamSpec("output_path", "Output GIF path", "str", "view.gif"),
            ParamSpec("gif_fps", "Frames per second", "float", 8.0, min=0.5, max=60.0),
            ParamSpec("time_axis", "Time axis (-1=auto)", "int", -1, min=-1, max=7),
            ParamSpec("canvas_only", "Canvas only (no window chrome)", "bool", True),
        ),
        run_mode="notify",
        description=(
            "Animate the 3D view over the time/cardiac slider (4D images or synced "
            "4D flow-vector overlays)."
        ),
    ),
    GuiToolSpec(
        "volume_projection",
        "Transform",
        "Volume projection",
        (
            ParamSpec("projection_axis", "Axis (-1=auto Z)", "int", -1, min=-1, max=7),
            ParamSpec(
                "projection_method",
                "Method",
                "choice",
                "max",
                choices=("max", "mean", "median", "min", "std", "sum"),
            ),
        ),
        description=(
            "Collapse a 3D or 4D image along one axis (max, mean, median, min, std, sum)."
        ),
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
        "reg_ants_register",
        "Registration",
        "ANTsPy register (ants.registration)",
        (
            ParamSpec("reference_layer", "Fixed / reference layer", "layer", ""),
            _OUTPUT_DIR,
            ParamSpec("type_of_transform", "type_of_transform", "str", "SyN"),
            ParamSpec("write_composite_transform", "Write composite transform", "bool", False),
            ParamSpec("verbose", "Verbose", "bool", False),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "reg_ants_apply",
        "Registration",
        "ANTsPy apply transforms",
        (
            ParamSpec("reference_layer", "Fixed / reference layer", "layer", ""),
            ParamSpec("transform_paths", "Transforms (comma-separated paths)", "str", ""),
            ParamSpec("out_path", "Output NIfTI path (empty=temp)", "str", ""),
            ParamSpec("interpolator", "Interpolator", "str", "linear"),
            ParamSpec("verbose", "Verbose", "bool", False),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "reg_fireants_register",
        "Registration",
        "FireANTs register (fireantsRegistration)",
        (
            ParamSpec("reference_layer", "Fixed / reference layer", "layer", ""),
            _OUTPUT_DIR,
            ParamSpec("device", "Device (e.g. cuda:0)", "str", "cuda:0"),
            ParamSpec("verbose", "Verbose", "bool", False),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "reg_fireants_apply",
        "Registration",
        "FireANTs apply transforms",
        (
            ParamSpec("reference_layer", "Fixed / reference layer", "layer", ""),
            ParamSpec("transform_paths", "Transforms (comma-separated paths)", "str", ""),
            ParamSpec("out_path", "Output NIfTI path (empty=temp)", "str", ""),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec(
        "orient_volume",
        "Transform",
        "View / reorient orientation",
        (
            ParamSpec("orient_mode", "Action", "choice", "view", choices=("view", "reorient")),
            ParamSpec(
                "target_orientation",
                "Target orientation",
                "choice",
                "RAS",
                choices=("RAS", "LPS", "LAS", "RPS", "RSA", "LPI", "LSA", "RPI", "LIA", "RIA"),
            ),
        ),
        needs_3d=True,
        description=(
            "Show NIfTI axis codes (RAS/LPS/…) or mirror/permute the Napari display "
            "to a target layout (e.g. RAS→LAS flips left/right in the viewer)."
        ),
    ),
    GuiToolSpec(
        "reorient_volume",
        "Transform",
        "Reorient (permute / reference / mouse)",
        (
            ParamSpec(
                "reorient_mode",
                "Mode",
                "choice",
                "mouse",
                choices=("mouse", "reference", "manual"),
            ),
            ParamSpec("reference_layer", "Reference layer (optional)", "layer", ""),
            ParamSpec(
                "target_orientation",
                "Target orientation",
                "choice",
                "LAS",
                choices=("RAS", "LPS", "LAS", "RPS", "RSA", "LPI", "LSA", "RPI", "LIA", "RIA"),
            ),
            ParamSpec("permute_order", "Permute order (e.g. 0,2,1)", "str", "0,1,2"),
            ParamSpec("flip_x", "Flip axis 0 (X)", "bool", False),
            ParamSpec("flip_y", "Flip axis 1 (Y)", "bool", False),
            ParamSpec("flip_z", "Flip axis 2 (Z)", "bool", False),
            ParamSpec(
                "reset_affine",
                "Reset affine to target codes (ignore wrong header)",
                "bool",
                False,
            ),
        ),
        needs_3d=True,
        description=(
            "Physically reorient voxel data: mouse preset (AP on Z → AP on Y, LAS), "
            "match a reference layer's axis codes, or manual permute/flips/target. "
            "Use reset-affine when the header axes do not match anatomy."
        ),
    ),
    GuiToolSpec(
        "rotate_volume",
        "Transform",
        "Rotate volume",
        (
            ParamSpec("angle_degrees", "Angle (degrees, CCW)", "float", 90.0, min=-360, max=360),
            ParamSpec("axis", "Rotate around axis", "int", 2, min=0, max=2),
            ParamSpec("order", "Interpolation order (0=labels)", "int", 1, min=0, max=5),
            ParamSpec("reshape", "Expand canvas to fit", "bool", False),
        ),
        needs_3d=True,
        description=(
            "Rotate the active 3D layer around axis 0/1/2 (default Z). "
            "Use order 0 for label masks. Keep reshape off to preserve shape."
        ),
    ),
    GuiToolSpec(
        "swap_axes",
        "Transform",
        "Swap axes",
        (
            ParamSpec("swap_axis0", "First axis", "int", 0, min=0, max=3),
            ParamSpec("swap_axis1", "Second axis", "int", 1, min=0, max=3),
        ),
        needs_3d=True,
        description=(
            "Exchange two array axes of the active volume (e.g. 0↔2 to turn XYZ into ZYX). "
            "Spacing/affine columns are updated when metadata is present."
        ),
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
            ParamSpec("measure_resegment", "Resegment in-plane", "bool", False),
            ParamSpec("cs_supersampling", "Supersample plane (~4×)", "bool", True),
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
            ParamSpec("measure_resegment", "Resegment in-plane", "bool", False),
            ParamSpec("cs_supersampling", "Supersample plane (~4×)", "bool", True),
        ),
        needs_reference_layer=True,
        run_mode="notify",
    ),
    GuiToolSpec("volume_mm3", "Measure", "Volume (mm³)", (), run_mode="notify"),
    GuiToolSpec("volume_cc", "Measure", "Volume (cc)", (), run_mode="notify"),
    GuiToolSpec(
        "measure_centerline_arc_length",
        "Measure",
        "Centerline arc length (debug)",
        (
            ParamSpec("label_id", "Label id (0 = all / longest)", "int", 0, min=0, max=9999),
            ParamSpec("reskeletonize", "Re-skeletonize mask (thick masks only)", "bool", False),
        ),
        needs_3d=True,
        run_mode="notify",
        description="Report polyline arc length in voxels and mm for debugging.",
    ),
    GuiToolSpec(
        "measure_morphometrics",
        "Measure",
        "Morphometrics",
        (
            ParamSpec("output_dir", "Output directory", "str", ""),
            ParamSpec(
                "topology",
                "Topology JSON",
                "choice",
                "none",
                choices=_morpho_topology_choices(),
            ),
            ParamSpec(
                "species",
                "Species",
                "choice",
                SPECIES_AUTO,
                choices=SPECIES_CHOICES,
            ),
            ParamSpec("n_workers", "Workers", "int", 1, min=1, max=64),
            ParamSpec("skip_existing", "Skip if Excel exists", "bool", False),
            ParamSpec(
                "input_already_smoothed",
                "Input already Taubin-smoothed",
                "bool",
                False,
            ),
        ),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Run vessel-wise TOF morphometrics on the selected multilabel Labels layer. "
            "Choose a topology JSON under measure/morpho/topology "
            "(eicab_topology.json for TOF/eICAB; mouse_root_topology.json for the "
            "Mouse TOF CoW labels; qvtpy_topology.json is the 4D-flow label "
            "reference), or 'none' for topology-agnostic per-label metrics. "
            "Species 'auto' reads the topology's _meta block. Leave Output "
            "directory empty to show results in the GUI only."
        ),
    ),
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
        "intensity_similarity",
        "Measure",
        "Image intensity similarity",
        (ParamSpec("reference_layer", "Second image layer", "layer", ""),),
        needs_reference_layer=True,
        run_mode="notify",
        description=(
            "Compare voxel intensities of the active image vs a second layer "
            "(Pearson, Spearman, MAE, RMSE). Resamples the second image onto the "
            "active grid when affines differ. No mask required."
        ),
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
    GuiToolSpec(
        "measure_voxelwise",
        "Measure",
        "Voxelwise analysis (FSL randomise)",
        (),
        run_mode="notify",
        cli_command="nvitk-voxelwise",
        requires_layer=False,
        description=(
            "Cohort voxelwise GLM with permutation FWE correction. Opens its own window to "
            "configure the image directory, cohort, EVs and contrasts, and run locally or on the "
            "cluster — or to load a finished results folder. Corrected maps land as layers over "
            "the MNI template; the 3-D scene is under Visualization."
        ),
    ),
    GuiToolSpec(
        "viz_voxelwise_3d",
        "Visualization",
        "Voxelwise 3D scene",
        (),
        run_mode="notify",
        requires_layer=False,
        description=(
            "Suprathreshold voxels inside a translucent brain shell, in 3-D. Opens its own window "
            "to pick the results folder, the map (corrected 1-p or the t-statistic), the "
            "contrasts, the threshold window and whether clusters are drawn as iso-surfaces or "
            "coloured points."
        ),
    ),
    GuiToolSpec(
        "lab_mouse_tof_cow",
        "Lab",
        "Mouse TOF CoW",
        (),
        needs_3d=True,
        run_mode="notify",
        description=(
            "Stage 1: N4 → blood flood from-scratch → label CCs on the active TOF volume. "
            "Stage 2: click CCs on the labeled layer; Add CC to tree / Tree done for "
            "Left ICA → Right ICA → Basilar (output labels 1/2/3)."
        ),
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
    """Display order of tool categories."""
    return list(_CATEGORY_ORDER)


def tools_for_category(category: str) -> list[GuiToolSpec]:
    """All registered tool specs belonging to *category*."""
    return [t for t in _TOOLS if t.category == category]


def tool_by_id(tool_id: str) -> GuiToolSpec | None:
    """Look up a registered :class:`GuiToolSpec` by its id, or ``None`` if unregistered."""
    for t in _TOOLS:
        if t.id == tool_id:
            return t
    return None


def operations_for_category(category: str) -> list[str]:
    """Display labels of every tool in *category*, for populating the operation dropdown."""
    return [t.label for t in tools_for_category(category)]


def tool_id_from_label(category: str, label: str) -> str | None:
    """Resolve a category/label combination back to its tool id, or ``None`` if unmatched."""
    for t in tools_for_category(category):
        if t.label == label:
            return t.id
    return None


def default_category() -> str:
    """The category selected by default when the Tools panel first loads."""
    return _CATEGORY_ORDER[0]


def default_operation(category: str) -> str:
    """The operation label selected by default for *category* (its first tool, or ``""`` if empty)."""
    ops = operations_for_category(category)
    return ops[0] if ops else ""


def params_for_tool(tool_id: str) -> tuple[ParamSpec, ...]:
    """Parameter specs registered for *tool_id*, empty if the tool is unknown or has none."""
    t = tool_by_id(tool_id)
    return t.params if t else ()


def operation_help_text(tool_id: str | None) -> str:
    """Brief description for the selected tool (Tools panel)."""
    from nvitk.gui.tools.descriptions import tool_description_text

    if not tool_id:
        return tool_description_text("", fallback_label="Select a category and operation.")
    spec = tool_by_id(tool_id)
    fallback = spec.label if spec else ""
    if spec and spec.description.strip():
        return spec.description.strip()
    return tool_description_text(tool_id, fallback_label=fallback)


# GUI SGE capability gate (layer-output tools only; v1 blocklist).
SGE_BLOCKLIST: frozenset[str] = frozenset({
    "centerline_detect_junctions",
    "centerline_cut_junctions",
    "centerline_to_polyline",
    "measure_generate_suv",
    "measure_centerline_arc_length",
    "measure_morphometrics",
    "measure_loc_hemodynamics",
    "volume_mm3",
    "volume_cc",
    "mean_intensity",
    "integrated_intensity",
    "label_stats",
    "intensity_similarity",
    "dice",
    "jaccard",
    "voxel_metrics",
    "surface_metrics",
    "qvtpy_locs",
    "viz_flowshow",
    "viz_flow_streamlines",
    "viz_pet_hotspots",
    "viz_vessel_hemo",
    "viz_pitc",
    "viz_pwv",
    "viz_tof_morphometrics",
    "seg_totalsegmentator",
    "seg_eicab",
    "orient_volume",
    "reorient_volume",
    "seg_mouse_brain",
    "seg_brain_extraction",
    "seg_mra_vessel",
    "seg_dkt",
    "mri_super_resolution",
    "seg_split_lr_cc",
    "seg_split_lr_midline",
    "seg_adjust_masks",
    "lab_mouse_tof_cow",
    "measure_voxelwise",
    "viz_voxelwise_3d",
    "reg_flirt_rigid",
    "reg_flirt_apply",
    "siphon_correct",
    "pet_ureter_seg",
})

SGE_BLOCKLIST_PREFIXES: tuple[str, ...] = ("qvtpy_stage",)


def is_sge_capable(tool_id: str | None) -> bool:
    """Return True when *tool_id* can run through :func:`run_gui_tool_headless`."""
    if not tool_id:
        return False
    tid = str(tool_id).strip()
    if tid in SGE_BLOCKLIST:
        return False
    if any(tid.startswith(p) for p in SGE_BLOCKLIST_PREFIXES):
        return False
    spec = tool_by_id(tid)
    if spec is None:
        return False
    if spec.run_mode != "layer":
        return False
    return True


def sge_block_reason(tool_id: str | None) -> str:
    """Human-readable reason *tool_id* can't run via remote SGE, or ``""`` if it can."""
    if is_sge_capable(tool_id):
        return ""
    if not tool_id:
        return "No tool selected."
    spec = tool_by_id(str(tool_id))
    if spec is None:
        return f"Unknown tool {tool_id!r}."
    if spec.run_mode != "layer":
        return f"{spec.label} uses run mode {spec.run_mode!r} (not supported on SGE yet)."
    if str(tool_id) in SGE_BLOCKLIST:
        return f"{spec.label} requires Napari or a dedicated cluster CLI."
    return f"{spec.label} is not supported for remote SGE in this version."
