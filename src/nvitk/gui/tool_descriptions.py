"""Short help text for GUI tools (shown under Operation in the Tools panel)."""

from __future__ import annotations

TOOL_DESCRIPTIONS: dict[str, str] = {
    "bilateral": "Edge-preserving Gaussian smoothing on the active image or mask.",
    "sliding_threshold": "Adaptive threshold along one axis (useful for uneven intensity).",
    "dilate": "Expand foreground voxels by a spherical footprint.",
    "erode": "Shrink foreground voxels by a spherical footprint.",
    "open": "Erosion followed by dilation (removes small protrusions).",
    "close": "Dilation followed by erosion (fills small holes).",
    "fill_holes": "Fill enclosed background regions in a binary mask.",
    "label_cc": "Label connected components in a mask.",
    "remove_small_components": "Remove connected components below a minimum size.",
    "morph_biggest_cc": "Keep only the largest connected component.",
    "skeletonize": "Reduce a mask to a 1-voxel-thick skeleton / centerline.",
    "centerline_detect_junctions": (
        "Mark skeleton branch points (degree ≥ N) on a 3D centerline mask."
    ),
    "centerline_cut_junctions": (
        "Split a label at junction markers from Detect skeleton junctions."
    ),
    "siphon_correct": "Correct ICA siphon centerlines using a TOF/MRA reference.",
    "mask_genus": "Report topological genus of a mask (handles / tunnels).",
    "seg_get_label": "Extract one label id into a binary mask layer.",
    "seg_combine_labels": "Merge selected labels into one output label.",
    "seg_remove_labels": "Zero out selected label ids in a label map.",
    "seg_pet_ureter": "PET-guided ureter segmentation from organ/body masks.",
    "seg_convex_hull_slice": "2D convex hull of a mask on one slice axis.",
    "seg_convex_hull_3d": "3D convex hull of a label volume.",
    "seg_distance_transform": "Euclidean distance transform inside/outside a mask.",
    "seg_mask_union": "Voxel-wise OR of two masks.",
    "seg_mask_intersection": "Voxel-wise AND of two masks.",
    "seg_mask_subtract": "Subtract one mask from another.",
    "seg_mask_xor": "Voxel-wise XOR of two masks.",
    "seg_mask_complement": "Invert a binary mask.",
    "seg_biggest_cc": "Keep largest connected component per label.",
    "seg_split_lr_cc": "Split mask into left/right by connected components.",
    "seg_split_lr_midline": "Split mask at a sagittal midline plane.",
    "seg_region_grow": (
        "Grow from seed voxels. Barriers block other label ids only; "
        "mask and centerline layers use separate dilation radii."
    ),
    "measure_centerline_arc_length": (
        "Report centerline polyline arc length in voxels and mm (debug)."
    ),
    "seg_adjust_masks": "Morphological adjust label masks (open/close per label).",
    "seg_totalsegmentator": "Run TotalSegmentator on the active image.",
    "seg_eicab": "Run EICAB segmentation (cluster or local).",
    "viz_pet_hotspots": "Highlight high-SUV voxels as Napari points.",
    "viz_flowshow": "Overlay 4D flow magnitude / vectors in Napari.",
    "reg_flirt_rigid": "Rigid FLIRT registration to a reference volume.",
    "reg_flirt_apply": "Apply a saved FLIRT transform to a volume.",
    "isotropy": "Resample to near-isotropic voxel spacing.",
    "resample_to": "Resample to a target voxel size.",
    "oblique_slice": "Extract an oblique 2D slice through the volume.",
    "qvtpy_locs": "Load or generate QVTpy LOC CSV and show as points.",
    "measure_loc_hemodynamics": "Hemodynamics along LOC centerlines (QVTpy-style).",
    "measure_mask_hemodynamics": "Mean flow statistics inside a vessel mask.",
    "volume_mm3": "Report label volume in mm³.",
    "volume_cc": "Report label volume in cc.",
    "masked_stats": "Intensity statistics inside a mask.",
    "integrated_intensity": "Sum of intensities inside a mask.",
    "suv_stats": "SUV statistics for a region on a PET layer.",
    "measure_generate_suv": "Build an SUV image from PET metadata.",
    "dice": "Dice overlap vs a reference mask.",
    "jaccard": "Jaccard index vs a reference mask.",
    "voxel_metrics": "Voxel-wise overlap metrics vs reference.",
    "surface_metrics": "Surface distance metrics vs reference.",
}


def tool_description_text(tool_id: str, *, fallback_label: str = "") -> str:
    if not tool_id:
        return "Select a category and operation."
    text = TOOL_DESCRIPTIONS.get(tool_id, "").strip()
    if text:
        return text
    if fallback_label:
        return fallback_label
    return ""
