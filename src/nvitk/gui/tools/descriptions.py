"""Short help text for GUI tools (shown under Operation in the Tools panel)."""

from __future__ import annotations

TOOL_DESCRIPTIONS: dict[str, str] = {
    "bilateral": "Edge-preserving Gaussian smoothing on the active image or mask.",
    "n4_bias": (
        "ANTs N4 bias-field correction on the active intensity volume. "
        "Optional mask layer restricts the bias estimate."
    ),
    "mri_super_resolution": (
        "ANTsPyNet MRI super-resolution on the active volume. "
        "Use integer expansion factors only (default 1,1,2 for thick-slice). "
        "Supported: 1,1,2 | 1,1,3 | 1,1,4 | 1,1,6 | 2,2,2 | 2,2,4; feature vgg or grader."
    ),
    "sliding_threshold": "Adaptive threshold along one axis (useful for uneven intensity).",
    "dilate": "Expand foreground voxels by a spherical footprint.",
    "erode": "Shrink foreground voxels by a spherical footprint.",
    "open": "Erosion followed by dilation (removes small protrusions).",
    "close": "Dilation followed by erosion (fills small holes).",
    "fill_holes": "Fill enclosed background regions in a binary mask.",
    "label_cc": "Label connected components in a mask.",
    "remove_small_components": "Remove connected components below a minimum size.",
    "morph_biggest_cc": "Keep only the largest connected component.",
    "skeletonize": (
        "Reduce a mask to a 1-voxel-thick skeleton / centerline; "
        "multilabel masks keep each label id on its skeleton voxels."
    ),
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
    "seg_blood_flood": (
        "Active layer = marker / seed labels; intensity layer = CD or TOF. "
        "Frangi vesselness → hysteresis tree → watershed (same algorithm as qvtpy "
        "distal vessel expansion). Optional barrier punches hard walls out of the tree."
    ),
    "seg_mouse_brain": (
        "ANTsPyNet mouse brain extraction or parcellation (T2w mouse MRI). "
        "No T1 model — use modality t2 (or ex5* for histology). "
        "If voxel spacing is ~1 mm (unit header), auto-rescales FOV to ~20 mm "
        "(mouse template size). N4 preprocess on by default."
    ),
    "seg_brain_extraction": (
        "ANTsPyNet multi-modal brain extraction on the active MRI volume. "
        "For modalities like t1t2infant, set a second modality layer."
    ),
    "seg_mra_vessel": (
        "ANTsPyNet MRA-TOF vessel segmentation (probability map) on the active "
        "volume. Leave mask at (none) to auto-extract a brain mask, or supply a "
        "binary brain mask layer (any positive foreground is binarized)."
    ),
    "seg_dkt": (
        "ANTsPyNet Desikan-Killiany-Tourville cortical parcellation on a T1w volume."
    ),
    "measure_centerline_arc_length": (
        "Report centerline polyline arc length in voxels and mm (debug)."
    ),
    "seg_adjust_masks": "Morphological adjust label masks (open/close per label).",
    "seg_totalsegmentator": "Run TotalSegmentator on the active image.",
    "seg_eicab": "Run EICAB segmentation (cluster or local).",
    "viz_pet_hotspots": "Highlight high-SUV voxels as Napari points.",
    "viz_flowshow": (
        "4D flow velocity vectors in Napari: all phases precomputed, arrow length and "
        "color from speed (mm/s). Syncs to the cardiac-phase slider; optional auto-play."
    ),
    "viz_flow_streamlines": (
        "4D flow streamlines or pathlines in Napari. Active layer = vessel segmentation mask. "
        "Streamlines: instantaneous flow at each cardiac phase. Pathlines: particle tracks "
        "integrated forward through time from the selected phase. Color by speed, arc length, "
        "integration time, or fixed; optional per-vertex gradient. Planar inlet seeding by default."
    ),
    "viz_vessel_cross_sections": (
        "Active layer = centerline mask; choose CD, AP/RL/FH phases, and optional segmentation. "
        "Per-label paths from the mask; click in 3D for oblique cross-sections with "
        "selected and ±2 neighboring flow waveforms when phase layers are provided. "
        "Enable 'Supersample plane (~4×)' for a finer in-plane grid (resegment on the "
        "supersampled plane, or upsample the stage-4 mask when not resegmenting). "
        "Toggle 'Pick cross-section on click' in the dock to rotate/pan the 3D view freely."
    ),
    "viz_pitc": (
        "Deprecated alias — use viz_vessel_hemo (PITC / PWV hemodynamics)."
    ),
    "viz_pwv": (
        "Deprecated alias — use viz_vessel_hemo (PITC / PWV hemodynamics)."
    ),
    "viz_vessel_hemo": (
        "Active layer = stage-4 multilabel segmentation. Requires AP/RL/FH and angio/CD "
        "layers. Computes PITC and PWV together (same defaults as qvtpy stage 6: "
        "supersampled plane, no in-plane resegmentation). After "
        "run, use the diagnostics dock to switch plots (PITC / PWV / Bjornfoot), color "
        "stations by feature, choose colormap / contrast caps, and toggle the legend."
    ),
    "viz_tof_morphometrics": (
        "Load qvtpy stage-7 morphometrics centerline VTPs for debugging. Set stage7_dir "
        "to <subject>/qvtpy/stage7_morphometrics. Color samples by radius, stenosis, or "
        "curvature; a left dock shows Path Summary scalars when available."
    ),
    "export_view_png": (
        "Save the Napari 3D render window as PNG (same camera and orientation as on screen)."
    ),
    "export_view_gif": (
        "Export an animated GIF: one 3D screenshot per cardiac phase (4D image or "
        "4D flow vectors overlay)."
    ),
    "volume_projection": (
        "Maximum / mean / median (etc.) intensity projection along a chosen axis."
    ),
    "reg_flirt_rigid": "Rigid FLIRT registration to a reference volume.",
    "reg_flirt_apply": "Apply a saved FLIRT transform to a volume.",
    "isotropy": "Resample to near-isotropic voxel spacing.",
    "resample_to": "Resample to a target voxel size.",
    "rotate_volume": (
        "Rotate the active volume around axis 0/1/2 (default Z = 2). "
        "Angle is counter-clockwise in the plane orthogonal to that axis. "
        "Use interpolation order 0 for label masks."
    ),
    "orient_volume": (
        "Show NIfTI axis codes or mirror/permute the Napari display to a target layout."
    ),
    "layer_metadata": (
        "Print spacing, FOV, origin, orientation, direction, and affine for the active layer."
    ),
    "reorient_volume": (
        "Reorient the active volume for mouse/ANTs layouts. "
        "Modes: mouse (permute 0,2,1 + LAS), reference (match another layer), "
        "or manual permute/flips/target codes. Enable reset-affine when the "
        "NIfTI header axes disagree with anatomy (e.g. AP stored on Z)."
    ),
    "swap_axes": (
        "Swap two array axes of the active volume (e.g. 0 and 2). "
        "Updates spacing/affine metadata when available."
    ),
    "oblique_slice": "Extract an oblique 2D slice through the volume.",
    "qvtpy_locs": "Load or generate QVTpy LOC CSV and show as points.",
    "measure_loc_hemodynamics": "Hemodynamics along LOC centerlines (QVTpy-style).",
    "measure_mask_hemodynamics": "Mean flow statistics inside a vessel mask.",
    "volume_mm3": "Report label volume in mm³.",
    "volume_cc": "Report label volume in cc.",
    "masked_stats": "Intensity statistics inside a mask.",
    "integrated_intensity": "Sum of intensities inside a mask.",
    "suv_stats": "SUV statistics for a region on a PET layer.",
    "intensity_similarity": (
        "Compare intensities of the active image vs another layer (Pearson, Spearman, MAE, RMSE)."
    ),
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
