"""Where every tool lives: task-first sections, subsections, and tags to filter by.

The registry's ``category`` grew with the code (18 flat lists, external suites as categories of
their own, mask operations under Segmentation). This module places each tool in a *task*
section (Segmentation, Masks & labels, Registration...), a subsection inside it, and tags it
with the imaging modalities it applies to, its domain (neuro, vascular, whole-body), species,
and the suite that does the work (FSL, FreeSurfer, FastSurfer, ANTs, ANTsPyNet, VMTK,
TotalSegmentator, eICAB, TopBrain...; ``nvitk`` for nvitk's own). The tool browser groups by
section, filters by the tags, and badges the suite, so an FSL tool sits next to its nvitk
equivalent instead of in an "FSL" drawer.

A tool may declare its own :class:`ToolMeta` (``GuiToolSpec.meta``); otherwise :data:`TABLE`
places it, and :func:`meta_for` falls back to the old category for anything new and unplaced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A task section of the tool browser."""

    key: str
    title: str
    icon: str
    hue: str
    description: str = ""


SECTIONS: tuple[Section, ...] = (
    Section("data", "Data & conversion", "convert", "grey", "Load subjects, convert and import."),
    Section("preprocess", "Preprocess & restoration", "restore_image", "cyan", "Bias field, denoising, distortion, cropping, anonymisation."),
    Section("filters", "Filters & enhancement", "sparkle", "blue", "Smoothing, edges, thresholds, vesselness."),
    Section("segmentation", "Segmentation", "wand", "purple", "Brain, vessels, whole-body and preclinical segmentation."),
    Section("masks", "Masks & labels", "masks", "green", "Extract, combine and edit masks and label maps."),
    Section("morphology", "Morphology", "blob", "olive", "Dilate, erode, fill, hulls, distance, topology."),
    Section("registration", "Registration & templates", "align", "orange", "Linear and deformable registration, transforms."),
    Section("transform", "Transform & resample", "move", "brown", "Orientation, rotation, resampling, projections."),
    Section("vessels", "Vessels & centerlines", "vessel", "red", "Centerlines, junctions, siphon correction."),
    Section("time", "Time, 4D flow & k-space", "wave", "cyan", "3D+t frames and curves, k-space."),
    Section("diffusion", "Diffusion & functional", "layers", "pink", "DWI preprocessing, tensors, tractography; fMRI ICA and timing."),
    Section("perfusion", "Perfusion (ASL)", "droplet", "pink", "Arterial spin labelling quantification."),
    Section("surfaces", "Surfaces & topology", "surface", "orange", "Cortical surfaces, topology correction."),
    Section("measure", "Measure & quantify", "ruler", "red", "Volumes, intensities, hemodynamics, overlaps, group statistics."),
    Section("visualization", "Visualization", "eye", "blue", "Views, CPR, cross-sections, flow, exports."),
    Section("pipelines", "Pipelines", "pipeline", "grey", "Multi-stage cohort pipelines."),
    Section("lab", "Preclinical (lab)", "cube", "brown", "Mouse and lab-specific workflows."),
    Section("utilities", "Utilities & installers", "tools", "grey", "Calculators, raw commands, installers."),
)
SECTIONS_BY_KEY: dict[str, Section] = {s.key: s for s in SECTIONS}

#: Modality tags -> chip label and accent.
MODALITIES: dict[str, tuple[str, str]] = {
    "mra": ("MRA / TOF", "red"),
    "4dflow": ("4D flow", "pink"),
    "smri": ("Structural MRI", "blue"),
    "dwi": ("Diffusion", "purple"),
    "fmri": ("fMRI", "cyan"),
    "asl": ("ASL", "olive"),
    "ct": ("CT / CTA", "orange"),
    "pet": ("PET", "brown"),
    "dixon": ("Dixon", "green"),
    "eye": ("Ophthalmic", "grey"),
}
DOMAINS: dict[str, tuple[str, str]] = {
    "neuro": ("Neuro", "purple"),
    "vascular": ("Vascular / cardiac", "red"),
    "whole-body": ("Whole-body", "orange"),
}
SPECIES: dict[str, tuple[str, str]] = {"human": ("Human", "blue"), "mouse": ("Mouse", "brown")}


@dataclass(frozen=True)
class ToolMeta:
    """Placement and tags of one tool."""

    section: str
    subsection: str = ""
    modalities: tuple[str, ...] = ()
    domain: str = "general"
    species: tuple[str, ...] = ("human",)
    suite: str = "nvitk"


def T(section: str, sub: str = "", mods: str = "", domain: str = "general", species: str = "human", suite: str = "nvitk") -> ToolMeta:
    """Compact constructor for :data:`TABLE` (space-separated modalities / species)."""
    return ToolMeta(section, sub, tuple(mods.split()), domain, tuple(species.split()), suite)


TABLE: dict[str, ToolMeta] = {
    # Preprocess & restoration
    "bilateral": T("preprocess", "Denoising"),
    "n4_bias": T("preprocess", "Bias field", "smri", suite="ANTs"),
    "mri_super_resolution": T("preprocess", "Super-resolution", "smri", "neuro", suite="ANTsPyNet"),
    "recon_background_correction": T("preprocess", "4D flow phase correction", "4dflow", "vascular"),
    "fsl_robustfov": T("preprocess", "Cropping", "smri", "neuro", suite="FSL"),
    "fsl_deface": T("preprocess", "Anonymisation", "smri", "neuro", suite="FSL"),
    "fsl_lesion_filling": T("preprocess", "Lesions", "smri", "neuro", suite="FSL"),
    "fsl_fugue": T("preprocess", "Distortion", "fmri dwi", "neuro", suite="FSL"),
    "fsl_prepare_fieldmap": T("preprocess", "Distortion", "fmri dwi", "neuro", suite="FSL"),
    # Filters & enhancement
    "sliding_threshold": T("filters", "Thresholds"),
    "sk_threshold_global": T("filters", "Thresholds"),
    "sk_threshold_multiotsu": T("filters", "Thresholds"),
    "sk_threshold_local": T("filters", "Thresholds"),
    "sk_threshold_hysteresis": T("filters", "Thresholds"),
    "hessian_filter": T("filters", "Vesselness", "mra ct", "vascular"),
    "jerman_filter": T("filters", "Vesselness", "mra ct", "vascular"),
    "sk_ridge": T("filters", "Vesselness", "mra ct", "vascular"),
    "vmtk_vessel_enhancement": T("filters", "Vesselness", "mra ct", "vascular", suite="VMTK"),
    "snakes_filter": T("filters", "Contours"),
    "sk_edge": T("filters", "Edges"),
    "sk_laplace": T("filters", "Edges"),
    "sk_blur": T("filters", "Smoothing & sharpening"),
    "sk_median": T("filters", "Smoothing & sharpening"),
    "sk_unsharp": T("filters", "Smoothing & sharpening"),
    "sk_butterworth": T("filters", "Smoothing & sharpening"),
    "sk_rank": T("filters", "Smoothing & sharpening"),
    "sk_gabor": T("filters", "Texture"),
    "fsl_susan": T("filters", "Smoothing & sharpening", "smri fmri", suite="FSL"),
    # Segmentation
    "seg_brain_extraction": T("segmentation", "Brain", "smri mra", "neuro", suite="ANTsPyNet"),
    "fsl_bet": T("segmentation", "Brain", "smri", "neuro", suite="FSL"),
    "fsl_fast": T("segmentation", "Tissue", "smri", "neuro", suite="FSL"),
    "fsl_first": T("segmentation", "Subcortical", "smri", "neuro", suite="FSL"),
    "seg_dkt": T("segmentation", "Parcellation", "smri", "neuro", suite="ANTsPyNet"),
    "fastsurfer_seg": T("segmentation", "Parcellation", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_cereb": T("segmentation", "Subcortical", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_hypothal": T("segmentation", "Subcortical", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_cc": T("segmentation", "Subcortical", "smri", "neuro", suite="FastSurfer"),
    "seg_mra_vessel": T("segmentation", "Vessels", "mra", "vascular", suite="ANTsPyNet"),
    "seg_topbrain": T("segmentation", "Vessels", "mra ct", "vascular", suite="TopBrain"),
    "seg_eicab": T("segmentation", "Vessels", "mra", "vascular", suite="eICAB"),
    "seg_eicab_postprocess": T("segmentation", "Vessels", "mra", "vascular", suite="eICAB"),
    "seg_blood_flood": T("segmentation", "Vessels", "mra ct", "vascular"),
    "seg_totalsegmentator": T("segmentation", "Whole-body", "ct smri", "whole-body", suite="TotalSegmentator"),
    "seg_pet_ureter": T("segmentation", "Whole-body", "pet", "whole-body"),
    "seg_region_grow": T("segmentation", "Interactive"),
    "seg_mouse_brain": T("segmentation", "Mouse", "smri", "neuro", "mouse", suite="ANTsPyNet"),
    # Masks & labels
    "seg_get_label": T("masks", "Labels"),
    "seg_combine_labels": T("masks", "Labels"),
    "seg_remove_labels": T("masks", "Labels"),
    "fastsurfer_reduce_aseg": T("masks", "Labels", "smri", "neuro", suite="FastSurfer"),
    "seg_mask_image": T("masks", "Masking"),
    "seg_mask_union": T("masks", "Boolean"),
    "seg_mask_intersection": T("masks", "Boolean"),
    "seg_mask_subtract": T("masks", "Boolean"),
    "seg_mask_xor": T("masks", "Boolean"),
    "seg_mask_complement": T("masks", "Boolean"),
    "label_cc": T("masks", "Components"),
    "remove_small_components": T("masks", "Components"),
    "morph_biggest_cc": T("masks", "Components"),
    "seg_biggest_cc": T("masks", "Components"),
    "seg_split_lr_cc": T("masks", "Split"),
    "seg_split_lr_midline": T("masks", "Split"),
    "seg_adjust_masks": T("masks", "Editing"),
    "interp_mask_slices": T("masks", "Editing"),
    # Morphology
    "dilate": T("morphology", "Basic"),
    "erode": T("morphology", "Basic"),
    "open": T("morphology", "Basic"),
    "close": T("morphology", "Basic"),
    "fill_holes": T("morphology", "Basic"),
    "seg_convex_hull_slice": T("morphology", "Shape"),
    "seg_convex_hull_3d": T("morphology", "Shape"),
    "seg_distance_transform": T("morphology", "Distance"),
    "mask_genus": T("morphology", "Topology"),
    # Registration & templates
    "reg_flirt_rigid": T("registration", "Linear", suite="FSL"),
    "reg_flirt_apply": T("registration", "Apply transforms", suite="FSL"),
    "reg_ants_register": T("registration", "Deformable", suite="ANTs"),
    "reg_ants_apply": T("registration", "Apply transforms", suite="ANTs"),
    "reg_fireants_register": T("registration", "Deformable", suite="FireANTs"),
    "reg_fireants_apply": T("registration", "Apply transforms", suite="FireANTs"),
    "fsl_fnirt": T("registration", "Deformable", "smri", "neuro", suite="FSL"),
    "fsl_applywarp": T("registration", "Apply transforms", suite="FSL"),
    "fsl_invwarp": T("registration", "Transforms", suite="FSL"),
    "fsl_convert_xfm": T("registration", "Transforms", suite="FSL"),
    "fsl_epi_reg": T("registration", "Linear", "fmri dwi", "neuro", suite="FSL"),
    # Transform & resample
    "orient_volume": T("transform", "Orientation"),
    "reorient_volume": T("transform", "Orientation"),
    "rotate_volume": T("transform", "Orientation"),
    "swap_axes": T("transform", "Orientation"),
    "oblique_slice": T("transform", "Orientation"),
    "fsl_reorient2std": T("transform", "Orientation", "smri", "neuro", suite="FSL"),
    "isotropy": T("transform", "Resampling"),
    "resample_to": T("transform", "Resampling"),
    "interp_resample_axes": T("transform", "Resampling"),
    "interp_block_reduce": T("transform", "Resampling"),
    "interp_fill_slices": T("transform", "Resampling"),
    "fastsurfer_conform": T("transform", "Resampling", "smri", "neuro", suite="FastSurfer"),
    "volume_projection": T("transform", "Projection"),
    # Vessels & centerlines
    "skeletonize": T("vessels", "Centerlines", "mra ct", "vascular"),
    "centerline_detect_junctions": T("vessels", "Centerlines", "mra ct", "vascular"),
    "centerline_to_polyline": T("vessels", "Centerlines", "mra ct", "vascular"),
    "polyline_to_centerline": T("vessels", "Centerlines", "mra ct", "vascular"),
    "vmtk_centerline_network": T("vessels", "Centerlines", "mra ct", "vascular", suite="VMTK"),
    "centerline_center": T("vessels", "Centerlines", "mra ct", "vascular"),
    "centerline_cut_junctions": T("vessels", "Centerlines", "mra ct", "vascular"),
    "measure_centerline_arc_length": T("vessels", "Centerlines", "mra ct", "vascular"),
    "siphon_correct": T("vessels", "Correction", "mra", "vascular"),
    # Time, 4D flow & k-space
    "time_extract_frame": T("time", "3D+t"),
    "time_projection": T("time", "3D+t"),
    "time_curve": T("time", "3D+t"),
    "time_interpolate_frames": T("time", "3D+t"),
    "time_stack_layers": T("time", "3D+t"),
    "kspace_fft": T("time", "K-space"),
    "kspace_ifft": T("time", "K-space"),
    "kspace_filter": T("time", "K-space"),
    # Diffusion & functional
    "fsl_mcflirt": T("diffusion", "fMRI", "fmri", "neuro", suite="FSL"),
    "fsl_slicetimer": T("diffusion", "fMRI", "fmri", "neuro", suite="FSL"),
    "fsl_melodic": T("diffusion", "fMRI", "fmri", "neuro", suite="FSL"),
    "fsl_topup": T("diffusion", "DWI preprocessing", "dwi", "neuro", suite="FSL"),
    "fsl_eddy": T("diffusion", "DWI preprocessing", "dwi", "neuro", suite="FSL"),
    "fsl_dtifit": T("diffusion", "Tensors & tracts", "dwi", "neuro", suite="FSL"),
    "fsl_bedpostx": T("diffusion", "Tensors & tracts", "dwi", "neuro", suite="FSL"),
    "fsl_probtrackx2": T("diffusion", "Tensors & tracts", "dwi", "neuro", suite="FSL"),
    "fsl_xtract": T("diffusion", "Tensors & tracts", "dwi", "neuro", suite="FSL"),
    "fsl_tbss": T("diffusion", "Group analysis", "dwi", "neuro", suite="FSL"),
    # Perfusion
    "fsl_asl_oxford": T("perfusion", "Quantification", "asl", "neuro", suite="FSL"),
    "fsl_asl_basil": T("perfusion", "Quantification", "asl", "neuro", suite="FSL"),
    "fsl_asl_file": T("perfusion", "Preparation", "asl", "neuro", suite="FSL"),
    "fsl_asl_calib": T("perfusion", "Calibration", "asl", "neuro", suite="FSL"),
    # Surfaces & topology
    "fastsurfer_surf": T("surfaces", "Cortical surfaces", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_full": T("surfaces", "Cortical surfaces", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_long": T("surfaces", "Cortical surfaces", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_spectral_sphere": T("surfaces", "Topology", "smri", "neuro", suite="FastSurfer"),
    "fastsurfer_topology_chain": T("surfaces", "Topology", "smri", "neuro", suite="FastSurfer"),
    "freesurfer_pretess": T("surfaces", "Topology", "smri", "neuro", suite="FreeSurfer"),
    "freesurfer_mc": T("surfaces", "Meshing", suite="FreeSurfer"),
    "freesurfer_main_component": T("surfaces", "Meshing", suite="FreeSurfer"),
    "freesurfer_euler": T("surfaces", "Topology", suite="FreeSurfer"),
    "freesurfer_smooth": T("surfaces", "Meshing", suite="FreeSurfer"),
    "freesurfer_inflate": T("surfaces", "Meshing", suite="FreeSurfer"),
    "freesurfer_qsphere": T("surfaces", "Topology", suite="FreeSurfer"),
    "freesurfer_fix_topology": T("surfaces", "Topology", suite="FreeSurfer"),
    "freesurfer_topo_fixer": T("surfaces", "Topology", suite="FreeSurfer"),
    # Measure & quantify
    "volume_mm3": T("measure", "Volume"),
    "volume_cc": T("measure", "Volume"),
    "fsl_siena": T("measure", "Atrophy", "smri", "neuro", suite="FSL"),
    "fsl_sienax": T("measure", "Atrophy", "smri", "neuro", suite="FSL"),
    "fastsurfer_segstats": T("measure", "Regional statistics", "smri", "neuro", suite="FastSurfer"),
    "masked_stats": T("measure", "Intensity"),
    "integrated_intensity": T("measure", "Intensity"),
    "fsl_stats": T("measure", "Intensity", suite="FSL"),
    "fsl_meants": T("measure", "Intensity", "fmri", suite="FSL"),
    "suv_stats": T("measure", "PET", "pet", "whole-body"),
    "measure_generate_suv": T("measure", "PET", "pet", "whole-body"),
    "qvtpy_locs": T("measure", "Hemodynamics", "4dflow", "vascular", suite="QVTpy"),
    "measure_loc_hemodynamics": T("measure", "Hemodynamics", "4dflow", "vascular", suite="QVTpy"),
    "measure_mask_hemodynamics": T("measure", "Hemodynamics", "4dflow", "vascular"),
    "measure_morphometrics": T("measure", "Morphometry", "mra ct", "vascular"),
    "intensity_similarity": T("measure", "Similarity"),
    "dice": T("measure", "Overlap & distance"),
    "volsim": T("measure", "Overlap & distance"),
    "mcc": T("measure", "Overlap & distance"),
    "jaccard": T("measure", "Overlap & distance"),
    "voxel_metrics": T("measure", "Overlap & distance"),
    "surface_metrics": T("measure", "Overlap & distance"),
    "measure_voxelwise": T("measure", "Group statistics", "smri asl", "neuro", suite="FSL"),
    "fsl_cluster": T("measure", "Group statistics", "fmri", "neuro", suite="FSL"),
    "fsl_smoothest": T("measure", "Group statistics", "fmri", "neuro", suite="FSL"),
    # Visualization
    "viz_ortho_views": T("visualization", "Views"),
    "export_view_png": T("visualization", "Export"),
    "export_view_gif": T("visualization", "Export"),
    "viz_vessel_cpr": T("visualization", "Vessels", "mra ct", "vascular"),
    "viz_cross_sections_generic": T("visualization", "Vessels"),
    "viz_vessel_cross_sections": T("visualization", "Vessels", "4dflow mra", "vascular"),
    "viz_vessel_hemo": T("visualization", "4D flow", "4dflow", "vascular", suite="QVTpy"),
    "viz_flowshow": T("visualization", "4D flow", "4dflow", "vascular"),
    "viz_flow_traces": T("visualization", "4D flow", "4dflow", "vascular", suite="VMTK"),
    "viz_tof_morphometrics": T("visualization", "Vessels", "mra", "vascular"),
    "viz_pet_hotspots": T("visualization", "PET", "pet", "whole-body"),
    "viz_voxelwise_3d": T("visualization", "Neuro", "smri asl", "neuro"),
    "viz_brain_vascular_map": T("visualization", "Neuro", "smri mra", "neuro"),
    # Pipelines
    "pipeline_pesa_fat_ctpet": T("pipelines", "Whole-body", "ct pet", "whole-body"),
    "pipeline_pesa_fat_dixon": T("pipelines", "Whole-body", "dixon", "whole-body"),
    "pipeline_qvtpy": T("pipelines", "Vascular", "4dflow mra", "vascular", suite="QVTpy"),
    "fsl_anat": T("pipelines", "Structural", "smri", "neuro", suite="FSL"),
    # Data
    "fastsurfer_load_subject": T("data", "Import", "smri", "neuro", suite="FastSurfer"),
    # Preclinical
    "lab_mouse_tof_cow": T("lab", "Mouse", "mra", "vascular", "mouse"),
    # Utilities
    "fsl_maths": T("utilities", "Calculators", suite="FSL"),
    "fsl_command": T("utilities", "Raw commands", suite="FSL"),
    "fsl_install": T("utilities", "Installers", suite="FSL"),
    "fastsurfer_install_freesurfer": T("utilities", "Installers", suite="FreeSurfer"),
}

#: Old registry category -> section, for tools added later without a :data:`TABLE` entry.
_CATEGORY_FALLBACK: dict[str, str] = {
    "Restoration": "preprocess", "Reconstruction": "preprocess", "Filters": "filters",
    "Morphology": "morphology", "Centerline": "vessels", "Segmentation": "segmentation",
    "FastSurfer": "segmentation", "FreeSurfer": "surfaces", "FSL": "utilities",
    "ASL perfusion": "perfusion", "Registration": "registration", "Visualization": "visualization",
    "Transform": "transform", "Interpolation": "transform", "Time & frequency": "time",
    "Measure": "measure", "Lab": "lab", "Pipelines": "pipelines",
}
_PREFIX_SUITES: tuple[tuple[str, str], ...] = (
    ("fsl_", "FSL"), ("fastsurfer_", "FastSurfer"), ("freesurfer_", "FreeSurfer"),
    ("antsxnet_", "ANTsPyNet"), ("ants_", "ANTs"), ("vmtk_", "VMTK"),
)


def meta_for(tool: object) -> ToolMeta:
    """Placement of a registry tool (a ``GuiToolSpec`` or anything with ``id`` / ``category``)."""
    declared = getattr(tool, "meta", None)
    if isinstance(declared, ToolMeta):
        return declared
    tool_id = str(getattr(tool, "id", ""))
    if tool_id in TABLE:
        return TABLE[tool_id]
    suite = next((s for prefix, s in _PREFIX_SUITES if tool_id.startswith(prefix)), "nvitk")
    section = _CATEGORY_FALLBACK.get(str(getattr(tool, "category", "")), "utilities")
    return ToolMeta(section, str(getattr(tool, "category", "")), suite=suite)


def matches(meta: ToolMeta, *, modalities: frozenset = frozenset(), domains: frozenset = frozenset(),
            species: frozenset = frozenset(), suites: frozenset = frozenset()) -> bool:
    """Whether a tool passes the chip filters (an empty filter passes everything; a tool with
    no modality tags applies to any modality)."""
    if modalities and meta.modalities and not (set(meta.modalities) & modalities):
        return False
    if domains and meta.domain != "general" and meta.domain not in domains:
        return False
    if species and not (set(meta.species) & species):
        return False
    if suites and meta.suite not in suites:
        return False
    return True


__all__ = [
    "DOMAINS",
    "MODALITIES",
    "SECTIONS",
    "SECTIONS_BY_KEY",
    "SPECIES",
    "Section",
    "TABLE",
    "ToolMeta",
    "matches",
    "meta_for",
]
