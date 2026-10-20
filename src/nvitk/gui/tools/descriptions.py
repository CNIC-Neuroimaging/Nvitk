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
    "hessian_filter": (
        "Hybrid Hessian ridge / vesselness filter (skimage.filters.hessian). "
        "Use bright ridges (default) for TOF/CD; enable black ridges for dark vessels."
    ),
    "jerman_filter": (
        "Jerman multiscale vesselness (IEEE TMI 2016). More uniform vessel response "
        "than Frangi/Hessian across scales; use bright ridges (default) for TOF/CD. "
        "Tau in [0.5, 1]: lower → stronger enhancement."
    ),
    "snakes_filter": (
        "Kass snakes / parametric active contours (IJCV 1988). Deforms an initial "
        "mask contour under tension/rigidity and image line/edge energies. "
        "Requires an init contour mask; 3-D volumes are processed slice-wise."
    ),
    "dilate": "Expand foreground voxels by a spherical footprint.",
    "erode": "Shrink foreground voxels by a spherical footprint.",
    "open": "Erosion followed by dilation (removes small protrusions).",
    "close": "Dilation followed by erosion (fills small holes).",
    "fill_holes": "Fill enclosed background regions in a binary mask.",
    "label_cc": "Label connected components in a mask.",
    "remove_small_components": "Remove connected components below a minimum size.",
    "morph_biggest_cc": (
        "Keep the N largest connected components of a binary mask (N=1 → single largest)."
    ),
    "skeletonize": (
        "Reduce a mask to a 1-voxel-thick skeleton / centerline; "
        "multilabel masks keep each label id on its skeleton voxels."
    ),
    "centerline_detect_junctions": (
        "Mark skeleton branch points (degree ≥ N) on a 3D centerline mask."
    ),
    "centerline_to_polyline": (
        "Convert a complete centerline mask into smoothed Napari path shapes. "
        "Per label keeps the main path and all unique branch edges through "
        "bifurcations. Min branch points (0 = keep all) is the only prune."
    ),
    "centerline_cut_junctions": (
        "Split a label at junction markers from Detect skeleton junctions."
    ),
    "siphon_correct": "Correct ICA siphon centerlines using a TOF/MRA reference.",
    "mask_genus": "Report topological genus of a mask (handles / tunnels).",
    "seg_get_label": "Extract one label id into a binary mask layer.",
    "seg_mask_image": (
        "Keep an image's voxels inside a mask's labels, or remove them (binary or "
        "multilabel mask). Either layer can be the active one; margin grows or shrinks "
        "the region in mm."
    ),
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
    "seg_biggest_cc": (
        "Keep the N largest connected components (N=1 → single largest)."
    ),
    "seg_split_lr_cc": "Split mask into left/right by connected components.",
    "seg_split_lr_midline": "Split mask at a sagittal midline plane.",
    "seg_region_grow": (
        "Grow from seed voxels. Barriers block other label ids only; "
        "mask and centerline layers use separate dilation radii."
    ),
    "seg_blood_flood": (
        "Two modes. expand: active layer = seed/marker labels, intensity layer = CD/TOF. "
        "With multiple selected seeds, grows each label via watershed (mutually exclusive) "
        "and returns a multilabel segmentation; unselected labels act as barriers. "
        "from_scratch: active layer = intensity only; Frangi→hysteresis tree + CC labels "
        "(optional ROI mask). Optional barrier punches hard walls out of the tree."
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
        "volume. Uses streaming patch inference to avoid OOM on large TOFs. "
        "Leave mask at (none) to auto-extract a brain mask, or supply a binary "
        "brain mask layer. Prefer small batch size (default 2) on CPU."
    ),
    "seg_dkt": (
        "ANTsPyNet Desikan-Killiany-Tourville cortical parcellation on a T1w volume."
    ),
    "measure_centerline_arc_length": (
        "Report centerline polyline arc length in voxels and mm (debug)."
    ),
    "measure_morphometrics": (
        "Run TOF morphometrics on the selected multilabel Labels layer. "
        "Topology dropdown lists JSON files under measure/morpho/topology "
        "(eicab_topology.json for TOF/eICAB labels, qvtpy_topology.json for "
        "4D-flow reference labels, mouse_root_topology.json, …); choose 'none' for "
        "vessel-wise metrics without CoW topology. Species selects the anatomical "
        "frame used to find each vessel's proximal end: 'auto' reads it from the "
        "topology's _meta block (mouse_root_topology.json declares 'mouse'), and a "
        "mouse is treated as a quadruped so 'caudal' resolves onto the scanner A/P "
        "axis instead of S/I. Leave Output directory empty "
        "to display results in the GUI only (nothing persisted)."
    ),
    "seg_adjust_masks": "Morphological adjust label masks (open/close per label).",
    "seg_totalsegmentator": "Run TotalSegmentator on the active image.",
    "seg_eicab": "Run EICAB segmentation (cluster or local).",
    "viz_pet_hotspots": "Highlight high-SUV voxels as Napari points.",
    "viz_voxelwise_3d": (
        "Suprathreshold voxels inside a translucent brain shell, in 3-D. Needs no active layer. "
        "Draws any of the maps randomise wrote - the corrected 1-p maps for where the evidence "
        "is, the t-statistic for how large and which direction the effect is - thresholded on a "
        "window picked from the map's own distribution. Clusters as iso-surfaces or as points "
        "coloured by value; the shell is drawn either way."
    ),
    "measure_voxelwise": (
        "Mass-univariate voxelwise GLM over a cohort (FSL randomise), with permutation-based "
        "family-wise-error correction. Needs no active layer: the images come from a flat "
        "directory of MNI-normalised volumes and the design matrix from database measurements, "
        "so the two can be different modalities. Or runs randomise on a prepared 4D stack and "
        "design.mat/.con/.fts built elsewhere. randomise_parallel runs its fragments on this "
        "machine, capped by 'jobs' (or $FSLSUB_PARALLEL). Opens its own window; also loads a "
        "finished results folder without re-running. Corrected maps are 1-p, so 0.95 means p < 0.05. The 3-D scene is a separate tool under Visualization."
    ),
    "viz_flowshow": (
        "4D flow velocity vectors in Napari: all phases precomputed, arrow length and "
        "color from speed (mm/s). Syncs to the cardiac-phase slider; optional auto-play."
    ),
    "viz_flow_traces": (
        "4D flow particle traces in Napari, traced by VMTK (vmtkparticletracer's temporal stream "
        "tracer, mm and seconds). Active layer = vessel segmentation mask. Streamlines: the "
        "frozen flow of each cardiac phase. Pathlines: particles released at the selected phase "
        "and advected through the cardiac cycle. Color by speed, arc length, integration time, "
        "or fixed; optional per-vertex gradient. Planar inlet seeding by default."
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
    "recon_background_correction": (
        "Polynomial background correction, as qvtpy applies it to the 4D-flow complex difference: "
        "fit a smooth spatial polynomial on the lowest-valued voxels (lowest |values| for a signed "
        "phase/velocity map), subtract it, floor magnitudes at 0. 3D, or 3D+t per frame / on the "
        "temporal mean."
    ),
    "time_extract_frame": "One time point of a 3D+t layer as a 3D layer (-1 = the frame on screen).",
    "time_projection": (
        "Collapse a 3D+t layer over time: max/mean/min/std/sum/median, or ttp (time to peak, "
        "seconds) and auc (area under the curve) maps for dynamic CT / perfusion."
    ),
    "time_curve": (
        "Time–intensity curve at the cursor voxel or over a mask label (HU vs keV on a "
        "monoenergetic stack), in a plot window that collects curves and saves them as CSV."
    ),
    "time_stack_layers": (
        "Stack same-grid 3D image layers into one 3D+t layer, ordered by energy (monoE) or "
        "cardiac phase when known."
    ),
    "kspace_fft": (
        "K-space of the active image: log-magnitude (or phase/real/imag) on the same grid; the "
        "complex data stays attached for Inverse FFT. 2D mode transforms slice by slice."
    ),
    "kspace_ifft": (
        "Inverse FFT of the active k-space layer; an optional mask keeps (or removes) the "
        "painted part of k-space."
    ),
    "kspace_filter": "Radial low/high/band-pass filtering in k-space, with a smooth or ideal edge.",
    "volume_projection": (
        "Maximum / mean / median (etc.) intensity projection along a chosen axis."
    ),
    "reg_flirt_rigid": "Rigid FLIRT registration to a reference volume.",
    "reg_flirt_apply": "Apply a saved FLIRT transform to a volume.",
    "isotropy": "Resample to near-isotropic voxel spacing.",
    "resample_to": (
        "Resample the active layer onto the reference layer's grid (shape, spacing, position and "
        "orientation), like `nvitk-transform resample`. The result is placed and saved on the "
        "reference grid; label maps are resampled nearest-neighbour."
    ),
    "rotate_volume": (
        "Rotate the active volume around axis 0/1/2 (default Z = 2). "
        "Angle is counter-clockwise in the plane orthogonal to that axis. "
        "Use interpolation order 0 for label masks."
    ),
    "orient_volume": (
        "Show NIfTI axis codes or mirror/permute the Napari display to a target layout."
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
    "lab_mouse_tof_cow": (
        "Lab recipe for mouse TOF Circle of Willis. Stage 1 (Run): N4 (shrink 2, spline 6, "
        "rescale on) → blood flood from-scratch → label CCs (connectivity 3); adds a labeled-CC "
        "layer only. Stage 2: click CCs on that Labels layer, Add CC to tree / Tree done for "
        "Left ICA → Right ICA → Basilar; then a final multilabel blood-flood expand "
        "(labels 1/2/3) on the raw TOF volume."
    ),
    # ── FastSurfer ───────────────────────────────────────────────────────────
    "fastsurfer_seg": (
        "FastSurfer (Deep-MI) on the active T1: whole-brain aparc.DKTatlas+aseg segmentation, and "
        "by default cerebellum (CerebNet), hypothalamus (HypVINN) and corpus callosum. Runs in the "
        "background (Jobs window, ~1-3 min on GPU); label maps open named and coloured, stats go to "
        "the results window. 'Native image' keeps every result on the input's own grid; off, they are "
        "on FastSurfer's conformed isotropic grid. Uses the GPU when GPU computing is on."
    ),
    "fastsurfer_cereb": (
        "CerebNet cerebellum sub-segmentation (lobules, vermis, white matter). Reuses the subject's "
        "whole-brain segmentation when the subjects directory + id already hold one; otherwise it is "
        "computed first from the active T1."
    ),
    "fastsurfer_hypothal": (
        "HypVINN hypothalamus sub-segmentation (hypothalamic nuclei, fornix, mammillary bodies, "
        "optic chiasm …). A T2 layer improves it. Reuses an existing whole-brain segmentation."
    ),
    "fastsurfer_cc": (
        "Corpus callosum and fornix segmentation with FastSurfer-CC shape analysis (thickness, "
        "area, curvature per segment → results window). Reuses an existing whole-brain segmentation."
    ),
    "fastsurfer_surf": (
        "FastSurfer's recon-surf on an already segmented subject: white/pial surfaces, cortical "
        "thickness, surface parcellation and stats. Needs FreeSurfer 7.4.1 and its license (see "
        "'Install FreeSurfer'); takes about an hour, in the background."
    ),
    "fastsurfer_full": (
        "The whole FastSurfer pipeline on the active T1: every segmentation module, then the "
        "surfaces (FreeSurfer 7.4.1 + license). About an hour, in the background."
    ),
    "fastsurfer_long": (
        "FastSurfer longitudinal processing: an unbiased within-subject template from the named "
        "time-point layers, then each time point initialised from it (surfaces included; needs "
        "FreeSurfer 7.4.1 + license). Several hours, in the background."
    ),
    "fastsurfer_load_subject": (
        "Open an existing FastSurfer / FreeSurfer subject directory: T1s, segmentations (named and "
        "coloured), white/pial surfaces (pial coloured by thickness) and every .stats table."
    ),
    "fastsurfer_segstats": (
        "Partial-volume corrected volume (and intensity) statistics per label of the active "
        "segmentation (FastSurfer's segstats); give the bias-corrected T1 as intensity layer for "
        "the PV correction."
    ),
    "fastsurfer_reduce_aseg": (
        "Reduce the active aparc+aseg segmentation to aseg (cortex one label per hemisphere) and a "
        "brain mask."
    ),
    "fastsurfer_conform": (
        "Conform the active image the way FastSurfer does before segmenting: isotropic voxels, "
        "256³ (or the field of view), LIA orientation, uint8. A label map is conformed "
        "nearest-neighbour."
    ),
    "fastsurfer_install_freesurfer": (
        "Download (≈9.5 GB, resumable) and unpack FreeSurfer 7.4.1, which FastSurfer's surface and "
        "longitudinal pipelines need. The license is free: register at surfer.nmr.mgh.harvard.edu."
    ),
    # ── FSL ──────────────────────────────────────────────────────────────────
    "fsl_bet": "FSL BET brain extraction of the active image: brain and brain-mask layers.",
    "fsl_fast": (
        "FSL FAST: CSF / GM / WM segmentation (Labels), partial-volume maps (hidden) and the "
        "bias-corrected image. Background job."
    ),
    "fsl_first": "FSL FIRST (run_first_all) subcortical segmentation, as named Labels. Background job.",
    "fsl_anat": (
        "fsl_anat: reorientation, FOV crop, bias correction, linear + non-linear registration to MNI, "
        "FAST and FIRST in one run. Background job; the .anat folder holds every intermediate."
    ),
    "fsl_siena": "SIENA percentage brain volume change between the active and a second time point.",
    "fsl_sienax": "SIENAX normalised brain, grey and white matter volumes (results window).",
    "fsl_reorient2std": "fslreorient2std: reorient to the MNI orientation (axis swaps / flips, no resampling).",
    "fsl_robustfov": "robustfov: crop the lower head and neck before registration or BET.",
    "fsl_deface": "fsl_deface: remove the face from a structural image (anonymisation).",
    "fsl_lesion_filling": "lesion_filling: fill lesions with surrounding white-matter intensities.",
    "fsl_fnirt": (
        "FNIRT non-linear registration of the active image to a reference layer. Start from a FLIRT "
        "affine (Registration ▸ FLIRT) for robust results. Background job."
    ),
    "fsl_applywarp": "applywarp: resample the active layer onto a reference through a warp and/or affine.",
    "fsl_invwarp": "invwarp: invert the active warp field.",
    "fsl_convert_xfm": "convert_xfm: invert a FLIRT matrix, or concatenate two.",
    "fsl_mcflirt": "MCFLIRT motion correction of a 4D series; motion summary in the results window.",
    "fsl_epi_reg": "epi_reg: boundary-based registration of an EPI to its T1.",
    "fsl_topup": "TOPUP: susceptibility field from reversed phase-encode b0 volumes (needs a datain file).",
    "fsl_eddy": (
        "EDDY eddy-current and motion correction of the active DWI series (GPU build when installed). "
        "bvecs / bvals are found next to the layer's source file when left blank."
    ),
    "fsl_dtifit": "DTIFIT tensor fit of the active DWI series: FA (shown), MD, L1, V1, S0.",
    "fsl_bedpostx": "BEDPOSTX fibre orientation distributions (GPU build when installed). Background job.",
    "fsl_probtrackx2": "PROBTRACKX2 probabilistic tractography seeded from the active mask layer.",
    "fsl_tbss": (
        "TBSS steps 1-4 on a cohort's FA maps: mean FA, skeleton, and the 4D skeletonised data for "
        "randomise (Measure ▸ Voxelwise)."
    ),
    "fsl_xtract": "XTRACT: standardised tractography of the major white-matter tracts from a BEDPOSTX run.",
    "fsl_melodic": "MELODIC single-session ICA of the active 4D series (component maps + HTML report).",
    "fsl_susan": "SUSAN edge-preserving noise reduction.",
    "fsl_slicetimer": "slicetimer: slice-timing correction of a 4D series.",
    "fsl_fugue": "FUGUE: unwarp an EPI with a field map in rad/s.",
    "fsl_prepare_fieldmap": "fsl_prepare_fieldmap: a field map in rad/s from Siemens phase difference / GE Hz data.",
    "fsl_maths": "fslmaths on the active layer; {layer:NAME} brings in another layer (e.g. -mas {layer:mask}).",
    "fsl_stats": "fslstats summary (range, robust range, mean, std, volume) of the active layer, optionally in a mask.",
    "fsl_meants": "fslmeants: the mean time series of the active 4D layer inside a mask.",
    "fsl_cluster": "cluster: connected clusters above a threshold, as Labels plus the cluster table.",
    "fsl_smoothest": "smoothest: smoothness (DLH, resels, FWHM) of the active z-stat inside a mask.",
    "fsl_command": (
        "Run any FSL command in the FSL environment — FEAT, or anything not wrapped. {layer:NAME} is "
        "replaced by that layer's file, {out:NAME} names an output that is opened afterwards."
    ),
    "fsl_install": (
        "Install nvitk's own FSL core (FSL's conda packages, pinned release) into this environment. "
        "An FSL already set up through $FSLDIR is used first."
    ),
    "fsl_asl_oxford": (
        "FSL BASIL perfusion pipeline (oxford_asl) on the active ASL series: Bayesian kinetic-model fit "
        "giving CBF (ml/100g/min with an M0 calibration layer or value), arrival time (ATT) and arterial "
        "blood volume (aCBV), with their variances. Single- or multi-delay pCASL / pASL, tag-control or "
        "differenced data. A structural layer or fsl_anat folder adds registration, GM / WM means and "
        "partial-volume correction; the fsl_anat folder also gives MNI-space maps and an atlas region "
        "analysis (tables). Background job."
    ),
    "fsl_asl_basil": (
        "BASIL alone on label-control differenced data (asl_file): perfusion in data units and arrival "
        "time from the variational Bayes fit, with the adaptive spatial prior. Calibrate with asl_calib."
    ),
    "fsl_asl_file": (
        "asl_file on the active ASL series: label-control subtraction (difference series + mean per delay), "
        "tag / control split, or averaging; 'tis' output order is what BASIL reads."
    ),
    "fsl_asl_calib": (
        "asl_calib on the active M0 / calibration layer: arterial-blood M0 from a reference region (CSF by "
        "default; a mask layer, or segmented from a structural), and the perfusion layer in ml/100g/min."
    ),
}


#: Appended to every per-label tool's description, so the behaviour is documented once instead of
#: being restated (and drifting) in a dozen strings.
MULTILABEL_NOTE: str = (
    "With several labels selected, runs once per label and combines the results, keeping the "
    "input's label ids."
)


def tool_description_text(tool_id: str, *, fallback_label: str = "") -> str:
    """Human-readable description for *tool_id* from ``TOOL_DESCRIPTIONS``, falling back to
    *fallback_label* (or a generic prompt) when no description or tool id is available.

    Per-label tools (``GuiToolSpec.multilabel``) get :data:`MULTILABEL_NOTE` appended."""
    if not tool_id:
        return "Select a category and operation."
    text = TOOL_DESCRIPTIONS.get(tool_id, "").strip() or fallback_label
    if not text:
        return ""

    from nvitk.gui.tools.registry import tool_by_id

    spec = tool_by_id(tool_id)
    if spec is not None and spec.multilabel:
        return f"{text} {MULTILABEL_NOTE}"
    return text
