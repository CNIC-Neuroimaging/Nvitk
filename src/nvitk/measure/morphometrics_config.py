"""Algorithm constants for TOF Circle-of-Willis morphometrics.

These are processing defaults ported from cow_morpho. Paths (segmentation input,
output directory, vessel topology) are resolved by the qvtpy stage-7 driver
(:mod:`nvitk.pipes.qvtpy.stage7_morphometrics`) via
:mod:`nvitk.pipes.qvtpy.util.eicab.morpho_paths` and
:mod:`nvitk.pipes.qvtpy.config` layout constants — not here.
"""

from __future__ import annotations

# =========================
# LABEL SELECTION
# =========================

LABELS = "auto"
PROCESS_SELECTED_TAGS_ONLY = False
SELECTED_TAGS = [15]

SAVE_CENTERLINES = True
SAVE_CENTERLINE_RADIUS = False
SAVE_SURFACES = True

RESAMPLE_CENTERLINES_BY_ARCLENGTH = True
CENTERLINE_RESAMPLE_STEP_MM = 0.10

EXPORT_ANATOMIC_SPLIT_CENTERLINES = True
ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS = 20

STENOSIS_THRESHOLD_PCT = 10.0
ENLARGEMENT_THRESHOLD_PCT = 10.0

# Radius source used for stenosis/enlargement detection. The exported VTPs keep
# both CrossSectionRadius and MaximumInscribedSphereRadius when available.
RADIUS_SOURCE_FOR_CALIBER_DETECTION = "maximum_inscribed_sphere"  # or "cross_section"


# =========================
# ADVANCED CONFIG
# =========================

# Surface refinement for VMTK. This can stabilize point snapping at narrow tips.
REFINE_SURFACE_FOR_VMTK = True
SURFACE_SUBDIVISION_LEVELS = 1
SURFACE_REFINEMENT_SMOOTH_ITERATIONS = 5
SAVE_PRE_REFINED_SURFACES = False

CENTERLINE_RESAMPLE_MIN_SEGMENT_MM = 1e-6

# VMTK retry for narrow/poorly meshed tips.
RETRY_VMTK_WITH_TRIMMED_SEEDS = True
VMTK_SEED_TRIM_RETRY_MM = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
VMTK_MIN_RETRIED_SEED_SEPARATION_MM = 2.0
VALIDATE_VMTK_CENTERLINE_GEOMETRY = True
VMTK_MIN_VALID_CENTERLINE_POINTS = 10
VMTK_VALIDATE_LENGTH_RATIO = False
VMTK_VALIDATE_ENDPOINT_SEEDS = False
VMTK_VALIDATE_REFERENCE_CONNECTION = False
VMTK_MIN_CENTERLINE_TO_SEED_PATH_LENGTH_RATIO = 0.35
VMTK_MAX_CENTERLINE_ENDPOINT_SEED_DIST_MM = None
VMTK_REFERENCE_CONNECTION_TOL_MM = None

ENABLE_TREE_MODE = True
PROCESS_ALL_CONNECTED_COMPONENTS = False
MIN_TREE_ENDPOINTS_FOR_TREE_MODE = 3

ENABLE_DONUT_LOOP_MODE = True
SAVE_DONUT_LOOP_DEBUG = False
USE_VMTK_FOR_DONUT_LOOP_PATH = True
SAVE_UNSELECTED_DONUT_ARM_SKELETON = True
RUN_VMTK_ON_ISOLATED_DONUT_ARM = True
DONUT_ARM_MASK_MARGIN_MM = 1.0
PRESERVE_DONUT_ARM_ENDPOINTS = True
TRIM_DONUT_ARM_VMTK_TO_SKELETON_ENDPOINTS = True
DONUT_ARM_ENDPOINT_BLEND_MM = 1.0
TRIM_DONUT_ARM_ENDS_BY_RADIUS = True
DONUT_ARM_END_TRIM_RADIUS_FACTOR = 0.2
USE_SKELETON_CONNECTORS_AFTER_TRIM = False
TRIM_DONUT_ARM_OVERLAP_WITH_MAIN_CENTERLINE = True
DONUT_ARM_MAIN_OVERLAP_TOL_MM = None
DONUT_ARM_MIN_POINTS_AFTER_MAIN_OVERLAP_TRIM = 3

DISCARD_SHORT_CENTERLINE_PATHS = True
MIN_CENTERLINE_PATH_LENGTH_MM = 7.5
PRUNE_OVERLAPPING_FINAL_CENTERLINE_PREFIXES = True
FINAL_CENTERLINE_OVERLAP_TOL_MM = None
FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE = 10

SPLIT_BIFURCATING_TREE_REGIONS = True
CENTERLINE_OVERLAP_TOL_MM = 0.25
MIN_COMMON_BASE_POINTS = 2
CENTERLINE_OVERLAP_MAX_GAP_POINTS = 2
DISCARD_SHORT_TREE_ARMS = True
MIN_TREE_ARM_LENGTH_MM = 4
VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM = {"LICA": 10.0, "RICA": 10.0, "BA": 10.0}

ENABLE_RECURSIVE_TREE_SEGMENTS = True
TREE_SEGMENT_ASSIGN_TOL_MM = None
REMOVE_ROOT_TO_TERMINAL_CENTERLINE_VTPS_AFTER_SPLIT = True

SKELETON_STEP_MM = None
SKELETON_WALL_TOL_MM = None
SKELETON_TANGENT_NPTS = 5
SKELETON_MAX_STEPS = 300

PRUNE_TERMINAL_SPURS = True
PRUNE_SPUR_LENGTH_MM = 2.0

STENOSIS_MIN_LEN_MM = 3.0
STENOSIS_EXCLUDE_END_MM = 2.0
# Larger end-exclusion applied only to the CoreCandidate / SupportCandidate
# diagnostic arrays.  Vessel labels often taper or end abruptly, which the
# taper reference can misread as a stenosis; 5 mm suppresses those artefacts
# without touching the main detection threshold.
STENOSIS_CANDIDATE_EXCLUDE_END_MM = 5.0
STENOSIS_MAX_INTERNAL_GAP_MM = 1.5
STENOSIS_SUPPORT_THRESHOLD_PCT = 25.0
STENOSIS_SEGMENT_REFERENCE_MARGIN_MM = 5.0
ENLARGEMENT_MIN_LEN_MM = 5.0
# Alternative acceptance criterion: a segment is kept when it spans at least
# this many consecutive support-threshold points, even if its arc length is
# below ENLARGEMENT_MIN_LEN_MM.  At 0.1 mm resampling, 3 points ≈ 0.3 mm.
# Set to None to disable (pure mm-length filter only).
ENLARGEMENT_MIN_CONSECUTIVE_SUPPORT_POINTS = 3
ENLARGEMENT_EXCLUDE_END_MM = 3.0
ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM = 5.0
ENLARGEMENT_MAX_INTERNAL_GAP_MM = STENOSIS_MAX_INTERNAL_GAP_MM
ENLARGEMENT_SUPPORT_THRESHOLD_PCT = 25.0
SUPPRESS_ENLARGEMENT_NEAR_CENTERLINE_STARTS = True
ENLARGEMENT_CENTERLINE_START_EXCLUDE_MM = None

# Idea A — taper-fit end-zone exclusion: points within this arc-length distance
# of each vessel endpoint are excluded from the taper reference *fit* but are
# still eligible for stenosis/enlargement detection.  The reference at those
# points is extrapolated from the healthy interior via fill_reference_gaps +
# the PAVA non-increasing constraint.  Must be >= the corresponding
# STENOSIS_EXCLUDE_END_MM / ENLARGEMENT_EXCLUDE_END_MM values.
STENOSIS_TAPER_FIT_EXCLUDE_END_MM = 10.0
ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM = 10.0

# Idea B — iterative taper reference: after each detection pass the detected
# lesion points are excluded from the reference fit and the reference is
# recomputed.  Iterations continue until the lesion mask stops changing
# (convergence) or TAPER_TWO_PASS_MAX_ITERATIONS is reached.  This breaks the
# self-contamination loop where a large lesion biases the local percentile
# envelope toward the lesion radius.
TAPER_TWO_PASS = True
TAPER_TWO_PASS_MAX_ITERATIONS = 5

# Taper (reference radius) estimation.
# The reference is a healthy-caliber envelope, not a least-squares baseline:
# local high-percentile radii estimate the expected non-stenotic caliber, then
# the curve is smoothed and constrained to taper along arc length.
# SIPHON_KAPPA_THRESHOLD identifies high-curvature regions (e.g. ICA cavernous
# siphon) that are excluded from the taper fit so the wide siphon loop cannot
# bias the slope toward zero.  Those same points are also suppressed from
# enlargement detection (the siphon is anatomy, not pathology).
# SIPHON_DILATION_MM extends the siphon mask bilaterally in arc-length.
SIPHON_KAPPA_THRESHOLD = 0.10   # mm^-1: curvature above this → siphon region
SIPHON_DILATION_MM = 5.0        # mm: bilateral arc-length dilation of siphon mask
# When True, siphon-region points are excluded from enlargement detection (detect_core
# mask).  Set False to detect enlargements in siphon regions — useful for diagnosing
# why EnlargementCoreCandidate points are not promoted to EnlargementBinary.
SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION = False
TAPER_FIT_OUTLIER_FRACTION = 0.60
TAPER_FIT_MAX_ITERATIONS = 3
TAPER_FIT_MIN_HEALTHY_FRACTION = 0.45
TAPER_FIT_ENFORCE_NONINCREASING = True
TAPER_REFERENCE_PERCENTILE = 85.0
TAPER_REFERENCE_WINDOW_MM = 20.0
TAPER_REFERENCE_SMOOTH_MM = 4.0

INFLECT_KAPPA_MIN = 0.02
INFLECT_SMOOTH_WIN = 7
BEND_KAPPA_PEAK = 0.05

TREE_REGION_CODES = {
    "unassigned": 0,
    "common_base": 1,
    "arm1": 2,
    "arm2": 3,
    "base_plus_arm1": 12,
    "base_plus_arm2": 13,
    "trunk": 1,
    "branch01": 2,
    "branch02": 3,
    "trunk_plus_branch01": 12,
    "trunk_plus_branch02": 13,
}

TREE_REGION_ROLE_CODES = {
    "unassigned": 0,
    "trunk": 1,
    "branch": 2,
    "fused_trunk_branch": 3,
    "donut_selected_arm": 4,
    "donut_alternate_arm": 5,
}


from dataclasses import dataclass


@dataclass(frozen=True)
class MorphometricsConfig:
    """Runtime configuration for TOF morphometrics (defaults mirror module constants)."""

    labels: str = LABELS
    process_selected_tags_only: bool = PROCESS_SELECTED_TAGS_ONLY
    selected_tags: tuple[int, ...] = tuple(SELECTED_TAGS)
    save_centerlines: bool = SAVE_CENTERLINES
    save_centerline_radius: bool = SAVE_CENTERLINE_RADIUS
    save_surfaces: bool = SAVE_SURFACES
    resample_centerlines_by_arclength: bool = RESAMPLE_CENTERLINES_BY_ARCLENGTH
    centerline_resample_step_mm: float = CENTERLINE_RESAMPLE_STEP_MM
    export_anatomic_split_centerlines: bool = EXPORT_ANATOMIC_SPLIT_CENTERLINES
    anatomic_branch_z_lookahead_points: int = ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS
    stenosis_threshold_pct: float = STENOSIS_THRESHOLD_PCT
    enlargement_threshold_pct: float = ENLARGEMENT_THRESHOLD_PCT
    radius_source_for_caliber_detection: str = RADIUS_SOURCE_FOR_CALIBER_DETECTION
    enable_tree_mode: bool = ENABLE_TREE_MODE
    process_all_connected_components: bool = PROCESS_ALL_CONNECTED_COMPONENTS
    enable_donut_loop_mode: bool = ENABLE_DONUT_LOOP_MODE
    discard_short_centerline_paths: bool = DISCARD_SHORT_CENTERLINE_PATHS
    min_centerline_path_length_mm: float = MIN_CENTERLINE_PATH_LENGTH_MM
    split_bifurcating_tree_regions: bool = SPLIT_BIFURCATING_TREE_REGIONS
    enable_recursive_tree_segments: bool = ENABLE_RECURSIVE_TREE_SEGMENTS
    prune_terminal_spurs: bool = PRUNE_TERMINAL_SPURS
    prune_spur_length_mm: float = PRUNE_SPUR_LENGTH_MM
    taubin_iters: int = 20
    taubin_lambda: float = 0.65
    taubin_mu: float = -0.65
    keep_largest_component_taubin: bool = False
    run_tortuosity: bool = True
    run_histograms: bool = True
    n_workers: int | None = None


def default_morphometrics_config() -> MorphometricsConfig:
    """Return morphometrics settings with cow_morpho defaults."""
    return MorphometricsConfig()


__all__ = [
    "MorphometricsConfig",
    "default_morphometrics_config",
]
