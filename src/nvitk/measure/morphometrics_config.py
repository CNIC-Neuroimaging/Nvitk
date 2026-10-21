"""Morphometrics configuration: named, mask-specific parameter profiles.

Every tunable of the TOF Circle-of-Willis morphometrics lives in a **profile** — a mapping from
the UPPER_CASE parameter names below to values — and :class:`MorphometricsConfig` bundles one
profile with the per-run settings (Taubin pre-smoothing, workers, species).

The algorithm modules read these parameters as attributes of this module at call time
(``from nvitk.measure import morphometrics_config as _cfg`` … ``_cfg.STENOSIS_THRESHOLD_PCT``).
:func:`apply_config` / :func:`active_config` install a profile's values here, and
:mod:`nvitk.measure.morpho.run_case` re-applies it inside every spawned worker, so a run uses
exactly the profile it was given. The module-level values are the default profile.

Profiles
--------
``eicab_cw`` (default)
    eICAB **Circle-of-Willis** (``*_eICAB_CW*``) TOF masks. The values are cow_morpho's, which
    were tuned on the PESA eICAB CW cohort (``/home/imarcoss/cow_morpho-main/config.py``), with the
    deliberate changes listed in the profile comments (siphon-curvature fix; non-overlapping
    connected-edge segment export). A whole-brain (WB) profile, tuned on its own, is pending.

Paths (segmentation input, output directory, topology) are resolved by the qvtpy stage-7 driver
(:mod:`nvitk.pipes.qvtpy.stage7_morphometrics`), not here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping

# ──────────────────────────────────────────────────────────────────────────────
# Output schema constants (not tunable; identical for every profile)
# ──────────────────────────────────────────────────────────────────────────────

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

#: Centerline extraction backends. ``vmtk`` is cow_morpho's (and the default); ``native`` is the
#: voxel-skeleton path re-centred on the lumen (:mod:`nvitk.morphology.centering`); ``raycast`` is
#: the experimental EXPIGEO ray-probing skeleton (:mod:`nvitk.morphology.raycast_skeleton`).
CENTERLINE_BACKENDS: tuple[str, ...] = ("vmtk", "native", "raycast")
#: Radius the enlargement test reads (``ENLARGEMENT_RADIUS_SOURCE``). ``cow`` follows cow_morpho:
#: the cross-section radius on a path's first measurement, the detection radius on paths
#: re-measured after trunk pruning and on donut arms. ``cross_section`` and ``detection`` use that
#: one radius on every path (``detection`` = the stenosis radius, RADIUS_SOURCE_FOR_CALIBER_DETECTION).
ENLARGEMENT_RADIUS_SOURCES: tuple[str, ...] = ("cow", "cross_section", "detection")
#: Cross-section radius estimators (``CROSS_SECTION_RADIUS_METHOD``): ``ray_median`` — the median
#: in-plane distance from the centerline point to the wall over ``CROSS_SECTION_RAYS`` rays;
#: ``largest_loop`` — cow_morpho's equivalent-circle radius of the largest loop of the cut.
CROSS_SECTION_RADIUS_METHODS: tuple[str, ...] = ("ray_median", "largest_loop")
#: Healthy reference of the lesion table's diameter / area % (``LESION_REFERENCE_METHOD``):
#: ``interpolated`` — proximal and distal windows, interpolated to the lesion (QCA-style, taper
#: corrected); ``proximal`` — the upstream window alone (distal fallback), as before 2026-10-21.
LESION_REFERENCE_METHODS: tuple[str, ...] = ("interpolated", "proximal")
ENLARGEMENT_RADIUS_SOURCE_LABELS: Mapping[str, str] = MappingProxyType({
    "cow": "cow_morpho (cross-section; inscribed on re-measured paths)",
    "cross_section": "cross-section on every path",
    "detection": "inscribed (detection radius) on every path",
})

# ──────────────────────────────────────────────────────────────────────────────
# Profile: eICAB Circle-of-Willis TOF masks (cow_morpho parity)
# ──────────────────────────────────────────────────────────────────────────────

_EICAB_CW: dict[str, Any] = {
    # ── Labels / outputs ─────────────────────────────────────────────────────
    "LABELS": "auto",                      # "auto" = every label id present in the mask
    "PROCESS_SELECTED_TAGS_ONLY": False,   # True → only SELECTED_TAGS
    "SELECTED_TAGS": (15,),
    "SAVE_CENTERLINES": True,
    "SAVE_CENTERLINE_RADIUS": False,       # extra radius-tube VTPs per path
    "SAVE_SURFACES": True,
    "SAVE_PRE_REFINED_SURFACES": False,    # QC copy of the surface before VMTK refinement
    "EXPORT_ANATOMIC_SPLIT_CENTERLINES": True,
    "ANATOMIC_BRANCH_Z_LOOKAHEAD_POINTS": 20,

    # ── Centerline extraction ────────────────────────────────────────────────
    "CENTERLINE_BACKEND": "vmtk",
    "RESAMPLE_CENTERLINES_BY_ARCLENGTH": True,
    "CENTERLINE_RESAMPLE_STEP_MM": 0.10,
    "CENTERLINE_RESAMPLE_MIN_SEGMENT_MM": 1e-6,
    # Surface refinement before VMTK (stabilises point snapping at narrow tips).
    "REFINE_SURFACE_FOR_VMTK": True,
    "SURFACE_SUBDIVISION_LEVELS": 1,
    "SURFACE_REFINEMENT_SMOOTH_ITERATIONS": 5,
    # VMTK retry for narrow / poorly meshed tips, and the attempt validation.
    "RETRY_VMTK_WITH_TRIMMED_SEEDS": True,
    "VMTK_SEED_TRIM_RETRY_MM": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    "VMTK_MIN_RETRIED_SEED_SEPARATION_MM": 2.0,
    "VALIDATE_VMTK_CENTERLINE_GEOMETRY": True,
    "VMTK_MIN_VALID_CENTERLINE_POINTS": 10,
    "VMTK_VALIDATE_LENGTH_RATIO": False,
    "VMTK_VALIDATE_ENDPOINT_SEEDS": False,
    "VMTK_VALIDATE_REFERENCE_CONNECTION": False,
    "VMTK_MIN_CENTERLINE_TO_SEED_PATH_LENGTH_RATIO": 0.35,
    "VMTK_MAX_CENTERLINE_ENDPOINT_SEED_DIST_MM": None,
    "VMTK_REFERENCE_CONNECTION_TOL_MM": None,
    # Native backend: re-centring of the voxel-skeleton path (see nvitk.morphology.centering).
    "NATIVE_CENTERING_ITERATIONS": 3,
    "NATIVE_CENTERING_SMOOTH_MM": 0.75,

    # ── Cross-section radius (surface cut by the plane normal to the centerline) ─
    # The plane's normal is the chord from the point this many mm upstream to the one this many mm
    # downstream (arc length; at least the neighbouring point on each side, one-sided at the ends).
    # 0.1 = cow_morpho: the immediate neighbours (±1 point) at the 0.1 mm resampling step. A wider
    # window tilts the plane less with centerline jitter (fewer oblique, oversized cuts) but turns
    # late in tight bends.
    "CROSS_SECTION_TANGENT_HALF_WINDOW_MM": 1.0,
    # How the cut becomes a radius (CROSS_SECTION_RADIUS_METHODS). "ray_median": from the centerline
    # point, CROSS_SECTION_RAYS rays spread evenly in the plane; each stops at its first wall
    # crossing, and the radius is the median distance — a circle between the nearest and farthest
    # wall that a branch opening (a few long rays) does not move, and that never sees the other
    # vessels the plane also cuts. "largest_loop" (cow_morpho): the largest closed loop's area as an
    # equivalent circle, which near the siphon or a bifurcation is often another limb, a
    # neighbouring branch or the merged carina, so the radius spikes and climbs towards junctions.
    "CROSS_SECTION_RADIUS_METHOD": "ray_median",
    "CROSS_SECTION_RAYS": 36,

    # ── Skeleton / tree / donut-loop handling ────────────────────────────────
    "ENABLE_TREE_MODE": True,
    "MIN_TREE_ENDPOINTS_FOR_TREE_MODE": 3,
    # cow parity: keep only the largest component per label and do not bridge fragments.
    "PROCESS_ALL_CONNECTED_COMPONENTS": False,
    "BRIDGE_LABEL_GAPS_BEFORE_CENTERLINES": False,
    "BRIDGE_LABEL_MAX_GAP_VOXELS": 12,
    "BRIDGE_LABEL_CLOSE_RADIUS": 0,
    "ENABLE_DONUT_LOOP_MODE": True,
    "SAVE_DONUT_LOOP_DEBUG": False,
    "SAVE_UNSELECTED_DONUT_ARM_SKELETON": True,
    "RUN_VMTK_ON_ISOLATED_DONUT_ARM": True,
    "DONUT_ARM_MASK_MARGIN_MM": 1.0,
    "PRESERVE_DONUT_ARM_ENDPOINTS": True,
    "TRIM_DONUT_ARM_VMTK_TO_SKELETON_ENDPOINTS": True,
    "DONUT_ARM_ENDPOINT_BLEND_MM": 1.0,
    "TRIM_DONUT_ARM_ENDS_BY_RADIUS": True,
    "DONUT_ARM_END_TRIM_RADIUS_FACTOR": 0.2,
    "USE_SKELETON_CONNECTORS_AFTER_TRIM": False,
    "TRIM_DONUT_ARM_OVERLAP_WITH_MAIN_CENTERLINE": True,
    "DONUT_ARM_MAIN_OVERLAP_TOL_MM": None,
    "DONUT_ARM_MIN_POINTS_AFTER_MAIN_OVERLAP_TRIM": 3,
    "DISCARD_SHORT_CENTERLINE_PATHS": True,
    "MIN_CENTERLINE_PATH_LENGTH_MM": 7.5,
    # cow parity: later root→terminal paths drop the trunk they share with earlier ones and their
    # stenosis / enlargement detection is re-run on the remaining branch (its own taper
    # reference); 0 keeps one shared point, as cow_morpho. The exported geometry is unaffected:
    # the non-overlapping segments (nvitk design, docs/pipelines/qvtpy-morphometrics.md) are cut
    # junction-to-junction from whichever measured path runs along them.
    "PRUNE_OVERLAPPING_FINAL_CENTERLINE_PREFIXES": True,
    "KEEP_SHARED_PREFIX_AT_BIFURCATION_MM": 0.0,
    "FINAL_CENTERLINE_OVERLAP_TOL_MM": None,
    "FINAL_CENTERLINE_MIN_POINTS_AFTER_OVERLAP_PRUNE": 10,
    "SPLIT_BIFURCATING_TREE_REGIONS": True,
    "CENTERLINE_OVERLAP_TOL_MM": 0.25,
    "MIN_COMMON_BASE_POINTS": 2,
    "CENTERLINE_OVERLAP_MAX_GAP_POINTS": 2,
    "DISCARD_SHORT_TREE_ARMS": True,
    "MIN_TREE_ARM_LENGTH_MM": 4.0,
    "VESSEL_SPECIFIC_MIN_TREE_ARM_LENGTH_MM": {"LICA": 10.0, "RICA": 10.0, "BA": 10.0},
    "ENABLE_RECURSIVE_TREE_SEGMENTS": True,
    "TREE_SEGMENT_ASSIGN_TOL_MM": None,
    "REMOVE_ROOT_TO_TERMINAL_CENTERLINE_VTPS_AFTER_SPLIT": True,
    "SKELETON_STEP_MM": None,
    "SKELETON_WALL_TOL_MM": None,
    "SKELETON_TANGENT_NPTS": 5,
    "SKELETON_MAX_STEPS": 300,
    "PRUNE_TERMINAL_SPURS": True,
    "PRUNE_SPUR_LENGTH_MM": 2.0,

    # ── Stenosis / enlargement detection (cow_morpho rules) ──────────────────
    "STENOSIS_THRESHOLD_PCT": 20.0,
    "ENLARGEMENT_THRESHOLD_PCT": 20.0,
    "RADIUS_SOURCE_FOR_CALIBER_DETECTION": "maximum_inscribed_sphere",  # or "cross_section"
    "STENOSIS_MIN_LEN_MM": 3.0,
    "STENOSIS_EXCLUDE_END_MM": 2.0,
    # Diagnostic CoreCandidate / SupportCandidate arrays only.
    "STENOSIS_CANDIDATE_EXCLUDE_END_MM": 5.0,
    "STENOSIS_MAX_INTERNAL_GAP_MM": 1.5,
    # The support tier is min(support, core): with cow's 25 % against the 20 % core the support and
    # core contours coincide, i.e. a stenosis is a ≥20 % run of ≥3 mm (gaps ≤1.5 mm merged). A
    # support below the core would grow lesions into their shoulders (hysteresis).
    "STENOSIS_SUPPORT_THRESHOLD_PCT": 25.0,
    "STENOSIS_SEGMENT_REFERENCE_MARGIN_MM": 5.0,
    "ENLARGEMENT_MIN_LEN_MM": 5.0,
    # Alternative acceptance, in arc length: a shorter enlargement still counts when its run spans
    # ≥ this. It replaces cow_morpho's "≥ 3 consecutive points", which at the 0.1 mm step accepted
    # 0.2 mm runs — single cross-section spikes; the effective minimum is the smaller of the two.
    # None = ENLARGEMENT_MIN_LEN_MM alone.
    "ENLARGEMENT_MIN_SUPPORT_LENGTH_MM": 1.5,
    "ENLARGEMENT_EXCLUDE_END_MM": 3.0,
    "ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM": 5.0,
    "ENLARGEMENT_MAX_INTERNAL_GAP_MM": 1.5,
    "ENLARGEMENT_SUPPORT_THRESHOLD_PCT": 25.0,
    # Which radius enlargements are tested on (ENLARGEMENT_RADIUS_SOURCES). "cow" keeps cow_morpho's
    # path-dependent choice: cross-section on a first measurement, detection radius on re-measured
    # (pruned) paths and donut arms, so the same bulge can pass on one path and fail on another.
    # "cross_section" or "detection" makes it one radius everywhere.
    "ENLARGEMENT_RADIUS_SOURCE": "cow",
    # Where enlargements are tested on the cross-section (area-equivalent) radius, it spikes for one or
    # two 0.1 mm samples wherever a cut catches a branch opening or sits obliquely in a bend. True
    # tests a running median of it over ENLARGEMENT_CROSS_SECTION_MEDIAN_MM of arc length instead
    # (a bulge longer than half the window survives); False keeps the raw radius, as cow_morpho.
    "ENLARGEMENT_SMOOTH_CROSS_SECTION": False,
    "ENLARGEMENT_CROSS_SECTION_MEDIAN_MM": 1.0,
    "SUPPRESS_ENLARGEMENT_NEAR_CENTERLINE_STARTS": True,
    "ENLARGEMENT_CENTERLINE_START_EXCLUDE_MM": None,
    # Taper-fit end-zone exclusion (must be ≥ the *_EXCLUDE_END_MM values).
    "STENOSIS_TAPER_FIT_EXCLUDE_END_MM": 10.0,
    "ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM": 10.0,
    # Iterative taper reference: refit without the detected lesions until stable.
    "TAPER_TWO_PASS": True,
    "TAPER_TWO_PASS_MAX_ITERATIONS": 5,
    # Healthy-calibre envelope (local high-percentile radius, smoothed, non-increasing).
    "TAPER_FIT_OUTLIER_FRACTION": 0.60,
    "TAPER_FIT_MAX_ITERATIONS": 3,
    "TAPER_FIT_MIN_HEALTHY_FRACTION": 0.45,
    "TAPER_FIT_ENFORCE_NONINCREASING": True,
    "TAPER_REFERENCE_PERCENTILE": 85.0,
    "TAPER_REFERENCE_WINDOW_MM": 20.0,
    "TAPER_REFERENCE_SMOOTH_MM": 4.0,

    # ── Siphon (high-curvature) mask: excluded from the taper fit ────────────
    # cow_morpho measured curvature on the 0.1 mm-resampled centerline (7-point smoothing), where
    # sampling noise alone exceeds its 0.10 threshold: the mask covered ~100 % of every vessel, the
    # taper fit had no points left, and the reference collapsed to the vessel's median radius. The
    # fix measures true curvature (1/mm) on the centerline smoothed at SIPHON_CURVATURE_SMOOTH_MM
    # (anatomical scale). SIPHON_LEGACY_MASK=True reproduces cow_morpho exactly, including its
    # half-valued curvature (metrics.discrete_curvature) against SIPHON_LEGACY_KAPPA_THRESHOLD.
    # Calibrated by scripts/morpho/calibrate_siphon.py on 20 random PESA CW masks (2026-10-16):
    # cow's threshold (0.10 on κ/2 = 0.20 true) and dilation are kept, only the smoothing σ is new;
    # the mask then covers 40 % of the ICAs and 9 % of the other vessels (legacy: 100 % / 100 %).
    # Curvature alone cannot confine it to the ICA siphon — PCAs and SCAs bend as tightly — so it
    # marks tight bends wherever they are, which is what the taper fit has to skip.
    "SIPHON_LEGACY_MASK": False,
    "SIPHON_LEGACY_KAPPA_THRESHOLD": 0.10,
    "SIPHON_CURVATURE_SMOOTH_MM": 3.0,
    "SIPHON_KAPPA_THRESHOLD": 0.20,
    "SIPHON_DILATION_MM": 5.0,
    "SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION": False,

    # ── Shape descriptors ────────────────────────────────────────────────────
    "INFLECT_KAPPA_MIN": 0.02,
    "INFLECT_SMOOTH_WIN": 7,
    "BEND_KAPPA_PEAK": 0.05,
    # Scale-aware descriptors of the vessel tables (vessel_features): curvature is measured on
    # the centerline smoothed at this scale, and diameter statistics skip points within
    # BLANKING_MM (or BLANKING_RADIUS_FACTOR × the local radius, whichever is larger) of a junction.
    "FEATURE_CURVATURE_SMOOTH_MM": 1.0,
    "FEATURE_END_WINDOW_MM": 5.0,
    # Calibre statistics skip max(this, local radius) at both tips (VMTK appends the seed point on
    # the wall, where the inscribed radius is ~0).
    "FEATURE_TIP_EXCLUDE_MM": 1.0,
    "BLANKING_MM": 2.0,
    "BLANKING_RADIUS_FACTOR": 1.5,

    # ── Circle-of-Willis variants (subject table) ────────────────────────────
    # A present A1 / PComm / P1 / AComm thinner than this (median diameter) is "hypoplastic".
    "HYPOPLASIA_DIAMETER_MM": 1.0,

    # ── Lesion severity / confidence (lesions table) ─────────────────────────
    # Healthy reference radius of the lesion table's diameter / area % (LESION_REFERENCE_METHODS).
    # "interpolated": the median radius of a proximal window (LESION_PROXIMAL_REFERENCE_WINDOW_MM
    # upstream of the lesion) and of a distal one (the same distances downstream), interpolated in
    # arc length to the lesion's narrowest (widest) point, so the vessel's natural taper between
    # them is not counted as stenosis. Windows keep to healthy points of the lesion's own
    # junction-to-junction stretch (no other lesions, no bifurcation blanking zone, no junction
    # crossed); a distal median above the proximal one is capped at it; one window only → that one.
    # "proximal": the upstream window alone (downstream when there is no room), as before 2026-10-21.
    "LESION_REFERENCE_METHOD": "interpolated",
    # Window distances from the lesion (mm): upstream for the proximal window, mirrored downstream
    # for the distal one.
    "LESION_PROXIMAL_REFERENCE_WINDOW_MM": (2.0, 10.0),
    # No reference window reaches into the first / last this many mm of the path: the tips hold
    # VMTK's seed point on the wall and the end taper.
    "LESION_REFERENCE_EXCLUDE_END_MM": 3.0,
    # "interpolated": proximal and distal medians differing by more than this fraction (more than
    # taper explains: distal collapse, dilatation, an unseen branch) flag reference_inconsistent.
    "LESION_REFERENCE_DISAGREEMENT_FRACTION": 0.25,
    "LESION_NEAR_BIFURCATION_MM": 3.0,
    "LESION_NEAR_END_MM": 5.0,
    # Cross-section / inscribed radius ratio inside the lesion vs that path's median ratio.
    "LESION_RADIUS_DISAGREEMENT_FRACTION": 0.20,
}

#: Named profiles. Values are plain dicts (configs travel pickled to spawned workers); copy them
#: through :class:`MorphometricsConfig` / :meth:`MorphometricsConfig.with_params`, never mutate.
PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "eicab_cw": _EICAB_CW,
})
DEFAULT_PROFILE = "eicab_cw"

#: Every tunable parameter name (all profiles share one schema).
PARAMETER_NAMES: frozenset[str] = frozenset(_EICAB_CW)

# The active values, as module attributes (the default profile until apply_config changes them).
globals().update(_EICAB_CW)
# Named here as well so readers (and linters) see the attributes the algorithm modules use.
CENTERLINE_BACKEND: str


@dataclass(frozen=True)
class MorphometricsConfig:
    """One morphometrics run's configuration: a named profile plus per-run settings.

    ``params`` holds every algorithm parameter (UPPER_CASE names, see :data:`PARAMETER_NAMES`);
    build variants with :meth:`with_params` rather than editing it.
    """

    profile: str = DEFAULT_PROFILE
    params: Mapping[str, Any] = field(default_factory=lambda: dict(PROFILES[DEFAULT_PROFILE]))
    # Taubin pre-smoothing of the label mask (skipped when the input is already smoothed).
    taubin_iters: int = 20
    taubin_lambda: float = 0.65
    taubin_mu: float = -0.65
    keep_largest_component_taubin: bool = False
    # Post-processing steps.
    run_tortuosity: bool = True
    run_histograms: bool = True
    n_workers: int | None = None
    #: ``"auto"`` defers to the topology JSON's ``_meta`` (see nvitk.measure.morpho.anatomy_axes).
    species: str = "auto"
    axes_override: str | None = None

    def __post_init__(self) -> None:
        unknown = set(self.params) - PARAMETER_NAMES
        missing = PARAMETER_NAMES - set(self.params)
        if unknown or missing:
            raise ValueError(
                f"Morphometrics profile {self.profile!r}: unknown parameters {sorted(unknown)}, "
                f"missing parameters {sorted(missing)}."
            )
        if self.params["CENTERLINE_BACKEND"] not in CENTERLINE_BACKENDS:
            raise ValueError(
                f"CENTERLINE_BACKEND must be one of {CENTERLINE_BACKENDS}, "
                f"got {self.params['CENTERLINE_BACKEND']!r}."
            )
        if not float(self.params["CROSS_SECTION_TANGENT_HALF_WINDOW_MM"]) >= 0.0:
            raise ValueError(
                "CROSS_SECTION_TANGENT_HALF_WINDOW_MM must be ≥ 0 mm, "
                f"got {self.params['CROSS_SECTION_TANGENT_HALF_WINDOW_MM']!r}."
            )
        if str(self.params["CROSS_SECTION_RADIUS_METHOD"]).lower() not in CROSS_SECTION_RADIUS_METHODS:
            raise ValueError(
                f"CROSS_SECTION_RADIUS_METHOD must be one of {CROSS_SECTION_RADIUS_METHODS}, "
                f"got {self.params['CROSS_SECTION_RADIUS_METHOD']!r}."
            )
        if int(self.params["CROSS_SECTION_RAYS"]) < 8:
            raise ValueError(f"CROSS_SECTION_RAYS must be ≥ 8, got {self.params['CROSS_SECTION_RAYS']!r}.")
        if str(self.params["LESION_REFERENCE_METHOD"]).lower() not in LESION_REFERENCE_METHODS:
            raise ValueError(
                f"LESION_REFERENCE_METHOD must be one of {LESION_REFERENCE_METHODS}, "
                f"got {self.params['LESION_REFERENCE_METHOD']!r}."
            )
        if str(self.params["ENLARGEMENT_RADIUS_SOURCE"]).lower() not in ENLARGEMENT_RADIUS_SOURCES:
            raise ValueError(
                f"ENLARGEMENT_RADIUS_SOURCE must be one of {ENLARGEMENT_RADIUS_SOURCES}, "
                f"got {self.params['ENLARGEMENT_RADIUS_SOURCE']!r}."
            )

    @property
    def centerline_backend(self) -> str:
        return str(self.params["CENTERLINE_BACKEND"])

    def with_params(self, **overrides: Any) -> "MorphometricsConfig":
        """Copy with some algorithm parameters changed (names validated)."""
        merged = dict(self.params)
        merged.update(overrides)
        return replace(self, params=merged)


def morphometrics_config(profile: str = DEFAULT_PROFILE, **run_settings: Any) -> MorphometricsConfig:
    """The :class:`MorphometricsConfig` of a named profile (run settings as keyword arguments)."""
    key = str(profile or DEFAULT_PROFILE).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown morphometrics profile {profile!r}; available: {sorted(PROFILES)}.")
    return MorphometricsConfig(profile=key, params=dict(PROFILES[key]), **run_settings)


def default_morphometrics_config() -> MorphometricsConfig:
    """The default profile (``eicab_cw``)."""
    return morphometrics_config(DEFAULT_PROFILE)


def current_params() -> dict[str, Any]:
    """The parameter values currently installed in this module."""
    g = globals()
    return {name: g[name] for name in PARAMETER_NAMES}


def apply_config(config: MorphometricsConfig) -> None:
    """Install *config*'s parameters as this module's attributes (what the algorithms read)."""
    globals().update(config.params)


@contextmanager
def active_config(config: MorphometricsConfig) -> Iterator[MorphometricsConfig]:
    """Apply *config* for the duration of the block, then restore the previous values."""
    previous = current_params()
    apply_config(config)
    try:
        yield config
    finally:
        globals().update(previous)


__all__ = [
    "CENTERLINE_BACKENDS",
    "DEFAULT_PROFILE",
    "PARAMETER_NAMES",
    "PROFILES",
    "TREE_REGION_CODES",
    "TREE_REGION_ROLE_CODES",
    "MorphometricsConfig",
    "active_config",
    "apply_config",
    "current_params",
    "default_morphometrics_config",
    "morphometrics_config",
]
