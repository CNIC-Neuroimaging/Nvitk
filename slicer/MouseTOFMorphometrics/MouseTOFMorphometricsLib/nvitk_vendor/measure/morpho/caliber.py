# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morpho/caliber.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Taper reference, stenosis detection, enlargement detection, and flag filtering."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from nvitk_vendor.measure.morphometrics_config import (
    ENLARGEMENT_MAX_INTERNAL_GAP_MM,
    ENLARGEMENT_MIN_LEN_MM,
    ENLARGEMENT_MIN_SUPPORT_LENGTH_MM,
    ENLARGEMENT_SUPPORT_THRESHOLD_PCT,
    ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM,
    ENLARGEMENT_THRESHOLD_PCT,
    INFLECT_SMOOTH_WIN,
    RADIUS_SOURCE_FOR_CALIBER_DETECTION,
    SIPHON_DILATION_MM,
    SIPHON_KAPPA_THRESHOLD,
    SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION,
    STENOSIS_MAX_INTERNAL_GAP_MM,
    STENOSIS_SEGMENT_REFERENCE_MARGIN_MM,
    STENOSIS_SUPPORT_THRESHOLD_PCT,
    STENOSIS_TAPER_FIT_EXCLUDE_END_MM,
    STENOSIS_THRESHOLD_PCT,
    TAPER_FIT_ENFORCE_NONINCREASING,
    TAPER_FIT_MAX_ITERATIONS,
    TAPER_FIT_MIN_HEALTHY_FRACTION,
    TAPER_FIT_OUTLIER_FRACTION,
    TAPER_REFERENCE_PERCENTILE,
    TAPER_REFERENCE_SMOOTH_MM,
    TAPER_REFERENCE_WINDOW_MM,
    TAPER_TWO_PASS,
    TAPER_TWO_PASS_MAX_ITERATIONS,
)
from .metrics import discrete_curvature, smooth_1d
from .models import EnlargementResult, StenosisResult

def stenosis_valid_mask_from_ends(s, exclude_end_mm) -> np.ndarray:
    """Boolean mask excluding the first/last *exclude_end_mm* of arc length *s* (avoid end-effect false positives)."""
    if len(s) == 0:
        return np.zeros(0, dtype=bool)
    s = np.asarray(s, dtype=float); length = float(s[-1]) if len(s) else 0.0
    return (s >= exclude_end_mm) & (s <= (length - exclude_end_mm))


def compute_siphon_mask(
    pts: np.ndarray,
    s: np.ndarray,
    kappa_threshold: Optional[float] = None,
    dilation_mm: Optional[float] = None,
) -> np.ndarray:
    """Boolean mask of high-curvature (siphon) regions, dilated by dilation_mm."""
    kappa_threshold = SIPHON_KAPPA_THRESHOLD if kappa_threshold is None else kappa_threshold
    dilation_mm = SIPHON_DILATION_MM if dilation_mm is None else dilation_mm
    kappa = discrete_curvature(np.asarray(pts, dtype=float))
    kappa_sm = smooth_1d(np.where(np.isfinite(kappa), kappa, 0.0), win=INFLECT_SMOOTH_WIN)
    high_k = kappa_sm >= kappa_threshold
    if not high_k.any() or dilation_mm <= 0.0:
        return high_k
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    if ds < 1e-6:
        ds = 1.0
    radius = max(1, int(np.round(dilation_mm / ds)))
    struct = np.ones(2 * radius + 1, dtype=bool)
    return ndi.binary_dilation(high_k, structure=struct)


_siphon_mask = compute_siphon_mask


def _pava_nondecreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool-adjacent-violators fit for a nondecreasing 1D sequence."""
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    values: List[float] = []
    weights: List[float] = []
    starts: List[int] = []
    ends: List[int] = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        values.append(float(yi))
        weights.append(float(max(wi, 1e-12)))
        starts.append(i)
        ends.append(i)
        while len(values) >= 2 and values[-2] > values[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_value = (values[-2] * weights[-2] + values[-1] * weights[-1]) / merged_weight
            values[-2] = merged_value
            weights[-2] = merged_weight
            ends[-2] = ends[-1]
            values.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    out = np.empty(len(y), dtype=float)
    for value, start, end in zip(values, starts, ends):
        out[start:end + 1] = value
    return out


def _enforce_nonincreasing_reference(ref: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Make the valid part of ref nonincreasing along arc length."""
    out = np.asarray(ref, dtype=float).copy()
    valid_idx = np.flatnonzero(np.asarray(valid_mask, dtype=bool) & np.isfinite(out))
    if len(valid_idx) < 2:
        return out
    out[valid_idx] = -_pava_nondecreasing(-out[valid_idx])
    return out


def _healthy_radius_envelope(
    s: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Estimate expected healthy radius from the high-percentile local envelope."""
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(s) & np.isfinite(r)

    out = np.full_like(r, np.nan, dtype=float)
    if fit_mask.sum() < 3:
        vals = r[valid_mask & np.isfinite(r)]
        med = float(np.nanmedian(vals)) if vals.size else np.nan
        out[valid_mask] = med
        return out

    half_window = max(float(TAPER_REFERENCE_WINDOW_MM) * 0.5, 1e-6)
    fit_s = s[fit_mask]
    fit_r = r[fit_mask]
    whole_fit_percentile = float(np.nanpercentile(fit_r, TAPER_REFERENCE_PERCENTILE))

    for i in np.flatnonzero(valid_mask & np.isfinite(s)):
        local = np.abs(fit_s - s[i]) <= half_window
        if local.sum() >= 3:
            out[i] = float(np.nanpercentile(fit_r[local], TAPER_REFERENCE_PERCENTILE))
        else:
            out[i] = whole_fit_percentile

    if TAPER_REFERENCE_SMOOTH_MM > 0 and len(out) >= 3:
        ds = float(np.nanmedian(np.diff(s))) if len(s) > 1 else 1.0
        if not np.isfinite(ds) or ds <= 1e-6:
            ds = 1.0
        win = max(3, int(round(float(TAPER_REFERENCE_SMOOTH_MM) / ds)))
        win = win | 1
        filled = out.copy()
        idx = np.flatnonzero(np.isfinite(filled))
        if len(idx) >= 2:
            filled = np.interp(np.arange(len(out)), idx, filled[idx])
            out[valid_mask] = smooth_1d(filled, win=win)[valid_mask]

    if TAPER_FIT_ENFORCE_NONINCREASING:
        out = _enforce_nonincreasing_reference(out, valid_mask)
    return np.where(valid_mask, np.clip(out, 1e-6, None), np.nan).astype(float)


def _linear_taper_reference(
    s: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Simple robust-enough taper used only to pre-remove obvious outliers."""
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(s) & np.isfinite(r)
    out = np.full_like(r, np.nan, dtype=float)
    vals = r[fit_mask]
    if fit_mask.sum() < 3 or not np.isfinite(vals).any():
        med = float(np.nanmedian(r[valid_mask & np.isfinite(r)]))
        out[valid_mask] = med
        return np.where(valid_mask, np.clip(out, 1e-6, None), np.nan).astype(float)

    slope, intercept = np.polyfit(s[fit_mask], r[fit_mask], 1)
    slope = float(slope)
    intercept = float(intercept)
    if TAPER_FIT_ENFORCE_NONINCREASING and slope > 0:
        slope = 0.0
        intercept = float(np.nanmedian(vals))
    out[valid_mask] = intercept + slope * s[valid_mask]
    return np.where(valid_mask, np.clip(out, 1e-6, None), np.nan).astype(float)


def _compute_vessel_taper(
    s: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    low_outlier_pct: Optional[float] = None,
    high_outlier_pct: Optional[float] = None,
    fit_basis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reference radius via an iterative healthy-caliber envelope.

    A stenosis should be judged against the expected healthy caliber, not
    against a regression line pulled downward by the stenosis itself.  The
    reference is therefore the high-percentile local radius envelope, optionally
    made non-increasing along the vessel, with low/high outlier regions removed
    from the envelope mask over a few iterations.

    fit_basis: optional mask of points allowed to contribute to the fit.
        Defaults to valid_mask.  Pass a stricter mask (e.g. interior-only) to
        prevent end-zone lesions from biasing the local percentile envelope.
        The reference is still *computed* (and interpolated) at all valid_mask
        positions regardless of fit_basis.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    _basis = np.asarray(fit_basis, dtype=bool) if fit_basis is not None else valid_mask
    fit_mask = _basis & np.isfinite(r) & np.isfinite(s)
    if exclude_mask is not None:
        fit_mask = fit_mask & ~np.asarray(exclude_mask, dtype=bool)

    def _constant_ref() -> np.ndarray:
        """Fallback reference radius when too few points are usable for a taper fit: flat median."""
        vals = r[valid_mask & np.isfinite(r)]
        med = float(np.nanmedian(vals)) if vals.size else np.nan
        return np.where(valid_mask, np.clip(med, 1e-6, None), np.nan).astype(float)

    original_fit_mask = fit_mask.copy()
    if fit_mask.sum() < 3:
        return _constant_ref()

    low_cut = None if low_outlier_pct is None else float(low_outlier_pct) * float(TAPER_FIT_OUTLIER_FRACTION)
    high_cut = None if high_outlier_pct is None else float(high_outlier_pct) * float(TAPER_FIT_OUTLIER_FRACTION)
    min_fit_points = max(3, int(np.ceil(float(TAPER_FIT_MIN_HEALTHY_FRACTION) * float(original_fit_mask.sum()))))

    pre_ref = _linear_taper_reference(s, r, valid_mask, fit_mask)
    pre_deviation_pct = (r / pre_ref - 1.0) * 100.0
    prehealthy = original_fit_mask.copy()
    if low_cut is not None:
        prehealthy &= ~(np.isfinite(pre_deviation_pct) & (pre_deviation_pct < -low_cut))
    if high_cut is not None:
        prehealthy &= ~(np.isfinite(pre_deviation_pct) & (pre_deviation_pct > high_cut))
    if prehealthy.sum() >= min_fit_points:
        fit_mask = prehealthy

    ref = _healthy_radius_envelope(s, r, valid_mask, fit_mask)
    for _ in range(max(1, int(TAPER_FIT_MAX_ITERATIONS))):
        if fit_mask.sum() < min_fit_points:
            fit_mask = original_fit_mask.copy()
        if fit_mask.sum() < 3:
            return _constant_ref()

        ref = _healthy_radius_envelope(s, r, valid_mask, fit_mask)
        deviation_pct = (r / ref - 1.0) * 100.0
        healthy = original_fit_mask.copy()
        if low_cut is not None:
            healthy &= ~(np.isfinite(deviation_pct) & (deviation_pct < -low_cut))
        if high_cut is not None:
            healthy &= ~(np.isfinite(deviation_pct) & (deviation_pct > high_cut))
        if healthy.sum() < min_fit_points or np.array_equal(healthy, fit_mask):
            break
        fit_mask = healthy

    return np.where(valid_mask, np.clip(ref, 1e-6, None), np.nan).astype(float)


def stenosis_segment_percent(
    s: np.ndarray,
    r: np.ndarray,
    r_ref_per_point: np.ndarray,
    segments: List[Tuple[int, int]],
    margin_mm: Optional[float] = None,
) -> np.ndarray:
    """Reported stenosis severity, using one reference radius per lesion.

    The moving taper reference is useful to *find* candidate lesions, but it is
    a confusing denominator for reporting because two points in the same lesion
    can be compared against different reference radii.  For accepted stenotic
    segments we therefore use the maximum expected healthy reference in and
    around that lesion.  Outside accepted segments the reported percentage is 0.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    r_ref_per_point = np.asarray(r_ref_per_point, dtype=float)
    pct = np.zeros_like(r, dtype=float)
    if len(r) == 0:
        return pct
    margin = STENOSIS_SEGMENT_REFERENCE_MARGIN_MM if margin_mm is None else float(margin_mm)
    for start, end in segments:
        start = max(0, int(start))
        end = min(len(r) - 1, int(end))
        if end < start:
            continue
        lo = float(s[start]) - margin
        hi = float(s[end]) + margin
        ref_mask = (s >= lo) & (s <= hi) & np.isfinite(r_ref_per_point)
        if not ref_mask.any():
            ref_mask = np.zeros(len(r), dtype=bool)
            ref_mask[start:end + 1] = np.isfinite(r_ref_per_point[start:end + 1])
        if not ref_mask.any():
            continue
        lesion_ref = float(np.nanmax(r_ref_per_point[ref_mask]))
        if not np.isfinite(lesion_ref) or lesion_ref <= 1e-8:
            continue
        seg = slice(start, end + 1)
        valid = np.isfinite(r[seg])
        seg_pct = np.zeros(end - start + 1, dtype=float)
        seg_pct[valid] = (1.0 - r[seg][valid] / lesion_ref) * 100.0
        pct[seg] = np.maximum(seg_pct, 0.0)
    return pct


def stenosis_raw_percent(
    s: np.ndarray,
    r: np.ndarray,
    r_ref_per_point: np.ndarray,
    exclude_end_mm: float = 0.0,
) -> np.ndarray:
    """Raw pointwise radius deficit against the taper reference."""
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    r_ref_per_point = np.asarray(r_ref_per_point, dtype=float)
    valid_core = stenosis_valid_mask_from_ends(s, exclude_end_mm)
    pct = np.full_like(r, np.nan, dtype=float)
    pct_valid = np.isfinite(r) & np.isfinite(r_ref_per_point) & valid_core & (r_ref_per_point > 1e-8)
    pct[pct_valid] = (1.0 - r[pct_valid] / r_ref_per_point[pct_valid]) * 100.0
    return pct


def fill_reference_gaps(ref: np.ndarray) -> np.ndarray:
    """Fill NaN reference-radius values by nearest/interpolated finite values."""
    ref = np.asarray(ref, dtype=float)
    out = ref.copy()
    idx = np.flatnonzero(np.isfinite(out))
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[:] = out[idx[0]]
        return out
    missing = ~np.isfinite(out)
    if missing.any():
        out[missing] = np.interp(np.flatnonzero(missing), idx, out[idx])
    return out


def detect_stenosis_segments(
    s,
    r,
    threshold_pct,
    min_segment_mm,
    exclude_end_mm=0.0,
    max_internal_gap_mm: Optional[float] = None,
    pts=None,
    taper_fit_exclude_end_mm: Optional[float] = None,
    **_ignored,
) -> StenosisResult:
    """Detect stenosis segments along a vessel by comparing local radius to a fitted taper reference.

    Builds a per-point reference radius (excluding a wider end-zone and, if
    curvature points are given, siphon regions), then flags points whose radius
    drops *threshold_pct* below that reference. When ``TAPER_TWO_PASS`` is set,
    iteratively excludes detected lesion points from the reference fit and
    re-detects until the lesion mask stabilizes. Segments are merged via
    hysteresis (:func:`hysteresis_segments_from_min_length`) using a lower
    support threshold so a strict core lesion can extend into a softer halo.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    valid_core = stenosis_valid_mask_from_ends(s, exclude_end_mm)
    if not (np.isfinite(r) & valid_core).any():
        return StenosisResult(np.nan, np.nan, np.nan, [], [], np.full_like(r, np.nan, dtype=float))

    siphon = (
        compute_siphon_mask(np.asarray(pts), s)
        if pts is not None and len(np.asarray(pts)) == len(s)
        else None
    )

    # Idea A: use a larger end-zone exclusion for the fit so end-zone lesions
    # cannot pull the local percentile envelope down toward the lesion radius.
    fit_excl = float(STENOSIS_TAPER_FIT_EXCLUDE_END_MM) if taper_fit_exclude_end_mm is None else float(taper_fit_exclude_end_mm)
    fit_core = stenosis_valid_mask_from_ends(s, fit_excl) if fit_excl > float(exclude_end_mm) else valid_core

    r_ref_per_point = _compute_vessel_taper(
        s, r, valid_core,
        exclude_mask=siphon,
        fit_basis=fit_core,
        low_outlier_pct=threshold_pct,
        high_outlier_pct=ENLARGEMENT_THRESHOLD_PCT,
    )
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    max_gap = STENOSIS_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)

    # Idea B: iterative re-fit — exclude detected lesion points from the
    # reference fit, recompute, re-detect.  Repeat until the lesion mask
    # stabilises (convergence) or TAPER_TWO_PASS_MAX_ITERATIONS is reached.
    if TAPER_TWO_PASS:
        lesion_mask = np.zeros(len(r), dtype=bool)
        for _ in range(int(TAPER_TWO_PASS_MAX_ITERATIONS)):
            pct_iter = stenosis_raw_percent(s, r, r_ref_per_point, exclude_end_mm=0.0)
            core_iter = (pct_iter >= float(threshold_pct)) & np.isfinite(pct_iter) & valid_core
            support_iter = (
                (pct_iter >= min(float(STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
                & np.isfinite(pct_iter) & valid_core
            )
            segs_iter = hysteresis_segments_from_min_length(s, core_iter, support_iter, min_segment_mm, max_gap_mm=max_gap)
            new_mask = mask_from_segments(len(r), segs_iter) if segs_iter else np.zeros(len(r), dtype=bool)
            if np.array_equal(new_mask, lesion_mask):
                break  # converged — reference and lesion map are stable
            lesion_mask = new_mask
            if not lesion_mask.any():
                break
            excl_iter = (siphon | lesion_mask) if siphon is not None else lesion_mask
            r_ref_per_point = _compute_vessel_taper(
                s, r, valid_core,
                exclude_mask=excl_iter,
                fit_basis=fit_core & ~lesion_mask,
                low_outlier_pct=threshold_pct,
                high_outlier_pct=ENLARGEMENT_THRESHOLD_PCT,
            )
            r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    r_min = float(np.nanmin(np.where(valid_core, r, np.nan)))
    pct_per_point = stenosis_raw_percent(s, r, r_ref_per_point, exclude_end_mm=0.0)

    core = (pct_per_point >= float(threshold_pct)) & np.isfinite(pct_per_point) & valid_core
    support_domain = stenosis_valid_mask_from_ends(s, float(exclude_end_mm))
    support = (
        (pct_per_point >= min(float(STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
        & np.isfinite(pct_per_point)
        & support_domain
    )
    # Accept on the support contour (so a short severe lesion is not lost to the length rule),
    # then report on the core contour.
    segments_point_idx = clip_segments_to_core(
        hysteresis_segments_from_min_length(
            s, core, support, min_segment_mm, max_gap_mm=max_gap,
        ),
        core,
    )
    accepted = core & mask_from_segments(len(r), segments_point_idx)
    segments_s_mm = [(float(s[a]), float(s[b])) for a, b in segments_point_idx]
    reported_pct = stenosis_segment_percent(s, r, r_ref_per_point, segments_point_idx)
    pct_values = reported_pct[accepted & np.isfinite(reported_pct)]
    pct = float(np.nanmax(pct_values)) if len(pct_values) else 0.0
    rr_ref = r_ref_per_point[np.isfinite(r_ref_per_point)]
    r_ref = float(np.mean(rr_ref)) if len(rr_ref) else np.nan
    return StenosisResult(r_ref, r_min, pct, segments_s_mm, segments_point_idx, r_ref_per_point)


def stenosis_pointwise(
    s,
    r,
    r_ref_per_point,
    threshold_pct,
    exclude_end_mm=0.0,
    min_segment_mm: Optional[float] = None,
    max_internal_gap_mm: Optional[float] = None,
    **_ignored,
):
    """Per-point stenosis percentage and flag against an already-computed reference radius.

    Lighter-weight sibling of :func:`detect_stenosis_segments` for cases (e.g.
    interactive review) where the taper reference was already fit and just needs
    re-evaluating; optionally re-applies hysteresis segment merging.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    r_ref_per_point = np.asarray(r_ref_per_point, dtype=float)
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)
    valid_core = stenosis_valid_mask_from_ends(s, exclude_end_mm)
    pct = stenosis_raw_percent(s, r, r_ref_per_point, exclude_end_mm=0.0)
    is_stenotic = ((pct >= float(threshold_pct)) & np.isfinite(pct) & valid_core).astype(int)
    if min_segment_mm is not None:
        max_gap = STENOSIS_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)
        support_domain = stenosis_valid_mask_from_ends(s, float(exclude_end_mm))
        support = (
            (pct >= min(float(STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
            & np.isfinite(pct)
            & support_domain
        )
        core = is_stenotic == 1
        segments = clip_segments_to_core(
            hysteresis_segments_from_min_length(
                s,
                core,
                support,
                float(min_segment_mm),
                max_gap_mm=max_gap,
            ),
            core,
        )
        is_stenotic = (core & mask_from_segments(len(r), segments)).astype(int)
        pct_reported = stenosis_segment_percent(s, r, r_ref_per_point, segments)
        pct_reported[is_stenotic != 1] = 0.0
        return pct_reported, is_stenotic
    return pct, is_stenotic


def detect_enlargement_segments(
    s,
    r,
    threshold_pct,
    min_segment_mm,
    exclude_end_mm=0.0,
    max_internal_gap_mm: Optional[float] = None,
    pts=None,
    taper_fit_exclude_end_mm: Optional[float] = None,
    **_ignored,
) -> EnlargementResult:
    """Detect enlargement (aneurysm-like) segments: mirror of :func:`detect_stenosis_segments`.

    Same taper-reference-fit + iterative-refit + hysteresis-merge strategy, but
    flags points *above* the reference by *threshold_pct* instead of below it.
    Siphon (high-curvature) regions can optionally be excluded from detection
    (not just the reference fit) via ``SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION``,
    since curvature can locally widen the apparent radius without true enlargement.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    valid_core = stenosis_valid_mask_from_ends(s, exclude_end_mm)
    if not (np.isfinite(r) & valid_core).any():
        return EnlargementResult(np.nan, np.nan, np.nan, [], [], np.full_like(r, np.nan, dtype=float))

    siphon = (
        compute_siphon_mask(np.asarray(pts), s)
        if pts is not None and len(np.asarray(pts)) == len(s)
        else None
    )

    # Idea A: larger end-zone exclusion for the fit so end-zone aneurysms
    # cannot pull the local percentile envelope up toward the lesion radius.
    fit_excl = float(ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM) if taper_fit_exclude_end_mm is None else float(taper_fit_exclude_end_mm)
    fit_core = stenosis_valid_mask_from_ends(s, fit_excl) if fit_excl > float(exclude_end_mm) else valid_core

    r_ref_per_point = _compute_vessel_taper(
        s, r, valid_core,
        exclude_mask=siphon,
        fit_basis=fit_core,
        low_outlier_pct=STENOSIS_THRESHOLD_PCT,
        high_outlier_pct=threshold_pct,
    )
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)
    detect_core = (
        valid_core
        if (siphon is None or not SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION)
        else (valid_core & ~siphon)
    )

    max_gap = ENLARGEMENT_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)

    # Idea B: iterative re-fit — exclude detected lesion points from the
    # reference fit, recompute, re-detect.  Repeat until the lesion mask
    # stabilises (convergence) or TAPER_TWO_PASS_MAX_ITERATIONS is reached.
    if TAPER_TWO_PASS:
        lesion_mask = np.zeros(len(r), dtype=bool)
        for _ in range(int(TAPER_TWO_PASS_MAX_ITERATIONS)):
            pct_iter = np.full_like(r, np.nan, dtype=float)
            pct_valid_iter = detect_core & np.isfinite(r) & np.isfinite(r_ref_per_point) & (r_ref_per_point > 1e-8)
            pct_iter[pct_valid_iter] = (r[pct_valid_iter] / r_ref_per_point[pct_valid_iter] - 1.0) * 100.0
            core_iter = (pct_iter >= float(threshold_pct)) & np.isfinite(pct_iter) & detect_core
            support_iter = (
                (pct_iter >= min(float(ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
                & np.isfinite(pct_iter) & detect_core
            )
            segs_iter = hysteresis_segments_from_min_length(
                s, core_iter, support_iter, min_segment_mm, max_gap_mm=max_gap,
                min_points=points_for_length(s, ENLARGEMENT_MIN_SUPPORT_LENGTH_MM),
            )
            new_mask = mask_from_segments(len(r), segs_iter) if segs_iter else np.zeros(len(r), dtype=bool)
            if np.array_equal(new_mask, lesion_mask):
                break  # converged — reference and lesion map are stable
            lesion_mask = new_mask
            if not lesion_mask.any():
                break
            excl_iter = (siphon | lesion_mask) if siphon is not None else lesion_mask
            r_ref_per_point = _compute_vessel_taper(
                s, r, valid_core,
                exclude_mask=excl_iter,
                fit_basis=fit_core & ~lesion_mask,
                low_outlier_pct=STENOSIS_THRESHOLD_PCT,
                high_outlier_pct=threshold_pct,
            )
            r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    r_max = float(np.nanmax(np.where(valid_core, r, np.nan)))
    pct_per_point = np.full_like(r, np.nan, dtype=float)
    pct_valid = detect_core & np.isfinite(r) & np.isfinite(r_ref_per_point) & (r_ref_per_point > 1e-8)
    pct_per_point[pct_valid] = (r[pct_valid] / r_ref_per_point[pct_valid] - 1.0) * 100.0

    core = (pct_per_point >= float(threshold_pct)) & np.isfinite(pct_per_point) & detect_core
    support = (
        (pct_per_point >= min(float(ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
        & np.isfinite(pct_per_point)
        & detect_core
    )
    # Accept on the support contour, report on the core contour (mirrors the stenosis path).
    segments_point_idx = clip_segments_to_core(
        hysteresis_segments_from_min_length(
            s, core, support, min_segment_mm, max_gap_mm=max_gap,
            min_points=points_for_length(s, ENLARGEMENT_MIN_SUPPORT_LENGTH_MM),
        ),
        core,
    )
    accepted = core & mask_from_segments(len(r), segments_point_idx)
    segments_s_mm = [(float(s[a]), float(s[b])) for a, b in segments_point_idx]
    pct_values = pct_per_point[accepted & np.isfinite(pct_per_point)]
    pct = float(np.nanmax(pct_values)) if len(pct_values) else 0.0
    rr_ref = r_ref_per_point[np.isfinite(r_ref_per_point)]
    r_ref = float(np.mean(rr_ref)) if len(rr_ref) else np.nan
    return EnlargementResult(r_ref, r_max, pct, segments_s_mm, segments_point_idx, r_ref_per_point)


def enlargement_pointwise(
    s,
    r,
    r_ref_per_point,
    threshold_pct,
    exclude_end_mm=0.0,
    min_segment_mm: Optional[float] = None,
    max_internal_gap_mm: Optional[float] = None,
    pts=None,
    **_ignored,
):
    """Per-point enlargement percentage and flag against an already-computed reference radius.

    Mirror of :func:`stenosis_pointwise` for the enlargement (aneurysm) direction.
    """
    s = np.asarray(s, dtype=float)
    r = np.asarray(r, dtype=float)
    r_ref_per_point = np.asarray(r_ref_per_point, dtype=float)
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)
    valid_core = stenosis_valid_mask_from_ends(s, exclude_end_mm)
    siphon = (
        compute_siphon_mask(np.asarray(pts), s)
        if pts is not None and len(np.asarray(pts)) == len(s)
        else None
    )
    detect_core = (
        valid_core
        if (siphon is None or not SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION)
        else (valid_core & ~siphon)
    )
    # Compute pct over all valid_core so siphon-region points are not NaN in the output
    # (is_enlarged detection still uses detect_core, which excludes siphon when the flag is set).
    pct = np.full_like(r, np.nan, dtype=float)
    pct_valid = np.isfinite(r) & np.isfinite(r_ref_per_point) & valid_core & (r_ref_per_point > 1e-8)
    pct[pct_valid] = (r[pct_valid] / r_ref_per_point[pct_valid] - 1.0) * 100.0
    is_enlarged = ((pct >= float(threshold_pct)) & np.isfinite(pct) & detect_core).astype(int)
    if min_segment_mm is not None:
        max_gap = ENLARGEMENT_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)
        support = (
            (pct >= min(float(ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
            & np.isfinite(pct)
            & detect_core
        )
        core = is_enlarged == 1
        segments = clip_segments_to_core(
            hysteresis_segments_from_min_length(
                s,
                core,
                support,
                float(min_segment_mm),
                max_gap_mm=max_gap,
                min_points=points_for_length(s, ENLARGEMENT_MIN_SUPPORT_LENGTH_MM),
            ),
            core,
        )
        is_enlarged = (core & mask_from_segments(len(r), segments)).astype(int)
        # Return 0 outside enlarged segments (mirrors stenosis_pointwise behaviour so
        # downstream interpolation in anatomic-split VTPs never sees all-NaN slices).
        pct_reported = np.zeros_like(r, dtype=float)
        enlarged_finite = (is_enlarged == 1) & np.isfinite(pct)
        pct_reported[enlarged_finite] = pct[enlarged_finite]
        return pct_reported, is_enlarged
    return pct, is_enlarged


def resolve_stenosis_enlargement_overlap(
    stenosis_pct: np.ndarray,
    is_stenotic: np.ndarray,
    enlargement_pct: np.ndarray,
    is_enlarged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clear both flags at points where stenosis and enlargement detections coincide (an ambiguous fit, not a real lesion)."""
    stenosis_pct = np.asarray(stenosis_pct, dtype=float).copy()
    enlargement_pct = np.asarray(enlargement_pct, dtype=float).copy()
    is_stenotic = np.asarray(is_stenotic, dtype=int).copy()
    is_enlarged = np.asarray(is_enlarged, dtype=int).copy()
    overlap = (is_stenotic == 1) & (is_enlarged == 1)
    if not overlap.any():
        return stenosis_pct, is_stenotic, enlargement_pct, is_enlarged

    is_stenotic[overlap] = 0
    stenosis_pct[overlap] = 0.0
    is_enlarged[overlap] = 0
    enlargement_pct[overlap] = 0.0
    return stenosis_pct, is_stenotic, enlargement_pct, is_enlarged


def stenosis_total_length(s, flag) -> float:
    """Total arc length (mm) covered by consecutive-flagged point pairs."""
    if len(s) < 2:
        return 0.0
    return float(np.sum(np.diff(s)[(flag[:-1] & flag[1:]).astype(bool)]))


def flag_segments_from_min_length(
    s: np.ndarray,
    flag: np.ndarray,
    min_segment_mm: float,
    max_gap_mm: float = 0.0,
    min_points: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Group flagged points into ``(start, end)`` index segments, merging runs separated by ≤*max_gap_mm*.

    A merged segment survives if it spans ≥*min_segment_mm* of arc length OR
    (when given) has at least *min_points* points.
    """
    s = np.asarray(s, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    if len(s) == 0 or len(flag) != len(s) or not flag.any():
        return []
    idx = np.where(flag)[0]
    raw_segments = []
    start = prev = int(idx[0])
    for j in idx[1:]:
        j = int(j)
        if j == prev + 1:
            prev = j
            continue
        raw_segments.append((int(start), int(prev)))
        start = prev = j
    raw_segments.append((int(start), int(prev)))

    merged = []
    max_gap_mm = max(0.0, float(max_gap_mm))
    for start, end in raw_segments:
        if not merged:
            merged.append([start, end])
            continue
        prev_start, prev_end = merged[-1]
        gap_mm = float(s[start] - s[prev_end]) if start < len(s) and prev_end < len(s) else np.inf
        if gap_mm <= max_gap_mm:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        (int(start), int(end))
        for start, end in merged
        if (float(s[end] - s[start]) >= float(min_segment_mm))
        or (min_points is not None and (end - start + 1) >= int(min_points))
    ]


def hysteresis_segments_from_min_length(
    s: np.ndarray,
    core_flag: np.ndarray,
    support_flag: np.ndarray,
    min_segment_mm: float,
    max_gap_mm: float = 0.0,
    min_points: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Hysteresis segment merging: build segments from the looser *support_flag*, keep only those
    that contain at least one point from the stricter *core_flag*.
    """
    support_segments = flag_segments_from_min_length(
        s, support_flag, min_segment_mm, max_gap_mm=max_gap_mm, min_points=min_points,
    )
    core_flag = np.asarray(core_flag, dtype=bool)
    return [
        (start, end)
        for start, end in support_segments
        if bool(np.any(core_flag[int(start):int(end) + 1]))
    ]


def mask_from_segments(n_points: int, segments: List[Tuple[int, int]]) -> np.ndarray:
    """Boolean point mask of length *n_points* that is ``True`` inside each ``(start, end)`` segment."""
    keep = np.zeros(int(n_points), dtype=bool)
    for start, end in segments:
        keep[int(start):int(end) + 1] = True
    return keep


def points_for_length(s: np.ndarray, length_mm: Optional[float]) -> Optional[int]:
    """Point count spanning *length_mm* at this path's own sampling step, or ``None``.

    Lets a policy be stated in mm while :func:`flag_segments_from_min_length` still counts points,
    so changing ``CENTERLINE_RESAMPLE_STEP_MM`` cannot silently retune a detection rule.
    """
    if length_mm is None:
        return None
    s = np.asarray(s, dtype=float)
    if s.size < 2:
        return None
    step = float(np.nanmedian(np.diff(s)))
    if not np.isfinite(step) or step <= 1e-9:
        return None
    # A run of n points spans (n-1) steps. The epsilon absorbs the float error in the median step
    # (an arange at 0.1 mm medians to 0.0999…, which would otherwise round the count up by one).
    n_intervals = int(np.ceil(float(length_mm) / step - 1e-6))
    return max(2, n_intervals + 1)


def clip_segments_to_core(
    segments: List[Tuple[int, int]], core_flag: np.ndarray
) -> List[Tuple[int, int]]:
    """Trim each accepted segment to the span between its first and last core point.

    The support contour decides *whether* a lesion is real and merges runs across small internal
    dips; it should not widen what gets **reported**. Reporting the halo would inflate
    ``*_length_total_mm`` and make "stenosis length" mean something other than "length narrowed by
    at least the core threshold". Trimming keeps the merge (an internal sub-core dip stays inside
    one segment) while pinning both ends to the core contour.

    Segments holding no core point are dropped — they cannot occur via
    :func:`hysteresis_segments_from_min_length`, which already requires one, but the guard keeps
    this usable on any segment list.
    """
    core_flag = np.asarray(core_flag, dtype=bool)
    out: List[Tuple[int, int]] = []
    for start, end in segments:
        start, end = int(start), int(end)
        local = np.flatnonzero(core_flag[start:end + 1])
        if local.size == 0:
            continue
        out.append((start + int(local[0]), start + int(local[-1])))
    return out


def filter_flag_by_min_length(
    s: np.ndarray,
    flag: np.ndarray,
    min_segment_mm: float,
    max_gap_mm: float = 0.0,
) -> np.ndarray:
    """Drop flagged runs shorter than *min_segment_mm* (single-threshold version of the hysteresis merge)."""
    flag = np.asarray(flag, dtype=bool)
    segments = flag_segments_from_min_length(s, flag, min_segment_mm, max_gap_mm=max_gap_mm)
    return flag & mask_from_segments(len(flag), segments)


def segment_detail_json(s: np.ndarray, percent_point: np.ndarray, segments_point_idx: List[Tuple[int, int]]) -> str:
    """JSON list of per-segment detail (index range, arc-length span, peak percent) for export columns."""
    s = np.asarray(s, dtype=float)
    percent_point = np.asarray(percent_point, dtype=float)
    details = []
    for start, end in segments_point_idx:
        start = int(start)
        end = int(end)
        if start < 0 or end >= len(s) or end < start:
            continue
        seg_pct = percent_point[start:end + 1]
        finite_pct = seg_pct[np.isfinite(seg_pct)]
        details.append({
            "point_idx": [start, end],
            "s_start_mm": float(s[start]),
            "s_end_mm": float(s[end]),
            "length_mm": float(max(0.0, s[end] - s[start])),
            "degree_pct": float(np.nanmax(finite_pct)) if len(finite_pct) else np.nan,
        })
    return json.dumps(details)


def select_caliber_detection_radius(cross_section_radius: np.ndarray, mis_radius: Optional[np.ndarray] = None) -> np.ndarray:
    """Pick the radius signal used for stenosis/enlargement detection, per ``RADIUS_SOURCE_FOR_CALIBER_DETECTION``.

    Falls back to *cross_section_radius* wherever the maximum-inscribed-sphere
    (VMTK MIS) radius is unavailable or non-finite.
    """
    cross_section_radius = np.asarray(cross_section_radius, dtype=float)
    if str(RADIUS_SOURCE_FOR_CALIBER_DETECTION).lower() in {"maximum_inscribed_sphere", "mis", "vmtk"}:
        if mis_radius is not None:
            mis_radius = np.asarray(mis_radius, dtype=float)
            valid = np.isfinite(mis_radius) & (mis_radius > 1e-6)
            if valid.any():
                out = cross_section_radius.copy()
                out[valid] = mis_radius[valid]
                return out
    return cross_section_radius.copy()


def refresh_enlargement_summary_from_flags(res: dict) -> None:
    """In-place: recompute a path result's summary enlargement fields from its per-point flags/percent arrays."""
    s = np.asarray(res.get("s_mm", []), dtype=float)
    radius = np.asarray(res.get("radius_mm", []), dtype=float)
    pct = np.asarray(res.get("enlargement_percent_point", np.full(len(s), np.nan)), dtype=float)
    flag = np.asarray(res.get("is_enlarged", np.zeros(len(s), dtype=int)), dtype=int)
    if len(s) == 0 or len(flag) != len(s):
        return
    pct = np.where(flag == 1, np.nan_to_num(pct, nan=0.0), 0.0)
    segments = flag_segments_from_min_length(
        s,
        flag == 1,
        ENLARGEMENT_MIN_LEN_MM,
        max_gap_mm=ENLARGEMENT_MAX_INTERNAL_GAP_MM,
        min_points=points_for_length(s, ENLARGEMENT_MIN_SUPPORT_LENGTH_MM),
    )
    if segments:
        keep = np.zeros(len(flag), dtype=bool)
        for start, end in segments:
            keep[int(start):int(end) + 1] = True
        flag = (keep & (flag == 1)).astype(int)
        pct = np.where(flag == 1, pct, 0.0)
    else:
        flag = np.zeros(len(flag), dtype=int)
        pct = np.zeros(len(flag), dtype=float)

    enlarged_radius = radius[(flag == 1) & np.isfinite(radius)]
    enlarged_pct = pct[(flag == 1) & np.isfinite(pct)]
    res["enlargement_percent_point"] = pct
    res["is_enlarged"] = flag
    res["enlargement_percent_max"] = float(np.max(enlarged_pct)) if enlarged_pct.size else 0.0
    res["enlargement_length_total_mm"] = float(stenosis_total_length(s, flag))
    res["enlargement_segments_n"] = int(len(segments))
    res["enlargement_segments_point_idx"] = json.dumps(segments)
    res["enlargement_segments_detail_json"] = segment_detail_json(s, pct, segments)
    res["radius_max_enlarged_mm"] = float(np.max(enlarged_radius)) if enlarged_radius.size else np.nan
