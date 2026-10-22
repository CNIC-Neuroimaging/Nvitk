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

from nvitk_vendor.measure import morphometrics_config as _cfg
from .metrics import discrete_curvature, smooth_1d, smoothed_curvature
from .models import EnlargementResult, StenosisResult

def enlargement_min_length_mm(min_segment_mm: Optional[float] = None) -> float:
    """Shortest accepted enlargement: *min_segment_mm* (default ``ENLARGEMENT_MIN_LEN_MM``), or the
    alternative ``ENLARGEMENT_MIN_SUPPORT_LENGTH_MM`` when that is shorter."""
    base = float(_cfg.ENLARGEMENT_MIN_LEN_MM if min_segment_mm is None else min_segment_mm)
    alt = _cfg.ENLARGEMENT_MIN_SUPPORT_LENGTH_MM
    return min(base, float(alt)) if alt is not None else base


def detection_radius_is_cross_section() -> bool:
    """Whether the caliber detection radius is the cross-section one (``RADIUS_SOURCE_FOR_CALIBER_DETECTION``)."""
    return str(_cfg.RADIUS_SOURCE_FOR_CALIBER_DETECTION).lower() not in {"maximum_inscribed_sphere", "mis", "vmtk"}


def enlargement_cross_section_radius(s, r) -> np.ndarray:
    """The cross-section radius an enlargement test reads: *r* itself, or — with
    ``ENLARGEMENT_SMOOTH_CROSS_SECTION`` — its running median over ``ENLARGEMENT_CROSS_SECTION_MEDIAN_MM``
    of arc length (NaNs ignored; the ends padded with their own value).

    A single bad cut (the plane catching a branch opening, or oblique in a bend) spikes the
    area-equivalent radius for one or two 0.1 mm samples; the median removes it, while a bulge
    longer than half the window survives.
    """
    import warnings

    r = np.asarray(r, dtype=float)
    if not bool(_cfg.ENLARGEMENT_SMOOTH_CROSS_SECTION) or len(r) < 3:
        return r.copy()
    s = np.asarray(s, dtype=float)
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 0.0
    if not np.isfinite(ds) or ds <= 0.0:
        return r.copy()
    half = max(1, int(round(0.5 * float(_cfg.ENLARGEMENT_CROSS_SECTION_MEDIAN_MM) / ds)))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(r, half, mode="edge"), 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows stay NaN
        out = np.nanmedian(windows, axis=1)
    return np.where(np.isfinite(out), out, r)


def enlargement_radius(s, cross_section_r, detection_r, *, cow_source: str) -> Tuple[np.ndarray, str]:
    """The radius one measurement's enlargement test reads, and its source (``"cross_section"`` or
    ``"detection"``), per ``ENLARGEMENT_RADIUS_SOURCE``.

    *cow_source* is what cow_morpho reads on the calling code path, used under ``"cow"``. A
    cross-section signal goes through :func:`enlargement_cross_section_radius` (the optional running
    median), including a detection radius that is itself the cross-section one.
    """
    source = str(_cfg.ENLARGEMENT_RADIUS_SOURCE).lower()
    if source not in _cfg.ENLARGEMENT_RADIUS_SOURCES:
        raise ValueError(f"ENLARGEMENT_RADIUS_SOURCE must be one of {_cfg.ENLARGEMENT_RADIUS_SOURCES}, got {source!r}.")
    if source == "cow":
        source = cow_source
    if source == "detection" and not detection_radius_is_cross_section():
        return np.asarray(detection_r, dtype=float).copy(), source
    return enlargement_cross_section_radius(s, detection_r if source == "detection" else cross_section_r), source


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
    """Boolean mask of high-curvature (siphon) regions, dilated by dilation_mm.

    Curvature is measured on the centerline smoothed at ``SIPHON_CURVATURE_SMOOTH_MM`` (anatomical
    scale). ``SIPHON_LEGACY_MASK`` restores cow_morpho's measurement (0.1 mm Menger curvature,
    ``INFLECT_SMOOTH_WIN``-point moving average), whose sampling noise exceeds the 0.10 mm⁻¹
    threshold almost everywhere — the mask then covers whole vessels and the taper fit collapses
    to a flat median reference.
    """
    if kappa_threshold is None:
        kappa_threshold = (
            _cfg.SIPHON_LEGACY_KAPPA_THRESHOLD if _cfg.SIPHON_LEGACY_MASK else _cfg.SIPHON_KAPPA_THRESHOLD
        )
    dilation_mm = _cfg.SIPHON_DILATION_MM if dilation_mm is None else dilation_mm
    pts = np.asarray(pts, dtype=float)
    if _cfg.SIPHON_LEGACY_MASK:
        kappa = discrete_curvature(pts)
        kappa_sm = smooth_1d(np.where(np.isfinite(kappa), kappa, 0.0), win=_cfg.INFLECT_SMOOTH_WIN)
    else:
        kappa = smoothed_curvature(pts, float(_cfg.SIPHON_CURVATURE_SMOOTH_MM))
        kappa_sm = np.where(np.isfinite(kappa), kappa, 0.0)
    high_k = kappa_sm >= kappa_threshold
    if not high_k.any() or dilation_mm <= 0.0:
        return high_k
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    if ds < 1e-6:
        ds = 1.0
    radius = max(1, int(np.round(dilation_mm / ds)))
    struct = np.ones(2 * radius + 1, dtype=bool)
    return ndi.binary_dilation(high_k, structure=struct)


# The junctions (branch points, mm) of the component being measured: set by the orchestration
# around one component, read by the stretch-wise taper reference of every path measured in it.
_JUNCTIONS: dict = {"points": None}


def set_detection_junctions(points_mm: Optional[np.ndarray]) -> None:
    """Install the junctions of the component about to be measured (``None`` clears them)."""
    pts = None if points_mm is None else np.asarray(points_mm, dtype=float).reshape(-1, 3)
    _JUNCTIONS["points"] = pts if pts is not None and len(pts) else None


def detection_junctions() -> Optional[np.ndarray]:
    """The junctions installed by :func:`set_detection_junctions` (``None`` when none)."""
    return _JUNCTIONS["points"]


def blanking_mask(points: np.ndarray, radius: np.ndarray, junctions: Optional[np.ndarray]) -> np.ndarray:
    """True where a point lies within the bifurcation blanking zone of any junction:
    ``max(BLANKING_MM, BLANKING_RADIUS_FACTOR × local radius)``."""
    n = len(points)
    if junctions is None or len(junctions) == 0 or n == 0:
        return np.zeros(n, dtype=bool)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(junctions, dtype=float)).query(np.asarray(points, dtype=float))
    r = np.where(np.isfinite(radius), radius, 0.0)
    zone = np.maximum(float(_cfg.BLANKING_MM), float(_cfg.BLANKING_RADIUS_FACTOR) * r)
    return d <= zone


def junction_crossings(points: np.ndarray, radius: np.ndarray, junctions: Optional[np.ndarray]) -> np.ndarray:
    """Sorted indices of the path points nearest each junction the path passes through (the
    junction lies within that point's blanking zone)."""
    pts = np.asarray(points, dtype=float)
    if junctions is None or len(junctions) == 0 or len(pts) == 0:
        return np.empty(0, dtype=int)
    from scipy.spatial import cKDTree

    d, idx = cKDTree(pts).query(np.asarray(junctions, dtype=float))
    r = np.where(np.isfinite(radius), radius, 0.0)
    zone = np.maximum(float(_cfg.BLANKING_MM), float(_cfg.BLANKING_RADIUS_FACTOR) * r[idx])
    return np.unique(idx[d <= zone])


def path_stretches(pts: Optional[np.ndarray], r: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """``(stretch id per point, blanking mask)`` for the ``stretch`` taper reference: the path cut
    at the installed junctions it passes through. A stretch shorter than
    ``TAPER_STRETCH_MIN_LENGTH_MM`` joins its upstream neighbour (the first one, its downstream
    neighbour): two branch points close together would otherwise give a few millimetres — often
    exactly a narrowing — a reference of their own. ``(None, all-False)`` for the ``envelope``
    method or without points."""
    n = len(r)
    if str(_cfg.TAPER_REFERENCE_METHOD).lower() != "stretch" or pts is None or len(pts) != n:
        return None, np.zeros(n, dtype=bool)
    pts = np.asarray(pts, dtype=float)
    junctions = detection_junctions()
    cuts = list(junction_crossings(pts, r, junctions))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]) if n > 1 else np.zeros(n)
    min_len = float(_cfg.TAPER_STRETCH_MIN_LENGTH_MM)
    while cuts:
        bounds = [0, *cuts, n - 1]
        lengths = [s[bounds[k + 1]] - s[bounds[k]] for k in range(len(bounds) - 1)]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_len:
            break
        cuts.pop(k - 1 if k > 0 else 0)  # drop the cut upstream of it (the first: downstream)
    stretch = np.searchsorted(np.asarray(cuts, dtype=int), np.arange(n), side="left").astype(int)
    return stretch, blanking_mask(pts, r, junctions)


def taper_stretch_ids(pts: Optional[np.ndarray], r: np.ndarray) -> np.ndarray:
    """Per-point stretch ids of :func:`path_stretches` as floats for the point tables (all 0 for
    the ``envelope`` method): where the taper reference may step (lesion inspection)."""
    stretch, _ = path_stretches(pts, np.asarray(r, dtype=float))
    return np.zeros(len(r)) if stretch is None else stretch.astype(float)


def _quantile_line(s: np.ndarray, r: np.ndarray, tau: float) -> Tuple[float, float]:
    """``(a, b)`` of the τ-quantile line ``r ≈ a + b·s`` (pinball loss), its slope limited to
    ``[−TAPER_STRETCH_MAX_DECLINE_PCT_PER_MM % × level, 0]`` (non-increasing when
    ``TAPER_FIT_ENFORCE_NONINCREASING``; ``level`` = the τ-quantile radius)."""
    level = float(np.quantile(r, tau))
    cap = _cfg.TAPER_STRETCH_MAX_DECLINE_PCT_PER_MM
    span = float(s.max() - s.min())
    b_min = -(float(cap) / 100.0) * level if cap is not None else (-(float(np.ptp(r)) / span * 2.0 if span > 0 else 0.0))
    b_max = 0.0 if _cfg.TAPER_FIT_ENFORCE_NONINCREASING else -b_min
    if span <= 1e-6 or b_min >= b_max:
        return level, 0.0
    best = (np.inf, level, 0.0)
    lo, hi = b_min, b_max
    for _ in range(2):  # a coarse grid, then a finer one around the best slope
        slopes = np.linspace(lo, hi, 41)
        resid = r[None, :] - slopes[:, None] * s[None, :]
        a = np.quantile(resid, tau, axis=1)
        u = resid - a[:, None]
        loss = np.sum(np.where(u >= 0, tau * u, (tau - 1.0) * u), axis=1)
        k = int(np.argmin(loss))
        if loss[k] < best[0]:
            best = (float(loss[k]), float(a[k]), float(slopes[k]))
        step = (hi - lo) / 40.0
        lo, hi = max(b_min, slopes[k] - step), min(b_max, slopes[k] + step)
    return best[1], best[2]


def _local_linear_envelope(s_fit: np.ndarray, r_fit: np.ndarray, knots: np.ndarray, tau: float) -> np.ndarray:
    """At each knot, the τ-quantile line of the fit points within ±``TAPER_REFERENCE_WINDOW_MM``/2
    evaluated there (NaN with fewer than 3 points): a local-*linear* envelope, which follows a
    taper without the lead a local percentile has on a sloping vessel, and continues the local
    slope past the last point."""
    half = max(float(_cfg.TAPER_REFERENCE_WINDOW_MM) * 0.5, 1e-6)
    out = np.full(len(knots), np.nan)
    for j, c in enumerate(knots):
        local = np.abs(s_fit - c) <= half
        if local.sum() >= 3:
            out[j] = _quantile_line(s_fit[local] - c, r_fit[local], tau)[0]
    return out


def _stretch_reference(
    s: np.ndarray,
    r: np.ndarray,
    valid_mask: np.ndarray,
    fit_mask: np.ndarray,
    stretch: np.ndarray,
    *,
    envelope: bool = True,
) -> np.ndarray:
    """Healthy calibre stretch by stretch (``TAPER_REFERENCE_METHOD = "stretch"``).

    In each junction-to-junction stretch, the reference is the envelope method's local
    ``TAPER_REFERENCE_PERCENTILE`` envelope made **local-linear** (:func:`_local_linear_envelope`,
    every 0.5 mm): at each place, the upper-quantile line of the stretch's fit points within
    ``TAPER_REFERENCE_WINDOW_MM``, its decline capped at ``TAPER_STRETCH_MAX_DECLINE_PCT_PER_MM``.
    It follows a natural taper (and a flat-then-declining vessel) without the ~7 mm lead a local
    percentile has on a slope, and is then smoothed and kept non-increasing. At the path's distal
    end it continues the local decline into the vessel's end instead of freezing (before a junction
    it is held); gaps inside the stretch (siphon,
    excluded lesions, junction zones) are bridged linearly from both sides; upstream of the first
    fitted point it is held, never extrapolated upwards. *envelope* False gives the outlier pre-pass its centre
    line (the local median). A stretch whose fit points span less than
    ``TAPER_STRETCH_MIN_LENGTH_MM`` gets the median of its valid radius, and so does a whole vessel
    shorter than ``TAPER_STRETCH_SHORT_VESSEL_MM`` (the envelope method's fallback when nothing is
    left to fit); a stretch with fewer than 3 fit points is left NaN (filled from its neighbours by
    :func:`fill_reference_gaps`). The reference may step at a junction.
    """
    tau = float(_cfg.TAPER_REFERENCE_PERCENTILE) / 100.0 if envelope else 0.5
    out = np.full_like(r, np.nan, dtype=float)
    finite = np.isfinite(s) & np.isfinite(r)
    if len(s) and float(np.nanmax(s) - np.nanmin(s)) < float(_cfg.TAPER_STRETCH_SHORT_VESSEL_MM):
        # A short vessel (A1, PComm, AChA…) has no taper to fit: its median, as the envelope
        # method (whose 10 mm end exclusions leave such a vessel nothing to fit).
        vals = r[valid_mask & finite]
        out[:] = float(np.median(vals)) if vals.size else np.nan
        return np.where(valid_mask, np.clip(out, 1e-6, None), np.nan).astype(float)
    for k in np.unique(stretch):
        in_k = stretch == k
        fk = in_k & fit_mask & finite
        if fk.sum() < 3:
            continue
        idx = np.flatnonzero(in_k)
        if float(s[fk].max() - s[fk].min()) < float(_cfg.TAPER_STRETCH_MIN_LENGTH_MM):
            # Too little vessel for a taper (a short vessel on its own): the median of its valid
            # radius, as the envelope method falls back to when nothing is left to fit.
            vals = r[in_k & valid_mask & finite]
            out[idx] = float(np.median(vals)) if vals.size else np.nan
            continue
        s_k = s[idx]
        knots = np.arange(s_k[0], s_k[-1] + 0.5, 0.5)
        env = _local_linear_envelope(s[fk], r[fk], knots, tau)
        # Only where the radius was fitted, and past the last fitted point of the path's last
        # stretch (the natural decline into the vessel's end; before a junction the reference is
        # held). Gaps inside the stretch (siphon, excluded lesions,
        # junction zones) are bridged linearly from both sides rather than by one side's slope;
        # upstream of the first fitted point the reference is held, never extrapolated upwards.
        s_fit = np.sort(s[fk])
        near = np.abs(s_fit[np.clip(np.searchsorted(s_fit, knots), 0, len(s_fit) - 1)] - knots)
        near = np.minimum(near, np.abs(s_fit[np.clip(np.searchsorted(s_fit, knots) - 1, 0, len(s_fit) - 1)] - knots))
        last = k == stretch.max()  # only the vessel's own end declines on; a junction is held
        covered = (near <= 1.0) | ((knots > s_fit[-1]) & last)
        env[~covered | (knots < s_fit[0])] = np.nan
        ok = np.isfinite(env)
        ref_k = np.interp(s_k, knots[ok], env[ok])  # bridges gaps; holds past the end knots
        if envelope and _cfg.TAPER_REFERENCE_SMOOTH_MM > 0 and len(idx) >= 3:
            ds = float(np.nanmedian(np.diff(s_k))) if len(idx) > 1 else 1.0
            win = max(3, int(round(float(_cfg.TAPER_REFERENCE_SMOOTH_MM) / ds))) if ds > 1e-6 else 3
            win = min(win | 1, (len(idx) - 1) | 1 if len(idx) > 3 else 3)
            ref_k = smooth_1d(ref_k, win=win)
        if _cfg.TAPER_FIT_ENFORCE_NONINCREASING:
            ref_k = _enforce_nonincreasing_reference(ref_k, np.ones(len(idx), dtype=bool))
        out[idx] = ref_k
    return np.where(valid_mask, np.clip(out, 1e-6, None), np.nan).astype(float)


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

    half_window = max(float(_cfg.TAPER_REFERENCE_WINDOW_MM) * 0.5, 1e-6)
    fit_s = s[fit_mask]
    fit_r = r[fit_mask]
    whole_fit_percentile = float(np.nanpercentile(fit_r, _cfg.TAPER_REFERENCE_PERCENTILE))

    for i in np.flatnonzero(valid_mask & np.isfinite(s)):
        local = np.abs(fit_s - s[i]) <= half_window
        if local.sum() >= 3:
            out[i] = float(np.nanpercentile(fit_r[local], _cfg.TAPER_REFERENCE_PERCENTILE))
        else:
            out[i] = whole_fit_percentile

    if _cfg.TAPER_REFERENCE_SMOOTH_MM > 0 and len(out) >= 3:
        ds = float(np.nanmedian(np.diff(s))) if len(s) > 1 else 1.0
        if not np.isfinite(ds) or ds <= 1e-6:
            ds = 1.0
        win = max(3, int(round(float(_cfg.TAPER_REFERENCE_SMOOTH_MM) / ds)))
        win = win | 1
        filled = out.copy()
        idx = np.flatnonzero(np.isfinite(filled))
        if len(idx) >= 2:
            filled = np.interp(np.arange(len(out)), idx, filled[idx])
            out[valid_mask] = smooth_1d(filled, win=win)[valid_mask]

    if _cfg.TAPER_FIT_ENFORCE_NONINCREASING:
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
    if _cfg.TAPER_FIT_ENFORCE_NONINCREASING and slope > 0:
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
    stretch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reference radius via an iterative healthy-caliber envelope.

    stretch: per-point junction-to-junction stretch ids (``TAPER_REFERENCE_METHOD = "stretch"``,
        :func:`path_stretches`): the line pre-fit and the envelope are then computed stretch by
        stretch (:func:`_stretch_reference`); ``None`` keeps cow_morpho's whole-path envelope.

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

    low_cut = None if low_outlier_pct is None else float(low_outlier_pct) * float(_cfg.TAPER_FIT_OUTLIER_FRACTION)
    high_cut = None if high_outlier_pct is None else float(high_outlier_pct) * float(_cfg.TAPER_FIT_OUTLIER_FRACTION)
    min_fit_points = max(3, int(np.ceil(float(_cfg.TAPER_FIT_MIN_HEALTHY_FRACTION) * float(original_fit_mask.sum()))))

    if stretch is not None:
        def _envelope(fm: np.ndarray) -> np.ndarray:
            return fill_reference_gaps(_stretch_reference(s, r, valid_mask, fm, stretch))

        pre_ref = fill_reference_gaps(_stretch_reference(s, r, valid_mask, fit_mask, stretch, envelope=False))
    else:
        def _envelope(fm: np.ndarray) -> np.ndarray:
            return _healthy_radius_envelope(s, r, valid_mask, fm)

        pre_ref = _linear_taper_reference(s, r, valid_mask, fit_mask)
    pre_deviation_pct = (r / pre_ref - 1.0) * 100.0
    prehealthy = original_fit_mask.copy()
    if low_cut is not None:
        prehealthy &= ~(np.isfinite(pre_deviation_pct) & (pre_deviation_pct < -low_cut))
    if high_cut is not None:
        prehealthy &= ~(np.isfinite(pre_deviation_pct) & (pre_deviation_pct > high_cut))
    if prehealthy.sum() >= min_fit_points:
        fit_mask = prehealthy

    ref = _envelope(fit_mask)
    for _ in range(max(1, int(_cfg.TAPER_FIT_MAX_ITERATIONS))):
        if fit_mask.sum() < min_fit_points:
            fit_mask = original_fit_mask.copy()
        if fit_mask.sum() < 3:
            return _constant_ref()

        ref = _envelope(fit_mask)
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
    margin = _cfg.STENOSIS_SEGMENT_REFERENCE_MARGIN_MM if margin_mm is None else float(margin_mm)
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
    fit_excl = float(_cfg.STENOSIS_TAPER_FIT_EXCLUDE_END_MM) if taper_fit_exclude_end_mm is None else float(taper_fit_exclude_end_mm)
    fit_core = stenosis_valid_mask_from_ends(s, fit_excl) if fit_excl > float(exclude_end_mm) else valid_core
    # TAPER_REFERENCE_METHOD = "stretch": junction-to-junction stretches, junction zones unfitted.
    stretch, blank = path_stretches(None if pts is None else np.asarray(pts), r)
    if stretch is not None:
        # The local-linear envelope sees the whole stretch but its tips and junction zones: it
        # carries the local decline into the end zones rather than freezing there.
        fit_core = valid_core & ~blank

    r_ref_per_point = _compute_vessel_taper(
        s, r, valid_core,
        exclude_mask=siphon,
        fit_basis=fit_core,
        low_outlier_pct=threshold_pct,
        high_outlier_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT,
        stretch=stretch,
    )
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    max_gap = _cfg.STENOSIS_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)

    # Idea B: iterative re-fit — exclude detected lesion points from the
    # reference fit, recompute, re-detect.  Repeat until the lesion mask
    # stabilises (convergence) or TAPER_TWO_PASS_MAX_ITERATIONS is reached.
    if _cfg.TAPER_TWO_PASS:
        lesion_mask = np.zeros(len(r), dtype=bool)
        for _ in range(int(_cfg.TAPER_TWO_PASS_MAX_ITERATIONS)):
            pct_iter = stenosis_raw_percent(s, r, r_ref_per_point, exclude_end_mm=0.0)
            core_iter = (pct_iter >= float(threshold_pct)) & np.isfinite(pct_iter) & valid_core
            support_iter = (
                (pct_iter >= min(float(_cfg.STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
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
                high_outlier_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT,
                stretch=stretch,
            )
            r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    r_min = float(np.nanmin(np.where(valid_core, r, np.nan)))
    pct_per_point = stenosis_raw_percent(s, r, r_ref_per_point, exclude_end_mm=0.0)

    core = (pct_per_point >= float(threshold_pct)) & np.isfinite(pct_per_point) & valid_core
    support_domain = stenosis_valid_mask_from_ends(s, float(exclude_end_mm))
    support = (
        (pct_per_point >= min(float(_cfg.STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
        & np.isfinite(pct_per_point)
        & support_domain
    )
    segments_point_idx = hysteresis_segments_from_min_length(
        s, core, support, min_segment_mm, max_gap_mm=max_gap,
    )
    accepted = support & mask_from_segments(len(r), segments_point_idx)
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
        max_gap = _cfg.STENOSIS_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)
        support_domain = stenosis_valid_mask_from_ends(s, float(exclude_end_mm))
        support = (
            (pct >= min(float(_cfg.STENOSIS_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
            & np.isfinite(pct)
            & support_domain
        )
        segments = hysteresis_segments_from_min_length(
            s,
            is_stenotic == 1,
            support,
            float(min_segment_mm),
            max_gap_mm=max_gap,
        )
        is_stenotic = (support & mask_from_segments(len(r), segments)).astype(int)
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
    fit_excl = float(_cfg.ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM) if taper_fit_exclude_end_mm is None else float(taper_fit_exclude_end_mm)
    fit_core = stenosis_valid_mask_from_ends(s, fit_excl) if fit_excl > float(exclude_end_mm) else valid_core
    # TAPER_REFERENCE_METHOD = "stretch": junction-to-junction stretches, junction zones unfitted.
    stretch, blank = path_stretches(None if pts is None else np.asarray(pts), r)
    if stretch is not None:
        # The local-linear envelope sees the whole stretch but its tips and junction zones: it
        # carries the local decline into the end zones rather than freezing there.
        fit_core = valid_core & ~blank

    r_ref_per_point = _compute_vessel_taper(
        s, r, valid_core,
        exclude_mask=siphon,
        fit_basis=fit_core,
        low_outlier_pct=_cfg.STENOSIS_THRESHOLD_PCT,
        high_outlier_pct=threshold_pct,
        stretch=stretch,
    )
    r_ref_per_point = fill_reference_gaps(r_ref_per_point)
    detect_core = (
        valid_core
        if (siphon is None or not _cfg.SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION)
        else (valid_core & ~siphon)
    )

    max_gap = _cfg.ENLARGEMENT_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)

    # Idea B: iterative re-fit — exclude detected lesion points from the
    # reference fit, recompute, re-detect.  Repeat until the lesion mask
    # stabilises (convergence) or TAPER_TWO_PASS_MAX_ITERATIONS is reached.
    if _cfg.TAPER_TWO_PASS:
        lesion_mask = np.zeros(len(r), dtype=bool)
        for _ in range(int(_cfg.TAPER_TWO_PASS_MAX_ITERATIONS)):
            pct_iter = np.full_like(r, np.nan, dtype=float)
            pct_valid_iter = detect_core & np.isfinite(r) & np.isfinite(r_ref_per_point) & (r_ref_per_point > 1e-8)
            pct_iter[pct_valid_iter] = (r[pct_valid_iter] / r_ref_per_point[pct_valid_iter] - 1.0) * 100.0
            core_iter = (pct_iter >= float(threshold_pct)) & np.isfinite(pct_iter) & detect_core
            support_iter = (
                (pct_iter >= min(float(_cfg.ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
                & np.isfinite(pct_iter) & detect_core
            )
            segs_iter = hysteresis_segments_from_min_length(
                s, core_iter, support_iter, enlargement_min_length_mm(min_segment_mm), max_gap_mm=max_gap,
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
                low_outlier_pct=_cfg.STENOSIS_THRESHOLD_PCT,
                high_outlier_pct=threshold_pct,
                stretch=stretch,
            )
            r_ref_per_point = fill_reference_gaps(r_ref_per_point)

    r_max = float(np.nanmax(np.where(valid_core, r, np.nan)))
    pct_per_point = np.full_like(r, np.nan, dtype=float)
    pct_valid = detect_core & np.isfinite(r) & np.isfinite(r_ref_per_point) & (r_ref_per_point > 1e-8)
    pct_per_point[pct_valid] = (r[pct_valid] / r_ref_per_point[pct_valid] - 1.0) * 100.0

    core = (pct_per_point >= float(threshold_pct)) & np.isfinite(pct_per_point) & detect_core
    support = (
        (pct_per_point >= min(float(_cfg.ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
        & np.isfinite(pct_per_point)
        & detect_core
    )
    segments_point_idx = hysteresis_segments_from_min_length(
        s, core, support, enlargement_min_length_mm(min_segment_mm), max_gap_mm=max_gap,
    )
    accepted = support & mask_from_segments(len(r), segments_point_idx)
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
        if (siphon is None or not _cfg.SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION)
        else (valid_core & ~siphon)
    )
    # Compute pct over all valid_core so siphon-region points are not NaN in the output
    # (is_enlarged detection still uses detect_core, which excludes siphon when the flag is set).
    pct = np.full_like(r, np.nan, dtype=float)
    pct_valid = np.isfinite(r) & np.isfinite(r_ref_per_point) & valid_core & (r_ref_per_point > 1e-8)
    pct[pct_valid] = (r[pct_valid] / r_ref_per_point[pct_valid] - 1.0) * 100.0
    is_enlarged = ((pct >= float(threshold_pct)) & np.isfinite(pct) & detect_core).astype(int)
    if min_segment_mm is not None:
        max_gap = _cfg.ENLARGEMENT_MAX_INTERNAL_GAP_MM if max_internal_gap_mm is None else float(max_internal_gap_mm)
        support = (
            (pct >= min(float(_cfg.ENLARGEMENT_SUPPORT_THRESHOLD_PCT), float(threshold_pct)))
            & np.isfinite(pct)
            & detect_core
        )
        segments = hysteresis_segments_from_min_length(
            s,
            is_enlarged == 1,
            support,
            enlargement_min_length_mm(min_segment_mm),
            max_gap_mm=max_gap,
        )
        is_enlarged = (support & mask_from_segments(len(r), segments)).astype(int)
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
    if str(_cfg.RADIUS_SOURCE_FOR_CALIBER_DETECTION).lower() in {"maximum_inscribed_sphere", "mis", "vmtk"}:
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
        enlargement_min_length_mm(),
        max_gap_mm=_cfg.ENLARGEMENT_MAX_INTERNAL_GAP_MM,
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
