"""Stenosis and enlargement detection against one robust healthy reference
(``LESION_DETECTION_METHOD = "robust_reference"``).

The ``taper`` method (:mod:`nvitk.measure.morpho.caliber`, cow_morpho's) fits two references per path:
a high (q85) envelope for stenoses and another one for enlargements, each on its own radius, and
tests them separately. Next to a real lesion, a mild bulge or the siphon's natural widening then
reads as the other kind (a hump before a C7 stenosis was reported as an enlargement), and the two
detections can overlap.

Here each path gets **one** reference and **one** signed deviation profile:

1. **Signal.** One radius for both kinds (``ROBUST_RADIUS_SOURCE``): the cross-section radius
   (``ray_median``, the inscribed radius where it is missing) or the detection radius, with a
   running median over ``ROBUST_RADIUS_MEDIAN_MM`` against single-cut spikes.
2. **Reference.** Per junction-to-junction stretch, a robust local-linear fit (LOWESS, tricube
   window ``ROBUST_REFERENCE_BANDWIDTH_MM``) of log r, iterated ``ROBUST_REFERENCE_ITERATIONS``
   times with symmetric Tukey-bisquare weights (c = 4.685 σ, σ = 1.4826 · MAD of the residuals,
   at least ``ROBUST_MIN_SIGMA``). Bumps and dips are down-weighted alike, so neither pulls it.
   Points in a junction's blanking zone or within the end exclusion carry no weight; the curve is
   bridged across them. The siphon keeps its weight: it is 25-30 mm of naturally wider artery, and
   bridging across it reads the whole cavernous / clinoid ICA as an enlargement and hides a C7
   stenosis behind the step down at its end. Then, with ``ROBUST_REFERENCE_MONOTONE``, a weighted
   pool-adjacent-violators projection lets it rise by at most ``ROBUST_MAX_RISE_PER_MM`` along the
   stretch: a healthy artery does not widen downstream, except gently (the petrous ICA into the
   cavernous one), so a bulge cannot lift its own reference. A stretch shorter than
   ``TAPER_STRETCH_MIN_LENGTH_MM`` gets its weighted median.
3. **Detection.** ``d = r / ref − 1``. A stenosis core is ``d ≤ −max(T_s, k σ_path)``, an
   enlargement core ``d ≥ max(T_e, k σ_path)`` (T = the profile's thresholds, k =
   ``ROBUST_SIGMA_FACTOR``, σ_path the path's robust spread around the unconstrained fit): a noisy
   path needs a larger
   deviation. The run extends over the support (``ROBUST_SUPPORT_FRACTION`` of the core threshold,
   same sign) and must reach the kind's minimum length; the end exclusions are the profile's, plus
   the tips' wall ramp (two reference radii), no core lies in a bifurcation's blanking zone (the
   flare where branches leave, including those shared with the other half of a split vessel), and
   none where the reference radius is under
   ``ROBUST_MIN_REFERENCE_RADIUS_MM`` (a lumen a voxel or two across). A
   point has one sign, so it is never both kinds, and a bump next to a dip is kept only when it
   stands out from the shared reference on its own.

:func:`nvitk.measure.morpho.lesion_detection.detect_caliber_lesions` dispatches here and fills the same
fields as the ``taper`` method (both reference columns hold the robust reference), plus ``robust_*``
point columns; the lesion table measures both kinds on the signal and against this reference
(:func:`nvitk.measure.morpho.vessel_features.lesion_rows`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from nvitk.measure import morphometrics_config as _cfg

from .caliber import (
    blanking_mask,
    detection_junctions,
    enlargement_min_length_mm,
    fill_reference_gaps,
    hysteresis_segments_from_min_length,
    junction_stretches,
    mask_from_segments,
    running_median,
    stenosis_valid_mask_from_ends,
)
from .models import EnlargementResult, StenosisResult

#: Tukey bisquare tuning constant (95 % efficiency under Gaussian noise).
_BISQUARE_C = 4.685


def robust_signal(s: np.ndarray, cross_section_r: np.ndarray, detection_r: np.ndarray) -> np.ndarray:
    """The one radius both kinds are tested on (module docstring, step 1)."""
    cs = np.asarray(cross_section_r, dtype=float)
    det = np.asarray(detection_r, dtype=float)
    if str(_cfg.ROBUST_RADIUS_SOURCE).lower() == "detection":
        base = det.copy()
    else:
        base = np.where(np.isfinite(cs) & (cs > 0), cs, det)
    return running_median(s, base, float(_cfg.ROBUST_RADIUS_MEDIAN_MM))


def _weighted_median(y: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(y)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float(np.median(y))
    return float(y[order][int(np.searchsorted(cw, 0.5 * cw[-1]))])


def _pava_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(max(float(wi), 1e-12))
        lens.append(1)
        while len(vals) >= 2 and vals[-2] < vals[-1]:
            tw = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            wts[-2] = tw
            lens[-2] += lens[-1]
            vals.pop()
            wts.pop()
            lens.pop()
    return np.repeat(np.asarray(vals), np.asarray(lens))


def _robust_scale(res: np.ndarray, w: np.ndarray) -> float:
    m = w > 0
    if m.sum() < 3:
        return float(_cfg.ROBUST_MIN_SIGMA)
    return max(1.4826 * float(np.median(np.abs(res[m] - np.median(res[m])))), float(_cfg.ROBUST_MIN_SIGMA))


def _lowess(s: np.ndarray, y: np.ndarray, w: np.ndarray, half_window: float, step: float) -> np.ndarray:
    """Weighted local-linear fit of *y* on *s* (tricube kernel of *half_window* mm), evaluated on
    knots every *step* mm and interpolated back to the points; NaN where a knot has no support."""
    lo, hi = float(s[0]), float(s[-1])
    knots = np.arange(lo, hi + step, step) if hi > lo else np.array([lo])
    vals = np.full(len(knots), np.nan)
    for k, x0 in enumerate(knots):
        u = np.abs(s - x0) / half_window
        kern = np.where(u < 1.0, (1.0 - u ** 3) ** 3, 0.0) * w
        sw = kern.sum()
        if sw < 1e-9 or (kern > 0).sum() < 3:
            continue
        sup = s[kern > 0]
        if x0 < sup.min() - step or x0 > sup.max() + step:
            # Data on one side only (a path end, a junction zone): no slope is
            # extrapolated into it; the gap is bridged / held from the supported knots instead.
            continue
        xm = (kern * s).sum() / sw
        ym = (kern * y).sum() / sw
        sxx = (kern * (s - xm) ** 2).sum()
        b = (kern * (s - xm) * (y - ym)).sum() / sxx if sxx > 1e-12 else 0.0
        vals[k] = ym + b * (x0 - xm)
    ok = np.isfinite(vals)
    if not ok.any():
        return np.full(len(s), np.nan)
    return np.interp(s, knots[ok], vals[ok])


def _stretch_reference(s: np.ndarray, r: np.ndarray, fit_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(reference, final weights, unconstrained fit)`` of one stretch (module docstring, step 2).
    The unconstrained robust fit (before the rise bound) is what the path's noise is measured
    against."""
    n = len(s)
    good = np.isfinite(r) & (r > 0)
    w0 = np.where(good, fit_w, 0.0)
    y = np.log(np.where(good, r, 1.0))
    if w0.sum() < 3:
        return np.full(n, np.nan), w0, np.full(n, np.nan)
    length = float(s[-1] - s[0])
    if length < float(_cfg.TAPER_STRETCH_MIN_LENGTH_MM) or (w0 > 0).sum() < 10:
        m = _weighted_median(y[w0 > 0], w0[w0 > 0])
        res = y - m
        sig = _robust_scale(res, w0)
        u = np.abs(res) / (_BISQUARE_C * sig)
        ref = np.full(n, np.exp(m))
        return ref, w0 * np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0), ref
    half = 0.5 * float(_cfg.ROBUST_REFERENCE_BANDWIDTH_MM)
    w = w0.copy()
    fit = np.full(n, np.nan)
    for _ in range(int(_cfg.ROBUST_REFERENCE_ITERATIONS)):
        fit = _lowess(s, y, w, half, 0.5)
        if not np.isfinite(fit).any():
            break
        res = y - fit
        sig = _robust_scale(res, w0)
        u = np.abs(res) / (_BISQUARE_C * sig)
        w = w0 * np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    free = np.exp(fit)
    if bool(_cfg.ROBUST_REFERENCE_MONOTONE) and np.isfinite(fit).all():
        # Rising by at most ROBUST_MAX_RISE_PER_MM (log units per mm): a long, gentle widening (the
        # petrous ICA into the cavernous one) is followed, a bulge a few millimetres long is not.
        eps = float(_cfg.ROBUST_MAX_RISE_PER_MM) * (s - s[0])
        fit = _pava_nonincreasing(fit - eps, np.maximum(w, 1e-3)) + eps
    return np.exp(fit), w, free


@dataclass
class RobustLesions:
    """One path's robust detection: the reference, the signal, the deviation, both kinds' results
    and per-point flags / percentages (0 outside the accepted runs, as the ``taper`` method)."""

    signal: np.ndarray
    reference: np.ndarray
    deviation_pct: np.ndarray
    sigma: float
    stenosis: StenosisResult
    enlargement: EnlargementResult
    stenosis_pct_point: np.ndarray
    is_stenotic: np.ndarray
    enlargement_pct_point: np.ndarray
    is_enlarged: np.ndarray
    thresholds_pct: Tuple[float, float] = (np.nan, np.nan)


def _runs_result(kind: str, s, r, ref, segments, pct_point, valid) -> StenosisResult | EnlargementResult:
    rr = ref[np.isfinite(ref)]
    r_ref = float(np.mean(rr)) if len(rr) else np.nan
    seg_s = [(float(s[a]), float(s[b])) for a, b in segments]
    vals = pct_point[np.isfinite(pct_point) & (pct_point > 0)]
    pmax = float(np.max(vals)) if len(vals) else 0.0
    if kind == "stenosis":
        rmin = float(np.nanmin(np.where(valid, r, np.nan))) if (valid & np.isfinite(r)).any() else np.nan
        return StenosisResult(r_ref, rmin, pmax, seg_s, list(segments), ref)
    rmax = float(np.nanmax(np.where(valid, r, np.nan))) if (valid & np.isfinite(r)).any() else np.nan
    return EnlargementResult(r_ref, rmax, pmax, seg_s, list(segments), ref)


def detect_lesions_robust(
    s: np.ndarray,
    cross_section_r: np.ndarray,
    detection_r: np.ndarray,
    pts: Optional[np.ndarray],
) -> RobustLesions:
    """Both kinds of one path against its robust reference (module docstring)."""
    s = np.asarray(s, dtype=float)
    n = len(s)
    sig_r = robust_signal(s, cross_section_r, detection_r)
    det = np.asarray(detection_r, dtype=float)
    pts_a = None if pts is None else np.asarray(pts, dtype=float)
    # Fit weights: off the path ends and the junction blanking zones.
    end_ex = max(float(_cfg.STENOSIS_EXCLUDE_END_MM), float(_cfg.ENLARGEMENT_EXCLUDE_END_MM))
    fit_w = stenosis_valid_mask_from_ends(s, end_ex).astype(float)
    stretch = np.zeros(n, dtype=int)
    blank = np.zeros(n, dtype=bool)
    if pts_a is not None and len(pts_a) == n:
        blank = blanking_mask(pts_a, det, detection_junctions())
        fit_w[blank] = 0.0
        stretch = junction_stretches(pts_a, det)
    ref = np.full(n, np.nan)
    free = np.full(n, np.nan)
    weights = np.zeros(n)
    for sid in np.unique(stretch):
        idx = np.flatnonzero(stretch == sid)
        ref[idx], weights[idx], free[idx] = _stretch_reference(s[idx], sig_r[idx], fit_w[idx])
    ref = fill_reference_gaps(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(np.isfinite(sig_r) & (ref > 1e-8), sig_r / ref - 1.0, np.nan)
        noise = np.where(np.isfinite(sig_r) & (free > 1e-8), sig_r / free - 1.0, np.nan)
    # The path's noise: the robust spread around the unconstrained fit (local scatter, not the
    # vessel's own shape).
    sigma = _robust_scale(np.nan_to_num(noise), np.where(np.isfinite(noise), fit_w, 0.0))
    k = float(_cfg.ROBUST_SIGMA_FACTOR)
    t_s = max(float(_cfg.STENOSIS_THRESHOLD_PCT) / 100.0, k * sigma)
    t_e = max(float(_cfg.ENLARGEMENT_THRESHOLD_PCT) / 100.0, k * sigma)
    frac = float(_cfg.ROBUST_SUPPORT_FRACTION)
    finite = np.isfinite(dev)

    # VMTK seeds a path's tips on the wall, so the radius ramps to ~0 over about one radius there:
    # no lesion may start inside that ramp (twice the reference radius at each tip).
    tip = np.zeros(n, dtype=bool)
    if n > 1 and np.isfinite(ref).any():
        r0 = float(ref[np.isfinite(ref)][0])
        r1 = float(ref[np.isfinite(ref)][-1])
        tip = (s < s[0] + 2.0 * r0) | (s > s[-1] - 2.0 * r1)

    # Resolution floor: a lumen under ROBUST_MIN_REFERENCE_RADIUS_MM is a voxel or two across, where
    # a 20 % change is one voxel of quantisation, not a lesion.
    resolvable = np.nan_to_num(ref, nan=0.0) >= float(_cfg.ROBUST_MIN_REFERENCE_RADIUS_MM)

    def _detect(sign: float, thr: float, exclude_end: float, min_len: float, max_gap: float):
        valid = stenosis_valid_mask_from_ends(s, exclude_end) & finite & ~tip
        core = valid & resolvable & ~blank & (sign * dev >= thr)  # never at a bifurcation
        support = valid & (sign * dev >= frac * thr)
        segs = hysteresis_segments_from_min_length(s, core, support, min_len, max_gap_mm=max_gap)
        flag = support & mask_from_segments(n, segs)
        pct = np.where(flag, sign * dev * 100.0, 0.0)
        return segs, flag.astype(int), pct, valid

    sten_segs, is_sten, sten_pct, v_s = _detect(-1.0, t_s, float(_cfg.STENOSIS_EXCLUDE_END_MM),
                                               float(_cfg.STENOSIS_MIN_LEN_MM), float(_cfg.STENOSIS_MAX_INTERNAL_GAP_MM))
    enl_segs, is_enl, enl_pct, v_e = _detect(1.0, t_e, float(_cfg.ENLARGEMENT_EXCLUDE_END_MM),
                                            enlargement_min_length_mm(), float(_cfg.ENLARGEMENT_MAX_INTERNAL_GAP_MM))
    return RobustLesions(
        signal=sig_r, reference=ref, deviation_pct=dev * 100.0, sigma=float(sigma),
        stenosis=_runs_result("stenosis", s, sig_r, ref, sten_segs, sten_pct, v_s),
        enlargement=_runs_result("enlargement", s, sig_r, ref, enl_segs, enl_pct, v_e),
        stenosis_pct_point=sten_pct, is_stenotic=is_sten,
        enlargement_pct_point=enl_pct, is_enlarged=is_enl,
        thresholds_pct=(100.0 * t_s, 100.0 * t_e),
    )


__all__ = ["RobustLesions", "detect_lesions_robust", "robust_signal"]
