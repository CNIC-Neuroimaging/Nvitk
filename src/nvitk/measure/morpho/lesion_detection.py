"""Stenosis / enlargement detection of one measured path, by the profile's method.

:func:`detect_caliber_lesions` is the one entry point the path measurements call
(:mod:`~nvitk.measure.morpho.centerlines`, :mod:`~nvitk.measure.morpho.donut_loops`). It runs
``LESION_DETECTION_METHOD``:

- ``taper``: cow_morpho's separate stenosis / enlargement taper fits
  (:mod:`~nvitk.measure.morpho.caliber`), each on its own radius;
- ``robust_reference``: one robust reference and one signed deviation for both kinds
  (:mod:`~nvitk.measure.morpho.robust_reference`).

Then, whatever the method, a run whose extreme point lies in a bifurcation's blanking zone (the
installed junctions, including those shared with the other half of a split vessel) or in a path
tip's wall ramp (two local radii, at least the kind's end exclusion) is dropped: a calibre change
there is the flare where branches leave, or VMTK's seed on the wall, not a lesion.

The result (:class:`CaliberLesions`) carries every per-point array and summary the point tables,
VTPs and lesion table read, so the call sites write one set of fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, List, Tuple

import numpy as np

from nvitk.measure import morphometrics_config as _cfg

from .caliber import (
    blanking_mask,
    detect_enlargement_segments,
    detect_stenosis_segments,
    detection_junctions,
    enlargement_cross_section_radius,
    enlargement_pointwise,
    enlargement_radius,
    mask_from_segments,
    resolve_stenosis_enlargement_overlap,
    segment_detail_json,
    stenosis_pointwise,
    stenosis_raw_percent,
    stenosis_total_length,
    stenosis_valid_mask_from_ends,
)
from .models import EnlargementResult, StenosisResult
from .robust_reference import detect_lesions_robust


@dataclass
class CaliberLesions:
    """One path's stenoses and enlargements (module docstring). Per-point arrays have the path's
    length; ``*_pct_point`` are 0 outside the accepted runs."""

    stenosis: StenosisResult
    enlargement: EnlargementResult
    stenosis_pct_point: np.ndarray
    is_stenotic: np.ndarray
    stenosis_raw_pct_point: np.ndarray
    stenosis_core_candidate: np.ndarray
    stenosis_support_candidate: np.ndarray
    stenosis_threshold_radius: np.ndarray
    enlargement_pct_point: np.ndarray
    is_enlarged: np.ndarray
    enlargement_raw_pct_point: np.ndarray
    enlargement_core_candidate: np.ndarray
    enlargement_support_candidate: np.ndarray
    enlargement_threshold_radius: np.ndarray
    #: The radius the enlargements were tested on, and which one (``cross_section`` /
    #: ``detection`` / ``robust``).
    enlargement_radius: np.ndarray
    enlargement_source: str
    #: Diagnostics: the enlargement flags on the cross-section and on the inscribed radius.
    is_enlarged_cs: np.ndarray
    is_enlarged_mis: np.ndarray
    #: Extra point fields of the robust method (``robust_*``, ``deviation_percent_point``).
    extra_fields: dict[str, Any] = field(default_factory=dict)


def _candidates(raw_pct: np.ndarray, core_pct: float, support_pct: float, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Core / support candidate flags (diagnostic point columns) of a raw % profile."""
    finite = np.isfinite(raw_pct)
    core = (raw_pct >= core_pct) & finite & valid
    support = (raw_pct >= support_pct) & finite & valid
    return core.astype(np.float64), support.astype(np.float64)


def _enlargement_flags_on(s, r, pts, is_stenotic) -> np.ndarray:
    """Diagnostic taper enlargement flags tested on radius *r* alone, cleared where a stenosis is
    flagged."""
    found = detect_enlargement_segments(
        s=s, r=r, threshold_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT,
        min_segment_mm=_cfg.ENLARGEMENT_MIN_LEN_MM, exclude_end_mm=_cfg.ENLARGEMENT_EXCLUDE_END_MM, pts=pts,
    )
    _, flags = enlargement_pointwise(
        s=s, r=r, r_ref_per_point=found.r_ref_per_point, threshold_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT,
        exclude_end_mm=_cfg.ENLARGEMENT_EXCLUDE_END_MM, min_segment_mm=_cfg.ENLARGEMENT_MIN_LEN_MM, pts=pts,
    )
    return np.where(is_stenotic == 1, 0, flags).astype(int)


def _taper(s, cs_r, det_r, pts, cow_source: str) -> CaliberLesions:
    """cow_morpho's taper detection (``LESION_DETECTION_METHOD = "taper"``)."""
    sten = detect_stenosis_segments(
        s=s, r=det_r, threshold_pct=_cfg.STENOSIS_THRESHOLD_PCT, min_segment_mm=_cfg.STENOSIS_MIN_LEN_MM,
        exclude_end_mm=_cfg.STENOSIS_EXCLUDE_END_MM, pts=pts,
    )
    sten_pct, is_sten = stenosis_pointwise(
        s=s, r=det_r, r_ref_per_point=sten.r_ref_per_point, threshold_pct=_cfg.STENOSIS_THRESHOLD_PCT,
        exclude_end_mm=_cfg.STENOSIS_EXCLUDE_END_MM, min_segment_mm=_cfg.STENOSIS_MIN_LEN_MM,
    )
    sten_raw = stenosis_raw_percent(s=s, r=det_r, r_ref_per_point=sten.r_ref_per_point, exclude_end_mm=0.0)
    # cow tests a path's first measurement on the cross-section radius and re-measured paths on the
    # detection radius (ENLARGEMENT_RADIUS_SOURCE = "cow").
    enl_r, enl_source = enlargement_radius(s, cs_r, det_r, cow_source=cow_source)
    enl = detect_enlargement_segments(
        s=s, r=enl_r, threshold_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT, min_segment_mm=_cfg.ENLARGEMENT_MIN_LEN_MM,
        exclude_end_mm=_cfg.ENLARGEMENT_EXCLUDE_END_MM, pts=pts,
    )
    enl_pct, is_enl = enlargement_pointwise(
        s=s, r=enl_r, r_ref_per_point=enl.r_ref_per_point, threshold_pct=_cfg.ENLARGEMENT_THRESHOLD_PCT,
        exclude_end_mm=_cfg.ENLARGEMENT_EXCLUDE_END_MM, min_segment_mm=_cfg.ENLARGEMENT_MIN_LEN_MM, pts=pts,
    )
    enl_raw = np.full(len(s), np.nan)
    ok = np.isfinite(enl_r) & np.isfinite(enl.r_ref_per_point) & (enl.r_ref_per_point > 1e-8)
    enl_raw[ok] = (enl_r[ok] / enl.r_ref_per_point[ok] - 1.0) * 100.0
    sten_pct, is_sten, enl_pct, is_enl = resolve_stenosis_enlargement_overlap(sten_pct, is_sten, enl_pct, is_enl)
    sten_valid = stenosis_valid_mask_from_ends(s, float(_cfg.STENOSIS_CANDIDATE_EXCLUDE_END_MM))
    enl_valid = stenosis_valid_mask_from_ends(s, float(_cfg.ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM))
    t_s, t_e = float(_cfg.STENOSIS_THRESHOLD_PCT), float(_cfg.ENLARGEMENT_THRESHOLD_PCT)
    # Diagnostic flags per radius: the tested one is is_enl, the other is re-detected alone.
    if enl_source == "cross_section":
        is_enl_cs, is_enl_mis = is_enl, _enlargement_flags_on(s, det_r, pts, is_sten)
    else:
        is_enl_mis, is_enl_cs = is_enl, _enlargement_flags_on(s, enlargement_cross_section_radius(s, cs_r), pts, is_sten)
    sten_core, sten_support = _candidates(sten_raw, t_s, min(float(_cfg.STENOSIS_SUPPORT_THRESHOLD_PCT), t_s), sten_valid)
    enl_core, enl_support = _candidates(enl_raw, t_e, min(float(_cfg.ENLARGEMENT_SUPPORT_THRESHOLD_PCT), t_e), enl_valid)
    return CaliberLesions(
        stenosis=sten, enlargement=enl,
        stenosis_pct_point=sten_pct, is_stenotic=is_sten, stenosis_raw_pct_point=sten_raw,
        stenosis_core_candidate=sten_core, stenosis_support_candidate=sten_support,
        stenosis_threshold_radius=(1.0 - t_s / 100.0) * sten.r_ref_per_point,
        enlargement_pct_point=enl_pct, is_enlarged=is_enl, enlargement_raw_pct_point=enl_raw,
        enlargement_core_candidate=enl_core, enlargement_support_candidate=enl_support,
        enlargement_threshold_radius=(1.0 + t_e / 100.0) * enl.r_ref_per_point,
        enlargement_radius=enl_r, enlargement_source=enl_source,
        is_enlarged_cs=is_enl_cs, is_enlarged_mis=is_enl_mis,
    )


def _robust(s, cs_r, det_r, pts) -> CaliberLesions:
    """The robust shared-reference detection (``LESION_DETECTION_METHOD = "robust_reference"``)."""
    rob = detect_lesions_robust(s, cs_r, det_r, pts)
    t_s, t_e = rob.thresholds_pct
    frac = float(_cfg.ROBUST_SUPPORT_FRACTION)
    sten_raw, enl_raw = -rob.deviation_pct, rob.deviation_pct.copy()
    sten_valid = stenosis_valid_mask_from_ends(s, float(_cfg.STENOSIS_CANDIDATE_EXCLUDE_END_MM))
    enl_valid = stenosis_valid_mask_from_ends(s, float(_cfg.ENLARGEMENT_CANDIDATE_EXCLUDE_END_MM))
    sten_core, sten_support = _candidates(sten_raw, t_s, frac * t_s, sten_valid)
    enl_core, enl_support = _candidates(enl_raw, t_e, frac * t_e, enl_valid)
    return CaliberLesions(
        stenosis=rob.stenosis, enlargement=rob.enlargement,
        stenosis_pct_point=rob.stenosis_pct_point, is_stenotic=rob.is_stenotic, stenosis_raw_pct_point=sten_raw,
        stenosis_core_candidate=sten_core, stenosis_support_candidate=sten_support,
        stenosis_threshold_radius=(1.0 - t_s / 100.0) * rob.reference,
        enlargement_pct_point=rob.enlargement_pct_point, is_enlarged=rob.is_enlarged, enlargement_raw_pct_point=enl_raw,
        enlargement_core_candidate=enl_core, enlargement_support_candidate=enl_support,
        enlargement_threshold_radius=(1.0 + t_e / 100.0) * rob.reference,
        enlargement_radius=rob.signal, enlargement_source="robust",
        # One signal for both kinds: the flags are the same on either radius.
        is_enlarged_cs=rob.is_enlarged, is_enlarged_mis=rob.is_enlarged,
        extra_fields={
            "robust_reference_radius_point": rob.reference,
            "robust_radius_point": rob.signal,
            "deviation_percent_point": rob.deviation_pct,
            "robust_sigma": float(rob.sigma),
        },
    )


def _keep_runs(
    runs: List[Tuple[int, int]], radius: np.ndarray, s: np.ndarray, blank: np.ndarray, end_ex: float, narrowest: bool
) -> List[Tuple[int, int]]:
    """*runs* without those whose extreme point (the narrowest / widest of *radius*) is in a
    blanking zone or within two local radii (at least *end_ex*) of a path end."""
    to_end = np.minimum(s - s[0], s[-1] - s)
    kept = []
    for a, b in runs:
        seg = radius[a:b + 1]
        if np.isfinite(seg).any():
            x = a + int(np.nanargmin(seg) if narrowest else np.nanargmax(seg))
        else:
            x = (a + b) // 2
        r_x = float(radius[x]) if np.isfinite(radius[x]) else 0.0
        if blank[x] or to_end[x] < max(float(end_ex), 2.0 * r_x):
            continue
        kept.append((int(a), int(b)))
    return kept


def _restrict(result, runs, pct_point, flags, s, n):
    """*result*, its point % and flags kept to *runs* (a dropped run leaves no trace)."""
    keep = mask_from_segments(n, runs)
    flags = (np.asarray(flags) * keep).astype(int)
    pct_point = np.where(keep, pct_point, 0.0)
    vals = pct_point[np.isfinite(pct_point) & (pct_point > 0)]
    pmax = float(np.max(vals)) if len(vals) else 0.0
    seg_s = [(float(s[a]), float(s[b])) for a, b in runs]
    if isinstance(result, StenosisResult):
        result = replace(result, percent_stenosis_max=pmax, segments_s_mm=seg_s, segments_point_idx=list(runs))
    else:
        result = replace(result, percent_enlargement_max=pmax, segments_s_mm=seg_s, segments_point_idx=list(runs))
    return result, pct_point, flags


def detect_caliber_lesions(
    s: np.ndarray, cross_section_r: np.ndarray, detection_r: np.ndarray, pts: np.ndarray, *, cow_source: str
) -> CaliberLesions:
    """Stenoses and enlargements of one path (module docstring). *cross_section_r* /
    *detection_r* are the path's cross-section and detection (inscribed) radii; *cow_source* is the
    radius cow_morpho tests enlargements on along this code path (``cross_section`` on a first
    measurement, ``detection`` on a re-measured one; ``taper`` method only)."""
    s = np.asarray(s, dtype=float)
    cs_r = np.asarray(cross_section_r, dtype=float)
    det_r = np.asarray(detection_r, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if str(_cfg.LESION_DETECTION_METHOD).lower() == "robust_reference":
        out = _robust(s, cs_r, det_r, pts)
    else:
        out = _taper(s, cs_r, det_r, pts, cow_source)
    n = len(s)
    if n < 2:
        return out
    blank = blanking_mask(pts, det_r, detection_junctions()) if len(pts) == n else np.zeros(n, dtype=bool)
    sten_r = out.extra_fields.get("robust_radius_point", det_r)
    sten_runs = _keep_runs(list(out.stenosis.segments_point_idx), np.asarray(sten_r, dtype=float), s, blank,
                           float(_cfg.STENOSIS_EXCLUDE_END_MM), narrowest=True)
    enl_runs = _keep_runs(list(out.enlargement.segments_point_idx), out.enlargement_radius, s, blank,
                          float(_cfg.ENLARGEMENT_EXCLUDE_END_MM), narrowest=False)
    out.stenosis, out.stenosis_pct_point, out.is_stenotic = _restrict(
        out.stenosis, sten_runs, out.stenosis_pct_point, out.is_stenotic, s, n)
    out.enlargement, out.enlargement_pct_point, out.is_enlarged = _restrict(
        out.enlargement, enl_runs, out.enlargement_pct_point, out.is_enlarged, s, n)
    return out


def lesion_result_fields(les: CaliberLesions, s: np.ndarray, detection_r: np.ndarray) -> dict[str, Any]:
    """The path-result fields of *les* (point arrays, run summaries, JSON run lists) that the point
    tables, the VTPs and the lesion table read."""
    s = np.asarray(s, dtype=float)
    det_r = np.asarray(detection_r, dtype=float)
    sten, enl = les.stenosis, les.enlargement
    enl_r = les.enlargement_radius
    stenotic_r = det_r[(les.is_stenotic == 1) & np.isfinite(det_r)]
    enlarged_r = enl_r[(les.is_enlarged == 1) & np.isfinite(enl_r)]
    return {
        "stenosis_percent_point": les.stenosis_pct_point,
        "stenosis_raw_percent_point": les.stenosis_raw_pct_point,
        "stenosis_core_candidate_point": les.stenosis_core_candidate,
        "stenosis_support_candidate_point": les.stenosis_support_candidate,
        "is_stenotic": les.is_stenotic,
        "stenosis_reference_radius_point": sten.r_ref_per_point,
        "stenosis_threshold_radius_point": les.stenosis_threshold_radius,
        "enlargement_raw_percent_point": les.enlargement_raw_pct_point,
        "enlargement_core_candidate_point": les.enlargement_core_candidate,
        "enlargement_support_candidate_point": les.enlargement_support_candidate,
        "enlargement_percent_point": les.enlargement_pct_point,
        "is_enlarged": les.is_enlarged,
        "is_enlarged_cs": les.is_enlarged_cs,
        "is_enlarged_mis": les.is_enlarged_mis,
        "enlargement_reference_radius_point": enl.r_ref_per_point,
        "enlargement_threshold_radius_point": les.enlargement_threshold_radius,
        # The radius the enlargement test used; the lesion table measures severity on the same one.
        "enlargement_radius_mm": enl_r,
        "radius_ref_mm": float(sten.r_ref),
        "radius_min_mm": float(sten.r_min),
        "radius_min_stenotic_mm": float(np.min(stenotic_r)) if stenotic_r.size else np.nan,
        "stenosis_percent_max": float(sten.percent_stenosis_max),
        "degree_of_stenosis_pct": float(sten.percent_stenosis_max),
        "stenosis_length_total_mm": float(stenosis_total_length(s, les.is_stenotic)),
        "stenosis_segments_n": int(len(sten.segments_point_idx)),
        "stenosis_segments_point_idx": json.dumps(sten.segments_point_idx),
        "stenosis_segments_detail_json": segment_detail_json(s, les.stenosis_pct_point, sten.segments_point_idx),
        "enlargement_radius_ref_mm": float(enl.r_ref),
        "enlargement_radius_max_mm": float(enl.r_max),
        "radius_max_enlarged_mm": float(np.max(enlarged_r)) if enlarged_r.size else np.nan,
        "enlargement_percent_max": float(enl.percent_enlargement_max),
        "enlargement_length_total_mm": float(stenosis_total_length(s, les.is_enlarged)),
        "enlargement_segments_n": int(len(enl.segments_point_idx)),
        "enlargement_segments_point_idx": json.dumps(enl.segments_point_idx),
        "enlargement_segments_detail_json": segment_detail_json(s, les.enlargement_pct_point, enl.segments_point_idx),
        **les.extra_fields,
    }


__all__ = ["CaliberLesions", "detect_caliber_lesions", "lesion_result_fields"]
