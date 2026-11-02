"""Scale-aware vessel descriptors and lesion severity / confidence for the compact vessel tables.

Inspired by the segment features of FLOWCAT ``arterial`` (feature_extraction: proximal / distal
and end-window diameters, min/max diameter ratio, tortuosity of the first / last N mm, bending
length, cumulative curvature, curvature energy, direction angles, bifurcation blanking) and
re-implemented here from their definitions; no code is shared (``arterial`` is PolyForm
Noncommercial). Two deliberate differences: curvature integrals are arc-length integrals
(``∫κ ds``, ``∫κ² ds``) rather than sums over nodes, so they do not depend on the sampling step;
and curvature is measured on the centerline smoothed at ``FEATURE_CURVATURE_SMOOTH_MM`` (true
1/mm, :func:`~nvitk.measure.morpho.metrics.smoothed_curvature`).

Radii: ``mis`` is the maximum-inscribed-sphere radius (VMTK, or its native equivalent), ``cs`` the
cross-section equivalent radius. Diameters are ``2 × radius``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from nvitk.measure import morphometrics_config as _cfg

from .caliber import blanking_mask, enlargement_min_length_mm, junction_crossings
from .metrics import discrete_torsion, smooth_polyline, smoothed_curvature


def _arc(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _stat(a: np.ndarray, fn) -> float:
    v = _finite(a)
    return float(fn(v)) if v.size else np.nan


def _window_mean(values: np.ndarray, s: np.ndarray, lo: float, hi: float) -> float:
    sel = (s >= lo) & (s <= hi)
    return _stat(values[sel], np.mean)


def _dm(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.nan
    chord = float(np.linalg.norm(points[-1] - points[0]))
    return float(_arc(points)[-1] / chord) if chord > 1e-8 else np.nan


def _sub_path(points: np.ndarray, s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return points[(s >= lo) & (s <= hi)]


def bending_length(points: np.ndarray) -> float:
    """Largest distance of the path from its chord (mm): how far the vessel bows out."""
    if len(points) < 3:
        return 0.0
    a, b = points[0], points[-1]
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-8:
        return float(np.max(np.linalg.norm(points - a, axis=1)))
    rel = points - a
    proj = (rel @ ab) / n
    perp = np.sqrt(np.maximum((rel ** 2).sum(axis=1) - proj ** 2, 0.0))
    return float(perp.max())


def _direction_angles(vec: np.ndarray, axes: Any | None) -> tuple[float, float]:
    """Polar angle from the subject's superior axis and azimuth in the axial plane (degrees,
    azimuth 0 = right, 90 = anterior). Array-axis fallback when the anatomical axes are unknown."""
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.nan, np.nan
    v = v / n
    if axes is not None:
        r = v[axes.lr_axis] * axes.lr_sign
        a = v[axes.ap_axis] * axes.ap_sign
        s = v[axes.si_axis] * axes.si_sign
    else:
        r, a, s = v
    polar = float(np.degrees(np.arccos(np.clip(s, -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(a, r)))
    return polar, azimuth


def _turn_count(kappa: np.ndarray, s: np.ndarray) -> int:
    """Successive turns: curvature peaks above ``BEND_KAPPA_PEAK × 2`` (true units) at least one
    smoothing scale apart (FLOWCAT's curve-id idea on the smoothed curvature)."""
    from scipy.signal import find_peaks

    k = np.where(np.isfinite(kappa), kappa, 0.0)
    if len(k) < 5 or len(s) < 2:
        return 0
    ds = float(np.median(np.diff(s))) or 1.0
    distance = max(1, int(round(float(_cfg.FEATURE_CURVATURE_SMOOTH_MM) * 2.0 / ds)))
    peaks, _ = find_peaks(k, height=2.0 * float(_cfg.BEND_KAPPA_PEAK), distance=distance)
    return int(len(peaks))


def path_features(
    points: np.ndarray,
    mis_radius: np.ndarray,
    cs_radius: np.ndarray,
    *,
    junctions: np.ndarray | None = None,
    axes: Any | None = None,
) -> dict[str, float]:
    """FLOWCAT-style geometry / calibre / shape descriptors of one centerline path (mm, 1/mm)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    out: dict[str, float] = {"n_points": float(n)}
    if n < 2:
        return out
    s = _arc(pts)
    length = float(s[-1])
    mis = np.asarray(mis_radius, dtype=float)
    cs = np.asarray(cs_radius, dtype=float)
    # Prefer the inscribed radius; fall back to cross-section where it is missing.
    r = np.where(np.isfinite(mis), mis, cs)
    end = float(_cfg.FEATURE_END_WINDOW_MM)
    # The tips are excluded from calibre statistics: VMTK appends the seed point on the wall
    # (inscribed radius -> 0) and the cross-section there cuts the cap, not the lumen.
    tip = np.maximum(float(_cfg.FEATURE_TIP_EXCLUDE_MM), np.where(np.isfinite(r), r, 0.0))
    core = (s >= tip) & (s <= length - tip)
    if core.sum() < 2:
        core = np.ones(n, dtype=bool)
    mis = np.where(core, mis, np.nan)
    cs = np.where(core, cs, np.nan)
    r = np.where(core, r, np.nan)

    out.update({
        "length_mm": length,
        "chord_length_mm": float(np.linalg.norm(pts[-1] - pts[0])),
        "tortuosity_dm": _dm(pts),
        "bending_length_mm": bending_length(pts),
    })
    for tag, rad in (("", r), ("cs_", cs)):
        d = 2.0 * rad
        out[f"{tag}diameter_mean_mm"] = _stat(d, np.mean)
        out[f"{tag}diameter_median_mm"] = _stat(d, np.median)
        out[f"{tag}diameter_std_mm"] = _stat(d, np.std)
        out[f"{tag}diameter_min_mm"] = _stat(d, np.min)
        out[f"{tag}diameter_max_mm"] = _stat(d, np.max)
        out[f"{tag}diameter_p05_mm"] = _stat(d, lambda v: np.percentile(v, 5))
        out[f"{tag}diameter_p95_mm"] = _stat(d, lambda v: np.percentile(v, 95))
    d = 2.0 * r
    out["diameter_proximal_mm"] = _window_mean(d, s, 0.0, end)
    out["diameter_distal_mm"] = _window_mean(d, s, length - end, length)
    out["diameter_last_5mm_mm"] = _window_mean(d, s, length - 5.0, length)
    out["diameter_last_10mm_mm"] = _window_mean(d, s, length - 10.0, length)
    dmin, dmax = out["diameter_min_mm"], out["diameter_max_mm"]
    out["diameter_min_max_ratio"] = float(dmin / dmax) if np.isfinite(dmin) and np.isfinite(dmax) and dmax > 0 else np.nan
    out["cross_section_area_mean_mm2"] = _stat(np.pi * cs ** 2, np.mean)
    valid = np.isfinite(r)
    if valid.sum() >= 3:
        out["taper_mm_per_mm"] = float(np.polyfit(s[valid], 2.0 * r[valid], 1)[0])
    blank = blanking_mask(pts, r, junctions)
    out["blanked_fraction"] = float(blank.mean())
    out["diameter_mean_blanked_mm"] = _stat(d[~blank], np.mean)
    out["diameter_min_blanked_mm"] = _stat(d[~blank], np.min)

    for name, lo, hi in (
        ("tortuosity_first_10mm", 0.0, 10.0),
        ("tortuosity_last_10mm", length - 10.0, length),
        ("tortuosity_last_20mm", length - 20.0, length),
    ):
        out[name] = _dm(_sub_path(pts, s, lo, hi)) if length >= (hi - lo) else np.nan

    sigma = float(_cfg.FEATURE_CURVATURE_SMOOTH_MM)
    kappa = smoothed_curvature(pts, sigma)
    tau = discrete_torsion(smooth_polyline(pts, sigma))
    ds = np.gradient(s) if n > 1 else np.zeros(n)
    k0 = np.where(np.isfinite(kappa), kappa, 0.0)
    out["curvature_mean_1_per_mm"] = _stat(kappa, np.mean)
    out["curvature_p95_1_per_mm"] = _stat(kappa, lambda v: np.percentile(v, 95))
    out["curvature_max_1_per_mm"] = _stat(kappa, np.max)
    out["total_curvature_rad"] = float(np.sum(k0 * ds))           # ∫κ ds
    out["curvature_energy_1_per_mm"] = float(np.sum(k0 ** 2 * ds))  # ∫κ² ds
    out["sum_of_angles_deg_per_mm"] = float(np.degrees(out["total_curvature_rad"]) / length) if length > 0 else np.nan
    out["torsion_mean_abs_1_per_mm"] = _stat(np.abs(tau), np.mean)
    out["n_turns"] = float(_turn_count(kappa, s))

    polar, azimuth = _direction_angles(pts[-1] - pts[0], axes)
    out["chord_polar_angle_deg"] = polar
    out["chord_azimuth_angle_deg"] = azimuth
    smooth = smooth_polyline(pts, sigma)
    tang = np.gradient(smooth, axis=0)
    polars = np.asarray([_direction_angles(t, axes)[0] for t in tang])
    out["polar_angle_min_deg"] = _stat(polars, np.min)
    out["polar_angle_accumulated_change_deg"] = float(np.nansum(np.abs(np.diff(polars)))) if len(polars) > 1 else 0.0
    return out


# Lesions


def _segment_name_at(point: np.ndarray, segments: Sequence[Mapping[str, Any]]) -> str:
    """Name of the non-overlapping segment closest to *point*."""
    best, best_d = "", np.inf
    for seg in segments:
        p = np.column_stack([seg["x_mm"], seg["y_mm"], seg["z_mm"]]).astype(float)
        if len(p) == 0:
            continue
        d = float(np.min(np.linalg.norm(p - point, axis=1)))
        if d < best_d:
            best, best_d = str(seg.get("anatomic_segment_name") or seg.get("tree_label") or ""), d
    return best


def _parse_runs(raw: Any) -> list[tuple[int, int]]:
    import json

    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    try:
        return [(int(a), int(b)) for a, b in raw]
    except Exception:  # noqa: BLE001
        return []


def _window_span(mask: np.ndarray) -> tuple[float, float]:
    """First and last index of *mask* (NaN, NaN when empty)."""
    idx = np.flatnonzero(mask)
    return (float(idx[0]), float(idx[-1])) if idx.size else (np.nan, np.nan)


def _window(s: np.ndarray, eligible: np.ndarray, a: int, b: int, upstream: bool) -> np.ndarray:
    """One reference window of the lesion run ``a..b``: the nearest
    ``LESION_REFERENCE_WINDOW_LENGTH_MM`` of *eligible* points from ``LESION_REFERENCE_GAP_MM`` to
    ``LESION_REFERENCE_SEARCH_MM`` away, upstream or downstream."""
    gap = float(_cfg.LESION_REFERENCE_GAP_MM)
    search = float(_cfg.LESION_REFERENCE_SEARCH_MM)
    dist = (s[a] - s) if upstream else (s - s[b])
    cand = eligible & (dist >= gap) & (dist <= search)
    out = np.zeros(len(s), dtype=bool)
    if not cand.any():
        return out
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 0.1
    need = max(3, int(round(float(_cfg.LESION_REFERENCE_WINDOW_LENGTH_MM) / max(ds, 1e-6))))
    idx = np.flatnonzero(cand)
    out[idx[np.argsort(dist[idx], kind="stable")[:need]]] = True
    return out


def _lesion_reference(
    s: np.ndarray,
    kind_r: np.ndarray,
    a: int,
    b: int,
    x: int,
    *,
    ref_ok: np.ndarray,
    healthy: np.ndarray,
    junction_idx: np.ndarray,
    robust_ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """The healthy reference radius of the lesion run ``a..b`` (extreme point *x*), per
    ``LESION_REFERENCE_METHOD`` (or the robust reference curve, *robust_ref*, when given); see
    :func:`lesion_rows`."""
    finite = np.isfinite(kind_r)
    method = str(_cfg.LESION_REFERENCE_METHOD).lower()
    fallback = crosses = False
    if method == "interpolated" or robust_ref is not None:
        # Healthy points, kept to the lesion's own junction-to-junction stretch when it has room.
        before = junction_idx[junction_idx < a]
        after = junction_idx[junction_idx > b]
        lo_i = int(before.max()) if before.size else -1
        hi_i = int(after.min()) if after.size else len(s)
        idx = np.arange(len(s))
        ok = ref_ok & healthy & finite
        own = ok & (idx > lo_i) & (idx < hi_i)
        up, down = _window(s, own, a, b, True), _window(s, own, a, b, False)
        if up.sum() < 3:
            alt = _window(s, ok, a, b, True)
            if alt.sum() >= 3:
                up, crosses = alt, True
        if down.sum() < 3:
            alt = _window(s, ok, a, b, False)
            if alt.sum() >= 3:
                down, crosses = alt, True
        if up.sum() < 3 and down.sum() < 3:
            # No healthy window (a lesion hemmed in by ends or other lesions): the plain windows,
            # as the proximal method, rather than no severity.
            up, down = _window(s, ref_ok & finite, a, b, True), _window(s, ref_ok & finite, a, b, False)
            if robust_ref is None:
                method = "proximal"
            fallback, crosses = True, False
    else:
        up, down = _window(s, ref_ok & finite, a, b, True), _window(s, ref_ok & finite, a, b, False)
    r_p = _stat(kind_r[up], np.median) if up.sum() >= 3 else np.nan
    r_d = _stat(kind_r[down], np.median) if down.sum() >= 3 else np.nan
    out: dict[str, Any] = {
        "reference_proximal_radius_mm": r_p,
        "reference_distal_radius_mm": r_d,
        "inconsistent": False,
        "fallback": fallback,
        "crosses_junction": crosses,
    }
    p0, p1 = _window_span(up) if np.isfinite(r_p) else (np.nan, np.nan)
    d0, d1 = _window_span(down) if np.isfinite(r_d) else (np.nan, np.nan)
    out.update(reference_proximal_start_point=p0, reference_proximal_end_point=p1,
               reference_distal_start_point=d0, reference_distal_end_point=d1)
    if robust_ref is not None:
        # The robust reference itself; the windows only check it: their ratio against the one
        # the reference expects between them (its taper and junction steps).
        out["ref_r"] = float(robust_ref[x]) if np.isfinite(robust_ref[x]) else np.nan
        out["ref_where"] = "robust" if np.isfinite(out["ref_r"]) else "none"
        if np.isfinite(r_p) and np.isfinite(r_d) and r_p > 0:
            mp, md = int(round(0.5 * (p0 + p1))), int(round(0.5 * (d0 + d1)))
            expect = robust_ref[md] / robust_ref[mp] if robust_ref[mp] > 0 else np.nan
            if np.isfinite(expect) and expect > 0:
                out["inconsistent"] = abs((r_d / r_p) / expect - 1.0) > float(_cfg.LESION_REFERENCE_DISAGREEMENT_FRACTION)
        return out
    if method != "interpolated":
        # The method before 2026-10-21: the proximal window whenever it has room.
        if up.sum() >= 3:
            out["ref_r"], out["ref_where"] = r_p, "proximal"
        elif down.sum() >= 3:
            out["ref_r"], out["ref_where"] = r_d, "distal"
        else:
            out["ref_r"], out["ref_where"] = np.nan, "none"
    elif np.isfinite(r_p) and np.isfinite(r_d) and r_p > 0:
        # Linear in arc length between the two windows' midpoints, at the lesion's extreme point.
        # A healthy vessel does not widen downstream: a wider distal window (post-stenotic
        # dilatation) is capped at the proximal radius.
        out["inconsistent"] = abs(r_d / r_p - 1.0) > float(_cfg.LESION_REFERENCE_DISAGREEMENT_FRACTION)
        s_p = 0.5 * (s[int(p0)] + s[int(p1)])
        s_d = 0.5 * (s[int(d0)] + s[int(d1)])
        t = float(np.clip((s[x] - s_p) / (s_d - s_p), 0.0, 1.0)) if s_d > s_p else 0.0
        out["ref_r"] = r_p + (min(r_d, r_p) - r_p) * t
        out["ref_where"] = "interpolated"
    elif np.isfinite(r_p):
        out["ref_r"], out["ref_where"] = r_p, "proximal"
    else:
        out["ref_r"], out["ref_where"] = r_d, "distal"
    return out


def lesion_rows(
    paths: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]] = (),
    *,
    junctions: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """One row per detected stenosis / enlargement of a vessel component.

    Lesions are exactly the detection segments of the measured paths
    (``stenosis_segments_point_idx`` / ``enlargement_segments_point_idx``), counted once where
    overlapping paths share them, and named after the non-overlapping segment that contains their
    midpoint. Severity is reported three ways: against the taper reference (``taper_percent``, the
    detection measure) and against a **healthy reference radius** as diameter and area stenosis; for
    enlargements both are signed so that positive means wider than the reference. The reference
    (``LESION_REFERENCE_METHOD``):

    Each window is the nearest ``LESION_REFERENCE_WINDOW_LENGTH_MM`` of eligible points from
    ``LESION_REFERENCE_GAP_MM`` to ``LESION_REFERENCE_SEARCH_MM`` away from the lesion (upstream for
    the proximal one, downstream for the distal one), so it reaches past a neighbouring lesion or a
    short eroded stretch instead of measuring it.

    ``robust`` (``LESION_DETECTION_METHOD = "robust_reference"``)
        the path's robust reference at the lesion's extreme point
        (:mod:`nvitk.measure.morpho.robust_reference`); both kinds are measured on its signal. The
        windows below are still reported, and flag ``reference_inconsistent`` when their ratio
        departs from the one the reference expects between them.
    ``interpolated`` (``taper`` method, default)
        the median radius of a proximal and a distal window, interpolated linearly in arc length to
        the lesion's narrowest (widest) point (QCA's interpolated reference), so the vessel's natural
        taper between them is not counted as stenosis. The windows keep to healthy points (other
        lesions and the bifurcation blanking zones left out) of the lesion's own junction-to-junction
        stretch; a side with no room there crosses the junction (``reference_crosses_junction``). A
        distal median above the proximal one is capped at it (a healthy vessel does not widen
        downstream). With one window only, that window's median (``proximal`` / ``distal``). With
        neither, the ``proximal`` method's plain windows (``reference_fallback``).
    ``proximal``
        the median radius of the upstream window (downstream when there is no upstream room): the
        method before 2026-10-21.

    Neither window comes within ``LESION_REFERENCE_EXCLUDE_END_MM`` of a path end. Both windows'
    medians and spans are reported (``reference_proximal_*`` / ``reference_distal_*``). With the
    ``taper`` method each kind is measured on the radius its test used: stenoses on the detection
    (inscribed) radius, enlargements on ``enlargement_radius_mm`` (``ENLARGEMENT_RADIUS_SOURCE``;
    cow tests them on the cross-section radius, or on the detection radius on re-measured paths).

    Confidence ∈ [0, 1] multiplies a penalty for each condition that makes a calibre change
    unreliable (the applied ones are listed in ``confidence_flags``):

    ===================== ======= ====================================================================
    flag                  factor  condition
    ===================== ======= ====================================================================
    reference_distal/none 0.8     no proximal reference window (distal one used, or none)
    reference_inconsistent 0.8    ``interpolated``: the proximal and distal medians differ by more
                                  than ``LESION_REFERENCE_DISAGREEMENT_FRACTION``; more than taper
                                  explains (distal collapse, dilatation, an unseen branch)
    reference_fallback    0.8     ``interpolated``: no healthy window in the lesion's stretch, so the
                                  plain windows of the ``proximal`` method were used
    reference_crosses_    0.9     a window had to leave the lesion's stretch
    junction
    in_siphon             0.7     overlaps the siphon (high-curvature) mask (robust method: only
                                  together with radius_disagreement)
    noisy_path            0.8     robust method: the path's robust spread of r / ref − 1 exceeds
                                  ``ROBUST_NOISY_SIGMA``
    near_vessel_end       0.6     within ``LESION_NEAR_END_MM`` of a path end
    short                 0.8     shorter than 1.5 × the detection minimum length (stenoses:
                                  ``STENOSIS_MIN_LEN_MM``; enlargements: the shorter of
                                  ``ENLARGEMENT_MIN_LEN_MM`` / ``ENLARGEMENT_MIN_SUPPORT_LENGTH_MM``)
    near_bifurcation      0.5     within ``LESION_NEAR_BIFURCATION_MM`` of a junction
    radius_disagreement   0.7     cross-section / inscribed radius ratio departs from the path's
                                  median ratio by more than ``LESION_RADIUS_DISAGREEMENT_FRACTION``
                                  (one estimator sees the lesion, the other does not)
    cross_section_gaps    0.8     more than 20 % of the lesion has no valid cross-section
    ===================== ======= ====================================================================

    Runs at a bifurcation or a path tip never reach here: detection drops them
    (:func:`nvitk.measure.morpho.lesion_detection.detect_caliber_lesions`).

    Classes: high >= 0.7 (at most one mild penalty), moderate >= 0.4, low below.
    """
    end_ex = float(_cfg.LESION_REFERENCE_EXCLUDE_END_MM or 0.0)
    rows: list[dict[str, Any]] = []
    seen: dict[str, list[np.ndarray]] = {"stenosis": [], "enlargement": []}
    tree = None
    if junctions is not None and len(junctions):
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(junctions, dtype=float))
    for path in paths:
        pts = np.column_stack([path["x_mm"], path["y_mm"], path["z_mm"]]).astype(float)
        n = len(pts)
        if n < 2:
            continue
        s = _arc(pts)
        det_r = np.asarray(path.get("stenosis_detection_radius_mm", path.get("radius_mm")), dtype=float)
        # Enlargements are tested on their own radius (cow: the cross-section one on a path's first
        # measurement); their severity is measured on that same radius.
        enl_r = np.asarray(path.get("enlargement_radius_mm", path.get("radius_mm", det_r)), dtype=float)
        robust_ref = path.get("robust_reference_radius_point")
        robust = robust_ref is not None and len(robust_ref) == n
        if robust:  # robust_reference: both kinds on one signal, against one reference
            robust_ref = np.asarray(robust_ref, dtype=float)
            det_r = enl_r = np.asarray(path.get("robust_radius_point", enl_r), dtype=float)
        sigma = float(path.get("robust_sigma", np.nan)) if robust else np.nan
        mis = np.asarray(path.get("maximum_inscribed_sphere_radius_mm", np.full(n, np.nan)), dtype=float)
        cs = np.asarray(path.get("radius_mm", np.full(n, np.nan)), dtype=float)
        siphon = np.asarray(path.get("siphon_mask_point", np.zeros(n)), dtype=float) >= 0.5
        # The area-equivalent cross-section radius normally exceeds the inscribed radius by 10-25 %
        # (non-circular lumen); a lesion is suspicious when its ratio departs from the path's own.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mis > 0, cs / mis, np.nan)
        path_ratio = _stat(ratio, np.median)
        ref_ok = (s >= end_ex) & (s <= s[-1] - end_ex)  # where a reference window may lie
        # Healthy reference points (interpolated method): off every lesion run of this path and off
        # the bifurcation blanking zones; junction_idx bounds the windows to the lesion's stretch.
        lesioned = np.zeros(n, dtype=bool)
        for key in ("stenosis_segments_point_idx", "enlargement_segments_point_idx"):
            for p, q in _parse_runs(path.get(key)):
                lesioned[max(0, min(p, q)):min(n, max(p, q) + 1)] = True
        healthy = ~lesioned & ~blanking_mask(pts, det_r, junctions)
        junction_idx = junction_crossings(pts, det_r, junctions)
        for kind, runs_key, pct_key, min_len, kind_r in (
            ("stenosis", "stenosis_segments_point_idx", "stenosis_percent_point", float(_cfg.STENOSIS_MIN_LEN_MM),
             det_r),
            ("enlargement", "enlargement_segments_point_idx", "enlargement_percent_point",
             enlargement_min_length_mm(), enl_r),
        ):
            pct = np.asarray(path.get(pct_key, np.full(n, np.nan)), dtype=float)
            for a, b in _parse_runs(path.get(runs_key)):
                a, b = max(0, min(a, n - 1)), max(0, min(b, n - 1))
                if b < a:
                    a, b = b, a
                mid = pts[(a + b) // 2]
                if any(float(np.linalg.norm(mid - m)) < 1.0 for m in seen[kind]):
                    continue  # the same lesion, measured again on an overlapping path
                sl = slice(a, b + 1)
                extreme_r = _stat(kind_r[sl], np.min) if kind == "stenosis" else _stat(kind_r[sl], np.max)
                seg_r = np.where(np.isfinite(kind_r[sl]), kind_r[sl], np.nan)
                if np.isfinite(seg_r).any():
                    x = a + int(np.nanargmin(seg_r) if kind == "stenosis" else np.nanargmax(seg_r))
                else:
                    x = (a + b) // 2
                seen[kind].append(mid)
                flags: list[str] = []
                conf = 1.0
                ref = _lesion_reference(s, kind_r, a, b, x, ref_ok=ref_ok, healthy=healthy,
                                        junction_idx=junction_idx, robust_ref=robust_ref if robust else None)
                ref_r, ref_where = ref.pop("ref_r"), ref.pop("ref_where")
                if ref_where not in ("proximal", "interpolated", "robust"):
                    conf *= 0.8
                    flags.append(f"reference_{ref_where}")
                if ref.pop("inconsistent"):
                    conf *= 0.8
                    flags.append("reference_inconsistent")
                if ref.pop("fallback"):
                    conf *= 0.8
                    flags.append("reference_fallback")
                if ref.pop("crosses_junction"):
                    conf *= 0.9
                    flags.append("reference_crosses_junction")
                lesion_ratio = _stat(ratio[sl], np.median)
                disagree = bool(
                    np.isfinite(lesion_ratio) and np.isfinite(path_ratio) and path_ratio > 0
                    and abs(lesion_ratio / path_ratio - 1.0) > float(_cfg.LESION_RADIUS_DISAGREEMENT_FRACTION)
                )
                # The siphon's tight bend corrupts the cross-sections; with the robust method it
                # only counts when the radii disagree there (a real paraclinoid lesion stays high).
                if siphon[sl].any() and (disagree or not robust):
                    conf *= 0.7
                    flags.append("in_siphon")
                if robust and np.isfinite(sigma) and sigma > float(_cfg.ROBUST_NOISY_SIGMA):
                    conf *= 0.8
                    flags.append("noisy_path")
                if min(float(s[a]), float(s[-1] - s[b])) < float(_cfg.LESION_NEAR_END_MM):
                    conf *= 0.6
                    flags.append("near_vessel_end")
                length = float(s[b] - s[a])
                if length < 1.5 * min_len:
                    conf *= 0.8
                    flags.append("short")
                if tree is not None and float(tree.query(pts[sl])[0].min()) < float(_cfg.LESION_NEAR_BIFURCATION_MM):
                    conf *= 0.5
                    flags.append("near_bifurcation")
                if disagree:
                    conf *= 0.7
                    flags.append("radius_disagreement")
                if float(np.mean(~np.isfinite(cs[sl]))) > 0.2:
                    conf *= 0.8
                    flags.append("cross_section_gaps")
                if np.isfinite(ref_r) and ref_r > 0 and np.isfinite(extreme_r):
                    diam_pct = (1.0 - extreme_r / ref_r) * 100.0
                    area_pct = (1.0 - (extreme_r / ref_r) ** 2) * 100.0
                    if kind == "enlargement":
                        diam_pct, area_pct = -diam_pct, -area_pct
                else:
                    diam_pct = area_pct = np.nan
                rows.append({
                    "kind": kind,
                    "detection_method": "robust_reference" if robust else "taper",
                    "segment": _segment_name_at(mid, segments) if segments else str(path.get("tree_label") or ""),
                    "path_id": str(path.get("path_id", "")),
                    "point_start": int(a),
                    "point_end": int(b),
                    "start_s_from_vessel_origin_mm": float(s[a]),
                    "length_mm": length,
                    "taper_percent": _stat(pct[sl], np.max),
                    "extreme_radius_mm": extreme_r,
                    "reference_radius_mm": float(ref_r),
                    "reference": ref_where,
                    **{k: float(v) for k, v in ref.items()},
                    "diameter_percent": float(diam_pct),
                    "area_percent": float(area_pct),
                    "confidence": float(conf),
                    "confidence_class": "high" if conf >= 0.7 else ("moderate" if conf >= 0.4 else "low"),
                    "confidence_flags": ";".join(flags),
                    "x_mm": float(mid[0]), "y_mm": float(mid[1]), "z_mm": float(mid[2]),
                })
    return rows


__all__ = ["bending_length", "blanking_mask", "lesion_rows", "path_features"]
