# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morpho/vessel_features.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Scale-aware vessel descriptors and lesion severity / confidence for the compact vessel tables.

Inspired by the segment features of FLOWCAT ``arterial`` (feature_extraction: proximal / distal
and end-window diameters, min/max diameter ratio, tortuosity of the first / last N mm, bending
length, cumulative curvature, curvature energy, direction angles, bifurcation blanking) and
re-implemented here from their definitions — no code is shared (``arterial`` is PolyForm
Noncommercial). Two deliberate differences: curvature integrals are arc-length integrals
(``∫κ ds``, ``∫κ² ds``) rather than sums over nodes, so they do not depend on the sampling step;
and curvature is measured on the centerline smoothed at ``FEATURE_CURVATURE_SMOOTH_MM`` (true
1/mm, :func:`~nvitk_vendor.measure.morpho.metrics.smoothed_curvature`).

Radii: ``mis`` is the maximum-inscribed-sphere radius (VMTK, or its native equivalent), ``cs`` the
cross-section equivalent radius. Diameters are ``2 × radius``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from nvitk_vendor.measure import morphometrics_config as _cfg

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


def blanking_mask(points: np.ndarray, radius: np.ndarray, junctions: np.ndarray | None) -> np.ndarray:
    """True where a point lies within the bifurcation blanking zone of any junction:
    ``max(BLANKING_MM, BLANKING_RADIUS_FACTOR × local radius)``."""
    n = len(points)
    if junctions is None or len(junctions) == 0 or n == 0:
        return np.zeros(n, dtype=bool)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(junctions, dtype=float)).query(points)
    r = np.where(np.isfinite(radius), radius, 0.0)
    zone = np.maximum(float(_cfg.BLANKING_MM), float(_cfg.BLANKING_RADIUS_FACTOR) * r)
    return d <= zone


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
    # (inscribed radius → 0) and the cross-section there cuts the cap, not the lumen.
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


# ──────────────────────────────────────────────────────────────────────────────
# Lesions
# ──────────────────────────────────────────────────────────────────────────────


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
    detection measure) and against the **proximal healthy reference** — the median radius over
    ``LESION_PROXIMAL_REFERENCE_WINDOW_MM`` upstream on the same path (downstream when there is no
    upstream room) — as diameter and area stenosis (NASCET-like); for enlargements both are signed
    so that positive means wider than the reference. Each kind is measured on the radius its test
    used: stenoses on the detection (inscribed) radius, enlargements on ``enlargement_radius_mm``
    (cow tests them on the cross-section radius, or on the detection radius on re-measured paths).

    Confidence ∈ [0, 1] multiplies a penalty for each condition that makes a calibre change
    unreliable (the applied ones are listed in ``confidence_flags``):

    ===================== ======= ====================================================================
    flag                  factor  condition
    ===================== ======= ====================================================================
    reference_distal/none 0.8     no proximal reference window (distal one used, or none)
    in_siphon             0.7     overlaps the siphon (high-curvature) mask
    near_vessel_end       0.6     within ``LESION_NEAR_END_MM`` of a path end
    short                 0.8     shorter than 1.5 × the detection minimum length
    near_bifurcation      0.5     within ``LESION_NEAR_BIFURCATION_MM`` of a junction
    radius_disagreement   0.7     cross-section / inscribed radius ratio departs from the path's
                                  median ratio by more than ``LESION_RADIUS_DISAGREEMENT_FRACTION``
                                  (one estimator sees the lesion, the other does not)
    cross_section_gaps    0.8     more than 20 % of the lesion has no valid cross-section
    ===================== ======= ====================================================================

    Classes: high ≥ 0.7 (at most one mild penalty), moderate ≥ 0.4, low below.
    """
    lo_w, hi_w = (float(v) for v in _cfg.LESION_PROXIMAL_REFERENCE_WINDOW_MM)
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
        mis = np.asarray(path.get("maximum_inscribed_sphere_radius_mm", np.full(n, np.nan)), dtype=float)
        cs = np.asarray(path.get("radius_mm", np.full(n, np.nan)), dtype=float)
        siphon = np.asarray(path.get("siphon_mask_point", np.zeros(n)), dtype=float) >= 0.5
        # The area-equivalent cross-section radius normally exceeds the inscribed radius by 10–25 %
        # (non-circular lumen); a lesion is suspicious when its ratio departs from the path's own.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mis > 0, cs / mis, np.nan)
        path_ratio = _stat(ratio, np.median)
        for kind, runs_key, pct_key, min_len, kind_r in (
            ("stenosis", "stenosis_segments_point_idx", "stenosis_percent_point", float(_cfg.STENOSIS_MIN_LEN_MM),
             det_r),
            ("enlargement", "enlargement_segments_point_idx", "enlargement_percent_point",
             float(_cfg.ENLARGEMENT_MIN_LEN_MM), enl_r),
        ):
            pct = np.asarray(path.get(pct_key, np.full(n, np.nan)), dtype=float)
            for a, b in _parse_runs(path.get(runs_key)):
                a, b = max(0, min(a, n - 1)), max(0, min(b, n - 1))
                if b < a:
                    a, b = b, a
                mid = pts[(a + b) // 2]
                if any(float(np.linalg.norm(mid - m)) < 1.0 for m in seen[kind]):
                    continue  # the same lesion, measured again on an overlapping path
                seen[kind].append(mid)
                sl = slice(a, b + 1)
                extreme_r = _stat(kind_r[sl], np.min) if kind == "stenosis" else _stat(kind_r[sl], np.max)
                flags: list[str] = []
                conf = 1.0
                up = (s >= s[a] - hi_w) & (s <= s[a] - lo_w)
                if up.sum() >= 3:
                    ref_r, ref_where = _stat(kind_r[up], np.median), "proximal"
                else:
                    down = (s >= s[b] + lo_w) & (s <= s[b] + hi_w)
                    ref_r, ref_where = (_stat(kind_r[down], np.median), "distal") if down.sum() >= 3 else (np.nan, "none")
                if ref_where != "proximal":
                    conf *= 0.8
                    flags.append(f"reference_{ref_where}")
                if siphon[sl].any():
                    conf *= 0.7
                    flags.append("in_siphon")
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
                lesion_ratio = _stat(ratio[sl], np.median)
                if (
                    np.isfinite(lesion_ratio) and np.isfinite(path_ratio) and path_ratio > 0
                    and abs(lesion_ratio / path_ratio - 1.0) > float(_cfg.LESION_RADIUS_DISAGREEMENT_FRACTION)
                ):
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
                    "diameter_percent": float(diam_pct),
                    "area_percent": float(area_pct),
                    "confidence": float(conf),
                    "confidence_class": "high" if conf >= 0.7 else ("moderate" if conf >= 0.4 else "low"),
                    "confidence_flags": ";".join(flags),
                    "x_mm": float(mid[0]), "y_mm": float(mid[1]), "z_mm": float(mid[2]),
                })
    return rows


__all__ = ["bending_length", "blanking_mask", "lesion_rows", "path_features"]
