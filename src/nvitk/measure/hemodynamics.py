"""Phase-contrast / 4D Flow hemodynamic indices and velocity helpers.

PC-MRI velocity conventions match :mod:`nvitk.io.conversors.phase2volume`.
PI/RI definitions follow QVTplus-style ratios on time-resolved flow or velocity
series (see :func:`pulsatility_index`, :func:`resistivity_index`).

Uses :func:`nvitk.core.backend.setup` so ``np`` follows the active NumPy or CuPy
backend; inputs are coerced with :func:`~nvitk.core.array.as_backend_array`.
"""

from __future__ import annotations

import math
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using

setup(globals())

# PWV physiological acceptance window (m/s), matching QVTplus ``enc_PWV``.
PWV_MIN_M_S: float = 0.0
PWV_MAX_M_S: float = 30.0
# Cross-section quality is scored on a 0-4 scale (QVTplus ``StdvFromMean`` range);
# points below this threshold are excluded from PITC / PWV fits.
QUALITY_SCALE_MAX: float = 4.0
QUALITY_THRESH_DEFAULT: float = 2.5
# Reporting unit for flow: pipelines compute ml/s, literature bands and the DB
# (``flow_mean`` / ``flow_tseries``) are in mL/min.
ML_S_TO_ML_MIN: float = 60.0


def flow_pulsatile_ml_s(velocity_ts_mm_s, area_mm2: float) -> np.ndarray:
    """Time-resolved flow Q(t) in ml/s (``paramMap_params_threshS``: v_mean * area)."""
    v = as_backend_array(velocity_ts_mm_s).astype(np.float64).reshape(-1)
    return v * (float(area_mm2) / 1000.0)


def flow_per_heart_cycle_ml_s(flow_pulsatile: np.ndarray) -> float:
    """Cardiac time-averaged flow (ml/s)."""
    x = as_backend_array(flow_pulsatile).astype(np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def mean_flow_ml_min(flow_pulsatile) -> float:
    """
    Cardiac time-averaged flow **magnitude** in mL/min.

    Same quantity as :func:`flow_per_heart_cycle_ml_s` scaled by
    :data:`ML_S_TO_ML_MIN`, but ``|Q(t)|`` is taken first so a flipped plane
    normal (tangent polarity) cannot cancel the mean — this matches how
    ``loc_mean_flow_ml_s`` and the ``flow_mean`` DB variable are reported.
    """
    x = np.abs(as_backend_array(flow_pulsatile).astype(np.float64).reshape(-1))
    return flow_per_heart_cycle_ml_s(x) * ML_S_TO_ML_MIN


def pulsatility_index_qvt(flow_pulsatile, *, eps: float = 1e-9):
    """PI = (max - min) / mean(Q) on a non-negative flow series.

    Callers should pass ``abs(Q(t))`` so tangent polarity cannot flip the sign of
    mean flow / PI (QVTplus-style magnitude reporting).
    """
    x = np.abs(as_backend_array(flow_pulsatile).astype(np.float64).reshape(-1))
    if x.size == 0:
        return float("nan")
    den = float(np.mean(x))
    if den <= eps:
        return float("nan")
    return float((float(np.max(x)) - float(np.min(x))) / den)


def pulsatility_index(flow_t, *, eps: float = 1e-9):
    """PI = (max_t - min_t) / mean(Q) per row on non-negative flow.

    Absolute value is applied so signed through-plane polarity does not invert
    the denominator. Prefer abs'ing ``Q(t)`` once upstream when possible.
    """
    x = np.abs(as_backend_array(flow_t).astype(np.float64))
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1)
    mn = np.min(x, axis=1)
    mu = np.mean(x, axis=1)
    return ((mx - mn) / np.maximum(mu, eps)).astype(np.float64)


def resistivity_index(flow_t, *, eps: float = 1e-9):
    """RI = (max_t - min_t) / max(|flow|) per row."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1)
    mn = np.min(x, axis=1)
    den = np.maximum(np.max(np.abs(x), axis=1), eps)
    return (np.abs(mx - mn) / den).astype(np.float64)


def mean_flow_ml_s(flow_t, temporal_resolution_s: float | None = None):
    """Time-mean flow proxy (same units as *flow_t* per frame)."""
    del temporal_resolution_s
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.mean(x, axis=1)


def mean_velocity_mm_s(velocity_t):
    """Temporal mean of a 1D through-plane velocity series (mm/s)."""
    x = as_backend_array(velocity_t).astype(np.float64).reshape(-1)
    return float(np.mean(x))


def velocity_mm_s_from_phases(ap, rl, fh):
    """PC velocity components in mm/s (same convention as :mod:`nvitk.io.conversors.phase2volume`)."""
    ap = as_backend_array(ap).astype(np.float64)
    rl = as_backend_array(rl).astype(np.float64)
    fh = as_backend_array(fh).astype(np.float64)
    vx = -rl * 10.0  # R (Left 2 Right) -> RL (Right 2 Left)
    vy = -ap * 10.0  # A (Posterior 2 Anterior) -> AP (Anterior 2
    vz = fh * 10.0   # S (Inferior 2 Superior) = FH (Feet 2 Head)
    return vx, vy, vz


def through_plane_velocity_series(
    vx,
    vy,
    vz,
    *,
    i: int,
    j: int,
    k: int,
    tangent,
):
    """Time series of velocity projected onto unit *tangent* at voxel (i,j,k)."""
    t = as_backend_array(tangent).astype(np.float64).reshape(3)
    t = t / (np.linalg.norm(t) + 1e-12)
    nt = int(vx.shape[3])
    out = np.empty(nt, dtype=np.float64)
    for ti in range(nt):
        v = np.array([vx[i, j, k, ti], vy[i, j, k, ti], vz[i, j, k, ti]], dtype=np.float64)
        out[ti] = float(np.dot(v, t))
    return out


# ---------------------------------------------------------------------------
# Cross-section waveform quality (QVTplus StdvFromMean analogue)
# ---------------------------------------------------------------------------


def waveform_quality_score(flow_t, *, eps: float = 1e-9) -> float:
    """Quality of a single flow waveform on a 0-4 scale (higher = cleaner).

    Analogue of the QVTplus ``StdvFromMean`` metric adapted to a per-vessel mask:
    a clean, high-amplitude cardiac waveform (small temporal roughness relative to
    its pulsatile amplitude) scores near 4; a noisy or flat waveform scores near 0.
    """
    x = as_backend_array(flow_t).astype(np.float64).reshape(-1)
    if x.size < 3:
        return 0.0
    amp = float(np.max(x) - np.min(x))
    if amp <= eps:
        return 0.0
    roughness = float(np.std(np.diff(np.diff(x))))
    roughness_ratio = roughness / (amp + eps)
    return float(QUALITY_SCALE_MAX / (1.0 + roughness_ratio))


def branch_window_slices(length: int) -> list[slice]:
    """0-based half-open slices per station (MATLAB ``paramMap_params_threshS``)."""
    l_id = np.ones(length, dtype=int)
    if length >= 4:
        rhs = np.arange(1, length - 1, dtype=int)
        lhs = l_id[3:]
        l_id[3:] = lhs + rhs[: lhs.size]
    r_id = np.arange(3, length + 3, dtype=int)
    if length >= 3:
        r_id[-3:] = length
    return [slice(int(l_id[m]) - 1, int(r_id[m])) for m in range(length)]


def stdv_from_mean_station(
    flow_per_cycle: np.ndarray,
    area: np.ndarray,
    diam: np.ndarray,
    flow_pulsatile: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """QVTplus ``StdvFromMean`` for one station given its local window arrays.

    Each term is clipped before summing so near-zero mean flow cannot explode the
    score; the result is clamped to ``[0, QUALITY_SCALE_MAX]`` for Dempsey weights.
    """
    fpc = as_backend_array(flow_per_cycle).astype(np.float64).reshape(-1)
    ar = as_backend_array(area).astype(np.float64).reshape(-1)
    di = as_backend_array(diam).astype(np.float64).reshape(-1)
    fp = as_backend_array(flow_pulsatile).astype(np.float64)
    if fp.ndim == 1:
        fp = fp.reshape(1, -1)
    if fpc.size == 0:
        return 0.0
    mu_f = float(np.mean(fpc))
    mu_a = float(np.mean(ar))
    # Robust floors: avoid 1/eps blow-ups when mean flow/area ≈ 0.
    floor_f = max(abs(mu_f), float(np.max(np.abs(fpc))) * 0.05, eps)
    floor_a = max(abs(mu_a), float(np.max(np.abs(ar))) * 0.05 if ar.size else eps, eps)
    # qv_meanflow = float(np.clip(1.0 - float(np.std(fpc)) / floor_f, -1.0, 1.0))
    # qv_area = float(np.clip(1.0 - float(np.std(ar)) / floor_a, -1.0, 1.0))
    # qv_circ = float(np.clip(float(np.mean(di)) if di.size else 0.0, 0.0, 1.0))
    qv_meanflow = max(-1.0, min(1.0, 1.0 - float(np.std(fpc)) / floor_f))     # Clipped to [-1, 1]
    qv_area     = max(-1.0, min(1.0, 1.0 - float(np.std(ar)) / floor_a))      # Clipped to [-1, 1]
    qv_circ     = max(0.0, min(1.0, float(np.mean(di)) if di.size else 0.0))  # Clipped to [0, 1]

    minmax_phase = np.max(fp, axis=0) - np.min(fp, axis=0)
    qv_tight = float(max(-1.0, min(1.0, 1.0 - float(np.mean(minmax_phase)) / floor_f)))          # Clipped to [-1, 1]
    return max(0.0, min(float(QUALITY_SCALE_MAX), qv_meanflow + qv_area + qv_circ + qv_tight))   # Clipped to [0, QUALITY_SCALE_MAX]


def stdv_from_mean_branch(
    flow_per_cycle: np.ndarray,
    area: np.ndarray,
    diam: np.ndarray,
    flow_pulsatile: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """``StdvFromMean`` along one ordered branch (``paramMap_params_threshS``)."""
    fpc = as_backend_array(flow_per_cycle).astype(np.float64).reshape(-1)
    ar = as_backend_array(area).astype(np.float64).reshape(-1)
    di = as_backend_array(diam).astype(np.float64).reshape(-1)
    fp = as_backend_array(flow_pulsatile).astype(np.float64)
    if fp.ndim == 1:
        fp = fp.reshape(1, -1)
    n = int(fpc.size)
    out = np.empty(n, dtype=np.float64)
    for m, sl in enumerate(branch_window_slices(n)):
        out[m] = stdv_from_mean_station(
            fpc[sl], ar[sl], di[sl], fp[sl, :], eps=eps
        )
    return out


PitcQualityMetric = str  # "stdv_from_mean" | "waveform"


def station_quality_scores(
    flow_pulsatile_rows: np.ndarray,
    *,
    metric: str = "stdv_from_mean",
    flow_per_cycle: np.ndarray | None = None,
    area: np.ndarray | None = None,
    diam: np.ndarray | None = None,
) -> np.ndarray:
    """Per-station quality on one vessel branch."""
    fp = as_backend_array(flow_pulsatile_rows).astype(np.float64)
    if fp.ndim == 1:
        fp = fp.reshape(1, -1)
    n = fp.shape[0]
    if metric == "waveform":
        return np.array([waveform_quality_score(fp[i]) for i in range(n)], dtype=np.float64)
    if flow_per_cycle is None:
        flow_per_cycle = fp.mean(axis=1)
    if area is None:
        area = np.ones(n, dtype=np.float64)
    if diam is None:
        diam = np.ones(n, dtype=np.float64)
    return stdv_from_mean_branch(flow_per_cycle, area, diam, fp)


# ---------------------------------------------------------------------------
# Weighted linear regression + PITC
# ---------------------------------------------------------------------------


def weighted_linear_fit(x, y, weights=None, *, eps: float = 1e-12) -> dict[str, float]:
    """Weighted least-squares fit ``y = slope * x + intercept``.

    Returns ``slope``, ``intercept``, ``r2`` (weighted), and ``n`` (number of points).
    """
    xv = as_backend_array(x).astype("float64").reshape(-1)
    yv = as_backend_array(y).astype("float64").reshape(-1)
    n = int(min(xv.size, yv.size))
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": n}
    xv = xv[:n]
    yv = yv[:n]
    if weights is None:
        wv = _np_ones_like(yv)
    else:
        wv = as_backend_array(weights).astype("float64").reshape(-1)[:n]
    wsum = float(wv.sum())
    if wsum <= eps:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": n}
    xm = float((wv * xv).sum() / wsum)
    ym = float((wv * yv).sum() / wsum)
    sxx = float((wv * (xv - xm) ** 2).sum())
    sxy = float((wv * (xv - xm) * (yv - ym)).sum())
    if abs(sxx) <= eps:
        return {"slope": float("nan"), "intercept": float(ym), "r2": float("nan"), "n": n}
    slope = sxy / sxx
    intercept = ym - slope * xm
    pred = slope * xv + intercept
    ss_res = float((wv * (yv - pred) ** 2).sum())
    ss_tot = float((wv * (yv - ym) ** 2).sum())
    r2 = float("nan") if ss_tot <= eps else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": n}


def _np_ones_like(a):
    """Array of ones with the same shape/dtype as *a* (default weights when none are supplied)."""
    return np.ones_like(a)


def quality_weights(quality, *, thresh: float = QUALITY_THRESH_DEFAULT):
    """Dempsey-style weights ``(Q - thresh) / (scale_max - thresh)`` clipped to ``[0, 1]``."""
    q = (as_backend_array(quality)).astype("float64").reshape(-1)
    denom = max(QUALITY_SCALE_MAX - float(thresh), 1e-9)
    return np.clip((q - float(thresh)) / denom, 0.0, 1.0)


def pitc_fit(
    pi_values,
    distances_mm,
    quality=None,
    *,
    thresh: float = QUALITY_THRESH_DEFAULT,
) -> dict[str, float]:
    """Pulsatility Index Transmission Coefficient: slope of PI vs distance from root.

    ``PI(d) = pitc_slope * d + pitc_intercept``. High-quality points are up-weighted
    with :func:`quality_weights`. Returns the slope (1/mm), intercept (PI at root),
    weighted R2, point count, and quality-weighted mean PI (``global_pi``).
    """
    pi = (as_backend_array(pi_values)).astype("float64").reshape(-1)
    dist = (as_backend_array(distances_mm)).astype("float64").reshape(-1)
    n = int(min(pi.size, dist.size))
    pi = pi[:n]
    dist = dist[:n]
    if quality is None:
        weights = np.ones(n, dtype="float64")
    else:
        weights = quality_weights(quality, thresh=thresh)[:n]
    keep = weights > 0
    if int(keep.sum()) < 2:
        return {
            "pitc_slope": float("nan"),
            "pitc_intercept": float("nan"),
            "r2": float("nan"),
            "n": int(keep.sum()),
            "global_pi": float("nan"),
        }
    fit = weighted_linear_fit(dist[keep], pi[keep], weights[keep])
    wsum = float(weights[keep].sum())
    global_pi = float((weights[keep] * pi[keep]).sum() / wsum) if wsum > 0 else float("nan")
    return {
        "pitc_slope": fit["slope"],
        "pitc_intercept": fit["intercept"],
        "r2": fit["r2"],
        "n": fit["n"],
        "global_pi": global_pi,
    }


def damping_index(pi_proximal: float, pi_distal: float, *, eps: float = 1e-9) -> float:
    """Pulsatility damping index ``(PI_prox - PI_dist) / PI_prox``."""
    p = float(pi_proximal)
    if abs(p) <= eps:
        return float("nan")
    return float((p - float(pi_distal)) / p)


# ---------------------------------------------------------------------------
# Pulse wave velocity (Bjornfoot optimizer + Fielding cross-correlation)
# ---------------------------------------------------------------------------

# QVTplus ``enc_PWV_XCor`` / Rivera-style upsample of one cardiac cycle.
_XCOR_UPSAMPLE: int = 500
# QVTplus ``enc_PWV_WO`` / paper: unbounded least-squares (fminunc); initial PWV.
_BJORNFOOT_PWV0: float = 10.0


def bjornfoot_prepare_waveforms(
    flow_matrix,
    areas,
    qualities=None,
    *,
    thresh: float = QUALITY_THRESH_DEFAULT,
    weight_mode: str = "area",
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QVTplus ``enc_PWV_WO`` / Björnfot normalization and station weights.

    Per-station (matching MATLAB): convert flow→velocity ``F/A``, zero-mean,
    unit-std normalize; area weights ``A / scaling²`` then ``/ max``.
    ``weight_mode='area'`` (tag=0) uses those area weights; ``'quality'`` (tag=1)
    filters ``Q < thresh`` and uses Dempsey weights on the remaining rows.

    Returns ``(F_norm, W, keep_mask)`` where *keep_mask* selects rows kept for
    Bjornfoot (always all rows for ``area``; quality-filtered for ``quality``).
    """
    F = as_backend_array(flow_matrix).astype(np.float64)
    if F.ndim != 2:
        F = F.reshape(1, -1)
    n, _m = F.shape
    A = as_backend_array(areas).astype(np.float64).reshape(-1)[:n]
    Q = (
        as_backend_array(qualities).astype(np.float64).reshape(-1)[:n]
        if qualities is not None
        else np.full(n, QUALITY_SCALE_MAX, dtype=np.float64)
    )
    keep = np.ones(n, dtype=bool)
    if weight_mode == "quality":
        keep = Q >= float(thresh)
    F_out = F.copy()
    A_out = A.copy()
    for i in range(n):
        if not keep[i]:
            continue
        # QVTplus enc_PWV_WO: Ftemp = F./A  (velocity), then mean-subtract / unit-std.
        area_i = float(A_out[i])
        vel = F_out[i] / (area_i if abs(area_i) > eps else eps)
        ft = vel - float(np.mean(vel))
        scaling = 1.0 / (float(np.std(ft)) + eps)
        F_out[i] = ft * scaling
        A_out[i] = area_i / (scaling * scaling)
    amax = float(np.max(as_backend_array(A_out[keep]))) if int(keep.sum()) else 0.0
    if amax > eps:
        A_out = A_out / amax
    if weight_mode == "quality":
        W = quality_weights(Q, thresh=thresh)
    else:
        W = A_out.copy()
    W = np.maximum(W, 0.0)
    W[~keep] = 0.0
    return F_out, W, keep


def _pwvest3_share_components(
    in_params: np.ndarray,
    distances_m: np.ndarray,
    flows_norm: np.ndarray,
    temporal_resolution_s: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return fitted waveforms, observations, and weights for ``PWVest3_share``."""
    params = as_backend_array(in_params).astype(np.float64).reshape(-1)
    m = int(params.size - 1)
    if m < 2:
        return None
    velocity = params[:m]
    pwv = float(params[m])
    if pwv <= 0.0 or not np.isfinite(pwv):
        return None
    tr = float(temporal_resolution_s)
    D = as_backend_array(distances_m).astype(np.float64).reshape(-1)
    F = as_backend_array(flows_norm).astype(np.float64)
    Qw = as_backend_array(weights).astype(np.float64).reshape(-1)
    n = int(min(D.size, F.shape[0], Qw.size))
    if n < 1:
        return None
    D = D[:n]
    F = F[:n, :m]
    Qw = Qw[:n]
    # tV = 0:tres:(tres*3*m - tres)  → 3*m samples (MATLAB 1-based region m+1:2m).
    tV = np.arange(0.0, tr * 3.0 * m, tr, dtype=np.float64)
    if tV.size != 3 * m:
        tV = np.linspace(0.0, tr * (3 * m - 1), num=3 * m, dtype=np.float64)
    vel3 = np.tile(velocity, 3)
    region = np.arange(m, 2 * m)
    delta_t = tV[np.newaxis, :] - D[:, np.newaxis] / pwv

    from scipy.interpolate import interp1d
    interp = interp1d(
        to_numpy(tV),
        to_numpy(vel3),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    v_shift = as_backend_array(interp(to_numpy(delta_t))).astype(np.float64)[:, region]
    if not np.all(np.isfinite(v_shift)) or not np.all(np.isfinite(F)):
        return None
    return v_shift, F, np.maximum(Qw, 0.0)


def _pwvest3_share_diff(
    in_params: np.ndarray,
    distances_m: np.ndarray,
    flows_norm: np.ndarray,
    temporal_resolution_s: float,
    weights: np.ndarray,
) -> np.ndarray | None:
    """Weighted residual matrix from QVTplus ``PWVest3_share.m`` (or None if invalid)."""
    components = _pwvest3_share_components(
        in_params, distances_m, flows_norm, temporal_resolution_s, weights
    )
    if components is None:
        return None
    fitted, observed, station_weights = components
    return np.sqrt(station_weights[:, np.newaxis]) * (fitted - observed)


def _pwvest3_share_diagnostics(
    in_params: np.ndarray,
    distances_m: np.ndarray,
    flows_norm: np.ndarray,
    temporal_resolution_s: float,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-station diagnostics from the fitted Bjornfoot shared-template model."""
    components = _pwvest3_share_components(
        in_params, distances_m, flows_norm, temporal_resolution_s, weights
    )
    if components is None:
        return {}
    fitted, observed, station_weights = components
    raw_diff = fitted - observed
    weighted_rms = np.sqrt(
        np.mean(station_weights[:, np.newaxis] * raw_diff * raw_diff, axis=1)
    )
    fitted_centered = fitted - np.mean(fitted, axis=1, keepdims=True)
    observed_centered = observed - np.mean(observed, axis=1, keepdims=True)
    denom = np.sqrt(
        np.sum(fitted_centered * fitted_centered, axis=1)
        * np.sum(observed_centered * observed_centered, axis=1)
    )
    correlations = np.divide(
        np.sum(fitted_centered * observed_centered, axis=1),
        denom,
        out=np.full(fitted.shape[0], np.nan, dtype=np.float64),
        where=denom > 1e-12,
    )
    return {
        "template_norm": as_backend_array(in_params[:-1]).astype(np.float64),
        "fitted_waveforms_norm": fitted,
        "observed_waveforms_norm": observed,
        "weighted_residual_rms": weighted_rms,
        "waveform_correlation": correlations,
    }


def pwvest3_share_cost(
    in_params: np.ndarray,
    distances_m: np.ndarray,
    flows_norm: np.ndarray,
    temporal_resolution_s: float,
    weights: np.ndarray,
) -> float:
    """Scalar SSE cost from QVTplus ``PWVest3_share.m`` (joint template + PWV)."""
    diff = _pwvest3_share_diff(
        in_params, distances_m, flows_norm, temporal_resolution_s, weights
    )
    if diff is None:
        return 1e12
    return float(np.sum(diff * diff))


def pwvest3_share_residuals(
    in_params: np.ndarray,
    distances_m: np.ndarray,
    flows_norm: np.ndarray,
    temporal_resolution_s: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Flattened weighted residuals for least-squares / ML fitting."""
    m = int(as_backend_array(in_params).astype(np.float64).reshape(-1).size - 1)
    n = int(as_backend_array(flows_norm).shape[0]) if np.ndim(flows_norm) else 0
    n_res = max(n * max(m, 0), 1)
    diff = _pwvest3_share_diff(
        in_params, distances_m, flows_norm, temporal_resolution_s, weights
    )
    if diff is None:
        # Large finite residuals so the LS solver can step away from invalid PWV≤0.
        return np.full(n_res, 1e6, dtype=np.float64)
    return diff.reshape(-1)


def pwv_bjornfoot_optimize(
    distances_m,
    flow_matrix,
    temporal_resolution_s: float,
    *,
    areas=None,
    qualities=None,
    quality_thresh: float = QUALITY_THRESH_DEFAULT,
    weight_mode: str = "area",
) -> dict[str, Any]:
    """Bjornfoot PWV via QVTplus ``enc_PWV_WO`` + ``PWVest3_share`` (unbounded LS).

    Jointly fits the shared velocity template and PWV by weighted least squares
    (paper ML estimator; QVTplus ``fminunc`` analogue via ``scipy.optimize.least_squares``).
    Returns ``pwv_m_s`` and residual cost; non-positive / ≥30 m/s → ``nan``.
    """
    dist = as_backend_array(distances_m).astype(np.float64).reshape(-1)
    flows = as_backend_array(flow_matrix).astype(np.float64)
    if flows.ndim != 2 or flows.shape[0] < 2:
        return {
            "pwv_m_s": float("nan"),
            "cost": float("nan"),
            "n": int(flows.shape[0] if flows.ndim else 0),
        }
    if areas is None:
        areas_arr = np.ones(flows.shape[0], dtype=np.float64)
    else:
        areas_arr = as_backend_array(areas).astype(np.float64).reshape(-1)
    F_norm, W, keep = bjornfoot_prepare_waveforms(
        flows,
        areas_arr,
        qualities=qualities,
        thresh=quality_thresh,
        weight_mode=weight_mode,
    )
    dist_k = dist[keep]
    F_k = F_norm[keep]
    W_k = W[keep]
    n_stations = int(F_k.shape[0])
    if n_stations < 2:
        return {"pwv_m_s": float("nan"), "cost": float("nan"), "n": n_stations}
    tr = float(temporal_resolution_s)
    if tr <= 0:
        return {"pwv_m_s": float("nan"), "cost": float("nan"), "n": n_stations}
    x0 = np.concatenate([np.mean(F_k, axis=0), np.array([_BJORNFOOT_PWV0])])

    def residuals(x: np.ndarray) -> np.ndarray:
        """Least-squares residual function for the Bjornfoot PWV fit, evaluated on the host backend."""
        with using("numpy"):
            return pwvest3_share_residuals(to_numpy(x), to_numpy(dist_k), to_numpy(F_k), tr, to_numpy(W_k))

    from scipy.optimize import least_squares

    res = least_squares(
        residuals,
        to_numpy(x0),
        method="lm",
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    pwv_raw = float(res.x[-1])
    cost = float(np.sum(res.fun * res.fun)) if res.fun is not None else float("nan")
    with using("numpy"):
        diagnostics = (
            _pwvest3_share_diagnostics(to_numpy(res.x), to_numpy(dist_k), to_numpy(F_k), tr, to_numpy(W_k))
            if np.isfinite(to_numpy(pwv_raw)) and to_numpy(pwv_raw) > 0.0
            else {}
        )

    
    expected_delay_s = (
        (dist_k - dist_k[0]) / pwv_raw
        if np.isfinite(pwv_raw) and pwv_raw > 0.0
        else np.full(n_stations, np.nan, dtype=np.float64)
    )
    pwv = pwv_raw
    if not np.isfinite(pwv):
        pwv = float("nan")
    elif pwv <= 0.0 or pwv >= PWV_MAX_M_S:
        # QVTplus enc_PWV_WO: Results(end)<0 → PWV=-1 (rejected downstream).
        pwv = float("nan")
    return {
        "pwv_m_s": pwv,
        "pwv_raw_m_s": pwv_raw,
        "cost": cost,
        "n": n_stations,
        "weights": W_k,
        "expected_delay_s": expected_delay_s,
        **diagnostics,
    }


def normalize_waveform(flow_t, *, eps: float = 1e-9):
    """Zero-mean, unit-std normalization along the last axis (per row)."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def _circular_fractional_shift(x, shift_frames: float):
    """Shift a periodic 1D waveform by *shift_frames* (fractional) via interpolation."""
    xv = (as_backend_array(x)).astype(np.float64).reshape(-1)
    nt = xv.size
    if nt == 0:
        return xv
    idx = (np.arange(nt, dtype="float64") - float(shift_frames)) % nt
    lo = np.floor(idx).astype(int) % nt
    hi = (lo + 1) % nt
    frac = idx - np.floor(idx)
    return xv[lo] * (1.0 - frac) + xv[hi] * frac


def upsample_periodic_cycle(
    flow_t,
    *,
    n_up: int = _XCOR_UPSAMPLE,
) -> tuple[np.ndarray, float]:
    """Spline-upsample one periodic cardiac cycle (QVTplus Rivera / enc_PWV_XCor).

    Triples the waveform for periodic boundaries, then interpolates onto ``n_up``
    samples spanning one cycle. Returns ``(upsampled, samples_per_frame)``.
    """
    x = as_backend_array(flow_t).astype(np.float64).reshape(-1)
    nt = int(x.size)
    n_up = max(int(n_up), max(nt * 2, 8))
    if nt < 2:
        return x.copy(), 1.0
    x3 = np.concatenate([x, x, x])
    t3 = np.arange(-nt, 2 * nt, dtype=np.float64)
    t_up = np.linspace(0.0, float(nt), num=n_up, endpoint=False)
    # Cubic spline on the tripled trace (matches MATLAB interp1(...,'spline')).
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(to_numpy(t3), to_numpy(x3), bc_type="not-a-knot")
    return as_backend_array(cs(to_numpy(t_up)).astype(np.float64)), float(n_up) / float(nt)


def circular_cross_correlation_lag(reference, signal) -> tuple[float, float]:
    """Integer lag (samples) maximizing circular cross-correlation and its correlation.

    Returns ``(lag_samples, corr)`` where *lag_samples* is how far to roll *signal*
    to match *reference* (wrapped to ``(-n/2, n/2]``). Prefer
    :func:`cross_correlation_delay_seconds` for PWV (sign + sub-frame upsample).
    """
    ref = (as_backend_array(reference)).astype(np.float64).reshape(-1)
    sig = (as_backend_array(signal)).astype(np.float64).reshape(-1)
    nt = int(min(ref.size, sig.size))
    if nt < 2:
        return 0.0, 0.0
    ref = ref[:nt]
    sig = sig[:nt]
    ref = (ref - ref.mean()) / (ref.std() + 1e-9)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(nt):
        shifted = np.roll(sig, lag)
        corr = float(np.dot(ref, shifted) / nt)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag > nt // 2:
        best_lag -= nt
    return float(best_lag), float(best_corr)


def cross_correlation_delay_seconds(
    reference,
    signal,
    temporal_resolution_s: float,
    *,
    n_up: int = _XCOR_UPSAMPLE,
) -> tuple[float, float]:
    """Transit delay (s) of *signal* relative to *reference* via upsampled XCor.

    Positive delay means *signal* arrives later than *reference* (distal later than
    proximal). Uses QVTplus-style spline upsampling of one cardiac cycle.
    """
    tr = float(temporal_resolution_s)
    if tr <= 0:
        return float("nan"), 0.0
    ref_u, spp = upsample_periodic_cycle(reference, n_up=n_up)
    sig_u, _ = upsample_periodic_cycle(signal, n_up=n_up)
    lag_samples, corr = circular_cross_correlation_lag(ref_u, sig_u)
    # lag_samples = roll(signal) to match ref. Delayed distal → negative roll →
    # arrival delay = -lag (in original frames).
    delay_frames = -float(lag_samples) / float(spp)
    return delay_frames * tr, float(corr)


def time_to_upstroke_seconds(
    flow_ts,
    temporal_resolution_s: float,
    *,
    n_up: int = _XCOR_UPSAMPLE,
) -> float:
    """Time (s) of maximal systolic upslope on an upsampled periodic waveform."""
    tr = float(temporal_resolution_s)
    if tr <= 0:
        return float("nan")
    up, spp = upsample_periodic_cycle(flow_ts, n_up=n_up)
    if up.size < 3:
        return float("nan")
    d = np.diff(up)
    i = int(np.argmax(d))
    return float(i / spp) * tr


def _mad_outlier_mask(y: np.ndarray, *, z_thresh: float = 3.5) -> np.ndarray:
    """True for inliers under a robust MAD z-score (QVTplus ``isoutlier`` analogue)."""
    yv = as_backend_array(y).astype(np.float64).reshape(-1)
    keep = np.isfinite(yv)
    if int(keep.sum()) < 4:
        return keep
    med = float(np.median(yv[keep]))
    mad = float(np.median(np.abs(yv[keep] - med)))
    if mad < 1e-12:
        return keep
    z = 0.6745 * (yv - med) / mad
    return keep & (np.abs(z) <= float(z_thresh))


def pwv_fielding_xcor(
    distances_m,
    flow_matrix,
    temporal_resolution_s: float,
    *,
    weights=None,
    reference_index: int = 0,
    n_up: int = _XCOR_UPSAMPLE,
    reject_outliers: bool = True,
) -> dict[str, float]:
    """Fielding-style PWV: upsampled cross-correlation delay vs distance.

    *flow_matrix* is ``(n_stations, n_frames)`` ordered along the vessel. Delays are
    measured relative to *reference_index*; ``tau = distance / PWV`` is fit so
    ``PWV = 1 / slope``. Returns ``pwv_m_s``, ``r`` (mean |correlation|), and ``n``.
    """
    dist = (as_backend_array(distances_m)).astype("float64").reshape(-1)
    flows = (as_backend_array(flow_matrix)).astype("float64")
    if flows.ndim != 2 or flows.shape[0] < 2:
        return {
            "pwv_m_s": float("nan"),
            "r": float("nan"),
            "n": int(flows.shape[0] if flows.ndim else 0),
        }
    tr = float(temporal_resolution_s)
    n = int(flows.shape[0])
    # ref_i = int(np.clip(reference_index, 0, n - 1))
    ref_i = max(0, min(reference_index, n - 1))
    ref = flows[ref_i]
    lags_s = np.zeros(n, dtype="float64")
    corrs = np.zeros(n, dtype="float64")
    for i in range(n):
        delay_s, corr = cross_correlation_delay_seconds(
            ref, flows[i], tr, n_up=n_up
        )
        lags_s[i] = delay_s
        corrs[i] = abs(corr)
    if weights is None:
        wv = np.ones(n, dtype="float64")
    else:
        wv = (as_backend_array(weights)).astype("float64").reshape(-1)[:n]
        wv = np.clip(wv, 0.0, None)
    keep = np.isfinite(lags_s) & np.isfinite(dist) & (wv > 0)
    if reject_outliers and int(keep.sum()) >= 4:
        keep &= _mad_outlier_mask(lags_s)
    if int(keep.sum()) < 2:
        return {"pwv_m_s": float("nan"), "r": float(np.mean(corrs)), "n": int(keep.sum())}
    fit = weighted_linear_fit(dist[keep], lags_s[keep], wv[keep])
    slope = fit["slope"]
    pwv = (
        1.0 / slope
        if slope and np.isfinite(slope) and abs(slope) > 1e-12
        else float("nan")
    )
    return {
        "pwv_m_s": float(pwv),
        "r": float(np.mean(corrs[keep])),
        "n": int(keep.sum()),
    }


def accept_pwv(pwv_m_s: float) -> bool:
    """QVTplus acceptance gate: ``0 < PWV < 30`` m/s."""
    try:
        v = float(pwv_m_s)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    return PWV_MIN_M_S < v < PWV_MAX_M_S



# ──────────────────────────────────────────────────────────────────────────────
# Automatic quality control — literature-grounded plausibility and consistency
# ──────────────────────────────────────────────────────────────────────────────
# The scores above answer "is this waveform smooth?". These answer "is this number
# physiologically possible, and is it consistent with the vessel's neighbours?" —
# which is what catches a mis-placed LOC, an aliased velocity or a leaked segmentation.
#
# Bands come from Zarrinkoob et al. 2015 (J Cereb Blood Flow Metab 35:648-654), 94
# healthy subjects by 2D PCMRI. They are **not** used verbatim: 4D flow reads 20-46%
# lower than 2D PC-MRI at matched levels (Neuroinformatics 2021, 7T ICA at C3/C7), so
# the soft band is widened to ``0.5*(mu - 3*sd)`` .. ``1.3*(mu + 3*sd)``. The point is to
# catch gross failures (flow 5-10x physiological, or ~0 in a patent vessel), not to grade
# normal variation - PI/RI and cohort statistics are for that.

#: Per-vessel time-averaged |flow| soft bands in mL/min, widened for the 4D-vs-2D bias.
FLOW_PLAUSIBILITY_ML_MIN = {
    "ICA":     (56.5, 521.3),
    "VA":      (17.0, 215.8),
    "BA":      (11.0, 348.4),   
    "MCA":     (26.5, 310.7),
    "MCAdist": (1.0,  110.5),   
    "ACA":     (14.0, 176.8),
    "ACAdist": (1.0,  78.0),    
    "PCA":     (9.0,  117.0),
}

#: Vessels deliberately given no band. A communicating artery carrying near-zero or
#: reversed flow is a normal circle of Willis, not a QC failure -- Krabbe-Hartkamp et al.
#: 1998 found the CoW anatomically complete in only ~51% of healthy subjects.
FLOW_PLAUSIBILITY_EXEMPT: frozenset[str] = frozenset({"ACOMM", "PCOMM", "LPCOMM", "RPCOMM"})

#: Anterior share of total cerebral inflow, and the screening tolerance around it.
#: Zarrinkoob et al. report 72/28% with SD ~4-5%, independent of age, sex and brain
#: volume; the wider tolerance here is because this is a per-subject screen and CoW
#: variants (fetal PCA, hypoplastic A1) legitimately shift the ratio.
ANTERIOR_SHARE_PCT: float = 72.0
ANTERIOR_SHARE_TOL_PCT: float = 10.0

#: How far outside its band a flow may sit before it scores zero, as a fraction of the bound it
#: crossed. 0.15 means 15% below the floor (or above the ceiling) is fully implausible. Small
#: because the bands are already widened well past any healthy cohort — see
#: :func:`flow_plausibility_score`. Raise it to be more forgiving, lower it to be stricter.
FLOW_BAND_TOLERANCE: float = 0.15

#: Krabbe-Hartkamp et al. 1998 definition of CoW hypoplasia on 3D TOF MRA.
HYPOPLASIA_DIAM_MM: float = 0.8

#: Default relative junction residual beyond which mass conservation is considered violated.
#: Cranial literature does **not** publish one universal threshold: ISMRM 2017 (Roberts et al.)
#: observed ~1–10% residuals at well-conserved proximal junctions and 11–55% where flow was
#: clearly broken, while venous confluence imbalances of ~4–9% are typical even in good data
#: (Sci Rep 2025). Class-specific tolerances below are preferred; this default is the arterial
#: gate used when a single number is needed (filters, CLI ``--conservation-tol``).
CONSERVATION_TOL: float = 0.10

#: Proximal arterial junctions (ICA → ACA+MCA, VA → BA). Upper end of the "good" 1–10% band.
CONSERVATION_TOL_ARTERIAL: float = 0.15

#: Distal / incomplete arterial junctions (BA → PCA). Smaller vessels plus unmeasured AICA/SCA
#: outflow systematically inflate the residual, so the gate is looser than the proximal one.
CONSERVATION_TOL_DISTAL: float = 0.20

#: Venous confluence (SSS + straight sinus → transverse sinuses). Direct cortical / petrosal /
#: emissary tributaries make a zero residual anatomically unreachable; ~20% catches gross
#: failures without flagging every healthy subject.
CONSERVATION_TOL_VENOUS: float = 0.20

#: Coefficient of variation of flow along a non-branching segment beyond which the
#: centerline is likely drifting or a side branch was missed. The QVT validation paper
#: reported along-segment percent variation with SD ≈ 3%; 0.15 is a soft upper gate
#: (≈ 5× that scatter) for automatic review rather than a physiological hard limit.
SEGMENT_CV_TOL: float = 0.15


def _vessel_band_key(vessel_name: str) -> str:
    """
    Reduce a qvtpy vessel label to the key its literature band is stored under.

    Side prefixes carry no physiology here -- the bands are reported right/left averaged --
    so ``LICA``, ``Right_ICA`` and ``ICA`` share one band. Distal segments keep their own.
    """
    raw = str(vessel_name or "").strip()
    if not raw:
        return ""
    token = raw.upper().replace("-", "_").replace(" ", "_")
    for prefix in ("LEFT_", "RIGHT_", "L_", "R_"):
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    else:
        # Bare side letter, but only when what follows is a known stem: ``LICA`` is left
        # ICA, while ``LSCA`` must not be stripped into a band that does not exist.
        if len(token) > 2 and token[0] in "LR" and token[1:] in {
            k.upper() for k in FLOW_PLAUSIBILITY_ML_MIN
        } | {v.upper() for v in FLOW_PLAUSIBILITY_EXEMPT}:
            token = token[1:]
    aliases = {"BASI": "BA", "BASILAR": "BA", "VERT": "VA", "VERTEBRAL": "VA",
               "MCADIST": "MCAdist", "ACADIST": "ACAdist", "M1": "MCA", "A1": "ACA",
               "P2": "PCA"}
    token = aliases.get(token, token)
    for key in FLOW_PLAUSIBILITY_ML_MIN:
        if token == key.upper():
            return key
    return token


def flow_plausibility_score(mean_flow_ml_min: float, vessel_name: str) -> float:
    """
    Literature-band plausibility of a time-averaged |flow|, in ``[0, 1]``.

    1.0 anywhere inside the vessel's soft band, decaying linearly to 0.0 one band
    half-width outside either edge -- which puts the zero point at the hard implausibility
    cap (2x the soft band) used elsewhere in the qvtpy flow-cap convention.

    Returns ``NaN`` for a vessel with no meaningful band (the communicating arteries, and
    anything unrecognised), so callers skip rather than silently pass or fail it.

    Parameters
    ----------
    mean_flow_ml_min : float
        Time-averaged flow magnitude in **mL/min**. Note ``loc_mean_flow_ml_s`` is per
        second; multiply by 60 first.
    """
    key = _vessel_band_key(vessel_name)
    if key in FLOW_PLAUSIBILITY_EXEMPT or key not in FLOW_PLAUSIBILITY_ML_MIN:
        return float("nan")
    lo, hi = FLOW_PLAUSIBILITY_ML_MIN[key]
    value = abs(float(mean_flow_ml_min))
    if not math.isfinite(value):
        return float("nan")
    if lo <= value <= hi:
        return 1.0

    # Outside the band the score falls steeply, because the band is already generous: it spans
    # ``0.5*(mu - 3*sd)`` to ``1.3*(mu + 3*sd)``, which is far wider than any healthy cohort. A
    # measurement outside it is not an unusual subject, it is a number that could not have been
    # produced by a correctly segmented vessel. Decaying gently over a band half-width (as this
    # once did) undid the widening: it let an ICA at 31 mL/min -- an eighth of a healthy one --
    # score 0.55 and pass a 0.5 gate.
    #
    # The distance is measured **relative to the bound that was crossed**, so one tolerance works
    # for every vessel whatever its scale, and the score reaches 0 at
    # :data:`FLOW_BAND_TOLERANCE` beyond it. Symmetric on both sides: the low end already carries
    # extra room for the known 4D-vs-2D underestimation, so it needs no further leniency here.
    bound = hi if value > hi else lo
    excess = abs(value - bound) / max(abs(bound), 1e-6)
    return float(max(0.0, 1.0 - excess / max(FLOW_BAND_TOLERANCE, 1e-6)))


def is_plausibly_hypoplastic(
    cross_section_area_mm2: float, *, diam_thresh_mm: float = HYPOPLASIA_DIAM_MM
) -> bool:
    """
    Whether a vessel's segmented caliber puts it at or under the hypoplasia threshold.

    Use this to *gate* :func:`flow_plausibility_score` and the anterior/posterior check: a
    hypoplastic vessel legitimately carries almost no flow, and scoring it against a
    healthy-cohort band would report normal anatomy as a data-quality failure.

    Threshold from Krabbe-Hartkamp et al. 1998 (<0.8 mm diameter on 3D TOF MRA).
    """
    area = float(cross_section_area_mm2)
    if not math.isfinite(area) or area <= 0.0:
        return True
    equivalent_diameter_mm = 2.0 * math.sqrt(area / math.pi)
    return bool(equivalent_diameter_mm < float(diam_thresh_mm))


def bifurcation_conservation_error(
    parent_flow_ml_s: float, branch_flows_ml_s: Any
) -> float:
    """
    Relative mass-conservation residual at one junction: ``(Qp - sum(Qb)) / Qp``.

    Zero is perfect conservation. This is the same junction mass-balance check used by the
    cranial QVT/CPS validation work and by cardiac 4D Flow CMR quality assurance
    (conservation of mass). Relative residual ``(Qp - sum(Qb)) / Qp``.

    A failure is genuinely ambiguous between "the flow measurement is wrong" and "the
    vessel tree is incomplete" -- a missed side branch removes real outflow and shows up
    identically. 

    Returns ``NaN`` when the parent flow is ~0, since the relative residual is undefined.
    """
    parent = float(parent_flow_ml_s)
    if not math.isfinite(parent) or abs(parent) < 1e-9:
        return float("nan")
    with using("cpu"):
        branches = to_numpy(as_backend_array(branch_flows_ml_s)).astype(float).reshape(-1)
        branches = branches[np.isfinite(branches)]
        if branches.size == 0:
            return float("nan")
        return float((parent - float(branches.sum())) / parent)


def segment_flow_consistency_cv(flow_per_cycle_stations: Any) -> float:
    """
    Coefficient of variation of time-averaged flow along one non-branching segment.

    Mass is conserved along a segment with no branches, so the station-to-station flow
    should barely move. A low CV (under ~0.10-0.15) says the centerline and segmentation
    track the vessel; a high one says the centerline drifted, partial volume ate the
    lumen, or a side branch was missed. Mirrors the along-segment Gaussian-fit consistency
    check from the cranial QVT/CPS validation paper.

    Returns ``NaN`` for fewer than three stations or a mean of ~0, where a CV means nothing.
    """
    with using("cpu"):
        values = to_numpy(as_backend_array(flow_per_cycle_stations)).astype(float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size < 3:
            return float("nan")
        mean = float(values.mean())
        if abs(mean) < 1e-9:
            return float("nan")
        return float(values.std() / abs(mean))


def anterior_posterior_share_pct(ica_flow_ml_min: float, va_flow_ml_min: float) -> float:
    """
    Anterior (carotid) share of total cerebral inflow, as a percentage.

    Both arguments are the **summed bilateral** flows: ICA left + right, and vertebral
    left + right (or the basilar, which carries their confluence). Returns ``NaN`` when the
    total is not positive.
    """
    anterior = float(ica_flow_ml_min)
    posterior = float(va_flow_ml_min)
    total = anterior + posterior
    if not math.isfinite(total) or total <= 0.0:
        return float("nan")
    return float(100.0 * anterior / total)


def anterior_posterior_split_flag(
    ica_flow_ml_min: float,
    va_flow_ml_min: float,
    *,
    expected_anterior_pct: float = ANTERIOR_SHARE_PCT,
    tolerance_pct: float = ANTERIOR_SHARE_TOL_PCT,
) -> bool:
    """
    Whether the anterior/posterior flow split falls outside its expected band.

    Zarrinkoob et al. report a 72/28% split with SD ~4-5%, stable across age, sex and brain
    volume -- which makes it a cheap subject-level screen that needs no reference scan. The
    tolerance is deliberately wider than the population SD because anatomic variants (fetal
    PCA, hypoplastic A1) shift the ratio without any measurement being wrong.

    ``True`` means "outside the band", including the degenerate no-inflow case.
    """
    share = anterior_posterior_share_pct(ica_flow_ml_min, va_flow_ml_min)
    if not math.isfinite(share):
        return True
    return bool(abs(share - float(expected_anterior_pct)) > float(tolerance_pct))


__all__ = [
    "ML_S_TO_ML_MIN",
    "PWV_MAX_M_S",
    "PWV_MIN_M_S",
    "QUALITY_SCALE_MAX",
    "QUALITY_THRESH_DEFAULT",
    "accept_pwv",
    "bjornfoot_prepare_waveforms",
    "circular_cross_correlation_lag",
    "cross_correlation_delay_seconds",
    "damping_index",
    "mean_flow_ml_min",
    "mean_flow_ml_s",
    "mean_velocity_mm_s",
    "normalize_waveform",
    "branch_window_slices",
    "flow_per_heart_cycle_ml_s",
    "ANTERIOR_SHARE_PCT",
    "ANTERIOR_SHARE_TOL_PCT",
    "CONSERVATION_TOL",
    "CONSERVATION_TOL_ARTERIAL",
    "CONSERVATION_TOL_DISTAL",
    "CONSERVATION_TOL_VENOUS",
    "FLOW_BAND_TOLERANCE",
    "FLOW_PLAUSIBILITY_EXEMPT",
    "FLOW_PLAUSIBILITY_ML_MIN",
    "HYPOPLASIA_DIAM_MM",
    "SEGMENT_CV_TOL",
    "anterior_posterior_share_pct",
    "anterior_posterior_split_flag",
    "bifurcation_conservation_error",
    "flow_plausibility_score",
    "is_plausibly_hypoplastic",
    "segment_flow_consistency_cv",
    "flow_pulsatile_ml_s",
    "pitc_fit",
    "pulsatility_index",
    "pulsatility_index_qvt",
    "pwv_bjornfoot_optimize",
    "pwv_fielding_xcor",
    "pwvest3_share_cost",
    "pwvest3_share_residuals",
    "quality_weights",
    "resistivity_index",
    "through_plane_velocity_series",
    "time_to_upstroke_seconds",
    "upsample_periodic_cycle",
    "velocity_mm_s_from_phases",
    "station_quality_scores",
    "stdv_from_mean_branch",
    "stdv_from_mean_station",
    "waveform_quality_score",
    "weighted_linear_fit",
]
