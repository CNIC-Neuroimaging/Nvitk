"""Phase-contrast / 4D Flow hemodynamic indices and velocity helpers.

PC-MRI velocity conventions match :mod:`nvitk.io.conversors.phase2volume`.
PI/RI definitions follow QVTplus-style ratios on time-resolved flow or velocity
series (see :func:`pulsatility_index`, :func:`resistivity_index`).

Uses :func:`nvitk.core.backend.setup` so ``np`` follows the active NumPy or CuPy
backend; inputs are coerced with :func:`~nvitk.core.array.as_backend_array`.
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup

setup(globals())

# PWV physiological acceptance window (m/s), matching QVTplus ``enc_PWV``.
PWV_MIN_M_S: float = 0.0
PWV_MAX_M_S: float = 30.0
# Cross-section quality is scored on a 0-4 scale (QVTplus ``StdvFromMean`` range);
# points below this threshold are excluded from PITC / PWV fits.
QUALITY_SCALE_MAX: float = 4.0
QUALITY_THRESH_DEFAULT: float = 2.5


def flow_pulsatile_ml_s(velocity_ts_mm_s, area_mm2: float) -> np.ndarray:
    """Time-resolved flow Q(t) in ml/s (``paramMap_params_threshS``: v_mean * area)."""
    v = to_numpy(as_backend_array(velocity_ts_mm_s)).astype(np.float64).reshape(-1)
    return v * (float(area_mm2) / 1000.0)


def flow_per_heart_cycle_ml_s(flow_pulsatile: np.ndarray) -> float:
    """Cardiac time-averaged flow (ml/s)."""
    x = to_numpy(as_backend_array(flow_pulsatile)).astype(np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def pulsatility_index_qvt(flow_pulsatile, *, eps: float = 1e-9):
    """PI per ``paramMap_params_threshS``: ``abs(max-min) / mean(Q)`` on signed flow."""
    x = to_numpy(as_backend_array(flow_pulsatile)).astype(np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    den = float(np.mean(x))
    if abs(den) <= eps:
        return float("nan")
    return float(abs(float(np.max(x)) - float(np.min(x))) / den)


def pulsatility_index(flow_t, *, eps: float = 1e-9):
    """PI = (max_t - min_t) / mean(|flow|) per row (legacy signed-flow variant)."""
    x = as_backend_array(flow_t).astype(np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mx = np.max(x, axis=1)
    mn = np.min(x, axis=1)
    mu = np.mean(np.abs(x), axis=1)
    return (np.abs(mx - mn) / np.maximum(mu, eps)).astype(np.float64)


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
    """QVTplus ``StdvFromMean`` for one station given its local window arrays."""
    fpc = to_numpy(as_backend_array(flow_per_cycle)).astype(np.float64).reshape(-1)
    ar = to_numpy(as_backend_array(area)).astype(np.float64).reshape(-1)
    di = to_numpy(as_backend_array(diam)).astype(np.float64).reshape(-1)
    fp = to_numpy(as_backend_array(flow_pulsatile)).astype(np.float64)
    if fp.ndim == 1:
        fp = fp.reshape(1, -1)
    if fpc.size == 0:
        return 0.0
    mu_f = float(np.mean(fpc))
    mu_a = float(np.mean(ar))
    qv_meanflow = 1.0 - float(np.std(fpc)) / max(abs(mu_f), eps)
    qv_area = 1.0 - float(np.std(ar)) / max(abs(mu_a), eps)
    qv_circ = float(np.mean(di)) if di.size else 0.0
    minmax_phase = np.max(fp, axis=0) - np.min(fp, axis=0)
    qv_tight = 1.0 - float(np.mean(minmax_phase)) / max(abs(mu_f), eps)
    return float(qv_meanflow + qv_area + qv_circ + qv_tight)


def stdv_from_mean_branch(
    flow_per_cycle: np.ndarray,
    area: np.ndarray,
    diam: np.ndarray,
    flow_pulsatile: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """``StdvFromMean`` along one ordered branch (``paramMap_params_threshS``)."""
    fpc = to_numpy(as_backend_array(flow_per_cycle)).astype(np.float64).reshape(-1)
    ar = to_numpy(as_backend_array(area)).astype(np.float64).reshape(-1)
    di = to_numpy(as_backend_array(diam)).astype(np.float64).reshape(-1)
    fp = to_numpy(as_backend_array(flow_pulsatile)).astype(np.float64)
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
    fp = to_numpy(as_backend_array(flow_pulsatile_rows)).astype(np.float64)
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
    xv = to_numpy(as_backend_array(x)).astype("float64").reshape(-1)
    yv = to_numpy(as_backend_array(y)).astype("float64").reshape(-1)
    n = int(min(xv.size, yv.size))
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": n}
    xv = xv[:n]
    yv = yv[:n]
    if weights is None:
        wv = _np_ones_like(yv)
    else:
        wv = to_numpy(as_backend_array(weights)).astype("float64").reshape(-1)[:n]
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
    import numpy as _np

    return _np.ones_like(a)


def quality_weights(quality, *, thresh: float = QUALITY_THRESH_DEFAULT):
    """Dempsey-style weights ``(Q - thresh) / (scale_max - thresh)`` clipped to >= 0."""
    import numpy as _np

    q = to_numpy(as_backend_array(quality)).astype("float64").reshape(-1)
    denom = max(QUALITY_SCALE_MAX - float(thresh), 1e-9)
    return _np.clip((q - float(thresh)) / denom, 0.0, None)


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
    import numpy as _np

    pi = to_numpy(as_backend_array(pi_values)).astype("float64").reshape(-1)
    dist = to_numpy(as_backend_array(distances_mm)).astype("float64").reshape(-1)
    n = int(min(pi.size, dist.size))
    pi = pi[:n]
    dist = dist[:n]
    if quality is None:
        weights = _np.ones(n, dtype="float64")
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
    import numpy as _np

    xv = to_numpy(as_backend_array(x)).astype("float64").reshape(-1)
    nt = xv.size
    if nt == 0:
        return xv
    idx = (_np.arange(nt, dtype="float64") - float(shift_frames)) % nt
    lo = _np.floor(idx).astype(int) % nt
    hi = (lo + 1) % nt
    frac = idx - _np.floor(idx)
    return xv[lo] * (1.0 - frac) + xv[hi] * frac


def circular_cross_correlation_lag(reference, signal) -> tuple[float, float]:
    """Integer lag (frames) maximizing circular cross-correlation and its correlation.

    Ported from QVTplus ``slidingxCor``: both waveforms are normalized, then the
    circular shift of *signal* that best matches *reference* is returned.
    """
    import numpy as _np

    ref = to_numpy(as_backend_array(reference)).astype("float64").reshape(-1)
    sig = to_numpy(as_backend_array(signal)).astype("float64").reshape(-1)
    nt = int(min(ref.size, sig.size))
    if nt < 2:
        return 0.0, 0.0
    ref = ref[:nt]
    sig = sig[:nt]
    ref = (ref - ref.mean()) / (ref.std() + 1e-9)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    best_lag = 0
    best_corr = -_np.inf
    for lag in range(nt):
        shifted = _np.roll(sig, lag)
        corr = float(_np.dot(ref, shifted) / nt)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag > nt // 2:
        best_lag -= nt
    return float(best_lag), float(best_corr)


def pwv_fielding_xcor(
    distances_m,
    flow_matrix,
    temporal_resolution_s: float,
    *,
    weights=None,
    reference_index: int = 0,
) -> dict[str, float]:
    """Fielding-style PWV: cross-correlation lag vs distance, weighted linear fit.

    *flow_matrix* is ``(n_stations, n_frames)`` ordered along the vessel. Lags are
    measured relative to *reference_index*; ``tau = distance / PWV`` is fit so
    ``PWV = 1 / slope``. Returns ``pwv_m_s``, ``r`` (mean |correlation|), and ``n``.
    """
    import numpy as _np

    dist = to_numpy(as_backend_array(distances_m)).astype("float64").reshape(-1)
    flows = to_numpy(as_backend_array(flow_matrix)).astype("float64")
    if flows.ndim != 2 or flows.shape[0] < 2:
        return {"pwv_m_s": float("nan"), "r": float("nan"), "n": int(flows.shape[0] if flows.ndim else 0)}
    tr = float(temporal_resolution_s)
    ref = flows[int(reference_index)]
    lags_s = _np.zeros(flows.shape[0], dtype="float64")
    corrs = _np.zeros(flows.shape[0], dtype="float64")
    for i in range(flows.shape[0]):
        lag_frames, corr = circular_cross_correlation_lag(ref, flows[i])
        lags_s[i] = lag_frames * tr
        corrs[i] = abs(corr)
    fit = weighted_linear_fit(dist, lags_s, weights)
    slope = fit["slope"]
    pwv = 1.0 / slope if slope and _np.isfinite(slope) and abs(slope) > 1e-12 else float("nan")
    return {"pwv_m_s": float(pwv), "r": float(_np.mean(corrs)), "n": int(flows.shape[0])}


def pwv_bjornfoot_optimize(
    distances_m,
    flow_matrix,
    temporal_resolution_s: float,
    *,
    weights=None,
    pwv_bounds: tuple[float, float] = (0.5, 30.0),
) -> dict[str, float]:
    """Bjornfoot waveform-shift PWV estimate (J Cereb Blood Flow Metab 2021).

    Each waveform is normalized; for a candidate PWV the per-station delay
    ``dt_i = d_i / PWV`` aligns the waveforms to a shared template, and the total
    weighted residual is minimized over PWV. Returns ``pwv_m_s`` and the residual cost.
    """
    import numpy as _np
    from scipy.optimize import minimize_scalar

    dist = to_numpy(as_backend_array(distances_m)).astype("float64").reshape(-1)
    flows = to_numpy(as_backend_array(normalize_waveform(flow_matrix)))
    if flows.ndim != 2 or flows.shape[0] < 2:
        return {"pwv_m_s": float("nan"), "cost": float("nan"), "n": int(flows.shape[0] if flows.ndim else 0)}
    tr = float(temporal_resolution_s)
    n_stations = flows.shape[0]
    if weights is None:
        wv = _np.ones(n_stations, dtype="float64")
    else:
        wv = to_numpy(as_backend_array(weights)).astype("float64").reshape(-1)[:n_stations]
    wsum = float(wv.sum()) or 1.0

    def cost(pwv: float) -> float:
        if pwv <= 0:
            return 1e12
        shift_frames = (dist / pwv) / tr
        aligned = _np.vstack(
            [_circular_fractional_shift(flows[i], shift_frames[i]) for i in range(n_stations)]
        )
        template = (wv[:, None] * aligned).sum(axis=0) / wsum
        total = 0.0
        for i in range(n_stations):
            model = _circular_fractional_shift(template, -shift_frames[i])
            total += wv[i] * float(_np.sum((model - flows[i]) ** 2))
        return total

    res = minimize_scalar(cost, bounds=tuple(pwv_bounds), method="bounded")
    return {"pwv_m_s": float(res.x), "cost": float(res.fun), "n": int(n_stations)}


def accept_pwv(pwv_m_s: float) -> bool:
    """QVTplus acceptance gate: ``0 < PWV < 30`` m/s."""
    try:
        v = float(pwv_m_s)
    except (TypeError, ValueError):
        return False
    return PWV_MIN_M_S < v < PWV_MAX_M_S


__all__ = [
    "PWV_MAX_M_S",
    "PWV_MIN_M_S",
    "QUALITY_SCALE_MAX",
    "QUALITY_THRESH_DEFAULT",
    "accept_pwv",
    "circular_cross_correlation_lag",
    "damping_index",
    "mean_flow_ml_s",
    "mean_velocity_mm_s",
    "normalize_waveform",
    "branch_window_slices",
    "flow_per_heart_cycle_ml_s",
    "flow_pulsatile_ml_s",
    "pitc_fit",
    "pulsatility_index",
    "pulsatility_index_qvt",
    "pwv_bjornfoot_optimize",
    "pwv_fielding_xcor",
    "quality_weights",
    "resistivity_index",
    "through_plane_velocity_series",
    "velocity_mm_s_from_phases",
    "station_quality_scores",
    "stdv_from_mean_branch",
    "stdv_from_mean_station",
    "waveform_quality_score",
    "weighted_linear_fit",
]
