"""Paper-style stage-6 measurement figures and PITC branch masks.

Renders the three figures used to report vessel-level 4D-flow hemodynamics:

- **PITC** — per-root pulsatility quality ``Q`` and pulsatility index ``p_pi`` vs
  distance-from-root with the transmission-coefficient fit line.
- **PWV** — per-root cross-correlation ``XCor`` time and time-to-upstroke vs
  distance with the quality- and correlation-weighted fits, plus the weights.
- **Flow waveforms** — per-vessel mean +/- std flow over the cardiac cycle.

Also writes, per root region, a NIfTI mask of the vessels (root + downstream
branches) contributing to that region's PITC fit.

All figures are opt-in via the stage-6 ``--save-plots`` flag; nothing here runs
by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.logger import Logger
from nvitk.io.imageio import imsave
from nvitk.measure.hemodynamics import weighted_linear_fit
from nvitk.pipes.qvtpy.labels import (
    QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID,
    qvtpy_vessel_name,
)
from nvitk.pipes.qvtpy.util.vessel_hemodynamics import ROOT_GROUPS

log = Logger()

# Region column order / display titles (paper layout: Left ICA, Right ICA, Basilar).
_REGION_ORDER: tuple[str, ...] = ("L_ICA", "R_ICA", "Basilar")
_REGION_TITLES: dict[str, str] = {
    "L_ICA": "Left ICA",
    "R_ICA": "Right ICA",
    "Basilar": "Basilar",
}

# Legacy 9-vessel paper grid (fallback when no all-label waveforms are supplied).
_WAVEFORM_LAYOUT: tuple[tuple[int, str], ...] = (
    (1, "LICA"),
    (2, "RICA"),
    (3, "BASILAR"),
    (7, "LMCA"),
    (4, "LACA"),
    (9, "LPCA"),
    (8, "RMCA"),
    (5, "RACA"),
    (10, "RPCA"),
)


def _ordered_waveform_labels(waveforms: dict[int, dict[str, Any]]) -> list[tuple[int, str]]:
    """Stable plot order: canonical qvtpy label order, then any extras."""
    present = {int(k) for k in waveforms.keys()}
    ordered: list[tuple[int, str]] = []
    for lid, name in sorted(QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID.items()):
        if int(lid) in present:
            ordered.append((int(lid), str(name)))
    for lid in sorted(present):
        if lid not in {x[0] for x in ordered}:
            ordered.append((lid, qvtpy_vessel_name(lid)))
    return ordered


def _fit_pwv_line(distance_mm: np.ndarray, time_s: np.ndarray, weights: np.ndarray):
    """Weighted fit of delay (s) vs distance (mm); return (slope_s_per_mm, intercept, pwv_m_s)."""
    finite = np.isfinite(distance_mm) & np.isfinite(time_s)
    if int(finite.sum()) < 2:
        return None
    fit = weighted_linear_fit(distance_mm[finite], time_s[finite], weights[finite])
    slope = fit["slope"]  # s per mm
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return fit["slope"], fit["intercept"], float("nan")
    pwv = (1.0 / slope) / 1000.0  # (mm/s) -> m/s
    return slope, fit["intercept"], float(pwv)


# ---------------------------------------------------------------------------
# Figure 1: PITC (Q + p_pi vs distance)
# ---------------------------------------------------------------------------


def plot_pitc_figure(region_plot_data: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    """Two-row PITC figure: quality Q and pulsatility index vs distance per root."""
    regions = [r for r in _REGION_ORDER if r in region_plot_data]
    if not regions:
        return None
    fig, axes = plt.subplots(2, len(regions), figsize=(4.2 * len(regions), 6.4), squeeze=False)
    for col, region in enumerate(regions):
        d = region_plot_data[region]
        dist = np.asarray(d["distance_mm"], dtype="float64")
        pi = np.asarray(d["pi"], dtype="float64")
        quality = np.asarray(d["quality"], dtype="float64")
        thresh = float(d.get("quality_thresh", 2.5))

        # Row 1: quality vs distance, split about the inclusion threshold.
        ax_q = axes[0][col]
        low = quality < thresh
        ax_q.scatter(dist[low], quality[low], s=14, c="0.6", label=f"Q<{thresh:g}")
        ax_q.scatter(dist[~low], quality[~low], s=14, c="royalblue", label=f"Q>{thresh:g}")
        ax_q.set_title(_REGION_TITLES.get(region, region), fontsize=13, fontweight="bold")
        ax_q.set_ylabel("Q (max 4)")
        ax_q.set_xlabel("d (mm)")
        ax_q.legend(loc="lower left", fontsize=8, framealpha=0.9)

        # Row 2: p_pi vs distance with the PITC fit, mean line, and x_p / x_d markers.
        ax_p = axes[1][col]
        ax_p.scatter(dist, pi, s=12, c="royalblue", alpha=0.7, label="Data")
        slope = d.get("pitc_slope")
        intercept = d.get("pitc_intercept")
        if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept) and dist.size:
            xline = np.array([float(np.min(dist)), float(np.max(dist))], dtype="float64")
            yline = slope * xline + intercept
            ax_p.plot(xline, yline, "k-", lw=1.6, label=r"$p_{tf}(d)=p_{tc}d+\beta$")
            ax_p.scatter(xline, yline, s=60, c="limegreen", zorder=5, label=r"$x_p$ and $x_d$")
        gpi = d.get("global_pi")
        if gpi is not None and np.isfinite(gpi):
            ax_p.axhline(float(gpi), color="firebrick", lw=1.2, label=r"$\mu(p_{pi})$")
        ax_p.set_ylabel(r"$p_{pi}$")
        ax_p.set_xlabel("d (mm)")
        if col == 0:
            ax_p.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: PWV (XCor time, time-to-upstroke, weights vs distance)
# ---------------------------------------------------------------------------


def plot_pwv_figure(region_plot_data: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    """Three-row PWV figure: XCor time, time-to-upstroke, and per-station weights."""
    regions = [r for r in _REGION_ORDER if r in region_plot_data]
    if not regions:
        return None
    fig, axes = plt.subplots(3, len(regions), figsize=(4.2 * len(regions), 9.0), squeeze=False)
    for col, region in enumerate(regions):
        d = region_plot_data[region]
        dist = np.asarray(d.get("pwv_distance_mm", []), dtype="float64")
        xcor = np.asarray(d.get("pwv_xcor_time_s", []), dtype="float64")
        upstroke = np.asarray(d.get("pwv_time_to_upstroke_s", []), dtype="float64")
        w1 = np.asarray(d.get("pwv_weight_quality", []), dtype="float64")
        w2 = np.asarray(d.get("pwv_weight_correlation", []), dtype="float64")

        ax_x = axes[0][col]
        ax_u = axes[1][col]
        ax_w = axes[2][col]
        ax_x.set_title(_REGION_TITLES.get(region, region), fontsize=13, fontweight="bold")

        if dist.size:
            ax_x.scatter(dist, xcor, s=16, c="orchid", alpha=0.7, label="raw data")
            _draw_pwv_fits(ax_x, dist, xcor, w1, w2)
            ax_u.scatter(dist, upstroke, s=16, c="teal", alpha=0.7, label="raw data")
            _draw_pwv_fits(ax_u, dist, upstroke, w1, w2)
            ax_w.scatter(dist, w1, s=16, c="limegreen", alpha=0.7, label=r"$W_1$")
            ax_w.scatter(dist, w2, s=16, c="royalblue", alpha=0.7, label=r"$W_2$")

        ax_x.set_ylabel("maximised XCor time (s)")
        ax_u.set_ylabel("time-to-upstroke (s)")
        ax_w.set_ylabel("Weight")
        ax_w.set_xlabel("d (mm)")
        ax_x.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax_w.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _draw_pwv_fits(ax, dist: np.ndarray, time_s: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> None:
    """Overlay the quality- and correlation-weighted delay-vs-distance fits."""
    xline = np.array([float(np.min(dist)), float(np.max(dist))], dtype="float64")
    for weights, style, tag in ((w1, "-.", "W_1"), (w2, "--", "W_2")):
        if weights.size != dist.size or not np.any(weights > 0):
            continue
        res = _fit_pwv_line(dist, time_s, weights)
        if res is None:
            continue
        slope, intercept, pwv = res
        yline = slope * xline + intercept
        label = rf"${tag}$: {pwv:.1f} m/s" if np.isfinite(pwv) else rf"${tag}$"
        ax.plot(xline, yline, style, color="black", lw=1.4, label=label)


# ---------------------------------------------------------------------------
# Figure 3: per-vessel flow waveforms
# ---------------------------------------------------------------------------


def plot_flow_waveforms(
    region_plot_data: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    all_label_waveforms: dict[int, dict[str, Any]] | None = None,
) -> Path | None:
    """Grid of per-vessel mean +/- std flow waveforms over the cardiac cycle."""
    waveforms: dict[int, dict[str, Any]] = {}
    tr: float | None = None
    if all_label_waveforms:
        waveforms = {int(k): v for k, v in all_label_waveforms.items()}
        for region in region_plot_data.values():
            if tr is None and region.get("temporal_resolution_s"):
                tr = float(region["temporal_resolution_s"])
    else:
        for region in region_plot_data.values():
            if tr is None and region.get("temporal_resolution_s"):
                tr = float(region["temporal_resolution_s"])
            for label, wf in region.get("vessel_waveforms", {}).items():
                prev = waveforms.get(int(label))
                if prev is None or int(wf["n_stations"]) > int(prev["n_stations"]):
                    waveforms[int(label)] = wf
    if not waveforms:
        return None

    layout = _ordered_waveform_labels(waveforms)
    n_panels = len(layout)
    ncols = min(4, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for idx, (label, title) in enumerate(layout):
        ax = axes[idx // ncols][idx % ncols]
        wf = waveforms.get(int(label))
        display = wf.get("vessel_name", title) if wf else title
        ax.set_title(str(display), fontsize=11, fontweight="bold")
        ax.set_ylabel("Flow (mL/s)")
        ax.set_xlabel("time (s)")
        if wf is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.6")
            continue
        mean = np.asarray(wf["mean"], dtype="float64")
        std = np.asarray(wf.get("std", np.zeros_like(mean)), dtype="float64")
        nt = mean.size
        t = np.arange(nt, dtype="float64") * (tr if tr else 1.0)
        if std.size == nt and np.any(std > 0):
            ax.fill_between(t, mean - std, mean + std, color="0.7", alpha=0.6)
        ax.plot(t, mean, color="black", lw=1.5)
        ax.set_ylim(bottom=0)
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PITC branch masks (one NIfTI per root region)
# ---------------------------------------------------------------------------


def save_pitc_region_masks(
    volume_seg: np.ndarray,
    out_dir: Path,
    *,
    metadata: dict | None = None,
) -> list[str]:
    """Write one multilabel NIfTI per root region with the vessels used for PITC.

    Each mask keeps the ``seg_4dflow`` labels of the region's root and downstream
    branches, so a reviewer can see exactly which branches fed each PITC fit.
    """
    seg = to_numpy(volume_seg).astype(np.int32, copy=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for group in ROOT_GROUPS:
        labels = [int(group.root_label), *[int(b) for b in group.branch_labels]]
        mask = np.zeros(seg.shape, dtype=np.int32)
        any_present = False
        for lid in labels:
            sel = to_numpy(seg == lid).astype(bool)
            if np.any(sel):
                mask[sel] = lid
                any_present = True
        if not any_present:
            continue
        path = out_dir / f"pitc_mask_{group.region_id}.nii.gz"
        imsave(path, mask, metadata=dict(metadata or {}))
        written.append(path.name)
    return written


__all__ = [
    "plot_flow_waveforms",
    "plot_pitc_figure",
    "plot_pwv_figure",
    "save_pitc_region_masks",
]
