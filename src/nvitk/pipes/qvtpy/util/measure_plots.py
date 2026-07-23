r"""Paper-style stage-6 measurement figures and PITC branch masks.

Renders the four figures used to report vessel-level 4D-flow hemodynamics:

- **PITC** — per-root pulsatility quality ``Q`` and pulsatility index ``p_pi`` vs
  distance-from-root with the transmission-coefficient fit line.
- **PWV** — per-root cross-correlation ``XCor`` time and time-to-upstroke vs
  distance with Bjornfoot area-weighted (\(W_1\)) and Dempsey quality-weighted
  (\(W_2\)) fits (QVTplus ``enc_PWV_XCor`` tag 0/1), plus the weights.
- **Bjornfoot QC** — per-station weighted template residual, XCor-minus-model
  delay residual, and observed-versus-fitted waveform correlation.
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


def _figure_axes(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float],
    fig=None,
):
    """Create a new figure or clear *fig* and build an ``(nrows, ncols)`` axes grid."""
    if fig is None:
        return plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.clear()
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _safe_tight_layout(fig) -> None:
    try:
        fig.tight_layout(pad=0.35)
    except Exception:
        pass

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


def make_pitc_figure(
    region_plot_data: dict[str, dict[str, Any]],
    *,
    show_legend: bool = True,
    fig=None,
):
    """Build a live Matplotlib PITC diagnostics figure.

    Pass an existing *fig* (e.g. a QtAgg canvas figure) to redraw in place
    without replacing the canvas figure / breaking zoom-pan.
    """
    regions = [r for r in _REGION_ORDER if r in region_plot_data]
    if not regions:
        return None
    fig, axes = _figure_axes(
        2,
        len(regions),
        figsize=(4.2 * len(regions), 6.4),
        fig=fig,
    )
    for col, region in enumerate(regions):
        d = region_plot_data[region]
        dist = to_numpy(d["distance_mm"]).astype("float64")
        pi = to_numpy(d["pi"]).astype("float64")
        quality = to_numpy(d["quality"]).astype("float64")
        thresh = float(d.get("quality_thresh", 2.5))

        ax_q = axes[0][col]
        low = quality < thresh
        ax_q.scatter(dist[low], quality[low], s=14, c="0.6", label=f"Q<{thresh:g}")
        ax_q.scatter(dist[~low], quality[~low], s=14, c="royalblue", label=f"Q>{thresh:g}")
        ax_q.set_title(_REGION_TITLES.get(region, region), fontsize=13, fontweight="bold")
        ax_q.set_ylabel("Q (StdvFromMean)")
        ax_q.set_xlabel("d (mm)")
        if show_legend:
            ax_q.legend(loc="best", fontsize=7, framealpha=0.9)

        ax_p = axes[1][col]
        used = quality > thresh
        ax_p.scatter(
            dist[~used], pi[~used], s=12, c="0.6", alpha=0.7, label=f"Q<{thresh:g}"
        )
        ax_p.scatter(
            dist[used], pi[used], s=12, c="royalblue", alpha=0.7, label=f"Q>{thresh:g}"
        )
        slope = d.get("pitc_slope")
        intercept = d.get("pitc_intercept")
        if (
            slope is not None
            and intercept is not None
            and np.isfinite(slope)
            and np.isfinite(intercept)
            and dist.size
        ):
            xline = np.array(
                [float(np.min(dist)), float(np.max(dist))], dtype="float64"
            )
            yline = slope * xline + intercept
            ax_p.plot(xline, yline, "k-", lw=1.6, label=r"$p_{tf}(d)=p_{tc}d+\beta$")
        if int(used.sum()) >= 2:
            iu = np.where(used)[0]
            order = np.argsort(dist[iu])
            i0, i1 = int(iu[order[0]]), int(iu[order[-1]])
            ax_p.scatter(
                [dist[i0], dist[i1]],
                [pi[i0], pi[i1]],
                s=60,
                c="limegreen",
                zorder=5,
                label=r"$x_p$ and $x_d$",
            )
        gpi = d.get("global_pi")
        if gpi is not None and np.isfinite(gpi):
            ax_p.axhline(float(gpi), color="firebrick", lw=1.2, label=r"$\mu(p_{pi})$")
        ax_p.set_ylabel(r"$p_{pi}$")
        ax_p.set_xlabel("d (mm)")
        if show_legend and col == 0:
            ax_p.legend(loc="best", fontsize=7, framealpha=0.9)
    _safe_tight_layout(fig)
    return fig


def plot_pitc_figure(region_plot_data: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    """Two-row PITC figure: quality Q and pulsatility index vs distance per root."""
    fig = make_pitc_figure(region_plot_data, show_legend=True)
    if fig is None:
        return None
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: PWV (XCor time, time-to-upstroke, weights vs distance)
# ---------------------------------------------------------------------------


def make_pwv_figure(
    region_plot_data: dict[str, dict[str, Any]],
    *,
    show_legend: bool = True,
    fig=None,
):
    """Build a live Matplotlib PWV diagnostics figure.

    Quality-excluded stations are shown in grey (same style as PITC); fits and
    colored markers use only stations with ``Q > quality_thresh``.

    Pass an existing *fig* to redraw in place (GUI dock) without replacing the
    canvas figure.
    """
    regions = [r for r in _REGION_ORDER if r in region_plot_data]
    if not regions:
        return None
    fig, axes = _figure_axes(
        3,
        len(regions),
        figsize=(4.2 * len(regions), 9.0),
        fig=fig,
    )
    for col, region in enumerate(regions):
        d = region_plot_data[region]
        dist = to_numpy(d.get("pwv_distance_mm", [])).astype("float64")
        xcor = to_numpy(d.get("pwv_xcor_time_s", [])).astype("float64")
        upstroke = to_numpy(d.get("pwv_time_to_upstroke_s", [])).astype("float64")
        excl_dist = to_numpy(d.get("pwv_excluded_distance_mm", [])).astype("float64")
        excl_xcor = to_numpy(d.get("pwv_excluded_xcor_time_s", [])).astype("float64")
        excl_up = to_numpy(d.get("pwv_excluded_time_to_upstroke_s", [])).astype("float64")
        thresh = float(d.get("quality_thresh", 2.5))
        if "pwv_weight_area" in d:
            w1 = to_numpy(d["pwv_weight_area"]).astype("float64")
            w2 = to_numpy(d.get("pwv_weight_quality", [])).astype("float64")
        else:
            w1 = to_numpy(d.get("pwv_weight_quality", [])).astype("float64")
            w2 = to_numpy(d.get("pwv_weight_correlation", [])).astype("float64")

        ax_x = axes[0][col]
        ax_u = axes[1][col]
        ax_w = axes[2][col]
        ax_x.set_title(_REGION_TITLES.get(region, region), fontsize=13, fontweight="bold")

        excl_label = f"Q≤{thresh:g}"
        if excl_dist.size:
            n_x = min(excl_dist.size, excl_xcor.size)
            if n_x:
                finite = np.isfinite(excl_dist[:n_x]) & np.isfinite(excl_xcor[:n_x])
                if np.any(finite):
                    ax_x.scatter(
                        excl_dist[:n_x][finite],
                        excl_xcor[:n_x][finite],
                        s=14,
                        c="0.6",
                        alpha=0.7,
                        zorder=1,
                        label=excl_label,
                    )
            n_u = min(excl_dist.size, excl_up.size)
            if n_u:
                finite = np.isfinite(excl_dist[:n_u]) & np.isfinite(excl_up[:n_u])
                if np.any(finite):
                    ax_u.scatter(
                        excl_dist[:n_u][finite],
                        excl_up[:n_u][finite],
                        s=14,
                        c="0.6",
                        alpha=0.7,
                        zorder=1,
                        label=excl_label,
                    )
            # Weights panel: mark excluded stations on the x-axis at y=0.
            ax_w.scatter(
                excl_dist[np.isfinite(excl_dist)],
                np.zeros(int(np.count_nonzero(np.isfinite(excl_dist)))),
                s=14,
                c="0.6",
                alpha=0.7,
                zorder=1,
                label=excl_label,
            )

        if dist.size:
            ax_x.scatter(
                dist, xcor, s=16, c="orchid", alpha=0.7, zorder=2, label="raw data"
            )
            _draw_pwv_fits(ax_x, dist, xcor, w1, w2)
            ax_u.scatter(
                dist, upstroke, s=16, c="teal", alpha=0.7, zorder=2, label="raw data"
            )
            _draw_pwv_fits(ax_u, dist, upstroke, w1, w2)
            ax_w.scatter(
                dist, w1, s=16, c="limegreen", alpha=0.7, zorder=2, label=r"$W_1$ (area)"
            )
            ax_w.scatter(
                dist,
                w2,
                s=16,
                c="royalblue",
                alpha=0.7,
                zorder=2,
                label=r"$W_2$ (quality)",
            )

        ax_x.set_ylabel("maximised XCor time (s)")
        ax_u.set_ylabel("time-to-upstroke (s)")
        ax_w.set_ylabel("Weight")
        ax_w.set_xlabel("d (mm)")
        if show_legend:
            ax_x.legend(loc="best", fontsize=7, framealpha=0.9)
            ax_w.legend(loc="best", fontsize=7, framealpha=0.9)
    _safe_tight_layout(fig)
    return fig


def plot_pwv_figure(region_plot_data: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    """Three-row PWV figure: XCor time, time-to-upstroke, and per-station weights."""
    fig = make_pwv_figure(region_plot_data, show_legend=True)
    if fig is None:
        return None
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def make_bjornfoot_qc_figure(
    region_plot_data: dict[str, dict[str, Any]],
    *,
    show_legend: bool = True,
    fig=None,
):
    """Build a three-row Bjornfoot shared-template fit QC figure.

    Pass an existing *fig* to redraw in place (GUI dock) without replacing the
    canvas figure.
    """
    regions = [r for r in _REGION_ORDER if r in region_plot_data]
    if not regions:
        return None
    fig, axes = _figure_axes(
        3,
        len(regions),
        figsize=(4.2 * len(regions), 9.0),
        fig=fig,
    )
    for col, region in enumerate(regions):
        data = region_plot_data[region]
        dist = to_numpy(data.get("pwv_distance_mm", [])).astype("float64").reshape(-1)
        weighted_rms = to_numpy(
            data.get("pwv_bjornfoot_weighted_rms", [])
        ).astype("float64").reshape(-1)
        delay_residual_ms = (
            to_numpy(data.get("pwv_bjornfoot_delay_residual_s", []))
            .astype("float64")
            .reshape(-1)
            * 1000.0
        )
        waveform_corr = to_numpy(
            data.get("pwv_bjornfoot_waveform_corr", [])
        ).astype("float64").reshape(-1)

        ax_rms = axes[0][col]
        ax_delay = axes[1][col]
        ax_corr = axes[2][col]
        raw_pwv = data.get("pwv_bjornfoot_raw_m_s")
        cost = data.get("pwv_bjornfoot_cost")
        pwv_text = (
            f"{float(raw_pwv):.2f} m/s"
            if raw_pwv is not None and np.isfinite(raw_pwv)
            else "n/a"
        )
        cost_text = (
            f"{float(cost):.3g}"
            if cost is not None and np.isfinite(cost)
            else "n/a"
        )
        ax_rms.set_title(
            f"{_REGION_TITLES.get(region, region)}\n"
            f"Bjornfoot={pwv_text}, cost={cost_text}",
            fontsize=12,
            fontweight="bold",
        )

        def _scatter_aligned(ax, values: np.ndarray, *, color: str, label: str) -> None:
            n = min(dist.size, values.size)
            if n < 1:
                ax.text(
                    0.5,
                    0.5,
                    "No Bjornfoot fit diagnostics",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.5",
                )
                return
            finite = np.isfinite(dist[:n]) & np.isfinite(values[:n])
            if np.any(finite):
                ax.scatter(
                    dist[:n][finite],
                    values[:n][finite],
                    s=18,
                    c=color,
                    alpha=0.75,
                    label=label,
                )

        _scatter_aligned(
            ax_rms,
            weighted_rms,
            color="coral",
            label="weighted template residual",
        )
        _scatter_aligned(
            ax_delay,
            delay_residual_ms,
            color="darkviolet",
            label="XCor − Bjornfoot delay",
        )
        _scatter_aligned(
            ax_corr,
            waveform_corr,
            color="seagreen",
            label="observed vs fitted",
        )
        ax_delay.axhline(0.0, color="0.35", lw=1.0, ls=":")
        ax_corr.axhline(0.0, color="0.65", lw=0.8, ls=":")
        ax_corr.set_ylim(-1.05, 1.05)
        ax_rms.set_ylabel("Weighted residual RMS\n(normalized velocity)")
        ax_delay.set_ylabel("XCor − model delay (ms)")
        ax_corr.set_ylabel("Waveform correlation")
        ax_corr.set_xlabel("d (mm)")
        if show_legend and col == 0:
            ax_rms.legend(loc="best", fontsize=7, framealpha=0.9)
            ax_delay.legend(loc="best", fontsize=7, framealpha=0.9)
            ax_corr.legend(loc="best", fontsize=7, framealpha=0.9)
    _safe_tight_layout(fig)
    return fig


def plot_bjornfoot_qc_figure(
    region_plot_data: dict[str, dict[str, Any]], out_path: Path
) -> Path | None:
    """Save standalone Bjornfoot options A/B/C QC as one PNG."""
    fig = make_bjornfoot_qc_figure(region_plot_data, show_legend=True)
    if fig is None:
        return None
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _draw_pwv_fits(ax, dist: np.ndarray, time_s: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> None:
    r"""Overlay area-weighted (\(W_1\)) and Dempsey quality-weighted (\(W_2\)) fits."""
    from nvitk.measure.hemodynamics import accept_pwv

    xline = np.array([float(np.min(dist)), float(np.max(dist))], dtype="float64")
    for weights, style, tag in ((w1, "-.", "W_1"), (w2, "--", "W_2")):
        if weights.size != dist.size or not np.any(weights > 0):
            continue
        res = _fit_pwv_line(dist, time_s, weights)
        if res is None:
            continue
        slope, intercept, pwv = res
        yline = slope * xline + intercept
        if accept_pwv(pwv):
            label = rf"${tag}$: {pwv:.1f} m/s"
        elif np.isfinite(pwv):
            label = rf"${tag}$: n/a ({pwv:.1f})"
        else:
            label = rf"${tag}$: n/a"
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
        mean = to_numpy(wf["mean"]).astype("float64")
        std = to_numpy(wf.get("std", np.zeros_like(mean))).astype("float64")
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
    Isolated islands should already be removed by
    :func:`~nvitk.pipes.qvtpy.util.mask_cleaning.clean_volume_seg_for_pitc`; this
    export also keeps the largest component per label as a safety net.
    """
    from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_per_label

    seg = keep_largest_component_per_label(to_numpy(volume_seg).astype(np.int32, copy=False))
    seg = to_numpy(seg).astype(np.int32, copy=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for group in ROOT_GROUPS:
        labels = [int(group.root_label), *[int(b) for b in group.branch_labels]]
        mask = np.zeros(seg.shape, dtype=np.int32)
        any_present = False
        for lid in labels:
            sel = seg == lid
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
    "make_bjornfoot_qc_figure",
    "make_pitc_figure",
    "make_pwv_figure",
    "plot_bjornfoot_qc_figure",
    "plot_flow_waveforms",
    "plot_pitc_figure",
    "plot_pwv_figure",
    "save_pitc_region_masks",
]
