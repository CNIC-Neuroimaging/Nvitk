"""Cohort violin + strip hemodynamics plots from qvtpy ``image_measurements``.

Shared by the CLI script ``scripts/pesa_brain/plotter/violin_hemodynamics.py``
and the QC GUI cohort-violin dock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from nvitk.core.logger import Logger
from nvitk.db.xnat import parse_subject_tokens
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token
from nvitk.pipes.qvtpy.common.db_publish import (
    QVTPY_PIPELINE_ALIASES,
    QVTPY_PIPELINE_ID,
    resolve_repo,
)
from nvitk.pipes.qvtpy.stage0_download import load_subjects

log = Logger()

# Display order for LOC-level metrics (existing qvtpy vessels only; no C1/C3 split).
VESSEL_SPECS: list[tuple[str, str, str]] = [
    # (display_label, db_region_id, territory_group)
    ("TCBF", "TCBF", "TCBF & ICAs"),
    ("LICA", "LICA", "TCBF & ICAs"),
    ("RICA", "RICA", "TCBF & ICAs"),
    ("LMCA", "LMCA", "Anterior Circ."),
    ("RMCA", "RMCA", "Anterior Circ."),
    ("LACA", "LACA", "Anterior Circ."),
    ("RACA", "RACA", "Anterior Circ."),
    ("LPCOMM", "LPCOMM", "Anterior Circ."),
    ("RPCOMM", "RPCOMM", "Anterior Circ."),
    ("BA", "BASILAR", "Posterior Circ."),
    ("LVA", "LVA", "Posterior Circ."),
    ("RVA", "RVA", "Posterior Circ."),
    ("LPCA", "LPCA", "Posterior Circ."),
    ("RPCA", "RPCA", "Posterior Circ."),
    ("SSS", "SSSV", "Venous Drainage"),
    ("STR", "STRV", "Venous Drainage"),
    ("LTS", "LTSV", "Venous Drainage"),
    ("RTS", "RTSV", "Venous Drainage"),
]

PITC_PWV_SPECS: list[tuple[str, str, str]] = [
    ("L_ICA", "L_ICA", "TCBF & ICAs"),
    ("R_ICA", "R_ICA", "TCBF & ICAs"),
    ("Basilar", "Basilar", "Posterior Circ."),
]

TERRITORY_COLORS: dict[str, str] = {
    "TCBF & ICAs": "#c9a0c9",
    "Anterior Circ.": "#8fbc8f",
    "Posterior Circ.": "#f0e68c",
    "Venous Drainage": "#b8b4d8",
}

METRICS: list[dict[str, Any]] = [
    {
        "key": "flow",
        "variable_id": "flow_mean",
        "title": "Blood Flow Rates in All Vessel Segments",
        "ylabel": "Volumetric Flow Rate (mL/min)",
        "filename": "flow_mean_violin.png",
        "panel": "A",
        "kind": "loc",
        "derive_tcbf": True,
    },
    {
        "key": "pi",
        "variable_id": "pi",
        "title": "Pulsatility Index in All Vessel Segments",
        "ylabel": "Pulsatility Index",
        "filename": "pi_violin.png",
        "panel": "B",
        "kind": "loc",
        "derive_tcbf": False,
    },
    {
        "key": "ri",
        "variable_id": "ri",
        "title": "Resistivity Index in All Vessel Segments",
        "ylabel": "Resistivity Index",
        "filename": "ri_violin.png",
        "panel": "C",
        "kind": "loc",
        "derive_tcbf": False,
    },
    {
        "key": "pitc_slope",
        "variable_id": "pitc_slope",
        "title": "PITC Slope by Root Territory",
        "ylabel": "PITC Slope (1/mm)",
        "filename": "pitc_slope_violin.png",
        "panel": "D",
        "kind": "root",
        "derive_tcbf": False,
    },
    {
        "key": "pwv_f",
        "variable_id": "pwv_fielding_xcor",
        "title": "PWV Fielding (XCor) by Root Territory",
        "ylabel": "PWV Fielding (m/s)",
        "filename": "pwv_fielding_violin.png",
        "panel": "E",
        "kind": "root",
        "derive_tcbf": False,
    },
    {
        "key": "pwv_b",
        "variable_id": "pwv",
        "title": "PWV Bjornfoot by Root Territory",
        "ylabel": "PWV Bjornfoot (m/s)",
        "filename": "pwv_bjornfoot_violin.png",
        "panel": "F",
        "kind": "root",
        "derive_tcbf": False,
    },
]

# Back-compat aliases used by the CLI script.
_VESSEL_SPECS = VESSEL_SPECS
_PITC_PWV_SPECS = PITC_PWV_SPECS
_TERRITORY_COLORS = TERRITORY_COLORS
_METRICS = METRICS


def resolve_pipeline_id(pipeline_version: str) -> str:
    token = str(pipeline_version or "").strip()
    if not token or token.lower() in {a.lower() for a in QVTPY_PIPELINE_ALIASES}:
        return QVTPY_PIPELINE_ID
    return token


def _resolve_subjects_filter(subjects: str | None) -> list[str] | None:
    if subjects is None or not str(subjects).strip():
        return None
    tokens = parse_subject_tokens(subjects)
    if len(tokens) == 1 and resolve_xnat_project_cohort_token(tokens[0]) is not None:
        return None
    return load_subjects(subjects=subjects, subjects_file=None)


def _cohort_id_from_subjects(subjects: str | None) -> str | bool:
    if subjects is None or not str(subjects).strip():
        return False
    tokens = parse_subject_tokens(subjects)
    if len(tokens) == 1 and resolve_xnat_project_cohort_token(tokens[0]) is not None:
        return tokens[0].strip()
    return False


def load_long_measurements(
    *,
    pipeline_id: str,
    variable_ids: list[str],
    subjects: str | None = None,
) -> pd.DataFrame:
    """Load long-format image_measurements for the given variables."""
    repo = resolve_repo(prefer_sge=False)
    cohort = _cohort_id_from_subjects(subjects)
    subject_filter = _resolve_subjects_filter(subjects)
    df = repo.image(
        modality="4dflow",
        variables=variable_ids,
        pipeline=pipeline_id,
        wide=False,
        cohort_id=cohort,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if subject_filter is not None:
        df = df[df["subject_uid"].astype(str).isin(subject_filter)].copy()
    if "pipeline_id" in df.columns:
        df = df[df["pipeline_id"].astype(str) == str(pipeline_id)].copy()
    return df


def derive_tcbf_rows(flow_df: pd.DataFrame) -> pd.DataFrame:
    """TCBF = LICA + RICA + BASILAR flow_mean per subject."""
    need = {"LICA", "RICA", "BASILAR"}
    rows: list[dict[str, Any]] = []
    for sid, g in flow_df.groupby("subject_uid", dropna=False):
        by_reg = {
            str(r): float(v)
            for r, v in zip(g["region_id"], g["value_num"], strict=False)
            if pd.notna(v)
        }
        if not need.issubset(by_reg):
            continue
        rows.append(
            {
                "subject_uid": sid,
                "region_id": "TCBF",
                "variable_id": "flow_mean",
                "value_num": by_reg["LICA"] + by_reg["RICA"] + by_reg["BASILAR"],
            }
        )
    return pd.DataFrame(rows)


def prepare_plot_frame(
    long_df: pd.DataFrame,
    *,
    variable_id: str,
    specs: list[tuple[str, str, str]],
    derive_tcbf: bool,
) -> pd.DataFrame:
    sub = long_df[long_df["variable_id"].astype(str) == variable_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=["subject_uid", "vessel", "group", "value"])

    if derive_tcbf and variable_id == "flow_mean":
        tcbf = derive_tcbf_rows(sub)
        if not tcbf.empty:
            sub = pd.concat([sub, tcbf], ignore_index=True)

    region_to_display = {db: disp for disp, db, _g in specs}
    region_to_group = {db: grp for _d, db, grp in specs}
    order = [disp for disp, _db, _g in specs]

    sub["region_id"] = sub["region_id"].astype(str)
    sub = sub[sub["region_id"].isin(region_to_display)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["subject_uid", "vessel", "group", "value"])

    out = pd.DataFrame(
        {
            "subject_uid": sub["subject_uid"].astype(str),
            "vessel": sub["region_id"].map(region_to_display),
            "group": sub["region_id"].map(region_to_group),
            "value": pd.to_numeric(sub["value_num"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["value", "vessel"])
    n_before = len(out)
    out = out[out["value"] != 0].copy()
    n_zeros = n_before - len(out)
    if n_zeros:
        log.info(
            "Dropped %d zero placeholder value(s) for variable=%s",
            n_zeros,
            variable_id,
        )
    out["vessel"] = pd.Categorical(out["vessel"], categories=order, ordered=True)
    out = out.sort_values(["vessel", "subject_uid"])
    return out


def flag_outliers_iqr(df: pd.DataFrame, *, k: float = 1.5) -> pd.Series:
    """Return boolean mask of IQR outliers, computed per vessel."""
    is_outlier = pd.Series(False, index=df.index)
    for _vessel, grp in df.groupby("vessel", observed=True):
        q1 = grp["value"].quantile(0.25)
        q3 = grp["value"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (grp["value"] < lo) | (grp["value"] > hi)
        is_outlier.loc[mask[mask].index] = True
    return is_outlier


def flag_low_values(df: pd.DataFrame, *, thresh: float) -> pd.Series:
    """Return boolean mask of values strictly below ``thresh``."""
    return df["value"].astype(float) < float(thresh)


def low_vals_export_df(df: pd.DataFrame, *, thresh: float) -> pd.DataFrame:
    """Build subject/vessel/value table for values below ``thresh``."""
    mask = flag_low_values(df, thresh=thresh)
    out = df.loc[mask, ["subject_uid", "vessel", "value"]].copy()
    out = out.rename(columns={"value": "highlighted_value"})
    out["threshold"] = float(thresh)
    out = out.sort_values(["vessel", "subject_uid", "highlighted_value"])
    return out.reset_index(drop=True)


def annotate_flagged_points(
    ax: Any,
    flagged_df: pd.DataFrame,
    *,
    vessels: list[str],
    edgecolor: str,
    text_color: str,
    marker: str = "D",
    annotate_uid: bool = True,
) -> None:
    if flagged_df.empty:
        return
    vessel_to_x = {v: i for i, v in enumerate(vessels)}
    for _, row in flagged_df.iterrows():
        xi = vessel_to_x.get(row["vessel"])
        if xi is None:
            continue
        ax.scatter(
            [xi],
            [row["value"]],
            marker=marker,
            s=40,
            facecolors="none",
            edgecolors=edgecolor,
            linewidths=1.5,
            zorder=5,
        )
        if annotate_uid:
            ax.annotate(
                str(row["subject_uid"]),
                xy=(xi, row["value"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=5.5,
                color=text_color,
                alpha=0.85,
            )


def draw_violin_figure(
    plot_df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    panel: str,
    outlier_rem: bool = False,
    outlier_high: bool = False,
    flag_low_vals: bool = False,
    low_val_thresh: float = 50.0,
    highlight_subject: str | None = None,
    fig: Figure | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | None:
    """Draw a territory-grouped violin plot onto *fig* (or a new figure).

    When ``outlier_rem`` is True and ``highlight_subject`` is set, that subject's
    points are retained even if they are IQR outliers, and highlighted on the plot.
    """
    if plot_df.empty:
        log.warning("Skipping empty figure: %s", title)
        return None

    work = plot_df.copy()
    highlight = str(highlight_subject).strip() if highlight_subject else None
    outlier_mask = flag_outliers_iqr(work)
    outlier_df = work[outlier_mask].copy() if outlier_high else pd.DataFrame()

    if flag_low_vals:
        low_mask = flag_low_values(work, thresh=low_val_thresh)
        n_low = int(low_mask.sum())
        if n_low:
            work = work[~low_mask].copy()
            log.info(
                "  flag-low-vals: dropped %d/%d points below %g for %s",
                n_low,
                n_low + len(work),
                low_val_thresh,
                title,
            )
            outlier_mask = flag_outliers_iqr(work)
            if outlier_high:
                outlier_df = work[outlier_mask].copy()

    if work.empty:
        log.warning("Skipping empty figure after low-val drop: %s", title)
        return None

    keep_subject_mask = pd.Series(False, index=work.index)
    if highlight:
        keep_subject_mask = work["subject_uid"].astype(str) == highlight

    if outlier_rem:
        n_before = len(work)
        drop_mask = outlier_mask & ~keep_subject_mask
        work = work[~drop_mask].copy()
        n_removed = n_before - len(work)
        if n_removed:
            log.info("  outlier-rem: removed %d/%d points for %s", n_removed, n_before, title)
        if outlier_high and not outlier_df.empty:
            outlier_df = outlier_df[outlier_df.index.isin(work.index)]

    vessels = [v for v in work["vessel"].cat.categories if v in set(work["vessel"])]
    if not vessels:
        log.warning("Skipping empty figure (no vessels): %s", title)
        return None
    n_by_vessel = work.groupby("vessel", observed=True)["value"].count().to_dict()

    sns.set_theme(style="whitegrid", font_scale=1.05)
    if fig is None:
        w = figsize[0] if figsize else max(10.0, 0.7 * len(vessels) + 4.0)
        h = figsize[1] if figsize else 6.5
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    palette = {
        g: TERRITORY_COLORS.get(g, "#aaaaaa")
        for g in work["group"].dropna().unique()
    }

    sns.violinplot(
        data=work,
        x="vessel",
        y="value",
        hue="group",
        order=vessels,
        hue_order=[g for g in TERRITORY_COLORS if g in palette],
        palette=palette,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=1.0,
        alpha=0.75,
        dodge=False,
        ax=ax,
    )

    sns.boxplot(
        data=work,
        x="vessel",
        y="value",
        order=vessels,
        width=0.18,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        ax=ax,
    )

    strip_df = work
    if highlight:
        strip_df = work[work["subject_uid"].astype(str) != highlight]
    if not strip_df.empty:
        sns.stripplot(
            data=strip_df,
            x="vessel",
            y="value",
            order=vessels,
            color="black",
            size=3.0,
            alpha=0.25,
            jitter=0.12,
            ax=ax,
            zorder=3,
        )

    if outlier_high and not outlier_df.empty:
        annotate_flagged_points(
            ax,
            outlier_df,
            vessels=vessels,
            edgecolor="red",
            text_color="red",
        )

    if highlight:
        hi_df = work[work["subject_uid"].astype(str) == highlight]
        if not hi_df.empty:
            annotate_flagged_points(
                ax,
                hi_df,
                vessels=vessels,
                edgecolor="#1d4ed8",
                text_color="#1d4ed8",
                marker="o",
                annotate_uid=True,
            )
            # Solid fill for selected subject.
            vessel_to_x = {v: i for i, v in enumerate(vessels)}
            for _, row in hi_df.iterrows():
                xi = vessel_to_x.get(row["vessel"])
                if xi is None:
                    continue
                ax.scatter(
                    [xi],
                    [row["value"]],
                    marker="o",
                    s=55,
                    facecolors="#2563eb",
                    edgecolors="#1e3a8a",
                    linewidths=1.2,
                    zorder=6,
                )

    if flag_low_vals:
        ax.axhline(
            float(low_val_thresh),
            color="#d97706",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            zorder=2,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(bottom=-50 if "Flow" in title else 0)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(True, alpha=0.35)

    tick_labels = []
    for v in vessels:
        n = int(n_by_vessel.get(v, 0) or 0)
        tick_labels.append(f"{v}\n(n={n})")
    ax.set_xticklabels(tick_labels)

    ax.text(
        0.98,
        0.98,
        str(panel),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="right",
        va="top",
    )

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, lab in zip(handles, labels, strict=False):
        if lab in seen or lab not in TERRITORY_COLORS:
            continue
        seen.add(lab)
        uniq_h.append(h)
        uniq_l.append(lab)
    if uniq_h:
        ax.legend(
            uniq_h,
            uniq_l,
            title="",
            loc="upper center",
            frameon=True,
            fontsize=8,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    return fig


def plot_violin_figure(
    plot_df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    panel: str,
    output_path: Path,
    outlier_rem: bool = False,
    outlier_high: bool = False,
    flag_low_vals: bool = False,
    low_val_thresh: float = 50.0,
    highlight_subject: str | None = None,
) -> Path | None:
    """Draw and save a violin figure to *output_path*."""
    fig = draw_violin_figure(
        plot_df,
        title=title,
        ylabel=ylabel,
        panel=panel,
        outlier_rem=outlier_rem,
        outlier_high=outlier_high,
        flag_low_vals=flag_low_vals,
        low_val_thresh=low_val_thresh,
        highlight_subject=highlight_subject,
    )
    if fig is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", output_path)
    return output_path


def metric_by_key(key: str) -> dict[str, Any] | None:
    for m in METRICS:
        if m["key"] == key:
            return m
    return None


__all__ = [
    "METRICS",
    "PITC_PWV_SPECS",
    "TERRITORY_COLORS",
    "VESSEL_SPECS",
    "derive_tcbf_rows",
    "draw_violin_figure",
    "flag_low_values",
    "flag_outliers_iqr",
    "load_long_measurements",
    "low_vals_export_df",
    "metric_by_key",
    "plot_violin_figure",
    "prepare_plot_frame",
    "resolve_pipeline_id",
]
