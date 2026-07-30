#!/usr/bin/env python3
"""Violin + scatter hemodynamics plots from qvtpy ``image_measurements``.

Builds one figure per metric (flow, PI, RI, PITC slope, PWV Fielding, PWV
Bjornfoot) with vessels ordered and colored by territory, matching the PESA-Brain
hemodynamics style. Annotates per-vessel sample size ``n``.

Examples::

    python scripts/pesa_brain/plotter/violin_hemodynamics.py \\
        --output-path /tmp/hemo_violins

    python scripts/pesa_brain/plotter/violin_hemodynamics.py \\
        --output-path /tmp/hemo_violins \\
        --pipeline-version 4dflow_v3 \\
        --subjects PESA-Brain
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.common.db_publish import (
    QVTPY_PIPELINE_ALIASES,
    QVTPY_PIPELINE_ID,
    resolve_repo,
)
from nvitk.pipes.qvtpy.stage0_download import load_subjects
from nvitk.db.xnat import parse_subject_tokens
from nvitk.db.xnat_projects import resolve_xnat_project_cohort_token

log = Logger()

# Display order for LOC-level metrics (existing qvtpy vessels only; no C1/C3 split).
# Territory groups match the attached PESA-Brain figure.
_VESSEL_SPECS: list[tuple[str, str, str]] = [
    # (display_label, db_region_id, territory_group)
    ("TCBF", "TCBF", "TCBF & ICAs"),
    ("LICA", "LICA", "TCBF & ICAs"),
    ("RICA", "RICA", "TCBF & ICAs"),
    ("LMCA", "LMCA", "Anterior Circ."),
    ("RMCA", "RMCA", "Anterior Circ."),
    ("LACA", "LACA", "Anterior Circ."),
    ("RACA", "RACA", "Anterior Circ."),
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

_PITC_PWV_SPECS: list[tuple[str, str, str]] = [
    ("L_ICA", "L_ICA", "TCBF & ICAs"),
    ("R_ICA", "R_ICA", "TCBF & ICAs"),
    ("Basilar", "Basilar", "Posterior Circ."),
]

_TERRITORY_COLORS: dict[str, str] = {
    "TCBF & ICAs": "#c9a0c9",       # light purple/pink
    "Anterior Circ.": "#8fbc8f",    # seafoam
    "Posterior Circ.": "#f0e68c",   # pale yellow
    "Venous Drainage": "#b8b4d8",   # periwinkle
}

_METRICS: list[dict[str, Any]] = [
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


def _resolve_pipeline_id(pipeline_version: str) -> str:
    token = str(pipeline_version or "").strip()
    if not token or token.lower() in {a.lower() for a in QVTPY_PIPELINE_ALIASES}:
        return QVTPY_PIPELINE_ID
    return token


def _resolve_subjects_filter(subjects: str | None) -> list[str] | None:
    if subjects is None or not str(subjects).strip():
        return None
    tokens = parse_subject_tokens(subjects)
    if len(tokens) == 1 and resolve_xnat_project_cohort_token(tokens[0]) is not None:
        return None  # cohort applied via DataRepo.image(..., cohort_id=...)
    return load_subjects(subjects=subjects, subjects_file=None)


def _cohort_id_from_subjects(subjects: str | None) -> str | bool:
    if subjects is None or not str(subjects).strip():
        return False
    tokens = parse_subject_tokens(subjects)
    if len(tokens) == 1 and resolve_xnat_project_cohort_token(tokens[0]) is not None:
        return tokens[0].strip()
    return False


def _load_long_measurements(
    *,
    pipeline_id: str,
    variable_ids: list[str],
    subjects: str | None,
) -> pd.DataFrame:
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


def _derive_tcbf(flow_df: pd.DataFrame) -> pd.DataFrame:
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


def _prepare_plot_frame(
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
        tcbf = _derive_tcbf(sub)
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


def _flag_outliers_iqr(df: pd.DataFrame, *, k: float = 1.5) -> pd.Series:
    """Return boolean mask of IQR outliers, computed per vessel."""
    is_outlier = pd.Series(False, index=df.index)
    for vessel, grp in df.groupby("vessel", observed=True):
        q1 = grp["value"].quantile(0.25)
        q3 = grp["value"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (grp["value"] < lo) | (grp["value"] > hi)
        is_outlier.loc[mask[mask].index] = True
    return is_outlier


def _flag_low_values(df: pd.DataFrame, *, thresh: float) -> pd.Series:
    """Return boolean mask of values strictly below ``thresh``."""
    return df["value"].astype(float) < float(thresh)


def _low_vals_export_df(df: pd.DataFrame, *, thresh: float) -> pd.DataFrame:
    """Build subject/vessel/value table for values below ``thresh``."""
    mask = _flag_low_values(df, thresh=thresh)
    out = df.loc[mask, ["subject_uid", "vessel", "value"]].copy()
    out = out.rename(columns={"value": "highlighted_value"})
    out["threshold"] = float(thresh)
    out = out.sort_values(["vessel", "subject_uid", "highlighted_value"])
    return out.reset_index(drop=True)


def _annotate_flagged_points(
    ax,
    flagged_df: pd.DataFrame,
    *,
    vessels: list[str],
    edgecolor: str,
    text_color: str,
    marker: str = "D",
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
        ax.annotate(
            str(row["subject_uid"]),
            xy=(xi, row["value"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=5.5,
            color=text_color,
            alpha=0.85,
        )


def _plot_violin_figure(
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
) -> Path | None:
    if plot_df.empty:
        log.warning("Skipping empty figure: %s", title)
        return None

    outlier_mask = _flag_outliers_iqr(plot_df)
    outlier_df = plot_df[outlier_mask].copy() if outlier_high else pd.DataFrame()

    if flag_low_vals:
        low_mask = _flag_low_values(plot_df, thresh=low_val_thresh)
        n_low = int(low_mask.sum())
        if n_low:
            plot_df = plot_df[~low_mask].copy()
            log.info(
                "  flag-low-vals: dropped %d/%d points below %g for %s",
                n_low,
                n_low + len(plot_df),
                low_val_thresh,
                title,
            )
            # Recompute IQR on the remaining points so outlier-rem matches the plot.
            outlier_mask = _flag_outliers_iqr(plot_df)
            if outlier_high:
                outlier_df = plot_df[outlier_mask].copy()

    if plot_df.empty:
        log.warning("Skipping empty figure after low-val drop: %s", title)
        return None

    if outlier_rem:
        n_before = len(plot_df)
        plot_df = plot_df[~outlier_mask].copy()
        n_removed = n_before - len(plot_df)
        if n_removed:
            log.info("  outlier-rem: removed %d/%d points for %s", n_removed, n_before, title)
        if outlier_high and not outlier_df.empty:
            outlier_df = outlier_df[outlier_df.index.isin(plot_df.index)]

    vessels = [v for v in plot_df["vessel"].cat.categories if v in set(plot_df["vessel"])]
    n_by_vessel = plot_df.groupby("vessel", observed=True)["value"].count().to_dict()

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * len(vessels) + 4.0), 6.5))

    palette = {
        g: _TERRITORY_COLORS.get(g, "#aaaaaa")
        for g in plot_df["group"].dropna().unique()
    }

    sns.violinplot(
        data=plot_df,
        x="vessel",
        y="value",
        hue="group",
        order=vessels,
        hue_order=[g for g in _TERRITORY_COLORS if g in palette],
        palette=palette,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=1.0,
        alpha=0.75,
        dodge=False,
        ax=ax,
    )

    # White boxplot inside each violin.
    sns.boxplot(
        data=plot_df,
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

    # Individual subject points (scatter).
    sns.stripplot(
        data=plot_df,
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
        _annotate_flagged_points(
            ax,
            outlier_df,
            vessels=vessels,
            edgecolor="red",
            text_color="red",
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

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(bottom=-50 if 'Flow' in title else 0)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.grid(True, alpha=0.35)

    # Per-vessel N under tick labels.
    tick_labels = []
    for v in vessels:
        n = int(n_by_vessel.get(v, 0) or 0)
        tick_labels.append(f"{v}\n(n={n})")
    ax.set_xticklabels(tick_labels)

    # Panel letter (top-right).
    ax.text(
        0.98,
        0.98,
        str(panel),
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="right",
        va="top",
    )

    # Territory legend (dedupe seaborn's hue legend).
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, lab in zip(handles, labels, strict=False):
        if lab in seen or lab not in _TERRITORY_COLORS:
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
            fontsize=9,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", output_path)
    return output_path


@click.command("violin-hemodynamics")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for output PNG figures.",
)
@click.option(
    "--pipeline-version",
    default=QVTPY_PIPELINE_ID,
    show_default=True,
    help=(
        "image_measurements pipeline_id (or alias: qvtpy / latest / v3). "
        f"Default is the current qvtpy id ({QVTPY_PIPELINE_ID})."
    ),
)
@click.option(
    "--subjects",
    default=None,
    help=(
        "Optional comma/space-separated subject ids, or a cohort alias "
        "(e.g. PESA-Brain). Omit to use all subjects for the pipeline."
    ),
)
@click.option(
    "--metrics",
    default="flow,pi,ri,pitc_slope,pwv_f,pwv_b",
    show_default=True,
    help="Comma-separated metric keys to plot.",
)
@click.option(
    "--outlier-rem/--no-outlier-rem",
    is_flag=True,
    default=True,
    help="Remove IQR outliers per vessel before plotting.",
)
@click.option(
    "--outlier-high/--no-outlier-high",
    is_flag=True,
    default=False,
    help="Highlight IQR outliers with subject uid labels.",
)
@click.option(
    "--flag-low-vals/--no-flag-low-vals",
    is_flag=True,
    default=False,
    help=(
        "Drop values below --low-val-thresh from the plot, draw a dashed "
        "threshold line, and write a CSV per metric listing those rows "
        "(subject_uid / vessel / highlighted_value)."
    ),
)
@click.option(
    "--low-val-thresh",
    type=float,
    default=25.0,
    show_default=True,
    help="Absolute threshold for --flag-low-vals (e.g. 50 mL/min for flow).",
)
def main(
    output_path: Path,
    pipeline_version: str,
    subjects: str | None,
    metrics: str,
    outlier_rem: bool,
    outlier_high: bool,
    flag_low_vals: bool,
    low_val_thresh: float,
) -> None:
    """Create territory-grouped violin plots for qvtpy hemodynamics."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_id = _resolve_pipeline_id(pipeline_version)
    wanted = {m.strip().lower() for m in metrics.split(",") if m.strip()}
    specs = [m for m in _METRICS if m["key"] in wanted]
    if not specs:
        raise click.UsageError(f"No known metrics in --metrics={metrics!r}")

    variable_ids = sorted({m["variable_id"] for m in specs})
    log.info(
        "Loading image_measurements (pipeline=%s, variables=%s)",
        pipeline_id,
        ",".join(variable_ids),
    )
    long_df = _load_long_measurements(
        pipeline_id=pipeline_id,
        variable_ids=variable_ids,
        subjects=subjects,
    )
    if long_df.empty:
        raise click.ClickException(
            f"No image_measurements rows for pipeline={pipeline_id!r}. "
            "Run scripts/pesa_brain/db/sync_db_results.py first."
        )
    log.info("Loaded %d measurement row(s)", len(long_df))

    written: list[Path] = []
    for meta in specs:
        vessel_specs = _VESSEL_SPECS if meta["kind"] == "loc" else _PITC_PWV_SPECS
        plot_df = _prepare_plot_frame(
            long_df,
            variable_id=meta["variable_id"],
            specs=vessel_specs,
            derive_tcbf=bool(meta.get("derive_tcbf")),
        )
        if flag_low_vals and not plot_df.empty:
            low_export = _low_vals_export_df(plot_df, thresh=low_val_thresh)
            csv_path = out_dir / f"{meta['key']}_low_vals.csv"
            low_export.to_csv(csv_path, index=False)
            log.info(
                "Wrote %d low-value row(s) (< %g) to %s",
                len(low_export),
                low_val_thresh,
                csv_path,
            )
        path = _plot_violin_figure(
            plot_df,
            title=meta["title"],
            ylabel=meta["ylabel"],
            panel=meta["panel"],
            output_path=out_dir / meta["filename"],
            outlier_rem=outlier_rem,
            outlier_high=outlier_high,
            flag_low_vals=flag_low_vals,
            low_val_thresh=low_val_thresh,
        )
        if path is not None:
            written.append(path)

    if not written:
        raise click.ClickException("No figures were written (empty data for all metrics).")
    log.info("Wrote %d figure(s) under %s", len(written), out_dir)


if __name__ == "__main__":
    main()
