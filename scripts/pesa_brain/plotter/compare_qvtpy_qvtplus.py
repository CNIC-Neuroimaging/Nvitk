#!/usr/bin/env python3
"""Compare per-subject, per-vessel flow_mean and PI between pipeline v2 and v3.

Only the intersection of subjects and vessels (region_ids) present in both
pipelines is compared. Generates Bland–Altman and correlation scatter plots.

Examples::

    python scripts/pesa_brain/plotter/compare_qvtpy_qvtplus.py \\
        --output-path /tmp/v2_vs_v3

    python scripts/pesa_brain/plotter/compare_qvtpy_qvtplus.py \\
        --output-path /tmp/v2_vs_v3 \\
        --old-pipeline 4dflow_v2 --new-pipeline 4dflow_v3
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.common.db_publish import resolve_repo

log = Logger()

_OLD_PIPELINE = "4dflow_v2"
_NEW_PIPELINE = "4dflow_v3"

_METRICS: list[dict[str, str]] = [
    {
        "variable_id": "flow_mean",
        "title": "Mean Flow",
        "unit": "mL/min",
        "filename": "compare_flow_mean",
    },
    {
        "variable_id": "pi",
        "title": "Pulsatility Index",
        "unit": "dimensionless",
        "filename": "compare_pi",
    },
]

_TERRITORY_ORDER = [
    "LICA",
    "RICA",
    "LMCA",
    "RMCA",
    "LACA",
    "RACA",
    "BASILAR",
    "LPCA",
    "RPCA",
    "LPCOMM",
    "RPCOMM",
    "SSSV",
    "STRV",
    "LTSV",
    "RTSV",
]

_TERRITORY_COLORS: dict[str, str] = {
    "LICA": "#c9a0c9", "RICA": "#c9a0c9",
    "LMCA": "#8fbc8f", "RMCA": "#8fbc8f", "LACA": "#8fbc8f", "RACA": "#8fbc8f",
    "BASILAR": "#f0e68c",
    "LPCA": "#f0e68c", "RPCA": "#f0e68c",
    # Communicating arteries are part of the anterior circulation in this context.
    "LPCOMM": "#8fbc8f", "RPCOMM": "#8fbc8f",
    "SSSV": "#b8b4d8", "STRV": "#b8b4d8", "LTSV": "#b8b4d8", "RTSV": "#b8b4d8",
}

def _map_region_id_v2_to_v3(region_id: str) -> str | None:
    """
    Map `4dflow_v2` region_id naming (snake_case, lower) to `4dflow_v3`
    naming (uppercase, e.g. `left_ica` -> `LICA`).
    """
    x = str(region_id).strip().lower()
    mapping = {
        # Arteries
        "basilar": "BASILAR",
        "left_ica": "LICA",
        "right_ica": "RICA",
        "left_mca": "LMCA",
        "right_mca": "RMCA",
        "left_aca": "LACA",
        "right_aca": "RACA",
        "left_pca": "LPCA",
        "right_pca": "RPCA",
        # Communicating / posterior communicating
        "left_communicating": "LPCOMM",
        "right_communicating": "RPCOMM",
        # Venous
        "sagital_sinus": "SSSV",
        "straight_sinus": "STRV",
        "left_transverse": "LTSV",
        "right_transverse": "RTSV",
    }
    return mapping.get(x)


def _load_long(repo, pipeline_id: str, variable_id: str) -> pd.DataFrame:
    """Load long-form image_measurements for a single variable + pipeline."""
    df = repo.get(
        "image_measurements",
        columns=["subject_uid", "region_id", "variable_id", "pipeline_id", "value_num"],
        filters={
            "pipeline_id": pipeline_id,
            "variable_id": variable_id,
        },
        cohort_id=False,
    )
    df = df.dropna(subset=["value_num"])
    df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")
    return df


def _merge_pipelines(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join on (subject_uid, region_id), returning v2/v3 value columns."""
    old = df_old[["subject_uid", "region_id", "value_num"]].rename(
        columns={"value_num": "v2"}
    )
    new = df_new[["subject_uid", "region_id", "value_num"]].rename(
        columns={"value_num": "v3"}
    )
    merged = old.merge(new, on=["subject_uid", "region_id"], how="inner")
    return merged


def _plot_scatter_per_vessel(
    merged: pd.DataFrame,
    metric: dict[str, str],
    out_dir: Path,
    *,
    old_pipeline: str,
    new_pipeline: str,
) -> None:
    """One correlation scatter PNG per vessel."""
    vessels_present = [v for v in _TERRITORY_ORDER if v in set(merged["region_id"])]
    color = "#4c72b0"
    for vessel in vessels_present:
        sub = merged[merged["region_id"] == vessel].dropna(subset=["v2", "v3"])
        if len(sub) < 2:
            log.warning("Skipping scatter for %s / %s (n=%d)", metric["variable_id"], vessel, len(sub))
            continue
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(
            sub["v2"], sub["v3"],
            alpha=0.55, s=28, linewidths=0,
            color=_TERRITORY_COLORS.get(vessel, color),
        )
        all_vals = pd.concat([sub["v2"], sub["v3"]])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        margin = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            lw=0.8,
            label="identity",
        )
        r, _p = stats.pearsonr(sub["v2"], sub["v3"])
        ax.set_title(
            f"{metric['title']} — {vessel}\n"
            f"n={len(sub)} subjects, r={r:.3f}"
        )
        ax.set_xlabel(f"{old_pipeline}  [{metric['unit']}]")
        ax.set_ylabel(f"{new_pipeline}  [{metric['unit']}]")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        out = out_dir / f"{metric['filename']}_{vessel}_scatter.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        log.info("Saved %s", out)


def _plot_bland_altman_per_vessel(
    merged: pd.DataFrame,
    metric: dict[str, str],
    out_dir: Path,
) -> None:
    """One Bland–Altman PNG per vessel."""
    vessels_present = [v for v in _TERRITORY_ORDER if v in set(merged["region_id"])]
    for vessel in vessels_present:
        sub = merged[merged["region_id"] == vessel].dropna(subset=["v2", "v3"])
        if len(sub) < 2:
            log.warning(
                "Skipping Bland–Altman for %s / %s (n=%d)",
                metric["variable_id"],
                vessel,
                len(sub),
            )
            continue
        mean = (sub["v2"] + sub["v3"]) / 2
        diff = sub["v3"] - sub["v2"]
        md = float(diff.mean())
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(
            mean, diff,
            alpha=0.55, s=28, linewidths=0,
            color=_TERRITORY_COLORS.get(vessel, "#4c72b0"),
        )
        ax.axhline(md, color="k", ls="-", lw=0.8, label=f"mean diff = {md:.3f}")
        ax.axhline(
            md + 1.96 * sd, color="r", ls="--", lw=0.7,
            label=f"+1.96 SD = {md + 1.96 * sd:.3f}",
        )
        ax.axhline(
            md - 1.96 * sd, color="r", ls="--", lw=0.7,
            label=f"−1.96 SD = {md - 1.96 * sd:.3f}",
        )
        ax.set_title(f"{metric['title']} — {vessel} Bland–Altman\nn={len(sub)} subjects")
        ax.set_xlabel(f"Mean of v2 & v3  [{metric['unit']}]")
        ax.set_ylabel(f"Difference (v3 − v2)  [{metric['unit']}]")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        out = out_dir / f"{metric['filename']}_{vessel}_bland_altman.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        log.info("Saved %s", out)


def _plot_per_vessel_box(
    merged: pd.DataFrame,
    metric: dict[str, str],
    out_dir: Path,
) -> None:
    """Side-by-side box plots of v2 vs v3 per vessel (overview)."""
    vessels_present = [v for v in _TERRITORY_ORDER if v in set(merged["region_id"])]
    if not vessels_present:
        return

    long = merged.melt(
        id_vars=["subject_uid", "region_id"],
        value_vars=["v2", "v3"],
        var_name="pipeline",
        value_name="value",
    )
    long["region_id"] = pd.Categorical(
        long["region_id"], categories=vessels_present, ordered=True
    )

    counts = merged.groupby("region_id").size()

    fig, ax = plt.subplots(figsize=(max(10, len(vessels_present) * 1.2), 6))
    sns_palette = {"v2": "#7cafc2", "v3": "#d97a6e"}

    import seaborn as sns
    sns.boxplot(
        data=long, x="region_id", y="value", hue="pipeline",
        palette=sns_palette, ax=ax, fliersize=2, linewidth=0.8,
        order=vessels_present,
    )

    labels = [f"{v}\n(n={int(counts.get(v, 0))})" for v in vessels_present]
    ax.set_xticks(range(len(vessels_present)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{metric['title']}  —  v2 vs v3 per vessel")
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric['unit']}")
    ax.legend(title="Pipeline", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric['filename']}_per_vessel.png", dpi=200)
    plt.close(fig)
    log.info("Saved %s", out_dir / f"{metric['filename']}_per_vessel.png")


@click.command("compare-qvtpy-qvtplus")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for output PNGs.",
)
@click.option("--old-pipeline", default=_OLD_PIPELINE, show_default=True,
              help="Pipeline id for the old (v2) measurements.")
@click.option("--new-pipeline", default=_NEW_PIPELINE, show_default=True,
              help="Pipeline id for the new (v3) measurements.")
def main(output_path: Path, old_pipeline: str, new_pipeline: str) -> None:
    """Compare flow_mean and PI between two 4D-flow pipeline versions."""
    output_path.mkdir(parents=True, exist_ok=True)
    repo = resolve_repo(prefer_sge=False)

    for metric in _METRICS:
        var = metric["variable_id"]
        log.info("Loading %s for pipelines %s / %s …", var, old_pipeline, new_pipeline)

        df_old = _load_long(repo, old_pipeline, var)
        df_new = _load_long(repo, new_pipeline, var)

        if df_old.empty:
            log.warning("No %s data for pipeline %s — skipping.", var, old_pipeline)
            continue
        if df_new.empty:
            log.warning("No %s data for pipeline %s — skipping.", var, new_pipeline)
            continue

        # `4dflow_v2` and `4dflow_v3` use different `region_id` naming conventions.
        # Normalize v2 names to v3 so we can intersect vessels correctly.
        if str(old_pipeline).lower() == "4dflow_v2" and str(new_pipeline).lower() == "4dflow_v3":
            df_old["region_id"] = df_old["region_id"].map(_map_region_id_v2_to_v3)
            before = len(df_old)
            df_old = df_old.dropna(subset=["region_id"]).copy()
            dropped = before - len(df_old)
            if dropped:
                log.info(
                    "Normalized v2 region_id for %s: dropped %d unmapped rows",
                    var,
                    dropped,
                )
            df_new["region_id"] = df_new["region_id"].astype(str).str.strip().str.upper()
        else:
            df_old["region_id"] = df_old["region_id"].astype(str).str.strip().str.upper()
            df_new["region_id"] = df_new["region_id"].astype(str).str.strip().str.upper()

        merged = _merge_pipelines(df_old, df_new)
        if merged.empty:
            log.warning("No intersecting (subject, vessel) pairs for %s — skipping.", var)
            continue

        n_subj = merged["subject_uid"].nunique()
        n_vessels = merged["region_id"].nunique()
        log.info(
            "%s: %d intersecting subjects, %d vessels, %d data points",
            var, n_subj, n_vessels, len(merged),
        )

        _plot_scatter_per_vessel(
            merged, metric, output_path,
            old_pipeline=old_pipeline, new_pipeline=new_pipeline,
        )
        _plot_bland_altman_per_vessel(merged, metric, output_path)
        _plot_per_vessel_box(merged, metric, output_path)

    log.ok("All comparison plots saved to %s", output_path)


if __name__ == "__main__":
    main()
