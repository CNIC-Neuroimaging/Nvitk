#!/usr/bin/env python3
"""Plot the automatic 4D-flow QC metrics published by the qvtpy stage 9.

Reads the ``qc_*`` variables straight out of the dataset and draws the four views that answer
"how healthy is this cohort's flow data, and where is the damage concentrated?":

1. **Per-vessel pass rates** — which vessels fail the literature band, ranked. A single vessel at the
   top usually means a segmentation or LOC-placement problem specific to it, not a bad cohort.
2. **Score distributions** — the plausibility and combined scores per vessel, so a vessel that is
   merely borderline is distinguishable from one that is failing outright.
3. **Conservation residuals** — the junction balances, with the ±15% tolerance drawn on. The spread
   matters more than the offset: a consistent bias is an unmeasured branch, scatter is measurement
   noise.
4. **Subject-level summary** — the anterior/posterior split against its 72% reference, and how many
   subjects carry at least one failing check.

Examples::

    python scripts/pesa_brain/plotter/autoqc_summary.py --output-path /tmp/qc

    python scripts/pesa_brain/plotter/autoqc_summary.py \\
        --output-path /tmp/qc --pipeline qvtpy --export-csv
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.measure.hemodynamics import ANTERIOR_SHARE_PCT, ANTERIOR_SHARE_TOL_PCT, CONSERVATION_TOL
from nvitk.pipes.qvtpy.stage9_autoqc import QC_LABELS, QC_VARIABLES

log = Logger()

#: Score at or below which a vessel counts as failing, matching the stage's own flag.
FAIL_BELOW = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_qc(repo, *, pipeline: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The published QC metrics, wide: one row per ``(subject, vessel)`` and one per subject.

    Read from the long tables directly rather than through ``DataRepo.image``, which applies
    catalog-default and cohort restrictions this report does not want — it should describe every
    row the stage wrote.

    Returns
    -------
    (vessel, subject)
        Both empty when the stage has not run.
    """
    image_vars = [v for v, table in QC_VARIABLES.items() if table == "image_measurements"]
    clinical_vars = [v for v, table in QC_VARIABLES.items() if table == "clinical_measurements"]

    image = repo.get("image_measurements", cohort_id=False)
    rows = image.loc[image["variable_id"].astype(str).isin(image_vars)]
    if pipeline and "pipeline_id" in rows.columns:
        rows = rows.loc[rows["pipeline_id"].astype(str) == str(pipeline)]
    vessel = (
        rows.pivot_table(
            index=["subject_uid", "region_id"], columns="variable_id",
            values="value_num", aggfunc="mean",
        ).reset_index()
        if not rows.empty else pd.DataFrame()
    )

    clinical = repo.get("clinical_measurements", cohort_id=False)
    crows = clinical.loc[clinical["variable_id"].astype(str).isin(clinical_vars)]
    subject = (
        crows.pivot_table(index="subject_uid", columns="variable_id",
                          values="value_num", aggfunc="mean").reset_index()
        if not crows.empty else pd.DataFrame()
    )
    return vessel, subject


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def plot_pass_rates(vessel: pd.DataFrame, out_dir: Path) -> Path | None:
    """Failure rate per vessel, ranked worst first."""
    import matplotlib.pyplot as plt

    if vessel.empty or "qc_flow_plausible" not in vessel.columns:
        return None
    per = vessel.groupby("region_id")["qc_flow_plausible"].agg(
        n="size",
        scored=lambda s: float(s.notna().sum()),
        failing=lambda s: float((s < FAIL_BELOW).sum()),
    )
    # A vessel with no band was never assessed — the communicating arteries and the venous sinuses.
    # Showing those at "0% failing" would read as a clean bill of health for exactly the vessels the
    # scoring deliberately declines to judge.
    per["rate"] = 100.0 * per["failing"] / per["scored"].where(per["scored"] > 0)
    per = per.sort_values("rate", ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.32 * len(per) + 1.5)))
    colours = [
        "#D9D9D9" if not np.isfinite(r)
        else "#C44E52" if r >= 20 else "#DD8452" if r >= 5 else "#55A868"
        for r in per["rate"]
    ]
    ax.barh(per.index.astype(str), per["rate"].fillna(0.0), color=colours)
    for i, (rate, n, scored) in enumerate(zip(per["rate"], per["n"], per["scored"])):
        text = (
            f"not scored — no band  (n={int(n)})" if not np.isfinite(rate)
            else f"{rate:.0f}%  ({int(per['failing'].iloc[i])}/{int(scored)})"
        )
        ax.annotate(text, (0 if not np.isfinite(rate) else rate, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#555555")
    ax.set_xlabel(f"% of measurements below a plausibility of {FAIL_BELOW:g}")
    ax.set_title("Flow plausibility failures by vessel")
    ax.set_xlim(0, 100.0)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir / "qc_pass_rates.png")


def plot_score_distributions(vessel: pd.DataFrame, out_dir: Path) -> Path | None:
    """Plausibility and combined score per vessel, as violins."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    columns = [c for c in ("qc_flow_plausible", "qc_score") if c in vessel.columns]
    if vessel.empty or not columns:
        return None

    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 4.4 * len(columns)), squeeze=False)
    order = sorted(vessel["region_id"].astype(str).unique())
    for ax, column in zip(axes.ravel(), columns):
        sns.violinplot(
            data=vessel, x="region_id", y=column, order=order, hue="region_id",
            legend=False, ax=ax, inner="box", cut=0, density_norm="width", palette="tab10",
        )
        ax.axhline(FAIL_BELOW, color="#C44E52", ls="--", lw=1.2,
                   label=f"fail below {FAIL_BELOW:g}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(QC_LABELS.get(column, column))
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.tight_layout()
    return _save(fig, out_dir / "qc_score_distributions.png")


def plot_conservation(vessel: pd.DataFrame, out_dir: Path) -> Path | None:
    """Junction mass-conservation residuals, with the tolerance band drawn on."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if vessel.empty or "qc_conservation" not in vessel.columns:
        return None
    rows = vessel.dropna(subset=["qc_conservation"])
    if rows.empty:
        return None

    order = sorted(rows["region_id"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(9, max(4.0, 1.1 * len(order) + 2.0)))
    sns.violinplot(
        data=rows, x="qc_conservation", y="region_id", order=order, hue="region_id",
        legend=False, ax=ax, inner="quartile", cut=0, density_norm="width", palette="tab10",
    )
    ax.axvline(0, color="#333333", lw=1.2)
    ax.axvspan(-CONSERVATION_TOL, CONSERVATION_TOL, color="#55A868", alpha=0.12,
               label=f"within ±{CONSERVATION_TOL:.0%}")
    ax.set_xlabel("(inflow − outflow) / inflow")
    ax.set_ylabel("parent vessel")
    ax.set_title("Junction mass-conservation residuals")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir / "qc_conservation.png")


def plot_subject_summary(subject: pd.DataFrame, out_dir: Path) -> Path | None:
    """Anterior/posterior split against its reference, and the subject flag counts."""
    import matplotlib.pyplot as plt

    if subject.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if "qc_ap_share" in subject.columns:
        share = pd.to_numeric(subject["qc_ap_share"], errors="coerce").dropna()
        ax.hist(share, bins=40, color="#4C72B0", alpha=0.8)
        lo = ANTERIOR_SHARE_PCT - ANTERIOR_SHARE_TOL_PCT
        hi = ANTERIOR_SHARE_PCT + ANTERIOR_SHARE_TOL_PCT
        ax.axvspan(lo, hi, color="#55A868", alpha=0.15, label=f"expected {lo:.0f}–{hi:.0f}%")
        ax.axvline(ANTERIOR_SHARE_PCT, color="#C44E52", ls="--", lw=1.5,
                   label=f"{ANTERIOR_SHARE_PCT:.0f}% (Zarrinkoob 2015)")
        outside = int(((share < lo) | (share > hi)).sum())
        ax.set_title(f"Anterior share of inflow — {outside} of {len(share)} outside the band")
        ax.set_xlabel("anterior share (%)")
        ax.set_ylabel("subjects")
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    counts, labels = [], []
    for column in ("qc_ap_flag", "qc_subject_flag"):
        if column in subject.columns:
            values = pd.to_numeric(subject[column], errors="coerce").dropna()
            counts.append(float((values >= 0.5).sum()))
            labels.append(QC_LABELS.get(column, column).split("(")[0].strip())
    if counts:
        total = len(subject)
        bars = ax.bar(labels, counts, color=["#DD8452", "#C44E52"][: len(counts)])
        for bar, value in zip(bars, counts):
            ax.annotate(f"{int(value)} / {total}\n({100*value/max(total,1):.0f}%)",
                        (bar.get_x() + bar.get_width() / 2, value), ha="center",
                        va="bottom", fontsize=9)
        ax.set_ylim(0, max(counts) * 1.25 if max(counts) else 1)
        ax.set_ylabel("subjects flagged")
        ax.set_title("Subject-level checks")
        ax.tick_params(axis="x", rotation=10)
    else:
        ax.set_axis_off()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir / "qc_subject_summary.png")


def _save(fig, path: Path) -> Path:
    """Write *fig* and close it, so a long run does not accumulate open figures."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)
    return path


def summary_table(vessel: pd.DataFrame, subject: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel counts and rates, for the console and the optional CSV."""
    if vessel.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for region, group in vessel.groupby("region_id"):
        entry = {"region_id": str(region), "n": int(len(group))}
        for column in ("qc_flow_plausible", "qc_score"):
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce")
                entry[f"{column}_median"] = float(values.median())
                entry[f"{column}_fail_pct"] = float(100.0 * (values < FAIL_BELOW).mean())
        if "qc_conservation" in group.columns:
            residual = pd.to_numeric(group["qc_conservation"], errors="coerce").dropna()
            if not residual.empty:
                entry["conservation_median"] = float(residual.median())
                entry["conservation_out_pct"] = float(
                    100.0 * (residual.abs() > CONSERVATION_TOL).mean()
                )
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("qc_flow_plausible_fail_pct", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
@click.command("autoqc-summary")
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for the output PNG figures.",
)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the one configured in .nvitk/settings.json.",
)
@click.option(
    "--pipeline",
    default="",
    help="Restrict to one image pipeline_id. Omit to use every published row.",
)
@click.option(
    "--export-csv/--no-export-csv",
    default=False,
    show_default=True,
    help="Also write the per-vessel summary table beside the figures.",
)
def main(output_path: Path, dataset: Path | None, pipeline: str, export_csv: bool) -> None:
    """Plot the automatic QC metrics for the 4D-flow measurements in a dataset."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(
            f"Could not open the dataset ({exc}). Pass --dataset PATH, or configure one in "
            f".nvitk/settings.json."
        ) from exc

    vessel, subject = load_qc(repo, pipeline=pipeline)
    if vessel.empty and subject.empty:
        raise click.ClickException(
            "No qc_* measurements in this dataset. Run 'nvitk-qvtpy-autoqc' first."
        )
    log.info(
        "Loaded %d vessel row(s) over %d subject(s), and %d subject-level row(s).",
        len(vessel), vessel["subject_uid"].nunique() if not vessel.empty else 0, len(subject),
    )

    out_dir = Path(output_path)
    written = [
        path for path in (
            plot_pass_rates(vessel, out_dir),
            plot_score_distributions(vessel, out_dir),
            plot_conservation(vessel, out_dir),
            plot_subject_summary(subject, out_dir),
        ) if path is not None
    ]

    table = summary_table(vessel, subject)
    if not table.empty:
        click.echo("\n" + table.round(2).to_string(index=False))
        if export_csv:
            csv_path = out_dir / "qc_summary.csv"
            table.to_csv(csv_path, index=False)
            log.info("Wrote %s", csv_path)

    if not written:
        raise click.ClickException("Nothing could be plotted — the QC columns are all empty.")
    log.ok("Wrote %d figure(s) under %s", len(written), out_dir)


if __name__ == "__main__":
    main()
