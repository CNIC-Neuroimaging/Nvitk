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

#: Compact plot labels for :func:`~nvitk.stats.vessel_network.canonical_node` ids.
#: Two importers routinely publish the same vessel as ``LICA`` and ``left_ica``; without this map
#: every figure would draw them as separate categories.
_PLOT_LABEL: dict[str, str] = {
    "lva": "LVA",
    "rva": "RVA",
    "basi": "BASILAR",
    "lpca": "LPCA",
    "rpca": "RPCA",
    "lica": "LICA",
    "rica": "RICA",
    "laca": "LACA",
    "raca": "RACA",
    "lmca": "LMCA",
    "rmca": "RMCA",
    "lpcomm": "LPCOMM",
    "rpcomm": "RPCOMM",
    "acomm": "ACOMM",
    "sss": "SSSV",
    "strs": "STRV",
    "lts": "LTSV",
    "rts": "RTSV",
}


def _display_region(region_id: object) -> str:
    """Canonical short label for a published region spelling, or the original when unknown."""
    from nvitk.stats.vessel_network import canonical_node

    node = canonical_node(region_id)
    if node is None:
        return str(region_id)
    return _PLOT_LABEL.get(node, node.upper())


def _canonicalize_vessel_frame(vessel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse mixed region spellings onto one display label per vessel.

    Without this, a cohort that mixed ``4dflow_v2`` (``left_ica``) and ``qvtpy`` (``LICA``) imports
    draws every vessel twice — once per spelling — and the pass-rate / violin plots look like
    duplicated anatomy.
    """
    if vessel.empty or "region_id" not in vessel.columns:
        return vessel
    out = vessel.copy()
    before = int(out["region_id"].nunique())
    out["region_id"] = out["region_id"].map(_display_region)
    after = int(out["region_id"].nunique())
    if before != after:
        log.info(
            "Canonicalized region labels for plotting: %d spellings → %d vessels.",
            before, after,
        )
    # Same subject can now carry two rows for one vessel (one per importer spelling); average them.
    keys = ["subject_uid", "region_id"]
    value_cols = [c for c in out.columns if c not in keys]
    if not value_cols:
        return out.drop_duplicates(subset=keys)
    return out.groupby(keys, as_index=False)[value_cols].mean(numeric_only=True)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_qc(repo, *, pipeline: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The published QC metrics, wide: one row per ``(subject, vessel)`` and one per subject.

    Read from the long tables directly rather than through ``DataRepo.image``, which applies
    catalog-default and cohort restrictions this report does not want — it should describe every
    row the stage wrote.

    Region spellings are canonicalized (``LICA`` / ``left_ica`` → ``LICA``) so mixed importers do
    not fragment the figures.

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
    vessel = _canonicalize_vessel_frame(vessel)

    clinical = repo.get("clinical_measurements", cohort_id=False)
    crows = clinical.loc[clinical["variable_id"].astype(str).isin(clinical_vars)]
    subject = (
        crows.pivot_table(index="subject_uid", columns="variable_id",
                          values="value_num", aggfunc="mean").reset_index()
        if not crows.empty else pd.DataFrame()
    )
    return vessel, subject


def load_flow(repo, *, pipeline: str = "", variable: str = "flow_mean") -> pd.DataFrame:
    """
    Long ``subject_uid`` / ``region_id`` / ``flow_ml_min`` rows for the consensus checks.

    The published QC columns are enough for the per-scan figures, but the cohort-level junction
    regression needs the flows themselves. The unit is inferred rather than assumed, exactly as
    stage 9 infers it — a dataset holding mL/s would otherwise regress fine and report nonsense
    intercepts.
    """
    from nvitk.pipes.qvtpy.stage9_autoqc import infer_flow_scale

    empty = pd.DataFrame(columns=["subject_uid", "region_id", "flow_ml_min"])
    image = repo.get("image_measurements", cohort_id=False)
    if image is None or image.empty or "variable_id" not in image.columns:
        return empty
    rows = image.loc[image["variable_id"].astype(str) == str(variable)]
    if pipeline and "pipeline_id" in rows.columns:
        rows = rows.loc[rows["pipeline_id"].astype(str) == str(pipeline)]
    if rows.empty:
        log.warning("No %s rows — the consensus junction check needs them.", variable)
        return empty

    flow = rows.loc[:, ["subject_uid", "region_id", "value_num"]].copy()
    flow["flow_ml_min"] = pd.to_numeric(flow["value_num"], errors="coerce") * infer_flow_scale(
        flow["value_num"]
    )
    return flow.drop(columns=["value_num"])


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def plot_consensus_junctions(flow: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Junction inflow against outflow, one panel per junction — the consensus consistency check.

    Each point is a subject. The dashed line is identity (perfect conservation); the solid line is
    the fitted regression. Read the **slope**: a pipeline that loses a constant fraction of outflow
    still correlates near-perfectly, so a high *r* on its own certifies nothing.
    """
    import matplotlib.pyplot as plt

    from nvitk.pipes.qvtpy.stage9_autoqc import CONSENSUS_JUNCTIONS, consensus_junction_report
    from nvitk.stats.vessel_network import canonical_node

    if flow.empty:
        return None
    report = consensus_junction_report(flow)
    if report.empty:
        return None

    wide = (
        flow.assign(node=flow["region_id"].map(canonical_node))
        .dropna(subset=["node"])
        .pivot_table(index="subject_uid", columns="node", values="flow_ml_min", aggfunc="mean")
    )
    checked = {row.junction: row for row in report.itertuples()}
    panels = [j for j in CONSENSUS_JUNCTIONS if j[0] in checked]
    if not panels:
        return None

    cols = min(3, len(panels))
    rows_n = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.6 * cols, 4.4 * rows_n), squeeze=False)
    for ax, (key, inlets, outlets) in zip(axes.ravel(), panels):
        stats = checked[key]
        block = wide.loc[:, [*inlets, *outlets]].dropna()
        x = block[list(inlets)].sum(axis=1)
        y = block[list(outlets)].sum(axis=1)
        ax.scatter(x, y, s=18, alpha=0.65, edgecolor="none", color="#4878cf")

        span = np.linspace(float(min(x.min(), y.min())), float(max(x.max(), y.max())), 2)
        ax.plot(span, span, "--", color="#888", lw=1.2, label="identity")
        ax.plot(span, stats.intercept + stats.slope * span, "-", color="#c44e52", lw=1.6,
                label=f"slope {stats.slope:.3f}")
        ok = "✓" if stats.slope_includes_one else "✗"
        ax.set_title(
            f"{stats.label}\n{ok} slope {stats.slope:.3f} "
            f"[{stats.slope_ci_low:.3f}, {stats.slope_ci_high:.3f}]  ·  r={stats.r:.3f}  ·  n={int(stats.n)}",
            fontsize=9,
        )
        ax.set_xlabel(f"inflow: {stats.inlets}  (mL/min)")
        ax.set_ylabel(f"outflow: {stats.outlets}  (mL/min)")
        ax.legend(fontsize=8, frameon=False)

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(
        "Junction internal consistency — inflow vs outflow across subjects "
        "(✓ = 95% CI on the slope includes 1)",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, out_dir / "qc_consensus_junctions.png")


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
    ax.set_ylabel("anchor vessel")
    ax.set_title("Junction mass-conservation residuals")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir / "qc_conservation.png")


def plot_segment_cv(vessel: pd.DataFrame, out_dir: Path) -> Path | None:
    """Along-segment flow CV, with the soft gate drawn on."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from nvitk.measure.hemodynamics import SEGMENT_CV_TOL

    if vessel.empty or "qc_segment_cv" not in vessel.columns:
        return None
    rows = vessel.dropna(subset=["qc_segment_cv"])
    if rows.empty:
        return None

    order = sorted(rows["region_id"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(9, max(4.0, 1.1 * len(order) + 2.0)))
    sns.violinplot(
        data=rows, x="qc_segment_cv", y="region_id", order=order, hue="region_id",
        legend=False, ax=ax, inner="quartile", cut=0, density_norm="width", palette="tab10",
    )
    ax.axvspan(0.0, SEGMENT_CV_TOL, color="#55A868", alpha=0.12,
               label=f"within {SEGMENT_CV_TOL:.0%}")
    ax.set_xlabel("flow CV along segment")
    ax.set_ylabel("vessel")
    ax.set_title("Along-segment flow consistency")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir / "qc_segment_cv.png")


def plot_subject_summary(subject: pd.DataFrame, out_dir: Path) -> Path | None:
    """Anterior/posterior split against its reference, and the subject flag counts."""
    import matplotlib.pyplot as plt

    if subject.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    n_subjects = int(len(subject))
    if "qc_ap_share" in subject.columns:
        share = pd.to_numeric(subject["qc_ap_share"], errors="coerce")
        valid = share.dropna()
        ax.hist(valid, bins=40, color="#4C72B0", alpha=0.8)
        lo = ANTERIOR_SHARE_PCT - ANTERIOR_SHARE_TOL_PCT
        hi = ANTERIOR_SHARE_PCT + ANTERIOR_SHARE_TOL_PCT
        ax.axvspan(lo, hi, color="#55A868", alpha=0.15, label=f"expected {lo:.0f}–{hi:.0f}%")
        ax.axvline(ANTERIOR_SHARE_PCT, color="#C44E52", ls="--", lw=1.5,
                   label=f"{ANTERIOR_SHARE_PCT:.0f}% (Zarrinkoob 2015)")
        # Prefer the published flag when present so the left title matches the right-hand bar
        # (same denominator / gate as ``qc_ap_flag``).
        if "qc_ap_flag" in subject.columns:
            outside = int(
                (pd.to_numeric(subject["qc_ap_flag"], errors="coerce").fillna(0.0) >= 0.5).sum()
            )
        else:
            outside = int(((share < lo) | (share > hi) | share.isna()).sum())
        ax.set_title(
            f"Anterior share of inflow — {outside} of {n_subjects} outside the band"
        )
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
            values = pd.to_numeric(subject[column], errors="coerce").fillna(0.0)
            counts.append(float((values >= 0.5).sum()))
            labels.append(QC_LABELS.get(column, column).split("(")[0].strip())
    if counts:
        total = n_subjects
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

    flow = load_flow(repo, pipeline=pipeline)

    out_dir = Path(output_path)
    written = [
        path for path in (
            plot_pass_rates(vessel, out_dir),
            plot_score_distributions(vessel, out_dir),
            plot_conservation(vessel, out_dir),
            plot_segment_cv(vessel, out_dir),
            plot_subject_summary(subject, out_dir),
            plot_consensus_junctions(flow, out_dir),
        ) if path is not None
    ]

    table = summary_table(vessel, subject)
    if not table.empty:
        click.echo("\n" + table.round(2).to_string(index=False))
        if export_csv:
            csv_path = out_dir / "qc_summary.csv"
            table.to_csv(csv_path, index=False)
            log.info("Wrote %s", csv_path)

    # ---- Cohort-level consistency: the validation statistic, not the per-scan gate ------------
    if not flow.empty:
        from nvitk.pipes.qvtpy.stage9_autoqc import consensus_junction_report

        consensus = consensus_junction_report(flow)
        if not consensus.empty:
            shown = consensus.loc[:, [
                "label", "n", "slope", "slope_ci_low", "slope_ci_high", "r",
                "mean_rel_residual", "slope_includes_one",
            ]]
            click.echo("\nJunction internal consistency (inflow regressed on outflow):")
            click.echo(shown.round(4).to_string(index=False))
            failed = consensus.loc[~consensus["slope_includes_one"], "label"].tolist()
            if failed:
                click.echo(
                    "\n  Slope CI excludes 1 at: " + ", ".join(failed)
                    + "\n  A systematic inflow/outflow imbalance — unmeasured side branches or a "
                      "scaling error, not per-scan noise."
                )
            if export_csv:
                consensus.to_csv(out_dir / "qc_consensus_junctions.csv", index=False)
                log.info("Wrote %s", out_dir / "qc_consensus_junctions.csv")

    if not written:
        raise click.ClickException("Nothing could be plotted — the QC columns are all empty.")
    log.ok("Wrote %d figure(s) under %s", len(written), out_dir)


if __name__ == "__main__":
    main()
