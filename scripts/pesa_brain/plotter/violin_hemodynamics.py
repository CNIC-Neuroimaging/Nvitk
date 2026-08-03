#!/usr/bin/env python3
"""Violin + scatter hemodynamics plots from qvtpy ``image_measurements``.

Thin CLI wrapper around :mod:`nvitk.stats.violin_hemodynamics`.

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

import click

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.stats.violin_hemodynamics import (
    METRICS,
    PITC_PWV_SPECS,
    VESSEL_SPECS,
    load_long_measurements,
    low_vals_export_df,
    plot_violin_figure,
    prepare_plot_frame,
    resolve_pipeline_id,
)

log = Logger()


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
    default=False,
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
@click.option(
    "--highlight-subject",
    default=None,
    help="Optional subject_uid to keep (even if IQR outlier) and highlight.",
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
    highlight_subject: str | None,
) -> None:
    """Create territory-grouped violin plots for qvtpy hemodynamics."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_id = resolve_pipeline_id(pipeline_version)
    wanted = {m.strip().lower() for m in metrics.split(",") if m.strip()}
    specs = [m for m in METRICS if m["key"] in wanted]
    if not specs:
        raise click.UsageError(f"No known metrics in --metrics={metrics!r}")

    variable_ids = sorted({m["variable_id"] for m in specs})
    log.info(
        "Loading image_measurements (pipeline=%s, variables=%s)",
        pipeline_id,
        ",".join(variable_ids),
    )
    long_df = load_long_measurements(
        pipeline_id=pipeline_id,
        variable_ids=variable_ids,
        subjects=subjects,
    )
    if long_df.empty:
        raise click.ClickException(
            f"No image_measurements rows for pipeline={pipeline_id!r}. "
            "Run scripts/pesa_brain/db/sync_db_measurements.py first."
        )
    log.info("Loaded %d measurement row(s)", len(long_df))

    written: list[Path] = []
    for meta in specs:
        vessel_specs = VESSEL_SPECS if meta["kind"] == "loc" else PITC_PWV_SPECS
        plot_df = prepare_plot_frame(
            long_df,
            variable_id=meta["variable_id"],
            specs=vessel_specs,
            derive_tcbf=bool(meta.get("derive_tcbf")),
        )
        if flag_low_vals and not plot_df.empty:
            low_export = low_vals_export_df(plot_df, thresh=low_val_thresh)
            csv_path = out_dir / f"{meta['key']}_low_vals.csv"
            low_export.to_csv(csv_path, index=False)
            log.info(
                "Wrote %d low-value row(s) (< %g) to %s",
                len(low_export),
                low_val_thresh,
                csv_path,
            )
        path = plot_violin_figure(
            plot_df,
            title=meta["title"],
            ylabel=meta["ylabel"],
            panel=meta["panel"],
            output_path=out_dir / meta["filename"],
            outlier_rem=outlier_rem,
            outlier_high=outlier_high,
            flag_low_vals=flag_low_vals,
            low_val_thresh=low_val_thresh,
            highlight_subject=highlight_subject,
        )
        if path is not None:
            written.append(path)

    if not written:
        raise click.ClickException("No figures were written (empty data for all metrics).")
    log.info("Wrote %d figure(s) under %s", len(written), out_dir)


if __name__ == "__main__":
    main()
