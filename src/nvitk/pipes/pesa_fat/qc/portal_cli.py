"""CLI: run the PESA-Fat QC portal (static HTML + Excel-backed reviews)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import DEFAULT_RESULTS_ROOT, layout
from nvitk.pipes.pesa_fat.common.stage4_qc import RES_QC_DIR
from nvitk.pipes.pesa_fat.qc.portal import create_qc_portal_app


@click.command("nvitk-pesa-fat-qc-portal")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week4').")
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8008, show_default=True)
@click.option(
    "--reviews-xlsx",
    type=click.Path(path_type=Path),
    default=None,
    help="Excel file to store reviews (default: RESULTS/res_qc/reviews.xlsx).",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    results_root: Path | None,
    host: str,
    port: int,
    reviews_xlsx: Path | None,
    log_level: str,
) -> None:
    """Serve QC HTML and accept review updates via POST /review."""
    Logger(level=log_level.upper())
    lay = layout(batch, results_root=results_root or DEFAULT_RESULTS_ROOT)
    qc_root = lay.results_dir / RES_QC_DIR
    if not qc_root.is_dir():
        raise click.ClickException(f"QC directory not found: {qc_root}")
    results_root_eff = (results_root or DEFAULT_RESULTS_ROOT)
    reviews = reviews_xlsx or (Path(results_root_eff) / RES_QC_DIR / "reviews.xlsx")

    app = create_qc_portal_app(qc_root=qc_root, reviews_xlsx=reviews)
    try:
        import uvicorn
    except Exception as exc:
        raise click.ClickException(
            'uvicorn is required. Install with: pip install "uvicorn>=0.27"'
        ) from exc
    uvicorn.run(app, host=str(host), port=int(port), log_level=str(log_level).lower())


__all__ = ["main"]

