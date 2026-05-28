"""CLI: run the PESA-Fat QC portal (static HTML + Excel/DB-backed reviews)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import DEFAULT_RESULTS_ROOT, layout
from nvitk.pipes.pesa_fat.common.stage4_qc import RES_QC_DIR
from nvitk.pipes.pesa_fat.qc.portal import create_qc_portal_app


@click.command("nvitk-pesa-fat-qc-portal")
@click.option("--batch", required=False, default=None, help="Optional batch name (e.g. '202602_Week4').")
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8008, show_default=True)
@click.option(
    "--reviews-xlsx",
    type=click.Path(path_type=Path),
    default=None,
    help="Excel file to store reviews (default: RESULTS/res_qc/reviews.xlsx).",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Skip NVITK database publish on review (Excel only).",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str | None,
    results_root: Path | None,
    host: str,
    port: int,
    reviews_xlsx: Path | None,
    no_db: bool,
    log_level: str,
) -> None:
    """Serve QC HTML and accept review updates via POST /review (Excel + DB)."""
    Logger(level=log_level.upper())
    results_root_eff = (results_root or DEFAULT_RESULTS_ROOT)
    reviews = reviews_xlsx or (Path(results_root_eff) / RES_QC_DIR / "reviews.xlsx")

    if batch:
        lay = layout(batch, results_root=results_root_eff)
        qc_root = lay.results_dir / RES_QC_DIR
        if not qc_root.is_dir():
            raise click.ClickException(f"QC directory not found: {qc_root}")
    else:
        # Dashboard mode (no single batch): use RESULTS root as the static tree.
        qc_root = Path(results_root_eff) / RES_QC_DIR  # unused for file serving; kept for app factory shape
        qc_root.mkdir(parents=True, exist_ok=True)

    app = create_qc_portal_app(
        qc_root=qc_root,
        reviews_xlsx=reviews,
        results_root=Path(results_root_eff),
        default_batch=batch,
        publish_db=not no_db,
    )
    try:
        import uvicorn
    except Exception as exc:
        raise click.ClickException(
            'uvicorn is required. Install with: pip install "uvicorn>=0.27"'
        ) from exc
    uvicorn.run(app, host=str(host), port=int(port), log_level=str(log_level).lower())


__all__ = ["main"]

