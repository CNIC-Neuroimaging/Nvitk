"""PESA-Fat stage 4: HTML QC report (CT-PET + Dixon)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT,
    layout,
    parse_subjects,
)
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.pipes.pesa_fat.qc.expected_ranges import EXPECTED_RANGES_CTPET, EXPECTED_RANGES_DIXON
from nvitk.pipes.pesa_fat.qc.hotspot_embed import (
    export_hotspot_gallery_for_batch,
    hotspot_gallery_control_html,
)
from nvitk.pipes.pesa_fat.qc.html_builder import build_report_html
from nvitk.pipes.pesa_fat.qc.measurements_table import (
    dataframe_to_html_table,
    load_per_subject_tables,
)
from nvitk.pipes.pesa_fat.qc.pet_axial import (
    build_ctpet_axial_section_html,
    build_dixon_axial_section_html,
)
from nvitk.pipes.pesa_fat.qc.pyvista_scenes import (
    export_ctpet_mask_strip_html,
    export_dixon_mask_strip_html,
)
from nvitk.pipes.pesa_fat.run_hotspot import _resolve_measure_ctpet

log = Logger()

PIPELINE_CHOICES = ("ct-pet-v5", "dixon-v5")
RES_QC_DIR = "res_qc"


def run_qc(
    batch: str,
    subjects: list[str],
    *,
    pipelines: list[str],
    nifti_root: Path | None = None,
    results_root: Path | None = None,
    out_html: Path | None = None,
    margin_vox: int = 3,
) -> Path:
    """Build the combined QC report; returns path to the main HTML file."""
    lay = layout(
        batch,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
    )

    qc_root = lay.results_dir / RES_QC_DIR
    assets = qc_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rel_assets = "assets"

    out_path = out_html or (qc_root / f"{batch}_qc_report.html")

    ctpet_masks: list[str] = []
    dixon_masks: list[str] = []
    ctpet_axial_parts: list[str] = []
    dixon_axial_parts: list[str] = []

    if "ct-pet-v5" in pipelines:
        for subj in subjects:
            _out_html = out_html / "ctpet" / f"masks_{subj}.html" if out_html else assets / "ctpet" / f"masks_{subj}.html"
            p = export_ctpet_mask_strip_html(
                lay, subj, _out_html, notebook=True,
            )
            if p is not None:
                ctpet_masks.append(f"{rel_assets}/ctpet/masks_{subj}.html")
            ctpet_axial_parts.append(
                build_ctpet_axial_section_html(
                    lay, subj, rel_assets, assets, margin_vox=margin_vox
                )
            )

    if "dixon-v5" in pipelines:
        for subj in subjects:
            _out_html = out_html / "dixon" / f"masks_{subj}.html" if out_html else assets / "dixon" / f"masks_{subj}.html"
            p = export_dixon_mask_strip_html(
                lay, subj, _out_html, notebook=True,
            )
            if p is not None:
                dixon_masks.append(f"{rel_assets}/dixon/masks_{subj}.html")
            dixon_axial_parts.append(
                build_dixon_axial_section_html(
                    lay, subj, rel_assets, assets, margin_vox=margin_vox
                )
            )

    if "ct-pet-v5" in pipelines:
        df_ct = load_per_subject_tables(
            lay.results_dir / ct_cfg.STAGE3_DIR / "per_subject", subjects
        )
        ct_table = dataframe_to_html_table(df_ct, EXPECTED_RANGES_CTPET)
    else:
        ct_table = "<p><em>CT-PET pipeline not selected.</em></p>"

    if "dixon-v5" in pipelines:
        df_dx = load_per_subject_tables(
            lay.results_dir / dx_cfg.STAGE3_DIR / "per_subject", subjects
        )
        dx_table = dataframe_to_html_table(df_dx, EXPECTED_RANGES_DIXON)
    else:
        dx_table = "<p><em>Dixon pipeline not selected.</em></p>"

    hotspot_dir = assets / "hotspot"
    ct_entries: list[tuple[str, str, str]] = []
    dx_entries: list[tuple[str, str, str]] = []
    if "ct-pet-v5" in pipelines or "dixon-v5" in pipelines:
        ent, err = export_hotspot_gallery_for_batch(
            lay,
            subjects,
            hotspot_dir,
            rel_assets_root=f"{rel_assets}/hotspot",
            notebook=True,
        )
        if err:
            log.debug("hotspot export skips: %d", len(err))
        for t in ent:
            if _resolve_measure_ctpet(t[1]) is not None:
                ct_entries.append(t)
            else:
                dx_entries.append(t)
        if "ct-pet-v5" not in pipelines:
            ct_entries = []
        if "dixon-v5" not in pipelines:
            dx_entries = []

    ct_hot = hotspot_gallery_control_html(ct_entries)
    dx_hot = hotspot_gallery_control_html(dx_entries)

    html = build_report_html(
        batch=batch,
        ctpet_masks_html=ctpet_masks,
        dixon_masks_html=dixon_masks,
        ctpet_measurements_table=ct_table,
        dixon_measurements_table=dx_table,
        ctpet_axial_html=ctpet_axial_parts,
        dixon_axial_html=dixon_axial_parts,
        ctpet_hotspot_gallery=ct_hot,
        dixon_hotspot_gallery=dx_hot,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html + '_report.html', encoding="utf-8")
    log.info("QC report written to %s", out_path)
    return out_path


@click.command("nvitk-pesa-fat-qc")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week4').")
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated PESA* subjects (default: all under nifti batch).",
)
@click.option(
    "--pipelines",
    default=",".join(PIPELINE_CHOICES),
    show_default=True,
    help="Comma-separated pipelines to include (ct-pet-v5, dixon-v5).",
)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--out",
    "out_html",
    type=click.Path(path_type=Path),
    default=None,
    help="Main HTML output path (default: RESULTS/<batch>/res_qc/<batch>_qc_report.html).",
)
@click.option("--margin-vox", type=int, default=3, show_default=True, help="Axial crop margin (voxels).")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    subjects: str | None,
    pipelines: str,
    nifti_root: Path | None,
    results_root: Path | None,
    out_html: Path | None,
    margin_vox: int,
    log_level: str,
) -> None:
    """Build HTML QC report for a PESA-Fat batch (or subset of subjects)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    lay = layout(
        batch,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
    )
    subj_list = parse_subjects(subjects) or list(lay.iter_subjects())
    if not subj_list:
        raise click.ClickException(
            f"No subjects found for batch {batch!r} (use --subjects or check nifti-root)."
        )

    pipes = [p.strip().lower() for p in pipelines.split(",") if p.strip()]
    bad = set(pipes) - set(PIPELINE_CHOICES)
    if bad:
        raise click.BadParameter(f"Unknown pipelines {bad}. Valid: {PIPELINE_CHOICES}")

    run_qc(
        batch,
        subj_list,
        pipelines=pipes,
        nifti_root=nifti_root,
        results_root=results_root,
        out_html=out_html,
        margin_vox=margin_vox,
    )


__all__ = ["main", "run_qc", "RES_QC_DIR"]
