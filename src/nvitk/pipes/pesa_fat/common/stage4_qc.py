"""PESA-Fat stage 4: HTML QC report (CT-PET + Dixon).

v2: one report per subject + batch index.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
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
from nvitk.pipes.pesa_fat.qc.embed import iframe_srcdoc
from nvitk.pipes.pesa_fat.qc.hotspot_embed import (
    export_hotspot_gallery_for_batch,
    hotspot_gallery_control_srcdoc,
)
from nvitk.pipes.pesa_fat.qc.html_builder import build_ctpet_report_html, build_dixon_report_html
from nvitk.pipes.pesa_fat.qc.measurements_table import (
    copy_measurements_xlsx_for_qc,
    dataframe_to_html_table,
    load_per_subject_tables,
    measurements_download_button_html,
)
from nvitk.pipes.pesa_fat.qc.pet_axial import (
    build_ctpet_slice_viewer_html,
    build_dixon_slice_viewer_html,
)
from nvitk.pipes.pesa_fat.qc.dixon_heatmap import build_dixon_measurement_heatmap_html
from nvitk.pipes.pesa_fat.qc.review_policy import expected_review_structures
from nvitk.pipes.pesa_fat.qc.slice_review import (
    report_footer_toolbar_html,
    report_header_toolbar_html,
    review_context,
)
from nvitk.pipes.pesa_fat.qc.pyvista_scenes import (
    export_ctpet_overview_html,
    export_dixon_overview_html,
)
from nvitk.pipes.pesa_fat.run_hotspot import _resolve_measure_ctpet

log = Logger()

PIPELINE_CHOICES = ("ct-pet-v5", "dixon-v5")
RES_QC_DIR = "res_qc"


def _index_html(batch: str, links: list[tuple[str, str, str]]) -> str:
    items = "\n".join(
        f"<li><code>{subj}</code> · <a href='{href_ct}'>CT-PET</a> · <a href='{href_dx}'>Dixon</a></li>"
        for subj, href_ct, href_dx in links
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>PESA-Fat QC index — {batch}</title></head>
<body>
<h1>PESA-Fat QC index — batch <code>{batch}</code></h1>
<ul>{items}</ul>
</body></html>"""


def run_qc_subject(
    lay,
    subject: str,
    *,
    pipelines: list[str],
    out_dir: Path,
    margin_vox: int = 3,
) -> tuple[Path | None, Path | None]:
    """Build CT-PET and/or Dixon QC pages for one subject; returns (ctpet_path, dixon_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rel_assets = "assets"

    out_ct = out_dir / f"qc_ctpet_{subject}.html"
    out_dx = out_dir / f"qc_dixon_{subject}.html"

    structures_ct = (
        expected_review_structures("ct-pet-v5") if "ct-pet-v5" in pipelines else []
    )
    structures_dx = expected_review_structures("dixon-v5") if "dixon-v5" in pipelines else []
    review_ctx_ct = (
        review_context(
            batch=lay.batch,
            subject=subject,
            pipeline="ct-pet-v5",
            report_relpath=f"{subject}/{out_ct.name}",
            structures=structures_ct,
        )
        if structures_ct
        else None
    )
    review_ctx_dx = (
        review_context(
            batch=lay.batch,
            subject=subject,
            pipeline="dixon-v5",
            report_relpath=f"{subject}/{out_dx.name}",
            structures=structures_dx,
        )
        if structures_dx
        else None
    )

    ctpet_masks: list[str] = []
    dixon_masks: list[str] = []
    ctpet_axial_parts: list[str] = []
    dixon_axial_parts: list[str] = []

    if "ct-pet-v5" in pipelines:
        p = export_ctpet_overview_html(
            lay, subject, assets / "ctpet" / f"overview_{subject}.html", notebook=True
        )
        if p is not None:
            # embed as srcdoc for portability
            html_txt = Path(p).read_text(encoding="utf-8", errors="ignore")
            ctpet_masks.append(iframe_srcdoc(html_txt, height_px=420, title="ctpet_overview"))
        ctpet_axial_parts.append(
            build_ctpet_slice_viewer_html(
                lay,
                subject,
                margin_vox=margin_vox,
                assets_dir=assets / "slices" / "ctpet",
                assets_rel=f"{rel_assets}/slices/ctpet",
                review_ctx=review_ctx_ct,
            )
        )

    if "dixon-v5" in pipelines:
        p = export_dixon_overview_html(
            lay, subject, assets / "dixon" / f"overview_{subject}.html", notebook=True
        )
        if p is not None:
            html_txt = Path(p).read_text(encoding="utf-8", errors="ignore")
            dixon_masks.append(iframe_srcdoc(html_txt, height_px=420, title="dixon_overview"))
        dixon_axial_parts.append(
            build_dixon_slice_viewer_html(
                lay,
                subject,
                margin_vox=margin_vox,
                assets_dir=assets / "slices" / "dixon",
                assets_rel=f"{rel_assets}/slices/dixon",
                review_ctx=review_ctx_dx,
            )
        )

    if "ct-pet-v5" in pipelines:
        df_ct = load_per_subject_tables(
            lay.results_dir / ct_cfg.STAGE3_DIR / "per_subject", [subject]
        )
        ct_table = dataframe_to_html_table(df_ct, EXPECTED_RANGES_CTPET)
    else:
        ct_table = "<p><em>CT-PET pipeline not selected.</em></p>"

    if "dixon-v5" in pipelines:
        df_dx = load_per_subject_tables(
            lay.results_dir / dx_cfg.STAGE3_DIR / "per_subject", [subject]
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
            [subject],
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

    # Portable: embed hotspot html via srcdoc map when possible.
    ct_map: dict[str, dict[str, str]] = {}
    dx_map: dict[str, dict[str, str]] = {}
    for s, m, rel in ct_entries:
        p = (out_dir / rel).resolve() if rel.startswith(rel_assets) else (out_dir / rel).resolve()
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = "<!doctype html><html><body><p>Hotspot missing.</p></body></html>"
        ct_map.setdefault(s, {})[m] = txt
    for s, m, rel in dx_entries:
        p = (out_dir / rel).resolve() if rel.startswith(rel_assets) else (out_dir / rel).resolve()
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = "<!doctype html><html><body><p>Hotspot missing.</p></body></html>"
        dx_map.setdefault(s, {})[m] = txt

    ct_hot = hotspot_gallery_control_srcdoc(ct_map, dom_prefix="ct")
    dx_hot = ""  # Dixon QC no longer shows hotspot section.

    wrote_ct: Path | None = None
    wrote_dx: Path | None = None

    if "ct-pet-v5" in pipelines:
        report_relpath_ct = f"{subject}/{out_ct.name}"
        header_ct = report_header_toolbar_html(
            batch=lay.batch,
            subject=subject,
            pipeline="ct-pet-v5",
            report_relpath=report_relpath_ct,
        )
        footer_ct = report_footer_toolbar_html(
            batch=lay.batch,
            subject=subject,
            pipeline="ct-pet-v5",
            report_relpath=report_relpath_ct,
        )
        ct_xlsx_href = copy_measurements_xlsx_for_qc(
            lay=lay,
            subject=subject,
            pipeline="ct-pet-v5",
            assets_dir=assets,
            rel_assets=rel_assets,
        )
        ct_download = measurements_download_button_html(ct_xlsx_href)
        html_ct = build_ctpet_report_html(
            batch=lay.batch,
            subject=subject,
            header_toolbar=header_ct,
            footer_toolbar=footer_ct,
            masks_html=ctpet_masks,
            hotspot_gallery=ct_hot,
            axial_html=ctpet_axial_parts,
            measurements_table=ct_table,
            measurements_download=ct_download,
        )
        out_ct.write_text(html_ct, encoding="utf-8")
        wrote_ct = out_ct
        log.info("CT-PET QC report written to %s", out_ct)

    if "dixon-v5" in pipelines:
        from nvitk.pipes.pesa_fat.qc.pet_axial import _safe_stem

        report_relpath_dx = f"{subject}/{out_dx.name}"
        header_dx = report_header_toolbar_html(
            batch=lay.batch,
            subject=subject,
            pipeline="dixon-v5",
            report_relpath=report_relpath_dx,
        )
        footer_dx = report_footer_toolbar_html(
            batch=lay.batch,
            subject=subject,
            pipeline="dixon-v5",
            report_relpath=report_relpath_dx,
        )
        heat = build_dixon_measurement_heatmap_html(
            lay,
            subject,
            margin_vox=margin_vox,
            assets_dir=assets / "heatmap" / "dixon",
            assets_rel=f"{rel_assets}/heatmap/dixon",
            slice_viewer_dom_id=f"dx_sv_{_safe_stem(subject)}",
        )
        extra = f'''
<div class="card">
  <div class="card-h"><h3>Measurement heatmap</h3><div class="muted">WATER underlay · colormap inside mask</div></div>
  <div class="card-b">{heat}</div>
</div>
'''.strip()
        dx_xlsx_href = copy_measurements_xlsx_for_qc(
            lay=lay,
            subject=subject,
            pipeline="dixon-v5",
            assets_dir=assets,
            rel_assets=rel_assets,
        )
        dx_download = measurements_download_button_html(dx_xlsx_href)
        html_dx = build_dixon_report_html(
            batch=lay.batch,
            subject=subject,
            header_toolbar=header_dx,
            footer_toolbar=footer_dx,
            masks_html=dixon_masks,
            hotspot_gallery=dx_hot,
            axial_html=dixon_axial_parts,
            measurements_table=dx_table,
            measurements_download=dx_download,
            extra_sections_html=extra,
        )
        out_dx.write_text(html_dx, encoding="utf-8")
        wrote_dx = out_dx
        log.info("Dixon QC report written to %s", out_dx)

    return wrote_ct, wrote_dx


def run_qc(
    batch: str,
    subjects: list[str],
    *,
    pipelines: list[str],
    nifti_root: Path | None = None,
    results_root: Path | None = None,
    margin_vox: int = 3,
) -> Path:
    """Build per-subject QC reports and a batch index; returns the index path."""
    from nvitk.pipes.pesa_fat.qc.headless import configure_headless_viz

    configure_headless_viz()
    lay = layout(
        batch,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
    )

    qc_root = lay.results_dir / RES_QC_DIR
    qc_root.mkdir(parents=True, exist_ok=True)

    links: list[tuple[str, str, str]] = []
    for subj in subjects:
        subj_dir = qc_root / subj
        out_ct, out_dx = run_qc_subject(
            lay,
            subj,
            pipelines=pipelines,
            out_dir=subj_dir,
            margin_vox=margin_vox,
        )
        href_ct = f"{subj}/{out_ct.name}" if out_ct is not None else "#"
        href_dx = f"{subj}/{out_dx.name}" if out_dx is not None else "#"
        links.append((subj, href_ct, href_dx))

    index = qc_root / "index.html"
    index.write_text(_index_html(batch, links), encoding="utf-8")
    log.info("QC index written to %s", index)
    return index


@click.command("nvitk-pesa-fat-qc")
@backend_click_option()
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
@click.option("--margin-vox", type=int, default=3, show_default=True, help="Axial crop margin (voxels).")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    subjects: str | None,
    pipelines: str,
    nifti_root: Path | None,
    results_root: Path | None,
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
        margin_vox=margin_vox,
    )


__all__ = ["main", "run_qc", "RES_QC_DIR"]


if __name__ == "__main__":
    main()
