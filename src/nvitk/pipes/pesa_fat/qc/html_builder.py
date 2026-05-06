"""Assemble the main QC HTML document."""

from __future__ import annotations

_CSS = """
body { font-family: system-ui, sans-serif; margin: 1rem 2rem; max-width: 1400px; }
h1 { border-bottom: 1px solid #ccc; }
h2 { margin-top: 2rem; color: #234; }
h3 { margin-top: 1.25rem; }
section { margin-bottom: 2.5rem; }
.qc-measurements { border-collapse: collapse; font-size: 0.85rem; }
.qc-measurements th { background: #f0f4f8; position: sticky; top: 0; }
.subsection { margin: 1rem 0; padding: 0.5rem; background: #fafafa; border-left: 3px solid #468; }
iframe { background: #111; }
.axial-slider { margin: 1rem 0; }
.axial-blocks { display: flex; flex-wrap: wrap; gap: 1rem; }
"""


def build_report_html(
    *,
    batch: str,
    ctpet_masks_html: list[str],
    dixon_masks_html: list[str],
    ctpet_measurements_table: str,
    dixon_measurements_table: str,
    ctpet_axial_html: list[str],
    dixon_axial_html: list[str],
    ctpet_hotspot_gallery: str,
    dixon_hotspot_gallery: str,
) -> str:
    def block(title: str, inner: str) -> str:
        return f'<div class="subsection"><h4>{title}</h4>{inner}</div>'

    def join_iframes(parts: list[str]) -> str:
        if not parts:
            return "<p><em>No mask overview exports.</em></p>"
        return "\n".join(
            f'<iframe title="masks" style="width:100%;height:420px;border:1px solid #333" src="{p}"></iframe>'
            for p in parts
        )

    ct_masks = join_iframes(ctpet_masks_html)
    dx_masks = join_iframes(dixon_masks_html)
    ct_ax = "\n".join(ctpet_axial_html) if ctpet_axial_html else "<p><em>No axial QC.</em></p>"
    dx_ax = "\n".join(dixon_axial_html) if dixon_axial_html else "<p><em>No axial QC.</em></p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PESA-Fat QC — {batch}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>PESA-Fat QC report — batch <code>{batch}</code></h1>

<section id="ctpet">
<h2>CT-PET pipeline</h2>
{block("1. Segmentation masks (PyVista)", ct_masks)}
{block("2. Measurements vs expected ranges", ctpet_measurements_table)}
{block("3. Axial PET + ROI contour", ct_ax)}
{block("4. SUV hotspots (interactive)", ctpet_hotspot_gallery)}
</section>

<section id="dixon">
<h2>Dixon pipeline</h2>
{block("1. Segmentation masks (PyVista)", dx_masks)}
{block("2. Measurements vs expected ranges", dixon_measurements_table)}
{block("3. Axial Dixon FF + ROI contour", dx_ax)}
{block("4. Map hotspots (interactive)", dixon_hotspot_gallery)}
</section>
</body>
</html>
"""


__all__ = ["build_report_html"]
