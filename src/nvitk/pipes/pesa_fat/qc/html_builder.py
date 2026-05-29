"""Assemble the main QC HTML document."""

from __future__ import annotations

_CSS = """
html, body { height: 100%; }
body {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  margin: 0;
  background: #14213d;
  color: #ffffff;
}
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}
.header {
  margin-bottom: 18px;
  padding: 18px 18px;
  border: 1px solid rgba(229,229,229,0.18);
  background: rgba(0,0,0,0.35);
  border-radius: 14px;
}
.header h1 { margin: 0 0 6px 0; font-size: 22px; }
.header .meta { color: rgba(229,229,229,0.90); font-size: 13px; }

section { margin-top: 18px; }
section > h2 { margin: 14px 0 10px; font-size: 18px; color: #fca311; }

.card {
  margin: 12px 0;
  border: 1px solid rgba(229,229,229,0.18);
  background: rgba(0,0,0,0.22);
  border-radius: 14px;
  overflow: hidden;
}
.card .card-h {
  padding: 12px 14px;
  border-bottom: 1px solid rgba(229,229,229,0.16);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.card .card-h h3 { margin: 0; font-size: 14px; letter-spacing: 0.2px; color: #ffffff; }
.card .card-b { padding: 14px; }
.muted { color: rgba(229,229,229,0.92); font-size: 12px; }

.iframe-wrap iframe { width: 100%; height: 420px; border: 0; background: #000000; border-radius: 10px; }
.two-col { display: grid; grid-template-columns: 1fr; gap: 12px; }
@media (min-width: 980px) { .two-col { grid-template-columns: 1fr 1fr; } }

.table-wrap { overflow: auto; border-radius: 10px; border: 1px solid rgba(229,229,229,0.18); background: rgba(0,0,0,0.25); }
.qc-measurements { border-collapse: collapse; font-size: 12px; min-width: 700px; width: 100%; }
.qc-measurements th, .qc-measurements td { padding: 6px 8px; border-bottom: 1px solid rgba(229,229,229,0.14); }
.qc-measurements th { position: sticky; top: 0; background: rgba(20,33,61,0.92); z-index: 1; text-align: left; }
.qc-measurements tr:hover td { background: rgba(252,163,17,0.10); }

.axial-blocks { display: grid; grid-template-columns: 1fr; gap: 10px; }
.scroll-x { overflow-x: auto; }
.qc-dl-btn {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid rgba(252,163,17,0.5);
  background: rgba(252,163,17,0.15);
  color: #fca311;
  font-weight: 600;
  font-size: 12px;
  text-decoration: none;
}
.qc-dl-btn:hover { background: rgba(252,163,17,0.28); text-decoration: none; }
"""


def _join_iframes(parts: list[str]) -> str:
    if not parts:
        return "<p><em>No mask overview exports.</em></p>"
    out: list[str] = []
    for p in parts:
        if p.lstrip().startswith("<iframe") or "<iframe" in p:
            out.append(p)
        else:
            out.append(f'<div class="iframe-wrap"><iframe title="masks" src="{p}"></iframe></div>')
    return "\n".join(out)


def _base_doc(*, title: str, batch: str, subject: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title} — {batch} — {subject}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{title}</h1>
    <div class="meta">Batch <code>{batch}</code> · Subject <code>{subject}</code></div>
  </div>
{body_html}
</div>
</body>
</html>
"""


def build_ctpet_report_html(
    *,
    batch: str,
    subject: str,
    review_widget: str = "",
    masks_html: list[str],
    hotspot_gallery: str,
    axial_html: list[str],
    measurements_table: str,
    measurements_download: str = "",
) -> str:
    masks = _join_iframes(masks_html)
    ax = "\n".join(axial_html) if axial_html else "<p><em>No slice QC.</em></p>"
    body = f"""
<section id="ctpet">
<h2>CT-PET pipeline</h2>

<div class="card">
  <div class="card-h"><h3>Segmentation overview (3D)</h3><div class="muted">interactive</div></div>
  <div class="card-b">{masks}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Hotspots</h3><div class="muted">interactive</div></div>
  <div class="card-b">{hotspot_gallery}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Slice views</h3><div class="muted">axial</div></div>
  <div class="card-b">{ax}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Measurements</h3><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">{measurements_download}<span class="muted">out-of-range highlighted</span></div></div>
  <div class="card-b"><div class="table-wrap">{measurements_table}</div></div>
</div>
</section>
{review_widget}
"""
    return _base_doc(title="PESA-Fat QC (CT-PET)", batch=batch, subject=subject, body_html=body)


def build_dixon_report_html(
    *,
    batch: str,
    subject: str,
    review_widget: str = "",
    masks_html: list[str],
    hotspot_gallery: str,
    axial_html: list[str],
    measurements_table: str,
    measurements_download: str = "",
    extra_sections_html: str = "",
) -> str:
    masks = _join_iframes(masks_html)
    ax = "\n".join(axial_html) if axial_html else "<p><em>No slice QC.</em></p>"
    hotspot_block = (
        f'''
<div class="card">
  <div class="card-h"><h3>Hotspots</h3><div class="muted">interactive</div></div>
  <div class="card-b">{hotspot_gallery}</div>
</div>
'''.strip()
        if str(hotspot_gallery).strip()
        else ""
    )

    body = f"""
<section id=\"dixon\">
<h2>Dixon pipeline</h2>

<div class=\"card\">
  <div class=\"card-h\"><h3>Segmentation overview (3D)</h3><div class=\"muted\">interactive</div></div>
  <div class=\"card-b\">{masks}</div>
</div>

{hotspot_block}

<div class=\"card\">
  <div class=\"card-h\"><h3>Slice views</h3><div class=\"muted\">axial</div></div>
  <div class=\"card-b\">{ax}</div>
</div>

{extra_sections_html}

<div class=\"card\">
  <div class=\"card-h\"><h3>Measurements</h3><div style=\"display:flex;align-items:center;gap:10px;flex-wrap:wrap\">{measurements_download}<span class=\"muted\">out-of-range highlighted</span></div></div>
  <div class=\"card-b\"><div class=\"table-wrap\">{measurements_table}</div></div>
</div>
</section>
{review_widget}
"""
    return _base_doc(title="PESA-Fat QC (Dixon)", batch=batch, subject=subject, body_html=body)


def build_report_html(**kwargs: object) -> str:
    """Backward-compatible wrapper for older callers (single combined page)."""
    return build_ctpet_report_html(  # type: ignore[arg-type]
        batch=str(kwargs.get("batch", "")),
        subject=str(kwargs.get("subject", "")),
        masks_html=list(kwargs.get("ctpet_masks_html") or []),
        hotspot_gallery=str(kwargs.get("ctpet_hotspot_gallery") or ""),
        axial_html=list(kwargs.get("ctpet_axial_html") or []),
        measurements_table=str(kwargs.get("ctpet_measurements_table") or ""),
    )


__all__ = [
    "build_report_html",
    "build_ctpet_report_html",
    "build_dixon_report_html",
]
