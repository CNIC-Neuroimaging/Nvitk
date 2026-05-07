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
"""


def build_report_html(
    *,
    batch: str,
    subject: str,
    ctpet_masks_html: list[str],
    dixon_masks_html: list[str],
    ctpet_measurements_table: str,
    dixon_measurements_table: str,
    ctpet_hotspot_gallery: str,
    dixon_hotspot_gallery: str,
    ctpet_axial_html: list[str],
    dixon_axial_html: list[str],
) -> str:
    def join_iframes(parts: list[str]) -> str:
        if not parts:
            return "<p><em>No mask overview exports.</em></p>"
        out: list[str] = []
        for p in parts:
            if p.lstrip().startswith("<iframe") or "<iframe" in p:
                out.append(p)
            else:
                out.append(f'<div class="iframe-wrap"><iframe title="masks" src="{p}"></iframe></div>')
        return "\n".join(out)

    ct_masks = join_iframes(ctpet_masks_html)
    dx_masks = join_iframes(dixon_masks_html)
    ct_ax = "\n".join(ctpet_axial_html) if ctpet_axial_html else "<p><em>No slice QC.</em></p>"
    dx_ax = "\n".join(dixon_axial_html) if dixon_axial_html else "<p><em>No slice QC.</em></p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PESA-Fat QC — {batch} — {subject}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>PESA-Fat QC report</h1>
    <div class="meta">Batch <code>{batch}</code> · Subject <code>{subject}</code></div>
  </div>

<section id="ctpet">
<h2>CT-PET pipeline</h2>

<div class="card">
  <div class="card-h"><h3>Segmentation overview (3D)</h3><div class="muted">interactive</div></div>
  <div class="card-b">{ct_masks}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Hotspots</h3><div class="muted">interactive</div></div>
  <div class="card-b">{ctpet_hotspot_gallery}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Slice views</h3><div class="muted">axial (current)</div></div>
  <div class="card-b">{ct_ax}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Measurements</h3><div class="muted">out-of-range highlighted</div></div>
  <div class="card-b"><div class="table-wrap">{ctpet_measurements_table}</div></div>
</div>
</section>

<section id="dixon">
<h2>Dixon pipeline</h2>

<div class="card">
  <div class="card-h"><h3>Segmentation overview (3D)</h3><div class="muted">interactive</div></div>
  <div class="card-b">{dx_masks}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Hotspots</h3><div class="muted">interactive</div></div>
  <div class="card-b">{dixon_hotspot_gallery}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Slice views</h3><div class="muted">axial (current)</div></div>
  <div class="card-b">{dx_ax}</div>
</div>

<div class="card">
  <div class="card-h"><h3>Measurements</h3><div class="muted">out-of-range highlighted</div></div>
  <div class="card-b"><div class="table-wrap">{dixon_measurements_table}</div></div>
</div>
</section>

</div>
</body>
</html>
"""


__all__ = ["build_report_html"]
