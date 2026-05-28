"""PESA-Fat QC portal (static HTML + Excel-backed review endpoint)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import json

from nvitk.core.exceptions import BackendUnavailableError
from nvitk.core.logger import Logger

log = Logger()

QcStatus = Literal["PENDING", "OK", "FAIL"]


def review_widget_html(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    structures: Iterable[str],
    report_relpath: str,
) -> str:
    structs = [str(s).strip() for s in structures if str(s).strip()]
    if not structs:
        return "<p><em>No QC structures defined.</em></p>"
    dom = f"qc_review_{_safe(batch)}_{_safe(subject)}_{_safe(pipeline)}"
    payload = {
        "batch": batch,
        "subject": subject,
        "pipeline": pipeline,
        "report_relpath": report_relpath,
        "structures": structs,
    }
    return f"""
<div class="card">
  <div class="card-h"><h3>Review</h3><div class="muted">writes to reviews.xlsx via portal</div></div>
  <div class="card-b">
    <div class="muted" style="margin-bottom:8px">Reviewer: <input id="{dom}_reviewer" placeholder="name" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;"/></div>
    <div id="{dom}_rows" style="display:grid;grid-template-columns:1fr;gap:8px"></div>
    <div class="muted" id="{dom}_status" style="margin-top:10px"></div>
  </div>
</div>
<script>
(() => {{
  const ctx = {json.dumps(payload)};
  const rows = document.getElementById('{dom}_rows');
  const reviewerInput = document.getElementById('{dom}_reviewer');
  const status = document.getElementById('{dom}_status');
  const mk = (tag, attrs={{}}, txt=null) => {{
    const el = document.createElement(tag);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (txt !== null) el.textContent = txt;
    return el;
  }};
  const post = async (structure, qc_status) => {{
    const reviewer = reviewerInput.value || '';
    status.textContent = `Saving ${{structure}} → ${{qc_status}}...`;
    const body = {{...ctx, structure, qc_status, reviewer, comment: ''}};
    const res = await fetch('/review', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(body),
    }});
    if (!res.ok) {{
      status.textContent = `Save failed (${{res.status}}).`;
      return;
    }}
    status.textContent = `Saved ${{structure}} → ${{qc_status}}.`;
  }};
  const addRow = (structure) => {{
    const wrap = mk('div', {{style:'display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 10px;border:1px solid rgba(229,229,229,0.18);border-radius:10px;background:rgba(0,0,0,0.20)'}});
    wrap.appendChild(mk('div', {{}}, structure));
    const sel = mk('select', {{style:'padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;'}});
    for (const opt of ['PENDING','OK','FAIL']) {{
      sel.appendChild(mk('option', {{value: opt}}, opt));
    }}
    sel.addEventListener('change', () => post(structure, sel.value));
    wrap.appendChild(sel);
    rows.appendChild(wrap);
  }};
  for (const s of ctx.structures) addRow(s);
}})();
</script>
""".strip()


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ReviewRow:
    batch: str
    subject: str
    pipeline: str
    structure: str
    qc_status: QcStatus
    reviewer: str = ""
    reviewed_at: str = ""
    comment: str = ""
    report_relpath: str = ""


_HEADERS = [
    "batch",
    "subject",
    "pipeline",
    "structure",
    "qc_status",
    "reviewer",
    "reviewed_at",
    "comment",
    "report_relpath",
]


def upsert_review_row_excel(path: Path, row: ReviewRow) -> None:
    try:
        import openpyxl
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError('openpyxl is required for QC review Excel writes.') from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        if ws.max_row < 1:
            ws.append(_HEADERS)
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "reviews"
        ws.append(_HEADERS)

    # Map header -> column index
    header = [str(c.value).strip() if c.value is not None else "" for c in next(ws.iter_rows(min_row=1, max_row=1))]
    if header != _HEADERS:
        # normalize: rewrite header row if mismatched
        for i, h in enumerate(_HEADERS, start=1):
            ws.cell(row=1, column=i, value=h)

    def key_match(r: int) -> bool:
        vals = tuple(ws.cell(row=r, column=i).value for i in range(1, 5))
        want = (row.batch, row.subject, row.pipeline, row.structure)
        return vals == want

    target_row = None
    for r in range(2, ws.max_row + 1):
        if key_match(r):
            target_row = r
            break
    if target_row is None:
        target_row = ws.max_row + 1

    values = [
        row.batch,
        row.subject,
        row.pipeline,
        row.structure,
        row.qc_status,
        row.reviewer,
        row.reviewed_at or _utc_now_iso(),
        row.comment,
        row.report_relpath,
    ]
    for i, v in enumerate(values, start=1):
        ws.cell(row=target_row, column=i, value=v)

    wb.save(path)


def create_qc_portal_app(*, qc_root: Path, reviews_xlsx: Path, results_root: Path | None = None, default_batch: str | None = None):
    """Return a FastAPI app serving QC HTML and POST /review.

    The app serves:
    - `GET /` dashboard (when `results_root` is provided by the CLI wrapper)
    - `GET /batch/{batch}` convenience redirect to a batch QC index
    - `GET /files/...` static file server rooted at `results_root`
    - `POST /review` Excel upsert endpoint used by the embedded widgets
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(
            'QC portal requires FastAPI. Install with: pip install "fastapi>=0.110" "uvicorn>=0.27"'
        ) from exc

    app = FastAPI(title="nvitk PESA-Fat QC portal")

    qc_root = Path(qc_root).resolve()
    # Keep backward compatibility: if invoked with a single batch qc_root,
    # serve it under /qc so we can also host a dashboard at / later.
    app.mount("/qc", StaticFiles(directory=str(qc_root), html=True), name="qc")

    results_root_eff = Path(results_root).resolve() if results_root is not None else None
    if results_root_eff is not None:
        # Static file server for full RESULTS tree so relative links work.
        app.mount("/files", StaticFiles(directory=str(results_root_eff), html=True), name="files")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            html = _dashboard_html(results_root=results_root_eff, reviews_xlsx=reviews_xlsx)
            if default_batch:
                # Offer a prominent link to the default batch when provided.
                html = html.replace(
                    "<h1>nvitk PESA-Fat QC portal</h1>",
                    "<h1>nvitk PESA-Fat QC portal</h1>"
                    f"<div class='muted' style='margin-top:6px'>Default batch: "
                    f"<a href='/batch/{_esc(default_batch)}'><code>{_esc(default_batch)}</code></a></div>",
                    1,
                )
            return HTMLResponse(html)

        @app.get("/batch/{batch}")
        async def go_batch(batch: str):
            batch = str(batch).strip()
            if not batch:
                return RedirectResponse(url="/")
            return RedirectResponse(url=f"/files/{batch}/res_qc/index.html")
    else:
        # No results root context: default to qc_root index.
        @app.get("/")
        async def root_redirect():
            return RedirectResponse(url="/qc/index.html")

    @app.post("/review")
    async def post_review(payload: dict[str, Any]):
        try:
            row = ReviewRow(
                batch=str(payload.get("batch", "")).strip(),
                subject=str(payload.get("subject", "")).strip(),
                pipeline=str(payload.get("pipeline", "")).strip(),
                structure=str(payload.get("structure", "")).strip(),
                qc_status=str(payload.get("qc_status", "PENDING")).strip().upper(),  # type: ignore[arg-type]
                reviewer=str(payload.get("reviewer", "")).strip(),
                reviewed_at=_utc_now_iso(),
                comment=str(payload.get("comment", "")).strip(),
                report_relpath=str(payload.get("report_relpath", "")).strip(),
            )
            if row.qc_status not in {"PENDING", "OK", "FAIL"}:
                row = ReviewRow(**{**row.__dict__, "qc_status": "PENDING"})  # type: ignore[misc]
            upsert_review_row_excel(reviews_xlsx, row)
            return JSONResponse({"ok": True})
        except Exception as exc:
            log.warning("review write failed: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return app


def _read_reviews_xlsx(path: Path) -> list[ReviewRow]:
    """Best-effort read of the shared reviews workbook."""
    try:
        import openpyxl
    except Exception:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        wb = openpyxl.load_workbook(p)
    except Exception:
        return []
    ws = wb.active
    rows: list[ReviewRow] = []
    # Map header to index
    header = [str(c.value).strip() if c.value is not None else "" for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = {h: i for i, h in enumerate(header)}
    required = {"batch", "subject", "pipeline", "structure", "qc_status"}
    if not required.issubset(idx.keys()):
        return []
    for r in ws.iter_rows(min_row=2, values_only=True):
        try:
            batch = str(r[idx["batch"]] or "").strip()
            subject = str(r[idx["subject"]] or "").strip()
            pipeline = str(r[idx["pipeline"]] or "").strip()
            structure = str(r[idx["structure"]] or "").strip()
            qc_status = str(r[idx["qc_status"]] or "PENDING").strip().upper()
            reviewer = str(r[idx.get("reviewer", -1)] or "").strip() if "reviewer" in idx else ""
            reviewed_at = str(r[idx.get("reviewed_at", -1)] or "").strip() if "reviewed_at" in idx else ""
            comment = str(r[idx.get("comment", -1)] or "").strip() if "comment" in idx else ""
            report_relpath = str(r[idx.get("report_relpath", -1)] or "").strip() if "report_relpath" in idx else ""
        except Exception:
            continue
        if not batch or not subject or not pipeline or not structure:
            continue
        if qc_status not in {"PENDING", "OK", "FAIL"}:
            qc_status = "PENDING"
        rows.append(
            ReviewRow(
                batch=batch,
                subject=subject,
                pipeline=pipeline,
                structure=structure,
                qc_status=qc_status,  # type: ignore[arg-type]
                reviewer=reviewer,
                reviewed_at=reviewed_at,
                comment=comment,
                report_relpath=report_relpath,
            )
        )
    return rows


def _overall_status(rows: list[ReviewRow]) -> str:
    """Collapse per-structure statuses into one status for a subject+pipeline."""
    if not rows:
        return "PENDING"
    statuses = {r.qc_status for r in rows}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses == {"OK"}:
        return "OK"
    return "PENDING"


def _discover_batches(results_root: Path) -> list[str]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    batches: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "res_qc" / "index.html").is_file():
            batches.append(p.name)
    return batches


def _discover_processed_subjects(results_root: Path) -> list[dict[str, str]]:
    """Return rows with links to CT/Dixon reports for any batch."""
    out: list[dict[str, str]] = []
    for batch in _discover_batches(results_root):
        qc_dir = Path(results_root) / batch / "res_qc"
        subj_dir_items = [p for p in qc_dir.iterdir() if p.is_dir()]
        for sd in sorted(subj_dir_items, key=lambda p: p.name):
            subj = sd.name
            ct = sd / f"qc_ctpet_{subj}.html"
            dx = sd / f"qc_dixon_{subj}.html"
            out.append(
                {
                    "batch": batch,
                    "subject": subj,
                    "ct_href": f"/files/{batch}/res_qc/{subj}/{ct.name}" if ct.is_file() else "",
                    "dx_href": f"/files/{batch}/res_qc/{subj}/{dx.name}" if dx.is_file() else "",
                }
            )
    return out


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dashboard_html(*, results_root: Path, reviews_xlsx: Path) -> str:
    batches = _discover_batches(results_root)
    review_rows = _read_reviews_xlsx(reviews_xlsx)

    # Index reviews by (batch, subject, pipeline)
    by_key: dict[tuple[str, str, str], list[ReviewRow]] = {}
    for r in review_rows:
        by_key.setdefault((r.batch, r.subject, r.pipeline), []).append(r)

    processed = _discover_processed_subjects(results_root)
    table_rows: list[str] = []
    for row in processed:
        b = row["batch"]
        s = row["subject"]
        ct_stat = _overall_status(by_key.get((b, s, "ct-pet-v5"), []))
        dx_stat = _overall_status(by_key.get((b, s, "dixon-v5"), []))
        ct_link = f"<a href='{_esc(row['ct_href'])}'>CT-PET</a>" if row["ct_href"] else "<span class='muted'>—</span>"
        dx_link = f"<a href='{_esc(row['dx_href'])}'>Dixon</a>" if row["dx_href"] else "<span class='muted'>—</span>"
        table_rows.append(
            "<tr>"
            f"<td><code>{_esc(b)}</code></td>"
            f"<td><code>{_esc(s)}</code></td>"
            f"<td class='st {ct_stat.lower()}'>{_esc(ct_stat)}</td>"
            f"<td class='st {dx_stat.lower()}'>{_esc(dx_stat)}</td>"
            f"<td>{ct_link} · {dx_link}</td>"
            "</tr>"
        )

    batch_links = "\n".join(
        f"<li><a href='/batch/{_esc(b)}'><code>{_esc(b)}</code></a></li>" for b in batches
    ) or "<li><em>No batches found under RESULTS root.</em></li>"

    rows_html = "\n".join(table_rows) or (
        "<tr><td colspan='5'><em>No processed subjects found yet.</em></td></tr>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>nvitk PESA-Fat QC portal</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      margin: 0; padding: 24px 20px 60px;
      background: #14213d; color: #fff;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px 0; font-size: 22px; }}
    .muted {{ color: rgba(229,229,229,0.9); font-size: 12px; }}
    .card {{
      margin: 14px 0;
      border: 1px solid rgba(229,229,229,0.18);
      background: rgba(0,0,0,0.22);
      border-radius: 14px;
      overflow: hidden;
    }}
    .card-h {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(229,229,229,0.16);
      display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
    }}
    .card-h h2 {{ margin: 0; font-size: 14px; }}
    .card-b {{ padding: 14px; }}
    a {{ color: #fca311; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .table-wrap {{ overflow: auto; border-radius: 10px; border: 1px solid rgba(229,229,229,0.18); background: rgba(0,0,0,0.25); }}
    table {{ border-collapse: collapse; width: 100%; min-width: 900px; font-size: 12px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid rgba(229,229,229,0.14); text-align: left; }}
    th {{ position: sticky; top: 0; background: rgba(20,33,61,0.92); }}
    code {{ color: #fff; }}
    .st {{ font-weight: 700; }}
    .st.ok {{ color: #7CFC9A; }}
    .st.fail {{ color: #FF6B6B; }}
    .st.pending {{ color: #FFE08A; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>nvitk PESA-Fat QC portal</h1>
    <div class="muted">Serving results root: <code>{_esc(str(results_root))}</code> · Reviews: <code>{_esc(str(reviews_xlsx))}</code></div>

    <div class="card">
      <div class="card-h"><h2>Batches</h2><div class="muted">click to open batch QC index</div></div>
      <div class="card-b"><ul style="margin:0;padding-left:18px">{batch_links}</ul></div>
    </div>

    <div class="card">
      <div class="card-h"><h2>Processed subjects (all batches)</h2><div class="muted">overall status aggregated from reviews.xlsx</div></div>
      <div class="card-b">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Batch</th>
                <th>Subject</th>
                <th>CT-PET</th>
                <th>Dixon</th>
                <th>Reports</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


__all__ = [
    "ReviewRow",
    "create_qc_portal_app",
    "review_widget_html",
    "upsert_review_row_excel",
]

