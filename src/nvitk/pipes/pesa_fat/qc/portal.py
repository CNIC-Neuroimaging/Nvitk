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
    status.textContent = `Saving ${structure} → ${qc_status}...`;
    const body = {{...ctx, structure, qc_status, reviewer, comment: ''}};
    const res = await fetch('/review', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(body),
    }});
    if (!res.ok) {{
      status.textContent = `Save failed (${res.status}).`;
      return;
    }}
    status.textContent = `Saved ${structure} → ${qc_status}.`;
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


def create_qc_portal_app(*, qc_root: Path, reviews_xlsx: Path):
    """Return a FastAPI app serving QC HTML and POST /review."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(
            'QC portal requires FastAPI. Install with: pip install "fastapi>=0.110" "uvicorn>=0.27"'
        ) from exc

    app = FastAPI(title="nvitk PESA-Fat QC portal")

    qc_root = Path(qc_root).resolve()
    app.mount("/", StaticFiles(directory=str(qc_root), html=True), name="qc")

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


__all__ = [
    "ReviewRow",
    "create_qc_portal_app",
    "review_widget_html",
    "upsert_review_row_excel",
]

