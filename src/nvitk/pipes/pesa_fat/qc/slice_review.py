"""Embedded per-ROI review UI for QC slice viewers + report header toolbar."""

from __future__ import annotations

import json
from typing import Iterable


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def report_header_toolbar_html(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    report_relpath: str,
) -> str:
    """Top toolbar: Mark as revised + reviewer name (right side in header)."""
    ctx = json.dumps(
        {
            "batch": batch,
            "subject": subject,
            "pipeline": pipeline,
            "report_relpath": report_relpath,
        }
    )
    dom = f"qc_hdr_{_safe(batch)}_{_safe(subject)}_{_safe(pipeline)}"
    return f"""
<div id="{dom}" style="display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap;margin-top:10px">
  <button type="button" id="{dom}_revise" class="qc-revise-btn">Mark as revised</button>
  <label class="muted" style="display:flex;align-items:center;gap:8px">Reviewer
    <input id="{dom}_reviewer" placeholder="name" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;min-width:140px"/>
  </label>
  <span class="muted" id="{dom}_status"></span>
</div>
<script>
(() => {{
  const ctx = {ctx};
  const dom = '{dom}';
  const reviewerInput = document.getElementById(dom + '_reviewer');
  const statusEl = document.getElementById(dom + '_status');
  const reviseBtn = document.getElementById(dom + '_revise');
  window.qcReviewCtx = window.qcReviewCtx || {{}};
  window.qcReviewCtx[ctx.pipeline] = ctx;
  window.qcGetReviewer = () => reviewerInput.value || '';
  fetch('/review/state?' + new URLSearchParams({{batch: ctx.batch, subject: ctx.subject, pipeline: ctx.pipeline}}))
    .then(r => r.ok ? r.json() : {{}})
    .then(st => {{ if (st.reviewer) reviewerInput.value = st.reviewer; }})
    .catch(() => {{}});
  reviseBtn.addEventListener('click', async () => {{
    reviseBtn.disabled = true;
    statusEl.textContent = 'Syncing to database...';
    try {{
      const res = await fetch('/review/sync-db', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{...ctx, reviewer: reviewerInput.value || ''}}),
      }});
      const data = await res.json().catch(() => ({{}}));
      if (!res.ok || !data.ok) {{
        statusEl.textContent = 'Sync failed: ' + (data.db_error || data.error || res.status);
        return;
      }}
      statusEl.textContent = 'Report marked revised (DB synced).';
    }} catch (e) {{
      statusEl.textContent = 'Sync failed: ' + e;
    }} finally {{
      reviseBtn.disabled = false;
    }}
  }});
}})();
</script>
""".strip()


def embedded_review_panel_js(dom_id: str, review_ctx: dict) -> str:
    """JS for per-ROI status/comment panel inside a slice viewer."""
    ctx_js = json.dumps(review_ctx)
    return f"""
  const reviewCtx = {ctx_js};
  const reviewPanel = document.getElementById('{dom_id}_rv_panel');
  const reviewStatus = document.getElementById('{dom_id}_rv_status');
  const reviewSel = document.getElementById('{dom_id}_rv_qc');
  const reviewComment = document.getElementById('{dom_id}_rv_comment');
  const reviewStructLabel = document.getElementById('{dom_id}_rv_struct');
  const reviewState = {{}};
  const reviewable = new Set(reviewCtx.structures || []);
  const getReviewer = () => (window.qcGetReviewer && window.qcGetReviewer()) || '';
  function isReviewable(roi) {{
    if (!roi) return false;
    if (String(roi).toUpperCase().endsWith('_LR')) return false;
    return reviewable.has(roi);
  }}
  async function loadReviewState() {{
    const q = new URLSearchParams({{batch: reviewCtx.batch, subject: reviewCtx.subject, pipeline: reviewCtx.pipeline}});
    try {{
      const res = await fetch('/review/state?' + q.toString());
      if (!res.ok) return;
      const st = await res.json();
      Object.assign(reviewState, st.structures || {{}});
    }} catch (e) {{}}
  }}
  async function saveReview(structure) {{
    if (!isReviewable(structure)) return;
    const qc_status = reviewSel.value;
    const comment = reviewComment.value || '';
    reviewStatus.textContent = 'Saving ' + structure + '...';
    const body = {{...reviewCtx, structure, qc_status, reviewer: getReviewer(), comment}};
    const res = await fetch('/review', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(body),
    }});
    if (!res.ok) {{
      reviewStatus.textContent = 'Save failed.';
      return;
    }}
    reviewState[structure] = {{qc_status, comment}};
    reviewStatus.textContent = 'Saved ' + structure + ' → ' + qc_status;
  }}
  function showReviewForRoi(roi) {{
    if (!isReviewable(roi)) {{
      if (reviewPanel) reviewPanel.style.display = 'none';
      return;
    }}
    if (reviewPanel) reviewPanel.style.display = 'flex';
    reviewStructLabel.textContent = roi;
    const saved = reviewState[roi] || {{}};
    reviewSel.value = saved.qc_status || 'PENDING';
    reviewComment.value = saved.comment || '';
    reviewStatus.textContent = '';
  }}
  reviewSel.addEventListener('change', () => saveReview(reviewStructLabel.textContent));
  reviewComment.addEventListener('blur', () => saveReview(reviewStructLabel.textContent));
  await loadReviewState();
"""


def embedded_review_panel_html(dom_id: str) -> str:
    return f"""
    <div id="{dom_id}_rv_panel" style="grid-column:2;grid-row:2;display:flex;flex-direction:column;gap:8px;padding:10px;border:1px solid rgba(229,229,229,0.18);border-radius:10px;background:rgba(0,0,0,0.25);min-height:180px">
      <div class="muted">Review (selected ROI)</div>
      <div id="{dom_id}_rv_struct" style="font-weight:600;font-size:13px">—</div>
      <label class="muted" style="display:flex;flex-direction:column;gap:4px">Status
        <select id="{dom_id}_rv_qc" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;">
          <option value="PENDING">PENDING</option>
          <option value="OK">OK</option>
          <option value="FAIL">FAIL</option>
        </select>
      </label>
      <label class="muted" style="display:flex;flex-direction:column;gap:4px;flex:1">Comment
        <textarea id="{dom_id}_rv_comment" rows="3" placeholder="optional" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;resize:vertical;flex:1"></textarea>
      </label>
      <div class="muted" id="{dom_id}_rv_status" style="font-size:11px"></div>
    </div>
"""


def review_context(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    report_relpath: str,
    structures: Iterable[str],
) -> dict:
    from nvitk.pipes.pesa_fat.qc.review_policy import filter_review_structures

    reviewable = filter_review_structures([str(s).strip() for s in structures if str(s).strip()])
    return {
        "batch": batch,
        "subject": subject,
        "pipeline": pipeline,
        "report_relpath": report_relpath,
        "structures": reviewable,
    }


__all__ = [
    "embedded_review_panel_html",
    "embedded_review_panel_js",
    "report_header_toolbar_html",
    "review_context",
]
