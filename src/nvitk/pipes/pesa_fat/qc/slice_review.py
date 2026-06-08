"""Embedded per-ROI review UI for QC slice viewers + report header toolbar."""

from __future__ import annotations

import json
from typing import Iterable


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _review_ctx_dict(*, batch: str, subject: str, pipeline: str, report_relpath: str) -> dict:
    return {
        "batch": batch,
        "subject": subject,
        "pipeline": pipeline,
        "report_relpath": report_relpath,
    }


def _revise_sync_js(*, dom: str, ctx_json: str, include_reviewer_setup: bool) -> str:
    reviewer_setup = ""
    if include_reviewer_setup:
        reviewer_setup = f"""
  window.qcReviewCtx = window.qcReviewCtx || {{}};
  window.qcReviewCtx[ctx.pipeline] = ctx;
  window.qcGetReviewer = () => reviewerInput.value || '';
  fetch('/review/state?' + new URLSearchParams({{batch: ctx.batch, subject: ctx.subject, pipeline: ctx.pipeline}}))
    .then(r => r.ok ? r.json() : {{}})
    .then(st => {{ if (st.reviewer) reviewerInput.value = st.reviewer; }})
    .catch(() => {{}});"""
    get_reviewer = (
        "reviewerInput.value || ''"
        if include_reviewer_setup
        else "(window.qcGetReviewer && window.qcGetReviewer()) || ''"
    )
    reviewer_decl = (
        f"const reviewerInput = document.getElementById('{dom}_reviewer');"
        if include_reviewer_setup
        else ""
    )
    return f"""
(() => {{
  const ctx = {ctx_json};
  const dom = '{dom}';
  {reviewer_decl}
  const statusEl = document.getElementById(dom + '_status');
  const reviseBtn = document.getElementById(dom + '_revise');
  {reviewer_setup}
  reviseBtn.addEventListener('click', async () => {{
    reviseBtn.disabled = true;
    statusEl.textContent = 'Syncing to database...';
    try {{
      const res = await fetch('/review/sync-db', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{...ctx, reviewer: {get_reviewer}}}),
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
""".strip()


def report_header_toolbar_html(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    report_relpath: str,
) -> str:
    """Top toolbar: Mark as revised + reviewer name (right side in header)."""
    ctx = json.dumps(_review_ctx_dict(batch=batch, subject=subject, pipeline=pipeline, report_relpath=report_relpath))
    dom = f"qc_hdr_{_safe(batch)}_{_safe(subject)}_{_safe(pipeline)}"
    sync_js = _revise_sync_js(dom=dom, ctx_json=ctx, include_reviewer_setup=True)
    return f"""
<div id="{dom}" style="display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap;margin-top:10px">
  <button type="button" id="{dom}_revise" class="qc-revise-btn">Mark as revised</button>
  <label class="muted" style="display:flex;align-items:center;gap:8px">Reviewer
    <input id="{dom}_reviewer" placeholder="name" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;min-width:140px"/>
  </label>
  <span class="muted" id="{dom}_status"></span>
</div>
<script>
{sync_js}
</script>
""".strip()


def report_footer_toolbar_html(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    report_relpath: str,
) -> str:
    """Bottom toolbar: Mark as revised (reuses header reviewer via window.qcGetReviewer)."""
    ctx = json.dumps(_review_ctx_dict(batch=batch, subject=subject, pipeline=pipeline, report_relpath=report_relpath))
    dom = f"qc_ftr_{_safe(batch)}_{_safe(subject)}_{_safe(pipeline)}"
    sync_js = _revise_sync_js(dom=dom, ctx_json=ctx, include_reviewer_setup=False)
    return f"""
<div id="{dom}" style="display:flex;align-items:center;justify-content:flex-end;gap:14px;flex-wrap:wrap">
  <button type="button" id="{dom}_revise" class="qc-revise-btn">Mark as revised</button>
  <span class="muted" id="{dom}_status"></span>
</div>
<script>
{sync_js}
</script>
""".strip()


def embedded_review_panel_js(dom_id: str, review_ctx: dict) -> str:
    """JS for per-ROI dual-aspect status/comment panel inside a slice viewer."""
    ctx_js = json.dumps(review_ctx)
    return f"""
  const reviewCtx = {ctx_js};
  const reviewPanel = document.getElementById('{dom_id}_rv_panel');
  const reviewStatus = document.getElementById('{dom_id}_rv_status');
  const reviewSegSel = document.getElementById('{dom_id}_rv_qc_seg');
  const reviewMeasSel = document.getElementById('{dom_id}_rv_qc_meas');
  const reviewComment = document.getElementById('{dom_id}_rv_comment');
  const reviewStructLabel = document.getElementById('{dom_id}_rv_struct');
  const reviewState = {{}};
  const reviewable = new Set(reviewCtx.structures || []);
  const reviewAspects = reviewCtx.aspects || ['SEGMENTATION', 'MEASUREMENT'];
  const getReviewer = () => (window.qcGetReviewer && window.qcGetReviewer()) || '';
  function isReviewable(roi) {{
    if (!roi) return false;
    if (String(roi).toUpperCase().endsWith('_LR')) return false;
    return reviewable.has(roi);
  }}
  function aspectLabel(aspect) {{
    return (reviewCtx.aspect_labels && reviewCtx.aspect_labels[aspect]) || aspect;
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
  async function saveReview(structure, review_aspect) {{
    if (!isReviewable(structure)) return;
    const sel = review_aspect === 'SEGMENTATION' ? reviewSegSel : reviewMeasSel;
    const qc_status = sel.value;
    const comment = reviewComment.value || '';
    reviewStatus.textContent = 'Saving ' + structure + ' (' + aspectLabel(review_aspect) + ')...';
    const body = {{...reviewCtx, structure, review_aspect, qc_status, reviewer: getReviewer(), comment}};
    const res = await fetch('/review', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(body),
    }});
    if (!res.ok) {{
      reviewStatus.textContent = 'Save failed.';
      return;
    }}
    reviewState[structure] = reviewState[structure] || {{}};
    reviewState[structure][review_aspect] = {{qc_status, comment}};
    reviewStatus.textContent = 'Saved ' + structure + ' (' + aspectLabel(review_aspect) + ') → ' + qc_status;
  }}
  function showReviewForRoi(roi) {{
    if (!isReviewable(roi)) {{
      if (reviewPanel) reviewPanel.style.display = 'none';
      return;
    }}
    if (reviewPanel) reviewPanel.style.display = 'flex';
    reviewStructLabel.textContent = roi;
    const saved = reviewState[roi] || {{}};
    reviewSegSel.value = (saved.SEGMENTATION && saved.SEGMENTATION.qc_status) || 'PENDING';
    reviewMeasSel.value = (saved.MEASUREMENT && saved.MEASUREMENT.qc_status) || 'PENDING';
    const comment = (saved.SEGMENTATION && saved.SEGMENTATION.comment) || (saved.MEASUREMENT && saved.MEASUREMENT.comment) || '';
    reviewComment.value = comment;
    reviewStatus.textContent = '';
  }}
  reviewSegSel.addEventListener('change', () => saveReview(reviewStructLabel.textContent, 'SEGMENTATION'));
  reviewMeasSel.addEventListener('change', () => saveReview(reviewStructLabel.textContent, 'MEASUREMENT'));
  reviewComment.addEventListener('blur', () => {{
    const structure = reviewStructLabel.textContent;
    if (!isReviewable(structure)) return;
    saveReview(structure, 'SEGMENTATION');
    saveReview(structure, 'MEASUREMENT');
  }});
  await loadReviewState();
"""


def _qc_select(dom_id: str, suffix: str) -> str:
    return f"""<select id="{dom_id}_rv_qc_{suffix}" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;">
          <option value="PENDING">PENDING</option>
          <option value="OK">OK</option>
          <option value="FAIL">FAIL</option>
        </select>"""


def embedded_review_panel_html(dom_id: str) -> str:
    from nvitk.pipes.pesa_fat.qc.review_policy import REVIEW_ASPECT_LABELS

    seg_label = REVIEW_ASPECT_LABELS["SEGMENTATION"]
    meas_label = REVIEW_ASPECT_LABELS["MEASUREMENT"]
    return f"""
    <div id="{dom_id}_rv_panel" style="grid-column:2;grid-row:2;display:flex;flex-direction:column;gap:8px;padding:10px;border:1px solid rgba(229,229,229,0.18);border-radius:10px;background:rgba(0,0,0,0.25);min-height:220px">
      <div class="muted">Review (selected ROI)</div>
      <div id="{dom_id}_rv_struct" style="font-weight:600;font-size:13px">—</div>
      <label class="muted" style="display:flex;flex-direction:column;gap:4px">{seg_label}
        {_qc_select(dom_id, "seg")}
      </label>
      <label class="muted" style="display:flex;flex-direction:column;gap:4px">{meas_label}
        {_qc_select(dom_id, "meas")}
      </label>
      <label class="muted" style="display:flex;flex-direction:column;gap:4px;flex:1">Comment
        <textarea id="{dom_id}_rv_comment" rows="2" placeholder="optional" style="padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;resize:vertical;flex:1"></textarea>
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
    from nvitk.pipes.pesa_fat.qc.review_policy import (
        REVIEW_ASPECTS,
        REVIEW_ASPECT_LABELS,
        filter_review_structures,
    )

    reviewable = filter_review_structures([str(s).strip() for s in structures if str(s).strip()])
    return {
        "batch": batch,
        "subject": subject,
        "pipeline": pipeline,
        "report_relpath": report_relpath,
        "structures": reviewable,
        "aspects": list(REVIEW_ASPECTS),
        "aspect_labels": REVIEW_ASPECT_LABELS,
    }


__all__ = [
    "embedded_review_panel_html",
    "embedded_review_panel_js",
    "report_footer_toolbar_html",
    "report_header_toolbar_html",
    "review_context",
]
