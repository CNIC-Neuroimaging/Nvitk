"""Dixon QC: measurement heatmap inside segmentation masks.

Per-ROI axial slice stacks with a range slider (same interaction model as slice views):
- Outside masks: Dixon WATER sequence (grayscale underlay, same as slice views).
- Inside mask: FF or T2 metric as a heatmap overlay.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvitk.core import to_numpy
from nvitk.io import imread
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, resolve_nii_optional
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.pipes.pesa_fat.qc.pet_axial import (
    _SliceImageStore,
    _dixon_roi_specs,
    _flip_lr_2d,
    _full_volume_display_range,
    _safe_stem,
)
from nvitk.segmentation.labels import get_label
from nvitk.types import Image

_METRICS = ("FF", "T2")
_CMAP = "viridis"


def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> np.ndarray:
    """Union of *label_ids* in *label_img* as a boolean array."""
    if len(label_ids) == 1:
        m = get_label(label_img, label_ids[0], missing="empty").data
        return to_numpy(m) > 0
    first = to_numpy(get_label(label_img, label_ids[0], missing="empty").data).copy()
    for lid in label_ids[1:]:
        extra = to_numpy(get_label(label_img, lid, missing="empty").data)
        first[extra > 0] = 1
    return first > 0


def _metric_suffix(metric: str) -> str:
    """NIfTI filename suffix for *metric* (``"FF"`` → ``"FAT_FRACTION"``, else ``"T2STAR"``)."""
    metric = str(metric).strip().upper()
    if metric not in _METRICS:
        metric = "FF"
    return "FAT_FRACTION" if metric == "FF" else "T2STAR"


def _heatmap_metric_range(value_xyz: np.ndarray, mask_xyz: np.ndarray) -> tuple[float, float]:
    """Colormap limits from masked metric voxels (2–98 percentile)."""
    sel = value_xyz[mask_xyz]
    finite = sel[np.isfinite(sel)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, (2, 98))
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return float(lo), float(hi)


def _render_heatmap_slice_png(
    *,
    base_2d: np.ndarray,
    value_2d: np.ndarray,
    mask_2d: np.ndarray,
    title: str,
    base_vmin: float,
    base_vmax: float,
    metric_vmin: float,
    metric_vmax: float,
    cmap: str,
) -> bytes:
    """Render one axial slice as PNG bytes: a grayscale WATER-sequence underlay with the metric
    heatmap composited only inside the ROI mask."""
    base = _flip_lr_2d(np.asarray(base_2d, dtype=np.float64))
    val = _flip_lr_2d(np.asarray(value_2d, dtype=np.float64))
    mask = _flip_lr_2d(np.asarray(mask_2d, dtype=bool))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(base.T, cmap="gray", origin="lower", vmin=base_vmin, vmax=base_vmax, interpolation="none")
    if mask.any():
        sel = np.where(mask, val, np.nan)
        ax.imshow(
            sel.T,
            cmap=cmap,
            origin="lower",
            alpha=0.85,
            vmin=metric_vmin,
            vmax=metric_vmax,
            interpolation="none",
        )
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def _heatmap_viewer_html(
    dom_id: str,
    *,
    roi_names: list[str],
    axial_by_metric: dict[str, dict[str, list[str]]],
    ax_mid_by_metric: dict[str, dict[str, int]],
    range_by_metric: dict[str, dict[str, list[float]]],
    default_roi: str | None = None,
    slice_viewer_dom_id: str = "",
) -> str:
    """Build the self-contained HTML/JS heatmap slice viewer widget (ROI + metric selectors, slice
    range slider, synced with the plain slice viewer if *slice_viewer_dom_id* is given)."""
    roi_opts = "".join(f"<option value='{_safe_stem(r)}'>{r}</option>" for r in roi_names)
    key_map = {_safe_stem(r): r for r in roi_names}
    key_js = json.dumps(key_map)
    axial_js = json.dumps(axial_by_metric)
    ax_mid_js = json.dumps(ax_mid_by_metric)
    range_js = json.dumps(range_by_metric)
    metric_opts = "".join(f"<option value='{m}'>{m}</option>" for m in _METRICS)

    return f"""
<div class="slice-viewer scroll-x" id="{dom_id}">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <strong>Measurement heatmap</strong>
    <label class="muted" style="display:flex;align-items:center;gap:6px">Metric
      <select id="{dom_id}_metric">{metric_opts}</select>
    </label>
    <label>ROI <select id="{dom_id}_roi">{roi_opts}</select></label>
    <label class="muted" style="display:flex;align-items:center;gap:6px">
      <input type="checkbox" id="{dom_id}_sync" /> sync views
    </label>
  </div>
  <div style="margin-top:10px;display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap">
    <div>
      <div class="muted">Axial</div>
      <img id="{dom_id}_ax_img" style="max-width:none;width:560px;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_ax_r" min="0" max="0" value="0" step="1" style="width:min(560px,100%)"/></div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:4px;min-width:48px">
      <div class="muted" style="font-size:11px">range</div>
      <div id="{dom_id}_cbar_max" style="font-size:11px;color:#fff"></div>
      <div id="{dom_id}_cbar" style="width:18px;height:220px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);background:linear-gradient(to top, #440154, #31688e, #35b779, #fde725)"></div>
      <div id="{dom_id}_cbar_min" style="font-size:11px;color:#fff"></div>
    </div>
  </div>
</div>
<script>
(function() {{
  var keyToRoi = {key_js};
  var axialByMetric = {axial_js};
  var axMidByMetric = {ax_mid_js};
  var rangeByMetric = {range_js};
  var sliceViewerDom = {json.dumps(slice_viewer_dom_id)};
  var sel = document.getElementById("{dom_id}_roi");
  var metricSel = document.getElementById("{dom_id}_metric");
  var syncCb = document.getElementById("{dom_id}_sync");
  var axImg = document.getElementById("{dom_id}_ax_img");
  var axR = document.getElementById("{dom_id}_ax_r");
  var cbarMin = document.getElementById("{dom_id}_cbar_min");
  var cbarMax = document.getElementById("{dom_id}_cbar_max");
  var syncing = false;

  function currentMetric() {{ return metricSel.value || 'FF'; }}

  function updateColorbar(roi) {{
    var metric = currentMetric();
    var rr = (rangeByMetric[metric] && rangeByMetric[metric][roi]) || [0, 1];
    cbarMin.textContent = Number(rr[0]).toFixed(3);
    cbarMax.textContent = Number(rr[1]).toFixed(3);
  }}

  function showAxial(roi, idx) {{
    var metric = currentMetric();
    var ax = (axialByMetric[metric] && axialByMetric[metric][roi]) || [];
    axR.max = Math.max(0, ax.length - 1);
    axR.value = Math.min(axR.max, idx);
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
    updateColorbar(roi);
  }}

  function updRoi(preferredIdx) {{
    var key = sel.value;
    var roi = keyToRoi[key];
    var metric = currentMetric();
    var axMid = (axMidByMetric[metric] && axMidByMetric[metric][roi]) || 0;
    showAxial(roi, preferredIdx !== undefined ? preferredIdx : axMid);
  }}

  metricSel.onchange = function() {{ updRoi(parseInt(axR.value, 10)); }};
  axR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var metric = currentMetric();
    var ax = (axialByMetric[metric] && axialByMetric[metric][roi]) || [];
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
  }};
  sel.onchange = function() {{ updRoi(); }};

  document.addEventListener('qcSliceViewerAxial', function(ev) {{
    if (!syncCb.checked || syncing) return;
    var d = ev.detail || {{}};
    if (sliceViewerDom && d.domId !== sliceViewerDom) return;
    if (!d.roi) return;
    syncing = true;
    for (var k in keyToRoi) {{
      if (keyToRoi[k] === d.roi) {{ sel.value = k; break; }}
    }}
    updRoi(typeof d.sliceIdx === 'number' ? d.sliceIdx : undefined);
    syncing = false;
  }});

  var preferred = {default_roi!r};
  if (preferred && Object.values(keyToRoi).indexOf(preferred) >= 0) {{
    for (var k in keyToRoi) {{ if (keyToRoi[k] === preferred) {{ sel.value = k; break; }} }}
  }}
  updRoi();
}})();
</script>
""".strip()


def build_dixon_measurement_heatmap_html(
    lay: BatchLayout,
    subject: str,
    *,
    cmap: str = _CMAP,
    margin_vox: int = 3,
    assets_dir: Path | None = None,
    assets_rel: str | None = None,
    slice_viewer_dom_id: str = "",
) -> str:
    """Return embeddable HTML with per-ROI axial slice scrolling for FF and T2."""
    stage2_dir = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    subject_nifti = lay.subject_nifti_dir(subject)

    roi_names: list[str] = []
    axial_by_metric: dict[str, dict[str, list[str]]] = {m: {} for m in _METRICS}
    ax_mid_by_metric: dict[str, dict[str, int]] = {m: {} for m in _METRICS}
    range_by_metric: dict[str, dict[str, list[float]]] = {m: {} for m in _METRICS}

    for disp, region, mask_file, label_ids in _dixon_roi_specs():
        region_u = str(region).strip().upper()
        mask_p = resolve_nii_optional(stage2_dir, Path(str(mask_file)).stem)
        if mask_p is None:
            continue
        try:
            label_img = imread(str(mask_p), axes="XYZ")
        except Exception:
            continue

        mask_xyz = _build_binary_mask(label_img, tuple(int(x) for x in label_ids))
        if not np.any(mask_xyz):
            continue

        coords = np.argwhere(mask_xyz)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        z0 = max(0, z0 - margin_vox)
        z1 = min(mask_xyz.shape[2] - 1, z1 + margin_vox)
        z_indices = [z for z in range(z0, z1 + 1) if np.any(mask_xyz[:, :, z])]
        if not z_indices:
            continue

        water_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region_u}_WATER")
        if water_p is None:
            continue
        try:
            water_xyz = to_numpy(imread(str(water_p), axes="XYZ").data)
        except Exception:
            continue
        if water_xyz.shape != mask_xyz.shape:
            continue
        base_vmin, base_vmax = _full_volume_display_range(water_xyz)

        roi_has_metric = False
        for metric in _METRICS:
            suffix = _metric_suffix(metric)
            val_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region_u}_{suffix}")
            if val_p is None:
                continue
            try:
                val_xyz = to_numpy(imread(str(val_p), axes="XYZ").data)
            except Exception:
                continue
            if val_xyz.shape != mask_xyz.shape:
                continue

            img_store = _SliceImageStore(
                assets_dir=assets_dir,
                assets_rel=assets_rel,
                prefix=f"dxhm_{_safe_stem(subject)}_{metric.lower()}",
            )
            metric_vmin, metric_vmax = _heatmap_metric_range(val_xyz, mask_xyz)
            ax_uris: list[str] = []
            for z in z_indices:
                png = _render_heatmap_slice_png(
                    base_2d=water_xyz[:, :, z],
                    value_2d=val_xyz[:, :, z],
                    mask_2d=mask_xyz[:, :, z],
                    title=f"{disp} axial z={z} ({metric})",
                    base_vmin=base_vmin,
                    base_vmax=base_vmax,
                    metric_vmin=metric_vmin,
                    metric_vmax=metric_vmax,
                    cmap=cmap,
                )
                ax_uris.append(img_store.add(png, view="ax", roi=disp, tag=f"z{z:04d}"))
            if not ax_uris:
                continue
            axial_by_metric[metric][disp] = ax_uris
            ax_mid_by_metric[metric][disp] = len(ax_uris) // 2
            range_by_metric[metric][disp] = [metric_vmin, metric_vmax]
            roi_has_metric = True

        if roi_has_metric and disp not in roi_names:
            roi_names.append(disp)

    if not roi_names:
        return "<p><em>No Dixon heatmap assets available.</em></p>"

    return _heatmap_viewer_html(
        f"dx_hm_{_safe_stem(subject)}",
        roi_names=roi_names,
        axial_by_metric=axial_by_metric,
        ax_mid_by_metric=ax_mid_by_metric,
        range_by_metric=range_by_metric,
        default_roi="LIVER",
        slice_viewer_dom_id=slice_viewer_dom_id,
    )


__all__ = ["build_dixon_measurement_heatmap_html"]
