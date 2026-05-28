"""Dixon QC: measurement heatmap inside segmentation masks.

Per-ROI axial slice stacks with a range slider (same interaction model as slice views):
- Outside masks: raw WATER in grayscale.
- Inside mask: selected map (FF or T2*) as a heatmap.
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
    _full_volume_display_range,
    _safe_stem,
)
from nvitk.segmentation.labels import get_label
from nvitk.types import Image


def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> np.ndarray:
    if len(label_ids) == 1:
        m = get_label(label_img, label_ids[0], missing="empty").data
        return to_numpy(m) > 0
    first = to_numpy(get_label(label_img, label_ids[0], missing="empty").data).copy()
    for lid in label_ids[1:]:
        extra = to_numpy(get_label(label_img, lid, missing="empty").data)
        first[extra > 0] = 1
    return first > 0


def _metric_suffix(metric: str) -> str:
    metric = str(metric).strip().upper()
    if metric not in {"FF", "T2"}:
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
    water_2d: np.ndarray,
    value_2d: np.ndarray,
    mask_2d: np.ndarray,
    title: str,
    water_vmin: float,
    water_vmax: float,
    metric_vmin: float,
    metric_vmax: float,
    cmap: str,
) -> bytes:
    water = np.asarray(water_2d, dtype=np.float64)
    val = np.asarray(value_2d, dtype=np.float64)
    mask = np.asarray(mask_2d, dtype=bool)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(water.T, cmap="gray", origin="lower", vmin=water_vmin, vmax=water_vmax, interpolation="none")
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
    metric: str,
    roi_names: list[str],
    roi_to_axial: dict[str, list[str]],
    roi_ax_mid: dict[str, int],
    default_roi: str | None = None,
) -> str:
    roi_opts = "".join(f"<option value='{_safe_stem(r)}'>{r}</option>" for r in roi_names)
    key_map = {_safe_stem(r): r for r in roi_names}
    key_js = json.dumps(key_map)

    return f"""
<div class="slice-viewer scroll-x" id="{dom_id}">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <strong>Measurement heatmap ({metric})</strong>
    <span class="muted">FF/T2 inside mask · WATER outside</span>
    <label>ROI <select id="{dom_id}_roi">{roi_opts}</select></label>
  </div>
  <div style="margin-top:10px">
    <div class="muted">Axial</div>
    <img id="{dom_id}_ax_img" style="max-width:none;width:560px;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
    <div><input type="range" id="{dom_id}_ax_r" min="0" max="0" value="0" step="1" style="width:min(560px,100%)"/></div>
  </div>
</div>
<script>
(function() {{
  var keyToRoi = {key_js};
  var axial = {json.dumps(roi_to_axial)};
  var axMid = {roi_ax_mid!r};
  var sel = document.getElementById("{dom_id}_roi");
  var axImg = document.getElementById("{dom_id}_ax_img");
  var axR = document.getElementById("{dom_id}_ax_r");

  function updRoi() {{
    var key = sel.value;
    var roi = keyToRoi[key];
    var ax = axial[roi] || [];
    axR.max = Math.max(0, ax.length - 1);
    axR.value = Math.min(axR.max, (axMid[roi] || 0));
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
  }}

  axR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var ax = axial[roi] || [];
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
  }};

  sel.onchange = updRoi;
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
    metric: str = "FF",
    cmap: str = "viridis",
    margin_vox: int = 3,
    assets_dir: Path | None = None,
    assets_rel: str | None = None,
) -> str:
    """Return embeddable HTML with per-ROI axial slice scrolling."""
    metric = str(metric).strip().upper()
    if metric not in {"FF", "T2"}:
        metric = "FF"
    suffix = _metric_suffix(metric)

    stage2_dir = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    subject_nifti = lay.subject_nifti_dir(subject)

    img_store = _SliceImageStore(
        assets_dir=assets_dir,
        assets_rel=assets_rel,
        prefix=f"dxhm_{_safe_stem(subject)}_{metric.lower()}",
    )

    roi_names: list[str] = []
    roi_to_ax: dict[str, list[str]] = {}
    roi_ax_mid: dict[str, int] = {}

    for disp, region, mask_file, label_ids in _dixon_roi_specs():
        region_u = str(region).strip().upper()
        water_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region_u}_WATER")
        val_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region_u}_{suffix}")
        mask_p = resolve_nii_optional(stage2_dir, Path(str(mask_file)).stem)
        if water_p is None or val_p is None or mask_p is None:
            continue
        try:
            water_img = imread(str(water_p), axes="XYZ")
            val_img = imread(str(val_p), axes="XYZ")
            label_img = imread(str(mask_p), axes="XYZ")
        except Exception:
            continue

        water_xyz = to_numpy(water_img.data)
        val_xyz = to_numpy(val_img.data)
        mask_xyz = _build_binary_mask(label_img, tuple(int(x) for x in label_ids))
        if water_xyz.shape != val_xyz.shape or water_xyz.shape != mask_xyz.shape:
            continue
        if not np.any(mask_xyz):
            continue

        water_vmin, water_vmax = _full_volume_display_range(water_xyz)
        metric_vmin, metric_vmax = _heatmap_metric_range(val_xyz, mask_xyz)

        coords = np.argwhere(mask_xyz)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        z0 = max(0, z0 - margin_vox)
        z1 = min(water_xyz.shape[2] - 1, z1 + margin_vox)
        z_indices = [z for z in range(z0, z1 + 1) if np.any(mask_xyz[:, :, z])]
        if not z_indices:
            continue

        ax_uris: list[str] = []
        for z in z_indices:
            png = _render_heatmap_slice_png(
                water_2d=water_xyz[:, :, z],
                value_2d=val_xyz[:, :, z],
                mask_2d=mask_xyz[:, :, z],
                title=f"{disp} axial z={z} ({metric})",
                water_vmin=water_vmin,
                water_vmax=water_vmax,
                metric_vmin=metric_vmin,
                metric_vmax=metric_vmax,
                cmap=cmap,
            )
            ax_uris.append(
                img_store.add(png, view="ax", roi=disp, tag=f"z{z:04d}")
            )
        roi_to_ax[disp] = ax_uris
        roi_ax_mid[disp] = len(ax_uris) // 2
        roi_names.append(disp)

    if not roi_names:
        return "<p><em>No Dixon heatmap assets available.</em></p>"

    return _heatmap_viewer_html(
        f"dx_hm_{_safe_stem(subject)}_{metric.lower()}",
        metric=metric,
        roi_names=roi_names,
        roi_to_axial=roi_to_ax,
        roi_ax_mid=roi_ax_mid,
        default_roi="LIVER",
    )

__all__ = ["build_dixon_measurement_heatmap_html"]
