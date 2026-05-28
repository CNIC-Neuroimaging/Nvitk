"""Dixon QC: measurement heatmap inside segmentation masks.

Generates a compact HTML widget (dropdown + PNG) where:
- Outside masks: raw WATER image in grayscale.
- Inside mask: selected map (FF or T2*) rendered as a heatmap.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvitk.io import imread
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, resolve_nii_optional
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.segmentation.labels import get_label
from nvitk.types import Image


def _png_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> np.ndarray:
    if len(label_ids) == 1:
        m = get_label(label_img, label_ids[0], missing="empty").data
        return (np.asarray(m) > 0)
    first = np.asarray(get_label(label_img, label_ids[0], missing="empty").data).copy()
    for lid in label_ids[1:]:
        extra = np.asarray(get_label(label_img, lid, missing="empty").data)
        first[extra > 0] = 1
    return (first > 0)


def _pick_slice_index(mask_xyz: np.ndarray) -> int:
    # Choose the axial slice (Z) with maximum mask area.
    if mask_xyz.ndim != 3:
        return 0
    areas = mask_xyz.sum(axis=(0, 1))
    if areas.size == 0:
        return 0
    return int(np.argmax(areas))


def _render_panel(
    *,
    water_xyz: np.ndarray,
    value_xyz: np.ndarray,
    mask_xyz: np.ndarray,
    z: int,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
) -> bytes:
    water = np.asarray(water_xyz[:, :, z], dtype=float)
    val = np.asarray(value_xyz[:, :, z], dtype=float)
    mask = np.asarray(mask_xyz[:, :, z], dtype=bool)

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.axis("off")
    ax.imshow(water.T, cmap="gray", origin="lower")

    if mask.any():
        sel = np.where(mask, val, np.nan)
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            finite = sel[np.isfinite(sel)]
            if finite.size:
                vmin_, vmax_ = np.nanpercentile(finite, (2, 98))
                vmin = float(vmin_) if np.isfinite(vmin_) else None
                vmax = float(vmax_) if np.isfinite(vmax_) else None
        ax.imshow(sel.T, cmap=cmap, origin="lower", alpha=0.85, vmin=vmin, vmax=vmax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return buf.getvalue()


def build_dixon_measurement_heatmap_html(
    lay: BatchLayout,
    subject: str,
    *,
    metric: str = "FF",
    cmap: str = "viridis",
) -> str:
    """Return an embeddable HTML widget string."""
    metric = str(metric).strip().upper()
    if metric not in {"FF", "T2"}:
        metric = "FF"

    # Map from metric to Dixon input suffix.
    suffix = "FAT_FRACTION" if metric == "FF" else "T2STAR"

    stage2_dir = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    subject_nifti = lay.subject_nifti_dir(subject)

    # Build ROI list from MEASURE_SPECS, grouped by region.
    rois: list[dict[str, Any]] = []
    for spec in dx_cfg.MEASURE_SPECS:
        roi = {
            "label": spec.prefix,
            "region": spec.region,
            "mask_file": spec.mask_file,
            "label_ids": spec.label_ids,
        }
        rois.append(roi)

    images: dict[str, str] = {}
    for roi in rois:
        region = str(roi["region"]).strip().upper()
        water_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region}_WATER")
        val_p = resolve_nii_optional(subject_nifti, f"{dx_cfg.INPUT_PREFIX}_{region}_{suffix}")
        mask_p = resolve_nii_optional(stage2_dir, Path(str(roi["mask_file"])).stem)
        if water_p is None or val_p is None or mask_p is None:
            continue
        try:
            water_img = imread(str(water_p), axes="XYZ")
            val_img = imread(str(val_p), axes="XYZ")
            label_img = imread(str(mask_p), axes="XYZ")
        except Exception:
            continue

        water_xyz = np.asarray(water_img.data)
        val_xyz = np.asarray(val_img.data)
        mask_xyz = _build_binary_mask(label_img, tuple(int(x) for x in roi["label_ids"]))
        if water_xyz.shape != val_xyz.shape or water_xyz.shape != mask_xyz.shape:
            continue
        z = _pick_slice_index(mask_xyz)
        png = _render_panel(
            water_xyz=water_xyz,
            value_xyz=val_xyz,
            mask_xyz=mask_xyz,
            z=z,
            vmin=None,
            vmax=None,
            cmap=cmap,
        )
        images[str(roi["label"])] = _png_data_uri(png)

    if not images:
        return "<p><em>No Dixon heatmap assets available.</em></p>"

    dom = f"dixon_heat_{_safe(subject)}_{metric.lower()}"
    payload = json.dumps(images)
    first_key = next(iter(images.keys()))
    options_html = "".join(
        f"<option value='{_esc(k)}'>{_esc(k)}</option>" for k in images.keys()
    )
    return f"""
<div class=\"axial-blocks\">
  <div class=\"two-col\">
    <div>
      <div class=\"muted\">Metric</div>
      <div><strong>{metric}</strong></div>
    </div>
    <div style=\"text-align:right\">
      <label class=\"muted\" for=\"{dom}_roi\">ROI</label><br/>
      <select id=\"{dom}_roi\" style=\"padding:6px 8px;border-radius:8px;border:1px solid rgba(229,229,229,0.18);background:rgba(0,0,0,0.25);color:#fff;\">
        {options_html}
      </select>
    </div>
  </div>
  <div style=\"margin-top:10px\">
    <img id=\"{dom}_img\" src=\"{_esc(images[first_key])}\" style=\"width:100%;max-width:820px;border-radius:10px;border:1px solid rgba(229,229,229,0.18);\"/>
  </div>
</div>
<script>
(() => {{
  const lut = {payload};
  const sel = document.getElementById('{dom}_roi');
  const img = document.getElementById('{dom}_img');
  const update = () => {{
    const k = sel.value;
    if (lut[k]) img.src = lut[k];
  }};
  sel.addEventListener('change', update);
}})();
</script>
""".strip()


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = ["build_dixon_measurement_heatmap_html"]

