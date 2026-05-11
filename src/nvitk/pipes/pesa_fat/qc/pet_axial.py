"""Slice viewers (axial + sagittal) with ROI dropdown.

v2: compact widget per pipeline (subject), embedding PNGs as data URIs.
"""

from __future__ import annotations

import io
import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvitk.core.array import to_numpy
from nvitk.io import imread
from nvitk.measure.suv import suv_image
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, resolve_nii
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.segmentation.labels import get_label
from nvitk.transform.resampling import resample_mask_to_pet
from nvitk.types import Image


def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> Image:
    if len(label_ids) == 1:
        return get_label(label_img, label_ids[0], missing="empty")
    first = get_label(label_img, label_ids[0], missing="empty").data.copy()
    for lid in label_ids[1:]:
        extra = get_label(label_img, lid, missing="empty").data
        first[extra > 0] = 1
    return label_img.with_data(first.astype("uint8"))


def _load_mask(stage2_dir: Path, filename: str) -> Image:
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return imread(str(resolve_nii(stage2_dir, stem)), axes="XYZ")


def _ctpet_roi_specs() -> list[tuple[str, str, tuple[int, ...]]]:
    """(display_name, mask_file, label_ids) unique over SUV + VOL specs."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    out: list[tuple[str, str, tuple[int, ...]]] = []
    for spec in ct_cfg.SUV_SPECS:
        key = (spec.mask_file, spec.label_ids)
        if key in seen:
            continue
        seen.add(key)
        out.append((spec.column_prefix, spec.mask_file, spec.label_ids))
    for spec in ct_cfg.VOL_SPECS:
        key = (spec.mask_file, spec.label_ids)
        if key in seen:
            continue
        seen.add(key)
        name = spec.column.replace("_VOL", "")
        out.append((name, spec.mask_file, spec.label_ids))
    return out


def _safe_stem(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _render_axial_png(
    vol_2d: np.ndarray,
    mask_2d: np.ndarray,
    *,
    title: str,
) -> bytes:
    """*vol_2d* shape (nx, ny) axial slice; *mask_2d* same shape binary."""
    fig, ax = plt.subplots(figsize=(4, 4))
    v = np.asarray(vol_2d, dtype=np.float64)
    m = mask_2d > 0
    if np.any(np.isfinite(v)) and v.size:
        lo, hi = np.nanpercentile(v[m] if np.any(m) else v, (5, 95))
        if hi <= lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        im = ax.imshow(v.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="none")
    else:
        ax.imshow(np.zeros_like(v).T, cmap="gray", origin="lower", interpolation="none")
    if np.any(m):
        ax.contour(m.T.astype(float), levels=[0.5], colors=["red"], linewidths=1.0, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def _png_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_slice_png(
    vol_2d: np.ndarray,
    mask_2d: np.ndarray,
    *,
    title: str,
    rotate90: bool = False,
) -> bytes:
    # same as axial helper but shared for sagittal
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    v = np.asarray(vol_2d, dtype=np.float64)
    m = mask_2d > 0
    if rotate90:
        v = np.rot90(v, k=1)
        m = np.rot90(m, k=1)
    if np.any(np.isfinite(v)) and v.size:
        lo, hi = np.nanpercentile(v[m] if np.any(m) else v, (2, 98))
        if hi <= lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        ax.imshow(v.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="none")
    else:
        ax.imshow(np.zeros_like(v).T, cmap="gray", origin="lower", interpolation="none")
    if np.any(m):
        ax.contour(m.T.astype(float), levels=[0.5], colors=["red"], linewidths=1.0, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


def _viewer_html(
    dom_id: str,
    title: str,
    roi_names: list[str],
    roi_to_axial: dict[str, list[str]],
    roi_to_cor: dict[str, list[str]],
    roi_to_sag: dict[str, list[str]],
    roi_to_ax_mid: dict[str, int],
    roi_to_cor_mid: dict[str, int],
    roi_to_sag_mid: dict[str, int],
    default_roi: str | None = None,
) -> str:
    # Build JS object literal safely via repr (strings are data URIs).
    def js_map(d: dict[str, list[str]]) -> str:
        items = ",\n".join(f"{k!r}: {v!r}" for k, v in d.items())
        return "{\n" + items + "\n}"

    roi_opts = "".join(f"<option value='{_safe_stem(r)}'>{r}</option>" for r in roi_names)
    # Use stem as key to avoid quotes in value; keep separate label map.
    key_map = { _safe_stem(r): r for r in roi_names }
    key_items = ",\n".join(f"{k!r}: {v!r}" for k, v in key_map.items())
    key_js = "{\n" + key_items + "\n}"

    return f"""
<div class="slice-viewer" id="{dom_id}">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <strong>{title}</strong>
    <label>ROI <select id="{dom_id}_roi">{roi_opts}</select></label>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;align-items:start">
    <div style="grid-column:1">
      <div class="muted">Axial</div>
      <img id="{dom_id}_ax_img" style="max-width:560px;width:100%;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_ax_r" min="0" max="0" value="0" step="1"/></div>
    </div>
    <div style="grid-column:2">
      <div class="muted">Sagittal</div>
      <img id="{dom_id}_sg_img" style="max-width:560px;width:100%;max-height:420px;object-fit:contain;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_sg_r" min="0" max="0" value="0" step="1"/></div>
    </div>
    <div style="grid-column:1 / span 2">
      <div class="muted">Coronal</div>
      <img id="{dom_id}_co_img" style="max-width:560px;width:100%;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_co_r" min="0" max="0" value="0" step="1"/></div>
    </div>
  </div>
</div>
<script>
(function() {{
  var keyToRoi = {key_js};
  var axial = {js_map(roi_to_axial)};
  var cor = {js_map(roi_to_cor)};
  var sag = {js_map(roi_to_sag)};
  var axMid = {roi_to_ax_mid!r};
  var coMid = {roi_to_cor_mid!r};
  var sgMid = {roi_to_sag_mid!r};
  var sel = document.getElementById("{dom_id}_roi");
  var axImg = document.getElementById("{dom_id}_ax_img");
  var coImg = document.getElementById("{dom_id}_co_img");
  var sgImg = document.getElementById("{dom_id}_sg_img");
  var axR = document.getElementById("{dom_id}_ax_r");
  var coR = document.getElementById("{dom_id}_co_r");
  var sgR = document.getElementById("{dom_id}_sg_r");

  function updRoi() {{
    var key = sel.value;
    var roi = keyToRoi[key];
    var ax = axial[roi] || [];
    var co = cor[roi] || [];
    var sg = sag[roi] || [];
    axR.max = Math.max(0, ax.length - 1);
    coR.max = Math.max(0, co.length - 1);
    sgR.max = Math.max(0, sg.length - 1);
    axR.value = Math.min(axR.max, (axMid[roi] || 0));
    coR.value = Math.min(coR.max, (coMid[roi] || 0));
    sgR.value = Math.min(sgR.max, (sgMid[roi] || 0));
    axImg.src = ax.length ? ax[parseInt(axR.value,10)] : "";
    coImg.src = co.length ? co[parseInt(coR.value,10)] : "";
    sgImg.src = sg.length ? sg[parseInt(sgR.value,10)] : "";
  }}

  axR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var ax = axial[roi] || [];
    axImg.src = ax.length ? ax[parseInt(axR.value,10)] : "";
  }};
  coR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var co = cor[roi] || [];
    coImg.src = co.length ? co[parseInt(coR.value,10)] : "";
  }};
  sgR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var sg = sag[roi] || [];
    sgImg.src = sg.length ? sg[parseInt(sgR.value,10)] : "";
  }};

  sel.onchange = updRoi;
  var preferred = {default_roi!r};
  if (preferred && Object.values(keyToRoi).indexOf(preferred) >= 0) {{
    for (var k in keyToRoi) {{ if (keyToRoi[k] === preferred) {{ sel.value = k; break; }} }}
  }}
  updRoi();
}})();
</script>
"""


def build_ctpet_slice_viewer_html(
    lay: BatchLayout,
    subject: str,
    *,
    margin_vox: int = 3,
) -> str:
    """Compact axial+sagittal viewer on CT (with mask contours)."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no CT-PET stage-2 directory.</em></p>"

    ct = imread(str(resolve_nii(nifti_dir, ct_cfg.INPUT_STEM)), axes="XYZ")
    ct_arr = to_numpy(ct.data)

    cache: dict[str, Image] = {}
    roi_names: list[str] = []
    roi_to_ax: dict[str, list[str]] = {}
    roi_to_co: dict[str, list[str]] = {}
    roi_to_sg: dict[str, list[str]] = {}
    roi_ax_mid: dict[str, int] = {}
    roi_co_mid: dict[str, int] = {}
    roi_sg_mid: dict[str, int] = {}

    for disp, mask_file, label_ids in _ctpet_roi_specs():
        try:
            if mask_file not in cache:
                cache[mask_file] = _load_mask(stage2, mask_file)
            label_img = cache[mask_file]
            bin_mask = _build_binary_mask(label_img, label_ids)
            if ct.data.shape != bin_mask.data.shape:
                # masks are on CT grid for stage2, but keep this robust
                bin_mask = resample_mask_to_pet(bin_mask, ct)
            m = to_numpy(bin_mask.data) > 0
        except Exception:
            continue
        if not np.any(m):
            continue
        roi_names.append(disp)

        coords = np.argwhere(m)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        # Only crop in Z (show full XY with neighbors)
        z0 = max(0, z0 - margin_vox)
        z1 = min(ct_arr.shape[2] - 1, z1 + margin_vox)

        # axial slices over z
        z_indices = [z for z in range(z0, z1 + 1) if np.any(m[:, :, z])]
        if not z_indices:
            continue
        ax_uris: list[str] = []
        for z in z_indices:
            sl = ct_arr[:, :, z]
            sl_m = m[:, :, z]
            ax_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} axial z={z}")))
        roi_to_ax[disp] = ax_uris
        roi_ax_mid[disp] = len(ax_uris) // 2

        # coronal slices over y (X x Z plane)
        y_indices = [y for y in range(0, ct_arr.shape[1]) if np.any(m[:, y, z0 : z1 + 1])]
        if not y_indices:
            y_indices = [ct_arr.shape[1] // 2]
        co_uris: list[str] = []
        for y in y_indices:
            sl = ct_arr[:, y, z0 : z1 + 1]
            sl_m = m[:, y, z0 : z1 + 1]
            co_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} coronal y={y}")))
        roi_to_co[disp] = co_uris
        roi_co_mid[disp] = len(co_uris) // 2

        # sagittal slices over x
        x_indices = [x for x in range(0, ct_arr.shape[0]) if np.any(m[x, :, z0 : z1 + 1])]
        sg_uris: list[str] = []
        for x in x_indices:
            sl = ct_arr[x, :, z0 : z1 + 1]
            sl_m = m[x, :, z0 : z1 + 1]
            sg_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} sagittal x={x}", rotate90=True)))
        roi_to_sg[disp] = sg_uris
        roi_sg_mid[disp] = len(sg_uris) // 2

    if not roi_names:
        return f"<p><em>{_safe_stem(subject)}: no CT-PET slice ROIs.</em></p>"

    return _viewer_html(
        dom_id=f"ct_sv_{_safe_stem(subject)}",
        title="CT-PET slices (CT underlay)",
        roi_names=roi_names,
        roi_to_axial=roi_to_ax,
        roi_to_cor=roi_to_co,
        roi_to_sag=roi_to_sg,
        roi_to_ax_mid=roi_ax_mid,
        roi_to_cor_mid=roi_co_mid,
        roi_to_sag_mid=roi_sg_mid,
        default_roi="HIGADO",
    )


def _dixon_roi_specs() -> list[tuple[str, str, str, tuple[int, ...]]]:
    """(display_name, region, mask_file, label_ids)."""
    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    out: list[tuple[str, str, str, tuple[int, ...]]] = []
    for spec in dx_cfg.MEASURE_SPECS:
        key = (spec.region, spec.mask_file, spec.label_ids)
        if key in seen:
            continue
        seen.add(key)
        name = spec.prefix.replace("DIXON_", "", 1) if spec.prefix.startswith("DIXON_") else spec.prefix
        out.append((name, spec.region, spec.mask_file, spec.label_ids))
    return out


def _load_dixon_ff_map(nifti_dir: Path, region: str) -> Image:
    stem = f"{dx_cfg.INPUT_PREFIX}_{region}_FAT_FRACTION"
    return imread(str(resolve_nii(nifti_dir, stem)), axes="XYZ")


def build_dixon_slice_viewer_html(
    lay: BatchLayout,
    subject: str,
    *,
    margin_vox: int = 3,
) -> str:
    """Compact axial+sagittal viewer on Dixon WATER (fallback FF)."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no Dixon stage-2 directory.</em></p>"

    cache_mask: dict[str, Image] = {}
    cache_water: dict[str, np.ndarray] = {}
    roi_names: list[str] = []
    roi_to_ax: dict[str, list[str]] = {}
    roi_to_co: dict[str, list[str]] = {}
    roi_to_sg: dict[str, list[str]] = {}
    roi_ax_mid: dict[str, int] = {}
    roi_co_mid: dict[str, int] = {}
    roi_sg_mid: dict[str, int] = {}

    for disp, region, mask_file, label_ids in _dixon_roi_specs():
        try:
            if mask_file not in cache_mask:
                cache_mask[mask_file] = _load_mask(stage2, mask_file)
            if region not in cache_water:
                stem = f"{dx_cfg.INPUT_PREFIX}_{region}_WATER"
                cache_water[region] = to_numpy(imread(str(resolve_nii(nifti_dir, stem)), axes="XYZ").data)
            label_img = cache_mask[mask_file]
            vol = cache_water[region]
            bin_mask = _build_binary_mask(label_img, label_ids)
            m = to_numpy(bin_mask.data) > 0
            if vol.shape != m.shape:
                continue
        except Exception:
            continue
        if not np.any(m):
            continue
        roi_names.append(disp)

        coords = np.argwhere(m)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        # Only crop in Z (show full XY with neighbors)
        z0 = max(0, z0 - margin_vox)
        z1 = min(vol.shape[2] - 1, z1 + margin_vox)

        z_indices = [z for z in range(z0, z1 + 1) if np.any(m[:, :, z])]
        if not z_indices:
            continue

        ax_uris: list[str] = []
        for z in z_indices:
            sl = vol[:, :, z]
            sl_m = m[:, :, z]
            ax_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} axial z={z}")))
        roi_to_ax[disp] = ax_uris
        roi_ax_mid[disp] = len(ax_uris) // 2

        # coronal slices over y
        y_indices = [y for y in range(0, vol.shape[1]) if np.any(m[:, y, z0 : z1 + 1])]
        if not y_indices:
            y_indices = [vol.shape[1] // 2]
        co_uris: list[str] = []
        for y in y_indices:
            sl = vol[:, y, z0 : z1 + 1]
            sl_m = m[:, y, z0 : z1 + 1]
            co_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} coronal y={y}")))
        roi_to_co[disp] = co_uris
        roi_co_mid[disp] = len(co_uris) // 2

        x_indices = [x for x in range(0, vol.shape[0]) if np.any(m[x, :, z0 : z1 + 1])]
        sg_uris: list[str] = []
        for x in x_indices:
            sl = vol[x, :, z0 : z1 + 1]
            sl_m = m[x, :, z0 : z1 + 1]
            sg_uris.append(_png_data_uri(_render_slice_png(sl, sl_m, title=f"{disp} sagittal x={x}", rotate90=True)))
        roi_to_sg[disp] = sg_uris
        roi_sg_mid[disp] = len(sg_uris) // 2

    if not roi_names:
        return f"<p><em>{_safe_stem(subject)}: no Dixon slice ROIs.</em></p>"

    return _viewer_html(
        dom_id=f"dx_sv_{_safe_stem(subject)}",
        title="Dixon slices (WATER underlay)",
        roi_names=roi_names,
        roi_to_axial=roi_to_ax,
        roi_to_cor=roi_to_co,
        roi_to_sag=roi_to_sg,
        roi_to_ax_mid=roi_ax_mid,
        roi_to_cor_mid=roi_co_mid,
        roi_to_sag_mid=roi_sg_mid,
        default_roi="LIVER",
    )


__all__ = [
    "build_ctpet_slice_viewer_html",
    "build_dixon_slice_viewer_html",
]
