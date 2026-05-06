"""Axial slice stacks (PET or Dixon maps) with ROI contour; slider-friendly PNGs."""

from __future__ import annotations

import io
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


def build_ctpet_axial_section_html(
    lay: BatchLayout,
    subject: str,
    rel_assets_root: str,
    out_dir: Path,
    *,
    margin_vox: int = 3,
) -> str:
    """Write PNGs under *out_dir* and return HTML fragment with one slider block per ROI."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no CT-PET stage-2 directory.</em></p>"

    pet = imread(str(resolve_nii(nifti_dir, ct_cfg.PET_STEM)), axes="XYZ")
    suv_img = suv_image(pet, pet.metadata)
    suv = to_numpy(suv_img.data)

    cache: dict[str, Image] = {}
    blocks: list[str] = []

    for disp, mask_file, label_ids in _ctpet_roi_specs():
        try:
            if mask_file not in cache:
                cache[mask_file] = _load_mask(stage2, mask_file)
            label_img = cache[mask_file]
            bin_mask = _build_binary_mask(label_img, label_ids)
            if pet.data.shape != bin_mask.data.shape:
                bin_mask = resample_mask_to_pet(bin_mask, pet)
            m = to_numpy(bin_mask.data) > 0
        except Exception:
            continue
        if not np.any(m):
            continue

        coords = np.argwhere(m)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        x0, x1 = int(coords[:, 0].min()), int(coords[:, 0].max())
        y0, y1 = int(coords[:, 1].min()), int(coords[:, 1].max())
        x0 = max(0, x0 - margin_vox)
        y0 = max(0, y0 - margin_vox)
        z0 = max(0, z0 - margin_vox)
        x1 = min(suv.shape[0] - 1, x1 + margin_vox)
        y1 = min(suv.shape[1] - 1, y1 + margin_vox)
        z1 = min(suv.shape[2] - 1, z1 + margin_vox)

        z_indices = [z for z in range(z0, z1 + 1) if np.any(m[x0 : x1 + 1, y0 : y1 + 1, z])]
        if not z_indices:
            continue

        stem = _safe_stem(f"{subject}_{disp}")
        subdir = out_dir / "axial_ctpet" / subject
        subdir.mkdir(parents=True, exist_ok=True)
        rel_paths: list[str] = []
        for z in z_indices:
            sl = suv[x0 : x1 + 1, y0 : y1 + 1, z]
            sl_m = m[x0 : x1 + 1, y0 : y1 + 1, z]
            png = _render_axial_png(sl, sl_m, title=f"{disp} z={z}")
            fn = subdir / f"{stem}_z{z}.png"
            fn.write_bytes(png)
            rel_paths.append(f"{rel_assets_root}/axial_ctpet/{subject}/{fn.name}")

        bid = f"ax_{_safe_stem(subject)}_{_safe_stem(disp)}"
        blocks.append(_slider_html(bid, disp, rel_paths, z_indices))

    if not blocks:
        return f"<p><em>{_safe_stem(subject)}: no CT-PET axial ROIs.</em></p>"
    return "<div class='axial-blocks'>" + "".join(blocks) + "</div>"


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


def build_dixon_axial_section_html(
    lay: BatchLayout,
    subject: str,
    rel_assets_root: str,
    out_dir: Path,
    *,
    margin_vox: int = 3,
) -> str:
    """Axial slices on Dixon FF map with ROI contour."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no Dixon stage-2 directory.</em></p>"

    cache_mask: dict[str, Image] = {}
    cache_vol: dict[str, np.ndarray] = {}
    blocks: list[str] = []

    for disp, region, mask_file, label_ids in _dixon_roi_specs():
        try:
            if mask_file not in cache_mask:
                cache_mask[mask_file] = _load_mask(stage2, mask_file)
            if region not in cache_vol:
                cache_vol[region] = to_numpy(_load_dixon_ff_map(nifti_dir, region).data)
            label_img = cache_mask[mask_file]
            vol = cache_vol[region]
            bin_mask = _build_binary_mask(label_img, label_ids)
            m = to_numpy(bin_mask.data) > 0
            if vol.shape != m.shape:
                continue
        except Exception:
            continue
        if not np.any(m):
            continue

        coords = np.argwhere(m)
        z0, z1 = int(coords[:, 2].min()), int(coords[:, 2].max())
        x0, x1 = int(coords[:, 0].min()), int(coords[:, 0].max())
        y0, y1 = int(coords[:, 1].min()), int(coords[:, 1].max())
        x0 = max(0, x0 - margin_vox)
        y0 = max(0, y0 - margin_vox)
        z0 = max(0, z0 - margin_vox)
        x1 = min(vol.shape[0] - 1, x1 + margin_vox)
        y1 = min(vol.shape[1] - 1, y1 + margin_vox)
        z1 = min(vol.shape[2] - 1, z1 + margin_vox)

        z_indices = [z for z in range(z0, z1 + 1) if np.any(m[x0 : x1 + 1, y0 : y1 + 1, z])]
        if not z_indices:
            continue

        stem = _safe_stem(f"{subject}_{disp}")
        subdir = out_dir / "axial_dixon" / subject
        subdir.mkdir(parents=True, exist_ok=True)
        rel_paths: list[str] = []
        for z in z_indices:
            sl = vol[x0 : x1 + 1, y0 : y1 + 1, z]
            sl_m = m[x0 : x1 + 1, y0 : y1 + 1, z]
            png = _render_axial_png(sl, sl_m, title=f"{disp} z={z} (FF)")
            fn = subdir / f"{stem}_z{z}.png"
            fn.write_bytes(png)
            rel_paths.append(f"{rel_assets_root}/axial_dixon/{subject}/{fn.name}")

        bid = f"axd_{_safe_stem(subject)}_{_safe_stem(disp)}"
        blocks.append(_slider_html(bid, disp, rel_paths, z_indices))

    if not blocks:
        return f"<p><em>{_safe_stem(subject)}: no Dixon axial ROIs.</em></p>"
    return "<div class='axial-blocks'>" + "".join(blocks) + "</div>"


def _slider_html(block_id: str, title: str, rel_png_urls: list[str], z_vals: list[int]) -> str:
    """Slider over precomputed PNG paths (relative to report HTML)."""
    if not rel_png_urls:
        return ""
    # Embed first image inline fallback; slider swaps ``src`` to sibling paths.
    n = len(rel_png_urls)
    zjs = ",".join(str(z) for z in z_vals)
    return f"""<div class="axial-slider" id="{block_id}">
<h4>{title}</h4>
<img id="{block_id}_img" src="{rel_png_urls[0]}" alt="{title}" style="max-width:420px;border:1px solid #333"/>
<div><label>Slice z <input type="range" id="{block_id}_r" min="0" max="{n-1}" value="0" step="1"/></label>
<span id="{block_id}_zlab">{z_vals[0]}</span></div>
<script>
(function() {{
  var paths = [{", ".join(repr(p) for p in rel_png_urls)}];
  var zs = [{zjs}];
  var r = document.getElementById("{block_id}_r");
  var im = document.getElementById("{block_id}_img");
  var lab = document.getElementById("{block_id}_zlab");
  r.oninput = function() {{
    var i = parseInt(r.value, 10);
    im.src = paths[i];
    lab.textContent = zs[i];
  }};
}})();
</script>
</div>
"""


__all__ = [
    "build_ctpet_axial_section_html",
    "build_dixon_axial_section_html",
]
