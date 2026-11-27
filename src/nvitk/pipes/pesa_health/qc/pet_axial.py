"""Slice viewers (axial + sagittal) with ROI dropdown.

v2: compact widget per pipeline (subject), embedding PNGs as data URIs.
"""

from __future__ import annotations

import io
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.io import imread
from nvitk.measure.suv import suv_image
from nvitk.pipes.pesa_health.common.paths import BatchLayout, resolve_nii, resolve_nii_optional
from nvitk.pipes.pesa_health.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_health.ct_pet_v5.labels import (
    BODY_LABELS,
    FAT_BATCH_LABELS,
    FAT_IM_LABELS,
    FAT_LABELS,
    MO_LABELS,
    MUSCLES_LABELS,
    ORGANS_LABELS,
)
from nvitk.pipes.pesa_health.dixon_v5 import config as dx_cfg
from nvitk.pipes.pesa_health.dixon_v5.labels import HEAD_LABELS, LEGS_LABELS, THORAX_LABELS
from nvitk.segmentation.labels import get_label
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.transform.resampling import resample_mask_to_pet
from nvitk.types import Image


def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> Image:
    """Union of *label_ids* in *label_img* as a binary uint8 mask image."""
    if len(label_ids) == 1:
        return get_label(label_img, label_ids[0], missing="empty")
    first = get_label(label_img, label_ids[0], missing="empty").data.copy()
    for lid in label_ids[1:]:
        extra = get_label(label_img, lid, missing="empty").data
        first[extra > 0] = 1
    return label_img.with_data(first.astype("uint8"))


def _load_mask(stage2_dir: Path, filename: str) -> Image:
    """Load the mask NIfTI matching *filename* (suffix stripped) from *stage2_dir*."""
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return imread(str(resolve_nii(stage2_dir, stem)), axes="XYZ")


def _ctpet_roi_specs() -> list[tuple[str, str, tuple[int, ...], bool]]:
    """(display_name, mask_file, label_ids, has_measure) unique over SUV + VOL specs.

    Deduplicated by display name, so a volume read from a ``*_VOL`` twin shows
    under its SUV ROI. Volume-only ROIs (CORP, MO_L3/MO_L4, GRASA_IM_*) keep their
    own entry with *has_measure* off: no SUV is measured on them.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, tuple[int, ...], bool]] = []
    candidates = [(spec.column_prefix, spec.mask_file, spec.label_ids, True) for spec in ct_cfg.SUV_SPECS]
    candidates += [
        (
            spec.column[: -len("_VOL")] if spec.column.endswith("_VOL") else spec.column,
            spec.mask_file,
            spec.label_ids,
            False,
        )
        for spec in ct_cfg.VOL_SPECS
    ]
    for name, mask_file, label_ids, has_measure in candidates:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, mask_file, label_ids, has_measure))
    return out


def _base_mask_file(mask_file: str) -> str:
    """*mask_file* with a ``_VOL`` twin suffix removed (``"MO_VOL.nii"`` -> ``"MO.nii"``)."""
    stem, dot, ext = mask_file.partition(".")
    if stem.endswith("_VOL"):
        stem = stem[: -len("_VOL")]
    return f"{stem}{dot}{ext}"


def _safe_stem(s: str) -> str:
    """*s* with any character outside ``[A-Za-z0-9_-]`` replaced by ``_``."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


_CT_MASK_TO_LABELS: dict[str, dict[str, int]] = {
    "MO.nii": MO_LABELS,
    "FAT.nii": FAT_LABELS,
    "FAT_BATCH.nii": FAT_BATCH_LABELS,
    "FAT_IM.nii": FAT_IM_LABELS,
    "BODY.nii": BODY_LABELS,
    "ORGANS.nii": ORGANS_LABELS,
    "MUSCLES.nii": MUSCLES_LABELS,
}

_DIXON_MASK_TO_LABELS: dict[str, dict[str, int]] = {
    "HEAD.nii": HEAD_LABELS,
    "THORAX.nii": THORAX_LABELS,
    "LEGS.nii": LEGS_LABELS,
}

def _pp_label_names(mask_file: str, label_ids: tuple[int, ...], *, pipeline: str) -> list[str]:
    """Human-readable label names for *label_ids* in *mask_file* (merges L/R deltoids into one)."""
    labels_dict = (
        _CT_MASK_TO_LABELS if pipeline == "ctpet" else _DIXON_MASK_TO_LABELS
    ).get(_base_mask_file(mask_file), {})
    inv = {v: k for k, v in labels_dict.items()}
    names: list[str] = []
    for lid in label_ids:
        name = inv.get(lid)
        if not name:
            continue
        if name in ("DELTOIDES_L", "DELTOIDES_R"):
            if "DELTOIDES" not in names:
                names.append("DELTOIDES")
        else:
            names.append(name)
    return names


def _vol_mask_file(mask_file: str) -> str:
    """``_VOL`` twin name of *mask_file* (``"MO.nii"`` -> ``"MO_VOL.nii"``)."""
    stem, dot, ext = _base_mask_file(mask_file).partition(".")
    return f"{stem}_VOL{dot}{ext}"


def _load_vol_mask(stage2_dir: Path, mask_file: str, cache: dict[str, Image]) -> Image:
    """Stage-2 label image the volumes of *mask_file*'s ROIs are measured on.

    That is the ``_VOL`` twin, or the regular mask when stage 2 wrote no twin (BODY,
    FAT_IM, batches older than the twins): the same fallback stage 3 applies.
    """
    base = _base_mask_file(mask_file)
    key = f"vol:{base}"
    if key not in cache:
        twin = _vol_mask_file(base)
        stem = twin.partition(".")[0]
        cache[key] = _load_mask(stage2_dir, twin if resolve_nii_optional(stage2_dir, stem) is not None else base)
    return cache[key]


def _mask_union(*masks: np.ndarray | None) -> np.ndarray | None:
    """Union of the given boolean masks, skipping the missing ones; None if all are missing."""
    out: np.ndarray | None = None
    for mask in masks:
        if mask is not None:
            out = mask.copy() if out is None else out | mask
    return out


def _binary_on_grid(label_img: Image, label_ids: tuple[int, ...], target: Image) -> np.ndarray:
    """Union of *label_ids* in *label_img* as a boolean array on *target*'s grid."""
    bin_mask = _build_binary_mask(label_img, label_ids)
    if target.data.shape != bin_mask.data.shape:
        bin_mask = resample_mask_to_pet(bin_mask, target)
    return to_numpy(bin_mask.data) > 0


# Bone-marrow ROIs are shown from the sacrum up to L1 (stage-1 classes below), so a
# reviewer can count the vertebrae and confirm the marrow really sits in L3/L4.
_CT_SPINE_RANGE_TS: tuple[str, ...] = ("sacrum", "vertebrae_S1", "vertebrae_L1")
_DIXON_SPINE_RANGE_TS: tuple[str, ...] = ("sacrum", "vertebrae_L1")
_DIXON_BM_LABELS: frozenset[str] = frozenset({"BN_L3", "BN_L4"})


def _spine_z_range(
    seg_path: Path | None,
    task: str,
    class_names: tuple[str, ...],
    *,
    target: Image,
) -> tuple[int, int] | None:
    """Axial (Z) extent, on *target*'s grid, of the *class_names* present in the stage-1
    segmentation *seg_path*; None when the file or every class is missing."""
    if seg_path is None:
        return None
    ids: list[int] = []
    for name in class_names:
        try:
            ids.append(int(get_class_id(name, task)))
        except Exception:
            continue
    try:
        seg = imread(str(seg_path), axes="XYZ")
    except Exception:
        return None
    present = np.isin(to_numpy(seg.data), ids)
    if not ids or not present.any():
        return None
    target_shape = tuple(int(v) for v in to_numpy(target.data).shape)
    if present.shape != target_shape:
        try:
            raw_img = seg.with_data(as_backend_array(present.astype(np.uint8)))
            present = to_numpy(resample_mask_to_pet(raw_img, target).data) > 0
        except Exception:
            return None
    z_hits = np.where(np.any(present, axis=(0, 1)))[0]
    if z_hits.size == 0:
        return None
    return int(z_hits.min()), int(z_hits.max())


def _axial_block(
    m: np.ndarray,
    margin_vox: int,
    spine_z: tuple[int, int] | None = None,
) -> tuple[int, int, list[int], int]:
    """Axial block for one ROI: ``(z0, z1, axial slice indices, index of the opening slice)``.

    By default the block is the mask's Z extent plus *margin_vox* and the axial stack
    lists only the slices holding the mask. With *spine_z* the block also spans that
    range and the stack lists every slice in it, opening on the mask's middle slice.
    """
    mask_z = np.where(np.any(m, axis=(0, 1)))[0]
    z0 = max(0, int(mask_z.min()) - margin_vox)
    z1 = min(m.shape[2] - 1, int(mask_z.max()) + margin_vox)
    if spine_z is None:
        z_indices = [int(z) for z in mask_z]
        return z0, z1, z_indices, len(z_indices) // 2
    z0, z1 = min(z0, spine_z[0]), max(z1, spine_z[1])
    z_indices = list(range(z0, z1 + 1))
    return z0, z1, z_indices, z_indices.index(int(mask_z[len(mask_z) // 2]))


def _flip_lr_2d(arr: np.ndarray) -> np.ndarray:
    """Flip left-right (patient R/L) for QC slice display."""
    return np.ascontiguousarray(np.asarray(arr)[::-1, ...])


def _full_volume_display_range(vol: np.ndarray) -> tuple[float, float]:
    """2nd/98th percentile intensity window over all finite voxels of *vol* (min/max fallback)."""
    v = np.asarray(vol, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, (2, 98))
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return float(lo), float(hi)


def _display_range_block(vol: np.ndarray, mask: np.ndarray, z0: int, z1: int) -> tuple[float, float]:
    """Intensity window from voxels inside *mask* over axial block ``z0..z1``."""
    sub = np.asarray(vol)[:, :, z0 : z1 + 1]
    m = np.asarray(mask)[:, :, z0 : z1 + 1]
    sel = sub[m]
    finite = sel[np.isfinite(sel)] if sel.size else np.array([], dtype=np.float64)
    if finite.size == 0:
        return _full_volume_display_range(sub)
    lo, hi = np.nanpercentile(finite, (2, 98))
    if hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    return float(lo), float(hi)


def _figsize_for_slice(vol_2d: np.ndarray, *, rotate90: bool = False, width_in: float = 5.2) -> tuple[float, float]:
    """Keep the displayed horizontal (X) extent fixed across axial/coronal views."""
    v = np.asarray(vol_2d)
    if rotate90:
        v = np.rot90(v, k=1)
    disp_rows, disp_cols = v.T.shape
    if disp_cols <= 0:
        return (width_in, width_in)
    return (width_in, max(width_in * (disp_rows / disp_cols), 1.0))


def _figsize_sagittal_match_y(
    vol_2d: np.ndarray,
    *,
    y_height_in: float,
    rotate90: bool = True,
) -> tuple[float, float]:
    """Sagittal panel height matches axial Y extent; width follows slice aspect."""
    v = np.asarray(vol_2d)
    if rotate90:
        v = np.rot90(v, k=1)
    disp_rows, disp_cols = v.T.shape
    if disp_rows <= 0:
        return (y_height_in, y_height_in)
    return (max(y_height_in * (disp_cols / disp_rows), 1.0), y_height_in)


def _png_data_uri(png_bytes: bytes) -> str:
    """Base64 ``data:image/png`` URI for *png_bytes*."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


class _SliceImageStore:
    """Write slice PNGs to disk (preferred) or fall back to inline data URIs."""

    def __init__(
        self,
        *,
        assets_dir: Path | None,
        assets_rel: str | None,
        prefix: str,
    ) -> None:
        """Store slice PNGs under *assets_dir* (URLs built from *assets_rel*) or inline if either
        is None; *prefix* namespaces filenames for this viewer instance."""
        self.assets_dir = assets_dir
        self.assets_rel = assets_rel.rstrip("/") if assets_rel else None
        self.prefix = _safe_stem(prefix)

    def add(self, png_bytes: bytes, *, view: str, roi: str, tag: str) -> str:
        """Store one slice PNG and return its URL (relative asset path) or inline data URI."""
        if self.assets_dir is None or self.assets_rel is None:
            return _png_data_uri(png_bytes)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.prefix}_{_safe_stem(roi)}_{view}_{tag}.png"
        (self.assets_dir / fname).write_bytes(png_bytes)
        return f"{self.assets_rel}/{fname}"


def _render_slice_png(
    vol_2d: np.ndarray,
    mask_meas_2d: np.ndarray | None,
    *,
    mask_vol_2d: np.ndarray | None = None,
    title: str,
    vmin: float,
    vmax: float,
    rotate90: bool = False,
    figsize: tuple[float, float] | None = None,
) -> bytes:
    """Render one slice to PNG bytes with the post-processed mask contours overlaid: the
    volumetry mask in blue and, on top, the measure mask in red."""
    if figsize is None:
        figsize = _figsize_for_slice(vol_2d, rotate90=rotate90)
    fig, ax = plt.subplots(figsize=figsize)
    v = np.asarray(vol_2d, dtype=np.float64)
    v = _flip_lr_2d(v)
    if rotate90:
        v = np.rot90(v, k=1)
    ax.imshow(v.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
    for mask_2d, color in ((mask_vol_2d, "blue"), (mask_meas_2d, "red")):
        if mask_2d is None:
            continue
        m2d = _flip_lr_2d(np.asarray(mask_2d) > 0)
        if rotate90:
            m2d = np.rot90(m2d, k=1)
        if np.any(m2d):
            ax.contour(
                m2d.T.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=1.0,
                origin="lower",
            )
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
    review_ctx: dict | None = None,
    sync_peer_dom_ids: list[str] | None = None,
) -> str:
    """Self-contained HTML/JS slice-viewer widget (ROI dropdown, axial/coronal/sagittal panels,
    optional review panel and cross-view sync) for one subject/pipeline."""
    from nvitk.pipes.pesa_health.qc.slice_review import embedded_review_panel_html, embedded_review_panel_js

    def js_map(d: dict[str, list[str]]) -> str:
        """JSON-encode *d* for inlining into the generated JavaScript."""
        return json.dumps(d)

    roi_opts = "".join(f"<option value='{_safe_stem(r)}'>{r}</option>" for r in roi_names)
    key_map = {_safe_stem(r): r for r in roi_names}
    key_js = json.dumps(key_map)
    sync_peers_js = json.dumps(sync_peer_dom_ids or [])
    sync_note = ""
    if sync_peer_dom_ids:
        sync_note = (
            '<span class="muted" style="font-size:11px">synced across CT / PET views</span>'
        )
    review_panel = embedded_review_panel_html(dom_id) if review_ctx else ""
    review_js = embedded_review_panel_js(dom_id, review_ctx) if review_ctx else ""
    fn_open = "(async function() {" if review_ctx else "(function() {"
    fn_close = "})();"

    return f"""
<div class="slice-viewer" id="{dom_id}">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
    <strong>{title}</strong>
    <label>ROI <select id="{dom_id}_roi">{roi_opts}</select></label>
    {sync_note}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:12px;margin-top:10px;align-items:start">
    <div style="grid-column:1;grid-row:1">
      <div class="muted">Axial</div>
      <img id="{dom_id}_ax_img" style="max-width:560px;width:100%;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_ax_r" min="0" max="0" value="0" step="1"/></div>
    </div>
    <div style="grid-column:2;grid-row:1">
      <div class="muted">Sagittal</div>
      <img id="{dom_id}_sg_img" style="max-width:560px;width:100%;max-height:420px;object-fit:contain;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_sg_r" min="0" max="0" value="0" step="1"/></div>
    </div>
    <div style="grid-column:1;grid-row:2">
      <div class="muted">Coronal</div>
      <img id="{dom_id}_co_img" style="max-width:560px;width:100%;border-radius:10px;border:1px solid rgba(255,255,255,0.15)"/>
      <div><input type="range" id="{dom_id}_co_r" min="0" max="0" value="0" step="1"/></div>
    </div>
    {review_panel}
  </div>
</div>
<script>
{fn_open}
  var keyToRoi = {key_js};
  var axial = {js_map(roi_to_axial)};
  var cor = {js_map(roi_to_cor)};
  var sag = {js_map(roi_to_sag)};
  var axMid = {roi_to_ax_mid!r};
  var coMid = {roi_to_cor_mid!r};
  var sgMid = {roi_to_sag_mid!r};
  var syncPeers = {sync_peers_js};
  var syncing = false;
  var sel = document.getElementById("{dom_id}_roi");
  var axImg = document.getElementById("{dom_id}_ax_img");
  var coImg = document.getElementById("{dom_id}_co_img");
  var sgImg = document.getElementById("{dom_id}_sg_img");
  var axR = document.getElementById("{dom_id}_ax_r");
  var coR = document.getElementById("{dom_id}_co_r");
  var sgR = document.getElementById("{dom_id}_sg_r");
  {review_js}

  function mapIdx(srcIdx, srcMax, tgtMax) {{
    srcMax = parseInt(srcMax, 10) || 0;
    tgtMax = parseInt(tgtMax, 10) || 0;
    if (tgtMax <= 0) return 0;
    if (srcMax <= 0) return Math.min(parseInt(srcIdx, 10) || 0, tgtMax);
    return Math.round((parseInt(srcIdx, 10) / srcMax) * tgtMax);
  }}

  function showSlices(roi, axI, coI, sgI) {{
    var ax = axial[roi] || [];
    var co = cor[roi] || [];
    var sg = sag[roi] || [];
    axR.max = Math.max(0, ax.length - 1);
    coR.max = Math.max(0, co.length - 1);
    sgR.max = Math.max(0, sg.length - 1);
    axR.value = Math.min(axI, axR.max);
    coR.value = Math.min(coI, coR.max);
    sgR.value = Math.min(sgI, sgR.max);
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
    coImg.src = co.length ? co[parseInt(coR.value, 10)] : "";
    sgImg.src = sg.length ? sg[parseInt(sgR.value, 10)] : "";
  }}

  function emitState() {{
    if (syncing) return;
    var roi = keyToRoi[sel.value];
    var detail = {{
      domId: '{dom_id}',
      roi: roi,
      axIdx: parseInt(axR.value, 10),
      coIdx: parseInt(coR.value, 10),
      sgIdx: parseInt(sgR.value, 10),
      axMax: parseInt(axR.max, 10),
      coMax: parseInt(coR.max, 10),
      sgMax: parseInt(sgR.max, 10)
    }};
    if (syncPeers.length) {{
      document.dispatchEvent(new CustomEvent('qcSliceViewerState', {{ detail: detail }}));
    }}
    document.dispatchEvent(new CustomEvent('qcSliceViewerAxial', {{
      detail: {{ domId: '{dom_id}', roi: roi, sliceIdx: detail.axIdx }}
    }}));
  }}

  function applyPeerState(d) {{
    if (!d || !d.roi || d.domId === '{dom_id}') return;
    if (!syncPeers.length) return;
    if (syncPeers.indexOf(d.domId) < 0) return;
    syncing = true;
    for (var k in keyToRoi) {{
      if (keyToRoi[k] === d.roi) {{ sel.value = k; break; }}
    }}
    var roi = d.roi;
    var axI = mapIdx(d.axIdx, d.axMax, Math.max(0, (axial[roi] || []).length - 1));
    var coI = mapIdx(d.coIdx, d.coMax, Math.max(0, (cor[roi] || []).length - 1));
    var sgI = mapIdx(d.sgIdx, d.sgMax, Math.max(0, (sag[roi] || []).length - 1));
    showSlices(roi, axI, coI, sgI);
    if (typeof showReviewForRoi === 'function') showReviewForRoi(roi);
    syncing = false;
  }}

  document.addEventListener('qcSliceViewerState', function(ev) {{
    applyPeerState(ev.detail);
  }});

  function updRoi() {{
    var key = sel.value;
    var roi = keyToRoi[key];
    showSlices(roi, axMid[roi] || 0, coMid[roi] || 0, sgMid[roi] || 0);
    if (typeof showReviewForRoi === 'function') showReviewForRoi(roi);
    emitState();
  }}
  axR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var ax = axial[roi] || [];
    axImg.src = ax.length ? ax[parseInt(axR.value, 10)] : "";
    emitState();
  }};
  coR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var co = cor[roi] || [];
    coImg.src = co.length ? co[parseInt(coR.value, 10)] : "";
    emitState();
  }};
  sgR.oninput = function() {{
    var roi = keyToRoi[sel.value];
    var sg = sag[roi] || [];
    sgImg.src = sg.length ? sg[parseInt(sgR.value, 10)] : "";
    emitState();
  }};
  sel.onchange = updRoi;
  var preferred = {default_roi!r};
  if (preferred && Object.values(keyToRoi).indexOf(preferred) >= 0) {{
    for (var k in keyToRoi) {{ if (keyToRoi[k] === preferred) {{ sel.value = k; break; }} }}
  }}
  updRoi();
{fn_close}
</script>
"""


#: CT display window (HU) of the CT slice views, the same for every ROI: abdomen W400/L40.
CT_ABDOMEN_WINDOW: tuple[float, float] = (-160.0, 240.0)


def _build_ctpet_slice_viewer_core(
    lay: BatchLayout,
    subject: str,
    *,
    vol_arr: np.ndarray,
    target_img: Image,
    stage2: Path,
    margin_vox: int,
    img_store: _SliceImageStore,
    dom_id: str,
    title: str,
    default_roi: str = "HIGADO",
    review_ctx: dict | None = None,
    sync_peer_dom_ids: list[str] | None = None,
    window: tuple[float, float] | None = None,
) -> str:
    """Render CT-PET slice stacks for *vol_arr* with masks resampled to *target_img*.

    *window* is a fixed ``(vmin, vmax)`` display window for every ROI; without it each
    ROI's window comes from its mask's voxels.
    """
    sagittal_y_height = _figsize_for_slice(
        vol_arr[:, :, min(vol_arr.shape[2] // 2, vol_arr.shape[2] - 1)]
    )[1]

    cache: dict[str, Image] = {}
    spine_cache: dict[str, tuple[int, int] | None] = {}
    roi_names: list[str] = []
    roi_to_ax: dict[str, list[str]] = {}
    roi_to_co: dict[str, list[str]] = {}
    roi_to_sg: dict[str, list[str]] = {}
    roi_ax_mid: dict[str, int] = {}
    roi_co_mid: dict[str, int] = {}
    roi_sg_mid: dict[str, int] = {}

    for disp, mask_file, label_ids, has_measure in _ctpet_roi_specs():
        m_meas: np.ndarray | None = None
        m_vol: np.ndarray | None = None
        if has_measure:
            try:
                if mask_file not in cache:
                    cache[mask_file] = _load_mask(stage2, mask_file)
                m_meas = _binary_on_grid(cache[mask_file], label_ids, target_img)
            except Exception:
                m_meas = None
        try:
            m_vol = _binary_on_grid(_load_vol_mask(stage2, mask_file, cache), label_ids, target_img)
        except Exception:
            m_vol = None
        m = _mask_union(m_meas, m_vol)
        if m is None or not np.any(m):
            continue

        spine_z = None
        if _base_mask_file(mask_file) == "MO.nii":
            if "z" not in spine_cache:
                spine_cache["z"] = _spine_z_range(
                    resolve_nii_optional(lay.results_dir / ct_cfg.STAGE1_DIR / subject / "CT", "total"),
                    "total",
                    _CT_SPINE_RANGE_TS,
                    target=target_img,
                )
            spine_z = spine_cache["z"]
        z0, z1, z_indices, ax_mid = _axial_block(m, margin_vox, spine_z)
        # Intensity window from the measured voxels when the ROI has any.
        m_win = m_meas if m_meas is not None and np.any(m_meas) else m
        vmin, vmax = window if window is not None else _display_range_block(vol_arr, m_win, z0, z1)
        ax_uris: list[str] = []
        for z in z_indices:
            sl = vol_arr[:, :, z]
            sl_m = m_meas[:, :, z] if m_meas is not None else None
            sl_vol = m_vol[:, :, z] if m_vol is not None else None
            ax_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} axial z={z}",
                        vmin=vmin,
                        vmax=vmax,
                    ),
                    view="ax",
                    roi=disp,
                    tag=f"z{z:04d}",
                )
            )
        roi_to_ax[disp] = ax_uris
        roi_ax_mid[disp] = ax_mid

        y_indices = [y for y in range(0, vol_arr.shape[1]) if np.any(m[:, y, z0 : z1 + 1])]
        if not y_indices:
            y_indices = [vol_arr.shape[1] // 2]
        co_uris: list[str] = []
        for y in y_indices:
            sl = vol_arr[:, y, z0 : z1 + 1]
            sl_m = m_meas[:, y, z0 : z1 + 1] if m_meas is not None else None
            sl_vol = m_vol[:, y, z0 : z1 + 1] if m_vol is not None else None
            co_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} coronal y={y}",
                        vmin=vmin,
                        vmax=vmax,
                    ),
                    view="co",
                    roi=disp,
                    tag=f"y{y:04d}",
                )
            )
        roi_to_co[disp] = co_uris
        roi_co_mid[disp] = len(co_uris) // 2

        x_indices = [x for x in range(0, vol_arr.shape[0]) if np.any(m[x, :, z0 : z1 + 1])]
        sg_uris: list[str] = []
        for x in x_indices:
            sl = vol_arr[x, :, z0 : z1 + 1]
            sl_m = m_meas[x, :, z0 : z1 + 1] if m_meas is not None else None
            sl_vol = m_vol[x, :, z0 : z1 + 1] if m_vol is not None else None
            sg_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} sagittal x={x}",
                        vmin=vmin,
                        vmax=vmax,
                        rotate90=True,
                        figsize=_figsize_sagittal_match_y(sl, y_height_in=sagittal_y_height),
                    ),
                    view="sg",
                    roi=disp,
                    tag=f"x{x:04d}",
                )
            )
        roi_to_sg[disp] = sg_uris
        roi_sg_mid[disp] = len(sg_uris) // 2
        roi_names.append(disp)

    if not roi_names:
        return f"<p><em>{_safe_stem(subject)}: no CT-PET slice ROIs.</em></p>"

    return _viewer_html(
        dom_id=dom_id,
        title=title,
        roi_names=roi_names,
        roi_to_axial=roi_to_ax,
        roi_to_cor=roi_to_co,
        roi_to_sag=roi_to_sg,
        roi_to_ax_mid=roi_ax_mid,
        roi_to_cor_mid=roi_co_mid,
        roi_to_sag_mid=roi_sg_mid,
        default_roi=default_roi,
        review_ctx=review_ctx,
        sync_peer_dom_ids=sync_peer_dom_ids,
    )


def build_ctpet_slice_viewer_html(
    lay: BatchLayout,
    subject: str,
    *,
    margin_vox: int = 3,
    assets_dir: Path | None = None,
    assets_rel: str | None = None,
    review_ctx: dict | None = None,
) -> str:
    """Compact axial+sagittal viewer on CT (with mask contours)."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no CT-PET stage-2 directory.</em></p>"

    ct = imread(str(resolve_nii(nifti_dir, ct_cfg.INPUT_STEM)), axes="XYZ")
    ct_arr = to_numpy(ct.data)
    subj_key = _safe_stem(subject)
    ct_dom = f"ct_sv_{subj_key}"
    pet_dom = f"ctpet_pet_sv_{subj_key}"
    img_store = _SliceImageStore(
        assets_dir=assets_dir,
        assets_rel=assets_rel,
        prefix=f"ct_{subj_key}",
    )
    return _build_ctpet_slice_viewer_core(
        lay,
        subject,
        vol_arr=ct_arr,
        target_img=ct,
        stage2=stage2,
        margin_vox=margin_vox,
        img_store=img_store,
        dom_id=ct_dom,
        title="CT-PET slices (CT underlay; red=post-processed for measure, blue=post-processed for volumetry)",
        default_roi="HIGADO",
        review_ctx=review_ctx,
        sync_peer_dom_ids=[pet_dom],
        window=CT_ABDOMEN_WINDOW,
    )


def build_ctpet_pet_slice_viewer_html(
    lay: BatchLayout,
    subject: str,
    *,
    margin_vox: int = 3,
    assets_dir: Path | None = None,
    assets_rel: str | None = None,
) -> str:
    """Axial/coronal/sagittal viewer on PET SUV with masks resampled to the PET grid."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no CT-PET stage-2 directory.</em></p>"

    pet_path = resolve_nii_optional(nifti_dir, ct_cfg.PET_STEM)
    if pet_path is None:
        return f"<p><em>{_safe_stem(subject)}: PET volume ({ct_cfg.PET_STEM}) not found.</em></p>"

    pet = imread(str(pet_path), axes="XYZ")
    suv = suv_image(pet, pet.metadata)
    suv_arr = to_numpy(suv.data)
    subj_key = _safe_stem(subject)
    ct_dom = f"ct_sv_{subj_key}"
    pet_dom = f"ctpet_pet_sv_{subj_key}"
    img_store = _SliceImageStore(
        assets_dir=assets_dir,
        assets_rel=assets_rel,
        prefix=f"pet_{subj_key}",
    )
    return _build_ctpet_slice_viewer_core(
        lay,
        subject,
        vol_arr=suv_arr,
        target_img=pet,
        stage2=stage2,
        margin_vox=margin_vox,
        img_store=img_store,
        dom_id=pet_dom,
        title="CT-PET slices (PET SUV underlay; red=post-processed for measure, blue=post-processed for volumetry)",
        default_roi="HIGADO",
        review_ctx=None,
        sync_peer_dom_ids=[ct_dom],
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
    """Load the fat-fraction map NIfTI for *region* under *nifti_dir*."""
    stem = f"{dx_cfg.INPUT_PREFIX}_{region}_FAT_FRACTION"
    return imread(str(resolve_nii(nifti_dir, stem)), axes="XYZ")


def build_dixon_slice_viewer_html(
    lay: BatchLayout,
    subject: str,
    *,
    margin_vox: int = 3,
    assets_dir: Path | None = None,
    assets_rel: str | None = None,
    review_ctx: dict | None = None,
) -> str:
    """Compact axial+sagittal viewer on Dixon WATER (fallback FF)."""
    nifti_dir = lay.subject_nifti_dir(subject)
    stage2 = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    if not stage2.exists():
        return f"<p><em>{_safe_stem(subject)}: no Dixon stage-2 directory.</em></p>"

    img_store = _SliceImageStore(
        assets_dir=assets_dir,
        assets_rel=assets_rel,
        prefix=f"dx_{_safe_stem(subject)}",
    )

    cache_mask: dict[str, Image] = {}
    cache_water: dict[str, Image] = {}
    cache_vrange: dict[str, tuple[float, float]] = {}
    cache_sag_y_height: dict[str, float] = {}
    cache_spine_z: dict[str, tuple[int, int] | None] = {}
    roi_names: list[str] = []
    roi_to_ax: dict[str, list[str]] = {}
    roi_to_co: dict[str, list[str]] = {}
    roi_to_sg: dict[str, list[str]] = {}
    roi_ax_mid: dict[str, int] = {}
    roi_co_mid: dict[str, int] = {}
    roi_sg_mid: dict[str, int] = {}

    for disp, region, mask_file, label_ids in _dixon_roi_specs():
        m_meas: np.ndarray | None = None
        m_vol: np.ndarray | None = None
        try:
            if region not in cache_water:
                stem = f"{dx_cfg.INPUT_PREFIX}_{region}_WATER"
                cache_water[region] = imread(str(resolve_nii(nifti_dir, stem)), axes="XYZ")
                vol0 = to_numpy(cache_water[region].data)
                cache_vrange[region] = _full_volume_display_range(vol0)
                zmid = min(vol0.shape[2] // 2, vol0.shape[2] - 1)
                cache_sag_y_height[region] = _figsize_for_slice(vol0[:, :, zmid])[1]
            water_img = cache_water[region]
            vol = to_numpy(water_img.data)
            vmin, vmax = cache_vrange[region]
        except Exception:
            continue
        try:
            if mask_file not in cache_mask:
                cache_mask[mask_file] = _load_mask(stage2, mask_file)
            m_meas = to_numpy(_build_binary_mask(cache_mask[mask_file], label_ids).data) > 0
        except Exception:
            m_meas = None
        try:
            m_vol = to_numpy(
                _build_binary_mask(_load_vol_mask(stage2, mask_file, cache_mask), label_ids).data
            ) > 0
        except Exception:
            m_vol = None
        # Masks off the WATER grid cannot be overlaid on it.
        if m_meas is not None and m_meas.shape != vol.shape:
            m_meas = None
        if m_vol is not None and m_vol.shape != vol.shape:
            m_vol = None
        m = _mask_union(m_meas, m_vol)
        if m is None or not np.any(m):
            continue

        spine_z = None
        bm_names = set(_pp_label_names(mask_file, label_ids, pipeline="dixon"))
        if bm_names and bm_names <= _DIXON_BM_LABELS:
            if region not in cache_spine_z:
                stage1_dir = lay.results_dir / dx_cfg.STAGE1_DIR / subject / f"{dx_cfg.INPUT_PREFIX}_{region}"
                cache_spine_z[region] = _spine_z_range(
                    resolve_nii_optional(stage1_dir, "vertebrae_mr"),
                    "vertebrae_mr",
                    _DIXON_SPINE_RANGE_TS,
                    target=water_img,
                )
            spine_z = cache_spine_z[region]
        # Only crop in Z (show full XY with neighbors)
        z0, z1, z_indices, ax_mid = _axial_block(m, margin_vox, spine_z)

        ax_uris: list[str] = []
        for z in z_indices:
            sl = vol[:, :, z]
            sl_m = m_meas[:, :, z] if m_meas is not None else None
            sl_vol = m_vol[:, :, z] if m_vol is not None else None
            ax_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} axial z={z}",
                        vmin=vmin,
                        vmax=vmax,
                    ),
                    view="ax",
                    roi=disp,
                    tag=f"z{z:04d}",
                )
            )
        roi_to_ax[disp] = ax_uris
        roi_ax_mid[disp] = ax_mid

        # coronal slices over y
        y_indices = [y for y in range(0, vol.shape[1]) if np.any(m[:, y, z0 : z1 + 1])]
        if not y_indices:
            y_indices = [vol.shape[1] // 2]
        co_uris: list[str] = []
        for y in y_indices:
            sl = vol[:, y, z0 : z1 + 1]
            sl_m = m_meas[:, y, z0 : z1 + 1] if m_meas is not None else None
            sl_vol = m_vol[:, y, z0 : z1 + 1] if m_vol is not None else None
            co_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} coronal y={y}",
                        vmin=vmin,
                        vmax=vmax,
                    ),
                    view="co",
                    roi=disp,
                    tag=f"y{y:04d}",
                )
            )
        roi_to_co[disp] = co_uris
        roi_co_mid[disp] = len(co_uris) // 2

        x_indices = [x for x in range(0, vol.shape[0]) if np.any(m[x, :, z0 : z1 + 1])]
        sg_uris: list[str] = []
        for x in x_indices:
            sl = vol[x, :, z0 : z1 + 1]
            sl_m = m_meas[x, :, z0 : z1 + 1] if m_meas is not None else None
            sl_vol = m_vol[x, :, z0 : z1 + 1] if m_vol is not None else None
            sg_uris.append(
                img_store.add(
                    _render_slice_png(
                        sl,
                        sl_m,
                        mask_vol_2d=sl_vol,
                        title=f"{disp} sagittal x={x}",
                        vmin=vmin,
                        vmax=vmax,
                        rotate90=True,
                        figsize=_figsize_sagittal_match_y(
                            sl, y_height_in=cache_sag_y_height[region]
                        ),
                    ),
                    view="sg",
                    roi=disp,
                    tag=f"x{x:04d}",
                )
            )
        roi_to_sg[disp] = sg_uris
        roi_sg_mid[disp] = len(sg_uris) // 2
        roi_names.append(disp)

    if not roi_names:
        return f"<p><em>{_safe_stem(subject)}: no Dixon slice ROIs.</em></p>"

    return _viewer_html(
        dom_id=f"dx_sv_{_safe_stem(subject)}",
        title="Dixon slices (WATER underlay; red=post-processed for measure, blue=post-processed for volumetry)",
        roi_names=roi_names,
        roi_to_axial=roi_to_ax,
        roi_to_cor=roi_to_co,
        roi_to_sag=roi_to_sg,
        roi_to_ax_mid=roi_ax_mid,
        roi_to_cor_mid=roi_co_mid,
        roi_to_sag_mid=roi_sg_mid,
        default_roi="LIVER",
        review_ctx=review_ctx,
    )


__all__ = [
    "build_ctpet_slice_viewer_html",
    "build_ctpet_pet_slice_viewer_html",
    "build_dixon_slice_viewer_html",
]
