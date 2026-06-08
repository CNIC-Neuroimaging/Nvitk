"""Precompute PyVista hotspot HTML exports for QC gallery."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
import base64

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.run_hotspot import (
    _ctpet_load_extra_mask_on_pet_grid,
    _list_valid_measures,
    _load_ctpet_inputs,
    _load_dixon_inputs,
    _require_pyvista,
    _resolve_measure_ctpet,
    _resolve_measure_dixon,
    _surface_from_binary,
)
from nvitk.viz import show_hotspots
from nvitk.viz.pet_hotspots import _roi_mask

log = Logger()

_GRASA_V_BATCH_SUV = "GRASA_V_BATCH_SUV"
_URETER_OVERLAY_OPACITY = 0.5


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _write_text_with_retries(path: Path, text: str, *, retries: int = 3, sleep_s: float = 1.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, int(retries) + 1):
        try:
            path.write_text(text, encoding="utf-8")
            return True
        except OSError as exc:
            log.warning(
                "write_text failed (%s) [attempt %d/%d]: %s",
                path,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(float(sleep_s))
    return False


def _ureter_on_roi_z_slices(ureter: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Keep ureter voxels only on axial slices where *roi* is non-empty."""
    ureter_bin = np.asarray(ureter) > 0
    z_has_roi = np.any(roi, axis=(0, 1))
    return ureter_bin & z_has_roi[np.newaxis, np.newaxis, :]


def _add_grasa_v_batch_ureter_overlay(pl, *, lay, subject: str, pet, mask_img, label_ids) -> None:
    """Overlay stage-2 ureter surface for GRASA_V_BATCH_SUV hotspot exports."""
    try:
        ureter_img = _ctpet_load_extra_mask_on_pet_grid(lay, subject, "ureter", pet)
    except ValidationError as exc:
        log.debug("GRASA_V_BATCH_SUV ureter overlay skipped: %s", exc)
        return
    if ureter_img is None:
        return

    mask_arr = to_numpy(mask_img.data)
    roi = _roi_mask(mask_arr, label_ids)
    ureter_bin = _ureter_on_roi_z_slices(to_numpy(ureter_img.data), roi)
    if not bool(np.any(ureter_bin)):
        log.debug("GRASA_V_BATCH_SUV ureter overlay empty after z-slice gating")
        return

    pv = _require_pyvista()
    surf = _surface_from_binary(pv, ureter_bin.astype(np.uint8))
    pl.add_mesh(
        surf,
        color="#00A6FB",
        opacity=float(_URETER_OVERLAY_OPACITY),
        show_scalar_bar=False,
    )
    pl.add_text("+ ureter", position="lower_left", font_size=10)


def _export_html_with_retries(pl, path: Path, *, retries: int = 3, sleep_s: float = 1.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, int(retries) + 1):
        try:
            pl.export_html(str(path))
            return True
        except OSError as exc:
            log.warning(
                "export_html failed (%s) [attempt %d/%d]: %s",
                path,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(float(sleep_s))
        except Exception as exc:
            log.warning("export_html failed (%s): %s", path, exc)
            break
    return False


def export_hotspot_gallery_for_batch(
    lay,
    subjects: list[str],
    out_dir: Path,
    *,
    rel_assets_root: str,
    notebook: bool = True,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    For each subject and each voxelwise measure, write ``hotspot_<subj>_<meas>.html``.

    Returns
    -------
    entries
        ``(subject, measure, rel_html_path)`` for the parent page selectors.
    errors
        Human-readable skip messages.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str, str]] = []
    errors: list[str] = []
    measures = _list_valid_measures()

    for subject in subjects:
        for measure in measures:
            ct = _resolve_measure_ctpet(measure)
            title = f"{'CT-PET' if ct else 'DIXON'} {measure} | {subject} | {lay.batch}"
            sb_title = "SUV" if ct is not None else measure.rsplit("_", 1)[-1]
            try:
                pet = None
                if ct is not None:
                    img, mask_img, label_ids, pet = _load_ctpet_inputs(lay, subject, ct)
                else:
                    dx = _resolve_measure_dixon(measure)
                    img, mask_img, label_ids = _load_dixon_inputs(lay, subject, dx)

                pl = show_hotspots(
                    img,
                    mask_img,
                    label_ids=label_ids,
                    hotspot="top_percent",
                    top_percent=1.0,
                    notebook=notebook,
                    show=False,
                    title=title,
                    auto_threshold_fallback=True,
                    allow_empty_hotspot=True,
                    scalar_bar_title=sb_title,
                )
                if ct is not None and measure == _GRASA_V_BATCH_SUV and pet is not None:
                    _add_grasa_v_batch_ureter_overlay(
                        pl,
                        lay=lay,
                        subject=subject,
                        pet=pet,
                        mask_img=mask_img,
                        label_ids=label_ids,
                    )
                fname = f"hotspot_{_safe_name(subject)}_{_safe_name(measure)}.html"
                path = out_dir / fname
                try:
                    ok = _export_html_with_retries(pl, path)
                    if not ok:
                        raise OSError(f"export_html failed for {path}")
                except Exception as exc:
                    import traceback
                    log.warning(traceback.format_exc())
                    log.warning("[%s] hotspot export failed %s: %s", subject, measure, exc)
                    _write_text_with_retries(
                        "<!DOCTYPE html><html><body><p>Hotspot HTML export failed: "
                        f"{exc!s}</p></body></html>",
                    )
                rel = f"{rel_assets_root}/{fname}"
                entries.append((subject, measure, rel))
            except (ValidationError, FileNotFoundError, OSError) as exc:
                msg = f"{subject} / {measure}: {exc}"
                log.debug(msg)
                errors.append(msg)
            except Exception as exc:
                msg = f"{subject} / {measure}: {exc}"
                log.warning(msg)
                errors.append(msg)

    return entries, errors


def hotspot_gallery_control_html(entries: list[tuple[str, str, str]], *, dom_prefix: str) -> str:
    """Two ``<select>`` elements + ``<iframe>``; measure list depends on subject.

    dom_prefix is required so CT-PET and Dixon widgets don't collide in the same page.
    """
    if not entries:
        return "<p><em>No hotspot exports generated.</em></p>"

    sub_to_meas: dict[str, dict[str, str]] = defaultdict(dict)
    for s, m, rel in entries:
        sub_to_meas[s][m] = rel
    subjects = sorted(sub_to_meas)
    first_sub = subjects[0]
    measures_for_first = sorted(sub_to_meas[first_sub].keys())
    first_meas = measures_for_first[0]
    first_rel = sub_to_meas[first_sub][first_meas]

    map_json = json.dumps({s: sub_to_meas[s] for s in subjects})
    meas_opts = "".join(
        f"<option value='{_esc(m)}'>{_esc(m)}</option>" for m in measures_for_first
    )
    subj_opts = "".join(f"<option value='{_esc(s)}'>{_esc(s)}</option>" for s in subjects)

    pid = _esc(dom_prefix)
    return f"""
<div id="hotspot-gallery">
<label>Subject <select id="{pid}_hg_sub">{subj_opts}</select></label>
<label>Measure <select id="{pid}_hg_meas">{meas_opts}</select></label>
<iframe id="{pid}_hg_frame" title="hotspot" style="width:100%;height:560px;border:1px solid rgba(255,255,255,0.15);border-radius:10px" src="{_esc(first_rel)}"></iframe>
</div>
<script>
(function() {{
  var subMap = {map_json};
  var fs = document.getElementById("{pid}_hg_frame");
  var ss = document.getElementById("{pid}_hg_sub");
  var sm = document.getElementById("{pid}_hg_meas");
  function refillMeasures() {{
    var sub = ss.value;
    var mm = subMap[sub];
    var keys = Object.keys(mm).sort();
    sm.innerHTML = keys.map(function(k) {{
      return "<option value='" + k.replace(/'/g, "&#39;") + "'>" + k.replace(/</g, "&lt;") + "</option>";
    }}).join("");
    sm.value = keys[0];
  }}
  function upd() {{
    var sub = ss.value;
    var me = sm.value;
    var url = subMap[sub] && subMap[sub][me];
    if (url) fs.src = url;
  }}
  ss.onchange = function() {{ refillMeasures(); upd(); }};
  sm.onchange = upd;
  // Ensure initial state is coherent in case browser restores state.
  refillMeasures(); upd();
}})();
</script>
"""


def hotspot_gallery_control_srcdoc(
    html_map: dict[str, dict[str, str]],
    *,
    dom_prefix: str,
) -> str:
    """Hotspot gallery where each (subject,measure) maps to an iframe srcdoc HTML string.

    NOTE: raw HTML must NOT be embedded directly into JS literals (it can contain quotes and
    even `</script>`). We base64-encode values and store them in a JSON blob.
    """
    if not html_map:
        return "<p><em>No hotspot exports generated.</em></p>"
    subjects = sorted(html_map)
    first_sub = subjects[0]
    measures = sorted(html_map[first_sub])
    if not measures:
        return "<p><em>No hotspot exports generated.</em></p>"
    first_meas = measures[0]
    pid = _esc(dom_prefix)

    # Base64 values to keep the JSON/script safe.
    b64_map: dict[str, dict[str, str]] = {}
    for s, mm in html_map.items():
        b64_map[s] = {}
        for m, html in mm.items():
            b64_map[s][m] = base64.b64encode(str(html).encode("utf-8", errors="ignore")).decode("ascii")
    map_json = json.dumps(b64_map)
    subj_opts = "".join(f"<option value='{_esc(s)}'>{_esc(s)}</option>" for s in subjects)
    meas_opts = "".join(f"<option value='{_esc(m)}'>{_esc(m)}</option>" for m in measures)

    # srcdoc is set dynamically to avoid giant HTML attribute at load time
    return f"""
<div id="hotspot-gallery">
<label>Subject <select id="{pid}_hg_sub">{subj_opts}</select></label>
<label>Measure <select id="{pid}_hg_meas">{meas_opts}</select></label>
<iframe id="{pid}_hg_frame" title="hotspot" style="width:100%;height:560px;border:1px solid rgba(255,255,255,0.15);border-radius:10px"></iframe>
</div>
<script type="application/json" id="{pid}_hg_blob">{map_json}</script>
<script>
(function() {{
  function b64ToUtf8(b64) {{
    // decode base64 -> utf8 string
    try {{
      return decodeURIComponent(escape(atob(b64)));
    }} catch (e) {{
      // fallback (may mangle unicode, but better than blank)
      try {{ return atob(b64); }} catch (e2) {{ return ""; }}
    }}
  }}

  var blob = document.getElementById("{pid}_hg_blob");
  var subMap = JSON.parse(blob.textContent || "{{}}");
  var fs = document.getElementById("{pid}_hg_frame");
  var ss = document.getElementById("{pid}_hg_sub");
  var sm = document.getElementById("{pid}_hg_meas");

  function refillMeasures() {{
    var sub = ss.value;
    var mm = subMap[sub] || {{}};
    var keys = Object.keys(mm).sort();
    sm.innerHTML = keys.map(function(k) {{
      return "<option value='" + k.replace(/'/g, \"&#39;\") + \"'>\" + k.replace(/</g, \"&lt;\") + \"</option>\";
    }}).join(\"\");
    if (keys.length) sm.value = keys[0];
  }}

  function upd() {{
    var sub = ss.value;
    var me = sm.value;
    var b64 = subMap[sub] && subMap[sub][me];
    if (b64) fs.srcdoc = b64ToUtf8(b64);
  }}

  ss.onchange = function() {{ refillMeasures(); upd(); }};
  sm.onchange = upd;
  refillMeasures(); upd();
}})();
</script>
"""


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
    )


__all__ = [
    "export_hotspot_gallery_for_batch",
    "hotspot_gallery_control_html",
    "hotspot_gallery_control_srcdoc",
]
