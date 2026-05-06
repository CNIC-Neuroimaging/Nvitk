"""Precompute PyVista hotspot HTML exports for QC gallery."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.run_hotspot import (
    _list_valid_measures,
    _load_ctpet_inputs,
    _load_dixon_inputs,
    _resolve_measure_ctpet,
    _resolve_measure_dixon,
)
from nvitk.viz import show_hotspots

log = Logger()


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


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
            try:
                if ct is not None:
                    img, mask_img, label_ids, _pet = _load_ctpet_inputs(lay, subject, ct)
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
                )
                fname = f"hotspot_{_safe_name(subject)}_{_safe_name(measure)}.html"
                path = out_dir / fname
                try:
                    pl.export_html(str(path))
                except Exception as exc:
                    import traceback
                    log.warning(traceback.format_exc())
                    log.warning("[%s] hotspot export failed %s: %s", subject, measure, exc)
                    path.write_text(
                        "<!DOCTYPE html><html><body><p>Hotspot HTML export failed: "
                        f"{exc!s}</p></body></html>",
                        encoding="utf-8",
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


def hotspot_gallery_control_html(entries: list[tuple[str, str, str]]) -> str:
    """Two ``<select>`` elements + ``<iframe>``; measure list depends on subject."""
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

    return f"""
<div id="hotspot-gallery">
<label>Subject <select id="hg_sub">{subj_opts}</select></label>
<label>Measure <select id="hg_meas">{meas_opts}</select></label>
<iframe id="hg_frame" title="hotspot" style="width:100%;height:560px;border:1px solid #444" src="{_esc(first_rel)}"></iframe>
</div>
<script>
(function() {{
  var subMap = {map_json};
  var fs = document.getElementById("hg_frame");
  var ss = document.getElementById("hg_sub");
  var sm = document.getElementById("hg_meas");
  function refillMeasures() {{
    var sub = ss.value;
    var mm = subMap[sub];
    var keys = Object.keys(mm).sort();
    sm.innerHTML = keys.map(function(k) {{
      return "<option value='" + k.replace(/'/g, "&#39;") + "'>" + k.replace(/</g, "&lt;") + "</option>";
    }}).join("");
  }}
  function upd() {{
    var sub = ss.value;
    var me = sm.value;
    var url = subMap[sub] && subMap[sub][me];
    if (url) fs.src = url;
  }}
  ss.onchange = function() {{ refillMeasures(); upd(); }};
  sm.onchange = upd;
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
]
