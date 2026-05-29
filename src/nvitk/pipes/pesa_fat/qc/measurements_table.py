"""Load stage-3 per-subject spreadsheets and render HTML tables with range highlighting."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import pandas as pd

from nvitk.pipes.pesa_fat.common.paths import BatchLayout
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

PesaFatQcPipeline = Literal["ct-pet-v5", "dixon-v5"]

from nvitk.pipes.pesa_fat.qc.expected_ranges import cell_level

WARN_BG = "#fca311"
BAD_BG = "#ff4d4d"


def load_per_subject_tables(per_subject_dir: Path, subjects: list[str]) -> pd.DataFrame:
    """Concatenate ``<subject>.xlsx`` files (one row each) into a single DataFrame."""
    rows: list[pd.DataFrame] = []
    for sub in subjects:
        p = per_subject_dir / f"{sub}.xlsx"
        if not p.exists():
            continue
        df = pd.read_excel(p)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def dataframe_to_html_table(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
) -> str:
    """Render a DataFrame as an HTML table with out-of-range cells highlighted.

    v2: expects *df* to contain a single subject row and reshapes it to
    ROI-per-row, metric-per-column for readability.
    """
    if df.empty:
        return "<p><em>No measurement rows loaded.</em></p>"

    row = df.iloc[0].to_dict()
    # Remove pesa_id if present
    row.pop("pesa_id", None)
    if not row:
        return "<p><em>No measurement columns.</em></p>"

    # Parse columns into (roi, metric)
    parsed: list[tuple[str, str, str, object]] = []
    for col, val in row.items():
        c = str(col)
        if "_" not in c:
            continue
        roi, metric = c.rsplit("_", 1)
        # CT-PET aliasing: MO_L3/MO_L4 volume+nslices should appear under L3/L4 rows.
        if roi == "MO_L3" and metric in ("VOL", "NSlices"):
            roi = "L3"
        elif roi == "MO_L4" and metric in ("VOL", "NSlices"):
            roi = "L4"
        parsed.append((roi, metric, c, val))

    if not parsed:
        return "<p><em>No parseable measurement columns.</em></p>"

    rois = sorted({p[0] for p in parsed})
    metrics = sorted({p[1] for p in parsed}, key=lambda x: ("VOL" not in x, x))
    # Build lookup
    lut: dict[tuple[str, str], tuple[str, object]] = {}
    for roi, metric, full_col, val in parsed:
        lut[(roi, metric)] = (full_col, val)

    cols = ["ROI"] + metrics
    thead = "<thead><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in cols) + "</tr></thead>"
    body_rows: list[str] = []
    for roi in rois:
        tds = [f"<td><strong>{_esc(roi)}</strong></td>"]
        for metric in metrics:
            full_col, val = lut.get((roi, metric), ("", None))
            style = ""
            if full_col:
                lvl = cell_level(full_col, val, ranges)
                if lvl == "warn":
                    style = f' style="background-color:{WARN_BG};color:#000000"'
                elif lvl == "bad":
                    style = f' style="background-color:{BAD_BG};color:#000000"'
            if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
                disp = ""
            else:
                disp = _esc(_format_cell(val))
            tds.append(f"<td{style}>{disp}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f'<table class="qc-measurements" border="0" cellspacing="0" cellpadding="0">{thead}{tbody}</table>'


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_cell(val: object) -> str:
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def stage3_measurements_xlsx_path(
    lay: BatchLayout,
    subject: str,
    pipeline: PesaFatQcPipeline,
) -> Path:
    """Path to the stage-3 per-subject measurements workbook."""
    subj = str(subject).strip()
    if pipeline == "ct-pet-v5":
        return lay.results_dir / ct_cfg.STAGE3_DIR / "per_subject" / f"{subj}.xlsx"
    return lay.results_dir / dx_cfg.STAGE3_DIR / "per_subject" / f"{subj}.xlsx"


def copy_measurements_xlsx_for_qc(
    *,
    lay: BatchLayout,
    subject: str,
    pipeline: PesaFatQcPipeline,
    assets_dir: Path,
    rel_assets: str,
) -> str | None:
    """Copy stage-3 Excel beside QC HTML assets; return relative download href."""
    src = stage3_measurements_xlsx_path(lay, subject, pipeline)
    if not src.is_file():
        return None
    subj = str(subject).strip()
    tag = "ctpet" if pipeline == "ct-pet-v5" else "dixon"
    fname = f"measurements_{tag}_{subj}.xlsx"
    dest_dir = Path(assets_dir) / "measurements"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / fname)
    rel = str(rel_assets).strip().rstrip("/")
    return f"{rel}/measurements/{fname}"


def measurements_download_button_html(
    href: str | None,
    *,
    label: str = "Download Table",
) -> str:
    """Link button to download the measurements Excel (relative to the QC report)."""
    if not href:
        return '<span class="muted">Excel not available</span>'
    safe_href = _esc(href)
    safe_label = _esc(label)
    return (
        f'<a class="qc-dl-btn" href="{safe_href}" download>{safe_label}</a>'
    )


__all__ = [
    "PesaFatQcPipeline",
    "copy_measurements_xlsx_for_qc",
    "dataframe_to_html_table",
    "load_per_subject_tables",
    "measurements_download_button_html",
    "stage3_measurements_xlsx_path",
]
