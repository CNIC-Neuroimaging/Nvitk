"""Load stage-3 per-subject spreadsheets and render HTML tables with range highlighting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvitk.pipes.pesa_fat.qc.expected_ranges import cell_out_of_range

WARN_BG = "#ffcccc"


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
            if full_col and cell_out_of_range(full_col, val, ranges):
                style = f' style="background-color:{WARN_BG}"'
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


__all__ = [
    "dataframe_to_html_table",
    "load_per_subject_tables",
]
