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
    """Render a DataFrame as an HTML table with out-of-range cells highlighted."""
    if df.empty:
        return "<p><em>No measurement rows loaded.</em></p>"

    cols = list(df.columns)
    thead = "<thead><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in cols) + "</tr></thead>"
    body_rows: list[str] = []
    for _, row in df.iterrows():
        tds: list[str] = []
        for c in cols:
            val = row[c]
            style = ""
            if cell_out_of_range(c, val, ranges):
                style = f' style="background-color:{WARN_BG}"'
            if pd.isna(val):
                disp = ""
            else:
                disp = _esc(_format_cell(val))
            tds.append(f"<td{style}>{disp}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f'<table class="qc-measurements" border="1" cellspacing="0" cellpadding="4">{thead}{tbody}</table>'


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
