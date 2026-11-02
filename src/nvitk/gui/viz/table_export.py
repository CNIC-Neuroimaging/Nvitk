"""The Results panel's table export: the rows and columns a ``QTableWidget`` shows, to a
DataFrame, then to an ``.xlsx`` workbook or a ``.csv`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def visible_table_frame(table: Any, payload: Mapping[Any, Any]) -> pd.DataFrame:
    """The rows (in their displayed, sorted order) and columns a ``QTableWidget`` shows, with the
    result *payload*'s own values instead of their formatted text where those are plain numbers or
    text. A matrix payload maps row keys (the first column) to ``{column: value}``; a flat one is
    shown as two columns, key and value."""
    cols = [c for c in range(table.columnCount()) if not table.isColumnHidden(c)]
    headers = [(table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) is not None else "")
               for c in cols]
    by_key = {str(k): v for k, v in payload.items()}
    matrix = bool(payload) and all(isinstance(v, Mapping) for v in payload.values())

    def plain(value: Any, shown: str) -> Any:
        return value if isinstance(value, (int, float, str, bool)) or value is None else shown

    rows = []
    for r in range(table.rowCount()):
        if table.isRowHidden(r):
            continue
        cells = [table.item(r, c) for c in cols]
        shown = [cell.text() if cell is not None else "" for cell in cells]
        key = shown[0] if shown else ""
        raw = by_key.get(key)
        if matrix and isinstance(raw, Mapping):
            rows.append([key] + [plain(raw.get(h, ""), t) for h, t in zip(headers[1:], shown[1:])])
        elif not matrix and len(cols) == 2:
            rows.append([key, plain(raw, shown[1])])
        else:
            rows.append(shown)
    return pd.DataFrame(rows, columns=headers)


def export_table(frame: pd.DataFrame, path: Any, *, title: str = "Results", csv: bool = False) -> Path:
    """Write *frame* to *path*: a ``.csv`` file (by extension, or *csv* when the path has none), else
    an ``.xlsx`` workbook with one sheet named after *title*, columns sized to their content.
    Returns the path written."""
    from nvitk.gui.panels.statmodels.export import autosize_columns
    from nvitk.measure.morpho.io_utils import safe_sheet_name

    out = Path(path)
    if out.suffix.lower() not in (".csv", ".xlsx"):
        out = out.with_suffix(".csv" if csv else ".xlsx")
    if out.suffix.lower() == ".csv":
        frame.to_csv(out, index=False)
        return out
    sheet = safe_sheet_name(str(title)).strip() or "Results"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet, index=False)
        autosize_columns(writer.sheets[sheet], frame)
    return out


def export_dialog(parent: Any, frame: pd.DataFrame, title: str) -> Path | None:
    """Ask where to save *frame* (``.xlsx`` or ``.csv``) and save it there; the path written, or
    ``None`` when cancelled or failed (the failure is shown in a message box)."""
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    stem = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in str(title)).strip() or "results"
    path, chosen = QFileDialog.getSaveFileName(
        parent, "Export the table", f"{stem}.xlsx", "Excel workbook (*.xlsx);;CSV (*.csv)")
    if not path:
        return None
    try:
        return export_table(frame, Path(path), title=title, csv="csv" in str(chosen).lower())
    except Exception as exc:  # noqa: BLE001 (report it, keep the window)
        QMessageBox.warning(parent, "Export failed", str(exc))
        return None


__all__ = ["export_dialog", "export_table", "visible_table_frame"]
