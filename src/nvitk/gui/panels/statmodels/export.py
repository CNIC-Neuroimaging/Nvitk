"""
Export the active analysis dataframe.

Description
-----------
Writes exactly what the table shows — measurements joined, derived columns computed, filter rules
applied — so a spreadsheet handed to a collaborator matches the rows the model was fitted on.

An ``.xlsx`` export carries a second **provenance** sheet recording how the frame was built: which
measurements and groupings, which covariates and the visits they came from, every active filter with
the rows it removed, and every derived column's definition. Without it a spreadsheet is a number
soup that nobody — including its author six months later — can reconstruct.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.stats.frame_ops import DerivedColumn, FilterRule

log = Logger()

# Excel tops out just under 2^20 rows; leave room for the header.
EXCEL_ROW_LIMIT = 1_048_575


def build_provenance_frame(
    *,
    frame: pd.DataFrame,
    source_rows: int,
    measurements: Sequence[Any] = (),
    join: str = "inner",
    covariates: Sequence[str] = (),
    visit_provenance: Mapping[str, Sequence[str]] | None = None,
    derived: Sequence[DerivedColumn] = (),
    filters: Sequence[FilterRule] = (),
    filter_report: Sequence[Mapping[str, Any]] = (),
    dataset: str = "",
) -> pd.DataFrame:
    """Two-column ``item``/``detail`` record of how the exported frame was assembled."""
    rows: list[tuple[str, str]] = [
        ("Exported", datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")),
        ("Dataset", str(dataset)),
        ("Rows exported", f"{len(frame)}"),
        ("Rows before filtering", f"{source_rows}"),
        ("Columns", f"{len(frame.columns)}"),
        ("Join", str(join)),
    ]
    for i, spec in enumerate(measurements, start=1):
        rows.append((f"Measurement {i}", getattr(spec, "label", lambda: str(spec))()))
    if covariates:
        rows.append(("Covariates", ", ".join(str(c) for c in covariates)))
    for column, visits in sorted(dict(visit_provenance or {}).items()):
        rows.append((f"Visit · {column}", ", ".join(str(v) for v in visits)))

    by_index = {i: entry for i, entry in enumerate(filter_report)}
    if not filters:
        rows.append(("Filters", "none"))
    for i, rule in enumerate(filters):
        entry = by_index.get(i) or {}
        if entry.get("skipped"):
            effect = f"not applied ({entry.get('reason', '')})"
        else:
            effect = f"removed {int(entry.get('removed', 0))} row(s)"
        rows.append((f"Filter {i + 1}", f"{rule.label()} — {effect}"))

    if not derived:
        rows.append(("Derived columns", "none"))
    for column in derived:
        rows.append((f"Derived · {column.name}", column.label()))

    return pd.DataFrame(rows, columns=["item", "detail"])


def export_analysis_frame(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    provenance: pd.DataFrame | None = None,
) -> Path:
    """
    Write *frame* to *path*, choosing the format from the extension.

    ``.xlsx`` gets a ``data`` sheet plus an optional ``provenance`` sheet; ``.csv`` and ``.tsv`` are
    written flat, with the provenance alongside as ``<name>.provenance.csv`` so it is not lost.

    Raises
    ------
    ValueError
        For an unsupported extension, an empty frame, or an Excel export that exceeds the sheet
        row limit.
    """
    out = Path(path)
    suffix = out.suffix.lower()
    if frame is None or frame.empty:
        raise ValueError("There is nothing to export — the analysis dataframe is empty.")

    # Categoricals and datetimes survive to_excel, but a tz-aware timestamp does not.
    export = frame.copy()
    for column in export.columns:
        if pd.api.types.is_datetime64tz_dtype(export[column]):
            export[column] = export[column].dt.tz_localize(None)

    if suffix in {".xlsx", ".xlsm"}:
        if len(export) > EXCEL_ROW_LIMIT:
            raise ValueError(
                f"{len(export)} rows exceeds the Excel sheet limit of {EXCEL_ROW_LIMIT}. "
                "Export to CSV instead, or filter the frame first."
            )
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            export.to_excel(writer, sheet_name="data", index=False)
            if provenance is not None and not provenance.empty:
                provenance.to_excel(writer, sheet_name="provenance", index=False)
            _autosize(writer.sheets["data"], export)
            if provenance is not None and not provenance.empty:
                _autosize(writer.sheets["provenance"], provenance)
    elif suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else ","
        export.to_csv(out, index=False, sep=separator)
        if provenance is not None and not provenance.empty:
            provenance.to_csv(out.with_suffix(f".provenance{suffix}"), index=False, sep=separator)
    else:
        raise ValueError(f"Unsupported export format {suffix!r} — use .xlsx, .csv or .tsv.")

    log.info("Exported %d rows x %d columns to %s", len(export), len(export.columns), out)
    return out



def export_group_summary(
    path: str | Path,
    summary: pd.DataFrame,
    *,
    provenance: pd.DataFrame | None = None,
    values: pd.DataFrame | None = None,
) -> Path:
    """
    Write a per-group statistics table, with its provenance and optionally the rows behind it.

    Three sheets rather than one: ``summary`` is the table, ``provenance`` records what it describes
    (a sheet of means with no note of which column, grouping or confidence level is unreconstructable
    later), and ``values`` carries the underlying rows so a reader can check a number without going
    back to the tool.

    Raises
    ------
    ValueError
        For an unsupported extension or an empty summary.
    """
    out = Path(path)
    suffix = out.suffix.lower()
    if summary is None or summary.empty:
        raise ValueError("There is nothing to export — the summary is empty.")

    if suffix in {".xlsx", ".xlsm"}:
        with pd.ExcelWriter(out, engine="openpyxl") as writer_:
            summary.to_excel(writer_, sheet_name="summary", index=False)
            _autosize(writer_.sheets["summary"], summary)
            if provenance is not None and not provenance.empty:
                provenance.to_excel(writer_, sheet_name="provenance", index=False)
                _autosize(writer_.sheets["provenance"], provenance)
            if values is not None and not values.empty and len(values) <= EXCEL_ROW_LIMIT:
                values.to_excel(writer_, sheet_name="values", index=False)
                _autosize(writer_.sheets["values"], values)
    elif suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else ","
        summary.to_csv(out, index=False, sep=separator)
        if provenance is not None and not provenance.empty:
            provenance.to_csv(out.with_suffix(f".provenance{suffix}"), index=False, sep=separator)
    else:
        raise ValueError(f"Unsupported export format {suffix!r} — use .xlsx, .csv or .tsv.")

    log.info("Exported a %d-group summary to %s", len(summary), out)
    return out


def _autosize(worksheet: Any, frame: pd.DataFrame, *, max_width: int = 42) -> None:
    """Widen each column to roughly fit its content, so the sheet opens readable."""
    from openpyxl.utils import get_column_letter

    for i, column in enumerate(frame.columns, start=1):
        sample = frame[column].astype(str).head(200)
        width = max(len(str(column)), int(sample.str.len().max()) if len(sample) else 0)
        worksheet.column_dimensions[get_column_letter(i)].width = min(width + 2, max_width)


__all__ = [
    "EXCEL_ROW_LIMIT",
    "build_provenance_frame",
    "export_analysis_frame",
    "export_group_summary",
]
