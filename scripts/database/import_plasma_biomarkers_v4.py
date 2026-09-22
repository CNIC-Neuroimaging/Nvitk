#!/usr/bin/env python3
"""Import visit-4 plasma biomarkers into ``clinical_measurements``.

Two Excel sources are supported:

1. Main biomarkers workbook, keyed directly by ``pesa_id`` (= ``subject_uid``),
   with columns::

       pesa_id
       plasma_bbrc_batch
       plasma_bbrc_ab40
       plasma_bbrc_ab42
       plasma_bbrc_ab4240
       plasma_bbrc_ptau181
       plasma_bbrc_nfl
       plasma_bbrc_gfap

2. Fujirebio pTau217 workbook, sheet ``pTau217Fujirebio``, keyed by
   ``Participant ID`` (= ``subject_uid``), with column::

       Fujirebio - pTau217 (pg/mL)

All imported rows are filed under ``visit_id = 4``.

Default mode is dry-run.  Use ``--write`` to publish rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import read_tabular_source

log = Logger()

DEFAULT_MAIN_SOURCE = Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/plasma_biomarkers_combined_batches.xlsx")
DEFAULT_PTAU217_SOURCE = Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/ptau217_BBRC_20260326.xlsx")
DEFAULT_MAIN_SHEET = "0"
DEFAULT_PTAU217_SHEET = "pTau217 Fujirebio"
SOURCE_BATCH_ID = "import_plasma_biomarkers_v4"
VISIT_ID = "4"

MAIN_COLUMNS = (
    "plasma_bbrc_batch",
    "plasma_bbrc_ab40",
    "plasma_bbrc_ab42",
    "plasma_bbrc_ab4240",
    "plasma_bbrc_ptau181",
    "plasma_bbrc_nfl",
    "plasma_bbrc_gfap",
)
PTAU217_SOURCE_COLUMN = "Fujirebio - pTau217 (pg/mL)"
PTAU217_VARIABLE_ID = "plasma_bbrc_ptau217"

LABELS = {
    "plasma_bbrc_batch": "Plasma BBRC batch",
    "plasma_bbrc_ab40": "Plasma BBRC Aβ40",
    "plasma_bbrc_ab42": "Plasma BBRC Aβ42",
    "plasma_bbrc_ab4240": "Plasma BBRC Aβ42/Aβ40",
    "plasma_bbrc_ptau181": "Plasma BBRC p-tau181",
    "plasma_bbrc_nfl": "Plasma BBRC NfL",
    "plasma_bbrc_gfap": "Plasma BBRC GFAP",
    PTAU217_VARIABLE_ID: "Plasma p-tau217 (Fujirebio)",
}


def _parse_sheet(value: str) -> str | int:
    return int(value) if value.strip().isdigit() else value


def _column_map(raw: pd.DataFrame) -> dict[str, Any]:
    return {str(c).strip().lower(): c for c in raw.columns}


def _normalize_subject_uid(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _extract_main(path: Path, *, sheet: str | int = DEFAULT_MAIN_SHEET) -> pd.DataFrame:
    raw = read_tabular_source(path, sheet_name=_parse_sheet(str(sheet)))
    cols = _column_map(raw)
    required = ["pesa_id", *MAIN_COLUMNS]
    missing = [name for name in required if name.lower() not in cols]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}")

    subject = raw[cols["pesa_id"]].map(_normalize_subject_uid)
    frames: list[pd.DataFrame] = []
    for variable_id in MAIN_COLUMNS:
        values = raw[cols[variable_id.lower()]]
        # The batch is kept as text because it is an identifier, while all
        # biomarker concentrations/ratios are numeric.
        if variable_id == "plasma_bbrc_batch":
            text_values = values.astype("string").str.strip()
            frame = pd.DataFrame(
                {
                    "subject_uid": subject,
                    "visit_id": VISIT_ID,
                    "value_num": pd.NA,
                    "value_text": text_values,
                    "measured_at": pd.NaT,
                    "source_column": variable_id,
                    "variable_id": variable_id,
                }
            )
            frame = frame[frame["value_text"].notna() & frame["value_text"].ne("")]
        else:
            frame = pd.DataFrame(
                {
                    "subject_uid": subject,
                    "visit_id": VISIT_ID,
                    "value_num": pd.to_numeric(values, errors="coerce"),
                    "value_text": pd.NA,
                    "measured_at": pd.NaT,
                    "source_column": variable_id,
                    "variable_id": variable_id,
                }
            )
            frame = frame[frame["value_num"].notna()]

        frame = frame[frame["subject_uid"].ne("")]
        if not frame.empty:
            frames.append(frame)
        log.info("%s: %d row(s) with a value.", variable_id, len(frame))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_ptau217(path: Path, *, sheet: str = DEFAULT_PTAU217_SHEET) -> pd.DataFrame:
    raw = read_tabular_source(path, sheet_name=_parse_sheet(str(sheet)))
    cols = _column_map(raw)

    participant_col = cols.get("participant id")
    value_col = cols.get(PTAU217_SOURCE_COLUMN.lower())
    missing = []
    if participant_col is None:
        missing.append("Participant ID")
    if value_col is None:
        missing.append(PTAU217_SOURCE_COLUMN)
    if missing:
        raise ValueError(f"{path.name} ({sheet}) is missing required column(s): {', '.join(missing)}")

    frame = pd.DataFrame(
        {
            "subject_uid": raw[participant_col].map(_normalize_subject_uid),
            "visit_id": VISIT_ID,
            "value_num": pd.to_numeric(raw[value_col], errors="coerce"),
            "value_text": pd.NA,
            "measured_at": pd.NaT,
            "source_column": PTAU217_SOURCE_COLUMN,
            "variable_id": PTAU217_VARIABLE_ID,
        }
    )
    frame = frame[frame["subject_uid"].ne("") & frame["value_num"].notna()]
    log.info("%s: %d pTau217 row(s) with a value.", path.name, len(frame))
    return frame


def _publish_frame(
    repo: Any,
    frame: pd.DataFrame,
    *,
    path: Path,
    sheet: str | int,
    write: bool,
    source_batch_id: str,
) -> int:
    if frame.empty:
        return 0

    total = 0
    for variable_id, sub in frame.groupby("variable_id", sort=False):
        source_column = str(sub["source_column"].iloc[0])
        rows = build_clinical_measurement_rows(
            sub,
            DerivedClinicalMeasurementSpec(
                variable_id=variable_id,
                source_file=path.name,
                source_sheet=str(sheet),
                source_column=source_column,
                unit="pg/mL" if variable_id == PTAU217_VARIABLE_ID else None,
                source_batch_id=source_batch_id,
            ),
        )
        # build_clinical_measurement_rows uses the value payload from the input
        # frame; the explicit value_text/value_num columns above preserve batch
        # identifiers as text and biomarkers as numeric values.
        if rows.empty:
            continue

        if write:
            publish_derived_measurements(
                repo,
                rows,
                table="clinical_measurements",
                register=DerivedVariableRegistration(
                    variable_id=variable_id,
                    domain="clinical",
                    table="clinical_measurements",
                    label=LABELS.get(variable_id, variable_id),
                    value_kind="categorical" if variable_id == "plasma_bbrc_batch" else "float",
                    source_file=path.name,
                    source_sheet=str(sheet),
                    source_column=source_column,
                    unit="pg/mL" if variable_id == PTAU217_VARIABLE_ID else None,
                ),
                provenance={
                    "importer": "import_plasma_biomarkers_v4",
                    "source_file": path.name,
                    "visit": VISIT_ID,
                },
                build_sqlite_index=True,
            )
        total += len(rows)
        log.info(
            "%s: %d row(s) %s.",
            variable_id,
            len(rows),
            "written" if write else "would be written",
        )

    return total


def publish_all(
    repo: Any,
    main_frame: pd.DataFrame,
    ptau217_frame: pd.DataFrame,
    *,
    main_path: Path,
    ptau217_path: Path,
    main_sheet: str | int,
    ptau217_sheet: str,
    write: bool,
    source_batch_id: str,
) -> int:
    total = _publish_frame(
        repo,
        main_frame,
        path=main_path,
        sheet=main_sheet,
        write=write,
        source_batch_id=source_batch_id,
    )
    total += _publish_frame(
        repo,
        ptau217_frame,
        path=ptau217_path,
        sheet=ptau217_sheet,
        write=write,
        source_batch_id=source_batch_id,
    )
    return total


@click.command("import-plasma-biomarkers-v4")
@click.option(
    "--main-source",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_MAIN_SOURCE,
    show_default=False,
    help="Excel workbook containing the main plasma BBRC biomarkers.",
)
@click.option(
    "--ptau217-source",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_PTAU217_SOURCE,
    show_default=False,
    help="Excel workbook containing the Fujirebio pTau217 sheet.",
)
@click.option("--main-sheet", default=DEFAULT_MAIN_SHEET, show_default=True, type=str, help="Main workbook sheet name or zero-based sheet index.")
@click.option("--ptau217-sheet", default=DEFAULT_PTAU217_SHEET, show_default=True)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the path configured in .nvitk/settings.json.",
)
@click.option("--source-batch-id", default=SOURCE_BATCH_ID, show_default=True)
@click.option("--write/--dry-run", default=False, show_default="--dry-run")
def main(
    main_source: Path,
    ptau217_source: Path,
    main_sheet: str | int,
    ptau217_sheet: str,
    dataset: Path | None,
    source_batch_id: str,
    write: bool,
) -> None:
    """Import all requested visit-4 plasma biomarkers into clinical_measurements."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
        main_frame = _extract_main(Path(main_source), sheet=main_sheet)
        ptau217_frame = _extract_ptau217(Path(ptau217_source), sheet=ptau217_sheet)
        total = publish_all(
            repo,
            main_frame,
            ptau217_frame,
            main_path=Path(main_source),
            ptau217_path=Path(ptau217_source),
            main_sheet=main_sheet,
            ptau217_sheet=ptau217_sheet,
            write=write,
            source_batch_id=source_batch_id,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    action = "Wrote" if write else "Dry run — would write"
    click.echo(f"{action} {total} plasma biomarker row(s) for visit 4 into clinical_measurements.")


if __name__ == "__main__":
    main()
