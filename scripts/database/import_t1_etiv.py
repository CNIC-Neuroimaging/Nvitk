#!/usr/bin/env python3
"""Import FreeSurfer eTIV (estimated total intracranial volume) into ``image_measurements``.

Description
-----------
The T1 volumetry exports carry one row per ``(subject, side, measurement, region)``. Among the
regions is ``eTIV`` — not a parcel but a whole-head scalar, FreeSurfer's estimate of intracranial
volume derived from the Talairach transform. It matters because it is the standard denominator for
head-size normalization: a raw hippocampal volume is not comparable between a 1.9 L and a 0.85 L
skull, and dividing by eTIV (or regressing it out) is what makes the comparison legitimate.

It is written as its own variable rather than as one region among the parcels, because it *is* a
different kind of quantity — a per-subject scalar, region ``etiv``, which every parcel is measured
against.

Source layout
-------------
Long format, in the same workbook as the cortical parcels::

    ID           side  measurement  region  value        subject      session

``side`` is empty for eTIV (it is not lateralized) and the same value repeats across a subject's
rows, so values are deduplicated per ``(subject, session)``.

Examples::

    # Report what would be written, touching nothing
    python scripts/database/import_t1_etiv.py

    # Write it
    python scripts/database/import_t1_etiv.py --write
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_image_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import read_tabular_source

log = Logger()

#: The cortical workbook carries eTIV; the subcortical one repeats the same value, so one source is
#: enough and using both would only add duplicate rows to deduplicate again.
DEFAULT_SOURCE = Path(
    "/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/"
    "PESA-Brain/DB/raw/PESABrain_T1_cortical_regs_20250605.xlsx"
)

T1_PIPELINE_ID = "t1_volumetry_v1"
VARIABLE_ID = "t1_etiv_volume"
REGION_ID = "etiv"
#: FreeSurfer reports eTIV in mm³ under the ``Volume_mm3`` measurement.
SOURCE_MEASUREMENT = "Volume_mm3"
SOURCE_REGION = "etiv"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_etiv(path: Path, *, sheet: str = "Sheet1") -> pd.DataFrame:
    """
    One row per ``(subject_uid, session)`` carrying that subject's eTIV in mm³.

    Raises
    ------
    ValueError
        When the workbook has none of the expected long-format columns, or holds no eTIV row —
        either means the export changed shape and importing it blindly would write nonsense.
    """
    raw = read_tabular_source(path, sheet_name=sheet)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    missing = [c for c in ("region", "measurement", "value", "subject") if c not in lower]
    if missing:
        raise ValueError(
            f"{path.name} is not in the expected long T1 layout — missing column(s): "
            f"{', '.join(missing)}. Found: {', '.join(str(c) for c in raw.columns)}."
        )

    region, measurement = raw[lower["region"]].astype(str), raw[lower["measurement"]].astype(str)
    hit = raw.loc[
        region.str.strip().str.lower().eq(SOURCE_REGION)
        & measurement.str.strip().str.lower().eq(SOURCE_MEASUREMENT.lower())
    ]
    if hit.empty:
        raise ValueError(
            f"No '{SOURCE_REGION}' / '{SOURCE_MEASUREMENT}' row in {path.name}. Regions present: "
            f"{', '.join(sorted(region.unique())[:8])}…"
        )

    session_column = lower.get("session")
    frame = pd.DataFrame({
        "subject_uid": hit[lower["subject"]].astype("string"),
        "session_id": (
            hit[session_column].astype("string") if session_column
            else pd.Series(pd.NA, index=hit.index, dtype="string")
        ),
        VARIABLE_ID: pd.to_numeric(hit[lower["value"]], errors="coerce"),
    }).dropna(subset=["subject_uid", VARIABLE_ID])

    # eTIV repeats on every row of a subject's export. Identical duplicates are expected; differing
    # ones are not, and would mean two runs of FreeSurfer got merged into one sheet.
    grouped = frame.groupby(["subject_uid", "session_id"], dropna=False)[VARIABLE_ID]
    spread = (grouped.max() - grouped.min()).abs()
    inconsistent = int((spread > 1.0).sum())
    if inconsistent:
        log.warning(
            "%d (subject, session) pair(s) carry more than one eTIV value (>1 mm³ apart); "
            "taking the mean. That usually means two FreeSurfer runs were concatenated.",
            inconsistent,
        )
    out = grouped.mean().reset_index()
    log.info(
        "eTIV: %d subject(s) over %d row(s); range %.0f–%.0f mm³.",
        out["subject_uid"].nunique(), len(hit),
        float(out[VARIABLE_ID].min()), float(out[VARIABLE_ID].max()),
    )
    return out


def publish_etiv(
    repo: Any, frame: pd.DataFrame, *, path: Path, sheet: str = "Sheet1", write: bool = False
) -> pd.DataFrame:
    """Build the ``image_measurements`` rows for *frame* and, with *write*, upsert them."""
    agg = frame.rename(columns={VARIABLE_ID: "value_num"}).assign(
        region_id=REGION_ID, region_label="Estimated total intracranial volume"
    )
    rows = build_image_measurement_rows(
        agg,
        DerivedImageMeasurementSpec(
            variable_id=VARIABLE_ID,
            modality="t1",
            pipeline_id=T1_PIPELINE_ID,
            pipeline_name=path.stem,
            source_file=path.name,
            source_sheet=sheet,
            source_column="value",
            unit="mm3",
            source_batch_id="import_t1_etiv",
        ),
    )
    if rows.empty:
        log.warning("No eTIV rows survived the frame build.")
        return rows

    if write:
        publish_derived_measurements(
            repo,
            rows,
            table="image_measurements",
            register=DerivedVariableRegistration(
                variable_id=VARIABLE_ID,
                domain="image",
                table="image_measurements",
                modality="t1",
                label="Estimated total intracranial volume",
                unit="mm3",
                value_kind="float",
                source_file=path.name,
                source_sheet=sheet,
                source_column="value",
            ),
            provenance={"importer": "import_t1_etiv", "source_file": path.name},
            build_sqlite_index=True,
            # The catalog key omits session_id, so a subject's repeat scan would overwrite the
            # first. Four subjects here have two T1 sessions and both are real measurements.
            upsert_key_columns=[
                "subject_uid", "session_id", "modality", "region_id", "frame_index",
                "variable_id", "pipeline_id", "source_file", "source_sheet", "source_column",
            ],
        )
        log.ok("Wrote %d eTIV row(s) as %s / region %s.", len(rows), VARIABLE_ID, REGION_ID)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-t1-etiv")
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SOURCE,
    show_default=False,
    help="T1 volumetry workbook holding the eTIV rows. Defaults to the cortical export.",
)
@click.option("--sheet", default="Sheet1", show_default=True, help="Worksheet to read.")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the one configured in .nvitk/settings.json.",
)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually write to the dataset. The default only reports what would be written.",
)
def main(source: Path, sheet: str, dataset: Path | None, write: bool) -> None:
    """Import eTIV from the T1 volumetry export into image_measurements."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(f"Could not open the dataset ({exc}).") from exc

    try:
        frame = extract_etiv(Path(source), sheet=sheet)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    rows = publish_etiv(repo, frame, path=Path(source), sheet=sheet, write=write)
    if write:
        click.echo(f"Wrote {len(rows)} eTIV row(s) to image_measurements.")
    else:
        click.echo(
            f"Dry run — {len(rows)} row(s) would be written as {VARIABLE_ID} "
            f"(region {REGION_ID}, pipeline {T1_PIPELINE_ID}).\n"
            f"Re-run with --write to apply."
        )


if __name__ == "__main__":
    main()
