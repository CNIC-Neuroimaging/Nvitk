#!/usr/bin/env python3
"""Import ``PSQEDUCA`` (educational attainment) into ``clinical_measurements``.

Description
-----------
Education is the standard proxy for cognitive reserve, so it belongs beside age and sex as a
covariate in essentially every cognitive model this dataset supports.

``PSQEDUCA`` is an ordinal level (2–6 in the current export), not a count of years. It is written
as a float so it can be used either as a continuous covariate or binned into groups, but treat the
spacing between levels as arbitrary unless the study codebook says otherwise.

Examples::

    # Report the match rate and what would be written
    python scripts/database/import_pqseduca.py

    # Write it
    python scripts/database/import_pqseduca.py --write
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
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import DEFAULT_VISIT_LABEL, read_tabular_source

log = Logger()

DEFAULT_SOURCE = Path(
    "/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/"
    "PESA-Brain/DB/raw/PESABrain_Clinical_AllXNAT_20260216.xlsx"
)

SOURCE_COLUMN = "PSQEDUCA"
VARIABLE_ID = "psqeduca"
#: The workbook carries the study visit and the date the psychosocial block was administered — the
#: block PSQEDUCA belongs to. Both are read rather than defaulted: this cohort is entirely visit 4
#: today, but the dataset already holds visit-3 clinical rows, and a hard-coded default would file
#: a future visit-3 export under the wrong visit without complaining.
VISIT_COLUMN = "VISITA"
DATE_COLUMN = "PSQDATE_PSICOSOCIAL"


# ---------------------------------------------------------------------------
# SEQN → subject_uid
# ---------------------------------------------------------------------------
def seqn_to_subject(repo: Any) -> dict[str, str]:
    """
    Map SEQN to ``subject_uid``, from the ``subject_ids`` registry first and ``subjects`` second.

    Both hold the key, with very different coverage: the ``seqn`` namespace of ``subject_ids``
    carries it for essentially every subject, while ``subjects.primary_seqn`` is only filled for the
    few hundred whose catalog row was built from a source that had it. Reading only the latter would
    silently discard two thirds of the cohort, so the registry leads and ``subjects`` fills gaps.

    Keys are normalized to plain digit strings: a SEQN read from Excel can arrive as ``28``,
    ``28.0`` or ``" 28 "`` depending on the column's dtype, and those must all resolve alike.
    """
    mapping: dict[str, str] = {}
    collisions = 0

    def _add(seqn: Any, subject: Any) -> None:
        nonlocal collisions
        key = _normalize_seqn(seqn)
        if not key or subject is None or pd.isna(subject):
            return
        value = str(subject)
        if key in mapping:
            if mapping[key] != value:
                collisions += 1
            return
        mapping[key] = value

    registry = repo.get("subject_ids", cohort_id=False)
    if registry is not None and not registry.empty and "id_namespace" in registry.columns:
        rows = registry.loc[registry["id_namespace"].astype(str).str.lower() == "seqn"]
        for seqn, subject in zip(rows["id_value"], rows["subject_uid"]):
            _add(seqn, subject)
        log.info("SEQN from 'subject_ids': %d pair(s).", len(mapping))

    from_registry = len(mapping)
    subjects = repo.get("subjects", cohort_id=False)
    if subjects is not None and not subjects.empty and "primary_seqn" in subjects.columns:
        for seqn, subject in zip(subjects["primary_seqn"], subjects["subject_uid"]):
            _add(seqn, subject)
        log.info("SEQN from 'subjects.primary_seqn': +%d new pair(s).", len(mapping) - from_registry)

    if not mapping:
        raise ValueError(
            "No SEQN → subject_uid mapping in this dataset: neither the 'seqn' namespace of "
            "'subject_ids' nor 'subjects.primary_seqn' is populated. Run the subject-id import "
            "first."
        )
    if collisions:
        log.warning(
            "%d SEQN value(s) resolve to more than one subject_uid; kept the first of each.",
            collisions,
        )
    log.info("Resolved %d SEQN → subject_uid pair(s) in total.", len(mapping))
    return mapping


def _normalize_seqn(value: Any) -> str:
    """``28``, ``28.0``, ``' 28 '`` → ``'28'``; anything unusable → ``''``."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Excel turns an integer id into a float as soon as one cell in the column is blank.
    if text.endswith(".0"):
        text = text[:-2]
    return text


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_psqeduca(
    path: Path, mapping: dict[str, str], *, sheet: str | int = 0
) -> pd.DataFrame:
    """
    One row per resolved subject: ``subject_uid``, ``visit_id`` and the education level.

    Raises
    ------
    ValueError
        When the workbook carries neither the SEQN key nor the value column.
    """
    raw = read_tabular_source(path, sheet_name=sheet)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    seqn_column = lower.get("seqn")
    value_column = lower.get(SOURCE_COLUMN.lower())
    if seqn_column is None or value_column is None:
        raise ValueError(
            f"{path.name} needs both a 'SEQN' and a '{SOURCE_COLUMN}' column; found "
            f"{', '.join(str(c) for c in raw.columns[:12])}…"
        )

    visit_column = lower.get(VISIT_COLUMN.lower())
    date_column = lower.get(DATE_COLUMN.lower())
    frame = pd.DataFrame({
        "seqn": raw[seqn_column].map(_normalize_seqn),
        "value_num": pd.to_numeric(raw[value_column], errors="coerce"),
        "visit_id": (
            raw[visit_column].map(_normalize_seqn).astype("string") if visit_column
            else pd.Series(DEFAULT_VISIT_LABEL, index=raw.index, dtype="string")
        ),
        "measured_at": (
            pd.to_datetime(raw[date_column], errors="coerce") if date_column
            else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        ),
    })
    if visit_column is None:
        log.warning(
            "No %r column — filing every row under visit %s.", VISIT_COLUMN, DEFAULT_VISIT_LABEL
        )
    if date_column is None:
        log.warning("No %r column — measured_at will be empty.", DATE_COLUMN)
    n_rows = len(frame)
    frame = frame[frame["seqn"].astype(bool) & frame["value_num"].notna()]
    n_valued = len(frame)

    frame["subject_uid"] = frame["seqn"].map(mapping)
    unmatched = frame.loc[frame["subject_uid"].isna(), "seqn"].tolist()
    frame = frame.dropna(subset=["subject_uid"])

    log.info(
        "%s: %d row(s) → %d with a value → %d resolved to a subject.",
        path.name, n_rows, n_valued, len(frame),
    )
    if unmatched:
        log.warning(
            "%d SEQN value(s) have no subject in this dataset and were skipped (e.g. %s). "
            "These are participants without imaging, or a key mismatch.",
            len(unmatched), ", ".join(unmatched[:8]),
        )
    if not frame.empty:
        levels = sorted(frame["value_num"].unique())
        log.info("%s levels present: %s", SOURCE_COLUMN, ", ".join(f"{v:g}" for v in levels))
        log.info(
            "visits: %s | measured_at: %d of %d dated.",
            frame["visit_id"].value_counts().to_dict(),
            int(frame["measured_at"].notna().sum()), len(frame),
        )
    frame["visit_id"] = frame["visit_id"].fillna(DEFAULT_VISIT_LABEL)
    return frame.loc[
        :, ["subject_uid", "visit_id", "value_num", "measured_at"]
    ].reset_index(drop=True)


def publish_psqeduca(
    repo: Any, frame: pd.DataFrame, *, path: Path, sheet: str | int = 0, write: bool = False
) -> pd.DataFrame:
    """Build the ``clinical_measurements`` rows and, with *write*, upsert them."""
    rows = build_clinical_measurement_rows(
        frame,
        DerivedClinicalMeasurementSpec(
            variable_id=VARIABLE_ID,
            source_file=path.name,
            source_sheet=str(sheet),
            source_column=SOURCE_COLUMN,
            unit=None,
            source_batch_id="import_pqseduca",
        ),
    )
    if rows.empty:
        log.warning("No %s rows to write.", VARIABLE_ID)
        return rows

    # The spec carries one measured_at for the whole batch; these dates are per subject, so they
    # are assigned after the build. Positions line up — the builder resets the index and preserves
    # row order.
    if "measured_at" in frame.columns:
        rows["measured_at"] = frame["measured_at"].reset_index(drop=True)

    if write:
        publish_derived_measurements(
            repo,
            rows,
            table="clinical_measurements",
            register=DerivedVariableRegistration(
                variable_id=VARIABLE_ID,
                domain="clinical",
                table="clinical_measurements",
                label="Educational attainment (PSQEDUCA, ordinal level)",
                value_kind="float",
                source_file=path.name,
                source_sheet=str(sheet),
                source_column=SOURCE_COLUMN,
            ),
            provenance={"importer": "import_pqseduca", "source_file": path.name},
            build_sqlite_index=True,
        )
        log.ok("Wrote %d %s row(s).", len(rows), VARIABLE_ID)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-pqseduca")
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SOURCE,
    show_default=False,
    help="Clinical workbook holding SEQN and PSQEDUCA.",
)
@click.option("--sheet", default="Datos", show_default=True, help="Worksheet to read.")
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
    """Import PSQEDUCA into clinical_measurements, resolving SEQN to subject_uid."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(f"Could not open the dataset ({exc}).") from exc

    try:
        mapping = seqn_to_subject(repo)
        frame = extract_psqeduca(Path(source), mapping, sheet=sheet)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    rows = publish_psqeduca(repo, frame, path=Path(source), sheet=sheet, write=write)
    if write:
        click.echo(f"Wrote {len(rows)} {VARIABLE_ID} row(s) to clinical_measurements.")
    else:
        click.echo(
            f"Dry run — {len(rows)} row(s) would be written as {VARIABLE_ID}.\n"
            f"Re-run with --write to apply."
        )


if __name__ == "__main__":
    main()
