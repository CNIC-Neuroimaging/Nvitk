#!/usr/bin/env python3
"""Import visit-4 hypertension/diabetes/dyslipemia/obesity into clinical_measurements.

Expected source columns:

    seqn
    visit
    hipertension  -> hypertension
    diabetes      -> diabetes
    dislipemia    -> dyslipemia
    obesidad      -> obesity

Only visit 4 is imported.  SEQN is resolved to ``subject_uid`` through the
dataset subject-id registries.

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

DEFAULT_SOURCE = Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/Brain_10_06_2026.xlsx")
DEFAULT_SHEET = "0"
SOURCE_BATCH_ID = "import_rest_v4"
VISIT_ID = "4"

VARIABLE_COLUMNS: dict[str, str] = {
    "hipertension": "hypertension",
    "diabetes": "diabetes",
    "dislipemia": "dyslipemia",
    "obesidad": "obesity",
}

LABELS = {
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipemia": "Dyslipemia",
    "obesity": "Obesity",
}


def _normalize_seqn(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return ""
    return text[:-2] if text.endswith(".0") else text


def seqn_to_subject(repo: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    collisions = 0

    def _add(seqn: Any, subject: Any) -> None:
        nonlocal collisions
        key = _normalize_seqn(seqn)
        if not key or subject is None or pd.isna(subject):
            return
        value = str(subject).strip()
        if not value:
            return
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

    subjects = repo.get("subjects", cohort_id=False)
    if subjects is not None and not subjects.empty and "primary_seqn" in subjects.columns:
        for seqn, subject in zip(subjects["primary_seqn"], subjects["subject_uid"]):
            _add(seqn, subject)

    if not mapping:
        raise ValueError("No SEQN -> subject_uid mapping exists in this dataset.")
    if collisions:
        log.warning("%d SEQN collision(s) detected; keeping the first mapping.", collisions)
    log.info("Resolved %d SEQN -> subject_uid pair(s).", len(mapping))
    return mapping


def _parse_sheet(value: str) -> str | int:
    return int(value) if value.strip().isdigit() else value


def _column_map(raw: pd.DataFrame) -> dict[str, Any]:
    return {str(c).strip().lower(): c for c in raw.columns}


def _is_visit_4(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"4", "4.0", "v4", "visit4", "visit 4", "visit_4", "visit-4"}:
        return True
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(numeric) and float(numeric) == 4.0)


def extract_rest(path: Path, mapping: dict[str, str], *, sheet: str | int = DEFAULT_SHEET) -> pd.DataFrame:
    raw = read_tabular_source(path, sheet_name=_parse_sheet(str(sheet)))
    cols = _column_map(raw)

    missing = [name for name in ("seqn", "visit") if name not in cols]
    missing.extend(name for name in VARIABLE_COLUMNS if name.lower() not in cols)
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}")

    visit_col = cols["visit"]
    work = raw.loc[raw[visit_col].map(_is_visit_4)].copy()
    log.info("%d row(s) belong to visit 4 out of %d total row(s).", len(work), len(raw))

    work["seqn"] = work[cols["seqn"]].map(_normalize_seqn)
    work["subject_uid"] = work["seqn"].map(mapping)
    unmatched = sorted(set(work.loc[work["subject_uid"].isna() & work["seqn"].astype(bool), "seqn"]))
    if unmatched:
        log.warning(
            "%d SEQN value(s) have no subject_uid match and will be skipped (e.g. %s).",
            len(unmatched), ", ".join(unmatched[:8]),
        )
    work = work.dropna(subset=["subject_uid"])

    frames: list[pd.DataFrame] = []
    for source_column, variable_id in VARIABLE_COLUMNS.items():
        source_col = cols[source_column]
        values = pd.to_numeric(work[source_col], errors="coerce")
        frame = pd.DataFrame(
            {
                "subject_uid": work["subject_uid"].astype("string"),
                "visit_id": VISIT_ID,
                "value_num": values,
                "measured_at": pd.NaT,
                "source_column": source_column,
                "variable_id": variable_id,
            },
            index=work.index,
        ).dropna(subset=["value_num"])
        if not frame.empty:
            frames.append(frame)
        log.info("%s -> %s: %d valued row(s).", source_column, variable_id, len(frame))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def publish_rest(
    repo: Any,
    frame: pd.DataFrame,
    *,
    path: Path,
    sheet: str | int,
    write: bool,
    source_batch_id: str = SOURCE_BATCH_ID,
) -> int:
    if frame.empty:
        log.warning("No visit-4 rest/clinical rows to write.")
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
                unit=None,
                source_batch_id=source_batch_id,
            ),
        )
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
                    value_kind="float",
                    source_file=path.name,
                    source_sheet=str(sheet),
                    source_column=source_column,
                ),
                provenance={
                    "importer": "import_rest_v4",
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


@click.command("import-rest-v4")
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_SOURCE,
    show_default=False,
    help="Excel workbook containing visit-4 clinical variables.",
)
@click.option("--sheet", default=DEFAULT_SHEET, show_default=True, type=str, help="Worksheet name or zero-based sheet index.")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the path configured in .nvitk/settings.json.",
)
@click.option("--source-batch-id", default=SOURCE_BATCH_ID, show_default=True)
@click.option("--write/--dry-run", default=False, show_default="--dry-run")
def main(source: Path, sheet: str | int, dataset: Path | None, source_batch_id: str, write: bool) -> None:
    """Import visit-4 rest clinical variables into clinical_measurements."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
        mapping = seqn_to_subject(repo)
        frame = extract_rest(Path(source), mapping, sheet=sheet)
        total = publish_rest(
            repo,
            frame,
            path=Path(source),
            sheet=sheet,
            write=write,
            source_batch_id=source_batch_id,
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    action = "Wrote" if write else "Dry run — would write"
    click.echo(f"{action} {total} rest/clinical row(s) for visit 4 into clinical_measurements.")


if __name__ == "__main__":
    main()
