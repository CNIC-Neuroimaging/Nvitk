#!/usr/bin/env python3
"""Import the longitudinal PESA clinical block — every visit — into ``clinical_measurements``.

Description
-----------
Same export as ``import_plaque_v4.py`` (one row per ``seqn`` × ``visit``, v1–v4), but this reads
every visit rather than visit 4 only: labs, demographics, anthropometry and blood pressure,
reproductive history, the CT calcium score, the psychosocial, sleep and physical-activity
questionnaires, accelerometry, the cardiovascular risk-factor flags and the 3D-ultrasound plaque
volumes. ``seqn`` is resolved to ``subject_uid`` through the ``subject_ids`` registry and
``subjects.primary_seqn``, as in the other SEQN importers.

Variable ids
------------
Each column keeps its own name, lowercased (``lbxglu``, ``psqde010``, …) — which is how the
dataset already files ``bpxdim``, ``lbdldl`` or ``psqeduca``. The exceptions are columns the
dataset already holds under another id, reused so the new visits line up with the old ones: the
risk-factor flags (``hipertension`` → ``hypertension``, …) and the plaque volumes
(``e3dburdenc`` → ``total_plaque_vol``, …).

Overwrite
---------
A value is written at ``(subject_uid, visit_id, variable_id)``, and any row already at that key is
removed first, whichever file it came from. Upserting alone would not do it: the table keys on the
source file/sheet/column too, so a value imported from another workbook would survive next to the
new one — and the wide pivot keeps the first row it meets, so it would keep returning the old
value. An empty cell never removes anything: what the workbook leaves blank keeps what the dataset
had.

Values
------
* Censored lab results (``"< 6"``, ``"> 60.00"``) keep the bound in ``value_num`` and the censoring
  in ``value_text`` (``"<6"``, ``">60"``), so they stay in the distribution and can still be found
  and excluded. Dropping them would bias the low end of Lp(a), where they are ~2 % of the values.
* The export stores part of its numbers as float32, which read back as ``0.029999999329447746``.
  Each value that is exactly a float32 is rounded to the shortest decimal float32 prints
  (``0.03``) — the value that was entered; float64 values are left as read.
* Date columns (``lbxdt``, ``deqbirth``, …) are stored as ISO ``YYYY-MM-DD`` text with
  ``value_kind = "date"``. The four exam dates also stamp ``measured_at`` on the block they date
  (see :data:`MEASURED_AT_BLOCKS`); every other block has no date column and stays undated.

Default mode is dry-run.  Use ``--write`` to publish rows.

Examples::

    # Report what would be written and replaced, per variable
    python scripts/database/import_clinical_longitudinal.py --source /path/to/export.xlsx

    # Write visits 3 and 4 only
    python scripts/database/import_clinical_longitudinal.py --source ... --visit 3 --visit 4 --write
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
)
from nvitk.db.importers import read_tabular_source

log = Logger()

DEFAULT_SOURCE = "/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/Brain_10_06_2026.xlsx"
DEFAULT_SHEET = "0"
SOURCE_BATCH_ID = "import_clinical_longitudinal"
TABLE = "clinical_measurements"
#: What "the same measurement" means for the overwrite: the source columns are left out on purpose.
REPLACE_KEY = ["subject_uid", "visit_id", "variable_id"]

#: Source columns to import, in workbook order. ``seqn`` and ``visit`` are the row keys.
COLUMNS: tuple[str, ...] = (
    # Labs — lbxdt is the sample extraction date
    "lbxdt", "lbdbano", "lbdeono", "lbdgfr", "lbxwbcsi", "lbdlymno", "lbdmono", "lbdneno",
    "lbxcystc", "lbxesr1", "lbxfb", "lbxfer", "lbxgh", "lbxglu", "lbxhct", "lbxhdd", "lbdldl",
    "lbxtc", "lbxtr", "lbxhgb", "lbxin", "lbxinr", "lbxlppa", "lbxmchsi", "lbxmcvsi", "lbxoxldl",
    "lbxpltsi", "lbxpselec", "lbxptba", "lbxptbt", "lbxrbcsi", "lbxcot", "lbxsassi", "lbxsatsi",
    "lbxscr", "lbxsgtsi", "lbxvcam1",
    # Demographics
    "deqbirth", "deqrace", "deqsex",
    # Physical exam
    "bmxht", "bmxwaist", "bmxwt", "bpxdim", "bpxpls", "bpxsym",
    # Reproductive history
    "edad_menarquia", "numero_embarazos", "rhq020", "rhq030a", "rhq040",
    "tiempo_tratamiento_pildora",
    # Risk score, exam date, CT calcium score
    "pedscore2", "peqdate", "tacdt", "tacsctot",
    # Psychosocial and sleep
    "psqto000", "slq070b", "slq130", "psdansco", "psddesco", "psdsssco", "psdstsco", "psqdate",
    "psqeduca",
    # Physical activity (IPAQ) and accelerometry
    "ipaq110", "ipaq112", "ipaq120", "ipaq122", "ipaq131", "ipaq133", "ipaq141",
    "inmvpa", "light", "sedentary", "totalmvpa", "vigorous", "stepsavcounts",
    # Biobank status and risk-factor flags
    "idestado_banco", "medicacion_hipertension", "hipertension_anterior", "hipertension",
    "diabetes", "dislipemia", "obesidad",
    # Psychosocial questionnaire items
    *(f"psqde{i:03d}" for i in range(10, 201, 10)),
    *(f"psqst{i:03d}" for i in range(10, 141, 10)),
    # 3D-ultrasound plaque volumes
    "e3dburdenc", "e3dcrawcd", "e3dcrawci", "e3dcrawcsum", "e3dfrawcd", "e3dfrawci", "e3dfrawcsum",
)

#: Columns the dataset already holds under another variable id. Everything else is lowercased.
VARIABLE_IDS: dict[str, str] = {
    "hipertension": "hypertension",
    "dislipemia": "dyslipemia",
    "obesidad": "obesity",
    "e3dburdenc": "total_plaque_vol",
    "e3dcrawcd": "right_carotid_plaque_vol",
    "e3dcrawci": "left_carotid_plaque_vol",
    "e3dcrawcsum": "total_carotid_plaque_vol",
    "e3dfrawcd": "right_femoral_plaque_vol",
    "e3dfrawci": "left_femoral_plaque_vol",
    "e3dfrawcsum": "total_femoral_plaque_vol",
}

#: Stored as ISO date text. A column the workbook types as a date is treated the same way.
DATE_COLUMNS = frozenset({"lbxdt", "deqbirth", "rhq040", "peqdate", "tacdt", "psqdate"})

#: Exam date -> prefixes of the source columns it dates (the date column itself included).
MEASURED_AT_BLOCKS: dict[str, tuple[str, ...]] = {
    "lbxdt": ("lbx", "lbd"),
    "peqdate": ("bmx", "bpx", "ped", "peq"),
    "tacdt": ("tac",),
    "psqdate": ("psq", "psd"),
}

#: Catalog labels for the variables whose meaning is unambiguous; the rest keep their id.
LABELS: dict[str, str] = {
    "lbxdt": "Blood sample extraction date",
    "lbdbano": "Basophils (10^3/µL)",
    "lbdeono": "Eosinophils (10^3/µL)",
    "lbdgfr": "Estimated glomerular filtration rate",
    "lbxwbcsi": "White blood cell count",
    "lbdlymno": "Lymphocytes (10^3/µL)",
    "lbdmono": "Monocytes (10^3/µL)",
    "lbdneno": "Neutrophils (10^3/µL)",
    "lbxcystc": "Cystatin C",
    "lbxesr1": "Erythrocyte sedimentation rate",
    "lbxfb": "Fibrinogen",
    "lbxfer": "Ferritin",
    "lbxgh": "Glycated hemoglobin (HbA1c)",
    "lbxglu": "Glucose",
    "lbxhct": "Hematocrit",
    "lbxhdd": "HDL cholesterol",
    "lbdldl": "LDL cholesterol",
    "lbxtc": "Total cholesterol",
    "lbxtr": "Triglycerides",
    "lbxhgb": "Hemoglobin",
    "lbxin": "Insulin",
    "lbxinr": "Prothrombin INR",
    "lbxlppa": "Lipoprotein(a)",
    "lbxmchsi": "Mean corpuscular hemoglobin",
    "lbxmcvsi": "Mean corpuscular volume",
    "lbxoxldl": "Oxidized LDL",
    "lbxpltsi": "Platelet count",
    "lbxpselec": "P-selectin",
    "lbxptba": "Prothrombin activity",
    "lbxptbt": "Prothrombin time",
    "lbxrbcsi": "Red blood cell count",
    "lbxcot": "Cotinine",
    "lbxsassi": "Aspartate aminotransferase (AST)",
    "lbxsatsi": "Alanine aminotransferase (ALT)",
    "lbxscr": "Creatinine",
    "lbxsgtsi": "Gamma-glutamyl transferase (GGT)",
    "lbxvcam1": "VCAM-1",
    "deqbirth": "Date of birth",
    "bmxht": "Height",
    "bmxwaist": "Waist circumference",
    "bmxwt": "Weight",
    "bpxdim": "Diastolic blood pressure",
    "bpxpls": "Pulse",
    "bpxsym": "Systolic blood pressure",
    "pedscore2": "SCORE2 cardiovascular risk",
    "peqdate": "Physical exam date",
    "tacdt": "CT calcium score date",
    "tacsctot": "Total coronary calcium score",
    "psqdate": "Psychosocial questionnaire date",
    "psqeduca": "Educational attainment (PSQEDUCA, ordinal level)",
    "medicacion_hipertension": "Antihypertensive medication",
    "hipertension_anterior": "Previous hypertension",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "dyslipemia": "Dyslipemia",
    "obesity": "Obesity",
    "total_plaque_vol": "Total plaque volume",
    "right_carotid_plaque_vol": "Right carotid plaque volume",
    "left_carotid_plaque_vol": "Left carotid plaque volume",
    "total_carotid_plaque_vol": "Total carotid plaque volume",
    "right_femoral_plaque_vol": "Right femoral plaque volume",
    "left_femoral_plaque_vol": "Left femoral plaque volume",
    "total_femoral_plaque_vol": "Total femoral plaque volume",
}

#: An optional ``<``/``>`` bound followed by a number with a point or comma decimal.
_BOUNDED_NUMBER = re.compile(r"^\s*([<>]=?)?\s*(\d+(?:[.,]\d+)?)\s*$")


# ---------------------------------------------------------------------------
# SEQN → subject_uid
# ---------------------------------------------------------------------------
def _normalize_seqn(value: Any) -> str:
    """Normalize Excel SEQN values such as 28, 28.0, or ' 28 ' to '28'."""
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
    if text.endswith(".0"):
        text = text[:-2]
    return text


def seqn_to_subject(repo: Any) -> dict[str, str]:
    """Build a SEQN -> subject_uid map from the dataset registries."""
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
        raise ValueError(
            "No SEQN -> subject_uid mapping in this dataset. Populate the 'seqn' namespace "
            "of subject_ids (or subjects.primary_seqn) before running this importer."
        )

    if collisions:
        log.warning("%d SEQN value(s) resolve to multiple subject_uid values; keeping the first.", collisions)

    log.info("Resolved %d SEQN -> subject_uid pair(s).", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
def _parse_sheet(value: str) -> str | int:
    return int(value) if value.strip().isdigit() else value


def _column_map(raw: pd.DataFrame) -> dict[str, Any]:
    return {str(c).strip().lower(): c for c in raw.columns}


def _normalize_visit(value: Any) -> str:
    """``4``, ``4.0``, ``"v4"`` or ``"visit 4"`` -> ``"4"``; ``""`` when there is no visit number."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    match = re.search(r"\d+", str(value))
    return str(int(match.group())) if match else ""


def numeric_values(series: pd.Series) -> tuple[pd.Series, pd.Series, list[str]]:
    """
    ``(value_num, value_text, unreadable)`` for one numeric column.

    Plain numbers pass through. A censored result keeps its bound as the number and the censoring
    as text: ``"< 6"`` -> ``(6.0, "<6")``. A comma decimal (``"33,5"``) is read as a number. Any
    other text is left missing and returned in *unreadable* so the caller can report it.
    """
    value_num = pd.to_numeric(series, errors="coerce").astype("float64")
    value_text = pd.Series(pd.NA, index=series.index, dtype="string")
    leftover = series.notna() & value_num.isna()
    if not leftover.any():
        return value_num, value_text, []

    raw = series[leftover].astype(str)
    parts = raw.str.extract(_BOUNDED_NUMBER)
    bound = pd.to_numeric(parts[1].str.replace(",", ".", regex=False), errors="coerce")
    parsed = bound.notna()
    value_num.loc[bound.index[parsed]] = bound[parsed]
    censored = parsed & parts[0].notna()
    value_text.loc[bound.index[censored]] = [
        f"{op}{value:g}" for op, value in zip(parts.loc[censored, 0], bound[censored])
    ]
    return value_num, value_text, sorted(set(raw[~parsed]))


def drop_float32_noise(values: pd.Series) -> pd.Series:
    """
    Round values stored as float32 to the decimals that were typed (``0.0299999993`` -> ``0.03``).

    Decided per value, because the export mixes both within a column (``0.0299999993`` next to
    ``0.05``). Only a value that is exactly a float32 is touched, and it is replaced by the shortest
    decimal that rounds to the same float32. A decimal typed into float64 is never exactly a
    float32 unless it is dyadic (an integer, ``0.5``), and those print unchanged — so float64
    values pass through as read.
    """
    present = values.dropna()
    if present.empty:
        return values
    as_float64 = present.to_numpy(dtype=np.float64)
    with np.errstate(over="ignore"):
        as_float32 = as_float64.astype(np.float32)
    exact = as_float32.astype(np.float64) == as_float64
    if not exact.any():
        return values
    out = values.copy()
    out.loc[present.index[exact]] = as_float32[exact].astype(str).astype(np.float64)
    return out


def _block_date(column: str) -> str | None:
    """The exam-date column that dates *column*, if any."""
    for date_column, prefixes in MEASURED_AT_BLOCKS.items():
        if column.startswith(prefixes):
            return date_column
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_longitudinal(
    path: Path,
    mapping: dict[str, str],
    *,
    sheet: str | int = DEFAULT_SHEET,
    visits: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Long frame of every valued cell: ``subject_uid``, ``visit_id``, ``variable_id``,
    ``source_column``, ``value_num``, ``value_text``, ``value_kind``, ``measured_at``.

    *visits* restricts the import to those visit numbers; empty means every visit in the workbook.
    """
    raw = read_tabular_source(path, sheet_name=_parse_sheet(str(sheet)))
    cols = _column_map(raw)
    missing = [name for name in ("seqn", "visit", *COLUMNS) if name not in cols]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}")

    work = raw.copy()
    work["_visit"] = work[cols["visit"]].map(_normalize_visit)
    if visits:
        wanted = {_normalize_visit(v) for v in visits}
        work = work.loc[work["_visit"].isin(wanted)]
    work = work.loc[work["_visit"].astype(bool)]
    log.info(
        "%d row(s) across visit(s) %s out of %d total row(s).",
        len(work), ", ".join(sorted(work["_visit"].unique(), key=int)), len(raw),
    )

    work["_seqn"] = work[cols["seqn"]].map(_normalize_seqn)
    work["_subject"] = work["_seqn"].map(mapping)
    unmatched = sorted(set(work.loc[work["_subject"].isna() & work["_seqn"].astype(bool), "_seqn"]))
    if unmatched:
        log.warning(
            "%d SEQN value(s) have no subject_uid match and will be skipped (e.g. %s).",
            len(unmatched), ", ".join(unmatched[:8]),
        )
    work = work.dropna(subset=["_subject"])

    duplicated = work.duplicated(subset=["_subject", "_visit"], keep="last")
    if duplicated.any():
        log.warning(
            "%d subject/visit pair(s) appear on more than one row; keeping the last (e.g. %s).",
            int(duplicated.sum()),
            ", ".join(f"{s} v{v}" for s, v in work.loc[duplicated, ["_subject", "_visit"]].values[:5]),
        )
        work = work.loc[~duplicated]

    dates = {
        column: pd.to_datetime(work[cols[column]], errors="coerce") for column in MEASURED_AT_BLOCKS
    }
    no_date = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")

    frames: list[pd.DataFrame] = []
    for column in COLUMNS:
        series = work[cols[column]]
        variable_id = VARIABLE_IDS.get(column, column)
        if column in DATE_COLUMNS or pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            value_num = pd.Series(np.nan, index=work.index, dtype="float64")
            value_text = parsed.dt.strftime("%Y-%m-%d").astype("string")
            kind = "date"
            unreadable = sorted(set(series[series.notna() & parsed.isna()].astype(str)))
        else:
            value_num, value_text, unreadable = numeric_values(series)
            value_num = drop_float32_noise(value_num)
            kind = "float"
        if unreadable:
            log.warning(
                "%s: %d unreadable value(s) left empty (e.g. %s).",
                column, len(unreadable), ", ".join(repr(v) for v in unreadable[:5]),
            )

        block = _block_date(column)
        frame = pd.DataFrame(
            {
                "subject_uid": work["_subject"].astype("string"),
                "visit_id": work["_visit"].astype("string"),
                "variable_id": variable_id,
                "source_column": str(cols[column]),
                "value_num": value_num,
                "value_text": value_text,
                "value_kind": kind,
                "measured_at": dates[block] if block else no_date,
            },
            index=work.index,
        )
        frame = frame.loc[frame["value_num"].notna() | frame["value_text"].notna()]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def build_rows(
    frame: pd.DataFrame,
    *,
    path: Path,
    sheet: str | int,
    source_batch_id: str = SOURCE_BATCH_ID,
) -> tuple[pd.DataFrame, list[DerivedVariableRegistration]]:
    """``clinical_measurements`` rows for every variable in *frame*, plus their catalog entries."""
    parts: list[pd.DataFrame] = []
    registrations: list[DerivedVariableRegistration] = []
    for variable_id, sub in frame.groupby("variable_id", sort=False):
        sub = sub.reset_index(drop=True)
        source_column = str(sub["source_column"].iloc[0])
        kind = str(sub["value_kind"].iloc[0])
        rows = build_clinical_measurement_rows(
            sub,
            DerivedClinicalMeasurementSpec(
                variable_id=str(variable_id),
                source_file=path.name,
                source_sheet=str(sheet),
                source_column=source_column,
                value_column="value_text" if kind == "date" else "value_num",
                value_kind=kind,
                source_batch_id=source_batch_id,
            ),
        )
        # The spec carries one measured_at for the whole batch, and numeric kinds get an empty
        # value_text; both vary per row here (exam dates, censoring marks), so they are assigned
        # after the build. Positions line up — the builder resets the index and keeps row order.
        rows["measured_at"] = sub["measured_at"].astype("datetime64[ns]")
        if kind != "date":
            rows["value_text"] = sub["value_text"].astype("string")
        parts.append(rows)
        registrations.append(
            DerivedVariableRegistration(
                variable_id=str(variable_id),
                domain="clinical",
                table=TABLE,
                label=LABELS.get(str(variable_id)),
                value_kind=kind,
                source_file=path.name,
                source_sheet=str(sheet),
                source_column=source_column,
            )
        )
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return rows, registrations


def superseded(existing: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """True for each *existing* row sitting on a ``(subject_uid, visit_id, variable_id)`` *rows* writes."""
    if existing.empty or rows.empty:
        return pd.Series(False, index=existing.index)
    incoming = pd.MultiIndex.from_frame(rows[REPLACE_KEY].astype("string").fillna(""))
    current = pd.MultiIndex.from_frame(existing[REPLACE_KEY].astype("string").fillna(""))
    return pd.Series(current.isin(incoming), index=existing.index)


def publish_replacing(
    repo: Any,
    rows: pd.DataFrame,
    registrations: list[DerivedVariableRegistration],
    *,
    write: bool,
    provenance: dict[str, Any],
) -> pd.Series:
    """
    Replace whatever the table holds at the keys *rows* writes, then add *rows* — in one write.

    Returns the number of replaced rows per variable. Reads Parquet rather than SQLite, which may
    lag a previous write that skipped the index rebuild.
    """
    existing = repo.get(TABLE, cohort_id=False, use_sqlite=False)
    stale = superseded(existing, rows)
    replaced = existing.loc[stale, "variable_id"].astype(str).value_counts()
    if not write or rows.empty:
        return replaced

    combined = pd.concat([existing.loc[~stale], rows], ignore_index=True)
    repo.write_table(
        TABLE,
        combined,
        provenance={**provenance, "rows_replaced": int(stale.sum()), "rows_written": len(rows)},
        build_sqlite_index=False,
    )
    repo.register_variables([entry.to_catalog_entry() for entry in registrations])
    # Once, after the catalog entries exist — the index is derived from Parquet and the catalog.
    repo.build_sqlite_index(tables=[TABLE])
    return replaced


def summarize(frame: pd.DataFrame, rows: pd.DataFrame, replaced: pd.Series) -> pd.DataFrame:
    """Per-variable rows by visit, censored and dated counts, and how many existing rows it replaces."""
    by_visit = (
        rows.assign(visit_id="v" + rows["visit_id"].astype(str))
        .pivot_table(index="variable_id", columns="visit_id", values="subject_uid", aggfunc="size", fill_value=0)
    )
    numeric = frame["value_kind"] != "date"
    extra = pd.DataFrame(
        {
            "censored": frame.loc[numeric & frame["value_text"].notna()].groupby("variable_id").size(),
            "dated": rows.loc[rows["measured_at"].notna()].groupby("variable_id").size(),
            "replaces": replaced,
        }
    )
    order = list(dict.fromkeys(frame["variable_id"]))
    out = by_visit.join(extra, how="left").fillna(0).astype(int).reindex(order)
    out.columns.name = None
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-clinical-longitudinal")
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SOURCE or None,
    required=not DEFAULT_SOURCE,
    help="Workbook with one row per seqn × visit (same export as import_plaque_v4).",
)
@click.option("--sheet", default=DEFAULT_SHEET, show_default=True, type=str, help="Worksheet name or zero-based sheet index.")
@click.option(
    "--visit",
    "visits",
    multiple=True,
    help="Only import this visit (repeatable: --visit 3 --visit 4). Default: every visit.",
)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the path configured in .nvitk/settings.json.",
)
@click.option("--source-batch-id", default=SOURCE_BATCH_ID, show_default=True)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually publish into clinical_measurements; default is dry-run.",
)
def main(
    source: Path,
    sheet: str,
    visits: tuple[str, ...],
    dataset: Path | None,
    source_batch_id: str,
    write: bool,
) -> None:
    """Import every visit of the longitudinal clinical block into clinical_measurements."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
        mapping = seqn_to_subject(repo)
        frame = extract_longitudinal(Path(source), mapping, sheet=sheet, visits=visits)
        if frame.empty:
            raise ValueError("No valued cells to import.")
        rows, registrations = build_rows(
            frame, path=Path(source), sheet=sheet, source_batch_id=source_batch_id
        )
        replaced = publish_replacing(
            repo,
            rows,
            registrations,
            write=write,
            provenance={"importer": "import_clinical_longitudinal", "source_file": Path(source).name},
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    with pd.option_context("display.max_rows", None, "display.width", 200):
        click.echo(summarize(frame, rows, replaced).to_string())
    action = "Wrote" if write else "Dry run — would write"
    click.echo(
        f"\n{action} {len(rows)} row(s) across {rows['variable_id'].nunique()} variable(s), "
        f"replacing {int(replaced.sum())} existing row(s) at the same subject/visit/variable."
    )
    if not write:
        click.echo("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
