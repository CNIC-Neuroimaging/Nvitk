#!/usr/bin/env python3
"""
Insert ``sex`` into ``clinical_measurements`` as a numeric variable (0 = female, 1 = male).

Sex is only stored as a per-subject identifier (``subject_ids`` rows with ``id_namespace='sex'``)
and on the ``subjects`` entity table, so it never reaches ``repo.clinical()`` and cannot be used as
a model covariate in the Statmodels tool. This script derives the encoded variable and upserts it
into ``clinical_measurements`` without touching any other variable's rows: the upsert de-duplicates
on the table's manifest key columns, so only rows whose ``variable_id`` is ``sex`` (with the source
triple below) are added or replaced.

One row is written per subject, at a single ``--visit`` (default ``"4"``, the visit that carries
age_at_mri / hematocrit / apoe / bmi and every other subject-level clinical variable in this
dataset). Sex is time-invariant, so the visit is only a placement choice — but it must match the
other covariates: ``repo.clinical(wide=True)`` pivots on ``(subject_uid, visit_id)``, and the
analysis-frame merge keeps one row per subject, so sex written at a *different* visit than the rest
would produce a second, otherwise-empty row and silently blank out the other covariates.

Re-running is idempotent: the key columns are stable, so a second run replaces its own rows rather
than appending duplicates.

Example::

    python scripts/database/import_sex.py --dry-run
    python scripts/database/import_sex.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import DEFAULT_VISIT_LABEL
from nvitk.db.repo import DataRepo, get_repo, get_repo_from_settings

VARIABLE_ID = "sex"
SEX_ID_NAMESPACE = "sex"

# Part of the ``clinical_measurements`` key tuple (subject_uid, visit_id, variable_id, source_file,
# source_sheet, source_column) — keep stable so re-runs replace these rows instead of duplicating.
SOURCE_FILE = "derived"
SOURCE_SHEET = "sex"
SOURCE_COLUMN = "sex"
SOURCE_BATCH_ID = "derived_sex"

# 0 = female, 1 = male.
FEMALE, MALE = 0.0, 1.0
_SEX_TOKENS: dict[str, float] = {
    "male": MALE,
    "m": MALE,
    "man": MALE,
    "hombre": MALE,
    "varon": MALE,
    "1": MALE,
    "female": FEMALE,
    "f": FEMALE,
    "woman": FEMALE,
    "mujer": FEMALE,
    "0": FEMALE,
}


def _open_repo(dataset_root: Path | None) -> DataRepo:
    """Open the dataset at *dataset_root*, or the one configured in ``.nvitk/settings.json``."""
    if dataset_root is not None:
        return get_repo(root=dataset_root)
    got = get_repo_from_settings()
    return got[0] if isinstance(got, tuple) else got


def _encode_sex(value: Any) -> float | None:
    """Encode a raw sex label (``\"Male\"``, ``\"F\"``, ``1``, …) as 1.0 = male / 0.0 = female, or
    ``None`` when the value is missing or unrecognized."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return None
    text = str(value).strip().lower()
    if not text or text in {"na", "nan", "none", "<na>", "unknown", "u"}:
        return None
    if text.endswith(".0"):  # numeric-looking labels round-tripped through float
        text = text[:-2]
    return _SEX_TOKENS.get(text)


def collect_sex(repo: DataRepo) -> tuple[pd.Series, dict[str, int]]:
    """
    One encoded sex value per ``subject_uid``, gathered from ``subject_ids`` then ``subjects``.

    Returns the ``subject_uid -> {0.0, 1.0}`` series plus per-source counts. Raises if any source
    holds a non-empty label that :func:`_encode_sex` does not recognize, so unmapped categories are
    never silently dropped.
    """
    resolved: dict[str, float] = {}
    unmapped: dict[str, int] = {}
    counts: dict[str, int] = {}

    def absorb(pairs: Iterable[tuple[Any, Any]], source: str) -> None:
        """Encode *pairs* of ``(subject_uid, raw_label)``, keeping the first value seen per subject."""
        added = 0
        for subject_uid, raw in pairs:
            uid = str(subject_uid).strip()
            if not uid or uid in resolved:
                continue
            encoded = _encode_sex(raw)
            if encoded is None:
                if raw is not None and str(raw).strip() and not pd.isna(raw):
                    token = str(raw).strip()
                    unmapped[token] = unmapped.get(token, 0) + 1
                continue
            resolved[uid] = encoded
            added += 1
        counts[source] = added

    if repo.catalog.table_exists("subject_ids"):
        sid = repo.get("subject_ids", cohort_id=False)
        if not sid.empty and {"subject_uid", "id_namespace", "id_value"} <= set(sid.columns):
            rows = sid[sid["id_namespace"].astype("string") == SEX_ID_NAMESPACE]
            absorb(zip(rows["subject_uid"], rows["id_value"]), "subject_ids")

    if repo.catalog.table_exists("subjects"):
        subjects = repo.get("subjects", cohort_id=False)
        if not subjects.empty and "sex" in subjects.columns:
            absorb(zip(subjects["subject_uid"], subjects["sex"]), "subjects")

    if unmapped:
        detail = ", ".join(f"{tok!r} ({n})" for tok, n in sorted(unmapped.items(), key=lambda kv: -kv[1]))
        raise ValueError(
            f"Unrecognized sex labels: {detail}. Add them to _SEX_TOKENS before importing."
        )

    series = pd.Series(resolved, dtype="float64")
    series.index.name = "subject_uid"
    return series.sort_index(), counts


def _align_dtypes(rows: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """
    Cast *rows* to the dtypes the table already uses, so appending them cannot rewrite the existing
    rows' storage types.

    Without this, a freshly built ``measured_at`` (``datetime64[ns]``) concatenated with a table
    stored as ``datetime64[ms]`` promotes the whole column — the values survive, but every
    pre-existing row is rewritten with a different dtype.
    """
    if existing.empty:
        return rows
    out = rows.copy()
    for column in out.columns:
        if column not in existing.columns:
            continue
        target = existing[column].dtype
        if out[column].dtype == target:
            continue
        try:
            out[column] = out[column].astype(target)
        except (TypeError, ValueError):
            pass
    return out


def build_rows(repo: DataRepo, sex: pd.Series, *, visit: str) -> pd.DataFrame:
    """Long-form ``clinical_measurements`` rows for *sex*: one row per subject at *visit*."""
    if sex.empty:
        return pd.DataFrame()

    clinical = pd.DataFrame()
    if repo.catalog.table_exists("clinical_measurements"):
        clinical = repo.get("clinical_measurements", cohort_id=False)

    agg = pd.DataFrame(
        {
            "subject_uid": pd.Series(list(sex.index), dtype="string"),
            "visit_id": pd.Series([visit] * len(sex), dtype="string"),
            "value_num": pd.Series(sex.to_numpy(), dtype="float64"),
        }
    )

    rows = build_clinical_measurement_rows(
        agg,
        DerivedClinicalMeasurementSpec(
            variable_id=VARIABLE_ID,
            source_file=SOURCE_FILE,
            source_sheet=SOURCE_SHEET,
            source_column=SOURCE_COLUMN,
            value_kind="numeric",
            source_batch_id=SOURCE_BATCH_ID,
            source_table=f"{SOURCE_FILE}::{SOURCE_SHEET}",
        ),
    )
    return _align_dtypes(rows, clinical)


def registration() -> DerivedVariableRegistration:
    """Catalog entry that makes ``sex`` resolvable by ``repo.clinical(variables=[\"sex\"])`` and
    visible in the Statmodels clinical-covariate list."""
    return DerivedVariableRegistration(
        variable_id=VARIABLE_ID,
        domain="clinical",
        table="clinical_measurements",
        label="Sex (0 = female, 1 = male)",
        source_column=SOURCE_COLUMN,
        source_file=SOURCE_FILE,
        source_sheet=SOURCE_SHEET,
        aliases=["sex", "Sex", "gender", "Gender"],
        value_kind="numeric",
    )


def verify(repo: DataRepo) -> None:
    """Read the freshly written variable back through the public API and print a coverage summary."""
    wide = repo.clinical(variables=[VARIABLE_ID], wide=True, cohort_id=False)
    if wide.empty or VARIABLE_ID not in wide.columns:
        print("  ! verification failed: repo.clinical(variables=['sex']) returned nothing")
        return
    values = pd.to_numeric(wide[VARIABLE_ID], errors="coerce")
    per_subject = (
        wide.assign(**{VARIABLE_ID: values})
        .dropna(subset=[VARIABLE_ID])
        .drop_duplicates(subset=["subject_uid"])[VARIABLE_ID]
    )
    print(f"  rows via repo.clinical : {len(wide)}")
    print(f"  subjects with sex      : {len(per_subject)}")
    print(
        f"  subjects male / female : {int((per_subject == MALE).sum())} / "
        f"{int((per_subject == FEMALE).sum())}"
    )
    unexpected = values.dropna()
    unexpected = unexpected[~unexpected.isin([FEMALE, MALE])]
    if len(unexpected):
        print(f"  ! {len(unexpected)} rows are neither 0 nor 1")


def main(argv: Iterable[str] | None = None) -> int:
    """Parse arguments, build the encoded ``sex`` rows, upsert them, and rebuild the SQLite index."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root (default: the one in .nvitk/settings.json)",
    )
    parser.add_argument(
        "--visit",
        type=str,
        default=DEFAULT_VISIT_LABEL,
        help=f"visit_id to write sex at; must match the other clinical covariates "
             f"(default: {DEFAULT_VISIT_LABEL!r})",
    )
    parser.add_argument(
        "--full-index",
        action="store_true",
        help="Rebuild the SQLite index for every table instead of just clinical_measurements",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the plan without writing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = _open_repo(args.dataset_root)
    print(f"Dataset: {repo.root}")

    sex, counts = collect_sex(repo)
    if sex.empty:
        print("No sex values found in subject_ids or subjects — nothing to import.")
        return 1
    for source, added in counts.items():
        print(f"  from {source:<12}: {added} subjects")
    print(f"  subjects total      : {len(sex)} "
          f"(male {int((sex == MALE).sum())}, female {int((sex == FEMALE).sum())})")

    rows = build_rows(repo, sex, visit=args.visit)
    if rows.empty:
        print("No rows built — nothing to import.")
        return 1
    print(f"  rows to upsert      : {len(rows)} at visit_id={args.visit!r}")

    if args.dry_run:
        print("\n--dry-run: nothing written. Preview:")
        print(rows.head(5).to_string(index=False))
        return 0

    publish_derived_measurements(
        repo,
        rows,
        table="clinical_measurements",
        register=registration(),
        provenance={"importer": "import_sex", "encoding": "0=female,1=male"},
        build_sqlite_index=False,  # rebuilt below, after the catalog entry is registered
    )
    print(f"Upserted {len(rows)} '{VARIABLE_ID}' rows into clinical_measurements.")

    tables = None if args.full_index else ["clinical_measurements"]
    repo.build_sqlite_index(tables=tables)
    print(f"Rebuilt SQLite index ({'all tables' if tables is None else 'clinical_measurements'}).")

    repo.catalog.refresh()
    print("Verification:")
    verify(repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
