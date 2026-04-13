from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError

from .repo import DEFAULT_COHORT_ID, DataRepo
from .storage import json_dumps, normalize_string, normalize_variable_id, utc_now_iso


def _resolve_import_pipeline_id(repo: DataRepo, modality: str) -> str | None:
    """Prefer ``NVITK_PIPELINE_ID_<MODALITY>`` or ``NVITK_PIPELINE_ID``, else catalog default for modality."""
    mod_key = str(modality).strip().upper().replace("-", "_")
    env_val = os.getenv(f"NVITK_PIPELINE_ID_{mod_key}") or os.getenv("NVITK_PIPELINE_ID")
    if env_val and str(env_val).strip():
        return str(env_val).strip()
    return repo.catalog.default_pipeline_id(str(modality))


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator

SUBJECT_UID_CANDIDATES = [
    "subject_uid",
    "patient_id",
    "patientid",
    "codigoimagen",
    "codigoimagen_x",
    "subject",
    "subject_id",
    "pesa",
    "seqn",
    "codi sub.",
    "codisub",
]
VISIT_CANDIDATES = ["visit_id", "visit", "visit_label", "visit_number", "visita"]
SESSION_CANDIDATES = ["session_uid", "session_id", "mri_id", "mr id", "experiment_label", "session"]
DATE_CANDIDATES = ["date", "imagingdate", "measured_at", "scan_date", "session_date", "psqdate_reconocimiento_medico"]


def _image_measurement_session_id_series(raw: pd.DataFrame) -> pd.Series:
    """Return a per-row session id for ``image_measurements`` with a stable empty-string sentinel.

    Subject-only Excel sheets have no session column. Using ``pd.NA`` for every row makes
    ``drop_duplicates`` / merges treat rows inconsistently after Parquet round-trips, which
    shows up as duplicated subjects. Missing or blank session is stored as ``""``.
    """
    column = _first_matching_column(raw, SESSION_CANDIDATES)
    if column is None:
        return pd.Series([""] * len(raw), dtype="string")
    return raw[column].astype("string").fillna("").str.strip().astype("string")


def _column_name_is_timeseries_frame_index(name: object) -> bool:
    """True if a column label is a non-negative integer frame index (0, 1, 2 or 0.0, '3', …)."""
    if isinstance(name, (int, np.integer)):
        return int(name) >= 0
    if isinstance(name, (float, np.floating)):
        if np.isnan(name):
            return False
        v = float(name)
        return v >= 0 and v == int(v)
    s = str(name).strip()
    if s.isdigit():
        return True
    try:
        v = float(s)
        return v >= 0 and v == int(v)
    except ValueError:
        return False


def _timeseries_wide_frame_columns(all_columns: list, *, reserved: set[str]) -> list:
    out: list = []
    for column in all_columns:
        if column in reserved:
            continue
        label = str(column)
        if label.startswith("Unnamed"):
            continue
        if _column_name_is_timeseries_frame_index(column):
            out.append(column)
    return out
ID_NAMESPACE_EXACT = {
    "patient_id",
    "patientid",
    "subject",
    "subject_id",
    "subject_uid",
    "session",
    "session_id",
    "session_uid",
    "visit",
    "visit_id",
    "seqn",
    "mri_id",
    "mr_id",
    "mrid",
    "medrecon_id",
    "fert_id",
    "psicosocial_id",
    "codigoimagen",
    "codigoimagen_x",
    "codi_sub",
    "pesa",
}
LOCAL_PROJECT_ID = "local_db"
DEFAULT_VISIT_LABEL = '4'

CLINICAL_METADATA_COLUMNS = {
    "age_at_mri",
    "sex",
    "weight",
    "height",
    "bmi",
    "bpxsym",
    "bpxdim",
    "bpxpls",
    "tas",
    "tad",
    "sys_dias_delta",
    "hematocrit",
    "psqto000",
    "lbxhdd",
    "lbdldl",
    "lbxtc",
    "score2",
    "pedframi10",
    "pedframi30",
    "visit",
    "tacsctot",
    "tacsctot_group",
    "total_plaque_vol",
    "left_carotid_plaque_vol",
    "right_carotid_plaque_vol",
    "total_femoral_plaque_vol",
    "total_carotid_plaque_vol",
    "apoe",
    "age_c",
}


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    sheet: str
    source_kind: str
    domain: str
    layout: str
    modality: str | None = None
    cohort_id: str | None = None
    batch_id: str | None = None
    default_visit_label: str | None = None
    default_pipeline_id: str | None = None


PESABRAIN_DB_SPECS: dict[str, list[SourceSpec]] = {
    "PESABrain_All_IDs.xlsx": [
        SourceSpec("PESABrain_All_IDs.xlsx", "Sheet1", "subject_ids", "metadata", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_All_4DFlow_IDs.xlsx": [
        SourceSpec("PESABrain_All_4DFlow_IDs.xlsx", "Sheet1", "subject_ids", "metadata", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx": [
        SourceSpec("PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx", "Datos", "subject_catalog", "metadata", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_Clinical_20260216.xlsx": [
        SourceSpec("PESABrain_Clinical_20260216.xlsx", "Sheet1", "clinical_wide", "clinical", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_Clinical_AllXNAT_20260216.xlsx": [
        SourceSpec("PESABrain_Clinical_AllXNAT_20260216.xlsx", "Datos", "clinical_wide", "clinical", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_APOE_20260318.xlsx": [
        SourceSpec("PESABrain_APOE_20260318.xlsx", "Sheet1", "clinical_wide", "clinical", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_TAC_20260318.xlsx": [
        SourceSpec("PESABrain_TAC_20260318.xlsx", "Sheet1", "clinical_wide", "clinical", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_Echography_CarotidePlaque_20260216.xlsx": [
        SourceSpec("PESABrain_Echography_CarotidePlaque_20260216.xlsx", "Sheet1", "clinical_wide", "clinical", "wide", cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_4DFlow_LocalizedPI_20260216.xlsx": [
        SourceSpec("PESABrain_4DFlow_LocalizedPI_20260216.xlsx", "PESABrain_AnalysisDB_Batch1", "image_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v1', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx": [
        SourceSpec("PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx", "PESABrain_AnalysisDB_Batch1", "image_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v1', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx": [
        SourceSpec("PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx", "Datos", "image_timeseries_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v1', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_4DFlowv2_LocalizedPI_202602410.xlsx": [
        SourceSpec("PESABrain_4DFlowv2_LocalizedPI_202602410.xlsx", "Sheet1", "image_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v2', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_4DFlowv2_LocalizedTimeAvgFlow_202602410.xlsx": [
        SourceSpec("PESABrain_4DFlowv2_LocalizedTimeAvgFlow_202602410.xlsx", "Sheet1", "image_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v2', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_4DFlowv2_LocalizedTimeseriesFlow_Wide_202602410.xlsx": [
        SourceSpec("PESABrain_4DFlowv2_LocalizedTimeseriesFlow_Wide_202602410.xlsx", "Sheet1", "image_timeseries_wide", "image", "wide", modality="4dflow", default_pipeline_id='4dflow_v2', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="4DFlow-Processed"),
    ],
    "PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx": [
        SourceSpec("PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx", "Sheet1", "image_wide", "image", "wide", modality="asl", cohort_id="PESA-Brain", default_pipeline_id='asl_v1', default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_ASLPerfusion_CovCBF_20260216.xlsx": [
        SourceSpec("PESABrain_ASLPerfusion_CovCBF_20260216.xlsx", "Sheet1", "image_wide", "image", "wide", modality="asl", cohort_id="PESA-Brain", default_pipeline_id='asl_v1', default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
    "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx": [
        SourceSpec("PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx", "Sheet1", "image_wide", "image", "wide", modality="asl", default_pipeline_id='asl_v1', cohort_id="PESA-Brain", default_visit_label=DEFAULT_VISIT_LABEL, batch_id="All"),
    ],
}


def list_pesabrain_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename, specs in sorted(PESABRAIN_DB_SPECS.items()):
        for spec in specs:
            rows.append(
                {
                    "filename": filename,
                    "sheet": spec.sheet,
                    "source_kind": spec.source_kind,
                    "domain": spec.domain,
                    "layout": spec.layout,
                    "modality": spec.modality,
                    "cohort_id": spec.cohort_id,
                    "batch_id": spec.batch_id,
                    "default_visit_label": spec.default_visit_label,
                }
            )
    return rows


def _matching_pesabrain_specs(
    filename: str,
    *,
    sheet: str | None = None,
    source_kind: str | None = None,
) -> list[SourceSpec]:
    matches = list(PESABRAIN_DB_SPECS.get(filename, []))
    if sheet is not None:
        matches = [spec for spec in matches if spec.sheet == sheet]
    if source_kind is not None:
        matches = [spec for spec in matches if spec.source_kind == source_kind]
    return matches


def _default_pesabrain_batch_id() -> str:
    return f"pesabrain_{utc_now_iso().replace(':', '').replace('-', '').replace('+00:00', 'z')}"


def _is_identifier_column(column: str) -> bool:
    normalized = normalize_variable_id(str(column))
    if normalized in ID_NAMESPACE_EXACT:
        return True
    if normalized.endswith("_id") or normalized.endswith("_uid"):
        return True
    if normalized.startswith("codigo") or normalized.startswith("codigoprueba"):
        return True
    return False


def _region_id(value: str | None) -> str | None:
    text = normalize_string(value)
    if text is None:
        return None
    return normalize_variable_id(text)


def read_tabular_source(path: str | Path, *, sheet_name: str | int = 0, header: int = 0) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet_name, header=header)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported source format: {source}")


def list_excel_sources(base_path: str | Path) -> list[Path]:
    base = Path(base_path)
    return sorted([path for path in base.iterdir() if path.suffix.lower() in {".xlsx", ".xls"}])


def _normalized_column_map(columns: list[str]) -> dict[str, str]:
    return {re.sub(r"[^0-9a-z]+", "", str(column).lower()): str(column) for column in columns}


def _first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = _normalized_column_map(list(df.columns))
    for candidate in candidates:
        key = re.sub(r"[^0-9a-z]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        sample = numeric.dropna()
        if not sample.empty and sample.between(20000, 60000).all():
            return pd.to_datetime(numeric, errors="coerce", unit="D", origin="1899-12-30")
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        numeric = pd.to_numeric(series.astype("string").str.replace(",", ".", regex=False), errors="coerce")
    return numeric.astype("float64")


def _series_value_payload(series: pd.Series) -> tuple[pd.Series, pd.Series, str]:
    numeric = _coerce_numeric(series)
    observed = int(series.notna().sum())
    numeric_ratio = float(numeric.notna().sum()) / float(observed) if observed else 0.0
    value_kind = "numeric" if numeric_ratio >= 0.70 else "text"
    if value_kind == "numeric":
        return numeric, pd.Series([pd.NA] * len(series), dtype="string"), value_kind
    return (
        pd.Series([np.nan] * len(series), dtype="float64"),
        series.astype("string").where(series.notna(), pd.NA),
        value_kind,
    )


def ensure_subject_uid(df: pd.DataFrame, *, candidates: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "subject_uid" in out.columns:
        out["subject_uid"] = out["subject_uid"].astype("string")
        return out

    subject_column = _first_matching_column(out, candidates or SUBJECT_UID_CANDIDATES)
    if subject_column is not None:
        out["subject_uid"] = out[subject_column].astype("string")
    else:
        out["subject_uid"] = pd.Series([f"row_{index:06d}" for index in range(len(out))], dtype="string")
    return out


def _source_table_name(source_path: Path, sheet_name: str) -> str:
    return f"{source_path.stem}::{sheet_name}"


def _inventory_row(source_path: Path, sheet_name: str, df: pd.DataFrame, spec: SourceSpec, source_batch_id: str) -> dict[str, Any]:
    return {
        "source_uid": f"{source_path.name}:{sheet_name}:{spec.source_kind}",
        "source_file": source_path.name,
        "source_sheet": sheet_name,
        "source_kind": spec.source_kind,
        "domain": spec.domain,
        "modality": spec.modality or pd.NA,
        "layout": spec.layout,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns_json": json_dumps(list(map(str, df.columns))),
        "source_batch_id": source_batch_id,
        "updated_at": utc_now_iso(),
    }


def _register_inventory_rows(repo: DataRepo, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    repo.upsert_table(
        "source_tables",
        df,
        provenance={"importer": "import_pesabrain_db_directory"},
    )
    return df


def _variable_entry(
    *,
    variable_id: str,
    source_column: str,
    domain: str,
    table: str,
    modality: str | None = None,
    value_kind: str | None = None,
    label: str | None = None,
    source_file: str | None = None,
    source_sheet: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    aliases = [source_column]
    return {
        "variable_id": variable_id,
        "source_column": source_column,
        "source_file": source_file,
        "source_sheet": source_sheet,
        "aliases": aliases,
        "domain": domain,
        "table": table,
        "modality": modality,
        "label": label or source_column,
        "value_kind": value_kind,
        **extra,
    }


def _upsert_subject_ids(repo: DataRepo, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    repo.upsert_table(
        "subject_ids",
        df,
        provenance={"importer": "import_pesabrain_db_directory"},
    )
    return df


def _upsert_sessions(repo: DataRepo, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    repo.upsert_table(
        "sessions",
        df,
        provenance={"importer": "import_pesabrain_db_directory"},
    )
    return df


def harvest_subject_ids_from_frame(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    id_source: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    now = utc_now_iso()
    for _, row in raw.iterrows():
        subject_uid = normalize_string(row.get("subject_uid"))
        if subject_uid is None:
            continue
        for column in raw.columns:
            if column == "subject_uid":
                continue
            if not _is_identifier_column(str(column)):
                continue
            normalized = normalize_variable_id(str(column))
            value = normalize_string(row.get(column))
            if value is None:
                continue
            rows.append(
                {
                    "subject_uid": subject_uid,
                    "id_namespace": normalized,
                    "id_value": value,
                    "id_source": id_source or _source_table_name(source_path, sheet_name),
                    "is_primary": normalized in {"patient_id", "seqn", "subject", "subject_id", "codigoimagen"},
                    "source_batch_id": source_batch_id,
                    "updated_at": now,
                }
            )
    return _upsert_subject_ids(repo, pd.DataFrame(rows))


def import_subject_ids_from_source(
    repo: DataRepo,
    source_path: str | Path,
    *,
    source_batch_id: str,
    id_source: str | None = None,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    raw = ensure_subject_uid(read_tabular_source(source_path, sheet_name=sheet_name))
    return harvest_subject_ids_from_frame(
        repo,
        raw,
        source_path=Path(source_path),
        sheet_name=sheet_name,
        source_batch_id=source_batch_id,
        id_source=id_source,
    )


def import_cohort_membership_from_source(
    repo: DataRepo,
    source_path: str | Path,
    *,
    cohort_id: str,
    source_batch_id: str,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    raw = ensure_subject_uid(read_tabular_source(source_path, sheet_name=sheet_name))
    df = pd.DataFrame(
        {
            "cohort_id": cohort_id,
            "subject_uid": raw["subject_uid"].astype("string"),
            "membership_source": _source_table_name(Path(source_path), sheet_name),
            "source_batch_id": source_batch_id,
            "updated_at": utc_now_iso(),
        }
    ).dropna(subset=["subject_uid"])
    repo.upsert_table(
        "cohort_membership",
        df,
        provenance={"source_path": str(source_path), "importer": "import_cohort_membership_from_source"},
    )
    return df


def upsert_cohort_membership_for_subjects(
    repo: DataRepo,
    cohort_id: str,
    subject_uids: Iterable[str],
    *,
    source_batch_id: str,
    membership_source: str = "import",
) -> pd.DataFrame:
    """Register ``subject_uid`` values as members of ``cohort_id`` (upserts ``cohort_membership``)."""
    uids = sorted({str(u).strip() for u in subject_uids if u is not None and str(u).strip()})
    if not uids:
        return pd.DataFrame()
    now = utc_now_iso()
    df = pd.DataFrame(
        {
            "cohort_id": cohort_id,
            "subject_uid": uids,
            "membership_source": membership_source,
            "source_batch_id": source_batch_id,
            "updated_at": now,
        }
    )
    return repo.upsert_table(
        "cohort_membership",
        df,
        provenance={"cohort_id": cohort_id, "importer": "upsert_cohort_membership_for_subjects"},
    )


def _collect_subject_uids_from_import_result(result: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(result, pd.DataFrame):
        if "subject_uid" in result.columns:
            out.update(result["subject_uid"].dropna().astype(str).str.strip().tolist())
        return {u for u in out if u}
    if isinstance(result, dict):
        for v in result.values():
            out |= _collect_subject_uids_from_import_result(v)
        return {u for u in out if u}
    if isinstance(result, (list, tuple)):
        for item in result:
            out |= _collect_subject_uids_from_import_result(item)
        return {u for u in out if u}
    return set()


def import_measurements_from_source(
    repo: DataRepo,
    source_path: str | Path,
    *,
    kind: str,
    source_batch_id: str,
    modality: str | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    source = Path(source_path)
    raw = read_tabular_source(source, sheet_name=sheet_name)
    resolved_sheet = str(sheet_name)
    if kind == "clinical":
        return _parse_generic_clinical_wide(
            repo,
            raw,
            source_path=source,
            sheet_name=resolved_sheet,
            source_batch_id=source_batch_id,
        )
    if kind == "image":
        return _parse_generic_image_wide(
            repo,
            raw,
            source_path=source,
            sheet_name=resolved_sheet,
            source_batch_id=source_batch_id,
            modality=modality or "image",
        )
    raise ValueError(f"Unsupported measurement kind: {kind}")


def harvest_sessions_from_frame(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    modality: str | None,
    default_visit_label: str | None = None,
) -> pd.DataFrame:
    session_column = _first_matching_column(raw, SESSION_CANDIDATES)
    if session_column is None:
        return pd.DataFrame()

    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    visit_column = _first_matching_column(raw, VISIT_CANDIDATES)
    scanner_column = _first_matching_column(raw, ["scanner"])
    scans_column = _first_matching_column(raw, ["scans"])

    frame = pd.DataFrame(
        {
            "session_uid": raw[session_column].astype("string"),
            "subject_uid": raw["subject_uid"].astype("string"),
            "project_id": LOCAL_PROJECT_ID,
            "experiment_label": raw[session_column].astype("string"),
            "modality": modality or "mr",
            "visit_label": raw[visit_column].astype("string") if visit_column else pd.Series([default_visit_label if default_visit_label else pd.NA] * len(raw), dtype="string"),
            "scanner": raw[scanner_column].astype("string") if scanner_column else pd.Series([pd.NA] * len(raw), dtype="string"),
            "available_scans": raw[scans_column].astype("string") if scans_column else pd.Series([pd.NA] * len(raw), dtype="string"),
            "acquired_at": _parse_datetime_series(raw[date_column]) if date_column else pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]"),
            "source_file": source_path.name,
            "source_sheet": sheet_name,
            "source_batch_id": source_batch_id,
            "updated_at": utc_now_iso(),
        }
    )
    frame = frame.dropna(subset=["session_uid", "subject_uid"]).drop_duplicates(subset=["session_uid"], keep="last")
    return _upsert_sessions(repo, frame)


def _upsert_measurements(repo: DataRepo, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    repo.upsert_table(
        table_name,
        df,
        provenance={"importer": "import_pesabrain_db_directory"},
    )
    return df


def _register_variables(repo: DataRepo, entries: list[dict[str, Any]]) -> None:
    if entries:
        repo.register_variables(entries)


def _clinical_frame(
    raw: pd.DataFrame,
    *,
    visit_column: str | None,
    date_column: str | None,
    column: str,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    variable_id: str | None = None,
    unit: str | None = None,
    default_visit_label: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    value_num, value_text, value_kind = _series_value_payload(raw[column])
    variable = _variable_entry(
        variable_id=variable_id or normalize_variable_id(column),
        source_column=column,
        domain="clinical",
        table="clinical_measurements",
        label=column,
        value_kind=value_kind,
        unit=unit,
        source_file=source_path.name,
        source_sheet=sheet_name,
    )
    frame = pd.DataFrame(
        {
            "subject_uid": raw["subject_uid"].astype("string").where(raw["subject_uid"].notna(), pd.NA),
            "visit_id": raw[visit_column].astype("string") if visit_column else pd.Series([DEFAULT_VISIT_LABEL] * len(raw), dtype="string"),
            "variable_id": variable_id or normalize_variable_id(column),
            "value_num": value_num,
            "value_text": value_text,
            "unit": pd.Series([unit if unit else pd.NA] * len(raw), dtype="string"),
            "value_kind": value_kind,
            "source_table": _source_table_name(source_path, sheet_name),
            "source_file": source_path.name,
            "source_sheet": sheet_name,
            "source_column": column,
            "source_batch_id": source_batch_id,
            "measured_at": _parse_datetime_series(raw[date_column]) if date_column else pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]"),
        }
    )
    if frame['visit_id'].isna().any():
        frame['visit_id'] = pd.Series([DEFAULT_VISIT_LABEL] * len(frame), dtype="string")
    frame = frame[(frame["value_num"].notna()) | (frame["value_text"].notna())]
    return frame, variable


def _image_frame(
    raw: pd.DataFrame,
    *,
    session_column: str | None,
    date_column: str | None,
    column: str,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    modality: str,
    variable_id: str,
    region_id: str | None,
    region_label: str | None,
    frame_index: pd.Series | None = None,
    unit: str | None = None,
    pipeline_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    value_num, value_text, value_kind = _series_value_payload(raw[column])
    variable = _variable_entry(
        variable_id=variable_id,
        source_column=column,
        domain="image",
        table="image_measurements",
        modality=modality,
        label=region_label or column,
        value_kind=value_kind,
        unit=unit,
        source_file=source_path.name,
        source_sheet=sheet_name,
    )
    frame = pd.DataFrame(
        {
            "subject_uid": raw["subject_uid"].astype("string").where(raw["subject_uid"].notna(), pd.NA),
            "session_id": _image_measurement_session_id_series(raw).astype("string").fillna(""),
            "modality": modality,
            "region_id": pd.Series([region_id if region_id else pd.NA] * len(raw), dtype="string"),
            "region_label": pd.Series([region_label if region_label else pd.NA] * len(raw), dtype="string"),
            "frame_index": frame_index if frame_index is not None else pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "variable_id": variable_id,
            "value_num": value_num,
            "value_text": value_text,
            "unit": pd.Series([unit if unit else pd.NA] * len(raw), dtype="string"),
            "value_kind": value_kind,
            "pipeline_name": source_path.stem,
            "pipeline_id": (
                pd.Series([pipeline_id] * len(raw), dtype="string")
                if pipeline_id is not None
                else pd.Series([pd.NA] * len(raw), dtype="string")
            ),
            "qc_status": pd.Series([pd.NA] * len(raw), dtype="string"),
            "source_asset": pd.Series([pd.NA] * len(raw), dtype="string"),
            "source_table": _source_table_name(source_path, sheet_name),
            "source_file": source_path.name,
            "source_sheet": sheet_name,
            "source_column": column,
            "source_batch_id": source_batch_id,
            "measured_at": _parse_datetime_series(raw[date_column]) if date_column else pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]"),
        }
    )
    frame = frame[(frame["value_num"].notna()) | (frame["value_text"].notna())]
    return frame, variable


def _parse_generic_clinical_wide(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    default_visit_label: str | None = None,
) -> pd.DataFrame:
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id)
    harvest_sessions_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id, modality=None, default_visit_label=default_visit_label)

    visit_column = _first_matching_column(raw, VISIT_CANDIDATES)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    skip_columns = {
        "subject_uid",
        _first_matching_column(raw, SUBJECT_UID_CANDIDATES),
        _first_matching_column(raw, SESSION_CANDIDATES),
        visit_column,
        date_column,
    }
    measurements: list[pd.DataFrame] = []
    variables: list[dict[str, Any]] = []
    for column in raw.columns:
        if column in skip_columns:
            continue
        if str(column).startswith("Unnamed:"):
            continue
        frame, variable = _clinical_frame(
            raw,
            visit_column=visit_column,
            date_column=date_column,
            column=column,
            source_path=source_path,
            sheet_name=sheet_name,
            source_batch_id=source_batch_id,
            default_visit_label=default_visit_label,
        )
        if not frame.empty:
            measurements.append(frame)
        if variable is not None:
            variables.append(variable)

    df = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    _upsert_measurements(repo, "clinical_measurements", df)
    _register_variables(repo, variables)
    return df


def _parse_image_wide_column(source_name: str, column: str) -> tuple[str, str | None, str | None]:
    if source_name in ["PESABrain_4DFlow_LocalizedPI_20260216.xlsx", "PESABrain_4DFlowv2_LocalizedPI_202602410.xlsx"]:
        region = column.replace("_PI", "")
        return "pi", _region_id(region), region
    if source_name in ["PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx", "PESABrain_4DFlowv2_LocalizedTimeAvgFlow_202602410.xlsx"]:
        if column.upper() == "TCBF":
            return "tcbf", None, None
        return "flow_mean", _region_id(column), column
    if source_name in ["PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx", "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx", "PESABrain_ASLPerfusion_CovCBF_20260216.xlsx"]:
        return "mean_cbf", _region_id(column), column
    if source_name == "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx":
        return "mean_cbf", _region_id(column), column
    if source_name == "PESABrain_ASLPerfusion_CovCBF_20260216.xlsx":
        return "cov_cbf", _region_id(column), column
    return normalize_variable_id(column), None, None


def _parse_generic_image_wide(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    modality: str,
    pipeline_id: str | None = None,
) -> pd.DataFrame:
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id)
    harvest_sessions_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id, modality=modality)

    session_column = _first_matching_column(raw, SESSION_CANDIDATES)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    skip_columns = {
        "subject_uid",
        _first_matching_column(raw, SUBJECT_UID_CANDIDATES),
        session_column,
        _first_matching_column(raw, VISIT_CANDIDATES),
        date_column,
    }
    measurements: list[pd.DataFrame] = []
    variables: list[dict[str, Any]] = []
    for column in raw.columns:
        if column in skip_columns:
            continue
        if str(column).startswith("Unnamed:"):
            continue
        variable_id, region_id, region_label = _parse_image_wide_column(source_path.name, column)
        pid = pipeline_id or _resolve_import_pipeline_id(repo, modality)
        frame, variable = _image_frame(
            raw,
            session_column=session_column,
            date_column=date_column,
            column=column,
            source_path=source_path,
            sheet_name=sheet_name,
            source_batch_id=source_batch_id,
            modality=modality,
            variable_id=variable_id,
            region_id=region_id,
            region_label=region_label,
            pipeline_id=pid,
        )
        if not frame.empty:
            measurements.append(frame)
        if variable is not None:
            variables.append(variable)

    df = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    _upsert_measurements(repo, "image_measurements", df)
    _register_variables(repo, variables)
    return df


def _parse_image_timeseries_long(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    modality: str,
) -> pd.DataFrame:
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id)

    frame_index = pd.to_numeric(raw.get("frame"), errors="coerce").astype("Int64")
    region_label = raw.get("vessel", pd.Series([pd.NA] * len(raw), dtype="string")).astype("string")
    region_id = raw.get("vessel_code", region_label).astype("string").map(_region_id)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)

    measurements: list[pd.DataFrame] = []
    variables: list[dict[str, Any]] = []
    _pv = _resolve_import_pipeline_id(repo, modality)
    mapping = {
        "flow": "flow_tseries",
        "phase": "phase_fraction",
    }
    for source_column, variable_id in mapping.items():
        if source_column not in raw.columns:
            continue
        value_num, value_text, value_kind = _series_value_payload(raw[source_column])
        frame = pd.DataFrame(
            {
                "subject_uid": raw["subject_uid"].astype("string"),
                "session_id": _image_measurement_session_id_series(raw).astype("string").fillna(""),
                "modality": modality,
                "region_id": region_id.astype("string"),
                "region_label": region_label,
                "frame_index": frame_index,
                "variable_id": variable_id,
                "value_num": value_num,
                "value_text": value_text,
                "unit": pd.Series([pd.NA] * len(raw), dtype="string"),
                "value_kind": value_kind,
                "pipeline_name": source_path.stem,
                "pipeline_id": (
                    pd.Series([_pv] * len(raw), dtype="string")
                    if _pv is not None
                    else pd.Series([pd.NA] * len(raw), dtype="string")
                ),
                "qc_status": pd.Series([pd.NA] * len(raw), dtype="string"),
                "source_asset": pd.Series([pd.NA] * len(raw), dtype="string"),
                "source_table": _source_table_name(source_path, sheet_name),
                "source_file": source_path.name,
                "source_sheet": sheet_name,
                "source_column": source_column,
                "source_batch_id": source_batch_id,
                "measured_at": _parse_datetime_series(raw[date_column]) if date_column else pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]"),
            }
        )
        frame = frame[(frame["value_num"].notna()) | (frame["value_text"].notna())]
        if not frame.empty:
            measurements.append(frame)
        ts_var = _variable_entry(
            variable_id=variable_id,
            source_column=source_column,
            domain="image",
            table="image_measurements",
            modality=modality,
            label=source_column,
            value_kind=value_kind,
            source_file=source_path.name,
            source_sheet=sheet_name,
        )
        ts_var["aliases"] = list(dict.fromkeys([ts_var["source_column"], variable_id]))
        variables.append(ts_var)

    df = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    _upsert_measurements(repo, "image_measurements", df)
    _register_variables(repo, variables)
    return df


def _parse_image_timeseries_wide(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    modality: str,
    pipeline_id: str | None = None,
) -> pd.DataFrame:
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id)

    wide = raw.copy()
    # Melt only canonical identifiers. Including ``patient_id`` alongside ``subject_uid`` duplicates
    # the subject grain and can confuse reshapes; subject_uid is the canonical key.
    melt_id_columns = ("subject_uid", "vessel", "vessel_code")
    id_vars = [column for column in melt_id_columns if column in wide.columns]
    reserved = set(id_vars)
    frame_columns = _timeseries_wide_frame_columns(list(wide.columns), reserved=reserved)
    if not frame_columns:
        return pd.DataFrame()
    melted = wide.melt(
        id_vars=id_vars,
        value_vars=frame_columns,
        var_name="frame",
        value_name="flow_value",
    )
    melted["frame"] = pd.to_numeric(melted["frame"], errors="coerce").astype("Int64")
    value_num, value_text, value_kind = _series_value_payload(melted["flow_value"])
    _pv = pipeline_id or _resolve_import_pipeline_id(repo, modality)
    _sid = _image_measurement_session_id_series(wide).astype("string").fillna("")
    session_expanded = np.repeat(_sid.to_numpy(), len(frame_columns))
    df = pd.DataFrame(
        {
            "subject_uid": melted["subject_uid"].astype("string"),
            "session_id": pd.Series(session_expanded, dtype="string"),
            "modality": modality,
            "region_id": melted["vessel_code"].astype("string").map(_region_id),
            "region_label": melted["vessel"].astype("string"),
            "frame_index": melted["frame"],
            "variable_id": "flow_tseries",
            "value_num": value_num,
            "value_text": value_text,
            "unit": pd.Series([pd.NA] * len(melted), dtype="string"),
            "value_kind": value_kind,
            "pipeline_name": source_path.stem,
            "pipeline_id": (
                pd.Series([_pv] * len(melted), dtype="string")
                if _pv is not None
                else pd.Series([pd.NA] * len(melted), dtype="string")
            ),
            "qc_status": pd.Series([pd.NA] * len(melted), dtype="string"),
            "source_asset": pd.Series([pd.NA] * len(melted), dtype="string"),
            "source_table": _source_table_name(source_path, sheet_name),
            "source_file": source_path.name,
            "source_sheet": sheet_name,
            "source_column": "wide_frame",
            "source_batch_id": source_batch_id,
            "measured_at": pd.Series([pd.NaT] * len(melted), dtype="datetime64[ns]"),
        }
    )
    df = df[df["value_num"].notna() | df["value_text"].notna()]
    _upsert_measurements(repo, "image_measurements", df)
    wide_var = _variable_entry(
        variable_id="flow_tseries",
        source_column="wide_frame",
        domain="image",
        table="image_measurements",
        modality=modality,
        label="Flow time series",
        value_kind=value_kind,
        source_file=source_path.name,
        source_sheet=sheet_name,
    )
    # Wide sheets only expose per-frame columns; keep the same aliases as long ``flow`` so
    # ``repo.image(variables=[\"flow\"])`` resolves after wide-only imports.
    wide_var["aliases"] = list(
        dict.fromkeys(
            [
                wide_var["source_column"],
                "flow",
                "flow_tseries",
                str(wide_var.get("label") or ""),
            ]
        )
    )
    wide_var["aliases"] = [a for a in wide_var["aliases"] if a]
    _register_variables(repo, [wide_var])
    return df


def _parse_hybrid_hemodynamic(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
) -> dict[str, pd.DataFrame]:
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id)
    harvest_sessions_from_frame(repo, raw, source_path=source_path, sheet_name=sheet_name, source_batch_id=source_batch_id, modality="4dflow")

    session_column = _first_matching_column(raw, SESSION_CANDIDATES)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    visit_column = _first_matching_column(raw, VISIT_CANDIDATES)
    skip_columns = {
        "subject_uid",
        _first_matching_column(raw, SUBJECT_UID_CANDIDATES),
        session_column,
        visit_column,
        date_column,
    }
    clinical_frames: list[pd.DataFrame] = []
    clinical_variables: list[dict[str, Any]] = []
    image_frames: list[pd.DataFrame] = []
    image_variables: list[dict[str, Any]] = []

    for column in raw.columns:
        if column in skip_columns:
            continue
        if str(column).startswith("Unnamed:"):
            continue
        normalized = normalize_variable_id(column)

        vessel_match = re.match(r"^([a-z0-9]+)_(flow|area|psv|pi|ri)$", normalized)
        asl_match = re.match(r"^([a-z0-9]+)_asl$", normalized)

        if vessel_match:
            region = vessel_match.group(1)
            metric = vessel_match.group(2)
            variable_id = "flow_mean" if metric == "flow" else metric
            frame, variable = _image_frame(
                raw,
                session_column=session_column,
                date_column=date_column,
                column=column,
                source_path=source_path,
                sheet_name=sheet_name,
                source_batch_id=source_batch_id,
                modality="4dflow",
                variable_id=variable_id,
                region_id=region,
                region_label=region.upper(),
                pipeline_id=_resolve_import_pipeline_id(repo, "4dflow"),
            )
            if not frame.empty:
                image_frames.append(frame)
            if variable is not None:
                image_variables.append(variable)
            continue

        if asl_match:
            region = asl_match.group(1)
            frame, variable = _image_frame(
                raw,
                session_column=session_column,
                date_column=date_column,
                column=column,
                source_path=source_path,
                sheet_name=sheet_name,
                source_batch_id=source_batch_id,
                modality="asl",
                variable_id="mean_cbf",
                region_id=region,
                region_label=region.upper(),
                pipeline_id=_resolve_import_pipeline_id(repo, "asl"),
            )
            if not frame.empty:
                image_frames.append(frame)
            if variable is not None:
                image_variables.append(variable)
            continue

        if normalized in {"ipb", "a2vpb", "apcpi"}:
            frame, variable = _image_frame(
                raw,
                session_column=session_column,
                date_column=date_column,
                column=column,
                source_path=source_path,
                sheet_name=sheet_name,
                source_batch_id=source_batch_id,
                modality="4dflow",
                variable_id=normalized,
                region_id=None,
                region_label=None,
                pipeline_id=_resolve_import_pipeline_id(repo, "4dflow"),
            )
            if not frame.empty:
                image_frames.append(frame)
            if variable is not None:
                image_variables.append(variable)
            continue

        if normalized in CLINICAL_METADATA_COLUMNS:
            frame, variable = _clinical_frame(
                raw,
                visit_column=visit_column,
                date_column=date_column,
                column=column,
                source_path=source_path,
                sheet_name=sheet_name,
                source_batch_id=source_batch_id,
                default_visit_label=DEFAULT_VISIT_LABEL,
                variable_id=normalized,
            )
            if not frame.empty:
                clinical_frames.append(frame)
            if variable is not None:
                clinical_variables.append(variable)

    clinical_df = pd.concat(clinical_frames, ignore_index=True) if clinical_frames else pd.DataFrame()
    image_df = pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
    _upsert_measurements(repo, "clinical_measurements", clinical_df)
    _upsert_measurements(repo, "image_measurements", image_df)
    _register_variables(repo, clinical_variables + image_variables)
    return {"clinical": clinical_df, "image": image_df}


def _resolve_metadata_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "name": _first_matching_column(df, ["Nombre", "Variable", "Glosario"]),
        "export_name": _first_matching_column(df, ["Nombre exportación", "Nombre exportacion", "Nombre exportación", "Nombre exportacion"]),
        "description": _first_matching_column(df, ["Descripción", "Descripcion", "Descripción (Significado de la variable)"]),
        "description_en": _first_matching_column(df, ["Description"]),
        "type": _first_matching_column(df, ["Tipo", "Tipo Texto/Cat/Num/Fecha"]),
        "unit": _first_matching_column(df, ["Unidades", "Unidades (Unidades en las que esta medida, Ej. cm, mg,)"]),
        "origin_name": _first_matching_column(df, ["Origen nombre"]),
        "glossary": _first_matching_column(df, ["Glosario"]),
        "nhanes": _first_matching_column(df, ["Significado NHANES"]),
        "min_value": _first_matching_column(df, ["Rango posibilidad (MIN)", "Rango posibilidad (MIN) Valor mínimo que puede tomar la variable"]),
        "max_value": _first_matching_column(df, ["Rango posibilidad (MAX)", "Rango posibilidad (MAX) Valor máximo que puede tomar la variable"]),
        "missing_allowed": _first_matching_column(df, ["Rango posibilidad (missing)", "Rango posibilidad (missing) 0: La variable no puede ser missing; 1: la variable puede ser missing"]),
        "allowed_values": _first_matching_column(df, ["Rango posibilidad (otros valores)", "Rango posibilidad (otros valores) Otros valores posibles separados por ;"]),
        "comments": _first_matching_column(df, ["Comentarios"]),
        "codebook": _first_matching_column(df, ["Codebook"]),
    }


def _metadata_type_to_value_kind(value: Any) -> str | None:
    text = normalize_string(value)
    if text is None:
        return None
    lowered = text.lower()
    if "num" in lowered:
        return "numeric"
    if "fecha" in lowered or "date" in lowered:
        return "datetime"
    if "cat" in lowered:
        return "categorical"
    if "text" in lowered:
        return "text"
    return lowered


def _split_allowed_values(value: Any) -> list[str]:
    text = normalize_string(value)
    if text is None:
        return []
    return [item.strip() for item in re.split(r"[;|]", text) if item.strip()]


def _parse_variable_dictionary(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    domain: str,
    modality: str | None,
) -> pd.DataFrame:
    columns = _resolve_metadata_columns(raw)
    entries: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        export_name = normalize_string(row.get(columns["export_name"])) if columns["export_name"] else None
        original_name = normalize_string(row.get(columns["name"])) if columns["name"] else None
        if export_name is None and original_name is None:
            continue
        if original_name and original_name.startswith("("):
            continue

        variable_id = normalize_variable_id(export_name or original_name or "")
        if not variable_id:
            continue

        missing_raw = normalize_string(row.get(columns["missing_allowed"])) if columns["missing_allowed"] else None
        missing_allowed: bool | None
        if missing_raw in {"0", "1"}:
            missing_allowed = missing_raw == "1"
        else:
            missing_allowed = None

        entry = _variable_entry(
            variable_id=variable_id,
            source_column=export_name or original_name or variable_id,
            source_file=source_path.name,
            source_sheet=sheet_name,
            domain=domain,
            table="image_measurements" if domain == "image" else "clinical_measurements",
            modality=modality,
            value_kind=_metadata_type_to_value_kind(row.get(columns["type"])) if columns["type"] else None,
            label=export_name or original_name,
            original_name=original_name,
            export_name=export_name,
            origin_name=normalize_string(row.get(columns["origin_name"])) if columns["origin_name"] else None,
            description=normalize_string(row.get(columns["description"])) if columns["description"] else None,
            description_en=normalize_string(row.get(columns["description_en"])) if columns["description_en"] else None,
            glossary=normalize_string(row.get(columns["glossary"])) if columns["glossary"] else None,
            unit=normalize_string(row.get(columns["unit"])) if columns["unit"] else None,
            min_value=normalize_string(row.get(columns["min_value"])) if columns["min_value"] else None,
            max_value=normalize_string(row.get(columns["max_value"])) if columns["max_value"] else None,
            missing_allowed=missing_allowed,
            allowed_values=_split_allowed_values(row.get(columns["allowed_values"])) if columns["allowed_values"] else [],
            comments=normalize_string(row.get(columns["comments"])) if columns["comments"] else None,
            codebook=normalize_string(row.get(columns["codebook"])) if columns["codebook"] else None,
            nhanes=normalize_string(row.get(columns["nhanes"])) if columns["nhanes"] else None,
        )
        aliases = [item for item in [original_name, export_name] if item]
        entry["aliases"] = aliases
        entries.append(entry)

    _register_variables(repo, entries)
    return pd.DataFrame(entries)


def _parse_dropdown_dictionary(
    repo: DataRepo,
    raw: pd.DataFrame,
    *,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    domain: str,
    modality: str | None,
) -> pd.DataFrame:
    entries: list[dict[str, Any]] = []
    for column in raw.columns:
        values = [normalize_string(item) for item in raw[column].tolist()]
        values = [item for item in values if item is not None]
        if not values:
            continue
        variable_name = values[0]
        allowed_values = values[1:]
        if variable_name is None or not allowed_values:
            continue
        variable_id = normalize_variable_id(variable_name)
        entries.append(
            {
                "variable_id": variable_id,
                "source_column": variable_name,
                "source_file": source_path.name,
                "source_sheet": sheet_name,
                "aliases": [variable_name],
                "domain": domain,
                "table": "image_measurements" if domain == "image" else "clinical_measurements",
                "modality": modality,
                "label": variable_name,
                "allowed_values": allowed_values,
            }
        )
    _register_variables(repo, entries)
    return pd.DataFrame(entries)


def rebuild_subjects_table(repo: DataRepo) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table_name in ("subject_ids", "clinical_measurements", "image_measurements", "sessions"):
        try:
            frame = repo.get(table_name)
        except Exception:
            continue
        if not frame.empty and "subject_uid" in frame.columns:
            frames.append(frame[["subject_uid"]].dropna())

    if not frames:
        return pd.DataFrame()

    subjects = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["subject_uid"]).reset_index(drop=True)
    subject_ids = repo.get("subject_ids") if repo.catalog.table_exists("subject_ids") else pd.DataFrame()
    clinical = repo.get("clinical_measurements") if repo.catalog.table_exists("clinical_measurements") else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for subject_uid in subjects["subject_uid"].astype("string"):
        subset = subject_ids[subject_ids["subject_uid"] == subject_uid] if not subject_ids.empty else pd.DataFrame()

        def pick_id(*names: str) -> str | None:
            for name in names:
                matches = subset[subset["id_namespace"] == name]
                if not matches.empty:
                    return normalize_string(matches.iloc[0]["id_value"])
            return None

        clinical_subset = clinical[clinical["subject_uid"] == subject_uid] if not clinical.empty else pd.DataFrame()
        sex_value = None
        birth_date = pd.NaT
        if not clinical_subset.empty:
            sex_rows = clinical_subset[clinical_subset["variable_id"] == "sex"]
            if not sex_rows.empty:
                sex_value = normalize_string(sex_rows.iloc[0].get("value_text")) or normalize_string(sex_rows.iloc[0].get("value_num"))
            birth_rows = clinical_subset[clinical_subset["variable_id"].isin(["birth_date", "fechanacimiento"])]
            if not birth_rows.empty:
                birth_date = pd.to_datetime(birth_rows.iloc[0].get("measured_at"), errors="coerce")

        rows.append(
            {
                "subject_uid": subject_uid,
                "primary_patient_id": pick_id("patient_id", "codigoimagen", "subject", "subject_id") or subject_uid,
                "primary_seqn": pick_id("seqn"),
                "sex": sex_value if sex_value is not None else pd.NA,
                "birth_date": birth_date,
                "notes": pd.NA,
                "source_batch_id": "derived_subjects",
                "updated_at": utc_now_iso(),
            }
        )

    df = pd.DataFrame(rows)
    repo.write_table("subjects", df, provenance={"importer": "rebuild_subjects_table"})
    return df


def import_source_spec(
    repo: DataRepo,
    source_path: Path,
    spec: SourceSpec,
    *,
    source_batch_id: str,
    pipeline_id: str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    raw = read_tabular_source(source_path, sheet_name=spec.sheet)
    _register_inventory_rows(repo, [_inventory_row(source_path, spec.sheet, raw, spec, source_batch_id)])

    if spec.source_kind == "subject_ids":
        return import_subject_ids_from_source(
            repo,
            source_path,
            source_batch_id=source_batch_id,
            id_source=_source_table_name(source_path, spec.sheet),
            sheet_name=spec.sheet,
        )
    if spec.source_kind == "cohort":
        return import_cohort_membership_from_source(
            repo,
            source_path,
            cohort_id=spec.cohort_id or source_path.stem,
            source_batch_id=source_batch_id,
            sheet_name=spec.sheet,
        )
    if spec.source_kind == "subject_catalog":
        raw = ensure_subject_uid(raw, candidates=["subject", "patient_id", "codigoimagen"])
        harvest_subject_ids_from_frame(repo, raw, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id)
        return harvest_sessions_from_frame(repo, raw, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, modality="mr")
    if spec.source_kind == "clinical_wide":
        return _parse_generic_clinical_wide(repo, raw, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id)
    if spec.source_kind == "image_wide":
        return _parse_generic_image_wide(repo, raw, pipeline_id=pipeline_id, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, modality=spec.modality or "image")
    if spec.source_kind == "image_timeseries_long":
        return _parse_image_timeseries_long(repo, raw, pipeline_id=pipeline_id, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, modality=spec.modality or "image")
    if spec.source_kind == "image_timeseries_wide":
        return _parse_image_timeseries_wide(repo, raw, pipeline_id=pipeline_id, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, modality=spec.modality or "image")
    if spec.source_kind == "variable_dictionary":
        return _parse_variable_dictionary(repo, raw, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, domain=spec.domain, modality=spec.modality)
    if spec.source_kind == "dropdown_dictionary":
        return _parse_dropdown_dictionary(repo, raw, source_path=source_path, sheet_name=spec.sheet, source_batch_id=source_batch_id, domain=spec.domain, modality=spec.modality)
    return pd.DataFrame()


def import_pesabrain_db_directory(
    repo: DataRepo,
    db_base_path: str | Path,
    *,
    source_batch_id: str | None = None,
    build_sqlite_index: bool = False,
    cohort_id: str | None = None,
) -> dict[str, Any]:
    base = Path(db_base_path)
    if not base.exists():
        raise FileNotFoundError(base)

    batch_id = source_batch_id or _default_pesabrain_batch_id()
    cohort_resolved = (cohort_id or "").strip() or DEFAULT_COHORT_ID
    imported: dict[str, Any] = {}
    inventory_rows: list[dict[str, Any]] = []
    cohort_uids: set[str] = set()

    for source_path in list_excel_sources(base):
        specs = PESABRAIN_DB_SPECS.get(source_path.name, [])
        matched_sheets = {spec.sheet for spec in specs}
        try:
            workbook = pd.ExcelFile(source_path)
            workbook_sheets = list(workbook.sheet_names)
        except Exception:
            workbook_sheets = [spec.sheet for spec in specs] if specs else []

        for spec in specs:
            result = import_source_spec(repo, source_path, spec, source_batch_id=batch_id)
            cohort_uids |= _collect_subject_uids_from_import_result(result)
            imported[f"{source_path.name}:{spec.sheet}:{spec.source_kind}"] = result

        for sheet_name in workbook_sheets:
            if sheet_name in matched_sheets:
                continue
            raw = read_tabular_source(source_path, sheet_name=sheet_name)
            inventory_rows.append(
                _inventory_row(
                    source_path,
                    sheet_name,
                    raw,
                    SourceSpec(source_path.name, sheet_name, "unmapped", "metadata", "wide"),
                    batch_id,
                )
            )

    _register_inventory_rows(repo, inventory_rows)
    subjects = rebuild_subjects_table(repo)
    imported["subjects"] = subjects

    if cohort_uids:
        upsert_cohort_membership_for_subjects(
            repo,
            cohort_resolved,
            cohort_uids,
            source_batch_id=batch_id,
            membership_source="import_pesabrain_db_directory",
        )

    if build_sqlite_index:
        repo.build_sqlite_index()
    return imported


def import_pesabrain_source(
    repo: DataRepo,
    db_base_path: str | Path,
    filename: str,
    *,
    sheet: str | None = None,
    source_kind: str | None = None,
    source_batch_id: str | None = None,
    rebuild_subjects: bool = True,
    build_sqlite_index: bool = False,
    pipeline_id: str | None = None,
    cohort_id: str | None = None,
) -> dict[str, Any]:
    base = Path(db_base_path)
    if not base.exists():
        raise FileNotFoundError(base)

    source_path = base / filename
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    matches = _matching_pesabrain_specs(filename, sheet=sheet, source_kind=source_kind)
    if not matches:
        raise ValueError(
            f"No configured PESA-Brain source matches filename={filename!r}, sheet={sheet!r}, source_kind={source_kind!r}."
        )
    if len(matches) > 1:
        options = ", ".join(f"{spec.sheet}:{spec.source_kind}" for spec in matches)
        raise ValueError(
            f"Multiple source specs match {filename!r}. Refine with 'sheet' and/or 'source_kind'. Matches: {options}"
        )

    spec = matches[0]
    batch_id = source_batch_id or _default_pesabrain_batch_id()
    cohort_resolved = (cohort_id or "").strip() or DEFAULT_COHORT_ID
    result = import_source_spec(repo, source_path, spec, source_batch_id=batch_id, pipeline_id=pipeline_id)
    uids = _collect_subject_uids_from_import_result(result)
    if uids:
        upsert_cohort_membership_for_subjects(
            repo,
            cohort_resolved,
            uids,
            source_batch_id=batch_id,
            membership_source=f"import_pesabrain_source:{filename}",
        )
    subjects = rebuild_subjects_table(repo) if rebuild_subjects else pd.DataFrame()
    if build_sqlite_index:
        repo.build_sqlite_index()
    return {
        "filename": filename,
        "sheet": spec.sheet,
        "source_kind": spec.source_kind,
        "cohort_id": cohort_resolved,
        "result": result,
        "subjects": subjects,
    }


def import_pesabrain_curated_tables(
    repo: DataRepo,
    db_base_path: str | Path,
    *,
    source_batch_id: str | None = None,
    build_sqlite_index: bool = False,
    cohort_id: str | None = None,
) -> dict[str, Any]:
    return import_pesabrain_db_directory(
        repo,
        db_base_path,
        source_batch_id=source_batch_id,
        build_sqlite_index=build_sqlite_index,
        cohort_id=cohort_id,
    )


@_click_command()
@_click_option(
    "--dataset-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=Path("dataset"),
    show_default=True,
    help="Dataset root that will receive the canonical Parquet tables.",
)
@_click_option(
    "--db-base-path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Directory containing the current curated PESA-Brain Excel files.",
)
@_click_option(
    "--build-sqlite-index",
    is_flag=True,
    help="Build or refresh the SQLite query cache after import.",
)
@_click_option(
    "--pipeline-id",
    type=str,
    default=None,
    help="Set NVITK_PIPELINE_ID for this run (overrides catalog default for all modalities unless NVITK_PIPELINE_ID_<MODALITY> is set).",
)
def main(dataset_root: Path, db_base_path: Path, build_sqlite_index: bool, pipeline_id: str | None) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    if pipeline_id and str(pipeline_id).strip():
        os.environ["NVITK_PIPELINE_ID"] = str(pipeline_id).strip()

    repo = DataRepo(dataset_root, auto_scaffold=True)
    imported = import_pesabrain_db_directory(repo, db_base_path, build_sqlite_index=build_sqlite_index)
    click.echo(f"Imported sources: {', '.join(sorted(imported)) if imported else 'none'}")


if __name__ == "__main__":
    main()
