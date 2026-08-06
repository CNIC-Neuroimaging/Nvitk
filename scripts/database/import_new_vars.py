#!/usr/bin/env python3
"""
Batch import: T1 cortical/subcortical volumetry, cognitive wide sheet, clinical renames/derivations,
ATT (long Desikan CSV + wide vascular Excel), WMH (wide CSV). Upserts into an existing nvitk
dataset (Parquet + catalog).

Example::

    python scripts/database/import_new_vars.py --dataset-root /path/to/dataset --build-sqlite-index
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    build_image_measurement_rows,
    DerivedImageMeasurementSpec,
    publish_derived_measurements,
)
from nvitk.db.importers import (
    DATE_CANDIDATES,
    DEFAULT_VISIT_LABEL,
    SESSION_CANDIDATES,
    SUBJECT_UID_CANDIDATES,
    VISIT_CANDIDATES,
    _first_matching_column,
    _image_frame,
    _parse_datetime_series,
    _region_id,
    _series_value_payload,
    _source_table_name,
    ensure_subject_uid,
    harvest_sessions_from_frame,
    harvest_subject_ids_from_frame,
    read_tabular_source,
)
from nvitk.db.repo import DataRepo
from nvitk.db.storage import normalize_variable_id, read_json, utc_now_iso, write_json
from nvitk.db.t1_atlases import register_t1_atlas_regions

DEFAULT_BATCH = "raw"

DEFAULT_PATHS = {
    "t1_cortical": Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/PESABrain_T1_cortical_regs_20250605.xlsx"),
    "t1_subcortical": Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/PESABrain_T1_subcortical_regs_20250605.xlsx"),
    "cognitive": Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/PESABrain_Cognitives_20260201.xlsx"),
    "att": Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/ATT_native_results.csv"),
    "att_vascular_mean": Path(
        "/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DB/raw/"
        "PESABrain_ASLPerfusion_VascularAtlas_MeanATT_20260216.xlsx"
    ),
    "att_vascular_median": Path(
        "/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DB/raw/"
        "PESABrain_ASLPerfusion_VascularAtlas_MedianATT_20260216.xlsx"
    ),
    "wmh": Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/WMH_MergedBatches_Averaged.csv"),
}

# Identifier / junk columns on the vascular ATT wide Excel sheets (not region values).
_VASCULAR_ATT_SKIP_COLUMNS = {
    "mri_id",
    "mri_id.1",
    "patient_id",
    "subject_uid",
    "session_id",
    "session_uid",
}

_ATT_VASCULAR_LABELS = {
    "att_mean": "ATT mean (vascular atlas)",
    "att_median": "ATT median (vascular atlas)",
}

T1_PIPELINE_ID = "t1_volumetry_v1"
WMH_PIPELINE_ID = "wmh_v1"
ASL_PIPELINE_FALLBACK = "asl_v1"


def _noop_print(*_a: Any, **_k: Any) -> None:
    pass


def ensure_cognitive_table(repo: DataRepo) -> None:
    if repo.catalog.table_exists("cognitive_measurements"):
        return
    repo.catalog.ensure_cognitive_measurements_table()


def ensure_measurement_pipeline(
    repo: DataRepo,
    *,
    pipeline_id: str,
    modality: str,
    is_default: bool = False,
    name: str | None = None,
) -> None:
    path = repo.catalog.pipelines_manifest_path
    if path is None or not path.exists():
        return
    data = read_json(path)
    pipelines = list(data.get("pipelines", []))
    ids = {str(p.get("pipeline_id")) for p in pipelines if p.get("pipeline_id")}
    if pipeline_id in ids:
        return
    pipelines.append(
        {
            "pipeline_id": pipeline_id,
            "modality": modality,
            "name": name or pipeline_id,
            "is_default": bool(is_default),
            "aliases": [],
        }
    )
    data["pipelines"] = pipelines
    data["last_updated"] = utc_now_iso()
    write_json(path, data)
    repo.catalog.refresh()


def _cognitive_measurement_frame(
    raw: pd.DataFrame,
    *,
    visit_column: str | None,
    date_column: str | None,
    column: str,
    source_path: Path,
    sheet_name: str,
    source_batch_id: str,
    default_visit_label: str | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    value_num, value_text, value_kind = _series_value_payload(raw[column])
    vid = normalize_variable_id(column)
    variable = {
        "variable_id": vid,
        "source_column": column,
        "domain": "cognitive",
        "table": "cognitive_measurements",
        "label": column,
        "value_kind": value_kind,
        "source_file": source_path.name,
        "source_sheet": sheet_name,
        "aliases": [column],
    }
    visit = (
        raw[visit_column].astype("string")
        if visit_column
        else pd.Series([default_visit_label or DEFAULT_VISIT_LABEL] * len(raw), dtype="string")
    )
    frame = pd.DataFrame(
        {
            "subject_uid": raw["subject_uid"].astype("string").where(raw["subject_uid"].notna(), pd.NA),
            "visit_id": visit,
            "variable_id": vid,
            "value_num": value_num,
            "value_text": value_text,
            "unit": pd.Series([pd.NA] * len(raw), dtype="string"),
            "value_kind": value_kind,
            "source_table": _source_table_name(source_path, sheet_name),
            "source_file": source_path.name,
            "source_sheet": sheet_name,
            "source_column": column,
            "source_batch_id": source_batch_id,
            "measured_at": pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]"),
        }
    )
    if date_column:
        frame["measured_at"] = _parse_datetime_series(raw[date_column])
    if frame["visit_id"].isna().any():
        frame["visit_id"] = pd.Series([default_visit_label or DEFAULT_VISIT_LABEL] * len(frame), dtype="string")
    frame = frame[(frame["value_num"].notna()) | (frame["value_text"].notna())]
    return frame, variable


def import_cognitive_wide(
    repo: DataRepo,
    path: Path,
    *,
    source_batch_id: str,
    log: Any = print,
) -> pd.DataFrame:
    ensure_cognitive_table(repo)
    raw = read_tabular_source(path, sheet_name="Sheet1")
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(
        repo, raw, source_path=path, sheet_name="Sheet1", source_batch_id=source_batch_id
    )
    harvest_sessions_from_frame(
        repo,
        raw,
        source_path=path,
        sheet_name="Sheet1",
        source_batch_id=source_batch_id,
        modality=None,
        default_visit_label=DEFAULT_VISIT_LABEL,
    )
    visit_column = _first_matching_column(raw, VISIT_CANDIDATES)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    skip = {
        "subject_uid",
        _first_matching_column(raw, SUBJECT_UID_CANDIDATES),
        _first_matching_column(raw, SESSION_CANDIDATES),
        visit_column,
        date_column,
    }
    frames: list[pd.DataFrame] = []
    variables: list[dict[str, Any]] = []
    for column in raw.columns:
        if column in skip or column is None:
            continue
        if str(column).startswith("Unnamed:"):
            continue
        fr, var = _cognitive_measurement_frame(
            raw,
            visit_column=visit_column,
            date_column=date_column,
            column=str(column),
            source_path=path,
            sheet_name="Sheet1",
            source_batch_id=source_batch_id,
            default_visit_label=DEFAULT_VISIT_LABEL,
        )
        if not fr.empty:
            frames.append(fr)
        if var is not None:
            variables.append(var)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        repo.upsert_table(
            "cognitive_measurements",
            out,
            provenance={"importer": "import_new_vars", "step": "cognitive"},
            build_sqlite_index=True,
        )
    if variables:
        repo.register_variables(variables)
    log(f"cognitive_measurements rows upserted: {len(out)}")
    return out


def import_t1_volumetry(
    repo: DataRepo,
    path: Path,
    *,
    variable_id: str,
    atlas_key: str,
    source_batch_id: str,
    log: Any = print,
) -> pd.DataFrame:
    ensure_measurement_pipeline(repo, pipeline_id=T1_PIPELINE_ID, modality="t1", name="T1 volumetry import")
    raw = read_tabular_source(path, sheet_name="Sheet1")
    raw = ensure_subject_uid(raw)
    harvest_subject_ids_from_frame(
        repo, raw, source_path=path, sheet_name="Sheet1", source_batch_id=source_batch_id
    )
    harvest_sessions_from_frame(
        repo,
        raw,
        source_path=path,
        sheet_name="Sheet1",
        source_batch_id=source_batch_id,
        modality="t1",
        default_visit_label=DEFAULT_VISIT_LABEL,
    )
    session_column = _first_matching_column(raw, SESSION_CANDIDATES)
    date_column = _first_matching_column(raw, DATE_CANDIDATES)
    skip = {
        "subject_uid",
        _first_matching_column(raw, SUBJECT_UID_CANDIDATES),
        session_column,
        _first_matching_column(raw, VISIT_CANDIDATES),
        date_column,
    }
    measurements: list[pd.DataFrame] = []
    variables: list[dict[str, Any]] = []
    region_ids: list[str] = []
    for column in raw.columns:
        if column in skip or column is None:
            continue
        if str(column).startswith("Unnamed:"):
            continue
        rid = _region_id(str(column))
        if rid:
            region_ids.append(rid)
        fr, var = _image_frame(
            raw,
            session_column=session_column,
            date_column=date_column,
            column=str(column),
            source_path=path,
            sheet_name="Sheet1",
            source_batch_id=source_batch_id,
            modality="t1",
            variable_id=variable_id,
            region_id=rid,
            region_label=str(column),
            pipeline_id=T1_PIPELINE_ID,
        )
        if not fr.empty:
            measurements.append(fr)
        if var is not None:
            variables.append(var)
    out = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    if not out.empty:
        repo.upsert_table(
            "image_measurements",
            out,
            provenance={"importer": "import_new_vars", "step": f"t1_{atlas_key}"},
            build_sqlite_index=True,
        )
    if variables:
        repo.register_variables(variables)
    register_t1_atlas_regions(atlas_key, region_ids)
    log(f"T1 {atlas_key} image_measurements rows: {len(out)}; atlas regions registered: {len(set(region_ids))}")
    return out


def rename_sys_dias_delta_to_pp(repo: DataRepo, *, log: Any = print) -> int:
    df = repo.get("clinical_measurements", wide=False)
    if df.empty or "variable_id" not in df.columns:
        return 0
    mask = df["variable_id"].astype("string") == "sys_dias_delta"
    n = int(mask.sum())
    if n:
        df = df.copy()
        df.loc[mask, "variable_id"] = "pp"
        repo.write_table(
            "clinical_measurements",
            df,
            provenance={"importer": "import_new_vars", "step": "rename_sys_dias_delta_to_pp"},
            build_sqlite_index=True,
        )
    repo.register_variables(
        [
            {
                "variable_id": "pp",
                "domain": "clinical",
                "table": "clinical_measurements",
                "label": "Pulse pressure (sys - dias)",
                "source_column": "pp",
                "aliases": ["pp", "sys_dias_delta"],
                "unit": "mmHg",
            }
        ]
    )
    log(f"Renamed sys_dias_delta -> pp on {n} rows")
    return n


def derive_pulse_pressure_map(repo: DataRepo, *, source_batch_id: str, log: Any = print) -> pd.DataFrame:
    long = repo.get(
        "clinical_measurements",
        filters={"variable_id": ["bpxdim", "pp"]},
        wide=False,
    )
    if long.empty:
        log("pulse_pressure_map: skip (no bpxdim/pp)")
        return pd.DataFrame()
    pivot = long.pivot_table(
        index=["subject_uid", "visit_id"],
        columns="variable_id",
        values="value_num",
        aggfunc="first",
    ).reset_index()
    if "bpxdim" not in pivot.columns or "pp" not in pivot.columns:
        log("pulse_pressure_map: skip (missing pivoted columns)")
        return pd.DataFrame()
    pivot["pulse_pressure_map"] = pd.to_numeric(pivot["bpxdim"], errors="coerce") + (1.0 / 3.0) * pd.to_numeric(
        pivot["pp"], errors="coerce"
    )
    agg = pivot.dropna(subset=["pulse_pressure_map"])
    spec = DerivedClinicalMeasurementSpec(
        variable_id="pulse_pressure_map",
        source_file="import_new_vars",
        source_sheet="derived",
        source_column="pulse_pressure_map",
        value_column="pulse_pressure_map",
        value_kind="float",
        unit="mmHg",
        source_batch_id=source_batch_id,
    )
    rows = build_clinical_measurement_rows(agg, spec)
    if rows.empty:
        return rows
    publish_derived_measurements(
        repo,
        rows,
        table="clinical_measurements",
        register=DerivedVariableRegistration(
            variable_id="pulse_pressure_map",
            domain="clinical",
            table="clinical_measurements",
            label="bpxdim + (1/3)·pp",
            aliases=["pulse_pressure_map", "map"],
            source_column="pulse_pressure_map",
            source_file="import_new_vars",
            source_sheet="derived",
            value_kind="float",
            unit="mmHg",
        ),
        provenance={"importer": "import_new_vars", "step": "pulse_pressure_map"},
        build_sqlite_index=True,
    )
    log(f"pulse_pressure_map rows: {len(rows)}")
    return rows


def classify_apoe_group(raw: Any) -> str:
    """Map APOE genotype string to apoe_group category (public for tests)."""
    return _classify_apoe(raw)


def _classify_apoe(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "unknown"
    t = str(raw).upper().replace(" ", "").replace("_", "/").replace("\\", "/")
    parts = [p.strip() for p in t.split("/") if p.strip()]
    if len(parts) != 2:
        return "unknown"
    a, b = parts[0], parts[1]
    pair = {a, b}
    if pair == {"E2", "E4"}:
        return "e2_e4"
    if (a == "E4" and b == "E4") or (pair == {"E3", "E4"}):
        return "any_e4"
    if (a == "E2" and b == "E2") or pair == {"E2", "E3"}:
        return "any_e2"
    if a == "E3" and b == "E3":
        return "only_e3"
    return "unknown"


def derive_apoe_group(repo: DataRepo, *, source_batch_id: str, log: Any = print) -> pd.DataFrame:
    long = repo.get("clinical_measurements", filters={"variable_id": ["apoe"]}, cohort_id=False, wide=False)
    if long.empty:
        log("apoe_group: skip (no apoe)")
        return pd.DataFrame()
    pick = long.sort_values(["subject_uid", "visit_id", "measured_at"], na_position="last").drop_duplicates(
        subset=["subject_uid", "visit_id"], keep="last"
    )
    vt = pick["value_text"].where(pick["value_text"].notna(), pick["value_num"].astype("string"))
    pick = pick.assign(_apoe_label=vt.astype("string"), apoe_group=vt.map(classify_apoe_group))
    n = len(pick)
    meas = pick["measured_at"] if "measured_at" in pick.columns else pd.Series([pd.NaT] * n, dtype="datetime64[ns]")
    rows = pd.DataFrame(
        {
            "subject_uid": pick["subject_uid"].astype("string"),
            "visit_id": pick["visit_id"].astype("string"),
            "variable_id": "apoe_group",
            "value_num": pd.Series([np.nan] * n, dtype="float64"),
            "value_text": pick["apoe_group"].astype("string"),
            "unit": pd.Series([pd.NA] * n, dtype="string"),
            "value_kind": pd.Series(["categorical"] * n, dtype="string"),
            "source_table": pd.Series(["import_new_vars::derived"] * n, dtype="string"),
            "source_file": pd.Series(["import_new_vars"] * n, dtype="string"),
            "source_sheet": pd.Series(["apoe_group"] * n, dtype="string"),
            "source_column": pd.Series(["apoe_group"] * n, dtype="string"),
            "source_batch_id": pd.Series([source_batch_id] * n, dtype="string"),
            "measured_at": pd.to_datetime(meas, errors="coerce"),
        }
    )
    publish_derived_measurements(
        repo,
        rows,
        table="clinical_measurements",
        register=DerivedVariableRegistration(
            variable_id="apoe_group",
            domain="clinical",
            table="clinical_measurements",
            label="APOE risk grouping",
            aliases=["apoe_group"],
            source_column="apoe_group",
            source_file="import_new_vars",
            source_sheet="derived",
            value_kind="categorical",
        ),
        provenance={"importer": "import_new_vars", "step": "apoe_group"},
        build_sqlite_index=False,
    )
    log(f"apoe_group rows: {len(rows)}")
    return rows


def _mri_subject_lookup(repo: DataRepo) -> pd.DataFrame:
    s = repo.get("sessions", cohort_id=False)
    if s.empty:
        return pd.DataFrame(columns=["mri_id", "subject_uid"])
    sid = s["subject_uid"].astype("string").str.strip()
    exp = s["experiment_label"].astype("string").str.strip()
    out = pd.DataFrame({"mri_id": exp, "subject_uid": sid}).dropna()
    if "session_uid" in s.columns:
        su = s["session_uid"].astype("string")
        tail = su.str.split(":").str[-1].str.strip()
        out = pd.concat(
            [out, pd.DataFrame({"mri_id": tail, "subject_uid": sid})],
            ignore_index=True,
        )
    return out.drop_duplicates(subset=["mri_id"], keep="first")


def _default_asl_pipeline(repo: DataRepo) -> str:
    pid = repo.catalog.default_pipeline_id("asl")
    return pid or ASL_PIPELINE_FALLBACK


def import_att_csv(repo: DataRepo, path: Path, *, source_batch_id: str, log: Any = print) -> pd.DataFrame:
    pid = _default_asl_pipeline(repo)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"mri_id", "region", "corrected_mean", "cov"} <= set(df.columns):
        raise ValueError(f"ATT CSV missing required columns: got {list(df.columns)}")
    lookup = _mri_subject_lookup(repo)
    merged = df.merge(lookup, on="mri_id", how="left")
    n_miss = int(merged["subject_uid"].isna().sum())
    if n_miss:
        log(f"ATT: {n_miss} rows without subject_uid match in sessions")
    merged = merged.dropna(subset=["subject_uid"])
    frames: list[pd.DataFrame] = []
    for variable_id, col in (("att_mean", "corrected_mean"), ("att_cov", "cov")):
        agg = merged.rename(columns={col: "value_num", "region": "_region_raw"}).copy()
        agg["session_id"] = agg["mri_id"].astype("string").str.strip()
        agg["region_id"] = agg["_region_raw"].map(lambda x: _region_id(str(x)) if pd.notna(x) else None)
        agg["region_label"] = agg["_region_raw"].astype("string")
        spec = DerivedImageMeasurementSpec(
            variable_id=variable_id,
            modality="asl",
            pipeline_id=pid,
            source_file=path.name,
            source_sheet="ATT_native_results",
            source_column=col,
            value_column="value_num",
            source_batch_id=source_batch_id,
        )
        fr = build_image_measurement_rows(agg, spec)
        if not fr.empty:
            frames.append(fr)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        repo.upsert_table(
            "image_measurements",
            out,
            provenance={"importer": "import_new_vars", "step": "att"},
            build_sqlite_index=True,
        )
        repo.register_variables(
            [
                {
                    "variable_id": "att_mean",
                    "domain": "image",
                    "table": "image_measurements",
                    "modality": "asl",
                    "label": "ATT corrected mean",
                    "source_column": "corrected_mean",
                    "source_file": path.name,
                },
                {
                    "variable_id": "att_cov",
                    "domain": "image",
                    "table": "image_measurements",
                    "modality": "asl",
                    "label": "ATT coefficient of variation",
                    "source_column": "cov",
                    "source_file": path.name,
                },
            ]
        )
    log(f"ATT image_measurements rows: {len(out)}")
    return out


def _vascular_att_region_columns(raw: pd.DataFrame) -> list[str]:
    """Region value columns on a vascular ATT wide sheet (exclude id / junk columns)."""
    cols: list[str] = []
    for column in raw.columns:
        name = str(column).strip()
        if not name or name.startswith("Unnamed:"):
            continue
        lower = name.lower()
        if lower in _VASCULAR_ATT_SKIP_COLUMNS or lower.startswith("mri_id"):
            continue
        cols.append(column if isinstance(column, str) else name)
    return cols


def import_att_vascular_xlsx(
    repo: DataRepo,
    path: Path,
    *,
    variable_id: str,
    source_batch_id: str,
    sheet_name: str = "Sheet1",
    log: Any = print,
) -> pd.DataFrame:
    """Import vascular-atlas ATT from a wide Excel sheet (``mri_id`` × region columns).

    Unlike the Desikan ``ATT_native_results.csv`` (long: region / corrected_mean / cov), these
    sheets are wide: one row per ``mri_id``, one column per vascular parcel (``Left_ACA-0``, …).
    ``mri_id`` is resolved to ``subject_uid`` via sessions (same lookup as :func:`import_att_csv`).
    """
    if variable_id not in _ATT_VASCULAR_LABELS:
        raise ValueError(f"Unsupported vascular ATT variable_id: {variable_id!r}")
    if not path.exists():
        raise FileNotFoundError(path)

    pid = _default_asl_pipeline(repo)
    raw = read_tabular_source(path, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "mri_id" not in raw.columns:
        raise ValueError(f"Vascular ATT Excel missing mri_id column: got {list(raw.columns)}")

    region_cols = _vascular_att_region_columns(raw)
    if not region_cols:
        raise ValueError(f"Vascular ATT Excel has no region columns: {path.name}")

    melted = raw.melt(
        id_vars=["mri_id"],
        value_vars=region_cols,
        var_name="_region_raw",
        value_name="value_num",
    )
    melted["mri_id"] = melted["mri_id"].astype("string").str.strip()
    melted["value_num"] = pd.to_numeric(melted["value_num"], errors="coerce")
    melted = melted.dropna(subset=["mri_id", "value_num"])
    melted = melted[melted["mri_id"] != ""]

    lookup = _mri_subject_lookup(repo)
    merged = melted.merge(lookup, on="mri_id", how="left")
    n_miss = int(merged["subject_uid"].isna().sum())
    if n_miss:
        log(f"ATT vascular ({variable_id}): {n_miss} value rows without subject_uid match in sessions")
    merged = merged.dropna(subset=["subject_uid"])

    frames: list[pd.DataFrame] = []
    for region_label, sub in merged.groupby("_region_raw", sort=False):
        agg = sub.copy()
        agg["session_id"] = agg["mri_id"].astype("string")
        agg["region_id"] = agg["_region_raw"].map(lambda x: _region_id(str(x)) if pd.notna(x) else None)
        agg["region_label"] = agg["_region_raw"].astype("string")
        spec = DerivedImageMeasurementSpec(
            variable_id=variable_id,
            modality="asl",
            pipeline_id=pid,
            source_file=path.name,
            source_sheet=sheet_name,
            source_column=str(region_label),
            value_column="value_num",
            source_batch_id=source_batch_id,
            unit="s",
            value_kind="numeric",
        )
        fr = build_image_measurement_rows(agg, spec)
        if not fr.empty:
            frames.append(fr)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        # Replace any prior rows for this source file (including broken subject_uid-NA imports).
        existing = repo.get("image_measurements", cohort_id=False, use_sqlite=False)
        if not existing.empty and "source_file" in existing.columns:
            keep = existing.loc[existing["source_file"].astype("string") != path.name].copy()
        else:
            keep = existing
        combined = pd.concat([keep, out], ignore_index=True) if keep is not None and not keep.empty else out
        repo.write_table(
            "image_measurements",
            combined,
            provenance={"importer": "import_new_vars", "step": f"att_vascular_{variable_id}"},
            build_sqlite_index=True,
        )
        repo.register_variables(
            [
                {
                    "variable_id": variable_id,
                    "domain": "image",
                    "table": "image_measurements",
                    "modality": "asl",
                    "label": _ATT_VASCULAR_LABELS[variable_id],
                    "source_file": path.name,
                    "unit": "s",
                    "aliases": [variable_id],
                }
            ]
        )
    log(
        f"ATT vascular {variable_id} rows: {len(out)} "
        f"({merged['mri_id'].nunique() if not merged.empty else 0} sessions, "
        f"{len(region_cols)} regions) from {path.name}"
    )
    return out


def import_att_vascular(
    repo: DataRepo,
    *,
    mean_path: Path,
    median_path: Path,
    source_batch_id: str,
    log: Any = print,
) -> pd.DataFrame:
    """Import vascular-atlas ``att_mean`` and ``att_median`` from the two wide Excel sources."""
    frames = [
        import_att_vascular_xlsx(
            repo,
            mean_path,
            variable_id="att_mean",
            source_batch_id=source_batch_id,
            log=log,
        ),
        import_att_vascular_xlsx(
            repo,
            median_path,
            variable_id="att_median",
            source_batch_id=source_batch_id,
            log=log,
        ),
    ]
    nonempty = [f for f in frames if not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


_WMH_METRIC = {"Les": "wmh_les", "Reg": "wmh_reg", "Freq": "wmh_freq", "Dist": "wmh_dist"}
_WMH_RE = re.compile(r"^(Les|Reg|Freq|Dist)([A-Za-z0-9]+)$")


def import_wmh_csv(repo: DataRepo, path: Path, *, source_batch_id: str, log: Any = print) -> pd.DataFrame:
    ensure_measurement_pipeline(repo, pipeline_id=WMH_PIPELINE_ID, modality="flair", name="WMH merged batches")
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().strip('"') for c in raw.columns]
    raw = raw.replace({"NA": np.nan, "": np.nan})
    id_col = "ID" if "ID" in raw.columns else raw.columns[0]
    lookup = _mri_subject_lookup(repo)
    long_rows: list[pd.DataFrame] = []
    for col in raw.columns:
        if col == id_col:
            continue
        m = _WMH_RE.match(str(col))
        if not m:
            continue
        prefix, region = m.group(1), m.group(2)
        vid = _WMH_METRIC.get(prefix)
        if not vid:
            continue
        piece = pd.DataFrame(
            {
                "mri_id": raw[id_col].astype("string").str.strip(),
                "value_num": pd.to_numeric(raw[col], errors="coerce"),
                "region_id": normalize_variable_id(region),
                "region_label": region,
                "_var": vid,
            }
        )
        long_rows.append(piece)
    if len(long_rows) == 0:
        log("WMH: no columns matched Les|Reg|Freq|Dist pattern")
        return pd.DataFrame()
    melted = pd.concat(long_rows, ignore_index=True)
    melted = melted.merge(lookup, on="mri_id", how="left")
    n_miss = int(melted["subject_uid"].isna().sum())
    if n_miss:
        log(f"WMH: {n_miss} value rows without subject_uid match in sessions")
    melted = melted.dropna(subset=["subject_uid"])
    melted["session_id"] = melted["mri_id"].astype("string")
    frames: list[pd.DataFrame] = []
    for vid, sub in melted.groupby("_var"):
        agg = sub.drop(columns=["_var"]).rename(columns={"value_num": "value_num"})
        spec = DerivedImageMeasurementSpec(
            variable_id=str(vid),
            modality="flair",
            pipeline_id=WMH_PIPELINE_ID,
            source_file=path.name,
            source_sheet="WMH_MergedBatches_Averaged",
            source_column=str(vid),
            value_column="value_num",
            source_batch_id=source_batch_id,
        )
        fr = build_image_measurement_rows(agg, spec)
        if not fr.empty:
            frames.append(fr)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        repo.upsert_table(
            "image_measurements",
            out,
            provenance={"importer": "import_new_vars", "step": "wmh"},
            build_sqlite_index=True,
        )
        for vid, label in (
            ("wmh_les", "WMH lesion volume"),
            ("wmh_reg", "WMH region volume"),
            ("wmh_freq", "WMH Les/Reg"),
            ("wmh_dist", "WMH Les/LesTot"),
        ):
            repo.register_variables(
                [
                    {
                        "variable_id": vid,
                        "domain": "image",
                        "table": "image_measurements",
                        "modality": "flair",
                        "label": label,
                        "source_file": path.name,
                    }
                ]
            )
    log(f"WMH image_measurements rows: {len(out)}")
    return out


STEPS = (
    "t1_cortical",
    "t1_subcortical",
    "cognitive",
    "rename_pp",
    "pulse_map",
    "apoe_group",
    "att",
    "att_vascular",
    "wmh",
)


def run_import_new_vars(
    repo: DataRepo,
    *,
    paths: dict[str, Path] | None = None,
    source_batch_id: str = DEFAULT_BATCH,
    steps: Iterable[str] | None = None,
    log: Any = print,
) -> None:
    """Run T1/cognitive imports, clinical derivations (pp, MAP, APOE group), ATT, vascular ATT, and WMH."""
    resolved_paths = dict(DEFAULT_PATHS)
    if paths:
        resolved_paths.update(paths)
    step_set = set(steps) if steps is not None else set(STEPS)

    if "t1_cortical" in step_set:
        import_t1_volumetry(
            repo,
            resolved_paths["t1_cortical"],
            variable_id="t1_cortical_volume",
            atlas_key="cortical",
            source_batch_id=source_batch_id,
            log=log,
        )
    if "t1_subcortical" in step_set:
        import_t1_volumetry(
            repo,
            resolved_paths["t1_subcortical"],
            variable_id="t1_subcortical_volume",
            atlas_key="subcortical",
            source_batch_id=source_batch_id,
            log=log,
        )
    if "cognitive" in step_set:
        import_cognitive_wide(repo, resolved_paths["cognitive"], source_batch_id=source_batch_id, log=log)
    if "rename_pp" in step_set:
        rename_sys_dias_delta_to_pp(repo, log=log)
    if "pulse_map" in step_set:
        derive_pulse_pressure_map(repo, source_batch_id=source_batch_id, log=log)
    if "apoe_group" in step_set:
        derive_apoe_group(repo, source_batch_id=source_batch_id, log=log)
    if "att" in step_set:
        import_att_csv(repo, resolved_paths["att"], source_batch_id=source_batch_id, log=log)
    if "att_vascular" in step_set:
        import_att_vascular(
            repo,
            mean_path=resolved_paths["att_vascular_mean"],
            median_path=resolved_paths["att_vascular_median"],
            source_batch_id=source_batch_id,
            log=log,
        )
    if "wmh" in step_set:
        import_wmh_csv(repo, resolved_paths["wmh"], source_batch_id=source_batch_id, log=log)


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True, help="Dataset root (catalog + tables/)")
    p.add_argument("--source-batch-id", type=str, default=DEFAULT_BATCH)
    p.add_argument("--build-sqlite-index", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Log planned actions only (still loads repo)")
    p.add_argument(
        "--steps",
        type=str,
        default=",".join(STEPS),
        help=f"Comma-separated subset of: {','.join(STEPS)}",
    )
    for key, default in DEFAULT_PATHS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=Path, default=default, help=f"Override path for {key}")
    args = p.parse_args(list(argv) if argv is not None else None)

    log = _noop_print if args.dry_run else print
    repo = DataRepo(args.dataset_root, auto_scaffold=True, use_sqlite=True)
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    if args.dry_run:
        print(f"dry-run: would execute steps {sorted(steps)} on {args.dataset_root}")
        return

    paths = {
        "t1_cortical": args.t1_cortical,
        "t1_subcortical": args.t1_subcortical,
        "cognitive": args.cognitive,
        "att": args.att,
        "att_vascular_mean": args.att_vascular_mean,
        "att_vascular_median": args.att_vascular_median,
        "wmh": args.wmh,
    }
    run_import_new_vars(repo, paths=paths, source_batch_id=args.source_batch_id, steps=steps, log=log)

    if args.build_sqlite_index:
        repo.build_sqlite_index()
        log("SQLite index rebuilt")


if __name__ == "__main__":
    main()
