"""Register measurement variable units in the catalog from curated metadata and table data."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from nvitk.db.repo import DataRepo

LogFn = Callable[..., Any]

MEASUREMENT_TABLE_DOMAINS: dict[str, str] = {
    "clinical_measurements": "clinical",
    "cognitive_measurements": "cognitive",
    "image_measurements": "image",
}

# Authoritative units for variables that lack a populated ``unit`` column in parquet.
CURATED_VARIABLE_UNITS: list[dict[str, Any]] = [
    {"variable_id": "age_at_mri", "domain": "clinical", "table": "clinical_measurements", "unit": "Years"},
    {"variable_id": "weight", "domain": "clinical", "table": "clinical_measurements", "unit": "kg"},
    {"variable_id": "height", "domain": "clinical", "table": "clinical_measurements", "unit": "cm"},
    {"variable_id": "bmi", "domain": "clinical", "table": "clinical_measurements", "unit": "kg/m2"},
    {
        "variable_id": "psqto000",
        "domain": "clinical",
        "table": "clinical_measurements",
        "unit": "0: non; 1: active; 2: former; 3: social",
    },
    {"variable_id": "lbxhdd", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "lbdldl", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "lbdlld", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "lbxtc", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "bpxsym", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "bpxdim", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "dpxdim", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "bpxpls", "domain": "clinical", "table": "clinical_measurements", "unit": "bpm"},
    {"variable_id": "tas", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "tad", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "sys_dias_delta", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "pp", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {
        "variable_id": "pulse_pressure_map",
        "domain": "clinical",
        "table": "clinical_measurements",
        "unit": "mmHg",
        "aliases": ["pulse_pressure_map", "map"],
    },
    {"variable_id": "hematocrit", "domain": "clinical", "table": "clinical_measurements", "unit": "%"},
    {"variable_id": "tacsctot", "domain": "clinical", "table": "clinical_measurements", "unit": "Agaston Units"},
    {"variable_id": "left_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "right_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "right_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "left_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {
        "variable_id": "apoe",
        "domain": "clinical",
        "table": "clinical_measurements",
        "unit": "Apolipoprotein E Aplotype Status",
    },
    {"variable_id": "pi", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "dimensionless"},
    {"variable_id": "ri", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "dimensionless"},
    {"variable_id": "flow_mean", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "mL/min"},
    {"variable_id": "flow_tseries", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "mL/min"},
    {
        "variable_id": "pwv",
        "domain": "image",
        "table": "image_measurements",
        "modality": "4dflow",
        "unit": "m/s",
        "aliases": ["pwv", "pwv_bjornfoot"],
    },
    {
        "variable_id": "pwv_fielding_xcor",
        "domain": "image",
        "table": "image_measurements",
        "modality": "4dflow",
        "unit": "m/s",
    },
    {"variable_id": "pitc_slope", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "1/mm"},
    {"variable_id": "pitc_intercept", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "dimensionless"},
    {"variable_id": "damping_index", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "dimensionless"},
    {"variable_id": "tcbf", "domain": "image", "table": "image_measurements", "modality": "4dflow", "unit": "mL/min"},
    {"variable_id": "length_mm", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "mm"},
    {"variable_id": "radius_mean_mm", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "mm"},
    {"variable_id": "radius_max_mm", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "mm"},
    {"variable_id": "tortuosity_dm", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "dimensionless"},
    {"variable_id": "stenosis_percent_max", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "%"},
    {"variable_id": "enlargement_percent_max", "domain": "image", "table": "image_measurements", "modality": "tof", "unit": "%"},
    {"variable_id": "mean_cbf", "domain": "image", "table": "image_measurements", "modality": "asl", "unit": "mL/100g/min"},
    {"variable_id": "cov_cbf", "domain": "image", "table": "image_measurements", "modality": "asl", "unit": "%"},
    {"variable_id": "att_mean", "domain": "image", "table": "image_measurements", "modality": "asl", "unit": "s"},
    {"variable_id": "att_median", "domain": "image", "table": "image_measurements", "modality": "asl", "unit": "s"},
    {"variable_id": "att_cov", "domain": "image", "table": "image_measurements", "modality": "asl", "unit": "%"},
    {
        "variable_id": "t1_cortical_volume",
        "domain": "image",
        "table": "image_measurements",
        "modality": "t1",
        "unit": "mm3",
    },
    {
        "variable_id": "t1_subcortical_volume",
        "domain": "image",
        "table": "image_measurements",
        "modality": "t1",
        "unit": "mm3",
    },
    {"variable_id": "wmh_les", "domain": "image", "table": "image_measurements", "modality": "flair", "unit": "mm3"},
    {"variable_id": "wmh_reg", "domain": "image", "table": "image_measurements", "modality": "flair", "unit": "mm3"},
    {"variable_id": "wmh_freq", "domain": "image", "table": "image_measurements", "modality": "flair", "unit": "ratio"},
    {"variable_id": "wmh_dist", "domain": "image", "table": "image_measurements", "modality": "flair", "unit": "ratio"},
]


def _harvest_units_from_table(repo: DataRepo, table: str) -> dict[str, str]:
    if not repo.catalog.table_exists(table):
        return {}
    df = repo.get(table, columns=["variable_id", "unit"], cohort_id=False, wide=False)
    if df.empty or "unit" not in df.columns:
        return {}
    units = df["unit"].astype("string").str.strip()
    sub = df.loc[units.notna() & (units != "")]
    if sub.empty:
        return {}
    grouped = sub.groupby("variable_id")["unit"].agg(lambda series: series.mode(dropna=True).iloc[0])
    return {str(variable_id): str(unit) for variable_id, unit in grouped.items() if str(unit).strip()}


def _catalog_variable_lookup(repo: DataRepo) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entry in repo.catalog.variables_manifest.get("variables", []):
        variable_id = str(entry.get("variable_id", "")).strip()
        if variable_id:
            lookup[variable_id] = dict(entry)
    return lookup


def register_all_variable_units(repo: DataRepo, *, log: LogFn = print) -> int:
    """Fill catalog ``unit`` fields from parquet data and curated overrides."""
    entries_by_id = _catalog_variable_lookup(repo)

    for table, domain in MEASUREMENT_TABLE_DOMAINS.items():
        harvested = _harvest_units_from_table(repo, table)
        for variable_id, unit in harvested.items():
            entry = entries_by_id.setdefault(
                variable_id,
                {"variable_id": variable_id, "domain": domain, "table": table},
            )
            if not str(entry.get("unit") or "").strip():
                entry["unit"] = unit

    for curated in CURATED_VARIABLE_UNITS:
        variable_id = str(curated["variable_id"])
        entry = entries_by_id.setdefault(
            variable_id,
            {
                "variable_id": variable_id,
                "domain": curated.get("domain"),
                "table": curated.get("table"),
            },
        )
        for key, value in curated.items():
            if key == "unit" or not str(entry.get(key) or "").strip():
                if value is not None and (not isinstance(value, str) or value.strip()):
                    entry[key] = value

    to_register = [
        entry
        for entry in entries_by_id.values()
        if str(entry.get("unit") or "").strip() and entry.get("variable_id")
    ]
    if to_register:
        repo.catalog.register_variables(to_register, merge=True)
    n = len(to_register)
    log(f"Registered units for {n} variables")
    return n
