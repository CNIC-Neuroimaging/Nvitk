"""Publish qvtpy (4D-Flow) stage-6 measurements into the NVITK ``image_measurements`` table.

Long-form rows are upserted under pipeline ``4dflow_v3`` (aliases ``qvtpy`` / ``latest``
/ ``v3``; see :func:`qvtpy_pipeline_catalog_spec`). Re-runs overwrite existing values
for the same ``(subject_uid, pipeline_id, variable_id, region_id, frame_index)``.

Sources (written by :mod:`nvitk.pipes.qvtpy.stage6_measure`):

- ``loc_measurements.csv`` — per-LOC ``pi``, ``ri``, ``flow_mean`` (mL/min), and the
  per-frame ``flow_tseries`` (mL/min).
- ``vessel_hemodynamics.csv`` — per-root ``pitc_slope`` / ``pitc_intercept`` / ``pwv``
  (Bjornfoot) / ``pwv_fielding_xcor`` and per-branch ``damping_index``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.exceptions import SettingsError, TableNotFoundError
from nvitk.db.repo import DataRepo, get_repo, get_repo_from_settings
from nvitk.db.settings_paths import sge_dataset_root_path
from nvitk.db.storage import utc_now_iso

log = Logger()

QVTPY_PIPELINE_ID = "4dflow_v3"
QVTPY_PIPELINE_NAME = "QVTPy"
QVTPY_MODALITY = "4dflow"
QVTPY_PIPELINE_ALIASES = ("qvtpy", "latest", "v3", "3")

_UPSERT_KEY = ["subject_uid", "pipeline_id", "variable_id", "region_id", "frame_index"]
_ML_S_TO_ML_MIN = 60.0


def qvtpy_pipeline_catalog_spec() -> dict[str, Any]:
    """Dataset catalog entry registering ``4dflow_v3`` as the default 4D-flow pipeline."""
    return {
        "pipeline_id": QVTPY_PIPELINE_ID,
        "pipeline_name": QVTPY_PIPELINE_NAME,
        "modality": QVTPY_MODALITY,
        "is_default": True,
        "aliases": list(QVTPY_PIPELINE_ALIASES),
    }


def resolve_repo(repo: DataRepo | None = None, *, prefer_sge: bool | None = None) -> DataRepo:
    """Resolve a :class:`DataRepo`, preferring the cluster dataset under SGE."""
    if repo is not None:
        return repo
    use_sge = prefer_sge
    if use_sge is None:
        use_sge = os.environ.get("NVITK_SGE", "").lower() in ("1", "true", "yes")
    if use_sge and sge_dataset_root_path(must_exist=True) is not None:
        return get_repo(prefer_sge=True)
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def _sge_db_publish_enabled() -> bool:
    if os.environ.get("NVITK_SGE", "").lower() not in ("1", "true", "yes"):
        return False
    return sge_dataset_root_path(must_exist=True) is not None


def _measurement_row(
    *,
    subject_uid: str,
    variable_id: str,
    value: float,
    region_id: str,
    frame_index: Any,
    unit: str,
    source_file: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "subject_uid": str(subject_uid),
        "session_id": pd.NA,
        "modality": QVTPY_MODALITY,
        "region_id": str(region_id) if region_id else pd.NA,
        "region_label": str(region_id) if region_id else pd.NA,
        "frame_index": frame_index if frame_index is not None else pd.NA,
        "variable_id": str(variable_id),
        "value_num": float(value),
        "value_text": pd.NA,
        "unit": unit,
        "value_kind": "float",
        "pipeline_id": QVTPY_PIPELINE_ID,
        "pipeline_name": QVTPY_PIPELINE_NAME,
        "qc_status": pd.NA,
        "source_asset": pd.NA,
        "source_table": f"qvtpy_stage6::{source_file}",
        "source_file": source_file,
        "source_sheet": pd.NA,
        "source_column": pd.NA,
        "source_batch_id": pd.NA,
        "measured_at": pd.NaT,
        "updated_at": updated_at,
    }


def _finite(value: Any) -> float | None:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return None
    return float(v)


def _rows_from_loc_measurements(
    subject_uid: str, loc_csv: Path, updated_at: str
) -> list[dict[str, Any]]:
    if not loc_csv.is_file():
        return []
    df = pd.read_csv(loc_csv)
    if df.empty:
        return []
    flow_cols = sorted(
        (c for c in df.columns if str(c).startswith("loc_flow_ml_s_t")),
        key=lambda c: int(str(c).rsplit("_t", 1)[1]),
    )
    out: list[dict[str, Any]] = []
    src = "loc_measurements.csv"
    for _, row in df.iterrows():
        region = str(row.get("vessel_name") or "").strip()
        if not region:
            continue
        scalar_map = {
            "pi": (_finite(row.get("loc_pi")), "dimensionless"),
            "ri": (_finite(row.get("loc_ri")), "dimensionless"),
        }
        mean_flow = _finite(row.get("loc_mean_flow_ml_s"))
        if mean_flow is not None:
            scalar_map["flow_mean"] = (mean_flow * _ML_S_TO_ML_MIN, "mL/min")
        for var, (val, unit) in scalar_map.items():
            if val is None:
                continue
            out.append(
                _measurement_row(
                    subject_uid=subject_uid,
                    variable_id=var,
                    value=val,
                    region_id=region,
                    frame_index=pd.NA,
                    unit=unit,
                    source_file=src,
                    updated_at=updated_at,
                )
            )
        for col in flow_cols:
            val = _finite(row.get(col))
            if val is None:
                continue
            frame = int(str(col).rsplit("_t", 1)[1])
            out.append(
                _measurement_row(
                    subject_uid=subject_uid,
                    variable_id="flow_tseries",
                    value=val * _ML_S_TO_ML_MIN,
                    region_id=region,
                    frame_index=frame,
                    unit="mL/min",
                    source_file=src,
                    updated_at=updated_at,
                )
            )
    return out


def _rows_from_vessel_hemodynamics(
    subject_uid: str, vessel_csv: Path, updated_at: str
) -> list[dict[str, Any]]:
    if not vessel_csv.is_file():
        return []
    df = pd.read_csv(vessel_csv)
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    src = "vessel_hemodynamics.csv"
    var_units = {
        "pitc_slope": "1/mm",
        "pitc_intercept": "dimensionless",
        "pwv": "m/s",
        "pwv_fielding_xcor": "m/s",
        "damping_index": "dimensionless",
    }
    col_to_var = {
        "pitc_slope": "pitc_slope",
        "pitc_intercept": "pitc_intercept",
        "pwv_bjornfoot_m_s": "pwv",
        "pwv_fielding_m_s": "pwv_fielding_xcor",
        "damping_index": "damping_index",
    }
    for _, row in df.iterrows():
        region = str(row.get("region_id") or "").strip()
        if not region:
            continue
        for col, var in col_to_var.items():
            if col not in df.columns:
                continue
            val = _finite(row.get(col))
            if val is None:
                continue
            out.append(
                _measurement_row(
                    subject_uid=subject_uid,
                    variable_id=var,
                    value=val,
                    region_id=region,
                    frame_index=pd.NA,
                    unit=var_units[var],
                    source_file=src,
                    updated_at=updated_at,
                )
            )
    return out


def build_image_measurement_rows_from_stage6(
    *,
    subject_uid: str,
    stage6_dir: Path,
) -> pd.DataFrame:
    """Long-form ``image_measurements`` rows from stage-6 CSV outputs."""
    stage6_dir = Path(stage6_dir)
    updated_at = utc_now_iso()
    rows = _rows_from_loc_measurements(
        subject_uid, stage6_dir / "loc_measurements.csv", updated_at
    )
    rows += _rows_from_vessel_hemodynamics(
        subject_uid, stage6_dir / "vessel_hemodynamics.csv", updated_at
    )
    return pd.DataFrame(rows)


def publish_stage6(
    *,
    subject_uid: str,
    stage6_dir: Path,
    repo: DataRepo | None = None,
    prefer_sge: bool | None = None,
    build_sqlite_index: bool = True,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Upsert stage-6 measurements into ``image_measurements`` (overwrite semantics)."""
    repo = resolve_repo(repo, prefer_sge=prefer_sge)
    rows = build_image_measurement_rows_from_stage6(
        subject_uid=subject_uid, stage6_dir=stage6_dir
    )
    if rows.empty:
        log.warning("No qvtpy stage6 measurements to publish: %s", stage6_dir)
        return rows
    if source_batch_id and "source_batch_id" in rows.columns:
        rows = rows.copy()
        rows["source_batch_id"] = str(source_batch_id).strip()
    return repo.upsert_table(
        "image_measurements",
        rows,
        key_columns=_UPSERT_KEY,
        provenance={
            "importer": "qvtpy_stage6_publish",
            "pipeline": QVTPY_PIPELINE_ID,
            "stage6_dir": str(stage6_dir),
        },
        build_sqlite_index=build_sqlite_index,
    )


def maybe_publish_stage6_on_sge(*, subject_uid: str, stage6_dir: Path) -> None:
    """Upsert stage-6 measurements into the cluster DB when running under SGE."""
    if not _sge_db_publish_enabled():
        return
    try:
        rows = publish_stage6(
            subject_uid=subject_uid,
            stage6_dir=stage6_dir,
            prefer_sge=True,
            build_sqlite_index=False,
        )
        if not rows.empty:
            log.info(
                "SGE DB: published %d image_measurements row(s) for %s / %s",
                len(rows),
                subject_uid,
                QVTPY_PIPELINE_ID,
            )
    except Exception as exc:
        log.warning(
            "SGE DB publish skipped for %s / %s (%s): %s",
            subject_uid,
            QVTPY_PIPELINE_ID,
            stage6_dir,
            exc,
        )


def try_publish_stage6(**kwargs: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Best-effort :func:`publish_stage6`; returns ``(rows, error_message)``."""
    try:
        return publish_stage6(**kwargs), None
    except SettingsError as exc:
        return None, f"database not configured ({exc})"
    except TableNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("qvtpy stage6 DB publish failed: %s", exc)
        return None, str(exc)


__all__ = [
    "QVTPY_MODALITY",
    "QVTPY_PIPELINE_ALIASES",
    "QVTPY_PIPELINE_ID",
    "QVTPY_PIPELINE_NAME",
    "build_image_measurement_rows_from_stage6",
    "maybe_publish_stage6_on_sge",
    "publish_stage6",
    "qvtpy_pipeline_catalog_spec",
    "resolve_repo",
    "try_publish_stage6",
]
