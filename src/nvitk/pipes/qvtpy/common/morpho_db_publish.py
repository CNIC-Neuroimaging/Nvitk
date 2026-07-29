"""Publish qvtpy stage-7 TOF morphometrics into the NVITK ``image_measurements`` table.

Long-form rows are upserted under pipeline ``tof_morpho_v1``. Per-vessel scalar
summaries are aggregated from the ``00_Path_Summary`` sheet of
``case_metrics_donut_tree.xlsx``.

When ``stenosis_percent_max`` / length totals are absent (older Excel exports),
values are recovered from ``*_segments_detail_json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.exceptions import SettingsError, TableNotFoundError
from nvitk.db.repo import DataRepo
from nvitk.db.storage import utc_now_iso
from nvitk.pipes.qvtpy.common.db_publish import _UPSERT_KEY, _finite, resolve_repo

log = Logger()

TOF_MORPHO_PIPELINE_ID = "tof_morpho_v1"
TOF_MORPHO_PIPELINE_NAME = "TOF Morphometrics"
TOF_MORPHO_MODALITY = "tof"
TOF_MORPHO_PIPELINE_ALIASES = ("tof_morpho", "morpho", "tof_morpho_v1")

# variable_id -> unit
_SCALAR_VARS: dict[str, str] = {
    "length_mm": "mm",
    "radius_mean_mm": "mm",
    "radius_max_mm": "mm",
    "tortuosity_dm": "dimensionless",
    "curvature_mean_1_per_mm": "1/mm",
    "curvature_p95_1_per_mm": "1/mm",
    "stenosis_percent_max": "%",
    "stenosis_segments_n": "count",
    "stenosis_length_total_mm": "mm",
    "radius_min_stenotic_mm": "mm",
    "enlargement_percent_max": "%",
    "enlargement_segments_n": "count",
    "enlargement_length_total_mm": "mm",
    "radius_max_enlarged_mm": "mm",
}


def tof_morpho_pipeline_catalog_spec() -> dict[str, Any]:
    """Dataset catalog entry for the TOF morphometrics pipeline."""
    return {
        "pipeline_id": TOF_MORPHO_PIPELINE_ID,
        "pipeline_name": TOF_MORPHO_PIPELINE_NAME,
        "modality": TOF_MORPHO_MODALITY,
        "is_default": True,
        "aliases": list(TOF_MORPHO_PIPELINE_ALIASES),
    }


def _sge_db_publish_enabled() -> bool:
    from nvitk.pipes.qvtpy.common.db_publish import _sge_db_publish_enabled

    return _sge_db_publish_enabled()


def _measurement_row(
    *,
    subject_uid: str,
    variable_id: str,
    value: float,
    region_id: str,
    unit: str,
    source_file: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "subject_uid": str(subject_uid),
        "session_id": pd.NA,
        "modality": TOF_MORPHO_MODALITY,
        "region_id": str(region_id) if region_id else pd.NA,
        "region_label": str(region_id) if region_id else pd.NA,
        "frame_index": pd.NA,
        "variable_id": str(variable_id),
        "value_num": float(value),
        "value_text": pd.NA,
        "unit": unit,
        "value_kind": "float",
        "pipeline_id": TOF_MORPHO_PIPELINE_ID,
        "pipeline_name": TOF_MORPHO_PIPELINE_NAME,
        "qc_status": pd.NA,
        "source_asset": pd.NA,
        "source_table": f"qvtpy_stage7::{source_file}",
        "source_file": source_file,
        "source_sheet": "00_Path_Summary",
        "source_column": pd.NA,
        "source_batch_id": pd.NA,
        "measured_at": pd.NaT,
        "updated_at": updated_at,
    }


def _parse_segment_details(raw: Any) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return []
    if isinstance(raw, list):
        return [d for d in raw if isinstance(d, dict)]
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _segment_degree_max(raw: Any) -> float | None:
    degrees = [
        float(d["degree_pct"])
        for d in _parse_segment_details(raw)
        if d.get("degree_pct") is not None and np.isfinite(float(d["degree_pct"]))
    ]
    return float(max(degrees)) if degrees else None


def _segment_length_sum(raw: Any) -> float | None:
    lengths = [
        float(d["length_mm"])
        for d in _parse_segment_details(raw)
        if d.get("length_mm") is not None and np.isfinite(float(d["length_mm"]))
    ]
    return float(sum(lengths)) if lengths else None


def _enrich_path_summary_from_segment_json(path_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing stenosis/enlargement scalars from segment detail JSON."""
    df = path_df.copy()

    if "stenosis_segments_detail_json" in df.columns:
        if "stenosis_percent_max" not in df.columns:
            df["stenosis_percent_max"] = np.nan
        if "stenosis_length_total_mm" not in df.columns:
            df["stenosis_length_total_mm"] = np.nan
        for idx in df.index:
            if not np.isfinite(pd.to_numeric(df.at[idx, "stenosis_percent_max"], errors="coerce")):
                deg = _segment_degree_max(df.at[idx, "stenosis_segments_detail_json"])
                if deg is not None:
                    df.at[idx, "stenosis_percent_max"] = deg
            if not np.isfinite(pd.to_numeric(df.at[idx, "stenosis_length_total_mm"], errors="coerce")):
                length = _segment_length_sum(df.at[idx, "stenosis_segments_detail_json"])
                if length is not None:
                    df.at[idx, "stenosis_length_total_mm"] = length

    if "enlargement_segments_detail_json" in df.columns:
        if "enlargement_percent_max" not in df.columns:
            df["enlargement_percent_max"] = np.nan
        if "enlargement_length_total_mm" not in df.columns:
            df["enlargement_length_total_mm"] = np.nan
        for idx in df.index:
            if not np.isfinite(pd.to_numeric(df.at[idx, "enlargement_percent_max"], errors="coerce")):
                deg = _segment_degree_max(df.at[idx, "enlargement_segments_detail_json"])
                if deg is not None:
                    df.at[idx, "enlargement_percent_max"] = deg
            if not np.isfinite(pd.to_numeric(df.at[idx, "enlargement_length_total_mm"], errors="coerce")):
                length = _segment_length_sum(df.at[idx, "enlargement_segments_detail_json"])
                if length is not None:
                    df.at[idx, "enlargement_length_total_mm"] = length

    return df


def _aggregate_vessel_metrics(path_df: pd.DataFrame) -> pd.DataFrame:
    if path_df.empty or "vessel_name" not in path_df.columns:
        return pd.DataFrame()

    path_df = _enrich_path_summary_from_segment_json(path_df)

    rows: list[dict[str, Any]] = []
    for vessel, group in path_df.groupby("vessel_name", dropna=True):
        region = str(vessel).strip()
        if not region:
            continue

        def _col(col: str) -> pd.Series:
            if col not in group.columns:
                return pd.Series(np.nan, index=group.index, dtype=float)
            return pd.to_numeric(group[col], errors="coerce")

        lengths = _col("length_mm")
        valid = lengths.notna() & np.isfinite(lengths) & (lengths > 0)
        if not valid.any():
            continue

        def _wmean(col: str) -> float | None:
            vals = _col(col)
            good = valid & vals.notna() & np.isfinite(vals)
            if not good.any():
                return None
            return float(np.average(vals[good], weights=lengths[good]))

        def _max(col: str) -> float | None:
            vals = _col(col)
            good = vals.notna() & np.isfinite(vals)
            if not good.any():
                return None
            return float(vals[good].max())

        def _min(col: str) -> float | None:
            vals = _col(col)
            good = vals.notna() & np.isfinite(vals)
            if not good.any():
                return None
            return float(vals[good].min())

        def _sum(col: str) -> float | None:
            vals = _col(col)
            good = vals.notna() & np.isfinite(vals)
            if not good.any():
                return None
            return float(vals[good].sum())

        radius_max = _max("radius_p95_mm")
        if radius_max is None:
            radius_max = _max("radius_max_enlarged_mm")

        row = {
            "region_id": region,
            "length_mm": float(lengths[valid].sum()),
            "radius_mean_mm": _wmean("radius_mean_mm"),
            "radius_max_mm": radius_max,
            "tortuosity_dm": _wmean("tortuosity_dm"),
            "curvature_mean_1_per_mm": _wmean("curvature_mean_1_per_mm"),
            "curvature_p95_1_per_mm": _wmean("curvature_p95_1_per_mm"),
            "stenosis_percent_max": _max("stenosis_percent_max"),
            "stenosis_segments_n": _sum("stenosis_segments_n"),
            "stenosis_length_total_mm": _sum("stenosis_length_total_mm"),
            "radius_min_stenotic_mm": _min("radius_min_stenotic_mm"),
            "enlargement_percent_max": _max("enlargement_percent_max"),
            "enlargement_segments_n": _sum("enlargement_segments_n"),
            "enlargement_length_total_mm": _sum("enlargement_length_total_mm"),
            "radius_max_enlarged_mm": _max("radius_max_enlarged_mm"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_image_measurement_rows_from_stage7(
    *,
    subject_uid: str,
    stage7_dir: Path,
) -> pd.DataFrame:
    """Long-form ``image_measurements`` rows from stage-7 morphometrics Excel."""
    stage7_dir = Path(stage7_dir)
    excel_path = stage7_dir / "case_metrics_donut_tree.xlsx"
    if not excel_path.is_file():
        return pd.DataFrame()

    try:
        path_df = pd.read_excel(excel_path, sheet_name="00_Path_Summary")
    except Exception as exc:
        log.warning("Failed reading %s: %s", excel_path, exc)
        return pd.DataFrame()

    agg = _aggregate_vessel_metrics(path_df)
    if agg.empty:
        return pd.DataFrame()

    updated_at = utc_now_iso()
    src = excel_path.name
    rows: list[dict[str, Any]] = []
    for _, vessel_row in agg.iterrows():
        region = str(vessel_row.get("region_id") or "").strip()
        if not region:
            continue
        for var, unit in _SCALAR_VARS.items():
            val = _finite(vessel_row.get(var))
            if val is None:
                continue
            rows.append(
                _measurement_row(
                    subject_uid=subject_uid,
                    variable_id=var,
                    value=val,
                    region_id=region,
                    unit=unit,
                    source_file=src,
                    updated_at=updated_at,
                )
            )
    return pd.DataFrame(rows)


def publish_stage7(
    *,
    subject_uid: str,
    stage7_dir: Path,
    repo: DataRepo | None = None,
    prefer_sge: bool | None = None,
    build_sqlite_index: bool = True,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Upsert stage-7 TOF morphometrics into ``image_measurements``."""
    repo = resolve_repo(repo, prefer_sge=prefer_sge)
    rows = build_image_measurement_rows_from_stage7(
        subject_uid=subject_uid, stage7_dir=stage7_dir
    )
    if rows.empty:
        log.warning("No qvtpy stage7 morphometrics to publish: %s", stage7_dir)
        return rows
    if source_batch_id and "source_batch_id" in rows.columns:
        rows = rows.copy()
        rows["source_batch_id"] = str(source_batch_id).strip()
    return repo.upsert_table(
        "image_measurements",
        rows,
        key_columns=_UPSERT_KEY,
        provenance={
            "importer": "qvtpy_stage7_publish",
            "pipeline": TOF_MORPHO_PIPELINE_ID,
            "stage7_dir": str(stage7_dir),
        },
        build_sqlite_index=build_sqlite_index,
    )


def maybe_publish_stage7_on_sge(*, subject_uid: str, stage7_dir: Path) -> None:
    """Upsert stage-7 measurements into the cluster DB when running under SGE."""
    if not _sge_db_publish_enabled():
        return
    try:
        rows = publish_stage7(
            subject_uid=subject_uid,
            stage7_dir=stage7_dir,
            prefer_sge=True,
            build_sqlite_index=False,
        )
        if not rows.empty:
            log.info(
                "SGE DB: published %d image_measurements row(s) for %s / %s",
                len(rows),
                subject_uid,
                TOF_MORPHO_PIPELINE_ID,
            )
    except Exception as exc:
        log.warning(
            "SGE DB publish skipped for %s / %s (%s): %s",
            subject_uid,
            TOF_MORPHO_PIPELINE_ID,
            stage7_dir,
            exc,
        )


def try_publish_stage7(**kwargs: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Best-effort :func:`publish_stage7`; returns ``(rows, error_message)``."""
    try:
        return publish_stage7(**kwargs), None
    except SettingsError as exc:
        return None, f"database not configured ({exc})"
    except TableNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("qvtpy stage7 DB publish failed: %s", exc)
        return None, str(exc)


__all__ = [
    "TOF_MORPHO_MODALITY",
    "TOF_MORPHO_PIPELINE_ALIASES",
    "TOF_MORPHO_PIPELINE_ID",
    "TOF_MORPHO_PIPELINE_NAME",
    "build_image_measurement_rows_from_stage7",
    "maybe_publish_stage7_on_sge",
    "publish_stage7",
    "tof_morpho_pipeline_catalog_spec",
    "try_publish_stage7",
]
