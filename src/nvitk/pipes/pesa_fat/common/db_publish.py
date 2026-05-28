"""Publish PESA-Fat derived measurements into the NVITK DB.

Current policy:
- On successful stage3 per-subject measurement export, upsert long-form rows
  into the ``image_measurements`` table.
- Re-runs overwrite existing values for the same (subject, pipeline, variable, region, frame).

Measurement versioning/history can be added later by expanding the key and/or
adding a version column; for now we keep only the latest value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.db.storage import utc_now_iso

log = Logger()

PesaFatPipelineId = Literal["pesa_fat_ct_pet_v5", "pesa_fat_dixon_v5"]


@dataclass(frozen=True)
class PublishContext:
    pipeline_id: PesaFatPipelineId
    modality: str
    source_file: str = "pesa_fat_stage3"
    source_sheet: str = "stage3"


def _infer_publish_context(pipeline: str) -> PublishContext:
    pl = str(pipeline).strip().lower()
    if pl in {"ct-pet-v5", "ct_pet_v5", "ctpet", "pesa_fat_ct_pet_v5"}:
        return PublishContext(pipeline_id="pesa_fat_ct_pet_v5", modality="ctpet")
    if pl in {"dixon-v5", "dixon_v5", "dixon", "pesa_fat_dixon_v5"}:
        return PublishContext(pipeline_id="pesa_fat_dixon_v5", modality="dixon")
    raise ValueError(f"Unknown PESA-Fat pipeline {pipeline!r}")


def resolve_repo(repo: DataRepo | None = None) -> DataRepo:
    if repo is not None:
        return repo
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def _stable_region_id(column: str) -> str | None:
    # Dixon columns include a prefix like DIXON_<...>_HEAD/... in the variable name,
    # but we keep region_id empty for now unless an explicit token is present.
    c = str(column).upper()
    for region in ("HEAD", "THORAX", "LEGS"):
        if f"_{region}_" in c or c.startswith(f"{region}_") or c.endswith(f"_{region}"):
            return region
    return None


def build_image_measurement_rows_from_stage3_excel(
    *,
    subject_uid: str,
    excel_path: Path,
    pipeline: str,
) -> pd.DataFrame:
    """Read a 1-row stage3 Excel file and return long-form ``image_measurements`` rows."""
    ctx = _infer_publish_context(pipeline)
    df = pd.read_excel(excel_path)
    if df.empty:
        return pd.DataFrame()
    row = df.iloc[0].to_dict()

    out_rows: list[dict[str, Any]] = []
    updated_at = utc_now_iso()
    for col, val in row.items():
        col_s = str(col).strip()
        if not col_s:
            continue
        if col_s.lower() in {"pesa_id", "pesaid", "subject", "subject_uid"}:
            continue
        # Keep numeric values as float when possible; non-numeric become NA in value_num.
        v = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        region_id = _stable_region_id(col_s)
        out_rows.append(
            {
                "subject_uid": str(subject_uid),
                "session_id": pd.NA,
                "modality": ctx.modality,
                "region_id": region_id if region_id is not None else pd.NA,
                "region_label": pd.NA,
                "frame_index": pd.NA,
                "variable_id": col_s,
                "value_num": float(v) if pd.notna(v) else float("nan"),
                "value_text": pd.NA,
                "unit": pd.NA,
                "value_kind": "float",
                "pipeline_name": ctx.pipeline_id,
                "pipeline_id": ctx.pipeline_id,
                "qc_status": pd.NA,
                "source_asset": pd.NA,
                "source_table": f"{ctx.source_file}::{ctx.source_sheet}",
                "source_file": ctx.source_file,
                "source_sheet": ctx.source_sheet,
                "source_column": col_s,
                "source_batch_id": pd.NA,
                "measured_at": pd.NaT,
                "updated_at": updated_at,
            }
        )
    return pd.DataFrame(out_rows)


def publish_stage3_excel(
    *,
    subject_uid: str,
    excel_path: Path,
    pipeline: str,
    repo: DataRepo | None = None,
    build_sqlite_index: bool = True,
) -> pd.DataFrame:
    """Upsert rows into ``image_measurements`` with overwrite semantics."""
    repo = resolve_repo(repo)
    rows = build_image_measurement_rows_from_stage3_excel(
        subject_uid=subject_uid,
        excel_path=excel_path,
        pipeline=pipeline,
    )
    if rows.empty:
        log.warning("No stage3 measurements to publish: %s", excel_path)
        return rows
    # Overwrite existing values for the same subject/pipeline/variable (region/frame included when present).
    key = ["subject_uid", "pipeline_id", "variable_id", "region_id", "frame_index"]
    return repo.upsert_table(
        "image_measurements",
        rows,
        key_columns=key,
        provenance={
            "importer": "pesa_fat_stage3_publish",
            "pipeline": str(pipeline),
            "excel_path": str(excel_path),
        },
        build_sqlite_index=build_sqlite_index,
    )


__all__ = [
    "publish_stage3_excel",
    "build_image_measurement_rows_from_stage3_excel",
    "resolve_repo",
]

