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
from nvitk.db.exceptions import SettingsError, TableNotFoundError
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.db.storage import empty_dataframe, utc_now_iso, write_json, write_parquet_table

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
    source_batch_id: str | None = None,
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
    if source_batch_id and "source_batch_id" in rows.columns:
        rows = rows.copy()
        rows["source_batch_id"] = str(source_batch_id).strip()
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


PESA_FAT_QC_REVIEWS_TABLE = "pesa_fat_qc_reviews"
PESA_FAT_QC_REVIEWS_COLUMNS: dict[str, str] = {
    "batch": "string",
    "subject_uid": "string",
    "pipeline": "string",
    "structure": "string",
    "review_aspect": "string",
    "qc_status": "string",
    "reviewer": "string",
    "reviewed_at": "string",
    "comment": "string",
    "report_relpath": "string",
    "updated_at": "string",
}


def _variable_matches_structure(variable_id: str, structure: str) -> bool:
    """True when *variable_id* belongs to the QC structure label (e.g. HIGADO, DIXON_LIVER)."""
    vid = str(variable_id).strip()
    s = str(structure).strip()
    if not vid or not s:
        return False
    if vid == s:
        return True
    return vid.startswith(f"{s}_")


def _ensure_pesa_fat_qc_reviews_table(repo: DataRepo) -> None:
    """Register ``pesa_fat_qc_reviews`` in the dataset catalog when missing."""
    catalog = repo.catalog
    if catalog.table_exists(PESA_FAT_QC_REVIEWS_TABLE):
        return
    table_root = str(catalog.repository_manifest.get("table_root", "tables")).strip().rstrip("/")
    rel_path = f"{table_root}/{PESA_FAT_QC_REVIEWS_TABLE}.parquet"
    dest = catalog.root / rel_path
    write_parquet_table(dest, empty_dataframe(PESA_FAT_QC_REVIEWS_COLUMNS))
    payload: dict[str, Any] = {
        "path": rel_path,
        "kind": "derived",
        "description": "PESA-Fat QC portal review decisions (mirrors reviews.xlsx).",
        "key_columns": ["batch", "subject_uid", "pipeline", "structure", "review_aspect"],
        "columns": dict(PESA_FAT_QC_REVIEWS_COLUMNS),
        "row_count": 0,
        "last_updated": utc_now_iso(),
    }
    tables = catalog.tables_manifest.setdefault("tables", {})
    tables[PESA_FAT_QC_REVIEWS_TABLE] = payload
    catalog.tables_manifest["last_updated"] = utc_now_iso()
    write_json(catalog.tables_manifest_path, catalog.tables_manifest)
    catalog.refresh()


def publish_qc_review(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    structure: str,
    review_aspect: str = "MEASUREMENT",
    qc_status: str,
    reviewer: str = "",
    reviewed_at: str | None = None,
    comment: str = "",
    report_relpath: str = "",
    repo: DataRepo | None = None,
    build_sqlite_index: bool = False,
) -> dict[str, int]:
    """Persist a QC review to DB tables (``image_measurements`` + ``pesa_fat_qc_reviews``).

    - For ``review_aspect=MEASUREMENT``, sets ``qc_status`` on matching ``image_measurements``.
    - ``review_aspect=SEGMENTATION`` is audit-only in ``pesa_fat_qc_reviews``.
    - Upserts one audit row into ``pesa_fat_qc_reviews`` (created on first use if absent).
    """
    from nvitk.pipes.pesa_fat.qc.review_policy import DEFAULT_REVIEW_ASPECT

    repo = resolve_repo(repo)
    ctx = _infer_publish_context(pipeline)
    status = str(qc_status).strip().upper()
    if status not in {"PENDING", "OK", "FAIL"}:
        status = "PENDING"
    reviewed_at_eff = reviewed_at or utc_now_iso()
    subject_uid = str(subject).strip()
    structure_s = str(structure).strip()
    aspect_s = str(review_aspect or DEFAULT_REVIEW_ASPECT).strip().upper()

    updated_measurements = 0
    if aspect_s == "MEASUREMENT" and repo.catalog.table_exists("image_measurements"):
        all_df = repo.get("image_measurements", cohort_id=False)
        if not all_df.empty and "variable_id" in all_df.columns:
            sp_mask = (
                all_df["subject_uid"].astype("string").fillna("") == subject_uid
            ) & (all_df["pipeline_id"].astype("string").fillna("") == ctx.pipeline_id)
            var_match = all_df["variable_id"].astype("string").fillna("").map(
                lambda vid: _variable_matches_structure(vid, structure_s)
            )
            update_mask = sp_mask & var_match
            if update_mask.any():
                all_df = all_df.copy()
                all_df.loc[update_mask, "qc_status"] = status
                if "updated_at" in all_df.columns:
                    all_df.loc[update_mask, "updated_at"] = reviewed_at_eff
                if "source_batch_id" in all_df.columns and str(batch).strip():
                    all_df.loc[update_mask, "source_batch_id"] = str(batch).strip()
                repo.write_table(
                    "image_measurements",
                    all_df,
                    provenance={
                        "importer": "pesa_fat_qc_review",
                        "batch": str(batch),
                        "subject_uid": subject_uid,
                        "pipeline": str(pipeline),
                        "structure": structure_s,
                        "qc_status": status,
                    },
                    build_sqlite_index=build_sqlite_index,
                )
                updated_measurements = int(update_mask.sum())
            else:
                log.warning(
                    "QC review: no image_measurements rows for subject=%s pipeline=%s structure=%s",
                    subject_uid,
                    ctx.pipeline_id,
                    structure_s,
                )
    else:
        log.warning("QC review: image_measurements table not found in dataset catalog")

    _ensure_pesa_fat_qc_reviews_table(repo)
    review_row = pd.DataFrame(
        [
            {
                "batch": str(batch).strip(),
                "subject_uid": subject_uid,
                "pipeline": str(pipeline).strip(),
                "structure": structure_s,
                "review_aspect": aspect_s,
                "qc_status": status,
                "reviewer": str(reviewer).strip(),
                "reviewed_at": reviewed_at_eff,
                "comment": str(comment).strip(),
                "report_relpath": str(report_relpath).strip(),
                "updated_at": utc_now_iso(),
            }
        ]
    )
    repo.upsert_table(
        PESA_FAT_QC_REVIEWS_TABLE,
        review_row,
        key_columns=["batch", "subject_uid", "pipeline", "structure", "review_aspect"],
        provenance={
            "importer": "pesa_fat_qc_review",
            "batch": str(batch),
            "subject_uid": subject_uid,
            "structure": structure_s,
        },
        build_sqlite_index=build_sqlite_index,
    )

    return {
        "updated_measurements": updated_measurements,
        "qc_reviews": 1,
    }


def sync_qc_reviews_for_report(
    *,
    batch: str,
    subject: str,
    pipeline: str,
    rows: list[dict[str, Any]],
    repo: DataRepo | None = None,
) -> dict[str, Any]:
    """Push all reviews for one report to DB tables; rebuild SQLite index once at the end."""
    repo = resolve_repo(repo)
    batch_s = str(batch).strip()
    subject_uid = str(subject).strip()
    pipeline_s = str(pipeline).strip()
    ctx = _infer_publish_context(pipeline_s)

    report_rows = [
        r
        for r in rows
        if str(r.get("batch", "")).strip() == batch_s
        and str(r.get("subject", "")).strip() == subject_uid
        and str(r.get("pipeline", "")).strip() == pipeline_s
    ]
    if not report_rows:
        return {
            "synced_structures": 0,
            "updated_measurements": 0,
            "qc_reviews": 0,
        }

    from nvitk.pipes.pesa_fat.qc.review_policy import DEFAULT_REVIEW_ASPECT

    updated_measurements = 0
    if repo.catalog.table_exists("image_measurements"):
        all_df = repo.get("image_measurements", cohort_id=False)
        if not all_df.empty and "variable_id" in all_df.columns:
            all_df = all_df.copy()
            sp_mask = (
                all_df["subject_uid"].astype("string").fillna("") == subject_uid
            ) & (all_df["pipeline_id"].astype("string").fillna("") == ctx.pipeline_id)
            for r in report_rows:
                aspect_s = str(r.get("review_aspect") or DEFAULT_REVIEW_ASPECT).strip().upper()
                if aspect_s != "MEASUREMENT":
                    continue
                status = str(r.get("qc_status", "PENDING")).strip().upper()
                if status not in {"OK", "FAIL"}:
                    continue
                structure_s = str(r.get("structure", "")).strip()
                if not structure_s:
                    continue
                reviewed_at_eff = str(r.get("reviewed_at") or utc_now_iso())
                var_match = all_df["variable_id"].astype("string").fillna("").map(
                    lambda vid, s=structure_s: _variable_matches_structure(vid, s)
                )
                update_mask = sp_mask & var_match
                if not update_mask.any():
                    continue
                all_df.loc[update_mask, "qc_status"] = status
                if "updated_at" in all_df.columns:
                    all_df.loc[update_mask, "updated_at"] = reviewed_at_eff
                if "source_batch_id" in all_df.columns and batch_s:
                    all_df.loc[update_mask, "source_batch_id"] = batch_s
                updated_measurements += int(update_mask.sum())
            if updated_measurements > 0:
                repo.write_table(
                    "image_measurements",
                    all_df,
                    provenance={
                        "importer": "pesa_fat_qc_review_sync",
                        "batch": batch_s,
                        "subject_uid": subject_uid,
                        "pipeline": pipeline_s,
                    },
                    build_sqlite_index=False,
                )

    _ensure_pesa_fat_qc_reviews_table(repo)
    audit_rows: list[dict[str, Any]] = []
    synced = 0
    for r in report_rows:
        status = str(r.get("qc_status", "PENDING")).strip().upper()
        if status not in {"OK", "FAIL", "PENDING"}:
            status = "PENDING"
        structure_s = str(r.get("structure", "")).strip()
        aspect_s = str(r.get("review_aspect") or DEFAULT_REVIEW_ASPECT).strip().upper()
        if not structure_s:
            continue
        synced += 1
        audit_rows.append(
            {
                "batch": batch_s,
                "subject_uid": subject_uid,
                "pipeline": pipeline_s,
                "structure": structure_s,
                "review_aspect": aspect_s,
                "qc_status": status,
                "reviewer": str(r.get("reviewer", "")).strip(),
                "reviewed_at": str(r.get("reviewed_at") or utc_now_iso()),
                "comment": str(r.get("comment", "")).strip(),
                "report_relpath": str(r.get("report_relpath", "")).strip(),
                "updated_at": utc_now_iso(),
            }
        )
    if audit_rows:
        repo.upsert_table(
            PESA_FAT_QC_REVIEWS_TABLE,
            pd.DataFrame(audit_rows),
            key_columns=["batch", "subject_uid", "pipeline", "structure", "review_aspect"],
            provenance={
                "importer": "pesa_fat_qc_review_sync",
                "batch": batch_s,
                "subject_uid": subject_uid,
                "pipeline": pipeline_s,
            },
            build_sqlite_index=True,
        )

    return {
        "synced_structures": synced,
        "updated_measurements": updated_measurements,
        "qc_reviews": len(audit_rows),
    }


def try_sync_qc_reviews_for_report(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return sync_qc_reviews_for_report(**kwargs), None
    except SettingsError as exc:
        return None, f"database not configured ({exc})"
    except TableNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:
        log.warning("QC review DB sync failed: %s", exc)
        return None, str(exc)


def try_publish_qc_review(**kwargs: Any) -> tuple[dict[str, int] | None, str | None]:
    """Best-effort :func:`publish_qc_review`; returns ``(stats, error_message)``."""
    try:
        return publish_qc_review(**kwargs), None
    except SettingsError as exc:
        return None, f"database not configured ({exc})"
    except TableNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:
        log.warning("QC review DB publish failed: %s", exc)
        return None, str(exc)


__all__ = [
    "PESA_FAT_QC_REVIEWS_TABLE",
    "publish_stage3_excel",
    "publish_qc_review",
    "sync_qc_reviews_for_report",
    "try_publish_qc_review",
    "try_sync_qc_reviews_for_report",
    "build_image_measurement_rows_from_stage3_excel",
    "resolve_repo",
    "_variable_matches_structure",
]

