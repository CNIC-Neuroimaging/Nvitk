"""Register qvtpy / eICAB pipeline outputs in the ``assets`` catalog table.

Supports local results trees (``<results>/<subject>/eicab`` and ``.../qvtpy``) and
rows produced by :mod:`nvitk.db.xnat_pipeline_resources` from XNAT experiment
resources with the same labels (``eicab``, ``qvtpy``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.storage import utc_now_iso
from nvitk.pipes.qvtpy import config as qvt_cfg
from nvitk.pipes.qvtpy.stage1_eicab import _output_has_segmentation
from nvitk.pipes.qvtpy.util.qc_report import check_subject_stages

from .repo import DataRepo

log = Logger()

XNAT_RESOURCE_EICAB = qvt_cfg.STAGE1_EICAB_DIR
XNAT_RESOURCE_QVTPY = qvt_cfg.QVT_SUBDIR

QVTPY_PIPELINE_RESOURCES: tuple[str, ...] = (XNAT_RESOURCE_EICAB, XNAT_RESOURCE_QVTPY)

PIPELINE_RESOURCE_TO_SLOT: dict[str, str] = {
    XNAT_RESOURCE_EICAB: "pipeline_eicab",
    XNAT_RESOURCE_QVTPY: "pipeline_qvtpy",
}

PIPELINE_RESOURCE_TO_PIPELINE_ID: dict[str, str] = {
    XNAT_RESOURCE_EICAB: "eicab_v1",
    XNAT_RESOURCE_QVTPY: "qvtpy_v1",
}

PIPELINE_RESOURCE_TO_NAME: dict[str, str] = {
    XNAT_RESOURCE_EICAB: "eICAB",
    XNAT_RESOURCE_QVTPY: "QVTPy",
}

_DEFAULT_QVTPY_QC_STAGES: tuple[str, ...] = (
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
)


def resource_label_to_asset_slot(resource_label: str) -> str:
    key = str(resource_label).strip().lower()
    return PIPELINE_RESOURCE_TO_SLOT.get(key, f"pipeline_{key}")


def resource_label_to_pipeline_id(resource_label: str) -> str:
    key = str(resource_label).strip().lower()
    return PIPELINE_RESOURCE_TO_PIPELINE_ID.get(key, f"pipeline_{key}")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _count_tree_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def _list_nifti_relpaths(root: Path, *, limit: int = 40) -> list[str]:
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not _is_nifti(p):
            continue
        out.append(p.relative_to(root).as_posix())
        if len(out) >= limit:
            break
    return out


def describe_local_eicab_resource(resource_dir: Path) -> dict[str, Any]:
    complete = _output_has_segmentation(resource_dir)
    return {
        "resource_label": XNAT_RESOURCE_EICAB,
        "complete": bool(complete),
        "n_files": _count_tree_files(resource_dir),
        "nifti_examples": _list_nifti_relpaths(resource_dir),
    }


def describe_local_qvtpy_resource(
    resource_dir: Path,
    *,
    required_stages: Iterable[str] = _DEFAULT_QVTPY_QC_STAGES,
) -> dict[str, Any]:
    subject = resource_dir.parent.name
    results_root = resource_dir.parent.parent
    checks = check_subject_stages(
        subject,
        list(required_stages),
        results_root=results_root,
        nifti_root=results_root,
    )
    stage_status = {c.stage: {"complete": c.complete, "detail": c.detail} for c in checks}
    complete = all(c.complete for c in checks)
    return {
        "resource_label": XNAT_RESOURCE_QVTPY,
        "complete": bool(complete),
        "n_files": _count_tree_files(resource_dir),
        "required_stages": list(required_stages),
        "stage_status": stage_status,
        "nifti_examples": _list_nifti_relpaths(resource_dir),
    }


def describe_local_pipeline_resource(
    resource_dir: Path,
    resource_label: str,
    *,
    required_qvtpy_stages: Iterable[str] = _DEFAULT_QVTPY_QC_STAGES,
) -> dict[str, Any]:
    label = str(resource_label).strip().lower()
    if label == XNAT_RESOURCE_EICAB:
        return describe_local_eicab_resource(resource_dir)
    if label == XNAT_RESOURCE_QVTPY:
        return describe_local_qvtpy_resource(
            resource_dir,
            required_stages=required_qvtpy_stages,
        )
    return {
        "resource_label": label,
        "complete": resource_dir.is_dir() and any(resource_dir.rglob("*")),
        "n_files": _count_tree_files(resource_dir),
        "nifti_examples": _list_nifti_relpaths(resource_dir),
    }


def _bundle_asset_row(
    *,
    subject_uid: str,
    resource_label: str,
    source: str,
    asset_path: str,
    exists_locally: bool,
    session_uid: Any = pd.NA,
    experiment_label: str = "",
    source_batch_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slot = resource_label_to_asset_slot(resource_label)
    pipeline_id = resource_label_to_pipeline_id(resource_label)
    pipeline_name = PIPELINE_RESOURCE_TO_NAME.get(resource_label, resource_label)
    meta: dict[str, Any] = {
        "resource_label": resource_label,
        "asset_slot": slot,
        "bundle": True,
    }
    if experiment_label:
        meta["experiment_label"] = experiment_label
    if extra_meta:
        meta.update(extra_meta)
    sess = session_uid if session_uid is not None else pd.NA
    asset_uid = f"{source}:{subject_uid}:{sess}:{resource_label}:bundle"
    return {
        "asset_uid": asset_uid,
        "subject_uid": subject_uid,
        "session_uid": sess,
        "modality": "pipeline",
        "asset_type": "pipeline_bundle",
        "asset_path": asset_path,
        "resource_label": resource_label,
        "source": source,
        "pipeline_name": pipeline_name,
        "pipeline_id": pipeline_id,
        "exists_locally": bool(exists_locally),
        "asset_slot": slot,
        "metadata_json": json.dumps(meta),
        "source_batch_id": source_batch_id,
        "updated_at": utc_now_iso(),
    }


def register_local_pipeline_tree(
    results_root: str | Path,
    *,
    resources: Iterable[str] = QVTPY_PIPELINE_RESOURCES,
    source: str = "local_pipeline",
    source_batch_id: str | None = None,
    required_qvtpy_stages: Iterable[str] = _DEFAULT_QVTPY_QC_STAGES,
) -> pd.DataFrame:
    """Build ``assets`` rows for local ``eicab`` / ``qvtpy`` subject directories."""
    root = Path(results_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results root is not a directory: {root}")

    batch = source_batch_id or f"local_pipeline_{utc_now_iso().replace(':', '').replace('-', '')}"
    rows: list[dict[str, Any]] = []
    resource_set = {str(r).strip().lower() for r in resources if str(r).strip()}

    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_uid = subject_dir.name
        for resource_label in QVTPY_PIPELINE_RESOURCES:
            if resource_label not in resource_set:
                continue
            resource_dir = subject_dir / resource_label
            if not resource_dir.is_dir():
                continue
            detail = describe_local_pipeline_resource(
                resource_dir,
                resource_label,
                required_qvtpy_stages=required_qvtpy_stages,
            )
            rows.append(
                _bundle_asset_row(
                    subject_uid=subject_uid,
                    resource_label=resource_label,
                    source=source,
                    asset_path=str(resource_dir.resolve()),
                    exists_locally=True,
                    source_batch_id=batch,
                    extra_meta=detail,
                )
            )

    return pd.DataFrame(rows)


def upsert_local_pipeline_assets(
    repo: DataRepo,
    results_root: str | Path,
    *,
    resources: Iterable[str] = QVTPY_PIPELINE_RESOURCES,
    source: str = "local_pipeline",
    source_batch_id: str | None = None,
    required_qvtpy_stages: Iterable[str] = _DEFAULT_QVTPY_QC_STAGES,
    dry_run: bool = False,
    build_sqlite_index: bool = False,
) -> pd.DataFrame:
    df = register_local_pipeline_tree(
        results_root,
        resources=resources,
        source=source,
        source_batch_id=source_batch_id,
        required_qvtpy_stages=required_qvtpy_stages,
    )
    if dry_run or df.empty:
        return df
    repo.upsert_table(
        "assets",
        df,
        provenance={
            "source": "local_pipeline",
            "results_root": str(Path(results_root).expanduser().resolve()),
        },
        build_sqlite_index=build_sqlite_index,
    )
    log.info(f"Indexed {len(df)} local pipeline asset row(s) from {results_root}")
    return df


__all__ = [
    "PIPELINE_RESOURCE_TO_PIPELINE_ID",
    "PIPELINE_RESOURCE_TO_SLOT",
    "QVTPY_PIPELINE_RESOURCES",
    "XNAT_RESOURCE_EICAB",
    "XNAT_RESOURCE_QVTPY",
    "describe_local_eicab_resource",
    "describe_local_pipeline_resource",
    "describe_local_qvtpy_resource",
    "register_local_pipeline_tree",
    "resource_label_to_asset_slot",
    "resource_label_to_pipeline_id",
    "upsert_local_pipeline_assets",
]
