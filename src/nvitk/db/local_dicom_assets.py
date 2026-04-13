"""Register local DICOM files under per-subject modality folders into ``assets``.

Expected layout::

    {dicom_root}/{subject_uid}/4Dflow_AP/**/*.dcm
    {dicom_root}/{subject_uid}/4Dflow_FH/**/*.dcm
    {dicom_root}/{subject_uid}/4Dflow_RL/**/*.dcm
    {dicom_root}/{subject_uid}/TOF/**/*.dcm

Each DICOM file becomes one ``assets`` row. ``asset_slot`` is a short, stable key per
sequence folder (e.g. ``tof``, ``4dflow_ap``), aligned with :mod:`nvitk.db.local_nifti_assets`
and XNAT sync. Wide tables use ``subject_uid`` + ``asset_slot`` (one column per sequence).

Files are included if they look like DICOM (``*.dcm``, ``*.dicom``, ``*.ima``, ``*.img``,
case-insensitive) or have no extension (common for raw exports); obvious junk names are skipped.

Do not store secrets in the dataset; this module only records file paths.

See also :mod:`nvitk.db.local_nifti_assets` for the NIfTI variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError

from .repo import DataRepo
from .storage import utc_now_iso

# (directory name under subject, modality, resource_label, asset_slot — same for all files in folder)
_SUBJECT_FOLDER_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("4Dflow_AP", "4dflow", "4Dflow/AP", "4dflow_ap"),
    ("4Dflow_FH", "4dflow", "4Dflow/FH", "4dflow_fh"),
    ("4Dflow_RL", "4dflow", "4Dflow/RL", "4dflow_rl"),
    ("TOF", "tof", "TOF", "tof"),
)

_JUNK_NAMES = frozenset(
    {
        "readme",
        "readme.txt",
        "readme.md",
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
    }
)


def _is_probably_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name
    low = name.lower()
    if low.startswith(".") or low in _JUNK_NAMES:
        return False
    suf = path.suffix.lower()
    if suf in (".dcm", ".dicom", ".ima", ".img"):
        return True
    if suf == "":
        return True
    return False


def _iter_dicoms_under(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return
    for p in sorted(folder.rglob("*")):
        if _is_probably_dicom_file(p):
            yield p


def register_dicom_tree(
    dicom_root: str | Path,
    *,
    source: str = "local_dicom",
    pipeline_id: str | None = None,
    pipeline_name: str | None = None,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Build ``assets`` rows for DICOM trees under ``dicom_root`` (one row per file)."""
    root = Path(dicom_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM root is not a directory: {root}")

    batch = source_batch_id or f"local_dicom_{utc_now_iso().replace(':', '').replace('-', '')}"
    now = utc_now_iso()
    rows: list[dict[str, Any]] = []

    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_uid = subject_dir.name

        for dir_name, modality, res_label, asset_slot in _SUBJECT_FOLDER_SPECS:
            mod_dir = subject_dir / dir_name
            if not mod_dir.is_dir():
                continue

            for fp in _iter_dicoms_under(mod_dir):
                rel = fp.relative_to(root).as_posix()
                try:
                    rel_within = fp.relative_to(mod_dir).as_posix()
                except ValueError:
                    rel_within = fp.name
                meta_obj: dict[str, Any] = {
                    "relative_path": rel,
                    "basename": fp.name,
                    "folder": dir_name,
                    "rel_within": rel_within.replace("\\", "/"),
                    "asset_slot": asset_slot,
                    "layout": "local_dicom_per_subject",
                }
                rows.append(
                    {
                        "asset_uid": f"local:{subject_uid}:dicom:{rel}",
                        "subject_uid": subject_uid,
                        "session_uid": pd.NA,
                        "modality": modality,
                        "asset_type": "dicom",
                        "asset_path": str(fp.resolve()),
                        "resource_label": res_label,
                        "source": source,
                        "pipeline_name": pipeline_name if pipeline_name is not None else pd.NA,
                        "pipeline_id": pipeline_id if pipeline_id is not None else pd.NA,
                        "exists_locally": fp.exists(),
                        "asset_slot": asset_slot,
                        "metadata_json": json.dumps(meta_obj),
                        "source_batch_id": batch,
                        "updated_at": now,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df


def upsert_dicom_assets(
    repo: DataRepo,
    dicom_root: str | Path,
    *,
    source: str = "local_dicom",
    pipeline_id: str | None = None,
    pipeline_name: str | None = None,
    source_batch_id: str | None = None,
    dry_run: bool = False,
    build_sqlite_index: bool = False,
) -> pd.DataFrame:
    """Compute DICOM asset rows and upsert into ``repo`` unless ``dry_run``."""
    df = register_dicom_tree(
        dicom_root,
        source=source,
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        source_batch_id=source_batch_id,
    )
    if dry_run or df.empty:
        return df
    repo.upsert_table(
        "assets",
        df,
        provenance={"source": "local_dicom", "dicom_root": str(Path(dicom_root).expanduser().resolve())},
        build_sqlite_index=build_sqlite_index,
    )
    return df


def _cli_decorator(*args: Any, **kwargs: Any):
    def decorator(func: Any) -> Any:
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


@_click_command()
@_click_option(
    "--dataset-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=Path("dataset"),
    show_default=True,
)
@_click_option(
    "--dicom-root",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Root containing {subject}/4Dflow_AP|4Dflow_FH|4Dflow_RL|TOF (DICOM trees).",
)
@_click_option("--source", type=str, default="local_dicom", show_default=True)
@_click_option("--pipeline-id", type=str, default=None)
@_click_option("--pipeline-name", type=str, default=None)
@_click_option("--source-batch-id", type=str, default=None)
@_click_option("--dry-run", is_flag=True, help="Print row count only; do not write Parquet.")
@_click_option("--build-sqlite-index", is_flag=True)
def main(
    dataset_root: Path,
    dicom_root: Path,
    source: str,
    pipeline_id: str | None,
    pipeline_name: str | None,
    source_batch_id: str | None,
    dry_run: bool,
    build_sqlite_index: bool,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    repo = DataRepo(dataset_root, auto_scaffold=True)
    df = upsert_dicom_assets(
        repo,
        dicom_root,
        source=source,
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        source_batch_id=source_batch_id,
        dry_run=dry_run,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Registered {len(df)} DICOM asset row(s).")


if __name__ == "__main__":
    main()
