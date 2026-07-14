"""Sync qvtpy / eICAB XNAT experiment resources into the dataset ``assets`` table."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError
from nvitk.core.logger import Logger

from .pipeline_assets import (
    QVTPY_PIPELINE_RESOURCES,
    XNAT_RESOURCE_EICAB,
    XNAT_RESOURCE_QVTPY,
    _bundle_asset_row,
    describe_local_eicab_resource,
    describe_local_qvtpy_resource,
    resource_label_to_asset_slot,
)
from .repo import DataRepo
from .storage import utc_now_iso
from .xnat import connect_xnat, resolve_subject_labels
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection
from .xnat_upload import (
    iter_upload_files,
    resolve_subject_experiment,
    xnat_resource_has_files,
)

log = Logger()

_DEFAULT_QVTPY_QC_STAGES: tuple[str, ...] = (
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
)


def _resource_file_count(resource: Any) -> int:
    files = getattr(resource, "files", None)
    if files is None:
        return 0
    if callable(files):
        try:
            files = files()
        except Exception:
            return 0
    try:
        return int(len(files))
    except TypeError:
        try:
            return len(list(files))
        except Exception:
            return 0


def _experiment_session_uid(
    repo: DataRepo,
    *,
    project_id: str,
    subject_uid: str,
    experiment_label: str,
) -> Any:
    if not repo.catalog.table_exists("sessions"):
        return pd.NA
    sessions = repo._load_table_frame(
        "sessions",
        filters={"project_id": str(project_id), "subject_uid": str(subject_uid)},
        use_sqlite=True,
    )
    if sessions.empty or "experiment_label" not in sessions.columns:
        return pd.NA
    match = sessions[
        sessions["experiment_label"].astype(str).str.strip() == str(experiment_label).strip()
    ]
    if match.empty:
        return pd.NA
    return match.iloc[0].get("session_uid", pd.NA)


def inspect_xnat_pipeline_resource(
    experiment: Any,
    resource_label: str,
) -> dict[str, Any]:
    """Return availability metadata for one experiment resource (no download)."""
    label = str(resource_label).strip().lower()
    has_files = xnat_resource_has_files(experiment, label)
    n_files = 0
    if has_files:
        resources = getattr(experiment, "resources", None) or {}
        if label in resources:
            n_files = _resource_file_count(resources[label])
    return {
        "resource_label": label,
        "available": bool(has_files),
        "n_files": int(n_files),
        "complete": bool(has_files and n_files > 0),
    }


def download_experiment_resource(
    experiment: Any,
    resource_label: str,
    dest_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download an experiment-level XNAT resource into *dest_dir*."""
    label = str(resource_label).strip()
    dest = Path(dest_dir).expanduser().resolve()
    if dest.exists():
        if not overwrite and any(dest.rglob("*")):
            return dest
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    resources = getattr(experiment, "resources", None) or {}
    if label not in resources:
        raise LookupError(f"Resource {label!r} not found on experiment")
    resource = resources[label]

    download_dir = getattr(resource, "download_dir", None)
    if callable(download_dir):
        download_dir(str(dest), extract=True)
        return dest

    with tempfile.TemporaryDirectory(prefix="nvitk-xnat-pipeline-") as tmp:
        zip_path = Path(tmp) / f"{label}.zip"
        download = getattr(resource, "download", None)
        if not callable(download):
            raise RuntimeError(
                f"XNAT resource {label!r} does not support download_dir or download"
            )
        result = download(zip_path, verbose=False)
        bundle = Path(result) if result else zip_path
        if bundle.suffix.lower() == ".zip" or zipfile.is_zipfile(bundle):
            with zipfile.ZipFile(bundle, "r") as zf:
                zf.extractall(dest)
        elif bundle.is_dir():
            shutil.copytree(bundle, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(bundle, dest / bundle.name)
    return dest


def _describe_downloaded_resource(resource_dir: Path, resource_label: str) -> dict[str, Any]:
    label = str(resource_label).strip().lower()
    if label == XNAT_RESOURCE_EICAB:
        return describe_local_eicab_resource(resource_dir)
    if label == XNAT_RESOURCE_QVTPY:
        return describe_local_qvtpy_resource(resource_dir, required_stages=_DEFAULT_QVTPY_QC_STAGES)
    files = iter_upload_files(resource_dir)
    return {
        "resource_label": label,
        "complete": bool(files),
        "n_files": len(files),
    }


def sync_xnat_pipeline_resources(
    repo: DataRepo,
    config: XnatConnectionConfig,
    *,
    catalog_path: str | Path | None = None,
    subjects: str | Iterable[str] | None = None,
    subjects_file: str | Path | None = None,
    id_type: str = "subject",
    resources: Iterable[str] = QVTPY_PIPELINE_RESOURCES,
    download_root: str | Path | None = None,
    download_resources: bool = False,
    overwrite_downloads: bool = False,
    skip_existing_downloads: bool = True,
    build_sqlite_index: bool = False,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Index qvtpy/eICAB experiment resources on XNAT into ``assets``."""
    project_id = str(config.project)
    subject_labels = resolve_subject_labels(
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
    )
    if not subject_labels and repo.catalog.table_exists("sessions"):
        subject_labels = sorted(
            {
                str(s)
                for s in repo._load_table_frame(
                    "sessions",
                    filters={"project_id": project_id},
                    use_sqlite=True,
                )["subject_uid"]
                .dropna()
                .unique()
            }
        )

    resource_labels = [str(r).strip().lower() for r in resources if str(r).strip()]
    batch = source_batch_id or f"xnat_pipeline_{utc_now_iso().replace(':', '').replace('-', '')}"
    rows: list[dict[str, Any]] = []

    with connect_xnat(config) as session:
        project = session.projects[project_id]
        for subject_uid in subject_labels:
            try:
                experiment, experiment_label = resolve_subject_experiment(project, subject_uid)
            except LookupError as exc:
                log.warning(f"[{subject_uid}] skip pipeline resources: {exc}")
                continue

            session_uid = _experiment_session_uid(
                repo,
                project_id=project_id,
                subject_uid=subject_uid,
                experiment_label=experiment_label,
            )

            for resource_label in resource_labels:
                info = inspect_xnat_pipeline_resource(experiment, resource_label)
                if not info.get("available"):
                    continue

                local_path = ""
                exists_locally = False
                extra = dict(info)
                extra["experiment_label"] = experiment_label

                if download_resources and download_root is not None:
                    dest = (
                        Path(download_root).expanduser().resolve()
                        / subject_uid
                        / resource_label
                    )
                    if skip_existing_downloads and dest.is_dir() and any(dest.rglob("*")):
                        local_path = str(dest)
                        exists_locally = True
                        extra.update(_describe_downloaded_resource(dest, resource_label))
                    else:
                        try:
                            download_experiment_resource(
                                experiment,
                                resource_label,
                                dest,
                                overwrite=overwrite_downloads,
                            )
                            local_path = str(dest)
                            exists_locally = True
                            extra.update(_describe_downloaded_resource(dest, resource_label))
                        except Exception as exc:
                            log.warning(
                                f"[{subject_uid}] download {resource_label!r} failed: {exc}"
                            )

                rows.append(
                    _bundle_asset_row(
                        subject_uid=subject_uid,
                        resource_label=resource_label,
                        source="xnat_pipeline",
                        asset_path=local_path or f"xnat://{project_id}/{subject_uid}/{resource_label}",
                        exists_locally=exists_locally,
                        session_uid=session_uid,
                        experiment_label=experiment_label,
                        source_batch_id=batch,
                        extra_meta=extra,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        log.info("No qvtpy/eICAB pipeline resources indexed from XNAT.")
        return df

    repo.upsert_table(
        "assets",
        df,
        provenance={"source": "xnat_pipeline", "project_id": project_id},
        build_sqlite_index=build_sqlite_index,
    )
    log.info(f"Indexed {len(df)} XNAT pipeline resource row(s) for project {project_id}")
    return df


def list_pipeline_assets_for_subject(
    repo: DataRepo,
    project_id: str,
    subject_uid: str,
) -> pd.DataFrame:
    """Return indexed pipeline bundle rows for *subject_uid*."""
    if not repo.catalog.table_exists("assets"):
        return pd.DataFrame()

    assets = repo._load_table_frame(
        "assets",
        filters={"subject_uid": str(subject_uid)},
        use_sqlite=True,
    )
    if assets.empty:
        return assets

    slots = {resource_label_to_asset_slot(r) for r in QVTPY_PIPELINE_RESOURCES}
    if "asset_slot" in assets.columns:
        assets = assets[assets["asset_slot"].astype(str).isin(slots)]
    elif "resource_label" in assets.columns:
        assets = assets[assets["resource_label"].astype(str).str.lower().isin(QVTPY_PIPELINE_RESOURCES)]
    else:
        return pd.DataFrame()

    if repo.catalog.table_exists("sessions"):
        sessions = repo._load_table_frame(
            "sessions",
            filters={"project_id": str(project_id), "subject_uid": str(subject_uid)},
            use_sqlite=True,
        )
        if not sessions.empty and "session_uid" in assets.columns:
            session_uids = {str(x) for x in sessions["session_uid"].dropna().unique()}
            mask = assets["session_uid"].isna() | assets["session_uid"].astype(str).isin(session_uids)
            assets = assets[mask]

    cols = [
        c
        for c in (
            "asset_uid",
            "subject_uid",
            "session_uid",
            "resource_label",
            "asset_slot",
            "asset_path",
            "exists_locally",
            "pipeline_id",
            "metadata_json",
        )
        if c in assets.columns
    ]
    return assets[cols].reset_index(drop=True)


def _cli_decorator(*args, **kwargs):
    def decorator(func):
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
    help="Dataset root to update.",
)
@_click_option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="YAML/JSON XNAT profile (NVITK_XNAT_CONFIG or ~/.config/nvitk/xnat.{yaml,yml,json}).",
)
@_click_option("--server", type=str, default=None, help="XNAT server URL.")
@_click_option("--project", type=str, default=None, help="XNAT project (e.g. PESA_Brain).")
@_click_option("--user", type=str, default=None, help="XNAT username.")
@_click_option("--password", type=str, default=None, help="XNAT password.")
@_click_option(
    "--netrc-file",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="Optional netrc file for authentication.",
)
@_click_option(
    "--catalog-path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    default=None,
    help="Optional catalog CSV to resolve subject IDs.",
)
@_click_option("--subjects", type=str, default=None, help="Comma/space separated subject labels.")
@_click_option(
    "--subjects-file",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    default=None,
    help="Text file with one subject per line.",
)
@_click_option(
    "--id-type",
    type=click.Choice(["subject", "mrid"], case_sensitive=False) if click is not None else None,
    default="subject",
    show_default=True,
)
@_click_option(
    "--resources",
    type=str,
    default="eicab,qvtpy",
    show_default=True,
    help="Comma-separated experiment resource labels to index.",
)
@_click_option(
    "--download-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="Download resources to {root}/{subject}/{resource}/ while syncing.",
)
@_click_option("--download", is_flag=True, help="Download experiment resources (requires --download-root).")
@_click_option(
    "--with-local",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="Also index local results tree at {root}/{subject}/eicab and .../qvtpy.",
)
@_click_option("--overwrite-downloads", is_flag=True, help="Replace existing download directories.")
@_click_option("--build-sqlite-index", is_flag=True, help="Rebuild SQLite query cache after sync.")
def main(
    dataset_root: Path,
    config_path: Path | None,
    server: str | None,
    project: str | None,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    catalog_path: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    id_type: str,
    resources: str,
    download_root: Path | None,
    download: bool,
    overwrite_downloads: bool,
    with_local: Path | None,
    build_sqlite_index: bool,
) -> None:
    """Index qvtpy/eICAB XNAT experiment resources into the dataset ``assets`` table."""
    if click is None:
        raise BackendUnavailableError(
            'click is not installed. Please install it with "pip install click".'
        )

    profile = load_xnat_profile(config_path)
    conn = resolve_xnat_connection(
        profile,
        server=server,
        project=project,
        user=user,
        password=password,
        netrc_file=str(netrc_file) if netrc_file else None,
    )
    repo = DataRepo(dataset_root, auto_scaffold=True)

    resource_labels = [r.strip().lower() for r in resources.split(",") if r.strip()]
    df = sync_xnat_pipeline_resources(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        resources=resource_labels,
        download_root=download_root,
        download_resources=download,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=False,
    )
    click.echo(f"Indexed {len(df)} XNAT pipeline resource row(s).")

    if with_local is not None:
        from .pipeline_assets import upsert_local_pipeline_assets

        local_df = upsert_local_pipeline_assets(
            repo,
            with_local,
            resources=resource_labels,
            build_sqlite_index=False,
        )
        click.echo(f"Indexed {len(local_df)} local pipeline asset row(s) from {with_local}.")

    if build_sqlite_index:
        repo.build_sqlite_index()
        click.echo("SQLite index rebuilt.")


if __name__ == "__main__":
    main()


__all__ = [
    "download_experiment_resource",
    "inspect_xnat_pipeline_resource",
    "list_pipeline_assets_for_subject",
    "sync_xnat_pipeline_resources",
]
