"""Index XNAT experiment-level ``4dflows`` derivative NIfTIs into ``assets``."""

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
from nvitk.core.logger import Logger

from .repo import DataRepo
from .storage import utc_now_iso
from .xnat import _coalesce_attr, connect_xnat, resolve_subject_labels
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection
from .xnat_pipeline_resources import download_experiment_resource
from .xnat_scan_nifti_assets import classify_4dflows_derived_asset
from .xnat_upload import resolve_subject_experiment

log = Logger()

FOURDFLOWS_RESOURCE_LABEL = "4dflows"


def _cli_decorator(*args: Any, **kwargs: Any):
    """No-op stand-in for ``click.command``/``click.option`` when ``click`` isn't installed."""

    def decorator(func: Any) -> Any:
        """Return *func* unchanged."""
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


def _experiment_session_uid(
    repo: DataRepo,
    *,
    project_id: str,
    subject_uid: str,
    experiment_label: str,
) -> Any:
    """Look up the ``session_uid`` in the ``sessions`` table matching *project_id*/*subject_uid*/
    *experiment_label*; returns ``pd.NA`` if the table is missing or no row matches."""
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


def _resource_files_collection(resource: Any) -> list[Any]:
    """Normalize an XNAT resource's ``files`` (attribute, callable, or dict/iterable) into a plain list."""
    files = getattr(resource, "files", None)
    if files is None:
        return []
    if callable(files):
        try:
            files = files()
        except Exception:
            return []
    if isinstance(files, dict):
        return list(files.values())
    try:
        return list(files)
    except TypeError:
        return []


def _resource_file_name(file_obj: Any) -> str:
    """Basename of an XNAT resource file object, checking ``name``/``label``/``path``/``id`` in order."""
    for attr in ("name", "label", "path", "id"):
        value = _coalesce_attr(file_obj, attr)
        if value:
            return Path(str(value)).name
    return ""


def _resource_file_uri(file_obj: Any) -> str:
    """URI (or path) for an XNAT resource file object, checking ``uri``/``fulluri``/``path`` in order."""
    return str(_coalesce_attr(file_obj, "uri", "fulluri", "path") or "").strip()


def _is_nifti_or_json(name: str) -> bool:
    """True if *name* has a ``.nii``, ``.nii.gz``, or ``.json`` extension."""
    lower = name.lower()
    return lower.endswith(".json") or lower.endswith(".nii") or lower.endswith(".nii.gz")


def _list_experiment_resource_files(
    experiment: Any,
    *,
    resource_label: str = FOURDFLOWS_RESOURCE_LABEL,
) -> list[dict[str, str]]:
    """List NIfTI/JSON files under *experiment*'s *resource_label* resource (default ``4dflows``) as
    ``{name, uri, xnat_path}`` dicts; empty if the experiment has no such resource."""
    resources = getattr(experiment, "resources", None) or {}
    if resource_label not in resources:
        return []
    resource = resources[resource_label]
    out: list[dict[str, str]] = []
    for file_obj in _resource_files_collection(resource):
        name = _resource_file_name(file_obj)
        if not name or not _is_nifti_or_json(name):
            continue
        out.append(
            {
                "name": name,
                "uri": _resource_file_uri(file_obj),
                "xnat_path": str(_coalesce_attr(file_obj, "path") or ""),
            }
        )
    return out


def sync_xnat_4dflows_assets(
    repo: DataRepo,
    config: XnatConnectionConfig,
    *,
    catalog_path: str | Path | None = None,
    subjects: str | Iterable[str] | None = None,
    subjects_file: str | Path | None = None,
    id_type: str = "subject",
    resource_label: str = FOURDFLOWS_RESOURCE_LABEL,
    download_root: str | Path | None = None,
    download_files: bool = False,
    overwrite_downloads: bool = False,
    skip_existing_downloads: bool = True,
    build_sqlite_index: bool = False,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Index experiment-level ``4dflows`` derivative files into ``assets``."""
    if download_files and download_root is None:
        raise ValueError("download_files=True requires download_root to be set.")

    project_id = str(config.project)
    subject_labels = resolve_subject_labels(
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
    )
    if not subject_labels and repo.catalog.table_exists("sessions"):
        sessions = repo._load_table_frame(
            "sessions",
            filters={"project_id": project_id},
            use_sqlite=True,
        )
        if not sessions.empty and "subject_uid" in sessions.columns:
            subject_labels = sorted({str(s) for s in sessions["subject_uid"].dropna().unique()})

    batch = source_batch_id or f"xnat_4dflows_{utc_now_iso().replace(':', '').replace('-', '')}"
    rows: list[dict[str, Any]] = []
    resolved_download_root = Path(download_root).expanduser().resolve() if download_root is not None else None

    with connect_xnat(config) as session:
        project = session.projects[project_id]
        if not subject_labels:
            subject_labels = list(project.subjects.keys())

        for subject_uid in subject_labels:
            if subject_uid not in project.subjects:
                log.warning(f"[{subject_uid}] subject not found in XNAT project {project_id}")
                continue
            try:
                experiment, experiment_label = resolve_subject_experiment(
                    project, subject_uid
                )
            except LookupError as exc:
                log.warning(f"[{subject_uid}] skip 4dflows: {exc}")
                continue

            resource_files = _list_experiment_resource_files(
                experiment, resource_label=resource_label
            )
            if not resource_files:
                continue

            session_uid = _experiment_session_uid(
                repo,
                project_id=project_id,
                subject_uid=subject_uid,
                experiment_label=experiment_label,
            )

            download_dir: Path | None = None
            if download_files and resolved_download_root is not None:
                download_dir = (
                    resolved_download_root / subject_uid / resource_label
                )
                should_download = True
                if skip_existing_downloads and download_dir.is_dir() and any(download_dir.rglob("*")):
                    should_download = False
                if should_download:
                    try:
                        download_experiment_resource(
                            experiment,
                            resource_label,
                            download_dir,
                            overwrite=overwrite_downloads,
                        )
                    except Exception as exc:
                        log.warning(f"[{subject_uid}] download {resource_label!r} failed: {exc}")
                        download_dir = None

            for file_info in resource_files:
                asset_slot, asset_type = classify_4dflows_derived_asset(file_info["name"])
                if asset_slot is None or asset_type is None:
                    continue

                local_path = ""
                exists_locally = False
                if download_dir is not None and download_dir.is_dir():
                    candidate = download_dir / file_info["name"]
                    if not candidate.is_file():
                        matches = list(download_dir.rglob(file_info["name"]))
                        candidate = matches[0] if matches else candidate
                    if candidate.is_file():
                        local_path = str(candidate.resolve())
                        exists_locally = True

                path_or_uri = local_path or file_info["uri"] or (
                    f"xnat://{project_id}/{subject_uid}/{experiment_label}/{resource_label}/{file_info['name']}"
                )
                meta = {
                    "project_id": project_id,
                    "subject_uid": subject_uid,
                    "experiment_label": experiment_label,
                    "resource_label": resource_label,
                    "xnat_file_name": file_info["name"],
                    "xnat_file_uri": file_info["uri"],
                    "xnat_file_path": file_info["xnat_path"],
                    "flow4d_kind": "derived_global",
                }
                rows.append(
                    {
                        "asset_uid": f"xnat_4dflows:{subject_uid}:{session_uid}:{asset_slot}:{file_info['name']}",
                        "subject_uid": subject_uid,
                        "session_uid": session_uid,
                        "modality": "4dflow",
                        "asset_type": asset_type,
                        "asset_path": path_or_uri,
                        "resource_label": resource_label,
                        "source": "xnat_4dflows",
                        "pipeline_name": pd.NA,
                        "pipeline_id": pd.NA,
                        "exists_locally": exists_locally,
                        "asset_slot": asset_slot,
                        "metadata_json": json.dumps(meta),
                        "source_batch_id": batch,
                        "updated_at": utc_now_iso(),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        log.info("No XNAT 4dflows assets indexed.")
        return df

    repo.upsert_table(
        "assets",
        df,
        provenance={
            "source": "xnat_4dflows",
            "project_id": project_id,
            "resource_label": resource_label,
        },
        build_sqlite_index=build_sqlite_index,
    )
    log.info(f"Indexed {len(df)} XNAT 4dflows asset row(s) for project {project_id}")
    return df


@_click_command()
@_click_option("--dataset-root", type=click.Path(path_type=Path) if click is not None else None, default=Path("dataset"), show_default=True)
@_click_option("--config", "config_path", type=click.Path(path_type=Path) if click is not None else None, default=None)
@_click_option("--server", type=str, default=None)
@_click_option("--project", type=str, default=None)
@_click_option("--user", type=str, default=None)
@_click_option("--password", type=str, default=None)
@_click_option("--netrc-file", type=click.Path(path_type=Path) if click is not None else None, default=None)
@_click_option("--catalog-path", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None)
@_click_option("--subjects", type=str, default=None)
@_click_option("--subjects-file", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None)
@_click_option("--id-type", type=click.Choice(["subject", "mrid"], case_sensitive=False) if click is not None else None, default="subject", show_default=True)
@_click_option("--resource-label", type=str, default=FOURDFLOWS_RESOURCE_LABEL, show_default=True)
@_click_option("--download-root", type=click.Path(path_type=Path) if click is not None else None, default=None)
@_click_option("--download", is_flag=True)
@_click_option("--overwrite-downloads", is_flag=True)
@_click_option("--build-sqlite-index", is_flag=True)
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
    resource_label: str,
    download_root: Path | None,
    download: bool,
    overwrite_downloads: bool,
    build_sqlite_index: bool,
) -> None:
    """CLI entry point: resolve the XNAT connection profile from CLI flags/config file, then run
    :func:`sync_xnat_4dflows_assets` against the dataset at ``dataset_root``."""
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')
    if download and download_root is None:
        raise click.ClickException("--download requires --download-root.")

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
    df = sync_xnat_4dflows_assets(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        resource_label=resource_label,
        download_root=download_root,
        download_files=download,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Indexed {len(df)} XNAT 4dflows asset row(s).")


__all__ = [
    "FOURDFLOWS_RESOURCE_LABEL",
    "sync_xnat_4dflows_assets",
    "main",
]


if __name__ == "__main__":
    main()
