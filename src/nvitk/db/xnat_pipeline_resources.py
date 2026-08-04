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
    GUI_PIPELINE_RESOURCES,
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


def _resource_key_ci(resources: Any, resource_label: str) -> str | None:
    """Return the actual resources-dict key matching *resource_label* (case-insensitive)."""
    label = str(resource_label).strip()
    if resources is None:
        return None
    try:
        if label in resources:
            return label
    except TypeError:
        return None
    label_l = label.lower()
    try:
        keys = list(resources.keys())
    except Exception:
        return None
    for key in keys:
        if str(key).strip().lower() == label_l:
            return str(key)
    return None


def _resource_file_count(resource: Any) -> int:
    """Number of files exposed by an XNAT *resource* (attribute or callable ``files``), 0 if unavailable."""
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


def inspect_xnat_pipeline_resource(
    experiment: Any,
    resource_label: str,
) -> dict[str, Any]:
    """Return availability metadata for one experiment resource (no download)."""
    label = str(resource_label).strip().lower()
    resources = getattr(experiment, "resources", None) or {}
    key = _resource_key_ci(resources, label)
    has_files = False
    n_files = 0
    if key is not None:
        try:
            resource = resources[key]
        except (KeyError, TypeError, AttributeError):
            resource = None
        if resource is not None:
            n_files = _resource_file_count(resource)
            has_files = n_files > 0 or xnat_resource_has_files(experiment, key)
    return {
        "resource_label": label,
        "available": bool(has_files),
        "n_files": int(n_files),
        "complete": bool(has_files and n_files > 0),
        "xnat_resource_key": key or label,
    }


def _find_xnat_files_root(dest: Path, resource_label: str) -> Path | None:
    """Locate the ``…/resources/<label>/files`` directory from an xnatpy ZIP unpack."""
    label = str(resource_label).strip().lower()
    if not dest.is_dir():
        return None
    candidates: list[Path] = []
    for files_dir in dest.rglob("files"):
        if not files_dir.is_dir():
            continue
        parent = files_dir.parent
        # Expect …/resources/<resource_label>/files
        if parent.parent.name.lower() != "resources":
            continue
        candidates.append(files_dir)
    if not candidates:
        return None
    labeled = [p for p in candidates if p.parent.name.lower() == label]
    pool = labeled or candidates
    pool.sort(key=lambda p: (len(p.parts), str(p)))
    return pool[0]


def unwrap_xnat_resource_download(dest_dir: Path, resource_label: str) -> Path:
    """Promote ``{experiment}/resources/{label}/files/*`` up to *dest_dir*.

    xnatpy ``download_dir`` extracts the REST ZIP with experiment/resource nesting.
    Local QC / pipeline loaders expect the uploaded payload (stage folders, NIfTIs)
    directly under the destination, matching the on-disk results layout.
    """
    dest = Path(dest_dir).expanduser().resolve()
    if not dest.is_dir():
        return dest
    files_root = _find_xnat_files_root(dest, resource_label)
    if files_root is None:
        return dest
    try:
        if files_root.resolve() == dest.resolve():
            return dest
    except Exception:
        pass
    # Already unwrapped: stage folders or NIfTIs sit directly under dest.
    if any(
        (dest / name).exists()
        for name in (
            "stage6_measure",
            "stage4_4dflow_segmentation",
            "stage1_eicab",
            "ComplexDifference_3D.nii.gz",
            "ComplexDifference_3D.nii",
        )
    ):
        return dest

    staging = dest.parent / f".{dest.name}_unwrap_tmp"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(files_root, staging)
    # Replace dest contents with the promoted payload.
    for child in list(dest.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except Exception:
                pass
    for child in staging.iterdir():
        target = dest / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        shutil.move(str(child), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    return dest


def download_experiment_resource(
    experiment: Any,
    resource_label: str,
    dest_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download an experiment-level XNAT resource into *dest_dir*.

    After download, unwraps xnatpy's ``{experiment}/resources/{label}/files`` nesting
    so *dest_dir* matches the local pipeline results layout.
    """
    label = str(resource_label).strip()
    dest = Path(dest_dir).expanduser().resolve()
    if dest.exists():
        if not overwrite and any(dest.rglob("*")):
            return unwrap_xnat_resource_download(dest, label)
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    resources = getattr(experiment, "resources", None) or {}
    key = _resource_key_ci(resources, label)
    if key is None:
        raise LookupError(f"Resource {label!r} not found on experiment")
    resource = resources[key]

    download_dir = getattr(resource, "download_dir", None)
    if callable(download_dir):
        # xnat>=0.6 download_dir already unpacks the ZIP; it does not accept extract=.
        download_dir(str(dest), verbose=False)
        return unwrap_xnat_resource_download(dest, label)

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
    return unwrap_xnat_resource_download(dest, label)

def _describe_downloaded_resource(resource_dir: Path, resource_label: str) -> dict[str, Any]:
    """Summarize a downloaded XNAT experiment resource: dispatch to the ``eicab``/``qvtpy``-specific
    describer, or fall back to a generic file-count summary for other resource labels."""
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
    subject_source = "cli/catalog"
    if not subject_labels and repo.catalog.table_exists("sessions"):
        sessions = repo._load_table_frame(
            "sessions",
            filters={"project_id": project_id},
            use_sqlite=True,
        )
        if not sessions.empty and "subject_uid" in sessions.columns:
            subject_labels = sorted(
                {str(s) for s in sessions["subject_uid"].dropna().unique()}
            )
            subject_source = f"sessions:{project_id}"

    resource_labels = [str(r).strip().lower() for r in resources if str(r).strip()]
    batch = source_batch_id or f"xnat_pipeline_{utc_now_iso().replace(':', '').replace('-', '')}"
    rows: list[dict[str, Any]] = []
    n_checked = 0
    n_missing_experiment = 0
    n_no_resource = 0

    with connect_xnat(config) as session:
        project = session.projects[project_id]
        if not subject_labels:
            subject_labels = sorted(str(s) for s in project.subjects.keys())
            subject_source = f"xnat:{project_id}"
        log.info(
            "Indexing pipeline resources %s for %d subject(s) from %s (project=%s)",
            ",".join(resource_labels),
            len(subject_labels),
            subject_source,
            project_id,
        )

        for subject_uid in subject_labels:
            n_checked += 1
            try:
                experiment, experiment_label = resolve_subject_experiment(project, subject_uid)
            except LookupError as exc:
                n_missing_experiment += 1
                log.warning(f"[{subject_uid}] skip pipeline resources: {exc}")
                continue

            session_uid = _experiment_session_uid(
                repo,
                project_id=project_id,
                subject_uid=subject_uid,
                experiment_label=experiment_label,
            )

            found_any = False
            for resource_label in resource_labels:
                info = inspect_xnat_pipeline_resource(experiment, resource_label)
                if not info.get("available"):
                    continue
                found_any = True

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
            if not found_any:
                n_no_resource += 1

    df = pd.DataFrame(rows)
    if df.empty:
        log.info(
            "No qvtpy/eICAB pipeline resources indexed from XNAT "
            "(checked=%d, missing_experiment=%d, no_eicab/qvtpy=%d).",
            n_checked,
            n_missing_experiment,
            n_no_resource,
        )
        return df

    repo.upsert_table(
        "assets",
        df,
        provenance={"source": "xnat_pipeline", "project_id": project_id},
        build_sqlite_index=build_sqlite_index,
    )
    log.info(
        "Indexed %d XNAT pipeline resource row(s) for project %s "
        "(checked=%d, missing_experiment=%d, no_eicab/qvtpy=%d)",
        len(df),
        project_id,
        n_checked,
        n_missing_experiment,
        n_no_resource,
    )
    return df


def list_pipeline_assets_for_subject(
    repo: DataRepo,
    project_id: str,
    subject_uid: str,
) -> pd.DataFrame:
    """Return indexed pipeline bundle rows for *subject_uid*.

    Includes ``eicab`` / ``qvtpy`` / ``4dflows`` experiment resources. Derived
    ``4dflows`` file rows (``source=xnat_4dflows``) are collapsed to a single
    downloadable ``pipeline_4dflows`` bundle entry when no bundle row exists.
    """
    if not repo.catalog.table_exists("assets"):
        return pd.DataFrame()

    assets = repo._load_table_frame(
        "assets",
        filters={"subject_uid": str(subject_uid)},
        use_sqlite=True,
    )
    if assets.empty:
        return assets

    slots = {resource_label_to_asset_slot(r) for r in GUI_PIPELINE_RESOURCES}
    bundle = pd.DataFrame()
    if "asset_slot" in assets.columns:
        bundle = assets[assets["asset_slot"].astype(str).isin(slots)].copy()
    elif "resource_label" in assets.columns:
        bundle = assets[
            assets["resource_label"].astype(str).str.lower().isin(GUI_PIPELINE_RESOURCES)
        ].copy()

    # Collapse per-file 4dflows assets into one downloadable pipeline row.
    has_4dflows_bundle = False
    if not bundle.empty and "asset_slot" in bundle.columns:
        has_4dflows_bundle = (
            bundle["asset_slot"].astype(str) == "pipeline_4dflows"
        ).any()
    if not has_4dflows_bundle and not bundle.empty and "resource_label" in bundle.columns:
        has_4dflows_bundle = (
            bundle["resource_label"].astype(str).str.lower() == "4dflows"
        ).any()

    if not has_4dflows_bundle:
        fourd_mask = pd.Series(False, index=assets.index)
        if "source" in assets.columns:
            fourd_mask = fourd_mask | (
                assets["source"].astype(str).str.lower() == "xnat_4dflows"
            )
        if "resource_label" in assets.columns:
            fourd_mask = fourd_mask | (
                assets["resource_label"].astype(str).str.lower() == "4dflows"
            )
        fourd = assets.loc[fourd_mask]
        if not fourd.empty:
            row0 = fourd.iloc[0].to_dict()
            row0["asset_slot"] = "pipeline_4dflows"
            row0["resource_label"] = "4dflows"
            row0["asset_uid"] = f"pipeline_bundle:{subject_uid}:4dflows"
            # Prefer a local directory that already holds the resource tree.
            local_dir = ""
            for path in fourd.get("asset_path", pd.Series(dtype=str)).dropna():
                p = Path(str(path))
                if p.is_dir() and any(p.rglob("*")):
                    local_dir = str(p)
                    break
                if p.is_file():
                    # Walk up to a plausible resource folder name.
                    for parent in p.parents:
                        if parent.name.lower() in {"4dflows", "qvtpy", "eicab"}:
                            local_dir = str(parent)
                            break
                    if local_dir:
                        break
            row0["asset_path"] = local_dir or pd.NA
            row0["exists_locally"] = bool(local_dir)
            bundle = pd.concat([bundle, pd.DataFrame([row0])], ignore_index=True)

    if bundle.empty:
        return pd.DataFrame()

    if repo.catalog.table_exists("sessions"):
        sessions = repo._load_table_frame(
            "sessions",
            filters={"project_id": str(project_id), "subject_uid": str(subject_uid)},
            use_sqlite=True,
        )
        if not sessions.empty and "session_uid" in bundle.columns:
            session_uids = {str(x) for x in sessions["session_uid"].dropna().unique()}
            mask = bundle["session_uid"].isna() | bundle["session_uid"].astype(str).isin(
                session_uids
            )
            bundle = bundle[mask]

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
        if c in bundle.columns
    ]
    return bundle[cols].reset_index(drop=True)


def _cli_decorator(*args, **kwargs):
    """No-op stand-in for ``click.command``/``click.option`` when ``click`` isn't installed."""

    def decorator(func):
        """Return *func* unchanged."""
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


@_click_command()
@_click_option(
    "--dataset-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=Path("dataset/nvitk-dataset"),
    show_default=True,
    help="Dataset root to update (use the nvitk-dataset tree, not the empty parent dataset/).",
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
    default="eicab,qvtpy,4dflows",
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
    "unwrap_xnat_resource_download",
]
