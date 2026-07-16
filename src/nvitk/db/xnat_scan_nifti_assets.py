"""Index XNAT scan-level NIfTI resources into the dataset ``assets`` table.

This complements :mod:`nvitk.db.xnat` scan metadata sync and
:mod:`nvitk.db.xnat_pipeline_resources` experiment-level pipeline bundles by
tracking the individual files exposed under each scan ``NIFTI`` resource.

PESA-Brain conventions:

* ``TOF`` scan resource holds the TOF NIfTI (and optional JSON sidecar).
* ``4DFLOW_AP`` / ``4DFLOW_RL`` / ``4DFLOW_FH`` scan resources hold the phase
  and magnitude NIfTIs (and optional JSON sidecars) for that direction.
* Shared 4D-flow derivatives live in a separate experiment-level ``4dflows``
  resource (see :mod:`nvitk.db.xnat_4dflows_assets`).
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
from nvitk.core.logger import Logger

from .repo import DataRepo
from .storage import utc_now_iso
from .xnat import (
    _coalesce_attr,
    connect_xnat,
    download_scan_niftis,
    requested_sequence_set,
    resolve_subject_labels,
)
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection
from .xnat_projects import (
    classify_experiment_ia_pet_v5,
    classify_scan_for_project,
    default_sequences_for_project,
    get_xnat_project,
)

log = Logger()

NIFTI_RESOURCE_LABEL = "NIFTI"
_JSON_SUFFIX = "_json"

DERIVED_SLOT_BY_STEM: dict[str, str] = {
    "angiography3d": "flow_angiography_3d",
    "angiography4d": "flow_angiography_4d",
    "complexdifference3d": "flow_complex_difference_3d",
    "complexdifference4d": "flow_complex_difference_4d",
    "velocitymagnitude3d": "flow_velocity_magnitude_3d",
    "velocitymagnitude4d": "flow_velocity_magnitude_4d",
    "velocitymeancomponents": "flow_velocity_mean_components",
}


def _cli_decorator(*args: Any, **kwargs: Any):
    def decorator(func: Any) -> Any:
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


def _is_nifti_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".nii.gz") or lower.endswith(".nii")


def _is_json_filename(name: str) -> bool:
    return name.lower().endswith(".json")


def _strip_known_suffixes(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    if lower.endswith(".json"):
        return name[:-5]
    return Path(name).stem


def _normalize_token(text: str) -> str:
    import re

    return re.sub(r"[^0-9a-z]+", "", str(text).strip().lower())


def _sequence_direction(sequence_label: str) -> str | None:
    upper = str(sequence_label).strip().upper()
    if upper == "4DFLOW_AP":
        return "ap"
    if upper == "4DFLOW_RL":
        return "rl"
    if upper == "4DFLOW_FH":
        return "fh"
    return None


def classify_scan_nifti_asset(sequence_label: str, filename: str) -> tuple[str | None, str | None]:
    """Return ``(asset_slot, asset_type)`` for one XNAT scan-resource file."""
    name = Path(filename).name
    stem_norm = _normalize_token(_strip_known_suffixes(name))
    is_nifti = _is_nifti_filename(name)
    is_json = _is_json_filename(name)
    if not is_nifti and not is_json:
        return None, None

    slot: str | None = None
    seq_upper = str(sequence_label).strip().upper()

    if seq_upper == "TOF":
        slot = "tof"
    else:
        direction = _sequence_direction(seq_upper)
        if direction is not None:
            if stem_norm.endswith("ph"):
                slot = f"flow_{direction}_ph"
            elif stem_norm.endswith("m"):
                slot = f"flow_{direction}_m"

    if slot is None:
        return None, None
    if is_json:
        return f"{slot}{_JSON_SUFFIX}", "json"
    return slot, "nifti"


def classify_4dflows_derived_asset(filename: str) -> tuple[str | None, str | None]:
    """Return ``(asset_slot, asset_type)`` for one experiment ``4dflows`` file."""
    name = Path(filename).name
    stem_norm = _normalize_token(_strip_known_suffixes(name))
    is_nifti = _is_nifti_filename(name)
    is_json = _is_json_filename(name)
    if not is_nifti and not is_json:
        return None, None
    slot = DERIVED_SLOT_BY_STEM.get(stem_norm)
    if slot is None:
        return None, None
    if is_json:
        return f"{slot}{_JSON_SUFFIX}", "json"
    return slot, "nifti"


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


def _resource_files_collection(resource: Any) -> list[Any]:
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
    for attr in ("name", "label", "path", "id"):
        value = _coalesce_attr(file_obj, attr)
        if value:
            return Path(str(value)).name
    return ""


def _resource_file_uri(file_obj: Any) -> str:
    return str(_coalesce_attr(file_obj, "uri", "fulluri", "path") or "").strip()


def _list_scan_nifti_resource_files(scan: Any, *, resource_label: str = NIFTI_RESOURCE_LABEL) -> list[dict[str, str]]:
    resources = getattr(scan, "resources", None) or {}
    if resource_label not in resources:
        return []
    resource = resources[resource_label]
    out: list[dict[str, str]] = []
    for file_obj in _resource_files_collection(resource):
        name = _resource_file_name(file_obj)
        if not name:
            continue
        if not (_is_nifti_filename(name) or _is_json_filename(name)):
            continue
        out.append(
            {
                "name": name,
                "uri": _resource_file_uri(file_obj),
                "xnat_path": str(_coalesce_attr(file_obj, "path") or ""),
            }
        )
    return out


def _classified_scan_items(project_id: str, experiment: Any, experiment_label: str) -> list[tuple[Any, str, str, str, dict[str, Any]]]:
    scans = list(getattr(experiment, "scans", {}).values())
    try:
        project_spec = get_xnat_project(project_id)
    except KeyError:
        project_spec = None

    if project_spec is not None and project_spec.classifier == "ia_pet_v5":
        return classify_experiment_ia_pet_v5(scans, experiment_label=experiment_label)

    items: list[tuple[Any, str, str, str, dict[str, Any]]] = []
    for scan in scans:
        scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
        series_description = str(_coalesce_attr(scan, "series_description", "type", "label") or "")
        quality = str(_coalesce_attr(scan, "quality") or "")
        classification = classify_scan_for_project(
            project_id,
            series_description,
            quality,
            scan_id=scan_id,
            experiment_label=experiment_label,
        )
        if classification is None:
            continue
        items.append((scan, scan_id, series_description, quality, classification))
    return items


def _local_download_dir(download_root: Path, subject_uid: str, sequence_label: str, scan_id: str, resource_label: str) -> Path:
    return download_root / subject_uid / sequence_label / scan_id / resource_label


def _match_downloaded_local_file(download_dir: Path, basename: str) -> Path | None:
    candidate = download_dir / Path(basename).name
    if candidate.is_file():
        return candidate
    return None


def sync_xnat_scan_nifti_assets(
    repo: DataRepo,
    config: XnatConnectionConfig,
    *,
    catalog_path: str | Path | None = None,
    subjects: str | Iterable[str] | None = None,
    subjects_file: str | Path | None = None,
    id_type: str = "subject",
    requested_sequences: str | Iterable[str] | None = None,
    resource_label: str = NIFTI_RESOURCE_LABEL,
    download_root: str | Path | None = None,
    download_files: bool = False,
    overwrite_downloads: bool = False,
    skip_existing_downloads: bool = True,
    build_sqlite_index: bool = False,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Index scan-level XNAT NIfTI resource files into ``assets``."""
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

    allowed_sequences = requested_sequence_set(requested_sequences)
    if not allowed_sequences:
        allowed_sequences = requested_sequence_set(",".join(default_sequences_for_project(project_id)))
    batch = source_batch_id or f"xnat_scan_nifti_{utc_now_iso().replace(':', '').replace('-', '')}"
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
            subject = project.subjects[subject_uid]
            for experiment in getattr(subject, "experiments", {}).values():
                experiment_label = str(_coalesce_attr(experiment, "label", "id") or "")
                if not experiment_label:
                    continue
                session_uid = _experiment_session_uid(
                    repo,
                    project_id=project_id,
                    subject_uid=subject_uid,
                    experiment_label=experiment_label,
                )
                for scan, scan_id, series_description, quality, classification in _classified_scan_items(
                    project_id, experiment, experiment_label
                ):
                    sequence_label = str(classification.get("sequence") or "").strip()
                    if not sequence_label or sequence_label not in allowed_sequences:
                        continue

                    scan_files = _list_scan_nifti_resource_files(scan, resource_label=resource_label)
                    if not scan_files:
                        continue

                    download_dir: Path | None = None
                    if download_files and resolved_download_root is not None:
                        download_dir = _local_download_dir(
                            resolved_download_root,
                            subject_uid,
                            sequence_label,
                            scan_id,
                            resource_label,
                        )
                        should_download = True
                        if skip_existing_downloads and download_dir.is_dir() and any(download_dir.iterdir()):
                            should_download = False
                        if should_download:
                            if overwrite_downloads and download_dir.exists():
                                import shutil

                                shutil.rmtree(download_dir, ignore_errors=True)
                            download_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                download_scan_niftis(scan, download_dir, resource_label=resource_label)
                            except Exception as exc:
                                log.warning(
                                    f"[{subject_uid}] download {sequence_label} scan {scan_id} NIFTI failed: {exc}"
                                )

                    for file_info in scan_files:
                        asset_slot, asset_type = classify_scan_nifti_asset(sequence_label, file_info["name"])
                        if asset_slot is None or asset_type is None:
                            continue
                        local_path = ""
                        exists_locally = False
                        if download_dir is not None:
                            local_file = _match_downloaded_local_file(download_dir, file_info["name"])
                            if local_file is not None:
                                local_path = str(local_file.resolve())
                                exists_locally = True
                        path_or_uri = local_path or file_info["uri"] or (
                            f"xnat://{project_id}/{subject_uid}/{experiment_label}/{scan_id}/{resource_label}/{file_info['name']}"
                        )
                        meta = {
                            "project_id": project_id,
                            "subject_uid": subject_uid,
                            "experiment_label": experiment_label,
                            "sequence": sequence_label,
                            "scan_id": scan_id,
                            "scan_label": str(_coalesce_attr(scan, "label", "name") or scan_id),
                            "series_description": series_description,
                            "quality": quality,
                            "resource_label": resource_label,
                            "xnat_file_name": file_info["name"],
                            "xnat_file_uri": file_info["uri"],
                            "xnat_file_path": file_info["xnat_path"],
                        }
                        rows.append(
                            {
                                "asset_uid": f"xnat_scan_nifti:{subject_uid}:{session_uid}:{scan_id}:{asset_slot}:{file_info['name']}",
                                "subject_uid": subject_uid,
                                "session_uid": session_uid,
                                "modality": classification.get("modality") or "mr",
                                "asset_type": asset_type,
                                "asset_path": path_or_uri,
                                "resource_label": resource_label,
                                "source": "xnat_scan_nifti",
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
        log.info("No XNAT scan NIfTI assets indexed.")
        return df

    repo.upsert_table(
        "assets",
        df,
        provenance={"source": "xnat_scan_nifti", "project_id": project_id, "resource_label": resource_label},
        build_sqlite_index=build_sqlite_index,
    )
    log.info(f"Indexed {len(df)} XNAT scan NIfTI asset row(s) for project {project_id}")
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
@_click_option("--sequences", type=str, default="TOF,4DFLOW_AP,4DFLOW_RL,4DFLOW_FH", show_default=True)
@_click_option("--resource-label", type=str, default=NIFTI_RESOURCE_LABEL, show_default=True)
@_click_option("--download-root", type=click.Path(path_type=Path) if click is not None else None, default=None)
@_click_option("--download", is_flag=True, help="Download scan NIfTI resources while indexing.")
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
    sequences: str,
    resource_label: str,
    download_root: Path | None,
    download: bool,
    overwrite_downloads: bool,
    build_sqlite_index: bool,
) -> None:
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
    df = sync_xnat_scan_nifti_assets(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        requested_sequences=sequences,
        resource_label=resource_label,
        download_root=download_root,
        download_files=download,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Indexed {len(df)} XNAT scan NIfTI asset row(s).")


__all__ = [
    "DERIVED_SLOT_BY_STEM",
    "NIFTI_RESOURCE_LABEL",
    "classify_4dflows_derived_asset",
    "classify_scan_nifti_asset",
    "sync_xnat_scan_nifti_assets",
    "main",
]


if __name__ == "__main__":
    main()
