from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
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

try:
    import xnat
except Exception:
    xnat = None


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


@dataclass(frozen=True)
class XnatConnectionConfig:
    server: str
    project: str
    user: str | None = None
    password: str | None = None
    netrc_file: str | None = None
    verify: bool = True
    default_timeout: int = 300


def _normalize_header(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def _detect_catalog_column(fieldnames: list[str], candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = _normalize_header(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_subject_catalog_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def parse_subject_tokens(subjects: str | Iterable[str] | None) -> list[str]:
    if subjects is None:
        return []
    if isinstance(subjects, str):
        tokens = re.split(r"[\s,]+", subjects.strip())
        return [token for token in tokens if token]
    return [str(item).strip() for item in subjects if str(item).strip()]


def resolve_subject_labels(
    *,
    catalog_path: str | Path | None = None,
    subjects: str | Iterable[str] | None = None,
    subjects_file: str | Path | None = None,
    id_type: str = "subject",
) -> list[str]:
    if subjects_file is not None:
        with Path(subjects_file).open("r", encoding="utf-8") as handle:
            values = [line.strip() for line in handle if line.strip()]
    elif subjects is not None:
        values = parse_subject_tokens(subjects)
    elif catalog_path is not None:
        rows = load_subject_catalog_rows(catalog_path)
        if not rows:
            return []
        subject_column = _detect_catalog_column(list(rows[0].keys()), ["subject", "pesa", "date subject"])
        mrid_column = _detect_catalog_column(list(rows[0].keys()), ["mrid", "mriid", "mr_id", "mri_id"])
        selected_column = mrid_column if id_type == "mrid" and mrid_column else subject_column
        if selected_column is None:
            selected_column = list(rows[0].keys())[0]
        values = [str(row.get(selected_column, "")).strip().strip('"') for row in rows if str(row.get(selected_column, "")).strip()]
    else:
        return []

    if id_type != "mrid" or catalog_path is None:
        return sorted(set(values))

    rows = load_subject_catalog_rows(catalog_path)
    if not rows:
        return sorted(set(values))
    subject_column = _detect_catalog_column(list(rows[0].keys()), ["subject", "pesa", "date subject"])
    mrid_column = _detect_catalog_column(list(rows[0].keys()), ["mrid", "mriid", "mr_id", "mri_id"])
    if subject_column is None or mrid_column is None:
        return sorted(set(values))

    mapping = {
        str(row.get(mrid_column, "")).strip().strip('"').lower(): str(row.get(subject_column, "")).strip().strip('"')
        for row in rows
        if str(row.get(mrid_column, "")).strip() and str(row.get(subject_column, "")).strip()
    }
    resolved = [mapping[item.lower()] for item in values if item.lower() in mapping]
    return sorted(set(resolved))


def infer_flow_orientation(description: str) -> str:
    desc = description.upper()
    if any(token in desc for token in ("AP", "PA", "FA")):
        return "AP"
    if any(token in desc for token in ("RL", "LR", "RC", "CR")):
        return "RL"
    if any(token in desc for token in ("FH", "HF", "SI", "IS")):
        return "FH"
    return "GENERIC"


def classify_scan(series_description: str | None, quality: str | None = None) -> dict[str, Any] | None:
    description = series_description or ""
    if quality is not None and str(quality).lower() != "usable":
        return None

    if re.search(r"cs3di_mc|tof|mra", description, flags=re.IGNORECASE):
        return {
            "modality": "tof",
            "orientation": None,
            "sequence": "TOF",
        }

    if re.search(r"4d.?q?flow", description, flags=re.IGNORECASE):
        orientation = infer_flow_orientation(description)
        sequence = f"4DFLOW_{orientation}" if orientation != "GENERIC" else "4DFLOW_GENERIC"
        return {
            "modality": "4dflow",
            "orientation": orientation,
            "sequence": sequence,
        }

    return None


def requested_sequence_set(requested: str | Iterable[str] | None) -> set[str]:
    tokens = parse_subject_tokens(requested)
    if not tokens:
        return set()
    normalized: set[str] = set()
    for token in tokens:
        upper = token.upper().replace("-", "_")
        if upper in {"4DFLOW", "4DFLOW_ALL"}:
            normalized.update({"4DFLOW_AP", "4DFLOW_RL", "4DFLOW_FH"})
        elif upper == "TOF":
            normalized.add("TOF")
        else:
            normalized.add(upper)
    return normalized


def _coalesce_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None:
                return value
    return None


def connect_xnat(config: XnatConnectionConfig):
    if xnat is None:
        raise BackendUnavailableError('xnat is not installed. Please install it with "pip install xnat".')

    kwargs: dict[str, Any] = {
        "server": config.server,
        "verify": config.verify,
        "default_timeout": config.default_timeout,
    }
    if config.user:
        kwargs["user"] = config.user
    if config.password:
        kwargs["password"] = config.password
    if config.netrc_file:
        kwargs["netrc_file"] = config.netrc_file
    return xnat.connect(**kwargs)


def _download_scan_bundle(scan: Any, zip_path: Path) -> Path:
    if hasattr(scan, "download"):
        result = scan.download(zip_path, verbose=False)
        return Path(result) if result is not None else zip_path

    resources = getattr(scan, "resources", None)
    if resources and "DICOM" in resources:
        result = resources["DICOM"].download(zip_path, verbose=False)
        return Path(result) if result is not None else zip_path
    raise RuntimeError("Scan object does not expose a downloadable DICOM resource.")


def download_scan_dicoms(scan: Any, output_dir: str | Path, *, keep_zip: bool = False) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nvitk_xnat_") as tmp_dir:
        zip_path = Path(tmp_dir) / "scan.zip"
        bundle_path = _download_scan_bundle(scan, zip_path)
        extracted: list[Path] = []
        with zipfile.ZipFile(bundle_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = destination / Path(member.filename).name
                stem = target.stem
                suffix = "".join(target.suffixes)
                counter = 1
                while target.exists():
                    target = destination / f"{stem}_{counter}{suffix}"
                    counter += 1
                with archive.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                extracted.append(target)

        if keep_zip:
            kept_zip = destination / "scan_bundle.zip"
            shutil.copy2(bundle_path, kept_zip)
            extracted.append(kept_zip)

    return extracted


def sync_xnat_project(
    repo: DataRepo,
    config: XnatConnectionConfig,
    *,
    catalog_path: str | Path | None = None,
    subjects: str | Iterable[str] | None = None,
    subjects_file: str | Path | None = None,
    id_type: str = "subject",
    requested_sequences: str | Iterable[str] | None = None,
    download_root: str | Path | None = None,
    download_dicoms: bool = False,
    skip_existing: bool = True,
    build_sqlite_index: bool = False,
) -> dict[str, pd.DataFrame]:
    subject_labels = resolve_subject_labels(
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
    )
    allowed_sequences = requested_sequence_set(requested_sequences)
    source_batch_id = f"xnat_{utc_now_iso().replace(':', '').replace('-', '')}"

    subject_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    subject_id_rows: list[dict[str, Any]] = []

    with connect_xnat(config) as session:
        project = session.projects[config.project]
        available_subjects = subject_labels or list(project.subjects.keys())

        for subject_label in available_subjects:
            if subject_label not in project.subjects:
                continue
            subject = project.subjects[subject_label]
            subject_uid = str(_coalesce_attr(subject, "label", "id", "name") or subject_label)

            subject_rows.append(
                {
                    "subject_uid": subject_uid,
                    "primary_patient_id": subject_uid,
                    "primary_seqn": pd.NA,
                    "sex": pd.NA,
                    "birth_date": pd.NaT,
                    "notes": pd.NA,
                    "source_batch_id": source_batch_id,
                    "updated_at": utc_now_iso(),
                }
            )
            subject_id_rows.append(
                {
                    "subject_uid": subject_uid,
                    "id_namespace": "xnat_subject",
                    "id_value": subject_label,
                    "id_source": config.project,
                    "is_primary": True,
                    "source_batch_id": source_batch_id,
                    "updated_at": utc_now_iso(),
                }
            )

            experiments = list(getattr(subject, "experiments", {}).values())
            for experiment in experiments:
                experiment_label = str(_coalesce_attr(experiment, "label", "id") or "")
                session_uid = f"{config.project}:{subject_uid}:{experiment_label}"
                session_rows.append(
                    {
                        "session_uid": session_uid,
                        "subject_uid": subject_uid,
                        "project_id": config.project,
                        "experiment_label": experiment_label,
                        "modality": "mr",
                        "visit_label": pd.NA,
                        "acquired_at": pd.to_datetime(_coalesce_attr(experiment, "date"), errors="coerce"),
                        "source_batch_id": source_batch_id,
                        "updated_at": utc_now_iso(),
                    }
                )

                scans = list(getattr(experiment, "scans", {}).values())
                for scan in scans:
                    scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
                    series_description = str(_coalesce_attr(scan, "series_description", "type", "label") or "")
                    quality = str(_coalesce_attr(scan, "quality") or "")
                    classification = classify_scan(series_description, quality)
                    if classification is None:
                        continue
                    if allowed_sequences and classification["sequence"] not in allowed_sequences:
                        if not (
                            classification["sequence"] == "4DFLOW_GENERIC"
                            and allowed_sequences.intersection({"4DFLOW_AP", "4DFLOW_RL", "4DFLOW_FH"})
                        ):
                            continue

                    scan_uid = f"{session_uid}:{scan_id}"
                    local_cache_path = pd.NA

                    if download_dicoms and download_root is not None:
                        sequence_label = classification["sequence"]
                        target_dir = Path(download_root) / subject_uid / sequence_label
                        if skip_existing and target_dir.exists() and any(target_dir.iterdir()):
                            extracted_files = list(target_dir.iterdir())
                        else:
                            extracted_files = download_scan_dicoms(scan, target_dir)
                        local_cache_path = str(target_dir)
                        for file_path in extracted_files:
                            asset_rows.append(
                                {
                                    "asset_uid": f"{scan_uid}:{file_path.name}",
                                    "subject_uid": subject_uid,
                                    "session_uid": session_uid,
                                    "modality": classification["modality"],
                                    "asset_type": "dicom",
                                    "asset_path": str(file_path),
                                    "resource_label": "DICOM",
                                    "source": "xnat",
                                    "pipeline_name": pd.NA,
                                    "pipeline_version": pd.NA,
                                    "exists_locally": True,
                                    "metadata_json": "{}",
                                    "source_batch_id": source_batch_id,
                                    "updated_at": utc_now_iso(),
                                }
                            )

                    scan_rows.append(
                        {
                            "scan_uid": scan_uid,
                            "session_uid": session_uid,
                            "subject_uid": subject_uid,
                            "scan_id": scan_id,
                            "scan_label": str(_coalesce_attr(scan, "label", "name") or scan_id),
                            "series_description": series_description,
                            "quality": quality,
                            "modality": classification["modality"],
                            "orientation": classification["orientation"],
                            "resource_label": "DICOM",
                            "local_cache_path": local_cache_path,
                            "xnat_uri": str(_coalesce_attr(scan, "uri") or ""),
                            "source_batch_id": source_batch_id,
                            "updated_at": utc_now_iso(),
                        }
                    )

    frames = {
        "subjects": pd.DataFrame(subject_rows).drop_duplicates(subset=["subject_uid"], keep="last"),
        "subject_ids": pd.DataFrame(subject_id_rows).drop_duplicates(subset=["subject_uid", "id_namespace", "id_value"], keep="last"),
        "sessions": pd.DataFrame(session_rows).drop_duplicates(subset=["session_uid"], keep="last"),
        "scans": pd.DataFrame(scan_rows).drop_duplicates(subset=["scan_uid"], keep="last"),
        "assets": pd.DataFrame(asset_rows).drop_duplicates(subset=["asset_uid"], keep="last"),
    }

    for table_name, frame in frames.items():
        if frame.empty:
            continue
        repo.upsert_table(
            table_name,
            frame,
            provenance={"source": "xnat", "project": config.project, "server": config.server},
        )

    if build_sqlite_index:
        repo.build_sqlite_index()
    return frames


@_click_command()
@_click_option("--dataset-root", type=click.Path(path_type=Path) if click is not None else None, default=Path("dataset"), show_default=True, help="Dataset root to update.")
@_click_option("--server", type=str, required=True, help="XNAT server URL.")
@_click_option("--project", type=str, required=True, help="XNAT project identifier.")
@_click_option("--user", type=str, default=None, help="XNAT username.")
@_click_option("--password", type=str, default=None, help="XNAT password.")
@_click_option("--netrc-file", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Optional netrc file to use for authentication.")
@_click_option("--catalog-path", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None, help="Optional catalog CSV used to resolve subject IDs.")
@_click_option("--subjects", type=str, default=None, help="Comma or space separated subject identifiers.")
@_click_option("--subjects-file", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None, help="Text file with one subject identifier per line.")
@_click_option("--id-type", type=click.Choice(["subject", "mrid"], case_sensitive=False) if click is not None else None, default="subject", show_default=True, help="Interpret the provided IDs as XNAT subject labels or MR IDs.")
@_click_option("--sequences", type=str, default=None, help="Optional sequence subset such as TOF,4DFLOW_AP,4DFLOW_RL,4DFLOW_FH.")
@_click_option("--download-root", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Local directory where DICOM bundles are extracted.")
@_click_option("--download-dicoms", is_flag=True, help="Download and extract DICOMs while syncing metadata.")
@_click_option("--skip-existing", is_flag=True, help="Skip scan downloads when the local cache directory is already populated.")
@_click_option("--build-sqlite-index", is_flag=True, help="Rebuild the SQLite query cache after the sync.")
def main(
    dataset_root: Path,
    server: str,
    project: str,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    catalog_path: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    id_type: str,
    sequences: str | None,
    download_root: Path | None,
    download_dicoms: bool,
    skip_existing: bool,
    build_sqlite_index: bool,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    config = XnatConnectionConfig(
        server=server,
        project=project,
        user=user or os.getenv("XNAT_USER"),
        password=password or os.getenv("XNAT_PASSWORD"),
        netrc_file=str(netrc_file) if netrc_file else None,
    )
    repo = DataRepo(dataset_root, auto_scaffold=True)
    frames = sync_xnat_project(
        repo,
        config,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        requested_sequences=sequences,
        download_root=download_root,
        download_dicoms=download_dicoms,
        skip_existing=skip_existing,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Synced tables: {', '.join(name for name, frame in frames.items() if not frame.empty)}")


if __name__ == "__main__":
    main()
