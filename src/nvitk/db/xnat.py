from __future__ import annotations

import csv
import json
import netrc
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError

from .importers import upsert_cohort_membership_for_subjects
from .repo import DEFAULT_COHORT_ID, DataRepo
from .storage import utc_now_iso
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection

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


def parse_requested_sequences(requested: str | Iterable[str] | None) -> list[str]:
    """Split ``requested_sequences`` on commas; spaces inside a token are kept until normalized."""
    if requested is None:
        return []
    if isinstance(requested, str):
        return [part.strip() for part in requested.split(",") if part.strip()]
    out: list[str] = []
    for item in requested:
        s = str(item).strip()
        if not s:
            continue
        out.extend([p.strip() for p in s.split(",") if p.strip()])
    return out


def _normalize_sequence_request_token(token: str) -> str:
    """Canonical sequence key for filtering (aligned with :func:`classify_scan` ``sequence``)."""
    t = str(token).strip()
    t = re.sub(r"\s+", "_", t)
    t = t.replace("-", "_").replace("/", "_")
    t = re.sub(r"_+", "_", t).strip("_")
    return t.upper()


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

    # --- Additional structural / functional sequences (order: specific before broad) ---

    if re.search(
        r"resting\s*state|resting-state|\brsfmri\b|\brsf?mri\b|"
        r"rest\s*fmri|fmri\s*rest|bold.*rest|epi.*rest|task\s*rest|mb\s*epi|multiband.*rest",
        description,
        flags=re.IGNORECASE,
    ):
        return {
            "modality": "fmri",
            "orientation": None,
            "sequence": "RESTING_STATE_MB",
        }

    if re.search(r"carotid.*qf|qf.*carotid|carotid_qf|carotid\s*4d", description, flags=re.IGNORECASE):
        return {
            "modality": "mra",
            "orientation": None,
            "sequence": "CAROTID_QF",
        }

    if re.search(r"\bqsm\b|quantitative\s*suscept|suscept.*map", description, flags=re.IGNORECASE) and not re.search(
        r"\bswi\b|susceptibility\s*weighted", description, flags=re.IGNORECASE
    ):
        return {
            "modality": "qsm",
            "orientation": None,
            "sequence": "QSM",
        }

    if re.search(r"\bswi\b|swi\s*/\s*qsm|susceptibility\s*weighted", description, flags=re.IGNORECASE):
        return {
            "modality": "swi",
            "orientation": None,
            "sequence": "SWI_QSM",
        }

    if re.search(r"3d.*flair|flair.*3d|space.*flair|\bflair\b.*\b3d\b", description, flags=re.IGNORECASE):
        return {
            "modality": "flair",
            "orientation": None,
            "sequence": "3D_FLAIR",
        }

    if re.search(
        r"3d.*t2.*(?:hr|high\s*res)|t2.*(?:hr|high\s*res).*(?:3d|space)|\bt2\s*space\b|space\s*t2",
        description,
        flags=re.IGNORECASE,
    ):
        return {
            "modality": "t2",
            "orientation": None,
            "sequence": "3D_T2_HR",
        }

    if re.search(
        r"\b3d\s*t1\b|\bt1\b.*\b3d\b|\bmprage\b|mp[_\s-]?rage|t1_mpr|t1\s*mprage|t1w?\s*3d",
        description,
        flags=re.IGNORECASE,
    ):
        return {
            "modality": "t1",
            "orientation": None,
            "sequence": "3D_T1",
        }

    return None


def resolve_xnat_scan_from_scan_row(session: Any, row: Mapping[str, Any]) -> Any:
    """Return the XNAT *scan* handle for a ``scans`` table row (uses ``session_uid`` and ``scan_id``)."""
    session_uid = row.get("session_uid")
    scan_id = str(row.get("scan_id") or "").strip()
    if session_uid is None or (isinstance(session_uid, float) and pd.isna(session_uid)):
        raise ValueError("scan row has no session_uid")
    if not scan_id:
        raise ValueError("scan row has no scan_id")
    parts = str(session_uid).split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"session_uid must be project:subject:experiment, got {session_uid!r}")
    project_id, subject_key, experiment_label = parts
    project = session.projects[project_id]
    subject = None
    if subject_key in project.subjects:
        subject = project.subjects[subject_key]
    else:
        for _lbl, subj in project.subjects.items():
            uid = str(_coalesce_attr(subj, "label", "id", "name") or _lbl)
            if uid == subject_key:
                subject = subj
                break
    if subject is None:
        raise LookupError(f"No subject matching {subject_key!r} in project {project_id!r}")

    experiments_map = getattr(subject, "experiments", None) or {}
    experiment = None
    if experiment_label in experiments_map:
        experiment = experiments_map[experiment_label]
    else:
        for exp in experiments_map.values():
            if str(_coalesce_attr(exp, "label", "id") or "") == experiment_label:
                experiment = exp
                break
    if experiment is None:
        raise LookupError(f"Experiment {experiment_label!r} not found for subject {subject_key!r}")

    scans_map = getattr(experiment, "scans", None) or {}
    if scan_id in scans_map:
        return scans_map[scan_id]
    for sc in scans_map.values():
        sid = str(_coalesce_attr(sc, "id", "label", "name") or "")
        if sid == scan_id:
            return sc
    raise LookupError(f"Scan {scan_id!r} not found under session {session_uid!r}")


def xnat_sequence_to_asset_slot(sequence_label: str) -> str:
    """Map XNAT sequence labels (see :func:`classify_scan`) to canonical ``asset_slot`` names."""
    key = str(sequence_label).strip().upper().replace("-", "_")
    mapping = {
        "TOF": "tof",
        "4DFLOW_AP": "4dflow_ap",
        "4DFLOW_FH": "4dflow_fh",
        "4DFLOW_RL": "4dflow_rl",
        "4DFLOW_GENERIC": "4dflow",
        "3D_T1": "t1_3d",
        "3D_T2_HR": "t2_hr_3d",
        "3D_FLAIR": "flair_3d",
        "SWI_QSM": "swi",
        "QSM": "qsm",
        "CAROTID_QF": "carotid_qf",
        "RESTING_STATE_MB": "resting_state_mb",
    }
    if key in mapping:
        return mapping[key]
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_") or "unknown"


def requested_sequence_set(requested: str | Iterable[str] | None) -> set[str]:
    """Normalize ``requested_sequences`` tokens to the same keys :func:`classify_scan` uses for ``sequence``."""
    tokens = parse_requested_sequences(requested)
    if not tokens:
        return set()
    normalized: set[str] = set()
    for token in tokens:
        upper = _normalize_sequence_request_token(token)
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


def _netrc_host_candidates(server: str) -> list[str]:
    """Bare hostnames to try (host, then parent domain ``a.b.c`` → ``b.c``)."""
    netloc = urlparse(server if "://" in server else f"https://{server}").netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    host = netloc
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            inner = host[1:end]
            rest = host[end + 1 :]
            host = inner if rest.startswith(":") else inner
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]

    ordered: list[str] = []
    if host:
        ordered.append(host)
    parts = host.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[1:])
        if parent not in ordered:
            ordered.append(parent)
    return list(dict.fromkeys(ordered))


def _netrc_machine_candidates(server: str) -> list[str]:
    """Ordered *machine* strings for ~/.netrc lookup."""
    s = server.strip().rstrip("/")
    out: list[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    add(s)

    parsed = urlparse(s if "://" in s else f"https://{s}")
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    if netloc.startswith("["):
        end = netloc.find("]")
        if end != -1:
            inner = netloc[1:end]
            rest = netloc[end + 1 :]
            host = inner if rest.startswith(":") else inner
        else:
            host = netloc
    elif netloc.count(":") == 1:
        host = netloc.split(":", 1)[0]
    else:
        host = netloc

    if parsed.scheme and host:
        add(f"{parsed.scheme}://{host}".rstrip("/"))

    if "://" not in s.strip() and host:
        add(f"https://{host}")
        add(f"http://{host}")

    for h in _netrc_host_candidates(server):
        add(h)

    return out


def _credentials_from_netrc(
    server: str,
    netrc_path: str | Path,
    *,
    preferred_user: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (login, password) from a netrc file for *server*, or (None, None).

    If *preferred_user* is set, only an entry whose login matches is returned
    (so explicit ``config.user`` can be paired with the password from netrc).
    """
    path = Path(netrc_path).expanduser()
    if not path.is_file():
        return None, None
    try:
        n = netrc.netrc(str(path))
    except OSError:
        return None, None

    machines = _netrc_machine_candidates(server)

    if preferred_user:
        for machine in machines:
            try:
                login, _account, password = n.authenticators(machine)
            except (KeyError, TypeError):
                continue
            if login == preferred_user and password:
                return login, password
        return None, None

    for machine in machines:
        try:
            login, _account, password = n.authenticators(machine)
        except (KeyError, TypeError):
            continue
        if login and password:
            return login, password
    return None, None


def connect_xnat(config: XnatConnectionConfig):
    if xnat is None:
        raise BackendUnavailableError('xnat is not installed. Please install it with "pip install xnat".')

    user = config.user
    password = config.password
    if config.netrc_file:
        nu, np = _credentials_from_netrc(
            config.server,
            config.netrc_file,
            preferred_user=user,
        )
        if nu is not None:
            user = nu
        if np is not None:
            password = np

    kwargs: dict[str, Any] = {
        "server": config.server,
        "verify": config.verify,
        "default_timeout": config.default_timeout,
    }
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password
    if config.netrc_file and not (user and password):
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


def _download_resource_bundle(scan: Any, zip_path: Path, resource_label: str) -> Path:
    """Download a scan resource (e.g. ``NIFTI``) to ``zip_path``; returns path to zip or bundle."""
    resources = getattr(scan, "resources", None)
    if not resources or resource_label not in resources:
        raise RuntimeError(
            f"Scan does not expose a downloadable resource {resource_label!r}. "
            f"Available: {list(resources.keys()) if resources else []}"
        )
    result = resources[resource_label].download(zip_path, verbose=False)
    return Path(result) if result is not None else zip_path


def _is_nifti_filename(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".nii.gz") or lower.endswith(".nii")


def _is_json_sidecar_filename(name: str) -> bool:
    return name.lower().endswith(".json")


def _extract_zip_member_flat(archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> Path:
    """Write one archive file to ``destination`` using basename only; resolve name collisions."""
    base_name = Path(member.filename).name
    target = destination / base_name
    stem = target.stem
    suffix = "".join(target.suffixes)
    counter = 1
    while target.exists():
        target = destination / f"{stem}_{counter}{suffix}"
        counter += 1
    with archive.open(member) as source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return target


def download_scan_niftis(
    scan: Any,
    output_dir: str | Path,
    *,
    resource_label: str = "NIFTI",
    keep_zip: bool = False,
) -> list[Path]:
    """Download and extract NIfTI files from an XNAT scan resource into ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nvitk_xnat_nifti_") as tmp_dir:
        zip_path = Path(tmp_dir) / "scan_nifti.zip"
        bundle_path = _download_resource_bundle(scan, zip_path, resource_label)
        extracted: list[Path] = []
        with zipfile.ZipFile(bundle_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                base_name = Path(member.filename).name
                if not _is_nifti_filename(base_name):
                    continue
                extracted.append(_extract_zip_member_flat(archive, member, destination))

            for member in archive.infolist():
                if member.is_dir():
                    continue
                base_name = Path(member.filename).name
                if not _is_json_sidecar_filename(base_name):
                    continue
                extracted.append(_extract_zip_member_flat(archive, member, destination))

        if keep_zip:
            kept_zip = destination / "scan_nifti_bundle.zip"
            shutil.copy2(bundle_path, kept_zip)
            extracted.append(kept_zip)

    return extracted


def _list_local_nifti_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and _is_nifti_filename(p.name))


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
    download_niftis: bool = False,
    nifti_resource_label: str = "NIFTI",
    nifti_download_root: str | Path | None = None,
    skip_existing: bool = True,
    build_sqlite_index: bool = False,
    cohort_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    if download_niftis:
        base = nifti_download_root if nifti_download_root is not None else download_root
        if base is None:
            raise ValueError("download_niftis requires download_root or nifti_download_root to be set.")

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
                    sequence_label = classification["sequence"]
                    asset_slot = xnat_sequence_to_asset_slot(sequence_label)

                    if download_dicoms and download_root is not None:
                        target_dir = Path(download_root) / subject_uid / sequence_label
                        if skip_existing and target_dir.exists() and any(target_dir.iterdir()):
                            extracted_files = list(target_dir.iterdir())
                        else:
                            extracted_files = download_scan_dicoms(scan, target_dir)
                        local_cache_path = str(target_dir)
                        dicom_meta = json.dumps({"sequence": sequence_label, "asset_slot": asset_slot})
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
                                    "pipeline_id": pd.NA,
                                    "exists_locally": True,
                                    "asset_slot": asset_slot,
                                    "metadata_json": dicom_meta,
                                    "source_batch_id": source_batch_id,
                                    "updated_at": utc_now_iso(),
                                }
                            )

                    if download_niftis:
                        nifti_base = Path(nifti_download_root) if nifti_download_root is not None else Path(download_root)  # type: ignore[arg-type]
                        nifti_dir = nifti_base / subject_uid / sequence_label / "nifti"
                        if skip_existing and nifti_dir.exists() and _list_local_nifti_files(nifti_dir):
                            nifti_files = _list_local_nifti_files(nifti_dir)
                        else:
                            nifti_files = download_scan_niftis(
                                scan,
                                nifti_dir,
                                resource_label=nifti_resource_label,
                            )
                        meta = json.dumps(
                            {"resource": nifti_resource_label, "sequence": sequence_label, "asset_slot": asset_slot}
                        )
                        for file_path in nifti_files:
                            asset_rows.append(
                                {
                                    "asset_uid": f"{scan_uid}:nifti:{nifti_resource_label}:{file_path.name}",
                                    "subject_uid": subject_uid,
                                    "session_uid": session_uid,
                                    "modality": classification["modality"],
                                    "asset_type": "nifti",
                                    "asset_path": str(file_path),
                                    "resource_label": nifti_resource_label,
                                    "source": "xnat",
                                    "pipeline_name": pd.NA,
                                    "pipeline_id": pd.NA,
                                    "exists_locally": bool(file_path.exists()),
                                    "asset_slot": asset_slot,
                                    "metadata_json": meta,
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
                            "asset_slot": asset_slot,
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

    cohort_resolved = (cohort_id or "").strip() or DEFAULT_COHORT_ID
    synced_subjects = sorted({str(r["subject_uid"]) for r in subject_rows if r.get("subject_uid")})
    if synced_subjects:
        upsert_cohort_membership_for_subjects(
            repo,
            cohort_resolved,
            synced_subjects,
            source_batch_id=source_batch_id,
            membership_source="sync_xnat_project",
        )

    if build_sqlite_index:
        repo.build_sqlite_index()
    return frames


@_click_command()
@_click_option("--dataset-root", type=click.Path(path_type=Path) if click is not None else None, default=Path("dataset"), show_default=True, help="Dataset root to update.")
@_click_option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="YAML/JSON file with server, project, and optional auth settings. "
    "If omitted, uses NVITK_XNAT_CONFIG or ~/.config/nvitk/xnat.{yaml,yml,json}.",
)
@_click_option("--server", type=str, default=None, help="XNAT server URL (overrides config / XNAT_SERVER).")
@_click_option("--project", type=str, default=None, help="XNAT project identifier (overrides config / XNAT_PROJECT).")
@_click_option("--user", type=str, default=None, help="XNAT username.")
@_click_option("--password", type=str, default=None, help="XNAT password (prefer env, keyring, or netrc).")
@_click_option("--netrc-file", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Optional netrc file to use for authentication.")
@_click_option("--catalog-path", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None, help="Optional catalog CSV used to resolve subject IDs.")
@_click_option("--subjects", type=str, default=None, help="Comma or space separated subject identifiers.")
@_click_option("--subjects-file", type=click.Path(exists=True, path_type=Path) if click is not None else None, default=None, help="Text file with one subject identifier per line.")
@_click_option("--id-type", type=click.Choice(["subject", "mrid"], case_sensitive=False) if click is not None else None, default="subject", show_default=True, help="Interpret the provided IDs as XNAT subject labels or MR IDs.")
@_click_option(
    "--sequences",
    type=str,
    default=None,
    help="Comma-separated sequence keys (spaces allowed inside a name), e.g. "
    "TOF,4DFLOW_AP,3D T1,3D_T2 HR,3D_FLAIR,SWI/QSM,QSM,carotid_QF,RESTING STATE MB.",
)
@_click_option("--download-root", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Local directory for DICOM / NIfTI downloads (see per-modality layout below).")
@_click_option("--download-dicoms", is_flag=True, help="Download and extract DICOMs while syncing metadata.")
@_click_option("--download-niftis", is_flag=True, help="Download NIfTI resource per scan into {root}/{subject}/{sequence}/nifti/.")
@_click_option("--nifti-resource-label", type=str, default="NIFTI", show_default=True, help="XNAT scan resource label for NIfTI files.")
@_click_option(
    "--nifti-download-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=None,
    help="Optional separate root for NIfTI files (default: same as --download-root).",
)
@_click_option("--skip-existing", is_flag=True, help="Skip scan downloads when the local cache directory is already populated.")
@_click_option("--build-sqlite-index", is_flag=True, help="Rebuild the SQLite query cache after the sync.")
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
    sequences: str | None,
    download_root: Path | None,
    download_dicoms: bool,
    download_niftis: bool,
    nifti_resource_label: str,
    nifti_download_root: Path | None,
    skip_existing: bool,
    build_sqlite_index: bool,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

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
    frames = sync_xnat_project(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        requested_sequences=sequences,
        download_root=download_root,
        download_dicoms=download_dicoms,
        download_niftis=download_niftis,
        nifti_resource_label=nifti_resource_label,
        nifti_download_root=nifti_download_root,
        skip_existing=skip_existing,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Synced tables: {', '.join(name for name, frame in frames.items() if not frame.empty)}")


if __name__ == "__main__":
    main()
