#!/usr/bin/env python3
"""Upload qvtpy NIfTI inputs to XNAT scan-level ``NIFTI`` resources.

The script targets PESA-Brain MR sessions resolved from the TOF + 4D-flow scans
already present on XNAT. Files are uploaded per scan resource:

* TOF scan: ``TOF.nii[.gz]`` + optional JSON sidecar
* 4DFLOW_AP / RL / FH scans: ``*_ph`` and ``*_m`` NIfTIs + optional JSON
* Shared 4D-flow derivatives (``Angiography_3D``, ``ComplexDifference_3D``,
  ``ComplexDifference_4D``, ...) go to an experiment-level ``4dflows`` resource

When ``--from-sge`` is used, the script first stages the required files from the
remote NIfTI root over SFTP, uploads them to XNAT, then removes the temporary
local staging copy.
"""

from __future__ import annotations

import stat
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import click

from nvitk.cluster.remote_transfer import download_directory_sftp
from nvitk.core.logger import Logger
from nvitk.db.xnat import _coalesce_attr, classify_scan, connect_xnat, resolve_subject_labels
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_upload import resolve_subject_experiment, upload_directory_to_xnat_resource
from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials
from nvitk.pipes.qvtpy.util.io.paths import CLUSTER_HOST_ALIASES

log = Logger()

RESOURCE_LABEL = "NIFTI"
FOURDFLOWS_RESOURCE_LABEL = "4dflows"
FLOW_SEQUENCES = ("4DFLOW_AP", "4DFLOW_RL", "4DFLOW_FH")
FLOW_DERIVED_STEMS = (
    "Angiography_3D",
    # "Angiography_4D",
    "ComplexDifference_3D",
    "ComplexDifference_4D",
    # "VelocityMagnitude_3D",
    # "VelocityMagnitude_4D",
    # "VelocityMeanComponents",
)


def _iter_existing_files(directory: Path, patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(directory.glob(pattern)))
    return [p for p in out if p.is_file()]


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _matching_jsons_for_nifti(path: Path) -> list[Path]:
    """Locate JSON sidecar(s) for a NIfTI (PESA-Brain naming conventions)."""
    parent = path.parent
    stem = _nifti_stem(path)
    stem_lower = stem.lower()
    candidates: list[Path] = []

    if stem_lower.endswith("_ph"):
        base = stem[: -len("_ph")]
        candidates.extend(
            (
                parent / f"{base}_PHASE.json",
                parent / f"{base}_phase.json",
            )
        )
    elif stem_lower.endswith("_m"):
        base = stem[: -len("_m")]
        candidates.extend(
            (
                parent / f"{base}_M_FFE.json",
                parent / f"{base}_m_ffe.json",
            )
        )
    else:
        if path.name.endswith(".nii.gz"):
            candidates.append(Path(str(path)[:-7] + ".json"))
        elif path.suffix == ".nii":
            candidates.append(path.with_suffix(".json"))
        else:
            candidates.append(parent / f"{stem}.json")

    for candidate in candidates:
        if candidate.is_file():
            return [candidate]
    return []


def _json_files_in_dir(directory: Path, nifti_files: Iterable[Path] | None = None) -> list[Path]:
    """Collect JSON sidecars for *directory* (matched + any extra ``*.json``)."""
    if not directory.is_dir():
        return []
    niftis = list(nifti_files) if nifti_files is not None else _iter_existing_files(
        directory, ("*.nii.gz", "*.nii")
    )
    found: list[Path] = []
    seen: set[str] = set()
    for nii in niftis:
        for js in _matching_jsons_for_nifti(nii):
            if js.name not in seen:
                seen.add(js.name)
                found.append(js)
    for js in sorted(directory.glob("*.json")):
        if js.is_file() and js.name not in seen:
            seen.add(js.name)
            found.append(js)
    return found


def _copy_files_into_stage(files: Iterable[Path], stage_dir: Path) -> int:
    stage_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen: set[str] = set()
    for src in files:
        if src.name in seen:
            continue
        seen.add(src.name)
        shutil.copy2(src, stage_dir / src.name)
        copied += 1
    return copied


def _copy_scan_dir_to_stage(
    source_dir: Path,
    stage_dir: Path,
    nifti_patterns: Iterable[str],
    *,
    only_json: bool = False,
) -> int:
    """Stage scan files from *source_dir* (NIfTIs + JSON sidecars, or JSON only)."""
    niftis = _iter_existing_files(source_dir, nifti_patterns)
    if only_json:
        return _copy_files_into_stage(_json_files_in_dir(source_dir, niftis), stage_dir)

    to_copy: list[Path] = list(niftis)
    for nii in niftis:
        to_copy.extend(_matching_jsons_for_nifti(nii))
    to_copy.extend(
        js
        for js in _json_files_in_dir(source_dir, niftis)
        if js not in to_copy
    )
    return _copy_files_into_stage(to_copy, stage_dir)


def _find_flow_derived_files(subject_dir: Path) -> list[Path]:
    flow_dir = subject_dir / "4DFlow"
    files: list[Path] = []
    for stem in FLOW_DERIVED_STEMS:
        files.extend(_iter_existing_files(flow_dir, (f"{stem}.nii.gz", f"{stem}.nii")))
    return files


def _build_upload_stage(
    subject_dir: Path,
    stage_root: Path,
    *,
    only_json: bool = False,
) -> tuple[dict[str, Path], Path | None]:
    uploads: dict[str, Path] = {}

    tof_dir = subject_dir / "TOF"
    tof_stage = stage_root / "TOF"
    tof_copied = _copy_scan_dir_to_stage(
        tof_dir, tof_stage, ("TOF.nii.gz", "TOF.nii"), only_json=only_json
    )
    if tof_copied > 0:
        uploads["TOF"] = tof_stage

    for sequence in FLOW_SEQUENCES:
        direction = sequence.rsplit("_", 1)[-1]
        stage_dir = stage_root / sequence
        copied = _copy_scan_dir_to_stage(
            subject_dir / "4DFlow" / direction,
            stage_dir,
            ("*_ph.nii.gz", "*_ph.nii", "*_m.nii.gz", "*_m.nii"),
            only_json=only_json,
        )
        if copied > 0:
            uploads[sequence] = stage_dir

    derived_stage: Path | None = None
    flow_dir = subject_dir / "4DFlow"
    if only_json:
        derived_jsons = _json_files_in_dir(flow_dir)
        if derived_jsons:
            derived_stage = stage_root / FOURDFLOWS_RESOURCE_LABEL
            _copy_files_into_stage(derived_jsons, derived_stage)
    else:
        derived_files = _find_flow_derived_files(subject_dir)
        if derived_files:
            derived_stage = stage_root / FOURDFLOWS_RESOURCE_LABEL
            _copy_scan_dir_to_stage(
                flow_dir,
                derived_stage,
                tuple(f"{stem}.nii.gz" for stem in FLOW_DERIVED_STEMS)
                + tuple(f"{stem}.nii" for stem in FLOW_DERIVED_STEMS),
                only_json=False,
            )

    return uploads, derived_stage


def _scan_base_uri(scan: Any) -> str:
    for attr in ("fulluri", "uri"):
        value = getattr(scan, attr, None)
        if value:
            return str(value).rstrip("/")
    scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "").strip()
    if scan_id:
        return f"/data/scans/{scan_id}"
    raise RuntimeError("Cannot determine REST URI for scan")


def _clear_scan_resource_cache(scan: Any) -> None:
    for obj in (scan, getattr(scan, "resources", None)):
        if obj is None:
            continue
        clearcache = getattr(obj, "clearcache", None)
        if callable(clearcache):
            clearcache()


def _create_scan_resource(scan: Any, resource_label: str) -> Any:
    session = getattr(scan, "xnat_session", None)
    if session is None:
        raise RuntimeError("Scan has no xnat_session")
    uri = f"{_scan_base_uri(scan)}/resources/{resource_label}"
    session.put(uri)
    _clear_scan_resource_cache(scan)
    resource = session.create_object(uri)
    resources = getattr(scan, "resources", None)
    if resources is not None and resource_label in resources:
        return resources[resource_label]
    return resource


def _get_or_create_scan_resource(scan: Any, resource_label: str) -> Any:
    resources = getattr(scan, "resources", None)
    if resources is None:
        raise RuntimeError("Scan has no resources collection")
    if resource_label in resources:
        return resources[resource_label]
    create_resource = getattr(scan, "create_resource", None)
    if callable(create_resource):
        return create_resource(resource_label)
    return _create_scan_resource(scan, resource_label)


def _resource_files(resource: Any) -> list[Any]:
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


def _delete_existing_scan_resource(scan: Any, resource_label: str) -> None:
    resources = getattr(scan, "resources", None) or {}
    if resource_label not in resources:
        return
    resource = resources[resource_label]
    delete_resource = getattr(resource, "delete", None)
    if callable(delete_resource):
        delete_resource()
        _clear_scan_resource_cache(scan)
        return
    for file_obj in _resource_files(resource):
        delete_file = getattr(file_obj, "delete", None)
        if callable(delete_file):
            delete_file()
    _clear_scan_resource_cache(scan)


def _upload_stage_to_scan(scan: Any, local_stage_dir: Path, *, resource_label: str = RESOURCE_LABEL, overwrite: bool = True) -> int:
    if overwrite:
        _delete_existing_scan_resource(scan, resource_label)
    resource = _get_or_create_scan_resource(scan, resource_label)
    upload_dir = getattr(resource, "upload_dir", None)
    if not callable(upload_dir):
        raise RuntimeError(f"Scan resource {resource_label!r} does not support upload_dir")
    n_files = sum(1 for p in local_stage_dir.iterdir() if p.is_file())
    log.info(f"Upload {local_stage_dir} -> scan {resource_label} ({n_files} file(s))")
    upload_dir(str(local_stage_dir), overwrite=overwrite)
    return n_files


def _resolve_sequence_scans(experiment: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scan in getattr(experiment, "scans", {}).values():
        description = str(_coalesce_attr(scan, "series_description", "type", "label") or "")
        quality = str(_coalesce_attr(scan, "quality") or "")
        classification = classify_scan(description, quality)
        if classification is None:
            continue
        sequence = str(classification.get("sequence") or "").strip().upper()
        if sequence in {"TOF", *FLOW_SEQUENCES} and sequence not in out:
            out[sequence] = scan
    return out


def _fetch_remote_subject_tree(subject: str, remote_nifti_root: Path, local_subject_dir: Path, *, host: str, user: str, password: str) -> Path:
    remote_subject_dir = f"{str(remote_nifti_root).rstrip('/')}/{subject}"
    from nvitk.cluster.remote_transfer import sftp_session

    local_subject_dir.mkdir(parents=True, exist_ok=True)
    with sftp_session(host=host, user=user, password=password) as (_ssh, sftp):
        log.info(f"[{subject}] SFTP fetch {remote_subject_dir} -> {local_subject_dir}")
        download_directory_sftp(sftp, remote_subject_dir, local_subject_dir)
    return local_subject_dir


def _iter_local_subject_dirs(nifti_root: Path) -> list[str]:
    if not nifti_root.is_dir():
        return []
    return sorted(p.name for p in nifti_root.iterdir() if p.is_dir())


def _iter_remote_subject_dirs(remote_nifti_root: Path, *, host: str, user: str, password: str) -> list[str]:
    remote_root = str(remote_nifti_root).rstrip("/")
    from nvitk.cluster.remote_transfer import sftp_session

    with sftp_session(host=host, user=user, password=password) as (_ssh, sftp):
        try:
            entries = sftp.listdir_attr(remote_root)
        except OSError as exc:
            raise FileNotFoundError(f"Remote nifti root not accessible: {remote_root}") from exc
    out: list[str] = []
    for e in entries:
        if stat.S_ISDIR(e.st_mode):
            name = str(e.filename)
            if name not in {".", ".."}:
                out.append(name)
    return sorted(out)


@click.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="XNAT config profile.")
@click.option("--server", type=str, default=None, help="XNAT server URL.")
@click.option("--project", type=str, default="PESA_Brain", show_default=True, help="XNAT project id.")
@click.option("--user", type=str, default=None, help="XNAT username.")
@click.option("--password", type=str, default=None, help="XNAT password.")
@click.option("--netrc-file", type=click.Path(path_type=Path), default=None, help="Optional netrc file.")
@click.option("--catalog-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--subjects", type=str, default=None, help="Comma/space separated subject ids.")
@click.option("--subjects-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--id-type", type=click.Choice(["subject", "mrid"], case_sensitive=False), default="subject", show_default=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True, help="Input NIfTI root (local root or remote root with --from-sge).")
@click.option("--resource-label", type=str, default=RESOURCE_LABEL, show_default=True, help="XNAT scan resource label.")
@click.option("--from-sge", is_flag=True, default=True, help="Fetch the input subject tree from a remote SGE server before upload.")
@click.option("--remote-host", type=str, default=None, help="SSH host or alias when --from-sge is used.")
@click.option("--remote-user", type=str, default=None, help="SSH user when --from-sge is used.")
@click.option(
    "--only-json",
    is_flag=True,
    default=False,
    help="Upload only JSON sidecars (merge into existing XNAT resources; do not delete NIfTIs).",
)
def main(
    config_path: Path | None,
    server: str | None,
    project: str,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    catalog_path: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    id_type: str,
    nifti_root: Path,
    resource_label: str,
    from_sge: bool,
    remote_host: str | None,
    remote_user: str | None,
    only_json: bool,
) -> None:
    auto_discover_subjects = subjects is None and subjects_file is None and catalog_path is None

    profile = load_xnat_profile(config_path)
    conn = resolve_xnat_connection(
        profile,
        server=server,
        project=project,
        user=user,
        password=password,
        netrc_file=str(netrc_file) if netrc_file else None,
    )
    ssh_host = ssh_user = ssh_password = None
    if from_sge:
        ssh_host, ssh_user, ssh_password = prompt_ssh_credentials(
            remote_host=remote_host,
            remote_user=remote_user,
            host_aliases=CLUSTER_HOST_ALIASES,
        )

    with connect_xnat(conn) as session:
        project_obj = session.projects[conn.project]

        if auto_discover_subjects:
            if from_sge:
                candidates = _iter_remote_subject_dirs(
                    nifti_root, host=ssh_host, user=ssh_user, password=ssh_password
                )
            else:
                candidates = _iter_local_subject_dirs(nifti_root)

            if not candidates:
                raise click.ClickException(
                    f"No subject directories found under nifti root: {nifti_root}"
                )

            subject_list = sorted(
                s for s in candidates if str(s) in getattr(project_obj, "subjects", {})
            )
            if not subject_list:
                raise click.ClickException(
                    "No discovered subjects exist in the target XNAT project."
                )
        else:
            subject_list = resolve_subject_labels(
                catalog_path=catalog_path,
                subjects=subjects,
                subjects_file=subjects_file,
                id_type=id_type,
            )
            if not subject_list:
                raise click.ClickException("No subjects resolved for upload.")

        uploaded_subjects = 0
        for subject in subject_list:
            try:
                experiment, experiment_label = resolve_subject_experiment(project_obj, subject)
                scans_by_sequence = _resolve_sequence_scans(experiment)
                if not scans_by_sequence:
                    log.warning(f"[{subject}] no TOF/4DFlow scans found in experiment {experiment_label}")
                    continue

                with tempfile.TemporaryDirectory(prefix=f"nvitk-upload-xnat-nifti-{subject}-") as tmp:
                    tmp_root = Path(tmp)
                    if from_sge:
                        subject_dir = _fetch_remote_subject_tree(
                            subject,
                            nifti_root,
                            tmp_root / subject,
                            host=ssh_host,
                            user=ssh_user,
                            password=ssh_password,
                        )
                    else:
                        subject_dir = nifti_root / subject
                        if not subject_dir.is_dir():
                            raise FileNotFoundError(f"Subject NIfTI directory missing: {subject_dir}")

                    stage_root = tmp_root / f"{subject}_upload"
                    uploads, derived_stage = _build_upload_stage(
                        subject_dir, stage_root, only_json=only_json
                    )
                    if not uploads and derived_stage is None:
                        kind = "JSON" if only_json else "NIfTI"
                        log.warning(f"[{subject}] no {kind} files found to upload")
                        continue

                    overwrite = not only_json
                    for sequence, local_stage_dir in uploads.items():
                        scan = scans_by_sequence.get(sequence)
                        if scan is None:
                            log.warning(f"[{subject}] XNAT scan missing for sequence {sequence}; skipping upload")
                            continue
                        _upload_stage_to_scan(
                            scan,
                            local_stage_dir,
                            resource_label=resource_label,
                            overwrite=overwrite,
                        )

                    if derived_stage is not None:
                        log.info(
                            f"[{subject}] upload derivatives -> experiment resource {FOURDFLOWS_RESOURCE_LABEL!r}"
                        )
                        upload_directory_to_xnat_resource(
                            experiment,
                            FOURDFLOWS_RESOURCE_LABEL,
                            derived_stage,
                            overwrite=overwrite,
                        )

                uploaded_subjects += 1
            except Exception as exc:
                log.warning(f"[{subject}] upload failed: {exc}")

    kind = "JSON sidecar" if only_json else "NIfTI"
    click.echo(f"Uploaded {kind} resources for {uploaded_subjects} subject(s).")


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except SystemExit as exc:
        raise SystemExit(exc.code) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
