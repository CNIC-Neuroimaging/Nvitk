"""Stage0: download gPET inputs from XNAT into the gpetpy DICOM layout.

Sources (locked by plan):
- ``IA_PET_V5``: CT + PET
- ``PESA_Brain``: 3D T1
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from nvitk.core.logger import Logger
from nvitk.db.xnat import connect_xnat, download_scan_dicoms
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import classify_scan_for_project
from nvitk.pipes.pesa_fat.common.xnat_inputs import batch_from_session_date

from .layout import DEFAULT_DICOM_ROOT, DEFAULT_NIFTI_ROOT, DEFAULT_RESULTS_ROOT, GpetLayout

log = Logger()


@dataclass(frozen=True)
class GpetXnatRequest:
    ia_project_id: str = "IA_PET_V5"
    brain_project_id: str = "PESA_Brain"
    ia_session_label: str | None = None
    brain_session_label: str | None = None
    skip_existing: bool = True


def _coalesce_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            v = getattr(obj, name)
            if callable(v):
                try:
                    v = v()
                except TypeError:
                    pass
            if v is not None:
                return v
    return None


def _experiment_label(experiment: Any) -> str:
    return str(_coalesce_attr(experiment, "label", "id") or "").strip()


def _experiment_date(experiment: Any) -> Any:
    return _coalesce_attr(experiment, "date")


def _classify_scans_for_experiment(project_id: str, experiment: Any) -> dict[str, Any]:
    scans = list(getattr(experiment, "scans", {}).values())
    selected: dict[str, Any] = {}
    exp_label = _experiment_label(experiment)
    for scan in scans:
        scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
        desc = str(_coalesce_attr(scan, "series_description", "type", "label") or "")
        quality = str(_coalesce_attr(scan, "quality") or "")
        cls = classify_scan_for_project(
            project_id,
            desc,
            quality,
            scan_id=scan_id,
            experiment_label=exp_label,
        )
        if cls is None:
            continue
        seq = str(cls.get("sequence") or "").strip()
        if not seq:
            continue
        selected.setdefault(seq, scan)
    return selected


def _collect_scans_across_experiments(
    *,
    project_id: str,
    subject: Any,
    required_sequences: set[str],
) -> dict[str, tuple[Any, Any]]:
    """Return mapping {sequence: (experiment, scan)} spanning multiple experiments."""
    experiments = list(getattr(subject, "experiments", {}).values())
    picked: dict[str, tuple[Any, Any]] = {}
    for exp in experiments:
        selected = _classify_scans_for_experiment(project_id, exp)
        for seq, scan in selected.items():
            if seq not in required_sequences:
                continue
            picked.setdefault(seq, (exp, scan))
    return picked


def _pick_experiment_with_sequences(
    *,
    project_id: str,
    subject: Any,
    session_label: str | None,
    required_sequences: set[str],
) -> tuple[Any, dict[str, Any]]:
    """Pick one experiment that contains all required sequences."""
    experiments = list(getattr(subject, "experiments", {}).values())
    if session_label:
        for exp in experiments:
            if _experiment_label(exp) == str(session_label).strip():
                selected = _classify_scans_for_experiment(project_id, exp)
                missing = required_sequences - set(selected.keys())
                if missing:
                    raise LookupError(
                        f"XNAT session {session_label!r} found but missing sequences: {sorted(missing)}"
                    )
                return exp, selected
        known = ", ".join(sorted({lab for lab in (_experiment_label(e) for e in experiments) if lab}))
        raise LookupError(f"XNAT session {session_label!r} not found. Known: {known}")

    # No label override: choose newest experiment satisfying required sequences.
    best: Any | None = None
    best_date: Any | None = None
    best_selected: dict[str, Any] = {}
    for exp in experiments:
        selected = _classify_scans_for_experiment(project_id, exp)
        if not required_sequences.issubset(set(selected.keys())):
            continue
        dt = _experiment_date(exp)
        if best is None:
            best, best_date, best_selected = exp, dt, selected
            continue
        try:
            if dt is not None and best_date is not None and dt >= best_date:
                best, best_date, best_selected = exp, dt, selected
        except Exception:
            # keep the first one if dates can't compare
            pass
    if best is None:
        # fallback: allow required sequences across experiments (useful if XNAT splits content)
        picked = _collect_scans_across_experiments(
            project_id=project_id, subject=subject, required_sequences=required_sequences
        )
        missing = sorted(required_sequences - set(picked.keys()))
        if missing:
            raise LookupError(
                f"No XNAT experiment for project {project_id!r} contained required sequences "
                f"{sorted(required_sequences)} (missing {missing})."
            )
        # derive an experiment/date from the first picked sequence (good enough for manifest)
        seq0 = sorted(picked.keys())[0]
        exp0, _scan0 = picked[seq0]
        selected0 = _classify_scans_for_experiment(project_id, exp0)
        return exp0, selected0
    return best, best_selected


def _should_skip_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file():
            return True
    return False


def download_subject_from_xnat(
    *,
    subject: str,
    batch: str | None,
    dicom_root: Path | None = None,
    nifti_root: Path | None = None,
    results_root: Path | None = None,
    xnat_config_path: Path | None = None,
    request: GpetXnatRequest | None = None,
) -> tuple[str, GpetLayout, dict[str, Path]]:
    """Download CT/PET/T1 dicoms for one subject into the gpetpy layout.

    Returns (batch, layout, local_dirs) where local_dirs maps:
    - ``IA_PET_V5/CT``, ``IA_PET_V5/PET``, ``PESA_Brain/3D_T1`` → download dirs.
    """
    subj = str(subject).strip()
    if not subj:
        raise ValueError("subject must be non-empty")

    req = request or GpetXnatRequest()
    profile = load_xnat_profile(xnat_config_path)

    cfg_ia = resolve_xnat_connection(profile, project=req.ia_project_id)
    cfg_br = resolve_xnat_connection(profile, project=req.brain_project_id)

    # --- IA (CT+PET) ---
    with connect_xnat(cfg_ia) as session:
        project = session.projects[req.ia_project_id]
        if subj not in project.subjects:
            raise LookupError(f"XNAT subject {subj!r} not found in project {req.ia_project_id!r}")
        xsubj = project.subjects[subj]
        exp_ia, selected_ia = _pick_experiment_with_sequences(
            project_id=req.ia_project_id,
            subject=xsubj,
            session_label=req.ia_session_label,
            required_sequences={"CT", "PET"},
        )
        ia_label = _experiment_label(exp_ia)
        ia_date = _experiment_date(exp_ia)

    batch_in = None if batch is None else str(batch).strip()
    batch_auto = (batch_in is None) or (batch_in.lower() == "auto")
    derived_batch = batch_from_session_date(ia_date) if batch_auto else batch_in

    lay = GpetLayout(
        batch=derived_batch,
        subject=subj,
        dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
        nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
        results_root=results_root or DEFAULT_RESULTS_ROOT,
    )

    # --- Brain (T1) ---
    with connect_xnat(cfg_br) as session:
        project = session.projects[req.brain_project_id]
        if subj not in project.subjects:
            raise LookupError(f"XNAT subject {subj!r} not found in project {req.brain_project_id!r}")
        xsubj = project.subjects[subj]
        exp_br, selected_br = _pick_experiment_with_sequences(
            project_id=req.brain_project_id,
            subject=xsubj,
            session_label=req.brain_session_label,
            required_sequences={"3D_T1"},
        )
        br_label = _experiment_label(exp_br)
        br_date = _experiment_date(exp_br)

    out_dirs: dict[str, Path] = {}

    # Download with new sessions so we don't keep stale objects.
    with connect_xnat(cfg_ia) as session:
        project = session.projects[req.ia_project_id]
        xsubj = project.subjects[subj]
        exp_ia, selected_ia = _pick_experiment_with_sequences(
            project_id=req.ia_project_id,
            subject=xsubj,
            session_label=req.ia_session_label or ia_label,
            required_sequences={"CT", "PET"},
        )
        scan_ct = selected_ia["CT"]
        scan_pet = selected_ia["PET"]
        ct_dir = lay.dicom_ia_ct_dir()
        pet_dir = lay.dicom_ia_pet_dir()
        if not (req.skip_existing and _should_skip_dir(ct_dir)):
            download_scan_dicoms(scan_ct, ct_dir, keep_zip=False)
        if not (req.skip_existing and _should_skip_dir(pet_dir)):
            download_scan_dicoms(scan_pet, pet_dir, keep_zip=False)
        out_dirs["IA_PET_V5/CT"] = ct_dir
        out_dirs["IA_PET_V5/PET"] = pet_dir

    with connect_xnat(cfg_br) as session:
        project = session.projects[req.brain_project_id]
        xsubj = project.subjects[subj]
        exp_br, selected_br = _pick_experiment_with_sequences(
            project_id=req.brain_project_id,
            subject=xsubj,
            session_label=req.brain_session_label or br_label,
            required_sequences={"3D_T1"},
        )
        scan_t1 = selected_br["3D_T1"]
        t1_dir = lay.dicom_pesabrain_t1_dir()
        if not (req.skip_existing and _should_skip_dir(t1_dir)):
            download_scan_dicoms(scan_t1, t1_dir, keep_zip=False)
        out_dirs["PESA_Brain/3D_T1"] = t1_dir

    # Manifest under the subject DICOM dir.
    manifest = {
        "subject": subj,
        "batch": derived_batch,
        "created_at": datetime.now().isoformat(),
        "ia_pet_v5": {"project": req.ia_project_id, "session_label": ia_label, "date": str(ia_date)},
        "pesa_brain": {"project": req.brain_project_id, "session_label": br_label, "date": str(br_date)},
        "dicom_dirs": {k: str(v) for k, v in out_dirs.items()},
    }
    lay.dicom_dir.mkdir(parents=True, exist_ok=True)
    (lay.dicom_dir / "xnat_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return derived_batch, lay, out_dirs


def download_from_xnat(
    *,
    subjects: Iterable[str],
    batch: str | None,
    dicom_root: Path | None = None,
    nifti_root: Path | None = None,
    results_root: Path | None = None,
    xnat_config_path: Path | None = None,
    request: GpetXnatRequest | None = None,
) -> dict[str, tuple[str, GpetLayout]]:
    """Download stage for multiple subjects; returns {subject: (batch, layout)}."""
    out: dict[str, tuple[str, GpetLayout]] = {}
    for s in subjects:
        subj = str(s).strip()
        if not subj:
            continue
        b, lay, _dirs = download_subject_from_xnat(
            subject=subj,
            batch=batch,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            results_root=results_root,
            xnat_config_path=xnat_config_path,
            request=request,
        )
        out[subj] = (b, lay)
        log.info("Downloaded gpetpy DICOMs for %s (batch=%s).", subj, b)
    return out

