"""XNAT input mode for PESA-Fat (IA_PET_V5).

This module downloads the required DICOM resources from XNAT into the existing
PESA-Fat DICOM layout so the rest of the pipeline (stage0_convert → stages 1-3)
can run unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from nvitk.core.logger import Logger
from nvitk.db.xnat import connect_xnat, download_scan_dicoms
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import (
    classify_experiment_ia_pet_v5,
    classify_scan_for_project,
    default_sequences_for_project,
    get_xnat_project,
)

log = Logger()


@dataclass(frozen=True)
class XnatPesaFatRequest:
    """Parameters for downloading one PESA-Fat session's DICOM scans from XNAT."""

    project_id: str = "IA_PET_V5"
    session_label: str | None = None
    requested_sequences: tuple[str, ...] | None = None
    skip_existing: bool = True


def _coalesce_attr(obj: Any, *names: str) -> Any:
    """Return the first non-``None`` attribute among *names* on *obj*, or ``None``."""
    for name in names:
        if hasattr(obj, name):
            v = getattr(obj, name)
            if v is not None:
                return v
    return None


def _experiment_label(experiment: Any) -> str:
    """XNAT experiment label (falling back to id), stripped."""
    return str(_coalesce_attr(experiment, "label", "id") or "").strip()


def _experiment_date(experiment: Any) -> Any:
    """*experiment*'s raw ``date`` attribute, or ``None`` if unset."""
    return _coalesce_attr(experiment, "date")


def batch_from_session_date(value: Any) -> str:
    """Convert an XNAT experiment date into PESA-Fat batch name: YYYYMM_Week1..Week4."""
    if value is None:
        raise ValueError("XNAT session date is missing; cannot derive batch.")
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            raise ValueError("XNAT session date is empty; cannot derive batch.")
        # XNAT commonly returns YYYY-MM-DD.
        try:
            dt = datetime.fromisoformat(s[:10])
        except Exception as exc:
            raise ValueError(f"Could not parse XNAT date {value!r}") from exc
    year = dt.year
    month = dt.month
    day = dt.day
    week = min(4, ((day - 1) // 7) + 1)
    return f"{year:04d}{month:02d}_Week{week}"


def _classify_scans_for_experiment(project_id: str, experiment: Any) -> dict[str, Any]:
    """Classify every scan in *experiment* and return ``{sequence_key: first_matching_scan}``, using
    the project's dedicated classifier (e.g. ``ia_pet_v5``) when registered."""
    scans = list(getattr(experiment, "scans", {}).values())
    exp_label = _experiment_label(experiment)
    if get_xnat_project(project_id).classifier == "ia_pet_v5":
        selected: dict[str, Any] = {}
        for scan, _scan_id, _desc, _quality, cls in classify_experiment_ia_pet_v5(
            scans,
            experiment_label=exp_label,
        ):
            seq = str(cls.get("sequence") or "").strip()
            if seq:
                selected.setdefault(seq, scan)
        return selected

    selected = {}
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


def _debug_experiment_scans(project_id: str, experiment: Any) -> str:
    """Human-readable scan inventory + IA_PET_V5 classification decisions."""
    scans = list(getattr(experiment, "scans", {}).values())
    lines: list[str] = []
    exp_label = _experiment_label(experiment)
    exp_date = _experiment_date(experiment)
    lines.append(f"experiment={exp_label!r} date={exp_date!r} n_scans={len(scans)}")
    for scan in scans:
        scan_id = str(_coalesce_attr(scan, "id", "label", "name") or "")
        desc = str(_coalesce_attr(scan, "series_description", "type", "label") or "")
        quality = str(_coalesce_attr(scan, "quality") or "")
        try:
            if get_xnat_project(project_id).classifier == "ia_pet_v5":
                matches = [
                    cls
                    for s, sid, d, q, cls in classify_experiment_ia_pet_v5(
                        scans,
                        experiment_label=exp_label,
                    )
                    if sid == scan_id
                ]
                cls = matches[0] if matches else classify_scan_for_project(
                    project_id,
                    desc,
                    quality,
                    scan_id=scan_id,
                    experiment_label=exp_label,
                )
            else:
                cls = classify_scan_for_project(
                    project_id,
                    desc,
                    quality,
                    scan_id=scan_id,
                    experiment_label=exp_label,
                )
        except Exception as exc:
            cls = {"error": str(exc)}
        lines.append(
            f"  scan_id={scan_id!r} quality={quality!r} desc={desc!r} -> {cls}"
        )
    return "\n".join(lines)


def _pick_experiment_for_subject(
    *,
    project_id: str,
    subject: Any,
    session_label: str | None,
    required_sequences: set[str],
) -> Any:
    """Select *subject*'s XNAT experiment: the one named *session_label* if given, else the newest
    experiment whose classified scans cover every required sequence; raises ``LookupError`` (with a
    detailed per-experiment diagnostic) if none match."""
    experiments = list(getattr(subject, "experiments", {}).values())
    if session_label:
        for exp in experiments:
            if _experiment_label(exp) == str(session_label).strip():
                return exp
        known = ", ".join(sorted({lab for lab in (_experiment_label(e) for e in experiments) if lab}))
        raise LookupError(f"XNAT session {session_label!r} not found. Known: {known}")

    best = None
    best_date = None
    for exp in experiments:
        classified = set(_classify_scans_for_experiment(project_id, exp).keys())
        if required_sequences and not required_sequences.issubset(classified):
            continue
        dt = _experiment_date(exp)
        if best is None:
            best = exp
            best_date = dt
            continue
        # Prefer newest when date is comparable; otherwise keep first match.
        try:
            if dt is not None and best_date is not None and dt >= best_date:
                best = exp
                best_date = dt
        except Exception:
            pass
    if best is None:
        # Emit a detailed diagnostic to help understand classification mismatches.
        try:
            subject_label = str(_coalesce_attr(subject, "label", "id", "name") or "").strip()
        except Exception:
            subject_label = ""
        wanted = sorted(required_sequences)
        blocks: list[str] = [f"Required sequences: {wanted}"]
        for exp in experiments:
            classified = set(_classify_scans_for_experiment(project_id, exp).keys())
            missing = sorted(required_sequences - classified)
            blocks.append(
                f"- experiment={_experiment_label(exp)!r} date={_experiment_date(exp)!r} "
                f"classified={sorted(classified)} missing={missing}"
            )
            blocks.append(_debug_experiment_scans(project_id, exp))
        raise LookupError(
            f"No XNAT experiment for subject {subject_label!r} matched required sequences.\n"
            + "\n".join(blocks)
        )
    return best


def _required_sequences(project_id: str, requested: Iterable[str] | None) -> set[str]:
    """Resolve the set of required sequence keys: *requested* if non-empty, else *project_id*'s
    default sequences."""
    seqs = [str(s).strip() for s in (requested or []) if str(s).strip()]
    if not seqs:
        seqs = list(default_sequences_for_project(project_id))
    return set(seqs)


def _collect_scans_across_experiments(
    *,
    project_id: str,
    subject: Any,
    required_sequences: set[str],
) -> tuple[dict[str, tuple[Any, Any]], dict[str, set[str]]]:
    """Return mapping {sequence: (experiment, scan)} possibly spanning multiple experiments."""
    experiments = list(getattr(subject, "experiments", {}).values())
    picked: dict[str, tuple[Any, Any]] = {}
    per_exp_classified: dict[str, set[str]] = {}
    for exp in experiments:
        exp_label = _experiment_label(exp)
        selected = _classify_scans_for_experiment(project_id, exp)
        classified = set(selected.keys())
        per_exp_classified[exp_label or repr(exp)] = classified
        for seq, scan in selected.items():
            if seq not in required_sequences:
                continue
            # Keep first hit; we can add smarter date/tie-breaking later.
            picked.setdefault(seq, (exp, scan))
    return picked, per_exp_classified


def download_pesa_fat_dicoms_from_xnat(
    *,
    batch: str | None,
    subjects: Iterable[str],
    dicom_root: Path,
    xnat_config_path: Path | None = None,
    request: XnatPesaFatRequest | None = None,
) -> dict[str, tuple[str, dict[str, Path]]]:
    """Download DICOMs from XNAT into ``dicom_root/<batch>/<subject>/<sequence>/``.

    If *batch* is None or \"auto\", the batch is derived per subject from the XNAT experiment date.

    Returns ``{subject: (batch, {sequence: local_dir})}``.
    """
    req = request or XnatPesaFatRequest()
    project_id = str(req.project_id).strip()
    required = _required_sequences(project_id, req.requested_sequences)
    batch_in = None if batch is None else str(batch).strip()
    batch_auto = (batch_in is None) or (batch_in.lower() == "auto")

    profile = load_xnat_profile(xnat_config_path)
    cfg = resolve_xnat_connection(profile, project=project_id)

    out: dict[str, tuple[str, dict[str, Path]]] = {}
    with connect_xnat(cfg) as session:
        project = session.projects[project_id]
        for subj_label in subjects:
            subj_label = str(subj_label).strip()
            if not subj_label:
                continue
            if subj_label not in project.subjects:
                raise LookupError(f"XNAT subject {subj_label!r} not found in project {project_id!r}")
            subject = project.subjects[subj_label]
            if req.session_label:
                exp = _pick_experiment_for_subject(
                    project_id=project_id,
                    subject=subject,
                    session_label=req.session_label,
                    required_sequences=required,
                )
                selected = _classify_scans_for_experiment(project_id, exp)
                picked: dict[str, tuple[Any, Any]] = {k: (exp, v) for k, v in selected.items()}
            else:
                picked, _per_exp = _collect_scans_across_experiments(
                    project_id=project_id,
                    subject=subject,
                    required_sequences=required,
                )
            missing = sorted(required - set(picked.keys()))
            if missing:
                # Emit the same detailed diagnostic as _pick_experiment_for_subject does.
                experiments = list(getattr(subject, "experiments", {}).values())
                blocks: list[str] = [f"Required sequences: {sorted(required)}"]
                for exp in experiments:
                    classified = set(_classify_scans_for_experiment(project_id, exp).keys())
                    blocks.append(
                        f"- experiment={_experiment_label(exp)!r} date={_experiment_date(exp)!r} "
                        f"classified={sorted(classified)} missing={sorted(required - classified)}"
                    )
                    blocks.append(_debug_experiment_scans(project_id, exp))
                raise LookupError(
                    f"XNAT subject {subj_label!r}: missing sequences after scanning all experiments: {missing}\n"
                    + "\n".join(blocks)
                )

            # Batch: for multi-experiment subjects, use the newest experiment date among those used.
            used_exps = [picked[s][0] for s in sorted(picked.keys())]
            used_dates = [d for d in (_experiment_date(e) for e in used_exps) if d is not None]
            batch_date = max(used_dates) if used_dates else None
            subj_batch = batch_from_session_date(batch_date) if batch_auto else str(batch_in)
            subj_out: dict[str, Path] = {}
            subject_root = Path(dicom_root) / subj_batch / subj_label
            for seq in sorted(required):
                target_dir = subject_root / seq
                if req.skip_existing and target_dir.exists():
                    try:
                        if any(target_dir.iterdir()):
                            subj_out[seq] = target_dir
                            continue
                    except OSError:
                        pass
                exp, scan = picked[seq]
                log.info(
                    "XNAT download %s | %s | %s -> %s",
                    subj_label,
                    _experiment_label(exp),
                    seq,
                    target_dir,
                )
                download_scan_dicoms(scan, target_dir)
                subj_out[seq] = target_dir
            out[subj_label] = (subj_batch, subj_out)
    return out


__all__ = [
    "XnatPesaFatRequest",
    "batch_from_session_date",
    "download_pesa_fat_dicoms_from_xnat",
]

