"""XNAT project registry and project-specific scan classifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from nvitk.pipes.pesa_fat.common.dixon_regions import classify_dixon_scans_by_slice_location

ClassifierName = Literal["pesabrain", "ia_pet_v5"]

VISIT_PLAQUE = "3"
VISIT_PESA_BRAIN = "4"
VISIT_IA_PET_V5 = "5"

VISIT_LABEL_BY_PROJECT: dict[str, str] = {
    "PESA_Brain": VISIT_PESA_BRAIN,
    "IA_PET_V5": VISIT_IA_PET_V5,
    "local_db": VISIT_PESA_BRAIN,
}


def visit_label_for_project(project_id: str) -> str:
    """Default visit label for imaging sessions tied to an XNAT (or local) project."""
    return VISIT_LABEL_BY_PROJECT.get(str(project_id).strip(), VISIT_PESA_BRAIN)


@dataclass(frozen=True)
class XnatProjectSpec:
    """Metadata for a mapped XNAT project."""

    project_id: str
    display_name: str
    cohort_id: str
    classifier: ClassifierName
    default_sequences: tuple[str, ...]


PESA_BRAIN_SEQUENCES: tuple[str, ...] = (
    "TOF",
    "4DFLOW_AP",
    "4DFLOW_RL",
    "4DFLOW_FH",
    "VWI_BB",
    "3D_T1",
    "3D_T2_HR",
    "3D_FLAIR",
    "SWI_QSM",
    "QSM",
    "CAROTID_QF",
    "RESTING_STATE_MB",
)

IA_PET_V5_SEQUENCES: tuple[str, ...] = (
    "DIXON_HEAD",
    "DIXON_THORAX",
    "DIXON_LEGS",
    "CT",
    "PET",
)

XNAT_PROJECTS: dict[str, XnatProjectSpec] = {
    "PESA_Brain": XnatProjectSpec(
        project_id="PESA_Brain",
        display_name="PESA Brain",
        cohort_id="PESA-Brain",
        classifier="pesabrain",
        default_sequences=PESA_BRAIN_SEQUENCES,
    ),
    "IA_PET_V5": XnatProjectSpec(
        project_id="IA_PET_V5",
        display_name="IA PET v5 (PESA-Fat)",
        cohort_id="IA-PET-V5",
        classifier="ia_pet_v5",
        default_sequences=IA_PET_V5_SEQUENCES,
    ),
}


def list_xnat_project_ids() -> list[str]:
    """Sorted ids of every registered :data:`XNAT_PROJECTS` entry."""
    return sorted(XNAT_PROJECTS.keys())


def _normalize_cohort_token(token: str) -> str:
    """Lower-case *token* and strip everything but alphanumerics, for loose project/cohort matching."""
    return re.sub(r"[^0-9a-z]+", "", str(token).strip().lower())


def resolve_xnat_project_cohort_token(subjects: str | None) -> str | None:
    """Return XNAT ``project_id`` when *subjects* is a single cohort/project alias.

    Recognizes ``PESA-Brain`` / ``PESA_Brain``, ``IA-PET-V5`` / ``IA_PET_V5``, and
    registered :data:`XNAT_PROJECTS` keys.
    """
    if subjects is None:
        return None
    from nvitk.db.xnat import parse_subject_tokens

    tokens = parse_subject_tokens(subjects)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if token in XNAT_PROJECTS:
        return token
    key = _normalize_cohort_token(token)
    for spec in XNAT_PROJECTS.values():
        if key in {
            _normalize_cohort_token(spec.project_id),
            _normalize_cohort_token(spec.cohort_id),
            _normalize_cohort_token(spec.display_name),
        }:
            return spec.project_id
    return None


def get_xnat_project(project_id: str) -> XnatProjectSpec:
    """Return the registered :class:`XnatProjectSpec` for *project_id*; raises ``KeyError`` (listing
    known ids) if unregistered."""
    key = str(project_id).strip()
    if key not in XNAT_PROJECTS:
        known = ", ".join(list_xnat_project_ids())
        raise KeyError(f"Unknown XNAT project {key!r}; known: {known}")
    return XNAT_PROJECTS[key]


def default_sequences_for_project(project_id: str) -> tuple[str, ...]:
    """Default sequence keys to sync for *project_id* (from its registered :class:`XnatProjectSpec`)."""
    return get_xnat_project(project_id).default_sequences


def sequences_csv(sequences: Iterable[str]) -> str:
    """Comma-separated sequence keys for :func:`~nvitk.db.xnat.sync_xnat_project` filters."""
    return ",".join(sequences)


def build_default_xnat_sequences_csv() -> str:
    """Union of PESA_Brain and IA_PET_V5 default sequence keys (deduplicated, stable order)."""
    merged: list[str] = []
    seen: set[str] = set()
    for seq in (*PESA_BRAIN_SEQUENCES, *IA_PET_V5_SEQUENCES):
        if seq not in seen:
            seen.add(seq)
            merged.append(seq)
    return sequences_csv(merged)


def _scan_attr(scan: Any, *names: str) -> str:
    """Return the first non-``None`` attribute among *names* on *scan* as a stripped string, or ``""``."""
    for name in names:
        if hasattr(scan, name):
            value = getattr(scan, name)
            if value is not None:
                return str(value).strip()
    return ""


def classify_scan_ia_pet_v5(
    series_description: str | None,
    quality: str | None = None,
    *,
    scan_id: str | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any] | None:
    """
    Classify a single PESA-Fat IA_PET_V5 scan (Dixon MR + CT/PET).

    Dixon scans return a candidate marker (``dixon=True``); region assignment
    requires :func:`classify_experiment_ia_pet_v5` at experiment scope.
    """
    del scan_id, experiment_label
    if quality is not None and str(quality).strip().lower() != "usable":
        return None

    description = series_description or ""

    if re.search(r"mdixon|dixon.?quant", description, flags=re.IGNORECASE):
        return {
            "modality": "mr",
            "orientation": None,
            "dixon": True,
        }

    if re.search(r"body.*low\s*dose\s*ct|low\s*dose\s*ct", description, flags=re.IGNORECASE):
        return {
            "modality": "ct",
            "orientation": None,
            "sequence": "CT",
        }

    if re.search(
        r"detailwb.*ctac|ctac.*vascular|\[detailwb_ctac\]",
        description,
        flags=re.IGNORECASE,
    ):
        return {
            "modality": "pt",
            "orientation": None,
            "sequence": "PET",
        }

    return None


def classify_experiment_ia_pet_v5(
    scans: list[Any],
    *,
    experiment_label: str | None = None,
) -> list[tuple[Any, str, str, str, dict[str, Any]]]:
    """Classify all scans in one IA_PET_V5 experiment.

    Returns ``(scan, scan_id, series_description, quality, classification)`` tuples
    with one entry per sequence label (first scan wins for duplicates).
    """
    del experiment_label
    dixon_candidates: list[tuple[Any, str]] = []
    preliminary: list[tuple[Any, str, str, str, dict[str, Any]]] = []

    for scan in scans:
        scan_id = _scan_attr(scan, "id", "label", "name")
        desc = _scan_attr(scan, "series_description", "type", "label")
        quality = _scan_attr(scan, "quality")
        cls = classify_scan_ia_pet_v5(desc, quality or None, scan_id=scan_id)
        if cls is None:
            continue
        if cls.get("dixon"):
            dixon_candidates.append((scan, scan_id))
            continue
        preliminary.append((scan, scan_id, desc, quality, cls))

    results: list[tuple[Any, str, str, str, dict[str, Any]]] = list(preliminary)

    if dixon_candidates:
        scan_ids = [scan_id for _, scan_id in dixon_candidates]
        scans_by_id = {scan_id: scan for scan, scan_id in dixon_candidates}
        seq_by_scan_id = classify_dixon_scans_by_slice_location(scan_ids, scans_by_id)
        for scan, scan_id in dixon_candidates:
            sequence = seq_by_scan_id.get(scan_id)
            if not sequence:
                continue
            desc = _scan_attr(scan, "series_description", "type", "label")
            quality = _scan_attr(scan, "quality")
            results.append(
                (
                    scan,
                    scan_id,
                    desc,
                    quality,
                    {
                        "modality": "mr",
                        "orientation": None,
                        "sequence": sequence,
                    },
                )
            )

    seen_sequences: set[str] = set()
    unique: list[tuple[Any, str, str, str, dict[str, Any]]] = []
    for item in results:
        sequence = str(item[4].get("sequence") or "").strip()
        if not sequence or sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        unique.append(item)
    return unique


def session_modality_from_classifications(classifications: list[dict[str, Any]]) -> str:
    """Infer session-level modality label from classified scans in one experiment."""
    modalities = {str(c.get("modality") or "").lower() for c in classifications}
    if "pt" in modalities:
        return "pet"
    if "ct" in modalities:
        return "ct"
    if modalities.intersection({"4dflow", "tof", "mri", "mr", "mra", "fmri", "flair", "t1", "t2", "qsm", "swi"}):
        return "mr"
    return "mr"


def get_scan_classifier(project_id: str) -> Callable[..., dict[str, Any] | None]:
    """Return the scan-classification function registered for *project_id*'s ``classifier`` (currently
    :func:`classify_scan_ia_pet_v5` or a PESA-Brain wrapper around :func:`~nvitk.db.xnat.classify_scan`)."""
    spec = get_xnat_project(project_id)
    if spec.classifier == "ia_pet_v5":
        return classify_scan_ia_pet_v5
    from .xnat import classify_scan

    def _pesabrain(
        series_description: str | None,
        quality: str | None = None,
        *,
        scan_id: str | None = None,
        experiment_label: str | None = None,
    ) -> dict[str, Any] | None:
        """Classify a PESA-Brain scan by series description/quality (ignoring scan_id/experiment_label,
        which the ia_pet_v5 classifier needs but this project's classifier does not)."""
        del scan_id, experiment_label
        return classify_scan(series_description, quality)

    return _pesabrain


def classify_scan_for_project(
    project_id: str,
    series_description: str | None,
    quality: str | None = None,
    *,
    scan_id: str | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any] | None:
    """Dispatch scan classification for a registered XNAT project."""
    fn = get_scan_classifier(project_id)
    return fn(
        series_description,
        quality,
        scan_id=scan_id,
        experiment_label=experiment_label,
    )


__all__ = [
    "IA_PET_V5_SEQUENCES",
    "PESA_BRAIN_SEQUENCES",
    "VISIT_IA_PET_V5",
    "VISIT_LABEL_BY_PROJECT",
    "VISIT_PESA_BRAIN",
    "VISIT_PLAQUE",
    "XNAT_PROJECTS",
    "XnatProjectSpec",
    "build_default_xnat_sequences_csv",
    "classify_experiment_ia_pet_v5",
    "classify_scan_for_project",
    "classify_scan_ia_pet_v5",
    "default_sequences_for_project",
    "get_scan_classifier",
    "get_xnat_project",
    "list_xnat_project_ids",
    "resolve_xnat_project_cohort_token",
    "sequences_csv",
    "session_modality_from_classifications",
    "visit_label_for_project",
]
