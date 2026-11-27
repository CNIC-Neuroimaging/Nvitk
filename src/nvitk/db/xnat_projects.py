"""XNAT project registry and project-specific scan classifiers.

The projects, their classifiers and the visit labels come from the XNAT rules in force
(:mod:`nvitk.db.config.xnat_rules`: ``xnat_rules.json``, or the packaged empty default). The names below are kept for the code that
imports them; they are read from those rules when this module loads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from nvitk.db.config.xnat_rules import load_xnat_rules

#: A classifier name from the rules (``"pesabrain"``, ``"ia_pet_v5"``... out of the box).
ClassifierName = str


def _visits() -> dict[str, str]:
    """Named visit labels from the dataset config (``plaque``, ``pesa_brain``, ``ia_pet_v5``...)."""
    try:
        from nvitk.db.config.dataset_config import load_dataset_config

        return dict(load_dataset_config().visits)
    except Exception:  # noqa: BLE001 (an unreadable dataset config must not break XNAT access)
        return {}


_RULES = load_xnat_rules()

VISIT_PLAQUE = _visits().get("plaque", "3")
VISIT_PESA_BRAIN = _visits().get("pesa_brain", _RULES.default_visit_label or "4")
VISIT_IA_PET_V5 = _visits().get("ia_pet_v5", "5")

VISIT_LABEL_BY_PROJECT: dict[str, str] = dict(_RULES.visit_labels)


def visit_label_for_project(project_id: str) -> str:
    """Default visit label for imaging sessions tied to an XNAT (or local) project."""
    return load_xnat_rules().visit_label(project_id)


@dataclass(frozen=True)
class XnatProjectSpec:
    """Metadata for a mapped XNAT project."""

    project_id: str
    display_name: str
    cohort_id: str
    classifier: ClassifierName
    default_sequences: tuple[str, ...]
    #: Scans can only be classified with their whole experiment in view (Dixon stations).
    experiment_level: bool = False


def _project_specs() -> dict[str, XnatProjectSpec]:
    rules = load_xnat_rules()
    out: dict[str, XnatProjectSpec] = {}
    for pid, project in rules.projects.items():
        classifier = rules.classifiers.get(project.classifier)
        out[pid] = XnatProjectSpec(
            project_id=project.project_id,
            display_name=project.display_name,
            cohort_id=project.cohort_id,
            classifier=project.classifier,
            default_sequences=project.default_sequences,
            experiment_level=bool(classifier is not None and classifier.experiment_level),
        )
    return out


XNAT_PROJECTS: dict[str, XnatProjectSpec] = _project_specs()

PESA_BRAIN_SEQUENCES: tuple[str, ...] = (
    XNAT_PROJECTS["PESA_Brain"].default_sequences if "PESA_Brain" in XNAT_PROJECTS else ()
)
IA_PET_V5_SEQUENCES: tuple[str, ...] = (
    XNAT_PROJECTS["IA_PET_V5"].default_sequences if "IA_PET_V5" in XNAT_PROJECTS else ()
)


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
    """Union of every registered project's default sequence keys (deduplicated, stable order)."""
    merged: list[str] = []
    seen: set[str] = set()
    for spec in XNAT_PROJECTS.values():
        for seq in spec.default_sequences:
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
    Classify a single PESA-Health IA_PET_V5 scan (Dixon MR + CT/PET), with the rules' ``ia_pet_v5``
    classifier.

    Dixon scans return a candidate marker (``dixon=True``); region assignment
    requires :func:`classify_experiment_ia_pet_v5` at experiment scope.
    """
    del scan_id, experiment_label
    return load_xnat_rules().classifiers["ia_pet_v5"].classify(series_description, quality)


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
    return load_xnat_rules().classifiers["ia_pet_v5"].classify_experiment(scans)


def classify_experiment(
    project_id: str,
    scans: list[Any],
    *,
    experiment_label: str | None = None,
) -> list[tuple[Any, str, str, str, dict[str, Any]]]:
    """Classify all scans of one experiment of *project_id* with its classifier, experiment steps
    (Dixon stations...) included. Same tuples as :func:`classify_experiment_ia_pet_v5`."""
    del experiment_label
    return load_xnat_rules().classifier_for(project_id).classify_experiment(scans)


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
    """Return the scan-classification function for *project_id*: its classifier in the XNAT rules,
    as ``fn(series_description, quality=None, *, scan_id=None, experiment_label=None)``."""
    get_xnat_project(project_id)
    classifier = load_xnat_rules().classifier_for(project_id)

    def _classify(
        series_description: str | None,
        quality: str | None = None,
        *,
        scan_id: str | None = None,
        experiment_label: str | None = None,
    ) -> dict[str, Any] | None:
        """Classify one scan by series description/quality (scan_id/experiment_label are accepted
        for a uniform signature; rules match on the description and quality)."""
        del scan_id, experiment_label
        return classifier.classify(series_description, quality)

    return _classify


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
    "classify_experiment",
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
