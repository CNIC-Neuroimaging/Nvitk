"""XNAT project registry and project-specific scan classifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

ClassifierName = Literal["pesabrain", "ia_pet_v5"]

_REGION_BY_SCAN_LAST_DIGIT: dict[str, str] = {"1": "HEAD", "2": "THORAX", "3": "LEGS"}


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
    return sorted(XNAT_PROJECTS.keys())


def get_xnat_project(project_id: str) -> XnatProjectSpec:
    key = str(project_id).strip()
    if key not in XNAT_PROJECTS:
        known = ", ".join(list_xnat_project_ids())
        raise KeyError(f"Unknown XNAT project {key!r}; known: {known}")
    return XNAT_PROJECTS[key]


def default_sequences_for_project(project_id: str) -> tuple[str, ...]:
    return get_xnat_project(project_id).default_sequences


def _dixon_region_from_scan_id(scan_id: str | None) -> str | None:
    if not scan_id:
        return None
    digits = re.sub(r"\D", "", str(scan_id))
    if not digits:
        return None
    return _REGION_BY_SCAN_LAST_DIGIT.get(digits[-1])


def classify_scan_ia_pet_v5(
    series_description: str | None,
    quality: str | None = None,
    *,
    scan_id: str | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any] | None:
    """
    Classify PESA-Fat IA_PET_V5 scans (Dixon MR + CT/PET).

    Dixon regions use the last digit of ``scan_id`` (401→HEAD, 402→THORAX, 403→LEGS).
    """
    del experiment_label  # reserved for future session-aware rules
    if quality is not None and str(quality).strip().lower() != "usable":
        return None

    description = series_description or ""

    if re.search(r"mdixon|dixon.?quant", description, flags=re.IGNORECASE):
        region = _dixon_region_from_scan_id(scan_id)
        if region is None:
            return None
        return {
            "modality": "mr",
            "orientation": None,
            "sequence": f"DIXON_{region}",
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
