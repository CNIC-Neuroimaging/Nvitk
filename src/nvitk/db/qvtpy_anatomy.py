"""
Manual anatomical-configuration annotations for the qvtpy (4D-Flow) pipeline.

Description
-----------
Two subject-level variables that no pipeline stage can measure: the
Circle-of-Willis configuration (``cow_config``) and the venous drainage
configuration (``venous_config``). A reviewer picks them in the QC measurements
panel while looking at the subject's vessels, and marking the subject as revised
publishes them alongside the automatic measurements.

Storage
-------
Rows land in ``image_measurements`` under the qvtpy pipeline id, as *categorical*
values (``value_text``, ``value_num`` empty) with **no** ``region_id``: they
describe the whole subject, not one vessel. That is what lets the Statmodels tool
offer them as subject-level factors next to the clinical covariates — see
:func:`nvitk.stats._statmodels_frames.subject_image_annotation_entries`.

The vocabularies are closed. A value outside a variable's choices is rejected
rather than stored, because a level created by a typo does not announce itself in
a model summary — it silently splits a cell in two.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_image_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.repo import DataRepo
from nvitk.pipes.qvtpy.common.db_publish import (
    QVTPY_MODALITY,
    QVTPY_PIPELINE_ID,
    QVTPY_PIPELINE_NAME,
    resolve_repo,
)

log = Logger()

#: Source tag written on every published row, so a manual annotation is always
#: distinguishable from a stage-computed measurement.
ANATOMY_SOURCE_FILE = "qc_anatomy_review"
ANATOMY_SOURCE_SHEET = "manual"

#: Upsert key: subject-level rows, so ``region_id`` / ``frame_index`` are both empty and
#: re-reviewing a subject overwrites its previous answer instead of appending a second one.
ANATOMY_UPSERT_KEY = ["subject_uid", "pipeline_id", "variable_id", "region_id", "frame_index"]


# ──────────────────────────────────────────────────────────────────────────────
# Vocabularies
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AnatomyConfigVariable:
    """One manually annotated anatomical configuration and its closed vocabulary."""

    variable_id: str
    label: str
    region_label: str
    description: str
    #: ``(code, human label)`` in the order the QC dropdown offers them.
    choices: tuple[tuple[str, str], ...]

    @property
    def codes(self) -> tuple[str, ...]:
        """The allowed values, in dropdown order."""
        return tuple(code for code, _label in self.choices)

    def label_for(self, code: str) -> str:
        """Human label for *code*, falling back to the code itself when it is unknown."""
        wanted = str(code).strip()
        for candidate, label in self.choices:
            if candidate == wanted:
                return label
        return wanted


COW_CONFIG = AnatomyConfigVariable(
    variable_id="cow_config",
    label="Circle-of-Willis configuration (manual)",
    region_label="Circle of Willis",
    description=(
        "Arterial anatomy of the circle of Willis: whether the ring is complete, which half is "
        "incomplete, and whether a posterior cerebral artery is of fetal origin."
    ),
    choices=(
        ("complete", "Complete circle"),
        ("incomplete_anterior", "Incomplete anterior (A1 / AComm)"),
        ("incomplete_posterior", "Incomplete posterior (P1 / PComm)"),
        ("incomplete_both", "Incomplete anterior and posterior"),
        ("fetal_pca_left", "Fetal PCA — left"),
        ("fetal_pca_right", "Fetal PCA — right"),
        ("fetal_pca_bilateral", "Fetal PCA — bilateral"),
        ("other", "Other variant"),
        ("undetermined", "Undetermined / not assessable"),
    ),
)

VENOUS_CONFIG = AnatomyConfigVariable(
    variable_id="venous_config",
    label="Venous drainage configuration (manual)",
    region_label="Venous sinuses",
    description=(
        "Dural venous drainage pattern: which transverse sinus dominates, or which one is "
        "hypoplastic / absent."
    ),
    choices=(
        ("right_dominant", "Right transverse sinus dominant"),
        ("left_dominant", "Left transverse sinus dominant"),
        ("codominant", "Codominant transverse sinuses"),
        ("left_ts_hypoplastic", "Left transverse sinus hypoplastic / absent"),
        ("right_ts_hypoplastic", "Right transverse sinus hypoplastic / absent"),
        ("other", "Other variant"),
        ("undetermined", "Undetermined / not assessable"),
    ),
)

#: Every manual configuration variable, keyed by ``variable_id`` in QC-table order.
ANATOMY_CONFIG_VARIABLES: dict[str, AnatomyConfigVariable] = {
    var.variable_id: var for var in (COW_CONFIG, VENOUS_CONFIG)
}


def anatomy_config_variable_ids() -> tuple[str, ...]:
    """The manual configuration ``variable_id``s, in QC-table order."""
    return tuple(ANATOMY_CONFIG_VARIABLES)


def normalize_anatomy_value(variable_id: str, value: Any) -> str:
    """
    Canonical code for *value*, or ``""`` when nothing was chosen.

    Case and surrounding whitespace are forgiven (a code pasted from a report, a label copied from
    the dropdown); anything else raises, because an unlisted level is a data error, not a new
    category.
    """
    var = ANATOMY_CONFIG_VARIABLES.get(str(variable_id).strip())
    if var is None:
        raise ValueError(f"Unknown anatomy configuration variable {variable_id!r}")
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for code in var.codes:
        if lowered == code.lower():
            return code
    raise ValueError(
        f"{var.variable_id}={text!r} is not one of {', '.join(var.codes)}"
    )


def normalize_anatomy_values(values: Mapping[str, Any]) -> dict[str, str]:
    """Normalize a ``{variable_id: value}`` mapping, dropping the entries left blank."""
    out: dict[str, str] = {}
    for variable_id, value in dict(values).items():
        code = normalize_anatomy_value(variable_id, value)
        if code:
            out[str(variable_id).strip()] = code
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Catalog registration + publishing
# ──────────────────────────────────────────────────────────────────────────────
def _image_spec(var: AnatomyConfigVariable, *, pipeline_id: str) -> DerivedImageMeasurementSpec:
    """Row metadata for one configuration variable (categorical → ``value_text``)."""
    return DerivedImageMeasurementSpec(
        variable_id=var.variable_id,
        modality=QVTPY_MODALITY,
        pipeline_id=str(pipeline_id),
        pipeline_name=QVTPY_PIPELINE_NAME,
        source_file=ANATOMY_SOURCE_FILE,
        source_sheet=ANATOMY_SOURCE_SHEET,
        source_column=var.variable_id,
        value_column="value_text",
        value_kind="categorical",
        # Set by a human who looked at the subject — reviewed by construction.
        qc_status="OK",
    )


def anatomy_variable_registration(var: AnatomyConfigVariable) -> DerivedVariableRegistration:
    """
    Catalog entry for one configuration variable.

    ``scope="subject"`` is what marks it as an image variable carrying one value per subject rather
    than one per vessel; the Statmodels covariate list keys off that flag. ``choices`` travels with
    the entry so a consumer can render the vocabulary without importing this module.
    """
    return DerivedVariableRegistration(
        variable_id=var.variable_id,
        domain="image",
        table="image_measurements",
        label=var.label,
        source_column=var.variable_id,
        source_file=ANATOMY_SOURCE_FILE,
        source_sheet=ANATOMY_SOURCE_SHEET,
        modality=QVTPY_MODALITY,
        value_kind="categorical",
        extra={
            "scope": "subject",
            "description": var.description,
            "choices": list(var.codes),
        },
    )


def register_anatomy_config_variables(repo: DataRepo | None = None) -> int:
    """Register both configuration variables in the dataset catalog; returns how many were written."""
    repo = resolve_repo(repo)
    entries = [
        anatomy_variable_registration(var).to_catalog_entry()
        for var in ANATOMY_CONFIG_VARIABLES.values()
    ]
    repo.register_variables(entries)
    return len(entries)


def publish_anatomy_configs(
    *,
    subject_uid: str,
    values: Mapping[str, Any],
    repo: DataRepo | None = None,
    pipeline_id: str = QVTPY_PIPELINE_ID,
    build_sqlite_index: bool = True,
) -> dict[str, str]:
    """
    Upsert one subject's manual configurations into ``image_measurements``.

    Parameters
    ----------
    subject_uid
        Subject the annotations belong to.
    values
        ``{variable_id: code}``; blank entries are skipped, unknown codes raise (see
        :func:`normalize_anatomy_value`). Skipping rather than writing an empty row keeps
        "not annotated yet" distinguishable from ``undetermined``, which is a reviewer's verdict.

    Returns
    -------
    dict
        The ``{variable_id: code}`` actually published (empty when nothing was chosen).
    """
    subject = str(subject_uid).strip()
    if not subject:
        raise ValueError("subject_uid is required")
    chosen = normalize_anatomy_values(values)
    if not chosen:
        return {}

    repo = resolve_repo(repo)
    published: dict[str, str] = {}
    for variable_id, code in chosen.items():
        var = ANATOMY_CONFIG_VARIABLES[variable_id]
        spec = _image_spec(var, pipeline_id=pipeline_id)
        agg = pd.DataFrame({"subject_uid": [subject], "value_text": [code]})
        rows = build_image_measurement_rows(agg, spec)
        publish_derived_measurements(
            repo,
            rows,
            table="image_measurements",
            register=anatomy_variable_registration(var),
            provenance={
                "importer": ANATOMY_SOURCE_FILE,
                "subject_uid": subject,
                "pipeline_id": str(pipeline_id),
                "variable_id": variable_id,
            },
            upsert_key_columns=ANATOMY_UPSERT_KEY,
            # Rebuilt once by the caller after the last variable — each rebuild re-reads the
            # whole table, so doing it per variable doubles the cost of a two-row write.
            build_sqlite_index=False,
        )
        published[variable_id] = code

    if build_sqlite_index:
        repo.build_sqlite_index(tables=["image_measurements"])
    log.info(
        "Anatomy configs for %s: %s",
        subject,
        ", ".join(f"{k}={v}" for k, v in sorted(published.items())),
    )
    return published


def load_anatomy_configs(
    subject_uid: str,
    *,
    repo: DataRepo | None = None,
    pipeline_id: str = QVTPY_PIPELINE_ID,
) -> dict[str, str]:
    """
    Configurations already stored for *subject_uid*, as ``{variable_id: code}``.

    Never raises: no dataset, no table, or an unreadable one all mean "nothing annotated yet",
    which is the normal state for a subject reaching QC for the first time and must not stop the
    panel from opening.
    """
    subject = str(subject_uid).strip()
    if not subject:
        return {}
    try:
        if repo is None:
            repo = resolve_repo(None)
        if not repo.catalog.table_exists("image_measurements"):
            return {}
        rows = repo.get(
            "image_measurements",
            cohort_id=False,
            filters={
                "subject_uid": subject,
                "variable_id": list(anatomy_config_variable_ids()),
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("Anatomy configs unavailable for %s (%s).", subject, exc)
        return {}
    if rows is None or rows.empty or "variable_id" not in rows.columns:
        return {}
    if pipeline_id and "pipeline_id" in rows.columns:
        rows = rows.loc[rows["pipeline_id"].astype(str) == str(pipeline_id)]

    out: dict[str, str] = {}
    for variable_id, group in rows.groupby(rows["variable_id"].astype(str)):
        for value in group.get("value_text", pd.Series(dtype="string")):
            try:
                code = normalize_anatomy_value(variable_id, value)
            except ValueError:
                # A stored value outside today's vocabulary (an edited list, a hand-written row):
                # report it rather than silently pre-selecting nothing.
                log.warning(
                    "Anatomy config %s=%r for %s is not in the current vocabulary.",
                    variable_id, value, subject,
                )
                continue
            if code:
                out[str(variable_id)] = code
                break
    return out


__all__ = [
    "ANATOMY_CONFIG_VARIABLES",
    "ANATOMY_SOURCE_FILE",
    "ANATOMY_SOURCE_SHEET",
    "ANATOMY_UPSERT_KEY",
    "AnatomyConfigVariable",
    "COW_CONFIG",
    "VENOUS_CONFIG",
    "anatomy_config_variable_ids",
    "anatomy_variable_registration",
    "load_anatomy_configs",
    "normalize_anatomy_value",
    "normalize_anatomy_values",
    "publish_anatomy_configs",
    "register_anatomy_config_variables",
]
