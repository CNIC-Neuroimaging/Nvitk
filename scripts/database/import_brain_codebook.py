#!/usr/bin/env python3
"""Import the PESA-Brain neuroradiology read (codebook ``Casos`` sheet) into ``clinical_measurements``.

Description
-----------
One row per subject, keyed by ``Codi Sub.`` (= ``subject_uid``) and filed under visit 4, the
PESA-Brain MRI visit. Covers the incidental lesions (``lession_*``), the atrophy and Fazekas
scales (``brain_atrophy_*``), Circle of Willis anatomy and venous drainage (``cow_*``), and
lacunes, microbleeds, strategic infarcts and enlarged perivascular spaces.

Harmonization
-------------
The sheet stores answers as the dropdown label — ``"1: Si"``, ``"2: moderada gyrus"`` — except
where a rater typed the bare code (``0``). Each cell is reduced to its code (the part before the
colon) and mapped through the variable's table in :data:`ITEMS`. A code with no entry aborts the
run and lists the values, so a new dropdown option cannot slip in as missing. Codes follow the
workbook's own ``DESPLEGABLES`` sheet, which is what the data uses where it and the variable
descriptions disagree (e.g. ``Enlarged_perivasc_space_loc``: 0 = none, 1 = BG, 2 = CSO,
3 = midbrain, 4 = BG + CSO).

The output takes one of four shapes:

* **yes/no findings** -> ``0``/``1`` in ``value_num``; "indeterminate" -> missing.
* **ordinal scales and counts** (atrophy, Fazekas, perivascular-space degree, lacunes,
  microbleeds) -> the grade or count in ``value_num``; "No Eval" -> missing.
* **nominal descriptors** (side, CoW anatomy, vessel, venous drainage) -> English snake_case
  labels in ``value_text`` (``value_kind = "categorical"``), e.g. ``fetal_right``,
  ``jugular+occipital``. A location variable reads ``"none"`` only where the rater chose "none";
  where they left it blank it stays missing.
* **free text** (``Other_findings``, ``Other_comments``) -> the same text with one vocabulary:
  typos fixed, Spanish translated, ``R``/``L`` spelled out, every microhemorrhage spelling made
  ``microbleed`` (see :func:`harmonize_text`). The dry run prints every raw -> harmonized pair.

``Aneurysm`` records the vessel of the aneurysm, with 0 for none, so it is split in two:
``cow_aneurysm`` (0/1) for modelling and ``cow_aneurysm_location`` for the vessel.

Subjects listed more than once keep their last row; the dry run lists where the reads disagree.

Overwrite
---------
Any row already at the same ``(subject_uid, visit_id, variable_id)`` is replaced, whichever file
it came from; a blank cell never removes an existing value.

Default mode is dry-run.  Use ``--write`` to publish rows.

Examples::

    python scripts/database/import_brain_codebook.py --source /path/to/AllCodebooks.xlsx
    python scripts/database/import_brain_codebook.py --source ... --write
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping

import click
import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
)
from nvitk.db.importers import read_tabular_source

log = Logger()

DEFAULT_SOURCE = "/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/PESABrain_AllCodebooks_PESA-H_6_JM-3_06092026_JM.xlsx"
DEFAULT_SHEET = "Casos"
SUBJECT_COLUMN = "Codi Sub."
VISIT_ID = "4"
SOURCE_BATCH_ID = "import_brain_codebook"
TABLE = "clinical_measurements"
#: What "the same measurement" means for the overwrite: the source columns are left out on purpose.
REPLACE_KEY = ["subject_uid", "visit_id", "variable_id"]


# ---------------------------------------------------------------------------
# Code tables — keys are the code before the colon; ``None`` = a known answer stored as missing
# ---------------------------------------------------------------------------
NO_YES = {"0": 0.0, "1": 1.0}
NO_YES_INDETERMINATE = {**NO_YES, "3": None}
#: Global cortical atrophy, frontal, Koedam parietal and Fazekas: 0–3, 4 = not evaluable.
GRADE_0_3 = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": None}
#: Medial temporal atrophy: 0–4, 5 = not evaluable.
GRADE_0_4 = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": None}
#: Perivascular-space degree: 0 = none, 1 = 1–10, 2 = 11–20, 3 = 21–40, 4 = >40.
EPVS_DEGREE = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0}
#: Side with the greater atrophy.
SIDE = {"0": "symmetric", "1": "right", "2": "left"}
VESSEL = {
    "1": "ica_right", "2": "ica_left", "3": "mca_right", "4": "mca_left",
    "5": "aca_right", "6": "aca_left", "7": "pca_right", "8": "pca_left",
    "9": "basilar", "10": "vertebral_right", "11": "vertebral_left",
}
#: Cervical venous drainage letters, in any order: J = jugular, O = occipital, S = spinal.
_VEINS = {"J": "jugular", "O": "occipital", "S": "spinal"}
DRAINAGE = {
    "N": "none",
    **{
        "".join(p): "+".join(_VEINS[v] for v in "JOS" if v in p)
        for n in range(1, 4)
        for p in permutations("JOS", n)
    },
}


@dataclass(frozen=True)
class Item:
    """One output variable: where it comes from, what it is called, and how cells map to values."""

    source: str
    variable_id: str
    label: str
    #: code -> value; ``None`` for a count (read as a number) or free text.
    codes: Mapping[str, float | str | None] | None = None
    text: bool = False

    @property
    def value_kind(self) -> str:
        if self.text:
            return "text"
        if self.codes is not None and any(isinstance(v, str) for v in self.codes.values()):
            return "categorical"
        return "float"


ITEMS: tuple[Item, ...] = (
    # Incidental lesions
    Item("WB_lesions_MS", "lession_wb_ms", "Lesions suggestive of multiple sclerosis (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Cortical_Infarct", "lession_cortical_infarct", "Cortical infarct (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Arachnoidal_cyst", "lession_arachnoidal_cyst", "Arachnoid cyst (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Extraparenchymal_tumor", "lession_extraparenchymal_tumor", "Meningeal or extraparenchymal tumour (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Arterio_venous_Malformation", "lession_av_malformation", "Vascular malformation: cavernoma, AVM or fistula (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Brain_Hematoma", "lession_brain_hematoma", "Acute or chronic brain hematoma (0 = no, 1 = yes)", NO_YES_INDETERMINATE),
    Item("Other_findings", "lession_other", "Other findings (free text, harmonized)", text=True),
    # Atrophy and white matter
    Item("Brain_atrophy", "brain_atrophy_global", "Global cortical atrophy, GCA scale (0-3)", GRADE_0_3),
    Item("Brain_atrophy_Symm", "brain_atrophy_global_side", "Side with greater global atrophy", SIDE),
    Item("Brain_atrophy_frontal", "brain_atrophy_frontal", "Frontal atrophy, modified scale (0-3)", GRADE_0_3),
    Item("Brain_atrophy_frontal_Symm", "brain_atrophy_frontal_side", "Side with greater frontal atrophy", SIDE),
    Item("Brain_atrophy_temporal_R", "brain_atrophy_temporal_right", "Right medial temporal atrophy, MTA scale (0-4)", GRADE_0_4),
    Item("Brain_atrophy_temporal_L", "brain_atrophy_temporal_left", "Left medial temporal atrophy, MTA scale (0-4)", GRADE_0_4),
    Item("Brain_atrophy_Parietal", "brain_atrophy_parietal", "Parietal atrophy, Koedam scale (0-3)", GRADE_0_3),
    Item("Periventricular_Fazekas", "brain_atrophy_fazekas_periventricular", "Periventricular white-matter hyperintensities, Fazekas (0-3)", GRADE_0_3),
    # Circle of Willis and venous anatomy
    Item("CoW_configutacion", "cow_configuration", "Circle of Willis configuration", {"0": "complete", "1": "incomplete"}),
    Item(
        "CoW_PCA", "cow_pca", "Posterior cerebral artery origin",
        {"0": "basilar", "1": "fetal_right", "2": "fetal_left", "3": "fetal_bilateral", "4": "other", "5": None},
    ),
    Item(
        "CoW_A1", "cow_a1", "ACA A1 segment (normal = both present)",
        {"0": "normal", "1": "hypoplastic_right", "2": "hypoplastic_left", "3": "hypoplastic_bilateral", "4": "other", "5": None},
    ),
    Item("CoW_stenosis", "cow_stenosis", "Circle of Willis stenosis (0 = no, 1 = yes)", NO_YES),
    Item("CoW_stenosis_location", "cow_stenosis_location", "Vessel with the largest stenosis", VESSEL),
    Item(
        "CoW_stenosis_feat_1", "cow_stenosis_morphology", "Vessel-wall morphology at the largest stenosis",
        {"1": "focal_concentric", "2": "focal_eccentric", "3": "segmental_concentric", "4": "segmental_eccentric"},
    ),
    Item("Aneurysm", "cow_aneurysm", "Intracranial aneurysm (0 = no, 1 = yes)", {"0": 0.0, **{code: 1.0 for code in VESSEL}}),
    Item("Aneurysm", "cow_aneurysm_location", "Vessel of the intracranial aneurysm", {"0": "none", **VESSEL}),
    Item(
        "SWIp_Cov_veins", "cow_swi_medullary_veins", "Medullary veins of both centra semiovale on SWI",
        {"0": "continuous_homogeneous", "1": "continuous_heterogeneous", "2": "discontinuous_hypointense", "3": "absent"},
    ),
    Item(
        "Lateral_sinus_symm", "cow_lateral_sinus_dominance", "Lateral and sigmoid sinus dominance",
        {"0": "symmetric", "1": "right_dominant", "2": "left_dominant", "3": "symmetric_hypoplastic"},
    ),
    Item("Cervical_vein_drainage_r", "cow_cervical_drainage_right", "Right cervical venous drainage", DRAINAGE),
    Item("Cervical_vein_drainage_l", "cow_cervical_drainage_left", "Left cervical venous drainage", DRAINAGE),
    Item("Other_comments", "cow_other_comments", "Other vascular comments (free text, harmonized)", text=True),
    # Small-vessel disease
    Item("Lacunar_infarcts_number", "lacunar_infarcts_number", "Number of lacunar infarcts"),
    Item(
        "Lacunar_infarcts_location", "lacunar_infarcts_location", "Location of the largest lacunar infarct",
        {
            "0": "none", "1": "basal_ganglia_right", "2": "basal_ganglia_left", "3": "thalamus_right",
            "4": "thalamus_left", "5": "centrum_semiovale_right", "6": "centrum_semiovale_left",
            "7": "pons", "8": "cerebellum",
        },
    ),
    Item("Microbleeds", "microbleeds", "Number of microbleeds"),
    Item("Microbleed_lobar", "microbleed_lobar", "Number of lobar (cortical or juxtacortical) microbleeds"),
    Item("Microbleed_subcortical", "microbleed_subcortical", "Number of subcortical (basal ganglia) microbleeds"),
    Item("Subarach_siderosis", "subarach_siderosis", "Subarachnoid siderosis (0 = no, 1 = yes)", NO_YES),
    Item("Strategic_Infarction", "strategic_infarction", "Number of strategic infarcts"),
    Item(
        "Strategic_Infarct_location", "strategic_infarct_location", "Location of the strategic infarct",
        {
            "0": "pca_right", "1": "pca_left", "2": "mca_temporooccipital_right", "3": "mca_temporooccipital_left",
            "4": "mca_angular_right", "5": "mca_angular_left", "6": "small_cortical", "7": "thalamus_bilateral",
        },
    ),
    Item("Enlarged_perivasc_space", "enlarged_perivasc_space", "Enlarged perivascular spaces (0 = no, 1 = yes)", NO_YES),
    Item(
        "Enlarged_perivasc_space_loc", "enlarged_perivasc_space_loc", "Location of enlarged perivascular spaces",
        {"0": "none", "1": "basal_ganglia", "2": "centrum_semiovale", "3": "midbrain", "4": "basal_ganglia+centrum_semiovale"},
    ),
    Item(
        "Enlarged_perivasc_space_degree", "enlarged_perivasc_space_degree",
        "Enlarged perivascular spaces degree (0 = none, 1 = 1-10, 2 = 11-20, 3 = 21-40, 4 = >40)", EPVS_DEGREE,
    ),
)


# ---------------------------------------------------------------------------
# Free-text harmonization
# ---------------------------------------------------------------------------
#: ``(pattern, replacement)`` applied in order to the upper-cased text: Spanish phrases first,
#: then misspellings, then word order fixes that the later steps rely on.
TEXT_RULES: tuple[tuple[str, str], ...] = (
    (r"AUSENCIA PARCIAL (?:DEL )?SEPTUM PELLUCIDUM", "PARTIAL ABSENCE OF SEPTUM PELLUCIDUM"),
    (r"VENTRICULOMEGALIA SUPRATENTORIAL", "SUPRATENTORIAL VENTRICULOMEGALY"),
    (r"XANTOGRANULOMAS PLEXOS COROIDEOS", "CHOROID PLEXUS XANTHOGRANULOMAS"),
    (r"CALCIFICACIONES CORTICALES CEREBELO", "CEREBELLAR CORTICAL CALCIFICATIONS"),
    (
        r"RESTOS CIRUGIA PARIETAL D CON AREA DE MALACIA POSIBLE ANTECEDENTE MENINGIOMA",
        "RIGHT PARIETAL POST-SURGICAL CHANGES WITH MALACIA, POSSIBLE PRIOR MENINGIOMA",
    ),
    (r"FALTA SECUENCIA (?:FLUJO )?CERVICAL", "MISSING CERVICAL SEQUENCE"),
    (r"VARIANTE ARTERIA TRIGEMINAL VESTIGIAL", "PERSISTENT TRIGEMINAL ARTERY VARIANT"),
    (r"VENAS CENTRO OVAL TORTUOSAS", "TORTUOUS CENTRUM SEMIOVALE VEINS"),
    (r"ASIMETRIA VELOCIDAD YUGULARES COINCIDE SENO >", "ASYMMETRIC JUGULAR VELOCITY, MATCHING THE DOMINANT SINUS"),
    (r"EJEMPLO OCCIPITAL Y JO\b", "EXAMPLE CASE: OCCIPITAL AND JO DRAINAGE"),
    (r"NO PHC CERVICAL", "NO CERVICAL PHASE CONTRAST"),
    (r"CAVERNOMATOSIS MULTIPLE", "MULTIPLE CAVERNOMATOSIS"),
    (r"VALORAR FAMILIAR", "CONSIDER FAMILIAL FORM"),
    (r"\bDOBLE PCA LEFT\b|\bPCA LEFT DOUBLE\b", "DUPLICATED LEFT PCA"),
    (r"ECTOPIA CEREBELLAR AMIGDALA", "CEREBELLAR TONSILLAR ECTOPIA"),
    (r"\bSOLO\b", "ONLY"),
    (r"\bMULTIPLES\b", "MULTIPLE"),
    (r"\bPOSIBLE\b", "POSSIBLE"),
    # Misspellings and variants
    (r"\bMICRO-?(?:HEM\w*|BLEEDS?)\b", "MICROBLEED"),
    (r"ANEU+RI?YSM", "ANEURYSM"),
    (r"\bRIGTH\b", "RIGHT"),
    (r"\bCEREBELL+AR\b", "CEREBELLAR"),
    (r"\bFROTAL\b", "FRONTAL"),
    (r"\bCHYASM\b", "CHIASM"),
    (r"\bPARASAGITAL\b", "PARASAGITTAL"),
    (r"\bSEPTUM VERGAE\b", "CAVUM VERGAE"),
    (r"\bVELLUM\b", "VELUM"),
    (r"\bECTOPICAL\b", "ECTOPIC"),
    (r"\bNEUROHIPOPHYSIS\b", "NEUROHYPOPHYSIS"),
    (r"\bEPENDIMOMA\b", "EPENDYMOMA"),
    (r"\bNEUROCITOMA\b", "NEUROCYTOMA"),
    (r"\bSUBARACH(?:NOID)?\s+C(?:YST)?\b", "ARACHNOID CYST"),
    (r"\bARACHNOIDAL CYST\b", "ARACHNOID CYST"),
    (r"\bPOST-SURGICAL-POSTTRAUMATIC\b", "POST-SURGICAL/POST-TRAUMATIC"),
    (r"\bVERSUS\b", "VS"),
    (r"\bACOA\b", "ACOM"),
    # Numbers and units: "33,5" -> "33.5", "4MM" / "30 MMFRONTAL" -> "4 MM" / "30 MM FRONTAL"
    (r"(\d),(\d)", r"\1.\2"),
    (r"(\d)\s*MM(?=[A-Z])", r"\1 MM "),
    (r"(\d)MM\b", r"\1 MM"),
    # Sides
    (r"\bR\b", "RIGHT"),
    (r"\bL\b", "LEFT"),
    # Plural after a count: "2 MICROBLEED" -> "2 MICROBLEEDS"
    (r"\b((?:[2-9]|[1-9]\d+|MULTIPLE) MICROBLEED)\b", r"\1S"),
    # Separators
    (r"\s*\+\s*", " + "),
    (r"\s*/\s*", " / "),
)

#: Words kept in a fixed case; everything else is lower-cased.
TEXT_CASE = {
    "DVA": "DVA", "ICA": "ICA", "MCA": "MCA", "ACA": "ACA", "PCA": "PCA", "ACOM": "AComA",
    "A1": "A1", "T1": "T1", "T2": "T2", "DWI": "DWI", "TOF": "TOF", "JO": "JO", "I": "I",
    "CHIARI": "Chiari", "GALASSI": "Galassi",
}


def harmonize_text(value: Any) -> str | None:
    """
    One vocabulary for a free-text finding; ``None`` for a blank cell.

    >>> harmonize_text("1 DVA + 2 MICROHEMORRAGHES TEMPORAL RIGHT + ICA LEFT ANEURYSM")
    '1 DVA + 2 microbleeds temporal right + ICA left aneurysm'
    >>> harmonize_text("Left temporal Subarach C")
    'left temporal arachnoid cyst'
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().upper()
    if not text:
        return None
    for pattern, replacement in TEXT_RULES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip(" ,;.")
    return re.sub(r"[A-Z][A-Z0-9]*", lambda m: TEXT_CASE.get(m.group(), m.group().lower()), text)


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------
_BOUNDED_COUNT = re.compile(r"^\s*([<>]=?)?\s*(\d+(?:[.,]\d+)?)\s*$")


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def cell_code(value: Any) -> str | None:
    """
    The code of a dropdown answer: ``"2: moderada gyrus"`` -> ``"2"``, ``0`` -> ``"0"``,
    ``"JO: yugular occipital"`` -> ``"JO"``, ``"JS"`` -> ``"JS"``; ``None`` for a blank cell.
    """
    if _is_blank(value):
        return None
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    head = str(value).split(":", 1)[0].strip().upper()
    if re.fullmatch(r"\d+\.0+", head):
        head = head.split(".", 1)[0]
    return head or None


def count_value(value: Any) -> tuple[float, str | None] | None:
    """
    ``(number, censoring)`` for a count cell — ``">20"`` -> ``(20.0, ">20")`` — or ``None`` when
    the cell is not a number.
    """
    match = _BOUNDED_COUNT.match(str(value))
    if match is None:
        return None
    number = float(match.group(2).replace(",", "."))
    return number, (f"{match.group(1)}{number:g}" if match.group(1) else None)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _parse_sheet(value: str) -> str | int:
    return int(value) if value.strip().isdigit() else value


def _known_subjects(repo: Any) -> set[str]:
    subjects = repo.get("subjects", cohort_id=False)
    if subjects is None or subjects.empty or "subject_uid" not in subjects.columns:
        raise ValueError("The dataset has no subjects table to match 'Codi Sub.' against.")
    return set(subjects["subject_uid"].dropna().astype(str).str.strip())


def load_cases(path: Path, known: set[str], *, sheet: str | int = DEFAULT_SHEET) -> pd.DataFrame:
    """
    The ``Casos`` rows of known subjects, one per subject, indexed by ``subject_uid``.

    Raises
    ------
    ValueError
        When a column that :data:`ITEMS` reads is missing.
    """
    raw = read_tabular_source(path, sheet_name=_parse_sheet(str(sheet)))
    cols = {str(c).strip().lower(): c for c in raw.columns}
    needed = [SUBJECT_COLUMN, *dict.fromkeys(item.source for item in ITEMS)]
    missing = [name for name in needed if name.lower() not in cols]
    if missing:
        raise ValueError(f"{path.name} ({sheet}) is missing required column(s): {', '.join(missing)}")

    frame = raw.rename(columns={cols[name.lower()]: name for name in needed}).loc[:, needed]
    frame["subject_uid"] = frame[SUBJECT_COLUMN].map(lambda v: "" if _is_blank(v) else str(v).strip().upper())
    frame = frame.loc[frame["subject_uid"].astype(bool)]

    unknown = sorted(set(frame["subject_uid"]) - known)
    if unknown:
        log.warning(
            "%d subject(s) are not in the dataset and will be skipped: %s.",
            len(unknown), ", ".join(unknown),
        )
    frame = frame.loc[frame["subject_uid"].isin(known)]

    repeated = frame.loc[frame["subject_uid"].duplicated(keep=False)]
    for subject, rows in repeated.groupby("subject_uid", sort=True):
        values = rows.drop(columns=["subject_uid", SUBJECT_COLUMN]).astype("string").fillna("")
        differ = [column for column in values.columns if values[column].nunique() > 1]
        log.warning(
            "%s is listed %d times; keeping the last row. %s",
            subject, len(rows),
            f"The reads differ on: {', '.join(differ)}." if differ else "The reads agree.",
        )
    frame = frame.drop_duplicates(subset=["subject_uid"], keep="last").set_index("subject_uid")
    log.info("%d subject(s) read from %s.", len(frame), path.name)
    return frame


def harmonize(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """
    Long frame of every valued item — ``subject_uid``, ``variable_id``, ``source_column``,
    ``value_num``, ``value_text``, ``value_kind`` — plus ``{variable_id: {raw: harmonized}}`` for
    the free-text items.

    Raises
    ------
    ValueError
        When a cell holds a code its variable has no mapping for; every such value is listed.
    """
    frames: list[pd.DataFrame] = []
    unmapped: dict[str, dict[str, int]] = {}
    text_maps: dict[str, dict[str, str]] = {}

    for item in ITEMS:
        cells = cases[item.source]
        value_num = pd.Series(np.nan, index=cases.index, dtype="float64")
        value_text = pd.Series(pd.NA, index=cases.index, dtype="string")
        bad: dict[str, int] = {}

        for subject, cell in cells.items():
            if _is_blank(cell):
                continue
            if item.text:
                harmonized = harmonize_text(cell)
                value_text[subject] = harmonized
                text_maps.setdefault(item.variable_id, {})[str(cell).strip()] = harmonized or ""
                continue
            if item.codes is None:
                parsed = count_value(cell)
                if parsed is None:
                    bad[str(cell)] = bad.get(str(cell), 0) + 1
                    continue
                value_num[subject], value_text[subject] = parsed
                continue
            code = cell_code(cell)
            if code not in item.codes:
                bad[str(cell)] = bad.get(str(cell), 0) + 1
                continue
            value = item.codes[code]
            if value is None:
                continue
            if isinstance(value, str):
                value_text[subject] = value
            else:
                value_num[subject] = value

        if bad:
            unmapped[f"{item.source} -> {item.variable_id}"] = bad
        frame = pd.DataFrame(
            {
                "subject_uid": cases.index.astype("string"),
                "variable_id": item.variable_id,
                "source_column": item.source,
                "value_num": value_num.to_numpy(),
                "value_text": value_text.to_numpy(),
                "value_kind": item.value_kind,
            }
        )
        frame = frame.loc[frame["value_num"].notna() | frame["value_text"].notna()]
        if not frame.empty:
            frames.append(frame)

    if unmapped:
        detail = "\n".join(
            f"  {where}: " + ", ".join(f"{raw!r} (n={n})" for raw, n in values.items())
            for where, values in unmapped.items()
        )
        raise ValueError(f"Values with no mapping — add them to ITEMS before importing:\n{detail}")
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long, text_maps


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def build_rows(
    frame: pd.DataFrame,
    *,
    path: Path,
    sheet: str | int,
    visit: str = VISIT_ID,
    source_batch_id: str = SOURCE_BATCH_ID,
) -> tuple[pd.DataFrame, list[DerivedVariableRegistration]]:
    """``clinical_measurements`` rows for every variable in *frame*, plus their catalog entries."""
    labels = {item.variable_id: item.label for item in ITEMS}
    parts: list[pd.DataFrame] = []
    registrations: list[DerivedVariableRegistration] = []
    for variable_id, sub in frame.groupby("variable_id", sort=False):
        sub = sub.reset_index(drop=True).assign(visit_id=visit)
        source_column = str(sub["source_column"].iloc[0])
        kind = str(sub["value_kind"].iloc[0])
        rows = build_clinical_measurement_rows(
            sub,
            DerivedClinicalMeasurementSpec(
                variable_id=str(variable_id),
                source_file=path.name,
                source_sheet=str(sheet),
                source_column=source_column,
                value_column="value_num" if kind == "float" else "value_text",
                value_kind=kind,
                source_batch_id=source_batch_id,
            ),
        )
        # Numeric kinds get an empty value_text from the builder; a censored count (">20") keeps
        # its mark there. Positions line up — the builder resets the index and keeps row order.
        if kind == "float":
            rows["value_text"] = sub["value_text"].astype("string")
        parts.append(rows)
        registrations.append(
            DerivedVariableRegistration(
                variable_id=str(variable_id),
                domain="clinical",
                table=TABLE,
                label=labels[str(variable_id)],
                value_kind=kind,
                source_file=path.name,
                source_sheet=str(sheet),
                # ``Aneurysm`` feeds two variables, and the catalog turns a source column into an
                # alias — so a shared column would make ``Aneurysm`` resolve to whichever was
                # registered last. Register each under its own id; the column stays on the rows.
                source_column=source_column if source_column != "Aneurysm" else str(variable_id),
            )
        )
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return rows, registrations


def superseded(existing: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """True for each *existing* row sitting on a ``(subject_uid, visit_id, variable_id)`` *rows* writes."""
    if existing.empty or rows.empty:
        return pd.Series(False, index=existing.index)
    incoming = pd.MultiIndex.from_frame(rows[REPLACE_KEY].astype("string").fillna(""))
    current = pd.MultiIndex.from_frame(existing[REPLACE_KEY].astype("string").fillna(""))
    return pd.Series(current.isin(incoming), index=existing.index)


def publish_replacing(
    repo: Any,
    rows: pd.DataFrame,
    registrations: list[DerivedVariableRegistration],
    *,
    write: bool,
    provenance: dict[str, Any],
) -> pd.Series:
    """
    Replace whatever the table holds at the keys *rows* writes, then add *rows* — in one write.

    Returns the number of replaced rows per variable. Reads Parquet rather than SQLite, which may
    lag a previous write that skipped the index rebuild.
    """
    existing = repo.get(TABLE, cohort_id=False, use_sqlite=False)
    stale = superseded(existing, rows)
    replaced = existing.loc[stale, "variable_id"].astype(str).value_counts()
    if not write or rows.empty:
        return replaced

    combined = pd.concat([existing.loc[~stale], rows], ignore_index=True)
    repo.write_table(
        TABLE,
        combined,
        provenance={**provenance, "rows_replaced": int(stale.sum()), "rows_written": len(rows)},
        build_sqlite_index=False,
    )
    repo.register_variables([entry.to_catalog_entry() for entry in registrations])
    # Once, after the catalog entries exist — the index is derived from Parquet and the catalog.
    repo.build_sqlite_index(tables=[TABLE])
    return replaced


def summarize(frame: pd.DataFrame, replaced: pd.Series) -> pd.DataFrame:
    """Per variable: subjects, value distribution (categorical/numeric) and replaced rows."""
    records = []
    for variable_id, sub in frame.groupby("variable_id", sort=False):
        kind = str(sub["value_kind"].iloc[0])
        if kind == "text":
            distribution = f"{sub['value_text'].nunique()} distinct text value(s)"
        else:
            values = sub["value_num"] if kind == "float" else sub["value_text"]
            counts = values.value_counts().sort_index()
            distribution = ", ".join(
                f"{f'{k:g}' if isinstance(k, float) else k}={n}" for k, n in counts.items()
            )
        records.append(
            {
                "variable_id": variable_id,
                "kind": kind,
                "n": len(sub),
                "replaces": int(replaced.get(variable_id, 0)),
                "values": distribution,
            }
        )
    return pd.DataFrame(records).set_index("variable_id")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-brain-codebook")
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SOURCE or None,
    required=not DEFAULT_SOURCE,
    help="PESA-Brain codebook workbook holding the 'Casos' sheet.",
)
@click.option("--sheet", default=DEFAULT_SHEET, show_default=True, type=str, help="Worksheet name or zero-based sheet index.")
@click.option("--visit", default=VISIT_ID, show_default=True, help="visit_id to file the read under.")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the path configured in .nvitk/settings.json.",
)
@click.option("--source-batch-id", default=SOURCE_BATCH_ID, show_default=True)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually publish into clinical_measurements; default is dry-run.",
)
def main(source: Path, sheet: str, visit: str, dataset: Path | None, source_batch_id: str, write: bool) -> None:
    """Import the harmonized PESA-Brain neuroradiology read into clinical_measurements."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
        cases = load_cases(Path(source), _known_subjects(repo), sheet=sheet)
        frame, text_maps = harmonize(cases)
        if frame.empty:
            raise ValueError("No valued cells to import.")
        rows, registrations = build_rows(
            frame, path=Path(source), sheet=sheet, visit=str(visit), source_batch_id=source_batch_id
        )
        replaced = publish_replacing(
            repo,
            rows,
            registrations,
            write=write,
            provenance={"importer": "import_brain_codebook", "source_file": Path(source).name},
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 140):
        click.echo(summarize(frame, replaced).to_string())
    if not write:
        for variable_id, pairs in text_maps.items():
            click.echo(f"\n{variable_id}: raw -> harmonized")
            for raw, harmonized in sorted(pairs.items()):
                click.echo(f"  {raw!r}\n    -> {harmonized!r}")

    action = "Wrote" if write else "Dry run — would write"
    click.echo(
        f"\n{action} {len(rows)} row(s) across {rows['variable_id'].nunique()} variable(s) at visit "
        f"{visit}, replacing {int(replaced.sum())} existing row(s) at the same subject/visit/variable."
    )
    if not write:
        click.echo("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
