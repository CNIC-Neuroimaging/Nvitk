"""Static choices and defaults for the Statmodels explorer."""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline kinds and atlases
# ──────────────────────────────────────────────────────────────────────────────
PIPELINE_KIND_QVTPY = "qvtpy"
PIPELINE_KIND_ASL = "asl"
PIPELINE_KIND_T1 = "t1"
PIPELINE_KIND_FLAIR = "flair"
PIPELINE_KIND_TOF = "tof"

PIPELINE_KIND_ITEMS: tuple[tuple[str, str], ...] = (
    ("qvtpy — 4D flow hemodynamics", PIPELINE_KIND_QVTPY),
    ("ASL — perfusion (CBF / ATT)", PIPELINE_KIND_ASL),
    ("T1 — volumetry", PIPELINE_KIND_T1),
    ("FLAIR — WMH", PIPELINE_KIND_FLAIR),
    ("TOF — morphometrics (eICAB)", PIPELINE_KIND_TOF),
)

ASL_ATLASES: tuple[tuple[str, str], ...] = (
    ("Desikan (cortical parcels)", "desikan"),
    ("Vascular atlas · smooth 0", "vascular-0"),
    ("Vascular atlas · smooth 8", "vascular-8"),
    ("Vascular atlas · smooth 12", "vascular-12"),
)

# How measurement frames are combined on (subject_uid, territory).
JOIN_MODES: tuple[tuple[str, str], ...] = (
    ("inner — only cells present in every measurement", "inner"),
    ("outer — keep every cell, fill gaps with NaN", "outer"),
    ("left — keep the first measurement's cells", "left"),
)

# ──────────────────────────────────────────────────────────────────────────────
# Analysis types
# ──────────────────────────────────────────────────────────────────────────────
ANALYSIS_MIXEDLM = "mixedlm"
ANALYSIS_MEDIATION = "mediation"

ANALYSIS_ITEMS: tuple[tuple[str, str], ...] = (
    ("Mixed linear model (MixedLM)", ANALYSIS_MIXEDLM),
    ("Mediation analysis", ANALYSIS_MEDIATION),
)

# ──────────────────────────────────────────────────────────────────────────────
# Model formulation defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FORMULA = (
    "flow_mean ~ C(tacsctot_group, Treatment('None')) "
    "* C(group_key) + age_c + sex + Hematocrit"
)
DEFAULT_GROUPS = "group_key"
DEFAULT_RE = "0"
DEFAULT_VC = '{"patient": "0 + C(subject_uid)"}'
DEFAULT_MODEL_NAME = "mixedlm_model"

# ──────────────────────────────────────────────────────────────────────────────
# Table / plot behaviour
# ──────────────────────────────────────────────────────────────────────────────
# Rows shown in the analysis-dataframe preview. Filters and fits always use the full frame.
TABLE_ROW_CAP = 2000
# A column with more distinct values than this gets the numeric/expression tabs only — a checkable
# list of thousands of levels is not a usable way to pick anything.
MAX_CATEGORICAL_LEVELS = 200
# Levels beyond this get a scrollable list without the per-level colour swatches.
MAX_PLOT_GROUP_LEVELS = 60

# Slider resolution for the axis-limit controls, and how far past the plotted data they may reach.
AXIS_SLIDER_STEPS = 1000
AXIS_SLIDER_MARGIN = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────
# Bumped when the saved-config schema changes; ``_migrate_config`` upgrades anything older.
CONFIG_VERSION = 2

__all__ = [
    "ANALYSIS_ITEMS",
    "ANALYSIS_MEDIATION",
    "ANALYSIS_MIXEDLM",
    "ASL_ATLASES",
    "AXIS_SLIDER_MARGIN",
    "AXIS_SLIDER_STEPS",
    "CONFIG_VERSION",
    "DEFAULT_FORMULA",
    "DEFAULT_GROUPS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RE",
    "DEFAULT_VC",
    "JOIN_MODES",
    "MAX_CATEGORICAL_LEVELS",
    "MAX_PLOT_GROUP_LEVELS",
    "PIPELINE_KIND_ASL",
    "PIPELINE_KIND_FLAIR",
    "PIPELINE_KIND_ITEMS",
    "PIPELINE_KIND_QVTPY",
    "PIPELINE_KIND_T1",
    "PIPELINE_KIND_TOF",
    "TABLE_ROW_CAP",
]
