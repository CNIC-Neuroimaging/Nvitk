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
ANALYSIS_LME4 = "lme4"
ANALYSIS_MMRM = "mmrm"
ANALYSIS_OLS = "ols"
ANALYSIS_LMROB = "lmrob"
ANALYSIS_GLM = "glm"
ANALYSIS_NONLINEAR = "nonlinear"
ANALYSIS_SEM = "sem"
ANALYSIS_MRF = "mrf"
ANALYSIS_MEDIATION = "mediation"

ANALYSIS_ITEMS: tuple[tuple[str, str], ...] = (
    ("Mixed linear model (MixedLM)", ANALYSIS_MIXEDLM),
    ("Mixed model — R · lme4", ANALYSIS_LME4),
    ("Repeated measures (MMRM) — R · mmrm", ANALYSIS_MMRM),
    ("Linear model (OLS)", ANALYSIS_OLS),
    ("Robust linear model — R · lmrob", ANALYSIS_LMROB),
    ("Generalized linear model (GLM)", ANALYSIS_GLM),
    ("Non-linear curve fit", ANALYSIS_NONLINEAR),
    ("Path model / SEM — vascular network", ANALYSIS_SEM),
    ("MRF smooth over the vessel graph — R · mgcv", ANALYSIS_MRF),
    ("Mediation analysis", ANALYSIS_MEDIATION),
)

# Analyses driven by a formula, which therefore share the formulation panel and the plot.
ANALYSIS_FORMULA_KINDS: frozenset[str] = frozenset(
    {ANALYSIS_MIXEDLM, ANALYSIS_LME4, ANALYSIS_MMRM, ANALYSIS_OLS, ANALYSIS_GLM,
     ANALYSIS_LMROB, ANALYSIS_MRF}
)

# Engines whose covariance/random structure is described by dropdowns rather than by the formula.
ANALYSIS_R_KINDS: frozenset[str] = frozenset(
    {ANALYSIS_LME4, ANALYSIS_MMRM, ANALYSIS_LMROB, ANALYSIS_MRF}
)

# Analyses whose plot is a family of curves over anatomical levels, and can therefore be split into
# a grid of panels. Mediation is the exception: its per-level figure is already a forest with one
# row per level on a shared effect axis, which panelling would fragment rather than clarify.
ANALYSIS_PANEL_KINDS: frozenset[str] = frozenset(
    {
        ANALYSIS_MIXEDLM,
        ANALYSIS_LME4,
        ANALYSIS_MMRM,
        ANALYSIS_OLS,
        ANALYSIS_GLM,
        ANALYSIS_LMROB,
        ANALYSIS_NONLINEAR,
    }
)

# lme4 accepts more response families than lmer alone; gaussian uses lmer, the rest glmer.
LME4_FAMILY_ITEMS: tuple[tuple[str, str], ...] = (
    ("gaussian — continuous (lmer)", "gaussian"),
    ("binomial — binary (glmer)", "binomial"),
    ("poisson — counts (glmer)", "poisson"),
    ("gamma — skewed positive (glmer)", "gamma"),
    ("inverse_gaussian (glmer)", "inverse_gaussian"),
)

ANALYSIS_HINTS: dict[str, str] = {
    ANALYSIS_MIXEDLM: (
        "Repeated measurements per subject or per territory: the random structure absorbs the "
        "within-cluster correlation that a plain linear model would treat as extra evidence."
    ),
    ANALYSIS_LME4: (
        "The same mixed models, written R's way: the whole random structure lives in the formula, "
        "as pi ~ age_c + (1 + age_c | territory) + (1 | subject_uid). Adds Satterthwaite p-values "
        "via lmerTest. Needs pymer4, rpy2 and R with lme4 installed."
    ),
    ANALYSIS_MMRM: (
        "Models the correlation between a subject's repeated measurements directly, instead of "
        "decomposing it into a random intercept. A random intercept assumes every pair of "
        "territories is equally correlated; an unstructured MMRM estimates all of them. Needs R "
        "with the mmrm package — no extra Python packages."
    ),
    ANALYSIS_OLS: (
        "One independent observation per row. Simpler and faster than a MixedLM, but it will "
        "understate the standard errors if the same subject appears in several rows."
    ),
    ANALYSIS_LMROB: (
        "Least squares lets one badly segmented vessel move every coefficient. lmrob estimates the "
        "same linear model by MM-estimation, which tolerates up to half the data being contaminated "
        "and gives each observation a robustness weight — read the Weights tab to see which scans "
        "the fit discounted. Unlike statsmodels' RLM it also resists outliers in the *predictors*. "
        "No random effects: for clustered rows the coefficients hold but the standard errors ignore "
        "the clustering. Needs R with the robustbase package."
    ),
    ANALYSIS_GLM: (
        "A non-linear link between the predictors and the response: logistic for binary outcomes, "
        "Poisson for counts, Gamma with a log link for skewed positive measures. Coefficients are "
        "effects on the link scale; plots are shown on the scale of the data."
    ),
    ANALYSIS_NONLINEAR: (
        "Fits an explicit curve of one measurement against one predictor. No covariates and no "
        "grouping — the parameters are the model."
    ),
    ANALYSIS_SEM: (
        "Writes the vasculature as a system of regressions — basi ~ lva + rva, lpca ~ basi + "
        "lpcomm — and fits every junction at once, so an exogenous effect splits into the part "
        "travelling through the network and the part that does not. The mediation analysis is the "
        "three-node special case. Needs semopy (pip) or R with lavaan.\n"
        "Cross-sectional data: the direction is an anatomical assumption written into the model, "
        "not something the fit establishes."
    ),
    ANALYSIS_MRF: (
        "A random intercept shrinks every territory toward one mean, as if the basilar and the left "
        "MCA were two draws from a population of vessels. A Markov random field instead penalises "
        "differences between *adjacent* vessels, so each estimate borrows strength from the ones it "
        "actually connects to. Undirected — for direction use the path model. Needs R with mgcv, "
        "which ships with almost every R install."
    ),
    ANALYSIS_MEDIATION: (
        "How much of X's effect on Y runs through M."
    ),
}

# Heteroscedasticity-consistent covariance options for OLS.
ROBUST_COV_ITEMS: tuple[tuple[str, str | None], ...] = (
    ("classical (assume constant variance)", None),
    ("HC0 — heteroscedasticity-consistent", "HC0"),
    ("HC1 — HC0 with a small-sample correction", "HC1"),
    ("HC2 — leverage-adjusted", "HC2"),
    ("HC3 — leverage-adjusted, conservative", "HC3"),
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
    "ANALYSIS_FORMULA_KINDS",
    "ANALYSIS_GLM",
    "ANALYSIS_HINTS",
    "ANALYSIS_ITEMS",
    "ANALYSIS_LME4",
    "ANALYSIS_MMRM",
    "ANALYSIS_LMROB",
    "ANALYSIS_MRF",
    "ANALYSIS_SEM",
    "ANALYSIS_PANEL_KINDS",
    "ANALYSIS_R_KINDS",
    "ANALYSIS_MEDIATION",
    "ANALYSIS_MIXEDLM",
    "LME4_FAMILY_ITEMS",
    "ANALYSIS_NONLINEAR",
    "ANALYSIS_OLS",
    "ROBUST_COV_ITEMS",
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
