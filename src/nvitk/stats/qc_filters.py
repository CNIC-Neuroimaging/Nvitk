"""
Ready-made quality-control filters for the analysis dataframe.

Description
-----------
The automatic QC metrics published by the qvtpy stage 9 are ordinary columns, so filtering on them
needs no machinery beyond the ordinary :class:`~nvitk.stats.frame_ops.FilterRule`. What it needs is
*knowing the right threshold*, which is exactly what a user should not have to look up mid-analysis:
the plausibility score fails below 0.5, the flag columns fail **above** 0.5 because 1 means "review",
and the conservation residual fails on its **magnitude** in either direction.

Each preset here turns one of those into the rules that express it, so applying a literature-band
check is a menu click rather than three fields typed correctly.

Presets are **soft by construction**. They drop rows from the analysis frame — and therefore from the
fit — which is a stronger action than the plot's grey-out. Reach for the grey-out when you want to
*see* what fails, and for these when you have decided the failures should not inform the model.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from nvitk.core.logger import Logger

from .frame_ops import FilterRule

log = Logger()


@dataclass(frozen=True)
class QcFilterPreset:
    """One named QC filter, and how to express it as filter rules."""

    key: str
    label: str
    description: str
    #: Columns that must be in the frame before the preset can be applied.
    requires: tuple[str, ...]
    #: Whether it applies to any column or only to a specific measurement.
    applies_to: str = "any"

    def rules(self, *, threshold: float | None = None) -> list[FilterRule]:
        """The filter rules this preset expands into."""
        return _PRESET_RULES[self.key](threshold)


def _plausibility_rules(threshold: float | None) -> list[FilterRule]:
    """Keep rows the subject-aware flow decision retained — see :func:`flow_qc_keep`."""
    return [FilterRule(column=KEEP_COLUMN, kind="compare", op=">=", value="0.5")]


def _hypoplastic_rules(_threshold: float | None) -> list[FilterRule]:
    """
    Drop vessels flagged as plausibly hypoplastic.

    ``keep_na``: a vessel with no measured caliber was never assessed, and dropping every one of
    them would empty a frame that simply has no area measurement loaded.
    """
    return [FilterRule(
        column="qc_hypoplastic", kind="range", low=-0.5, high=0.5, keep_na=True,
    )]


def _conservation_rules(threshold: float | None) -> list[FilterRule]:
    """
    Keep rows whose junction residual is small in magnitude.

    A residual fails in *either* direction, so this is a symmetric range rather than a comparison —
    ``> -0.15`` would keep every subject whose outflow exceeds inflow by any amount at all.
    """
    limit = abs(0.15 if threshold is None else threshold)
    # ``keep_na``: the residual is a property of a junction and is recorded on the parent vessel
    # only, so most rows have none. Failing those would turn "drop junctions that do not balance"
    # into "keep only the three parent vessels".
    return [FilterRule(
        column="qc_conservation", kind="range", low=-limit, high=limit, keep_na=True,
    )]


def _score_rules(threshold: float | None) -> list[FilterRule]:
    """Keep rows the subject-aware flow decision retained — see :func:`flow_qc_keep`."""
    return [FilterRule(column=KEEP_COLUMN, kind="compare", op=">=", value="0.5")]


def _flag_rules(_threshold: float | None) -> list[FilterRule]:
    """Keep rows the subject-aware flow decision retained — see :func:`flow_qc_keep`."""
    return [FilterRule(column=KEEP_COLUMN, kind="compare", op=">=", value="0.5")]


def _cbf_rules(_threshold: float | None) -> list[FilterRule]:
    """Keep ASL mean CBF inside its physiological window."""
    return [FilterRule(column=CBF_COLUMN, kind="range", low=CBF_MIN, high=CBF_MAX)]


#: ASL grey-matter perfusion window, in mL/100 g/min.
CBF_MIN: float = 20.0
CBF_MAX: float = 95.0
CBF_COLUMN: str = "mean_cbf"

_PRESET_RULES = {
    "flow_plausible": _plausibility_rules,
    "hypoplastic": _hypoplastic_rules,
    "conservation": _conservation_rules,
    "score": _score_rules,
    "flag": _flag_rules,
    "cbf_window": _cbf_rules,
}

#: The presets offered from a column's context menu, in the order they should appear.
QC_FILTER_PRESETS: tuple[QcFilterPreset, ...] = (
    QcFilterPreset(
        "flow_plausible", "Flow within the literature band",
        "Keep vessels whose time-averaged flow scores at least 0.5 against the per-vessel band "
        "(Zarrinkoob et al. 2015, widened for the 4D-vs-2D underestimation). Catches gross "
        "segmentation, aliasing and LOC-placement failures, not unusual physiology.",
        requires=("qc_flow_plausible",),
    ),
    QcFilterPreset(
        "hypoplastic", "Drop hypoplastic vessels",
        "Remove vessels whose segmented caliber is under 0.8 mm (Krabbe-Hartkamp et al. 1998). "
        "These are normal anatomy carrying almost no flow — excluding them stops a normal circle "
        "of Willis dragging a territory's mean toward zero.",
        requires=("qc_hypoplastic",),
    ),
    QcFilterPreset(
        "conservation", "Mass conservation within 15%",
        "Keep junctions whose inflow and outflow agree to within 15% either way, the 4D Flow CMR "
        "consensus statement's internal-consistency threshold. A failure is ambiguous between a "
        "bad flow measurement and an incomplete vessel tree.",
        requires=("qc_conservation",),
    ),
    QcFilterPreset(
        "score", "Combined QC score ≥ 0.5",
        "Keep vessels whose combined score — plausibility and conservation together — is at least "
        "0.5. The broadest of the flow filters; use it when you want one gate rather than three.",
        requires=("qc_score",),
    ),
    QcFilterPreset(
        "flag", "Drop everything flagged for review",
        "Remove every vessel the automatic QC flagged. The strictest option, and the bluntest: it "
        "gives no say over which check did the flagging.",
        requires=("qc_flag",),
    ),
    QcFilterPreset(
        "cbf_window", f"ASL mean CBF within {CBF_MIN:g}–{CBF_MAX:g}",
        f"Keep perfusion inside {CBF_MIN:g}–{CBF_MAX:g} mL/100 g/min. Values outside that window in "
        f"grey matter are almost always a labelling-efficiency, arrival-time or motion artefact "
        f"rather than real perfusion.",
        requires=(CBF_COLUMN,),
        applies_to=CBF_COLUMN,
    ),
)

QC_FILTER_BY_KEY: dict[str, QcFilterPreset] = {p.key: p for p in QC_FILTER_PRESETS}

#: Presets evaluated by :func:`flow_qc_keep` rather than by a plain column threshold.
SUBJECT_AWARE_KEYS: dict[str, str] = {
    "flow_plausible": "qc_flow_plausible",
    "score": "qc_score",
    "flag": "qc_flag",
}



# ---------------------------------------------------------------------------
# Flow QC as a subject-aware decision
# ---------------------------------------------------------------------------
#: Vessels whose measurement carries the subject's whole cerebral inflow. If one of these is
#: implausible, every other flow in that subject is suspect too: the conservation checks, the
#: anterior/posterior split and any total all read through them.
CRITICAL_NODES: frozenset[str] = frozenset({"lica", "rica", "basi"})

#: Vessels the flow bands deliberately do not cover — the communicating arteries, where near-zero
#: or reversed flow is a normal circle of Willis, and the venous sinuses, which have no arterial
#: band at all. They are never removed by a flow filter, and never trigger the subject cascade.
EXEMPT_NODES: frozenset[str] = frozenset({
    "lpcomm", "rpcomm", "acomm", "sss", "strs", "lts", "rts",
})

#: Column the flow presets filter on, holding the decision below.
KEEP_COLUMN = "qc_keep"


def flow_qc_keep(
    frame: pd.DataFrame,
    *,
    metric: str = "qc_flow_plausible",
    threshold: float = 0.5,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
) -> tuple[pd.Series, dict[str, object]]:
    """
    Which rows a flow quality filter should keep, as a subject-aware decision.

    A plain column threshold gets two things wrong on this data:

    1. **Exempt vessels are dropped for having no score.** A PComm or a sagittal sinus has no
       arterial band, so its metric is missing — and a ``>=`` comparison fails a missing value.
       They would vanish from the frame for the crime of not being scoreable.
    2. **A failing carotid only removes itself.** If a subject's ICA is measured at 31 mL/min when
       a healthy one is ~257, that subject's remaining flows are not trustworthy either: the
       conservation residuals, the anterior/posterior split and any total all read through it.

    So the decision is: exempt vessels are always kept; a failing **critical** vessel (ICA or
    basilar) removes every flow row for that subject; any other failing vessel removes only itself.

    Returns
    -------
    (keep, report)
        *keep* is a boolean Series aligned to *frame*. *report* carries ``n_failing``,
        ``n_exempt_kept``, ``subjects_dropped`` and ``critical_failures`` for the status line.
    """
    from .vessel_network import canonical_node

    if metric not in frame.columns:
        raise ValueError(f"{metric!r} is not in the frame.")
    keep = pd.Series(True, index=frame.index)
    report: dict[str, object] = {
        "n_failing": 0, "n_exempt_kept": 0, "subjects_dropped": 0, "critical_failures": [],
    }
    if region_column not in frame.columns:
        # No region column: fall back to the plain threshold, which is all the frame supports.
        values = pd.to_numeric(frame[metric], errors="coerce")
        failing = values < float(threshold)
        report["n_failing"] = int(failing.sum())
        return ~failing.fillna(False), report

    nodes = frame[region_column].map(canonical_node)
    exempt = nodes.isin(EXEMPT_NODES) | nodes.isna()
    values = pd.to_numeric(frame[metric], errors="coerce")
    # A missing score is "not checked", never "failed" — that is what protects the exempt vessels
    # and anything the band table does not recognise.
    failing = (values < float(threshold)).fillna(False) & ~exempt

    report["n_failing"] = int(failing.sum())
    report["n_exempt_kept"] = int(exempt.sum())

    critical_failure = failing & nodes.isin(CRITICAL_NODES)
    if subject_column in frame.columns and bool(critical_failure.any()):
        bad_subjects = set(frame.loc[critical_failure, subject_column].astype(str))
        report["subjects_dropped"] = len(bad_subjects)
        report["critical_failures"] = sorted(
            {str(v) for v in frame.loc[critical_failure, region_column]}
        )
        keep &= ~frame[subject_column].astype(str).isin(bad_subjects)

    keep &= ~failing
    return keep, report


def available_presets(
    columns: Sequence[str], *, column: str = ""
) -> list[tuple[QcFilterPreset, str]]:
    """
    Every applicable preset, with whether the frame can satisfy it.

    The second element is ``"ready"`` when the metric column is in the frame and ``"unavailable"``
    otherwise. The metrics are **read from the dataset**, never derived here: scoring on the fly
    would give one session's filter a different answer from the published one, and from every other
    session — the QC has to mean the same thing everywhere it is read. Run the qvtpy stage 9 to
    publish them.

    Returned rather than filtered so the menu can grey an entry out with a reason; silently omitting
    it leaves the user wondering where the QC filters went.

    Parameters
    ----------
    column : str
        The column the menu was opened on. A preset tied to a specific measurement is only offered
        there; the rest are offered anywhere, since they filter the frame rather than that column.
    """
    present = {str(c) for c in columns}
    out: list[tuple[QcFilterPreset, str]] = []
    for preset in QC_FILTER_PRESETS:
        if preset.applies_to != "any" and str(column) != preset.applies_to:
            continue
        state = "ready" if all(r in present for r in preset.requires) else "unavailable"
        out.append((preset, state))
    return out


def preset_rules(
    key: str, frame: pd.DataFrame, *, threshold: float | None = None
) -> list[FilterRule]:
    """
    Rules for one preset, validated against *frame*.

    Raises
    ------
    KeyError
        Unknown preset.
    ValueError
        The frame lacks a column the preset needs — with the reason, since "run stage 9" is the
        actionable answer and an empty filter would silently do nothing.
    """
    preset = QC_FILTER_BY_KEY[key]
    missing = [c for c in preset.requires if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{preset.label!r} needs {', '.join(missing)}, which this frame does not carry. "
            + (
                "Run the qvtpy stage 9 (autoqc) and reload."
                if any(c.startswith("qc_") for c in missing)
                else "Load that measurement first."
            )
        )
    return preset.rules(threshold=threshold)


__all__ = [
    "CBF_COLUMN",
    "CRITICAL_NODES",
    "EXEMPT_NODES",
    "KEEP_COLUMN",
    "SUBJECT_AWARE_KEYS",
    "flow_qc_keep",
    "CBF_MAX",
    "CBF_MIN",
    "QC_FILTER_BY_KEY",
    "QC_FILTER_PRESETS",
    "QcFilterPreset",
    "available_presets",
    "preset_rules",
]
