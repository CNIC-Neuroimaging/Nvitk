"""qvtpy stage 9: automatic quality control over published 4D-flow measurements.

**Inputs**

- ``image_measurements`` rows already published by stage 6 — ``flow_mean`` per vessel, and
  ``cross_section_area`` where available. Nothing is read from disk, so this stage can be
  re-run after a re-import without touching the pipeline outputs.

**Outputs**

- New ``image_measurements`` rows carrying the per-vessel QC scores, and
  ``clinical_measurements`` rows carrying the subject-level ones, so every metric is
  queryable and joinable exactly like a hemodynamic measurement.

Description
-----------
Stage 6 answers "what is the flow?". This stage answers "should we believe it?", using the
literature-grounded checks in :mod:`nvitk.measure.hemodynamics`:

======================  ======================================================  ==========
Metric                  What it catches                                         Level
======================  ======================================================  ==========
``qc_flow_plausible``   flow far outside the healthy band for that vessel        vessel
``qc_hypoplastic``      caliber under 0.8 mm — normal anatomy, not a failure     vessel
``qc_conservation``     junction inflow ≠ outflow                                vessel
``qc_segment_cv``       flow varies too much along a non-branching segment       vessel
``qc_ap_share``         anterior/posterior split away from 72/28                 subject
``qc_score``            the per-vessel scores combined                           vessel
``qc_flag``             any check failed                                         both
======================  ======================================================  ==========

The scores are deliberately **soft**: a low ``qc_flow_plausible`` marks a measurement worth
looking at, not one to delete. Single-subject pathology — a high-grade stenosis, an AVM,
moyamoya collateralisation — legitimately falls outside a healthy-cohort band without any
data-quality problem, which is why the pipeline flags and never drops.

Hypoplasia gates the rest: a vessel whose segmented caliber is already at or under the
Krabbe-Hartkamp 0.8 mm threshold carries almost no flow by anatomy, and scoring it against a
patent-vessel band would report a normal circle of Willis as a failure.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import click

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.measure.hemodynamics import (
    ANTERIOR_SHARE_PCT,
    ANTERIOR_SHARE_TOL_PCT,
    CONSERVATION_TOL,
    CONSERVATION_TOL_ARTERIAL,
    CONSERVATION_TOL_DISTAL,
    CONSERVATION_TOL_VENOUS,
    SEGMENT_CV_TOL,
    anterior_posterior_share_pct,
    anterior_posterior_split_flag,
    flow_plausibility_score,
    is_plausibly_hypoplastic,
    segment_flow_consistency_cv,
)

log = Logger()

STAGE_NAME = "stage9_autoqc"

#: Variable ids this stage publishes, and the table each belongs to.
QC_VARIABLES: dict[str, str] = {
    "qc_flow_plausible": "image_measurements",
    "qc_hypoplastic": "image_measurements",
    "qc_conservation": "image_measurements",
    "qc_segment_cv": "image_measurements",
    "qc_score": "image_measurements",
    "qc_flag": "image_measurements",
    "qc_ap_share": "clinical_measurements",
    "qc_ap_flag": "clinical_measurements",
    "qc_subject_flag": "clinical_measurements",
}

#: Human-readable labels, used for the catalog entries and the GUI colour picker.
QC_LABELS: dict[str, str] = {
    "qc_flow_plausible": "Flow plausibility (literature band, 0–1)",
    "qc_hypoplastic": "Plausibly hypoplastic (<0.8 mm)",
    "qc_conservation": "Junction mass-conservation residual",
    "qc_segment_cv": "Along-segment flow CV",
    "qc_score": "Combined per-vessel QC score (0–1)",
    "qc_flag": "Vessel QC flag (1 = review)",
    "qc_ap_share": "Anterior share of cerebral inflow (%)",
    "qc_ap_flag": "Anterior/posterior split flag (1 = outside 72±10%)",
    "qc_subject_flag": "Subject QC flag (1 = any vessel or subject check failed)",
}

#: Score at or below which a vessel is flagged for review.
QC_SCORE_FLAG_BELOW: float = 0.5

#: AutoQC mass-balance checks: ``(rule_key, anchor_nodes, tolerance)``.
#: Rules come from :data:`~nvitk.stats.vessel_network.CONSERVATION_RULES`. Communicating-artery
#: terms are dropped (unsigned LOC flows), so ICA and BA→PCA balances reduce to the classical
#: parent → daughters check used by the QVT validation paper. Residuals are written onto every
#: *anchor* vessel so arterial parents, VAs/PCAs and the venous confluence all colour in the GUI.
AUTOQC_CONSERVATION: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("left_carotid_split", ("lica",), CONSERVATION_TOL_ARTERIAL),
    ("right_carotid_split", ("rica",), CONSERVATION_TOL_ARTERIAL),
    ("basilar_inflow", ("basi", "lva", "rva"), CONSERVATION_TOL_ARTERIAL),
    ("posterior_split", ("basi", "lpca", "rpca"), CONSERVATION_TOL_DISTAL),
    ("venous_drainage", ("sss", "strs", "lts", "rts"), CONSERVATION_TOL_VENOUS),
)

#: Vessels whose along-segment flow CV is published. Matches the QVT paper's continuous-segment
#: check (L/R ICA, SSS) plus the basilar, which is long enough for a meaningful CV.
SEGMENT_CV_NODES: frozenset[str] = frozenset({"lica", "rica", "basi", "sss"})


@dataclass(frozen=True)
class AutoQcConfig:
    """Thresholds and inputs for one autoqc run."""

    flow_variable: str = "flow_mean"
    area_variable: str = "cross_section_area"
    #: Factor converting the stored flow to mL/min, which the literature bands are in. ``None``
    #: infers it from the magnitude — the right default, because what a dataset stores depends on
    #: which importer wrote it: stage 6 emits mL/s, but several published tables are already per
    #: minute. Assuming either one silently mis-scales the other by 60x and fails the whole cohort.
    flow_to_ml_min: float | None = None
    conservation_tol: float = CONSERVATION_TOL
    conservation_tol_arterial: float = CONSERVATION_TOL_ARTERIAL
    conservation_tol_distal: float = CONSERVATION_TOL_DISTAL
    conservation_tol_venous: float = CONSERVATION_TOL_VENOUS
    segment_cv_tol: float = SEGMENT_CV_TOL
    score_flag_below: float = QC_SCORE_FLAG_BELOW
    anterior_pct: float = ANTERIOR_SHARE_PCT
    anterior_tol_pct: float = ANTERIOR_SHARE_TOL_PCT


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_vessel_qc(
    flows: pd.DataFrame,
    areas: pd.DataFrame | None = None,
    *,
    config: AutoQcConfig | None = None,
    segment_cv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Per-vessel QC scores for one cohort's published flow measurements.

    Parameters
    ----------
    flows : pandas.DataFrame
        Long rows with ``subject_uid``, ``region_id`` and ``value`` — the shape
        :meth:`~nvitk.db.repo.DataRepo.image` returns for ``flow_mean``.
    areas : pandas.DataFrame, optional
        The same shape for the cross-sectional area, used only to gate hypoplastic vessels.
        Without it every vessel is scored as patent, which over-flags a normal circle of
        Willis — so its absence is reported rather than passed over.
    segment_cv : pandas.DataFrame, optional
        Long rows with ``subject_uid``, ``region_id`` and ``value`` holding the along-segment
        flow coefficient of variation (from ``pitc_profile.csv``). Missing → ``qc_segment_cv``
        is NaN and is skipped in the combined score.

    Returns
    -------
    pandas.DataFrame
        One row per ``(subject_uid, region_id)`` with ``qc_flow_plausible``,
        ``qc_hypoplastic``, ``qc_conservation``, ``qc_segment_cv``, ``qc_score`` and ``qc_flag``.
    """
    from nvitk.stats.vessel_network import canonical_node

    config = config or AutoQcConfig()
    required = {"subject_uid", "region_id", "value"}
    missing = required - set(flows.columns)
    if missing:
        raise ValueError(f"flows is missing {', '.join(sorted(missing))}.")

    out = flows.loc[:, ["subject_uid", "region_id", "value"]].copy()
    out["subject_uid"] = out["subject_uid"].astype(str)
    out["region_id"] = out["region_id"].astype(str)
    values = pd.to_numeric(out["value"], errors="coerce")
    scale = config.flow_to_ml_min
    if scale is None:
        scale = infer_flow_scale(values)
    out["flow_ml_min"] = values * float(scale)
    out = out.drop(columns=["value"])
    out["node"] = out["region_id"].map(canonical_node)

    # Two importers can populate the same dataset with different spellings — ``LICA`` from one and
    # ``left_ica`` from another — and both survive as separate rows. Every downstream pivot then
    # averages two pipelines together without saying so, which is worse than either alone.
    named = out.dropna(subset=["node"])
    clashes = named.groupby(["subject_uid", "node"])["region_id"].nunique()
    n_clashing = int((clashes > 1).sum())
    if n_clashing:
        examples = (
            named.groupby("node")["region_id"].unique().loc[
                clashes[clashes > 1].index.get_level_values("node").unique()
            ]
        )
        log.warning(
            "autoqc: %d (subject, vessel) cell(s) carry more than one region spelling — e.g. %s. "
            "Two imports are mixed in this dataset; restrict with --pipeline, or the conservation "
            "and anterior/posterior checks will average them together.",
            n_clashing,
            "; ".join(f"{n} = {sorted(v)}" for n, v in list(examples.items())[:3]),
        )

    # ---- 1. Hypoplasia gate — evaluated first, because it excuses everything else ------
    if areas is not None and not areas.empty and {"subject_uid", "region_id", "value"} <= set(areas.columns):
        area_lookup = (
            areas.assign(
                subject_uid=lambda d: d["subject_uid"].astype(str),
                region_id=lambda d: d["region_id"].astype(str),
            )
            .set_index(["subject_uid", "region_id"])["value"]
        )
        keys = list(zip(out["subject_uid"], out["region_id"]))
        area_values = pd.to_numeric(
            pd.Series([area_lookup.get(k, np.nan) for k in keys], index=out.index),
            errors="coerce",
        )
        out["area_mm2"] = area_values
        out["qc_hypoplastic"] = [
            float(is_plausibly_hypoplastic(a)) if pd.notna(a) else np.nan for a in area_values
        ]
    else:
        log.warning(
            "autoqc: no %r measurements available — hypoplastic vessels cannot be excused and "
            "will be scored against a patent-vessel band.", config.area_variable,
        )
        out["area_mm2"] = np.nan
        out["qc_hypoplastic"] = np.nan

    # ---- 2. Literature flow band, skipped for hypoplastic and exempt vessels ----------
    scores: list[float] = []
    for flow, region, hypoplastic in zip(out["flow_ml_min"], out["region_id"], out["qc_hypoplastic"]):
        if hypoplastic == 1.0:
            scores.append(np.nan)
            continue
        scores.append(flow_plausibility_score(flow, region))
    out["qc_flow_plausible"] = scores

    # ---- 3. Mass conservation at each junction (ICA, VA→BA→PCA, venous) ----------------
    residual, tol_used = _conservation_columns(out, config)
    out["qc_conservation"] = residual
    out["_conservation_tol"] = tol_used

    # ---- 4. Along-segment flow CV (from pitc_profile stations, when available) --------
    out["qc_segment_cv"] = _segment_cv_column(out, segment_cv)

    # ---- 5. Combined score: the mean of whichever checks applied ----------------------
    # Residuals / CVs are errors, not scores — map each onto [0, 1] by its own tolerance
    # before averaging with the literature-band plausibility.
    conservation_score = 1.0 - (
        out["qc_conservation"].abs() / out["_conservation_tol"].clip(lower=1e-9)
    )
    conservation_score = conservation_score.clip(lower=0.0, upper=1.0)
    segment_score = 1.0 - (out["qc_segment_cv"].abs() / max(config.segment_cv_tol, 1e-9))
    segment_score = segment_score.clip(lower=0.0, upper=1.0)
    parts = pd.concat(
        [out["qc_flow_plausible"], conservation_score, segment_score], axis=1
    )
    out["qc_score"] = parts.mean(axis=1, skipna=True)
    out["qc_flag"] = (out["qc_score"] <= config.score_flag_below).astype(float)
    # A vessel with no applicable check is unknown, not passing.
    out.loc[out["qc_score"].isna(), "qc_flag"] = np.nan
    return out.drop(columns=["node", "_conservation_tol"])


def _autoqc_conservation_specs(
    config: AutoQcConfig,
) -> tuple[tuple[str, tuple[str, ...], float], ...]:
    """Resolve rule anchors and class-specific tolerances for this run."""
    # Allow a single CLI ``--conservation-tol`` to retune the arterial gate without losing the
    # distal/venous offsets: keep the relative gaps when the caller overrides the default.
    arterial = float(config.conservation_tol_arterial)
    distal = float(config.conservation_tol_distal)
    venous = float(config.conservation_tol_venous)
    if abs(float(config.conservation_tol) - CONSERVATION_TOL) > 1e-12:
        # Caller overrode the single gate — treat it as the arterial baseline and scale the others.
        scale = float(config.conservation_tol) / max(CONSERVATION_TOL_ARTERIAL, 1e-9)
        arterial = float(config.conservation_tol)
        distal = CONSERVATION_TOL_DISTAL * scale
        venous = CONSERVATION_TOL_VENOUS * scale
    return (
        ("left_carotid_split", ("lica",), arterial),
        ("right_carotid_split", ("rica",), arterial),
        ("basilar_inflow", ("basi", "lva", "rva"), arterial),
        ("posterior_split", ("basi", "lpca", "rpca"), distal),
        ("venous_drainage", ("sss", "strs", "lts", "rts"), venous),
    )


def _conservation_columns(
    frame: pd.DataFrame, config: AutoQcConfig
) -> tuple[pd.Series, pd.Series]:
    """
    Relative conservation residual and the tolerance that scored it, per vessel row.

    Each :data:`AUTOQC_CONSERVATION` rule is evaluated as
    ``(Σ inflow − Σ outflow) / Σ inflow`` (communicating-artery terms dropped — LOC flows are
    unsigned). The residual is written onto every anchor vessel for that rule. When a vessel
    participates in more than one rule (the basilar is both VA confluence and PCA split), the
    **worst** residual by magnitude is kept, together with that rule's class tolerance.
    """
    from nvitk.stats.vessel_network import CONSERVATION_RULES, canonical_node

    node = frame["region_id"].map(canonical_node)
    wide = (
        frame.assign(node=node)
        .dropna(subset=["node"])
        .pivot_table(index="subject_uid", columns="node", values="flow_ml_min", aggfunc="mean")
    )

    # subject → node → (residual, tol)
    best: dict[tuple[str, str], tuple[float, float]] = {}
    for rule_key, anchors, tol in _autoqc_conservation_specs(config):
        rule = CONSERVATION_RULES.get(rule_key)
        if rule is None:
            continue
        terms = {n: c for n, c in rule.terms.items() if n not in rule.signed_terms}
        missing = [n for n in terms if n not in wide.columns]
        if missing:
            log.info(
                "autoqc: junction %s not checkable (missing %s).",
                rule.label, ", ".join(missing),
            )
            continue
        for subject, row in wide.iterrows():
            values = {n: float(row[n]) for n in terms}
            if not all(np.isfinite(v) for v in values.values()):
                continue
            residual = sum(c * values[n] for n, c in terms.items())
            inflow = sum(values[n] for n, c in terms.items() if c > 0)
            if abs(inflow) < 1e-9:
                continue
            rel = float(residual / inflow)
            for anchor in anchors:
                if anchor not in wide.columns:
                    continue
                key = (str(subject), anchor)
                prev = best.get(key)
                if prev is None or abs(rel) > abs(prev[0]):
                    best[key] = (rel, float(tol))

    if not best:
        nan = pd.Series(np.nan, index=frame.index)
        return nan, nan

    residuals = []
    tols = []
    for subject, node_id in zip(frame["subject_uid"], node):
        hit = best.get((str(subject), node_id)) if node_id else None
        if hit is None:
            residuals.append(np.nan)
            tols.append(np.nan)
        else:
            residuals.append(hit[0])
            tols.append(hit[1])
    return (
        pd.Series(residuals, index=frame.index, dtype=float),
        pd.Series(tols, index=frame.index, dtype=float),
    )


#: Junctions the 4D Flow consensus / QVT validation work reports as its internal-consistency
#: check, as ``rule key -> (inlet nodes, outlet nodes)``. The two carotid splits and the venous
#: confluence are exactly the three that paper analyses; the vertebrobasilar pair is ours, and is
#: reported alongside because the same physics applies to it.
CONSENSUS_JUNCTIONS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("left_carotid_split", ("lica",), ("laca", "lmca")),
    ("right_carotid_split", ("rica",), ("raca", "rmca")),
    ("venous_drainage", ("sss", "strs"), ("lts", "rts")),
    ("basilar_inflow", ("lva", "rva"), ("basi",)),
    ("posterior_split", ("basi",), ("lpca", "rpca")),
)

#: Segments the validation work samples for along-segment consistency. Long, non-branching, and
#: large enough that partial volume does not dominate the station-to-station scatter.
CONSENSUS_SEGMENTS: tuple[str, ...] = ("lica", "rica", "sss")


def robust_range(values: Any, *, k: float = 3.0) -> tuple[float, float]:
    """
    Tukey fences ``[Q1 − k·IQR, Q3 + k·IQR]`` — the window holding the bulk of a distribution.

    Preferred over a fixed percentile trim because it does not need the *proportion* of outliers to
    be known in advance: clipping the outer 1% does nothing when 2% of the values are absurd, while
    the fences sit at a fixed distance from the quartiles however many outliers there are. ``k=3``
    is the conventional "extreme outlier" fence.

    Returns ``(-inf, inf)`` when fewer than four finite values make quartiles meaningless.
    """
    x = pd.to_numeric(pd.Series(values), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if x.size < 4:
        return (float("-inf"), float("inf"))
    q1, q3 = (float(v) for v in np.percentile(x, [25, 75]))
    iqr = q3 - q1
    if iqr <= 0:
        return (float("-inf"), float("inf"))
    return (q1 - k * iqr, q3 + k * iqr)


def consensus_junction_report(
    frame: pd.DataFrame,
    *,
    flow_column: str = "flow_ml_min",
    junctions: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]] | None = None,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Cohort-level junction consistency: inlet flow regressed on outlet flow, one row per junction.

    This is the *validation* view of mass conservation, distinct from the per-scan ``qc_conservation``
    residual that stage 9 publishes. The residual asks "is this scan self-consistent"; the
    regression asks "does this pipeline conserve flow across its measuring range" — the question the
    4D Flow consensus statement's internal-consistency check is designed to answer, and the one the
    QVT/CPS validation paper reports as slope, intercept, 95% CI and Pearson *r*.

    Read the **slope**, not the correlation. A pipeline that systematically loses a fixed fraction of
    outflow still correlates near-perfectly — *r* stays above 0.99 while the slope sits at 0.88 —
    so *r* alone certifies nothing about conservation.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long ``subject_uid`` / ``region_id`` / *flow_column* rows, as
        :func:`load_flow_measurements` returns them after renaming.
    junctions : sequence, optional
        ``(key, inlet_nodes, outlet_nodes)`` triples. Defaults to :data:`CONSENSUS_JUNCTIONS`.
    robust : bool
        Drop pairs outside the :func:`robust_range` fences of **either** axis before fitting. A
        single implausible flow — a segmentation that leaked and reported 10⁶ mL/min — sits at
        enormous leverage and can set the slope on its own, so the fenced fit is the honest
        sensitivity check. It is a *different estimate*, not a redrawn view: report it beside the
        full one, never instead of it.

    Returns
    -------
    pandas.DataFrame
        One row per checkable junction: ``junction``, ``label``, ``inlets``, ``outlets``, ``n``,
        ``slope`` with CI, ``intercept`` with CI, ``r``, ``p_value``, ``mean_rel_residual``,
        ``slope_includes_one`` — the actual pass/fail of the check — and ``n_trimmed``, how many
        pairs the trim removed.
    """
    from nvitk.measure.hemodynamics import junction_consistency_regression
    from nvitk.stats.vessel_network import CONSERVATION_RULES, canonical_node

    columns = [
        "junction", "label", "inlets", "outlets", "n", "n_trimmed", "slope", "slope_ci_low",
        "slope_ci_high", "intercept", "intercept_ci_low", "intercept_ci_high", "r", "p_value",
        "mean_rel_residual", "slope_includes_one",
    ]
    if frame is None or frame.empty or flow_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    wide = (
        frame.assign(node=frame["region_id"].map(canonical_node))
        .dropna(subset=["node"])
        .pivot_table(index="subject_uid", columns="node", values=flow_column, aggfunc="mean")
    )

    rows: list[dict[str, Any]] = []
    for key, inlets, outlets in (junctions or CONSENSUS_JUNCTIONS):
        missing = [n for n in (*inlets, *outlets) if n not in wide.columns]
        if missing:
            log.info("consensus: junction %s not checkable (missing %s).", key, ", ".join(missing))
            continue
        # A subject counts only when every vessel of the junction was measured — summing over a
        # missing outlet would manufacture a conservation failure out of an absent column.
        block = wide.loc[:, [*inlets, *outlets]].dropna()
        x = block[list(inlets)].sum(axis=1)
        y = block[list(outlets)].sum(axis=1)

        n_full = int(len(x))
        if robust and n_full >= 10:
            keep = x.between(*robust_range(x)) & y.between(*robust_range(y))
            x, y = x[keep], y[keep]
        stats = junction_consistency_regression(x, y)
        rule = CONSERVATION_RULES.get(key)
        includes_one = (
            bool(stats["slope_ci_low"] <= 1.0 <= stats["slope_ci_high"])
            if np.isfinite(stats["slope_ci_low"]) and np.isfinite(stats["slope_ci_high"])
            else False
        )
        rows.append({
            "junction": key,
            "label": rule.label if rule is not None else key,
            "inlets": " + ".join(inlets),
            "outlets": " + ".join(outlets),
            "slope_includes_one": includes_one,
            "n_trimmed": n_full - int(len(x)),
            **stats,
        })

    if not rows:
        log.warning("consensus: no junction had every one of its vessels measured.")
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).loc[:, columns]


def consensus_segment_report(
    segment_stations: Mapping[tuple[str, str], Sequence[float]] | pd.DataFrame,
    *,
    segments: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Cohort-level along-segment consistency: percent-from-mean flow variation fitted to a Gaussian.

    Mass is conserved along a non-branching segment, so station-to-station flow should barely move.
    Each segment is centred on its own mean and expressed in percent — which puts a sagittal sinus
    and a vertebral on one scale — then all stations are pooled and fitted. The **SD** is the
    result; the QVT validation work reports roughly 3%. A mean far from zero would mean the pooling
    itself is lopsided, not that flow is drifting.

    Parameters
    ----------
    segment_stations : mapping or DataFrame
        Either ``{(subject_uid, region_id): [station flows]}`` or a long frame with
        ``subject_uid`` / ``region_id`` / ``value`` and one row per station.
    segments : sequence of str, optional
        Canonical nodes to include. Defaults to :data:`CONSENSUS_SEGMENTS`; pass ``()`` for all.

    Returns
    -------
    pandas.DataFrame
        One row per segment plus a pooled ``all`` row: ``segment``, ``n_subjects``, ``n_stations``,
        ``mean_pct`` with CI, ``sd_pct`` with CI.
    """
    from nvitk.measure.hemodynamics import gaussian_mvue_fit, percent_variation_from_mean
    from nvitk.stats.vessel_network import canonical_node

    columns = [
        "segment", "n_subjects", "n_stations", "mean_pct", "mean_ci_low", "mean_ci_high",
        "sd_pct", "sd_ci_low", "sd_ci_high",
    ]

    # ---- Normalize either input shape into {(subject, node): stations} -----------------------
    stations: dict[tuple[str, str], list[float]] = {}
    if isinstance(segment_stations, pd.DataFrame):
        needed = {"subject_uid", "region_id", "value"}
        if segment_stations.empty or not needed <= set(segment_stations.columns):
            return pd.DataFrame(columns=columns)
        for (subject, region), group in segment_stations.groupby(
            ["subject_uid", "region_id"], observed=True
        ):
            stations[(str(subject), str(region))] = [
                float(v) for v in pd.to_numeric(group["value"], errors="coerce")
            ]
    else:
        for (subject, region), values in dict(segment_stations).items():
            stations[(str(subject), str(region))] = [float(v) for v in values]

    wanted = CONSENSUS_SEGMENTS if segments is None else tuple(segments)
    by_segment: dict[str, list[np.ndarray]] = {}
    subjects: dict[str, set[str]] = {}
    for (subject, region), values in stations.items():
        node = canonical_node(region)
        if node is None or (wanted and node not in wanted):
            continue
        percent = percent_variation_from_mean(values)
        if percent.size == 0:
            continue
        by_segment.setdefault(node, []).append(percent)
        subjects.setdefault(node, set()).add(subject)

    if not by_segment:
        log.warning("consensus: no along-segment stations for %s.", ", ".join(wanted) or "any node")
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for node in sorted(by_segment):
        pooled = np.concatenate(by_segment[node])
        fit = gaussian_mvue_fit(pooled)
        rows.append({
            "segment": node,
            "n_subjects": len(subjects[node]),
            "n_stations": int(fit["n"]),
            "mean_pct": fit["mean"],
            "mean_ci_low": fit["mean_ci_low"],
            "mean_ci_high": fit["mean_ci_high"],
            "sd_pct": fit["sd"],
            "sd_ci_low": fit["sd_ci_low"],
            "sd_ci_high": fit["sd_ci_high"],
        })

    everything = np.concatenate([p for group in by_segment.values() for p in group])
    fit = gaussian_mvue_fit(everything)
    rows.append({
        "segment": "all",
        "n_subjects": len({s for group in subjects.values() for s in group}),
        "n_stations": int(fit["n"]),
        "mean_pct": fit["mean"],
        "mean_ci_low": fit["mean_ci_low"],
        "mean_ci_high": fit["mean_ci_high"],
        "sd_pct": fit["sd"],
        "sd_ci_low": fit["sd_ci_low"],
        "sd_ci_high": fit["sd_ci_high"],
    })
    return pd.DataFrame(rows).loc[:, columns]


def _segment_cv_column(
    frame: pd.DataFrame, segment_cv: pd.DataFrame | None
) -> pd.Series:
    """Align a long segment-CV frame onto *frame*'s rows (NaN when unavailable)."""
    if segment_cv is None or segment_cv.empty:
        return pd.Series(np.nan, index=frame.index)
    required = {"subject_uid", "region_id", "value"}
    if not required <= set(segment_cv.columns):
        return pd.Series(np.nan, index=frame.index)

    from nvitk.stats.vessel_network import canonical_node

    lookup: dict[tuple[str, str], float] = {}
    for subject, region, value in zip(
        segment_cv["subject_uid"].astype(str),
        segment_cv["region_id"].astype(str),
        pd.to_numeric(segment_cv["value"], errors="coerce"),
    ):
        if not np.isfinite(value):
            continue
        lookup[(subject, region)] = float(value)
        lookup[(subject, region.upper())] = float(value)
        node = canonical_node(region)
        if node:
            lookup.setdefault((subject, node), float(value))

    node = frame["region_id"].map(canonical_node)
    values = []
    for subject, region, node_id in zip(frame["subject_uid"], frame["region_id"], node):
        hit = lookup.get((str(subject), str(region)))
        if hit is None and node_id:
            hit = lookup.get((str(subject), node_id))
        values.append(np.nan if hit is None else hit)
    return pd.Series(values, index=frame.index, dtype=float)


def compute_segment_cv_from_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Along-segment flow CV from a concatenated ``pitc_profile.csv`` frame.

    Groups stations by ``(subject_uid, vessel)``, keeps only the vessels in
    :data:`SEGMENT_CV_NODES`, and returns a long ``(subject_uid, region_id, value)`` frame
    ready for :func:`compute_vessel_qc`.
    """
    from nvitk.stats.vessel_network import canonical_node

    if profiles is None or profiles.empty:
        return pd.DataFrame(columns=["subject_uid", "region_id", "value"])

    work = profiles.copy()
    if "subject_uid" not in work.columns:
        raise ValueError("profiles must carry subject_uid.")
    region_col = next(
        (c for c in ("vessel_name", "region_id", "vessel_id") if c in work.columns),
        None,
    )
    flow_col = next(
        (c for c in ("flow_mean_ml_s", "flow_mean", "value") if c in work.columns),
        None,
    )
    if region_col is None or flow_col is None:
        return pd.DataFrame(columns=["subject_uid", "region_id", "value"])

    work["subject_uid"] = work["subject_uid"].astype(str)
    work["region_id"] = work[region_col].astype(str)
    work["node"] = work["region_id"].map(canonical_node)
    work = work.loc[work["node"].isin(SEGMENT_CV_NODES)].copy()
    work["_flow"] = pd.to_numeric(work[flow_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (subject, region), group in work.groupby(["subject_uid", "region_id"], sort=False):
        cv = segment_flow_consistency_cv(group["_flow"].to_numpy())
        if not np.isfinite(cv):
            continue
        rows.append({"subject_uid": str(subject), "region_id": str(region), "value": float(cv)})
    return pd.DataFrame(rows, columns=["subject_uid", "region_id", "value"])


def compute_subject_qc(
    vessel_qc: pd.DataFrame, *, config: AutoQcConfig | None = None
) -> pd.DataFrame:
    """
    Subject-level QC: the anterior/posterior inflow split, and a roll-up of the vessel flags.

    The split is the cheapest screen available — 72/28% with an SD of 4–5%, stable across age,
    sex and brain volume — and needs no reference scan. It is computed from the summed
    bilateral carotids against the basilar (or the vertebrals when the basilar is absent),
    and skipped when either side is missing rather than computed from half the inflow.
    """
    from nvitk.stats.vessel_network import canonical_node

    config = config or AutoQcConfig()
    node = vessel_qc["region_id"].map(canonical_node)
    wide = (
        vessel_qc.assign(node=node)
        .dropna(subset=["node"])
        .pivot_table(index="subject_uid", columns="node", values="flow_ml_min", aggfunc="mean")
    )

    anterior = _sum_if_all(wide, ("lica", "rica"))
    posterior = _sum_if_all(wide, ("basi",))
    if posterior is None:
        posterior = _sum_if_all(wide, ("lva", "rva"))

    out = pd.DataFrame(index=wide.index)
    if anterior is None or posterior is None:
        log.warning(
            "autoqc: cannot compute the anterior/posterior split — need both carotids and "
            "either the basilar or both vertebrals."
        )
        out["qc_ap_share"] = np.nan
        out["qc_ap_flag"] = np.nan
    else:
        out["qc_ap_share"] = [
            anterior_posterior_share_pct(a, p) for a, p in zip(anterior, posterior)
        ]
        out["qc_ap_flag"] = [
            float(anterior_posterior_split_flag(
                a, p, expected_anterior_pct=config.anterior_pct,
                tolerance_pct=config.anterior_tol_pct,
            ))
            for a, p in zip(anterior, posterior)
        ]

    vessel_flags = (
        vessel_qc.groupby("subject_uid")["qc_flag"].max(numeric_only=False).reindex(out.index)
    )
    out["qc_subject_flag"] = (
        (vessel_flags.fillna(0.0) > 0) | (out["qc_ap_flag"].fillna(0.0) > 0)
    ).astype(float)
    return out.reset_index()


def _sum_if_all(wide: pd.DataFrame, nodes: Sequence[str]) -> pd.Series | None:
    """Row-wise sum of *nodes*, or ``None`` when any of them is absent from the frame."""
    if not all(n in wide.columns for n in nodes):
        return None
    return wide.loc[:, list(nodes)].sum(axis=1, min_count=len(nodes))


# ---------------------------------------------------------------------------
# On-the-fly scoring, for a frame that has not been through the stage
# ---------------------------------------------------------------------------
#: Flow at or above which a cohort's median is taken to be mL/min rather than mL/s.
#: A healthy ICA is ~257 mL/min = ~4.3 mL/s, so the two scales are two orders of magnitude apart
#: and anything in between would be a cohort no unit assignment could rescue.
FLOW_SCALE_BOUNDARY: float = 20.0


def infer_flow_scale(values: Any) -> float:
    """
    Factor converting *values* to mL/min, inferred from their magnitude.

    Stage 6 publishes flow per **second**; the literature bands are per **minute**. A frame assembled
    in the GUI carries whichever the dataset stored, and scoring mL/s against a mL/min band would
    fail every vessel in the cohort — so the scale is inferred rather than assumed, and logged.

    Returns 1.0 when the values already look like mL/min, 60.0 when they look like mL/s.
    """
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").abs().dropna()
    if numeric.empty:
        return 1.0
    median = float(numeric.median())
    scale = 1.0 if median >= FLOW_SCALE_BOUNDARY else 60.0
    log.info(
        "autoqc: median flow %.4g → treating the column as %s (×%g to mL/min).",
        median, "mL/min" if scale == 1.0 else "mL/s", scale,
    )
    return scale


def compute_qc_columns(
    frame: pd.DataFrame,
    *,
    flow_column: str = "flow_mean",
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    area_column: str = "",
    config: AutoQcConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Score an analysis frame in place, without going through the dataset.

    The stage is the right way to do this — it writes the metrics once, for everyone. But a frame
    assembled in the GUI on a dataset where the stage has not run yet still deserves its QC filters,
    and computing them at the moment they are asked for is better than greying the option out.

    The results are **not** published: they are columns on this frame only. Running the stage remains
    the way to make them available to every later session.

    Parameters
    ----------
    flow_column : str
        The measurement to score. Its unit is inferred by :func:`infer_flow_scale`.
    area_column : str
        Cross-sectional area, used only to excuse hypoplastic vessels. Without it every vessel is
        scored as patent, which over-flags a normal circle of Willis.

    Returns
    -------
    (frame, added)
        A copy carrying the new ``qc_*`` columns, and their names. Existing ``qc_*`` columns are
        replaced, so re-running after a reload refreshes rather than duplicating them.

    Raises
    ------
    ValueError
        When the frame lacks the subject, region or flow column.
    """
    config = config or AutoQcConfig()
    for column, role in ((subject_column, "subject"), (region_column, "region"), (flow_column, "flow")):
        if column not in frame.columns:
            raise ValueError(
                f"Cannot score this frame: no {role} column {column!r}. Quality control needs a "
                f"vessel-wise flow measurement — load one, or run the qvtpy stage 9 (autoqc)."
            )

    long = pd.DataFrame({
        "subject_uid": frame[subject_column].astype(str),
        "region_id": frame[region_column].astype(str),
        "value": pd.to_numeric(frame[flow_column], errors="coerce"),
    }).dropna(subset=["value"])
    if long.empty:
        raise ValueError(f"{flow_column!r} has no numeric values to score.")

    scoring = AutoQcConfig(
        flow_variable=config.flow_variable,
        area_variable=config.area_variable,
        flow_to_ml_min=infer_flow_scale(long["value"]),  # explicit: the frame is already assembled
        conservation_tol=config.conservation_tol,
        score_flag_below=config.score_flag_below,
        anterior_pct=config.anterior_pct,
        anterior_tol_pct=config.anterior_tol_pct,
    )

    areas = None
    if area_column and area_column in frame.columns:
        areas = pd.DataFrame({
            "subject_uid": frame[subject_column].astype(str),
            "region_id": frame[region_column].astype(str),
            "value": pd.to_numeric(frame[area_column], errors="coerce"),
        }).dropna(subset=["value"])

    vessel = compute_vessel_qc(long, areas, config=scoring)
    subject = compute_subject_qc(vessel, config=scoring)

    vessel_columns = [c for c in QC_VARIABLES if c in vessel.columns]
    subject_columns = [c for c in QC_VARIABLES if c in subject.columns]

    out = frame.copy()
    added = [*vessel_columns, *subject_columns]
    out = out.drop(columns=[c for c in added if c in out.columns], errors="ignore")
    # Merge on the frame's own key names, which need not be the canonical ones.
    out["_qc_subject"] = out[subject_column].astype(str)
    out["_qc_region"] = out[region_column].astype(str)
    out = out.merge(
        vessel.loc[:, ["subject_uid", "region_id", *vessel_columns]].rename(
            columns={"subject_uid": "_qc_subject", "region_id": "_qc_region"}
        ),
        on=["_qc_subject", "_qc_region"], how="left",
    )
    if subject_columns:
        out = out.merge(
            subject.loc[:, ["subject_uid", *subject_columns]].rename(
                columns={"subject_uid": "_qc_subject"}
            ),
            on="_qc_subject", how="left",
        )
    out = out.drop(columns=["_qc_subject", "_qc_region"])
    log.info(
        "autoqc: scored %d row(s) on the fly from %r — added %s.",
        len(out), flow_column, ", ".join(added),
    )
    return out, added


# ---------------------------------------------------------------------------
# Reading and writing the dataset
# ---------------------------------------------------------------------------
def load_flow_measurements(
    repo: Any, *, variable_id: str, pipeline: str = "latest"
) -> pd.DataFrame:
    """
    Long ``(subject_uid, region_id, value)`` rows for one image variable, or an empty frame.

    Wraps :meth:`~nvitk.db.repo.DataRepo.image` so a missing variable is a reported empty
    result rather than an exception — the area measurement is optional, and a run that has it
    should not be blocked by a cohort that does not.
    """
    empty = pd.DataFrame(columns=["subject_uid", "region_id", "value"])
    # The long table directly rather than ``DataRepo.image``: that accessor applies catalog
    # default-pipeline and cohort restrictions and expands regions through atlas presets, none of
    # which this stage wants — it needs every published row for one variable, whatever cohort the
    # subject belongs to and whether or not the pipeline is a catalog default.
    try:
        frame = repo.get("image_measurements", cohort_id=False)
    except Exception as exc:
        log.warning("autoqc: could not read image_measurements (%s).", exc)
        log.debug("image_measurements read failed", exc_info=True)
        return empty
    if frame is None or frame.empty or "variable_id" not in frame.columns:
        return empty

    rows = frame.loc[frame["variable_id"].astype(str) == str(variable_id)]
    wanted = str(pipeline).strip().lower()
    if wanted not in {"", "latest", "any"} and "pipeline_id" in rows.columns:
        selected = rows.loc[rows["pipeline_id"].astype(str) == str(pipeline)]
        if selected.empty:
            available = sorted({str(p) for p in rows["pipeline_id"].dropna()})
            log.warning(
                "autoqc: no %r rows for pipeline %r — available: %s. Using every pipeline.",
                variable_id, pipeline, ", ".join(available) or "none",
            )
        else:
            rows = selected
    if rows.empty:
        return empty

    value_column = next(
        (c for c in ("value_num", "value") if c in rows.columns), None
    )
    if value_column is None or not {"subject_uid", "region_id"} <= set(rows.columns):
        log.warning(
            "autoqc: %r rows lack a value or region column (have %s).",
            variable_id, list(rows.columns),
        )
        return empty
    out = rows.loc[:, ["subject_uid", "region_id", value_column]].rename(
        columns={value_column: "value"}
    )
    # One subject can carry several frames of the same vessel; the QC works on the time-average.
    return out.groupby(["subject_uid", "region_id"], as_index=False)["value"].mean()


def purge_subject_qc(
    repo: Any,
    subjects: Sequence[str],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Drop every ``qc_*`` row this stage owns for *subjects*, so a re-run cannot leave stale values.

    The upsert that follows overwrites any row whose
    ``(subject_uid, region_id, variable_id, frame_index)`` it re-emits — but a metric that was
    computable last time and is **not** this time is simply not re-emitted, and its old value would
    survive. That is the dangerous case: a junction that no longer has all its vessels stops
    producing a ``qc_conservation`` row, and without this purge the dataset keeps reporting the
    previous run's residual as if it were current.

    Absence is the honest representation of "not evaluated", so the subjects in scope are cleared
    first and only what this run actually computed is written back.

    Only the variables in :data:`QC_VARIABLES` and only the subjects passed in are touched — every
    other measurement, and every other subject's QC, is left alone. Rows are matched on
    ``variable_id`` regardless of ``pipeline_id``: this stage is the sole producer of ``qc_*``, so a
    row carrying a different pipeline id is a leftover from an earlier naming, not a parallel result.

    Returns
    -------
    dict
        ``{table: rows_removed}`` for the tables that actually changed.
    """
    wanted = {str(s) for s in subjects}
    removed: dict[str, int] = {}
    if not wanted:
        return removed

    for table in sorted({t for t in QC_VARIABLES.values()}):
        variables = {v for v, t in QC_VARIABLES.items() if t == table}
        try:
            # Parquet, not SQLite: the index may lag when a prior write skipped the rebuild.
            frame = repo.get(table, cohort_id=False, use_sqlite=False)
        except Exception as exc:
            log.warning("autoqc: could not read %s to clear previous QC rows (%s).", table, exc)
            continue
        if frame is None or frame.empty:
            continue
        if not {"variable_id", "subject_uid"} <= set(frame.columns):
            continue
        stale = frame["variable_id"].astype(str).isin(variables) & frame[
            "subject_uid"
        ].astype(str).isin(wanted)
        n_stale = int(stale.sum())
        if not n_stale:
            continue
        removed[table] = n_stale
        if not dry_run:
            repo.write_table(
                table,
                frame.loc[~stale],
                provenance={"importer": STAGE_NAME, "action": "purge_stale_qc"},
                build_sqlite_index=False,
            )
    if removed:
        log.info(
            "autoqc: cleared %s previous QC row(s) for %d subject(s)%s.",
            ", ".join(f"{n} from {t}" for t, n in sorted(removed.items())),
            len(wanted),
            " (dry run — nothing removed)" if dry_run else "",
        )
    return removed


def publish_autoqc(
    repo: Any,
    vessel_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    *,
    pipeline_id: str = "qvtpy",
    dry_run: bool = False,
    purge_existing: bool = True,
) -> dict[str, int]:
    """
    Write the QC metrics into the dataset as ordinary measurements.

    Everything goes through :mod:`nvitk.db.derived_measurements`, so the rows land with the
    same schema, provenance and catalog registration as any importer-produced variable and
    become immediately queryable — which is what lets the Statmodels filter and the QC panel's
    colour picker use them without knowing this stage exists.

    With *purge_existing* (the default) every ``qc_*`` row for the subjects being published is
    dropped first — see :func:`purge_subject_qc` for why an upsert alone is not enough. Pass
    ``False`` only to add QC for new subjects without touching what is already stored.

    The SQLite index is rebuilt **once, after every table is written**, not per variable. Each
    rebuild re-reads the whole Parquet table, so leaving the default on would do it eight times for
    a single pass — the dominant cost of the run on a full cohort.

    Returns
    -------
    dict
        ``{variable_id: rows_written}``.
    """
    from nvitk.db.derived_measurements import (
        DerivedClinicalMeasurementSpec,
        DerivedImageMeasurementSpec,
        DerivedVariableRegistration,
        build_clinical_measurement_rows,
        build_image_measurement_rows,
        publish_derived_measurements,
    )

    written: dict[str, int] = {}
    touched: set[str] = set()
    provenance = {"importer": STAGE_NAME, "parent_variable": "flow_mean"}

    if purge_existing:
        subjects = {
            str(s)
            for frame in (vessel_qc, subject_qc)
            if frame is not None and "subject_uid" in getattr(frame, "columns", [])
            for s in frame["subject_uid"].astype(str)
        }
        # A purge that removes rows is itself a change to the table, so the SQLite index has to be
        # rebuilt even when no variable ends up producing a row to write.
        touched.update(purge_subject_qc(repo, sorted(subjects), dry_run=dry_run))

    for variable in (v for v, t in QC_VARIABLES.items() if t == "image_measurements"):
        if variable not in vessel_qc.columns:
            continue
        agg = vessel_qc.loc[:, ["subject_uid", "region_id", variable]].dropna(subset=[variable])
        if agg.empty:
            log.info("autoqc: %s has no values to publish.", variable)
            continue
        agg = agg.rename(columns={variable: "value_num"})
        spec = DerivedImageMeasurementSpec(
            variable_id=variable, modality="4dflow", pipeline_id=pipeline_id,
            source_file=STAGE_NAME, source_sheet="autoqc", source_column=variable,
            value_kind="float", pipeline_name=f"{pipeline_id}_autoqc",
        )
        rows = build_image_measurement_rows(agg, spec)
        written[variable] = int(len(rows))
        if not dry_run:
            publish_derived_measurements(
                repo, rows, table="image_measurements",
                register=DerivedVariableRegistration.from_image_spec(
                    spec, label=QC_LABELS.get(variable, variable), aliases=[variable],
                ),
                provenance=provenance,
                upsert_key_columns=["subject_uid", "region_id", "variable_id", "frame_index"],
                build_sqlite_index=False,
            )
            touched.add("image_measurements")

    for variable in (v for v, t in QC_VARIABLES.items() if t == "clinical_measurements"):
        if variable not in subject_qc.columns:
            continue
        agg = subject_qc.loc[:, ["subject_uid", variable]].dropna(subset=[variable])
        if agg.empty:
            continue
        agg = agg.rename(columns={variable: "value_num"})
        spec = DerivedClinicalMeasurementSpec(
            variable_id=variable, source_file=STAGE_NAME, source_sheet="autoqc",
            source_column=variable, value_kind="float",
        )
        rows = build_clinical_measurement_rows(agg, spec)
        written[variable] = int(len(rows))
        if not dry_run:
            publish_derived_measurements(
                repo, rows, table="clinical_measurements",
                register=DerivedVariableRegistration(
                    variable_id=variable, domain="clinical", table="clinical_measurements",
                    label=QC_LABELS.get(variable, variable), source_column=variable,
                    source_file=STAGE_NAME, source_sheet="autoqc", value_kind="float",
                ),
                provenance=provenance,
                upsert_key_columns=["subject_uid", "visit_id", "variable_id"],
                build_sqlite_index=False,
            )
            touched.add("clinical_measurements")

    if touched and not dry_run:
        # Once, for the tables actually written. Without this the Parquet holds the new metrics but
        # every SQLite-backed read still returns the old ones — the GUI reads through SQLite, so the
        # filters would silently score against stale values.
        log.info("Rebuilding the SQLite index for %s …", ", ".join(sorted(touched)))
        repo.build_sqlite_index(tables=sorted(touched))
        log.ok("SQLite index rebuilt.")
    return written


def run_autoqc(
    repo: Any,
    *,
    config: AutoQcConfig | None = None,
    pipeline: str = "latest",
    dry_run: bool = False,
    results: Any = None,
    subjects: Sequence[str] | None = None,
    purge_existing: bool = True,
) -> dict[str, Any]:
    """
    Compute and publish every automatic QC metric for the cohort in *repo*.

    Parameters
    ----------
    purge_existing : bool
        Clear each published subject's previous ``qc_*`` rows before writing, so a metric that is
        no longer computable disappears instead of keeping its stale value
        (:func:`purge_subject_qc`). Turn off only to add QC for new subjects without touching
        what is already stored.
    results : ResultsSource, optional
        Where to look for measurements the dataset does not carry, and for ``pitc_profile.csv``
        station flows used by the along-segment CV check. Stage 6 writes the numbers to disk
        before anything imports them, so a dataset whose import has not run — or has run only
        for the flows, leaving no areas to excuse hypoplastic vessels — can still be scored.
        Without it, a missing variable is simply missing and ``qc_segment_cv`` stays NaN.

    Returns
    -------
    dict
        ``vessel`` and ``subject`` frames, ``written`` counts, a short ``summary``, and ``recovery``
        describing anything read from the results tree.
    """
    config = config or AutoQcConfig()
    flows = load_flow_measurements(repo, variable_id=config.flow_variable, pipeline=pipeline)
    areas = load_flow_measurements(repo, variable_id=config.area_variable, pipeline=pipeline)

    recovery: dict[str, Any] = {}
    missing = [
        name for name, frame in (
            (config.flow_variable, flows), (config.area_variable, areas)
        ) if frame.empty
    ]
    segment_cv = pd.DataFrame(columns=["subject_uid", "region_id", "value"])
    if results is not None:
        from .autoqc_sources import long_measurements, load_pitc_profiles, open_results

        try:
            loc, profile_root = open_results(results, subjects)
            recovery = {
                "root": str(profile_root),
                "n_subjects": int(loc["subject_uid"].nunique()) if not loc.empty else 0,
                "recovered": [],
                "unavailable": [],
            }
            if missing:
                log.info(
                    "autoqc: %s not in the dataset — recovering from the results tree.",
                    ", ".join(missing),
                )
                recovered: dict[str, pd.DataFrame] = {}
                for variable in missing:
                    frame = long_measurements(loc, variable)
                    if not frame.empty:
                        recovered[variable] = frame
                recovery["recovered"] = sorted(recovered)
                recovery["unavailable"] = sorted(set(missing) - set(recovered))
                if config.flow_variable in recovered:
                    flows = recovered[config.flow_variable]
                if config.area_variable in recovered:
                    areas = recovered[config.area_variable]
                if recovered:
                    log.ok(
                        "autoqc: recovered %s from the results tree (%d subject(s)).",
                        ", ".join(recovery["recovered"]), recovery["n_subjects"],
                    )

            wanted = subjects
            if wanted is None and not flows.empty:
                wanted = sorted(flows["subject_uid"].astype(str).unique())
            profiles = load_pitc_profiles(profile_root, subjects=wanted)
            segment_cv = compute_segment_cv_from_profiles(profiles)
            if segment_cv.empty:
                log.info(
                    "autoqc: no along-segment CV computable from pitc_profile.csv under %s "
                    "(need ≥3 stations on ICA / basilar / SSS).",
                    profile_root,
                )
            else:
                log.info(
                    "autoqc: along-segment CV for %d vessel(s) from pitc_profile.csv.",
                    len(segment_cv),
                )
        except Exception as exc:
            log.warning("autoqc: results tree unavailable for recovery / segment CV (%s).", exc)
        finally:
            results.cleanup()

    if flows.empty:
        raise ValueError(
            f"No {config.flow_variable!r} measurements in the dataset or the results tree — run "
            f"stage 6 (and its import), or point --results-root at the tree that has them."
        )

    vessel = compute_vessel_qc(
        flows,
        areas if not areas.empty else None,
        config=config,
        segment_cv=segment_cv if not segment_cv.empty else None,
    )
    subject = compute_subject_qc(vessel, config=config)
    written = publish_autoqc(
        repo, vessel, subject, dry_run=dry_run, purge_existing=purge_existing
    )

    n_flagged = int(pd.to_numeric(vessel["qc_flag"], errors="coerce").fillna(0).sum())
    n_subjects_flagged = int(pd.to_numeric(subject["qc_subject_flag"], errors="coerce").fillna(0).sum())
    summary = (
        f"{n_flagged} of {len(vessel)} vessel measurements flagged; "
        f"{n_subjects_flagged} of {len(subject)} subjects have at least one failing check."
    )
    log.ok("autoqc: %s%s", summary, " (dry run — nothing written)" if dry_run else "")
    return {
        "vessel": vessel, "subject": subject, "written": written, "summary": summary,
        "dry_run": bool(dry_run), "recovery": recovery,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_subjects(text: str | None) -> list[str] | None:
    """Comma/space-separated subject ids, or ``None`` for every subject found."""
    if not text:
        return None
    parts = [p.strip() for p in str(text).replace(",", " ").split()]
    return [p for p in parts if p] or None


def _results_source(
    *,
    submit: str,
    results_root: Any = None,
    remote_host: str | None = None,
    remote_user: str | None = None,
    xnat_config_path: Any = None,
    xnat_server: str | None = None,
    xnat_project: str | None = None,
    xnat_user: str | None = None,
    xnat_password: str | None = None,
) -> Any:
    """
    Build the recovery source, resolving credentials only for the mode actually asked for.

    Lazily and never stored: a local run must not prompt, and a remote one should ask once at the
    point it needs to connect.
    """
    from .autoqc_sources import ResultsSource

    source = ResultsSource(submit=submit, results_root=results_root)
    if not source.is_remote():
        return source

    if source.is_xnat():
        from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

        profile = load_xnat_profile(xnat_config_path) if xnat_config_path else load_xnat_profile()
        source.xnat_config = resolve_xnat_connection(
            profile, server=xnat_server, project=xnat_project,
            user=xnat_user, password=xnat_password,
        )
        source.xnat_project = xnat_project or ""
        return source

    from nvitk.pipes.qvtpy import config as _cfg
    from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials

    host, user, password = prompt_ssh_credentials(
        remote_host=remote_host, remote_user=remote_user,
        host_aliases=getattr(_cfg, "CLUSTER_HOST_ALIASES", {}),
    )
    source.host, source.user, source.password = host, user, password
    return source


def _open_repo(dataset: Any = None) -> Any:
    """The dataset to work on: an explicit root, else whatever the settings resolve to."""
    from nvitk.db.repo import DataRepo, get_repo_from_settings

    if dataset:
        root = Path(dataset).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return DataRepo(root)
    return get_repo_from_settings()


@click.command("nvitk-qvtpy-autoqc")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the one configured in .nvitk/settings.json.",
)
@click.option(
    "--pipeline",
    default="latest",
    show_default=True,
    help="image_measurements pipeline to read (alias or explicit pipeline_id).",
)
@click.option(
    "--flow-variable",
    default="flow_mean",
    show_default=True,
    help="Time-averaged flow variable the plausibility and conservation checks read.",
)
@click.option(
    "--area-variable",
    default="cross_section_area",
    show_default=True,
    help=(
        "Cross-sectional area variable used to excuse hypoplastic vessels. Without it every "
        "vessel is scored as patent, which over-flags a normal circle of Willis."
    ),
)
@click.option(
    "--flow-scale",
    type=float,
    default=None,
    help=(
        "Factor converting the stored flow to mL/min. Omit to infer it from the magnitude "
        "(recommended): stage 6 emits mL/s, but several published tables are already per minute, "
        "and assuming the wrong one mis-scales the whole cohort by 60x."
    ),
)
@click.option(
    "--conservation-tol",
    type=float,
    default=CONSERVATION_TOL,
    show_default=True,
    help=(
        "Relative residual gate for proximal arterial junctions (ICA, VA→BA). Distal "
        "(BA→PCA) and venous gates scale with this; defaults are 10% / 15% / 20%."
    ),
)
@click.option(
    "--segment-cv-tol",
    type=float,
    default=SEGMENT_CV_TOL,
    show_default=True,
    help="Along-segment flow CV beyond which a vessel is considered inconsistent.",
)
@click.option(
    "--score-flag-below",
    type=float,
    default=QC_SCORE_FLAG_BELOW,
    show_default=True,
    help="Combined score at or below which a vessel is flagged for review.",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge", "xnat"]),
    default="local",
    show_default=True,
    help=(
        "Where to recover measurements the dataset is missing from. 'local' reads the results root "
        "on this machine; 'sge' fetches the stage-6 CSVs from the cluster over SFTP; 'xnat' "
        "downloads each session's qvtpy resource. Both remote modes stage into a temporary "
        "directory that is removed afterwards."
    ),
)
@click.option(
    "--results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Results root to recover from. Defaults to the configured root for the chosen --submit.",
)
@click.option(
    "--subjects",
    default=None,
    help="Optional comma/space-separated subject ids to restrict the recovery to.",
)
@click.option("--remote-host", default=None, help="(sge) SSH host or alias.")
@click.option("--remote-user", default=None, help="(sge) SSH user.")
@click.option(
    "--xnat-config",
    "xnat_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="(xnat) Connection profile JSON. Falls back to the configured default.",
)
@click.option("--xnat-server", default=None, help="(xnat) Server URL override.")
@click.option("--xnat-project", default=None, help="(xnat) Restrict to one project.")
@click.option("--xnat-user", default=None, help="(xnat) Username override.")
@click.option("--xnat-password", default=None, help="(xnat) Password override.")
@click.option(
    "--no-recover",
    is_flag=True,
    default=False,
    help="Score only what the dataset carries; never read the results tree.",
)
@click.option(
    "--purge-existing/--no-purge-existing",
    default=True,
    show_default=True,
    help=(
        "Clear each scored subject's previous qc_* rows before writing. Keeps the dataset honest "
        "on a re-run: a metric that is no longer computable disappears instead of keeping its "
        "stale value. Disable to add QC for new subjects without touching existing rows."
    ),
)
@click.option(
    "--dry-run/--write",
    default=False,
    show_default=True,
    help="Compute and report without writing anything to the dataset.",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional CSV to write the per-vessel scores to, for inspection outside the GUI.",
)
def main(
    dataset: Path | None,
    pipeline: str,
    flow_variable: str,
    area_variable: str,
    flow_scale: float | None,
    conservation_tol: float,
    segment_cv_tol: float,
    score_flag_below: float,
    submit: str,
    results_root: Path | None,
    subjects: str | None,
    remote_host: str | None,
    remote_user: str | None,
    xnat_config_path: Path | None,
    xnat_server: str | None,
    xnat_project: str | None,
    xnat_user: str | None,
    xnat_password: str | None,
    no_recover: bool,
    purge_existing: bool,
    dry_run: bool,
    report: Path | None,
) -> None:
    """
    Score published 4D-flow measurements against literature bands and physiological consistency.

    Reads what stage 6 published, writes the QC metrics back as ordinary measurements, and leaves
    the pipeline outputs untouched — so it is safe to re-run after any re-import.
    """
    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(
            f"Could not open the dataset ({exc}). Pass --dataset PATH, or configure one in "
            f".nvitk/settings.json."
        ) from exc

    config = AutoQcConfig(
        flow_variable=flow_variable,
        area_variable=area_variable,
        flow_to_ml_min=flow_scale,
        conservation_tol=conservation_tol,
        segment_cv_tol=segment_cv_tol,
        score_flag_below=score_flag_below,
    )
    source = None
    if not no_recover:
        source = _results_source(
            submit=submit, results_root=results_root,
            remote_host=remote_host, remote_user=remote_user,
            xnat_config_path=xnat_config_path, xnat_server=xnat_server,
            xnat_project=xnat_project, xnat_user=xnat_user, xnat_password=xnat_password,
        )
    wanted = _parse_subjects(subjects)

    try:
        result = run_autoqc(
            repo, config=config, pipeline=pipeline, dry_run=dry_run,
            results=source, subjects=wanted, purge_existing=purge_existing,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    recovery = result.get("recovery") or {}
    if recovery.get("recovered"):
        log.info(
            "Recovered %s from %s (%d subject(s)).",
            ", ".join(recovery["recovered"]), recovery["root"], recovery["n_subjects"],
        )
    if recovery.get("unavailable"):
        log.warning(
            "Could not recover %s — the results tree has no stage-6 column for it.",
            ", ".join(recovery["unavailable"]),
        )

    if report is not None:
        out = Path(report)
        out.parent.mkdir(parents=True, exist_ok=True)
        result["vessel"].to_csv(out, index=False)
        log.info("Wrote the per-vessel scores to %s", out)

    written = result["written"]
    if dry_run:
        log.info(
            "Dry run — would write %d row(s) across %d variable(s).",
            sum(written.values()), len(written),
        )
    else:
        log.info(
            "Wrote %d row(s) across %d variable(s): %s",
            sum(written.values()), len(written), ", ".join(sorted(written)),
        )


__all__ = [
    "AUTOQC_CONSERVATION",
    "FLOW_SCALE_BOUNDARY",
    "QC_LABELS",
    "QC_SCORE_FLAG_BELOW",
    "QC_VARIABLES",
    "SEGMENT_CV_NODES",
    "STAGE_NAME",
    "AutoQcConfig",
    "compute_qc_columns",
    "compute_segment_cv_from_profiles",
    "compute_subject_qc",
    "compute_vessel_qc",
    "infer_flow_scale",
    "load_flow_measurements",
    "publish_autoqc",
    "purge_subject_qc",
    "main",
    "run_autoqc",
]


if __name__ == "__main__":  # pragma: no cover
    main()
