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
    anterior_posterior_share_pct,
    anterior_posterior_split_flag,
    bifurcation_conservation_error,
    flow_plausibility_score,
    is_plausibly_hypoplastic,
)

log = Logger()

STAGE_NAME = "stage9_autoqc"

#: Variable ids this stage publishes, and the table each belongs to.
QC_VARIABLES: dict[str, str] = {
    "qc_flow_plausible": "image_measurements",
    "qc_hypoplastic": "image_measurements",
    "qc_conservation": "image_measurements",
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
    "qc_score": "Combined per-vessel QC score (0–1)",
    "qc_flag": "Vessel QC flag (1 = review)",
    "qc_ap_share": "Anterior share of cerebral inflow (%)",
    "qc_ap_flag": "Anterior/posterior split flag (1 = outside 72±10%)",
    "qc_subject_flag": "Subject QC flag (1 = any vessel or subject check failed)",
}

#: Score at or below which a vessel is flagged for review.
QC_SCORE_FLAG_BELOW: float = 0.5

#: Junctions checked for mass conservation, as ``parent: (branches...)`` in canonical ids.
#: The vertebrals feed the basilar; each carotid splits into its ACA and MCA.
CONSERVATION_JUNCTIONS: dict[str, tuple[str, ...]] = {
    "basi": ("lva", "rva"),
    "lica": ("laca", "lmca"),
    "rica": ("raca", "rmca"),
}


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

    Returns
    -------
    pandas.DataFrame
        One row per ``(subject_uid, region_id)`` with ``qc_flow_plausible``,
        ``qc_hypoplastic``, ``qc_conservation``, ``qc_score`` and ``qc_flag``.
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

    # ---- 3. Mass conservation at each junction, attributed to the parent vessel -------
    out["qc_conservation"] = _conservation_column(out, config)

    # ---- 4. Combined score: the mean of whichever checks applied ----------------------
    # A conservation residual is an error, not a score, so it is mapped onto [0, 1] by how
    # far past the tolerance it sits before being averaged with the plausibility.
    conservation_score = 1.0 - (out["qc_conservation"].abs() / max(config.conservation_tol, 1e-9))
    conservation_score = conservation_score.clip(lower=0.0, upper=1.0)
    parts = pd.concat([out["qc_flow_plausible"], conservation_score], axis=1)
    out["qc_score"] = parts.mean(axis=1, skipna=True)
    out["qc_flag"] = (out["qc_score"] <= config.score_flag_below).astype(float)
    # A vessel with no applicable check is unknown, not passing.
    out.loc[out["qc_score"].isna(), "qc_flag"] = np.nan
    return out.drop(columns=["node"])


def _conservation_column(frame: pd.DataFrame, config: AutoQcConfig) -> pd.Series:
    """
    Relative conservation residual per subject, written onto the *parent* vessel's row.

    Attributing it to the parent is a choice: the residual is a property of the junction, and
    the parent is the one row every junction has. A branch's own row carries NaN, so a
    conservation failure never double-counts against the vessels downstream of it.
    """
    from nvitk.stats.vessel_network import canonical_node

    node = frame["region_id"].map(canonical_node)
    wide = (
        frame.assign(node=node)
        .dropna(subset=["node"])
        .pivot_table(index="subject_uid", columns="node", values="flow_ml_min", aggfunc="mean")
    )

    residuals: dict[tuple[str, str], float] = {}
    for parent, branches in CONSERVATION_JUNCTIONS.items():
        if parent not in wide.columns or not all(b in wide.columns for b in branches):
            log.info(
                "autoqc: junction %s → %s not checkable (missing vessels).",
                parent, " + ".join(branches),
            )
            continue
        for subject, row in wide.iterrows():
            residuals[(str(subject), parent)] = bifurcation_conservation_error(
                row[parent], [row[b] for b in branches]
            )

    if not residuals:
        return pd.Series(np.nan, index=frame.index)
    return pd.Series(
        [residuals.get((s, n), np.nan) for s, n in zip(frame["subject_uid"], node)],
        index=frame.index,
    )


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


def publish_autoqc(
    repo: Any,
    vessel_qc: pd.DataFrame,
    subject_qc: pd.DataFrame,
    *,
    pipeline_id: str = "qvtpy",
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Write the QC metrics into the dataset as ordinary measurements.

    Everything goes through :mod:`nvitk.db.derived_measurements`, so the rows land with the
    same schema, provenance and catalog registration as any importer-produced variable and
    become immediately queryable — which is what lets the Statmodels filter and the QC panel's
    colour picker use them without knowing this stage exists.

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
) -> dict[str, Any]:
    """
    Compute and publish every automatic QC metric for the cohort in *repo*.

    Parameters
    ----------
    results : ResultsSource, optional
        Where to look for measurements the dataset does not carry. Stage 6 writes the numbers to
        disk before anything imports them, so a dataset whose import has not run — or has run only
        for the flows, leaving no areas to excuse hypoplastic vessels — can still be scored. Without
        it, a missing variable is simply missing.

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
    if missing and results is not None:
        from .autoqc_sources import recover_missing

        log.info(
            "autoqc: %s not in the dataset — recovering from the results tree.",
            ", ".join(missing),
        )
        try:
            recovered, recovery = recover_missing(missing, results, subjects=subjects)
        finally:
            results.cleanup()
        if config.flow_variable in recovered:
            flows = recovered[config.flow_variable]
        if config.area_variable in recovered:
            areas = recovered[config.area_variable]

    if flows.empty:
        raise ValueError(
            f"No {config.flow_variable!r} measurements in the dataset or the results tree — run "
            f"stage 6 (and its import), or point --results-root at the tree that has them."
        )

    vessel = compute_vessel_qc(flows, areas if not areas.empty else None, config=config)
    subject = compute_subject_qc(vessel, config=config)
    written = publish_autoqc(repo, vessel, subject, dry_run=dry_run)

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
    help="Relative junction residual beyond which mass conservation is considered violated.",
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
            results=source, subjects=wanted,
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
    "CONSERVATION_JUNCTIONS",
    "FLOW_SCALE_BOUNDARY",
    "QC_LABELS",
    "QC_SCORE_FLAG_BELOW",
    "QC_VARIABLES",
    "STAGE_NAME",
    "AutoQcConfig",
    "compute_qc_columns",
    "compute_subject_qc",
    "compute_vessel_qc",
    "infer_flow_scale",
    "load_flow_measurements",
    "publish_autoqc",
    "main",
    "run_autoqc",
]


if __name__ == "__main__":  # pragma: no cover
    main()
