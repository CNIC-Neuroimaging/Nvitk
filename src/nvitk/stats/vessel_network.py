"""
The cerebral vascular network — topology, conservation, adjacency and composition.

Description
-----------
Every statistical model in this toolkit so far has treated territories as *exchangeable levels*: a
list of unrelated categories that happen to appear in the same frame. Anatomically they are nothing
of the sort. The vertebrals merge into the basilar, which splits into the posterior cerebrals; each
carotid splits into an anterior and a middle cerebral; the sagittal and straight sinuses drain into
the transverse sinuses. Flow through those junctions is *conserved*, and that constraint is
information no exchangeable-levels model can use.

This module holds the topology once, so the four ways of exploiting it share one definition:

============================  ===========================================================
Use                           Entry point
============================  ===========================================================
conservation residuals        :data:`CONSERVATION_RULES`, :func:`conservation_frame`
path analysis / SEM           :func:`network_edges`, :func:`sem_model_syntax`
MRF smooth over the graph     :func:`neighbour_list`
compositional (flow shares)   :func:`flow_fractions`, :func:`clr_transform`
============================  ===========================================================

Communicating arteries
----------------------
The circle of Willis is not a tree. The posterior communicating arteries join each carotid to the
ipsilateral posterior cerebral, and the anterior communicating joins the two anterior cerebrals —
collaterals whose flow **direction is subject-specific**. That is why they are held separately from
the tree edges in :data:`COLLATERAL_EDGES` and enter the conservation equations as *signed* terms:
a positive PComm flow moves blood from the carotid to the posterior circulation, a negative one the
other way. Modelling them as tree edges would force a direction the anatomy does not have.

Sign convention
---------------
Tree flows are magnitudes and always positive. Collateral flows are signed, positive in the
direction named by the edge (``lica → lpca`` for the left PComm). Whether a published measurement
already follows that convention is a property of the pipeline, not of this module: pass
``signed_collaterals=False`` to :func:`conservation_frame` when the values are magnitudes and the
direction is unknown, and the affected balances are reported as ranges rather than as residuals.

Units
-----
Flow terms must share one unit (qvtpy publishes mL/min). Conservation is a statement about volume
per unit time; mixing a velocity or an index into a balance is meaningless, and
:func:`conservation_frame` refuses measurements it does not recognize as flows unless told
otherwise.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()


# ---------------------------------------------------------------------------
# Canonical nodes
# ---------------------------------------------------------------------------
#: Canonical node ids and their display labels. Everything else in this module speaks these.
VESSEL_NODES: dict[str, str] = {
    "lva": "Left vertebral",
    "rva": "Right vertebral",
    "basi": "Basilar",
    "lpca": "Left PCA",
    "rpca": "Right PCA",
    "lica": "Left ICA",
    "rica": "Right ICA",
    "laca": "Left ACA",
    "raca": "Right ACA",
    "lmca": "Left MCA",
    "rmca": "Right MCA",
    "lpcomm": "Left PComm",
    "rpcomm": "Right PComm",
    "acomm": "AComm",
    "sss": "Superior sagittal sinus",
    "strs": "Straight sinus",
    "lts": "Left transverse sinus",
    "rts": "Right transverse sinus",
}

#: Published spellings → canonical node. Normalized on lookup, so case and separators do not matter.
_NODE_ALIASES: dict[str, str] = {
    "lva": "lva", "left_va": "lva", "left_vertebral": "lva", "l_va": "lva",
    "rva": "rva", "right_va": "rva", "right_vertebral": "rva", "r_va": "rva",
    "basi": "basi", "basilar": "basi", "ba": "basi",
    "lpca": "lpca", "left_pca": "lpca", "l_pca": "lpca",
    "rpca": "rpca", "right_pca": "rpca", "r_pca": "rpca",
    "lica": "lica", "left_ica": "lica", "l_ica": "lica",
    "rica": "rica", "right_ica": "rica", "r_ica": "rica",
    "laca": "laca", "left_aca": "laca", "l_aca": "laca",
    "raca": "raca", "right_aca": "raca", "r_aca": "raca",
    "lmca": "lmca", "left_mca": "lmca", "l_mca": "lmca",
    "rmca": "rmca", "right_mca": "rmca", "r_mca": "rmca",
    "lpcomm": "lpcomm", "left_pcomm": "lpcomm", "lcomm": "lpcomm",
    "left_communicating": "lpcomm", "left_posterior_communicating": "lpcomm",
    "rpcomm": "rpcomm", "right_pcomm": "rpcomm", "rcomm": "rpcomm",
    "right_communicating": "rpcomm", "right_posterior_communicating": "rpcomm",
    "acomm": "acomm", "anterior_communicating": "acomm",
    "sss": "sss", "sssv": "sss", "sagital_sinus": "sss", "sagittal_sinus": "sss",
    "superior_sagittal_sinus": "sss",
    "strs": "strs", "strv": "strs", "straight_sinus": "strs",
    "lts": "lts", "ltsv": "lts", "left_transverse": "lts", "left_transverse_sinus": "lts",
    "rts": "rts", "rtsv": "rts", "right_transverse": "rts", "right_transverse_sinus": "rts",
}


def _normalize(name: Any) -> str:
    """Lowercase, underscore-separated form of a region label (``Left-ICA`` → ``left_ica``)."""
    text = re.sub(r"[\s\-.]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def canonical_node(region_id: Any) -> str | None:
    """
    Canonical vessel node for a published region label, or ``None`` when it is not one.

    Examples
    --------
    >>> canonical_node("Left_ICA"), canonical_node("LICA"), canonical_node("lica")
    ('lica', 'lica', 'lica')
    >>> canonical_node("ctx-lh-superiorfrontal") is None
    True
    """
    token = _normalize(region_id)
    if not token:
        return None
    if token in _NODE_ALIASES:
        return _NODE_ALIASES[token]
    # ASL parcels carry a smoothing kernel suffix that is not anatomy.
    stripped = re.sub(r"_(?:0|8|12)$", "", token)
    return _NODE_ALIASES.get(stripped)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VesselEdge:
    """One directed connection, ``source`` feeding ``target``."""

    source: str
    target: str
    kind: str = "tree"          # "tree" (fixed direction) | "collateral" (signed, variable)
    compartment: str = "arterial"

    def label(self) -> str:
        """Human-readable arrow, for a legend or a path diagram."""
        arrow = "→" if self.kind == "tree" else "↔"
        return f"{VESSEL_NODES.get(self.source, self.source)} {arrow} {VESSEL_NODES.get(self.target, self.target)}"


#: Fixed-direction edges. Blood cannot run the other way through these without a pathology.
TREE_EDGES: tuple[VesselEdge, ...] = (
    VesselEdge("lva", "basi"),
    VesselEdge("rva", "basi"),
    VesselEdge("basi", "lpca"),
    VesselEdge("basi", "rpca"),
    VesselEdge("lica", "laca"),
    VesselEdge("lica", "lmca"),
    VesselEdge("rica", "raca"),
    VesselEdge("rica", "rmca"),
    VesselEdge("sss", "lts", compartment="venous"),
    VesselEdge("sss", "rts", compartment="venous"),
    VesselEdge("strs", "lts", compartment="venous"),
    VesselEdge("strs", "rts", compartment="venous"),
)

#: Circle-of-Willis collaterals. Direction depends on the subject's pressure gradients, so these
#: carry a sign rather than a fixed orientation, and they are frequently absent or hypoplastic.
COLLATERAL_EDGES: tuple[VesselEdge, ...] = (
    VesselEdge("lica", "lpca", kind="collateral"),   # via the left PComm
    VesselEdge("rica", "rpca", kind="collateral"),   # via the right PComm
    VesselEdge("laca", "raca", kind="collateral"),   # via the AComm
)

#: Which measured vessel carries each collateral, so its flow can be read from the frame.
COLLATERAL_CONDUIT: dict[tuple[str, str], str] = {
    ("lica", "lpca"): "lpcomm",
    ("rica", "rpca"): "rpcomm",
    ("laca", "raca"): "acomm",
}


def network_edges(*, include_collaterals: bool = True, compartment: str = "") -> tuple[VesselEdge, ...]:
    """
    The network's edges, optionally restricted to one compartment.

    Parameters
    ----------
    include_collaterals : bool
        Include the communicating arteries. Excluding them leaves a strict DAG, which is what a
        path model needs when the collateral flows were not measured.
    compartment : {"", "arterial", "venous"}
        Empty keeps both.
    """
    edges = TREE_EDGES + (COLLATERAL_EDGES if include_collaterals else ())
    if compartment:
        edges = tuple(e for e in edges if e.compartment == compartment)
    return edges


def neighbour_list(
    nodes: Sequence[str] | None = None,
    *,
    include_collaterals: bool = True,
    compartment: str = "",
) -> dict[str, list[str]]:
    """
    Symmetric adjacency, as ``mgcv``'s Markov-random-field smooth expects it.

    ``bs="mrf"`` needs an *undirected* neighbourhood: each node listing the nodes it touches, with
    the relation symmetric. Directions are dropped here on purpose — an MRF shares information
    between anatomically adjacent vessels, it does not model flow direction. Use
    :func:`sem_model_syntax` when the direction is the point.

    Parameters
    ----------
    nodes : sequence of str, optional
        Restrict to these canonical nodes (typically the ones actually present in the frame).
        Edges to absent nodes are dropped, which can leave a node isolated — reported, because an
        MRF over a disconnected graph estimates those levels independently.

    Returns
    -------
    dict
        ``{node: [neighbour, ...]}``, every node present as a key even when it has no neighbours.
    """
    edges = network_edges(include_collaterals=include_collaterals, compartment=compartment)
    keep = {str(n) for n in nodes} if nodes is not None else None

    out: dict[str, list[str]] = {}
    if keep is not None:
        out = {n: [] for n in sorted(keep)}

    def link(a: str, b: str) -> None:
        """Record an undirected adjacency, if both ends are being kept."""
        if keep is not None and (a not in keep or b not in keep):
            return
        out.setdefault(a, [])
        out.setdefault(b, [])
        if b not in out[a]:
            out[a].append(b)
        if a not in out[b]:
            out[b].append(a)

    for edge in edges:
        conduit = COLLATERAL_CONDUIT.get((edge.source, edge.target))
        if edge.kind == "collateral" and conduit and (keep is None or conduit in keep):
            # The measured vessel *is* the connection, so it sits between the two endpoints rather
            # than beside them. Linking the endpoints directly would leave the conduit isolated,
            # and an isolated level is exactly what an MRF cannot borrow strength for.
            link(edge.source, conduit)
            link(conduit, edge.target)
        else:
            link(edge.source, edge.target)

    isolated = sorted(n for n, nb in out.items() if not nb)
    if isolated:
        log.warning(
            "MRF neighbourhood: %s have no neighbour in the retained graph and will be smoothed "
            "independently.", ", ".join(isolated),
        )
    return {node: sorted(neighbours) for node, neighbours in sorted(out.items())}


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConservationRule:
    """
    One mass-balance equation, written as a signed sum that should come to zero.

    ``terms`` maps a canonical node to its coefficient, so
    ``{"lva": 1, "rva": 1, "basi": -1}`` is *left vertebral + right vertebral − basilar = 0*.
    """

    key: str
    label: str
    terms: Mapping[str, float]
    #: Why it may not close exactly — the branches nobody measured.
    caveat: str = ""
    #: Nodes whose measurement must be signed for the equation to hold as written.
    signed_terms: tuple[str, ...] = ()
    compartment: str = "arterial"

    def required_nodes(self) -> tuple[str, ...]:
        """Canonical nodes the rule needs before it can be evaluated."""
        return tuple(sorted(self.terms))

    def expression(self) -> str:
        """The equation as text, e.g. ``lva + rva − basi``."""
        parts: list[str] = []
        for node, coefficient in self.terms.items():
            sign = "−" if coefficient < 0 else "+"
            magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient):g}·"
            parts.append(f"{sign} {magnitude}{node}")
        text = " ".join(parts)
        return text[2:] if text.startswith("+ ") else text


#: Standard balances. Each is *inflow − outflow*, so a healthy residual sits near zero.
CONSERVATION_RULES: dict[str, ConservationRule] = {
    "basilar_inflow": ConservationRule(
        "basilar_inflow", "Vertebrals → basilar",
        {"lva": 1.0, "rva": 1.0, "basi": -1.0},
        caveat="PICA arises from the vertebrals below the confluence, so the residual is expected "
               "to be positive by the PICA territory's supply.",
    ),
    "posterior_split": ConservationRule(
        "posterior_split", "Basilar (+ PComms) → posterior cerebrals",
        {"basi": 1.0, "lpcomm": 1.0, "rpcomm": 1.0, "lpca": -1.0, "rpca": -1.0},
        caveat="AICA and SCA leave the basilar before the terminal bifurcation, so the residual "
               "runs positive. The PComm terms are signed: positive means the carotid is feeding "
               "the posterior circulation.",
        signed_terms=("lpcomm", "rpcomm"),
    ),
    "left_carotid_split": ConservationRule(
        "left_carotid_split", "Left ICA → left ACA + MCA (+ PComm)",
        {"lica": 1.0, "laca": -1.0, "lmca": -1.0, "lpcomm": -1.0},
        caveat="The anterior choroidal and ophthalmic arteries are not measured. The PComm term is "
               "signed and leaves the carotid when positive.",
        signed_terms=("lpcomm",),
    ),
    "right_carotid_split": ConservationRule(
        "right_carotid_split", "Right ICA → right ACA + MCA (+ PComm)",
        {"rica": 1.0, "raca": -1.0, "rmca": -1.0, "rpcomm": -1.0},
        caveat="The anterior choroidal and ophthalmic arteries are not measured. The PComm term is "
               "signed and leaves the carotid when positive.",
        signed_terms=("rpcomm",),
    ),
    "global_arterial": ConservationRule(
        "global_arterial", "Total inflow → terminal branches",
        {"lica": 1.0, "rica": 1.0, "basi": 1.0,
         "laca": -1.0, "raca": -1.0, "lmca": -1.0, "rmca": -1.0, "lpca": -1.0, "rpca": -1.0},
        caveat="Collaterals cancel out of this one — whatever the PComms carry stays inside the "
               "circle — so it is the cleanest check of overall measurement consistency.",
    ),
    "venous_drainage": ConservationRule(
        "venous_drainage", "Sagittal + straight sinus → transverse sinuses",
        {"sss": 1.0, "strs": 1.0, "lts": -1.0, "rts": -1.0},
        caveat="Cortical, petrosal and emissary tributaries join the transverse sinuses directly, "
               "so this balance is expected to be substantially negative rather than zero.",
        compartment="venous",
    ),
}


def wide_flow_frame(
    df: pd.DataFrame,
    *,
    value_column: str,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
) -> pd.DataFrame:
    """
    Pivot a long analysis frame to one row per subject and one column per canonical vessel node.

    Conservation, composition and path models all need a subject's vessels side by side; the
    analysis frame stores them stacked. Regions that are not vessels are dropped, and duplicate
    (subject, node) cells are averaged with a warning — that happens when a melted grouping put two
    vessels under one key, where a balance no longer means anything.

    Returns
    -------
    pandas.DataFrame
        Indexed by *subject_column*, columns named by canonical node.
        ``frame.attrs["dropped_regions"]`` lists labels that resolved to no node.
    """
    for column in (value_column, region_column, subject_column):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} is not in the frame.")

    work = df[[subject_column, region_column, value_column]].copy()
    work["_node"] = work[region_column].map(canonical_node)
    dropped = sorted(
        {str(r) for r, n in zip(work[region_column], work["_node"]) if n is None}
    )
    work = work.dropna(subset=["_node"])
    if work.empty:
        raise ValueError(
            f"None of the {df[region_column].nunique()} {region_column!r} levels are recognized "
            f"vessels, so there is no network to work with. This needs a vessel-wise frame "
            f"(grouping = 'vessel'), not a melted one."
        )

    duplicated = work.duplicated(subset=[subject_column, "_node"]).sum()
    if duplicated:
        log.warning(
            "%d (subject, vessel) cells appear more than once — averaging them. A melted grouping "
            "collapses several vessels into one key, which makes conservation meaningless.",
            int(duplicated),
        )
    wide = work.pivot_table(
        index=subject_column, columns="_node", values=value_column, aggfunc="mean"
    )
    wide.columns = [str(c) for c in wide.columns]
    wide.attrs["dropped_regions"] = dropped
    return wide


def network_frame(
    df: pd.DataFrame,
    *,
    value_column: str,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    covariates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    One row per subject: a column per vessel, plus the subject-level covariates carried across.

    :func:`wide_flow_frame` gives the vessels; a path model also needs ``age_c``, ``sex`` and the
    rest of the design on the same row. Those live in the long frame repeated once per territory,
    so they are folded back by taking the single value each subject carries.

    A covariate that *varies within a subject* is not subject-level — it is another vessel-wise
    measurement, and averaging it would invent a number that was never observed. Those are dropped
    with a warning rather than silently collapsed; name them in the syntax as vessels if that is
    what they are.

    Parameters
    ----------
    covariates : sequence of str, optional
        Columns to carry over. ``None`` takes every column of *df* that is constant within subject,
        which is usually what the design table contributes.

    Returns
    -------
    pandas.DataFrame
        Indexed by *subject_column*. ``frame.attrs`` carries ``vessels``, ``carried``,
        ``dropped_regions`` and ``dropped_covariates``.

    Examples
    --------
    >>> wide = network_frame(long, value_column="flow_mean")   # doctest: +SKIP
    >>> sorted(wide.columns)[:3]                               # doctest: +SKIP
    ['age_c', 'basi', 'laca']
    """
    wide = wide_flow_frame(
        df,
        value_column=value_column,
        region_column=region_column,
        subject_column=subject_column,
    )
    vessels = list(wide.columns)
    dropped_regions = list(wide.attrs.get("dropped_regions", []))

    # ---- Candidate covariates: everything that is not the pivot's own three columns ----------
    reserved = {value_column, region_column, subject_column}
    if covariates is None:
        candidates = [c for c in df.columns if c not in reserved]
    else:
        candidates = [str(c) for c in covariates if str(c) not in reserved]
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            raise ValueError(f"Covariate column(s) not in the frame: {', '.join(missing)}.")

    # ---- Keep only what is genuinely subject-level -------------------------------------------
    # nunique(dropna=False) > 1 means the column moves between a subject's territories, so it is a
    # per-vessel measurement wearing a covariate's name.
    carried: list[str] = []
    varying: list[str] = []
    grouped = df.groupby(subject_column, observed=True, sort=False)
    for column in candidates:
        if column in vessels:
            continue
        try:
            spread = grouped[column].nunique(dropna=False).max()
        except TypeError:                       # unhashable cells (lists, arrays) — never a covariate
            continue
        if spread is not None and spread > 1:
            varying.append(column)
            continue
        carried.append(column)

    if varying:
        log.warning(
            "%d column(s) vary within subject and were left out of the network frame: %s. They are "
            "vessel-wise measurements, not subject-level covariates.",
            len(varying), ", ".join(sorted(varying)[:8]),
        )

    if carried:
        design = grouped[carried].first()
        wide = wide.join(design, how="left")

    wide.attrs["vessels"] = vessels
    wide.attrs["carried"] = carried
    wide.attrs["dropped_regions"] = dropped_regions
    wide.attrs["dropped_covariates"] = sorted(varying)
    log.info(
        "Network frame: %d subject(s) × %d vessel(s), %d covariate(s) carried.",
        len(wide), len(vessels), len(carried),
    )
    return wide


def conservation_frame(
    df: pd.DataFrame,
    *,
    value_column: str,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    rules: Sequence[str] | Sequence[ConservationRule] | None = None,
    signed_collaterals: bool = True,
    relative: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the mass-balance residuals, one column per rule, one row per subject.

    A residual is *inflow − outflow*: near zero means the measurements are internally consistent,
    and a systematic departure is either an unmeasured branch (see each rule's ``caveat``) or a
    measurement bias. Either way it is a quantity worth modelling in its own right — a subject whose
    carotid balance is 40 mL/min out has something wrong with a segmentation or with an artery.

    Parameters
    ----------
    rules : sequence, optional
        Rule keys or :class:`ConservationRule` objects. Defaults to every rule whose nodes are all
        present in the frame; rules missing a node are skipped and reported.
    signed_collaterals : bool
        Whether the communicating-artery measurements carry a direction. When ``False``, rules that
        need a signed collateral are still evaluated but with the collateral term dropped, and their
        ``caveat`` records that the residual is therefore uncertain by that flow.
    relative : bool
        Also emit ``<rule>_rel``, the residual as a fraction of the rule's total inflow. Absolute
        residuals scale with the subject's cardiac output; relative ones are comparable between
        subjects and are usually what you want as an outcome.

    Returns
    -------
    pandas.DataFrame
        Indexed by subject. One column per rule (``<key>_residual``), optionally ``<key>_rel``, plus
        the vessel columns used. ``frame.attrs["rules"]`` describes what was evaluated and skipped.
    """
    wide = wide_flow_frame(
        df, value_column=value_column, region_column=region_column, subject_column=subject_column
    )

    if rules is None:
        selected = list(CONSERVATION_RULES.values())
    else:
        selected = [CONSERVATION_RULES[r] if isinstance(r, str) else r for r in rules]

    out = wide.copy()
    evaluated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for rule in selected:
        terms = dict(rule.terms)
        note = rule.caveat
        if not signed_collaterals and rule.signed_terms:
            for node in rule.signed_terms:
                terms.pop(node, None)
            note = (
                f"{note} Collateral terms ({', '.join(rule.signed_terms)}) were dropped because the "
                f"measurements are unsigned, so the residual is uncertain by their magnitude."
            ).strip()

        missing = [node for node in terms if node not in wide.columns]
        if missing:
            skipped.append({"rule": rule.key, "label": rule.label, "missing": missing})
            continue

        residual = sum(coefficient * wide[node] for node, coefficient in terms.items())
        out[f"{rule.key}_residual"] = residual
        if relative:
            inflow = sum(wide[node] for node, c in terms.items() if c > 0)
            # A subject with no measured inflow would give ±inf; NaN is the honest answer.
            out[f"{rule.key}_rel"] = residual / inflow.where(inflow > 0)
        evaluated.append({
            "rule": rule.key, "label": rule.label, "expression": rule.expression(),
            "caveat": note, "n_subjects": int(residual.notna().sum()),
        })

    if skipped:
        log.warning(
            "Conservation: skipped %d rule(s) for missing vessels — %s",
            len(skipped),
            "; ".join(f"{s['label']} needs {', '.join(s['missing'])}" for s in skipped),
        )
    out.attrs["rules"] = evaluated
    out.attrs["skipped"] = skipped
    out.attrs["dropped_regions"] = wide.attrs.get("dropped_regions", [])
    return out


def conservation_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy per-rule summary of a :func:`conservation_frame` result: how far each balance is from zero.

    ``bias`` is the median residual — a systematic offset, usually an unmeasured branch.
    ``scatter`` is the median absolute deviation, which is the part that varies subject to subject
    and therefore the part that reflects measurement noise rather than anatomy.
    """
    rows: list[dict[str, Any]] = []
    for entry in residuals.attrs.get("rules", []):
        key = entry["rule"]
        column = f"{key}_residual"
        if column not in residuals.columns:
            continue
        values = pd.to_numeric(residuals[column], errors="coerce").dropna()
        relative = pd.to_numeric(residuals.get(f"{key}_rel"), errors="coerce")
        rows.append({
            "rule": key,
            "label": entry["label"],
            "expression": entry["expression"],
            "n": int(len(values)),
            "bias": float(values.median()) if len(values) else np.nan,
            "scatter": float((values - values.median()).abs().median()) if len(values) else np.nan,
            "bias_rel": float(relative.dropna().median()) if relative is not None and relative.notna().any() else np.nan,
            "caveat": entry["caveat"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
#: Sets of vessels whose flows partition a supply, and therefore form a composition.
COMPOSITION_SETS: dict[str, tuple[str, ...]] = {
    "terminal": ("laca", "raca", "lmca", "rmca", "lpca", "rpca"),
    "inflow": ("lica", "rica", "basi"),
    "left_carotid": ("laca", "lmca"),
    "right_carotid": ("raca", "rmca"),
    "anterior_posterior": ("lica", "rica", "basi"),
    "venous_outflow": ("lts", "rts"),
}


def flow_fractions(
    df: pd.DataFrame,
    *,
    value_column: str,
    nodes: Sequence[str] | str = "terminal",
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    prefix: str = "frac",
) -> pd.DataFrame:
    """
    Each vessel's share of a set's total flow — one row per subject, one column per vessel.

    Absolute flows carry the subject's whole cardiac output, so a model of ``lmca ~ age`` is partly
    a model of how much blood the person has. Fractions remove that: they ask *how the available
    flow is distributed*, which is the question most territory comparisons are really about, and
    they are far less sensitive to global scaling errors.

    Parameters
    ----------
    nodes : sequence of str or key of :data:`COMPOSITION_SETS`
        Which vessels form the composition. They must partition one supply for the fractions to
        mean anything — mixing arterial inflow with venous outflow does not.

    Returns
    -------
    pandas.DataFrame
        ``<prefix>_<node>`` columns summing to 1 within each row, plus ``<prefix>_total`` holding the
        denominator so the absolute scale is not lost. Subjects missing any component are NaN
        throughout, since a partial denominator would silently inflate every share.
    """
    members = COMPOSITION_SETS[nodes] if isinstance(nodes, str) else tuple(str(n) for n in nodes)
    wide = wide_flow_frame(
        df, value_column=value_column, region_column=region_column, subject_column=subject_column
    )
    missing = [n for n in members if n not in wide.columns]
    if missing:
        raise ValueError(
            f"Cannot build a composition: {', '.join(missing)} are not in the frame. A composition "
            f"needs every component, or the shares do not sum to one."
        )

    parts = wide[list(members)]
    total = parts.sum(axis=1, min_count=len(members))
    out = pd.DataFrame(index=wide.index)
    for node in members:
        out[f"{prefix}_{node}"] = parts[node] / total.where(total > 0)
    out[f"{prefix}_total"] = total
    return out


def clr_transform(
    fractions: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    prefix: str = "clr",
) -> pd.DataFrame:
    """
    Centred log-ratio of a set of shares, so they can go into an ordinary linear model.

    Shares live on a simplex: they are bounded, they sum to one, and so they are *not* independent —
    if the MCA's share rises, something else's must fall. Regressing them directly gives correlated
    residuals and predictions outside [0, 1]. The CLR maps them to a real space where a linear model
    is meaningful; a positive CLR coefficient means that vessel's share grew *relative to the
    geometric mean of all of them*, which is the only kind of statement a composition supports.

    Zeros have no logarithm, so any subject with a zero share is returned as NaN rather than being
    silently offset — an imputed zero would change every other component's value in the row.
    """
    columns = list(columns) if columns is not None else [
        c for c in fractions.columns if not c.endswith("_total")
    ]
    values = fractions[columns].astype(float)
    invalid = (values <= 0).any(axis=1) | values.isna().any(axis=1)
    if int(invalid.sum()):
        log.warning(
            "CLR: %d subject(s) have a zero or missing share and are returned as NaN — a zero has "
            "no logarithm, and substituting one would move every other component in the row.",
            int(invalid.sum()),
        )

    logs = np.log(values.where(values > 0))
    centred = logs.sub(logs.mean(axis=1), axis=0)
    centred[invalid] = np.nan
    centred.columns = [f"{prefix}_{c.split('_', 1)[-1]}" for c in columns]
    return centred


# ---------------------------------------------------------------------------
# Path model syntax
# ---------------------------------------------------------------------------
def sem_model_syntax(
    *,
    nodes: Sequence[str] | None = None,
    include_collaterals: bool = True,
    compartment: str = "arterial",
    covariates: Sequence[str] = (),
    outcome_covariates: Sequence[str] = (),
    conduit_names: bool = True,
) -> str:
    """
    Path-model syntax for the vascular network, in the ``lavaan``/``semopy`` dialect.

    Each edge becomes a regression of the downstream vessel on its upstream feeder, so the model is
    the anatomy written as equations::

        basi ~ lva + rva
        lpca ~ basi + lpcomm
        lmca ~ lica

    *covariates* are added to every structural equation — the usual ``age_c + sex`` — which turns the
    fit into "how does age move flow along each edge, holding the upstream vessel fixed". That is
    the question a stacked territory model cannot answer, because it has no notion of upstream.

    Parameters
    ----------
    nodes : sequence of str, optional
        Restrict to vessels present in the data. Edges touching an absent vessel are dropped.
    include_collaterals : bool
        Add the communicating arteries as extra predictors of their downstream vessel. They enter as
        ordinary regressors: the model does not claim a direction for them, it estimates how the
        collateral's flow relates to the territory it can supply.
    conduit_names : bool
        Name a collateral by the vessel that was measured (``lpcomm``) rather than by its endpoints.
        This is what matches the frame's columns.

    Returns
    -------
    str
        Model syntax, one equation per line, with comments naming each block.
    """
    keep = {str(n) for n in nodes} if nodes is not None else None
    edges = network_edges(include_collaterals=include_collaterals, compartment=compartment)

    incoming: dict[str, list[str]] = {}
    for edge in edges:
        source = edge.source
        if edge.kind == "collateral" and conduit_names:
            source = COLLATERAL_CONDUIT.get((edge.source, edge.target), edge.source)
        if keep is not None and (edge.target not in keep or source not in keep):
            continue
        incoming.setdefault(edge.target, [])
        if source not in incoming[edge.target]:
            incoming[edge.target].append(source)

    if not incoming:
        raise ValueError(
            "No edges survive: none of the network's vessel pairs are both present in the data. "
            "A path model needs at least one upstream/downstream pair."
        )

    covariates = [str(c) for c in covariates]
    lines = ["# Structural model — one equation per vascular junction"]
    for target in sorted(incoming):
        predictors = incoming[target] + covariates
        lines.append(f"{target} ~ " + " + ".join(predictors))

    if outcome_covariates:
        lines.append("")
        lines.append("# Downstream outcomes regressed on the network")
        for outcome in outcome_covariates:
            lines.append(f"{outcome} ~ " + " + ".join(sorted(incoming) + covariates))

    # Bilateral pairs share an anatomy and a scanner session, so their residuals covary whatever the
    # structural part says. Leaving that out pushes the correlation into the path coefficients.
    pairs = [("laca", "raca"), ("lmca", "rmca"), ("lpca", "rpca"), ("lts", "rts")]
    residual = [
        f"{a} ~~ {b}" for a, b in pairs
        if a in incoming and b in incoming and (keep is None or {a, b} <= keep)
    ]
    if residual:
        lines.append("")
        lines.append("# Residual covariance between bilateral counterparts")
        lines.extend(residual)
    return "\n".join(lines)


@dataclass(frozen=True)
class CollateralSpec:
    """How a communicating artery is treated in a model."""

    node: str
    label: str
    #: Threshold under which the vessel is considered absent/hypoplastic rather than carrying flow.
    patency_threshold: float = 1.0
    description: str = ""


#: The three communicating arteries, with what modelling them usually requires.
COLLATERAL_SPECS: dict[str, CollateralSpec] = {
    "lpcomm": CollateralSpec(
        "lpcomm", "Left posterior communicating", 1.0,
        "Present and flow-carrying in roughly half of subjects. Model as a signed flow plus a "
        "patency indicator: the indicator says whether the pathway exists, the flow says how much "
        "it is used, and the two answer different questions.",
    ),
    "rpcomm": CollateralSpec(
        "rpcomm", "Right posterior communicating", 1.0,
        "As the left. The two sides are frequently asymmetric, so a single 'PComm present' variable "
        "loses most of the information.",
    ),
    "acomm": CollateralSpec(
        "acomm", "Anterior communicating", 1.0,
        "Midline. Its flow is near zero in a balanced circle and rises when one carotid is "
        "compromised, so it behaves as a marker of asymmetry rather than as a supply route.",
    ),
}


def collateral_features(
    df: pd.DataFrame,
    *,
    value_column: str,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    nodes: Sequence[str] = ("lpcomm", "rpcomm", "acomm"),
    signed: bool = True,
) -> pd.DataFrame:
    """
    Turn the communicating arteries into the variables a model can actually use.

    A PComm is not a vessel like the others. In perhaps half of subjects it carries no meaningful
    flow at all, so its measurement is a mixture of "absent" and "present but small" — and a plain
    linear term treats those as the same thing. This builds the three features that separate them:

    ``<node>``            the flow itself (signed when the pipeline provides a direction);
    ``<node>_patent``     whether it carries flow above the threshold — the pathway's *existence*;
    ``<node>_abs``        the magnitude, for questions about how hard a collateral is working
                          regardless of which way it runs.

    Plus, across the pair, ``pcomm_config`` — a four-level factor (none / left / right / both) which
    is the circle-of-Willis variant most papers actually stratify on, and ``pcomm_asymmetry``.

    Returns
    -------
    pandas.DataFrame
        Indexed by subject.
    """
    wide = wide_flow_frame(
        df, value_column=value_column, region_column=region_column, subject_column=subject_column
    )
    out = pd.DataFrame(index=wide.index)
    present: list[str] = []

    for node in nodes:
        spec = COLLATERAL_SPECS.get(node, CollateralSpec(node, node))
        if node not in wide.columns:
            log.info("Collateral %s is not in the frame — skipping its features.", node)
            continue
        values = pd.to_numeric(wide[node], errors="coerce")
        out[node] = values if signed else values.abs()
        out[f"{node}_abs"] = values.abs()
        # NaN means "not measured", which is not the same as "absent"; only a measured small value
        # is evidence of a non-patent vessel.
        out[f"{node}_patent"] = np.where(
            values.isna(), np.nan, (values.abs() >= spec.patency_threshold).astype(float)
        )
        present.append(node)

    if "lpcomm" in present and "rpcomm" in present:
        left = out["lpcomm_patent"]
        right = out["rpcomm_patent"]
        config = pd.Series(pd.NA, index=out.index, dtype="object")
        known = left.notna() & right.notna()
        config[known & (left == 0) & (right == 0)] = "none"
        config[known & (left == 1) & (right == 0)] = "left"
        config[known & (left == 0) & (right == 1)] = "right"
        config[known & (left == 1) & (right == 1)] = "both"
        out["pcomm_config"] = pd.Categorical(
            config, categories=["none", "left", "right", "both"], ordered=False
        )
        out["pcomm_asymmetry"] = out["lpcomm_abs"] - out["rpcomm_abs"]
    return out


__all__ = [
    "COLLATERAL_CONDUIT",
    "COLLATERAL_EDGES",
    "COLLATERAL_SPECS",
    "COMPOSITION_SETS",
    "CONSERVATION_RULES",
    "TREE_EDGES",
    "VESSEL_NODES",
    "CollateralSpec",
    "ConservationRule",
    "VesselEdge",
    "canonical_node",
    "clr_transform",
    "collateral_features",
    "conservation_frame",
    "conservation_summary",
    "flow_fractions",
    "neighbour_list",
    "network_edges",
    "sem_model_syntax",
    "wide_flow_frame",
]
