"""
Arithmetic *across* regions — combining a measurement's rows into a new quantity.

Description
-----------
The derived columns in :mod:`~nvitk.stats.frame_ops` operate **within a row**: ``log(pi)``, ``pi /
flow_mean``, a binned ``tacsctot``. That covers everything expressible from one row's own values,
and none of the quantities that matter most in a vascular analysis, because those are sums *over*
rows:

.. code-block:: text

    TCBF          = flow(RICA) + flow(LICA) + flow(BASI)
    basilar gap   = flow(LVA) + flow(RVA) − flow(BASI)
    MCA share     = flow(LMCA) / (flow(LMCA) + flow(LACA))

The analysis frame is long — one row per subject × territory — so all three are combinations of
several rows of one subject. This module is that operation: pick a measurement, pick regions, give
each a coefficient, and get back either a new **column** (the same value on every row of the
subject, ready to use as a covariate) or a new **row** (a synthetic territory, ready to model
alongside the real ones).

Relationship to the rest
------------------------
:data:`~nvitk.stats.vessel_network.CONSERVATION_RULES` are exactly the combinations whose
coefficients encode mass balance, so :func:`conservation_combinations` turns them into
:class:`RegionCombination` objects and the same machinery evaluates them. Nothing here knows about
anatomy; it is the arithmetic, and the anatomy lives in
:mod:`~nvitk.stats.vessel_network`.
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

#: How the selected regions are combined.
COMBINE_OPS: dict[str, str] = {
    "sum": "Weighted sum — Σ cᵢ·xᵢ, e.g. TCBF = RICA + LICA + BASI, or a balance with −1 terms",
    "mean": "Weighted mean — the sum divided by the sum of |coefficients|",
    "ratio": "Ratio — the positive terms over the negative ones, e.g. MCA / (MCA + ACA)",
    "share": "Share — each positive term over the total of every listed region",
    "difference": "Difference — the first region minus the second, ignoring other coefficients",
}

#: Where the result goes.
COMBINE_MODES: dict[str, str] = {
    "column": "New column — the same value on every row of the subject, usable as a covariate",
    "row": "New territory row — a synthetic region modelled alongside the real ones",
}

#: A combination's name has to work as a formula term.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RegionCombination:
    """
    One quantity built by combining a measurement across regions.

    ``terms`` maps a region label to its coefficient, so ``{"Left_ICA": 1, "Right_ICA": 1,
    "Basilar": 1}`` with ``op="sum"`` is total cerebral blood flow, and ``{"Left_VA": 1,
    "Right_VA": 1, "Basilar": -1}`` is the vertebrobasilar mass-balance residual.
    """

    name: str
    value_column: str
    terms: Mapping[str, float]
    op: str = "sum"
    mode: str = "column"
    region_column: str = "territory"
    subject_column: str = "subject_uid"
    #: Refuse to produce a value for a subject missing any listed region. Off, a partial sum is
    #: emitted — which is almost never what you want for a total, and never for a balance.
    require_all: bool = True
    unit: str = ""
    description: str = ""

    def validate(self) -> str:
        """Empty when the combination is well-formed, otherwise the reason it is not."""
        if not NAME_RE.match(self.name or ""):
            return (
                "the name must start with a letter or underscore and contain only letters, digits "
                "and underscores — it is used as a formula term"
            )
        if not self.value_column:
            return "no measurement column chosen"
        if len(self.terms) < 1:
            return "no regions selected"
        if self.op not in COMBINE_OPS:
            return f"unknown operation {self.op!r}"
        if self.mode not in COMBINE_MODES:
            return f"unknown mode {self.mode!r}"
        if self.op in {"ratio", "difference"} and len(self.terms) < 2:
            return f"{self.op!r} needs at least two regions"
        if self.op == "ratio" and not any(c < 0 for c in self.terms.values()):
            return "'ratio' needs at least one region with a negative coefficient (the denominator)"
        return ""

    def expression(self) -> str:
        """The combination as readable text, for a label or a tooltip."""
        parts: list[str] = []
        for region, coefficient in self.terms.items():
            sign = "−" if coefficient < 0 else "+"
            magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient):g}·"
            parts.append(f"{sign} {magnitude}{region}")
        body = " ".join(parts)
        body = body[2:] if body.startswith("+ ") else body
        if self.op == "ratio":
            numerator = " + ".join(r for r, c in self.terms.items() if c > 0)
            denominator = " + ".join(r for r, c in self.terms.items() if c < 0)
            body = f"({numerator}) / ({denominator})"
        elif self.op == "share":
            numerator = " + ".join(r for r, c in self.terms.items() if c > 0)
            body = f"({numerator}) / ({' + '.join(self.terms)})"
        elif self.op == "mean":
            body = f"mean({body})"
        elif self.op == "difference":
            first, second = list(self.terms)[:2]
            body = f"{first} − {second}"
        return f"{self.name} = {body}   [{self.value_column}]"

    def to_dict(self) -> dict[str, Any]:
        """Config-serializable form."""
        return {
            "name": self.name, "value_column": self.value_column,
            "terms": {str(k): float(v) for k, v in self.terms.items()},
            "op": self.op, "mode": self.mode, "region_column": self.region_column,
            "subject_column": self.subject_column, "require_all": bool(self.require_all),
            "unit": self.unit, "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegionCombination:
        """Rebuild from :meth:`to_dict`, tolerating keys an older config did not have."""
        return cls(
            name=str(data.get("name", "")),
            value_column=str(data.get("value_column", "")),
            terms={str(k): float(v) for k, v in dict(data.get("terms", {})).items()},
            op=str(data.get("op", "sum")),
            mode=str(data.get("mode", "column")),
            region_column=str(data.get("region_column", "territory")),
            subject_column=str(data.get("subject_column", "subject_uid")),
            require_all=bool(data.get("require_all", True)),
            unit=str(data.get("unit", "")),
            description=str(data.get("description", "")),
        )


def _matched_regions(
    present: Iterable[Any], wanted: Iterable[str]
) -> dict[str, str]:
    """
    Map each wanted region to the frame's own spelling, comparing case- and separator-insensitively.

    A combination saved against ``Left_ICA`` should still resolve after a reload that publishes
    ``LICA``; matching on the raw string would silently produce an all-NaN column instead.
    """
    from .vessel_network import canonical_node

    def key(value: Any) -> str:
        """Comparison key: letters and digits only."""
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    present = [str(p) for p in present]
    literal = {key(p): p for p in present}
    # A second index through the vessel resolver, so ``LICA``, ``Left_ICA`` and ``lica`` all reach
    # the same row. Without it a combination written in one spelling silently matches nothing after
    # a reload that publishes another — an all-NaN column rather than an error.
    canonical: dict[str, str] = {}
    for p in present:
        node = canonical_node(p)
        if node is not None:
            canonical.setdefault(node, p)

    out: dict[str, str] = {}
    for name in wanted:
        hit = literal.get(key(name))
        if hit is None:
            node = canonical_node(name)
            hit = canonical.get(node) if node else None
        if hit is not None:
            out[str(name)] = hit
    return out


def evaluate_region_combination(
    df: pd.DataFrame, combo: RegionCombination
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Compute one combination, returning a per-subject series and a report of what it used.

    Returns
    -------
    (values, report)
        *values* is indexed by subject. *report* carries ``matched``, ``missing``,
        ``n_subjects``, ``n_incomplete`` and ``expression``.

    Raises
    ------
    ValueError
        When the combination is malformed, its columns are absent, or no listed region is in the
        frame at all.
    """
    problem = combo.validate()
    if problem:
        raise ValueError(problem)
    for column in (combo.value_column, combo.region_column, combo.subject_column):
        if column not in df.columns:
            raise ValueError(f"column {column!r} is not in the frame")

    matched = _matched_regions(df[combo.region_column].dropna().unique(), combo.terms)
    missing = [r for r in combo.terms if r not in matched]
    if not matched:
        raise ValueError(
            f"none of the regions ({', '.join(combo.terms)}) are in {combo.region_column!r}"
        )
    if missing and combo.require_all:
        # A sum over *some* of the intended regions is a different quantity wearing the same name.
        # For a balance it is worse than useless: dropping one outflow term turns a residual that
        # should sit near zero into that term's whole magnitude.
        raise ValueError(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in "
            f"{combo.region_column!r}. Load a measurement that covers {'it' if len(missing) == 1 else 'them'}, "
            f"or clear 'require every region' to accept a partial result."
        )

    work = df[[combo.subject_column, combo.region_column, combo.value_column]].copy()
    work[combo.region_column] = work[combo.region_column].astype(str)
    wide = work.pivot_table(
        index=combo.subject_column, columns=combo.region_column,
        values=combo.value_column, aggfunc="mean",
    )

    coefficients = {matched[r]: float(c) for r, c in combo.terms.items() if r in matched}
    columns = list(coefficients)
    block = wide.reindex(columns=columns)
    incomplete = block.isna().any(axis=1)

    if combo.op == "sum":
        values = sum(c * block[col] for col, c in coefficients.items())
    elif combo.op == "mean":
        weight = sum(abs(c) for c in coefficients.values()) or 1.0
        values = sum(c * block[col] for col, c in coefficients.items()) / weight
    elif combo.op == "difference":
        first, second = columns[0], columns[1]
        values = block[first] - block[second]
    elif combo.op == "ratio":
        numerator = sum(c * block[col] for col, c in coefficients.items() if c > 0)
        denominator = sum(-c * block[col] for col, c in coefficients.items() if c < 0)
        # A zero denominator is a missing answer, not an infinite one; ±inf survives dropna and
        # silently poisons every parameter of the model it lands in.
        values = numerator / denominator.where(denominator != 0)
    else:  # share
        numerator = sum(c * block[col] for col, c in coefficients.items() if c > 0)
        total = block.sum(axis=1, min_count=len(columns) if combo.require_all else 1)
        values = numerator / total.where(total != 0)

    values = values.replace([np.inf, -np.inf], np.nan)
    if combo.require_all:
        values = values.where(~incomplete)

    report = {
        "name": combo.name,
        "expression": combo.expression(),
        "matched": matched,
        "missing": missing,
        "n_subjects": int(values.notna().sum()),
        "n_incomplete": int(incomplete.sum()),
    }
    if missing:
        log.warning(
            "%s: %s not found in %r — computed from the %d region(s) that are present.",
            combo.name, ", ".join(missing), combo.region_column, len(matched),
        )
    return values.rename(combo.name), report


def apply_region_combinations(
    df: pd.DataFrame, combinations: Sequence[RegionCombination]
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """
    Apply each combination to a copy of *df*, in order.

    ``mode="column"`` merges the per-subject value back onto every row of that subject, so it can be
    used as a covariate beside the per-territory outcome. ``mode="row"`` appends synthetic rows
    carrying the combination's name as their region, so it can be modelled as another territory —
    note those rows are **not independent** of the ones they were built from, which matters if the
    model treats territories as replicates.

    A combination that fails is skipped with its error collected, so one bad definition does not
    cost the rest.

    Returns
    -------
    (frame, errors, reports)
    """
    if not combinations:
        return df, [], []

    out = df.copy()
    errors: list[str] = []
    reports: list[dict[str, Any]] = []
    new_rows: list[pd.DataFrame] = []

    for combo in combinations:
        try:
            values, report = evaluate_region_combination(out, combo)
        except Exception as exc:
            errors.append(f"{combo.name or '(unnamed)'}: {exc}")
            log.debug("Region combination %r failed: %s", combo.name, exc, exc_info=True)
            continue
        reports.append(report)

        if combo.mode == "column":
            if combo.name in out.columns:
                out = out.drop(columns=[combo.name])
            out = out.merge(
                values.reset_index(), on=combo.subject_column, how="left", validate="many_to_one"
            )
        else:
            frame = values.reset_index()
            frame[combo.region_column] = combo.name
            frame = frame.rename(columns={combo.name: combo.value_column})
            # Carry the subject-level covariates over so the synthetic rows are usable in a model;
            # anything that varies within a subject has no defined value here and is left out.
            constant = _subject_constant_columns(out, combo.subject_column)
            extras = [
                c for c in constant
                if c not in {combo.subject_column, combo.region_column, combo.value_column}
            ]
            if extras:
                lookup = out.groupby(combo.subject_column)[extras].first().reset_index()
                frame = frame.merge(lookup, on=combo.subject_column, how="left")
            new_rows.append(frame.dropna(subset=[combo.value_column]))

    if new_rows:
        out = pd.concat([out, *new_rows], ignore_index=True)
    return out, errors, reports


def _subject_constant_columns(df: pd.DataFrame, subject_column: str) -> list[str]:
    """Columns that take a single value within every subject — the ones safe to copy onto a new row."""
    candidates = [c for c in df.columns if c != subject_column]
    if not candidates:
        return []
    counts = df.groupby(subject_column)[candidates].nunique(dropna=True)
    return [c for c in candidates if int(counts[c].max() or 0) <= 1]


# ---------------------------------------------------------------------------
# Ready-made combinations
# ---------------------------------------------------------------------------
def conservation_combinations(
    value_column: str = "flow_mean",
    *,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
    rules: Sequence[str] | None = None,
    relative: bool = False,
) -> list[RegionCombination]:
    """
    The mass-balance rules as ready-to-apply combinations.

    This is the answer to "how do I get conservation residuals as derived columns": each rule in
    :data:`~nvitk.stats.vessel_network.CONSERVATION_RULES` is already a weighted sum over regions,
    so it maps directly onto a :class:`RegionCombination` with ``op="sum"``. The result is one
    column per balance, on every row of the subject, ready to use as an outcome or a covariate.

    Parameters
    ----------
    relative : bool
        Emit the balance as a *share* of the rule's inflow instead of an absolute residual.
        Absolute residuals scale with cardiac output; relative ones are comparable between subjects.
    """
    from .vessel_network import CONSERVATION_RULES, VESSEL_NODES

    selected = list(CONSERVATION_RULES.values()) if rules is None else [
        CONSERVATION_RULES[r] for r in rules
    ]
    out: list[RegionCombination] = []
    for rule in selected:
        # The rules are written in canonical node ids; the frame carries published labels, and
        # ``_matched_regions`` normalizes both, so the ids can be used as-is.
        out.append(RegionCombination(
            name=f"{rule.key}_{'rel' if relative else 'residual'}",
            value_column=value_column,
            terms=dict(rule.terms),
            op="share" if relative else "sum",
            mode="column",
            region_column=region_column,
            subject_column=subject_column,
            require_all=True,
            description=(
                f"{rule.label}: {rule.expression()}. {rule.caveat}"
                if rule.caveat else f"{rule.label}: {rule.expression()}"
            ),
        ))
    del VESSEL_NODES  # imported for the docstring's benefit only
    return out


def composite_combinations(
    value_column: str = "flow_mean",
    *,
    region_column: str = "territory",
    subject_column: str = "subject_uid",
) -> list[RegionCombination]:
    """
    The standard whole-brain composites: total inflow, and the anterior/posterior split.

    ``TCBF`` is the one asked for most often — total cerebral blood flow as the sum of both internal
    carotids and the basilar. It is a *column*, not a territory: it is the subject's total, so it
    belongs beside the per-territory rows rather than among them.
    """
    return [
        RegionCombination(
            name="TCBF", value_column=value_column,
            terms={"lica": 1.0, "rica": 1.0, "basi": 1.0},
            op="sum", mode="column",
            region_column=region_column, subject_column=subject_column,
            description="Total cerebral blood flow — both internal carotids plus the basilar.",
        ),
        RegionCombination(
            name="anterior_inflow", value_column=value_column,
            terms={"lica": 1.0, "rica": 1.0},
            op="sum", mode="column",
            region_column=region_column, subject_column=subject_column,
            description="Carotid (anterior) inflow.",
        ),
        RegionCombination(
            name="posterior_share", value_column=value_column,
            terms={"basi": 1.0, "lica": 0.0, "rica": 0.0},
            op="share", mode="column",
            region_column=region_column, subject_column=subject_column,
            description="Basilar flow as a share of total inflow — how posterior-dominant the "
                        "circulation is, independent of its absolute size.",
        ),
    ]


__all__ = [
    "COMBINE_MODES",
    "COMBINE_OPS",
    "NAME_RE",
    "RegionCombination",
    "apply_region_combinations",
    "composite_combinations",
    "conservation_combinations",
    "evaluate_region_combination",
]
