#!/usr/bin/env python3
"""Derive APOE carrier-status and allele-count groupings from the existing ``apoe`` genotype.

Description
-----------
``apoe`` is stored as a genotype string — ``E3/E4``, ``E2/E3``, and so on. That is the right thing
to store, but it is the wrong thing to model: six unordered levels spend five degrees of freedom on
a contrast nobody is asking about, and the two rare homozygotes (``E2/E2``, ``E4/E4``) carry so few
subjects that their coefficients are noise.

So this writes two families of derived variables from the same genotype, and which one to use is a
modelling decision, not a data one:

**Carrier status** (0/1) — ``apoe_e2_carrier``, ``apoe_e4_carrier``, ``apoe_e3e3``. The usual
epidemiological coding: does this person carry the risk allele at all. One degree of freedom each.

**Allele dose** (0/1/2) — ``apoe_e2_count``, ``apoe_e3_count``, ``apoe_e4_count``. Treats the
genotype as additive, which is the better-powered choice when the effect really is per-allele —
an ``E4/E4`` homozygote carries twice the dose of an ``E3/E4`` heterozygote, and carrier status
throws that distinction away.

Two things to know before using them
------------------------------------
``E2/E4`` is a carrier of **both** e2 and e4, so ``apoe_e2_carrier`` and ``apoe_e4_carrier`` are not
mutually exclusive and must not be summed into one "carrier group" column. It is also the genotype
where the protective and risk alleles are thought to partly cancel, which is why it is conventionally
either excluded or given its own level rather than folded into e4-carrier.

``apoe_e3e3`` is the reference genotype — the complement of "carries e2 or e4". It is written as a
convenience for filtering, not so it can be entered into a model alongside the other two, where it
would be collinear with them.

Source is the dataset itself, not an Excel: this reads whatever ``apoe`` currently holds, so
re-running after a genotype correction re-derives cleanly.

Examples::

    # Show the genotype distribution and what would be written
    python scripts/database/import_apoe_grouping.py

    # Write it
    python scripts/database/import_apoe_grouping.py --write
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import DEFAULT_VISIT_LABEL

log = Logger()

SOURCE_VARIABLE = "apoe"

#: ``variable_id -> (label, how to compute it from the allele multiset)``. Carrier flags first,
#: then doses; both are derived from the same parsed pair so they can never disagree.
DERIVATIONS: dict[str, tuple[str, str]] = {
    "apoe_e2_carrier": ("APOE e2 carrier (1 = at least one e2 allele)", "carrier:2"),
    "apoe_e4_carrier": ("APOE e4 carrier (1 = at least one e4 allele)", "carrier:4"),
    "apoe_e3e3": ("APOE e3/e3 homozygote (1 = reference genotype)", "e3e3"),
    "apoe_e2_count": ("APOE e2 allele count (0–2)", "count:2"),
    "apoe_e3_count": ("APOE e3 allele count (0–2)", "count:3"),
    "apoe_e4_count": ("APOE e4 allele count (0–2)", "count:4"),
}


# ---------------------------------------------------------------------------
# Genotype parsing
# ---------------------------------------------------------------------------
def parse_alleles(genotype: Any) -> tuple[int, ...]:
    """
    The two allele numbers in a genotype string, or ``()`` when it cannot be read.

    Accepts the spellings that turn up in practice — ``E3/E4``, ``e3e4``, ``3/4``, ``E3-E4`` — by
    pulling out the digits rather than matching a fixed format. Anything that does not yield exactly
    two alleles from {2, 3, 4} is refused rather than guessed at.

    Examples
    --------
    >>> parse_alleles("E3/E4"), parse_alleles("e2e2"), parse_alleles("")
    ((3, 4), (2, 2), ())
    """
    if genotype is None or pd.isna(genotype):
        return ()
    digits = [int(d) for d in re.findall(r"[234]", str(genotype))]
    if len(digits) != 2:
        return ()
    return tuple(sorted(digits))


def derive(alleles: tuple[int, ...], rule: str) -> float | None:
    """Value of one derivation *rule* for a parsed genotype, or ``None`` when unparseable."""
    if not alleles:
        return None
    kind, _, arg = rule.partition(":")
    if kind == "carrier":
        return float(int(arg) in alleles)
    if kind == "count":
        return float(alleles.count(int(arg)))
    if kind == "e3e3":
        return float(alleles == (3, 3))
    raise ValueError(f"Unknown derivation rule: {rule!r}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def load_genotypes(repo: Any) -> pd.DataFrame:
    """
    ``subject_uid`` / ``visit_id`` / ``genotype`` for every subject with an ``apoe`` value.

    Raises
    ------
    ValueError
        When ``apoe`` is absent — there is nothing to derive from, and writing empty groupings
        would look like "no carriers" rather than "no data".
    """
    clinical = repo.get("clinical_measurements", cohort_id=False)
    if clinical is None or clinical.empty or "variable_id" not in clinical.columns:
        raise ValueError("clinical_measurements is empty — nothing to derive from.")

    rows = clinical.loc[clinical["variable_id"].astype(str) == SOURCE_VARIABLE].copy()
    if rows.empty:
        raise ValueError(
            f"No {SOURCE_VARIABLE!r} rows in clinical_measurements. Import the APOE workbook first."
        )

    # The genotype is text; a numerically-coded import would land in value_num instead.
    genotype = rows["value_text"] if "value_text" in rows.columns else pd.Series(dtype="string")
    if genotype.isna().all() and "value_num" in rows.columns:
        genotype = rows["value_num"]

    frame = pd.DataFrame({
        "subject_uid": rows["subject_uid"].astype("string"),
        "visit_id": (
            rows["visit_id"].astype("string") if "visit_id" in rows.columns
            else pd.Series(DEFAULT_VISIT_LABEL, index=rows.index, dtype="string")
        ),
        "genotype": genotype.astype("string"),
    }).dropna(subset=["subject_uid"])
    frame["visit_id"] = frame["visit_id"].fillna(DEFAULT_VISIT_LABEL)

    frame["alleles"] = frame["genotype"].map(parse_alleles)
    unparsed = frame.loc[frame["alleles"].map(len) == 0, "genotype"]
    if len(unparsed):
        log.warning(
            "%d %s value(s) could not be parsed into two alleles and are skipped (e.g. %s).",
            len(unparsed), SOURCE_VARIABLE,
            ", ".join(sorted({str(v) for v in unparsed})[:6]),
        )
    frame = frame[frame["alleles"].map(len) == 2].reset_index(drop=True)

    counts = frame["genotype"].astype(str).value_counts().sort_index()
    log.info(
        "%s: %d subject(s) — %s",
        SOURCE_VARIABLE, len(frame),
        ", ".join(f"{k} n={v}" for k, v in counts.items()),
    )
    return frame


def build_groupings(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{variable_id: agg frame}`` — one aggregate frame per derived variable."""
    out: dict[str, pd.DataFrame] = {}
    for variable, (_label, rule) in DERIVATIONS.items():
        values = frame["alleles"].map(lambda a, r=rule: derive(a, r))
        agg = pd.DataFrame({
            "subject_uid": frame["subject_uid"],
            "visit_id": frame["visit_id"],
            "value_num": pd.to_numeric(values, errors="coerce"),
        }).dropna(subset=["value_num"])
        out[variable] = agg.reset_index(drop=True)
    return out


def publish_groupings(
    repo: Any, groupings: dict[str, pd.DataFrame], *, write: bool = False
) -> dict[str, int]:
    """Build and, with *write*, upsert every derived grouping. Returns rows written per variable."""
    written: dict[str, int] = {}
    variables = list(groupings)
    for i, variable in enumerate(variables):
        agg = groupings[variable]
        label, _rule = DERIVATIONS[variable]
        rows = build_clinical_measurement_rows(
            agg,
            DerivedClinicalMeasurementSpec(
                variable_id=variable,
                source_file="clinical_measurements",
                source_sheet="derived",
                source_column=SOURCE_VARIABLE,
                source_batch_id="import_apoe_grouping",
            ),
        )
        written[variable] = int(len(rows))
        if rows.empty or not write:
            continue
        publish_derived_measurements(
            repo,
            rows,
            table="clinical_measurements",
            register=DerivedVariableRegistration(
                variable_id=variable,
                domain="clinical",
                table="clinical_measurements",
                label=label,
                value_kind="float",
                source_file="clinical_measurements",
                source_sheet="derived",
                source_column=SOURCE_VARIABLE,
            ),
            provenance={"importer": "import_apoe_grouping", "derived_from": SOURCE_VARIABLE},
            # Rebuild the index once, after the last variable: it is derived from Parquet, so
            # rebuilding per variable repeats the same work six times.
            build_sqlite_index=(i == len(variables) - 1),
        )
        log.ok("%s: %d row(s).", variable, len(rows))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-apoe-grouping")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the one configured in .nvitk/settings.json.",
)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually write to the dataset. The default only reports what would be written.",
)
def main(dataset: Path | None, write: bool) -> None:
    """Derive APOE carrier flags and allele counts from the stored genotype."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(f"Could not open the dataset ({exc}).") from exc

    try:
        frame = load_genotypes(repo)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    groupings = build_groupings(frame)
    written = publish_groupings(repo, groupings, write=write)

    click.echo("")
    summary = pd.DataFrame(
        [
            {
                "variable": variable,
                "n": written[variable],
                "mean": round(float(groupings[variable]["value_num"].mean()), 4),
                "distribution": ", ".join(
                    f"{int(k)}:{v}"
                    for k, v in sorted(groupings[variable]["value_num"].value_counts().items())
                ),
            }
            for variable in DERIVATIONS
            if written.get(variable)
        ]
    )
    click.echo(summary.to_string(index=False))

    both = int(
        (
            (groupings["apoe_e2_carrier"]["value_num"] > 0)
            & (groupings["apoe_e4_carrier"]["value_num"] > 0)
        ).sum()
    )
    if both:
        click.echo(
            f"\n  {both} subject(s) are E2/E4 — carriers of both alleles. The two carrier flags "
            "overlap by design; do not add them into one group column."
        )

    total = sum(written.values())
    if write:
        click.echo(f"\nWrote {total} row(s) across {len(written)} variable(s).")
    else:
        click.echo(f"\nDry run — {total} row(s) would be written. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
