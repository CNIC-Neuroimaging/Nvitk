#!/usr/bin/env python3
"""Remove a subject's qvtpy results from the dataset, leaving everything else about them intact.

Description
-----------
When a 4D-flow acquisition turns out to be unusable — a segmentation that leaked into a
neighbouring structure, a mis-scaled reconstruction, a failed gating — the subject should drop out
of the flow analyses without disappearing from the study. Their clinical variables, cognitive
scores, morphometrics and imaging assets are all still valid; only the qvtpy-derived measurements
are not.

So this deletes exactly two things:

===================================  =========================================================
``image_measurements``               rows whose ``pipeline_id`` is ``4dflow_v3`` (stage-6
                                     hemodynamics) or ``qvtpy`` (stage-9 automatic QC)
``clinical_measurements``            the three subject-level autoQC variables
                                     (``qc_ap_share``, ``qc_ap_flag``, ``qc_subject_flag``)
===================================  =========================================================

Every other row of that subject, in these tables and all others, is left untouched — including
``tof_morpho_v1`` morphometrics, which come from a different acquisition and survive a bad 4D-flow
scan. Nothing is written until ``--write`` is passed.

The SQLite index is rebuilt once at the end, for the touched tables only.

Examples::

    # See what would go, without touching anything
    python scripts/pesa_brain/db/drop_subject_db.py --subjects PESA5745609

    # Actually remove it
    python scripts/pesa_brain/db/drop_subject_db.py --subjects PESA5745609 --write

    # Several subjects, explicit dataset
    python scripts/pesa_brain/db/drop_subject_db.py \\
        --subjects "PESA5745609, PESA0091204" --dataset /data/PESA-Brain --write
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.xnat import parse_subject_tokens
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.pipes.qvtpy.stage9_autoqc import QC_VARIABLES

log = Logger()

#: qvtpy writes its measurements under two pipeline ids: stage 6 publishes the hemodynamics as
#: ``4dflow_v3``, stage 9 publishes the automatic QC variables as ``qvtpy``. Morphometrics
#: (``tof_morpho_v1``) are deliberately absent — they come from the TOF acquisition, not the 4D
#: flow, and stay valid when the flow scan does not.
IMAGE_PIPELINES: tuple[str, ...] = (QVTPY_PIPELINE_ID, "qvtpy")

#: Subject-level autoQC variables, taken from the stage-9 registry so the two cannot drift apart.
CLINICAL_VARIABLES: tuple[str, ...] = tuple(
    variable for variable, table in QC_VARIABLES.items() if table == "clinical_measurements"
)


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------
def rows_to_drop(
    frame: pd.DataFrame, *, subjects: list[str], column: str, values: tuple[str, ...]
) -> pd.Series:
    """
    Boolean mask of *frame*'s rows belonging to *subjects* and matching *values* in *column*.

    Both conditions must hold: a subject's rows from another pipeline stay, and another subject's
    qvtpy rows stay. Comparison is on strings because ``pipeline_id`` and ``variable_id`` can come
    back as categoricals or objects depending on how the Parquet was written.
    """
    if frame.empty or column not in frame.columns or "subject_uid" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return (
        frame["subject_uid"].astype(str).isin(subjects)
        & frame[column].astype(str).isin(values)
    )


def describe_drop(frame: pd.DataFrame, mask: pd.Series, *, column: str) -> str:
    """One-line breakdown of what *mask* selects, by subject and by *column* value."""
    if not mask.any():
        return "nothing"
    hit = frame.loc[mask]
    by_value = hit[column].astype(str).value_counts().to_dict()
    subjects = sorted(hit["subject_uid"].astype(str).unique())
    parts = ", ".join(f"{key}={count}" for key, count in sorted(by_value.items()))
    return f"{int(mask.sum())} row(s) over {len(subjects)} subject(s) [{parts}]"


# ---------------------------------------------------------------------------
# Dataset surgery
# ---------------------------------------------------------------------------
def drop_subject_qvtpy(
    repo: Any, *, subjects: list[str], write: bool = False
) -> dict[str, int]:
    """
    Delete the qvtpy rows of *subjects* from both measurement tables.

    Reads with ``cohort_id=False`` so a subject outside the catalog's default cohort is still
    reachable — a scan being excluded is exactly the kind of row a cohort filter might already be
    hiding. Writes go through :meth:`DataRepo.write_table` with the index rebuild deferred, then the
    index is rebuilt once for whichever tables actually changed.

    Returns
    -------
    dict
        ``{table: rows_removed}``, empty values included, so the caller can report a clean no-op.
    """
    targets = (
        ("image_measurements", "pipeline_id", IMAGE_PIPELINES),
        ("clinical_measurements", "variable_id", CLINICAL_VARIABLES),
    )

    removed: dict[str, int] = {}
    touched: list[str] = []
    for table, column, values in targets:
        try:
            frame = repo.get(table, cohort_id=False)
        except Exception as exc:
            log.warning("Could not read %s (%s) — skipping it.", table, exc)
            log.debug("read failed for %s", table, exc_info=True)
            removed[table] = 0
            continue
        if frame is None or frame.empty:
            log.info("%s is empty — nothing to drop.", table)
            removed[table] = 0
            continue

        mask = rows_to_drop(frame, subjects=subjects, column=column, values=values)
        removed[table] = int(mask.sum())
        log.info("%s: %s", table, describe_drop(frame, mask, column=column))
        if not mask.any():
            continue

        if write:
            kept = frame.loc[~mask].reset_index(drop=True)
            repo.write_table(
                table,
                kept,
                provenance={
                    "operation": "drop_subject_qvtpy",
                    "subjects": ",".join(subjects),
                    "rows_removed": int(mask.sum()),
                },
                build_sqlite_index=False,
            )
            log.ok("%s: %d row(s) removed, %d kept.", table, int(mask.sum()), len(kept))
            touched.append(table)

    # One rebuild at the end: the index is derived from Parquet, so rebuilding per table would
    # repeat the same work for no gain.
    if write and touched:
        repo.build_sqlite_index(tables=sorted(touched))
        log.ok("SQLite index rebuilt for %s.", ", ".join(sorted(touched)))
    return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("drop-subject-db")
@click.option(
    "--subjects",
    required=True,
    help="Comma/space-separated subject_uid(s) whose qvtpy results should be removed.",
)
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
    help="Actually delete the rows. The default only reports what would go.",
)
def main(subjects: str, dataset: Path | None, write: bool) -> None:
    """Drop a subject's qvtpy 4D-flow measurements and autoQC variables from the dataset."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    wanted = parse_subject_tokens(subjects)
    if not wanted:
        raise click.ClickException("--subjects resolved to no subject id.")

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(
            f"Could not open the dataset ({exc}). Pass --dataset PATH, or configure one in "
            f".nvitk/settings.json."
        ) from exc

    log.info(
        "Dropping qvtpy results for %d subject(s): %s", len(wanted), ", ".join(wanted)
    )
    log.info(
        "image_measurements pipelines: %s  |  clinical variables: %s",
        ", ".join(IMAGE_PIPELINES), ", ".join(CLINICAL_VARIABLES),
    )

    removed = drop_subject_qvtpy(repo, subjects=wanted, write=write)
    total = sum(removed.values())

    if not total:
        click.echo("Nothing matched — these subjects have no qvtpy rows in this dataset.")
        return
    if write:
        click.echo(
            f"Removed {total} row(s): "
            + ", ".join(f"{table} {count}" for table, count in removed.items() if count)
        )
    else:
        click.echo(
            f"Dry run — {total} row(s) would be removed: "
            + ", ".join(f"{table} {count}" for table, count in removed.items() if count)
            + "\nRe-run with --write to apply."
        )


if __name__ == "__main__":
    main()
