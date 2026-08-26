"""Patient-grouped cross-validation splits for the TopBrain release.

Description
-----------
The release holds 50 volumes from **25 patients**: every patient contributes one CTA and one
MRA. Splitting on the case id would put a patient's CT in training and their MR in validation,
which leaks the subject's anatomy across the split and inflates every metric — the two scans
show the same vessels, so a model that memorised one has largely seen the other.

:func:`grouped_folds` therefore assigns whole *patients* to folds. Assignment is deterministic
(seeded shuffle) so a re-run, or a second stage reading the same dataset, reproduces the split
rather than quietly generating a new one.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain.util.paths import ReleaseCase

log = Logger()


def group_by_patient(cases: Iterable[ReleaseCase]) -> dict[str, list[str]]:
    """Map patient id → sorted case ids belonging to that patient."""
    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        groups[case.patient_id].append(case.case_id)
    return {pid: sorted(ids) for pid, ids in sorted(groups.items())}


def grouped_folds(
    cases: Sequence[ReleaseCase],
    *,
    num_folds: int = 5,
    seed: int = 12345,
) -> list[dict[str, list[str]]]:
    """Build *num_folds* nnU-Net-shaped splits, grouped by patient.

    Parameters
    ----------
    num_folds
        Number of folds. Must be at least 2 and no more than the number of patients.
    seed
        Seed for the patient shuffle. Fixed by default so splits are reproducible.

    Returns
    -------
    list of dict
        One ``{"train": [case_id, ...], "val": [case_id, ...]}`` per fold, case ids sorted —
        the exact shape nnU-Net expects in ``splits_final.json``.

    Raises
    ------
    ValueError
        If there are no cases, or *num_folds* is outside ``2..n_patients``. Silently clamping
        would produce a split that does not match what the caller asked for.
    """
    if not cases:
        raise ValueError("No cases to split.")

    groups = group_by_patient(cases)
    patients = list(groups)
    if not 2 <= num_folds <= len(patients):
        raise ValueError(
            f"num_folds must be between 2 and the number of patients ({len(patients)}); "
            f"got {num_folds}."
        )

    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)

    # Round-robin rather than contiguous chunks: with 25 patients and 5 folds the two differ
    # only in which patients land together, but round-robin keeps fold sizes within one of
    # each other for any num_folds that does not divide the cohort evenly.
    held_out: list[list[str]] = [[] for _ in range(num_folds)]
    for index, patient in enumerate(shuffled):
        held_out[index % num_folds].append(patient)

    folds: list[dict[str, list[str]]] = []
    for fold_index, val_patients in enumerate(held_out):
        val_set = set(val_patients)
        val_cases = sorted(cid for pid in val_patients for cid in groups[pid])
        train_cases = sorted(
            cid for pid, ids in groups.items() if pid not in val_set for cid in ids
        )
        folds.append({"train": train_cases, "val": val_cases})
        log.debug(
            "fold %d: %d val patients (%d cases), %d train cases",
            fold_index,
            len(val_patients),
            len(val_cases),
            len(train_cases),
        )

    return folds


def check_no_patient_leak(folds: Sequence[dict[str, list[str]]]) -> None:
    """Assert no patient appears on both sides of any fold.

    Cheap enough to run every time the splits are written; the failure it catches is invisible
    in the metrics (they just look good) and would invalidate every number downstream.

    Raises
    ------
    ValueError
        Naming the fold and the leaking patients.
    """
    from nvitk.pipes.topbrain.util.paths import parse_case_id

    for index, fold in enumerate(folds):
        train = {parse_case_id(cid)[1] for cid in fold["train"]}
        val = {parse_case_id(cid)[1] for cid in fold["val"]}
        overlap = train & val
        if overlap:
            raise ValueError(
                f"Fold {index} leaks {len(overlap)} patient(s) across train/val: "
                f"{sorted(overlap)}."
            )


__all__ = ["check_no_patient_leak", "group_by_patient", "grouped_folds"]
