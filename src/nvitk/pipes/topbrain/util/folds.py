"""Patient-grouped cross-validation splits for the TopBrain release.

Description
-----------
The release holds 50 volumes from **25 patients**: every patient contributes one CTA and one
MRA. Splitting on the case id would put a patient's CT in training and their MR in validation,
which leaks the subject's anatomy across the split and inflates every metric — the two scans
show the same vessels, so a model that memorised one has largely seen the other.

:func:`grouped_folds` therefore assigns whole *patients* to folds. Assignment is deterministic
(seeded) so a re-run, or a second stage reading the same dataset, reproduces the split rather
than quietly generating a new one.

Why the assignment is stratified
--------------------------------
With 25 patients and 36 classes, a plain shuffle is not good enough. The side roads (AChA, OA,
Pcom, 3rd-A2/A3) are anatomically absent or hypoplastic in a large minority of people, so some
classes are present in only a handful of subjects. A shuffle will regularly produce a fold whose
**validation split contains no positive example of a class at all** — that class then contributes
nothing (or a NaN) to the class-average metrics for that fold — or, worse, a fold whose
*training* split has none, so the model cannot possibly learn it and is then scored on it.

Either way the fold-to-fold variance of the headline metric swamps the few points that separate
two configurations, and the cross-validation stops being able to rank anything.
:func:`stratified_patient_folds` therefore spreads the rare classes across folds first, rarest
first, and only then fills the folds up with the remaining patients.

Presence is a **patient**-level property here: a class counts as present for a patient if it
appears in either of their scans, because both scans move together between folds.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain.util.paths import ReleaseCase

log = Logger()


def group_by_patient(cases: Iterable[ReleaseCase]) -> dict[str, list[str]]:
    """Map patient id → sorted case ids belonging to that patient."""
    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        groups[case.patient_id].append(case.case_id)
    return {pid: sorted(ids) for pid, ids in sorted(groups.items())}


def patient_presence(
    cases: Iterable[ReleaseCase], class_presence: Mapping[str, Iterable[int]]
) -> dict[str, set[int]]:
    """Patient id → the union of foreground classes present in that patient's scans.

    Parameters
    ----------
    class_presence
        Case id → foreground label values present in that case's mask, as stage 0 records it
        while converting. Cases missing from the mapping contribute nothing rather than raising:
        a partially converted dataset should still produce a usable (if less well stratified)
        split, and the caller warns about the gap.
    """
    presence: dict[str, set[int]] = defaultdict(set)
    for case in cases:
        presence[case.patient_id].update(
            int(v) for v in class_presence.get(case.case_id, ()) if int(v) != 0
        )
    return dict(presence)


def _fold_capacities(num_patients: int, num_folds: int) -> list[int]:
    """Target patient count per fold, as evenly as the cohort divides."""
    base, remainder = divmod(int(num_patients), int(num_folds))
    return [base + (1 if index < remainder else 0) for index in range(int(num_folds))]


def stratified_patient_folds(
    presence: Mapping[str, Iterable[int]],
    *,
    num_folds: int = 5,
    seed: int = 12345,
) -> list[list[str]]:
    """Assign patients to held-out folds, spreading rare classes evenly.

    Strategy
    --------
    1. Order the classes by how many patients carry them, **rarest first**. The rarest class has
       the least room to be spread badly, so it gets first claim on the layout.
    2. For each class in that order, hand its still-unassigned patients to the folds that are
       furthest below their fair share of that class, subject to a hard per-fold capacity so
       the folds stay the same size.
    3. Patients carrying no class that needed spreading (or already placed) fill the emptiest
       folds.

    This is iterative stratification restricted to the labels that actually matter here. It is
    not optimal — no cheap algorithm is, and with 25 patients some classes simply cannot be
    spread across 5 folds — but it removes the avoidable empty folds, which is the whole point.

    Parameters
    ----------
    presence
        Patient id → foreground classes present in that patient's scans.
    seed
        Fixed so the split is reproducible. Only breaks ties and shuffles the input order; it
        does not otherwise steer the assignment.

    Returns
    -------
    list of list of str
        Held-out patient ids, one list per fold.
    """
    patients = sorted(presence)
    rng = random.Random(seed)
    rng.shuffle(patients)

    capacity = _fold_capacities(len(patients), num_folds)
    folds: list[list[str]] = [[] for _ in range(int(num_folds))]
    assigned_counts: list[dict[int, int]] = [defaultdict(int) for _ in range(int(num_folds))]
    unassigned = list(patients)

    # ---- 1. Classes, rarest first -------------------------------------------
    carriers: dict[int, list[str]] = defaultdict(list)
    for patient in patients:
        for value in presence[patient]:
            carriers[int(value)].append(patient)
    order = sorted(carriers, key=lambda v: (len(carriers[v]), v))

    def _place(patient: str, fold: int) -> None:
        """Put *patient* in *fold* and record what classes that fold gained."""
        folds[fold].append(patient)
        for value in presence[patient]:
            assigned_counts[fold][int(value)] += 1
        unassigned.remove(patient)

    for value in order:
        pending = [p for p in carriers[value] if p in unassigned]
        if not pending:
            continue
        ideal = len(carriers[value]) / float(num_folds)
        for patient in pending:
            open_folds = [f for f in range(int(num_folds)) if len(folds[f]) < capacity[f]]
            if not open_folds:
                break
            # Largest deficit in this class wins; ties go to the emptiest fold, then to the
            # lowest index so the result is a pure function of (presence, seed).
            best = max(
                open_folds,
                key=lambda f: (ideal - assigned_counts[f][value], -len(folds[f]), -f),
            )
            _place(patient, best)

    # ---- 2. Everyone else fills the emptiest folds --------------------------
    for patient in list(unassigned):
        open_folds = [f for f in range(int(num_folds)) if len(folds[f]) < capacity[f]]
        target = min(open_folds or range(int(num_folds)), key=lambda f: (len(folds[f]), f))
        _place(patient, target)

    return [sorted(fold) for fold in folds]


def grouped_folds(
    cases: Sequence[ReleaseCase],
    *,
    num_folds: int = 5,
    seed: int = 12345,
    class_presence: Mapping[str, Iterable[int]] | None = None,
    train_only: Iterable[str] | None = None,
) -> list[dict[str, list[str]]]:
    """Build *num_folds* nnU-Net-shaped splits, grouped by patient.

    Parameters
    ----------
    num_folds
        Number of folds. Must be at least 2 and no more than the number of patients.
    seed
        Seed for the patient assignment. Fixed by default so splits are reproducible.
    class_presence
        Case id → foreground labels present in that case. When given, patients are assigned by
        :func:`stratified_patient_folds` so the rare classes are spread across folds; without
        it the assignment falls back to a seeded round-robin, which is reproducible but blind
        to which folds end up with no example of a class.
    train_only
        Case ids that belong in the **training** half of every fold and in no validation half.
        Pseudo-labelled cases must be passed here: their masks are model output, so scoring a
        model against them measures agreement with the previous model, not accuracy, and mixing
        them into validation silently redefines what the cross-validation reports.

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

    # Train-only cases take no part in the fold layout at all: they are not held out, so they
    # neither need a patient slot nor may influence how the scored patients are distributed.
    fixed = {str(c) for c in (train_only or ())}
    splittable = [case for case in cases if case.case_id not in fixed]
    if fixed and not splittable:
        raise ValueError(
            f"All {len(fixed)} case(s) are marked train-only; there is nothing left to hold out."
        )
    if fixed:
        log.info(
            "%d case(s) are train-only and will appear in every fold's training half.",
            len(fixed),
        )
    cases = splittable

    groups = group_by_patient(cases)
    patients = list(groups)
    if not 2 <= num_folds <= len(patients):
        raise ValueError(
            f"num_folds must be between 2 and the number of patients ({len(patients)}); "
            f"got {num_folds}."
        )

    if class_presence is not None:
        held_out = stratified_patient_folds(
            patient_presence(cases, class_presence), num_folds=num_folds, seed=seed
        )
    else:
        log.warning(
            "No class presence supplied; falling back to a blind round-robin split. Rare "
            "classes may end up absent from a fold's train or validation half."
        )
        shuffled = list(patients)
        random.Random(seed).shuffle(shuffled)
        # Round-robin rather than contiguous chunks: with 25 patients and 5 folds the two
        # differ only in which patients land together, but round-robin keeps fold sizes within
        # one of each other for any num_folds that does not divide the cohort evenly.
        held_out = [[] for _ in range(num_folds)]
        for index, patient in enumerate(shuffled):
            held_out[index % num_folds].append(patient)

    folds: list[dict[str, list[str]]] = []
    for fold_index, val_patients in enumerate(held_out):
        val_set = set(val_patients)
        val_cases = sorted(cid for pid in val_patients for cid in groups[pid])
        train_cases = sorted(
            cid for pid, ids in groups.items() if pid not in val_set for cid in ids
        )
        folds.append({"train": sorted(train_cases + sorted(fixed)), "val": val_cases})
        log.debug(
            "fold %d: %d val patients (%d cases), %d train cases",
            fold_index,
            len(val_patients),
            len(val_cases),
            len(train_cases),
        )

    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Per-class × per-fold presence
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassFoldPresence:
    """How many cases carry one class, cohort-wide and on each side of each fold."""

    label: int
    name: str
    num_cases: int
    """Cases anywhere in the cohort whose mask contains this class."""

    num_patients: int
    train_counts: tuple[int, ...]
    """Per fold: cases in the *training* half carrying this class."""

    val_counts: tuple[int, ...]
    """Per fold: cases in the *validation* half carrying this class."""

    @property
    def empty_train_folds(self) -> tuple[int, ...]:
        """Folds that would train on no example of this class — it cannot be learned there."""
        return tuple(i for i, n in enumerate(self.train_counts) if n == 0)

    @property
    def empty_val_folds(self) -> tuple[int, ...]:
        """Folds that score no example of this class — it contributes nothing to their metrics."""
        return tuple(i for i, n in enumerate(self.val_counts) if n == 0)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "label": self.label, "name": self.name, "num_cases": self.num_cases,
            "num_patients": self.num_patients, "train_counts": list(self.train_counts),
            "val_counts": list(self.val_counts),
            "empty_train_folds": list(self.empty_train_folds),
            "empty_val_folds": list(self.empty_val_folds),
        }


def class_presence_table(
    cases: Sequence[ReleaseCase],
    folds: Sequence[Mapping[str, Sequence[str]]],
    class_presence: Mapping[str, Iterable[int]],
    *,
    label_map: Mapping[int, str],
) -> list[ClassFoldPresence]:
    """Cross-tabulate every class against every fold.

    This is the table that tells you whether a cross-validation can support the comparison you
    are about to make. A class with two positive cases in the whole cohort will have empty folds
    no matter how well they are stratified, and its per-class Dice is not evidence of anything —
    far better to see that in a table before the run than to read a confident-looking mean
    afterwards.

    Rows are sorted rarest first, because that is the end of the table that matters.
    """
    presence = {cid: {int(v) for v in values if int(v) != 0}
                for cid, values in class_presence.items()}
    patient_of = {case.case_id: case.patient_id for case in cases}

    table: list[ClassFoldPresence] = []
    for label in sorted(label_map):
        carriers = {cid for cid, values in presence.items() if label in values}
        table.append(ClassFoldPresence(
            label=int(label),
            name=str(label_map[label]),
            num_cases=len(carriers),
            num_patients=len({patient_of.get(cid, cid) for cid in carriers}),
            train_counts=tuple(
                sum(1 for cid in fold["train"] if cid in carriers) for fold in folds
            ),
            val_counts=tuple(
                sum(1 for cid in fold["val"] if cid in carriers) for fold in folds
            ),
        ))
    return sorted(table, key=lambda row: (row.num_cases, row.label))


def format_presence_table(table: Sequence[ClassFoldPresence]) -> str:
    """Render :func:`class_presence_table` as fixed-width text for the log and a sidecar.

    ``train`` / ``val`` columns are per fold; a ``.`` marks a zero, so the holes are visible at
    a glance rather than needing to be read off as digits.
    """
    if not table:
        return "(no classes)"
    num_folds = len(table[0].val_counts)
    header = (
        f"{'lbl':>4} {'class':<16} {'cases':>5} {'pats':>4}  "
        + " ".join(f"f{i}(tr/val)" for i in range(num_folds))
    )
    lines = [header, "-" * len(header)]
    for row in table:
        cells = " ".join(
            f"{(str(t) if t else '.'):>3}/{(str(v) if v else '.'):<5}"
            for t, v in zip(row.train_counts, row.val_counts)
        )
        lines.append(
            f"{row.label:>4} {row.name[:16]:<16} {row.num_cases:>5} {row.num_patients:>4}  {cells}"
        )
    return "\n".join(lines)


def warn_empty_classes(table: Sequence[ClassFoldPresence]) -> list[str]:
    """Log (and return) one warning per class with an empty train or validation half.

    Not an error: with 25 patients some classes genuinely cannot be spread over five folds, and
    refusing to build the dataset would be unhelpful. But every one of these is a class whose
    per-fold numbers must not be read as a measurement.
    """
    messages: list[str] = []
    for row in table:
        if row.empty_train_folds:
            messages.append(
                f"class {row.label} ({row.name}) is absent from the TRAINING half of fold(s) "
                f"{list(row.empty_train_folds)} — those folds cannot learn it, yet score it."
            )
        if row.empty_val_folds:
            messages.append(
                f"class {row.label} ({row.name}) is absent from the VALIDATION half of fold(s) "
                f"{list(row.empty_val_folds)} — it contributes nothing to their class averages."
            )
    for message in messages:
        log.warning("%s", message)
    if not messages:
        log.ok("every class has at least one case on both sides of every fold")
    return messages


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

    def _patient(case_id: str) -> str:
        """Patient key for *case_id*, tolerating cohorts that do not use the release scheme.

        ``--extra-train`` cohorts get ``<cohort>_<stem>`` ids, which ``parse_case_id`` rejects
        by design. Falling back to the id itself is right for them: each such case is its own
        patient, so it cannot leak.
        """
        try:
            return parse_case_id(case_id)[1]
        except ValueError:
            return case_id

    for index, fold in enumerate(folds):
        val_ids = set(fold["val"])
        # A train-only case appears in every fold's training half and in no validation half, so
        # it cannot leak; comparing on patient keys alone would not know that.
        train = {_patient(cid) for cid in fold["train"] if cid not in val_ids}
        val = {_patient(cid) for cid in val_ids}
        overlap = train & val
        if overlap:
            raise ValueError(
                f"Fold {index} leaks {len(overlap)} patient(s) across train/val: "
                f"{sorted(overlap)}."
            )


__all__ = [
    "ClassFoldPresence",
    "check_no_patient_leak",
    "class_presence_table",
    "format_presence_table",
    "group_by_patient",
    "grouped_folds",
    "patient_presence",
    "stratified_patient_folds",
    "warn_empty_classes",
]
