"""Left-right mirroring as an augmentation, for label sets whose classes are lateralised.

nnU-Net's default mirroring is disabled for this pipeline (see
:mod:`...nnUNetTrainer.topbrain.topbrain_trainers`): flipping a volume left to right leaves the
mask claiming that what is now on the left is still ``R-ICA``, which teaches the network that
laterality carries no information. That is the right call for plain mirroring.

Mirroring becomes *correct* the moment the label values are swapped along with the voxels --
every ``R-x`` becomes ``L-x`` and vice versa. The result is an anatomically plausible volume
with a consistent mask, and it doubles the effective size of a cohort that, for a single
modality, is 25 cases.

Two things make this safe rather than clever:

* the swap map is derived from the label names, not hard-coded, so classes with no mirrored
  partner (BA, Acom, the 3rd-A2 variants) are left alone by construction;
* the axis to flip is **measured**, not assumed. Which array axis runs left-right depends on
  the acquisition and on nnU-Net's ``transpose_forward``, and flipping the wrong one produces
  a volume that is upside down or back to front with a mask that looks fine. If the
  measurement is not unambiguous, the augmentation declines to run.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain import labels as lbl

log = Logger()

#: Environment variable carrying the flag into the trainer, alongside the sampling spec.
LATERAL_SWAP_ENV: str = "TOPBRAIN_LATERAL_SWAP"

#: Carries the label set to the trainer, which needs it to build the swap map.
LABEL_SET_ENV: str = "TOPBRAIN_LABEL_SET"

#: Probability a training patch is mirrored when the augmentation is on.
DEFAULT_PROBABILITY: float = 0.5

#: How much the left-right axis has to dominate the other two before it is believed. The R and L
#: centroids of a head differ mostly along one axis and only slightly along the others; a factor
#: of three is comfortably above that noise and far below a genuine separation.
AXIS_DOMINANCE: float = 3.0


def swap_map(label_set: str) -> dict[int, int]:
    """Label value → its mirrored partner, both directions.

    Built from :func:`nvitk.pipes.topbrain.labels.lateral_pairs`, which reads the ``R-``/``L-``
    prefixes off the label map. Unpaired classes are absent, and a mirrored volume keeps them
    unchanged -- which is correct: they have no side to swap.
    """
    pairs = lbl.lateral_pairs(label_set)
    mapping: dict[int, int] = {}
    for right, left in pairs.items():
        mapping[int(right)] = int(left)
        mapping[int(left)] = int(right)
    return mapping


def apply_swap(segmentation: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Return *segmentation* with every lateralised value replaced by its partner.

    Uses a lookup table rather than successive ``where`` calls: chaining the replacements would
    swap a value and then swap it back the moment its partner is processed.
    """
    if not mapping:
        return segmentation
    table = np.arange(int(max(segmentation.max(), max(mapping))) + 1, dtype=segmentation.dtype)
    for source, target in mapping.items():
        if source < len(table):
            table[source] = target
    return table[segmentation]


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    """Centre of mass of a boolean mask, or ``None`` when it is empty."""
    coordinates = np.argwhere(mask)
    return coordinates.mean(axis=0) if coordinates.size else None


def detect_flip_axis(
    segmentations: Iterable[np.ndarray], mapping: dict[int, int], *, label_set: str = ""
) -> int | None:
    """The spatial axis along which left and right sit, measured from the masks.

    For each segmentation, the centroid of every ``R-`` class is compared with the centroid of
    every ``L-`` class. In a head those two points differ substantially along exactly one axis.
    The axis is accepted when it is the same for every readable case *and* dominates the other
    two by :data:`AXIS_DOMINANCE` in each of them.

    Returns
    -------
    int or None
        The axis index into the spatial dimensions, or ``None`` when the cases disagree, none
        of them contains both sides, or no axis dominates. ``None`` means "do not mirror": a
        wrong axis silently trains on volumes that are upside down.
    """
    rights = {value for value in mapping if value in lbl.lateral_pairs(label_set or "ta36")}
    lefts = {mapping[value] for value in rights}
    if not rights:
        return None

    votes: list[int] = []
    for segmentation in segmentations:
        array = np.asarray(segmentation)
        array = array[0] if array.ndim == 4 else array  # drop a leading channel if present
        right_centroid = _centroid(np.isin(array, list(rights)))
        left_centroid = _centroid(np.isin(array, list(lefts)))
        if right_centroid is None or left_centroid is None:
            continue  # one side absent: this case cannot vote
        separation = np.abs(right_centroid - left_centroid)
        order = np.argsort(separation)[::-1]
        best, runner_up = separation[order[0]], separation[order[1]]
        if runner_up > 0 and best < AXIS_DOMINANCE * runner_up:
            continue  # no axis stands out; an ambiguous case is not evidence
        votes.append(int(order[0]))

    if not votes:
        log.warning("Lateral swap: could not measure the left-right axis from any case.")
        return None
    if len(set(votes)) > 1:
        log.warning(
            "Lateral swap: cases disagree on the left-right axis (%s); not mirroring.",
            ", ".join(str(v) for v in sorted(set(votes))),
        )
        return None
    return votes[0]


def sample_segmentations(loader, identifiers: Sequence[str], *, limit: int = 8) -> list:
    """Load up to *limit* preprocessed segmentations through *loader*.

    The axis is measured on the arrays the dataloader actually serves, not on the raw NIfTIs:
    preprocessing applies the plan's ``transpose_forward``, so the raw orientation says nothing
    about which axis the trainer will see.
    """
    sampled = []
    for identifier in list(identifiers)[:limit]:
        try:
            _data, segmentation, _properties = loader.load_case(identifier)
        except Exception:  # noqa: BLE001 - a case that will not load is simply not evidence
            continue
        sampled.append(segmentation)
    return sampled


__all__ = [
    "AXIS_DOMINANCE",
    "DEFAULT_PROBABILITY",
    "LABEL_SET_ENV",
    "LATERAL_SWAP_ENV",
    "apply_swap",
    "detect_flip_axis",
    "sample_segmentations",
    "swap_map",
]
