"""
Segmentation metrics for thin, tubular, multi-class structures.

Description
-----------
Overlap alone is a poor summary of a vessel segmentation: a result can score well on Dice while
being fragmented into disconnected pieces, because a one-voxel gap costs almost no overlap but
destroys the topology a clinician reads the image for. These metrics add the topological and
detection views that vessel-segmentation benchmarks score alongside Dice:

- :func:`dice` — volumetric overlap
- :func:`cl_dice` — centerline Dice, overlap measured on the skeletons
- :func:`betti_zero_error` — difference in connected-component count
- :func:`hd95` — 95th-percentile Hausdorff distance, in millimetres
- :func:`invalid_neighbour_error` — adjacencies the anatomy does not allow
- :func:`detection_f1` — per-component detection at an IoU threshold

Conventions
-----------
Masks are integer label maps of identical shape. Physical distances are **millimetres**, taken
from an explicit ``spacing`` (z, y, x order matching the array axes) — never voxel counts.

Absent classes
--------------
A class missing from both masks is *correctly* absent and is excluded from class averages
rather than scored as perfect; scoring it 1.0 would let a model inflate its average by
predicting nothing for the classes that are usually absent. A class present in exactly one mask
is a real error and scores worst-case: 0 for Dice and clDice, :data:`HD95_MISSING_PENALTY` for
HD95. This matches how these benchmarks aggregate.

I/O and arrays: host NumPy. These are QC and reporting metrics run once per case, where clarity
matters more than throughput, and the distance transforms and skeletonisation they rely on are
CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger

log = Logger()

#: HD95 substituted when a class is present in exactly one of the two masks — roughly the
#: largest distance available inside a human head, in millimetres.
HD95_MISSING_PENALTY: float = 290.0

#: 26-connectivity: vessels run diagonally through the voxel grid, and 6-connectivity would
#: count a single smoothly-curving vessel as several components.
_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def _as_array(mask: Any) -> np.ndarray:
    """Host NumPy view of *mask*, accepting Image, CuPy or NumPy input."""
    data = mask.data if hasattr(mask, "data") and not isinstance(mask, np.ndarray) else mask
    return to_numpy(data)


def _check_pair(reference: np.ndarray, prediction: np.ndarray) -> None:
    """Validate that two masks describe the same voxel grid."""
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Reference shape {reference.shape} != prediction shape {prediction.shape}."
        )


def _spacing_tuple(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    """Normalise *spacing* to a tuple of length *ndim*, defaulting to isotropic 1 mm."""
    if spacing is None:
        return (1.0,) * ndim
    values = tuple(float(s) for s in spacing)
    if len(values) != ndim:
        raise ValueError(f"Spacing {values} does not match {ndim}-dimensional data.")
    return values


# ---------------------------------------------------------------------------
# Per-class primitives (binary masks)
# ---------------------------------------------------------------------------


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Dice similarity of two boolean masks; 1.0 when both are empty."""
    ref_sum, pred_sum = int(reference.sum()), int(prediction.sum())
    if ref_sum == 0 and pred_sum == 0:
        return 1.0
    return 2.0 * float(np.logical_and(reference, prediction).sum()) / float(ref_sum + pred_sum)


def cl_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Centerline Dice: the harmonic mean of topology precision and sensitivity.

    Skeletonises both masks and asks how much of each skeleton lies inside the *other* mask.
    High only when the prediction reproduces the reference's connectivity, so a fragmented
    result is penalised even when its volumetric overlap is good.
    """
    ref_sum, pred_sum = int(reference.sum()), int(prediction.sum())
    if ref_sum == 0 and pred_sum == 0:
        return 1.0
    if ref_sum == 0 or pred_sum == 0:
        return 0.0

    skel_ref = skeletonize(reference)
    skel_pred = skeletonize(prediction)
    n_ref, n_pred = int(skel_ref.sum()), int(skel_pred.sum())
    if n_ref == 0 or n_pred == 0:
        return 0.0

    precision = float(np.logical_and(skel_pred, reference).sum()) / n_pred
    sensitivity = float(np.logical_and(skel_ref, prediction).sum()) / n_ref
    if precision + sensitivity == 0.0:
        return 0.0
    return 2.0 * precision * sensitivity / (precision + sensitivity)


def count_components(mask: np.ndarray) -> int:
    """Number of 26-connected components in a boolean mask."""
    if not mask.any():
        return 0
    return int(label(mask, structure=_CONNECTIVITY)[1])


def betti_zero_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Absolute difference in connected-component count (the 0th Betti number).

    Zero means the prediction is as fragmented as the reference — not that it is correct — so
    read it alongside Dice rather than on its own.
    """
    return float(abs(count_components(reference) - count_components(prediction)))


def hd95(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    missing_penalty: float = HD95_MISSING_PENALTY,
) -> float:
    """95th-percentile symmetric Hausdorff distance in millimetres.

    The 95th percentile rather than the maximum, so a single stray voxel does not dominate.
    Returns 0.0 when both masks are empty and *missing_penalty* when exactly one is.
    """
    ref_any, pred_any = bool(reference.any()), bool(prediction.any())
    if not ref_any and not pred_any:
        return 0.0
    if not ref_any or not pred_any:
        return float(missing_penalty)

    sampling = _spacing_tuple(spacing, reference.ndim)
    # Distance from every voxel to the nearest foreground voxel of the other mask; sampling
    # makes those distances physical rather than voxel counts.
    dist_to_ref = distance_transform_edt(~reference, sampling=sampling)
    dist_to_pred = distance_transform_edt(~prediction, sampling=sampling)

    surface_distances = np.concatenate(
        [dist_to_ref[prediction], dist_to_pred[reference]]
    )
    return float(np.percentile(surface_distances, 95))


def detection_matches(
    reference: np.ndarray, prediction: np.ndarray, *, iou_threshold: float
) -> tuple[int, int, int]:
    """Component-level ``(true_positives, false_positives, false_negatives)`` at *iou_threshold*.

    Each reference component counts as detected when some predicted component overlaps it with
    at least *iou_threshold* IoU. Used for the small "side road" vessels, where *whether* a
    branch was found matters more than how precisely it was outlined.
    """
    ref_labels, n_ref = label(reference, structure=_CONNECTIVITY)
    pred_labels, n_pred = label(prediction, structure=_CONNECTIVITY)
    if n_ref == 0 and n_pred == 0:
        return 0, 0, 0

    matched_pred: set[int] = set()
    true_positives = 0
    for ref_id in range(1, n_ref + 1):
        ref_component = ref_labels == ref_id
        overlapping = np.unique(pred_labels[ref_component])
        best_iou, best_id = 0.0, None
        for pred_id in overlapping:
            if pred_id == 0 or int(pred_id) in matched_pred:
                continue
            pred_component = pred_labels == pred_id
            union = int(np.logical_or(ref_component, pred_component).sum())
            iou = float(np.logical_and(ref_component, pred_component).sum()) / union if union else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, int(pred_id)
        if best_id is not None and best_iou >= iou_threshold:
            matched_pred.add(best_id)
            true_positives += 1

    return true_positives, n_pred - len(matched_pred), n_ref - true_positives


def f1_from_counts(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """F1 from detection counts; 1.0 when there was nothing to detect and nothing was predicted."""
    if true_positives == 0 and false_positives == 0 and false_negatives == 0:
        return 1.0
    denominator = 2 * true_positives + false_positives + false_negatives
    return (2.0 * true_positives / denominator) if denominator else 0.0


# ---------------------------------------------------------------------------
# Anatomical adjacency
# ---------------------------------------------------------------------------


def neighbouring_labels(mask: np.ndarray, *, labels: Iterable[int] | None = None) -> set[frozenset[int]]:
    """Unordered pairs of labels that touch each other in *mask* (26-connectivity).

    Two labels "touch" when a voxel of one is within one voxel of the other in any direction.
    Implemented by dilating each label by a single step and intersecting, which is exact for
    26-connectivity and avoids materialising a full pairwise adjacency matrix.
    """
    from scipy.ndimage import binary_dilation

    present = sorted(labels) if labels is not None else [
        int(v) for v in np.unique(mask) if int(v) != 0
    ]
    pairs: set[frozenset[int]] = set()
    dilated = {value: binary_dilation(mask == value, structure=_CONNECTIVITY) for value in present}
    for i, first in enumerate(present):
        for second in present[i + 1:]:
            if np.logical_and(dilated[first], mask == second).any():
                pairs.add(frozenset((first, second)))
    return pairs


def invalid_neighbour_error(
    prediction: np.ndarray,
    *,
    valid_neighbours: Mapping[int, Iterable[int]],
    labels: Iterable[int] | None = None,
) -> float:
    """Count of touching label pairs that the anatomy does not permit.

    Parameters
    ----------
    valid_neighbours
        Label → labels it may legitimately touch. Pairs absent from the mapping are counted as
        violations, so an incomplete mapping over-reports rather than silently passing errors.

    A low score means the prediction is anatomically plausible: vessels only meet where they
    actually branch. It catches the characteristic failure of multi-class vessel models — two
    labels bleeding into each other where they run side by side.
    """
    violations = 0
    for pair in neighbouring_labels(prediction, labels=labels):
        first, second = sorted(pair)
        # Adjacency is symmetric, but a hand-written table often lists a pair only one way, so
        # accept the edge if either direction permits it.
        permitted = second in set(valid_neighbours.get(first, ())) or first in set(
            valid_neighbours.get(second, ())
        )
        if not permitted:
            violations += 1
    return float(violations)


# ---------------------------------------------------------------------------
# Case-level aggregation
# ---------------------------------------------------------------------------


@dataclass
class CaseMetrics:
    """All metrics for one case, per class and class-averaged."""

    case_id: str
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "case_id": self.case_id,
            "aggregate": self.aggregate,
            "per_class": {str(k): v for k, v in self.per_class.items()},
        }


def evaluate_case(
    reference: Any,
    prediction: Any,
    *,
    case_id: str,
    labels: Sequence[int],
    spacing: Sequence[float] | None = None,
    sideroad_labels: Sequence[int] = (),
    valid_neighbours: Mapping[int, Iterable[int]] | None = None,
    iou_threshold: float = 0.25,
) -> CaseMetrics:
    """Compute every metric for one reference/prediction pair.

    Classes absent from **both** masks are skipped in the class averages — see the module
    docstring on why that matters when most classes are absent from most cases.
    """
    ref = _as_array(reference).astype(np.int32, copy=False)
    pred = _as_array(prediction).astype(np.int32, copy=False)
    _check_pair(ref, pred)

    metrics = CaseMetrics(case_id=case_id)
    scored: dict[str, list[float]] = {"dice": [], "cl_dice": [], "b0_error": [], "hd95": []}

    for value in labels:
        ref_mask, pred_mask = ref == value, pred == value
        if not ref_mask.any() and not pred_mask.any():
            continue  # correctly absent
        entry = {
            "dice": dice(ref_mask, pred_mask),
            "cl_dice": cl_dice(ref_mask, pred_mask),
            "b0_error": betti_zero_error(ref_mask, pred_mask),
            "hd95": hd95(ref_mask, pred_mask, spacing=spacing),
        }
        metrics.per_class[int(value)] = entry
        for key, value_ in entry.items():
            scored[key].append(value_)

    for key, values in scored.items():
        metrics.aggregate[f"class_avg_{key}"] = float(np.mean(values)) if values else float("nan")

    # Side-road detection is pooled across classes: the F1 answers "how many small branches did
    # we find", which is a cohort-level question, not a per-class one.
    if sideroad_labels:
        tp = fp = fn = 0
        for value in sideroad_labels:
            case_tp, case_fp, case_fn = detection_matches(
                ref == value, pred == value, iou_threshold=iou_threshold
            )
            tp, fp, fn = tp + case_tp, fp + case_fp, fn + case_fn
        metrics.aggregate["sideroad_f1"] = f1_from_counts(tp, fp, fn)
        metrics.aggregate["sideroad_tp"] = float(tp)
        metrics.aggregate["sideroad_fp"] = float(fp)
        metrics.aggregate["sideroad_fn"] = float(fn)

    if valid_neighbours is not None:
        metrics.aggregate["invalid_neighbours"] = invalid_neighbour_error(
            pred, valid_neighbours=valid_neighbours, labels=labels
        )

    return metrics


def aggregate_cases(cases: Sequence[CaseMetrics]) -> dict[str, float]:
    """Cohort means of each aggregate metric, ignoring cases where a metric was undefined."""
    if not cases:
        return {}
    keys = sorted({key for case in cases for key in case.aggregate})
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            case.aggregate[key]
            for case in cases
            if key in case.aggregate and not np.isnan(case.aggregate[key])
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


__all__ = [
    "HD95_MISSING_PENALTY",
    "CaseMetrics",
    "aggregate_cases",
    "betti_zero_error",
    "cl_dice",
    "count_components",
    "detection_matches",
    "dice",
    "evaluate_case",
    "f1_from_counts",
    "hd95",
    "invalid_neighbour_error",
    "neighbouring_labels",
]
