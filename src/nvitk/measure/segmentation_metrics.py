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


#: Whether a larger value of each aggregate metric is better. Used to give a paired comparison
#: a consistent sign, so "positive delta = the candidate won" holds for every row of the table.
METRIC_POLARITY: dict[str, bool] = {
    "class_avg_dice": True,
    "class_avg_cl_dice": True,
    "sideroad_f1": True,
    "class_avg_b0_error": False,
    "class_avg_hd95": False,
    "invalid_neighbours": False,
}


@dataclass(frozen=True)
class PairedComparison:
    """One metric, compared case by case between two models on the same cases.

    A cohort mean hides the thing you need to know when 50 cases are all you have: whether the
    candidate beat the baseline *on the same case*. Two models within a point of each other on
    average can disagree on two thirds of the cases, and a mean cannot tell you that.
    """

    metric: str
    num_cases: int
    """Cases where both models produced a finite value for this metric."""

    mean_delta: float
    """Mean of (candidate − baseline), sign-corrected so positive always favours the candidate."""

    median_delta: float
    num_better: int
    num_worse: int
    num_tied: int
    wilcoxon_p: float
    """Two-sided Wilcoxon signed-rank p-value, or NaN when it cannot be computed.

    Paired and non-parametric, which is what 50 correlated cases with a skewed metric warrant;
    a t-test would assume a normality that HD95 and β0 error plainly do not have.
    """

    baseline_mean: float
    candidate_mean: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "metric": self.metric, "num_cases": self.num_cases,
            "mean_delta": self.mean_delta, "median_delta": self.median_delta,
            "num_better": self.num_better, "num_worse": self.num_worse,
            "num_tied": self.num_tied, "wilcoxon_p": self.wilcoxon_p,
            "baseline_mean": self.baseline_mean, "candidate_mean": self.candidate_mean,
        }


def _wilcoxon_p(deltas: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for *deltas*; NaN when undefined.

    SciPy raises when every difference is zero and warns when the sample is tiny; both mean
    "no evidence here", which is better reported as NaN than as an exception in the middle of
    a reporting run.
    """
    non_zero = [d for d in deltas if d != 0.0]
    if len(non_zero) < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(non_zero).pvalue)
    except Exception as exc:  # degenerate sample; not worth aborting a report for
        log.debug("Wilcoxon test unavailable (%s).", exc)
        return float("nan")


def paired_comparison(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    *,
    metrics: Sequence[str] | None = None,
    polarity: Mapping[str, bool] | None = None,
) -> list[PairedComparison]:
    """Compare two models case by case on the cases they share.

    Parameters
    ----------
    baseline, candidate
        Case id → metric → value, e.g. the ``aggregate`` dict of each :class:`CaseMetrics`.
        Only case ids present in both are compared; anything else is not a pair.
    metrics
        Which metrics to compare. Defaults to every metric both models report.
    polarity
        Metric → whether larger is better. Defaults to :data:`METRIC_POLARITY`; metrics absent
        from it are assumed larger-is-better.

    Returns
    -------
    list of PairedComparison
        One row per metric, in the order given (or sorted when defaulted).

    Raises
    ------
    ValueError
        If the two mappings share no case ids at all — almost always a mismatched run rather
        than a genuinely empty comparison.
    """
    polarity = dict(METRIC_POLARITY if polarity is None else polarity)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError(
            f"No case ids in common between the two runs ({len(baseline)} vs "
            f"{len(candidate)} cases). They did not score the same cohort."
        )
    if len(shared) < max(len(baseline), len(candidate)):
        log.warning(
            "Comparing %d case(s) present in both runs (baseline has %d, candidate %d).",
            len(shared), len(baseline), len(candidate),
        )

    if metrics is None:
        metrics = sorted(
            {k for cid in shared for k in baseline[cid]}
            & {k for cid in shared for k in candidate[cid]}
        )

    rows: list[PairedComparison] = []
    for metric in metrics:
        sign = 1.0 if polarity.get(metric, True) else -1.0
        pairs = [
            (float(baseline[cid][metric]), float(candidate[cid][metric]))
            for cid in shared
            if metric in baseline[cid] and metric in candidate[cid]
            and not np.isnan(baseline[cid][metric]) and not np.isnan(candidate[cid][metric])
        ]
        if not pairs:
            continue
        deltas = [sign * (new - old) for old, new in pairs]
        rows.append(PairedComparison(
            metric=metric,
            num_cases=len(pairs),
            mean_delta=float(np.mean(deltas)),
            median_delta=float(np.median(deltas)),
            num_better=sum(1 for d in deltas if d > 0),
            num_worse=sum(1 for d in deltas if d < 0),
            num_tied=sum(1 for d in deltas if d == 0),
            wilcoxon_p=_wilcoxon_p(deltas),
            baseline_mean=float(np.mean([old for old, _ in pairs])),
            candidate_mean=float(np.mean([new for _, new in pairs])),
        ))
    return rows


def fold_spread(
    cases: Sequence[CaseMetrics],
    case_folds: Mapping[str, int],
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-fold means of each metric, plus their spread across folds.

    The spread is the number that decides whether a difference between two runs means anything.
    If the candidate is 0.02 Dice ahead on the cohort mean but the fold-to-fold standard
    deviation is 0.05, the comparison has not resolved anything and needs either more folds or
    a paired test — see :func:`paired_comparison`.

    Parameters
    ----------
    case_folds
        Case id → index of the fold that held that case out. Cases absent from the mapping are
        skipped, so a partial cross-validation still reports on the folds it does have.
    """
    grouped: dict[int, list[CaseMetrics]] = {}
    for case in cases:
        fold = case_folds.get(case.case_id)
        if fold is not None:
            grouped.setdefault(int(fold), []).append(case)
    if not grouped:
        return {}

    folds = sorted(grouped)
    per_fold = {fold: aggregate_cases(grouped[fold]) for fold in folds}
    if metrics is None:
        metrics = sorted({key for summary in per_fold.values() for key in summary})

    spread: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [
            per_fold[fold][metric] for fold in folds
            if metric in per_fold[fold] and not np.isnan(per_fold[fold][metric])
        ]
        if not values:
            continue
        spread[metric] = {
            "per_fold": {str(fold): per_fold[fold].get(metric) for fold in folds},
            "mean": float(np.mean(values)),
            # Sample standard deviation: five folds are a sample of the splits we could have
            # drawn, not the population, and ddof=0 would understate the spread we care about.
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "num_folds": len(values),
        }
    return spread


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
    "METRIC_POLARITY",
    "CaseMetrics",
    "PairedComparison",
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
    "fold_spread",
    "neighbouring_labels",
    "paired_comparison",
]
