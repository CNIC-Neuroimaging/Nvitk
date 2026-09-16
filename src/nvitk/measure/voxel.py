"""
Voxel-based overlap metrics (``dice``, ``jaccard``, ``precision``, ...).

All primitives accept :class:`~nvitk.types.Image` or raw arrays. Unlike the
BioImaging legacy implementation, the dispatching ``voxel_metrics`` function has
a single unambiguous signature (it no longer silently interprets a ``dict``
as a callable registry).

Counts are computed on the active backend; a single host hop is used to
produce the final Python ``int`` for each confusion cell.
"""

from __future__ import annotations

from typing import Any, Iterable

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup

from ._common import backend_array, bool_mask, ensure_same_shape, label_mask

setup(globals())


def label_ids_present(*masks: Any) -> list[int]:
    """Sorted non-zero label ids occurring in any of *masks*.

    The union rather than one mask's values: a label the prediction invented and
    one it missed entirely are both real errors, and scoring only over the
    reference's labels would hide the first while scoring only over the
    prediction's would hide the second.
    """
    found: set[int] = set()
    for mask in masks:
        if mask is None:
            continue
        arr = backend_array(mask)
        for value in to_numpy(np.unique(arr)).tolist():
            ivalue = int(value)
            if ivalue != 0:
                found.add(ivalue)
    return sorted(found)


def confusion_counts(
    label_true: Any, label_pred: Any, *, label: int | None = None
) -> dict[str, int]:
    """
    Return ``{'TP','TN','FP','FN'}`` counts from two binary masks.

    With *label*, both sides are reduced to that label id (one-vs-rest) instead
    of to "any non-zero voxel". On multi-label segmentations the default is
    almost never what you want: it scores a prediction that puts every voxel
    under the *wrong* label as a perfect match, because binarising erases the
    distinction it is supposed to be measuring.
    """
    ensure_same_shape(label_true, label_pred)
    a = label_mask(label_true, label).ravel()
    b = label_mask(label_pred, label).ravel()
    # Backend math; materialize only the final Python ints.
    tp = int(to_numpy(np.logical_and(a, b).sum()))
    tn = int(to_numpy(np.logical_and(~a, ~b).sum()))
    fp = int(to_numpy(np.logical_and(~a, b).sum()))
    fn = int(to_numpy(np.logical_and(a, ~b).sum()))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def dice(label_true: Any, label_pred: Any) -> float:
    """Dice overlap ``2·TP / (2·TP + FP + FN)`` in ``[0, 1]`` (``0.0`` when both masks are empty)."""
    c = confusion_counts(label_true, label_pred)
    denom = 2 * c["TP"] + c["FP"] + c["FN"]
    return (2 * c["TP"]) / denom if denom > 0 else 0.0


def jaccard(label_true: Any, label_pred: Any) -> float:
    """Jaccard / IoU ``TP / (TP + FP + FN)`` in ``[0, 1]``."""
    c = confusion_counts(label_true, label_pred)
    denom = c["TP"] + c["FP"] + c["FN"]
    return c["TP"] / denom if denom > 0 else 0.0


def precision(label_true: Any, label_pred: Any) -> float:
    """Fraction of predicted-positive voxels that are correct ``TP / (TP + FP)``."""
    c = confusion_counts(label_true, label_pred)
    return c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else 0.0


def recall(label_true: Any, label_pred: Any) -> float:
    """Fraction of true-positive voxels that were recovered ``TP / (TP + FN)`` (sensitivity)."""
    c = confusion_counts(label_true, label_pred)
    return c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else 0.0


def fpr(label_true: Any, label_pred: Any) -> float:
    """False-positive rate ``FP / (FP + TN)``."""
    c = confusion_counts(label_true, label_pred)
    return c["FP"] / (c["FP"] + c["TN"]) if (c["FP"] + c["TN"]) > 0 else 0.0


def fnr(label_true: Any, label_pred: Any) -> float:
    """False-negative rate ``FN / (FN + TP)``."""
    c = confusion_counts(label_true, label_pred)
    return c["FN"] / (c["FN"] + c["TP"]) if (c["FN"] + c["TP"]) > 0 else 0.0


def volume_similarity(label_true: Any, label_pred: Any) -> float:
    """Relative volume difference ``|n_pred − n_true| / n_true`` (0 = identical volumes)."""
    c = confusion_counts(label_true, label_pred)
    n_true = c["TP"] + c["FN"]
    n_pred = c["TP"] + c["FP"]
    return abs(n_pred - n_true) / n_true if n_true > 0 else 0.0


def volsim(label_true: Any, label_pred: Any) -> float:
    """Volume similarity (Taha & Hanbury): ``1 - |FN - FP| / (2·TP + FP + FN)``.

    Bounded in ``[0, 1]``, 1.0 for identical volumes. Distinct from
    :func:`volume_similarity` in this module, which is a relative volume
    *difference* (0 = identical, unbounded above) — check which one you want.

    Deliberately blind to overlap: two masks of equal size score 1.0 even when
    disjoint. It answers "is the volume right?", not "is it in the right place",
    so pair it with an overlap metric rather than reporting it alone.
    """
    c = confusion_counts(label_true, label_pred)
    denominator = 2 * c["TP"] + c["FP"] + c["FN"]
    if denominator == 0:
        return 1.0
    return 1.0 - abs(c["FN"] - c["FP"]) / float(denominator)


def mcc(label_true: Any, label_pred: Any) -> float:
    """Matthews correlation coefficient, in ``[-1, 1]``; 0.0 is chance agreement.

    A correlation over the whole confusion matrix, so unlike Dice it also has to
    get the true negatives right — which is what makes it informative on a mask
    that is mostly background. Returns 0.0 where it is undefined (an all-one or
    all-zero mask leaves a row or column of the matrix empty).
    """
    import math

    c = confusion_counts(label_true, label_pred)
    tp, tn, fp, fn = float(c["TP"]), float(c["TN"]), float(c["FP"]), float(c["FN"])
    # Multiply as floats: these products overflow int64 on a large volume.
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0.0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denominator)


def _dice_c(c: dict[str, int]) -> float:
    denom = 2 * c["TP"] + c["FP"] + c["FN"]
    return (2 * c["TP"]) / denom if denom > 0 else 0.0


def _jaccard_c(c: dict[str, int]) -> float:
    denom = c["TP"] + c["FP"] + c["FN"]
    return c["TP"] / denom if denom > 0 else 0.0


def _precision_c(c: dict[str, int]) -> float:
    return c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else 0.0


def _recall_c(c: dict[str, int]) -> float:
    return c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else 0.0


def _fpr_c(c: dict[str, int]) -> float:
    return c["FP"] / (c["FP"] + c["TN"]) if (c["FP"] + c["TN"]) > 0 else 0.0


def _fnr_c(c: dict[str, int]) -> float:
    return c["FN"] / (c["FN"] + c["TP"]) if (c["FN"] + c["TP"]) > 0 else 0.0


def _volume_similarity_c(c: dict[str, int]) -> float:
    n_true = c["TP"] + c["FN"]
    n_pred = c["TP"] + c["FP"]
    return abs(n_pred - n_true) / n_true if n_true > 0 else 0.0


def _volsim_c(c: dict[str, int]) -> float:
    denominator = 2 * c["TP"] + c["FP"] + c["FN"]
    if denominator == 0:
        return 1.0
    return 1.0 - abs(c["FN"] - c["FP"]) / float(denominator)


def _mcc_c(c: dict[str, int]) -> float:
    import math

    tp, tn, fp, fn = float(c["TP"]), float(c["TN"]), float(c["FP"]), float(c["FN"])
    # Multiply as floats: these products overflow int64 on a large volume.
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0.0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(denominator)


#: Metric name -> function of a confusion-counts dict. The public one-shot
#: functions above are thin wrappers over these.
_FROM_COUNTS: dict[str, Any] = {
    "TP": lambda c: float(c["TP"]),
    "TN": lambda c: float(c["TN"]),
    "FP": lambda c: float(c["FP"]),
    "FN": lambda c: float(c["FN"]),
    "dice": _dice_c,
    "jaccard": _jaccard_c,
    "precision": _precision_c,
    "recall": _recall_c,
    "fpr": _fpr_c,
    "fnr": _fnr_c,
    "vs": _volume_similarity_c,
    "volsim": _volsim_c,
    "mcc": _mcc_c,
}

_METRICS: dict[str, Any] = {
    "TP": lambda t, p: confusion_counts(t, p)["TP"],
    "TN": lambda t, p: confusion_counts(t, p)["TN"],
    "FP": lambda t, p: confusion_counts(t, p)["FP"],
    "FN": lambda t, p: confusion_counts(t, p)["FN"],
    "dice": dice,
    "jaccard": jaccard,
    "precision": precision,
    "recall": recall,
    "fpr": fpr,
    "fnr": fnr,
    "vs": volume_similarity,
    "volsim": volsim,
    "mcc": mcc,
}


def voxel_metrics(
    label_true: Any,
    label_pred: Any,
    *,
    metrics: Iterable[str] | None = None,
    label: int | None = None,
) -> dict[str, float]:
    """
    Compute a named subset of voxel-based metrics.

    Parameters
    ----------
    metrics
        Iterable of names from
        ``{'TP','TN','FP','FN','dice','jaccard','precision','recall','fpr','fnr',
        'vs','volsim','mcc'}``.
        Default: all of them.
    label
        Score only this label id on both sides, rather than every non-zero
        voxel. See :func:`multilabel_metrics` to sweep every label at once.
    """
    requested = tuple(_METRICS.keys()) if metrics is None else tuple(m for m in metrics)
    unknown = set(requested) - set(_METRICS.keys())
    if unknown:
        raise ValueError(
            f"Unknown voxel metrics: {unknown}. Supported: {set(_METRICS.keys())}"
        )
    counts = confusion_counts(label_true, label_pred, label=label)
    return {m: float(_FROM_COUNTS[m](counts)) for m in requested}


def multilabel_metrics(
    label_true: Any,
    label_pred: Any,
    *,
    labels: Iterable[int] | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Score two label maps label by label instead of collapsing them to foreground.

    Returns ``{'labels', 'per_label', 'macro', 'foreground'}``:

    ``per_label``
        ``{label_id: {metric: value}}``, each label scored one-vs-rest against
        *the same label id* in the other map.
    ``macro``
        The unweighted mean of each metric across labels — every structure counts
        the same regardless of size, which is usually what you want when a few
        large labels would otherwise drown out the small ones.
    ``foreground``
        The any-label-vs-any-label numbers. Kept because they answer a real
        question ("did it find the right voxels at all?"), and reported under a
        name that makes clear they say nothing about label agreement.

    *labels* defaults to every non-zero id in either map.
    """
    requested = tuple(_FROM_COUNTS.keys()) if metrics is None else tuple(metrics)
    unknown = set(requested) - set(_FROM_COUNTS.keys())
    if unknown:
        raise ValueError(
            f"Unknown voxel metrics: {unknown}. Supported: {set(_FROM_COUNTS.keys())}"
        )
    ids = (
        [int(v) for v in labels]
        if labels is not None
        else label_ids_present(label_true, label_pred)
    )
    per_label = {
        lid: voxel_metrics(label_true, label_pred, metrics=requested, label=lid)
        for lid in ids
    }
    macro = {
        m: (sum(per_label[lid][m] for lid in ids) / len(ids) if ids else 0.0)
        for m in requested
    }
    foreground = voxel_metrics(label_true, label_pred, metrics=requested)
    return {
        "labels": ids,
        "per_label": per_label,
        "macro": macro,
        "foreground": foreground,
    }


__all__ = [
    "confusion_counts",
    "label_ids_present",
    "multilabel_metrics",
    "dice",
    "jaccard",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "mcc",
    "volsim",
    "volume_similarity",
    "voxel_metrics",
]
