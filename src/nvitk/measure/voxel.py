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

from ._common import bool_mask, ensure_same_shape

setup(globals())


def confusion_counts(label_true: Any, label_pred: Any) -> dict[str, int]:
    """
    Return ``{'TP','TN','FP','FN'}`` counts from two binary masks.
    """
    ensure_same_shape(label_true, label_pred)
    a = bool_mask(label_true).ravel()
    b = bool_mask(label_pred).ravel()
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
    """
    requested = tuple(_METRICS.keys()) if metrics is None else tuple(m for m in metrics)
    unknown = set(requested) - set(_METRICS.keys())
    if unknown:
        raise ValueError(
            f"Unknown voxel metrics: {unknown}. Supported: {set(_METRICS.keys())}"
        )
    counts = confusion_counts(label_true, label_pred)
    out: dict[str, float] = {}
    for m in requested:
        if m in counts:
            out[m] = float(counts[m])
        else:
            out[m] = float(_METRICS[m](label_true, label_pred))
    return out


__all__ = [
    "confusion_counts",
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
