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
    c = confusion_counts(label_true, label_pred)
    denom = 2 * c["TP"] + c["FP"] + c["FN"]
    return (2 * c["TP"]) / denom if denom > 0 else 0.0


def jaccard(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    denom = c["TP"] + c["FP"] + c["FN"]
    return c["TP"] / denom if denom > 0 else 0.0


def precision(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    return c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else 0.0


def recall(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    return c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else 0.0


def fpr(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    return c["FP"] / (c["FP"] + c["TN"]) if (c["FP"] + c["TN"]) > 0 else 0.0


def fnr(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    return c["FN"] / (c["FN"] + c["TP"]) if (c["FN"] + c["TP"]) > 0 else 0.0


def volume_similarity(label_true: Any, label_pred: Any) -> float:
    c = confusion_counts(label_true, label_pred)
    n_true = c["TP"] + c["FN"]
    n_pred = c["TP"] + c["FP"]
    return abs(n_pred - n_true) / n_true if n_true > 0 else 0.0


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
        ``{'TP','TN','FP','FN','dice','jaccard','precision','recall','fpr','fnr','vs'}``.
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
    "volume_similarity",
    "voxel_metrics",
]
