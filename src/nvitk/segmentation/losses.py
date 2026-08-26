"""
Differentiable segmentation objectives for thin, tubular structures.

Description
-----------
Overlap losses (Dice, cross-entropy) treat every foreground voxel alike. For vessel trees that
is a poor match to how the result is judged: a segmentation can score well on Dice while being
broken into disconnected fragments, because a one-voxel gap costs almost no overlap but
destroys the topology. Challenges in this area score connectivity explicitly — centerline Dice,
connected-component error — so the training objective should too.

This module provides the topology-aware terms, plus a focal loss for the extreme foreground
imbalance typical of angiography (vessels occupy well under 1 % of a head volume):

- :func:`soft_skeletonize` — differentiable morphological skeleton (iterative soft opening)
- :class:`SoftClDiceLoss` — centerline Dice, the differentiable analogue of the clDice metric
- :class:`SkeletonRecallLoss` — recall measured on the reference skeleton only
- :class:`FocalLoss` — down-weights the easy background that dominates the gradient
- :class:`CompoundLoss` — weighted sum of any of the above with a base loss

Torch dependency
----------------
This module imports :mod:`torch` at module scope and is therefore **deliberately excluded**
from ``nvitk.segmentation.__init__``: importing the package must not require a deep-learning
stack. Import it explicitly::

    from nvitk.segmentation.losses import SoftClDiceLoss

Conventions
-----------
Every loss follows the nnU-Net contract: ``forward(net_output, target)`` where *net_output* is
raw logits of shape ``(B, C, *spatial)`` and *target* holds integer class indices of shape
``(B, 1, *spatial)``. Losses return a scalar. Deep supervision is applied by the caller
(``DeepSupervisionWrapper``), never here.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from nvitk.core.logger import Logger

log = Logger()

#: Default number of soft-skeletonisation iterations. Each iteration erodes by one voxel, so
#: this bounds the half-thickness of a structure that can be fully skeletonised. Intracranial
#: vessels at 0.3-0.6 mm spacing are a few voxels across, so a small value suffices and each
#: extra iteration costs three more pooling passes over the patch.
DEFAULT_SKEL_ITERS: int = 3

#: Foreground classes skeletonised at once when ``per_class`` is enabled. Peak memory scales
#: with this rather than with the class count, which is what makes per-class topology losses
#: usable on a 16 GB card; the loss value itself is unaffected.
DEFAULT_CLASS_CHUNK: int = 4

#: Numerical floor for ratio denominators.
_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Soft morphology
# ---------------------------------------------------------------------------


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Grayscale erosion by a 3D cross, as the min of three axis-aligned min-pools.

    A separable cross rather than a full 3x3x3 cube: it is three pooling calls instead of one
    but erodes more gently, which matters for structures only a couple of voxels thick.
    """
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Grayscale dilation by a 3x3x3 cube."""
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """Grayscale opening: erosion followed by dilation."""
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = DEFAULT_SKEL_ITERS) -> torch.Tensor:
    """Differentiable morphological skeleton of a soft mask.

    Implements the iterative soft-skeleton of the clDice formulation: repeatedly erode, and
    accumulate whatever the opening removes — those are the medial voxels that would
    disappear, i.e. the centerline.

    Parameters
    ----------
    img
        Soft mask in ``[0, 1]``, shape ``(B, C, D, H, W)``. Probabilities, not logits.
    iters
        Erosion iterations. Cost is linear in this and each step is four pooling passes.

    Returns
    -------
    torch.Tensor
        Soft skeleton in ``[0, 1]``, same shape as *img*.
    """
    if img.ndim != 5:
        raise ValueError(f"soft_skeletonize expects (B, C, D, H, W); got shape {tuple(img.shape)}.")
    opened = _soft_open(img)
    skel = F.relu(img - opened)
    for _ in range(int(iters)):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        # Union rather than sum, so repeatedly-detected voxels do not accumulate above 1.
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ---------------------------------------------------------------------------
# Helpers shared by the topology losses
# ---------------------------------------------------------------------------


def _foreground_pairs(
    net_output: torch.Tensor,
    target: torch.Tensor,
    *,
    per_class: bool,
    class_chunk: int,
):
    """Yield ``(pred, true)`` tensor pairs of soft foreground probability and reference mask.

    Never materialises a full one-hot encoding of *target*. At 37 classes and a 128³ patch that
    tensor alone is ~600 MB, which is what made a naive per-class implementation unusable on a
    16 GB card; the reference masks are built by comparison (``target == c``) instead, one
    chunk at a time.

    With *per_class* false a single pair is yielded for the merged vessel tree — much cheaper,
    and still targeting the connectivity that the topology metrics measure. With *per_class*
    true the foreground classes are yielded in groups of *class_chunk*, so peak memory is set
    by the chunk size rather than by the class count.
    """
    probs = torch.softmax(net_output, dim=1)
    num_classes = net_output.shape[1]

    target_long = target.long()
    if target_long.shape[1] == 1:
        target_long = target_long[:, 0]
    target_long = target_long.unsqueeze(1)

    if not per_class:
        # "Any foreground" = 1 - P(background); reference is any non-zero label.
        fg_prob = (1.0 - probs[:, :1]).clamp(0.0, 1.0)
        with torch.no_grad():
            fg_true = (target_long > 0).float()
        yield fg_prob, fg_true
        return

    step = max(1, int(class_chunk))
    for start in range(1, num_classes, step):
        stop = min(start + step, num_classes)
        pred = probs[:, start:stop]
        with torch.no_grad():
            classes = torch.arange(start, stop, device=target_long.device).view(1, -1, *([1] * (target_long.ndim - 2)))
            true = (target_long == classes).float()
        yield pred, true


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor, smooth: float) -> torch.Tensor:
    """``(num + smooth) / (den + smooth)`` — 1.0 when both are empty, never NaN."""
    return (numerator + smooth) / (denominator + smooth)


def _skeletonize_pred(pred: torch.Tensor, iters: int, use_checkpoint: bool) -> torch.Tensor:
    """Soft-skeletonise a *differentiable* prediction, optionally under gradient checkpointing.

    Skeletonising the prediction means back-propagating through ~4 pooling ops per iteration,
    each of which stores an activation the size of the input. At 37 classes and a 128³ patch
    that is what exhausts a 16 GB card. Checkpointing recomputes the skeleton during backward
    instead of storing it: roughly 2x the time for a large constant-factor drop in memory.
    """
    if not use_checkpoint or not pred.requires_grad:
        return soft_skeletonize(pred, iters)
    return checkpoint(soft_skeletonize, pred, iters, use_reentrant=False)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class SoftClDiceLoss(nn.Module):
    """Soft centerline Dice — the differentiable analogue of the clDice metric.

    Combines two ratios: how much of the *predicted* skeleton falls inside the reference mask
    (topology precision), and how much of the *reference* skeleton falls inside the prediction
    (topology sensitivity). Their harmonic mean is high only when the prediction reproduces the
    reference's connectivity, so a fragmented segmentation is penalised even when its overlap
    is good.

    Parameters
    ----------
    iters
        Soft-skeletonisation iterations.
    per_class
        Skeletonise each foreground class separately instead of the merged vessel tree.
        Costly — see :func:`_foreground_probabilities`.
    smooth
        Added to numerator and denominator so an empty patch yields 0 rather than NaN. The
        0.2-0.5 % foreground rate of angiographic data makes empty patches common.
    class_chunk
        Foreground classes skeletonised per group when *per_class* is set.
    checkpoint_skeleton
        Recompute the predicted skeleton during backward instead of storing it. Defaults to
        *per_class*: without it, per-class clDice does not fit a 16 GB card at a 128³ patch.
    """

    def __init__(
        self,
        *,
        iters: int = DEFAULT_SKEL_ITERS,
        per_class: bool = False,
        smooth: float = 1.0,
        class_chunk: int = DEFAULT_CLASS_CHUNK,
        checkpoint_skeleton: bool | None = None,
    ) -> None:
        super().__init__()
        self.iters = int(iters)
        self.per_class = bool(per_class)
        self.smooth = float(smooth)
        self.class_chunk = int(class_chunk)
        # Only the per-class path is memory-constrained; the merged path skeletonises a single
        # channel, so paying the recompute there would be pure overhead.
        self.checkpoint_skeleton = (
            bool(per_class) if checkpoint_skeleton is None else bool(checkpoint_skeleton)
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar soft-clDice loss in ``[0, 1]`` (0 = perfect topology agreement)."""
        scores = []
        for pred, true in _foreground_pairs(
            net_output, target, per_class=self.per_class, class_chunk=self.class_chunk
        ):
            skel_pred = _skeletonize_pred(pred, self.iters, self.checkpoint_skeleton)
            with torch.no_grad():
                skel_true = soft_skeletonize(true, self.iters)

            dims = tuple(range(2, pred.ndim))
            precision = _ratio((skel_pred * true).sum(dims), skel_pred.sum(dims), self.smooth)
            sensitivity = _ratio((skel_true * pred).sum(dims), skel_true.sum(dims), self.smooth)
            scores.append(2.0 * precision * sensitivity / (precision + sensitivity + _EPS))

        return 1.0 - torch.cat(scores, dim=1).mean()


class SkeletonRecallLoss(nn.Module):
    """Recall measured only on the reference skeleton.

    Asks one question: of the voxels on the reference centerline, how many did the model
    predict as foreground? Unlike Dice it cannot be satisfied by getting thick vessels right —
    every branch contributes centerline voxels roughly in proportion to its *length*, not its
    calibre, so thin side branches carry as much weight as the large trunks. That matches a
    scoring scheme whose detection metric is dominated by small vessels.

    .. note::
       This computes the reference skeleton on the fly with :func:`soft_skeletonize`. The
       published Skeleton Recall method precomputes tubed skeletons in the dataloader, which is
       cheaper per step and gives a slightly thicker target region. This is an approximation of
       that idea, not a reimplementation of it.
    """

    def __init__(
        self,
        *,
        iters: int = DEFAULT_SKEL_ITERS,
        per_class: bool = False,
        smooth: float = 1.0,
        class_chunk: int = DEFAULT_CLASS_CHUNK,
    ) -> None:
        super().__init__()
        self.iters = int(iters)
        self.per_class = bool(per_class)
        self.smooth = float(smooth)
        self.class_chunk = int(class_chunk)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar skeleton-recall loss in ``[0, 1]`` (0 = the whole centerline was recovered)."""
        recalls = []
        for pred, true in _foreground_pairs(
            net_output, target, per_class=self.per_class, class_chunk=self.class_chunk
        ):
            with torch.no_grad():
                skel_true = soft_skeletonize(true, self.iters)
            dims = tuple(range(2, pred.ndim))
            recalls.append(_ratio((skel_true * pred).sum(dims), skel_true.sum(dims), self.smooth))
        return 1.0 - torch.cat(recalls, dim=1).mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss — cross-entropy with easy examples down-weighted.

    At a 0.2-0.5 % foreground rate, background voxels the model already classifies confidently
    still supply most of the gradient. The ``(1 - p_t) ** gamma`` factor suppresses them so the
    remaining signal comes from the vessels and the boundary.

    Parameters
    ----------
    gamma
        Focusing strength. 0 reduces to plain cross-entropy; 2 is the usual choice.
    alpha
        Optional per-class weights, length ``C``. Registered as a buffer so it follows the
        module across devices.
    ignore_index
        Label value excluded from the loss, or ``None``.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Sequence[float] | None = None,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        weight = torch.as_tensor(list(alpha), dtype=torch.float32) if alpha is not None else None
        self.register_buffer("alpha", weight, persistent=False)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar focal loss."""
        target_long = target.long()
        if target_long.shape[1] == 1:
            target_long = target_long[:, 0]
        ce = F.cross_entropy(
            net_output,
            target_long,
            weight=self.alpha,
            ignore_index=-100 if self.ignore_index is None else int(self.ignore_index),
            reduction="none",
        )
        # p_t = exp(-CE) holds exactly for the true class under a softmax.
        focal = (1.0 - torch.exp(-ce)) ** self.gamma * ce
        if self.ignore_index is not None:
            valid = target_long != int(self.ignore_index)
            return focal[valid].mean() if valid.any() else focal.sum() * 0.0
        return focal.mean()


class CompoundLoss(nn.Module):
    """Weighted sum of several losses sharing the nnU-Net ``(net_output, target)`` contract.

    Lets a base objective be combined with topology terms without a bespoke class per
    combination::

        CompoundLoss([(dice_ce, 1.0), (SoftClDiceLoss(), 0.5)])

    Components are held in a :class:`~torch.nn.ModuleList` so their buffers and any parameters
    move with the parent module.
    """

    def __init__(self, components: Sequence[tuple[nn.Module, float]]) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompoundLoss needs at least one component.")
        self.components = nn.ModuleList([module for module, _ in components])
        self.weights = tuple(float(weight) for _, weight in components)
        if all(weight == 0.0 for weight in self.weights):
            raise ValueError("CompoundLoss needs at least one non-zero weight.")

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted sum of the component losses."""
        total = None
        for module, weight in zip(self.components, self.weights):
            if weight == 0.0:
                continue
            value = weight * module(net_output, target)
            total = value if total is None else total + value
        return total

    def extra_repr(self) -> str:
        """Show the component weights in ``print(loss)``."""
        return f"weights={self.weights}"


__all__ = [
    "DEFAULT_CLASS_CHUNK",
    "DEFAULT_SKEL_ITERS",
    "CompoundLoss",
    "FocalLoss",
    "SkeletonRecallLoss",
    "SoftClDiceLoss",
    "soft_skeletonize",
]
