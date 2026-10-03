"""
Selectable training objectives for nnU-Net / nnssl based learning pipelines.

Description
-----------
A torch-free registry of named losses shared by the topbrain and CViT pipelines, so a ``--loss``
flag can be validated, listed and documented without importing torch. A name resolves in three
ways, in order:

1. a **built-in registry name** — see :data:`SEGMENTATION_LOSSES` and :data:`SSL_LOSSES`;
2. a **dotted path** ``package.module:Callable`` for a user-supplied loss;
3. nothing — an unknown name is an error listing the valid ones, never a silent fallback.

:func:`build_segmentation_loss` and :func:`build_ssl_loss` import torch (and the in-tree
nnU-Net / nnssl) only when a loss is actually constructed, inside the training worker. The
segmentation losses return the bare objective; the trainer applies deep-supervision wrapping.

The loss implementations beyond nnU-Net's own live in :mod:`nvitk.segmentation.losses`.
"""

from __future__ import annotations

import json
import pydoc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nvitk.core.logger import Logger

log = Logger()


@dataclass(frozen=True)
class LossSpec:
    """Metadata for one selectable loss. Deliberately holds no torch objects."""

    name: str
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossContext:
    """Everything a segmentation loss needs from the trainer to configure itself.

    Mirrors what ``nnUNetTrainer._build_loss`` reads off ``self``, so a registry factory can be
    written without a live trainer and tested standalone.
    """

    batch_dice: bool = False
    has_regions: bool = False
    ignore_label: int | None = None
    is_ddp: bool = False
    num_classes: int = 2


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: Supervised segmentation objectives.
SEGMENTATION_LOSSES: dict[str, LossSpec] = {
    "dice_ce": LossSpec(
        "dice_ce", "nnU-Net default: soft Dice + cross-entropy, equally weighted."
    ),
    "dice_ce_nosmooth": LossSpec(
        "dice_ce_nosmooth", "Dice + CE with no Dice smoothing term."
    ),
    "dice": LossSpec("dice", "Soft Dice only."),
    "ce": LossSpec("ce", "Cross-entropy only."),
    "dice_topk10": LossSpec(
        "dice_topk10", "Soft Dice + top-10% hardest-voxel cross-entropy."
    ),
    "topk10": LossSpec("topk10", "Top-10% hardest-voxel cross-entropy only."),
    "focal": LossSpec(
        "focal",
        "Focal loss only — down-weights the easy background that dominates a 0.3% foreground.",
        {"gamma": 2.0},
    ),
    "dice_focal": LossSpec(
        "dice_focal", "Soft Dice + focal.", {"gamma": 2.0, "weight_dice": 1.0, "weight_focal": 1.0}
    ),
    "dice_ce_cldice": LossSpec(
        "dice_ce_cldice",
        "Dice + CE + soft centerline Dice; targets the clDice and connected-component metrics.",
        {"weight_cldice": 1.0, "iters": 3, "per_class": False},
    ),
    "dice_ce_skelrec": LossSpec(
        "dice_ce_skelrec",
        "Dice + CE + skeleton recall; weights thin side-road vessels by length, not calibre.",
        {"weight_skelrec": 1.0, "iters": 3, "per_class": False},
    ),
    "dice_ce_cldice_focal": LossSpec(
        "dice_ce_cldice_focal",
        "Dice + CE + soft clDice + focal — topology and imbalance together.",
        {"weight_cldice": 1.0, "weight_focal": 1.0, "gamma": 2.0, "iters": 3, "per_class": False},
    ),
}

#: Self-supervised (nnssl) reconstruction objectives.
SSL_LOSSES: dict[str, LossSpec] = {
    "mse": LossSpec("mse", "Masked-region MSE — the nnssl MAE default."),
    "mse_masked": LossSpec(
        "mse_masked", "MSE excluding de-faced regions via the anonymisation mask."
    ),
    "l1": LossSpec("l1", "Masked-region L1; less sensitive to bright-vessel outliers than MSE."),
    "ssim": LossSpec("ssim", "Structural similarity on the reconstructed patch."),
    "ms_ssim": LossSpec("ms_ssim", "Multi-scale structural similarity."),
    "spark": LossSpec("spark", "SparK sparse-encoder reconstruction loss."),
}

#: Maps a registry name to the nnssl loss class that implements it.
_SSL_LOSS_CLASSES: dict[str, str] = {
    "mse": "MAEMSELoss",
    "mse_masked": "LossMaskMSELoss",
    "l1": "MAEL1Loss",
    "ssim": "MAESSIMLoss",
    "ms_ssim": "MAE_MS_SSIMLoss",
    "spark": "SparkLoss",
}


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def is_dotted_path(name: str) -> bool:
    """Whether *name* looks like a ``module:Callable`` custom-loss reference."""
    return ":" in name


def resolve_dotted(name: str) -> Callable[..., Any]:
    """Import a ``package.module:Callable`` reference.

    Raises
    ------
    ValueError
        If the reference is malformed or does not import. The message names the reference, so a
        typo in ``--custom-loss`` is obvious rather than surfacing as a later ``NoneType`` call.
    """
    module_name, _, attribute = name.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"Custom loss {name!r} is malformed; expected 'package.module:Callable'."
        )
    target = pydoc.locate(f"{module_name}.{attribute}")
    if target is None:
        raise ValueError(
            f"Could not import custom loss {name!r}. Is {module_name!r} on the PYTHONPATH?"
        )
    if not callable(target):
        raise ValueError(f"Custom loss {name!r} resolved to a non-callable {type(target)!r}.")
    return target


def validate_loss_name(name: str, *, registry: dict[str, LossSpec]) -> str:
    """Return *name* if it is usable, else raise listing the valid options.

    Dotted paths are accepted without importing them — validation happens on the worker, which
    is where the user's module is actually on the path.
    """
    if is_dotted_path(name):
        return name
    if name in registry:
        return name
    raise ValueError(
        f"Unknown loss {name!r}. Valid names: {', '.join(sorted(registry))}, "
        f"or a custom 'package.module:Callable'."
    )


def loss_spec_payload(name: str, config: dict[str, Any] | None = None) -> str:
    """JSON payload for :data:`LOSS_SPEC_ENV` describing *name* and its kwargs."""
    return json.dumps({"loss": name, "config": dict(config or {})}, sort_keys=True)


def parse_loss_config(value: str | Path | None) -> dict[str, Any]:
    """Parse a ``--loss-config`` argument: inline JSON, a path to a JSON file, or ``None``."""
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    candidate = Path(text)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--loss-config is neither a readable file nor valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"--loss-config must be a JSON object; got {type(parsed).__name__}.")
    return parsed


# ---------------------------------------------------------------------------
# Construction (torch is imported here, not at module scope)
# ---------------------------------------------------------------------------


def build_segmentation_loss(
    name: str,
    ctx: LossContext,
    config: dict[str, Any] | None = None,
):
    """Build the segmentation loss *name*, **without** deep-supervision wrapping.

    The caller (the trainer) applies ``DeepSupervisionWrapper``, so a custom loss only has to
    implement ``forward(net_output, target)``.

    Parameters
    ----------
    ctx
        Trainer-derived settings — see :class:`LossContext`.
    config
        Overrides for the registry entry's defaults.

    Returns
    -------
    torch.nn.Module
    """
    from nnunetv2.training.loss.compound_losses import (
        DC_and_BCE_loss,
        DC_and_CE_loss,
        DC_and_topk_loss,
    )
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    from nvitk.segmentation.losses import (
        CompoundLoss,
        FocalLoss,
        SkeletonRecallLoss,
        SoftClDiceLoss,
    )

    if is_dotted_path(name):
        factory = resolve_dotted(name)
        log.info("Building custom loss %s with %s", name, config or {})
        return factory(**(config or {}))

    spec = SEGMENTATION_LOSSES.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown loss {name!r}. Valid: {', '.join(sorted(SEGMENTATION_LOSSES))}."
        )
    options = dict(spec.defaults) | dict(config or {})

    dice_kwargs = {
        "batch_dice": ctx.batch_dice,
        "smooth": float(options.pop("smooth", 1e-5)),
        "do_bg": False,
        "ddp": ctx.is_ddp,
    }

    def _dice_ce() -> Any:
        """nnU-Net's own objective, region-aware exactly as ``_build_loss`` builds it."""
        if ctx.has_regions:
            return DC_and_BCE_loss(
                {},
                {"batch_dice": ctx.batch_dice, "do_bg": True, "smooth": dice_kwargs["smooth"],
                 "ddp": ctx.is_ddp},
                use_ignore_label=ctx.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        return DC_and_CE_loss(
            dice_kwargs,
            {},
            weight_ce=float(options.get("weight_ce", 1.0)),
            weight_dice=float(options.get("weight_dice", 1.0)),
            ignore_label=ctx.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    def _skel_kwargs() -> dict[str, Any]:
        """Shared constructor kwargs for the topology losses."""
        return {
            "iters": int(options.get("iters", 3)),
            "per_class": bool(options.get("per_class", False)),
        }

    if name == "dice_ce":
        return _dice_ce()
    if name == "dice_ce_nosmooth":
        dice_kwargs["smooth"] = 0.0
        return _dice_ce()
    if name == "dice":
        return MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **dice_kwargs)
    if name == "ce":
        return RobustCrossEntropyLoss()
    if name == "topk10":
        return TopKLoss(k=10)
    if name == "dice_topk10":
        return DC_and_topk_loss(
            dice_kwargs, {"k": 10}, weight_ce=1, weight_dice=1,
            ignore_label=ctx.ignore_label,
        )
    if name == "focal":
        return FocalLoss(gamma=float(options.get("gamma", 2.0)), ignore_index=ctx.ignore_label)
    if name == "dice_focal":
        return CompoundLoss([
            (MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **dice_kwargs),
             float(options.get("weight_dice", 1.0))),
            (FocalLoss(gamma=float(options.get("gamma", 2.0)), ignore_index=ctx.ignore_label),
             float(options.get("weight_focal", 1.0))),
        ])
    if name == "dice_ce_cldice":
        return CompoundLoss([
            (_dice_ce(), 1.0),
            (SoftClDiceLoss(**_skel_kwargs()), float(options.get("weight_cldice", 1.0))),
        ])
    if name == "dice_ce_skelrec":
        return CompoundLoss([
            (_dice_ce(), 1.0),
            (SkeletonRecallLoss(**_skel_kwargs()), float(options.get("weight_skelrec", 1.0))),
        ])
    if name == "dice_ce_cldice_focal":
        return CompoundLoss([
            (_dice_ce(), 1.0),
            (SoftClDiceLoss(**_skel_kwargs()), float(options.get("weight_cldice", 1.0))),
            (FocalLoss(gamma=float(options.get("gamma", 2.0)), ignore_index=ctx.ignore_label),
             float(options.get("weight_focal", 1.0))),
        ])

    raise ValueError(f"Loss {name!r} is registered but has no factory; this is a bug.")


def build_ssl_loss(name: str, config: dict[str, Any] | None = None):
    """Build a self-supervised reconstruction loss for an nnssl trainer.

    Requires the vendored nnssl clone to be importable — call
    :func:`~nvitk.pipes._engines.env.apply_nnssl_env` first.
    """
    if is_dotted_path(name):
        return resolve_dotted(name)(**(config or {}))

    class_name = _SSL_LOSS_CLASSES.get(name)
    if class_name is None:
        raise ValueError(f"Unknown SSL loss {name!r}. Valid: {', '.join(sorted(SSL_LOSSES))}.")

    import importlib

    for module_name in (
        "nnssl.training.loss.mse_loss",
        "nnssl.training.loss.spark_loss",
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            return getattr(module, class_name)(**(config or {}))
    raise ValueError(f"nnssl loss class {class_name!r} not found in the vendored clone.")


__all__ = [
    "SEGMENTATION_LOSSES",
    "SSL_LOSSES",
    "LossContext",
    "LossSpec",
    "build_segmentation_loss",
    "build_ssl_loss",
    "is_dotted_path",
    "loss_spec_payload",
    "parse_loss_config",
    "resolve_dotted",
    "validate_loss_name",
]
