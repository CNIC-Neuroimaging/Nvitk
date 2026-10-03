"""
CViT weight transfer — export / load pretrained encoders and build fine-tuning parameter groups.

Description
-----------
Pretraining (:class:`~nvitk.nn.cvit.model.CViTMIM`) and fine-tuning
(:class:`~nvitk.nn.cvit.model.CViT`) share the ``tokenizer.*`` and ``encoder.*`` parameter names,
so transfer is a filtered ``load_state_dict`` with three adaptations:

Positional embedding
    A different patch size changes the token grid; ``encoder.pos_embed`` is resampled with
    (bi/tri)linear interpolation (the same treatment nnU-Net's ``handle_pos_embed_resize`` gives
    Primus).
Input channels
    A model pretrained on 1 channel can seed a 2-channel one (and vice versa). Every conv
    that consumes the raw input (``model.keys_to_in_proj``) has its input-channel axis adapted:
    ``"repeat"`` tiles the kernels and divides by the expansion factor so the response to a
    channel-replicated input is unchanged; ``"mean"`` averages them into each new channel.
Checkpoint formats
    Plain state dicts, nnU-Net / nnssl checkpoints (``network_weights``), and ``module.`` /
    ``_orig_mod.`` prefixes from DDP / ``torch.compile`` are all accepted.

Layer-wise learning-rate decay
------------------------------
:func:`layerwise_lr_groups` implements BEiT-style decay for fine-tuning: the decoder trains at the
base rate, transformer block ``i`` at ``base · decay^(depth − i)``, and the tokenizer /
embeddings at ``base · decay^(depth + 1)``. Norms, biases, embeddings and gates get no weight
decay.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .transformer import resize_pos_embed

#: Parameter-name prefixes that make up the transferable encoder.
ENCODER_PREFIXES: tuple[str, ...] = ("tokenizer.", "encoder.")
_STRIP_PREFIXES: tuple[str, ...] = ("module.", "_orig_mod.")

# ──────────────────────────────────────────────────────────────────────────────
# State-dict helpers
# ──────────────────────────────────────────────────────────────────────────────


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP (``.module``) and ``torch.compile`` (``._orig_mod``) wrappers."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        else:
            return model


def _clean_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _STRIP_PREFIXES:
            if key.startswith(p):
                key, changed = key[len(p):], True
    return key


def read_state_dict(source: Mapping[str, Any] | str | Path) -> dict[str, torch.Tensor]:
    """Tensor state dict from a path or mapping, unwrapping checkpoint containers and prefixes."""
    if isinstance(source, (str, Path)):
        source = torch.load(str(source), map_location="cpu", weights_only=False)
    for key in ("network_weights", "state_dict", "model"):
        if isinstance(source, Mapping) and key in source and isinstance(source[key], Mapping):
            source = source[key]
            break
    return {_clean_key(k): v for k, v in source.items() if isinstance(v, torch.Tensor)}


def encoder_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The transferable ``tokenizer.*`` + ``encoder.*`` tensors of a CViT / CViTMIM (on CPU)."""
    sd = _unwrap(model).state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if k.startswith(ENCODER_PREFIXES)}


def _adapt_in_channels(w: torch.Tensor, c_new: int, how: str) -> torch.Tensor:
    """Adapt the input-channel axis (dim 1) of a conv kernel from ``w.shape[1]`` to ``c_new``."""
    c_old = w.shape[1]
    if how == "repeat":
        reps = ceil(c_new / c_old)
        return w.repeat(1, reps, *([1] * (w.ndim - 2)))[:, :c_new] * (c_old / c_new)
    if how == "mean":
        return w.mean(dim=1, keepdim=True).expand(-1, c_new, *w.shape[2:]).clone() * (c_old / c_new)
    raise ValueError(f"adapt_in_channels must be 'repeat', 'mean' or 'skip', got {how!r}.")


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_pretrained_encoder(
    model: nn.Module,
    source: Mapping[str, Any] | str | Path,
    *,
    resize_pos_embed_ok: bool = True,
    adapt_in_channels: str = "repeat",
    strict: bool = True,
) -> dict[str, list[str]]:
    """Load the encoder part of a pretrained checkpoint into *model*.

    Parameters
    ----------
    model
        :class:`~nvitk.nn.cvit.model.CViT` (possibly wrapped by DDP / ``torch.compile``).
    source
        Checkpoint path or state dict (from :class:`CViTMIM` pretraining or another CViT).
    resize_pos_embed_ok
        Resample ``encoder.pos_embed`` to the model's grid when sizes differ.
    adapt_in_channels
        ``"repeat"`` / ``"mean"`` adapt input-consuming convs to a different channel count;
        ``"skip"`` leaves those layers at their fresh initialisation.
    strict
        Raise if any other encoder tensor has a shape that cannot be reconciled, or if the
        checkpoint holds no encoder tensors at all — a silent partial load is how a
        "pretrained" run quietly trains from scratch.

    Returns
    -------
    dict
        ``loaded``, ``resized``, ``adapted``, ``skipped`` (shape mismatch, non-strict) and
        ``missing`` (encoder keys absent from the checkpoint) parameter names.
    """
    net = _unwrap(model)
    src = read_state_dict(source)
    own = net.state_dict()
    in_proj = tuple(getattr(net, "keys_to_in_proj", ()))
    report: dict[str, list[str]] = {k: [] for k in ("loaded", "resized", "adapted", "skipped", "missing")}
    new: dict[str, torch.Tensor] = {}

    for key, target in own.items():
        if not key.startswith(ENCODER_PREFIXES):
            continue
        if key not in src:
            report["missing"].append(key)
            continue
        value = src[key]
        if value.shape == target.shape:
            new[key] = value
            report["loaded"].append(key)
        elif key.endswith("encoder.pos_embed") and resize_pos_embed_ok and value.ndim == target.ndim:
            new[key] = resize_pos_embed(value.float(), target.shape[2:]).to(target.dtype)
            report["resized"].append(key)
        elif (
            any(key.startswith(p + ".") for p in in_proj)
            and key.endswith(".weight")
            and value.ndim == target.ndim
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and adapt_in_channels != "skip"
        ):
            new[key] = _adapt_in_channels(value, target.shape[1], adapt_in_channels).to(target.dtype)
            report["adapted"].append(key)
        elif any(key.startswith(p + ".") for p in in_proj) and adapt_in_channels == "skip":
            report["skipped"].append(key)
        else:
            if strict:
                raise ValueError(
                    f"Pretrained tensor {key!r} has shape {tuple(value.shape)}, model expects "
                    f"{tuple(target.shape)}. Architectures differ beyond patch size / input channels."
                )
            report["skipped"].append(key)

    if strict and not new:
        raise ValueError("The checkpoint contains no tokenizer./encoder. tensors to load.")
    net.load_state_dict(new, strict=False)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning parameter groups
# ──────────────────────────────────────────────────────────────────────────────


def _layer_id(name: str, depth: int) -> int:
    """0 = tokenizer / embeddings, 1…depth = transformer blocks, depth+1 = everything after."""
    if name.startswith("tokenizer.") or name in (
        "encoder.pos_embed", "encoder.registers", "encoder.mask_token"
    ):
        return 0
    if name.startswith("encoder.blocks."):
        return int(name.split(".")[2]) + 1
    return depth + 1


def _no_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or name.endswith(("pos_embed", "registers", "mask_token", "gate_logits"))


def layerwise_lr_groups(
    model: nn.Module, base_lr: float, *, decay: float = 0.75, weight_decay: float = 0.05
) -> list[dict[str, Any]]:
    """Optimizer parameter groups with layer-wise LR decay (BEiT / MAE fine-tuning recipe).

    Each group carries an ``lr_scale`` key so schedulers that rewrite ``lr`` every step can
    re-apply it (``group["lr"] = scheduled_lr * group["lr_scale"]``).
    """
    net = _unwrap(model)
    depth = len(net.encoder.blocks)
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name, depth)
        nd = _no_decay(name, p)
        g = groups.setdefault((lid, nd), {
            "params": [],
            "lr_scale": decay ** (depth + 1 - lid),
            "weight_decay": 0.0 if nd else weight_decay,
            "layer_id": lid,
        })
        g["params"].append(p)
    out = sorted(groups.values(), key=lambda g: (g["layer_id"], g["weight_decay"]))
    for g in out:
        g["lr"] = base_lr * g["lr_scale"]
    return out


__all__ = [
    "ENCODER_PREFIXES",
    "encoder_state_dict",
    "layerwise_lr_groups",
    "load_pretrained_encoder",
    "read_state_dict",
]
