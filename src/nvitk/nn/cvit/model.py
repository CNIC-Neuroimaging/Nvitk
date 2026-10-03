"""
CViT networks — segmentation (:class:`CViT`) and masked-image-modelling (:class:`CViTMIM`).

Description
-----------
``x → tokenizer → (token grid, conv skips) → transformer encoder → decoder → logits``

:class:`CViT` is constructible the way nnU-Net builds networks from a plans file
(``cls(input_channels=…, num_classes=…, deep_supervision=…, **arch_kwargs)`` followed by
``network.apply(network.initialize)``), so a plans file whose
``network_class_name`` is ``"nvitk.nn.cvit.CViT"`` and whose ``arch_kwargs`` is a serialised
:class:`~nvitk.nn.cvit.config.CViTConfig` works even with stock nnU-Net trainers.

:class:`CViTMIM` shares the tokenizer and encoder **with identical parameter names**, so its
``tokenizer.*`` / ``encoder.*`` weights load straight into a :class:`CViT` for fine-tuning.

Weight-transfer contract (nnssl ``AdaptationPlan``)
---------------------------------------------------
``key_to_encoder = "encoder"``, ``key_to_stem = "tokenizer"``, ``key_to_lpe = "encoder.pos_embed"``
and ``keys_to_in_proj`` = the tokenizer modules that consume the raw input.

Masked image modelling
----------------------
Masked voxels are zeroed **at the input** and masked tokens are replaced by ``mask_token``
(SimMIM) or dropped (MAE). Zeroing the input is what makes masking meaningful with a conv
tokenizer: otherwise the receptive field of a visible token would see straight into its masked
neighbour. The reconstruction head has no skips, forcing the signal through attention.
"""

from __future__ import annotations

import warnings
from math import prod
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import init_weights
from .config import CViTConfig
from .decoders import PatchDecoder, UNetDecoder
from .tokenizers import build_tokenizer
from .transformer import TransformerEncoder

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation network
# ──────────────────────────────────────────────────────────────────────────────


class CViT(nn.Module):
    """Convolutional Vision Transformer for dense segmentation.

    Parameters
    ----------
    input_channels, num_classes
        As in nnU-Net's network constructor contract.
    deep_supervision
        Return a list of logits (highest resolution first) instead of one tensor. Ignored by
        the patch decoder.
    **arch_kwargs
        Any :class:`CViTConfig` field. ``preset="CViTB"`` fills the transformer size first.

    Examples
    --------
    >>> net = CViT(1, 3, preset="CViTS", input_shape=(64, 64, 64), deep_supervision=False)
    >>> net(torch.zeros(1, 1, 64, 64, 64)).shape
    torch.Size([1, 3, 64, 64, 64])
    """

    key_to_encoder = "encoder"
    key_to_stem = "tokenizer"
    key_to_lpe = "encoder.pos_embed"

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        **arch_kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config_from_kwargs(
            arch_kwargs,
            input_channels=input_channels,
            num_classes=num_classes,
            deep_supervision=bool(deep_supervision) and arch_kwargs.get("decoder", "unet") == "unet",
        )
        self.config = cfg
        self.tokenizer = build_tokenizer(cfg)
        self.encoder = TransformerEncoder(cfg)
        if cfg.decoder == "unet":
            self.decoder: nn.Module = UNetDecoder(cfg)
        else:
            self.decoder = PatchDecoder(cfg, cfg.num_classes)
        self.keys_to_in_proj = tuple(f"tokenizer.{k}" for k in self.tokenizer.in_proj_keys)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        """Per-module initialiser for ``network.apply`` (see :func:`nvitk.nn.blocks.init_weights`)."""
        init_weights(module)

    @classmethod
    def from_config(cls, cfg: CViTConfig) -> "CViT":
        d = cfg.to_dict()
        return cls(d.pop("input_channels"), d.pop("num_classes"), d.pop("deep_supervision"), **d)

    # ---- skip / probe convenience ----------------------------------------------------------
    def set_skip_scale(self, scale: float) -> None:
        self.decoder.set_skip_scale(scale)

    def gate_penalty(self) -> torch.Tensor:
        return self.decoder.gate_penalty()

    # ---- forward ---------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        grid, skips = self.tokenizer(x)
        tokens = self.encoder(grid)
        return self.decoder(tokens, skips)


# ──────────────────────────────────────────────────────────────────────────────
# Masked image modelling
# ──────────────────────────────────────────────────────────────────────────────


class CViTMIM(nn.Module):
    """CViT tokenizer + encoder with a skip-free reconstruction head for self-supervised pretraining.

    Parameters
    ----------
    input_channels
        Channels to reconstruct (= network input channels).
    method
        ``"simmim"`` (mask tokens replaced, full sequence through the encoder) or ``"mae"``
        (masked tokens dropped; a mask token fills them back in before the decoder).
    **arch_kwargs
        :class:`CViTConfig` fields. ``decoder``/``skips``/``num_classes`` are forced.
    """

    key_to_encoder = "encoder"
    key_to_stem = "tokenizer"
    key_to_lpe = "encoder.pos_embed"

    def __init__(self, input_channels: int, method: str = "simmim", **arch_kwargs: Any) -> None:
        super().__init__()
        if method not in ("simmim", "mae"):
            raise ValueError(f"method must be 'simmim' or 'mae', got {method!r}.")
        arch_kwargs = {k: v for k, v in arch_kwargs.items() if k not in ("num_classes", "deep_supervision")}
        cfg = config_from_kwargs(
            arch_kwargs,
            input_channels=input_channels,
            num_classes=input_channels,
            decoder="patch",
            skips="none",
            deep_supervision=False,
        )
        if method == "mae" and cfg.tokenizer == "hierarchical":
            warnings.warn(
                "MAE with the hierarchical tokenizer: conv receptive fields still see across "
                "token borders (input zeroing limits, but does not remove, the leak). SimMIM is "
                "the recommended method for this tokenizer.",
                stacklevel=2,
            )
        self.config = cfg
        self.method = method
        self.tokenizer = build_tokenizer(cfg)
        self.encoder = TransformerEncoder(cfg)
        self.decoder = PatchDecoder(cfg, input_channels)
        self.keys_to_in_proj = tuple(f"tokenizer.{k}" for k in self.tokenizer.in_proj_keys)
        self.apply(CViT.initialize)

    # ---- masks -----------------------------------------------------------------------------
    @staticmethod
    def random_token_mask(
        batch: int,
        grid_shape: Sequence[int],
        ratio: float,
        *,
        device=None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Bool ``(B, *grid)`` mask with exactly ``round(ratio·N)`` masked tokens per sample.

        A fixed count per sample keeps MAE's kept-token tensor rectangular.
        """
        n = prod(int(g) for g in grid_shape)
        n_mask = int(round(float(ratio) * n))
        if not 0 < n_mask < n:
            raise ValueError(f"mask ratio {ratio} masks {n_mask} of {n} tokens; need 0 < masked < N.")
        noise = torch.rand(batch, n, device=device, generator=generator)
        idx = noise.argsort(dim=1)[:, :n_mask]
        mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return mask.reshape(batch, *grid_shape)

    def voxel_mask(self, token_mask: torch.Tensor) -> torch.Tensor:
        """Upsample a ``(B, *g)`` token mask to ``(B, 1, *S)`` voxels (nearest)."""
        m = token_mask[:, None].float()
        return F.interpolate(m, scale_factor=self.config.token_stride, mode="nearest").bool()

    # ---- forward ---------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct *x* from its visible part.

        Returns
        -------
        recon : Tensor
            ``(B, C, *S)`` reconstruction.
        vox_mask : Tensor
            ``(B, 1, *S)`` bool, ``True`` where the input was masked (where the loss applies).
        """
        vox_mask = self.voxel_mask(token_mask)
        grid, _ = self.tokenizer(x.masked_fill(vox_mask, 0.0))
        b, e, *g = grid.shape
        if self.method == "simmim":
            tokens = self.encoder(grid, token_mask=token_mask)
        else:
            flat = token_mask.reshape(b, -1)
            keep_idx = (~flat).nonzero()[:, 1].reshape(b, -1)              # equal count per sample
            kept = self.encoder(grid, keep_idx=keep_idx)                    # (B, K, E)
            full = self.encoder.mask_token.to(kept.dtype).expand(b, flat.shape[1], e).clone()
            full.scatter_(1, keep_idx[..., None].expand(-1, -1, e), kept)
            tokens = full.transpose(1, 2).reshape(b, e, *g)
        return self.decoder(tokens), vox_mask

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor, vox_mask: torch.Tensor) -> torch.Tensor:
        """Mean squared error over masked voxels only (all channels)."""
        m = vox_mask.to(recon.dtype)
        err = (recon.float() - target.float()).pow(2) * m
        return err.sum() / (m.sum() * recon.shape[1]).clamp_min(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Factories
# ──────────────────────────────────────────────────────────────────────────────


def config_from_kwargs(arch_kwargs: dict[str, Any], **forced: Any) -> CViTConfig:
    """Build a :class:`CViTConfig` from ``arch_kwargs`` (optionally naming a ``preset``) plus *forced* fields.

    ``strides`` is accepted as an alias of ``stem_strides``: nnU-Net reads a plan's
    ``pool_op_kernel_sizes`` (hence its deep-supervision scales) from ``arch_kwargs["strides"]``,
    so CViT plans carry it under that name. If both are given they must agree.
    """
    kwargs = dict(arch_kwargs)
    kwargs.update(forced)
    strides = kwargs.pop("strides", None)
    if strides is not None:
        given = kwargs.get("stem_strides")
        as_lists = [list(s) for s in strides]
        if given is not None and [list(s) for s in given] != as_lists:
            raise ValueError(f"'strides' {as_lists} disagrees with 'stem_strides' {given}.")
        kwargs["stem_strides"] = as_lists
    preset = kwargs.pop("preset", None)
    if preset:
        return CViTConfig.from_preset(preset, **kwargs)
    return CViTConfig(**kwargs)


def build_cvit(
    arch_kwargs: dict[str, Any],
    *,
    input_channels: int,
    num_classes: int,
    input_patch_size: Sequence[int] | None = None,
    deep_supervision: bool = True,
) -> CViT:
    """nnU-Net-facing factory: plans ``arch_kwargs`` (+ patch size) → initialised :class:`CViT`.

    ``input_patch_size`` overrides ``arch_kwargs["input_shape"]`` so the positional-embedding grid
    always matches the configuration being trained or predicted.
    """
    kwargs = dict(arch_kwargs)
    for k in ("input_channels", "num_classes", "deep_supervision"):
        kwargs.pop(k, None)
    if input_patch_size is not None:
        kwargs["input_shape"] = tuple(int(v) for v in input_patch_size)
    return CViT(input_channels, num_classes, deep_supervision, **kwargs)


__all__ = ["CViT", "CViTMIM", "build_cvit", "config_from_kwargs"]
