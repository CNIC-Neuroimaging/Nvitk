"""
CViT tokenizers — turn a volume into a token grid plus multi-scale skip features.

Description
-----------
All three tokenizers share one contract::

    grid, skips = tokenizer(x)
    # grid  : (B, E, *g)      token grid, g = input_shape // token_stride
    # skips : list[Tensor | None], one per level, (B, C_i, *input_shape // level_stride_i)

``skips[i]`` is ``None`` when the level is not needed (skip disabled and the tokenizer does not
need it for the tokens either), so the decoder never pays for features it ignores.

Modes
-----
``hierarchical``  (:class:`HierarchicalConvTokenizer`)
    Residual conv stages over the whole volume; a 1^n projection of the deepest stage gives
    the tokens. Receptive fields cross patch borders ("early convolutions" ViT).
``intra_patch``  (:class:`IntraPatchConvTokenizer`)
    The volume is cut into non-overlapping token patches; the same weight-shared CNN runs on
    each patch *independently* (zero padding at the patch border, :class:`LayerNormNd` instead
    of InstanceNorm) and its stride schedule ends at one voxel = one token. Intermediate maps
    are folded back into full-volume skips — exact, because patches do not overlap. No
    information crosses patches before attention.
``linear``  (:class:`LinearPatchTokenizer`)
    The ViT/Primus baseline: one ``Conv(k=s=token_stride)`` projection of raw voxels. When the
    decoder wants skips, a separate hierarchical stem produces them, so this ablation differs
    from ``hierarchical`` only in *where the tokens come from*.
"""

from __future__ import annotations

from math import prod

import torch
from torch import nn

from ..blocks import ResStage, conv_nd, make_norm
from .config import CViTConfig

# ──────────────────────────────────────────────────────────────────────────────
# Shared conv stem
# ──────────────────────────────────────────────────────────────────────────────


class _ConvStem(nn.Module):
    """``n_levels`` residual stages; returns every level's feature map up to ``last_level``."""

    def __init__(self, cfg: CViTConfig, *, norm: str, n_levels: int | None = None) -> None:
        super().__init__()
        n = cfg.n_levels if n_levels is None else n_levels
        chans = (cfg.input_channels,) + cfg.stem_channels
        self.stages = nn.ModuleList(
            ResStage(chans[i], chans[i + 1], cfg.ndim,
                     n_blocks=cfg.stem_blocks[i], stride=cfg.stem_strides[i], norm=norm)
            for i in range(n)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


def _check_input(x: torch.Tensor, cfg: CViTConfig) -> None:
    """Fail loudly on a spatial size the token stride cannot tile (instead of a shape error later)."""
    spatial = tuple(x.shape[2:])
    if len(spatial) != cfg.ndim:
        raise ValueError(f"Expected a {cfg.ndim}D input (B, C, *spatial), got shape {tuple(x.shape)}.")
    bad = [s for s, t in zip(spatial, cfg.token_stride) if s % t]
    if bad:
        raise ValueError(
            f"Input spatial size {spatial} is not divisible by the token stride {cfg.token_stride}."
        )
    if x.shape[1] != cfg.input_channels:
        raise ValueError(f"Expected {cfg.input_channels} input channel(s), got {x.shape[1]}.")


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizers
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalConvTokenizer(nn.Module):
    """Strided residual conv stem over the whole volume → 1^n projection → tokens."""

    mode = "hierarchical"

    def __init__(self, cfg: CViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = _ConvStem(cfg, norm="instance")
        self.proj = conv_nd(cfg.ndim)(cfg.stem_channels[-1], cfg.embed_dim, 1)
        self.norm = make_norm("layer", cfg.embed_dim, cfg.ndim)

    @property
    def in_proj_keys(self) -> tuple[str, ...]:
        """Modules consuming the raw input (for in-channel adaptation of pretrained weights)."""
        return ("stem.stages.0.blocks.0.conv1.conv", "stem.stages.0.blocks.0.shortcut.conv")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        _check_input(x, self.cfg)
        feats = self.stem(x)
        grid = self.norm(self.proj(feats[-1]))
        skips = [f if keep else None for f, keep in zip(feats, self.cfg.skips)]
        return grid, skips


class IntraPatchConvTokenizer(nn.Module):
    """Weight-shared CNN applied to each non-overlapping token patch independently."""

    mode = "intra_patch"

    def __init__(self, cfg: CViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # LayerNorm: InstanceNorm is degenerate on the 1-voxel deepest map, and LayerNorm uses
        # no spatial statistics, so isolation between patches is preserved exactly.
        self.stem = _ConvStem(cfg, norm="layer")
        self.proj = conv_nd(cfg.ndim)(cfg.stem_channels[-1], cfg.embed_dim, 1)
        self.norm = make_norm("layer", cfg.embed_dim, cfg.ndim)

    @property
    def in_proj_keys(self) -> tuple[str, ...]:
        return ("stem.stages.0.blocks.0.conv1.conv", "stem.stages.0.blocks.0.shortcut.conv")

    # ---- patch <-> volume reshapes -----------------------------------------------------------
    @staticmethod
    def _unfold(x: torch.Tensor, patch: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
        """``(B, C, *S)`` → ``(B·N, C, *patch)`` with patches in row-major grid order."""
        b, c, *spatial = x.shape
        g = tuple(s // p for s, p in zip(spatial, patch))
        nd = len(patch)
        shape = [b, c]
        for gi, pi in zip(g, patch):
            shape += [gi, pi]
        x = x.reshape(shape)
        # (B, C, g0, p0, g1, p1, ...) → (B, g0, g1, ..., C, p0, p1, ...)
        perm = [0] + [2 + 2 * i for i in range(nd)] + [1] + [3 + 2 * i for i in range(nd)]
        x = x.permute(perm).reshape(b * prod(g), c, *patch)
        return x, g

    @staticmethod
    def _fold(x: torch.Tensor, batch: int, g: tuple[int, ...]) -> torch.Tensor:
        """Inverse of :meth:`_unfold` for a feature map of any per-patch size."""
        _, c, *q = x.shape
        nd = len(g)
        x = x.reshape(batch, *g, c, *q)
        # (B, g0, g1, ..., C, q0, q1, ...) → (B, C, g0, q0, g1, q1, ...)
        perm = [0, 1 + nd]
        for i in range(nd):
            perm += [1 + i, 2 + nd + i]
        x = x.permute(perm)
        return x.reshape(batch, c, *(gi * qi for gi, qi in zip(g, q)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        _check_input(x, self.cfg)
        b = x.shape[0]
        patches, g = self._unfold(x, self.cfg.token_stride)
        feats = self.stem(patches)                       # each (B·N, C_i, *p / level_stride_i)
        tokens = self.norm(self.proj(feats[-1]))         # (B·N, E, 1, ..., 1)
        grid = self._fold(tokens, b, g)                  # (B, E, *g)
        skips = [self._fold(f, b, g) if keep else None for f, keep in zip(feats, self.cfg.skips)]
        return grid, skips


class LinearPatchTokenizer(nn.Module):
    """ViT baseline: ``Conv(k=s=token_stride)`` on raw voxels; optional stem for decoder skips."""

    mode = "linear"

    def __init__(self, cfg: CViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        stride = cfg.token_stride
        self.proj = conv_nd(cfg.ndim)(cfg.input_channels, cfg.embed_dim, stride, stride=stride)
        self.norm = make_norm("layer", cfg.embed_dim, cfg.ndim)
        # The skip stem only goes as deep as the deepest enabled skip.
        deepest = max((i for i, keep in enumerate(cfg.skips) if keep), default=-1)
        self.stem = _ConvStem(cfg, norm="instance", n_levels=deepest + 1) if deepest >= 0 else None

    @property
    def in_proj_keys(self) -> tuple[str, ...]:
        keys = ("proj",)
        if self.stem is not None:
            keys += ("stem.stages.0.blocks.0.conv1.conv", "stem.stages.0.blocks.0.shortcut.conv")
        return keys

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        _check_input(x, self.cfg)
        grid = self.norm(self.proj(x))
        skips: list[torch.Tensor | None] = [None] * self.cfg.n_levels
        if self.stem is not None:
            for i, f in enumerate(self.stem(x)):
                skips[i] = f if self.cfg.skips[i] else None
        return grid, skips


_TOKENIZERS = {
    "hierarchical": HierarchicalConvTokenizer,
    "intra_patch": IntraPatchConvTokenizer,
    "linear": LinearPatchTokenizer,
}


def build_tokenizer(cfg: CViTConfig) -> nn.Module:
    """Instantiate the tokenizer named by ``cfg.tokenizer``."""
    return _TOKENIZERS[cfg.tokenizer](cfg)


__all__ = [
    "HierarchicalConvTokenizer",
    "IntraPatchConvTokenizer",
    "LinearPatchTokenizer",
    "build_tokenizer",
]
