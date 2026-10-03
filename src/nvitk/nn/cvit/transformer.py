"""
CViT transformer encoder — global attention over the conv token grid.

Description
-----------
Pre-norm blocks (``x + LS(Attn(LN x))``, ``x + LS(SwiGLU(LN x))``) with stochastic depth,
fused ``scaled_dot_product_attention`` (flash / memory-efficient kernels when available),
axial N-D rotary position embeddings, an optional learned absolute embedding that is
trilinearly resampled to any token grid, and optional register tokens.

Masked image modelling
----------------------
``token_mask`` (SimMIM) replaces masked tokens by a learned ``mask_token`` before the position
embedding is added. ``keep_idx`` (MAE) runs the encoder on the kept tokens only; rotary and
absolute embeddings are gathered at the kept positions so every token still knows where it is.

Probe hooks
-----------
The attention-usage probe (:mod:`nvitk.nn.cvit.probe`) flips plain attributes rather than
patching code: ``Block.attn_enabled``, ``Attention.mode`` (``"normal"``/``"uniform"``),
``TransformerEncoder.bypass`` and ``TransformerEncoder.local_mask_spec``. Their defaults
reproduce the trained network exactly.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import DropPath, LayerScale, SwiGLU
from .config import CViTConfig

# ──────────────────────────────────────────────────────────────────────────────
# Positional encodings
# ──────────────────────────────────────────────────────────────────────────────


def grid_coords(grid_shape: Sequence[int], device=None) -> torch.Tensor:
    """Integer coordinates of a row-major token grid, ``(N, ndim)`` float."""
    axes = [torch.arange(int(s), device=device, dtype=torch.float32) for s in grid_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


class RoPE(nn.Module):
    """Axial rotary embedding over an N-D grid.

    The first ``rot_dim = 2·ndim·n_freq`` channels of each head are rotated, one frequency band
    per axis; any remainder of ``head_dim`` passes through unrotated, so every head size works
    (Primus-style sizes with ``head_dim % 6 == 0`` rotate every channel in 3D).
    """

    def __init__(self, head_dim: int, ndim: int, theta: float = 100.0) -> None:
        super().__init__()
        self.ndim = ndim
        self.n_freq = head_dim // (2 * ndim)
        if self.n_freq < 1:
            raise ValueError(f"head_dim {head_dim} too small for {ndim}D rotary embedding.")
        self.rot_dim = 2 * ndim * self.n_freq
        inv = 1.0 / (theta ** (torch.arange(self.n_freq, dtype=torch.float32) / self.n_freq))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, grid_shape: Sequence[int], device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)``, each ``(N, rot_dim // 2)``."""
        coords = grid_coords(grid_shape, device)                         # (N, ndim)
        ang = (coords[:, :, None] * self.inv_freq.to(device)[None, None]).reshape(coords.shape[0], -1)
        return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``2·cos.shape[-1]`` channels of ``x`` (…, N, head_dim)."""
    half = cos.shape[-1]
    rot, rest = x[..., : 2 * half], x[..., 2 * half:]
    x1, x2 = rot[..., :half], rot[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return torch.cat((out, rest), dim=-1) if rest.shape[-1] else out


def resize_pos_embed(pe: torch.Tensor, grid_shape: Sequence[int]) -> torch.Tensor:
    """Resample a channels-first ``(1, E, *g)`` embedding to ``grid_shape`` (linear interpolation)."""
    grid_shape = tuple(int(s) for s in grid_shape)
    if tuple(pe.shape[2:]) == grid_shape:
        return pe
    mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[len(grid_shape)]
    return F.interpolate(pe, size=grid_shape, mode=mode, align_corners=False)


# ──────────────────────────────────────────────────────────────────────────────
# Attention + block
# ──────────────────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    """Multi-head self-attention through ``F.scaled_dot_product_attention``.

    Probe attribute ``mode``: ``"normal"`` or ``"uniform"`` (every query receives the mean of
    all values — global pooling with no learned pattern).
    """

    def __init__(self, dim: int, num_heads: int, *, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = "normal"

    def qkv_heads(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, n_prefix: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``q, k, v`` as ``(B, H, N, head_dim)`` with rotary applied to the grid tokens."""
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if rope is not None:
            cos, sin = rope
            q = torch.cat((q[:, :, :n_prefix], apply_rope(q[:, :, n_prefix:], cos, sin)), dim=2)
            k = torch.cat((k[:, :, :n_prefix], apply_rope(k[:, :, n_prefix:], cos, sin)), dim=2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_prefix: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv_heads(x, rope, n_prefix)
        if self.mode == "uniform":
            out = v.mean(dim=2, keepdim=True).expand_as(v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0.0
            )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    """Pre-norm transformer block. Probe attribute ``attn_enabled=False`` drops the attention branch."""

    def __init__(self, cfg: CViTConfig, drop_path: float) -> None:
        super().__init__()
        dim = cfg.embed_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, cfg.num_heads, attn_drop=cfg.attn_drop, proj_drop=cfg.proj_drop)
        self.ls1 = LayerScale(dim, cfg.layer_scale_init)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, int(dim * cfg.mlp_ratio), drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, cfg.layer_scale_init)
        self.drop_path = DropPath(drop_path)
        self.attn_enabled = True

    def attn_branch(self, x, rope=None, n_prefix: int = 0, attn_mask=None) -> torch.Tensor:
        """The attention residual update ``LS(Attn(LN x))`` (before stochastic depth)."""
        return self.ls1(self.attn(self.norm1(x), rope, n_prefix, attn_mask))

    def forward(self, x, rope=None, n_prefix: int = 0, attn_mask=None) -> torch.Tensor:
        if self.attn_enabled:
            x = x + self.drop_path(self.attn_branch(x, rope, n_prefix, attn_mask))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────


def local_attention_mask(
    grid_shape: Sequence[int],
    token_spacing_mm: Sequence[float],
    radius_mm: float,
    n_prefix: int,
    device=None,
    index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean ``(N+P, N+P)`` mask: grid tokens attend only to tokens within ``radius_mm``.

    Register (prefix) tokens keep full attention in both directions — they carry no position.
    ``index`` restricts the grid tokens to a subset (MAE kept tokens; must be shared by the batch).
    """
    coords = grid_coords(grid_shape, device) * torch.as_tensor(
        token_spacing_mm, dtype=torch.float32, device=device
    )
    if index is not None:
        coords = coords[index]
    near = torch.cdist(coords, coords) <= float(radius_mm)
    n = near.shape[0] + n_prefix
    mask = torch.ones((n, n), dtype=torch.bool, device=device)
    mask[n_prefix:, n_prefix:] = near
    return mask


class TransformerEncoder(nn.Module):
    """Transformer over the token grid; returns a grid of the same shape (or kept tokens for MAE)."""

    def __init__(self, cfg: CViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, dim, *cfg.grid_shape)) if cfg.use_abs_pos_embed else None
        )
        self.registers = (
            nn.Parameter(torch.zeros(1, cfg.num_registers, dim)) if cfg.num_registers > 0 else None
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.rope = RoPE(cfg.head_dim, cfg.ndim) if cfg.use_rope else None
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(Block(cfg, p) for p in dpr)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        for p in (self.pos_embed, self.registers, self.mask_token):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)
        # ---- probe state (defaults = trained behaviour) ----
        self.bypass = False
        self.local_mask_spec: tuple[float, tuple[float, ...]] | None = None   # (radius_mm, token_spacing_mm)

    @property
    def n_prefix(self) -> int:
        return 0 if self.registers is None else self.registers.shape[1]

    def forward(
        self,
        grid: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        keep_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a token grid.

        Parameters
        ----------
        grid
            ``(B, E, *g)`` tokenizer output.
        token_mask
            ``(B, *g)`` bool, ``True`` = masked (SimMIM): replaced by ``mask_token``.
        keep_idx
            ``(B, K)`` long indices of kept tokens (MAE). Returns ``(B, K, E)`` in that case.

        Returns
        -------
        Tensor
            ``(B, E, *g)``, or ``(B, K, E)`` when ``keep_idx`` is given.
        """
        if self.bypass:
            return grid
        b, e, *g = grid.shape
        n = prod(g)
        tokens = grid.flatten(2).transpose(1, 2)                                   # (B, N, E)

        # ---- 1. Masking + absolute position ----------------------------------------------
        if token_mask is not None:
            m = token_mask.reshape(b, n, 1)
            tokens = torch.where(m, self.mask_token.to(tokens.dtype).expand(b, n, e), tokens)
        if self.pos_embed is not None:
            tokens = tokens + resize_pos_embed(self.pos_embed, g).flatten(2).transpose(1, 2)

        # ---- 2. Rotary tables (gathered at kept positions for MAE) ----------------------
        rope = None
        if self.rope is not None:
            cos, sin = self.rope(g, grid.device)                                    # (N, r)
            if keep_idx is not None:
                cos, sin = cos[keep_idx][:, None], sin[keep_idx][:, None]           # (B, 1, K, r)
            rope = (cos, sin)
        if keep_idx is not None:
            tokens = torch.gather(tokens, 1, keep_idx[..., None].expand(-1, -1, e))

        # ---- 3. Registers, optional locality mask, blocks --------------------------------
        if self.registers is not None:
            tokens = torch.cat((self.registers.to(tokens.dtype).expand(b, -1, -1), tokens), dim=1)
        attn_mask = None
        if self.local_mask_spec is not None:
            radius, spacing = self.local_mask_spec
            if keep_idx is not None and not bool((keep_idx == keep_idx[:1]).all()):
                raise ValueError("A local attention mask needs the same kept tokens across the batch.")
            attn_mask = local_attention_mask(
                g, spacing, radius, self.n_prefix, grid.device,
                index=None if keep_idx is None else keep_idx[0],
            )
        for blk in self.blocks:
            tokens = blk(tokens, rope, self.n_prefix, attn_mask)
        tokens = self.norm(tokens)[:, self.n_prefix:]

        if keep_idx is not None:
            return tokens
        return tokens.transpose(1, 2).reshape(b, e, *g)


__all__ = [
    "Attention",
    "Block",
    "RoPE",
    "TransformerEncoder",
    "apply_rope",
    "grid_coords",
    "local_attention_mask",
    "resize_pos_embed",
]
