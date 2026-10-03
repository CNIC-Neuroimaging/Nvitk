"""
Attention-usage probe — how much does a trained CViT actually rely on its transformer?

Description
-----------
Hybrid CNN-transformer segmenters can learn to route everything through the conv skips and
leave the transformer idle (the critique behind Primus, Wald et al. 2025). This module measures
that from two angles.

Interventions (:func:`intervene`)
    Context manager that changes the network's behaviour at inference and restores it on exit.
    Compare a segmentation metric under each mode against ``"full"``:

    =====================  ========================================================================
    ``full``               unchanged (reference)
    ``attn_off``           drop the attention residual branch (MLP kept) in all / selected layers
    ``attn_local``         tokens may only attend within ``radius_mm`` (physical units)
    ``attn_uniform``       attention weights replaced by a uniform average (global pooling)
    ``transformer_off``    the token grid bypasses the whole encoder
    ``tokens_off``         the token path into the decoder is zeroed (decoder sees skips only)
    ``skips_off``          selected skip levels zeroed (decoder sees tokens only)
    =====================  ========================================================================

Passive statistics (:func:`attention_stats`)
    Explicit-softmax attention on a random subset of query tokens: mean attention distance in
    **mm** per layer and head, normalised entropy, attention mass on register tokens, and the
    residual ratio ``‖LS(Attn(LN x))‖ / ‖x‖`` — how much each layer's attention changes its
    input. Plus learned skip-gate values (:func:`skip_gate_values`).

Units
-----
``spacing_mm`` is the **voxel** spacing of the network input, in the network's axis order; the
token spacing is ``spacing_mm × token_stride``. Distances are therefore physical, so models with
different strides or datasets with different resolutions are comparable.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import log
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from .transformer import grid_coords
from .weights import _unwrap

MODES: tuple[str, ...] = (
    "full", "attn_off", "attn_local", "attn_uniform", "transformer_off", "tokens_off", "skips_off",
)

# ──────────────────────────────────────────────────────────────────────────────
# Interventions
# ──────────────────────────────────────────────────────────────────────────────


def _token_spacing(net: nn.Module, spacing_mm: Sequence[float]) -> tuple[float, ...]:
    cfg = net.config
    if len(spacing_mm) != cfg.ndim:
        raise ValueError(f"spacing_mm needs {cfg.ndim} values (network axis order), got {spacing_mm}.")
    return tuple(float(s) * t for s, t in zip(spacing_mm, cfg.token_stride))


@contextmanager
def intervene(
    model: nn.Module,
    mode: str,
    *,
    layers: Sequence[int] | None = None,
    levels: Sequence[int] | None = None,
    radius_mm: float | None = None,
    spacing_mm: Sequence[float] | None = None,
) -> Iterator[nn.Module]:
    """Temporarily apply one probe intervention to a CViT (DDP / compiled wrappers accepted).

    Parameters
    ----------
    mode
        One of :data:`MODES`.
    layers
        ``attn_off`` only: transformer block indices (default: all).
    levels
        ``skips_off`` only: skip levels to zero (default: all enabled levels).
    radius_mm, spacing_mm
        ``attn_local`` only: neighbourhood radius and the input voxel spacing.

    Yields
    ------
    nn.Module
        The unwrapped network (already modified).
    """
    if mode not in MODES:
        raise ValueError(f"Unknown probe mode {mode!r}; choose from {MODES}.")
    net = _unwrap(model)
    enc, dec = net.encoder, net.decoder
    saved: dict[str, Any] = {
        "attn_enabled": [b.attn_enabled for b in enc.blocks],
        "attn_mode": [b.attn.mode for b in enc.blocks],
        "bypass": enc.bypass,
        "local": enc.local_mask_spec,
        "token_scale": dec.token_scale,
        "skip_off": dec.skip_off_levels,
    }
    try:
        if mode == "attn_off":
            idx = range(len(enc.blocks)) if layers is None else layers
            for i in idx:
                enc.blocks[int(i)].attn_enabled = False
        elif mode == "attn_local":
            if radius_mm is None or spacing_mm is None:
                raise ValueError("attn_local needs radius_mm and spacing_mm.")
            enc.local_mask_spec = (float(radius_mm), _token_spacing(net, spacing_mm))
        elif mode == "attn_uniform":
            for b in enc.blocks:
                b.attn.mode = "uniform"
        elif mode == "transformer_off":
            enc.bypass = True
        elif mode == "tokens_off":
            dec.token_scale = 0.0
        elif mode == "skips_off":
            enabled = [i for i, e in enumerate(getattr(dec, "enabled", ())) if e]
            dec.skip_off_levels = frozenset(int(i) for i in (enabled if levels is None else levels))
        yield net
    finally:
        for b, on, m in zip(enc.blocks, saved["attn_enabled"], saved["attn_mode"]):
            b.attn_enabled, b.attn.mode = on, m
        enc.bypass = saved["bypass"]
        enc.local_mask_spec = saved["local"]
        dec.token_scale = saved["token_scale"]
        dec.skip_off_levels = saved["skip_off"]


# ──────────────────────────────────────────────────────────────────────────────
# Passive statistics
# ──────────────────────────────────────────────────────────────────────────────


def skip_gate_values(model: nn.Module) -> dict[int, float]:
    """Learned skip-gate values per enabled level (empty without learned gates)."""
    dec = _unwrap(model).decoder
    gates = dec.gates()
    if gates is None:
        return {}
    return {i: float(g) for i, (g, e) in enumerate(zip(gates.detach().cpu(), dec.enabled)) if e}


@torch.no_grad()
def attention_stats(
    model: nn.Module,
    x: torch.Tensor,
    *,
    spacing_mm: Sequence[float],
    n_queries: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-layer attention statistics on input *x* (run in eval mode; restores the mode after).

    Parameters
    ----------
    x
        ``(B, C, *S)`` network input (e.g. one validation batch).
    spacing_mm
        Voxel spacing of *x* in network axis order.
    n_queries
        Grid tokens sampled as queries (explicit attention is ``O(Q·N)`` instead of ``O(N²)``).

    Returns
    -------
    dict
        ``{"layers": [{"layer", "mean_distance_mm", "mean_distance_mm_per_head", "entropy",
        "register_mass", "residual_ratio"}, ...], "skip_gates": {...}, "token_spacing_mm": [...],
        "grid_shape": [...]}``
    """
    net = _unwrap(model)
    enc = net.encoder
    token_spacing = _token_spacing(net, spacing_mm)
    captured: list[tuple] = []
    grid_shape: list[tuple[int, ...]] = []

    def _grab_block(_mod, args):
        captured.append(args)

    def _grab_grid(_mod, args):
        grid_shape.append(tuple(args[0].shape[2:]))

    handles = [b.register_forward_pre_hook(_grab_block) for b in enc.blocks]
    handles.append(enc.register_forward_pre_hook(_grab_grid))
    was_training = net.training
    net.eval()
    try:
        net(x)
    finally:
        for h in handles:
            h.remove()
        net.train(was_training)
    if not captured:
        return {"layers": [], "skip_gates": skip_gate_values(net), "token_spacing_mm": list(token_spacing),
                "grid_shape": list(grid_shape[0]) if grid_shape else []}

    g = grid_shape[0]
    coords = grid_coords(g, x.device) * torch.as_tensor(token_spacing, device=x.device)
    n_grid = coords.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    q_idx = torch.randperm(n_grid, generator=gen)[: min(int(n_queries), n_grid)].to(x.device)
    dist = torch.cdist(coords[q_idx], coords)                                  # (Q, N)

    layers = []
    for i, (blk, args) in enumerate(zip(enc.blocks, captured)):
        tokens, rope, n_prefix = args[0], args[1], int(args[2])
        attn_mask = args[3] if len(args) > 3 else None
        h = blk.norm1(tokens)
        q, k, _ = blk.attn.qkv_heads(h, rope, n_prefix)
        qq = q[:, :, n_prefix + q_idx].float()                                   # (B, H, Q, d)
        logits = qq @ k.float().transpose(-1, -2) * (blk.attn.head_dim ** -0.5)  # (B, H, Q, N+P)
        if attn_mask is not None:
            logits = logits.masked_fill(~attn_mask[n_prefix + q_idx][None, None], float("-inf"))
        if blk.attn.mode == "uniform":
            probs = torch.full_like(logits, 1.0 / logits.shape[-1])
        else:
            probs = logits.softmax(dim=-1)
        p_grid = probs[..., n_prefix:]
        reg_mass = probs[..., :n_prefix].sum(-1) if n_prefix else probs.new_zeros(probs.shape[:-1])
        p_norm = p_grid / p_grid.sum(-1, keepdim=True).clamp_min(1e-12)
        mean_dist = (p_norm * dist[None, None]).sum(-1)                         # (B, H, Q)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / log(max(probs.shape[-1], 2))
        delta = blk.attn_branch(tokens, rope, n_prefix, attn_mask)
        ratio = (delta.float().norm(dim=-1) / tokens.float().norm(dim=-1).clamp_min(1e-12)).mean()
        layers.append({
            "layer": i,
            "attn_enabled": bool(blk.attn_enabled),
            "mean_distance_mm": float(mean_dist.mean()),
            "mean_distance_mm_per_head": [float(v) for v in mean_dist.mean(dim=(0, 2))],
            "entropy": float(ent.mean()),
            "register_mass": float(reg_mass.mean()),
            "residual_ratio": float(ratio),
        })
    return {
        "layers": layers,
        "skip_gates": skip_gate_values(net),
        "token_spacing_mm": list(token_spacing),
        "grid_shape": list(g),
    }


__all__ = ["MODES", "attention_stats", "intervene", "skip_gate_values"]
