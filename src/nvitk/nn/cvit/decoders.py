"""
CViT decoders — token grid (+ conv skips) back to voxel resolution.

Description
-----------
:class:`UNetDecoder`
    nnU-Net-style decoder. Level ``L-1`` (token resolution) fuses the projected transformer
    output ``T`` with the deepest conv feature; every shallower level upsamples with a
    transposed conv and concatenates the matching conv skip. Deep-supervision heads sit on
    levels ``0 … L-2``, highest resolution first — the order and count nnU-Net's
    ``DeepSupervisionWrapper`` expects from ``pool_op_kernel_sizes``.
:class:`PatchDecoder`
    Primus-style transposed-conv stack from the token grid alone (no skips). Used as the
    masked-image-modelling reconstruction head and as a skip-free segmentation ablation.

Skip control
------------
Applied per level ``i`` to the skip ``F_i`` before concatenation::

    F_i ← F_i · skip_scale · gate_i · drop_i

``skips``
    Hard enable; disabled levels are built without the concat channels.
``skip_scale``
    Global scalar set each epoch by the trainer (``skip_schedule="warmup"``).
``gate_i``
    ``sigmoid(g_i)`` when ``skip_gate="learned"``; :meth:`UNetDecoder.gate_penalty` returns
    ``sum(gate_i)`` for the trainer's L1 term.
``drop_i``
    Training only: per-sample Bernoulli keep with probability ``1 - skip_drop_prob``, rescaled
    by ``1/(1-p)`` (inverted dropout); ``p = 1`` zeroes the skip.

There is deliberately no fixed per-skip multiplier: ``concat → conv → InstanceNorm`` absorbs a
constant factor, so it would be a control that does nothing.

Probe hooks
-----------
``token_scale`` (``0`` = decoder sees skips only) and ``skip_off_levels`` (levels zeroed at
inference) are plain attributes flipped by :func:`nvitk.nn.cvit.probe.intervene`.
"""

from __future__ import annotations

from math import log

import torch
from torch import nn

from ..blocks import ConvBlock, ConvNormAct, LayerNormNd, conv_nd, conv_transpose_nd
from .config import CViTConfig

# ──────────────────────────────────────────────────────────────────────────────
# U-Net decoder
# ──────────────────────────────────────────────────────────────────────────────


class UNetDecoder(nn.Module):
    """Skip-connected decoder with deep supervision and skip controls."""

    def __init__(self, cfg: CViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        nd, n = cfg.ndim, cfg.n_levels
        ch = cfg.stem_channels
        self.enabled = tuple(cfg.skips)
        self.deep_supervision = cfg.deep_supervision

        # ---- level L-1: token fusion; levels L-2 … 0: upsample + skip ----------------------
        # stages[i] is the conv body at level i; ups[i] upsamples level i+1 → level i using
        # the stride the tokenizer applied when it went from level i to i+1.
        self.token_proj = ConvNormAct(cfg.embed_dim, ch[-1], nd, kernel_size=1)
        self.stages = nn.ModuleList(
            ConvBlock(ch[i] * (2 if self.enabled[i] else 1), ch[i], nd, n_convs=cfg.decoder_convs)
            for i in range(n)
        )
        self.ups = nn.ModuleList(
            conv_transpose_nd(nd)(ch[i + 1], ch[i], cfg.stem_strides[i + 1], stride=cfg.stem_strides[i + 1])
            for i in range(n - 1)
        )

        # ---- segmentation heads on levels 0 … L-2 -------------------------------------------
        self.seg_layers = nn.ModuleList(
            conv_nd(nd)(ch[i], cfg.num_classes, 1) for i in range(n - 1)
        )

        # ---- skip control ------------------------------------------------------------------
        if cfg.skip_gate == "learned":
            logit = log(cfg.skip_gate_init / (1.0 - cfg.skip_gate_init))
            self.gate_logits = nn.Parameter(torch.full((n,), float(logit)))
        else:
            self.gate_logits = None
        self.skip_drop_prob = float(cfg.skip_drop_prob)
        self.skip_scale = 1.0
        # ---- probe state ----
        self.token_scale = 1.0
        self.skip_off_levels: frozenset[int] = frozenset()

    # ---- control API ---------------------------------------------------------------------
    def set_skip_scale(self, scale: float) -> None:
        """Global skip multiplier in ``[0, 1]`` (warm-up schedule)."""
        self.skip_scale = float(min(max(scale, 0.0), 1.0))

    def gates(self) -> torch.Tensor | None:
        """Current ``sigmoid`` gate values per level, or ``None`` without learned gates."""
        return None if self.gate_logits is None else torch.sigmoid(self.gate_logits)

    def gate_penalty(self) -> torch.Tensor:
        """``sum(gate_i)`` over enabled levels (0 without learned gates); differentiable."""
        g = self.gates()
        if g is None:
            return self.token_proj.conv.weight.new_zeros(())
        mask = torch.tensor(self.enabled, device=g.device, dtype=g.dtype)
        return (g * mask).sum()

    def _apply_skip(self, i: int, f: torch.Tensor) -> torch.Tensor:
        if i in self.skip_off_levels:
            return torch.zeros_like(f)
        scale = self.skip_scale
        out = f * scale if scale != 1.0 else f
        g = self.gates()
        if g is not None:
            out = out * g[i].to(out.dtype)
        p = self.skip_drop_prob
        if self.training and p > 0.0:
            if p >= 1.0:
                return torch.zeros_like(out)
            keep = out.new_empty((out.shape[0],) + (1,) * (out.ndim - 1)).bernoulli_(1.0 - p)
            out = out * keep / (1.0 - p)
        return out

    def _cat(self, i: int, x: torch.Tensor, skips: list[torch.Tensor | None]) -> torch.Tensor:
        if not self.enabled[i]:
            return x
        f = skips[i]
        if f is None:
            raise RuntimeError(f"Skip level {i} is enabled but the tokenizer produced no feature.")
        return torch.cat((x, self._apply_skip(i, f)), dim=1)

    # ---- forward ---------------------------------------------------------------------------
    def forward(
        self, tokens: torch.Tensor, skips: list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor]:
        n = self.cfg.n_levels
        t = tokens * self.token_scale if self.token_scale != 1.0 else tokens
        x = self.stages[n - 1](self._cat(n - 1, self.token_proj(t), skips))
        outs: list[torch.Tensor] = []
        for i in range(n - 2, -1, -1):
            x = self.stages[i](self._cat(i, self.ups[i](x), skips))
            if self.deep_supervision or i == 0:
                outs.append(self.seg_layers[i](x))
        outs = outs[::-1]                                  # highest resolution first
        return outs if self.deep_supervision else outs[0]


# ──────────────────────────────────────────────────────────────────────────────
# Patch decoder
# ──────────────────────────────────────────────────────────────────────────────


class PatchDecoder(nn.Module):
    """Transposed-conv stack token grid → voxels, one step per strided stage, no skips.

    Widths halve each step from ``embed_dim`` (floor 16). ``deep_supervision`` is accepted for
    interface parity with :class:`UNetDecoder` but there is only one output.
    """

    def __init__(self, cfg: CViTConfig, out_channels: int) -> None:
        super().__init__()
        nd = cfg.ndim
        steps = [s for s in reversed(cfg.stem_strides) if any(v > 1 for v in s)]
        layers: list[nn.Module] = []
        c = cfg.embed_dim
        for k, s in enumerate(steps):
            c_out = max(cfg.embed_dim // 2 ** (k + 1), 16)
            layers += [conv_transpose_nd(nd)(c, c_out, s, stride=s), LayerNormNd(c_out), nn.GELU()]
            c = c_out
        self.body = nn.Sequential(*layers)
        self.head = conv_nd(nd)(c, out_channels, 1)
        self.deep_supervision = False
        # ---- probe state (interface parity) ----
        self.token_scale = 1.0
        self.skip_off_levels: frozenset[int] = frozenset()

    def set_skip_scale(self, scale: float) -> None:  # noqa: D401 - no skips to scale
        """No-op (the patch decoder has no skips)."""

    def gates(self) -> None:
        return None

    def gate_penalty(self) -> torch.Tensor:
        return self.head.weight.new_zeros(())

    def forward(self, tokens: torch.Tensor, skips: list | None = None) -> torch.Tensor:
        t = tokens * self.token_scale if self.token_scale != 1.0 else tokens
        return self.head(self.body(t))


__all__ = ["PatchDecoder", "UNetDecoder"]
