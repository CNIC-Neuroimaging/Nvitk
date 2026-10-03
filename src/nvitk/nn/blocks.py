"""
Dimension-agnostic convolutional and transformer building blocks.

Description
-----------
Small, composable modules shared by the nvitk networks. Every conv block takes ``ndim``
(1, 2 or 3) and picks the matching ``ConvNd`` / ``InstanceNormNd`` so one implementation serves
2D slices and 3D volumes.

Array / axis conventions
------------------------
Channels-first tensors, ``(B, C, *spatial)``; ``spatial`` is ``(X, Y)`` or ``(X, Y, Z)`` in the
order the data loader delivers it. Nothing here assumes isotropy: strides and kernel sizes are
per-axis tuples wherever they appear.

Initialisation
--------------
:func:`init_weights` follows nnU-Net (He-normal with ``neg_slope=1e-2`` for convolutions, so the
LeakyReLU stacks start well scaled) and ViT practice (truncated normal, std 0.02, for linear
layers). It only touches Conv/Linear/Norm modules, so applying it with ``Module.apply`` never
resets parameters owned by other modules (positional embeddings, LayerScale, gates).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Operator lookup
# ──────────────────────────────────────────────────────────────────────────────

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_CONV_T = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_INORM = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}


def conv_nd(ndim: int) -> type[nn.Module]:
    """``nn.Conv{ndim}d``; raises ``ValueError`` for an unsupported dimensionality."""
    try:
        return _CONV[int(ndim)]
    except KeyError:
        raise ValueError(f"Unsupported spatial dimensionality {ndim}; expected 1, 2 or 3.")


def conv_transpose_nd(ndim: int) -> type[nn.Module]:
    """``nn.ConvTranspose{ndim}d``."""
    conv_nd(ndim)
    return _CONV_T[int(ndim)]


def as_tuple(value: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Broadcast an int to an ``ndim``-tuple, or validate a sequence's length."""
    if isinstance(value, int):
        return (int(value),) * ndim
    out = tuple(int(v) for v in value)
    if len(out) != ndim:
        raise ValueError(f"Expected {ndim} values, got {out}.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

class LayerNormNd(nn.Module):
    """LayerNorm over the channel axis of a channels-first tensor, per spatial location.

    Unlike ``InstanceNorm`` it uses no spatial statistics, so it is well defined on a 1-voxel
    map and never mixes information between locations (required by the intra-patch tokenizer).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.view(shape) + self.bias.view(shape)


def make_norm(kind: str, num_channels: int, ndim: int) -> nn.Module:
    """``"instance"`` (affine InstanceNorm, nnU-Net default) or ``"layer"`` (:class:`LayerNormNd`)."""
    if kind == "instance":
        return _INORM[int(ndim)](num_channels, eps=1e-5, affine=True)
    if kind == "layer":
        return LayerNormNd(num_channels)
    raise ValueError(f"Unknown norm {kind!r}; expected 'instance' or 'layer'.")


# ──────────────────────────────────────────────────────────────────────────────
# Convolutional blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvNormAct(nn.Module):
    """``Conv → Norm → LeakyReLU(0.01)`` with 'same' padding for odd kernels.

    Parameters
    ----------
    act
        ``False`` drops the nonlinearity (used for the second conv of a residual block).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        norm: str = "instance",
        act: bool = True,
    ) -> None:
        super().__init__()
        k = as_tuple(kernel_size, ndim)
        s = as_tuple(stride, ndim)
        self.conv = conv_nd(ndim)(
            in_channels, out_channels, k, stride=s, padding=tuple(i // 2 for i in k), bias=True
        )
        self.norm = make_norm(norm, out_channels, ndim)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Basic residual block: two 3^n convs, the first possibly strided, plus a projected shortcut.

    The shortcut is a strided 1^n conv + norm whenever stride or width changes, as in nnU-Net's
    residual encoders. It is the second module that consumes the block input, which matters for
    in-channel weight adaptation (see :mod:`nvitk.nn.cvit.weights`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        *,
        stride: int | Sequence[int] = 1,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        s = as_tuple(stride, ndim)
        self.conv1 = ConvNormAct(in_channels, out_channels, ndim, stride=s, norm=norm)
        self.conv2 = ConvNormAct(out_channels, out_channels, ndim, norm=norm, act=False)
        if any(i != 1 for i in s) or in_channels != out_channels:
            self.shortcut = ConvNormAct(
                in_channels, out_channels, ndim, kernel_size=1, stride=s, norm=norm, act=False
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResStage(nn.Module):
    """``n_blocks`` :class:`ResBlock` s; only the first is strided / changes width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int,
        *,
        n_blocks: int,
        stride: int | Sequence[int],
        norm: str = "instance",
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"A stage needs at least one block, got {n_blocks}.")
        blocks = [ResBlock(in_channels, out_channels, ndim, stride=stride, norm=norm)]
        blocks += [ResBlock(out_channels, out_channels, ndim, norm=norm) for _ in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvBlock(nn.Sequential):
    """``n_convs`` × :class:`ConvNormAct` (decoder stage body)."""

    def __init__(self, in_channels: int, out_channels: int, ndim: int, *, n_convs: int = 2) -> None:
        layers = [ConvNormAct(in_channels, out_channels, ndim)]
        layers += [ConvNormAct(out_channels, out_channels, ndim) for _ in range(n_convs - 1)]
        super().__init__(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer helpers
# ──────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample with probability ``p``."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    """Per-channel learnable residual scale (CaiT); ``init=None`` makes it the identity."""

    def __init__(self, dim: int, init: float | None = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init))) if init is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.gamma is None else x * self.gamma


class SwiGLU(nn.Module):
    """Gated MLP ``fc2(silu(a) * b)`` with ``[a, b] = fc1(x)``."""

    def __init__(self, dim: int, hidden: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(self.drop(F.silu(a) * b)))


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_weights(module: nn.Module, neg_slope: float = 1e-2) -> None:
    """Initialise one module in place (use with ``Module.apply``).

    Conv / ConvTranspose → He-normal (``a=neg_slope``), zero bias. Linear → truncated normal
    (std 0.02), zero bias. Affine norms → weight 1, bias 0. Everything else is left alone.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                           nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerNormNd,
                             nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "ConvBlock",
    "ConvNormAct",
    "DropPath",
    "LayerNormNd",
    "LayerScale",
    "ResBlock",
    "ResStage",
    "SwiGLU",
    "as_tuple",
    "conv_nd",
    "conv_transpose_nd",
    "init_weights",
    "make_norm",
]
