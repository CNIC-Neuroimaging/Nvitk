"""
CViT configuration — the serialisable architecture plan.

Description
-----------
:class:`CViTConfig` holds every architectural choice of a CViT network. It round-trips through
plain JSON (:meth:`CViTConfig.to_dict` / :meth:`CViTConfig.from_dict`), which is what nnU-Net
stores as ``arch_kwargs`` in a plans file and what nnssl stores in its adaptation plan, so a
checkpoint always carries the exact recipe needed to rebuild its network.

Array / axis conventions
------------------------
``input_shape`` is the training patch size in the data loader's axis order; its length sets the
spatial dimensionality (2 or 3). ``stem_strides`` is one per-axis stride tuple per tokenizer
level; level 0 is normally stride 1 (full-resolution features for the finest skip). The **token
stride** per axis is the product of all stage strides, and ``input_shape`` must be divisible by
it — anisotropic data simply uses per-axis strides such as ``(1, 2, 2)``.

Presets
-------
Sizes follow Primus (Wald et al., 2025) so that ``head_dim`` is divisible by 6 and every channel
of every head carries a 3D rotary frequency:

======  =========  =====  =====
preset  embed_dim  depth  heads
======  =========  =====  =====
CViTS   396        12     6
CViTB   792        12     12
CViTM   864        16     12
CViTL   1056       24     16
======  =========  =====  =====
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from math import prod
from typing import Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TOKENIZERS: tuple[str, ...] = ("hierarchical", "intra_patch", "linear")
DECODERS: tuple[str, ...] = ("unet", "patch")
SKIP_SCHEDULES: tuple[str, ...] = ("constant", "warmup")
SKIP_GATES: tuple[str, ...] = ("none", "learned")

#: Transformer sizes by preset name.
PRESETS: dict[str, dict[str, int]] = {
    "CViTS": {"embed_dim": 396, "depth": 12, "num_heads": 6},
    "CViTB": {"embed_dim": 792, "depth": 12, "num_heads": 12},
    "CViTM": {"embed_dim": 864, "depth": 16, "num_heads": 12},
    "CViTL": {"embed_dim": 1056, "depth": 24, "num_heads": 16},
}


def parse_skips(spec: str | Sequence[bool] | Sequence[int], n_levels: int) -> tuple[bool, ...]:
    """Normalise a skip specification to one bool per tokenizer level.

    Accepts ``"all"``, ``"none"``, a bit string such as ``"0011"`` (level 0 first), or a
    sequence of bools/ints.

    Raises
    ------
    ValueError
        On an unknown keyword or a length that does not match ``n_levels``.
    """
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "all":
            return (True,) * n_levels
        if key == "none":
            return (False,) * n_levels
        if key and set(key) <= {"0", "1"}:
            spec = [c == "1" for c in key]
        else:
            raise ValueError(f"Unknown skip spec {spec!r}; use 'all', 'none' or bits like '0011'.")
    out = tuple(bool(v) for v in spec)
    if len(out) != n_levels:
        raise ValueError(f"Skip spec has {len(out)} entries but the tokenizer has {n_levels} levels.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CViTConfig:
    """Complete architecture description of a CViT network.

    Parameters
    ----------
    input_channels, num_classes
        Network input channels (modalities) and output channels (segmentation heads, or the
        reconstructed channels for masked image modelling).
    input_shape
        Training patch size; sets the positional-embedding grid (resampled for other sizes).
    tokenizer
        ``"hierarchical"`` — strided residual conv stages over the whole volume (receptive
        fields cross patch borders). ``"intra_patch"`` — the same CNN run independently inside
        each non-overlapping patch (no information crosses patches before attention).
        ``"linear"`` — classic ViT ``Conv(k=s=patch)`` projection; the conv stem then only
        feeds the decoder skips.
    stem_channels, stem_blocks, stem_strides
        Per tokenizer level: width, residual blocks, per-axis stride. ``stem_strides=None``
        means ``(1, 2, 2, …)`` isotropic.
    embed_dim, depth, num_heads, mlp_ratio
        Transformer size (SwiGLU hidden width = ``embed_dim * mlp_ratio``).
    use_rope, use_abs_pos_embed, num_registers
        3D rotary embeddings, learned absolute embedding, extra register tokens.
    drop_path_rate, attn_drop, proj_drop, layer_scale_init
        Regularisation; ``layer_scale_init=None`` disables LayerScale.
    decoder, decoder_convs, deep_supervision
        ``"unet"`` (skips + deep supervision) or ``"patch"`` (transposed-conv patch decoder,
        no skips). ``decoder_convs`` convs per U-Net decoder stage.
    skips
        Per-level skip enable: ``"all"``, ``"none"``, bits ``"0011"`` or bools. Disabled levels
        are built without their concat channels.
    skip_drop_prob
        Training-time probability of zeroing a whole skip level per sample (inverted dropout).
    skip_schedule, skip_warmup_epochs
        ``"warmup"`` ramps the skip scale 0 → 1 over the first epochs (applied by the trainer
        through ``set_skip_scale``).
    skip_gate, skip_gate_init, skip_gate_l1
        ``"learned"`` adds a per-level gate ``sigmoid(g)`` (initial value ``skip_gate_init``);
        the trainer adds ``skip_gate_l1 * sum(gates)`` to the loss.
    """

    input_channels: int = 1
    num_classes: int = 2
    input_shape: tuple[int, ...] = (128, 128, 128)
    # ---- tokenizer -----------------------------------------------------------------------
    tokenizer: str = "hierarchical"
    stem_channels: tuple[int, ...] = (32, 64, 128, 256)
    stem_blocks: tuple[int, ...] = (1, 1, 2, 2)
    stem_strides: tuple[tuple[int, ...], ...] | None = None
    # ---- transformer ---------------------------------------------------------------------
    embed_dim: int = 792
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 8.0 / 3.0
    use_rope: bool = True
    use_abs_pos_embed: bool = True
    num_registers: int = 0
    drop_path_rate: float = 0.1
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    layer_scale_init: float | None = 0.1
    # ---- decoder -------------------------------------------------------------------------
    decoder: str = "unet"
    decoder_convs: int = 2
    deep_supervision: bool = True
    # ---- skip control --------------------------------------------------------------------
    skips: Any = "all"
    skip_drop_prob: float = 0.0
    skip_schedule: str = "constant"
    skip_warmup_epochs: int = 50
    skip_gate: str = "none"
    skip_gate_init: float = 0.5
    skip_gate_l1: float = 0.0
    # ---- provenance ----------------------------------------------------------------------
    preset: str | None = field(default=None)

    # ---- construction --------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.input_shape = tuple(int(v) for v in self.input_shape)
        self.stem_channels = tuple(int(v) for v in self.stem_channels)
        self.stem_blocks = tuple(int(v) for v in self.stem_blocks)
        if self.stem_strides is None:
            self.stem_strides = tuple(
                (1 if i == 0 else 2,) * self.ndim for i in range(len(self.stem_channels))
            )
        else:
            self.stem_strides = tuple(
                (int(s),) * self.ndim if isinstance(s, int) else tuple(int(v) for v in s)
                for s in self.stem_strides
            )
        self.skips = parse_skips(self.skips, self.n_levels)
        self.validate()

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "CViTConfig":
        """Config for a named preset (``CViTS/B/M/L``) with field overrides."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}.")
        kwargs = {**PRESETS[name], "preset": name}
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CViTConfig":
        """Inverse of :meth:`to_dict`; unknown keys raise so typos never pass silently."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown CViTConfig field(s): {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (tuples become lists)."""
        def _plain(v: Any) -> Any:
            if isinstance(v, (tuple, list)):
                return [_plain(i) for i in v]
            return v
        return {k: _plain(v) for k, v in asdict(self).items()}

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, text: str) -> "CViTConfig":
        return cls.from_dict(json.loads(text))

    def replace(self, **changes: Any) -> "CViTConfig":
        """A copy with *changes* applied (re-validated)."""
        return CViTConfig.from_dict({**self.to_dict(), **changes})

    # ---- derived geometry ----------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.input_shape)

    @property
    def n_levels(self) -> int:
        return len(self.stem_channels)

    @property
    def token_stride(self) -> tuple[int, ...]:
        """Per-axis product of all stage strides (voxels per token along each axis)."""
        return tuple(prod(s[a] for s in self.stem_strides) for a in range(self.ndim))

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Token grid for ``input_shape``."""
        return tuple(i // t for i, t in zip(self.input_shape, self.token_stride))

    @property
    def level_strides(self) -> list[tuple[int, ...]]:
        """Cumulative per-axis stride of each level relative to the input."""
        out, cur = [], [1] * self.ndim
        for s in self.stem_strides:
            cur = [c * v for c, v in zip(cur, s)]
            out.append(tuple(cur))
        return out

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def pool_op_kernel_sizes(self) -> list[list[int]]:
        """Stage strides in nnU-Net's plans format.

        nnU-Net derives deep-supervision scales as ``1 / cumprod(pool_op_kernel_sizes)[:-1]``,
        i.e. one target per level except the deepest — exactly the U-Net decoder's outputs.
        """
        return [list(s) for s in self.stem_strides]

    # ---- validation ----------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` on any inconsistent combination."""
        if self.ndim not in (2, 3):
            raise ValueError(f"input_shape must be 2D or 3D, got {self.input_shape}.")
        if self.tokenizer not in TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got {self.tokenizer!r}.")
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {self.decoder!r}.")
        if self.skip_schedule not in SKIP_SCHEDULES:
            raise ValueError(f"skip_schedule must be one of {SKIP_SCHEDULES}.")
        if self.skip_gate not in SKIP_GATES:
            raise ValueError(f"skip_gate must be one of {SKIP_GATES}.")
        n = self.n_levels
        if n < 2:
            raise ValueError("The tokenizer needs at least two levels.")
        if len(self.stem_blocks) != n or len(self.stem_strides) != n:
            raise ValueError(
                f"stem_channels/stem_blocks/stem_strides must have equal length "
                f"(got {n}, {len(self.stem_blocks)}, {len(self.stem_strides)})."
            )
        for s in self.stem_strides:
            if len(s) != self.ndim or any(v < 1 for v in s):
                raise ValueError(f"Invalid stage stride {s} for a {self.ndim}D input.")
        bad = [i for i, t in zip(self.input_shape, self.token_stride) if i % t]
        if bad:
            raise ValueError(
                f"input_shape {self.input_shape} must be divisible by the token stride "
                f"{self.token_stride} on every axis."
            )
        if self.embed_dim % self.num_heads:
            raise ValueError(f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}.")
        if not 0.0 <= self.skip_drop_prob <= 1.0:
            raise ValueError(f"skip_drop_prob must be in [0, 1], got {self.skip_drop_prob}.")
        if not 0.0 < self.skip_gate_init < 1.0:
            raise ValueError("skip_gate_init is a sigmoid value and must lie in (0, 1).")
        if self.skip_gate_l1 < 0:
            raise ValueError("skip_gate_l1 must be >= 0.")


__all__ = [
    "CViTConfig",
    "DECODERS",
    "PRESETS",
    "SKIP_GATES",
    "SKIP_SCHEDULES",
    "TOKENIZERS",
    "parse_skips",
]
