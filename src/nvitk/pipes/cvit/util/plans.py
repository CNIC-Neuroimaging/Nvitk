"""
CViT plans — derive an nnU-Net plans file for a CViT network from a baseline nnU-Net plan.

Description
-----------
nnU-Net's experiment planner turns the dataset fingerprint into everything a network needs
around it: target spacing, normalisation, resampling, patch size, batch size and an
anisotropy-aware pooling schedule. CViT keeps all of that and replaces only the architecture:

``spacing`` / ``normalization_schemes`` / resampling / ``data_identifier``
    Copied unchanged. Keeping the baseline ``data_identifier`` means every CViT variant derived
    from the same baseline shares **one** preprocessed copy of the dataset.
``architecture``
    ``network_class_name = "nvitk.nn.cvit.CViT"`` and ``arch_kwargs`` = the serialised
    :class:`~nvitk.nn.cvit.config.CViTConfig`, plus ``strides`` (nnU-Net reads
    ``pool_op_kernel_sizes`` — and so the deep-supervision scales — from that key).
Stage strides
    The first ``n_levels`` entries of the baseline's pooling schedule, so anisotropic data gets
    per-axis strides such as ``(1, 2, 2)`` decided by nnU-Net's own heuristics. The token stride
    is their product.
Patch size
    The baseline patch (or an explicit one), rounded *down* to a multiple of the token stride on
    every axis.

Plans naming
------------
nnU-Net names a results folder ``<trainer>__<plans>__<configuration>``, so every architectural
variant gets its own plans identifier (:func:`variant_tag`) and therefore its own results folder
— an ablation never overwrites another.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from math import log2, prod
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.logger import Logger
from nvitk.nn.cvit.config import PRESETS, CViTConfig

log = Logger()

#: ``network_class_name`` written into CViT plans.
CVIT_NETWORK_CLASS = "nvitk.nn.cvit.CViT"

#: Plans-identifier prefix.
PLANS_PREFIX = "cvitPlans"

#: Keys never stored in ``arch_kwargs`` (nnU-Net passes them to the constructor itself).
_RUNTIME_KEYS = ("input_channels", "num_classes", "deep_supervision")

_TOKENIZER_TAG = {"hierarchical": "hier", "intra_patch": "intra", "linear": "lin"}


# ──────────────────────────────────────────────────────────────────────────────
# Stem schedule helpers
# ──────────────────────────────────────────────────────────────────────────────


def default_stem_channels(n_levels: int) -> tuple[int, ...]:
    """``32 · 2^i`` capped at 320 — nnU-Net's feature schedule."""
    return tuple(min(32 * 2 ** i, 320) for i in range(n_levels))


def default_stem_blocks(n_levels: int) -> tuple[int, ...]:
    """One block on the two shallowest levels, two below."""
    return tuple(1 if i < 2 else 2 for i in range(n_levels))


def levels_for_token_stride(token_stride: int) -> int:
    """Tokenizer levels for an isotropic power-of-two token stride (8 → 4 levels: 1,2,2,2)."""
    t = int(token_stride)
    if t < 2 or t & (t - 1):
        raise ValueError(f"token stride must be a power of two >= 2, got {token_stride}.")
    return int(log2(t)) + 1


def stem_strides_from_baseline(pool_op_kernel_sizes: Sequence[Sequence[int]], n_levels: int) -> list[list[int]]:
    """The first *n_levels* stages of nnU-Net's pooling schedule (level 0 has stride 1).

    Raises
    ------
    ValueError
        If the baseline network has fewer stages than requested — the patch would be too small to
        reach the token stride.
    """
    pool = [list(int(v) for v in s) for s in pool_op_kernel_sizes]
    if len(pool) < n_levels:
        raise ValueError(
            f"The baseline plan has only {len(pool)} stages; a CViT with {n_levels} tokenizer "
            f"levels needs at least that many. Use a smaller --token-stride."
        )
    return pool[:n_levels]


def round_patch(patch: Sequence[int], token_stride: Sequence[int]) -> list[int]:
    """Round each axis down to a multiple of the token stride (at least one token)."""
    out = [max(t, (int(p) // t) * t) for p, t in zip(patch, token_stride)]
    if out != [int(p) for p in patch]:
        log.info("Patch size %s rounded to %s (token stride %s).", list(patch), out, list(token_stride))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Naming
# ──────────────────────────────────────────────────────────────────────────────


def variant_tag(cfg: CViTConfig, extra: str | None = None) -> str:
    """Short, filesystem-safe description of the architectural variant.

    ``CViTB_hier_unet_s1111`` / ``CViTS_intra_unet_s0011_drop0.3_warm50_gate1e-3`` / …, with a
    hash of the full config appended so two configs that differ only in a field the tag does not
    spell out (depth, widths, registers, …) still get different plans.
    """
    name = cfg.preset or f"E{cfg.embed_dim}D{cfg.depth}H{cfg.num_heads}"
    parts = [name, _TOKENIZER_TAG[cfg.tokenizer], cfg.decoder]
    if cfg.decoder == "unet":
        parts.append("s" + "".join("1" if s else "0" for s in cfg.skips))
        if cfg.skip_drop_prob > 0:
            parts.append(f"drop{cfg.skip_drop_prob:g}")
        if cfg.skip_schedule == "warmup":
            parts.append(f"warm{cfg.skip_warmup_epochs}")
        if cfg.skip_gate == "learned":
            parts.append(f"gate{cfg.skip_gate_l1:g}")
    if extra:
        parts.append("".join(c for c in extra if c.isalnum() or c in "-."))
    digest = hashlib.sha1(json.dumps(_arch_kwargs(cfg), sort_keys=True).encode()).hexdigest()[:6]
    parts.append(digest)
    return "_".join(parts)


def plans_identifier(cfg: CViTConfig, extra: str | None = None) -> str:
    return f"{PLANS_PREFIX}_{variant_tag(cfg, extra)}"


# ──────────────────────────────────────────────────────────────────────────────
# Plans derivation
# ──────────────────────────────────────────────────────────────────────────────


def _arch_kwargs(cfg: CViTConfig) -> dict[str, Any]:
    d = cfg.to_dict()
    for k in _RUNTIME_KEYS:
        d.pop(k, None)
    d["strides"] = [list(s) for s in cfg.stem_strides]
    return d


def build_cvit_config(
    *,
    arch: str | None,
    overrides: dict[str, Any],
    stem_strides: Sequence[Sequence[int]],
    patch_size: Sequence[int],
) -> CViTConfig:
    """A :class:`CViTConfig` for one plan (placeholder in/out channels; nnU-Net sets the real ones)."""
    n = len(stem_strides)
    kwargs: dict[str, Any] = {
        "stem_channels": default_stem_channels(n),
        "stem_blocks": default_stem_blocks(n),
        **overrides,
        "stem_strides": [list(s) for s in stem_strides],
        "input_shape": tuple(int(p) for p in patch_size),
        "input_channels": 1,
        "num_classes": 2,
    }
    if arch:
        if arch not in PRESETS:
            raise ValueError(f"Unknown --arch {arch!r}; choose from {sorted(PRESETS)}.")
        return CViTConfig.from_preset(arch, **kwargs)
    return CViTConfig(**kwargs)


def derive_cvit_plans(
    baseline: dict[str, Any],
    *,
    configuration: str,
    arch: str | None,
    overrides: dict[str, Any] | None = None,
    token_stride: int = 8,
    patch_size: Sequence[int] | None = None,
    batch_size: int | None = None,
    tag: str | None = None,
    stem_strides: Sequence[Sequence[int]] | None = None,
) -> tuple[dict[str, Any], CViTConfig]:
    """CViT plans derived from a baseline nnU-Net plans dict.

    Parameters
    ----------
    baseline
        Parsed baseline plans JSON (e.g. ``nnUNetResEncUNetMPlans.json``).
    configuration
        ``3d_fullres`` / ``2d`` / ``3d_lowres``.
    arch
        Preset name (``CViTS/B/M/L``) or ``None`` for explicit sizes in *overrides*.
    overrides
        Any :class:`CViTConfig` field (tokenizer, decoder, skips, gates, …).
    token_stride
        Isotropic-equivalent token stride (power of two); the per-axis stride follows the
        baseline's pooling schedule.
    patch_size, batch_size
        Overrides of the baseline values.
    tag
        Extra text in the plans identifier (e.g. a run name).
    stem_strides
        Explicit per-level strides (e.g. adopted from a pre-training bundle) instead of the
        baseline's pooling schedule; *token_stride* is then ignored.

    Returns
    -------
    (plans, cfg)
        The new plans dict (single configuration) and its CViT config.
    """
    if configuration not in baseline.get("configurations", {}):
        raise KeyError(
            f"Configuration {configuration!r} not in the baseline plans "
            f"({sorted(baseline.get('configurations', {}))})."
        )
    conf = deepcopy(baseline["configurations"][configuration])
    if "inherits_from" in conf:
        parent = deepcopy(baseline["configurations"][conf.pop("inherits_from")])
        parent.update(conf)
        conf = parent
    base_arch = conf["architecture"]["arch_kwargs"]
    if stem_strides is not None:
        stem = [list(int(v) for v in s) for s in stem_strides]
    else:
        stem = stem_strides_from_baseline(base_arch["strides"], levels_for_token_stride(token_stride))
    token = [prod(s[a] for s in stem) for a in range(len(stem[0]))]
    patch = round_patch(patch_size or conf["patch_size"], token)

    cfg = build_cvit_config(arch=arch, overrides=dict(overrides or {}), stem_strides=stem, patch_size=patch)

    conf["architecture"] = {
        "network_class_name": CVIT_NETWORK_CLASS,
        "arch_kwargs": _arch_kwargs(cfg),
        "_kw_requires_import": [],
    }
    conf["patch_size"] = patch
    if batch_size is not None:
        conf["batch_size"] = int(batch_size)

    plans = deepcopy(baseline)
    plans["plans_name"] = plans_identifier(cfg, tag)
    plans["configurations"] = {configuration: conf}
    plans["experiment_planner_used"] = "nvitk.pipes.cvit.util.plans.derive_cvit_plans"
    plans["cvit"] = {
        "baseline_plans": baseline.get("plans_name"),
        "token_stride": token,
        "grid_shape": list(cfg.grid_shape),
        "num_tokens": int(prod(cfg.grid_shape)),
    }
    return plans, cfg


def write_plans(plans: dict[str, Any], directory: Path) -> Path:
    """Write ``<directory>/<plans_name>.json``; returns the path."""
    path = Path(directory) / f"{plans['plans_name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plans, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CVIT_NETWORK_CLASS",
    "PLANS_PREFIX",
    "build_cvit_config",
    "default_stem_blocks",
    "default_stem_channels",
    "derive_cvit_plans",
    "levels_for_token_stride",
    "plans_identifier",
    "round_patch",
    "stem_strides_from_baseline",
    "variant_tag",
    "write_plans",
]
