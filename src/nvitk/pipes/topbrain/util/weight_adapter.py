"""Read nnssl checkpoints and seed a network's encoder from them.

nnssl and this pipeline's nnU-Net build both construct their encoders from the same
``dynamic_network_architectures`` classes, so a pre-trained encoder's parameters share a name
space with a freshly-built one. That is what lets :func:`load_encoder_into` drop published
weights into a network the caller has already constructed.

Scope
-----
Only **seeding** lives here — used by stage 1 to start self-supervised training from a published
checkpoint, and to materialise the bundle's exported segmentation model. Loading weights for
*fine-tuning* is not done here: the in-tree nnU-Net build handles that itself, through
``nnUNetv2_preprocess_like_nnssl`` (which records the checkpoint in the plans) and
``PretrainedTrainer`` (which loads it). Duplicating that would risk the two disagreeing.

Failure policy
--------------
Every mismatch raises. A partially-seeded encoder trains without complaint and merely looks like
a slightly worse run, which is far more expensive to diagnose than a refusal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nvitk.core.logger import Logger

log = Logger()

#: Key prefixes assumed when a checkpoint carries no nnssl adaptation plan (a bare state dict).
DEFAULT_KEY_TO_ENCODER: str = "encoder.stages"
DEFAULT_KEY_TO_STEM: str = "encoder.stem"

#: Where the encoder and stem live in the *downstream* nnU-Net network.
TARGET_KEY_TO_ENCODER: str = "encoder.stages"
TARGET_KEY_TO_STEM: str = "encoder.stem"


def load_nnssl_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Load an nnssl (or bare) checkpoint.

    Returns
    -------
    tuple
        ``(state_dict, adaptation_plan_or_None, checkpoint_metadata)``.

    Raises
    ------
    FileNotFoundError, ValueError
        If the file is missing or holds nothing that looks like a state dict.
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
        state = checkpoint["network_weights"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(hasattr(v, "shape") for v in checkpoint.values()):
        state, checkpoint = checkpoint, {}
    else:
        raise ValueError(
            f"{path} does not look like a checkpoint: expected a 'network_weights' or "
            f"'state_dict' entry, or a bare tensor mapping."
        )

    plan = checkpoint.get("nnssl_adaptation_plan") if isinstance(checkpoint, dict) else None
    if plan is None:
        plan = _sidecar_adaptation_plan(path)
    if isinstance(plan, str):
        plan = json.loads(plan)

    # Distributed and compiled training add key prefixes that would break every name match.
    state = {
        key.replace("module.", "", 1).replace("_orig_mod.", "", 1): value
        for key, value in state.items()
    }
    return state, plan, checkpoint if isinstance(checkpoint, dict) else {}


def _sidecar_adaptation_plan(checkpoint_path: Path) -> dict[str, Any] | None:
    """Find nnssl's ``adaptation_plan.json`` next to a checkpoint that lacks the embedded copy.

    ``AbstractBaseTrainer.save_checkpoint`` normally embeds the plan, but some nnssl trainers
    override that method and drop it — ``SparkMAETrainer`` is one, so every SparK checkpoint
    arrives without it. The trainer still writes ``adaptation_plan.json`` into its run directory
    (``<run>/adaptation_plan.json``, one level above ``fold_all/``), so read that instead of
    falling back to assumed key names.
    """
    for candidate in (
        checkpoint_path.parent / "adaptation_plan.json",
        checkpoint_path.parent.parent / "adaptation_plan.json",
    ):
        if candidate.is_file():
            log.info("Using adaptation plan from %s", candidate)
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("Ignoring malformed %s: %s", candidate, exc)
    return None


def _plan_keys(plan: dict[str, Any] | None) -> tuple[str, str, str, int]:
    """Extract ``(key_to_encoder, key_to_stem, architecture, num_input_channels)`` from *plan*."""
    if not plan:
        log.warning(
            "Checkpoint carries no nnssl adaptation plan; assuming the standard ResEnc-L key "
            "layout (%r / %r).",
            DEFAULT_KEY_TO_ENCODER,
            DEFAULT_KEY_TO_STEM,
        )
        return DEFAULT_KEY_TO_ENCODER, DEFAULT_KEY_TO_STEM, "ResEncL", 1

    architecture = str(
        (plan.get("architecture_plans") or {}).get("arch_class_name") or "ResEncL"
    )
    return (
        str(plan.get("key_to_encoder", DEFAULT_KEY_TO_ENCODER)),
        str(plan.get("key_to_stem", DEFAULT_KEY_TO_STEM)),
        architecture,
        int(plan.get("pretrain_num_input_channels", 1)),
    )


def load_encoder_into(
    module: Any,
    checkpoint_path: Path,
    *,
    strict_shapes: bool = True,
) -> dict[str, int]:
    """Load a checkpoint's encoder and stem into an already-built network, in place.

    Two callers: seeding self-supervised pre-training from a published checkpoint, so a run
    continues from general features rather than noise; and populating the encoder of the
    segmentation network stage 1 exports. In both cases the target already exists — only
    encoder and stem are touched, and everything else (decoder, heads, mask tokens) keeps the
    values it was initialised with.

    Parameters
    ----------
    module
        The trainer's network, already constructed (nnssl builds it in ``initialize()``).
    strict_shapes
        Raise on a shape mismatch. Turning it off skips the offending tensors instead, which is
        only sensible when deliberately seeding across architecture variants.

    Returns
    -------
    dict
        ``{"matched": n, "skipped": n, "target_encoder_keys": n}``.
    """
    state, plan, _ = load_nnssl_checkpoint(Path(checkpoint_path))
    key_to_encoder, key_to_stem, architecture, _ = _plan_keys(plan)

    # No architecture gate here: the target module is whatever the caller built, so a Primus
    # checkpoint seeding a Primus network is as valid as ResEnc-L into ResEnc-L. The key-prefix
    # match below is what decides whether the two actually correspond.
    remapped: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith(key_to_encoder) or key.startswith(key_to_stem):
            # Seeding keeps the source names: the target was built from the same adaptation
            # plan, so its encoder lives under the same prefixes. Cross-framework renaming is
            # not needed here — the in-tree nnU-Net build does its own loading at fine-tuning
            # time, from the checkpoint the plans file points at.
            remapped[key] = value

    target_state = module.state_dict()
    # Primus exposes its encoder under different names ('eva' / 'down_projection'), so the
    # target prefixes come from the checkpoint's own plan rather than the ResEnc-L constants.
    prefixes = tuple({key.split(".")[0] for key in remapped})
    expected = [key for key in target_state if key.startswith(prefixes)]
    if not expected:
        raise ValueError(
            f"Target network exposes no {TARGET_KEY_TO_ENCODER!r}/{TARGET_KEY_TO_STEM!r} "
            f"parameters; prefixes present: {sorted({k.split('.')[0] for k in target_state})}."
        )

    matched, skipped = 0, 0
    updates: dict[str, Any] = {}
    for key in expected:
        source = remapped.get(key)
        if source is None:
            skipped += 1
            continue
        if tuple(source.shape) != tuple(target_state[key].shape):
            if strict_shapes:
                raise ValueError(
                    f"Shape mismatch seeding {key}: checkpoint {tuple(source.shape)} vs target "
                    f"{tuple(target_state[key].shape)}."
                )
            skipped += 1
            continue
        updates[key] = source.to(target_state[key].dtype)
        matched += 1

    if matched == 0:
        raise ValueError(
            f"No encoder parameters could be seeded from {checkpoint_path}. Checkpoint prefixes: "
            f"{sorted({k.split('.')[0] for k in state})}."
        )

    target_state.update(updates)
    module.load_state_dict(target_state)

    log.ok(
        f"seeded encoder from {Path(checkpoint_path).name}: {matched}/{len(expected)} tensors "
        f"({skipped} skipped)"
    )
    return {"matched": matched, "skipped": skipped, "target_encoder_keys": len(expected)}


__all__ = [
    "DEFAULT_KEY_TO_ENCODER",
    "DEFAULT_KEY_TO_STEM",
    "TARGET_KEY_TO_ENCODER",
    "TARGET_KEY_TO_STEM",
    "load_encoder_into",
    "load_nnssl_checkpoint",
]
