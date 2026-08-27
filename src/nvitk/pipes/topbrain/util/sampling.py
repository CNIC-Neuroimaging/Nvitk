"""
Rare-class-aware patch sampling for the ToPBrain fine-tuning stage.

Description
-----------
nnU-Net's sampler is already fairer than it first looks: when a patch is one of the 33 % forced
to contain foreground, it picks a class **uniformly among the classes present in that case** and
centres the patch on a voxel of it. So a case carrying a Pcom does give the Pcom a fair share of
that case's patches.

What it does *not* do is notice that the Pcom is only in a handful of cases at all. The chance a
given training patch is centred on a rare class is roughly::

    P(case carries it) x 0.33 x 1 / (classes present in that case)

With 40 training volumes, a class present in three of them and ~28 classes present per case,
that is well under one patch in a thousand — the network sees it a few times per epoch, against
tens of thousands of views of the ICA.

This module supplies the two weightings that close that gap, leaving nnU-Net's machinery to do
the actual sampling:

``case``
    Per-case probabilities for ``nnUNetDataLoader``'s ``sampling_probabilities``, so volumes
    carrying rare classes come up more often.
``class``
    Per-class weights for the choice nnU-Net makes uniformly, so within such a case the rare
    class is what the patch actually centres on.

Both are **inverse-frequency with a tunable exponent**, normalised. The exponent matters: at
``1.0`` a class in 3 of 40 cases is weighted ~13x a universal one, which trains it at the cost
of everything else. :data:`DEFAULT_TEMPERATURE` is deliberately below 1 — enough to stop the
rare classes being invisible, not enough to distort the classes the score mostly rides on.

This is a *sampling* change, not a loss change. It alters which patches the network sees, not
what it is asked to optimise on them, and the two compose: it is worth running against the same
loss rather than instead of a loss experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from nvitk.core.logger import Logger

log = Logger()

#: Environment variable carrying the sampling specification into the training subprocess.
#: nnU-Net's CLI can only pass a trainer *name*, so this travels out of band exactly as the
#: loss specification does (see :mod:`nvitk.pipes.topbrain.util.losses`).
SAMPLING_SPEC_ENV: str = "TOPBRAIN_SAMPLING_SPEC"

#: Written by stage 0 beside the dataset: case id → the foreground labels in its mask.
CASE_CLASSES_FILE: str = "case_classes.json"

#: Selectable strategies. ``default`` leaves nnU-Net untouched, and is the control this has to
#: be measured against.
SAMPLING_MODES: tuple[str, ...] = ("default", "rare_aware")

#: Inverse-frequency exponent. Below 1 on purpose — see the module docstring.
DEFAULT_TEMPERATURE: float = 0.5

#: Ceiling on how much more often any one case or class may be drawn than the mean. Without it
#: a class present in a single volume would collapse training onto that volume.
DEFAULT_MAX_WEIGHT: float = 10.0

#: Fraction of patches forced to contain foreground when rare-aware sampling is on. Higher than
#: nnU-Net's 0.33 because with 0.3 % foreground the unforced patches are almost all background,
#: and they are what the rare classes are competing against.
DEFAULT_OVERSAMPLE_PERCENT: float = 0.5


def read_case_classes(dataset_dir: Path) -> dict[str, list[int]]:
    """Read stage 0's ``case_classes.json``; empty when it is absent.

    Absent means the dataset predates this sidecar. That is a reason to fall back to nnU-Net's
    own sampling with a warning, not to fail a training run — the caller decides.
    """
    path = Path(dataset_dir) / CASE_CLASSES_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s (%s); falling back to default sampling.", path, exc)
        return {}
    return {str(k): [int(v) for v in values] for k, values in raw.items()}


def class_frequencies(
    case_classes: Mapping[str, Iterable[int]],
    *,
    identifiers: Sequence[str] | None = None,
) -> dict[int, int]:
    """Label → number of cases carrying it.

    Parameters
    ----------
    identifiers
        Restrict the count to these cases. Pass the **training** split: frequencies computed
        over the whole cohort would leak the validation half's class distribution into how the
        model is trained.
    """
    keys = list(case_classes) if identifiers is None else [
        cid for cid in identifiers if cid in case_classes
    ]
    frequencies: dict[int, int] = {}
    for case_id in keys:
        for value in case_classes[case_id]:
            if int(value) != 0:
                frequencies[int(value)] = frequencies.get(int(value), 0) + 1
    return dict(sorted(frequencies.items()))


def _normalise(weights: Mapping[int, float], *, max_weight: float) -> dict[int, float]:
    """Scale *weights* to mean 1 and clip to *max_weight*, then rescale to mean 1 again."""
    if not weights:
        return {}
    mean = sum(weights.values()) / len(weights)
    scaled = {k: (v / mean if mean else 1.0) for k, v in weights.items()}
    clipped = {k: min(v, float(max_weight)) for k, v in scaled.items()}
    mean = sum(clipped.values()) / len(clipped)
    return {k: (v / mean if mean else 1.0) for k, v in clipped.items()}


def class_weights(
    frequencies: Mapping[int, int],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> dict[int, float]:
    """Inverse-frequency weight per class, mean-normalised to 1.

    A weight of 1 means "as often as nnU-Net would have". Classes absent from *frequencies*
    are absent from the result: they cannot be sampled, so they need no weight.

    Parameters
    ----------
    temperature
        Exponent on the inverse frequency. ``0`` reproduces uniform (nnU-Net's own behaviour),
        ``1`` is full inverse-frequency.
    max_weight
        Clip, so a single-case class cannot dominate.
    """
    if not frequencies:
        return {}
    raw = {
        int(label): (1.0 / max(int(count), 1)) ** float(temperature)
        for label, count in frequencies.items()
    }
    return _normalise(raw, max_weight=max_weight)


def case_weights(
    case_classes: Mapping[str, Iterable[int]],
    weights: Mapping[int, float],
    *,
    identifiers: Sequence[str],
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> list[float]:
    """Sampling probability per case, in the order of *identifiers*; sums to 1.

    A case is scored by the **rarest** class it carries, not by the sum over its classes. Every
    case here carries the twenty-odd common vessels, so a sum is nearly constant across the
    cohort and would wash the signal out; the maximum asks the question that matters — is this
    volume one of the few that shows the class we are short of?

    Cases with no recorded classes get the mean weight rather than zero: an unknown case should
    be sampled normally, not excluded from training.
    """
    if not weights:
        return [1.0 / len(identifiers)] * len(identifiers)
    scores: dict[str, float] = {}
    for case_id in identifiers:
        present = [int(v) for v in case_classes.get(case_id, ()) if int(v) != 0]
        candidates = [weights[v] for v in present if v in weights]
        scores[case_id] = max(candidates) if candidates else 1.0
    normalised = _normalise(
        {index: scores[cid] for index, cid in enumerate(identifiers)}, max_weight=max_weight
    )
    total = sum(normalised.values())
    return [normalised[index] / total for index in range(len(identifiers))]


def sampling_spec_payload(
    mode: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    oversample_percent: float | None = None,
) -> str:
    """JSON payload for :data:`SAMPLING_SPEC_ENV`.

    Raises
    ------
    ValueError
        For an unknown *mode*. Silently falling back to ``default`` would report a rare-aware
        run in the provenance while training an ordinary one.
    """
    mode = str(mode).strip().lower()
    if mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling mode {mode!r}; expected {', '.join(SAMPLING_MODES)}.")
    return json.dumps({
        "mode": mode,
        "temperature": float(temperature),
        "max_weight": float(max_weight),
        "oversample_percent": (
            DEFAULT_OVERSAMPLE_PERCENT if oversample_percent is None
            else float(oversample_percent)
        ),
    })


def parse_sampling_spec(raw: str | None) -> dict:
    """Parse :data:`SAMPLING_SPEC_ENV`; the default strategy when unset or malformed."""
    fallback = {
        "mode": "default", "temperature": DEFAULT_TEMPERATURE,
        "max_weight": DEFAULT_MAX_WEIGHT, "oversample_percent": DEFAULT_OVERSAMPLE_PERCENT,
    }
    if not raw:
        return fallback
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("%s is not valid JSON (%s); using default sampling.", SAMPLING_SPEC_ENV, exc)
        return fallback
    if str(spec.get("mode")) not in SAMPLING_MODES:
        log.warning("Unknown sampling mode %r; using default sampling.", spec.get("mode"))
        return fallback
    return {**fallback, **spec}


def describe_weights(
    frequencies: Mapping[int, int],
    weights: Mapping[int, float],
    *,
    label_map: Mapping[int, str] | None = None,
    limit: int = 10,
) -> str:
    """One-line-per-class summary of the *limit* rarest classes, for the training log."""
    rarest = sorted(frequencies, key=lambda v: (frequencies[v], v))[:limit]
    lines = [f"{'lbl':>4} {'class':<16} {'cases':>5} {'weight':>7}"]
    for value in rarest:
        name = (label_map or {}).get(value, "?")
        lines.append(
            f"{value:>4} {str(name)[:16]:<16} {frequencies[value]:>5} {weights.get(value, 1.0):>7.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "CASE_CLASSES_FILE",
    "DEFAULT_MAX_WEIGHT",
    "DEFAULT_OVERSAMPLE_PERCENT",
    "DEFAULT_TEMPERATURE",
    "SAMPLING_MODES",
    "SAMPLING_SPEC_ENV",
    "case_weights",
    "class_frequencies",
    "class_weights",
    "describe_weights",
    "parse_sampling_spec",
    "read_case_classes",
    "sampling_spec_payload",
]
