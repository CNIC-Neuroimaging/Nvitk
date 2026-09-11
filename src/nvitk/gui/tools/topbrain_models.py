"""Trained ToPBrain models as GUI dropdown entries.

Stage 2 records what it trained in a provenance marker, and inference needs those
facts back: the plans identifier embeds a spacing ``preprocess_like_nnssl`` only
settles at run time, and the trainer family follows from the pre-trained
checkpoint. None of it can be reconstructed from a flag, so the dropdown lists
what is actually on disk rather than a hardcoded set.

Each entry carries enough provenance to tell two models apart — how many classes,
which loss, how many folds — because "ta36" and "ta36_ct" say nothing about which
one is worth running, and the alternative is an operator opening a terminal to
find out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Separates the ``--model`` selector from the description in a dropdown entry.
#: Wide enough that it cannot occur inside a label-set name.
SELECTOR_SEPARATOR = "  —  "

#: Offered when no provenance marker is reachable — a cluster mount that is down,
#: or a workstation where stage 2 has never run. A dropdown that is empty until
#: the mount returns is worse than one that lets the run fail naming the model.
FALLBACK_CHOICES: tuple[str, ...] = ("ta36", "ta36_ct", "ta36_mr", "binary")

#: ``auto`` detects CT vs MR from the intensities; the named ones force it. The
#: last entry skips harmonisation, for inputs already in nnU-Net raw form —
#: predicting on unharmonised scanner intensities is silently wrong rather than an
#: error, so it has to be a deliberate choice.
MODALITY_CHOICES: tuple[str, ...] = ("auto", "mr", "ct", "already harmonised")

#: The modality entry meaning "do not harmonise".
MODALITY_NONE = "already harmonised"


#: Roots an SGE job exports after binding them into the container, keyed by the
#: layout field each one overrides. Set by :func:`~nvitk.gui.sge.job.tool_models_binding`.
CONTAINER_ROOT_ENV: dict[str, str] = {
    "results_root": "TOPBRAIN_RESULTS_ROOT",
    "nnunet_results": "TOPBRAIN_NNUNET_RESULTS",
    "nnunet_raw": "TOPBRAIN_NNUNET_RAW",
    "nnunet_preprocessed": "TOPBRAIN_NNUNET_PREPROCESSED",
}


def container_overrides() -> dict[str, Path]:
    """Roots an enclosing SGE container has bound, as layout overrides."""
    import os

    out: dict[str, Path] = {}
    for field, variable in CONTAINER_ROOT_ENV.items():
        value = os.environ.get(variable, "").strip()
        if value:
            out[field] = Path(value)
    return out


def host_layout() -> tuple[Any, str]:
    """``(paths, origin)`` for the roots this host should read.

    Inside an SGE container the configured cluster paths are host paths that were
    never mounted, so the bind target exported by the job wins. ``layout_local``
    rather than ``layout_cluster`` because only the local half treats a passed
    root as authoritative — the cluster half deliberately lets config override a
    flag, which here would hand back the unmounted path the override exists to
    replace.
    """
    from nvitk.pipes.topbrain.util.paths import layout_auto, layout_local

    overrides = container_overrides()
    if overrides:
        return layout_local(**overrides), "container"
    return layout_auto()


def discovered_models() -> list[Any]:
    """Every trained model this host can see, newest first; empty when none are."""
    try:
        from nvitk.pipes.topbrain.util.models import discover_models

        paths, _origin = host_layout()
        return list(discover_models(paths.results_root))
    except Exception:
        return []


def describe(model: Any) -> str:
    """One dropdown entry: the selector, then what distinguishes this model."""
    classes = int(getattr(model, "num_output_channels", 0) or 0)
    # Channels count background; what an operator cares about is vessel classes.
    kind = "binary vessel" if classes <= 2 else f"{max(classes - 1, 0)} vessel classes"
    folds = len(tuple(getattr(model, "folds", ()) or ()))
    created = str(getattr(model, "created", ""))[:10]
    parts = [kind, str(getattr(model, "loss", "") or "?"), f"{folds} fold(s)"]
    if created:
        parts.append(created)
    return f"{getattr(model, 'label_set', '?')}{SELECTOR_SEPARATOR}{' · '.join(parts)}"


def model_choices() -> tuple[str, ...]:
    """Dropdown entries for every reachable model, or the fallback when none are."""
    found = discovered_models()
    return tuple(describe(m) for m in found) if found else FALLBACK_CHOICES


def selector_from_choice(choice: Any) -> str:
    """The ``--model`` selector inside a dropdown entry."""
    text = str(choice or "").strip()
    if SELECTOR_SEPARATOR in text:
        return text.split(SELECTOR_SEPARATOR, 1)[0].strip()
    return text


def modality_argument(choice: Any) -> str | None:
    """The ``--modality`` value for a dropdown entry; ``None`` skips harmonisation."""
    text = str(choice or "auto").strip().lower()
    if not text or text == MODALITY_NONE:
        return None
    return text if text in ("auto", "mr", "ct") else None


def model_listing() -> str:
    """The full provenance listing, for the log when a run starts."""
    try:
        from nvitk.pipes.topbrain.util.models import describe_models

        paths, origin = host_layout()
        return (
            f"ToPBrain models ({origin} roots, {paths.results_root}):\n"
            + describe_models(paths.results_root, paths.nnunet_results)
        )
    except Exception as exc:  # noqa: BLE001 — advisory only
        return f"ToPBrain model listing unavailable: {exc}"


__all__ = [
    "CONTAINER_ROOT_ENV",
    "FALLBACK_CHOICES",
    "MODALITY_CHOICES",
    "MODALITY_NONE",
    "SELECTOR_SEPARATOR",
    "container_overrides",
    "describe",
    "discovered_models",
    "host_layout",
    "modality_argument",
    "model_choices",
    "model_listing",
    "selector_from_choice",
]
