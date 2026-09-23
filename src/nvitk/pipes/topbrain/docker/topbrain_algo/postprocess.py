"""Connected-component clean-up, applied to the predicted label map.

Scope
-------------------------------------------------
``islands`` removes speckle: components below ``min_volume_mm3`` in physical volume, judged per
class so a small artery is never compared against a large one. ``largest`` is stricter — one
component per class — and is off by default because several TA36 classes legitimately appear as
more than one component in a given field of view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

# Steps this slim runtime can apply, in the order they are applied.
SUPPORTED_STEPS: tuple[str, ...] = ("islands", "largest")

# Full 26-neighbourhood. Vessels run diagonally through the voxel grid far more often than they
# run along an axis, so face-only connectivity fragments a perfectly continuous artery.
_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


@dataclass
class PostProcessReport:
    """What post-processing actually did, for the submission log."""

    steps: list[str] = field(default_factory=list)
    removed_components: int = 0
    removed_voxels: int = 0
    per_class: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "steps": list(self.steps),
            "removed_components": int(self.removed_components),
            "removed_voxels": int(self.removed_voxels),
            "per_class": {int(k): int(v) for k, v in sorted(self.per_class.items())},
        }

    def describe(self) -> str:
        """One line for the log."""
        if not self.steps:
            return "post-processing: none (raw argmax)"
        return (
            f"post-processing [{', '.join(self.steps)}]: removed "
            f"{self.removed_components} component(s), {self.removed_voxels} voxel(s)"
        )


def read_config(directory: Path) -> tuple[tuple[str, ...], float]:
    """Read ``postprocess.json``; returns ``(steps, min_volume_mm3)``.

    Raises
    ------
    RuntimeError
        When the file selects a step this runtime cannot apply. stage 5 checks the same thing at
        build time, so reaching this means the image was assembled some other way — and running
        on anyway would apply a different pipeline from the one that was measured.
    """
    path = Path(directory) / "postprocess.json"
    if not path.is_file():
        return ("islands",), 5.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = tuple(payload.get("steps") or ())
    unsupported = [s for s in steps if s not in SUPPORTED_STEPS]
    if unsupported:
        raise RuntimeError(
            f"{path} selects {', '.join(unsupported)}, which the container does not implement "
            f"(it has: {', '.join(SUPPORTED_STEPS)}). Rebuild with a --postprocess the image "
            f"can honour rather than shipping one that quietly does less."
        )
    minimum = payload.get("min_volume_mm3")
    return steps, 5.0 if minimum is None else float(minimum)


def apply(labelmap: np.ndarray, *, steps: Sequence[str],
          spacing: Sequence[float] | None = None,
          min_volume_mm3: float = 5.0) -> tuple[np.ndarray, PostProcessReport]:
    """Apply *steps* to one label map; returns ``(result, report)``.

    *spacing* is ``(z, y, x)`` in millimetres, matching the array axis order. Without it the
    threshold is read in voxels, which is only meaningful for isotropic data.
    """
    report = PostProcessReport(steps=list(steps))
    if not steps:
        return labelmap, report

    result = np.asarray(labelmap).copy()
    voxel_mm3 = float(np.prod(spacing)) if spacing is not None else 1.0
    min_voxels = max(1, int(round(float(min_volume_mm3) / voxel_mm3))) if "islands" in steps else 0

    for value in (int(v) for v in np.unique(result) if v != 0):
        mask = result == value
        components, count = ndi.label(mask, structure=_STRUCTURE)
        if count <= 1 and "largest" not in steps:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0  # background of this class's own labelling

        drop = np.zeros(sizes.shape, dtype=bool)
        if "islands" in steps:
            drop |= sizes < min_voxels
        if "largest" in steps:
            drop |= np.arange(sizes.size) != int(sizes.argmax())
        drop[0] = False

        if not drop.any():
            continue
        removed = drop[components] & mask
        removed_voxels = int(removed.sum())
        if removed_voxels == 0:
            continue
        result[removed] = 0
        report.removed_components += int(drop.sum())
        report.removed_voxels += removed_voxels
        report.per_class[value] = removed_voxels

    return result, report


__all__ = ["SUPPORTED_STEPS", "PostProcessReport", "apply", "read_config"]
