"""One selectable post-processing pipeline, shared by stages 3, 4, 5 and 6.

The steps themselves live in :mod:`nvitk.segmentation.vessel_postprocess` and
:mod:`nvitk.segmentation.vessel_topology`. What this module adds is a single way to *choose*
them, so the same selection can be evaluated in stage 3, applied in stage 4, baked into the
submission container in stage 5 and used to clean pseudo-labels in stage 6 -- and so the choice
that was measured is provably the choice that ships.

Selection syntax
----------------
``--postprocess`` takes a comma-separated list of step names, or one of two words::

    none                      nothing at all; the raw argmax
    islands                   the default: drop components under --min-volume-mm3
    islands,bridge            islands, then reconnect same-class gaps
    all                       every step, including the ones that are off by default

Order is **not** taken from the list. The steps are ordered by
:func:`~nvitk.segmentation.vessel_topology.repair_topology`'s own reasoning -- islands before
anything that counts components, bridging before anything that judges a fragment by what it
touches -- and writing them in a different order changes nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain import labels as lbl

log = Logger()

#: Step names, in the order they are applied. ``islands`` and ``largest`` are component
#: clean-up; the rest are topology repair.
STEPS: tuple[str, ...] = ("islands", "largest", "bridge", "adjacency", "lateral")

#: What ``--postprocess`` does when it is not given: nnU-Net's argmax plus speckle removal.
#: The topology steps stay opt-in because each can trade one metric for another.
DEFAULT_STEPS: tuple[str, ...] = ("islands",)

#: Written into the stage 5 build context so the container applies what was measured.
CONTAINER_CONFIG_NAME: str = "postprocess.json"


@dataclass(frozen=True)
class PostProcessSpec:
    """A chosen set of steps and their parameters."""

    steps: tuple[str, ...] = DEFAULT_STEPS
    min_volume_mm3: float | None = 5.0
    bridge_gaps_mm: float | None = 3.0
    bridge_radius: int = 1
    close_radius: int = 0
    max_fragment_fraction: float = 0.25

    @property
    def enabled(self) -> bool:
        """Whether anything at all will be applied."""
        return bool(self.steps)

    def has(self, step: str) -> bool:
        """Whether *step* is selected."""
        return step in self.steps

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for provenance and for the container."""
        return {**asdict(self), "steps": list(self.steps)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PostProcessSpec:
        """Rebuild from :meth:`as_dict`, ignoring unknown keys."""
        fields = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in dict(payload).items() if k in fields}
        data["steps"] = tuple(data.get("steps") or ())
        return cls(**data)

    def describe(self) -> str:
        """One line naming the steps and the parameters that matter to them."""
        if not self.steps:
            return "post-processing: none (raw argmax)"
        parts = []
        for step in self.steps:
            if step == "islands":
                parts.append(f"islands(<{self.min_volume_mm3} mm3)")
            elif step == "bridge":
                parts.append(
                    f"bridge(<={self.bridge_gaps_mm} mm, r={self.bridge_radius}"
                    + (f", close={self.close_radius}" if self.close_radius else "")
                    + ")"
                )
            elif step in ("adjacency", "lateral"):
                parts.append(f"{step}(<={self.max_fragment_fraction:.0%} of host)")
            else:
                parts.append(step)
        return "post-processing: " + " -> ".join(parts)


def parse_steps(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a ``--postprocess`` value into an ordered, de-duplicated step tuple.

    Raises
    ------
    ValueError
        On an unknown step name, listing the valid ones. Silently dropping a typo would ship a
        container doing less than the run that was measured, which is the failure this whole
        module exists to prevent.
    """
    if spec is None:
        return DEFAULT_STEPS
    tokens = (
        [t.strip().lower() for t in str(spec).split(",")]
        if isinstance(spec, str) else [str(t).strip().lower() for t in spec]
    )
    tokens = [t for t in tokens if t]
    if not tokens or tokens == ["none"]:
        return ()
    if tokens == ["all"]:
        return STEPS
    unknown = [t for t in tokens if t not in STEPS]
    if unknown:
        raise ValueError(
            f"Unknown post-processing step(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(STEPS)}, or 'none'/'all'."
        )
    # Ordered by STEPS, not by how the user typed them: the order is a correctness property.
    return tuple(step for step in STEPS if step in tokens)


def apply_postprocess(
    labelmap: Any,
    *,
    label_set: str,
    spec: PostProcessSpec,
    spacing: Sequence[float] | None = None,
    affine: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply *spec* to one label map; returns ``(result, report)``.

    The single entry point every stage goes through, so none of them can drift into a different
    order or a different default.
    """
    from nvitk.segmentation.vessel_postprocess import postprocess_labelmap
    from nvitk.segmentation.vessel_topology import repair_topology

    if not spec.enabled:
        return labelmap, {"steps": [], "applied": False}

    result = labelmap
    if spec.has("islands") or spec.has("largest"):
        result = postprocess_labelmap(
            result,
            spacing=spacing,
            min_volume_mm3=spec.min_volume_mm3 if spec.has("islands") else None,
            largest_only=spec.has("largest"),
        )

    report: dict[str, Any] = {"steps": list(spec.steps), "applied": True}
    if spec.has("bridge") or spec.has("adjacency") or spec.has("lateral"):
        result, repair = repair_topology(
            result,
            spacing=spacing,
            affine=affine,
            valid_neighbours=lbl.valid_neighbours(label_set) if spec.has("adjacency") else None,
            lateral_pairs=lbl.lateral_pairs(label_set) if spec.has("lateral") else None,
            bridge_gaps_mm=spec.bridge_gaps_mm if spec.has("bridge") else None,
            bridge_radius=spec.bridge_radius,
            close_radius=spec.close_radius,
            max_fragment_fraction=spec.max_fragment_fraction,
        )
        report["repair"] = repair.as_dict()
    return result, report


def spec_from_options(**options: Any) -> PostProcessSpec:
    """Build a spec from CLI option values, tolerating absent keys."""
    return PostProcessSpec(
        steps=parse_steps(options.get("postprocess")),
        min_volume_mm3=options.get("min_volume_mm3", 5.0),
        bridge_gaps_mm=options.get("repair_gaps_mm") or 3.0,
        bridge_radius=int(options.get("repair_bridge_radius", 1) or 1),
        close_radius=int(options.get("repair_close_radius", 0) or 0),
        max_fragment_fraction=float(options.get("repair_fragment_fraction", 0.25) or 0.25),
    )


def write_container_config(context_dir: Any, spec: PostProcessSpec) -> Any:
    """Write *spec* into a stage 5 build context; returns the path."""
    from pathlib import Path

    path = Path(context_dir) / CONTAINER_CONFIG_NAME
    path.write_text(json.dumps(spec.as_dict(), indent=2) + "\n", encoding="utf-8")
    log.info("Container will apply -> %s", spec.describe())
    return path


def read_container_config(context_dir: Any) -> PostProcessSpec:
    """Read the spec a container was built with, falling back to the default."""
    from pathlib import Path

    path = Path(context_dir) / CONTAINER_CONFIG_NAME
    if not path.is_file():
        return PostProcessSpec()
    try:
        return PostProcessSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        log.warning("%s is unreadable; falling back to the default post-processing.", path)
        return PostProcessSpec()


__all__ = [
    "CONTAINER_CONFIG_NAME",
    "DEFAULT_STEPS",
    "STEPS",
    "PostProcessSpec",
    "apply_postprocess",
    "parse_steps",
    "read_container_config",
    "spec_from_options",
    "write_container_config",
]
