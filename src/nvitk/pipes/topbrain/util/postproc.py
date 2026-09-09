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
#: clean-up; ``flood`` grows the labels along image evidence; the rest are topology repair.
#:
#: ``flood`` sits before ``bridge``: it closes gaps using the intensities, which is better
#: evidence than the tube ``bridge`` draws between fragments, and whatever it adds is then still
#: subject to the adjacency and laterality checks.
STEPS: tuple[str, ...] = ("islands", "largest", "flood", "bridge", "adjacency", "lateral")

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
    # ---- flood: a pass-through to blood_flood's own parameters, nothing more ----
    flood_hyst_low_factor: float = 3.0
    flood_hyst_high_factor: float = 0.5
    flood_thin_percentile: float | None = 55.0
    flood_thicken_iter: int = 0
    flood_connectivity: int = 3

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
            elif step == "flood":
                parts.append(
                    f"flood(low={self.flood_hyst_low_factor}, "
                    f"thin={self.flood_thin_percentile})"
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
    intensity: Any = None,
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

    if spec.has("flood"):
        if intensity is None:
            # Not an error: every other step works on the mask alone, so a caller that cannot
            # supply the image should still get the rest rather than nothing.
            log.warning(
                "Post-processing step 'flood' needs the source image and none was given; "
                "skipping it. The other steps still ran."
            )
            report["flood"] = {"skipped": "no intensity image"}
        else:
            result, report["flood"] = _flood_step(
                result, intensity, spec=spec, label_set=label_set
            )

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


def _flood_step(
    labelmap: Any, intensity: Any, *, spec: PostProcessSpec, label_set: str
) -> tuple[Any, dict[str, Any]]:
    """Grow the labels along vessel evidence with :func:`blood_flood`.

    A plain pass-through: every parameter here is one blood_flood already had, so the step does
    exactly what the GUI tool does and nothing else. The existing labelling is the marker set,
    so the watershed is seeded from what is already there.
    """
    import numpy as np

    from nvitk.core.array import as_backend_array, to_numpy
    from nvitk.segmentation.blood_flood import blood_flood

    labels_in = to_numpy(as_backend_array(
        labelmap.data if hasattr(labelmap, "data") else labelmap
    ))
    image = to_numpy(as_backend_array(
        intensity.data if hasattr(intensity, "data") else intensity
    ))
    if labels_in.shape != image.shape:
        log.warning(
            "flood: the image is %s and the mask is %s; skipping the step rather than "
            "guessing an alignment.", image.shape, labels_in.shape,
        )
        return labelmap, {"skipped": "shape mismatch"}

    outcome = blood_flood(
        image,
        labels_in,
        hyst_low_factor=spec.flood_hyst_low_factor,
        hyst_high_factor=spec.flood_hyst_high_factor,
        thin_vesselness_percentile=spec.flood_thin_percentile,
        thicken_iter=spec.flood_thicken_iter,
        connectivity=spec.flood_connectivity,
    )
    # Where the flood assigned nothing, the input stands. A post-processing pass must never be
    # able to erase the model's own output.
    merged = np.where(outcome.labels != 0, outcome.labels, labels_in).astype(labels_in.dtype)
    added = int((merged != 0).sum() - (labels_in != 0).sum())
    log.info("flood: %+d voxel(s).", added)
    report = {
        "added_voxels": added,
        "vesselness_mode": outcome.vesselness_mode,
        "tree_voxels": outcome.info.get("n_tree_voxels"),
        "tree_marker_cc": outcome.info.get("tree_marker_cc"),
    }
    source = labelmap if hasattr(labelmap, "with_data") else None
    return (source.with_data(merged) if source is not None else merged), report


def spec_from_options(**options: Any) -> PostProcessSpec:
    """Build a spec from CLI option values, tolerating absent keys."""
    return PostProcessSpec(
        steps=parse_steps(options.get("postprocess")),
        flood_hyst_low_factor=float(options.get("flood_hyst_low_factor", 3.0) or 3.0),
        flood_hyst_high_factor=float(options.get("flood_hyst_high_factor", 0.5) or 0.5),
        flood_thin_percentile=options.get("flood_thin_percentile", 55.0),
        flood_thicken_iter=int(options.get("flood_thicken_iter", 0) or 0),
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
