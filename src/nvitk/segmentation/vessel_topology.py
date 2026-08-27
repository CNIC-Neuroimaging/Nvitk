"""
Graph-based topology repair for multi-class vessel segmentations.

Description
-----------
:mod:`nvitk.segmentation.vessel_postprocess` removes what should not be there. This module
repairs what *is* there but is wrong in a way a voxelwise loss cannot see, and that vessel
benchmarks score directly:

:func:`bridge_class_gaps`
    A signal dropout splits one vessel into two components. Costs almost no Dice, doubles the
    β0 error and cuts the centerline. Bridged per class along the shortest inter-component path,
    with a hard ceiling in **millimetres** so a genuine anatomical discontinuity is never welded
    shut.
:func:`resolve_invalid_adjacencies`
    Two lateral neighbours bleed into each other and a strip of one class ends up embedded in
    the other, touching labels the anatomy does not permit. Reassigned to the label it is most
    in contact with, which is nearly always what it actually was.
:func:`enforce_lateral_consistency`
    A component of ``R-M2`` sitting in the left hemisphere. Relabelled to its mirror class.

All three are **conservative by construction**: each acts only on components that are small
relative to their class's main body, because the failure mode being corrected is a fragment, and
a rule that can move the main body of a vessel is a rule that can lose it entirely.

Left / right without guessing
-----------------------------
Which array axis is lateral comes from the **affine** (the axis whose world direction is most
left-right). Which *end* of it is the patient's right comes from the **labels themselves**: the
mean lateral coordinate of the ``R-`` classes is compared with the ``L-`` classes and the
convention read off. Nothing assumes LPS, RAS, or an axis order, and when the two families do
not separate cleanly the step declines to act rather than mirroring labels on a guess — a
left-right flip is the one post-processing error that is worse than doing nothing.

Array / axis conventions
------------------------
Shape-agnostic 3D integer label maps. Distances are millimetres, converted with an explicit
``spacing``. Accepts and returns :class:`~nvitk.types.Image` or a bare array, preserving
geometry.

I/O and arrays: backend ``np`` after ``setup(globals())``; the component and distance work is
CPU-only (SciPy / scikit-image) and runs inside ``with using("cpu")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.segmentation.vessel_postprocess import (
    DEFAULT_CONNECTIVITY,
    _unwrap,
    voxel_volume_mm3,
)
from nvitk.types import Image

setup(globals())

log = Logger()

#: Largest gap that may be bridged, in millimetres. Intracranial vessels are 0.3-0.6 mm across
#: and dropouts are short; beyond a few millimetres a break is far more likely to be a genuine
#: occlusion or an edge of the field of view than a segmentation failure.
DEFAULT_MAX_GAP_MM: float = 3.0

#: Radius of the bridging tube, in voxels. One is a hairline — enough to restore connectivity
#: for the β0 and centerline metrics without inventing calibre.
DEFAULT_BRIDGE_RADIUS: int = 1

#: A component may be reassigned or mirrored only if it holds at most this fraction of its
#: class's voxels. Above it, the "fragment" premise does not hold.
DEFAULT_FRAGMENT_FRACTION: float = 0.25


@dataclass
class RepairReport:
    """What each repair step changed, for the stage's provenance."""

    bridged_voxels: int = 0
    bridged_classes: list[int] = field(default_factory=list)
    reassigned_components: int = 0
    reassigned_voxels: int = 0
    mirrored_components: int = 0
    mirrored_voxels: int = 0
    lateral_axis: int | None = None
    lateral_right_is_low: bool | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "bridged_voxels": self.bridged_voxels,
            "bridged_classes": list(self.bridged_classes),
            "reassigned_components": self.reassigned_components,
            "reassigned_voxels": self.reassigned_voxels,
            "mirrored_components": self.mirrored_components,
            "mirrored_voxels": self.mirrored_voxels,
            "lateral_axis": self.lateral_axis,
            "lateral_right_is_low": self.lateral_right_is_low,
            "notes": list(self.notes),
        }


def _labels_present(data: Any, labels: Sequence[int] | None) -> list[int]:
    """Foreground labels to operate on."""
    if labels is not None:
        return [int(v) for v in labels]
    return [int(v) for v in to_numpy(np.unique(data)) if int(v) != 0]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Gap bridging
# ──────────────────────────────────────────────────────────────────────────────


def bridge_class_gaps(
    labelmap: Any,
    *,
    labels: Sequence[int] | None = None,
    spacing: Sequence[float] | None = None,
    max_gap_mm: float = DEFAULT_MAX_GAP_MM,
    bridge_radius: int = DEFAULT_BRIDGE_RADIUS,
    close_radius: int = 0,
    report: RepairReport | None = None,
) -> Any:
    """Reconnect each class's fragments across gaps shorter than *max_gap_mm*.

    Delegates the geometry to :func:`~nvitk.morphology.mst_bridge.fill_multilabel_gaps_mst`,
    which builds a minimum spanning tree over each class's components and draws a tube along
    the edges it keeps. Only background voxels are ever filled, so a bridge can never overwrite
    another class.

    The gap ceiling is given in **millimetres** and converted with *spacing*: this cohort spans
    0.3-0.6 mm data, and a fixed voxel budget would bridge twice as far on one as on the other.

    Parameters
    ----------
    close_radius
        Morphological closing applied per class before bridging. ``0`` by default — closing
        thickens thin vessels, which helps β0 but costs Dice and clDice precision.

    Returns
    -------
    Image or array
        Same type and geometry as the input.
    """
    from nvitk.morphology.mst_bridge import fill_multilabel_gaps_mst

    data, source = _unwrap(labelmap)
    targets = _labels_present(data, labels)
    if not targets:
        return labelmap

    # Voxel budget from the *finest* spacing: bridging is measured along a path that may run in
    # any direction, and using the coarsest axis would permit a longer real-world gap than asked.
    finest = min(float(s) for s in (spacing or (1.0,))) if spacing else 1.0
    max_gap_voxels = max(1, int(round(float(max_gap_mm) / finest)))

    with using("cpu"):
        host = to_numpy(data).astype("int32", copy=False)
        bridged = fill_multilabel_gaps_mst(
            host,
            close_radius=int(close_radius),
            bridge_max_gap=max_gap_voxels,
            bridge_radius=int(bridge_radius),
            label_ids=targets,
        )
        filled = int((to_numpy(bridged) != host).sum())
        changed = [
            int(v) for v in targets
            if int(((bridged == v) & (host != v)).sum()) > 0
        ]

    if report is not None:
        report.bridged_voxels += filled
        report.bridged_classes = sorted(set(report.bridged_classes) | set(changed))
    if filled:
        log.info(
            "Bridged %d voxel(s) across gaps up to %.2f mm (%d voxel(s)) in %d class(es).",
            filled, float(max_gap_mm), max_gap_voxels, len(changed),
        )
    out = as_backend_array(bridged)
    return source.with_data(out) if source is not None else out


# ──────────────────────────────────────────────────────────────────────────────
# 2. Invalid adjacency
# ──────────────────────────────────────────────────────────────────────────────


def resolve_invalid_adjacencies(
    labelmap: Any,
    *,
    valid_neighbours: Mapping[int, Iterable[int]],
    labels: Sequence[int] | None = None,
    max_fragment_fraction: float = DEFAULT_FRAGMENT_FRACTION,
    connectivity: int = DEFAULT_CONNECTIVITY,
    report: RepairReport | None = None,
) -> Any:
    """Reassign fragments that touch labels the anatomy does not permit.

    The failure this targets is two vessels running side by side, where the network hands a
    strip of one to the other. That strip is a small component of class A embedded in class B,
    touching things A can never touch. Reassigning it to whichever label it shares the most
    surface with recovers the adjacency *and* usually the correct label, because the label it
    is buried in is the one it belongs to.

    A component is eligible only when it is small **relative to the label it would be merged
    into** — at most *max_fragment_fraction* of it. That is the measure that matches the
    premise: "a fragment embedded in something bigger". Measuring against the component's own
    class would not work, because a class the network hallucinated in exactly one place is 100 %
    of itself and would never qualify, which is precisely the case worth fixing.

    Components whose invalid contacts cannot be resolved by any single reassignment are left
    alone — deleting them would trade an adjacency error for a missing-structure error, which
    scores worse on every other metric.

    Parameters
    ----------
    valid_neighbours
        Label → labels it may legitimately touch, as in
        :func:`~nvitk.measure.segmentation_metrics.invalid_neighbour_error`. Treated
        symmetrically: a pair listed in either direction is permitted.
    max_fragment_fraction
        Ceiling on ``fragment / host`` size. At the default a 1000-voxel vessel absorbs strips
        of up to 250 voxels and nothing larger.
    """
    data, source = _unwrap(labelmap)
    targets = _labels_present(data, labels)
    permitted = {
        int(k): {int(v) for v in values} for k, values in valid_neighbours.items()
    }

    def _allowed(first: int, second: int) -> bool:
        """Whether *first* and *second* may touch, in either listing direction."""
        return second in permitted.get(first, set()) or first in permitted.get(second, set())

    with using("cpu"):
        from scipy.ndimage import binary_dilation, label as cc_label

        host = to_numpy(data).astype("int32", copy=False)
        out = host.copy()
        structure = np.ones((3,) * host.ndim, dtype="uint8")
        sizes = {int(v): int((host == v).sum()) for v in targets}
        moved_components = moved_voxels = 0

        for value in targets:
            mask = host == value
            if not mask.any():
                continue
            components, count = cc_label(mask, structure=structure)
            for index in range(1, count + 1):
                fragment = components == index
                size = int(fragment.sum())

                # Which labels does this fragment actually touch, and how much of each?
                halo = binary_dilation(fragment, structure=structure) & ~fragment
                touched = to_numpy(np.unique(host[halo]))
                contacts = {
                    int(other): int((host[halo] == other).sum())
                    for other in touched
                    if int(other) not in (0, int(value))
                }
                if not contacts:
                    continue
                if all(_allowed(int(value), o) for o in contacts):
                    continue

                # Reassign to the label it is most embedded in, but only if that label is big
                # enough for "embedded" to be true, and only if the move actually resolves the
                # violations rather than relocating them.
                candidate = max(contacts, key=lambda o: (contacts[o], -o))
                if size > max_fragment_fraction * sizes.get(candidate, 0):
                    continue
                if any(o != candidate and not _allowed(candidate, o) for o in contacts):
                    continue
                out[fragment] = candidate
                moved_components += 1
                moved_voxels += size

    if report is not None:
        report.reassigned_components += moved_components
        report.reassigned_voxels += moved_voxels
    if moved_components:
        log.info(
            "Reassigned %d fragment(s) (%d voxel(s)) with anatomically impossible neighbours.",
            moved_components, moved_voxels,
        )
    result = as_backend_array(out)
    return source.with_data(result) if source is not None else result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Left / right consistency
# ──────────────────────────────────────────────────────────────────────────────


def lateral_axis_from_affine(affine: Any, ndim: int = 3) -> int | None:
    """Array axis that runs most nearly left-right in world space.

    Read from the affine's rotation rather than assumed, because this cohort's MR is frequently
    oblique and its array axis order is not guaranteed. ``None`` when there is no affine.
    """
    if affine is None:
        return None
    matrix = to_numpy(affine)[:3, :3]
    # Row 0 of the affine maps array axes onto world x (the left-right world direction); the
    # array axis contributing most to it is the lateral one.
    return int(np.argmax(np.abs(matrix[0, :ndim])))


def lateral_convention(
    labelmap: Any,
    pairs: Mapping[int, int],
    *,
    axis: int,
    min_separation_voxels: float = 2.0,
) -> bool | None:
    """Whether the patient's **right**-side classes sit at the *low* end of *axis*.

    Determined from the data: the mean lateral coordinate of every right-side label is compared
    with that of its left-side partner and the majority wins. This sidesteps orientation
    conventions entirely — no assumption about LPS vs RAS, or which way the affine points.

    Returns ``None`` when the two families do not separate by at least
    *min_separation_voxels*, which means the prediction is too poor (or too sparse) to read a
    convention off. Callers must then decline to mirror anything: acting on a coin toss here
    produces exactly the left-right flip the step exists to fix.
    """
    data, _ = _unwrap(labelmap)
    with using("cpu"):
        host = to_numpy(data)
        votes: list[float] = []
        for right, left in pairs.items():
            right_mask, left_mask = host == int(right), host == int(left)
            if not right_mask.any() or not left_mask.any():
                continue
            right_mean = float(np.argwhere(right_mask)[:, axis].mean())
            left_mean = float(np.argwhere(left_mask)[:, axis].mean())
            votes.append(right_mean - left_mean)

    if not votes:
        return None
    separation = abs(float(np.mean(votes)))
    if separation < float(min_separation_voxels):
        log.warning(
            "Left and right classes separate by only %.2f voxel(s) along axis %d; the "
            "left/right convention cannot be read reliably. Not mirroring anything.",
            separation, axis,
        )
        return None
    return bool(float(np.mean(votes)) < 0)


def enforce_lateral_consistency(
    labelmap: Any,
    *,
    pairs: Mapping[int, int],
    affine: Any = None,
    axis: int | None = None,
    max_fragment_fraction: float = DEFAULT_FRAGMENT_FRACTION,
    connectivity: int = DEFAULT_CONNECTIVITY,
    report: RepairReport | None = None,
) -> Any:
    """Relabel fragments of a lateralised class that sit on the wrong side of the midline.

    The midline is the mean of the two families' lateral centroids, not the centre of the
    volume: the head is rarely centred in the field of view, and the classes themselves are a
    better reference than the array bounds.

    Parameters
    ----------
    pairs
        Right label → left label, e.g. ``{4: 6, 5: 7}``. Used in both directions.
    affine
        Used to find the lateral axis when *axis* is not given.

    Returns
    -------
    Image or array
        Unchanged when the left/right convention cannot be established — see
        :func:`lateral_convention`.
    """
    data, source = _unwrap(labelmap)
    if axis is None:
        axis = lateral_axis_from_affine(
            affine if affine is not None
            else (source.affine if isinstance(source, Image) else None),
            ndim=int(data.ndim),
        )
    if axis is None:
        if report is not None:
            report.notes.append("no affine: lateral axis unknown, left/right check skipped")
        log.warning("No affine available; cannot identify the lateral axis. Skipping.")
        return labelmap

    right_is_low = lateral_convention(labelmap, pairs, axis=int(axis))
    if report is not None:
        report.lateral_axis = int(axis)
        report.lateral_right_is_low = right_is_low
    if right_is_low is None:
        if report is not None:
            report.notes.append("left/right convention unreadable, nothing mirrored")
        return labelmap

    mirror = {int(r): int(l) for r, l in pairs.items()}
    mirror.update({int(l): int(r) for r, l in pairs.items()})
    right_labels = {int(r) for r in pairs}

    with using("cpu"):
        from scipy.ndimage import label as cc_label

        host = to_numpy(data).astype("int32", copy=False)
        out = host.copy()
        structure = np.ones((3,) * host.ndim, dtype="uint8")

        # Midline: halfway between the two families' centroids along the lateral axis.
        coordinates = np.argwhere(np.isin(host, list(mirror)))
        if coordinates.size == 0:
            return labelmap
        rights = np.argwhere(np.isin(host, list(right_labels)))
        lefts = np.argwhere(np.isin(host, [mirror[r] for r in right_labels]))
        if rights.size == 0 or lefts.size == 0:
            return labelmap
        midline = 0.5 * (
            float(rights[:, axis].mean()) + float(lefts[:, axis].mean())
        )

        moved_components = moved_voxels = 0
        for value, partner in mirror.items():
            mask = host == value
            total = int(mask.sum())
            if total == 0:
                continue
            expect_low = (value in right_labels) == bool(right_is_low)
            components, count = cc_label(mask, structure=structure)
            for index in range(1, count + 1):
                fragment = components == index
                size = int(fragment.sum())
                if size > max_fragment_fraction * total:
                    continue
                centre = float(np.argwhere(fragment)[:, axis].mean())
                on_low_side = centre < midline
                if on_low_side != expect_low:
                    out[fragment] = partner
                    moved_components += 1
                    moved_voxels += size

    if report is not None:
        report.mirrored_components += moved_components
        report.mirrored_voxels += moved_voxels
    if moved_components:
        log.info(
            "Mirrored %d fragment(s) (%d voxel(s)) found on the wrong side of the midline.",
            moved_components, moved_voxels,
        )
    result = as_backend_array(out)
    return source.with_data(result) if source is not None else result


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────


def repair_topology(
    labelmap: Any,
    *,
    labels: Sequence[int] | None = None,
    spacing: Sequence[float] | None = None,
    affine: Any = None,
    valid_neighbours: Mapping[int, Iterable[int]] | None = None,
    lateral_pairs: Mapping[int, int] | None = None,
    bridge_gaps_mm: float | None = None,
    bridge_radius: int = DEFAULT_BRIDGE_RADIUS,
    close_radius: int = 0,
    max_fragment_fraction: float = DEFAULT_FRAGMENT_FRACTION,
) -> tuple[Any, RepairReport]:
    """Run the enabled repair steps in order; returns ``(labelmap, report)``.

    Order matters and is not negotiable:

    1. **bridge** first, so a class is whole before anything reasons about its components. A
       fragment that is really one end of a broken vessel must not be reassigned as a stray.
    2. **adjacency** next, while wrong-label fragments are still identifiable by what they
       touch.
    3. **left/right** last, because it reads the midline off the class centroids and wants them
       as clean as they are going to get.

    Each step is opt-in: passing no ``valid_neighbours`` skips step 2, no ``lateral_pairs``
    skips step 3, and ``bridge_gaps_mm=None`` skips step 1.
    """
    report = RepairReport()
    result = labelmap

    if bridge_gaps_mm is not None:
        result = bridge_class_gaps(
            result, labels=labels, spacing=spacing, max_gap_mm=float(bridge_gaps_mm),
            bridge_radius=bridge_radius, close_radius=close_radius, report=report,
        )
    if valid_neighbours:
        result = resolve_invalid_adjacencies(
            result, valid_neighbours=valid_neighbours, labels=labels,
            max_fragment_fraction=max_fragment_fraction, report=report,
        )
    if lateral_pairs:
        result = enforce_lateral_consistency(
            result, pairs=lateral_pairs, affine=affine,
            max_fragment_fraction=max_fragment_fraction, report=report,
        )
    return result, report


__all__ = [
    "DEFAULT_BRIDGE_RADIUS",
    "DEFAULT_FRAGMENT_FRACTION",
    "DEFAULT_MAX_GAP_MM",
    "RepairReport",
    "bridge_class_gaps",
    "enforce_lateral_consistency",
    "lateral_axis_from_affine",
    "lateral_convention",
    "repair_topology",
    "resolve_invalid_adjacencies",
]
