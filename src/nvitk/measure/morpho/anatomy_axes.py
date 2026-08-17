"""Affine- and species-aware resolution of anatomical direction rules.

Topology JSONs express where a vessel's proximal ("init") end sits with a
``no_upstream_start`` rule such as ``"inferior"`` or ``"caudal"``. Turning that
word into an array axis + sign needs two pieces of information:

1. **The image affine** — which array axis runs L/R, A/P and S/I, and in which
   direction the index grows. Historically this was hardcoded to
   ``axis0 = L→R, axis1 = A/P, axis2 = S/I`` (an implicit LAS assumption), which
   is right for RAS-coded human TOF and wrong for anything else.
2. **The species** — a mouse is a quadruped, so its rostro-caudal axis lies along
   scanner A/P, not S/I. ``caudal`` and ``inferior`` are the same direction in a
   human and orthogonal directions in a mouse.

Terms therefore split into two groups:

- **Scanner-anatomical** (species-independent): ``superior``, ``inferior``,
  ``anterior``, ``posterior``, ``lateral_R``, ``lateral_L``.
- **Animal-frame** (species-dependent): ``rostral``, ``caudal``, ``dorsal``,
  ``ventral``.

Rules may be combined with ``+`` (e.g. ``"caudal+ventral"``); each component is
range-normalised before summing so neither axis dominates.

This module deliberately depends only on ``numpy`` + ``nibabel`` (already a hard
dependency of :mod:`nvitk.measure.morpho.io_utils`) so that
:mod:`nvitk.measure.morpho` stays importable without the rest of nvitk — the
Slicer ``MouseTOFMorphometrics`` module relies on that.

.. note::
   :class:`AnatomicalAxes` intentionally mirrors ``RasAxes`` /
   ``resolve_ras_axes`` in
   :mod:`nvitk.pipes.qvtpy.util.centerline.venous_heuristics`. It is not shared
   because importing ``pipes`` from ``measure`` would invert the package
   layering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

SPECIES_AUTO = "auto"
SPECIES_HUMAN = "human"
SPECIES_MOUSE = "mouse"
SPECIES_CHOICES: tuple[str, ...] = (SPECIES_AUTO, SPECIES_HUMAN, SPECIES_MOUSE)

DEFAULT_AXCODES = "RAS"

# Scanner-anatomical rule -> (plane, sign) where plane is "lr" / "ap" / "si" and
# sign +1 means "the positive end of that plane" (R / A / S).
_SCANNER_RULES: dict[str, tuple[str, int]] = {
    "right": ("lr", +1),
    "lateral_r": ("lr", +1),
    "left": ("lr", -1),
    "lateral_l": ("lr", -1),
    "anterior": ("ap", +1),
    "posterior": ("ap", -1),
    "superior": ("si", +1),
    "inferior": ("si", -1),
}

# Animal-frame rule -> scanner-anatomical rule, per species.
_ANIMAL_FRAMES: dict[str, dict[str, str]] = {
    # Biped: the body axis is vertical in the scanner.
    SPECIES_HUMAN: {
        "rostral": "superior",
        "caudal": "inferior",
        "dorsal": "posterior",
        "ventral": "anterior",
    },
    # Quadruped: the body axis runs along the scanner A/P axis, and the animal's
    # dorso-ventral axis maps onto scanner S/I.
    SPECIES_MOUSE: {
        "rostral": "anterior",
        "caudal": "posterior",
        "dorsal": "superior",
        "ventral": "inferior",
    },
}

ANIMAL_FRAME_RULES: tuple[str, ...] = ("rostral", "caudal", "dorsal", "ventral")
DIRECTION_RULES: tuple[str, ...] = tuple(_SCANNER_RULES) + ANIMAL_FRAME_RULES


def normalize_species(species: Optional[str], *, fallback: str = SPECIES_HUMAN) -> str:
    """Normalise a species token; ``None`` / ``""`` / ``"auto"`` resolve to *fallback*."""
    token = (species or "").strip().lower()
    if not token or token == SPECIES_AUTO:
        return fallback
    if token in _ANIMAL_FRAMES:
        return token
    raise ValueError(f"Unknown species {species!r}; expected one of {SPECIES_CHOICES}.")


def _axis_for_code(codes: Sequence[str], letter: str) -> tuple[int, int]:
    """Return ``(array_axis, sign)`` for anatomical *letter* (R/A/S), or ``(-1, 1)``."""
    up = letter.upper()
    opposite = {"R": "L", "A": "P", "S": "I"}[up]
    for i, code in enumerate(list(codes)[:3]):
        c = str(code).upper()
        if c == up:
            return i, 1
        if c == opposite:
            return i, -1
    return -1, 1


def axcodes_from_affine(affine) -> Optional[str]:
    """Return axis codes such as ``"RAS"`` / ``"LPS"`` from a 4x4 affine, else ``None``."""
    if affine is None:
        return None
    try:
        import nibabel as nib

        aff = np.asarray(affine, dtype=float)
        if aff.shape != (4, 4):
            return None
        return "".join(str(c) for c in nib.orientations.aff2axcodes(aff))
    except Exception:
        return None


@dataclass(frozen=True)
class AnatomicalAxes:
    """Array-axis + sign resolution for anatomical direction rules.

    Signs follow the "positive end" convention: ``lr_sign == +1`` means an
    increasing array index moves toward the subject's **R**ight, ``ap_sign == +1``
    toward **A**nterior and ``si_sign == +1`` toward **S**uperior.
    """

    species: str = SPECIES_HUMAN
    axcodes: str = DEFAULT_AXCODES
    lr_axis: int = 0
    lr_sign: int = 1
    ap_axis: int = 1
    ap_sign: int = 1
    si_axis: int = 2
    si_sign: int = 1

    # -- rule resolution ----------------------------------------------------
    def _plane(self, plane: str) -> tuple[int, int]:
        if plane == "lr":
            return self.lr_axis, self.lr_sign
        if plane == "ap":
            return self.ap_axis, self.ap_sign
        return self.si_axis, self.si_sign

    def canonical_rule(self, rule: Optional[str]) -> Optional[str]:
        """Map an animal-frame rule onto its scanner-anatomical equivalent."""
        token = (rule or "").strip().lower()
        if not token:
            return None
        token = _ANIMAL_FRAMES.get(self.species, {}).get(token, token)
        return token if token in _SCANNER_RULES else None

    def direction(self, rule: Optional[str]) -> Optional[tuple[int, int]]:
        """``(array_axis, sign)`` for *rule*; ``sign == +1`` ⇒ larger index is more *rule*."""
        canonical = self.canonical_rule(rule)
        if canonical is None:
            return None
        plane, want = _SCANNER_RULES[canonical]
        axis, axis_sign = self._plane(plane)
        if axis < 0:
            return None
        return int(axis), int(want * axis_sign)

    def score(self, pt_mm, rule: Optional[str]) -> float:
        """Ranking score for a point under *rule* — **lower means more extreme** toward it.

        Single rules read one coordinate. Combined rules (``"caudal+ventral"``)
        sum their components; use :meth:`score_many` when several points are
        ranked together so each component can be range-normalised first.
        """
        pt = np.asarray(pt_mm, dtype=float)
        total = 0.0
        matched = False
        for part in str(rule or "").split("+"):
            resolved = self.direction(part)
            if resolved is None:
                continue
            axis, sign = resolved
            total += -sign * float(pt[axis])
            matched = True
        return total if matched else 0.0

    def score_many(self, pts_mm, rule: Optional[str]) -> np.ndarray:
        """Per-point scores for *rule*, range-normalising each term of a combined rule."""
        pts = np.asarray(pts_mm, dtype=float)
        if pts.ndim != 2:
            pts = np.atleast_2d(pts)
        parts = [p for p in (self.direction(part) for part in str(rule or "").split("+")) if p]
        if not parts:
            return np.zeros(len(pts), dtype=float)
        if len(parts) == 1:
            axis, sign = parts[0]
            return -sign * pts[:, axis]

        total = np.zeros(len(pts), dtype=float)
        for axis, sign in parts:
            vals = -sign * pts[:, axis]
            span = float(np.nanmax(vals) - np.nanmin(vals))
            total += (vals - float(np.nanmin(vals))) / span if span > 0 else 0.0
        return total

    # -- provenance ---------------------------------------------------------
    def describe_rule(self, rule: Optional[str]) -> str:
        """Human-readable resolution of *rule*, e.g. ``"caudal→axis1(+)"``."""
        parts = []
        for part in str(rule or "").split("+"):
            token = part.strip()
            if not token:
                continue
            resolved = self.direction(token)
            if resolved is None:
                parts.append(f"{token}→?")
            else:
                axis, sign = resolved
                parts.append(f"{token}→axis{axis}({'+' if sign > 0 else '-'})")
        return " ".join(parts) if parts else "none"

    def describe(self) -> str:
        """One-line summary for logs and Excel provenance columns."""
        return (
            f"species={self.species} axcodes={self.axcodes} "
            f"LR=axis{self.lr_axis}({'+' if self.lr_sign > 0 else '-'}) "
            f"AP=axis{self.ap_axis}({'+' if self.ap_sign > 0 else '-'}) "
            f"SI=axis{self.si_axis}({'+' if self.si_sign > 0 else '-'})"
        )


def resolve_anatomical_axes(
    affine=None,
    *,
    species: Optional[str] = None,
    axes_override: Optional[str] = None,
) -> AnatomicalAxes:
    """Resolve L/R, A/P and S/I array axes for a volume.

    Parameters
    ----------
    affine
        4x4 voxel-to-world affine (nibabel convention, IJK→RAS). ``None`` or an
        unparseable affine falls back to ``RAS``.
    species
        ``"human"`` (default), ``"mouse"``, or ``"auto"``/``None`` for human.
        Selects the animal-frame → scanner mapping for ``rostral``/``caudal``/
        ``dorsal``/``ventral``.
    axes_override
        Axis codes such as ``"LSA"`` that replace the header-derived ones, for
        volumes whose header labels are known to be wrong.
    """
    resolved_species = normalize_species(species)
    codes = (axes_override or "").strip().upper() or axcodes_from_affine(affine) or DEFAULT_AXCODES

    lr_axis, lr_sign = _axis_for_code(codes, "R")
    ap_axis, ap_sign = _axis_for_code(codes, "A")
    si_axis, si_sign = _axis_for_code(codes, "S")
    if lr_axis < 0 or ap_axis < 0 or si_axis < 0:
        codes, lr_axis, lr_sign, ap_axis, ap_sign, si_axis, si_sign = (
            DEFAULT_AXCODES, 0, 1, 1, 1, 2, 1,
        )

    return AnatomicalAxes(
        species=resolved_species,
        axcodes=str(codes),
        lr_axis=int(lr_axis), lr_sign=int(lr_sign),
        ap_axis=int(ap_axis), ap_sign=int(ap_sign),
        si_axis=int(si_axis), si_sign=int(si_sign),
    )


@dataclass(frozen=True)
class MorphoContext:
    """Per-case settings that must reach spawned workers.

    Frozen and picklable: :func:`nvitk.measure.morpho.run_case.run_case` spawns
    (not forks) its workers, so anything the algorithm modules need beyond the
    import-time constants has to travel inside the job tuple.
    """

    axes: AnatomicalAxes = AnatomicalAxes()
    length_scale: float = 1.0

    def scaled(self, value_mm: float) -> float:
        """Scale a human-calibrated millimetre threshold to this case's anatomy."""
        return float(value_mm) * float(self.length_scale)

    def describe(self) -> str:
        return f"{self.axes.describe()} length_scale={self.length_scale:g}"


def default_morpho_context() -> MorphoContext:
    """Back-compat context: RAS axes, human frame, unscaled thresholds."""
    return MorphoContext()


__all__ = [
    "ANIMAL_FRAME_RULES",
    "DEFAULT_AXCODES",
    "DIRECTION_RULES",
    "SPECIES_AUTO",
    "SPECIES_CHOICES",
    "SPECIES_HUMAN",
    "SPECIES_MOUSE",
    "AnatomicalAxes",
    "MorphoContext",
    "axcodes_from_affine",
    "default_morpho_context",
    "normalize_species",
    "resolve_anatomical_axes",
]
