"""TopBrain vessel label maps, label-set identities, and challenge metric constants.

Description
-----------
The TopBrain release ships **three** label maps that share the value range ``1..34`` but
diverge from ``35`` upwards. Mixing them silently produces anatomically wrong training data,
so every label set is spelled out here and :func:`label_map` is the only way to get one.

.. list-table:: Value collisions above 34
   :header-rows: 1

   * - value
     - ``v1_ct``
     - ``v1_mr``
     - ``ta36``
   * - 35
     - VoG (vein of Galen)
     - R-ECA
     - R-ICA-C1-C5
   * - 36
     - StS (straight sinus)
     - L-ECA
     - L-ICA-C1-C5
   * - 37-40
     - ICVs, R/L-BVR, SSS
     - R/L-STA, R/L-MaxA
     - —
   * - 41-42
     - —
     - R/L-MMA
     - —

``ta36`` also renames values 4 and 6 from ``R-ICA``/``L-ICA`` to ``R-ICA-C6-C7``/``L-ICA-C6-C7``
(the supraclinoid segment), because the infraclinoid segment becomes 35/36.

Scoring constants
-----------------
Mirrors ``topbrain25_eval/constants.py`` from `TopBrain_Eval_Metrics
<https://github.com/CoWBenchmark/TopBrain_Eval_Metrics>`_: the "side road" vessel sets used by
the detection F1 metric, the detection IoU threshold, and the HD95 penalty applied when a
class is missing from either mask.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

# ──────────────────────────────────────────────────────────────────────────────
# Label-set identities
# ──────────────────────────────────────────────────────────────────────────────

#: Selectable label sets. ``ta36`` is the ToPBrain-2026 / ToPAneu track and the pipeline default.
LabelSet = Literal["ta36", "v1_ct", "v1_mr"]

#: Directory of the release holding each label set's masks.
LABEL_SET_DIRS: dict[str, str] = {
    "ta36": "labelsTr_topbrain_v2_topaneu36class",
    "v1_ct": "labelsTr_topbrain_v1_ct",
    "v1_mr": "labelsTr_topbrain_v1_mr",
}

#: Release ``labelmap_jsons/`` file matching each label set (used to cross-check our tables).
LABEL_SET_JSONS: dict[str, str] = {
    "ta36": "labels_topbrain_v2_topaneu36class.json",
    "v1_ct": "labels_topbrain_v1_ct.json",
    "v1_mr": "labels_topbrain_v1_mr.json",
}

#: Modalities each label set covers. ``ta36`` is modality-agnostic; the v1 sets are not.
#: Foreground of the binary vessel label set: one class, "is this voxel a vessel".
#: Not a released label set — it is derived by collapsing any TA36/v1 mask, which is what makes
#: silver-standard masks usable as a first fine-tuning stage (see :mod:`stage2_train`).
BINARY_LABELS: dict[int, str] = {1: "vessel"}

LABEL_SET_MODALITIES: dict[str, tuple[str, ...]] = {
    "binary": ("ct", "mr"),
    "binary_ct": ("ct",),
    "binary_mr": ("mr",),
    "ta36": ("ct", "mr"),
    "v1_ct": ("ct",),
    "v1_mr": ("mr",),
}

# ──────────────────────────────────────────────────────────────────────────────
# Label maps
# ──────────────────────────────────────────────────────────────────────────────

#: Values 1-34: identical anatomy in every label set (modulo the ICA rename in ``ta36``).
_COMMON: dict[int, str] = {
    1: "BA",
    2: "R-P1P2",
    3: "L-P1P2",
    4: "R-ICA",
    5: "R-M1",
    6: "L-ICA",
    7: "L-M1",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-A1A2",
    12: "L-A1A2",
    13: "R-A3",
    14: "L-A3",
    15: "3rd-A2",
    16: "3rd-A3",
    17: "R-M2",
    18: "R-M3",
    19: "L-M2",
    20: "L-M3",
    21: "R-P3P4",
    22: "L-P3P4",
    23: "R-VA",
    24: "L-VA",
    25: "R-SCA",
    26: "L-SCA",
    27: "R-AICA",
    28: "L-AICA",
    29: "R-PICA",
    30: "L-PICA",
    31: "R-AChA",
    32: "L-AChA",
    33: "R-OA",
    34: "L-OA",
}

#: 36 foreground classes, modality-agnostic. ICA is split into infra- (35/36) and
#: supraclinoid (4/6) segments, and every v1 vein / extracranial class is dropped.
TA36_LABELS: dict[int, str] = _COMMON | {
    4: "R-ICA-C6-C7",
    6: "L-ICA-C6-C7",
    35: "R-ICA-C1-C5",
    36: "L-ICA-C1-C5",
}

#: 40 foreground classes: the common arteries plus the deep and dural venous system.
V1_CT_LABELS: dict[int, str] = _COMMON | {
    35: "VoG",
    36: "StS",
    37: "ICVs",
    38: "R-BVR",
    39: "L-BVR",
    40: "SSS",
}

#: 42 foreground classes: the common arteries plus the extracranial carotid branches.
V1_MR_LABELS: dict[int, str] = _COMMON | {
    35: "R-ECA",
    36: "L-ECA",
    37: "R-STA",
    38: "L-STA",
    39: "R-MaxA",
    40: "L-MaxA",
    41: "R-MMA",
    42: "L-MMA",
}

#: Name of the modality-agnostic derived binary label set.
BINARY_LABEL_SET: str = "binary"

#: Binary label set per multi-class set it is meant to seed. Each gets its **own** dataset and
#: its own stage 2 provenance, so a CT-only and an MR-only curriculum can be trained side by
#: side without overwriting each other's binary model — one shared slot would mean the second
#: run silently replaced the first, and ``--init-from-binary`` would then resolve to whichever
#: finished last.
BINARY_LABEL_SET_FOR: dict[str, str] = {
    "ta36": "binary",
    "v1_ct": "binary_ct",
    "v1_mr": "binary_mr",
}

#: Modality each binary variant covers, used to filter its silver cohort.
BINARY_SET_MODALITY: dict[str, str | None] = {
    "binary": None, "binary_ct": "ct", "binary_mr": "mr",
}

_LABEL_MAPS: dict[str, dict[int, str]] = {
    **{name: BINARY_LABELS for name in BINARY_SET_MODALITY},
    "ta36": TA36_LABELS,
    "v1_ct": V1_CT_LABELS,
    "v1_mr": V1_MR_LABELS,
}

# ──────────────────────────────────────────────────────────────────────────────
# Scoring constants (topbrain25_eval/constants.py)
# ──────────────────────────────────────────────────────────────────────────────

#: "Side road" vessels present in every label set — small, anatomically variable branches
#: whose *detection* (not overlap) is scored by the F1 metric.
SIDEROAD_COMMON: tuple[int, ...] = (8, 9, 10, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)

#: Per-label-set side-road vessels. ``v1_ct`` adds ICVs and the basal veins of Rosenthal;
#: ``v1_mr`` adds the middle meningeal arteries. ``ta36`` has no extra side roads: its 35/36
#: are ICA segments, which are "highway" vessels.
SIDEROAD_LABELS: dict[str, tuple[int, ...]] = {
    "ta36": SIDEROAD_COMMON,
    "v1_ct": SIDEROAD_COMMON + (37, 38, 39),
    "v1_mr": SIDEROAD_COMMON + (41, 42),
}

#: IoU above which a side-road component counts as detected. Deliberately lenient upstream.
IOU_THRESHOLD: float = 0.25

#: HD95 substituted when a class is absent from the prediction or the reference — roughly the
#: largest distance available inside a human head, in millimetres.
HD95_UPPER_BOUND: float = 290.0


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical adjacency
# ──────────────────────────────────────────────────────────────────────────────

#: Directory holding the published adjacency tables copied from ``TopBrain_Eval_Metrics``.
DATA_DIR: Path = Path(__file__).resolve().parent / "data"

#: Published adjacency table per label set. ``ta36`` has none — see :func:`valid_neighbours`.
_NEIGHBOUR_FILES: dict[str, str] = {
    "v1_ct": "valid_neighbors_ct_all.json",
    "v1_mr": "valid_neighbors_mr_all.json",
}


@lru_cache(maxsize=None)
def is_binary(label_set: str) -> bool:
    """Whether *label_set* is one of the derived single-class vessel sets."""
    return str(label_set) in BINARY_SET_MODALITY


def binary_label_set_for(label_set: str) -> str:
    """The binary set that seeds *label_set*'s multi-class model.

    Raises
    ------
    ValueError
        For a label set with no binary counterpart, rather than silently falling back to the
        shared ``binary`` slot and clobbering another experiment's model.
    """
    try:
        return BINARY_LABEL_SET_FOR[str(label_set)]
    except KeyError:
        raise ValueError(
            f"No binary label set defined for {label_set!r}. Known: "
            f"{', '.join(sorted(BINARY_LABEL_SET_FOR))}."
        ) from None


def binary_modality(label_set: str) -> str | None:
    """Modality a binary set is restricted to, or ``None`` when it covers both."""
    return BINARY_SET_MODALITY.get(str(label_set))


def valid_neighbours(label_set: str) -> dict[int, tuple[int, ...]]:
    """Label → labels it may anatomically touch, for the invalid-neighbour metric.

    ``v1_ct`` and ``v1_mr`` are the challenge's published tables, copied verbatim from
    `TopBrain_Eval_Metrics <https://github.com/CoWBenchmark/TopBrain_Eval_Metrics>`_.

    **``ta36`` is derived, not published.** The challenge ships tables only for the two v1 label
    sets, and TA36 renumbers everything above 34 (35/36 become the infraclinoid ICA segments
    rather than veins or extracranial arteries), so neither table can be applied to it as-is.
    The derivation keeps the 1-34 adjacencies the three label sets share — taking the union of
    the CT and MR tables, since a pair valid in either modality is anatomically possible — and
    adds the two links the new classes introduce by construction: the infraclinoid ICA
    (35 / 36) is continuous with the supraclinoid ICA (4 / 6) on the same side, and inherits its
    ipsilateral vertebral/basilar relationships.

    Treat the TA36 invalid-neighbour numbers as a self-consistent internal signal for comparing
    runs, not as a reproduction of the organisers' score.
    """
    if is_binary(label_set):
        # One class cannot touch another, so every adjacency is trivially valid and the metric
        # is meaningless rather than perfect. Returning an empty table makes callers score 0
        # violations, which is the honest answer.
        return {}

    if label_set in _NEIGHBOUR_FILES:
        path = DATA_DIR / _NEIGHBOUR_FILES[label_set]
        if not path.is_file():
            raise FileNotFoundError(f"Adjacency table missing: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): tuple(int(v) for v in values) for k, values in raw.items()}

    if label_set != "ta36":
        raise ValueError(f"Unknown label set {label_set!r}.")

    shared: dict[int, set[int]] = {}
    for source in ("v1_ct", "v1_mr"):
        for key, values in valid_neighbours(source).items():
            if key > 34:
                continue
            shared.setdefault(key, set()).update(v for v in values if v <= 34)

    # The infraclinoid ICA is continuous with the supraclinoid ICA on the same side.
    for infra, supra in ((35, 4), (36, 6)):
        shared.setdefault(supra, set()).add(infra)
        shared[infra] = {supra} | {v for v in shared.get(supra, set()) if v <= 34}

    return {key: tuple(sorted(values)) for key, values in sorted(shared.items())}


# ──────────────────────────────────────────────────────────────────────────────
# Accessors
# ──────────────────────────────────────────────────────────────────────────────


def label_map(label_set: str) -> dict[int, str]:
    """Value → anatomy name for *label_set* (foreground only; background 0 is implicit).

    Raises
    ------
    ValueError
        If *label_set* is not one of ``ta36``, ``v1_ct``, ``v1_mr``. Deliberately strict: a
        typo that silently fell back to a default would train on the wrong anatomy.
    """
    try:
        return dict(_LABEL_MAPS[label_set])
    except KeyError:
        raise ValueError(
            f"Unknown label set {label_set!r}. Valid: {', '.join(sorted(_LABEL_MAPS))}."
        ) from None


def lateral_pairs(label_set: str) -> dict[int, str]:
    """Right label → left label, for the classes that exist as a mirrored pair.

    Derived from the ``R-`` / ``L-`` name prefixes rather than hard-coded, so it follows the
    label map instead of drifting from it. Unpaired classes (BA, Acom, the 3rd-A2/A3 variants)
    are absent by construction: they have no side, and giving them one is how a post-processing
    step invents anatomy.
    """
    names = label_map(label_set)
    by_name = {name: value for value, name in names.items()}
    pairs: dict[int, str] = {}
    for value, name in sorted(names.items()):
        if not name.startswith("R-"):
            continue
        partner = by_name.get("L-" + name[2:])
        if partner is not None:
            pairs[int(value)] = int(partner)
    return pairs


def num_foreground(label_set: str) -> int:
    """Number of foreground classes in *label_set* (36, 40 or 42)."""
    return len(label_map(label_set))


def max_label(label_set: str) -> int:
    """Largest label value in *label_set*; label values are contiguous from 1."""
    return max(label_map(label_set))


def sideroad_labels(label_set: str) -> tuple[int, ...]:
    """Side-road vessel values scored by the detection-F1 metric for *label_set*."""
    label_map(label_set)  # validate
    if is_binary(label_set):
        return ()  # no side roads to distinguish when there is one class
    return SIDEROAD_LABELS[label_set]


def nnunet_labels(label_set: str) -> dict[str, int]:
    """``dataset.json``-shaped ``{name: value}`` map, background first.

    nnU-Net wants name → value (the inverse of :func:`label_map`) with an explicit
    ``background`` entry.
    """
    return {"background": 0} | {name: value for value, name in sorted(label_map(label_set).items())}


__all__ = [
    "DATA_DIR",
    "HD95_UPPER_BOUND",
    "IOU_THRESHOLD",
    "LABEL_SET_DIRS",
    "LABEL_SET_JSONS",
    "LABEL_SET_MODALITIES",
    "LabelSet",
    "SIDEROAD_COMMON",
    "SIDEROAD_LABELS",
    "BINARY_LABELS",
    "BINARY_LABEL_SET",
    "BINARY_LABEL_SET_FOR",
    "BINARY_SET_MODALITY",
    "binary_label_set_for",
    "binary_modality",
    "TA36_LABELS",
    "is_binary",
    "V1_CT_LABELS",
    "V1_MR_LABELS",
    "label_map",
    "lateral_pairs",
    "max_label",
    "nnunet_labels",
    "num_foreground",
    "sideroad_labels",
    "valid_neighbours",
]
