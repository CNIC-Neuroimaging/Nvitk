"""
Brain parcellation vocabularies — region names, lobes and the parenchyma mask.

Description
-----------
The brain-vascular mapping needs three things from a parcellation, whatever produced it:

1. a **name** per region id (``ctx-lh-insula``, ``Left-Thalamus``, ``Left frontal lobe``...),
2. the **parenchyma**: the voxels a vessel can irrigate (no ventricles, CSF or sinuses),
3. optionally a **coarser scheme**: lobes, obtained by collapsing the Desikan-Killiany(-Tourville)
   cortical parcels and filling the cerebral white matter from the nearest cortical lobe.

The producers it knows are ANTsPyNet DKT and FastSurfer (both emit FreeSurfer's
``aparc+aseg`` numbering: aseg ids below 1000, cortex at ``1000 + offset`` / ``2000 + offset``)
and TotalSegmentator's ``brain_structures`` task (already lobe-level, but bilateral). Any other
label map works as "as labelled": the names come from the layer and every label counts as tissue.

Lobes
-----
FreeSurfer's ``mri_annotation2label --lobesStrict`` division, with cingulate and insula kept as
lobes of their own because they belong to different arterial territories (ACA / MCA) than the
frontal and parietal convexity they would otherwise be merged into. Deep grey matter is split
into basal ganglia (lenticulostriate, MCA) and thalamus / diencephalon (thalamoperforators,
PCA); hippocampus and amygdala join the temporal lobe. Lobe ids: left ``1..9``, right
``11..19``, brainstem ``20``, cerebellar vermis ``21``.

Array / axis conventions
------------------------
Label maps are 3D integer arrays in any axis order; the only geometry used is the per-axis
*spacing* (mm) for the nearest-lobe white-matter fill. I/O and arrays: backend ``np`` / ``ndi``
after ``setup(globals())``.
"""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.segmentation.dkt import APARC_LABELS, APARC_PARCEL_OFFSETS

setup(globals())

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Schemes
# ──────────────────────────────────────────────────────────────────────────────

#: ``aparc+aseg``-numbered label map (DKT / Desikan-Killiany + FreeSurfer aseg).
SCHEME_APARC = "aparc"
#: Lobes (:data:`LOBE_LABELS`), collapsed from an ``aparc`` map.
SCHEME_LOBES = "lobes"
#: TotalSegmentator ``brain_structures`` (lobe-level, split into hemispheres by the caller).
SCHEME_TOTALSEG = "brain_structures"
#: Any other label map: names from the layer, every label counts as tissue.
SCHEME_GENERIC = "generic"

PARCELLATION_SCHEMES: tuple[str, ...] = (SCHEME_APARC, SCHEME_LOBES, SCHEME_TOTALSEG, SCHEME_GENERIC)

# ──────────────────────────────────────────────────────────────────────────────
# Label tables
# ──────────────────────────────────────────────────────────────────────────────

#: FreeSurfer ``aseg`` structures, as written into ``aparc+aseg`` by FreeSurfer, FastSurfer and
#: ANTsPyNet's DKT network (which also emits the basal forebrain, 91 / 92).
ASEG_LABELS: dict[int, str] = {
    2: "Left-Cerebral-White-Matter", 3: "Left-Cerebral-Cortex", 4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent", 7: "Left-Cerebellum-White-Matter", 8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus", 11: "Left-Caudate", 12: "Left-Putamen", 13: "Left-Pallidum",
    14: "3rd-Ventricle", 15: "4th-Ventricle", 16: "Brain-Stem", 17: "Left-Hippocampus",
    18: "Left-Amygdala", 24: "CSF", 26: "Left-Accumbens-area", 28: "Left-VentralDC",
    30: "Left-vessel", 31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter", 42: "Right-Cerebral-Cortex", 43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent", 46: "Right-Cerebellum-White-Matter", 47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus", 50: "Right-Caudate", 51: "Right-Putamen", 52: "Right-Pallidum",
    53: "Right-Hippocampus", 54: "Right-Amygdala", 58: "Right-Accumbens-area", 60: "Right-VentralDC",
    62: "Right-vessel", 63: "Right-choroid-plexus", 72: "5th-Ventricle", 77: "WM-hypointensities",
    85: "Optic-Chiasm", 91: "Left-Basal-Forebrain", 92: "Right-Basal-Forebrain",
    251: "CC_Posterior", 252: "CC_Mid_Posterior", 253: "CC_Central", 254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    # ANTsPyNet DKT's cerebellum: FreeSurfer's older "exterior" ids and the vermis lobules.
    6: "Left-Cerebellum-Exterior", 45: "Right-Cerebellum-Exterior",
    630: "Cerebellar-Vermal-Lobules-I-V", 631: "Cerebellar-Vermal-Lobules-VI-VII",
    632: "Cerebellar-Vermal-Lobules-VIII-X",
}

#: Full ``aparc+aseg`` vocabulary: aseg structures and Desikan-Killiany cortex.
APARC_ASEG_LABELS: dict[int, str] = {**ASEG_LABELS, **APARC_LABELS}

#: aseg ids that are not parenchyma: ventricles, CSF, choroid plexus and FreeSurfer's "vessel".
APARC_NON_PARENCHYMA: frozenset[int] = frozenset({4, 5, 14, 15, 24, 30, 31, 43, 44, 62, 63, 72})

_LOBES: tuple[str, ...] = (
    "frontal lobe", "parietal lobe", "temporal lobe", "occipital lobe", "cingulate",
    "insula", "basal ganglia", "thalamus / diencephalon", "cerebellum",
)
#: Lobe id within a hemisphere (left = id, right = id + :data:`RIGHT_LOBE_OFFSET`).
LOBE_IDS: dict[str, int] = {name: i + 1 for i, name in enumerate(_LOBES)}
RIGHT_LOBE_OFFSET = 10
BRAINSTEM_LOBE_ID = 20
VERMIS_LOBE_ID = 21

#: Lobe scheme vocabulary.
LOBE_LABELS: dict[int, str] = {
    **{i: f"Left {name}" for name, i in LOBE_IDS.items()},
    **{i + RIGHT_LOBE_OFFSET: f"Right {name}" for name, i in LOBE_IDS.items()},
    BRAINSTEM_LOBE_ID: "Brainstem",
    VERMIS_LOBE_ID: "Cerebellar vermis",
}

#: Desikan-Killiany parcel -> lobe (FreeSurfer ``--lobesStrict``, cingulate and insula apart).
APARC_PARCEL_LOBE: dict[str, str] = {
    **dict.fromkeys(
        ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
         "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
         "precentral", "paracentral", "frontalpole"),
        "frontal lobe",
    ),
    **dict.fromkeys(
        ("superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"),
        "parietal lobe",
    ),
    **dict.fromkeys(
        ("superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
         "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"),
        "temporal lobe",
    ),
    **dict.fromkeys(("lateraloccipital", "lingual", "cuneus", "pericalcarine"), "occipital lobe"),
    **dict.fromkeys(
        ("rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
         "isthmuscingulate"),
        "cingulate",
    ),
    "insula": "insula",
}

#: aseg structure -> (lobe, side) for the structures that are a lobe's member outright.
_ASEG_LOBE: dict[int, tuple[str, str]] = {
    10: ("thalamus / diencephalon", "L"), 28: ("thalamus / diencephalon", "L"),
    49: ("thalamus / diencephalon", "R"), 60: ("thalamus / diencephalon", "R"),
    11: ("basal ganglia", "L"), 12: ("basal ganglia", "L"), 13: ("basal ganglia", "L"),
    26: ("basal ganglia", "L"), 91: ("basal ganglia", "L"),
    50: ("basal ganglia", "R"), 51: ("basal ganglia", "R"), 52: ("basal ganglia", "R"),
    58: ("basal ganglia", "R"), 92: ("basal ganglia", "R"),
    17: ("temporal lobe", "L"), 18: ("temporal lobe", "L"),
    53: ("temporal lobe", "R"), 54: ("temporal lobe", "R"),
    6: ("cerebellum", "L"), 7: ("cerebellum", "L"), 8: ("cerebellum", "L"),
    45: ("cerebellum", "R"), 46: ("cerebellum", "R"), 47: ("cerebellum", "R"),
}

#: Cerebral white matter / unparcellated cortex, filled from the nearest cortical lobe of its side
#: (``""``: either side, the corpus callosum and WM hypointensities).
_ASEG_FILL_SIDE: dict[int, str] = {
    2: "L", 3: "L", 41: "R", 42: "R", 77: "", 251: "", 252: "", 253: "", 254: "", 255: "",
}

#: TotalSegmentator ``brain_structures`` structures that are not parenchyma.
TOTALSEG_NON_PARENCHYMA: frozenset[int] = frozenset({2, 3, 10})
#: TotalSegmentator ``brain_structures`` structures that straddle the midline and are split into
#: hemispheres (:func:`lateralize_by_midline`).
TOTALSEG_BILATERAL: frozenset[int] = frozenset({5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16})
#: Id offset of the right half of a lateralized structure.
RIGHT_HALF_OFFSET = 100


# ──────────────────────────────────────────────────────────────────────────────
# Names and parenchyma
# ──────────────────────────────────────────────────────────────────────────────


def totalseg_brain_structure_labels(*, lateralized: bool = True) -> dict[int, str]:
    """TotalSegmentator ``brain_structures`` names, with the right halves at ``id + 100``."""
    from nvitk.segmentation.total_segmentator.class_maps import get_class_map

    base = {int(k): str(v).replace("_", " ") for k, v in get_class_map(SCHEME_TOTALSEG).items()}
    if not lateralized:
        return base
    out: dict[int, str] = {}
    for lid, name in base.items():
        if lid in TOTALSEG_BILATERAL:
            out[lid] = f"Left {name}"
            out[lid + RIGHT_HALF_OFFSET] = f"Right {name}"
        else:
            out[lid] = name
    return out


def scheme_label_names(scheme: str) -> dict[int, str]:
    """``{id: name}`` for *scheme* (empty for :data:`SCHEME_GENERIC`)."""
    if scheme == SCHEME_APARC:
        return dict(APARC_ASEG_LABELS)
    if scheme == SCHEME_LOBES:
        return dict(LOBE_LABELS)
    if scheme == SCHEME_TOTALSEG:
        return totalseg_brain_structure_labels()
    return {}


def non_parenchyma_ids(scheme: str) -> frozenset[int]:
    """Region ids of *scheme* that are not tissue a vessel irrigates (ventricles, CSF, sinuses)."""
    if scheme == SCHEME_APARC:
        return APARC_NON_PARENCHYMA
    if scheme == SCHEME_TOTALSEG:
        return TOTALSEG_NON_PARENCHYMA
    return frozenset()


def parenchyma_mask(labels: Any, scheme: str) -> Any:
    """Voxels of *labels* a vessel can irrigate: labelled, minus ventricles / CSF / sinuses."""
    lab = as_backend_array(labels)
    mask = lab > 0
    for lid in sorted(non_parenchyma_ids(scheme)):
        mask &= lab != lid
    return mask


def region_side(name: str) -> str:
    """``"L"`` / ``"R"`` read from a region name, ``""`` when it names no side."""
    low = str(name or "").strip().lower()
    for prefix, side in (("ctx-lh-", "L"), ("ctx-rh-", "R"), ("left", "L"), ("right", "R"),
                         ("lh.", "L"), ("rh.", "R"), ("l_", "L"), ("r_", "R")):
        if low.startswith(prefix):
            return side
    for suffix, side in (("_left", "L"), ("_right", "R"), (" left", "L"), (" right", "R")):
        if low.endswith(suffix):
            return side
    return ""


#: Region categories, as :func:`region_category` names them.
REGION_CATEGORIES: tuple[str, ...] = (
    "cortex", "subcortical", "white matter", "cerebellum", "brainstem", "other",
)

_SUBCORTICAL_WORDS: tuple[str, ...] = (
    "thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala", "accumbens",
    "ventraldc", "ventral dc", "basal-forebrain", "basal forebrain", "basal ganglia", "lentiform",
    "diencephalon",
)


def region_category(name: str) -> str:
    """
    What kind of tissue a parcellation region is, from its name: ``"cortex"``, ``"subcortical"``
    (deep grey matter), ``"white matter"``, ``"cerebellum"``, ``"brainstem"`` or ``"other"``.

    Works on the aparc+aseg (``ctx-lh-insula``, ``Left-Putamen``), lobe (``Left frontal lobe``)
    and TotalSegmentator (``Left insular cortex``) vocabularies. A lobe counts as cortex (its
    white matter included: the lobe scheme does not separate them).
    """
    low = str(name or "").strip().lower().replace("_", " ")
    if low.startswith("ctx-") or low.startswith("ctx "):
        return "cortex"
    if "cerebell" in low or "vermal" in low or "vermis" in low:
        return "cerebellum"
    if "brain-stem" in low or "brainstem" in low or "brain stem" in low:
        return "brainstem"
    if "white-matter" in low or "white matter" in low or low.startswith(("wm-", "cc ", "cc_")) \
            or "internal capsule" in low:
        return "white matter"
    if any(word in low for word in _SUBCORTICAL_WORDS):
        return "subcortical"
    if "cortex" in low or "lobe" in low or "cingulate" in low or "insula" in low:
        return "cortex"
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Lobes from aparc+aseg
# ──────────────────────────────────────────────────────────────────────────────


def _aparc_to_lobe_table() -> dict[int, int]:
    """``{aparc+aseg id: lobe id}`` for every id that maps to a lobe outright."""
    table: dict[int, int] = {}
    for side, base, offset in (("L", 1000, 0), ("R", 2000, RIGHT_LOBE_OFFSET)):
        for parcel_offset, parcel in APARC_PARCEL_OFFSETS.items():
            lobe = APARC_PARCEL_LOBE.get(parcel)
            if lobe is not None:
                table[base + parcel_offset] = LOBE_IDS[lobe] + offset
    for lid, (lobe, side) in _ASEG_LOBE.items():
        table[lid] = LOBE_IDS[lobe] + (RIGHT_LOBE_OFFSET if side == "R" else 0)
    table[16] = BRAINSTEM_LOBE_ID
    for vermis in (630, 631, 632):
        table[vermis] = VERMIS_LOBE_ID
    return table


def lobes_from_aparc(aparc: Any, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> Any:
    """
    Collapse an ``aparc+aseg`` label map into lobes (:data:`LOBE_LABELS`).

    Strategy
    --------
    1. Cortical parcels and the deep structures map to their lobe outright.
    2. Cerebral white matter (and unparcellated cortex) takes the lobe of the nearest cortical
       lobe voxel **of its own hemisphere**, Euclidean in mm, so the lobes extend from the
       cortex down to the ventricles instead of stopping at the grey matter. The corpus
       callosum and WM hypointensities take the nearest of either side.
    3. Ventricles, CSF and anything unknown stay 0.

    Parameters
    ----------
    aparc : array
        3D ``aparc+aseg``-numbered label map.
    spacing : sequence of float
        Voxel size per array axis (mm), for the nearest-lobe distances.

    Returns
    -------
    array
        int16 lobe map on the same grid, on the active backend.
    """
    lab = as_backend_array(aparc).astype(np.int32)
    out = np.zeros(lab.shape, dtype=np.int16)

    # ---- 1. Direct members --------------------------------------------------------
    for lid, lobe in sorted(_aparc_to_lobe_table().items()):
        out[lab == lid] = lobe

    # ---- 2. White matter from the nearest cortical lobe of its side ----------------
    cortical = {
        "L": [LOBE_IDS[n] for n in ("frontal lobe", "parietal lobe", "temporal lobe",
                                     "occipital lobe", "cingulate", "insula")],
    }
    cortical["R"] = [i + RIGHT_LOBE_OFFSET for i in cortical["L"]]
    cortical[""] = cortical["L"] + cortical["R"]
    sampling = tuple(float(s) for s in spacing)
    for side in ("L", "R", ""):
        fill = np.zeros(lab.shape, dtype=bool)
        for lid, fill_side in _ASEG_FILL_SIDE.items():
            if fill_side == side:
                fill |= lab == lid
        if not bool(fill.any()):
            continue
        seeds = np.zeros(lab.shape, dtype=bool)
        for lobe in cortical[side]:
            seeds |= out == lobe
        if not bool(seeds.any()):
            log.warning(f"No cortical lobe on side {side or 'L/R'}: its white matter stays unlabelled.")
            continue
        _, idx = ndi.distance_transform_edt(~seeds, sampling=sampling, return_indices=True)
        out[fill] = out[tuple(axis[fill] for axis in idx)]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Hemispheres for bilateral labels
# ──────────────────────────────────────────────────────────────────────────────


def lateralize_by_midline(labels: Any, voxel_to_world: Any, ids: Sequence[int],
                          *, offset: int = RIGHT_HALF_OFFSET) -> Any:
    """
    Split bilateral structures at the brain's midline: the right halves get ``id + offset``.

    The midline is the median world x (RAS: +x = subject's right) of every labelled voxel, a
    plane perpendicular to world x. Good enough for an upright head; a strongly rolled head
    would need a fitted mid-sagittal plane instead.

    Parameters
    ----------
    labels : array
        3D label map.
    voxel_to_world : (4, 4) array
        Voxel index -> RAS world (mm) of *labels*.
    ids : sequence of int
        The labels to split.
    """
    lab = as_backend_array(labels).astype(np.int32)
    with using("cpu"):
        aff = to_numpy(voxel_to_world).astype(float)[-4:, -4:]
    # World x of every voxel is affine row 0 applied to its index: a sum of per-axis ramps.
    ramps = [np.arange(n, dtype=np.float32) for n in lab.shape]
    world_x = (float(aff[0, 3])
               + float(aff[0, 0]) * ramps[0][:, None, None]
               + float(aff[0, 1]) * ramps[1][None, :, None]
               + float(aff[0, 2]) * ramps[2][None, None, :])
    labelled = lab > 0
    if not bool(labelled.any()):
        return lab
    midline = float(np.median(np.broadcast_to(world_x, lab.shape)[labelled]))
    right = np.broadcast_to(world_x > midline, lab.shape)
    out = lab.copy()
    for lid in ids:
        out[(lab == int(lid)) & right] = int(lid) + int(offset)
    return out


__all__ = [
    "APARC_ASEG_LABELS",
    "APARC_NON_PARENCHYMA",
    "APARC_PARCEL_LOBE",
    "ASEG_LABELS",
    "BRAINSTEM_LOBE_ID",
    "LOBE_IDS",
    "LOBE_LABELS",
    "PARCELLATION_SCHEMES",
    "REGION_CATEGORIES",
    "RIGHT_HALF_OFFSET",
    "RIGHT_LOBE_OFFSET",
    "SCHEME_APARC",
    "SCHEME_GENERIC",
    "SCHEME_LOBES",
    "SCHEME_TOTALSEG",
    "TOTALSEG_BILATERAL",
    "TOTALSEG_NON_PARENCHYMA",
    "VERMIS_LOBE_ID",
    "lateralize_by_midline",
    "lobes_from_aparc",
    "non_parenchyma_ids",
    "parenchyma_mask",
    "region_category",
    "region_side",
    "scheme_label_names",
    "totalseg_brain_structure_labels",
]
