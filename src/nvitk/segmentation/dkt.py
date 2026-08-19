"""Desikan-Killiany-Tourville cortical parcellation via ANTsPyNet."""

from __future__ import annotations

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()

# ---------------------------------------------------------------------------
# Label tables
# ---------------------------------------------------------------------------
#: Desikan–Killiany cortical parcels, in FreeSurfer's ``aparc`` numbering: the offset within a
#: hemisphere. The published ids are ``1000 + offset`` on the left and ``2000 + offset`` on the
#: right, which is the convention used by ``aparc+aseg.nii.gz``, by FreeSurfer's colour LUT, **and**
#: by the label map ANTsPyNet's DKT network emits — so one table serves a generated parcellation and
#: a user-supplied FreeSurfer volume alike.
APARC_PARCEL_OFFSETS: dict[int, str] = {
    1: "bankssts",
    2: "caudalanteriorcingulate",
    3: "caudalmiddlefrontal",
    4: "corpuscallosum",
    5: "cuneus",
    6: "entorhinal",
    7: "fusiform",
    8: "inferiorparietal",
    9: "inferiortemporal",
    10: "isthmuscingulate",
    11: "lateraloccipital",
    12: "lateralorbitofrontal",
    13: "lingual",
    14: "medialorbitofrontal",
    15: "middletemporal",
    16: "parahippocampal",
    17: "paracentral",
    18: "parsopercularis",
    19: "parsorbitalis",
    20: "parstriangularis",
    21: "pericalcarine",
    22: "postcentral",
    23: "posteriorcingulate",
    24: "precentral",
    25: "precuneus",
    26: "rostralanteriorcingulate",
    27: "rostralmiddlefrontal",
    28: "superiorfrontal",
    29: "superiorparietal",
    30: "superiortemporal",
    31: "supramarginal",
    32: "frontalpole",
    33: "temporalpole",
    34: "transversetemporal",
    35: "insula",
}

#: Parcels the DKT protocol does not define. DKT merges ``bankssts`` into the neighbouring temporal
#: parcels and drops both poles, so a DKT label map genuinely has no geometry for them — a
#: measurement in one of these has to be drawn as absent, never approximated onto a neighbour.
#: ``corpuscallosum`` is not cortex at all and is absent from both protocols.
DKT_MISSING_PARCELS: frozenset[str] = frozenset(
    {"bankssts", "frontalpole", "temporalpole", "corpuscallosum"}
)


def _aparc_label_table(*, exclude: frozenset[str] = frozenset()) -> dict[int, str]:
    """``{atlas id: "ctx-lh-<parcel>"}`` over both hemispheres, skipping *exclude*."""
    table: dict[int, str] = {}
    for side, base in (("lh", 1000), ("rh", 2000)):
        for offset, parcel in APARC_PARCEL_OFFSETS.items():
            if parcel in exclude:
                continue
            table[base + offset] = f"ctx-{side}-{parcel}"
    return table


#: Full Desikan–Killiany table, as written by FreeSurfer into ``aparc+aseg``.
APARC_LABELS: dict[int, str] = _aparc_label_table()

#: The subset a DKT parcellation actually contains — see :data:`DKT_MISSING_PARCELS`.
DKT_LABELS: dict[int, str] = _aparc_label_table(exclude=DKT_MISSING_PARCELS)


def desikan_killiany_tourville_labeling(
    image: Image | np.ndarray,
    *,
    do_preprocessing: bool = True,
    do_lobar_parcellation: bool = False,
    do_denoising: bool = True,
    version: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet Desikan-Killiany-Tourville (DKT) labeling on T1w MRI.

    Returns the segmentation label map (probability maps are discarded).
    """
    antspynet = require_antspynet()
    ants_t1 = to_ants_image(image)
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)
    log.info(
        f"DKT labeling: shape={shape}, preprocessing={bool(do_preprocessing)}, "
        f"lobar={bool(do_lobar_parcellation)}, denoising={bool(do_denoising)}, "
        f"version={int(version)}"
    )
    out = antspynet.desikan_killiany_tourville_labeling(
        ants_t1,
        do_preprocessing=bool(do_preprocessing),
        return_probability_images=False,
        do_lobar_parcellation=bool(do_lobar_parcellation),
        do_denoising=bool(do_denoising),
        version=int(version),
        verbose=bool(verbose),
    )
    return ants_result_to_array(out)


__all__ = [
    "APARC_LABELS",
    "APARC_PARCEL_OFFSETS",
    "DKT_LABELS",
    "DKT_MISSING_PARCELS",
    "desikan_killiany_tourville_labeling",
]
