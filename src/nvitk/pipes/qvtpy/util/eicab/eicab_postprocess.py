"""eICAB post-process (stage 1): fit the labels to the TOF, recover the vessels' missing stretches,
and open the ICA siphon loops. Writes the ``*_pp`` mask stage 7 prefers.

On TOF, eICAB's circle-of-Willis labels are too thick and stop short. The post-process runs, in
order (each step can be switched off):

1. **Drop the anterior choroidal arteries** (``remove_ids``, default LACHA / RACHA).
2. **Local re-segmentation** (``resegment``). Each label is re-drawn at its local half maximum,
   from the intensities of that vessel and its surroundings only, with the other labels as
   barriers (:func:`nvitk.segmentation.vessel_refine.resegment_local`).
3. **Trunks to the floor** (``extend``). eICAB's box cuts the ICAs and the basilar at its lower
   face while the lumen goes on. Each trunk follows the minimum-cost path through a multi-scale
   Frangi tree down to the image's last slices
   (:func:`nvitk.segmentation.vessel_refine.extend_trunks`). The basilar follows up to two
   branches, the vertebral arteries. They are then split off with the 4D-flow pipeline's
   vertebral logic (:func:`nvitk.pipes.qvtpy.util.segmentation.vertebral_split.split_vertebral_from_basilar`:
   a centerline Y below the confluence, L/R by side) as LVA / RVA (eICAB ids 19 / 20, new).
4. **Distal growth** (``distal``). The ACAs, MCAs, PCAs (P2) and SCAs grow along the Frangi tree
   beyond eICAB's box, all at once and in competition, each voxel going to the vessel that
   reaches it first (:func:`nvitk.segmentation.vessel_refine.grow_distal`). The fronts advance
   together, step for step, so each of the two A2s, running side by side, is reached first by its
   own ACA (``aca_guard``):
   - each side of the AComm starts that side's front;
   - neither ACA may grow more than 5 mm past a midline plane oriented as the head's mid-sagittal
     plane and anchored at the AComm.
5. **Siphon correction** (``siphon``). Where an ICA siphon's limbs touch and close a loop, only
   the contact is cut (:mod:`nvitk.morphology.siphon_cut`).

``postprocess_eicab_directory`` writes ``eicab_postprocess.json`` beside the mask: per label, the
voxels before and after, and the log of each step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import is_cupy_array, setup, using
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.pipes.qvtpy.labels import (
    EICAB_ACOMM,
    EICAB_BASILAR,
    EICAB_LACA,
    EICAB_LACHA,
    EICAB_LICA,
    EICAB_LMCA,
    EICAB_LPCA_P2,
    EICAB_LSCA,
    EICAB_LVA,
    EICAB_RACA,
    EICAB_RACHA,
    EICAB_RICA,
    EICAB_RMCA,
    EICAB_RPCA_P2,
    EICAB_RSCA,
    EICAB_RVA,
)
from nvitk.pipes.qvtpy.util.centerline.centerline_io import rasterize_centerlines_mask
from nvitk.pipes.qvtpy.util.eicab.eicab_masks import (
    CENTERLINES_MASK_PP_NIFTI,
    eicab_pp_path,
    find_tof_resampled_volume,
    resolve_eicab_mask,
)

setup(globals())

log = Logger()

_DEFAULT_ICA_IDS: tuple[int, ...] = (EICAB_LICA, EICAB_RICA)
#: The trunks eICAB's box cuts at its lower face: both ICAs and the basilar.
_DEFAULT_TRUNK_IDS: tuple[int, ...] = (EICAB_LICA, EICAB_RICA, EICAB_BASILAR)
#: Labels dropped first: the anterior choroidal arteries.
_DEFAULT_REMOVE_IDS: tuple[int, ...] = (EICAB_LACHA, EICAB_RACHA)
#: Labels grown distally along the Frangi tree: ACAs, MCAs, PCAs (P2) and SCAs.
_DEFAULT_DISTAL_IDS: tuple[int, ...] = (EICAB_LACA, EICAB_RACA, EICAB_LMCA, EICAB_RMCA, EICAB_LPCA_P2,
                                        EICAB_RPCA_P2, EICAB_LSCA, EICAB_RSCA)
PP_REPORT_JSON = "eicab_postprocess.json"


def _compute_pp_centerlines(labels: np.ndarray, *, min_points: int = 5) -> np.ndarray:
    """Rasterize centerlines for all foreground labels in the post-processed eICAB mask."""
    shape = tuple(int(s) for s in labels.shape[:3])
    seg_np = as_backend_array(labels).astype(np.int32, copy=False)
    present = [
        int(lid)
        for lid in to_numpy(np.unique(seg_np))
        if int(lid) > 0
    ]
    if not present:
        return rasterize_centerlines_mask(shape, {})
    seed_cls = compute_centerlines(seg_np, labels=present, min_points=int(min_points))
    return rasterize_centerlines_mask(shape, seed_cls)


def _isin(arr: Any, ids: Sequence[int]) -> Any:
    """``np.isin`` against a list of ids (CuPy needs an array, not a list)."""
    if not len(ids):
        return np.zeros(arr.shape, dtype=bool)
    return np.isin(arr, as_backend_array([int(i) for i in ids]))


def _label_counts(seg: Any) -> dict[str, int]:
    """Voxels per non-zero label, as ``{"id": n}`` (one bincount, not one pass per label)."""
    counts = to_numpy(np.bincount(seg.ravel()))
    return {str(i): int(n) for i, n in enumerate(counts) if i > 0 and n > 0}


def _repair_icas_mincut(
    intensity: Any, labels: Any, ica_ids: Sequence[int], affine: Any, *, pretess: bool
) -> dict[str, Any]:
    """Pretess + loop cuts on each ICA of *labels* (in place, host or CuPy); compact per-ICA log.
    The cuts run on the host (:mod:`nvitk.morphology.siphon_cut`); the TOF crosses once."""
    from nvitk.morphology.siphon_cut import repair_label

    inten = to_numpy(intensity)
    out: dict[str, Any] = {}
    for lid in ica_ids:
        if not bool((labels == int(lid)).any()):
            continue
        _m, _cut, entry = repair_label(labels, int(lid), inten, affine=affine,
                                       pretess_mode="remove" if pretess else None, label_name=f"ICA_{int(lid)}")
        rep = entry.get("repair") or {}
        out[str(int(lid))] = {
            "genus_before": entry["genus_before"], "genus_after": entry["genus_after"],
            "pretess": entry.get("pretess"), "loops_cut": rep.get("n_cut_loops", 0),
            "loops_left": rep.get("unresolved", 0), "cut_mm3": entry.get("cut_mm3"),
            "removed_mm3": entry.get("removed_mm3"), "volume_change_pct": entry.get("volume_change_pct"),
        }
    return out


def split_vertebrals(labels: Any) -> tuple[Any, dict[str, Any]]:
    """Split the vertebral arteries off the basilar with the 4D-flow pipeline's logic
    (:func:`~nvitk.pipes.qvtpy.util.segmentation.vertebral_split.split_vertebral_from_basilar`) and
    label them LVA / RVA (eICAB 19 / 20). The ICAs and the basilar share their ids with qvtpy's
    labels, which the split reads for left / right. Nothing changes without a centerline Y."""
    from nvitk.pipes.qvtpy.labels import QVTPY_LVA, QVTPY_RVA
    from nvitk.pipes.qvtpy.util.segmentation.vertebral_split import split_vertebral_from_basilar

    seg = as_backend_array(labels).astype(np.int32)
    trunks = to_numpy(np.where(_isin(seg, [EICAB_LICA, EICAB_RICA, EICAB_BASILAR]), seg, 0))
    with using("cpu"):  # qvtpy's split (skeleton graph) is host code
        split, result = split_vertebral_from_basilar(trunks)
        split = to_numpy(split)
    split = as_backend_array(split)
    out = seg.copy()
    if result.split_applied:
        out[(seg == EICAB_BASILAR) & (split == QVTPY_LVA)] = EICAB_LVA
        out[(seg == EICAB_BASILAR) & (split == QVTPY_RVA)] = EICAB_RVA
    info = {k: v for k, v in result.as_dict().items() if not k.endswith("_centerline")}
    return out, info


def postprocess_eicab_labels(
    intensity: Any,
    labels: np.ndarray,
    *,
    affine: Any | None = None,
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    trunk_ids: Sequence[int] = _DEFAULT_TRUNK_IDS,
    distal_ids: Sequence[int] = _DEFAULT_DISTAL_IDS,
    remove_ids: Sequence[int] = _DEFAULT_REMOVE_IDS,
    resegment: bool = True,
    extend: bool = True,
    vertebrals: bool = True,
    distal: bool = True,
    aca_guard: bool = True,
    siphon: bool = True,
    pretess: bool = True,
) -> dict[str, Any]:
    """Post-process eICAB *labels* against the TOF *intensity* (module docstring); *labels* is
    modified in place. *affine* (voxel → world) gives the voxel size and the superior direction.

    Steps, each switchable: drop *remove_ids*; re-segment every label (*resegment*); extend
    *trunk_ids* to the image floor (*extend*; the basilar into the vertebrals, split off when
    *vertebrals*); grow *distal_ids* along the Frangi tree (*distal*; with *aca_guard*, the ACAs
    start their A2s from their own side of the AComm and may not grow more than 5 mm past the
    midline); siphon loop cuts on *ica_ids* (*siphon*; *pretess* first).

    The volumes work on the active backend (:mod:`nvitk.segmentation.vessel_refine`); the result
    is written back into *labels* whichever kind it is (NumPy or CuPy)."""
    from nvitk.segmentation.vessel_refine import extend_trunks, grow_distal, resegment_local

    t0 = time.time()
    inten = as_backend_array(intensity).astype(np.float64, copy=False)
    seg = as_backend_array(labels).astype(np.int32, copy=True)
    before = _label_counts(seg)
    summary: dict[str, Any] = {"resegment": bool(resegment), "extend": bool(extend), "vertebrals": bool(vertebrals),
                               "distal": bool(distal), "siphon": bool(siphon),
                               "removed_labels": [int(i) for i in remove_ids]}
    if remove_ids:
        seg[_isin(seg, remove_ids)] = 0
    if resegment:
        seg, summary["resegmentation"] = resegment_local(inten, seg, affine=affine)
    if extend:
        branches = {int(EICAB_BASILAR): 2} if vertebrals else {}
        seg, summary["extension"] = extend_trunks(inten, seg, trunk_ids, affine=affine, floor_branches=branches,
                                                  split_ratio={int(i): (0.5 if int(i) in ica_ids else 0.2)
                                                               for i in trunk_ids})
        if vertebrals and int(EICAB_BASILAR) in [int(i) for i in trunk_ids]:
            seg, summary["vertebral_split"] = split_vertebrals(seg)
    if distal:
        pairs = [(int(EICAB_LACA), int(EICAB_RACA), int(EICAB_ACOMM))] if aca_guard else []
        refs = [(int(EICAB_LICA), int(EICAB_RICA)), (int(EICAB_LMCA), int(EICAB_RMCA)),
                (int(EICAB_LPCA_P2), int(EICAB_RPCA_P2)), (int(EICAB_LSCA), int(EICAB_RSCA))]
        seg, summary["distal_growth"] = grow_distal(inten, seg, distal_ids, affine=affine, hemisphere_pairs=pairs,
                                                    midline_refs=refs)
    if siphon:
        summary["siphon_correction"] = _repair_icas_mincut(inten, seg, ica_ids, affine, pretess=pretess)
    after = _label_counts(seg)
    summary["voxels"] = {k: {"before": before.get(k, 0), "after": after.get(k, 0)}
                         for k in sorted(set(before) | set(after), key=int)}
    summary["seconds"] = round(time.time() - t0, 1)
    labels[...] = as_backend_array(seg, backend="cupy" if is_cupy_array(labels) else "numpy")
    return summary


def postprocess_eicab_directory(
    eicab_dir: Path,
    *,
    tof_path: Path | None = None,
    preference: str = "cw",
    resegment: bool = True,
    extend: bool = True,
    distal: bool = True,
    siphon: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load eICAB mask + TOF, write the ``*_pp`` mask, ``centerlines_mask_pp.nii.gz`` and
    ``eicab_postprocess.json`` (:func:`postprocess_eicab_labels`; *kwargs* go there too)."""
    eicab_dir = Path(eicab_dir)
    res = resolve_eicab_mask(
        eicab_dir,
        preference=preference,
        prefer_postprocessed=False,
    )
    intensity_p = Path(tof_path) if tof_path is not None else find_tof_resampled_volume(eicab_dir)
    if intensity_p is None or not intensity_p.is_file():
        raise FileNotFoundError(
            f"No TOF intensity for eICAB post-process under {eicab_dir} "
            "(expected TOF_resampled.nii.gz)."
        )

    seg_img = imread(res.path)
    tof_img = imread(intensity_p)
    labels = to_numpy(seg_img.data).astype(np.int32, copy=True)
    wvi = to_numpy(tof_img.data)
    if labels.shape[:3] != wvi.shape[:3]:
        raise ValueError(
            f"Shape mismatch: eICAB {labels.shape[:3]} vs TOF {wvi.shape[:3]} "
            f"({res.path.name} vs {intensity_p.name})"
        )

    log.info(f"eICAB post-process: mask={res.path.name} intensity={intensity_p.name}")
    summary = postprocess_eicab_labels(
        wvi,
        labels,
        affine=seg_img.affine if seg_img.affine is not None else tof_img.affine,
        resegment=resegment,
        extend=extend,
        distal=distal,
        siphon=siphon,
        **kwargs,
    )
    pp_path = eicab_pp_path(res.path)
    cl_pp_path = eicab_dir / CENTERLINES_MASK_PP_NIFTI
    cl_pp = _compute_pp_centerlines(labels)

    summary["mask_path"] = str(res.path)
    summary["mask_pp_path"] = str(pp_path)
    summary["centerlines_mask_pp_path"] = str(cl_pp_path)
    summary["intensity_path"] = str(intensity_p)

    imsave(pp_path, labels, metadata=dict(seg_img.metadata or {}))
    imsave(cl_pp_path, cl_pp, metadata=dict(seg_img.metadata or {}))
    report = eicab_dir / PP_REPORT_JSON
    report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["report_path"] = str(report)
    log.ok(f"eICAB post-process written: {pp_path} + {cl_pp_path.name} + {report.name}")
    return summary


__all__ = [
    "PP_REPORT_JSON",
    "postprocess_eicab_directory",
    "postprocess_eicab_labels",
    "split_vertebrals",
]
