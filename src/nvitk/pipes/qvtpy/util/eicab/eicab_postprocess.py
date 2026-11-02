"""eICAB post-process (stage 1): fit the labels to the TOF, recover the vessels' missing stretches,
and open the ICA siphon loops. Writes two masks (:func:`postprocess_eicab_directory`):

- ``*_eICAB_CW_pp``: steps 1-3 and 5, eICAB's labels without the distal growth (the ACAs / MCAs
  read as A1 / M1);
- ``*_eICAB_WB_pp`` (with *distal*): steps 1-5. What the ACAs / MCAs gain in step 4 is written as
  their A2 / M2 (eICAB ids 21-24), so they split as the PCA does into P1 / P2.

Stage 7 measures the WB_pp mask when it exists, else the CW_pp one.

eICAB resamples the TOF without the rotation of an oblique acquisition's affine (it keeps the voxel
sizes and the origin), so its ``TOF_resampled`` and masks sit up to ~2 cm off the acquired
``TOF.nii.gz`` in world space. When the original TOF is given, the pp masks are written with the
scanner-frame affine instead (:func:`scanner_frame_affine`): same voxels, true positions. The
pipeline's own registration (FLIRT, stage 2/3) works on voxels and is unaffected.

On TOF, eICAB's circle-of-Willis labels are too thick and stop short. The post-process runs, in
order (each step can be switched off):

1. **Drop the anterior choroidal arteries** (``remove_ids``, default LACHA / RACHA).
2. **Local re-segmentation** (``resegment``). Each label is re-drawn at its local half maximum,
   from the intensities of that vessel and its surroundings only, with the other labels as
   barriers (:func:`nvitk.segmentation.vessel_refine.resegment_local`).
3. **ICAs to the floor** (``extend``). eICAB's box cuts the ICAs at its lower face while the
   lumen goes on. Each ICA follows the minimum-cost path through a multi-scale Frangi tree down to
   the image's lowest slice (:func:`nvitk.segmentation.vessel_refine.extend_trunks`). The basilar
   is left as eICAB drew it (re-segmented only): no extension, no vertebral arteries.
   Then the **ICA loops below the siphon** (``loop_fill``): in a tight loop the blood runs along the
   outer wall and the inner wall comes out darker on TOF, a crescent the mask misses. Along each
   ICA's centerline from the floor up to 2 mm before its siphon (the siphon: the cluster of
   bends, >= 0.20 / mm at a 3 mm scale as in the morphometrics, around its sharpest bend in the
   upper 60 % of the ICA), the cross-sections' notches at least 0.35 mm deep are filled; the
   outer wall does not move (:func:`nvitk.segmentation.cross_section_fill.fill_cross_section_notches`).
4. **Distal growth** (``distal``, WB_pp only). The ACAs, MCAs, PCAs (P2) and SCAs grow along the Frangi tree
   beyond eICAB's box, all at once and in competition, each voxel going to the vessel that
   reaches it first (:func:`nvitk.segmentation.vessel_refine.grow_distal`). The fronts advance
   together, step for step, so each of the two A2s, running side by side, is reached first by its
   own ACA (``aca_guard``):
   - each side of the AComm starts that side's front;
   - neither ACA may grow more than 5 mm past a midline plane oriented as the head's mid-sagittal
     plane and anchored at the AComm.
5. **Siphon correction** (``siphon``). Where an ICA siphon's limbs touch and close a loop, only
   the contact is cut (:mod:`nvitk.morphology.siphon_cut`). It runs on each mask separately, so the
   distal growth never fills a cut.

After each step (re-segmentation, trunks, distal growth, siphon) every label is reduced to its
largest connected component and its enclosed holes are filled
(:func:`nvitk.segmentation.vessel_refine.tidy_labels`; unlabelled voxels only; an A2 / M2 keeps
every piece on its A1 / M1). No step smooths the image beyond the light denoising of the
half-maximum threshold, which would merge touching siphon limbs.

``postprocess_eicab_directory`` writes ``eicab_postprocess.json`` beside the masks: the shared
steps once, then a ``cw`` and a ``wb`` section (per label, the voxels before and after, and the
log of each step).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as numpy_host  # host geometry (module ``np`` may be a CuPy proxy)

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import is_cupy_array, setup, using
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.pipes.qvtpy.labels import (
    EICAB_DISTAL_SPLIT,
    EICAB_ACOMM,
    EICAB_LACA,
    EICAB_LACHA,
    EICAB_LICA,
    EICAB_LMCA,
    EICAB_LPCA_P2,
    EICAB_LSCA,
    EICAB_RACA,
    EICAB_RACHA,
    EICAB_RICA,
    EICAB_RMCA,
    EICAB_RPCA_P2,
    EICAB_RSCA,
)
from nvitk.pipes.qvtpy.util.centerline.centerline_io import rasterize_centerlines_mask
from nvitk.pipes.qvtpy.util.eicab.eicab_masks import (
    CENTERLINES_MASK_PP_NIFTI,
    eicab_pp_variant_path,
    find_tof_resampled_volume,
    resolve_eicab_mask,
)

setup(globals())

log = Logger()

_DEFAULT_ICA_IDS: tuple[int, ...] = (EICAB_LICA, EICAB_RICA)
#: Labels dropped first: the anterior choroidal arteries.
_DEFAULT_REMOVE_IDS: tuple[int, ...] = (EICAB_LACHA, EICAB_RACHA)
#: Labels grown distally along the Frangi tree: ACAs, MCAs, PCAs (P2) and SCAs.
_DEFAULT_DISTAL_IDS: tuple[int, ...] = (EICAB_LACA, EICAB_RACA, EICAB_LMCA, EICAB_RMCA, EICAB_LPCA_P2,
                                        EICAB_RPCA_P2, EICAB_LSCA, EICAB_RSCA)
PP_REPORT_JSON = "eicab_postprocess.json"
#: The ICA siphon, for the loop fill: bends of at least this curvature (1/mm, at a 3 mm scale),
#: grouped around the sharpest one in the ICA's upper 60 % with the bends up to 8 mm before it.
SIPHON_KAPPA = 0.20
SIPHON_SMOOTH_MM = 3.0
SIPHON_GAP_MM = 8.0
#: The loop fill stops this far (mm) before the siphon.
LOOP_FILL_MARGIN_MM = 2.0
#: After each step, a label's pieces smaller than this (mm³) go: the loose voxels a growth step
#: leaves around a vessel. Real separate pieces (an M2 branch on its M1) are ~20 mm³ and more.
MIN_ISLAND_MM3 = 5.0


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
    """Pretess + loop cuts on each ICA of *labels* (:func:`nvitk.morphology.siphon_cut.repair_icas`,
    in place); compact per-ICA log."""
    from nvitk.morphology.siphon_cut import repair_icas

    return repair_icas(intensity, labels, ica_ids, affine=affine, pretess=pretess)


def _ica_below_siphon(mask: Any, affine: Any) -> tuple[Any, dict[str, Any]]:
    """The centerline of one ICA (host *mask*) from its lowest point up to ``LOOP_FILL_MARGIN_MM``
    before its siphon (module docstring, step 3), in mm on the voxel axes; ``None`` without a
    centerline or a siphon. Host code."""
    from nvitk.measure.morpho.geometry import cumulative_s, resample_polyline_by_arclength
    from nvitk.measure.morpho.metrics import smoothed_curvature
    from nvitk.morphology.centerline import compute_centerlines

    with using("cpu"):
        cl = compute_centerlines(mask.astype(numpy_host.int32), labels=[1], min_points=10).get(1)
    if cl is None or len(cl) < 10:
        return None, {"reason": "no centerline"}
    aff = numpy_host.asarray(to_numpy(affine), dtype=float) if affine is not None else numpy_host.eye(4)
    sp = numpy_host.sqrt((aff[:3, :3] ** 2).sum(0))
    cl = numpy_host.asarray(to_numpy(cl), dtype=float)
    if float(cl[0] @ aff[2, :3]) > float(cl[-1] @ aff[2, :3]):
        cl = cl[::-1]  # the inferior end first
    pts = resample_polyline_by_arclength(cl * sp, 0.25)
    s = cumulative_s(pts)
    k = smoothed_curvature(pts, SIPHON_SMOOTH_MM)
    upper = s >= 0.4 * s[-1]
    i_max = int(numpy_host.argmax(numpy_host.where(upper, k, -1.0)))
    if k[i_max] < SIPHON_KAPPA:
        return None, {"reason": "no siphon bend", "length_mm": round(float(s[-1]), 1)}
    high = numpy_host.concatenate([[False], k >= SIPHON_KAPPA, [False]])
    edges = numpy_host.flatnonzero(numpy_host.diff(high.astype(int)))
    runs = [(s[a], s[b - 1]) for a, b in zip(edges[::2], edges[1::2])]
    start = next(a for a, b in runs if a <= s[i_max] <= b)
    for a, b in reversed([r for r in runs if r[1] < start]):
        if start - b > SIPHON_GAP_MM:
            break
        start = a
    end = start - LOOP_FILL_MARGIN_MM
    info = {"length_mm": round(float(s[-1]), 1), "siphon_start_mm": round(float(start), 1),
            "filled_stretch_mm": round(max(0.0, float(end)), 1)}
    return (pts[s <= end] if end > 1.0 else None), info


def _fill_ica_loops(labels: Any, ica_ids: Sequence[int], affine: Any) -> dict[str, Any]:
    """Fill the cross-sectional notches of each ICA below its siphon (module docstring, step 3),
    *labels* in place (host or CuPy). Host geometry; the label volume crosses once."""
    from nvitk.segmentation.cross_section_fill import fill_cross_section_notches
    from nvitk.segmentation.vessel_refine import spacing_of

    host = to_numpy(labels).astype(numpy_host.int32)
    sp = spacing_of(affine)
    out: dict[str, Any] = {}
    for lid in ica_ids:
        mask = host == int(lid)
        if not mask.any():
            continue
        path, info = _ica_below_siphon(mask, affine)
        if path is not None:
            add, counts = fill_cross_section_notches(mask, path, sp, blocked=host != 0)
            host[add] = int(lid)
            info.update(counts, voxels_added=int(add.sum()))
        out[str(int(lid))] = info
    labels[...] = as_backend_array(host, backend="cupy" if is_cupy_array(labels) else "numpy")
    return out


def postprocess_eicab_variants(
    intensity: Any,
    labels: Any,
    *,
    affine: Any | None = None,
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    distal_ids: Sequence[int] = _DEFAULT_DISTAL_IDS,
    remove_ids: Sequence[int] = _DEFAULT_REMOVE_IDS,
    resegment: bool = True,
    extend: bool = True,
    loop_fill: bool = True,
    distal: bool = True,
    cw: bool = True,
    aca_guard: bool = True,
    siphon: bool = True,
    pretess: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Post-process eICAB *labels* against the TOF *intensity* (module docstring) into its two
    masks: ``{"cw": …, "wb": …}`` (``"wb"`` only with *distal*), int32 on the active backend, and
    the report. *affine* (voxel -> world) gives the voxel size and the superior direction.

    Shared steps, each switchable: drop *remove_ids*; re-segment every label (*resegment*); extend
    the ICAs (*ica_ids*) to the image floor (*extend*); fill the notches of the ICAs' loops below
    the siphon (*loop_fill*). Then the CW mask gets the siphon loop cuts on
    *ica_ids* (*siphon*; *pretess* first), and the WB mask the distal growth of *distal_ids* along
    the Frangi tree first (with *aca_guard*, the ACAs start their A2s from their own side of the
    AComm and may not grow more than 5 mm past the midline; the ACAs' and MCAs' growth is written
    as A2 / M2) and then the same cuts. *cw* False skips the CW mask (only with *distal*)."""
    from nvitk.segmentation.vessel_refine import (
        drop_small_islands,
        extend_trunks,
        grow_distal,
        resegment_local,
        spacing_of,
        tidy_labels,
    )

    t0 = time.time()
    inten = as_backend_array(intensity).astype(np.float64, copy=False)
    seg = as_backend_array(labels).astype(np.int32, copy=True)
    before = _label_counts(seg)
    summary: dict[str, Any] = {"resegment": bool(resegment), "extend": bool(extend), "loop_fill": bool(loop_fill),
                               "distal": bool(distal), "siphon": bool(siphon),
                               "removed_labels": [int(i) for i in remove_ids]}
    if remove_ids:
        seg[_isin(seg, remove_ids)] = 0
    tidy_log: dict[str, Any] = summary.setdefault("tidy", {})

    min_island = math.ceil(MIN_ISLAND_MM3 / math.prod(spacing_of(affine)))

    def _tidy(vol: Any, step: str) -> Any:
        """Every label to its largest component (an A2 / M2: its pieces on the A1 / M1), holes
        filled, pieces under MIN_ISLAND_MM3 dropped, after a step (logged per step)."""
        vol, tidy_log[step] = tidy_labels(vol, anchors={int(d): int(p) for p, d in EICAB_DISTAL_SPLIT.items()})
        vol, islands = drop_small_islands(vol, min_voxels=min_island)
        if islands:
            tidy_log[step] = {**tidy_log[step], "islands_cleared": islands}
        return vol

    if resegment:
        seg, summary["resegmentation"] = resegment_local(inten, seg, affine=affine)
        seg = _tidy(seg, "resegment")
    if extend:
        seg, summary["extension"] = extend_trunks(inten, seg, ica_ids, affine=affine,
                                                  split_ratio={int(i): 0.5 for i in ica_ids},
                                                  edge_ids=[int(i) for i in ica_ids])
        seg = _tidy(seg, "extend")
    if loop_fill:
        summary["loop_fill"] = _fill_ica_loops(seg, ica_ids, affine)

    def _tail(vol: Any, key: str) -> Any:
        """The siphon cuts and the voxel counts of one variant (*vol* in place)."""
        part: dict[str, Any] = summary.setdefault(key, {})
        if siphon:
            part["siphon_correction"] = _repair_icas_mincut(inten, vol, ica_ids, affine, pretess=pretess)
        vol[...] = _tidy(vol, f"{key}_final")
        after = _label_counts(vol)
        part["voxels"] = {k: {"before": before.get(k, 0), "after": after.get(k, 0)}
                          for k in sorted(set(before) | set(after), key=int)}
        return vol

    out: dict[str, Any] = {}
    if distal:
        pairs = [(int(EICAB_LACA), int(EICAB_RACA), int(EICAB_ACOMM))] if aca_guard else []
        refs = [(int(EICAB_LICA), int(EICAB_RICA)), (int(EICAB_LMCA), int(EICAB_RMCA)),
                (int(EICAB_LPCA_P2), int(EICAB_RPCA_P2)), (int(EICAB_LSCA), int(EICAB_RSCA))]
        wb, summary.setdefault("wb", {})["distal_growth"] = grow_distal(
            inten, seg.copy(), distal_ids, affine=affine, hemisphere_pairs=pairs, midline_refs=refs,
            relabel=dict(EICAB_DISTAL_SPLIT))
        out["wb"] = _tail(_tidy(wb, "distal"), "wb")
    if cw or not distal:
        out["cw"] = _tail(seg, "cw")
    summary["seconds"] = round(time.time() - t0, 1)
    return out, summary


def postprocess_eicab_labels(intensity: Any, labels: np.ndarray, *, distal: bool = True, **kwargs: Any) -> dict[str, Any]:
    """One post-processed mask (:func:`postprocess_eicab_variants`) written back into *labels*
    (NumPy or CuPy): the distal-grown WB one with *distal* (its A2 / M2 as ids 21-24), else the CW
    one. Only that mask is computed. Returns the report."""
    masks, summary = postprocess_eicab_variants(intensity, labels, distal=distal, cw=not distal, **kwargs)
    seg = masks["wb" if distal else "cw"]
    labels[...] = as_backend_array(seg, backend="cupy" if is_cupy_array(labels) else "numpy")
    return summary


def scanner_frame_affine(
    affine: Any, tof_original: Any, tof_original_affine: Any, *, intensity: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The scanner-frame affine of eICAB's grid: eICAB drops the rotation of the original TOF's
    affine (``A``) and keeps its voxel sizes and origin (``A0``), so a voxel's true position is
    ``A · A0⁻¹ · affine``. With *intensity* (``TOF_resampled``), the model is checked first: the
    original TOF sampled through each frame is correlated with it, and the correction applies only
    when the de-rotated frame matches better. Returns the affine (unchanged when not applied) and a
    report (``correction``: the 4 × 4 matrix to left-multiply eICAB's affines with). A flipped
    (improper) direction matrix is left alone. Host only."""
    from scipy import ndimage as ndi_h  # host sampling of one volume, once

    with using("cpu"):
        aff = to_numpy(affine).astype(float)
        a_full = to_numpy(tof_original_affine).astype(float)
        zooms = np.sqrt((a_full[:3, :3] ** 2).sum(0))
        a0 = np.eye(4)
        a0[:3, :3] = np.diag(zooms)
        a0[:3, 3] = a_full[:3, 3]
        rot = a_full[:3, :3] / zooms
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))
        info: dict[str, Any] = {"rotation_deg": round(angle, 2), "applied": False}
        if np.linalg.det(rot) < 0:  # a flip, not a rotation: FLIRT would mirror the labels
            return affine, {**info, "reason": "the TOF's direction matrix is improper (a flip); left as eICAB wrote it"}
        if angle < 0.05:
            return affine, {**info, "reason": "no rotation in the TOF's affine"}
        if intensity is not None:
            img = to_numpy(intensity)
            org = to_numpy(tof_original)
            rng = np.random.default_rng(0)
            idx = np.stack([rng.integers(0, n, 60000) for n in img.shape[:3]], 1)
            q = img[tuple(idx.T)].astype(float)
            corr = {}
            for name, a_src in (("eicab", a0), ("scanner", a_full)):
                src = (np.c_[idx, np.ones(len(idx))] @ (np.linalg.inv(a_src) @ aff).T)[:, :3]
                v = ndi_h.map_coordinates(org, src.T, order=1, mode="constant", cval=0.0)
                m = (q > 0) & (v > 0)
                corr[name] = float(np.corrcoef(q[m], v[m])[0, 1]) if m.sum() > 1000 else float("nan")
            info["correlation"] = {k: round(v, 3) for k, v in corr.items()}
            if not corr["eicab"] > corr["scanner"]:
                return affine, {**info, "reason": "TOF_resampled already matches the scanner frame"}
        correction = a_full @ np.linalg.inv(a0)
        # Stored so eICAB's own files (TOF_resampled, its masks) can be shown in the same frame.
        info.update(applied=True, correction=[[round(float(v), 9) for v in row] for row in correction])
        return correction @ aff, info


def postprocess_eicab_directory(
    eicab_dir: Path,
    *,
    tof_path: Path | None = None,
    tof_original_path: Path | None = None,
    preference: str = "cw",
    resegment: bool = True,
    extend: bool = True,
    distal: bool = True,
    siphon: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load the eICAB mask + TOF and write ``*_eICAB_CW_pp`` (always), ``*_eICAB_WB_pp`` (with
    *distal*; a stale one is removed without), ``centerlines_mask_pp.nii.gz`` (of the CW_pp mask)
    and ``eicab_postprocess.json`` (:func:`postprocess_eicab_variants`; *kwargs* go there too).

    *tof_original_path* (the acquired ``TOF.nii.gz``) gives the scanner frame the masks are written
    in (:func:`scanner_frame_affine`)."""
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
    affine = seg_img.affine if seg_img.affine is not None else tof_img.affine
    frame: dict[str, Any] = {"applied": False}
    meta = dict(seg_img.metadata or {})
    if tof_original_path is not None and Path(tof_original_path).is_file():
        orig = imread(tof_original_path)
        true_aff, frame = scanner_frame_affine(affine, to_numpy(orig.data), orig.affine, intensity=wvi)
        if frame.get("applied"):
            meta["affine"] = to_numpy(true_aff)
            log.info(f"eICAB post-process: masks written in the scanner frame of {Path(tof_original_path).name} "
                     f"(eICAB dropped a {frame['rotation_deg']}° rotation)")
    elif tof_original_path is not None:
        log.warning(f"eICAB post-process: original TOF {tof_original_path} not found; "
                    "the masks keep eICAB's (de-rotated) frame")

    log.info(f"eICAB post-process: mask={res.path.name} intensity={intensity_p.name}")
    masks, summary = postprocess_eicab_variants(
        wvi,
        labels,
        affine=affine,
        resegment=resegment,
        extend=extend,
        distal=distal,
        siphon=siphon,
        **kwargs,
    )
    summary["scanner_frame"] = frame
    summary["mask_path"] = str(res.path)
    summary["intensity_path"] = str(intensity_p)
    if tof_original_path is not None:
        summary["tof_original_path"] = str(tof_original_path)
    written: list[str] = []
    for kind in ("cw", "wb"):
        path = eicab_pp_variant_path(res.path, kind)  # type: ignore[arg-type]
        if kind in masks:
            imsave(path, to_numpy(masks[kind]).astype(np.int32), metadata=meta)
            summary[kind]["mask_pp_path"] = str(path)
            written.append(path.name)
        elif path.is_file():  # left by an earlier run with the distal growth: it would be measured
            path.unlink()
            log.info(f"eICAB post-process: removed the stale {path.name} (distal growth off)")
    summary["mask_pp_path"] = summary["cw"]["mask_pp_path"]

    cl_pp_path = eicab_dir / CENTERLINES_MASK_PP_NIFTI
    imsave(cl_pp_path, _compute_pp_centerlines(masks["cw"]), metadata=meta)
    summary["centerlines_mask_pp_path"] = str(cl_pp_path)
    report = eicab_dir / PP_REPORT_JSON
    report.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["report_path"] = str(report)
    log.ok(f"eICAB post-process written: {' + '.join(written)} + {cl_pp_path.name} + {report.name}")
    return summary


__all__ = [
    "PP_REPORT_JSON",
    "scanner_frame_affine",
    "postprocess_eicab_variants",
    "postprocess_eicab_directory",
    "postprocess_eicab_labels",
]
