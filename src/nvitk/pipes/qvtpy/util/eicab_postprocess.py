"""ICA Otsu resegmentation and region growing after eICAB (stage 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.morphology.centerline_siphon import (
    EROSION_ITERS,
    compute_mask_genus,
    ica_otsu_mask,
)
from nvitk.pipes.qvtpy.labels import EICAB_LICA, EICAB_RICA
from nvitk.pipes.qvtpy.util.centerline_io import rasterize_centerlines_mask
from nvitk.pipes.qvtpy.util.eicab_masks import find_tof_resampled_volume, resolve_eicab_mask
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    _DEFAULT_RG_INTENSITY_FRAC,
    _dilated_other_segmentation_barrier,
)
from nvitk.segmentation.region_growing import region_grow_into_label_volume

setup(globals())

log = Logger()

_DEFAULT_ICA_IDS: tuple[int, ...] = (EICAB_LICA, EICAB_RICA)


def _merge_ica_masks_into_labels(
    labels: np.ndarray,
    ica_masks: dict[int, np.ndarray],
    ica_ids: Sequence[int],
) -> None:
    for lid in ica_ids:
        rep = ica_masks.get(int(lid))
        if rep is None:
            continue
        rep_np = as_backend_array(rep).astype(bool)
        if not rep_np.any():
            continue
        labels[labels == int(lid)] = 0
        labels[rep_np] = int(lid)


def _seed_centerline_mask(
    labels: np.ndarray,
    ica_ids: Sequence[int],
    *,
    min_points: int = 5,
) -> np.ndarray:
    shape = tuple(int(s) for s in labels.shape[:3])
    seed_cls = compute_centerlines(
        labels.astype(np.int32, copy=False),
        labels=[int(lid) for lid in ica_ids],
        min_points=int(min_points),
    )
    return rasterize_centerlines_mask(shape, seed_cls)


def resegment_icas_otsu(
    intensity: Any,
    labels: np.ndarray,
    *,
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    cl_mask: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Otsu-resegment each ICA; erode only when the raw eICAB ICA mask is suspect."""
    wvi = as_backend_array(intensity)
    seg_np = as_backend_array(labels).astype(np.int32)
    clm = cl_mask if cl_mask is not None else _seed_centerline_mask(seg_np, ica_ids)

    out: dict[int, np.ndarray] = {}
    for lid in ica_ids:
        name = f"ICA_{int(lid)}"
        raw = as_backend_array(seg_np == int(lid)).astype(bool)
        if not raw.any():
            log.warning(f"[{name}] empty eICAB mask — skipping Otsu")
            continue
        rep = compute_mask_genus(raw, label_name=name)
        erode_iters = int(EROSION_ITERS) if rep.suspect else 0
        log.step(
            f"[{name}] Otsu resegment erode_iters={erode_iters} "
            f"(raw β₁={rep.beta1} suspect={rep.suspect})"
        )
        eroded, _pre, info = ica_otsu_mask(
            wvi, clm, int(lid), erode_iters=erode_iters
        )
        if info.get("warning"):
            log.warning(f"[{name}] Otsu: {info['warning']}")
            continue
        out[int(lid)] = to_numpy(eroded).astype(bool)
    return out


def region_grow_icas(
    intensity: Any,
    labels: np.ndarray,
    *,
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    rg_barrier_radius: int = 1,
    intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
) -> dict[int, int]:
    """Grow ICA labels into empty voxels; barriers are other labels dilated by *rg_barrier_radius*."""
    int_np = as_backend_array(intensity).astype(np.float64)
    seg_np = as_backend_array(labels).astype(np.int32)
    added: dict[int, int] = {}
    for lid in ica_ids:
        if not np.any(seg_np == int(lid)):
            continue
        forbidden = _dilated_other_segmentation_barrier(
            seg_np, int(lid), radius=int(rg_barrier_radius)
        )
        n = region_grow_into_label_volume(
            seg_np,
            int_np,
            int(lid),
            intensity_frac=float(intensity_frac),
            forbidden=forbidden,
            polarity="hyperintense",
        )
        added[int(lid)] = int(n)
        log.step(f"[ICA id={lid}] region growing added {n} voxels")
    return added


def postprocess_eicab_labels(
    intensity: Any,
    labels: np.ndarray,
    *,
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    rg_barrier_radius: int = 1,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
) -> dict[str, Any]:
    """Otsu ICA masks (conditional erosion) then hyperintense region growing."""
    seg_np = as_backend_array(labels).astype(np.int32)
    ica_masks = resegment_icas_otsu(intensity, seg_np, ica_ids=ica_ids)
    _merge_ica_masks_into_labels(seg_np, ica_masks, ica_ids)
    rg_added = region_grow_icas(
        intensity,
        seg_np,
        ica_ids=ica_ids,
        rg_barrier_radius=rg_barrier_radius,
        intensity_frac=rg_intensity_frac,
    )
    labels[:] = to_numpy(seg_np)
    return {
        "ica_otsu_voxels": {str(k): int(v.sum()) for k, v in ica_masks.items()},
        "region_grow_added": {str(k): v for k, v in rg_added.items()},
    }


def postprocess_eicab_directory(
    eicab_dir: Path,
    *,
    tof_path: Path | None = None,
    preference: str = "cw",
    ica_ids: Sequence[int] = _DEFAULT_ICA_IDS,
    rg_barrier_radius: int = 1,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
) -> dict[str, Any]:
    """Load eICAB mask + TOF, post-process in place, return summary metadata."""
    eicab_dir = Path(eicab_dir)
    res = resolve_eicab_mask(eicab_dir, preference=preference)  # type: ignore[arg-type]
    intensity_p = Path(tof_path) if tof_path is not None else find_tof_resampled_volume(eicab_dir)
    if intensity_p is None or not intensity_p.is_file():
        raise FileNotFoundError(
            f"No TOF intensity for eICAB post-process under {eicab_dir} "
            "(expected TOF_resampled.nii.gz)."
        )

    seg_img = imread(res.path)
    tof_img = imread(intensity_p)
    labels = to_numpy(seg_img.data).astype(np.int32)
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
        ica_ids=ica_ids,
        rg_barrier_radius=rg_barrier_radius,
        rg_intensity_frac=rg_intensity_frac,
    )
    summary["mask_path"] = str(res.path)
    summary["intensity_path"] = str(intensity_p)
    imsave(res.path, labels, metadata=dict(seg_img.metadata or {}))
    log.ok(f"eICAB post-process written: {res.path}")
    return summary


__all__ = [
    "postprocess_eicab_directory",
    "postprocess_eicab_labels",
    "region_grow_icas",
    "resegment_icas_otsu",
]
