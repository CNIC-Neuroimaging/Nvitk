"""CT-PET v5 stage 2 (per-subject): post-process TotalSegmentator outputs.

Port of :code:`BioImaging/src/pesa_fat/ct_pet/__post_process.py` /
:code:`2_post_processing.py`.

Produces the following per-subject label files under
``RESULTS/<batch>/res_post_processing_ct/<SUBJECT>/CT/``::

    MO.nii.gz       L4=1, L3=2                                          (bone narrow)
    FAT.nii.gz      GRASA_V=1, GRASA_SC=2                               (cleaned fat)
    BODY.nii.gz     BODY=1                                              (trunk+ext)
    ORGANS.nii.gz   HIGADO=1, BAZO=2, PANCREAS=3
    MUSCLES.nii.gz  CUADRICEPS_L=1, CUADRICEPS_R=2, PARAVERTEBRAL_L=3,
                    PARAVERTEBRAL_R=4, DELTOIDES_L=5, DELTOIDES_R=6,
                    TRAPECIOS=7                                         (v5 hemisphere-split)

v5 changes
----------
* Muscles are kept hemisphere-split; there is no L+R merging. Deltoid
  (bilateral TS label 9) is split into L/R via
  :func:`nvitk.segmentation.hemisphere.split_lr_by_cc`. Trapezius (14)
  remains a single bilateral mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from nvitk.core.backend import (
    get_current_backend,
    set_global_backend,
    setup,
    to_cupy,
    to_numpy,
    using,
)
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.morphology import dilate, erode, fill_holes
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, layout, resolve_nii
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
from nvitk.pipes.pesa_fat.ct_pet_v5.labels import (
    BODY_LABELS,
    FAT_LABELS,
    MO_LABELS,
    MUSCLES_LABELS,
    ORGANS_LABELS,
)
from nvitk.segmentation.hemisphere import split_lr_by_cc
from nvitk.segmentation.labels import biggest_cc, combine_labels, get_label
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.transform import resample_pet_to_mask
from nvitk.types import Image

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BN_ERODE = 5
_BODY_DILATATION = 5
_ORGANS_TO_REMOVE = (
    "kidney_right",
    "kidney_left",
    "small_bowel",
    "colon",
    "urinary_bladder",
    "liver",
)
_ORGANS_TO_DILATE = {
    "kidney_right": 5,
    "kidney_left": 5,
    "liver": 1,
    "urinary_bladder": 7,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _spacing(img: Image) -> tuple[float, float, float]:
    sp = img.metadata.get("spacing") if img.metadata else None
    if sp is None:
        sp = (
            img.metadata.get("x_res", 1.0) if img.metadata else 1.0,
            img.metadata.get("y_res", 1.0) if img.metadata else 1.0,
            img.metadata.get("z_res", 1.0) if img.metadata else 1.0,
        )
    return tuple(float(v) for v in sp)


def _chull_organ_3d(organ: Image) -> Image:
    """Slicewise 2D convex hull along Z (falls back to the binary mask if skimage missing)."""
    try:
        from skimage.morphology import convex_hull_image
    except Exception:
        return organ

    host = to_numpy(organ.data).astype("uint8")
    for i in range(host.shape[-1]):
        if host[..., i].any():
            host[..., i] = convex_hull_image(host[..., i])
    out = host
    if get_current_backend() == "cupy":
        out = to_cupy(out)
    return organ.with_data(out.astype(np.uint8))


def _process_bladder(bladder: Image, pet: Image) -> Image:
    """Expand the urinary bladder seeds along high-uptake PET signal."""
    pet_to_mask = resample_pet_to_mask(pet, bladder, order=1)
    seeds = pet_to_mask.data[bladder.data > 0]
    if seeds.size == 0:
        return bladder

    with using("numpy"):
        from skimage import filters

        pet_host = to_numpy(pet_to_mask.data)
        thr = max(float(filters.threshold_otsu(pet_host)), 2.5)

    potential = (pet_to_mask.data >= thr).astype(np.uint8)
    labeled, _ = ndi.label(potential, structure=np.ones((3, 3, 3)))
    overlap_ids = np.unique(labeled[bladder.data > 0])
    overlap_ids = overlap_ids[overlap_ids != 0]
    expanded = np.isin(labeled, overlap_ids).astype(np.uint8)
    filled = fill_holes(expanded, axis=2)
    return bladder.with_data(filled.astype(np.uint8))


def _vertebra_narrow(vertebra: Image, radius: int = _BN_ERODE) -> Image:
    """Return the bone narrow (eroded biggest CC) of a vertebra mask."""
    binary = (vertebra.data > 0).astype(np.uint8)
    eroded = erode(vertebra.with_data(binary), footprint=radius)
    return biggest_cc(eroded)


# ---------------------------------------------------------------------------
# Build each output mask
# ---------------------------------------------------------------------------


def build_mo_mask(total: Image) -> Image:
    """L3/L4 bone narrow."""
    l4 = get_label(total, get_class_id("vertebrae_L4", "total"), missing="empty")
    l3 = get_label(total, get_class_id("vertebrae_L3", "total"), missing="empty")
    l4n = _vertebra_narrow(l4)
    l3n = _vertebra_narrow(l3)

    out = np.zeros_like(total.data, dtype=np.uint8)
    out[l4n.data > 0] = MO_LABELS["L4"]
    out[l3n.data > 0] = MO_LABELS["L3"]
    return total.with_data(out)


def build_body_mask(body: Image) -> Image:
    """Merge body_trunc + body_extremities into a single BODY=1 mask."""
    trunk = get_label(body, get_class_id("body_trunc", "body"), missing="empty")
    ext = get_label(body, get_class_id("body_extremities", "body"), missing="empty")
    merged = np.zeros_like(body.data, dtype=np.uint8)
    merged[trunk.data > 0] = 1
    merged[ext.data > 0] = 1
    bc = biggest_cc(body.with_data(merged))

    out = np.zeros_like(body.data, dtype=np.uint8)
    out[bc.data > 0] = BODY_LABELS["BODY"]
    return body.with_data(out)


def _remove_extremities(fat: Any, body: Image) -> Any:
    """Zero out voxels inside body_extremities (id=2)."""
    ext = get_label(body, get_class_id("body_extremities", "body"), missing="empty").data
    out = fat.copy()
    out[ext > 0] = 0
    return out


def _remove_organs(fat_arr: Any, total: Image, pet: Image) -> Any:
    """Remove organ interiors (incl. dilated kidneys, PET-guided bladder) from fat."""
    ids_to_remove = [get_class_id(n, "total") for n in _ORGANS_TO_REMOVE]
    organs = combine_labels(total, ids_to_remove).data

    kr = get_label(total, get_class_id("kidney_right", "total"), missing="empty")
    kl = get_label(total, get_class_id("kidney_left", "total"), missing="empty")
    if bool(kr.data.any()):
        kr_ch = _chull_organ_3d(kr).data
        organs[kr_ch > 0] = 1
    if bool(kl.data.any()):
        kl_ch = _chull_organ_3d(kl).data
        organs[kl_ch > 0] = 1

    bladder = get_label(total, get_class_id("urinary_bladder", "total"), missing="empty")
    if bool(bladder.data.any()):
        bladder = _process_bladder(bladder, pet)
        organs[bladder.data > 0] = 1

    spacing = _spacing(total)
    for name, radius in _ORGANS_TO_DILATE.items():
        try:
            single = get_label(total, get_class_id(name, "total"), missing="empty").data
        except ValueError:
            continue
        if not bool(single.any()):
            continue
        dilated = dilate(
            total.with_data(single),
            footprint=radius,
            isotropic=True,
            spacing=spacing,
        ).data
        organs[dilated > 0] = 1

    out = fat_arr.copy()
    out[organs > 0] = 0
    return out


def build_fat_mask(
    tissue_types: Image,
    total: Image,
    body: Image,
    pet: Image,
) -> Image:
    """Visceral/subcutaneous fat clean-up (extremities, organs, PET-guided bladder)."""
    visceral_id = get_class_id("torso_fat", "tissue_types")
    subcutaneous_id = get_class_id("subcutaneous_fat", "tissue_types")
    fat_v = (tissue_types.data == visceral_id).astype(np.uint8)
    fat_s = (tissue_types.data == subcutaneous_id).astype(np.uint8)

    any_fat = (tissue_types.data > 0).astype(np.uint8)
    body_grown = dilate(
        tissue_types.with_data(any_fat),
        footprint=_BODY_DILATATION,
    )
    body_cc = biggest_cc(body_grown)
    body_filled = fill_holes(body_cc, axis=2).data
    fat_v = fat_v * body_filled
    fat_s = fat_s * body_filled

    fat_s = _remove_extremities(fat_s, body)

    fat_v = _remove_organs(fat_v, total, pet)
    fat_s = _remove_organs(fat_s, total, pet)

    out = np.zeros_like(tissue_types.data, dtype=np.uint8)
    out[fat_v > 0] = FAT_LABELS["GRASA_V"]
    out[fat_s > 0] = FAT_LABELS["GRASA_SC"]
    return tissue_types.with_data(out)


def build_organs_mask(total: Image) -> Image:
    """ORGANS.nii.gz: HIGADO=1, BAZO=2, PANCREAS=3."""
    out = np.zeros_like(total.data, dtype=np.uint8)
    mapping = [
        ("liver", ORGANS_LABELS["HIGADO"]),
        ("spleen", ORGANS_LABELS["BAZO"]),
        ("pancreas", ORGANS_LABELS["PANCREAS"]),
    ]
    for name, out_id in mapping:
        m = get_label(total, get_class_id(name, "total"), missing="empty").data
        out[m > 0] = out_id
    return total.with_data(out)


def build_muscles_mask(total: Image, muscles: Image) -> Image:
    """Hemisphere-preserving MUSCLES.nii.gz.

    * ``quadriceps_femoris_left/right`` (TS IDs 1,2) -> CUADRICEPS_L/R
    * ``autochthon_left/right`` (TS IDs 86,87 in 'total')   -> PARAVERTEBRAL_L/R
    * ``deltoid`` (TS ID 9)                           -> split L/R via CC
    * ``trapezius`` (TS ID 14)                        -> bilateral TRAPECIOS
    """
    out = np.zeros_like(muscles.data, dtype=np.uint8)

    q_l = get_label(
        muscles,
        get_class_id("quadriceps_femoris_left", "thigh_shoulder_muscles"),
        missing="empty",
    ).data
    q_r = get_label(
        muscles,
        get_class_id("quadriceps_femoris_right", "thigh_shoulder_muscles"),
        missing="empty",
    ).data
    out[q_l > 0] = MUSCLES_LABELS["CUADRICEPS_L"]
    out[q_r > 0] = MUSCLES_LABELS["CUADRICEPS_R"]

    p_l = get_label(total, get_class_id("autochthon_left", "total"), missing="empty").data
    p_r = get_label(total, get_class_id("autochthon_right", "total"), missing="empty").data
    out[p_l > 0] = MUSCLES_LABELS["PARAVERTEBRAL_L"]
    out[p_r > 0] = MUSCLES_LABELS["PARAVERTEBRAL_R"]

    deltoid = get_label(
        muscles, get_class_id("deltoid", "thigh_shoulder_muscles"), missing="empty"
    )
    if bool(deltoid.data.any()):
        try:
            d_left, d_right = split_lr_by_cc(deltoid, n=2)
            out[d_left.data > 0] = MUSCLES_LABELS["DELTOIDES_L"]
            out[d_right.data > 0] = MUSCLES_LABELS["DELTOIDES_R"]
        except Exception as exc:
            log.warning(f"deltoid CC split failed ({exc}); keeping bilateral")
            out[deltoid.data > 0] = MUSCLES_LABELS["DELTOIDES_L"]

    trap = get_label(
        muscles, get_class_id("trapezius", "thigh_shoulder_muscles"), missing="empty"
    ).data
    out[trap > 0] = MUSCLES_LABELS["TRAPECIOS"]

    return muscles.with_data(out)


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------


def _imread(path_parent: Path, stem: str, axes: str = "XYZ") -> Image:
    return imread(str(resolve_nii(path_parent, stem)), axes=axes)


def _process(segmentation_dir: Path, nifti_dir: Path, output_dir: Path) -> None:
    total = _imread(segmentation_dir, "total")
    tissue_types = _imread(segmentation_dir, "tissue_types")
    muscles = _imread(segmentation_dir, "thigh_shoulder_muscles")
    body = _imread(segmentation_dir, "body")
    pet = _imread(nifti_dir, cfg.PET_STEM)

    mo = build_mo_mask(total)
    fat = build_fat_mask(tissue_types, total, body, pet)
    bod = build_body_mask(body)
    organs = build_organs_mask(total)
    muscles_out = build_muscles_mask(total, muscles)

    output_dir.mkdir(parents=True, exist_ok=True)
    imsave(str(output_dir / "MO.nii.gz"), mo, axes="XYZ")
    imsave(str(output_dir / "FAT.nii.gz"), fat, axes="XYZ")
    imsave(str(output_dir / "BODY.nii.gz"), bod, axes="XYZ")
    imsave(str(output_dir / "ORGANS.nii.gz"), organs, axes="XYZ")
    imsave(str(output_dir / "MUSCLES.nii.gz"), muscles_out, axes="XYZ")


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    backend: str = "cupy",
) -> Path:
    """Build the five stage-2 outputs for a single subject."""
    try:
        set_global_backend(backend, allow_fallback=True)
    except Exception as exc:
        log.warning(f"Backend '{backend}' unavailable, falling back: {exc}")

    seg_dir = lay.results_dir / cfg.STAGE1_DIR / subject / "CT"
    nifti_dir = lay.subject_nifti_dir(subject)
    out_dir = lay.results_dir / cfg.STAGE2_DIR / subject / "CT"

    if not seg_dir.exists():
        raise FileNotFoundError(f"Expected stage-1 outputs under {seg_dir}")

    log.info(
        f"CT-PET v5 stage 2 | subject={subject} | backend={get_current_backend()}"
    )
    _process(seg_dir, nifti_dir, out_dir)
    log.info(f"[{subject}] ok -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("ctpet-v5-stage2")
@click.option("--batch", required=True)
@click.option("--subject", required=True)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--backend",
    type=click.Choice(["cupy", "numpy"], case_sensitive=False),
    default="cupy",
    show_default=True,
)
@click.option("--log-level", default="INFO")
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    backend: str,
    log_level: str,
) -> None:
    """CT-PET v5 stage 2 worker (single subject)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
    )
    run_subject(subject, lay, backend=backend)


if __name__ == "__main__":
    main()
