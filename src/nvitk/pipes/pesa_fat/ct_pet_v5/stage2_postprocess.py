"""CT-PET v5 stage 2 (per-subject): post-process TotalSegmentator outputs.

Port of :code:`BioImaging/src/pesa_fat/ct_pet/__post_process.py` /
:code:`2_post_processing.py`.

Produces the following per-subject label files under
``RESULTS/<batch>/res_post_processing_ct/<SUBJECT>/CT/``::

    MO.nii.gz       L4=1, L3=2                                          (bone narrow)
    FAT.nii.gz      GRASA_V=1, GRASA_SC=2                               (cleaned fat)
    FAT_BATCH.nii.gz GRASA_V_BATCH=1, GRASA_SC_BATCH=2                  (L3-L4 slab)
    FAT_IM.nii.gz   GRASA_IM_PIERNAS=1, GRASA_IM_TORAX=2                (intermuscular fat)
    BODY.nii.gz     BODY=1                                              (trunk+ext)
    ORGANS.nii.gz   HIGADO=1, BAZO=2, PANCREAS=3
    MUSCLES.nii.gz  CUADRICEPS_L=1, CUADRICEPS_R=2, PARAVERTEBRAL_L=3,
                    PARAVERTEBRAL_R=4, DELTOIDES_L=5, DELTOIDES_R=6,
                    TRAPECIOS=7                                         (v5 hemisphere-split)
    SKELETON.nii.gz SKELETON=1                                          (TS bone classes)

FAT, FAT_BATCH, ORGANS and MUSCLES are the SUV masks: each label is eroded by its
:data:`SUV_ERODE` iterations (top of this module) as a margin against organ motion
between CT and PET.

MO, FAT, FAT_BATCH, ORGANS and MUSCLES also get a ``<NAME>_VOL.nii.gz`` twin with
the same labels, read by the stage-3 volumes. It skips the erosions of the
measured structure (including :data:`SUV_ERODE`) and the dilations of structures
subtracted from it; the bone marrow twin is instead the raw vertebra below
:data:`cfg.MO_VOL_MAX_HU`.

v5 changes
----------
* Muscles are kept hemisphere-split; there is no L+R merging. Deltoid
  (bilateral TS label 9) is split into L/R via
  :func:`nvitk.segmentation.hemisphere.split_lr_by_cc`. Trapezius (14)
  remains a single bilateral mask.
* The skeleton (union of the ``total`` bone classes in
  :data:`cfg.SKELETON_ROIS`) is subtracted from every muscle label, so bone
  marrow uptake stays out of the muscle SUV statistics. Trapezius keeps its
  biggest component *per side* rather than overall, and quadriceps are eroded by
  one iteration afterwards.
* Fat comes from ``tissue_4_types``, whose intermuscular-fat class no longer
  inflates the subcutaneous label and feeds FAT_IM.
* The bone narrow gets :data:`cfg.MO_XY_ERODE` extra in-plane erosion steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from nvitk.core import as_backend_array
from nvitk.core.click_backend import backend_click_option, set_default_backend
from nvitk.core.backend import (
    get_current_backend,
    setup,
    to_cupy,
    to_numpy,
    using,
)
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.morphology import dilate, erode, fill_holes, label_connected
from nvitk.pipes.pesa_fat.common.paths import (
    BatchLayout,
    layout,
    resolve_nii,
    resolve_nii_optional,
)
from nvitk.pipes.pesa_fat.common.label_erosion import erode_labels
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
from nvitk.pipes.pesa_fat.ct_pet_v5.labels import (
    BODY_LABELS,
    FAT_LABELS,
    MO_LABELS,
    MUSCLES_LABELS,
    ORGANS_LABELS,
    SKELETON_LABELS,
    FAT_BATCH_LABELS,
    FAT_IM_LABELS,
)
from nvitk.segmentation.hemisphere import lr_axis_and_sign, split_lr_by_cc
from nvitk.segmentation.labels import biggest_cc, combine_labels, get_label
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.segmentation.hull_edt import convex_hull_3d
from nvitk.transform import resample_pet_to_mask
from nvitk.types import Image

# SUV-mask erosion: binary erosion iterations (6-connected, on the CT grid) per output
# label, a margin against organ motion between the CT and PET acquisitions so the SUV
# statistics stay inside the structure. Applied to FAT, FAT_BATCH, ORGANS and MUSCLES,
# which the stage-3 SUVs and the QC report (red contours) both read; the ``*_VOL``
# twins used for volumes are never eroded. The bone marrow (MO) is left out: it is
# already deep inside the vertebra (``_BN_ERODE`` + ``cfg.MO_XY_ERODE``). The
# quadriceps also keep their ``cfg.MUSCLE_ERODE_AFTER_SKELETON`` iteration before
# this. 0 (or a missing label) means no erosion; a label that erosion would empty is
# kept as is.
SUV_ERODE: dict[str, int] = {
    # FAT / FAT_BATCH (thin layers and pockets). The batch labels are eroded after
    # the ureter is cleared (margin from its uptake) but before the L3-L4 cut, so
    # the slab keeps its full height.
    "GRASA_V": 1,
    "GRASA_SC": 1,
    "GRASA_V_BATCH": 1,
    "GRASA_SC_BATCH": 1,
    # ORGANS
    "HIGADO": 2,
    "BAZO": 2,
    "PANCREAS": 1,
    # MUSCLES
    "CUADRICEPS_L": 2,
    "CUADRICEPS_R": 2,
    "PARAVERTEBRAL_L": 2,
    "PARAVERTEBRAL_R": 2,
    "DELTOIDES_L": 2,
    "DELTOIDES_R": 2,
    "TRAPECIOS": 2,
}

# Output mask file → label name(s) to clear where PET ureter overlaps.
URETER_EXCLUSION_BY_MASK: dict[str, tuple[str, ...]] = {
    "FAT_BATCH": ("GRASA_V_BATCH",),
}

_LABEL_MAPS_BY_MASK_FILE: dict[str, dict[str, int]] = {
    "FAT": FAT_LABELS,
    "FAT_BATCH": FAT_BATCH_LABELS,
}

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
    "kidney_right": 7,
    "kidney_left": 7,
    "liver": 1,
    "urinary_bladder": 10,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _spacing(img: Image) -> tuple[float, float, float]:
    """*img*'s (x, y, z) voxel spacing from its ``spacing`` metadata, or from ``x_res``/``y_res``/
    ``z_res`` as a fallback (defaulting to 1.0)."""
    sp = img.metadata.get("spacing") if img.metadata else None
    if sp is None:
        sp = (
            img.metadata.get("x_res", 1.0) if img.metadata else 1.0,
            img.metadata.get("y_res", 1.0) if img.metadata else 1.0,
            img.metadata.get("z_res", 1.0) if img.metadata else 1.0,
        )
    return tuple(float(v) for v in sp)


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
    # 26-connectivity CC labeling (base tool).
    labeled, _ = label_connected(potential, connectivity=3)
    overlap_ids = np.unique(labeled[bladder.data > 0])
    overlap_ids = overlap_ids[overlap_ids != 0]
    expanded = np.isin(labeled, overlap_ids).astype(np.uint8)
    filled = fill_holes(expanded, axis=2)
    return bladder.with_data(filled.astype(np.uint8))


def _xy_cross() -> Any:
    """Structuring element that erodes along X and Y only (XYZ arrays), never along Z."""
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[:, :, 1] = ndi.generate_binary_structure(2, 1)
    return struct


def _vertebra_narrow(
    vertebra: Image,
    radius: int = _BN_ERODE,
    xy_steps: int = cfg.MO_XY_ERODE,
) -> Image:
    """Return the bone narrow (eroded biggest CC) of a vertebra mask.

    The *radius* erosion is isotropic in voxels, so on thick-slice CT it trims far
    more in Z than in XY; *xy_steps* in-plane-only erosion steps then pull the
    marrow away from the cortex. A step that would empty the mask is skipped.
    """
    binary = (vertebra.data > 0).astype(np.uint8)
    eroded = erode(vertebra.with_data(binary), footprint=radius)
    if int(xy_steps) > 0 and bool(eroded.data.any()):
        shrunk = erode(eroded, footprint=_xy_cross(), iterations=int(xy_steps))
        if bool(shrunk.data.any()):
            eroded = shrunk
        else:
            log.warning(
                f"In-plane erosion by {int(xy_steps)} step(s) would empty the bone narrow; "
                "keeping the isotropic erosion only."
            )
    return biggest_cc(eroded)


def _vertebrae_l3_l4_labels(total: Image) -> Any:
    """Labeled L3/L4 mask on the ``total`` grid (values = :data:`MO_LABELS`)."""
    out = np.zeros_like(total.data, dtype=np.uint8)
    l4 = get_label(total, get_class_id("vertebrae_L4", "total"), missing="empty")
    l3 = get_label(total, get_class_id("vertebrae_L3", "total"), missing="empty")
    out[l4.data > 0] = MO_LABELS["L4"]
    out[l3.data > 0] = MO_LABELS["L3"]
    return total.with_data(out)


def _mask_bbox_slices(binary: Any) -> tuple[slice, slice, slice]:
    """Tight bounding box of *binary* as a 3-tuple of slices."""
    bounds = []
    for axis in range(3):
        projection = np.any(binary, axis=tuple(a for a in range(3) if a != axis))
        present = np.where(projection)[0]
        bounds.append(slice(int(present[0]), int(present[-1]) + 1))
    return tuple(bounds)


def _biggest_cc_per_side(base_img: Image, binary: Any) -> Any:
    """Largest connected component on each side of *binary*'s left-right centroid.

    A plain biggest-CC keep silently deletes one half of a bilateral label whose
    sides are not connected -- which is how trapezius usually comes out. Here each
    component is assigned whole to the side its own centroid falls on, and the
    largest component per side is kept, so a label connected across the midline
    still survives intact as one component. The split point is the mask's
    voxel-weighted centroid rather than its bounding-box midpoint, which a single
    stray speck would drag off-centre.

    A side is dropped when its component is smaller than
    :data:`cfg.MUSCLES_SIDE_MIN_RATIO` of the other's, so specks cannot pose as a
    missing half.
    """
    binary = (binary > 0)
    if not bool(np.any(binary)):
        return binary.astype(np.uint8)

    affine = base_img.affine
    if affine is None:
        log.warning("No affine available; keeping a single biggest CC for the bilateral label.")
        return (biggest_cc(base_img.with_data(binary.astype(np.uint8))).data > 0).astype(np.uint8)

    axis, _sign = lr_axis_and_sign(affine)
    # Everything below runs on the label's bounding box: whole-body grids are
    # large and this only ever looks at one muscle.
    box = _mask_bbox_slices(binary)
    labeled, num = label_connected(binary[box].astype(np.uint8), connectivity=3)
    if int(num) == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    # Per-component voxel counts and summed left-right index, accumulated one
    # slab at a time so no volume-sized coordinate array is ever allocated.
    sizes = np.zeros(int(num) + 1, dtype=np.float64)
    coord_sums = np.zeros(int(num) + 1, dtype=np.float64)
    offset = box[axis].start
    for i in range(labeled.shape[axis]):
        slab = np.take(labeled, i, axis=axis)
        counts = np.bincount(slab.ravel(), minlength=int(num) + 1).astype(np.float64)
        sizes += counts
        coord_sums += counts * float(i + offset)
    sizes[0] = 0.0
    coord_sums[0] = 0.0

    midline = float(coord_sums.sum()) / float(sizes.sum())
    centroids = coord_sums[1:] / np.maximum(sizes[1:], 1.0)

    keep: list[int] = []
    kept_sizes: list[float] = []
    for on_side in (centroids <= midline, centroids > midline):
        candidates = np.where(on_side, sizes[1:], 0.0)
        best = int(np.argmax(candidates))
        if float(candidates[best]) <= 0.0:
            continue
        keep.append(best + 1)
        kept_sizes.append(float(candidates[best]))

    largest = max(kept_sizes) if kept_sizes else 0.0
    kept = np.zeros_like(labeled, dtype=bool)
    for cc_id, size in zip(keep, kept_sizes):
        if size < cfg.MUSCLES_SIDE_MIN_RATIO * largest:
            continue
        kept |= labeled == cc_id

    out = np.zeros_like(binary)
    out[box] = kept
    return out.astype(np.uint8)


def _muscles_keep_biggest_cc_per_label(base_img: Image, out_labels: Image) -> Image:
    """Per muscle label ID, keep only the largest 3D connected component.

    Labels in :data:`cfg.MUSCLES_BIGGEST_CC_PER_SIDE` keep their largest
    component on each side instead (see :func:`_biggest_cc_per_side`).
    """
    per_side_ids = {
        MUSCLES_LABELS[name]
        for name in cfg.MUSCLES_BIGGEST_CC_PER_SIDE
        if name in MUSCLES_LABELS
    }
    arr = as_backend_array(out_labels.data).copy()
    for lid in sorted(set(MUSCLES_LABELS.values())):
        bin_mask = (arr == lid).astype(np.uint8)
        if not np.any(bin_mask):
            continue
        if lid in per_side_ids:
            cc = _biggest_cc_per_side(base_img, bin_mask) > 0
        else:
            cc = biggest_cc(base_img.with_data(bin_mask)).data > 0
        arr[arr == lid] = 0
        arr[cc] = lid
    return out_labels.with_data(arr)


def _subtract_skeleton(out_labels: Image, skeleton: Image | None) -> Image:
    """Clear :data:`cfg.SKELETON_SUBTRACT_FROM` labels wherever *skeleton* is positive."""
    if skeleton is None:
        return out_labels
    if tuple(skeleton.data.shape) != tuple(out_labels.data.shape):
        log.warning(
            f"Skeleton grid {tuple(skeleton.data.shape)} does not match the muscle grid "
            f"{tuple(out_labels.data.shape)}; skipping skeleton subtraction."
        )
        return out_labels
    bone = as_backend_array(skeleton.data) > 0
    if not bool(np.any(bone)):
        return out_labels
    arr = as_backend_array(out_labels.data).copy()
    for name in cfg.SKELETON_SUBTRACT_FROM:
        lid = MUSCLES_LABELS.get(name)
        if lid is None:
            continue
        arr[(arr == lid) & bone] = 0
    return out_labels.with_data(arr)


def _erode_after_skeleton(out_labels: Image) -> Image:
    """Shave :data:`cfg.MUSCLE_ERODE_AFTER_SKELETON` labels by their configured iterations.

    Runs after the skeleton subtraction, so the border being removed is the one
    left behind once bone is gone -- the voxels most likely to be contaminated by
    neighbouring tissue. A label that erosion would empty is kept intact instead.
    """
    return erode_labels(out_labels, MUSCLES_LABELS, cfg.MUSCLE_ERODE_AFTER_SKELETON)


# ---------------------------------------------------------------------------
# Build each output mask
# ---------------------------------------------------------------------------

def limit_vertebrae_axial(
    img: np.ndarray,
    vertebrae: np.ndarray,
    min_vertebrae: int,
    max_vertebrae: int,
    ref_space: Image,
) -> np.ndarray:
    """Keep only axial slices between the L3 and L4 extent (biggest CC each)."""
    bin_min = get_label(vertebrae, min_vertebrae, missing="empty")
    bin_max = get_label(vertebrae, max_vertebrae, missing="empty")
    cc_min = biggest_cc(ref_space.with_data(bin_min)).data > 0
    cc_max = biggest_cc(ref_space.with_data(bin_max)).data > 0
    min_slices = np.where(np.any(cc_min, axis=(0, 1)))[0]
    max_slices = np.where(np.any(cc_max, axis=(0, 1)))[0]
    if min_slices.size == 0 or max_slices.size == 0:
        raise ValueError("No vertebrae found in the image.")
    z0, z1 = int(np.min(min_slices)), int(np.max(max_slices))
    if z0 > z1:
        raise ValueError("Min slice is greater than max slice.")
    out = np.zeros_like(img)
    out[..., z0 : z1 + 1] = img[..., z0 : z1 + 1]
    return ref_space.with_data(out)


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


def build_mo_volume_mask(total: Image, ct: Image) -> Image:
    """MO_VOL.nii.gz: L3/L4 bone marrow for volumes, stripped of cortex by density.

    Volumes skip the erosions that define :func:`build_mo_mask`, so the marrow is
    the raw vertebra's voxels below :data:`cfg.MO_VOL_MAX_HU` (biggest component).
    Falls back to the raw vertebra, with a warning, when the CT grid differs.
    """
    hu = as_backend_array(ct.data)
    if tuple(hu.shape) != tuple(total.data.shape):
        log.warning(
            f"CT grid {tuple(hu.shape)} does not match the 'total' grid "
            f"{tuple(total.data.shape)}; MO_VOL keeps the raw vertebrae."
        )
        hu = None
    out = np.zeros_like(total.data, dtype=np.uint8)
    for name in ("L4", "L3"):
        vert = get_label(total, get_class_id(f"vertebrae_{name}", "total"), missing="empty")
        binary = as_backend_array(vert.data) > 0
        if hu is not None:
            binary &= hu < cfg.MO_VOL_MAX_HU
        if not bool(binary.any()):
            continue
        marrow = biggest_cc(total.with_data(binary.astype(np.uint8)))
        out[as_backend_array(marrow.data) > 0] = MO_LABELS[name]
    return total.with_data(out)


def build_skeleton_mask(total: Image) -> Image:
    """SKELETON.nii.gz: union of the :data:`cfg.SKELETON_ROIS` bone classes.

    Returns an empty mask (with a warning) when the ``total`` segmentation holds
    none of them -- the sign of a stage-1 run that predates the skeleton ROIs.
    """
    ids = []
    for name in cfg.SKELETON_ROIS:
        try:
            ids.append(get_class_id(name, "total"))
        except ValueError:
            log.debug(f"Skeleton ROI {name!r} is not a 'total' class; skipping.")

    arr = as_backend_array(total.data)
    present = [cid for cid in ids if bool(np.any(arr == cid))]
    if not present:
        log.warning(
            "No skeleton bone labels found in the stage-1 'total' segmentation. "
            "Re-run stage 1 so it saves cfg.SKELETON_ROIS; muscle masks are left "
            "unchanged by the skeleton subtraction."
        )
        return total.with_data(np.zeros_like(arr, dtype=np.uint8))

    bone = combine_labels(total, present, new_id=SKELETON_LABELS["SKELETON"])
    return total.with_data(as_backend_array(bone.data).astype(np.uint8))


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


def _organs_mask(total: Image, pet: Image) -> Any:
    """Organ interiors to clear from the fat: organs, kidney hulls, PET-guided bladder.

    Undilated; :func:`_with_dilated_organs` adds the safety margins the SUV masks use.
    """
    ids_to_remove = [get_class_id(n, "total") for n in _ORGANS_TO_REMOVE]
    organs = combine_labels(total, ids_to_remove).data

    kr = get_label(total, get_class_id("kidney_right", "total"), missing="empty")
    kl = get_label(total, get_class_id("kidney_left", "total"), missing="empty")
    if bool(kr.data.any()):
        kr_ch = convex_hull_3d(kr).data
        organs[kr_ch > 0] = 1
    if bool(kl.data.any()):
        kl_ch = convex_hull_3d(kl).data
        organs[kl_ch > 0] = 1

    bladder = get_label(total, get_class_id("urinary_bladder", "total"), missing="empty")
    if bool(bladder.data.any()):
        bladder = _process_bladder(bladder, pet)
        organs[bladder.data > 0] = 1
    return organs


def _with_dilated_organs(organs: Any, total: Image) -> Any:
    """Copy of *organs* with the :data:`_ORGANS_TO_DILATE` organs grown by their radius."""
    organs = organs.copy()
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
            isotropic=False,
            spacing=spacing,
        ).data
        organs[dilated > 0] = 1
    return organs


def _ureter_exclusion_label_ids(mask_file: str) -> list[int]:
    """Resolve configured label names for *mask_file* to integer IDs."""
    label_names = URETER_EXCLUSION_BY_MASK.get(mask_file, ())
    label_map = _LABEL_MAPS_BY_MASK_FILE.get(mask_file, {})
    return [int(label_map[name]) for name in label_names if name in label_map]


def _apply_ureter_exclusion(
    label_img: np.ndarray,
    ureter_mask: np.ndarray,
    *,
    mask_file: str,
) -> None:
    """Zero configured label(s) in *label_img* where *ureter_mask* is positive."""
    ureter = ureter_mask > 0
    if not ureter.any():
        return
    for label_id in _ureter_exclusion_label_ids(mask_file):
        sel = (label_img == label_id) & ureter
        label_img[sel] = 0


def _fat_image(fat_v: Any, fat_s: Any, template: Image) -> Image:
    """FAT label image (whole body): GRASA_V / GRASA_SC."""
    out = np.zeros_like(template.data, dtype=np.uint8)
    out[fat_v > 0] = FAT_LABELS["GRASA_V"]
    out[fat_s > 0] = FAT_LABELS["GRASA_SC"]
    return template.copy().with_data(out)


def _fat_batch_image(
    fat_v: Any,
    fat_s: Any,
    template: Image,
    vertebrae_l3_l4: Image,
    total: Image,
    ureter: Any | None,
    *,
    suv_erode: bool = False,
) -> Image:
    """FAT_BATCH label image: the fat cut to the L3-L4 slab, ureter cleared from GRASA_V_BATCH.

    With *suv_erode* the labels are eroded by their :data:`SUV_ERODE` iterations after
    the ureter is cleared (so the fat also keeps clear of the ureter's uptake) but
    before the slab cut (so the slab keeps its full height).
    """
    out_batch = np.zeros_like(template.data, dtype=np.uint8)
    out_batch[fat_v > 0] = FAT_BATCH_LABELS["GRASA_V_BATCH"]
    out_batch[fat_s > 0] = FAT_BATCH_LABELS["GRASA_SC_BATCH"]
    if ureter is not None:
        _apply_ureter_exclusion(out_batch, ureter, mask_file="FAT_BATCH")
    batch = template.copy().with_data(out_batch)
    if suv_erode:
        batch = erode_labels(batch, FAT_BATCH_LABELS, SUV_ERODE)
    slab = limit_vertebrae_axial(batch.data, vertebrae_l3_l4, MO_LABELS["L4"], MO_LABELS["L3"], total)
    return template.copy().with_data(slab.data)


def build_fat_mask(
    tissue_types: Image,
    total: Image,
    body: Image,
    pet: Image,
    exclude_ureter: bool = True,
    output_dir: Path | None = None,
    *,
    tissue_task: str = cfg.FAT_TASK,
) -> tuple[Image, Image, Image, Image]:
    """Visceral/subcutaneous fat clean-up (extremities, organs, PET-guided bladder).

    Returns ``(FAT, FAT_BATCH, FAT_VOL, FAT_BATCH_VOL)``. FAT and FAT_BATCH are the
    SUV masks: organs cleared with their safety dilation, then each depot eroded by
    its :data:`SUV_ERODE` iterations (see :func:`_fat_batch_image` for the batch). The
    ``_VOL`` pair clears the undilated organs and is not eroded.
    *tissue_task* names the TotalSegmentator task *tissue_types* came from.
    """
    visceral_id = get_class_id("torso_fat", tissue_task)
    subcutaneous_id = get_class_id("subcutaneous_fat", tissue_task)
    fat_v = (tissue_types.data == visceral_id).astype(np.uint8)
    fat_s = (tissue_types.data == subcutaneous_id).astype(np.uint8)

    # ---- Body mask --------------------------------------------------------
    any_fat = (tissue_types.data > 0).astype(np.uint8)
    body_grown = dilate(
        tissue_types.with_data(any_fat),
        footprint=_BODY_DILATATION,
    )
    body_cc = biggest_cc(body_grown)
    body_filled = fill_holes(body_cc, axis=2).data
    fat_v = fat_v * body_filled
    fat_s = fat_s * body_filled

    # ---- Extremities exclusion --------------------------------------------
    fat_s = _remove_extremities(fat_s, body)

    # ---- Organs exclusion -------------------------------------------------
    organs_raw = _organs_mask(total, pet)
    organs = _with_dilated_organs(organs_raw, total)
    fat_v_vol, fat_s_vol = fat_v.copy(), fat_s.copy()
    fat_v[organs > 0] = 0
    fat_s[organs > 0] = 0
    fat_v_vol[organs_raw > 0] = 0
    fat_s_vol[organs_raw > 0] = 0

    # ---- Ureter segmentation & exclusion --------------------------------
    ureter = None
    if exclude_ureter:
        from nvitk.segmentation.pet.ureter_segmentation import segment_ureter
        from nvitk.transform.resampling import resample_mask_to_pet
        from nvitk.measure.suv import suv_image

        log.info("Running ureter segmentation...")
        _kidney_r = get_label(total, get_class_id("kidney_right", "total"), missing="empty")
        _kidney_l = get_label(total, get_class_id("kidney_left", "total"), missing="empty")
        _bladder  = get_label(total, get_class_id("urinary_bladder", "total"), missing="empty")

        _resampled_kidney_r = resample_mask_to_pet(_kidney_r, pet, order=0)
        _resampled_kidney_l = resample_mask_to_pet(_kidney_l, pet, order=0)
        _resampled_bladder  = resample_mask_to_pet(_bladder, pet, order=0)
        _resampled_body     = resample_mask_to_pet(body, pet, order=0)
        _raw_suv            = suv_image(pet, pet.metadata, philips=False)
        _mask, _, _         = segment_ureter(
            _raw_suv,
            _resampled_kidney_r,
            _resampled_kidney_l,
            _resampled_bladder,
            _resampled_body,
            radius_mm=8.0
        )

        ureter = _mask.data > 0
        ureter = pet.copy().with_data(ureter)
        resampled_ureter = resample_mask_to_pet(ureter, total, order=0)
        if output_dir:
            _resampled_ureter = resampled_ureter.copy().with_data(resampled_ureter.data.astype(np.uint8))
            imsave(str(output_dir / "_URETER.nii.gz"), _resampled_ureter, axes="XYZ")
        ureter = resampled_ureter.data

    # ---- FAT / FAT BATCH (SUV and volume variants) -------------------------
    vertebrae_l3_l4 = _vertebrae_l3_l4_labels(total)
    fat = erode_labels(_fat_image(fat_v, fat_s, tissue_types), FAT_LABELS, SUV_ERODE)
    fat_batch = _fat_batch_image(
        fat_v, fat_s, tissue_types, vertebrae_l3_l4, total, ureter, suv_erode=True
    )
    fat_vol = _fat_image(fat_v_vol, fat_s_vol, tissue_types)
    fat_batch_vol = _fat_batch_image(fat_v_vol, fat_s_vol, tissue_types, vertebrae_l3_l4, total, ureter)
    return fat, fat_batch, fat_vol, fat_batch_vol


def _superior_sign(img: Image) -> int:
    """``+1`` when increasing the Z index moves toward the subject's head, else ``-1``."""
    affine = img.affine
    if affine is None:
        log.warning("No affine available; assuming the Z index increases toward the head.")
        return 1
    return 1 if float(to_numpy(affine)[2, 2]) >= 0 else -1


def _trunk_lr_range(body: Image | None, total: Image | None, ref: Image) -> tuple[int, int, int] | None:
    """``(lr_axis, first, last)``: the left-right extent of ``body_trunc``, projected axially over
    its slices from the trunk's inferior end up to the top of L3.

    The legs lie within it and arms hanging beside the hips outside it, so it tells the
    two apart where an axial split cannot. ``None``, with a warning, when an input is
    missing, off *ref*'s grid, or holds no trunk / L3.
    """
    if body is None or total is None:
        log.warning("No 'body' / 'total' segmentation; arm fat is not separated from the legs.")
        return None
    shape = tuple(ref.data.shape)
    if tuple(body.data.shape) != shape or tuple(total.data.shape) != shape:
        log.warning("'body' / 'total' grid differs from the tissue grid; arm fat is not separated from the legs.")
        return None
    trunc = as_backend_array(body.data) == get_class_id("body_trunc", "body")
    l3 = as_backend_array(total.data) == get_class_id("vertebrae_L3", "total")
    z_trunc = np.where(np.any(trunc, axis=(0, 1)))[0]
    z_l3 = np.where(np.any(l3, axis=(0, 1)))[0]
    if int(z_trunc.size) == 0 or int(z_l3.size) == 0:
        log.warning("No body_trunc or L3 found; arm fat is not separated from the legs.")
        return None
    if _superior_sign(ref) > 0:
        z_lo, z_hi = int(z_trunc.min()), int(z_l3.max())
    else:
        z_lo, z_hi = int(z_l3.min()), int(z_trunc.max())
    if z_lo > z_hi:
        log.warning("body_trunc starts above L3; arm fat is not separated from the legs.")
        return None
    lr_axis = 0 if ref.affine is None else lr_axis_and_sign(ref.affine)[0]
    if lr_axis == 2:
        log.warning("The left-right axis is the slice axis; arm fat is not separated from the legs.")
        return None
    projection = np.any(trunc[..., z_lo : z_hi + 1], axis=2)
    lr = np.where(np.any(projection, axis=1 - lr_axis))[0]
    return lr_axis, int(lr.min()), int(lr.max())


def build_fat_im_mask(
    tissue_types: Image,
    muscles: Image,
    *,
    body: Image | None = None,
    total: Image | None = None,
    tissue_task: str = cfg.FAT_TASK,
) -> Image:
    """FAT_IM.nii.gz: intermuscular fat split into legs and trunk along the axial axis.

    * ``GRASA_IM_PIERNAS`` -- the axial slices spanned by the quadriceps
      (CUADRICEPS_L + CUADRICEPS_R in *muscles*), extended by every whole
      26-connected intermuscular-fat component touching them, so fat running from
      the thighs into the hips stays with the legs;
    * ``GRASA_IM_TORAX``   -- the rest of the fat above the quadriceps' superior end.

    Arms hanging beside the body share the quadriceps' slices, so the legs are kept
    within the trunk's left-right extent (:func:`_trunk_lr_range`, from the stage-1
    *body* and *total*): fat beside it neither seeds the component extension nor
    stays in PIERNAS, and goes to TORAX with the rest of the arms. Fat entirely
    below the quadriceps is in neither label.

    Empty, with a warning, when *tissue_task* has no intermuscular-fat class (the
    3-class fallback) or *muscles* holds no quadriceps.
    """
    out = np.zeros_like(tissue_types.data, dtype=np.uint8)
    try:
        im_id = get_class_id("intermuscular_fat", tissue_task)
    except ValueError:
        log.warning(f"'{tissue_task}' has no intermuscular_fat class; FAT_IM is left empty.")
        return tissue_types.copy().with_data(out)
    if tuple(muscles.data.shape) != tuple(tissue_types.data.shape):
        log.warning(
            f"Muscle grid {tuple(muscles.data.shape)} does not match the tissue grid "
            f"{tuple(tissue_types.data.shape)}; FAT_IM is left empty."
        )
        return tissue_types.copy().with_data(out)

    muscle_arr = as_backend_array(muscles.data)
    quads = (muscle_arr == MUSCLES_LABELS["CUADRICEPS_L"]) | (muscle_arr == MUSCLES_LABELS["CUADRICEPS_R"])
    z_slices = np.where(np.any(quads, axis=(0, 1)))[0]
    if int(z_slices.size) == 0:
        log.warning("No quadriceps in MUSCLES; FAT_IM is left empty.")
        return tissue_types.copy().with_data(out)
    z0, z1 = int(z_slices.min()), int(z_slices.max())

    fat_im = as_backend_array(tissue_types.data) == im_id
    trunk = np.zeros_like(fat_im)
    if _superior_sign(tissue_types) > 0:
        trunk[..., z1 + 1 :] = True
    else:
        trunk[..., :z0] = True

    # Beside the trunk: left and right of its left-right extent (the arms).
    beside = np.zeros_like(fat_im)
    lr_range = _trunk_lr_range(body, total, tissue_types)
    if lr_range is not None:
        lr_axis, first, last = lr_range
        for part in (slice(None, first), slice(last + 1, None)):
            index = [slice(None)] * fat_im.ndim
            index[lr_axis] = part
            beside[tuple(index)] = True

    # Legs: every component with a voxel in the quadriceps' slices, i.e. those
    # slices' fat plus the whole components touching it -- seeded and kept only
    # within the trunk's extent. What the slices reach beside it is arm fat.
    labeled, _ = label_connected(fat_im, connectivity=3)
    slab = labeled[..., z0 : z1 + 1]
    seeds = np.unique(slab[~beside[..., z0 : z1 + 1]])
    legs = np.isin(labeled, seeds[seeds > 0]) & ~beside
    reached = np.unique(slab)
    arms = np.isin(labeled, reached[reached > 0]) & beside

    out[(fat_im & trunk) | arms] = FAT_IM_LABELS["GRASA_IM_TORAX"]
    out[legs] = FAT_IM_LABELS["GRASA_IM_PIERNAS"]
    return tissue_types.copy().with_data(out)


def build_organs_mask(total: Image, *, dilate_kidneys: bool = True) -> Image:
    """ORGANS.nii.gz: HIGADO=1, BAZO=2, PANCREAS=3.

    The kidneys' convex hulls are cleared from the liver; *dilate_kidneys* grows
    them first (the volume variant, ORGANS_VOL, does not).
    """
    out = np.zeros_like(total.data, dtype=np.uint8)
    mapping = [
        ("liver", ORGANS_LABELS["HIGADO"]),
        ("spleen", ORGANS_LABELS["BAZO"]),
        ("pancreas", ORGANS_LABELS["PANCREAS"]),
    ]
    for name, out_id in mapping:
        m = get_label(total, get_class_id(name, "total"), missing="empty").data

        # Liver: clear the (dilated) kidney hulls before writing it.
        if name == "liver":
            for kidney in ("kidney_right", "kidney_left"):
                k = get_label(total, get_class_id(kidney, "total"), missing="empty")
                if not bool(k.data.any()):
                    continue
                hull = convex_hull_3d(k).data
                if dilate_kidneys:
                    hull = dilate(total.copy().with_data(hull), footprint=5).data
                m[hull > 0] = 0

        out[m > 0] = out_id

    return total.with_data(out)


def build_muscles_mask(
    total: Image,
    muscles: Image,
    skeleton: Image | None = None,
    *,
    erode_after_skeleton: bool = True,
) -> Image:
    """Hemisphere-preserving MUSCLES.nii.gz.

    * ``quadriceps_femoris_left/right`` (TS IDs 1,2) -> CUADRICEPS_L/R
    * ``autochthon_left/right`` (TS IDs 86,87 in 'total')   -> PARAVERTEBRAL_L/R
    * ``deltoid`` (TS ID 9)                           -> split L/R via CC
    * ``trapezius`` (TS ID 14)                        -> bilateral TRAPECIOS

    With a *skeleton* mask (on the ``total`` grid) the bones are subtracted from
    the labels in :data:`cfg.SKELETON_SUBTRACT_FROM`. The labels in
    :data:`cfg.MUSCLE_ERODE_AFTER_SKELETON` are then eroded, unless
    *erode_after_skeleton* is off (the MUSCLES_VOL variant). Components are
    resolved *before* the subtraction, so splitting a muscle around a bone cannot
    cost it half its volume.
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
            import traceback
            log.warning(traceback.format_exc())
            log.warning(f"deltoid CC split failed ({exc}); keeping bilateral")
            out[deltoid.data > 0] = MUSCLES_LABELS["DELTOIDES_L"]

    trap = get_label(
        muscles, get_class_id("trapezius", "thigh_shoulder_muscles"), missing="empty"
    ).data
    out[trap > 0] = MUSCLES_LABELS["TRAPECIOS"]

    labels = _muscles_keep_biggest_cc_per_label(total, muscles.copy().with_data(out))
    labels = _subtract_skeleton(labels, skeleton)
    return _erode_after_skeleton(labels) if erode_after_skeleton else labels


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------


def _imread(path_parent: Path, stem: str, axes: str = "XYZ") -> Image:
    """Read the ``<stem>.nii[.gz]`` file resolved under *path_parent* as an Image with axis order *axes*."""
    return imread(str(resolve_nii(path_parent, stem)), axes=axes)


def _tissue_task(segmentation_dir: Path) -> str:
    """Stage-1 tissue task to read: :data:`cfg.FAT_TASK`, else the 3-class fallback."""
    if resolve_nii_optional(segmentation_dir, cfg.FAT_TASK) is not None:
        return cfg.FAT_TASK
    log.warning(
        f"{cfg.FAT_TASK}.nii(.gz) not found in {segmentation_dir}; reading {cfg.FAT_TASK_FALLBACK} "
        "instead (intermuscular fat stays in GRASA_SC and FAT_IM is empty). Re-run stage 1."
    )
    return cfg.FAT_TASK_FALLBACK


def _process(segmentation_dir: Path, nifti_dir: Path, output_dir: Path, exclude_ureter: bool = True) -> None:
    """Build and write the post-processed masks for one subject from its TotalSegmentator
    outputs, CT and PET volumes: MO/FAT/FAT_BATCH/ORGANS/MUSCLES plus their ``_VOL``
    variants, FAT_IM, BODY and SKELETON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue_task = _tissue_task(segmentation_dir)
    total = _imread(segmentation_dir, "total")
    tissue_types = _imread(segmentation_dir, tissue_task)
    muscles = _imread(segmentation_dir, "thigh_shoulder_muscles")
    body = _imread(segmentation_dir, "body")
    ct = _imread(nifti_dir, cfg.INPUT_STEM)
    pet = _imread(nifti_dir, cfg.PET_STEM)

    mo = build_mo_mask(total)
    mo_vol = build_mo_volume_mask(total, ct)
    fat, fat_batch, fat_vol, fat_batch_vol = build_fat_mask(
        tissue_types,
        total,
        body,
        pet,
        exclude_ureter=exclude_ureter,
        output_dir=output_dir,
        tissue_task=tissue_task,
    )
    bod = build_body_mask(body)
    organs = erode_labels(build_organs_mask(total), ORGANS_LABELS, SUV_ERODE)
    organs_vol = build_organs_mask(total, dilate_kidneys=False)
    skeleton = build_skeleton_mask(total)
    muscles_vol = build_muscles_mask(total, muscles, skeleton=skeleton, erode_after_skeleton=False)
    muscles_out = erode_labels(_erode_after_skeleton(muscles_vol), MUSCLES_LABELS, SUV_ERODE)
    fat_im = build_fat_im_mask(
        tissue_types, muscles_vol, body=body, total=total, tissue_task=tissue_task
    )

    imsave(str(output_dir / "MO.nii.gz"), mo, axes="XYZ")
    imsave(str(output_dir / "MO_VOL.nii.gz"), mo_vol, axes="XYZ")
    imsave(str(output_dir / "FAT.nii.gz"), fat, axes="XYZ")
    imsave(str(output_dir / "FAT_VOL.nii.gz"), fat_vol, axes="XYZ")
    imsave(str(output_dir / "FAT_BATCH.nii.gz"), fat_batch, axes="XYZ")
    imsave(str(output_dir / "FAT_BATCH_VOL.nii.gz"), fat_batch_vol, axes="XYZ")
    imsave(str(output_dir / "FAT_IM.nii.gz"), fat_im, axes="XYZ")
    imsave(str(output_dir / "BODY.nii.gz"), bod, axes="XYZ")
    imsave(str(output_dir / "ORGANS.nii.gz"), organs, axes="XYZ")
    imsave(str(output_dir / "ORGANS_VOL.nii.gz"), organs_vol, axes="XYZ")
    imsave(str(output_dir / "MUSCLES.nii.gz"), muscles_out, axes="XYZ")
    imsave(str(output_dir / "MUSCLES_VOL.nii.gz"), muscles_vol, axes="XYZ")
    imsave(str(output_dir / "SKELETON.nii.gz"), skeleton, axes="XYZ")


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    backend: str = "cupy",
    exclude_ureter: bool = True,
) -> Path:
    """Build the stage-2 outputs for a single subject."""
    try:
        set_default_backend(backend, allow_fallback=True)
    except Exception as exc:
        log.warning(f"Backend '{backend}' unavailable, falling back: {exc}")

    seg_dir = lay.results_dir / cfg.STAGE1_DIR / subject / "CT"
    nifti_dir = lay.subject_nifti_dir(subject)
    out_dir = lay.results_dir / cfg.STAGE2_DIR / subject / "CT"

    if not seg_dir.exists():
        raise FileNotFoundError(f"Expected stage-1 outputs under {seg_dir}")

    log.info(f"CT-PET v5 stage 2 | subject={subject} | backend={get_current_backend()}")
    _process(seg_dir, nifti_dir, out_dir, exclude_ureter=exclude_ureter)
    log.info(f"[{subject}] ok -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("ctpet-v5-stage2")
@backend_click_option()
@click.option("--batch", required=True)
@click.option("--subject", required=True)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Model root. Unused by this stage, but keeps BatchLayout resolvable without sge.json.",
)
@click.option("--log-level", default="INFO")
@click.option(
    "--exclude-ureter/--no-exclude-ureter",
    default=True,
    help="Exclude PET ureter from configured BATCH visceral fat labels (default: on).",
)
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    model_dir: Path | None,
    backend: str,
    log_level: str,
    exclude_ureter: bool = True,
) -> None:
    """CT-PET v5 stage 2 worker (single subject)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
        model_root=model_dir,
    )
    run_subject(subject, lay, backend=backend, exclude_ureter=exclude_ureter)


if __name__ == "__main__":
    main()
