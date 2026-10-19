"""Dixon v5 stage 2 (per-subject): post-process TotalSegmentator outputs.

Port of :code:`BioImaging/src/pesa_fat/dixon/__post_process.py` /
:code:`2_post_processing.py`.

Produces three per-subject, per-region label files under
``RESULTS/<batch>/res_post_processing_dixon/<SUBJECT>/``::

    HEAD.nii.gz    H_PVM_L=1, H_PVM_R=2
    THORAX.nii.gz  LIVER=1, PANCREAS=2, KIDNEY_L=3, KIDNEY_R=4,
                   T_PVM_L=5, T_PVM_R=6,
                   BN_L3=7, BN_L4=8
    LEGS.nii.gz    L_QM_L=1, L_QM_R=2

Two inputs do not come from the region's FAT contrast:

* the liver is taken from the WATER-contrast ``total_mr`` run
  (``cfg.THORAX_WATER_STEM``), where it is far better defined than on FAT;
* the LEGS quadriceps have the femur/hip skeleton (``cfg.SKELETON_ROIS_MR``)
  subtracted, the femurs dilated by ``cfg.LEGS_SKELETON_DILATE`` first, so bone
  marrow stays out of their fat-fraction statistics.

The liver also loses its vessels: voxels whose ``DIXON_THORAX_T2STAR`` exceeds
``cfg.LIVER_VESSEL_T2STAR_MS``.

Every label of HEAD, THORAX and LEGS is then eroded by its :data:`MEASURE_ERODE`
iterations (top of this module): those masks feed the FF / T2* / R2* / WF statistics.

Each region also gets a ``<REGION>_VOL.nii.gz`` twin with the same labels, read by
the stage-3 volumes: no erosions of the measured structures (including
:data:`MEASURE_ERODE`), no dilations of the structures subtracted from them and no
vessel cut, so BN_L3/BN_L4 there are the raw vertebrae.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option, set_default_backend
from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.morphology import dilate, erode
from nvitk.pipes.pesa_fat.common.label_erosion import erode_labels
from nvitk.pipes.pesa_fat.common.paths import (
    BatchLayout,
    layout,
    resolve_nii_optional,
)
from nvitk.pipes.pesa_fat.dixon_v5 import config as cfg
from nvitk.pipes.pesa_fat.dixon_v5.labels import (
    HEAD_LABELS,
    LEGS_LABELS,
    THORAX_LABELS,
)
from nvitk.segmentation.labels import biggest_cc, combine_labels, get_label
from nvitk.segmentation.hull_edt import convex_hull_3d
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.types import Image

# Measurement-mask erosion: binary erosion iterations (6-connected, on the region's
# grid) per output label, applied last to HEAD, THORAX and LEGS -- the masks the FF /
# T2* / R2* / WF statistics, the QC report (red contours) and the hotspots read -- so
# they stay clear of partial volume at the structure's border. The ``*_VOL`` twins
# used for VOL / NSlices are never eroded. 0 (or a missing label) means no erosion; a
# label that erosion would empty is kept as is (BN_L3 / BN_L4, already eroded to
# bone narrow, are often only 2-3 slices thick).
MEASURE_ERODE: dict[str, int] = {
    # HEAD
    "H_PVM_L": 1,
    "H_PVM_R": 1,
    # THORAX
    "LIVER": 1,
    "PANCREAS": 1,
    "KIDNEY_L": 1,
    "KIDNEY_R": 1,
    "T_PVM_L": 1,
    "T_PVM_R": 1,
    "BN_L3": 1,
    "BN_L4": 1,
    # LEGS
    "L_QM_L": 1,
    "L_QM_R": 1,
}

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BN_ERODE = 3  # legacy dixon used 3 (ct-pet used 5)
_KIDNEY_ERODE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _imread_opt(parent: Path, stem: str, axes: str = "XYZ") -> Image | None:
    """Read the ``<stem>.nii[.gz]`` file resolved under *parent*, or ``None`` if it doesn't exist."""
    path = resolve_nii_optional(parent, stem)
    if path is None:
        return None
    return imread(str(path), axes=axes)


def _vertebra_narrow(vertebra: Image, radius: int = _BN_ERODE) -> Image:
    """Erode the vertebra mask by *radius* voxels and keep only its largest connected component
    (empty mask if erosion removes everything)."""
    binary = (vertebra.data > 0).astype(np.uint8)
    eroded = erode(vertebra.with_data(binary), footprint=radius)
    if not bool(eroded.data.any()):
        return vertebra.with_data(np.zeros_like(vertebra.data, dtype=np.uint8))
    return biggest_cc(eroded)


def _biggest_cc_or_empty(label_img: Image, label_id: int) -> Image:
    """Extract *label_id* from *label_img* and keep only its largest connected component (empty mask
    if the label is absent)."""
    m = get_label(label_img, label_id, missing="empty")
    if not bool(m.data.any()):
        return m
    return biggest_cc(m)


def _muscles_keep_biggest_cc_per_label(base_img: Image, out_labels: Image) -> Image:
    """Per muscle label ID, keep only the largest 3D connected component."""
    for lid in np.unique(out_labels.data):
        bin_mask = (out_labels.data == lid)
        if not np.any(bin_mask):
            continue
        cc = biggest_cc(base_img.with_data(bin_mask))
        out_labels.data[out_labels.data == lid] = 0
        out_labels.data[cc.data > 0] = lid
    return out_labels


def _kidney_erode(kidney: Image, iterations: int = _KIDNEY_ERODE) -> Image:
    """Erode the kidney mask by *iterations*, falling back to the original (unwarned-empty) mask if
    erosion removes everything."""
    binary = (kidney.data > 0).astype(np.uint8)
    eroded = erode(kidney.with_data(binary), iterations=iterations)
    if not bool(eroded.data.any()):
        return kidney.with_data(np.zeros_like(kidney.data, dtype=np.uint8))
    log.warning(f"Kidney not eroded, using original mask [Empty mask post-erode]")
    return kidney


def _kidney_remove_pelvis(kidney: Image, *, dilate_iters: int = 1) -> Image:
    """Remove estimated renal pelvis from a kidney mask.

    Steps:
    - Compute 3D convex hull of the kidney
    - Pelvis candidate = hull \\ kidney
    - Keep biggest CC of candidate (if any)
    - Dilate candidate *dilate_iters* iterations (none when 0)
    - Subtract candidate from kidney
    """
    binary = (kidney.data > 0).astype(np.uint8)
    if not bool(binary.any()):
        return kidney.with_data(binary)

    hull = convex_hull_3d(kidney.with_data(binary))
    hull_arr = hull.data if hasattr(hull, "data") else hull
    pelvis = ((as_backend_array(hull_arr) > 0) & (binary == 0)).astype(np.uint8)
    if not bool(pelvis.any()):
        return kidney.with_data(binary)

    pelvis_img = biggest_cc(kidney.with_data(pelvis))
    if not bool(pelvis_img.data.any()):
        return kidney.with_data(binary)

    if int(dilate_iters) > 0:
        pelvis_img = dilate(pelvis_img, footprint=1, iterations=int(dilate_iters), mode="binary")
    pelvis_mask = (pelvis_img.data > 0)
    cleaned = (binary > 0) & (~pelvis_mask)
    return kidney.with_data(cleaned.astype(np.uint8))


def build_skeleton_mask(
    total_mr: Image | None,
    *,
    dilate_rois: dict[str, int] | None = None,
) -> Image | None:
    """Union of the :data:`cfg.SKELETON_ROIS_MR` bone classes, or ``None``.

    ``None`` means there is nothing to subtract: either the region has no
    ``total_mr`` output (a stage-1 run predating the bone ROIs) or the model
    found none of those bones. Classes in *dilate_rois* are grown by their radius
    (voxels) before joining the union.
    """
    if total_mr is None:
        log.warning(
            "No 'total_mr' output for this region; skipping skeleton subtraction. "
            "Re-run stage 1 so it segments cfg.SKELETON_ROIS_MR."
        )
        return None

    ids = []
    for name in cfg.SKELETON_ROIS_MR:
        try:
            ids.append(get_class_id(name, "total_mr"))
        except ValueError:
            log.debug(f"Skeleton ROI {name!r} is not a 'total_mr' class; skipping.")

    present = [cid for cid in ids if bool(np.any(total_mr.data == cid))]
    if not present:
        log.warning("No skeleton bone labels found in 'total_mr'; nothing to subtract.")
        return None

    bone = combine_labels(total_mr, present, new_id=1).data
    for name, radius in (dilate_rois or {}).items():
        try:
            cid = get_class_id(name, "total_mr")
        except ValueError:
            continue
        if cid not in present or int(radius) <= 0:
            continue
        single = total_mr.with_data((total_mr.data == cid).astype(np.uint8))
        bone[dilate(single, footprint=int(radius)).data > 0] = 1
    return total_mr.with_data(bone.astype(np.uint8))


def _subtract_skeleton(out_labels: Image, skeleton: Image | None, label_names: tuple[str, ...],
                       labels_map: dict[str, int]) -> Image:
    """Clear the named labels of *out_labels* wherever *skeleton* is positive."""
    if skeleton is None:
        return out_labels
    if skeleton.data.shape != out_labels.data.shape:
        log.warning(
            f"Skeleton grid {tuple(skeleton.data.shape)} does not match the mask grid "
            f"{tuple(out_labels.data.shape)}; skipping skeleton subtraction."
        )
        return out_labels
    bone = skeleton.data > 0
    if not bool(np.any(bone)):
        return out_labels
    arr = out_labels.data.copy()
    for name in label_names:
        lid = labels_map.get(name)
        if lid is None:
            continue
        arr[(arr == lid) & bone] = 0
    return out_labels.with_data(arr)


# ---------------------------------------------------------------------------
# Per-region mask builders
# ---------------------------------------------------------------------------


def build_head_mask(head_total_mr: Image) -> Image:
    """HEAD: H_PVM_L / H_PVM_R (autochthon L/R, largest CC each)."""
    left = _biggest_cc_or_empty(
        head_total_mr, get_class_id("autochthon_left", "total_mr")
    )
    right = _biggest_cc_or_empty(
        head_total_mr, get_class_id("autochthon_right", "total_mr")
    )
    out = np.zeros_like(head_total_mr.data, dtype=np.uint8)
    out[left.data > 0] = HEAD_LABELS["H_PVM_L"]
    out[right.data > 0] = HEAD_LABELS["H_PVM_R"]
    return _muscles_keep_biggest_cc_per_label(head_total_mr, head_total_mr.with_data(out))


def _liver_without_vessels(liver: Image, t2star: Image | None) -> Image:
    """*liver* minus its vessels: voxels whose T2* exceeds :data:`cfg.LIVER_VESSEL_T2STAR_MS`."""
    if t2star is None:
        log.warning("DIXON_THORAX_T2STAR not found; liver vessels are not removed.")
        return liver
    if tuple(t2star.data.shape) != tuple(liver.data.shape):
        log.warning(
            f"T2STAR grid {tuple(t2star.data.shape)} does not match the liver grid "
            f"{tuple(liver.data.shape)}; liver vessels are not removed."
        )
        return liver
    vessels = as_backend_array(t2star.data) > cfg.LIVER_VESSEL_T2STAR_MS
    return liver.with_data(((as_backend_array(liver.data) > 0) & ~vessels).astype(np.uint8))


def build_thorax_mask(
    thorax_total_mr: Image,
    thorax_vertebrae_mr: Image,
    thorax_total_mr_water: Image | None = None,
    t2star: Image | None = None,
) -> tuple[Image, Image]:
    """THORAX and THORAX_VOL: liver, pancreas, kidneys L/R, paravertebral L/R and BN_L3/L4.

    The liver comes from *thorax_total_mr_water* (the WATER-contrast ``total_mr``
    run). Everything else comes from the FAT-contrast run. When the WATER output
    is missing the liver falls back to the FAT run with a warning, so batches
    segmented before the split still process.

    THORAX (FF/T2*/R2*/WF) drops the liver vessels (*t2star* above
    :data:`cfg.LIVER_VESSEL_T2STAR_MS`), clears each kidney's pelvis dilated by
    two iterations and erodes the vertebrae to bone narrow. THORAX_VOL (volumes)
    keeps the vessels, clears the pelvis undilated and keeps the raw vertebrae.
    """
    liver_source = thorax_total_mr_water
    if liver_source is None:
        log.warning(
            f"{cfg.THORAX_WATER_STEM}.nii(.gz) not found; segmenting the liver from the FAT "
            "contrast. Re-run stage 1 to get the WATER-based liver."
        )
        liver_source = thorax_total_mr
    elif liver_source.data.shape != thorax_total_mr.data.shape:
        log.warning(
            f"WATER liver grid {tuple(liver_source.data.shape)} does not match the FAT grid "
            f"{tuple(thorax_total_mr.data.shape)}; falling back to the FAT-contrast liver."
        )
        liver_source = thorax_total_mr

    liver = _biggest_cc_or_empty(liver_source, get_class_id("liver", "total_mr"))
    pancreas = _biggest_cc_or_empty(thorax_total_mr, get_class_id("pancreas", "total_mr"))
    kidney_l = _biggest_cc_or_empty(thorax_total_mr, get_class_id("kidney_left", "total_mr"))
    kidney_r = _biggest_cc_or_empty(thorax_total_mr, get_class_id("kidney_right", "total_mr"))
    pvm_l = _biggest_cc_or_empty(thorax_total_mr, get_class_id("autochthon_left", "total_mr"))
    pvm_r = _biggest_cc_or_empty(thorax_total_mr, get_class_id("autochthon_right", "total_mr"))
    vert_l3 = get_label(
        thorax_vertebrae_mr, get_class_id("vertebrae_L3", "vertebrae_mr"), missing="empty"
    )
    vert_l4 = get_label(
        thorax_vertebrae_mr, get_class_id("vertebrae_L4", "vertebrae_mr"), missing="empty"
    )

    def assemble(liver_m: Image, kidney_l_m: Image, kidney_r_m: Image, l3: Image, l4: Image) -> Image:
        """Label image from the per-structure masks (later labels win overlaps)."""
        out = np.zeros_like(thorax_total_mr.data, dtype=np.uint8)
        out[liver_m.data > 0] = THORAX_LABELS["LIVER"]
        out[pancreas.data > 0] = THORAX_LABELS["PANCREAS"]
        out[kidney_l_m.data > 0] = THORAX_LABELS["KIDNEY_L"]
        out[kidney_r_m.data > 0] = THORAX_LABELS["KIDNEY_R"]
        out[pvm_l.data > 0] = THORAX_LABELS["T_PVM_L"]
        out[pvm_r.data > 0] = THORAX_LABELS["T_PVM_R"]
        out[l3.data > 0] = THORAX_LABELS["BN_L3"]
        out[l4.data > 0] = THORAX_LABELS["BN_L4"]
        return _muscles_keep_biggest_cc_per_label(thorax_total_mr, thorax_total_mr.with_data(out))

    thorax = assemble(
        _liver_without_vessels(liver, t2star),
        _kidney_remove_pelvis(kidney_l, dilate_iters=2),
        _kidney_remove_pelvis(kidney_r, dilate_iters=2),
        _vertebra_narrow(vert_l3),
        _vertebra_narrow(vert_l4),
    )
    thorax_vol = assemble(
        liver,
        _kidney_remove_pelvis(kidney_l, dilate_iters=0),
        _kidney_remove_pelvis(kidney_r, dilate_iters=0),
        vert_l3,
        vert_l4,
    )
    return thorax, thorax_vol


def build_legs_mask(legs_muscles_mr: Image, skeleton: Image | None = None) -> Image:
    """LEGS: quadriceps L/R (``L_QM_L`` / ``L_QM_R``, largest CC each).

    With a *skeleton* mask the femur/hip bones are subtracted from both
    quadriceps. Components are resolved first, so splitting a muscle around the
    femur cannot cost it half its volume.
    """
    left = _biggest_cc_or_empty(
        legs_muscles_mr,
        get_class_id("quadriceps_femoris_left", "thigh_shoulder_muscles_mr"),
    )
    right = _biggest_cc_or_empty(
        legs_muscles_mr,
        get_class_id("quadriceps_femoris_right", "thigh_shoulder_muscles_mr"),
    )
    out = np.zeros_like(legs_muscles_mr.data, dtype=np.uint8)
    out[left.data > 0] = LEGS_LABELS["L_QM_L"]
    out[right.data > 0] = LEGS_LABELS["L_QM_R"]
    labels = _muscles_keep_biggest_cc_per_label(legs_muscles_mr, legs_muscles_mr.with_data(out))
    return _subtract_skeleton(labels, skeleton, cfg.SKELETON_SUBTRACT_FROM_LEGS, LEGS_LABELS)


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------


def _process(segmentation_dir: Path, nifti_dir: Path, output_dir: Path) -> None:
    """Build and write the per-region post-processed Dixon masks for one subject from its
    TotalSegmentator outputs and Dixon maps: HEAD/THORAX/LEGS (eroded by :data:`MEASURE_ERODE`)
    and their uneroded ``_VOL`` twins, skipping any region whose inputs are missing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # HEAD
    head_dir = segmentation_dir / f"{cfg.INPUT_PREFIX}_HEAD"
    head_total = _imread_opt(head_dir, "total_mr")
    if head_total is not None:
        head_vol = build_head_mask(head_total)
        head_mask = erode_labels(head_vol, HEAD_LABELS, MEASURE_ERODE)
        imsave(str(output_dir / "HEAD.nii.gz"), head_mask, axes="XYZ")
        imsave(str(output_dir / "HEAD_VOL.nii.gz"), head_vol, axes="XYZ")
    else:
        log.warning(f"HEAD/total_mr missing in {segmentation_dir} - skipping HEAD")

    # THORAX
    thorax_dir = segmentation_dir / f"{cfg.INPUT_PREFIX}_THORAX"
    thorax_total = _imread_opt(thorax_dir, "total_mr")
    if thorax_total is not None:
        thorax_vert = _imread_opt(thorax_dir, "vertebrae_mr")
        if thorax_vert is None:
            thorax_vert = thorax_total.with_data(
                np.zeros_like(thorax_total.data, dtype=np.uint8)
            )
        thorax_water = _imread_opt(thorax_dir, cfg.THORAX_WATER_STEM)
        thorax_t2star = _imread_opt(nifti_dir, f"{cfg.INPUT_PREFIX}_THORAX_T2STAR")
        thorax_mask, thorax_vol = build_thorax_mask(
            thorax_total, thorax_vert, thorax_water, thorax_t2star
        )
        thorax_mask = erode_labels(thorax_mask, THORAX_LABELS, MEASURE_ERODE)
        imsave(str(output_dir / "THORAX.nii.gz"), thorax_mask, axes="XYZ")
        imsave(str(output_dir / "THORAX_VOL.nii.gz"), thorax_vol, axes="XYZ")
    else:
        log.warning(f"THORAX/total_mr missing in {segmentation_dir} - skipping THORAX")

    # LEGS
    legs_dir = segmentation_dir / f"{cfg.INPUT_PREFIX}_LEGS"
    legs_muscles = _imread_opt(legs_dir, "thigh_shoulder_muscles_mr")
    if legs_muscles is not None:
        legs_total = _imread_opt(legs_dir, "total_mr")
        legs_skeleton_raw = build_skeleton_mask(legs_total)
        legs_skeleton = (
            build_skeleton_mask(legs_total, dilate_rois=cfg.LEGS_SKELETON_DILATE)
            if legs_skeleton_raw is not None
            else None
        )
        legs_mask = erode_labels(
            build_legs_mask(legs_muscles, skeleton=legs_skeleton), LEGS_LABELS, MEASURE_ERODE
        )
        legs_vol = build_legs_mask(legs_muscles, skeleton=legs_skeleton_raw)
        imsave(str(output_dir / "LEGS.nii.gz"), legs_mask, axes="XYZ")
        imsave(str(output_dir / "LEGS_VOL.nii.gz"), legs_vol, axes="XYZ")
        if legs_skeleton is not None:
            imsave(str(output_dir / "LEGS_SKELETON.nii.gz"), legs_skeleton, axes="XYZ")
    else:
        log.warning(
            f"LEGS/thigh_shoulder_muscles_mr missing in {segmentation_dir} - skipping LEGS"
        )


def run_subject(subject: str, lay: BatchLayout, *, backend: str = "cupy") -> Path:
    """Build HEAD/THORAX/LEGS stage-2 outputs for a single subject."""
    try:
        set_default_backend(backend, allow_fallback=True)
    except Exception as exc:
        log.warning(f"Backend '{backend}' unavailable, falling back: {exc}")

    seg_dir = lay.results_dir / cfg.STAGE1_DIR / subject
    if not seg_dir.exists():
        raise FileNotFoundError(f"Expected stage-1 outputs under {seg_dir}")

    out_dir = lay.results_dir / cfg.STAGE2_DIR / subject
    log.info(f"Dixon v5 stage 2 | subject={subject}")
    _process(seg_dir, lay.subject_nifti_dir(subject), out_dir)
    log.info(f"[{subject}] ok -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("dixon-v5-stage2")
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
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    model_dir: Path | None,
    backend: str,
    log_level: str,
) -> None:
    """Dixon v5 stage 2 worker (single subject)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
        model_root=model_dir,
    )
    run_subject(subject, lay, backend=backend)


if __name__ == "__main__":
    main()
