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
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option, set_default_backend
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.morphology import dilate, erode
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
from nvitk.segmentation.labels import biggest_cc, get_label
from nvitk.segmentation.hull_edt import convex_hull_3d
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.types import Image

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
    path = resolve_nii_optional(parent, stem)
    if path is None:
        return None
    return imread(str(path), axes=axes)


def _vertebra_narrow(vertebra: Image, radius: int = _BN_ERODE) -> Image:
    binary = (vertebra.data > 0).astype(np.uint8)
    eroded = erode(vertebra.with_data(binary), footprint=radius)
    if not bool(eroded.data.any()):
        return vertebra.with_data(np.zeros_like(vertebra.data, dtype=np.uint8))
    return biggest_cc(eroded)


def _biggest_cc_or_empty(label_img: Image, label_id: int) -> Image:
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
    - Dilate candidate one iter
    - Subtract candidate from kidney
    """
    binary = (kidney.data > 0).astype(np.uint8)
    if not bool(binary.any()):
        return kidney.with_data(binary)

    hull = convex_hull_3d(kidney.with_data(binary))
    hull_arr = hull.data if hasattr(hull, "data") else hull
    pelvis = ((np.asarray(hull_arr) > 0) & (binary == 0)).astype(np.uint8)
    if not bool(pelvis.any()):
        return kidney.with_data(binary)

    pelvis_img = biggest_cc(kidney.with_data(pelvis))
    if not bool(pelvis_img.data.any()):
        return kidney.with_data(binary)

    pelvis_dil = dilate(pelvis_img, footprint=1, iterations=int(dilate_iters), mode="binary")
    pelvis_mask = (pelvis_dil.data > 0)
    cleaned = (binary > 0) & (~pelvis_mask)
    return kidney.with_data(cleaned.astype(np.uint8))


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


def build_thorax_mask(
    thorax_total_mr: Image,
    thorax_vertebrae_mr: Image,
) -> Image:
    """THORAX: liver, pancreas, kidneys L/R, paravertebral L/R and BN_L3/L4."""
    out = np.zeros_like(thorax_total_mr.data, dtype=np.uint8)

    liver = _biggest_cc_or_empty(thorax_total_mr, get_class_id("liver", "total_mr"))
    pancreas = _biggest_cc_or_empty(thorax_total_mr, get_class_id("pancreas", "total_mr"))
    kidney_l = _biggest_cc_or_empty(thorax_total_mr, get_class_id("kidney_left", "total_mr"))
    kidney_r = _biggest_cc_or_empty(thorax_total_mr, get_class_id("kidney_right", "total_mr"))
    kidney_l = _kidney_remove_pelvis(kidney_l, dilate_iters=2)
    kidney_r = _kidney_remove_pelvis(kidney_r, dilate_iters=2)

    out[liver.data > 0] = THORAX_LABELS["LIVER"]
    out[pancreas.data > 0] = THORAX_LABELS["PANCREAS"]
    out[kidney_l.data > 0] = THORAX_LABELS["KIDNEY_L"]
    out[kidney_r.data > 0] = THORAX_LABELS["KIDNEY_R"]

    pvm_l = _biggest_cc_or_empty(thorax_total_mr, get_class_id("autochthon_left", "total_mr"))
    pvm_r = _biggest_cc_or_empty(thorax_total_mr, get_class_id("autochthon_right", "total_mr"))
    out[pvm_l.data > 0] = THORAX_LABELS["T_PVM_L"]
    out[pvm_r.data > 0] = THORAX_LABELS["T_PVM_R"]

    bn_l3 = _vertebra_narrow(
        get_label(
            thorax_vertebrae_mr,
            get_class_id("vertebrae_L3", "vertebrae_mr"),
            missing="empty",
        )
    )
    bn_l4 = _vertebra_narrow(
        get_label(
            thorax_vertebrae_mr,
            get_class_id("vertebrae_L4", "vertebrae_mr"),
            missing="empty",
        )
    )
    out[bn_l3.data > 0] = THORAX_LABELS["BN_L3"]
    out[bn_l4.data > 0] = THORAX_LABELS["BN_L4"]

    return _muscles_keep_biggest_cc_per_label(thorax_total_mr, thorax_total_mr.with_data(out))


def build_legs_mask(legs_muscles_mr: Image) -> Image:
    """LEGS: quadriceps L/R (``L_QM_L`` / ``L_QM_R``, largest CC each)."""
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
    return _muscles_keep_biggest_cc_per_label(legs_muscles_mr, legs_muscles_mr.with_data(out))


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------


def _process(segmentation_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # HEAD
    head_dir = segmentation_dir / f"{cfg.INPUT_PREFIX}_HEAD"
    head_total = _imread_opt(head_dir, "total_mr")
    if head_total is not None:
        head_mask = build_head_mask(head_total)
        imsave(str(output_dir / "HEAD.nii.gz"), head_mask, axes="XYZ")
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
        thorax_mask = build_thorax_mask(thorax_total, thorax_vert)
        imsave(str(output_dir / "THORAX.nii.gz"), thorax_mask, axes="XYZ")
    else:
        log.warning(f"THORAX/total_mr missing in {segmentation_dir} - skipping THORAX")

    # LEGS
    legs_dir = segmentation_dir / f"{cfg.INPUT_PREFIX}_LEGS"
    legs_muscles = _imread_opt(legs_dir, "thigh_shoulder_muscles_mr")
    if legs_muscles is not None:
        legs_mask = build_legs_mask(legs_muscles)
        imsave(str(output_dir / "LEGS.nii.gz"), legs_mask, axes="XYZ")
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
    _process(seg_dir, out_dir)
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
    """Dixon v5 stage 2 worker (single subject)."""
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
