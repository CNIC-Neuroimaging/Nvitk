"""Brain mask from TotalSegmentator for qvtpy venous centerline filtering.

Runs ``total_mr`` on ``Angiography_3D`` with ROI subset ``brain``, using
qvtpy-configured TotalSegmentator model roots (see :mod:`nvitk.pipes.qvtpy.config`).
"""

from __future__ import annotations

from pathlib import Path

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy.util.paths import resolve_totalseg_model_dir
from nvitk.segmentation.total_segmentator import run_totalsegmentator
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.transform.resampling import resample_to
from nvitk.types import Image

setup(globals())

log = Logger()

TOTAL_MR_TASK = "total_mr"
BRAIN_ROI = ("brain",)
BRAIN_MASK_NIFTI = "brain_mask_from_angio.nii.gz"
TOTAL_MR_MULTILABEL_NIFTI = "total_mr_brain_subset.nii.gz"


def find_angio_volume(nifti_root: Path, subject: str) -> Path:
    """Locate ``4DFlow/Angiography_3D.nii[.gz]`` for *subject*."""
    flow_dir = nifti_root / subject / "4DFlow"
    for name in ("Angiography_3D.nii.gz", "Angiography_3D.nii"):
        p = flow_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No Angiography_3D under {flow_dir}")


def _resolve_totalseg_output(path: Path) -> Path:
    if path.is_file():
        return path
    for name in (f"{TOTAL_MR_TASK}.nii.gz", f"{TOTAL_MR_TASK}.nii"):
        candidate = path.parent / name if path.suffix in {"", ".nii", ".gz"} else path / name
        if candidate.is_file():
            return candidate
    if path.with_suffix(".nii.gz").is_file():
        return path.with_suffix(".nii.gz")
    if path.with_suffix(".nii").is_file():
        return path.with_suffix(".nii")
    raise FileNotFoundError(f"TotalSegmentator output not found near {path}")


def extract_brain_binary(multilabel: np.ndarray) -> np.ndarray:
    """Binary brain mask from a ``total_mr`` multilabel volume."""
    arr = to_numpy(as_backend_array(multilabel)).astype(np.int32, copy=False)
    brain_id = int(get_class_id("brain", TOTAL_MR_TASK))
    mask = arr == brain_id
    if not bool(mask.any()):
        mask = arr > 0
    return as_backend_array(mask.astype(bool, copy=False))


def align_brain_mask_to_target(brain_mask: np.ndarray, *, source_img: Image, target_img: Image) -> np.ndarray:
    """Resample *brain_mask* onto *target_img* grid when shape/affine differ."""
    src = source_img.with_data(as_backend_array(brain_mask.astype(np.uint8, copy=False)))
    tgt_shape = tuple(int(s) for s in target_img.shape[:3])
    src_shape = tuple(int(s) for s in source_img.shape[:3])
    if src_shape == tgt_shape:
        src_aff = getattr(source_img, "affine", None)
        tgt_aff = getattr(target_img, "affine", None)
        if src_aff is not None and tgt_aff is not None:
            if np.allclose(np.asarray(src_aff), np.asarray(tgt_aff), atol=1e-3):
                return as_backend_array(brain_mask.astype(bool, copy=False))
    resampled = resample_to(src, target_img, order=0, prefilter=False)
    return as_backend_array(to_numpy(resampled.data) > 0)


def segment_brain_mask_from_angio(
    angio_path: Path,
    out_dir: Path,
    *,
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = True,
) -> tuple[np.ndarray, Image]:
    """Run TotalSegmentator (``total_mr`` / ``brain``) and return a binary brain mask."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / BRAIN_MASK_NIFTI
    multilabel_path = out_dir / TOTAL_MR_MULTILABEL_NIFTI

    model_root = resolve_totalseg_model_dir(model_dir=model_dir)
    angio_img = imread(angio_path)

    if not overwrite and mask_path.is_file():
        log.info(f"brain mask: reuse cached {mask_path}")
        cached = imread(mask_path)
        mask = as_backend_array(to_numpy(cached.data) > 0)
        return mask, angio_img

    log.step(f"TotalSegmentator {TOTAL_MR_TASK} (roi=brain) on {angio_path.name}")
    log.info(f"  models: {model_root}")
    run_totalsegmentator(
        angio_path,
        multilabel_path,
        task=TOTAL_MR_TASK,
        device=device,
        roi_subset=list(BRAIN_ROI),
        multilabel=True,
        statistics=False,
        model_dir=model_root,
        check=True,
        capture_output=False,
    )
    ml_path = _resolve_totalseg_output(multilabel_path)
    ml_img = imread(ml_path)
    brain_on_angio = extract_brain_binary(ml_img.data)
    imsave(mask_path, brain_on_angio.astype(np.uint8), metadata=dict(angio_img.metadata or {}))
    log.info(f"brain mask: {int(brain_on_angio.sum())} voxels -> {mask_path}")
    return brain_on_angio, angio_img


def brain_mask_for_reference(
    nifti_root: Path,
    subject: str,
    out_dir: Path,
    target_img: Image,
    *,
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = True,
) -> np.ndarray:
    """Brain mask resampled onto *target_img* (typically ComplexDifference_3D)."""
    angio_path = find_angio_volume(nifti_root, subject)
    brain_on_angio, angio_img = segment_brain_mask_from_angio(
        angio_path,
        out_dir,
        device=device,
        model_dir=model_dir,
        overwrite=overwrite,
    )
    return align_brain_mask_to_target(
        brain_on_angio,
        source_img=angio_img,
        target_img=target_img,
    )


__all__ = [
    "BRAIN_MASK_NIFTI",
    "BRAIN_ROI",
    "TOTAL_MR_MULTILABEL_NIFTI",
    "TOTAL_MR_TASK",
    "align_brain_mask_to_target",
    "brain_mask_for_reference",
    "extract_brain_binary",
    "find_angio_volume",
    "segment_brain_mask_from_angio",
]
