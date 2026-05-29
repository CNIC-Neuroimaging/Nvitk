"""Stage1: crop whole-body PET to brain region (Z-only) using CT TotalSegmentator mask.

Locked methodology (plan):
- Run TotalSegmentator on CT (IA_PET_V5) to obtain a stable head ROI (brain+skull).
- Resample CT mask → PET grid with nearest-neighbor (assumes PET/CT are co-registered).
- Keep X/Y unchanged; crop only Z with:
  - margin_mm = 10
  - Nz_fixed = 128 slices
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.pipes.pesa_fat.common.paths import DEFAULT_MODEL_ROOT
from nvitk.segmentation.labels import combine_labels
from nvitk.segmentation.total_segmentator import run_totalsegmentator
from nvitk.segmentation.total_segmentator.class_maps import get_class_id
from nvitk.transform.resampling import resample_mask_to_pet
from nvitk.types import Image

from .layout import GpetLayout

log = Logger()


@dataclass(frozen=True)
class CropMeta:
    z0: int
    z1: int
    nz_fixed: int
    margin_mm: float
    pet_spacing_z: float
    mask_zmin: int
    mask_zmax: int
    ts_task: str
    ts_label_ids: tuple[int, ...]


def _spacing_z(img: Image) -> float:
    md = img.metadata or {}
    if "spacing" in md and md["spacing"] is not None:
        try:
            return float(md["spacing"][2])
        except Exception:
            pass
    for k in ("z_res", "z_spacing", "spacing_z"):
        if k in md and md[k] is not None:
            try:
                return float(md[k])
            except Exception:
                pass
    # Fallback: infer from affine column norm.
    aff = img.affine
    if aff is None:
        raise ValueError("PET image missing spacing metadata and affine.")
    return float((aff[:3, 2] ** 2).sum() ** 0.5)


def _crop_affine_z(affine: "np.ndarray", z0: int) -> "np.ndarray":
    if affine.shape != (4, 4):
        raise ValueError(f"Expected affine (4,4), got {affine.shape}")
    new_aff = affine.astype(float).copy()
    new_origin = (affine @ np.array([0.0, 0.0, float(z0), 1.0]))[:3]
    new_aff[:3, 3] = new_origin
    return new_aff


def _compute_fixed_z_window(
    *,
    pet_nz: int,
    mask_zmin: int,
    mask_zmax: int,
    spacing_z: float,
    margin_mm: float,
    nz_fixed: int,
) -> tuple[int, int]:
    """Return (z0, z1) half-open indices on the PET z-axis."""
    if pet_nz <= 0:
        raise ValueError("pet_nz must be > 0")
    if nz_fixed <= 0:
        raise ValueError("nz_fixed must be > 0")
    if pet_nz < nz_fixed:
        raise ValueError(f"PET nz={pet_nz} is smaller than nz_fixed={nz_fixed}")
    if mask_zmin > mask_zmax:
        raise ValueError(f"mask_zmin={mask_zmin} > mask_zmax={mask_zmax}")

    margin_slices = int(round(float(margin_mm) / float(spacing_z))) if spacing_z > 0 else 0
    lo = max(0, int(mask_zmin) - margin_slices)
    hi = min(pet_nz - 1, int(mask_zmax) + margin_slices)
    center = int(round((lo + hi) / 2.0))

    half = nz_fixed // 2
    if nz_fixed % 2 == 0:
        z0 = center - half
        z1 = center + half
    else:
        z0 = center - half
        z1 = center + half + 1

    # shift inside bounds
    if z0 < 0:
        z1 = min(pet_nz, z1 - z0)
        z0 = 0
    if z1 > pet_nz:
        shift = z1 - pet_nz
        z0 = max(0, z0 - shift)
        z1 = pet_nz

    # enforce exact length
    if (z1 - z0) != nz_fixed:
        # last resort: clamp to a valid fixed window
        z0 = min(max(0, z0), pet_nz - nz_fixed)
        z1 = z0 + nz_fixed
    if (z1 - z0) != nz_fixed:
        raise RuntimeError(f"Failed to compute fixed window: got [{z0}:{z1}] (len={z1-z0})")

    return int(z0), int(z1)


def run_subject(
    subject: str,
    lay: GpetLayout,
    *,
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = True,
    margin_mm: float = 10.0,
    nz_fixed: int = 128,
) -> CropMeta:
    subj = str(subject).strip()
    if not subj:
        raise ValueError("subject must be non-empty")

    ct_path = lay.nifti_ct()
    pet_path = lay.nifti_pet()
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT NIfTI not found: {ct_path}")
    if not pet_path.is_file():
        raise FileNotFoundError(f"PET NIfTI not found: {pet_path}")

    out_dir = lay.stage_dir("stage1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pet = lay.stage1_pet_brain()
    out_mask = lay.stage1_brain_mask_pet()
    out_meta = lay.stage1_meta()
    if not overwrite and out_pet.exists() and out_mask.exists() and out_meta.exists():
        meta = json.loads(out_meta.read_text(encoding="utf-8"))
        return CropMeta(**meta)

    # 1) TotalSegmentator on CT → multilabel seg
    ts_task = "total"
    ts_out = out_dir / "totalseg_total.nii.gz"
    if overwrite or not ts_out.exists():
        model_dir = model_dir or DEFAULT_MODEL_ROOT
        run_totalsegmentator(
            ct_path,
            ts_out,
            task=ts_task,
            device=device,
            multilabel=True,
            statistics=False,
            model_dir=model_dir,
            check=True,
            capture_output=False,
        )

    seg = imread(ts_out)
    if not isinstance(seg, Image):
        raise TypeError(f"Expected Image from imread({ts_out}), got {type(seg)}")

    # 2) Build brain+skull mask on CT grid
    brain_id = get_class_id("brain", ts_task)
    skull_id = get_class_id("skull", ts_task)
    mask_ct = combine_labels(seg, (brain_id, skull_id), new_id=1)
    if not isinstance(mask_ct, Image):
        raise TypeError("combine_labels returned non-Image unexpectedly.")

    # 3) Resample CT mask → PET grid
    pet = imread(pet_path)
    if not isinstance(pet, Image):
        raise TypeError(f"Expected Image from imread({pet_path}), got {type(pet)}")
    mask_pet = resample_mask_to_pet(mask_ct, pet, order=0)

    # 4) Compute Z bounds in PET voxels
    m = to_numpy(mask_pet.data)
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={m.shape}")
    z_any = to_numpy((m > 0).any(axis=(0, 1)))
    if not bool(z_any.any()):
        raise RuntimeError("Brain/skull mask is empty after resampling to PET grid.")
    z_idx = to_numpy(z_any.nonzero()[0])
    zmin = int(z_idx.min())
    zmax = int(z_idx.max())

    # 5) Fixed window selection
    spacing_z = _spacing_z(pet)
    z0, z1 = _compute_fixed_z_window(
        pet_nz=int(pet.shape[2]),
        mask_zmin=zmin,
        mask_zmax=zmax,
        spacing_z=spacing_z,
        margin_mm=float(margin_mm),
        nz_fixed=int(nz_fixed),
    )

    # 6) Crop PET + mask; update affine
    pet_data = as_backend_array(pet.data)[:, :, z0:z1]
    mask_data = as_backend_array(mask_pet.data)[:, :, z0:z1]
    new_aff = _crop_affine_z(pet.affine, z0)

    pet_md = dict(pet.metadata or {})
    pet_md["affine"] = new_aff
    pet_md["shape"] = tuple(getattr(pet_data, "shape", ()))
    pet_crop = Image(
        data=pet_data,
        metadata=pet_md,
        axes=pet.axes,
        name=pet.name,
        orientation=pet_md.get("orientation"),
    )

    mask_md = dict(mask_pet.metadata or {})
    mask_md["affine"] = new_aff
    mask_md["shape"] = tuple(getattr(mask_data, "shape", ()))
    mask_crop = Image(
        data=mask_data,
        metadata=mask_md,
        axes=mask_pet.axes,
        name=mask_pet.name,
        orientation=mask_md.get("orientation"),
    )

    imsave(pet_crop, out_pet)
    imsave(mask_crop, out_mask)

    meta = CropMeta(
        z0=z0,
        z1=z1,
        nz_fixed=int(nz_fixed),
        margin_mm=float(margin_mm),
        pet_spacing_z=float(spacing_z),
        mask_zmin=zmin,
        mask_zmax=zmax,
        ts_task=ts_task,
        ts_label_ids=(int(brain_id), int(skull_id)),
    )
    out_meta.write_text(json.dumps({**meta.__dict__, "created_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")
    log.info("[%s] gpetpy stage1 crop OK (z=%s:%s)", subj, z0, z1)
    return meta


__all__ = ["CropMeta", "run_subject", "_compute_fixed_z_window"]

