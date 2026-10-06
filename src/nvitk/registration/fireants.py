"""FireANTs registration backend (GPU).

FireANTs is a PyTorch registration library. The pip package ships a Python API
(``fireants.registration``) but not the ``fireantsRegistration`` command-line
tool the first version of this wrapper drove, so registration now runs through
the API: an optional moments initialisation, then any chain of rigid → affine →
greedy / SyN stages, each stage initialised from the one before.

Transforms are written in ANTs format (``.mat`` for rigid/affine, a displacement
``.nii.gz`` for greedy/SyN — which already *includes* the affine it was
initialised with), so :func:`fireants_apply` maps further images with
``ants.apply_transforms`` and the files interoperate with every ANTs tool.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.exceptions import BackendUnavailableError

#: Stages FireANTs can chain, in the order they may appear.
FIREANTS_STAGES: tuple[str, ...] = ("moments", "rigid", "affine", "greedy", "syn")

#: Named stage chains offered by the GUI (``+``-joined :data:`FIREANTS_STAGES`).
FIREANTS_TRANSFORMS: tuple[str, ...] = (
    "rigid",
    "affine",
    "rigid+affine",
    "moments+rigid+affine",
    "greedy",
    "affine+greedy",
    "rigid+affine+greedy",
    "syn",
    "affine+syn",
    "rigid+affine+syn",
    "moments+rigid+affine+syn",
)

#: Image similarity losses understood by every FireANTs stage.
FIREANTS_LOSSES: tuple[str, ...] = ("cc", "mi", "mse")


@dataclass(frozen=True)
class FireAntsResult:
    """Outputs of a FireANTs registration: the warped moving image and the transforms."""

    output_prefix: Path
    warped_moving_path: Path
    #: ANTs-format transforms mapping MOVING → FIXED, in ``ants.apply_transforms`` order.
    transforms: tuple[Path, ...] = field(default_factory=tuple)
    #: Stages that ran, e.g. ``("rigid", "affine", "syn")``.
    stages: tuple[str, ...] = field(default_factory=tuple)


def parse_stages(transform: str | Sequence[str]) -> tuple[str, ...]:
    """``"rigid+affine+syn"`` (or a sequence) → validated stage tuple, in canonical order."""
    if isinstance(transform, str):
        parts = [p.strip().lower() for p in transform.replace(",", "+").split("+")]
    else:
        parts = [str(p).strip().lower() for p in transform]
    parts = [p for p in parts if p]
    unknown = [p for p in parts if p not in FIREANTS_STAGES]
    if unknown:
        raise ValueError(f"Unknown FireANTs stage(s) {unknown}; choose from {FIREANTS_STAGES}.")
    if not parts:
        raise ValueError("Select at least one FireANTs stage.")
    if "greedy" in parts and "syn" in parts:
        raise ValueError("Use either a greedy or a SyN deformable stage, not both.")
    if parts == ["moments"]:
        raise ValueError("Moments is an initialiser: add a rigid or affine stage after it.")
    return tuple(s for s in FIREANTS_STAGES if s in parts)


def _int_list(values: Sequence[Any] | str, name: str) -> list[int]:
    """``"4,2,1"`` / ``[4, 2, 1]`` → ``[4, 2, 1]``."""
    if isinstance(values, str):
        values = [v for v in values.replace("x", ",").split(",") if v.strip()]
    out = [int(float(v)) for v in values]
    if not out:
        raise ValueError(f"{name} must list at least one value.")
    return out


def _require_fireants() -> dict[str, Any]:
    """Import the FireANTs API pieces, or raise an install hint."""
    try:
        from fireants.io.image import BatchedImages, Image
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
        from fireants.registration.moments import MomentsRegistration
        from fireants.registration.rigid import RigidRegistration
        from fireants.registration.syn import SyNRegistration
    except Exception as exc:  # noqa: BLE001 — torch / CUDA import problems surface here too
        raise BackendUnavailableError(
            f"FireANTs is not usable ({exc}). Install with: pip install fireants"
        ) from exc
    return {
        "Image": Image,
        "BatchedImages": BatchedImages,
        "moments": MomentsRegistration,
        "rigid": RigidRegistration,
        "affine": AffineRegistration,
        "greedy": GreedyRegistration,
        "syn": SyNRegistration,
    }


def _write_like(array: Any, reference_itk: Any, path: Path) -> None:
    """Write a ``[1, C, ...]`` torch tensor on the fixed grid as a NIfTI image."""
    import numpy as np
    import SimpleITK as sitk

    arr = array.detach().float().cpu().numpy()
    arr = arr[0]
    if arr.shape[0] == 1:
        arr = arr[0]
    else:  # channels last for SimpleITK vector images
        arr = np.moveaxis(arr, 0, -1)
    out = sitk.GetImageFromArray(arr.astype(np.float32), isVector=arr.ndim == reference_itk.GetDimension() + 1)
    out.CopyInformation(reference_itk)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path))


def fireants_register(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    transform: str | Sequence[str] = "rigid+affine+syn",
    loss: str = "cc",
    scales: Sequence[int] | str = (4, 2, 1),
    iterations: Sequence[int] | str = (200, 100, 50),
    deformable_iterations: Sequence[int] | str | None = None,
    learning_rate: float | None = None,
    cc_kernel_size: int = 5,
    smooth_warp_sigma: float = 0.5,
    smooth_grad_sigma: float = 1.0,
    device: str = "cuda:0",
    verbose: bool = False,
) -> FireAntsResult:
    """Register MOVING to FIXED with FireANTs and write outputs under *out_dir*.

    *transform* chains stages from :data:`FIREANTS_STAGES` (``"rigid+affine+syn"``);
    each stage starts from the previous one's result. *scales*/*iterations* are the
    multi-resolution pyramid (downsampling factors and iterations per level) for the
    linear stages; *deformable_iterations* overrides the iterations of the greedy/SyN
    stage (defaults to *iterations*). *learning_rate* overrides FireANTs' per-stage
    default. On a machine without CUDA, ``device="cpu"`` works (slowly).
    """
    api = _require_fireants()
    stages = parse_stages(transform)
    loss = str(loss).strip().lower() or "cc"
    if loss not in FIREANTS_LOSSES:
        raise ValueError(f"loss must be one of {FIREANTS_LOSSES}; got {loss!r}.")
    scale_list = _int_list(scales, "scales")
    iter_list = _int_list(iterations, "iterations")
    if len(iter_list) != len(scale_list):
        raise ValueError(
            f"Give one iteration count per scale ({len(scale_list)} scales, {len(iter_list)} iterations)."
        )
    def_iters = _int_list(deformable_iterations, "deformable iterations") if deformable_iterations else iter_list
    if len(def_iters) != len(scale_list):
        raise ValueError("Give one deformable iteration count per scale.")

    try:
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except Exception:  # noqa: BLE001 — the API import above already needs torch
        pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = api["Image"].load_file(str(fixed_path), device=device)
    moving = api["Image"].load_file(str(moving_path), device=device)
    fixed_b = api["BatchedImages"]([fixed])
    moving_b = api["BatchedImages"]([moving])

    common: dict[str, Any] = {"loss_type": loss, "fixed_images": fixed_b, "moving_images": moving_b}
    if loss == "cc":
        common["cc_kernel_size"] = int(cc_kernel_size)

    def _lr(kw: dict[str, Any]) -> dict[str, Any]:
        if learning_rate is not None and float(learning_rate) > 0:
            kw["optimizer_lr"] = float(learning_rate)
        return kw

    init: dict[str, Any] = {}
    last: Any = None
    rigid_matrix = None
    affine_matrix = None
    for stage in stages:
        if stage == "moments":
            reg = api["moments"](scale=float(scale_list[0]), **common)
            reg.optimize()
            init = {"init_translation": reg.get_rigid_transl_init(), "init_moment": reg.get_rigid_moment_init()}
            affine_init = reg.get_affine_init()
            affine_matrix = affine_init
            last = reg
            continue
        if stage == "rigid":
            kw = _lr({"scales": scale_list, "iterations": iter_list, **common})
            kw.update({k: v for k, v in init.items() if k in ("init_translation", "init_moment")})
            reg = api["rigid"](**kw)
            reg.optimize()
            rigid_matrix = reg.get_rigid_matrix()
            affine_matrix = rigid_matrix
            last = reg
            continue
        if stage == "affine":
            kw = _lr({"scales": scale_list, "iterations": iter_list, **common})
            if rigid_matrix is not None:
                kw["init_rigid"] = rigid_matrix
            elif affine_matrix is not None:
                kw["init_rigid"] = affine_matrix
            reg = api["affine"](**kw)
            reg.optimize()
            affine_matrix = reg.get_affine_matrix()
            last = reg
            continue
        # greedy / syn: the deformable stage, last by construction
        kw = _lr({
            "scales": scale_list,
            "iterations": def_iters,
            "smooth_warp_sigma": float(smooth_warp_sigma),
            "smooth_grad_sigma": float(smooth_grad_sigma),
            **common,
        })
        if affine_matrix is not None:
            kw["init_affine"] = affine_matrix
        reg = api[stage](**kw)
        reg.optimize()
        last = reg

    prefix = out_dir / "fireants_"
    warped_path = out_dir / "moving_warped.nii.gz"
    import torch

    with torch.no_grad():
        moved = last.evaluate(fixed_b, moving_b)
    _write_like(moved, fixed.itk_image, warped_path)

    transforms: list[Path] = []
    final = stages[-1]
    if final in ("greedy", "syn"):
        warp = out_dir / "fireants_warp.nii.gz"
        last.save_as_ants_transforms([str(warp)])
        transforms.append(warp)
    elif final in ("rigid", "affine"):
        mat = out_dir / f"fireants_{final}.mat"
        last.save_as_ants_transforms([str(mat)])
        transforms.append(mat)
    if verbose:
        from nvitk.core.logger import Logger

        Logger().info(f"FireANTs {'+'.join(stages)} done → {warped_path}")
    return FireAntsResult(
        output_prefix=prefix,
        warped_moving_path=warped_path,
        transforms=tuple(transforms),
        stages=stages,
    )


def fireants_apply(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_path: Path,
    transforms: list[Path],
    interpolator: str = "linear",
) -> Path:
    """Map MOVING into FIXED space with transforms written by :func:`fireants_register`.

    They are ANTs-format files, so ANTsPy applies them when installed; otherwise the
    ``fireantsApplyTransforms`` tool is used if a FireANTs install provides it.
    """
    try:
        from nvitk.registration.ants import ants_apply

        return ants_apply(
            fixed_path=Path(fixed_path),
            moving_path=Path(moving_path),
            out_path=Path(out_path),
            transforms=[Path(p) for p in transforms],
            interpolator=interpolator,
        )
    except BackendUnavailableError:
        pass
    exe = shutil.which("fireantsApplyTransforms")
    if exe is None:
        raise BackendUnavailableError(
            "Applying FireANTs transforms needs ANTsPy (pip install antspyx) or a FireANTs "
            "install that provides fireantsApplyTransforms."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "--fixed",
        str(fixed_path),
        "--moving",
        str(moving_path),
        "--output",
        str(out_path),
        "--transform",
        ",".join(str(p) for p in transforms),
    ]
    subprocess.run(cmd, check=True)
    return out_path


__all__ = [
    "FIREANTS_LOSSES",
    "FIREANTS_STAGES",
    "FIREANTS_TRANSFORMS",
    "FireAntsResult",
    "fireants_apply",
    "fireants_register",
    "parse_stages",
]
