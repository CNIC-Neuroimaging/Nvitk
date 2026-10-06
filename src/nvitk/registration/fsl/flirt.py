"""Rigid registration with FSL FLIRT via NiPype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvitk.core.logger import Logger

log = Logger()


def _ensure_fsl_env() -> None:
    """FSL defaults to uncompressed NIFTI; NiPype validates the exact ``out_file`` path."""
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    os.environ.setdefault("FSLMULTIFILEQUIT", "TRUE")


def _resolve_nifti_path(path: Path) -> Path | None:
    """Return *path* or a sibling ``.nii`` / ``.nii.gz`` if FSL wrote a different extension."""
    if path.is_file():
        return path
    if path.suffix == ".gz" and path.name.endswith(".nii.gz"):
        alt = path.with_name(path.name[: -len(".gz")])
        if alt.is_file():
            return alt
    if path.suffix == ".nii":
        alt = Path(f"{path}.gz")
        if alt.is_file():
            return alt
    return None


#: FLIRT degrees of freedom: 6 rigid, 7 rigid + global scale, 9 traditional
#: (rigid + per-axis scale), 12 full affine.
FLIRT_DOF: tuple[int, ...] = (6, 7, 9, 12)
#: FLIRT cost functions (``bbr`` needs a white-matter segmentation and is not offered).
FLIRT_COSTS: tuple[str, ...] = ("corratio", "mutualinfo", "normmi", "normcorr", "leastsq", "labeldiff")
#: Interpolation of the resliced output.
FLIRT_INTERPS: tuple[str, ...] = ("trilinear", "nearestneighbour", "sinc", "spline")


@dataclass(frozen=True)
class FlirtRigidResult:
    """Paths and NiPype runtime from a FLIRT rigid run."""

    matrix_path: Path
    warped_path: Path | None
    runtime: Any


def _require_paths(*paths: Path) -> None:
    """Assert every path exists as a file, raising ``FileNotFoundError`` for the first that doesn't."""
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")


def flirt_register_rigid(
    moving: str | Path,
    fixed: str | Path,
    out_dir: str | Path,
    *,
    dof: int = 6,
    cost: str = "corratio",
    warped_name: str = "moving_warped.nii.gz",
    matrix_name: str = "affine.mat",
    searchr_x: float | None = None,
    searchr: float | None = None,
    no_search: bool = False,
    interp: str | None = None,
    bins: int | None = None,
) -> FlirtRigidResult:
    """Run FLIRT alignment of *moving* towards *fixed* (rigid at ``dof=6``, affine at 12).

    Writes ``matrix_name`` under *out_dir* and optionally a warped moving image.
    *searchr* sets the angular search range (±degrees) on all three axes —
    *searchr_x* on X only, kept for older callers — and *no_search* skips the
    search entirely (images already roughly aligned). *interp* is the output's
    interpolation, *bins* the cost-function histogram size.
    Requires FSL on ``PATH`` and ``FSLDIR`` set (NiPype delegates to ``flirt``).
    """
    from nipype.interfaces.fsl import FLIRT

    moving_p = Path(moving).resolve()
    fixed_p = Path(fixed).resolve()
    out_d = Path(out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    mat_path = out_d / matrix_name
    warped_path = out_d / warped_name

    _require_paths(moving_p, fixed_p)
    _ensure_fsl_env()

    fl = FLIRT()
    fl.inputs.in_file = str(moving_p)
    fl.inputs.reference = str(fixed_p)
    fl.inputs.out_matrix_file = str(mat_path)
    fl.inputs.out_file = str(warped_path)
    fl.inputs.dof = int(dof)
    if hasattr(fl.inputs, "cost"):
        setattr(fl.inputs, "cost", cost)
    elif hasattr(fl.inputs, "cost_fun"):
        setattr(fl.inputs, "cost_fun", cost)
    elif hasattr(fl.inputs, "cost_func"):
        setattr(fl.inputs, "cost_func", cost)
    if searchr is not None and float(searchr) > 0:
        rng = [-int(round(float(searchr))), int(round(float(searchr)))]
        fl.inputs.searchr_x = rng
        fl.inputs.searchr_y = rng
        fl.inputs.searchr_z = rng
    elif searchr_x is not None and hasattr(fl.inputs, "searchr_x"):
        fl.inputs.searchr_x = [-int(round(abs(float(searchr_x)))), int(round(abs(float(searchr_x))))]
    if no_search:
        fl.inputs.no_search = True
    if interp:
        fl.inputs.interp = str(interp)
    if bins:
        fl.inputs.bins = int(bins)

    log.info(f"FLIRT rigid: moving={moving_p} reference={fixed_p} dof={dof}, cost={cost}")
    try:
        runtime = fl.run()
    except FileNotFoundError as exc:
        resolved = _resolve_nifti_path(warped_path)
        if resolved is not None and mat_path.is_file():
            log.warning(
                f"FLIRT wrote {resolved.name} (expected {warped_path.name}); "
                "set FSLOUTPUTTYPE=NIFTI_GZ for consistent outputs."
            )
            runtime = None
        else:
            raise exc from None
    if not mat_path.is_file():
        raise RuntimeError(f"FLIRT did not produce matrix file: {mat_path}")
    resolved_warped = _resolve_nifti_path(warped_path)
    return FlirtRigidResult(matrix_path=mat_path, warped_path=resolved_warped, runtime=runtime)


def flirt_apply_rigid(
    in_file: str | Path,
    reference: str | Path,
    mat_file: str | Path,
    out_file: str | Path,
    *,
    interp: str = "trilinear",
) -> Path:
    """Apply an existing FLIRT ``*.mat`` transform (``-applyxfm``)."""
    from nipype.interfaces.fsl import FLIRT

    inp = Path(in_file).resolve()
    ref = Path(reference).resolve()
    mat = Path(mat_file).resolve()
    outp = Path(out_file).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    _require_paths(inp, ref, mat)
    _ensure_fsl_env()

    fl = FLIRT()
    fl.inputs.in_file = str(inp)
    fl.inputs.reference = str(ref)
    fl.inputs.in_matrix_file = str(mat)
    fl.inputs.apply_xfm = True
    fl.inputs.interp = interp
    fl.inputs.out_file = str(outp)
    log.info(f"FLIRT apply: in={inp} ref={ref} mat={mat} interp={interp}")
    try:
        fl.run()
    except FileNotFoundError as exc:
        resolved = _resolve_nifti_path(outp)
        if resolved is None:
            raise exc from None
        log.warning(
            f"FLIRT apply wrote {resolved.name} (expected {outp.name}); "
            "set FSLOUTPUTTYPE=NIFTI_GZ for consistent outputs."
        )
        return resolved
    resolved = _resolve_nifti_path(outp)
    if resolved is None:
        raise RuntimeError(f"FLIRT apply did not produce: {outp}")
    return resolved
