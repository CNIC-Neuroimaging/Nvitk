"""FSL registration beyond FLIRT (which lives in :mod:`nvitk.registration.fsl`): FNIRT and its warp
tools, convert_xfm, MCFLIRT, epi_reg. Each function returns an :class:`~nvitk.fsl.run.FslCommand`."""

from __future__ import annotations

from pathlib import Path

from .run import FslCommand, base, nii

#: FNIRT configurations shipped with FSL (``$FSLDIR/etc/flirtsch/<name>.cnf``).
FNIRT_CONFIGS: tuple[str, ...] = ("T1_2_MNI152_2mm", "FA_2_FMRIB58_1mm", "none")
APPLYWARP_INTERP: tuple[str, ...] = ("trilinear", "nn", "sinc", "spline")
MCFLIRT_COSTS: tuple[str, ...] = ("normcorr", "mutualinfo", "woods", "corratio", "normmi", "leastsquares")


def fnirt(
    input: Path,
    ref: Path,
    out_base: Path,
    *,
    affine: Path | None = None,
    config: str = "T1_2_MNI152_2mm",
    ref_mask: Path | None = None,
    in_mask: Path | None = None,
) -> FslCommand:
    """Non-linear registration of *input* to *ref*: warped image, coefficients and displacement field.

    Give the FLIRT *affine* (input → ref) as the starting point; FNIRT does not search.
    """
    out = base(out_base)
    argv = ["fnirt", f"--in={input}", f"--ref={ref}", f"--iout={nii(f'{out}_warped')}",
            f"--cout={nii(f'{out}_warpcoef')}", f"--fout={nii(f'{out}_field')}"]
    if affine is not None:
        argv.append(f"--aff={affine}")
    if config and config != "none":
        argv.append(f"--config={config}")
    if ref_mask is not None:
        argv.append(f"--refmask={ref_mask}")
    if in_mask is not None:
        argv.append(f"--inmask={in_mask}")
    outputs = {"warped": nii(f"{out}_warped"), "warpcoef": nii(f"{out}_warpcoef"), "field": nii(f"{out}_field")}
    return FslCommand(argv, "FNIRT non-linear registration", outputs, long=True)


def applywarp(
    input: Path,
    ref: Path,
    output: Path,
    *,
    warp: Path | None = None,
    premat: Path | None = None,
    postmat: Path | None = None,
    interp: str = "trilinear",
    relative: bool = True,
) -> FslCommand:
    """Resample *input* onto *ref* through a warp field and/or affine(s)."""
    argv = ["applywarp", f"--in={input}", f"--ref={ref}", f"--out={nii(output)}", f"--interp={interp}"]
    if warp is not None:
        argv.append(f"--warp={warp}")
        argv.append("--rel" if relative else "--abs")
    if premat is not None:
        argv.append(f"--premat={premat}")
    if postmat is not None:
        argv.append(f"--postmat={postmat}")
    return FslCommand(argv, "applywarp", {"image": nii(output)}, label_outputs=("image",) if interp == "nn" else ())


def invwarp(warp: Path, ref: Path, output: Path) -> FslCommand:
    """Invert a warp field; *ref* is the space the inverse maps into (the original input's)."""
    return FslCommand(["invwarp", f"--warp={warp}", f"--ref={ref}", f"--out={nii(output)}"], "invwarp",
                      {"warp": nii(output)})


def convertwarp(
    ref: Path,
    output: Path,
    *,
    warp1: Path | None = None,
    warp2: Path | None = None,
    premat: Path | None = None,
    midmat: Path | None = None,
    postmat: Path | None = None,
) -> FslCommand:
    """Combine affines and warps into one warp field."""
    argv = ["convertwarp", f"--ref={ref}", f"--out={nii(output)}"]
    for key, value in (("warp1", warp1), ("warp2", warp2), ("premat", premat), ("midmat", midmat),
                       ("postmat", postmat)):
        if value is not None:
            argv.append(f"--{key}={value}")
    return FslCommand(argv, "convertwarp", {"warp": nii(output)})


def convert_xfm(output: Path, *, inverse: Path | None = None, concat: tuple[Path, Path] | None = None) -> FslCommand:
    """Invert an affine, or concatenate two (``concat=(a, b)`` gives *b* after *a*)."""
    if (inverse is None) == (concat is None):
        raise ValueError("Give exactly one of inverse= or concat=.")
    argv = ["convert_xfm", "-omat", str(output)]
    if inverse is not None:
        argv += ["-inverse", str(inverse)]
    else:
        first, second = concat  # type: ignore[misc]
        argv += ["-concat", str(second), str(first)]
    return FslCommand(argv, "convert_xfm", {"matrix": Path(output)})


def mcflirt(
    input: Path,
    out_base: Path,
    *,
    ref_vol: int | None = None,
    mean_vol: bool = False,
    cost: str = "normcorr",
    stages: int = 3,
    save_mats: bool = False,
    sinc_final: bool = False,
) -> FslCommand:
    """Motion correction of a 4D series: the corrected series and the ``.par`` motion parameters."""
    out = base(out_base)
    argv = ["mcflirt", "-in", str(input), "-out", str(out), "-cost", cost, "-stages", str(int(stages)), "-plots"]
    if mean_vol:
        argv.append("-meanvol")
    elif ref_vol is not None:
        argv += ["-refvol", str(int(ref_vol))]
    if save_mats:
        argv.append("-mats")
    if sinc_final:
        argv.append("-sinc_final")
    return FslCommand(argv, "MCFLIRT motion correction", {"image": nii(out), "par": Path(f"{out}.par")}, long=True)


def epi_reg(
    epi: Path,
    t1: Path,
    t1_brain: Path,
    out_base: Path,
    *,
    wmseg: Path | None = None,
    fmap: Path | None = None,
    fmap_mag: Path | None = None,
    fmap_mag_brain: Path | None = None,
    echo_spacing: float | None = None,
    pe_dir: str | None = None,
) -> FslCommand:
    """EPI → T1 boundary-based registration (with optional fieldmap unwarping)."""
    out = base(out_base)
    argv = ["epi_reg", f"--epi={epi}", f"--t1={t1}", f"--t1brain={t1_brain}", f"--out={out}"]
    for key, value in (("wmseg", wmseg), ("fmap", fmap), ("fmapmag", fmap_mag), ("fmapmagbrain", fmap_mag_brain)):
        if value is not None:
            argv.append(f"--{key}={value}")
    if echo_spacing:
        argv.append(f"--echospacing={echo_spacing:g}")
    if pe_dir:
        argv.append(f"--pedir={pe_dir}")
    return FslCommand(argv, "epi_reg", {"image": nii(out), "matrix": Path(f"{out}.mat")}, long=True,
                      requires=("epi_reg", "flirt", "fast"))


__all__ = [
    "APPLYWARP_INTERP",
    "FNIRT_CONFIGS",
    "MCFLIRT_COSTS",
    "applywarp",
    "convert_xfm",
    "convertwarp",
    "epi_reg",
    "fnirt",
    "invwarp",
    "mcflirt",
]
