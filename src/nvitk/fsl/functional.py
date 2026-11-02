"""Functional / field-map FSL: MELODIC ICA, SUSAN smoothing, slice timing, FUGUE unwarping and
field-map preparation. Each function returns an :class:`~nvitk.fsl.run.FslCommand`."""

from __future__ import annotations

from pathlib import Path

from .run import FslCommand, base, nii

MELODIC_APPROACHES: tuple[str, ...] = ("symm", "concat", "tica")
SLICE_ORDERS: tuple[str, ...] = ("bottom-up", "top-down", "interleaved")
UNWARP_DIRECTIONS: tuple[str, ...] = ("y", "y-", "x", "x-", "z", "z-")
FIELDMAP_SCANNERS: tuple[str, ...] = ("SIEMENS", "GEHC_FIELDMAPHZ")


def melodic(
    input: Path,
    out_dir: Path,
    *,
    dim: int = 0,
    approach: str = "symm",
    tr: float | None = None,
    no_bet: bool = False,
    report: bool = True,
    mask: Path | None = None,
) -> FslCommand:
    """Single-session ICA: independent component maps and a report (``dim`` 0 = automatic)."""
    argv = ["melodic", "-i", str(input), "-o", str(out_dir), "-a", approach]
    if dim:
        argv += ["-d", str(int(dim))]
    if tr:
        argv.append(f"--tr={tr:g}")
    if no_bet:
        argv.append("--nobet")
    if mask is not None:
        argv += ["-m", str(mask)]
    if report:
        argv.append("--report")
    out = Path(out_dir)
    return FslCommand(argv, "MELODIC ICA", {"ics": out / "melodic_IC.nii.gz", "mean": out / "mean.nii.gz",
                                             "report": out / "report" / "00index.html", "dir": out}, long=True)


def susan(
    input: Path,
    output: Path,
    *,
    sigma_mm: float,
    brightness_threshold: float = -1.0,
    dim: int = 3,
    use_median: bool = True,
) -> FslCommand:
    """Edge-preserving noise reduction. A negative *brightness_threshold* lets SUSAN pick 10% of
    the robust range; FEAT uses 75% of the median brain intensity."""
    argv = ["susan", str(input), f"{brightness_threshold:g}", f"{sigma_mm:g}", str(int(dim)),
            "1" if use_median else "0", "0", str(nii(output))]
    return FslCommand(argv, "SUSAN smoothing", {"image": nii(output)})


def slicetimer(
    input: Path,
    output: Path,
    *,
    tr: float = 3.0,
    order: str = "bottom-up",
    direction: int = 3,
    timings: Path | None = None,
    custom_order: Path | None = None,
) -> FslCommand:
    """Slice-timing correction of a 4D series."""
    argv = ["slicetimer", "-i", str(input), "-o", str(nii(output)), "-r", f"{tr:g}", "-d", str(int(direction))]
    if timings is not None:
        argv.append(f"--tcustom={timings}")
    elif custom_order is not None:
        argv.append(f"--ocustom={custom_order}")
    elif order == "top-down":
        argv.append("--down")
    elif order == "interleaved":
        argv.append("--odd")
    return FslCommand(argv, "Slice-timing correction", {"image": nii(output)})


def fugue(
    input: Path,
    output: Path,
    fieldmap: Path,
    *,
    dwell_time: float,
    unwarp_dir: str = "y",
    smooth3: float = 0.0,
) -> FslCommand:
    """Unwarp an EPI with a field map in rad/s (``fsl_prepare_fieldmap`` output)."""
    argv = ["fugue", "-i", str(input), f"--dwell={dwell_time:g}", f"--loadfmap={fieldmap}",
            f"--unwarpdir={unwarp_dir}", "-u", str(nii(output))]
    if smooth3:
        argv.append(f"--smooth3={smooth3:g}")
    return FslCommand(argv, "FUGUE unwarping", {"image": nii(output)})


def prepare_fieldmap(
    phase: Path, magnitude_brain: Path, output: Path, *, delta_te_ms: float = 2.46, scanner: str = "SIEMENS",
) -> FslCommand:
    """A field map in rad/s from a phase-difference (SIEMENS) or Hz (GE) image."""
    argv = ["fsl_prepare_fieldmap", scanner, str(phase), str(magnitude_brain), str(base(output)), f"{delta_te_ms:g}"]
    return FslCommand(argv, "Prepare field map", {"fieldmap": nii(output)})


__all__ = [
    "FIELDMAP_SCANNERS",
    "MELODIC_APPROACHES",
    "SLICE_ORDERS",
    "UNWARP_DIRECTIONS",
    "fugue",
    "melodic",
    "prepare_fieldmap",
    "slicetimer",
    "susan",
]
