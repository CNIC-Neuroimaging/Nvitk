"""Structural FSL: BET, FAST, FIRST, fsl_anat, SIENA/SIENAX, reorientation, cropping, defacing,
lesion filling. Each function returns an :class:`~nvitk.fsl.run.FslCommand` (nothing runs here)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .run import FslCommand, base, nii

#: BET modes (one of BET's exclusive options).
BET_MODES: dict[str, str] = {
    "standard": "",
    "robust (-R)": "-R",
    "eye & optic nerve cleanup (-S)": "-S",
    "bias field & neck cleanup (-B)": "-B",
    "small FOV in z (-Z)": "-Z",
    "4D / fMRI (-F)": "-F",
}
FAST_TYPES: dict[str, int] = {"T1": 1, "T2": 2, "PD": 3}
#: FIRST structure names (``run_first_all -s``).
FIRST_STRUCTURES: tuple[str, ...] = (
    "L_Accu", "L_Amyg", "L_Caud", "L_Hipp", "L_Pall", "L_Puta", "L_Thal",
    "R_Accu", "R_Amyg", "R_Caud", "R_Hipp", "R_Pall", "R_Puta", "R_Thal", "BrStem",
)
#: FIRST's segmentation ids (CMA labels), for naming the label map.
FIRST_LABELS: dict[int, str] = {
    10: "Left-Thalamus", 11: "Left-Caudate", 12: "Left-Putamen", 13: "Left-Pallidum", 16: "Brain-Stem",
    17: "Left-Hippocampus", 18: "Left-Amygdala", 26: "Left-Accumbens", 49: "Right-Thalamus",
    50: "Right-Caudate", 51: "Right-Putamen", 52: "Right-Pallidum", 53: "Right-Hippocampus",
    54: "Right-Amygdala", 58: "Right-Accumbens",
}


def bet(
    input: Path,
    output: Path,
    *,
    f: float = 0.5,
    g: float = 0.0,
    mode: str = "standard",
    mask: bool = True,
    skull: bool = False,
    surfaces: bool = False,
    center: Sequence[float] | None = None,
    radius: float | None = None,
) -> FslCommand:
    """Brain extraction: ``<out>`` (brain), ``<out>_mask``, optionally skull / surfaces."""
    out = base(output)
    argv = ["bet", str(input), str(out), "-f", f"{f:g}", "-g", f"{g:g}"]
    flag = BET_MODES.get(mode, mode if str(mode).startswith("-") else "")
    if flag:
        argv.append(flag)
    if mask:
        argv.append("-m")
    if skull:
        argv.append("-s")
    if surfaces:
        argv.append("-A")
    if center is not None:
        argv += ["-c", *[f"{float(v):g}" for v in center]]
    if radius:
        argv += ["-r", f"{radius:g}"]
    outputs = {"brain": nii(out)}
    if mask or surfaces:
        outputs["mask"] = nii(f"{out}_mask")
    if skull or surfaces:
        outputs["skull"] = nii(f"{out}_skull")
    return FslCommand(argv, "BET brain extraction", outputs, label_outputs=("mask",),
                      long=mode == "bias field & neck cleanup (-B)" or surfaces, requires=("bet", "bet2"))


def fast(
    input: Path,
    out_base: Path,
    *,
    classes: int = 3,
    img_type: str = "T1",
    bias_corrected: bool = True,
    bias_field: bool = False,
    no_bias: bool = False,
    iterations: int | None = None,
) -> FslCommand:
    """Tissue segmentation (and bias correction): ``_seg``, ``_pve_<k>``, ``_restore``."""
    out = base(out_base)
    argv = ["fast", "-n", str(int(classes)), "-t", str(FAST_TYPES.get(img_type, 1)), "-o", str(out)]
    if bias_corrected and not no_bias:
        argv.append("-B")
    if bias_field and not no_bias:
        argv.append("-b")
    if no_bias:
        argv.append("-N")
    if iterations:
        argv += ["-I", str(int(iterations))]
    argv.append(str(input))
    outputs = {"seg": nii(f"{out}_seg")}
    outputs.update({f"pve_{k}": nii(f"{out}_pve_{k}") for k in range(int(classes))})
    if bias_corrected and not no_bias:
        outputs["restore"] = nii(f"{out}_restore")
    if bias_field and not no_bias:
        outputs["bias"] = nii(f"{out}_bias")
    return FslCommand(argv, "FAST tissue segmentation", outputs, label_outputs=("seg",), long=True)


def first(
    input: Path,
    out_base: Path,
    *,
    structures: Sequence[str] = (),
    brain_extracted: bool = False,
    method: str = "auto",
    three_stage: bool = False,
) -> FslCommand:
    """Subcortical segmentation (``run_first_all``): ``<out>_all_<method>_firstseg``."""
    out = base(out_base)
    argv = ["run_first_all", "-i", str(input), "-o", str(out), "-m", method]
    if brain_extracted:
        argv.append("-b")
    if structures:
        argv += ["-s", ",".join(structures)]
    if three_stage:
        argv.append("-3")
    # The boundary-correction method names the file (``_all_fast_firstseg`` for auto/fast,
    # ``_all_none_...``, ``_all_th<t>_...``): matched by pattern.
    outputs = {"seg": Path(f"{out}_all_*_firstseg.nii.gz"), "origsegs": Path(f"{out}_all_*_origsegs.nii.gz")}
    return FslCommand(argv, "FIRST subcortical segmentation", outputs,
                      label_outputs=("seg",), long=True, requires=("run_first_all", "first", "first_flirt"))


def fsl_anat(
    input: Path,
    out_base: Path,
    *,
    img_type: str = "T1",
    strong_bias: bool = False,
    no_reorient: bool = False,
    no_crop: bool = False,
    no_bias: bool = False,
    no_reg: bool = False,
    no_nonlinreg: bool = False,
    no_seg: bool = False,
    no_subcortseg: bool = False,
    no_search: bool = False,
) -> FslCommand:
    """The structural pipeline (reorient, crop, bias, MNI registration, FAST, FIRST) -> ``<out>.anat``."""
    out = base(out_base)
    argv = ["fsl_anat", "-i", str(input), "-o", str(out), "-t", img_type, "--clobber"]
    for flag, on in (
        ("--strongbias", strong_bias), ("--noreorient", no_reorient), ("--nocrop", no_crop),
        ("--nobias", no_bias), ("--noreg", no_reg), ("--nononlinreg", no_nonlinreg), ("--noseg", no_seg),
        ("--nosubcortseg", no_subcortseg), ("--nosearch", no_search),
    ):
        if on:
            argv.append(flag)
    anat = Path(f"{out}.anat")
    t = img_type
    outputs = {
        "biascorr": anat / f"{t}_biascorr.nii.gz",
        "brain": anat / f"{t}_biascorr_brain.nii.gz",
        "brain_mask": anat / f"{t}_biascorr_brain_mask.nii.gz",
        "fast_seg": anat / f"{t}_fast_seg.nii.gz",
        "subcort_seg": anat / f"{t}_subcort_seg.nii.gz",
        "to_mni_nonlin": anat / f"{t}_to_MNI_nonlin.nii.gz",
        "dir": anat,
    }
    return FslCommand(argv, "fsl_anat structural pipeline", outputs,
                      label_outputs=("brain_mask", "fast_seg", "subcort_seg"), long=True,
                      requires=("fsl_anat", "bet", "fast", "flirt", "fnirt", "run_first_all"))


def siena(
    input1: Path,
    input2: Path,
    out_dir: Path,
    *,
    bet_opts: str = "",
    two_class: bool = False,
    t2: bool = False,
    std_mask: bool = False,
    ventricles: bool = False,
) -> FslCommand:
    """Two-time-point atrophy (percentage brain volume change) -> ``<out>/report.siena``."""
    argv = ["siena", str(input1), str(input2), "-o", str(out_dir)]
    if bet_opts:
        argv += ["-B", bet_opts]
    for flag, on in (("-2", two_class), ("-t2", t2), ("-m", std_mask), ("-V", ventricles)):
        if on:
            argv.append(flag)
    return FslCommand(argv, "SIENA atrophy", {"report": Path(out_dir) / "report.siena", "dir": Path(out_dir)},
                      long=True, requires=("siena", "bet", "flirt", "fast"))


def sienax(
    input: Path,
    out_dir: Path,
    *,
    bet_opts: str = "",
    two_class: bool = False,
    t2: bool = False,
    regional: bool = False,
    lesion_mask: Path | None = None,
) -> FslCommand:
    """Single-time-point normalised brain / GM / WM volumes -> ``<out>/report.sienax``."""
    argv = ["sienax", str(input), "-o", str(out_dir)]
    if bet_opts:
        argv += ["-B", bet_opts]
    for flag, on in (("-2", two_class), ("-t2", t2), ("-r", regional)):
        if on:
            argv.append(flag)
    if lesion_mask is not None:
        argv += ["-lm", str(lesion_mask)]
    out = Path(out_dir)
    # SIENAX deletes its intermediates (the segmentation included) unless run with -d; the
    # report and its HTML page (with renders) are what stays.
    return FslCommand(argv, "SIENAX brain volume", {"report": out / "report.sienax", "html": out / "report.html",
                                                     "dir": out},
                      long=True, requires=("sienax", "bet", "flirt", "fast"))


def parse_sienax_report(path: Path) -> dict[str, dict[str, float]]:
    """``{tissue: {"normalised": mm3, "unnormalised": mm3}}`` and the VSCALING factor."""
    out: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "VSCALING" and len(parts) > 1:
            out["VSCALING"] = {"value": float(parts[1])}
        elif parts[0] in ("GREY", "WHITE", "BRAIN", "pgrey", "vcsf") and len(parts) >= 3:
            try:
                out[parts[0]] = {"normalised": float(parts[1]), "unnormalised": float(parts[2])}
            except ValueError:
                continue
    return out


def parse_siena_report(path: Path) -> dict[str, float]:
    """``{"PBVC": percentage brain volume change, ...}`` from ``report.siena``."""
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("finalPBVC", "PBVC"):
            try:
                out["PBVC"] = float(parts[-1])
            except ValueError:
                pass
    return out


def reorient2std(input: Path, output: Path) -> FslCommand:
    """Reorient to MNI orientation (``fslreorient2std``; axis swaps/flips only, no resampling)."""
    return FslCommand(["fslreorient2std", str(input), str(nii(output))], "Reorient to standard",
                      {"image": nii(output)})


def robustfov(input: Path, output: Path, *, brain_size_mm: float = 170.0, matrix: Path | None = None) -> FslCommand:
    """Crop the lower head and neck."""
    argv = ["robustfov", "-i", str(input), "-r", str(nii(output)), "-b", f"{brain_size_mm:g}"]
    outputs = {"image": nii(output)}
    if matrix is not None:
        argv += ["-m", str(matrix)]
        outputs["matrix"] = Path(matrix)
    return FslCommand(argv, "Robust FOV crop", outputs)


def deface(input: Path, output: Path, *, f: float = 0.5, bias_correct: bool = False, mask: Path | None = None) -> FslCommand:
    """Remove the face (``fsl_deface``)."""
    argv = ["fsl_deface", str(input), str(nii(output)), "-f", f"{f:g}"]
    outputs = {"image": nii(output)}
    if bias_correct:
        argv.append("-B")
    if mask is not None:
        argv += ["-d", str(nii(mask))]
        outputs["mask"] = nii(mask)
    return FslCommand(argv, "Deface", outputs, label_outputs=("mask",), long=True,
                      requires=("fsl_deface", "bet", "flirt"))


def lesion_filling(input: Path, lesion_mask: Path, output: Path, *, wm_mask: Path | None = None) -> FslCommand:
    """Fill lesions with surrounding white-matter intensities (before FAST / FIRST / registration)."""
    argv = ["lesion_filling", "-i", str(input), "-l", str(lesion_mask), "-o", str(nii(output))]
    if wm_mask is not None:
        argv += ["-w", str(wm_mask)]
    return FslCommand(argv, "Lesion filling", {"image": nii(output)})


__all__ = [
    "BET_MODES",
    "FAST_TYPES",
    "FIRST_LABELS",
    "FIRST_STRUCTURES",
    "bet",
    "deface",
    "fast",
    "first",
    "fsl_anat",
    "lesion_filling",
    "parse_siena_report",
    "parse_sienax_report",
    "reorient2std",
    "robustfov",
    "siena",
    "sienax",
]
