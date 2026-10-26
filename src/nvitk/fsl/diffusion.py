"""FSL diffusion (FDT): TOPUP, EDDY, DTIFIT, BEDPOSTX, PROBTRACKX2, TBSS, XTRACT.

Each function returns an :class:`~nvitk.fsl.run.FslCommand`. EDDY, BEDPOSTX and
PROBTRACKX2 use their GPU builds when the FSL installation has them (``cuda=None``).
BEDPOSTX and TBSS work in directories laid out their way; :func:`prepare_bedpostx_dir`
and :func:`prepare_tbss_dir` build those from plain files.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Sequence

from . import env
from .run import FslCommand, base, nii

TOPUP_CONFIGS: tuple[str, ...] = ("b02b0.cnf", "b02b0_1.cnf", "b02b0_2.cnf", "b02b0_4.cnf")
XTRACT_SPECIES: tuple[str, ...] = ("HUMAN", "MACAQUE", "HUMAN_BABY")


def topup(imain: Path, datain: Path, out_base: Path, *, config: str = "b02b0.cnf") -> FslCommand:
    """Susceptibility field from reversed phase-encode b0s; also the corrected b0s and the field."""
    out = base(out_base)
    argv = ["topup", f"--imain={imain}", f"--datain={datain}", f"--config={config}", f"--out={out}",
            f"--iout={nii(f'{out}_b0_corrected')}", f"--fout={nii(f'{out}_field')}"]
    outputs = {"corrected": nii(f"{out}_b0_corrected"), "field": nii(f"{out}_field"),
               "fieldcoef": nii(f"{out}_fieldcoef"), "movpar": Path(f"{out}_movpar.txt")}
    return FslCommand(argv, "TOPUP", outputs, long=True)


def applytopup(
    images: Sequence[Path], datain: Path, inindex: Sequence[int], topup_base: Path, output: Path,
    *, method: str = "jac",
) -> FslCommand:
    """Apply a TOPUP field to *images* (one row of *datain* each, by *inindex*)."""
    argv = ["applytopup", f"--imain={','.join(str(p) for p in images)}", f"--datain={datain}",
            f"--inindex={','.join(str(int(i)) for i in inindex)}", f"--topup={base(topup_base)}",
            f"--out={nii(output)}", f"--method={method}"]
    return FslCommand(argv, "applytopup", {"image": nii(output)})


def eddy(
    imain: Path,
    mask: Path,
    acqp: Path,
    index: Path,
    bvecs: Path,
    bvals: Path,
    out_base: Path,
    *,
    topup_base: Path | None = None,
    repol: bool = True,
    cuda: bool | None = None,
    extra: Sequence[str] = (),
) -> FslCommand:
    """Eddy-current and motion correction (GPU build when available and *cuda* is not False)."""
    out = base(out_base)
    command = "eddy"
    if cuda is not False:
        gpu = env.cuda_command("eddy")
        if gpu:
            command = gpu
        elif cuda:
            raise env.FslNotFound("No eddy_cuda in this FSL installation.")
    if command == "eddy" and env.which("eddy_cpu"):
        command = "eddy_cpu"
    argv = [command, f"--imain={imain}", f"--mask={mask}", f"--acqp={acqp}", f"--index={index}",
            f"--bvecs={bvecs}", f"--bvals={bvals}", f"--out={out}"]
    if topup_base is not None:
        argv.append(f"--topup={base(topup_base)}")
    if repol:
        argv.append("--repol")
    argv += list(extra)
    outputs = {"image": nii(out), "rotated_bvecs": Path(f"{out}.eddy_rotated_bvecs")}
    return FslCommand(argv, f"EDDY ({command})", outputs, long=True, requires=(command,))


def dtifit(
    data: Path, mask: Path, bvecs: Path, bvals: Path, out_base: Path, *, wls: bool = False, save_tensor: bool = False,
    sse: bool = False,
) -> FslCommand:
    """Diffusion tensor fit: FA, MD, eigenvalues L1-L3, eigenvectors V1-V3, S0, MO."""
    out = base(out_base)
    argv = ["dtifit", f"--data={data}", f"--mask={mask}", f"--bvecs={bvecs}", f"--bvals={bvals}", f"--out={out}"]
    if wls:
        argv.append("--wls")
    if save_tensor:
        argv.append("--save_tensor")
    if sse:
        argv.append("--sse")
    outputs = {key: nii(f"{out}_{key}") for key in ("FA", "MD", "L1", "L2", "L3", "MO", "S0", "V1", "V2", "V3")}
    return FslCommand(argv, "DTIFIT tensor fit", outputs)


def prepare_bedpostx_dir(folder: Path, data: Path, mask: Path, bvecs: Path, bvals: Path) -> Path:
    """The folder BEDPOSTX expects: ``data``, ``nodif_brain_mask``, ``bvecs``, ``bvals``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for src, name in ((data, "data.nii.gz"), (mask, "nodif_brain_mask.nii.gz"), (bvecs, "bvecs"), (bvals, "bvals")):
        target = folder / name
        if Path(src).resolve() != target.resolve():
            shutil.copyfile(src, target)
    return folder


def bedpostx(folder: Path, *, fibres: int = 3, weight: float = 1.0, burnin: int = 1000, model: int = 2,
             gpu: bool | None = None) -> FslCommand:
    """Fibre orientation distributions (``<folder>.bedpostX``); hours on CPU, minutes on GPU."""
    command = "bedpostx"
    if gpu is not False and env.cuda_command("bedpostx"):
        command = "bedpostx_gpu"
    elif gpu:
        raise env.FslNotFound("No bedpostx_gpu in this FSL installation.")
    folder = Path(folder)
    argv = [command, str(folder), "-n", str(int(fibres)), "-w", f"{weight:g}", "-b", str(int(burnin)),
            "-model", str(int(model))]
    out = Path(f"{str(folder).rstrip('/')}.bedpostX")
    outputs = {"dir": out, "dyads1": out / "dyads1.nii.gz", "mean_f1": out / "mean_f1samples.nii.gz",
               "mean_d": out / "mean_dsamples.nii.gz"}
    return FslCommand(argv, f"BEDPOSTX ({command})", outputs, long=True, requires=(command, "xfibres"))


def probtrackx2(
    bedpostx_dir: Path,
    seed: Path,
    out_dir: Path,
    *,
    samples: int = 5000,
    curvature: float = 0.2,
    step_length: float = 0.5,
    waypoints: Path | None = None,
    exclusion: Path | None = None,
    target_masks: Path | None = None,
    seed_to_diff: Path | None = None,
    seed_ref: Path | None = None,
    gpu: bool | None = None,
) -> FslCommand:
    """Probabilistic tractography from *seed* through a BEDPOSTX result (``fdt_paths``)."""
    command = "probtrackx2"
    if gpu is not False and env.cuda_command("probtrackx2"):
        command = "probtrackx2_gpu"
    bpx = Path(bedpostx_dir)
    argv = [command, "-x", str(seed), "-s", str(bpx / "merged"), "-m", str(bpx / "nodif_brain_mask"),
            f"--dir={out_dir}", "--forcedir", "--opd", "-l", "-c", f"{curvature:g}", f"--steplength={step_length:g}",
            "-P", str(int(samples))]
    if waypoints is not None:
        argv.append(f"--waypoints={waypoints}")
    if exclusion is not None:
        argv.append(f"--avoid={exclusion}")
    if target_masks is not None:
        argv += [f"--targetmasks={target_masks}", "--os2t"]
    if seed_to_diff is not None:
        argv.append(f"--xfm={seed_to_diff}")
    if seed_ref is not None:
        argv.append(f"--seedref={seed_ref}")
    out = Path(out_dir)
    return FslCommand(argv, f"PROBTRACKX2 ({command})", {"paths": out / "fdt_paths.nii.gz", "waytotal": out / "waytotal",
                                                          "dir": out}, long=True, requires=(command,))


def prepare_tbss_dir(folder: Path, fa_images: Sequence[Path]) -> Path:
    """A TBSS working folder holding the subjects' FA maps (TBSS renames and moves them itself)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for image in fa_images:
        shutil.copyfile(image, folder / Path(image).name)
    return folder


def tbss(folder: Path, *, target: str = "FMRIB58", threshold: float = 0.2,
         non_fa: Sequence[str] = ()) -> FslCommand:
    """TBSS steps 1-4 in *folder* (FA maps in it); *target*: ``FMRIB58``, ``most_representative``
    or an image path. *non_fa* runs ``tbss_non_FA`` for those measures (folders ``<folder>/<MD>``)."""
    folder = Path(folder)
    reg = {"FMRIB58": "-T", "most_representative": "-n"}.get(target, f"-t {shlex.quote(str(target))}")
    # -S: the skeleton of this study's mean FA (not FMRIB58's), as the TBSS guide recommends.
    steps = [f"cd {shlex.quote(str(folder))}", "tbss_1_preproc *.nii.gz", f"tbss_2_reg {reg}",
             "tbss_3_postreg -S", f"tbss_4_prestats {threshold:g}"]
    steps += [f"tbss_non_FA {shlex.quote(m)}" for m in non_fa]
    stats = folder / "stats"
    outputs = {"mean_FA": stats / "mean_FA.nii.gz", "skeleton": stats / "mean_FA_skeleton.nii.gz",
               "skeleton_mask": stats / "mean_FA_skeleton_mask.nii.gz",
               "all_FA_skeletonised": stats / "all_FA_skeletonised.nii.gz", "dir": stats}
    outputs.update({f"all_{m}_skeletonised": stats / f"all_{m}_skeletonised.nii.gz" for m in non_fa})
    return FslCommand([" && ".join(steps)], "TBSS", outputs, label_outputs=("skeleton_mask",), long=True, shell=True,
                      requires=("tbss_1_preproc", "tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats", "fnirt"))


def xtract(
    bedpostx_dir: Path,
    out_dir: Path,
    *,
    species: str = "HUMAN",
    std_warps: tuple[Path, Path] | None = None,
    tracts: Sequence[str] = (),
    native: bool = False,
    gpu: bool | None = None,
) -> FslCommand:
    """Standardised tractography of ~40 tracts. *std_warps* = (standard->diffusion, diffusion->standard);
    without it XTRACT expects them in ``<bedpostX>/xfms``."""
    argv = ["xtract", "-bpx", str(bedpostx_dir), "-out", str(out_dir), "-species", species]
    if std_warps is not None:
        argv += ["-stdwarp", str(std_warps[0]), str(std_warps[1])]
    if tracts:
        argv += ["-tract_list", ",".join(tracts)]
    if native:
        argv.append("-native")
    if gpu is not False and env.cuda_command("probtrackx2"):
        argv.append("-gpu")
    out = Path(out_dir)
    return FslCommand(argv, "XTRACT", {"dir": out, "tracts": out / "tracts"}, long=True,
                      requires=("xtract", "probtrackx2"))


__all__ = [
    "TOPUP_CONFIGS",
    "XTRACT_SPECIES",
    "applytopup",
    "bedpostx",
    "dtifit",
    "eddy",
    "prepare_bedpostx_dir",
    "prepare_tbss_dir",
    "probtrackx2",
    "tbss",
    "topup",
    "xtract",
]
