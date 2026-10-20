"""``nvitk-fsl``: the FSL suite from nvitk, on the FSL installation nvitk resolves.

Set-up::

    nvitk-fsl info                       # which FSL, which components, GPU builds
    nvitk-fsl install [--components structural,diffusion] [--no-cuda]
    nvitk-fsl use /path/to/fsl           # make nvitk use another installation

Structural: ``bet``, ``fast``, ``first``, ``anat``, ``siena``, ``sienax``, ``reorient2std``,
``robustfov``, ``deface``, ``lesion-filling``. Registration: ``fnirt``, ``applywarp``,
``invwarp``, ``convertwarp``, ``convert-xfm``, ``mcflirt``, ``epi-reg`` (FLIRT: ``nvitk-flirt``).
Diffusion: ``topup``, ``applytopup``, ``eddy``, ``dtifit``, ``bedpostx``, ``probtrackx2``,
``tbss``, ``xtract``. Functional: ``melodic``, ``susan``, ``slicetimer``, ``fugue``,
``prepare-fieldmap``. Utilities: ``maths``, ``stats``, ``meants``, ``cluster``, ``smoothest``,
``run`` (any FSL command in the FSL environment).

Every command runs here by default, or on the cluster with ``--submit sge`` (in the nvitk
container, which carries FSL; ``--emit-script``, ``--no-remote``, ``--dry-run`` as for the
other nvitk CLIs)::

    nvitk-fsl bet /data/t1.nii.gz /data/t1_brain -f 0.4 --submit sge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from nvitk.core.click_config import config_dir_click_option

from . import diffusion, env, functional, registration, structural, utils
from .run import FslCommand

_PATH = click.Path(path_type=Path)


class _FslInput(click.ParamType):
    """An existing file or folder — or, as FSL accepts, an image basename without ``.nii.gz``."""

    name = "path"

    def convert(self, value, param, ctx):  # noqa: ANN001 — click's signature
        path = Path(value).expanduser()
        if path.exists():
            return path
        for suffix in (".nii.gz", ".nii"):
            candidate = path.with_name(path.name + suffix)
            if candidate.exists():
                return candidate
        self.fail(f"Path '{value}' does not exist.", param, ctx)


_IN = _FslInput()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
def main() -> None:
    """FSL (FMRIB Software Library) tools, on $FSLDIR or nvitk's own FSL core."""


def _run(cmd: FslCommand) -> FslCommand:
    try:
        outputs = cmd.run()
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    for role, path in outputs.items():
        click.echo(f"{role:22s} {path}")
    return cmd


def _table(rows: dict[str, Any]) -> None:
    for key, value in rows.items():
        click.echo(f"{key:24s} {json.dumps(value) if isinstance(value, dict) else value}")


# ── set-up ───────────────────────────────────────────────────────────────────


@main.command("info")
@click.option("--json", "as_json", is_flag=True)
def info_cmd(as_json: bool) -> None:
    """Which FSL installation nvitk uses, its version, components and GPU builds."""
    report = env.report()
    if as_json:
        click.echo(json.dumps(report, indent=2))
    else:
        _table(report)


@main.command("install")
@click.option("--components", default=",".join(c for c in ("structural", "registration", "diffusion", "functional")),
              show_default=True, help="Comma-separated: structural, registration, diffusion, functional (base always).")
@click.option("--cuda/--no-cuda", default=None, help="GPU builds of eddy/bedpostx/probtrackx2 (default: if NVIDIA).")
@click.option("--version", default=None, help="FSL release (default: the pinned one).")
@click.option("--dry-run", is_flag=True, help="Only report what would be downloaded.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask.")
def install_cmd(components: str, cuda: bool | None, version: str | None, dry_run: bool, yes: bool) -> None:
    """Create nvitk's own FSL core in this environment ($PREFIX/share/nvitk/fsl)."""
    from .install import FSL_VERSION, estimate_download, install_fsl

    chosen = ["base", *[c.strip() for c in components.split(",") if c.strip() and c.strip() != "base"]]
    version = version or FSL_VERSION
    if dry_run or not yes:
        try:
            gb, count = estimate_download(chosen, cuda=bool(cuda), version=version)
            click.echo(f"FSL {version} ({', '.join(chosen)}): {count} packages, ≈{gb:.1f} GB to download.")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"(could not estimate the download: {exc})")
        if dry_run:
            return
        click.confirm("Install?", abort=True)
    try:
        install_fsl(chosen, cuda=cuda, version=version)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


@main.command("use")
@click.argument("fsldir", required=False, type=_IN)
@click.option("--clear", is_flag=True, help="Forget the choice ($FSLDIR / nvitk's core apply again).")
def use_cmd(fsldir: Path | None, clear: bool) -> None:
    """Make nvitk use this FSL installation (stored in external_tools.json)."""
    from nvitk.core import external_tools

    if clear:
        click.echo(f"Cleared in {external_tools.remember(fsldir=None)}")
        return
    if fsldir is None:
        raise click.UsageError("Give an FSL directory, or --clear.")
    click.echo(f"Saved in {external_tools.remember(fsldir=str(fsldir.resolve()))}")


# ── structural ───────────────────────────────────────────────────────────────


@main.command("bet")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("-f", default=0.5, show_default=True, help="Fractional intensity threshold (smaller = larger brain).")
@click.option("-g", default=0.0, show_default=True, help="Vertical gradient in f.")
@click.option("--mode", type=click.Choice(list(structural.BET_MODES)), default="standard", show_default=True)
@click.option("--no-mask", is_flag=True)
@click.option("--skull", is_flag=True)
@click.option("--surfaces", is_flag=True, help="Skull / scalp surfaces (-A, needs registration).")
def bet_cmd(input: Path, output: Path, f: float, g: float, mode: str, no_mask: bool, skull: bool, surfaces: bool) -> None:
    """Brain extraction."""
    _run(structural.bet(input, output, f=f, g=g, mode=mode, mask=not no_mask, skull=skull, surfaces=surfaces))


@main.command("fast")
@click.argument("input", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--classes", default=3, show_default=True)
@click.option("--type", "img_type", type=click.Choice(list(structural.FAST_TYPES)), default="T1", show_default=True)
@click.option("--no-bias", is_flag=True, help="No bias-field correction.")
def fast_cmd(input: Path, out_base: Path, classes: int, img_type: str, no_bias: bool) -> None:
    """Tissue segmentation (CSF/GM/WM) and bias correction."""
    _run(structural.fast(input, out_base, classes=classes, img_type=img_type, no_bias=no_bias))


@main.command("first")
@click.argument("input", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--structures", default="", help=f"Comma-separated subset of {', '.join(structural.FIRST_STRUCTURES)}.")
@click.option("--brain-extracted", is_flag=True)
@click.option("--method", default="auto", show_default=True, help="auto, fast, none, or a threshold.")
def first_cmd(input: Path, out_base: Path, structures: str, brain_extracted: bool, method: str) -> None:
    """Subcortical segmentation (run_first_all)."""
    names = [s.strip() for s in structures.split(",") if s.strip()]
    _run(structural.first(input, out_base, structures=names, brain_extracted=brain_extracted, method=method))


@main.command("anat")
@click.argument("input", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--type", "img_type", type=click.Choice(["T1", "T2", "PD"]), default="T1", show_default=True)
@click.option("--strong-bias", is_flag=True)
@click.option("--no-crop", is_flag=True)
@click.option("--no-reg", is_flag=True)
@click.option("--no-nonlinreg", is_flag=True)
@click.option("--no-seg", is_flag=True)
@click.option("--no-subcortseg", is_flag=True)
def anat_cmd(input: Path, out_base: Path, img_type: str, strong_bias: bool, no_crop: bool, no_reg: bool,
             no_nonlinreg: bool, no_seg: bool, no_subcortseg: bool) -> None:
    """fsl_anat: reorient, crop, bias-correct, register to MNI, FAST and FIRST (→ <out>.anat)."""
    _run(structural.fsl_anat(input, out_base, img_type=img_type, strong_bias=strong_bias, no_crop=no_crop,
                             no_reg=no_reg, no_nonlinreg=no_nonlinreg, no_seg=no_seg, no_subcortseg=no_subcortseg))


@main.command("siena")
@click.argument("input1", type=_IN)
@click.argument("input2", type=_IN)
@click.argument("out_dir", type=_PATH)
@click.option("--bet-opts", default="", help='e.g. "-f 0.3"')
@click.option("--std-mask", is_flag=True)
@click.option("--ventricles", is_flag=True, help="VIENA ventricular analysis.")
def siena_cmd(input1: Path, input2: Path, out_dir: Path, bet_opts: str, std_mask: bool, ventricles: bool) -> None:
    """Two-time-point brain atrophy (PBVC)."""
    cmd = _run(structural.siena(input1, input2, out_dir, bet_opts=bet_opts, std_mask=std_mask, ventricles=ventricles))
    report = cmd.existing_outputs().get("report")
    if report:
        _table(structural.parse_siena_report(report))


@main.command("sienax")
@click.argument("input", type=_IN)
@click.argument("out_dir", type=_PATH)
@click.option("--bet-opts", default="")
@click.option("--regional", is_flag=True)
@click.option("--lesion-mask", type=_IN, default=None)
def sienax_cmd(input: Path, out_dir: Path, bet_opts: str, regional: bool, lesion_mask: Path | None) -> None:
    """Normalised brain, grey and white matter volumes."""
    cmd = _run(structural.sienax(input, out_dir, bet_opts=bet_opts, regional=regional, lesion_mask=lesion_mask))
    report = cmd.existing_outputs().get("report")
    if report:
        _table(structural.parse_sienax_report(report))


@main.command("reorient2std")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
def reorient_cmd(input: Path, output: Path) -> None:
    """Reorient to the MNI orientation (axis swaps / flips only)."""
    _run(structural.reorient2std(input, output))


@main.command("robustfov")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--brain-size", default=170.0, show_default=True, help="Brain size in z (mm).")
def robustfov_cmd(input: Path, output: Path, brain_size: float) -> None:
    """Crop the lower head and neck."""
    _run(structural.robustfov(input, output, brain_size_mm=brain_size))


@main.command("deface")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("-f", default=0.5, show_default=True)
@click.option("--bias-correct", is_flag=True)
def deface_cmd(input: Path, output: Path, f: float, bias_correct: bool) -> None:
    """Remove the face from a structural image."""
    _run(structural.deface(input, output, f=f, bias_correct=bias_correct))


@main.command("lesion-filling")
@click.argument("input", type=_IN)
@click.argument("lesion_mask", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--wm-mask", type=_IN, default=None)
def lesion_cmd(input: Path, lesion_mask: Path, output: Path, wm_mask: Path | None) -> None:
    """Fill lesions with surrounding white-matter intensities."""
    _run(structural.lesion_filling(input, lesion_mask, output, wm_mask=wm_mask))


# ── registration ─────────────────────────────────────────────────────────────


@main.command("fnirt")
@click.argument("input", type=_IN)
@click.argument("ref", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--aff", type=_IN, default=None, help="FLIRT affine input→ref to start from.")
@click.option("--config", default="T1_2_MNI152_2mm", show_default=True)
@click.option("--ref-mask", type=_IN, default=None)
def fnirt_cmd(input: Path, ref: Path, out_base: Path, aff: Path | None, config: str, ref_mask: Path | None) -> None:
    """Non-linear registration (warped image, coefficients, field)."""
    _run(registration.fnirt(input, ref, out_base, affine=aff, config=config, ref_mask=ref_mask))


@main.command("applywarp")
@click.argument("input", type=_IN)
@click.argument("ref", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--warp", type=_IN, default=None)
@click.option("--premat", type=_IN, default=None)
@click.option("--postmat", type=_IN, default=None)
@click.option("--interp", type=click.Choice(list(registration.APPLYWARP_INTERP)), default="trilinear")
@click.option("--abs", "absolute", is_flag=True, help="The warp holds absolute coordinates.")
def applywarp_cmd(input: Path, ref: Path, output: Path, warp: Path | None, premat: Path | None, postmat: Path | None,
                  interp: str, absolute: bool) -> None:
    """Resample through a warp and/or affines."""
    _run(registration.applywarp(input, ref, output, warp=warp, premat=premat, postmat=postmat, interp=interp,
                                relative=not absolute))


@main.command("invwarp")
@click.argument("warp", type=_IN)
@click.argument("ref", type=_IN)
@click.argument("output", type=_PATH)
def invwarp_cmd(warp: Path, ref: Path, output: Path) -> None:
    """Invert a warp field."""
    _run(registration.invwarp(warp, ref, output))


@main.command("convertwarp")
@click.argument("ref", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--warp1", type=_IN, default=None)
@click.option("--warp2", type=_IN, default=None)
@click.option("--premat", type=_IN, default=None)
@click.option("--midmat", type=_IN, default=None)
@click.option("--postmat", type=_IN, default=None)
def convertwarp_cmd(ref: Path, output: Path, **kw: Any) -> None:
    """Combine affines and warps into one warp."""
    _run(registration.convertwarp(ref, output, **kw))


@main.command("convert-xfm")
@click.argument("output", type=_PATH)
@click.option("--inverse", type=_IN, default=None, help="Invert this matrix.")
@click.option("--concat", nargs=2, type=_IN, default=None, help="A then B: B∘A.")
def convert_xfm_cmd(output: Path, inverse: Path | None, concat: tuple[Path, Path] | None) -> None:
    """Invert or concatenate FLIRT matrices."""
    _run(registration.convert_xfm(output, inverse=inverse, concat=tuple(concat) if concat else None))


@main.command("mcflirt")
@click.argument("input", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--ref-vol", type=int, default=None)
@click.option("--mean-vol", is_flag=True)
@click.option("--cost", type=click.Choice(list(registration.MCFLIRT_COSTS)), default="normcorr")
def mcflirt_cmd(input: Path, out_base: Path, ref_vol: int | None, mean_vol: bool, cost: str) -> None:
    """Motion correction of a 4D series."""
    _run(registration.mcflirt(input, out_base, ref_vol=ref_vol, mean_vol=mean_vol, cost=cost))


@main.command("epi-reg")
@click.option("--epi", required=True, type=_IN)
@click.option("--t1", required=True, type=_IN)
@click.option("--t1brain", required=True, type=_IN)
@click.option("--out", "out_base", required=True, type=_PATH)
@click.option("--wmseg", type=_IN, default=None)
@click.option("--fmap", type=_IN, default=None)
@click.option("--fmapmag", type=_IN, default=None)
@click.option("--fmapmagbrain", type=_IN, default=None)
@click.option("--echospacing", type=float, default=None)
@click.option("--pedir", default=None)
def epi_reg_cmd(epi: Path, t1: Path, t1brain: Path, out_base: Path, wmseg: Path | None, fmap: Path | None,
                fmapmag: Path | None, fmapmagbrain: Path | None, echospacing: float | None, pedir: str | None) -> None:
    """EPI → T1 boundary-based registration."""
    _run(registration.epi_reg(epi, t1, t1brain, out_base, wmseg=wmseg, fmap=fmap, fmap_mag=fmapmag,
                              fmap_mag_brain=fmapmagbrain, echo_spacing=echospacing, pe_dir=pedir))


# ── diffusion ────────────────────────────────────────────────────────────────


@main.command("topup")
@click.argument("imain", type=_IN)
@click.argument("datain", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--config", default="b02b0.cnf", show_default=True)
def topup_cmd(imain: Path, datain: Path, out_base: Path, config: str) -> None:
    """Susceptibility field from reversed phase-encode b0s."""
    _run(diffusion.topup(imain, datain, out_base, config=config))


@main.command("applytopup")
@click.option("--imain", required=True, multiple=True, type=_IN)
@click.option("--datain", required=True, type=_IN)
@click.option("--inindex", required=True, help="Comma-separated rows of datain, one per image.")
@click.option("--topup", "topup_base", required=True, type=_PATH)
@click.option("--out", "output", required=True, type=_PATH)
@click.option("--method", default="jac", show_default=True)
def applytopup_cmd(imain: tuple[Path, ...], datain: Path, inindex: str, topup_base: Path, output: Path, method: str) -> None:
    """Apply a TOPUP field."""
    _run(diffusion.applytopup(list(imain), datain, [int(i) for i in inindex.split(",")], topup_base, output,
                              method=method))


@main.command("eddy")
@click.option("--imain", required=True, type=_IN)
@click.option("--mask", required=True, type=_IN)
@click.option("--acqp", required=True, type=_IN)
@click.option("--index", required=True, type=_IN)
@click.option("--bvecs", required=True, type=_IN)
@click.option("--bvals", required=True, type=_IN)
@click.option("--out", "out_base", required=True, type=_PATH)
@click.option("--topup", "topup_base", type=_PATH, default=None)
@click.option("--no-repol", is_flag=True)
@click.option("--cuda/--cpu", default=None)
def eddy_cmd(imain: Path, mask: Path, acqp: Path, index: Path, bvecs: Path, bvals: Path, out_base: Path,
             topup_base: Path | None, no_repol: bool, cuda: bool | None) -> None:
    """Eddy-current and motion correction."""
    _run(diffusion.eddy(imain, mask, acqp, index, bvecs, bvals, out_base, topup_base=topup_base, repol=not no_repol,
                        cuda=cuda))


@main.command("dtifit")
@click.option("--data", required=True, type=_IN)
@click.option("--mask", required=True, type=_IN)
@click.option("--bvecs", required=True, type=_IN)
@click.option("--bvals", required=True, type=_IN)
@click.option("--out", "out_base", required=True, type=_PATH)
@click.option("--wls", is_flag=True)
@click.option("--save-tensor", is_flag=True)
def dtifit_cmd(data: Path, mask: Path, bvecs: Path, bvals: Path, out_base: Path, wls: bool, save_tensor: bool) -> None:
    """Diffusion tensor fit (FA, MD, eigenvalues and eigenvectors)."""
    _run(diffusion.dtifit(data, mask, bvecs, bvals, out_base, wls=wls, save_tensor=save_tensor))


@main.command("bedpostx")
@click.argument("folder", type=_PATH)
@click.option("--data", type=_IN, default=None, help="Build FOLDER from these files first.")
@click.option("--mask", type=_IN, default=None)
@click.option("--bvecs", type=_IN, default=None)
@click.option("--bvals", type=_IN, default=None)
@click.option("--fibres", default=3, show_default=True)
@click.option("--gpu/--cpu", default=None)
def bedpostx_cmd(folder: Path, data: Path | None, mask: Path | None, bvecs: Path | None, bvals: Path | None,
                 fibres: int, gpu: bool | None) -> None:
    """Fibre orientation estimation (FOLDER holds data, nodif_brain_mask, bvecs, bvals)."""
    if data is not None:
        if None in (mask, bvecs, bvals):
            raise click.UsageError("--data needs --mask, --bvecs and --bvals.")
        diffusion.prepare_bedpostx_dir(folder, data, mask, bvecs, bvals)  # type: ignore[arg-type]
    _run(diffusion.bedpostx(folder, fibres=fibres, gpu=gpu))


@main.command("probtrackx2")
@click.argument("bedpostx_dir", type=_IN)
@click.argument("seed", type=_IN)
@click.argument("out_dir", type=_PATH)
@click.option("--samples", default=5000, show_default=True)
@click.option("--waypoints", type=_IN, default=None)
@click.option("--avoid", "exclusion", type=_IN, default=None)
@click.option("--targets", "target_masks", type=_IN, default=None, help="Text file listing target masks.")
@click.option("--xfm", "seed_to_diff", type=_IN, default=None, help="Seed space → diffusion transform.")
@click.option("--seedref", "seed_ref", type=_IN, default=None)
@click.option("--gpu/--cpu", default=None)
def probtrackx_cmd(bedpostx_dir: Path, seed: Path, out_dir: Path, samples: int, **kw: Any) -> None:
    """Probabilistic tractography from a seed mask."""
    _run(diffusion.probtrackx2(bedpostx_dir, seed, out_dir, samples=samples, **kw))


@main.command("tbss")
@click.argument("folder", type=_PATH)
@click.option("--fa", "fa_images", multiple=True, type=_IN, help="FA maps to copy into FOLDER (repeat).")
@click.option("--target", default="FMRIB58", show_default=True,
              help="FMRIB58, most_representative, or an image path.")
@click.option("--threshold", default=0.2, show_default=True)
@click.option("--non-fa", default="", help="Comma-separated measures (MD, L1 …) with folders in FOLDER.")
def tbss_cmd(folder: Path, fa_images: tuple[Path, ...], target: str, threshold: float, non_fa: str) -> None:
    """Tract-based spatial statistics, steps 1-4 (then run `nvitk-voxelwise` / randomise)."""
    if fa_images:
        diffusion.prepare_tbss_dir(folder, list(fa_images))
    _run(diffusion.tbss(folder, target=target, threshold=threshold,
                        non_fa=[m.strip() for m in non_fa.split(",") if m.strip()]))


@main.command("xtract")
@click.argument("bedpostx_dir", type=_IN)
@click.argument("out_dir", type=_PATH)
@click.option("--species", type=click.Choice(list(diffusion.XTRACT_SPECIES)), default="HUMAN")
@click.option("--stdwarp", nargs=2, type=_IN, default=None, help="standard→diffusion, diffusion→standard warps.")
@click.option("--tracts", default="", help="Comma-separated subset.")
@click.option("--native", is_flag=True)
@click.option("--gpu/--cpu", default=None)
def xtract_cmd(bedpostx_dir: Path, out_dir: Path, species: str, stdwarp: tuple[Path, Path] | None, tracts: str,
               native: bool, gpu: bool | None) -> None:
    """Standardised tractography of the major white-matter tracts."""
    _run(diffusion.xtract(bedpostx_dir, out_dir, species=species, std_warps=tuple(stdwarp) if stdwarp else None,
                          tracts=[t.strip() for t in tracts.split(",") if t.strip()], native=native, gpu=gpu))


# ── functional ───────────────────────────────────────────────────────────────


@main.command("melodic")
@click.argument("input", type=_IN)
@click.argument("out_dir", type=_PATH)
@click.option("--dim", default=0, show_default=True, help="Components (0 = automatic).")
@click.option("--approach", type=click.Choice(list(functional.MELODIC_APPROACHES)), default="symm")
@click.option("--tr", type=float, default=None)
@click.option("--no-bet", is_flag=True)
def melodic_cmd(input: Path, out_dir: Path, dim: int, approach: str, tr: float | None, no_bet: bool) -> None:
    """Independent component analysis of a 4D series."""
    _run(functional.melodic(input, out_dir, dim=dim, approach=approach, tr=tr, no_bet=no_bet))


@main.command("susan")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--sigma", "sigma_mm", required=True, type=float, help="Smoothing sigma (mm).")
@click.option("--bt", "brightness_threshold", default=-1.0, show_default=True,
              help="Brightness threshold (negative: 10% of the robust range).")
@click.option("--2d", "two_d", is_flag=True)
def susan_cmd(input: Path, output: Path, sigma_mm: float, brightness_threshold: float, two_d: bool) -> None:
    """Edge-preserving noise reduction."""
    _run(functional.susan(input, output, sigma_mm=sigma_mm, brightness_threshold=brightness_threshold,
                          dim=2 if two_d else 3))


@main.command("slicetimer")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--tr", default=3.0, show_default=True)
@click.option("--order", type=click.Choice(list(functional.SLICE_ORDERS)), default="bottom-up")
@click.option("--direction", default=3, show_default=True)
@click.option("--timings", type=_IN, default=None)
def slicetimer_cmd(input: Path, output: Path, tr: float, order: str, direction: int, timings: Path | None) -> None:
    """Slice-timing correction."""
    _run(functional.slicetimer(input, output, tr=tr, order=order, direction=direction, timings=timings))


@main.command("fugue")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--fmap", "fieldmap", required=True, type=_IN, help="Field map in rad/s.")
@click.option("--dwell", "dwell_time", required=True, type=float, help="Effective echo spacing (s).")
@click.option("--unwarpdir", type=click.Choice(list(functional.UNWARP_DIRECTIONS)), default="y")
def fugue_cmd(input: Path, output: Path, fieldmap: Path, dwell_time: float, unwarpdir: str) -> None:
    """Field-map unwarping of an EPI."""
    _run(functional.fugue(input, output, fieldmap, dwell_time=dwell_time, unwarp_dir=unwarpdir))


@main.command("prepare-fieldmap")
@click.argument("phase", type=_IN)
@click.argument("magnitude_brain", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--delta-te", default=2.46, show_default=True, help="Echo time difference (ms).")
@click.option("--scanner", type=click.Choice(list(functional.FIELDMAP_SCANNERS)), default="SIEMENS")
def prepare_fieldmap_cmd(phase: Path, magnitude_brain: Path, output: Path, delta_te: float, scanner: str) -> None:
    """A field map in rad/s from phase-difference (Siemens) or Hz (GE) data."""
    _run(functional.prepare_fieldmap(phase, magnitude_brain, output, delta_te_ms=delta_te, scanner=scanner))


# ── utilities ────────────────────────────────────────────────────────────────


@main.command("maths", context_settings={"ignore_unknown_options": True, "help_option_names": ["--help"]})
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.argument("operations", nargs=-1, required=True, type=click.UNPROCESSED)
def maths_cmd(input: Path, output: Path, operations: tuple[str, ...]) -> None:
    """fslmaths, e.g. `nvitk-fsl maths in.nii.gz out.nii.gz -thr 100 -bin` (operations after the output)."""
    _run(utils.maths(input, " ".join(operations), output))


@main.command("stats")
@click.argument("input", type=_IN)
@click.option("--mask", type=_IN, default=None)
@click.option("--options", default="-R -r -M -S -V", show_default=True)
def stats_cmd(input: Path, mask: Path | None, options: str) -> None:
    """fslstats summary of an image (optionally inside a mask)."""
    opts = options.split()
    cmd = utils.stats(input, opts, mask=mask)
    try:
        cmd.run(on_line=lambda _t: None)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    _table(utils.parse_stats(cmd.stdout, opts))


@main.command("meants")
@click.argument("input", type=_IN)
@click.argument("output", type=_PATH)
@click.option("--mask", type=_IN, default=None)
@click.option("--coord", nargs=3, type=float, default=None)
@click.option("--mm", is_flag=True)
def meants_cmd(input: Path, output: Path, mask: Path | None, coord: tuple[float, float, float] | None, mm: bool) -> None:
    """Mean time series in a mask, or at a coordinate."""
    _run(utils.meants(input, output, mask=mask, coord=coord, mm=mm))


@main.command("cluster")
@click.argument("input", type=_IN)
@click.argument("out_base", type=_PATH)
@click.option("--thresh", required=True, type=float)
@click.option("--connectivity", type=click.Choice(["6", "18", "26"]), default="26")
@click.option("--min-extent", default=0)
def cluster_cmd(input: Path, out_base: Path, thresh: float, connectivity: str, min_extent: int) -> None:
    """Clusters above a threshold (index map + table)."""
    cmd = _run(utils.cluster(input, thresh, out_base, connectivity=int(connectivity), min_extent=min_extent))
    _table(utils.parse_cluster_table(cmd.stdout))


@main.command("smoothest")
@click.option("--mask", required=True, type=_IN)
@click.option("--zstat", type=_IN, default=None)
@click.option("--res", "residuals", type=_IN, default=None)
@click.option("--dof", type=int, default=None)
def smoothest_cmd(mask: Path, zstat: Path | None, residuals: Path | None, dof: int | None) -> None:
    """Smoothness of a statistic image or of residuals."""
    cmd = utils.smoothest(zstat=zstat, residuals=residuals, dof=dof, mask=mask)
    try:
        cmd.run(on_line=lambda _t: None)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    _table(utils.parse_smoothest(cmd.stdout))


@main.command("run", context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False})
@click.argument("argv", nargs=-1, type=click.UNPROCESSED, required=True)
def run_cmd(argv: tuple[str, ...]) -> None:
    """Any FSL command in the FSL environment: `nvitk-fsl run feat design.fsf`."""
    _run(utils.command(list(argv), title=argv[0]))


# Every command except the set-up ones can also run as an SGE job in the nvitk container
# (``--submit sge``); the GPU builds of eddy / bedpostx / probtrackx2 ask for a GPU node.
from nvitk.cli._external_sge import enable_sge  # noqa: E402

enable_sge(
    main,
    module="nvitk.fsl.cli",
    title="nvitk FSL",
    gpu_commands=("eddy", "bedpostx", "probtrackx2", "xtract"),
    local_only=("info", "install", "use"),
)


if __name__ == "__main__":
    sys.exit(main())
