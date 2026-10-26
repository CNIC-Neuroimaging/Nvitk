"""``nvitk-fastsurfer``: FastSurfer from the command line, with this environment's install.

Deep-learning modules (no FreeSurfer needed)::

    nvitk-fastsurfer seg     --t1 T1.nii.gz --sd subjects --sid sub01 [--native-image]
    nvitk-fastsurfer cereb   --sd subjects --sid sub01        # reuses the asegdkt above
    nvitk-fastsurfer hypothal --sd subjects --sid sub01 [--t2 T2.nii.gz]
    nvitk-fastsurfer cc      --sd subjects --sid sub01

Surfaces and longitudinal processing (FreeSurfer 7.4.1 + license)::

    nvitk-fastsurfer install-freesurfer [--dest DIR] [--license license.txt]
    nvitk-fastsurfer surf --sd subjects --sid sub01
    nvitk-fastsurfer run  --t1 T1.nii.gz --sd subjects --sid sub01
    nvitk-fastsurfer long --sd subjects --tid base --t1s a.nii.gz b.nii.gz --tpids a b

Tools: ``segstats``, ``reduce-to-aseg``, ``conform``; set-up: ``info``, ``install``,
``checkpoints``, ``set-license``, ``set-freesurfer``.

Every processing command runs here by default, or on the cluster with ``--submit sge`` (in
the nvitk container, which carries FastSurfer and a pruned FreeSurfer; the license is bound
into the job)::

    nvitk-fastsurfer seg --t1 /data/T1.nii.gz --sd /data/subjects --sid sub01 --submit sge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from nvitk.core.click_config import config_dir_click_option

from . import config
from .runner import (
    DEVICE_CHOICES,
    REG_MODE_CHOICES,
    FastSurferRequest,
    conform_argv,
    fastsurfer_argv,
    long_argv,
    reduce_to_aseg_argv,
    run,
    segstats_argv,
)

_PATH = click.Path(path_type=Path)
_EXISTING = click.Path(exists=True, path_type=Path)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
def main() -> None:
    """FastSurfer (Deep-MI) whole-brain, cerebellum, hypothalamus and corpus-callosum
    segmentation, surfaces and longitudinal processing, run with this environment's install."""


# shared options


def _subject_options(fn: Any) -> Any:
    fn = click.option("--sid", required=True, help="Subject id (a folder inside --sd).")(fn)
    fn = click.option("--sd", required=True, type=_PATH, help="Subjects directory ($SUBJECTS_DIR).")(fn)
    return fn


def _compute_options(fn: Any) -> Any:
    fn = click.option("--threads", default=None, help="Threads: a number, 'auto' or 'max'.")(fn)
    fn = click.option("--batch", default=1, show_default=True, type=int, help="Inference batch size.")(fn)
    fn = click.option(
        "--viewagg-device", default="auto", show_default=True, type=click.Choice(DEVICE_CHOICES),
        help="Where the view aggregation runs.",
    )(fn)
    fn = click.option(
        "--device", default="auto", show_default=True, type=click.Choice(DEVICE_CHOICES),
        help="Network inference device.",
    )(fn)
    return fn


def _seg_options(fn: Any) -> Any:
    fn = click.option("--qc-snap", is_flag=True, help="Write QC snapshots (qc_snapshots/).")(fn)
    fn = click.option(
        "--tal-reg", is_flag=True, help="Talairach registration for eTIV (needs the FreeSurfer license)."
    )(fn)
    fn = click.option(
        "--biasfield/--no-biasfield", default=True, show_default=True,
        help="Bias-field correction and partial-volume corrected statistics.",
    )(fn)
    fn = click.option(
        "--native-image", is_flag=True,
        help="Write every output on the input's own grid (implies --vox-size keep; not with surfaces).",
    )(fn)
    fn = click.option(
        "--vox-size", default="min", show_default=True,
        help="Processing voxel size: a number in [0.7, 1], 'min' (the input's smallest) or 'keep'.",
    )(fn)
    return fn


def _surface_options(fn: Any) -> Any:
    fn = click.option("--ignore-fs-version", is_flag=True, help="Accept another FreeSurfer version.")(fn)
    fn = click.option("--fsaparc", is_flag=True, help="Also create FreeSurfer's own aparc segmentations.")(fn)
    fn = click.option("--no-surfreg", is_flag=True, help="Skip the spherical registration (faster).")(fn)
    fn = click.option("--parallel", is_flag=True, help="Process both hemispheres at once.")(fn)
    fn = click.option("--3T", "three_tesla", is_flag=True, help="Use the 3T Talairach atlas.")(fn)
    fn = click.option("--fs-license", type=_PATH, default=None, help="FreeSurfer license file.")(fn)
    fn = click.option("--freesurfer-home", type=_PATH, default=None, help="FreeSurfer installation.")(fn)
    return fn


def _execute(req: FastSurferRequest) -> None:
    try:
        argv, env = fastsurfer_argv(req)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    code = run(argv, env)
    if code != 0:
        raise click.ClickException(f"FastSurfer failed (exit code {code}). Log: {req.subject_dir / 'scripts'}")
    click.echo(f"Done: {req.subject_dir}")


def _seg_request(sd: Path, sid: str, t1: Path | None, **kw: Any) -> FastSurferRequest:
    return FastSurferRequest(
        sd=sd, sid=sid, t1=t1,
        vox_size=str(kw.pop("vox_size", "min")),
        native_image=bool(kw.pop("native_image", False)),
        biasfield=bool(kw.pop("biasfield", True)),
        tal_reg=bool(kw.pop("tal_reg", False)),
        qc_snap=bool(kw.pop("qc_snap", False)),
        device=kw.pop("device", "auto"),
        viewagg_device=kw.pop("viewagg_device", "auto"),
        batch=int(kw.pop("batch", 1)),
        threads=kw.pop("threads", None),
        **kw,
    )


# segmentation


@main.command("seg")
@_subject_options
@click.option("--t1", required=True, type=_EXISTING, help="T1-weighted full-head image.")
@click.option("--t2", type=_EXISTING, default=None, help="Optional T2 (improves HypVINN).")
@click.option("--reg-mode", default="coreg", show_default=True, type=click.Choice(REG_MODE_CHOICES),
              help="T1/T2 registration for HypVINN.")
@click.option("--cereb/--no-cereb", default=True, show_default=True, help="Cerebellum (CerebNet).")
@click.option("--hypothal/--no-hypothal", default=True, show_default=True, help="Hypothalamus (HypVINN).")
@click.option("--cc/--no-cc", default=True, show_default=True, help="Corpus callosum segmentation and shape.")
@_seg_options
@_compute_options
def seg_cmd(sd: Path, sid: str, t1: Path, t2: Path | None, reg_mode: str, cereb: bool, hypothal: bool,
            cc: bool, **kw: Any) -> None:
    """Whole-brain segmentation (aparc.DKTatlas+aseg) and, by default, every sub-module."""
    _execute(_seg_request(sd, sid, t1, t2=t2, reg_mode=reg_mode, cereb=cereb, hypothal=hypothal, cc=cc, **kw))


def _module_command(name: str, module: str, doc: str) -> None:
    @main.command(name, help=doc)
    @_subject_options
    @click.option("--t1", type=_EXISTING, default=None,
                  help="T1 image; needed only when the subject has no whole-brain segmentation yet.")
    @_seg_options
    @_compute_options
    def _cmd(sd: Path, sid: str, t1: Path | None, **kw: Any) -> None:
        from .outputs import SubjectOutputs

        subject = SubjectOutputs(Path(sd) / sid)
        reuse = subject.has_segmentation()
        if not reuse and t1 is None:
            raise click.ClickException(
                f"{subject.subject_dir} has no whole-brain segmentation yet: pass --t1 to compute it first."
            )
        flags = {m: (m == module) for m in ("cereb", "hypothal", "cc")}
        _execute(_seg_request(sd, sid, subject.input_t1() if reuse else t1, asegdkt=not reuse, **flags, **kw))


_module_command("cereb", "cereb", "Cerebellum sub-segmentation (CerebNet) on a FastSurfer subject.")
_module_command("hypothal", "hypothal", "Hypothalamus sub-segmentation (HypVINN) on a FastSurfer subject.")
_module_command("cc", "cc", "Corpus callosum segmentation and shape analysis on a FastSurfer subject.")


# surfaces / full / longitudinal


@main.command("surf")
@_subject_options
@_surface_options
@click.option("--threads", default=None, help="Threads: a number, 'auto' or 'max'.")
def surf_cmd(sd: Path, sid: str, freesurfer_home: Path | None, fs_license: Path | None, three_tesla: bool,
             parallel: bool, no_surfreg: bool, fsaparc: bool, ignore_fs_version: bool, threads: str | None) -> None:
    """Surface reconstruction (recon-surf) on an already segmented subject."""
    _execute(FastSurferRequest(
        sd=sd, sid=sid, segmentation=False, surfaces=True, freesurfer_home=freesurfer_home,
        fs_license=fs_license, three_tesla=three_tesla, parallel=parallel, no_surfreg=no_surfreg,
        fsaparc=fsaparc, ignore_fs_version=ignore_fs_version, threads=threads,
    ))


@main.command("run")
@_subject_options
@click.option("--t1", required=True, type=_EXISTING, help="T1-weighted full-head image.")
@click.option("--t2", type=_EXISTING, default=None, help="Optional T2 (improves HypVINN).")
@click.option("--no-surfaces", is_flag=True, help="Segmentation only (same as `seg`).")
@click.option("--vox-size", default="min", show_default=True, help="A number in [0.7, 1], or 'min'.")
@click.option("--biasfield/--no-biasfield", default=True, show_default=True)
@_surface_options
@_compute_options
def run_cmd(sd: Path, sid: str, t1: Path, t2: Path | None, no_surfaces: bool, vox_size: str, biasfield: bool,
            freesurfer_home: Path | None, fs_license: Path | None, three_tesla: bool, parallel: bool,
            no_surfreg: bool, fsaparc: bool, ignore_fs_version: bool, **kw: Any) -> None:
    """The full pipeline: every segmentation module, then the surfaces."""
    _execute(_seg_request(
        sd, sid, t1, t2=t2, vox_size=vox_size, biasfield=biasfield, surfaces=not no_surfaces,
        freesurfer_home=freesurfer_home, fs_license=fs_license, three_tesla=three_tesla, parallel=parallel,
        no_surfreg=no_surfreg, fsaparc=fsaparc, ignore_fs_version=ignore_fs_version, **kw,
    ))


@main.command("long")
@click.option("--sd", required=True, type=_PATH, help="Subjects directory.")
@click.option("--tid", required=True, help="Id of the person-specific template.")
@click.option("--t1s", required=True, multiple=True, type=_EXISTING, help="T1 of each time point (repeat).")
@click.option("--tpids", required=True, multiple=True, help="Id of each time point (repeat, same order).")
@click.option("--parallel", type=int, default=None, help="Process this many time points at once.")
@click.option("--freesurfer-home", type=_PATH, default=None)
@click.option("--fs-license", type=_PATH, default=None)
def long_cmd(sd: Path, tid: str, t1s: tuple[Path, ...], tpids: tuple[str, ...], parallel: int | None,
             freesurfer_home: Path | None, fs_license: Path | None) -> None:
    """Longitudinal processing: an unbiased template, then every time point (surfaces included)."""
    try:
        argv, env = long_argv(sd, tid, list(t1s), list(tpids), parallel=parallel,
                              freesurfer_home=freesurfer_home, fs_license=fs_license)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    code = run(argv, env)
    if code != 0:
        raise click.ClickException(f"long_fastsurfer.sh failed (exit code {code}).")


# module tools


def _run_module(argv_env: tuple[list[str], dict[str, str]], what: str) -> None:
    code = run(*argv_env)
    if code != 0:
        raise click.ClickException(f"{what} failed (exit code {code}).")


@main.command("segstats")
@click.option("--seg", "segfile", required=True, type=_EXISTING, help="Segmentation (label map).")
@click.option("--norm", "normfile", type=_EXISTING, default=None,
              help="Intensity image for partial-volume correction (bias-corrected T1).")
@click.option("-o", "--out", required=True, type=_PATH, help="Output .stats file.")
@click.option("--lut", type=_EXISTING, default=None, help="Colour LUT (default: FreeSurferColorLUT).")
@click.option("--ids", default="", help="Only these label ids (comma-separated).")
@click.option("--threads", type=int, default=None)
def segstats_cmd(segfile: Path, normfile: Path | None, out: Path, lut: Path | None, ids: str,
                 threads: int | None) -> None:
    """Partial-volume corrected volume (and intensity) statistics per label."""
    id_list = [int(v) for v in ids.split(",") if v.strip()]
    _run_module(segstats_argv(segfile, out, normfile=normfile, lut=lut, ids=id_list, threads=threads), "segstats")


@main.command("reduce-to-aseg")
@click.option("-i", "--input", "aparc", required=True, type=_EXISTING, help="aparc+aseg segmentation.")
@click.option("-o", "--out", required=True, type=_PATH, help="Output aseg.")
@click.option("--mask", type=_PATH, default=None, help="Also write a brain mask here.")
@click.option("--fixwm/--no-fixwm", default=True, show_default=True, help="Relabel disconnected WM.")
def reduce_cmd(aparc: Path, out: Path, mask: Path | None, fixwm: bool) -> None:
    """Reduce an aparc+aseg segmentation to aseg (and a brain mask)."""
    _run_module(reduce_to_aseg_argv(aparc, out, out_mask=mask, fixwm=fixwm), "reduce_to_aseg")


@main.command("conform")
@click.option("-i", "--input", "image", required=True, type=_EXISTING)
@click.option("-o", "--out", required=True, type=_PATH)
@click.option("--vox-size", default="1", show_default=True, help="A number, 'min' or 'any'.")
@click.option("--order", default=1, show_default=True, type=click.IntRange(0, 3), help="Interpolation order.")
@click.option("--seg", "seg_input", is_flag=True, help="The input is a label map (nearest neighbour).")
def conform_cmd(image: Path, out: Path, vox_size: str, order: int, seg_input: bool) -> None:
    """Conform an image to FastSurfer's space (isotropic, 256³, LIA, uint8)."""
    _run_module(conform_argv(image, out, vox_size=vox_size, order=order, seg_input=seg_input), "conform")


# set-up


@main.command("info")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def info_cmd(as_json: bool) -> None:
    """Where FastSurfer, its checkpoints, FreeSurfer and the license are."""
    report = config.doctor()
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    for key, value in report.items():
        if isinstance(value, list):
            value = ", ".join(value) if value else "none"
        click.echo(f"{key:20s} {value}")
    if not report["surfaces_ready"]:
        for problem in config.surface_requirements()[2]:
            click.echo(f"  surfaces: {problem}")


@main.command("install")
@click.option("--version", default=None, help=f"FastSurfer tag (default {config.FASTSURFER_VERSION}).")
@click.option("--no-checkpoints", is_flag=True, help="Do not download the network weights now.")
def install_cmd(version: str | None, no_checkpoints: bool) -> None:
    """Install or update FastSurfer (tree + checkpoints) in this environment."""
    from .install import install_fastsurfer

    if install_fastsurfer(version=version, checkpoints=not no_checkpoints) != 0:
        raise click.ClickException("FastSurfer installation failed.")


@main.command("checkpoints")
def checkpoints_cmd() -> None:
    """Download any missing FastSurfer network weights."""
    from .install import download_checkpoints

    if download_checkpoints() != 0:
        raise click.ClickException("Some checkpoints could not be downloaded.")


@main.command("install-freesurfer")
@click.option("--dest", type=_PATH, default=None, show_default=False,
              help=f"Where to unpack it (default {config.default_freesurfer_dest()}).")
@click.option("--license", "license_file", type=_EXISTING, default=None, help="FreeSurfer license to copy in.")
@click.option("--url", default=None, help="Another archive URL (default: the build for this Linux).")
@click.option("--keep-archive", is_flag=True, help="Keep the downloaded .tar.gz.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask before downloading.")
def install_freesurfer_cmd(dest: Path | None, license_file: Path | None, url: str | None, keep_archive: bool,
                           yes: bool) -> None:
    """Download and unpack FreeSurfer 7.4.1 (≈9.5 GB) for the surface pipelines."""
    from .install import freesurfer_url, install_freesurfer, remote_size

    url = url or freesurfer_url()
    if not yes:
        size = remote_size(url)
        click.confirm(
            f"Download {url}" + (f" ({size / 1e9:.1f} GB)" if size else "") + " and unpack it?", abort=True
        )
    try:
        install_freesurfer(dest, license_file=license_file, url=url, keep_archive=keep_archive)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc


@main.command("set-license")
@click.argument("license_file", type=_EXISTING)
def set_license_cmd(license_file: Path) -> None:
    """Remember the FreeSurfer license file."""
    from nvitk.core import external_tools

    click.echo(f"Saved in {external_tools.remember(fs_license=str(license_file.resolve()))}")


@main.command("set-freesurfer")
@click.argument("freesurfer_home", type=_EXISTING)
def set_freesurfer_cmd(freesurfer_home: Path) -> None:
    """Remember an existing FreeSurfer installation."""
    from nvitk.core import external_tools

    if not (freesurfer_home / "SetUpFreeSurfer.sh").is_file():
        raise click.ClickException(f"{freesurfer_home} is not a FreeSurfer installation (no SetUpFreeSurfer.sh).")
    click.echo(f"Saved in {external_tools.remember(freesurfer_home=str(freesurfer_home.resolve()))}")


def _prepare_sge(name: str, values: dict[str, Any]) -> None:
    """Before a cluster run: the container brings its own FreeSurfer, and the license (which only
    this machine's configuration may know) is sent along as a bound file."""
    if "freesurfer_home" in values:
        values["freesurfer_home"] = None
    needs_license = name in ("surf", "run", "long") or bool(values.get("tal_reg"))
    if needs_license and "fs_license" in values and not values.get("fs_license"):
        lic = config.fs_license()
        if lic is None:
            raise click.ClickException(
                "The surface pipeline needs a FreeSurfer license: pass --fs-license (it is bound into the job)."
            )
        values["fs_license"] = lic


# Every command except the set-up ones can also run as an SGE job in the nvitk container
# (``--submit sge``); the network-based ones ask for a GPU node.
from nvitk.cli._external_sge import enable_sge  # noqa: E402

enable_sge(
    main,
    module="nvitk.segmentation.fastsurfer.cli",
    title="nvitk FastSurfer",
    gpu_commands=("seg", "cereb", "hypothal", "cc", "run", "long"),
    local_only=("info", "install", "checkpoints", "install-freesurfer", "set-license", "set-freesurfer"),
    prepare=_prepare_sge,
)


if __name__ == "__main__":
    sys.exit(main())
