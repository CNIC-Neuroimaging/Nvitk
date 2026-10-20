"""FastSurfer command lines: ``run_fastsurfer.sh``, ``long_fastsurfer.sh`` and the module scripts.

Every builder returns an argv list plus the environment to run it in, so the same
call serves the CLI (run in the foreground), the GUI (an
:class:`~nvitk.gui.core.external_jobs.ExternalJob`) and tests (inspect the argv).
FastSurfer always runs with *this* environment's Python (``--py``), from its own tree
on ``PYTHONPATH`` — see :mod:`nvitk.segmentation.fastsurfer.config`.

The modules map onto ``run_fastsurfer.sh`` flags:

=============================  ==============================================
asegdkt (whole brain)          on unless ``asegdkt=False`` (``--no_asegdkt``)
CerebNet (cerebellum)          ``cereb``      (``--no_cereb`` when off)
HypVINN (hypothalamus)         ``hypothal``   (``--no_hypothal`` when off)
corpus callosum                ``cc``         (``--no_cc`` when off)
bias field + PV stats          ``biasfield``  (``--no_biasfield`` when off)
surfaces (recon-surf)          ``surfaces``   (``--seg_only`` when off;
                                ``surf_only`` skips the segmentation)
=============================  ==============================================
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from nvitk.core.logger import Logger

from . import config

log = Logger()

#: ``--vox_size`` choices besides a number in [0.7, 1].
VOX_SIZE_CHOICES = ("min", "1", "keep")
#: ``--device`` / ``--viewagg_device`` choices.
DEVICE_CHOICES = ("auto", "cuda", "cpu")
#: ``--reg_mode`` (T1/T2 registration for HypVINN).
REG_MODE_CHOICES = ("coreg", "robust", "none")


class FastSurferNotInstalled(RuntimeError):
    """No FastSurfer tree to run."""


@dataclass
class FastSurferRequest:
    """One ``run_fastsurfer.sh`` invocation (defaults: segmentation only, every DL module)."""

    sd: Path
    sid: str
    t1: Path | None = None
    t2: Path | None = None
    # pipelines
    segmentation: bool = True
    surfaces: bool = False
    # modules
    asegdkt: bool = True
    cereb: bool = True
    hypothal: bool = True
    cc: bool = True
    biasfield: bool = True
    tal_reg: bool = False
    qc_snap: bool = False
    reg_mode: str = "coreg"
    # geometry
    vox_size: str = "min"
    native_image: bool = False
    # resources
    device: str = "auto"
    viewagg_device: str = "auto"
    batch: int = 1
    threads: int | str | None = None
    # surfaces
    three_tesla: bool = False
    parallel: bool = False
    no_surfreg: bool = False
    fsaparc: bool = False
    no_fs_t1: bool = False
    edits: bool = False
    ignore_fs_version: bool = False
    freesurfer_home: Path | None = None
    fs_license: Path | None = None
    extra: list[str] = field(default_factory=list)

    @property
    def subject_dir(self) -> Path:
        return Path(self.sd) / self.sid


def _home() -> Path:
    home = config.fastsurfer_home()
    if home is None:
        raise FastSurferNotInstalled(
            "FastSurfer is not installed in this environment. Run `nvitk-fastsurfer install` "
            "(or set $NVITK_FASTSURFER_HOME to a FastSurfer checkout)."
        )
    return home


def python_command() -> str:
    """The ``--py`` FastSurfer is given: this environment's interpreter, user site ignored."""
    return f"{shlex.quote(sys.executable)} -s"


def base_env(
    home: Path | None = None,
    *,
    freesurfer_home: Path | None = None,
    fs_license: Path | None = None,
    subjects_dir: Path | None = None,
) -> dict[str, str]:
    """The environment FastSurfer runs in."""
    home = home or _home()
    env = dict(os.environ)
    env["FASTSURFER_HOME"] = str(home)
    env["PYTHONPATH"] = str(home) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # A user site-packages must not shadow the environment's torch / numpy.
    env["PYTHONNOUSERSITE"] = "1"
    if freesurfer_home is not None:
        env["FREESURFER_HOME"] = str(freesurfer_home)
    if fs_license is not None:
        env["FS_LICENSE"] = str(fs_license)
    if subjects_dir is not None:
        env["SUBJECTS_DIR"] = str(subjects_dir)
    return env


def _with_freesurfer(argv: list[str], freesurfer_home: Path) -> list[str]:
    """Run *argv* with FreeSurfer's environment sourced (recon-surf needs its PATH)."""
    setup = Path(freesurfer_home) / "SetUpFreeSurfer.sh"
    return [
        "bash",
        "-c",
        'source "$1" >/dev/null 2>&1 || { echo "could not source $1" >&2; exit 1; }; shift; exec "$@"',
        "nvitk-fastsurfer",
        str(setup),
        *argv,
    ]


def fastsurfer_argv(req: FastSurferRequest) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` for *req*. Raises when the surface pipeline lacks FreeSurfer or a license."""
    home = _home()
    fs_home: Path | None = None
    lic: Path | None = None
    if req.surfaces or req.tal_reg:
        fs_home, lic, problems = config.surface_requirements(req.freesurfer_home, req.fs_license)
        if req.ignore_fs_version:
            problems = [p for p in problems if "validated against" not in p]
        if problems:
            raise RuntimeError("The surface pipeline cannot run:\n  - " + "\n  - ".join(problems))
    argv = ["bash", str(home / "run_fastsurfer.sh"), "--sd", str(Path(req.sd).resolve()), "--sid", req.sid]
    if req.t1 is not None:
        argv += ["--t1", str(Path(req.t1).resolve())]
    if req.t2 is not None:
        argv += ["--t2", str(Path(req.t2).resolve()), "--reg_mode", req.reg_mode]
    if not req.segmentation and req.surfaces:
        argv.append("--surf_only")
    elif req.segmentation and not req.surfaces:
        argv.append("--seg_only")
    elif not req.segmentation and not req.surfaces:
        raise ValueError("Nothing to run: enable the segmentation, the surfaces, or both.")
    if req.segmentation:
        for flag, on in (
            ("--no_asegdkt", req.asegdkt),
            ("--no_cereb", req.cereb),
            ("--no_hypothal", req.hypothal),
            ("--no_cc", req.cc),
            ("--no_biasfield", req.biasfield),
        ):
            if not on:
                argv.append(flag)
        if req.tal_reg:
            argv.append("--tal_reg")
        if req.qc_snap:
            argv.append("--qc_snap")
        if req.native_image:
            if req.surfaces:
                raise ValueError("--native_image is not compatible with the surface pipeline.")
            argv.append("--native_image")
        else:
            argv += ["--vox_size", str(req.vox_size)]
        argv += ["--device", req.device, "--viewagg_device", req.viewagg_device, "--batch", str(int(req.batch))]
    if req.threads not in (None, "", 0):
        argv += ["--threads", str(req.threads)]
    if req.surfaces:
        for flag, on in (
            ("--3T", req.three_tesla),
            ("--parallel", req.parallel),
            ("--no_surfreg", req.no_surfreg),
            ("--fsaparc", req.fsaparc),
            ("--no_fs_T1", req.no_fs_t1),
            ("--ignore_fs_version", req.ignore_fs_version),
        ):
            if on:
                argv.append(flag)
    if req.edits:
        argv.append("--edits")
    if lic is not None:
        argv += ["--fs_license", str(lic)]
    argv += ["--py", python_command(), *req.extra]
    env = base_env(home, freesurfer_home=fs_home, fs_license=lic, subjects_dir=Path(req.sd).resolve())
    if fs_home is not None:
        argv = _with_freesurfer(argv, fs_home)
    return argv, env


def long_argv(
    sd: Path,
    tid: str,
    t1s: Sequence[Path],
    tpids: Sequence[str],
    *,
    parallel: int | None = None,
    freesurfer_home: Path | None = None,
    fs_license: Path | None = None,
    extra: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` for ``long_fastsurfer.sh`` (template + every time point, surfaces included)."""
    home = _home()
    if len(t1s) != len(tpids) or len(t1s) < 2:
        raise ValueError("The longitudinal pipeline needs at least two T1 images, one id per image.")
    fs_home, lic, problems = config.surface_requirements(freesurfer_home, fs_license)
    if problems:
        raise RuntimeError("The longitudinal pipeline cannot run:\n  - " + "\n  - ".join(problems))
    argv = ["bash", str(home / "long_fastsurfer.sh"), "--tid", tid, "--sd", str(Path(sd).resolve())]
    argv += ["--t1s", *[str(Path(p).resolve()) for p in t1s], "--tpids", *tpids]
    if parallel:
        argv += ["--parallel", str(parallel)]
    argv += ["--fs_license", str(lic), "--py", python_command(), *extra]
    env = base_env(home, freesurfer_home=fs_home, fs_license=lic, subjects_dir=Path(sd).resolve())
    return _with_freesurfer(argv, fs_home), env


def module_argv(script: str, args: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` for one FastSurfer Python script (``FastSurferCNN/segstats.py`` …)."""
    home = _home()
    return [sys.executable, "-s", str(home / script), *[str(a) for a in args]], base_env(home)


def segstats_argv(
    segfile: Path,
    out_stats: Path,
    *,
    normfile: Path | None = None,
    lut: Path | None = None,
    ids: Sequence[int] = (),
    threads: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Partial-volume corrected segmentation statistics (``FastSurferCNN/segstats.py``)."""
    home = _home()
    lut = lut or home / "FastSurferCNN" / "config" / "FreeSurferColorLUT.txt"
    args = ["--segfile", str(segfile), "--segstatsfile", str(out_stats), "--lut", str(lut)]
    if normfile is not None:
        args += ["--normfile", str(normfile)]
    if ids:
        args += ["--ids", *[str(int(i)) for i in ids]]
    if threads:
        args += ["--threads", str(int(threads))]
    return module_argv("FastSurferCNN/segstats.py", args)


def reduce_to_aseg_argv(
    aparc_aseg: Path, out_aseg: Path, *, out_mask: Path | None = None, fixwm: bool = True
) -> tuple[list[str], dict[str, str]]:
    """aparc+aseg → aseg (and a brain mask) (``FastSurferCNN/reduce_to_aseg.py``)."""
    args = ["-i", str(aparc_aseg), "-o", str(out_aseg)]
    if out_mask is not None:
        args += ["--outmask", str(out_mask)]
    if fixwm:
        args.append("--fixwm")
    return module_argv("FastSurferCNN/reduce_to_aseg.py", args)


def conform_argv(
    image: Path, out: Path, *, vox_size: str = "1", order: int = 1, seg_input: bool = False
) -> tuple[list[str], dict[str, str]]:
    """Conform an image the way FastSurfer does (``FastSurferCNN/data_loader/conform.py``):
    isotropic *vox_size* (a number, ``min`` or ``any``), 256³ (or the field of view), LIA,
    uint8 rescaled. *seg_input* conforms a label map instead (nearest neighbour, ids kept)."""
    args = ["-i", str(image), "-o", str(out), "--vox_size", str(vox_size)]
    args += ["--seg_input"] if seg_input else ["--order", str(int(order))]
    return module_argv("FastSurferCNN/data_loader/conform.py", args)


def run(
    argv: Sequence[str],
    env: dict[str, str] | None = None,
    *,
    on_line: Callable[[str], None] | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Run a FastSurfer command in the foreground, streaming its output; returns the exit code."""
    log.info("$ " + " ".join(shlex.quote(str(a)) for a in argv))
    proc = subprocess.Popen(
        [str(a) for a in argv],
        env=env,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\n")
        if on_line is not None:
            on_line(text)
        else:
            print(text, flush=True)
    return int(proc.wait())


#: Lines of FastSurfer's log that say which stage it is in, by substring.
_STAGES = (
    ("conform", "conforming the input"),
    ("Run AsegDKT", "whole-brain segmentation (asegdkt)"),
    ("Running view aggregation", "view aggregation"),
    ("N4", "bias-field correction"),
    ("biasfield", "bias-field correction"),
    ("talairach", "Talairach registration"),
    ("CerebNet", "cerebellum (CerebNet)"),
    ("HypVINN", "hypothalamus (HypVINN)"),
    ("FastSurferCC", "corpus callosum"),
    ("fastsurfer_cc", "corpus callosum"),
    ("segstats", "statistics"),
    ("recon-surf", "surfaces (recon-surf)"),
    ("mris_", "surfaces (recon-surf)"),
    ("mri_mc", "surfaces: tessellation"),
    ("Spherical", "surfaces: spherical mapping"),
    ("mris_sphere", "surfaces: spherical mapping"),
    ("mris_register", "surfaces: registration"),
    ("mris_anatomical_stats", "surfaces: statistics"),
)


def stage_of(line: str) -> str | None:
    """A short stage description for a FastSurfer log line, or ``None``."""
    for needle, stage in _STAGES:
        if needle in line:
            return stage
    return None


__all__ = [
    "DEVICE_CHOICES",
    "FastSurferNotInstalled",
    "FastSurferRequest",
    "REG_MODE_CHOICES",
    "VOX_SIZE_CHOICES",
    "base_env",
    "conform_argv",
    "fastsurfer_argv",
    "long_argv",
    "module_argv",
    "python_command",
    "reduce_to_aseg_argv",
    "run",
    "segstats_argv",
    "stage_of",
]
