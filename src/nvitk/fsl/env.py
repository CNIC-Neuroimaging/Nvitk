"""Which FSL installation nvitk drives, and the environment its commands run in.

Resolution order (first usable wins):

1. ``$NVITK_FSLDIR`` — an explicit choice for nvitk only;
2. ``fsldir`` in ``external_tools.json`` (``nvitk-fsl use DIR``);
3. ``$FSLDIR`` — an installation the shell already sources (``~/fsl``);
4. nvitk's own FSL core in this environment, ``$PREFIX/share/nvitk/fsl`` (``nvitk-fsl install``);
5. inferred from where FSL commands are on ``PATH``.

FSL is never modified here: commands run with ``FSLDIR`` set, ``$FSLDIR/share/fsl/bin``
(conda installs: one wrapper per FSL command, nothing else) ahead on ``PATH`` and
``$FSLDIR/bin`` behind it, and ``FSLOUTPUTTYPE=NIFTI_GZ`` so output names are predictable.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from nvitk.core import external_tools

#: Commands each component ships (enough to tell whether that component is installed).
COMPONENT_PROBES: dict[str, tuple[str, ...]] = {
    "base": ("fslmaths", "fslstats", "fslmerge", "cluster", "randomise"),
    "structural": ("bet", "fast", "run_first_all", "fsl_anat", "siena", "sienax"),
    "registration": ("flirt", "fnirt", "applywarp", "mcflirt", "epi_reg"),
    "diffusion": ("eddy", "topup", "dtifit", "bedpostx", "probtrackx2", "tbss_1_preproc", "xtract"),
    "functional": ("melodic", "susan", "slicetimer", "fugue"),
    "asl": ("oxford_asl", "basil", "fabber_asl", "asl_file", "asl_calib"),
}

#: GPU builds, used instead of the CPU ones when present.
CUDA_COMMANDS: dict[str, tuple[str, ...]] = {
    "eddy": ("eddy_cuda", "eddy_cuda11.0", "eddy_cuda10.2"),
    "bedpostx": ("bedpostx_gpu",),
    "probtrackx2": ("probtrackx2_gpu",),
}


def _is_fsldir(path: Path | None) -> bool:
    return path is not None and (
        (path / "etc" / "fslversion").is_file() or (path / "share" / "fsl" / "bin").is_dir()
    )


def managed_fsldir() -> Path:
    """Where ``nvitk-fsl install`` puts nvitk's FSL core."""
    return external_tools.managed_dir("fsl")


def resolve_fsldir() -> Path | None:
    """The FSL installation to use (see the module docstring for the order)."""
    for raw in (
        os.environ.get("NVITK_FSLDIR", "").strip(),
        str(external_tools.get_path("fsldir") or ""),
        os.environ.get("FSLDIR", "").strip(),
    ):
        if raw and _is_fsldir(Path(raw).expanduser()):
            return Path(raw).expanduser()
    if _is_fsldir(managed_fsldir()):
        return managed_fsldir()
    # FSL on PATH without FSLDIR (a GUI started without sourcing fsl.sh): the binaries'
    # location is enough — conda installs put wrappers in $FSLDIR/share/fsl/bin and the
    # real files in $FSLDIR/bin, and an ancestor holds etc/fslversion either way.
    for name in ("fslmaths", "randomise", "fslmerge", "bet"):
        exe = shutil.which(name)
        if not exe:
            continue
        for parent in Path(exe).resolve().parents:
            if (parent / "etc" / "fslversion").is_file():
                return parent
    return None


def fsl_version(fsldir: Path | None = None) -> str:
    """The FSL version (``etc/fslversion``), or ``""``."""
    fsldir = fsldir or resolve_fsldir()
    if fsldir is None:
        return ""
    for rel in ("etc/fslversion", "VERSION", "share/fsl/etc/fslversion"):
        candidate = Path(fsldir) / rel
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip().splitlines()[0]
        except (OSError, IndexError):
            continue
    return ""


def bin_dirs(fsldir: Path) -> list[Path]:
    """FSL's command directories: the wrapper directory first, then ``bin``."""
    return [d for d in (fsldir / "share" / "fsl" / "bin", fsldir / "bin") if d.is_dir()]


def _with_fsl_path(path: str, fsldir: Path) -> str:
    """*path* with FSL's wrapper directory first and its ``bin`` last.

    Only the wrappers go in front, as FSL's own ``fsl.sh`` does: ``$FSLDIR/bin`` also holds the
    FSL environment's python, curl, tar …, which must never shadow the caller's. It is kept at
    the end so a command without a wrapper is still found.
    """
    wrappers = fsldir / "share" / "fsl" / "bin"
    tail = fsldir / "bin"
    rest = [p for p in path.split(os.pathsep) if p and p not in (str(wrappers), str(tail))]
    front = [str(wrappers)] if wrappers.is_dir() else []
    back = [str(tail)] if tail.is_dir() else []
    return os.pathsep.join(front + rest + back)


def fsl_env(fsldir: Path | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment an FSL command runs in (a copy of *base*, default ``os.environ``)."""
    fsldir = fsldir or resolve_fsldir()
    env = dict(os.environ if base is None else base)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env.setdefault("FSLMULTIFILEQUIT", "TRUE")
    if fsldir is None:
        return env
    env["FSLDIR"] = str(fsldir)
    env["PATH"] = _with_fsl_path(env.get("PATH", ""), fsldir)
    for key, rel in (("FSLTCLSH", "bin/fsltclsh"), ("FSLWISH", "bin/fslwish")):
        if (fsldir / rel).is_file():
            env[key] = str(fsldir / rel)
    return env


def ensure_fsl_in_process() -> Path | None:
    """Make the resolved FSL the one this process (NiPype, ``shutil.which``) finds; returns it."""
    fsldir = resolve_fsldir()
    env = fsl_env(fsldir)
    for key in ("FSLDIR", "PATH", "FSLOUTPUTTYPE", "FSLMULTIFILEQUIT", "FSLTCLSH", "FSLWISH"):
        if key in env:
            os.environ[key] = env[key]
    return fsldir


def which(command: str, fsldir: Path | None = None) -> str | None:
    """Full path of an FSL *command* in *fsldir* (resolved when omitted), else ``None``."""
    fsldir = fsldir or resolve_fsldir()
    if fsldir is None:
        return None
    return shutil.which(command, path=os.pathsep.join(str(d) for d in bin_dirs(fsldir)))


def installed_components(fsldir: Path | None = None) -> dict[str, bool]:
    """``{component: present}`` by probing each component's commands."""
    fsldir = fsldir or resolve_fsldir()
    return {
        name: fsldir is not None and all(which(cmd, fsldir) for cmd in probes)
        for name, probes in COMPONENT_PROBES.items()
    }


def missing_commands(commands: tuple[str, ...] | list[str], fsldir: Path | None = None) -> list[str]:
    """Which of *commands* the resolved FSL lacks."""
    fsldir = fsldir or resolve_fsldir()
    return [c for c in commands if which(c, fsldir) is None]


def cuda_command(command: str, fsldir: Path | None = None) -> str | None:
    """The GPU build of *command* (``eddy`` → ``eddy_cuda…``) when installed, else ``None``."""
    for candidate in CUDA_COMMANDS.get(command, ()):
        if which(candidate, fsldir):
            return candidate
    return None


class FslNotFound(RuntimeError):
    """No FSL installation, or one without the commands a tool needs."""


def require_commands(*commands: str) -> Path:
    """The FSLDIR to use, raising :class:`FslNotFound` naming what is missing."""
    fsldir = resolve_fsldir()
    if fsldir is None:
        raise FslNotFound(
            "No FSL installation found. Run `nvitk-fsl install` for nvitk's own FSL core, or point "
            "to an existing one ($FSLDIR, or `nvitk-fsl use /path/to/fsl`)."
        )
    missing = missing_commands(commands, fsldir)
    if missing:
        raise FslNotFound(
            f"FSL at {fsldir} lacks {', '.join(missing)}. Run `nvitk-fsl install` (nvitk's own core "
            "with every component) or add the packages to that installation."
        )
    return fsldir


def report() -> dict[str, object]:
    """Everything worth knowing about the FSL in use."""
    fsldir = resolve_fsldir()
    out: dict[str, object] = {
        "fsldir": str(fsldir) if fsldir else None,
        "version": fsl_version(fsldir) if fsldir else None,
        "managed": str(managed_fsldir()) if _is_fsldir(managed_fsldir()) else None,
        "source": _source(fsldir),
    }
    out.update({f"component:{k}": v for k, v in installed_components(fsldir).items()})
    out["cuda"] = {c: cuda_command(c, fsldir) for c in CUDA_COMMANDS} if fsldir else {}
    return out


def _source(fsldir: Path | None) -> str:
    if fsldir is None:
        return "none"
    if os.environ.get("NVITK_FSLDIR", "").strip() and Path(os.environ["NVITK_FSLDIR"]).expanduser() == fsldir:
        return "$NVITK_FSLDIR"
    if external_tools.get_path("fsldir") == fsldir:
        return "external_tools.json"
    if os.environ.get("FSLDIR", "").strip() and Path(os.environ["FSLDIR"]).expanduser() == fsldir:
        return "$FSLDIR"
    if fsldir == managed_fsldir():
        return "nvitk (this environment)"
    return "PATH"


__all__ = [
    "COMPONENT_PROBES",
    "CUDA_COMMANDS",
    "FslNotFound",
    "bin_dirs",
    "cuda_command",
    "ensure_fsl_in_process",
    "fsl_env",
    "fsl_version",
    "installed_components",
    "managed_fsldir",
    "missing_commands",
    "report",
    "require_commands",
    "resolve_fsldir",
    "which",
]
