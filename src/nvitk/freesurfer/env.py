"""Finding FreeSurfer, its license, and the environment its programs run in.

The installation is ``$FREESURFER_HOME``, then ``freesurfer_home`` in ``external_tools.json``, then
the conventional places (``/usr/local/freesurfer``, ``/opt/freesurfer``,
``~/.local/opt/freesurfer``, ``~/freesurfer``) and the folder nvitk's FastSurfer installer writes
to. The license is ``$FS_LICENSE``, then ``fs_license`` in ``external_tools.json``, then
``license.txt`` / ``.license`` in the installation.

FreeSurfer before 6.0 reads its license only from ``$FREESURFER_HOME`` and ignores
``$FS_LICENSE``. When such an installation's license file lives elsewhere, :func:`fs_env` runs
its programs from an *overlay*: a folder of links to the installation plus ``license.txt``, kept
under ``~/.cache/nvitk``. The installation itself is never written to.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from nvitk.core import external_tools

LICENSE_URL = "https://surfer.nmr.mgh.harvard.edu/registration.html"


class FreeSurferNotFound(RuntimeError):
    """No FreeSurfer installation, or one without a needed program."""


class FreeSurferLicenseError(RuntimeError):
    """A FreeSurfer program refused to run for want of a valid license."""


def _expand(raw: str | os.PathLike[str] | None) -> Path | None:
    if not raw:
        return None
    return Path(os.path.expandvars(str(raw))).expanduser()


def is_freesurfer(path: Path | None) -> bool:
    """*path* is a FreeSurfer installation (it has ``SetUpFreeSurfer.sh``)."""
    return path is not None and (Path(path) / "SetUpFreeSurfer.sh").is_file()


def candidates() -> list[Path]:
    """Every place looked at for an installation, in order."""
    out = [
        _expand(os.environ.get("FREESURFER_HOME")),
        external_tools.get_path("freesurfer_home"),
        Path("/usr/local/freesurfer"),
        Path("/opt/freesurfer"),
        Path.home() / ".local" / "opt" / "freesurfer",
        Path.home() / "freesurfer",
    ]
    try:  # nvitk's FastSurfer installer (FreeSurfer 7.4.1)
        from nvitk.segmentation.fastsurfer.config import default_freesurfer_dest

        dest = default_freesurfer_dest()
        out += [dest, dest / "freesurfer"]
    except Exception:  # noqa: BLE001
        pass
    return [p for p in out if p is not None]


def freesurfer_home() -> Path | None:
    """The first FreeSurfer installation found (module docstring)."""
    for candidate in candidates():
        if is_freesurfer(candidate):
            return candidate
    return None


def freesurfer_version(home: Path | None = None) -> str | None:
    """The version in the installation's ``build-stamp.txt`` (``5.3.0``, ``7.4.1`` ...)."""
    home = home or freesurfer_home()
    stamp = (home / "build-stamp.txt") if home is not None else None
    if stamp is None or not stamp.is_file():
        return None
    text = stamp.read_text(encoding="utf-8", errors="replace").strip()
    match = re.search(r"(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else text or None


def major_version(home: Path | None = None) -> int | None:
    version = freesurfer_version(home)
    try:
        return int(str(version).split(".")[0]) if version else None
    except ValueError:
        return None


def fs_license(fs_home: Path | None = None) -> Path | None:
    """The license file (module docstring), or ``None``."""
    found: list[Path | None] = [_expand(os.environ.get("FS_LICENSE")), external_tools.get_path("fs_license")]
    fs_home = fs_home or freesurfer_home()
    if fs_home is not None:
        found += [fs_home / "license.txt", fs_home / ".license"]
    for candidate in found:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def bin_dir(home: Path | None = None) -> Path | None:
    home = home or freesurfer_home()
    return home / "bin" if home is not None else None


def which(program: str, home: Path | None = None) -> Path | None:
    """*program* in the installation's ``bin`` (``None`` when absent)."""
    b = bin_dir(home)
    if b is None:
        return None
    path = b / program
    return path if path.is_file() and os.access(path, os.X_OK) else None


def available(home: Path | None = None) -> bool:
    """A FreeSurfer installation with a ``bin`` folder was found."""
    b = bin_dir(home)
    return b is not None and b.is_dir()


def missing(programs: tuple[str, ...] | list[str], home: Path | None = None) -> list[str]:
    return [p for p in programs if which(p, home) is None]


def _overlay(home: Path, license_file: Path) -> Path:
    """A folder that looks like *home* with *license_file* as its ``license.txt`` (links only)."""
    key = hashlib.sha1(f"{home.resolve()}|{license_file.resolve()}".encode()).hexdigest()[:12]
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nvitk" / f"freesurfer-{key}"
    root.mkdir(parents=True, exist_ok=True)
    for entry in home.iterdir():
        link = root / entry.name
        if entry.name in ("license.txt", ".license") or link.exists() or link.is_symlink():
            continue
        link.symlink_to(entry)
    lic = root / "license.txt"
    if lic.is_symlink() or lic.exists():
        lic.unlink()
    lic.symlink_to(license_file.resolve())
    return root


def effective_home(home: Path | None = None, license_file: Path | None = None) -> Path | None:
    """The ``FREESURFER_HOME`` programs run with: the installation, or its overlay when it is
    older than 6.0 and its license lives elsewhere."""
    home = home or freesurfer_home()
    if home is None:
        return None
    lic = license_file or fs_license(home)
    in_home = any((home / n).is_file() for n in ("license.txt", ".license"))
    if lic is not None and not in_home and (major_version(home) or 99) < 6:
        return _overlay(home, Path(lic))
    return home


def fs_env(
    home: Path | None = None,
    *,
    subjects_dir: Path | None = None,
    license_file: Path | None = None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment FreeSurfer programs run in: ``FREESURFER_HOME`` (the overlay, when one is
    needed), ``FS_LICENSE``, ``SUBJECTS_DIR`` and FreeSurfer's ``bin`` first on ``PATH``."""
    home = home or freesurfer_home()
    if home is None:
        raise FreeSurferNotFound(not_found_message())
    env = dict(os.environ if base is None else base)
    run_home = effective_home(home, license_file) or home
    lic = license_file or fs_license(home)
    env["FREESURFER_HOME"] = str(run_home)
    env["FREESURFER"] = str(run_home)
    if lic is not None:
        env["FS_LICENSE"] = str(lic)
    if subjects_dir is not None:
        env["SUBJECTS_DIR"] = str(subjects_dir)
    else:
        env.setdefault("SUBJECTS_DIR", str(run_home / "subjects"))
    for key in ("MNI_DIR", "FSFAST_HOME", "FUNCTIONALS_DIR"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join([str(run_home / "bin"), env.get("PATH", "")])
    env.setdefault("FS_FREESURFERENV_NO_OUTPUT", "1")
    return env


def not_found_message() -> str:
    return (
        "FreeSurfer was not found. Set $FREESURFER_HOME (or `freesurfer_home` in external_tools.json) "
        "to an installation; nvitk looks in " + ", ".join(str(p) for p in candidates()[2:6]) + "."
    )


_LICENSE_PATTERNS = (
    re.compile(r"license file .* not found", re.I),
    re.compile(r"invalid freesurfer license", re.I),
    re.compile(r"license.*(expired|invalid|key)", re.I),
)


def license_problem(output: str) -> str | None:
    """A clear message when *output* (a program's stdout + stderr) is FreeSurfer refusing to run
    for want of a license; ``None`` otherwise."""
    if not any(p.search(output or "") for p in _LICENSE_PATTERNS):
        return None
    home = freesurfer_home()
    old = (major_version(home) or 99) < 6
    where = f"{home}/license.txt" if home else "$FREESURFER_HOME/license.txt"
    return (
        "FreeSurfer refused to run without a license. Get one (free) at " + LICENSE_URL + " and "
        + (f"save it as {where}, or point $FS_LICENSE / the tools' license field to it (nvitk links it "
           "into an overlay, since FreeSurfer 5.x reads only $FREESURFER_HOME)." if old else
           f"point $FS_LICENSE (or the tools' license field) to it, or save it as {where}.")
    )


def report() -> dict[str, Any]:
    """What was found: installation, version, license, the topology programs present."""
    home = freesurfer_home()
    programs = ("mri_pretess", "mri_mc", "mris_extract_main_component", "mris_euler_number", "mris_smooth",
                "mris_inflate", "mris_sphere", "mris_fix_topology", "mris_topo_fixer", "mris_remove_intersection",
                "defect-seg")
    return {
        "freesurfer_home": str(home) if home else None,
        "version": freesurfer_version(home),
        "license": str(fs_license(home)) if home and fs_license(home) else None,
        "programs": {p: bool(which(p, home)) for p in programs} if home else {},
        "tcsh": shutil.which("tcsh") is not None or shutil.which("csh") is not None,
    }


__all__ = [
    "FreeSurferLicenseError",
    "FreeSurferNotFound",
    "LICENSE_URL",
    "available",
    "candidates",
    "effective_home",
    "freesurfer_home",
    "freesurfer_version",
    "fs_env",
    "fs_license",
    "is_freesurfer",
    "license_problem",
    "major_version",
    "missing",
    "report",
    "which",
]
