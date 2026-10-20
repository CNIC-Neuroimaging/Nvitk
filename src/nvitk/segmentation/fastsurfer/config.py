"""Where FastSurfer, its checkpoints, FreeSurfer and the FreeSurfer license are.

FastSurfer runs from its source tree with this environment's Python (it is not a
pip package). nvitk installs the tree into the environment —
``$PREFIX/share/nvitk/fastsurfer/FastSurfer`` (``recipe/fastsurfer/install_fastsurfer.sh``)
— but another checkout can be used instead.

The deep-learning modules (whole-brain asegdkt, CerebNet, HypVINN, corpus callosum)
need nothing else. The surface pipeline (``recon-surf``) and the longitudinal one also
need FreeSurfer :data:`FREESURFER_VERSION` and a FreeSurfer license file (free, from
https://surfer.nmr.mgh.harvard.edu/registration.html).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from nvitk.core import external_tools

#: The FastSurfer release nvitk installs: the newest that runs on Python 3.11 (v2.6.0
#: and later need Python >= 3.12).
FASTSURFER_VERSION = "v2.5.4"
#: The FreeSurfer release FastSurfer's recon-surf of that version is validated against.
FREESURFER_VERSION = "7.4.1"
#: Modules whose network weights live in ``<tree>/checkpoints``.
CHECKPOINT_MODULES = ("FastSurferCNN", "CerebNet", "HypVINN", "CorpusCallosum")


def _expand(raw: str | os.PathLike[str] | None) -> Path | None:
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def _is_tree(path: Path | None) -> bool:
    return path is not None and (path / "run_fastsurfer.sh").is_file()


def managed_tree() -> Path:
    """Where nvitk installs FastSurfer in this environment."""
    return external_tools.managed_dir("fastsurfer") / "FastSurfer"


def fastsurfer_home() -> Path | None:
    """The FastSurfer tree to run.

    ``$NVITK_FASTSURFER_HOME``, then ``$FASTSURFER_HOME``, then ``fastsurfer_home`` in
    ``external_tools.json``, then the tree nvitk installed into this environment.
    """
    for candidate in (
        _expand(os.environ.get("NVITK_FASTSURFER_HOME")),
        _expand(os.environ.get("FASTSURFER_HOME")),
        external_tools.get_path("fastsurfer_home"),
        managed_tree(),
    ):
        if _is_tree(candidate):
            return candidate
    return None


def installed_version(home: Path | None = None) -> str | None:
    """The FastSurfer version of *home* (the ``VERSION`` marker nvitk writes, else the package's)."""
    home = home or fastsurfer_home()
    if home is None:
        return None
    marker = home / "VERSION"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or None
    try:
        import tomllib

        data = tomllib.loads((home / "pyproject.toml").read_text(encoding="utf-8"))
        version = data.get("project", {}).get("version")
        return f"v{version}" if version else None
    except Exception:  # noqa: BLE001
        return None


def checkpoint_files(home: Path | None = None) -> dict[str, list[Path]]:
    """Every checkpoint each module needs, ``{module: [path, ...]}``."""
    home = home or fastsurfer_home()
    out: dict[str, list[Path]] = {}
    if home is None:
        return out
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return out
    for module in CHECKPOINT_MODULES:
        config = home / module / "config" / "checkpoint_paths.yaml"
        if not config.is_file():
            continue
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        out[module] = [
            Path(rel) if Path(rel).is_absolute() else home / rel
            for rel in (data.get("checkpoint") or {}).values()
        ]
    return out


def missing_checkpoints(home: Path | None = None) -> list[Path]:
    """Checkpoints not on disk yet (FastSurfer would download them on first use)."""
    return [p for files in checkpoint_files(home).values() for p in files if not p.is_file()]


# ── FreeSurfer ───────────────────────────────────────────────────────────────


def _is_freesurfer(path: Path | None) -> bool:
    return path is not None and (path / "SetUpFreeSurfer.sh").is_file()


def default_freesurfer_dest() -> Path:
    """Where ``nvitk-fastsurfer install-freesurfer`` puts FreeSurfer unless told otherwise."""
    return Path.home() / f"freesurfer-{FREESURFER_VERSION}"


def freesurfer_home() -> Path | None:
    """A FreeSurfer installation.

    ``$FREESURFER_HOME``, then ``freesurfer_home`` in ``external_tools.json``, then the
    conventional locations and the default destination of the opt-in installer.
    """
    for candidate in (
        _expand(os.environ.get("FREESURFER_HOME")),
        external_tools.get_path("freesurfer_home"),
        Path("/usr/local/freesurfer"),
        Path("/opt/freesurfer"),
        default_freesurfer_dest(),
        default_freesurfer_dest() / "freesurfer",
    ):
        if _is_freesurfer(candidate):
            return candidate
    return None


def freesurfer_version(home: Path | None = None) -> str | None:
    """The version FreeSurfer stamps into ``build-stamp.txt`` (e.g. ``7.4.1``), when readable."""
    home = home or freesurfer_home()
    stamp = (home / "build-stamp.txt") if home is not None else None
    if stamp is None or not stamp.is_file():
        return None
    text = stamp.read_text(encoding="utf-8", errors="replace").strip()
    import re

    match = re.search(r"(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else text or None


def fs_license(fs_home: Path | None = None) -> Path | None:
    """The FreeSurfer license file.

    ``$FS_LICENSE``, then ``fs_license`` in ``external_tools.json``, then
    ``license.txt`` / ``.license`` in the FreeSurfer installation.
    """
    candidates = [_expand(os.environ.get("FS_LICENSE")), external_tools.get_path("fs_license")]
    fs_home = fs_home or freesurfer_home()
    if fs_home is not None:
        candidates += [fs_home / "license.txt", fs_home / ".license"]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def surface_requirements(
    freesurfer: str | Path | None = None, license_file: str | Path | None = None
) -> tuple[Path | None, Path | None, list[str]]:
    """``(FREESURFER_HOME, license, problems)`` for the surface / longitudinal pipelines.

    Explicit arguments win over the lookups; *problems* is empty when both are usable.
    """
    fs_home = _expand(freesurfer) if freesurfer else freesurfer_home()
    lic = _expand(license_file) if license_file else fs_license(fs_home)
    problems: list[str] = []
    if not _is_freesurfer(fs_home):
        problems.append(
            f"FreeSurfer {FREESURFER_VERSION} was not found. Install it with "
            f"`nvitk-fastsurfer install-freesurfer` (≈9.5 GB download), or point to an "
            "existing installation with --freesurfer-home / $FREESURFER_HOME."
        )
    else:
        version = freesurfer_version(fs_home)
        if version and version != FREESURFER_VERSION:
            problems.append(
                f"FreeSurfer at {fs_home} is {version}; FastSurfer {FASTSURFER_VERSION}'s surface "
                f"pipeline is validated against {FREESURFER_VERSION} (allow it with ignore_fs_version)."
            )
    if lic is None or not Path(lic).is_file():
        problems.append(
            "No FreeSurfer license file. Register (free) at "
            "https://surfer.nmr.mgh.harvard.edu/registration.html and pass the file with "
            "--fs-license, $FS_LICENSE, or put it in $FREESURFER_HOME/license.txt."
        )
    for tool in ("tcsh", "bc"):
        if shutil.which(tool) is None:
            problems.append(f"`{tool}` is not on PATH (FreeSurfer's scripts need it).")
    return fs_home, (Path(lic) if lic else None), problems


def doctor() -> dict[str, Any]:
    """Everything worth knowing before a run, as a flat report."""
    home = fastsurfer_home()
    report: dict[str, Any] = {
        "fastsurfer_home": str(home) if home else None,
        "fastsurfer_version": installed_version(home),
        "expected_version": FASTSURFER_VERSION,
        "missing_checkpoints": [p.name for p in missing_checkpoints(home)] if home else None,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        report["torch"] = f"unavailable ({exc})"
        report["cuda"] = False
    fs_home = freesurfer_home()
    report["freesurfer_home"] = str(fs_home) if fs_home else None
    report["freesurfer_version"] = freesurfer_version(fs_home) if fs_home else None
    lic = fs_license(fs_home)
    report["fs_license"] = str(lic) if lic else None
    report["surfaces_ready"] = not surface_requirements()[2]
    return report


__all__ = [
    "CHECKPOINT_MODULES",
    "FASTSURFER_VERSION",
    "FREESURFER_VERSION",
    "checkpoint_files",
    "default_freesurfer_dest",
    "doctor",
    "fastsurfer_home",
    "freesurfer_home",
    "freesurfer_version",
    "fs_license",
    "installed_version",
    "managed_tree",
    "missing_checkpoints",
    "surface_requirements",
]
