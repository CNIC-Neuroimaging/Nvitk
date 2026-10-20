"""nvitk's own FSL core: a curated FSL release in ``$PREFIX/share/nvitk/fsl``.

FSL is distributed as conda packages on FSL's own channel, each release pinned by an
environment file (``fsl-<version>_linux-64.yml``). This creates a separate prefix with
the packages of the chosen *components* at exactly that release's versions:

==============  ===========================================================================
base            fslmaths/fslstats/… (avwutils), cluster, randomise, fsl_sub, MNI templates
structural      BET, FAST, FIRST (+ models), SIENA/SIENAX, lesion filling, defacing
registration    FLIRT, FNIRT (+ applywarp/invwarp/convertwarp), FUGUE, MCFLIRT
diffusion       TOPUP, EDDY, FDT (dtifit/bedpostx), probtrackx2, TBSS, XTRACT (+ atlases)
functional      MELODIC, SUSAN, slicetimer
asl             BASIL: oxford_asl, basil, asl_file, asl_calib (fabber ASL models); brings the
                structural and registration components it calls, and the atlases its region
                analysis uses
==============  ===========================================================================

and, with ``cuda``, the GPU builds of eddy, bedpostx and probtrackx2.

A *separate* prefix, not the nvitk environment: ``fsl-base`` brings its own Python
(3.14 for 6.0.7.23) and fslpy, which must not meet nvitk's pins. Creating it needs a
conda executable — the one that manages this environment, else ``mamba``/``conda`` on
PATH, else a micromamba downloaded next to it — which is why a conda post-link script
cannot do it and ``nvitk-fsl install`` (or the GUI) does.

FSL is free for non-commercial use; see https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, Iterable

from nvitk.core import external_tools

from .env import managed_fsldir

FSL_VERSION = "6.0.7.23"
FSL_CHANNEL = "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/"
RELEASES_URL = "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"

COMPONENTS: dict[str, tuple[str, ...]] = {
    "base": (
        "fsl-base", "fsl-avwutils", "fsl-utils", "fsl-misc_c", "fsl-misc_scripts", "fsl-misc_tcl",
        "fsl-sub", "fsl-data_standard", "fsl-cluster", "fsl-randomise",
    ),
    "structural": (
        "fsl-bet2", "fsl-fast4", "fsl-first", "fsl-data_first_models_317_bin",
        "fsl-data_first_models_336_bin", "fsl-siena", "fsl-lesions", "fsl-deface",
    ),
    "registration": ("fsl-flirt", "fsl-fnirt", "fsl-fugue", "fsl-mcflirt"),
    "diffusion": (
        "fsl-topup", "fsl-eddy", "fsl-fdt", "fsl-ptx2", "fsl-tbss", "fsl-xtract", "fsl-xtract_data",
        "fsl-data_atlases",
    ),
    "functional": ("fsl-melodic", "fsl-susan", "fsl-slicetimer"),
    "asl": ("fsl-oxford_asl", "fsl-basil", "fsl-fabber_core", "fsl-fabber_models_asl", "fsl-data_atlases"),
}
#: Components another one runs commands of (oxford_asl calls BET, FAST, FLIRT, MCFLIRT …).
COMPONENT_REQUIRES: dict[str, tuple[str, ...]] = {"asl": ("structural", "registration")}
#: GPU builds added with ``cuda`` (only meaningful with the diffusion component).
CUDA_PACKAGES: tuple[str, ...] = ("fsl-eddy-cuda", "fsl-fdt-cuda", "fsl-ptx2-cuda")
ALL_COMPONENTS: tuple[str, ...] = tuple(COMPONENTS)


def release_environment(version: str = FSL_VERSION) -> dict[str, str]:
    """``{package: version spec}`` from FSL's environment file for *version*."""
    import requests

    url = f"{RELEASES_URL}/fsl-{version}_linux-64.yml"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    specs: dict[str, str] = {}
    in_deps = False
    for raw in response.text.splitlines():
        if raw.strip().startswith("dependencies:"):
            in_deps = True
            continue
        if not in_deps or not raw.strip().startswith("- "):
            continue
        parts = raw.strip()[2:].split()
        if parts:
            specs[parts[0]] = parts[1] if len(parts) > 1 else ""
    if not specs:
        raise RuntimeError(f"No packages in {url}")
    return specs


def _match_spec(name: str, version: str) -> str:
    if not version:
        return name
    if "*" in version or any(c in version for c in "<>=,"):
        return f"{name} {version}"
    return f"{name}=={version}"


def package_specs(
    components: Iterable[str] = ALL_COMPONENTS, *, cuda: bool = False, version: str = FSL_VERSION
) -> tuple[list[str], list[str]]:
    """``(conda match specs, packages missing from the release)`` for *components*."""
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown FSL component(s): {', '.join(unknown)} (known: {', '.join(COMPONENTS)})")
    chosen = set(components)
    for component in list(chosen):
        chosen |= set(COMPONENT_REQUIRES.get(component, ()))
    wanted: list[str] = []
    for component in ("base", *[c for c in COMPONENTS if c != "base" and c in chosen]):
        wanted += [p for p in COMPONENTS[component] if p not in wanted]
    if cuda and "diffusion" in set(components):
        wanted += list(CUDA_PACKAGES)
    release = release_environment(version)
    specs = [_match_spec(name, release[name]) for name in wanted if name in release]
    missing = [name for name in wanted if name not in release]
    if "python" in release:
        specs.append(_match_spec("python", release["python"]))
    return specs, missing


# ── a conda to create the prefix with ────────────────────────────────────────


def _env_root() -> Path | None:
    prefix = Path(sys.prefix)
    return prefix.parent.parent if prefix.parent.name == "envs" else prefix


def conda_executable(*, download: bool = True, on_line: Callable[[str], None] | None = None) -> tuple[str, bool]:
    """``(executable, is_micromamba)``: this environment's conda/mamba, else one on PATH, else micromamba."""
    candidates: list[str] = []
    for var in ("MAMBA_EXE", "CONDA_EXE"):
        if os.environ.get(var):
            candidates.append(os.environ[var])
    root = _env_root()
    if root is not None:
        candidates += [str(root / "bin" / "mamba"), str(root / "bin" / "conda")]
    candidates += [shutil.which("mamba") or "", shutil.which("conda") or ""]
    for exe in candidates:
        if exe and Path(exe).is_file() and os.access(exe, os.X_OK):
            return exe, Path(exe).name == "micromamba"
    local = external_tools.managed_dir("micromamba") / "bin" / "micromamba"
    if local.is_file():
        return str(local), True
    if not download:
        raise RuntimeError("No conda, mamba or micromamba to create the FSL prefix with.")
    return str(_download_micromamba(local, on_line)), True


def _download_micromamba(target: Path, on_line: Callable[[str], None] | None) -> Path:
    import requests

    (on_line or print)(f"Downloading micromamba to {target.parent} ...")
    response = requests.get(MICROMAMBA_URL, timeout=120)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:bz2") as archive:
        member = archive.getmember("bin/micromamba")
        data = archive.extractfile(member)
        assert data is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.read())
    target.chmod(0o755)
    return target


def create_argv(conda: str, prefix: Path, specs: list[str], *, micromamba: bool, dry_run: bool = False) -> list[str]:
    """The ``create`` command line for *prefix* (FSL's channel first, strict priority)."""
    argv = [conda, "create", "-y", "-p", str(prefix), "--override-channels", "-c", FSL_CHANNEL,
            "-c", "conda-forge", "--strict-channel-priority"]
    if dry_run:
        argv += ["--dry-run", "--json"]
    return argv + specs


def _create_env(prefix: Path, micromamba: bool) -> dict[str, str]:
    env = dict(os.environ)
    # Mirrors what FSL's installer sets; the wrappers themselves are made afterwards by
    # finalize_prefix, since no package ships the createFSLWrapper the post-link scripts call.
    env["FSLDIR"] = str(prefix)
    if micromamba:
        env.setdefault("MAMBA_ROOT_PREFIX", str(external_tools.managed_dir("micromamba")))
    return env


#: Stand-in for FSL's ``createFSLWrapper``. FSL's installer (not a package) puts that script in
#: ``$FSLDIR/share/fsl/sbin`` before creating the environment; every FSL package's post-link
#: script calls it to write one wrapper per command into ``$FSLDIR/share/fsl/bin``, the
#: directory that goes on PATH without the environment's python, curl, tar … A prefix created
#: by plain conda/mamba has neither, so the wrappers are made here, the same way: ``target`` or
#: ``target=name`` per argument, ``<Command>_gui`` meaning the Linux TCL GUI ``<Command>``,
#: Python scripts run isolated (``python -I``).
_CREATE_WRAPPER = r"""#!/usr/bin/env bash
# nvitk stand-in for FSL's createFSLWrapper (see nvitk.fsl.install).
[ -n "$FSLDIR" ] && [ -n "$PREFIX" ] && [ -n "$FSL_CREATE_WRAPPER_SCRIPTS" ] || exit 0
dest="$FSLDIR/share/fsl/bin"
mkdir -p "$dest"
for spec in "$@"; do
  target="${spec%%=*}"
  name="${spec#*=}"
  case "$target" in *_gui) target="${target%_gui}" ;; esac
  case "$name" in *_gui) name="${name%_gui}" ;; esac
  src="$PREFIX/bin/$target"
  [ -e "$src" ] || continue
  if head -n 1 "$src" 2>/dev/null | grep -q python; then
    printf '#!/usr/bin/env bash\n"%s/bin/python" -I "%s" "$@"\n' "$PREFIX" "$src" > "$dest/$name"
  else
    printf '#!/usr/bin/env bash\n"%s" "$@"\n' "$src" > "$dest/$name"
  fi
  chmod 755 "$dest/$name"
done
"""


def finalize_prefix(prefix: Path, version: str = FSL_VERSION, on_line: Callable[[str], None] | None = None) -> int:
    """Make a conda-created FSL prefix look like an installed FSL: command wrappers in
    ``share/fsl/bin`` and ``etc/fslversion``. Returns the number of wrappers."""
    prefix = Path(prefix)
    sbin = prefix / "share" / "fsl" / "sbin"
    sbin.mkdir(parents=True, exist_ok=True)
    creator = sbin / "createFSLWrapper"
    if not creator.is_file() or "nvitk stand-in" in creator.read_text(errors="replace"):
        creator.write_text(_CREATE_WRAPPER)
        creator.chmod(0o755)
    env = dict(os.environ, FSLDIR=str(prefix), PREFIX=str(prefix), FSL_CREATE_WRAPPER_SCRIPTS="1")
    for script in sorted((prefix / "bin").glob(".fsl-*-post-link.sh")):
        subprocess.run(["bash", str(script)], env=env, cwd=str(prefix), capture_output=True, text=True)
    (prefix / "etc").mkdir(exist_ok=True)
    (prefix / "etc" / "fslversion").write_text(f"{version}\n")
    count = len(list((prefix / "share" / "fsl" / "bin").iterdir()))
    (on_line or print)(f"FSL command wrappers: {count} in {prefix / 'share' / 'fsl' / 'bin'}")
    return count


def estimate_download(
    components: Iterable[str] = ALL_COMPONENTS, *, cuda: bool = False, version: str = FSL_VERSION
) -> tuple[float, int]:
    """``(GB to download, packages)`` from a dry-run solve (``(nan, 0)`` when it cannot tell)."""
    specs, _missing = package_specs(components, cuda=cuda, version=version)
    conda, micro = conda_executable(download=False)
    prefix = managed_fsldir()
    proc = subprocess.run(
        create_argv(conda, prefix, specs, micromamba=micro, dry_run=True),
        capture_output=True, text=True, env=_create_env(prefix, micro),
    )
    try:
        data = json.loads(proc.stdout)
        fetch = data.get("actions", {}).get("FETCH", []) or []
        link = data.get("actions", {}).get("LINK", []) or []
        return sum(float(p.get("size", 0) or 0) for p in fetch) / 1e9, len(link)
    except Exception:  # noqa: BLE001
        return float("nan"), 0


def install_fsl(
    components: Iterable[str] = ALL_COMPONENTS,
    *,
    cuda: bool | None = None,
    version: str = FSL_VERSION,
    prefix: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Create (or recreate) nvitk's FSL core and return its ``FSLDIR``.

    *cuda* defaults to "an NVIDIA driver is present" (``nvidia-smi`` on PATH).
    """
    say = on_line or (lambda text: print(text, flush=True))
    components = list(components)
    if cuda is None:
        cuda = shutil.which("nvidia-smi") is not None
    prefix = Path(prefix) if prefix else managed_fsldir()
    specs, missing = package_specs(components, cuda=cuda, version=version)
    if missing:
        say(f"Not in FSL {version}, skipped: {', '.join(missing)}")
    conda, micro = conda_executable(on_line=say)
    if prefix.exists():
        say(f"Replacing the existing FSL core in {prefix} ...")
        shutil.rmtree(prefix)
    say(f"Creating FSL {version} ({', '.join(components)}{', CUDA' if cuda else ''}) in {prefix} with {conda} ...")
    argv = create_argv(conda, prefix, specs, micromamba=micro)
    proc = subprocess.Popen(argv, env=_create_env(prefix, micro), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            say(line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Creating the FSL prefix failed ({Path(conda).name} exit {code}).")
    finalize_prefix(prefix, version, on_line=say)
    from .env import fsl_version

    say(f"FSL {fsl_version(prefix) or version} ready: FSLDIR={prefix}")
    if os.environ.get("FSLDIR") and Path(os.environ["FSLDIR"]) != prefix:
        say(
            f"Note: $FSLDIR={os.environ['FSLDIR']} still takes precedence; use `nvitk-fsl use {prefix}` "
            "or NVITK_FSLDIR to make nvitk use this one."
        )
    return prefix


__all__ = [
    "ALL_COMPONENTS",
    "COMPONENTS",
    "COMPONENT_REQUIRES",
    "CUDA_PACKAGES",
    "FSL_CHANNEL",
    "FSL_VERSION",
    "conda_executable",
    "create_argv",
    "estimate_download",
    "finalize_prefix",
    "install_fsl",
    "package_specs",
    "release_environment",
]
