"""Installing FastSurfer (into this environment) and, on request, FreeSurfer.

FastSurfer's tree and checkpoints are installed by ``install_fastsurfer.sh`` — the
script the conda package's post-link step and ``install.sh`` run — so this module
only finds it and runs it.

FreeSurfer is never installed implicitly: it is a ≈9.5 GB download (≈20 GB unpacked)
under its own license, and only the surface and longitudinal pipelines need it.
:func:`install_freesurfer` downloads the release FastSurfer's ``recon-surf`` is
validated against, resumably, unpacks it outside the environment and remembers where.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from nvitk.core import config_paths, external_tools

from . import config

FREESURFER_BASE_URL = "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer"
#: Free space wanted on the destination's filesystem: the archive and its contents
#: coexist until the archive is removed.
FREESURFER_SPACE_GB = 32.0


def _script(name: str) -> Path:
    """A recipe script: shipped into the environment by the conda package, else in the checkout."""
    candidates = [external_tools.managed_dir("fastsurfer-install") / name]
    root = config_paths.source_checkout_root()
    if root is not None:
        candidates.append(root / "recipe" / "fastsurfer" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{name} was not found (looked in {', '.join(str(c) for c in candidates)}). "
        "Reinstall nvitk, or run it from a source checkout."
    )


def install_fastsurfer(
    *, version: str | None = None, checkpoints: bool = True, on_line: Callable[[str], None] | None = None
) -> int:
    """Install (or update) FastSurfer into this environment; returns the exit code."""
    env = dict(os.environ, PREFIX=sys.prefix)
    if version:
        env["NVITK_FASTSURFER_VERSION"] = version
    if not checkpoints:
        env["NVITK_FASTSURFER_CHECKPOINTS"] = "0"
    return _stream(["bash", str(_script("install_fastsurfer.sh"))], env, on_line)


def download_checkpoints(*, on_line: Callable[[str], None] | None = None) -> int:
    """Fetch the network weights FastSurfer's tree is missing; returns the exit code."""
    home = config.fastsurfer_home()
    if home is None:
        raise RuntimeError("FastSurfer is not installed; run `nvitk-fastsurfer install` first.")
    return _stream([sys.executable, "-s", str(_script("download_checkpoints.py")), str(home)], None, on_line)


def _stream(argv: list[str], env: dict[str, str] | None, on_line: Callable[[str], None] | None) -> int:
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        (on_line or (lambda text: print(text, flush=True)))(line.rstrip("\n"))
    return int(proc.wait())


# ── FreeSurfer (opt-in) ──────────────────────────────────────────────────────


def freesurfer_build() -> str:
    """The FreeSurfer build for this Linux: ``ubuntu22_amd64``, ``ubuntu20_amd64``, ``centos8_x86_64`` …"""
    info: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            key, _, value = line.partition("=")
            info[key] = value.strip().strip('"')
    except OSError:
        pass
    distro = info.get("ID", "").lower()
    like = info.get("ID_LIKE", "").lower()
    try:
        major = int(info.get("VERSION_ID", "0").split(".")[0])
    except ValueError:
        major = 0
    if distro == "ubuntu" or "ubuntu" in like or distro == "debian":
        if distro == "ubuntu" and major and major < 20:
            return "ubuntu18_amd64"
        if distro == "ubuntu" and major == 20:
            return "ubuntu20_amd64"
        return "ubuntu22_amd64"
    if distro in {"centos", "rhel", "rocky", "almalinux", "fedora"} or "rhel" in like:
        return "centos7_x86_64" if major == 7 else "centos8_x86_64"
    return "ubuntu22_amd64"


def freesurfer_url(version: str = config.FREESURFER_VERSION, build: str | None = None) -> str:
    build = build or freesurfer_build()
    return f"{FREESURFER_BASE_URL}/{version}/freesurfer-linux-{build}-{version}.tar.gz"


def remote_size(url: str) -> int | None:
    """The archive's size from the server, in bytes (``None`` when it does not say)."""
    try:
        import requests

        response = requests.head(url, allow_redirects=True, timeout=30)
        size = int(response.headers.get("Content-Length", "0"))
        return size or None
    except Exception:  # noqa: BLE001
        return None


def free_gb(path: Path) -> float:
    """Free space on *path*'s filesystem (its nearest existing ancestor), in GB."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def freesurfer_download_argv(url: str, archive: Path) -> list[str]:
    """``curl`` resuming *archive* (``-C -``) and giving up on a stalled server."""
    return [
        "curl", "-fL", "--retry", "5", "--retry-delay", "10", "-C", "-",
        "--speed-limit", "20000", "--speed-time", "120", "-o", str(archive), url,
    ]


def install_freesurfer(
    dest: str | Path | None = None,
    *,
    license_file: str | Path | None = None,
    url: str | None = None,
    keep_archive: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Download and unpack FreeSurfer :data:`~config.FREESURFER_VERSION`; returns ``FREESURFER_HOME``.

    The archive is downloaded into *dest* (resumed when a partial one is there),
    checked against the server's size, unpacked (it holds a ``freesurfer/`` folder),
    and removed unless *keep_archive*. The location — and *license_file*, copied in as
    ``license.txt`` — are remembered in ``external_tools.json``.
    """
    say = on_line or (lambda text: print(text, flush=True))
    dest = Path(dest).expanduser() if dest else config.default_freesurfer_dest()
    url = url or freesurfer_url()
    home = dest / "freesurfer"
    if (home / "SetUpFreeSurfer.sh").is_file():
        say(f"FreeSurfer is already unpacked in {home}.")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / Path(url).name
        size = remote_size(url)
        have = archive.stat().st_size if archive.is_file() else 0
        needed = FREESURFER_SPACE_GB - have / 1e9
        if free_gb(dest) < needed:
            raise RuntimeError(
                f"Not enough free space in {dest}: {free_gb(dest):.0f} GB free, about {needed:.0f} GB needed."
            )
        if size is None or have != size:
            say(f"Downloading {url}" + (f" ({size / 1e9:.1f} GB)" if size else "") + f" to {archive} ...")
            code = subprocess.call(freesurfer_download_argv(url, archive))
            if code != 0:
                raise RuntimeError(f"The FreeSurfer download stopped (curl exit {code}); run again to resume.")
            if size is not None and archive.stat().st_size != size:
                raise RuntimeError(
                    f"The FreeSurfer archive is {archive.stat().st_size} bytes, the server says {size}; "
                    "run again to resume."
                )
        say(f"Unpacking {archive.name} into {dest} (this takes a while) ...")
        code = subprocess.call(["tar", "-xzf", str(archive), "-C", str(dest)])
        if code != 0 or not (home / "SetUpFreeSurfer.sh").is_file():
            raise RuntimeError(f"Unpacking {archive} failed (tar exit {code}).")
        if not keep_archive:
            archive.unlink(missing_ok=True)
    if license_file:
        target = home / "license.txt"
        shutil.copyfile(Path(license_file).expanduser(), target)
        say(f"License copied to {target}.")
    external_tools.remember(freesurfer_home=str(home))
    if license_file:
        external_tools.remember(fs_license=str(home / "license.txt"))
    say(f"FreeSurfer {config.freesurfer_version(home) or ''} ready: FREESURFER_HOME={home}")
    if config.fs_license(home) is None:
        say(
            "No license yet: register (free) at https://surfer.nmr.mgh.harvard.edu/registration.html, "
            f"then run `nvitk-fastsurfer set-license <file>` or copy it to {home / 'license.txt'}."
        )
    return home


__all__ = [
    "download_checkpoints",
    "free_gb",
    "freesurfer_build",
    "freesurfer_download_argv",
    "freesurfer_url",
    "install_fastsurfer",
    "install_freesurfer",
    "remote_size",
]
