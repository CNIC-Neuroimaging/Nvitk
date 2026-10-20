"""Download FastSurfer's network checkpoints into a FastSurfer tree, robustly.

FastSurfer ships its own ``FastSurferCNN/download_checkpoints.py``, but it reads
with no timeout (``requests.get(..., timeout=(5, None))``): a mirror that accepts
the connection and then stalls — b2share.fz-juelich.de does this — hangs the
install for good. This reads the same per-module ``config/checkpoint_paths.yaml``
(the URLs and file names FastSurfer itself uses), tries the Zenodo mirror first,
gives up on a stalled transfer, and writes each file atomically. FastSurfer then
finds every checkpoint present and downloads nothing at run time.

Standalone on purpose (``requests`` and ``pyyaml`` only): it runs from
``install_fastsurfer.sh`` before nvitk itself is importable.

Usage: python download_checkpoints.py <FastSurfer tree> [module ...]
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

MODULES = ("FastSurferCNN", "CerebNet", "HypVINN", "CorpusCallosum")
#: (connect, between-bytes) seconds: a mirror silent for this long is abandoned.
TIMEOUT = (15, 60)


def _wanted(tree: Path, modules: tuple[str, ...]) -> list[tuple[Path, list[str]]]:
    """``(destination, urls)`` for every checkpoint of *modules* not yet on disk."""
    out: list[tuple[Path, list[str]]] = []
    for module in modules:
        config = tree / module / "config" / "checkpoint_paths.yaml"
        if not config.is_file():
            continue
        data = yaml.safe_load(config.read_text()) or {}
        urls = [str(u) for u in data.get("url") or []]
        # Zenodo first: it serves; the other mirror stalls mid-transfer.
        urls.sort(key=lambda u: 0 if "zenodo" in u else 1)
        for rel in (data.get("checkpoint") or {}).values():
            dest = Path(rel) if Path(rel).is_absolute() else tree / rel
            if not dest.is_file() or dest.stat().st_size == 0:
                out.append((dest, urls))
    return out


def _fetch(dest: Path, urls: list[str]) -> str | None:
    """Download *dest* from the first URL that delivers it; an error message, or None."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors: list[str] = []
    for url in urls:
        try:
            with requests.get(f"{url}/{dest.name}", stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                with part.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            if part.stat().st_size == 0:
                raise OSError("empty download")
            part.replace(dest)
            print(f"  {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB from {url.split('/')[2]})", flush=True)
            return None
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            part.unlink(missing_ok=True)
    return f"{dest.name}: " + " | ".join(errors)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    tree = Path(argv[0]).resolve()
    modules = tuple(argv[1:]) or MODULES
    todo = _wanted(tree, modules)
    if not todo:
        print("all FastSurfer checkpoints present")
        return 0
    print(f"downloading {len(todo)} FastSurfer checkpoint(s)...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        failures = [msg for msg in pool.map(lambda item: _fetch(*item), todo) if msg]
    for msg in failures:
        print(f"FAILED {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
