"""Regenerate ``nvitk_vendor/`` from an nvitk source tree.

The Slicer module ships a self-contained copy of the morphometrics pipeline so
it runs without the nvitk repo (and without nvitk's install dependencies, which
cannot go into Slicer's Python). Keeping that copy honest is what this script is
for: it treats the vendored algorithm modules as **generated code**, so a drift
from upstream is one command away from being fixed rather than a fork that
quietly rots.

The transformation is deliberately minimal — a single token rename of the root
package, ``nvitk`` → ``nvitk_vendor`` — so every vendored file stays diffable
against upstream and produces identical numbers. Relative imports are untouched.

Hand-written files (``core/``, ``morphology/binary.py``,
``morphology/components.py``, every ``__init__.py``) are never overwritten.

Usage::

    python vendor_sync.py                      # find nvitk src automatically
    python vendor_sync.py --src /path/to/src   # explicit source tree
    python vendor_sync.py --check              # exit 1 if the copy is stale
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VENDOR_ROOT_NAME = "nvitk_vendor"

#: Modules copied verbatim (source path relative to ``src/``), with the root
#: package renamed. Everything the morphometrics entry point reaches eagerly,
#: plus the two lazily-imported ``morphology`` helpers.
SYNCED_MODULES: tuple[str, ...] = (
    "nvitk/measure/morphometrics.py",
    "nvitk/measure/morphometrics_config.py",
    "nvitk/morphology/centerline.py",
    "nvitk/morphology/mst_bridge.py",
    "nvitk/morphology/polyline_graph.py",
)

#: Whole directories copied verbatim (``*.py`` plus the listed data suffixes).
SYNCED_TREES: tuple[str, ...] = ("nvitk/measure/morpho",)
DATA_SUFFIXES: tuple[str, ...] = (".json",)

#: Never overwritten — the NumPy-only stand-ins and package markers.
HAND_WRITTEN: frozenset[str] = frozenset(
    {
        "__init__.py",
        "core/__init__.py",
        "core/array.py",
        "core/backend.py",
        "core/exceptions.py",
        "core/logger.py",
        "morphology/__init__.py",
        "morphology/binary.py",
        "morphology/components.py",
        "measure/__init__.py",
    }
)

_IMPORT_RE = re.compile(r"\bnvitk\b(?=\.|\s|$)")

_GENERATED_BANNER = (
    "# ─────────────────────────────────────────────────────────────────────────\n"
    "# VENDORED FROM nvitk — DO NOT EDIT.\n"
    "# Source: {source}\n"
    "# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py\n"
    "# The only change from upstream is the root package rename nvitk -> {root}.\n"
    "# ─────────────────────────────────────────────────────────────────────────\n"
)


def vendor_dir() -> Path:
    """``…/MouseTOFMorphometricsLib/nvitk_vendor``."""
    return Path(__file__).resolve().parent / VENDOR_ROOT_NAME


def default_src_dir() -> Path:
    """``<repo>/src`` — this file lives at ``<repo>/slicer/<Module>/<ModuleLib>/``."""
    return Path(__file__).resolve().parents[3] / "src"


def rewrite(text: str, *, source: str) -> str:
    """Rename the root package and prepend the generated-file banner."""
    body = _IMPORT_RE.sub(VENDOR_ROOT_NAME, text)
    return _GENERATED_BANNER.format(source=source, root=VENDOR_ROOT_NAME) + body


def _relative_targets(src: Path) -> list[tuple[Path, str]]:
    """``(source_file, target_relative_path)`` for everything that gets synced."""
    out: list[tuple[Path, str]] = []
    for rel in SYNCED_MODULES:
        out.append((src / rel, rel.split("/", 1)[1]))
    for tree in SYNCED_TREES:
        base = src / tree
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix != ".py" and path.suffix not in DATA_SUFFIXES:
                continue
            out.append((path, str(Path(tree.split("/", 1)[1]) / path.relative_to(base))))
    return out


def _source_revision(src: Path) -> str:
    """Short git description of the source tree, or ``"unknown"``."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(src), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if rev.returncode == 0:
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return rev.stdout.strip() + suffix
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def sync(src: Path, *, check_only: bool = False) -> int:
    """Copy/rewrite everything; return the number of files written (or stale, in check mode)."""
    if not (src / "nvitk" / "measure" / "morpho").is_dir():
        raise SystemExit(f"nvitk source tree not found under {src} (expected nvitk/measure/morpho).")

    root = vendor_dir()
    root.mkdir(parents=True, exist_ok=True)
    targets = _relative_targets(src)

    written: list[str] = []
    manifest: list[tuple[str, str]] = []
    for source_file, rel in targets:
        if rel in HAND_WRITTEN:
            raise SystemExit(f"Refusing to overwrite hand-written file: {rel}")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if source_file.suffix == ".py":
            content = rewrite(source_file.read_text(encoding="utf-8"), source=f"src/{source_file.relative_to(src)}")
            new_bytes = content.encode("utf-8")
        else:
            new_bytes = source_file.read_bytes()

        manifest.append((rel, hashlib.sha256(new_bytes).hexdigest()[:16]))
        if not target.is_file() or target.read_bytes() != new_bytes:
            written.append(rel)
            if not check_only:
                target.write_bytes(new_bytes)

    # Package markers for generated subpackages that have no hand-written __init__.
    # Data-only directories (topology JSONs) are left as plain folders.
    code_dirs = {str(Path(rel).parent) for _s, rel in targets if rel.endswith(".py")}
    for pkg in sorted(d for d in code_dirs if d != "."):
        init = root / pkg / "__init__.py"
        if f"{pkg}/__init__.py" in HAND_WRITTEN or init.is_file():
            continue
        if not check_only:
            init.write_text(f'"""Vendored ``nvitk.{pkg.replace("/", ".")}`` (generated)."""\n', encoding="utf-8")
        written.append(f"{pkg}/__init__.py")

    if not check_only:
        _write_provenance(root, src, manifest)

    return len(written)


def _write_provenance(root: Path, src: Path, manifest: list[tuple[str, str]]) -> None:
    lines = [
        "# Vendored nvitk provenance",
        "",
        "Generated by `MouseTOFMorphometricsLib/vendor_sync.py` — **do not edit the",
        "files under `nvitk_vendor/` by hand** (except the stand-ins listed below).",
        "",
        f"- Source tree: `{src}`",
        f"- Source revision: `{_source_revision(src)}`",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Transformation: root package rename `nvitk` → `{VENDOR_ROOT_NAME}` (nothing else)",
        "",
        "## Refresh",
        "",
        "```bash",
        "python MouseTOFMorphometricsLib/vendor_sync.py",
        "```",
        "",
        "`--check` exits non-zero when the copy has drifted from the source tree.",
        "",
        "## Hand-written stand-ins (never synced)",
        "",
        "NumPy-only replacements for nvitk's CuPy backend / Rich logger / Image type:",
        "",
    ]
    lines += [f"- `{name}`" for name in sorted(HAND_WRITTEN)]
    lines += ["", f"## Synced files ({len(manifest)})", "", "| file | sha256:16 |", "|---|---|"]
    lines += [f"| `{rel}` | `{digest}` |" for rel, digest in sorted(manifest)]
    (root / "VENDORED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=None, help="nvitk 'src' directory (default: sibling repo)")
    parser.add_argument("--check", action="store_true", help="report drift without writing")
    args = parser.parse_args(argv)

    src = (args.src or default_src_dir()).expanduser().resolve()
    count = sync(src, check_only=args.check)

    if args.check:
        if count:
            print(f"vendored copy is STALE: {count} file(s) differ from {src}")
            return 1
        print(f"vendored copy is up to date with {src}")
        return 0

    print(f"synced {VENDOR_ROOT_NAME} from {src}: {count} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
