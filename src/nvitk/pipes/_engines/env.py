"""
Locate the vendored nnU-Net / nnssl trees and build the environment their processes need.

Description
-----------
Pipeline-agnostic: every function takes the three data roots explicitly, so each pipeline keeps
its own path layout object (``TopBrainPaths``, ``CViTPaths``) and only adapts it here.

``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
    Read lazily through ``nnunetv2.paths``, so they only have to be right in the subprocess
    environment built by :func:`nnunet_env`.

``nnssl_raw`` / ``nnssl_preprocessed`` / ``nnssl_results``
    ``nnssl/paths.py`` reads them **eagerly at import** and most nnssl modules then bind the
    values into their own namespace. :func:`apply_nnssl_env` must therefore run **before** the
    first ``import nnssl``; it raises if nnssl was already imported against other roots rather
    than let a run silently write to the wrong directory. It also clears the DKFZ-internal
    ``rocket_preprocessed`` override.

Python version
--------------
nnssl declares ``requires-python >= 3.12`` and five of its trainer modules import
``typing.override``. Its trainer lookup imports *every* module under ``nnsslTrainer/``, so on
3.11 that one symbol breaks discovery entirely. :func:`install_typing_override_shim` backfills
it from ``typing_extensions`` (an exact runtime equivalent).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nvitk.core.logger import Logger

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Environment variables nnU-Net reads for its three data roots.
NNUNET_ENV_KEYS: tuple[str, ...] = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")

#: Environment variables nnssl reads for its three data roots.
NNSSL_ENV_KEYS: tuple[str, ...] = ("nnssl_raw", "nnssl_preprocessed", "nnssl_results")

#: DKFZ-internal variable that silently overrides ``nnssl_preprocessed`` if left set.
_ROCKET_OVERRIDE = "rocket_preprocessed"

#: Directory holding both vendored trees.
ENGINES_DIR: Path = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# Locating the trees
# ──────────────────────────────────────────────────────────────────────────────

def nnunet_root() -> Path:
    """The in-tree nnU-Net build (the directory holding ``nnunetv2/``).

    Raises
    ------
    FileNotFoundError
        Naming the expected path. Training is unusable without it, and a bare
        ``ModuleNotFoundError`` several frames deep in a subprocess is far harder to act on.
    """
    root = ENGINES_DIR / "nnunet"
    if not (root / "nnunetv2" / "run" / "run_training_from_pretrained.py").is_file():
        raise FileNotFoundError(
            f"In-tree nnU-Net build not found under {root}. It must provide "
            f"nnunetv2/run/run_training_from_pretrained.py (the nnssl fine-tuning entry point)."
        )
    return root


def nnssl_root() -> Path:
    """Directory of the vendored nnssl clone (the one holding ``src/nnssl``)."""
    return ENGINES_DIR / "nnssl"


def nnssl_src_dir() -> Path:
    """The nnssl clone's ``src`` directory — what must be on ``sys.path``/``PYTHONPATH``.

    Raises
    ------
    FileNotFoundError
        If the clone is missing or incomplete, naming the expected path.
    """
    src = nnssl_root() / "src"
    if not (src / "nnssl" / "paths.py").is_file():
        raise FileNotFoundError(
            f"Vendored nnssl clone not found under {src}. Clone "
            f"https://github.com/MIC-DKFZ/nnssl (branch 'openneuro') into {nnssl_root()}."
        )
    return src


def nvitk_src_dir() -> Path:
    """The ``src`` directory holding the ``nvitk`` package itself.

    Trainers inside the in-tree nnU-Net import ``nvitk`` (losses, CViT architecture). An
    installed nvitk already satisfies that, but a source checkout run without ``pip install -e``
    does not, so :func:`nnunet_env` appends this to the subprocess ``PYTHONPATH``.
    """
    return ENGINES_DIR.parents[2]


def _prepend_pythonpath(*entries: str, fallback: tuple[str, ...] = ()) -> str:
    """``PYTHONPATH`` with *entries* first, the existing value next, *fallback* last.

    De-duplicated. *fallback* entries only matter when nothing earlier provides the package,
    so appending them never shadows an installed copy.
    """
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    out: list[str] = []
    for item in [*entries, *existing, *fallback]:
        if item not in out:
            out.append(item)
    return os.pathsep.join(out)


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net
# ──────────────────────────────────────────────────────────────────────────────

def nnunet_env(
    raw: Path | str,
    preprocessed: Path | str,
    results: Path | str,
    *,
    num_processes: int | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables an in-tree nnU-Net subprocess needs, as a plain dict.

    Parameters
    ----------
    raw, preprocessed, results
        The three nnU-Net data roots.
    num_processes
        Caps both nnU-Net's general worker pool (``nnUNet_def_n_proc``) and its augmentation
        pool (``nnUNet_n_proc_DA``). Left unset, nnU-Net sizes them from the host's core count,
        which oversubscribes an SGE slot allocation.
    extra
        Additional variables (e.g. trainer configuration such as ``TOPBRAIN_LOSS_SPEC``).
    """
    env = {
        # The in-tree build must win over the released nnunetv2 for this subprocess only.
        "PYTHONPATH": _prepend_pythonpath(
            str(nnunet_root()), fallback=(str(nvitk_src_dir()),)
        ),
        "nnUNet_raw": str(raw),
        "nnUNet_preprocessed": str(preprocessed),
        "nnUNet_results": str(results),
    }
    if num_processes is not None:
        env["nnUNet_def_n_proc"] = str(int(num_processes))
        env["nnUNet_n_proc_DA"] = str(int(num_processes))
    if extra:
        env.update(extra)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# nnssl
# ──────────────────────────────────────────────────────────────────────────────

def install_typing_override_shim() -> bool:
    """Backfill ``typing.override`` on Python < 3.12; returns whether a shim was installed.

    ``typing.override`` (:pep:`698`) is a no-op decorator at runtime that only tags the function
    for static checkers, so the backport is exactly equivalent.
    """
    import typing

    if hasattr(typing, "override"):
        return False
    try:
        from typing_extensions import override
    except ImportError:  # pragma: no cover - typing_extensions is an nnssl dependency
        def override(method):  # type: ignore[misc]
            """Minimal stand-in: tag the method and return it unchanged."""
            try:
                method.__override__ = True
            except (AttributeError, TypeError):
                pass
            return method

    typing.override = override  # type: ignore[attr-defined]
    log.debug("Installed typing.override shim for Python < 3.12 (required by nnssl).")
    return True


def nnssl_env(
    raw: Path | str,
    preprocessed: Path | str,
    results: Path | str,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables an nnssl subprocess needs, as a plain dict.

    ``PYTHONPATH`` gets the clone's ``src`` prepended, keeping access to the installed nvitk.
    """
    env = {
        "nnssl_raw": str(raw),
        "nnssl_preprocessed": str(preprocessed),
        "nnssl_results": str(results),
        "PYTHONPATH": _prepend_pythonpath(str(nnssl_src_dir())),
    }
    if extra:
        env.update(extra)
    return env


def apply_nnssl_env(
    raw: Path | str,
    preprocessed: Path | str,
    results: Path | str,
    *,
    create: bool = True,
) -> None:
    """Export the nnssl roots into this process and put the clone on ``sys.path``.

    Must be called before the first ``import nnssl`` — see the module docstring.

    Parameters
    ----------
    create
        ``mkdir -p`` the three roots. nnssl assumes they exist and fails deep inside a worker
        otherwise.

    Raises
    ------
    RuntimeError
        If nnssl was already imported against a different configuration.
    """
    wanted = {
        "nnssl_raw": str(raw),
        "nnssl_preprocessed": str(preprocessed),
        "nnssl_results": str(results),
    }

    # ---- 1. Refuse a stale binding ------------------------------------------------------
    if "nnssl.paths" in sys.modules:
        bound = sys.modules["nnssl.paths"]
        stale = {
            key: getattr(bound, key, None)
            for key in NNSSL_ENV_KEYS
            if getattr(bound, key, None) != wanted[key]
        }
        if stale:
            raise RuntimeError(
                "nnssl was imported before its environment was configured; it is bound to "
                f"{stale!r} but this run needs {wanted!r}. Call apply_nnssl_env() before the "
                "first nnssl import."
            )

    # ---- 2. Environment + import path ---------------------------------------------------
    if _ROCKET_OVERRIDE in os.environ:
        log.warning(
            "Unsetting %s=%r — it silently overrides nnssl_preprocessed.",
            _ROCKET_OVERRIDE,
            os.environ[_ROCKET_OVERRIDE],
        )
        os.environ.pop(_ROCKET_OVERRIDE)

    os.environ.update(wanted)
    install_typing_override_shim()

    src = str(nnssl_src_dir())
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ["PYTHONPATH"] = _prepend_pythonpath(src)

    if create:
        for root in (raw, preprocessed, results):
            Path(root).mkdir(parents=True, exist_ok=True)

    log.debug("nnssl env: %s", wanted)


__all__ = [
    "ENGINES_DIR",
    "NNSSL_ENV_KEYS",
    "NNUNET_ENV_KEYS",
    "apply_nnssl_env",
    "install_typing_override_shim",
    "nnssl_env",
    "nnssl_root",
    "nnssl_src_dir",
    "nnunet_env",
    "nnunet_root",
    "nvitk_src_dir",
]
