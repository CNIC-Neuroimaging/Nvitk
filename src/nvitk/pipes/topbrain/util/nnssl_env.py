"""Make the vendored nnssl clone importable and point it at our data roots.

Description
-----------
``src/nvitk/pipes/topbrain/nnssl`` is a vendored clone of `MIC-DKFZ/nnssl
<https://github.com/MIC-DKFZ/nnssl>`_ (``openneuro`` branch). It ships **without** a
``pyproject.toml``, so it cannot be ``pip install``-ed, and ``setuptools.packages.find`` in
this repo only includes ``nvitk*`` — the clone would never make it into a wheel anyway. It is
therefore used off ``sys.path`` / ``PYTHONPATH`` rather than installed. Verified working on
Python 3.11 despite the clone's ``requires-python >= 3.12`` claim.

Import-order constraint
-----------------------
``nnssl/paths.py`` reads its three environment variables **eagerly at import**::

    nnssl_raw = os.environ.get("nnssl_raw")

and most nnssl modules then do ``from nnssl.paths import nnssl_preprocessed``, binding the
value into their own namespace. Setting the environment after nnssl has been imported is
therefore not enough. :func:`apply_nnssl_env` must run **before** the first nnssl import; it
raises if that has already happened with a different configuration, rather than letting the
run silently write to the wrong directory.

``nnssl/paths.py`` also lets ``rocket_preprocessed`` silently override ``nnssl_preprocessed``
— a DKFZ-internal leftover. :func:`apply_nnssl_env` clears it.

Python version
--------------
The clone declares ``requires-python >= 3.12`` and five of its trainer modules import
``typing.override``, which only exists from 3.12. Everything else in it runs fine on 3.11, and
nnssl's trainer lookup imports *every* module under ``nnsslTrainer/`` — so on 3.11 that single
import breaks trainer discovery entirely. :func:`install_typing_override_shim` backfills the
symbol from ``typing_extensions`` (which is exactly the backport it exists for) rather than
forcing the whole toolkit onto 3.12.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain.util.paths import TopBrainPaths

log = Logger()

#: Environment variables nnssl reads for its three data roots.
NNSSL_ENV_KEYS: tuple[str, ...] = ("nnssl_raw", "nnssl_preprocessed", "nnssl_results")

#: DKFZ-internal variable that silently overrides ``nnssl_preprocessed`` if left set.
_ROCKET_OVERRIDE = "rocket_preprocessed"


def nnssl_root() -> Path:
    """Directory of the vendored nnssl clone (the one holding ``src/nnssl``)."""
    return Path(__file__).resolve().parents[1] / "nnssl"


def nnssl_src_dir() -> Path:
    """The clone's ``src`` directory — what must be on ``sys.path``/``PYTHONPATH``.

    Raises
    ------
    FileNotFoundError
        If the vendored clone is missing or incomplete, naming the expected path. The SSL
        stages are unusable without it, and a bare ``ModuleNotFoundError: nnssl`` several
        frames deep is far harder to act on.
    """
    src = nnssl_root() / "src"
    if not (src / "nnssl" / "paths.py").is_file():
        raise FileNotFoundError(
            f"Vendored nnssl clone not found under {src}. Clone "
            f"https://github.com/MIC-DKFZ/nnssl (branch 'openneuro') into {nnssl_root()}."
        )
    return src


def install_typing_override_shim() -> bool:
    """Backfill ``typing.override`` on Python < 3.12; returns whether a shim was installed.

    ``typing.override`` (:pep:`698`) landed in 3.12. It is a pure no-op decorator that only
    tags the function with ``__override__`` for static checkers, so the backport is exactly
    equivalent at runtime. Five nnssl trainer modules import it, and because nnssl discovers
    trainers by importing every module in its trainer package, one missing symbol makes *all*
    of them unfindable.
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


def nnssl_env(paths: TopBrainPaths, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables an nnssl subprocess needs, as a plain dict.

    Includes ``PYTHONPATH`` with the clone's ``src`` prepended to whatever is already set, so
    a worker launched with these keeps access to the installed ``nvitk``.
    """
    src = str(nnssl_src_dir())
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([src, existing]) if existing else src
    env = {
        "nnssl_raw": str(paths.nnssl_raw),
        "nnssl_preprocessed": str(paths.nnssl_preprocessed),
        "nnssl_results": str(paths.nnssl_results),
        "PYTHONPATH": pythonpath,
    }
    if extra:
        env.update(extra)
    return env


def apply_nnssl_env(paths: TopBrainPaths, *, create: bool = True) -> None:
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
        If nnssl was already imported against a different configuration. Continuing would
        write preprocessed data and checkpoints to the previously-bound directories while the
        rest of the pipeline looked for them here.
    """
    wanted = {
        "nnssl_raw": str(paths.nnssl_raw),
        "nnssl_preprocessed": str(paths.nnssl_preprocessed),
        "nnssl_results": str(paths.nnssl_results),
    }

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
    existing = os.environ.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([src, existing]) if existing else src

    if create:
        paths.ensure_dirs(paths.nnssl_raw, paths.nnssl_preprocessed, paths.nnssl_results)

    log.debug("nnssl env: %s", wanted)


__all__ = [
    "NNSSL_ENV_KEYS",
    "apply_nnssl_env",
    "install_typing_override_shim",
    "nnssl_env",
    "nnssl_root",
    "nnssl_src_dir",
]
