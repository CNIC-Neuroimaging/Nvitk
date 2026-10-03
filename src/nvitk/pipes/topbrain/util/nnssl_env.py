"""Make the vendored nnssl clone importable and point it at the topbrain data roots.

Description
-----------
Thin adapter over :mod:`nvitk.pipes._engines.env`, which documents the import-order constraint
(nnssl binds its roots **at import**), the ``rocket_preprocessed`` override and the
``typing.override`` shim for Python < 3.12. This module only unpacks
:class:`~nvitk.pipes.topbrain.util.paths.TopBrainPaths`.
"""

from __future__ import annotations

from nvitk.pipes._engines import env as _engine_env
from nvitk.pipes._engines.env import (
    NNSSL_ENV_KEYS,
    install_typing_override_shim,
    nnssl_root,
    nnssl_src_dir,
)
from nvitk.pipes.topbrain.util.paths import TopBrainPaths


def nnssl_env(paths: TopBrainPaths, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables an nnssl subprocess needs, as a plain dict."""
    return _engine_env.nnssl_env(
        paths.nnssl_raw, paths.nnssl_preprocessed, paths.nnssl_results, extra=extra
    )


def apply_nnssl_env(paths: TopBrainPaths, *, create: bool = True) -> None:
    """Export the nnssl roots into this process and put the clone on ``sys.path``.

    Must be called before the first ``import nnssl``; see
    :func:`nvitk.pipes._engines.env.apply_nnssl_env`.
    """
    _engine_env.apply_nnssl_env(
        paths.nnssl_raw, paths.nnssl_preprocessed, paths.nnssl_results, create=create
    )


__all__ = [
    "NNSSL_ENV_KEYS",
    "apply_nnssl_env",
    "install_typing_override_shim",
    "nnssl_env",
    "nnssl_root",
    "nnssl_src_dir",
]
