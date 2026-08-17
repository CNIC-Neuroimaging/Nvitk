"""NumPy-only replacement for ``nvitk.core.backend`` (hand-written; not synced).

Upstream this module selects between NumPy and CuPy and rewrites each consuming
module's ``np`` / ``scipy`` / ``ndi`` globals when the backend changes. Inside
Slicer there is only ever NumPy, so :func:`setup` performs the same global
injection once and :func:`using` is a no-op context manager.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as _np
import scipy as _scipy
from scipy import ndimage as _ndi

BackendName = str


def get_current_backend() -> str:
    """Always ``"numpy"``: the vendored build has no CuPy path."""
    return "numpy"


def get_ops(name: str | None = None, allow_fallback: bool = True) -> tuple[Any, Any, Any]:
    """The ``(xp, scipy, ndi)`` triple — always the NumPy/SciPy one."""
    return _np, _scipy, _ndi


class _NullProxy:
    """Stands in for ``nvitk.core.proxy.BackendProxy``; nothing can change here."""

    np = _np
    scipy = _scipy
    ndi = _ndi

    def bind_globals(self, module_globals: dict[str, Any]) -> None:
        module_globals["np"] = _np
        module_globals["scipy"] = _scipy
        module_globals["ndi"] = _ndi

    def invalidate(self) -> None:
        return None

    def refresh_globals(self) -> None:
        return None


def setup(module_globals: dict[str, Any], module_name: str | None = None) -> _NullProxy:
    """Inject ``np`` / ``scipy`` / ``ndi`` into *module_globals* (call with ``globals()``)."""
    proxy = _NullProxy()
    proxy.bind_globals(module_globals)
    return proxy


def register_module_for_backend_updates(module_name: str, module_globals: dict | None = None) -> None:
    """No-op: the backend can never change here."""
    return None


@contextlib.contextmanager
def using(*args: Any, **kwargs: Any) -> Iterator[str]:
    """No-op stand-in for nvitk's backend-switching context manager."""
    yield "numpy"


using_backend = using


def set_global_backend(*args: Any, **kwargs: Any) -> str:
    """No-op; reports the only available backend."""
    return "numpy"


__all__ = [
    "BackendName",
    "get_current_backend",
    "get_ops",
    "register_module_for_backend_updates",
    "set_global_backend",
    "setup",
    "using",
    "using_backend",
]
