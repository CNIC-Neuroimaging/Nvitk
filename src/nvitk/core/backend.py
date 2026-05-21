"""
NumPy vs CuPy selection, global and context-scoped switching, and array conversion.

Environment variables:

- ``NVITK_BACKEND`` — ``auto``, ``numpy``, ``cupy``, ``cupy_required``, or aliases (``cpu``, ``gpu``, …).
- ``NVITK_CUDA_DEVICE`` — optional integer device index.
- ``NVITK_WARN_ON_FALLBACK`` — warn when falling back from CuPy to NumPy.

Use :class:`using` or :func:`using_backend` for temporary switches; :func:`set_global_backend`
for process-wide default. Modules can call :func:`setup` once to inject dynamic ``np``/``scipy``/``ndi``.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import numpy as np
import scipy
from scipy import ndimage as ndi

from .config import load_core_config
from .exceptions import BackendUnavailableError

try:
    import cupy as _cp
    import cupyx.scipy as _cp_scipy
    import cupyx.scipy.ndimage as _cp_ndi
except Exception:
    _cp = None
    _cp_scipy = None
    _cp_ndi = None


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

BackendName = Literal["numpy", "cupy"]


@dataclass(frozen=True)
class BackendModules:
    """Triple of ``(xp, scipy, ndi)`` for the resolved backend name."""

    xp: Any
    scipy: Any
    ndi: Any


# ──────────────────────────────────────────────────────────────────────────────
# CuPy availability
# ──────────────────────────────────────────────────────────────────────────────


def _cupy_runtime_available() -> bool:
    """True if CuPy is importable and reports at least one CUDA device."""
    if _cp is None:
        return False
    try:
        return int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def is_cupy_installed() -> bool:
    """True if the ``cupy`` package imported (does not guarantee a usable GPU)."""
    return _cp is not None


def is_gpu_available() -> bool:
    """True if CuPy is installed and a CUDA device is visible."""
    return _cupy_runtime_available()


def available_backends() -> tuple[BackendName, ...]:
    """Tuple of backend names usable on this machine (always includes ``numpy``)."""
    if is_gpu_available():
        return ("numpy", "cupy")
    return ("numpy",)


def _normalize_backend_name(name: str) -> BackendName:
    """Map user-facing strings to ``numpy`` or ``cupy``."""
    value = name.strip().lower()
    aliases = {
        "numpy": "numpy",
        "cpu": "numpy",
        "np": "numpy",
        "cupy": "cupy",
        "gpu": "cupy",
        "cp": "cupy",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise BackendUnavailableError(f"Unknown backend '{name}'. Expected one of numpy/cupy.")
    return normalized


def _resolve_backend(name: str, allow_fallback: bool = False) -> BackendName:
    """Resolve *name* to ``numpy`` or ``cupy``, optionally falling back to NumPy if no GPU."""
    target = _normalize_backend_name(name)
    if target == "cupy" and not is_gpu_available():
        if allow_fallback:
            return "numpy"
        raise BackendUnavailableError(
            "Requested backend 'cupy' is not available. "
            "Install CuPy and ensure a CUDA-capable device/driver is visible."
        )
    return target


def _initial_backend() -> BackendName:
    """Pick startup backend from :func:`~nvitk.core.config.load_core_config` and env."""
    cfg = load_core_config()

    env_backend = os.getenv("NVITK_BACKEND")
    if env_backend:
        try:
            return _resolve_backend(env_backend, allow_fallback=True)
        except Exception:
            pass

    pref = cfg.backend_preference
    if pref == "numpy":
        return "numpy"
    if pref in {"cupy", "cupy_required"}:
        return _resolve_backend("cupy", allow_fallback=(pref == "cupy"))
    return "cupy" if is_gpu_available() else "numpy"


_BACKENDS: dict[BackendName, tuple[Any, Any, Any]] = {"numpy": (np, scipy, ndi),}
if _cp is not None and _cp_scipy is not None and _cp_ndi is not None and _cupy_runtime_available():
    _BACKENDS["cupy"] = (_cp, _cp_scipy, _cp_ndi)

_global_backend: BackendName = _initial_backend()
_backend_context: ContextVar[BackendName] = ContextVar("nvitk_backend", default=_global_backend)
_backend_dependent_modules: set[str] = set()


def _update_backend_proxies() -> None:
    # Late import avoids circular dependency with proxy module.
    try:
        from .proxy import refresh_all_proxies

        refresh_all_proxies()
    except Exception:
        # Proxy system may not be initialized yet.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Global & context backend
# ──────────────────────────────────────────────────────────────────────────────


def get_global_backend() -> BackendName:
    """Process-wide default backend (updated by :func:`set_global_backend`)."""
    return _global_backend


def set_default_backend(name: str, allow_fallback: bool = True) -> BackendName:
    """Alias for :func:`set_global_backend` (default ``allow_fallback=True`` for CLI use)."""
    return set_global_backend(name, allow_fallback=allow_fallback)


def set_global_backend(name: str, allow_fallback: bool = False) -> BackendName:
    """
    Set the default backend for new code and refresh registered module proxies.

    Parameters
    ----------
    name
        ``numpy`` or ``cupy`` (aliases accepted).
    allow_fallback
        If True and CuPy is unavailable, use NumPy instead of raising.

    Returns
    -------
    BackendName
        The resolved backend actually in effect.
    """
    global _global_backend
    resolved = _resolve_backend(name, allow_fallback=allow_fallback)
    if resolved not in _BACKENDS:
        raise BackendUnavailableError(
            f"Backend '{resolved}' is not available. Available: {tuple(_BACKENDS.keys())}"
        )
    _global_backend = resolved
    _backend_context.set(resolved)
    _update_backend_proxies()
    return resolved


def get_current_backend() -> BackendName:
    """Active backend for this context (thread/async aware via :class:`contextvars.ContextVar`)."""
    return _backend_context.get()


def get_backend_modules(
    name: str | None = None,
    allow_fallback: bool = True,
) -> BackendModules:
    """
    Return :class:`BackendModules` for *name* or :func:`get_current_backend`.

    Parameters
    ----------
    name
        Optional explicit backend; default is current context.
    allow_fallback
        Passed to :func:`_resolve_backend` when *name* is given.
    """
    target = get_current_backend() if name is None else _resolve_backend(name, allow_fallback)
    mods = _BACKENDS[target]
    return BackendModules(xp=mods[0], scipy=mods[1], ndi=mods[2])


@contextlib.contextmanager
def using_backend(name: str, allow_fallback: bool = True) -> Iterator[BackendName]:
    """
    Context manager: temporarily set :func:`get_current_backend` to *name* (generator style).

    Prefer :class:`using` for the attribute-style API; this is the lower-level API.
    """
    resolved = _resolve_backend(name, allow_fallback=allow_fallback)
    if resolved not in _BACKENDS:
        raise BackendUnavailableError(
            f"Backend '{resolved}' is not available. Available: {tuple(_BACKENDS.keys())}"
        )
    token = _backend_context.set(resolved)
    _update_backend_proxies()
    try:
        yield resolved
    finally:
        _backend_context.reset(token)
        _update_backend_proxies()


class using:
    """
    Context manager for scoped backend switching (updates :class:`contextvars.ContextVar`).

    Examples
    --------
    .. code-block:: python

        with using("cupy"):
            x = np.array([1])  # CuPy array module
    """

    def __init__(self, backend_name: str, allow_fallback: bool = True) -> None:
        self.backend_name = backend_name
        self.allow_fallback = allow_fallback
        self._token: object | None = None

    def __enter__(self) -> "using":
        resolved = _resolve_backend(self.backend_name, allow_fallback=self.allow_fallback)
        if resolved not in _BACKENDS:
            raise BackendUnavailableError(
                f"Backend '{resolved}' is not available. Available: {tuple(_BACKENDS.keys())}"
            )
        self._token = _backend_context.set(resolved)
        _update_backend_proxies()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _backend_context.reset(self._token)
            _update_backend_proxies()

    def __repr__(self) -> str:
        return f"using('{self.backend_name}')"


def get_cupy_module() -> Any | None:
    """Return the ``cupy`` module or None if not installed."""
    return _cp


def get_array_module(array: Any) -> Any:
    """Return ``cupy`` or ``numpy`` depending on *array*'s type."""
    if _cp is not None and isinstance(array, _cp.ndarray):
        return _cp
    return np


def synchronize() -> None:
    """If current backend is CuPy, synchronize the default CUDA stream (no-op otherwise)."""
    if get_current_backend() == "cupy" and _cp is not None:
        try:
            _cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def register_module_for_backend_updates(
    module_name: str,
    module_globals: dict[str, Any] | None = None,
) -> None:
    """
    Register *module_name* so backend switches refresh its ``np``/``scipy``/``ndi`` globals.

    Typically used together with :func:`setup_backend_proxy` via :func:`setup`.
    """
    _backend_dependent_modules.add(module_name)
    if module_globals is not None:
        from .proxy import setup_backend_proxy

        setup_backend_proxy(module_globals, module_name=module_name)


def setup(module_globals: dict[str, Any], module_name: str | None = None):
    """
    One-call setup: :func:`~nvitk.core.proxy.setup_backend_proxy` + :func:`register_module_for_backend_updates`.

    Parameters
    ----------
    module_globals
        Pass ``globals()`` from the calling module.
    module_name
        Defaults to ``module_globals['__name__']``.

    Returns
    -------
    BackendProxy
        The proxy bound to *module_globals*.
    """
    if module_name is None:
        module_name = module_globals.get("__name__", "unknown")

    from .proxy import setup_backend_proxy

    proxy = setup_backend_proxy(module_globals, module_name=module_name)
    register_module_for_backend_updates(module_name, module_globals=module_globals)
    return proxy


def get_ops(name: str | None = None, allow_fallback: bool = True) -> tuple[Any, Any, Any]:
    """Return ``(xp, scipy, ndi)`` tuple for *name* or current backend."""
    target = get_current_backend() if name is None else _resolve_backend(name, allow_fallback=allow_fallback)
    return _BACKENDS[target]


def is_cupy_array(arr: Any) -> bool:
    """True if *arr* is a ``cupy.ndarray``."""
    return _cp is not None and isinstance(arr, _cp.ndarray)


def is_numpy_array(arr: Any) -> bool:
    """True if *arr* is a ``numpy.ndarray``."""
    return isinstance(arr, np.ndarray)


def to_numpy(arr: Any, copy: bool = False) -> np.ndarray:
    """Materialize *arr* on CPU as ``numpy.ndarray`` (``get()`` for CuPy)."""
    if is_cupy_array(arr):
        out = arr.get()
        return out.copy() if copy else out
    if isinstance(arr, np.ndarray):
        return arr.copy() if copy else arr
    out = np.asarray(arr)
    return out.copy() if copy else out


def to_cupy(arr: Any, copy: bool = False, strict: bool = False):
    """
    Move *arr* to CuPy, or return NumPy array if *strict* is False and CuPy is unavailable.

    Parameters
    ----------
    strict
        If True, raise :class:`~nvitk.core.exceptions.BackendUnavailableError` when CuPy cannot load.
    """
    if _cp is None or "cupy" not in _BACKENDS:
        if strict:
            raise BackendUnavailableError("CuPy backend is unavailable.")
        out = np.asarray(arr)
        return out.copy() if copy else out
    if is_cupy_array(arr):
        return arr.copy() if copy else arr
    out = _cp.asarray(arr)
    return out.copy() if copy else out
