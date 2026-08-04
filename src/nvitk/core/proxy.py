"""Per-module proxies so ``np`` / ``scipy`` / ``ndi`` globals track :mod:`nvitk.core.backend` switches."""

from __future__ import annotations

from typing import Any

_PROXY_GLOBAL_KEYS = ("np", "scipy", "ndi")


# ──────────────────────────────────────────────────────────────────────────────
# BackendProxy
# ──────────────────────────────────────────────────────────────────────────────


class BackendProxy:
    """
    Lazily bind ``np``, ``scipy``, ``ndi`` in a module's ``globals()`` to the active backend.

    On :meth:`refresh_globals`, inserts either NumPy/SciPy or CuPy/cupyx equivalents.
    Unknown attribute access forwards to the active array module (``proxy.np``).
    """

    def __init__(self, module_name: str) -> None:
        """Create an unbound proxy for *module_name* (call :meth:`bind_globals` to attach it)."""
        self.module_name = module_name
        self._cached_backend: str | None = None
        self._cached_modules: tuple[Any, Any, Any] | None = None
        self._module_globals: dict[str, Any] | None = None

    def _resolve_modules(self) -> tuple[Any, Any, Any]:
        """Return the ``(array, scipy, ndimage)`` triple for the active backend, cached per backend."""
        from .backend import _BACKENDS, get_current_backend

        backend = get_current_backend()
        if self._cached_modules is None or self._cached_backend != backend:
            self._cached_modules = _BACKENDS[backend]
            self._cached_backend = backend
        return self._cached_modules

    def bind_globals(self, module_globals: dict[str, Any]) -> None:
        """Store *module_globals* and populate ``np``/``scipy``/``ndi`` keys."""
        self._module_globals = module_globals
        self.refresh_globals()

    def invalidate(self) -> None:
        """Drop cached backend tuple so the next access re-reads :func:`~nvitk.core.backend.get_current_backend`."""
        self._cached_backend = None
        self._cached_modules = None

    def refresh_globals(self) -> None:
        """Write current ``(np, scipy, ndi)`` triple into bound module globals."""
        if self._module_globals is None:
            return
        np_mod, scipy_mod, ndi_mod = self._resolve_modules()
        self._module_globals["np"] = np_mod
        self._module_globals["scipy"] = scipy_mod
        self._module_globals["ndi"] = ndi_mod

    @property
    def np(self) -> Any:
        """Active array module — NumPy on the CPU backend, CuPy on the GPU backend."""
        return self._resolve_modules()[0]

    @property
    def scipy(self) -> Any:
        """Active SciPy module — ``scipy`` or ``cupyx.scipy``."""
        return self._resolve_modules()[1]

    @property
    def ndi(self) -> Any:
        """Active n-d image module — ``scipy.ndimage`` or ``cupyx.scipy.ndimage``."""
        return self._resolve_modules()[2]

    def __getattr__(self, name: str) -> Any:
        """Forward any unknown attribute to the active array module (so ``proxy.zeros`` works)."""
        return getattr(self.np, name)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_proxies: dict[str, BackendProxy] = {}


def get_backend_proxy(module_name: str | None = None) -> BackendProxy:
    """
    Return (or create) the :class:`BackendProxy` keyed by *module_name*.

    If *module_name* is omitted, uses the caller's ``__name__`` via the stack.
    """
    if module_name is None:
        import inspect

        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            module_name = "unknown"
        else:
            module_name = frame.f_back.f_globals.get("__name__", "unknown")

    if module_name not in _proxies:
        _proxies[module_name] = BackendProxy(module_name=module_name)
    return _proxies[module_name]


def setup_backend_proxy(module_globals: dict[str, Any], module_name: str | None = None) -> BackendProxy:
    """
    Bind *module_globals* to a proxy for this package (typically ``setup_backend_proxy(globals())``).

    Returns
    -------
    BackendProxy
        The proxy instance; call :meth:`BackendProxy.refresh_globals` after backend changes (normally automatic).
    """
    if module_name is None:
        module_name = module_globals.get("__name__", "unknown")

    proxy = get_backend_proxy(module_name=module_name)
    proxy.bind_globals(module_globals)
    return proxy


def refresh_all_proxies() -> None:
    """Invalidate and refresh every registered :class:`BackendProxy` (called from :mod:`nvitk.core.backend`)."""
    for proxy in _proxies.values():
        proxy.invalidate()
        proxy.refresh_globals()


__all__ = [
    "BackendProxy",
    "get_backend_proxy",
    "setup_backend_proxy",
    "refresh_all_proxies",
    "_proxies",
]
