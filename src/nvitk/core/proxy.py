from __future__ import annotations

from typing import Any

_PROXY_GLOBAL_KEYS = ("np", "scipy", "ndi")


class BackendProxy:
    """
    Dynamic proxy that resolves backend modules (`numpy/scipy/ndi` or
    `cupy/cupyx.scipy/cupyx.scipy.ndimage`) on demand.

    Each importing module should have its own proxy instance so module-level
    globals can be refreshed when backend changes.
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._cached_backend: str | None = None
        self._cached_modules: tuple[Any, Any, Any] | None = None
        self._module_globals: dict[str, Any] | None = None

    def _resolve_modules(self) -> tuple[Any, Any, Any]:
        from .backend import _BACKENDS, get_current_backend

        backend = get_current_backend()
        if self._cached_modules is None or self._cached_backend != backend:
            self._cached_modules = _BACKENDS[backend]
            self._cached_backend = backend
        return self._cached_modules

    def bind_globals(self, module_globals: dict[str, Any]) -> None:
        self._module_globals = module_globals
        self.refresh_globals()

    def invalidate(self) -> None:
        self._cached_backend = None
        self._cached_modules = None

    def refresh_globals(self) -> None:
        if self._module_globals is None:
            return
        np_mod, scipy_mod, ndi_mod = self._resolve_modules()
        self._module_globals["np"] = np_mod
        self._module_globals["scipy"] = scipy_mod
        self._module_globals["ndi"] = ndi_mod

    @property
    def np(self) -> Any:
        return self._resolve_modules()[0]

    @property
    def scipy(self) -> Any:
        return self._resolve_modules()[1]

    @property
    def ndi(self) -> Any:
        return self._resolve_modules()[2]

    def __getattr__(self, name: str) -> Any:
        # Forward unknown attrs to the active array module (np/cp)
        return getattr(self.np, name)


_proxies: dict[str, BackendProxy] = {}


def get_backend_proxy(module_name: str | None = None) -> BackendProxy:
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
    Backward-compatible setup function.

    Typical usage:
        setup_backend_proxy(globals())
    """
    if module_name is None:
        module_name = module_globals.get("__name__", "unknown")

    proxy = get_backend_proxy(module_name=module_name)
    proxy.bind_globals(module_globals)
    return proxy


def refresh_all_proxies() -> None:
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
