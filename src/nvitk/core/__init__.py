"""
Public core API: NumPy/CuPy backend switching, array helpers, and proxy setup.

Typical GPU-aware code::

    from nvitk.core import setup, get_current_backend
    setup(globals())

    def f(x):
        return np.asarray(x)  # np follows active backend after setup
"""

from __future__ import annotations

from .array import as_backend_array, ensure_same_backend, is_cupy_array, is_numpy_array, to_cupy, to_numpy
from .backend import (
    BackendModules,
    available_backends,
    get_array_module,
    get_backend_modules,
    get_current_backend,
    get_global_backend,
    get_ops,
    is_cupy_installed,
    is_gpu_available,
    register_module_for_backend_updates,
    set_default_backend,
    set_global_backend,
    setup,
    synchronize,
    using,
    using_backend,
)
from .proxy import BackendProxy, get_backend_proxy, setup_backend_proxy

__all__ = [
    "BackendModules",
    "BackendProxy",
    "available_backends",
    "is_cupy_installed",
    "is_gpu_available",
    "set_default_backend",
    "set_global_backend",
    "get_global_backend",
    "get_current_backend",
    "using",
    "using_backend",
    "get_backend_modules",
    "get_ops",
    "get_array_module",
    "synchronize",
    "setup_backend_proxy",
    "register_module_for_backend_updates",
    "setup",
    "get_backend_proxy",
    "to_numpy",
    "to_cupy",
    "is_cupy_array",
    "is_numpy_array",
    "as_backend_array",
    "ensure_same_backend",
]
