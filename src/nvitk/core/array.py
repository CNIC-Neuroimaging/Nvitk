from __future__ import annotations

from typing import Any

from .backend import is_cupy_array, is_numpy_array, to_cupy, to_numpy


def as_backend_array(
    arr: Any,
    backend: str | None = None,
    copy: bool = False,
    strict: bool = False,
) -> Any:
    from .backend import get_current_backend

    target = (backend or get_current_backend()).strip().lower()
    if target in {"numpy", "cpu", "np"}:
        return to_numpy(arr, copy=copy)
    if target in {"cupy", "gpu", "cp"}:
        return to_cupy(arr, copy=copy, strict=strict)
    raise ValueError(f"Unsupported backend '{backend}'. Expected numpy/cupy.")


def ensure_same_backend(*arrays: Any, backend: str | None = None) -> tuple[Any, ...]:
    return tuple(as_backend_array(a, backend=backend) for a in arrays)


__all__ = [
    "is_cupy_array",
    "is_numpy_array",
    "to_numpy",
    "to_cupy",
    "as_backend_array",
    "ensure_same_backend",
]
