"""High-level helpers to coerce arrays to the active NumPy or CuPy backend."""

from __future__ import annotations

from typing import Any

from .backend import is_cupy_array, is_numpy_array, to_cupy, to_numpy


# ──────────────────────────────────────────────────────────────────────────────
# Coercion
# ──────────────────────────────────────────────────────────────────────────────


def as_backend_array(
    arr: Any,
    backend: str | None = None,
    copy: bool = False,
    strict: bool = False,
) -> Any:
    """
    Convert *arr* to NumPy or CuPy using :func:`~nvitk.core.backend.get_current_backend` when *backend* is None.

    Parameters
    ----------
    arr
        Any array-like or CuPy/NumPy array.
    backend
        ``numpy``/``cpu``/``np`` or ``cupy``/``gpu``/``cp``; default follows current context.
    copy
        Force a copy on conversion.
    strict
        For CuPy: raise :class:`~nvitk.core.exceptions.BackendUnavailableError` if CuPy missing.

    Raises
    ------
    ValueError
        Unknown *backend* string.
    """
    from .backend import get_current_backend

    target = (backend or get_current_backend()).strip().lower()
    if target in {"numpy", "cpu", "np"}:
        return to_numpy(arr, copy=copy)
    if target in {"cupy", "gpu", "cp"}:
        return to_cupy(arr, copy=copy, strict=strict)
    raise ValueError(f"Unsupported backend '{backend}'. Expected numpy/cupy.")


def ensure_same_backend(*arrays: Any, backend: str | None = None) -> tuple[Any, ...]:
    """Apply :func:`as_backend_array` to each input with the same *backend* (or current)."""
    return tuple(as_backend_array(a, backend=backend) for a in arrays)


__all__ = [
    "is_cupy_array",
    "is_numpy_array",
    "to_numpy",
    "to_cupy",
    "as_backend_array",
    "ensure_same_backend",
]
