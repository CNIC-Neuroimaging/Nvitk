"""NumPy-only replacement for ``nvitk.core.array`` (hand-written; not synced)."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(arr: Any, copy: bool = False) -> np.ndarray:
    """Return *arr* as a host NumPy array (unwrapping an ``Image``-like ``.data``)."""
    data = getattr(arr, "data", arr)
    # CuPy arrays would arrive with .get(); harmless to support, never hit here.
    get = getattr(data, "get", None)
    if get is not None and type(data).__module__.startswith("cupy"):
        data = get()
    out = np.asarray(data)
    return out.copy() if copy else out


def to_cupy(arr: Any, copy: bool = False, strict: bool = False) -> np.ndarray:
    """CuPy is never available in the vendored build; returns the NumPy array."""
    if strict:
        raise RuntimeError("CuPy is not available in the vendored nvitk build.")
    return to_numpy(arr, copy=copy)


def as_backend_array(arr: Any, backend: str | None = None, copy: bool = False, strict: bool = False) -> np.ndarray:
    """Convert *arr* to the active backend's array type — always NumPy here."""
    target = (backend or "numpy").strip().lower()
    if target in {"numpy", "cpu", "np"}:
        return to_numpy(arr, copy=copy)
    if target in {"cupy", "gpu", "cp"}:
        return to_cupy(arr, copy=copy, strict=strict)
    raise ValueError(f"Unsupported backend '{backend}'. Expected numpy/cupy.")


def resolve_array(arr: Any) -> np.ndarray:
    """Array view of *arr*, unwrapping ``Image``-like containers."""
    return to_numpy(arr)


def ensure_same_backend(*arrays: Any, backend: str | None = None) -> tuple[np.ndarray, ...]:
    """Coerce every input to the active backend (NumPy)."""
    return tuple(as_backend_array(a, backend=backend) for a in arrays)


__all__ = [
    "as_backend_array",
    "ensure_same_backend",
    "resolve_array",
    "to_cupy",
    "to_numpy",
]
