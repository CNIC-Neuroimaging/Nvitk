"""Shared Click ``--backend`` option for nvitk CLIs and pipeline entry points."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, TypeVar

import click

from nvitk.core.backend import set_default_backend

F = TypeVar("F", bound=Callable[..., Any])


def apply_cli_backend(backend: str, *, allow_fallback: bool = True) -> str:
    """Apply *backend* from CLI (``cpu`` / ``gpu`` or ``numpy`` / ``cupy``)."""
    return set_default_backend(backend, allow_fallback=allow_fallback)


def backend_click_option(*, default: str = "gpu") -> Callable[[F], F]:
    """Decorator: add ``--backend cpu|gpu`` and call :func:`set_default_backend` before the command."""

    def decorator(f: F) -> F:
        f = click.option(
            "--backend",
            type=click.Choice(["cpu", "gpu"], case_sensitive=False),
            default=default,
            show_default=True,
            help="Array backend: cpu (NumPy) or gpu (CuPy).",
        )(f)

        accepts_backend = "backend" in inspect.signature(f).parameters

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            backend_val = kwargs.pop("backend", default)
            apply_cli_backend(backend_val)
            if accepts_backend:
                kwargs["backend"] = backend_val
            return f(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def sge_backend_env(src_bind: str, backend: str = "gpu") -> dict[str, str]:
    """``extra_env`` for SGE workers: ``PYTHONPATH`` plus ``NVITK_BACKEND``."""
    return {
        "PYTHONPATH": str(src_bind),
        "NVITK_BACKEND": str(backend).strip().lower(),
    }


__all__ = [
    "apply_cli_backend",
    "backend_click_option",
    "set_default_backend",
    "sge_backend_env",
]
