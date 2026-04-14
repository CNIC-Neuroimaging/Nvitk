"""Typed exceptions for NVITK (I/O, backends, validation)."""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────────────────────────────────────


class NvitkError(Exception):
    """Base class for all library-specific errors."""


class BackendError(NvitkError):
    """Raised when array backend selection or availability fails."""


class BackendUnavailableError(BackendError):
    """Requested backend (e.g. CuPy) cannot be used in this environment."""


class UnsupportedFormatError(NvitkError):
    """No registered reader/writer matches the path or ``force_type``."""


class ValidationError(NvitkError):
    """Inconsistent axes, shapes, metadata, or other semantic validation failure."""
