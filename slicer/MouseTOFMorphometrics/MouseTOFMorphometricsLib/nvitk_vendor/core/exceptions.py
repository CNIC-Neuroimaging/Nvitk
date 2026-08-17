"""Exception types mirroring ``nvitk.core.exceptions`` (hand-written; not synced)."""

from __future__ import annotations


class NvitkError(Exception):
    """Base class for nvitk errors."""


class ValidationError(NvitkError, ValueError):
    """Invalid argument / input shape / inconsistent metadata."""


class BackendUnavailableError(NvitkError, RuntimeError):
    """A requested array backend (CuPy) is not installed."""


__all__ = ["BackendUnavailableError", "NvitkError", "ValidationError"]
