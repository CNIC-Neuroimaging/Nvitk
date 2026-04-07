from __future__ import annotations


class NvitkError(Exception):
    """Base exception for nvitk."""


class BackendError(NvitkError):
    """Generic backend-related error."""


class BackendUnavailableError(BackendError):
    """Requested backend is not available in current runtime."""


class UnsupportedFormatError(NvitkError):
    """Input/output format is not supported by registered codecs."""


class ValidationError(NvitkError):
    """Validation error for shape, axes, metadata, etc."""