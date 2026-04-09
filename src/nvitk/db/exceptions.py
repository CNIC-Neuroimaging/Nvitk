from __future__ import annotations

from nvitk.core.exceptions import NvitkError


class DatasetError(NvitkError):
    """Base exception for dataset/repository operations."""


class ManifestError(DatasetError):
    """Raised when a dataset manifest is invalid or inconsistent."""


class TableNotFoundError(DatasetError):
    """Raised when a requested dataset table is not defined."""


class FilterError(DatasetError):
    """Raised when a filter specification cannot be applied."""


class XnatSyncError(DatasetError):
    """Raised when XNAT inventory or download steps fail."""
