"""Exceptions for dataset tables, manifests, filters, and XNAT sync."""

from __future__ import annotations

from nvitk.core.exceptions import NvitkError


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / repository
# ──────────────────────────────────────────────────────────────────────────────


class DatasetError(NvitkError):
    """Base class for :mod:`nvitk.db` catalog and :class:`~nvitk.db.repo.DataRepo` failures."""


class ManifestError(DatasetError):
    """Parquet manifest or cohort metadata is missing, malformed, or inconsistent."""


class TableNotFoundError(DatasetError):
    """No :class:`~nvitk.db.catalog.TableDefinition` matches the requested table name."""


class FilterError(DatasetError):
    """Filter keys or values cannot be applied to the selected table (e.g. unknown column)."""


class XnatSyncError(DatasetError):
    """XNAT download, extraction, or inventory update failed."""

class SettingsError(DatasetError):
    """Error reading settings file."""
