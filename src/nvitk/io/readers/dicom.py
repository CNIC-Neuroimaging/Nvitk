"""
Thin wrapper around :func:`~nvitk.io.conversors._dicom_conversion.load_dicom_series` for ``imread``.

Returns array(s) and metadata without writing NIfTI files to disk.
"""

from __future__ import annotations

from typing import Any

from ..conversors._dicom_conversion import load_dicom_series


def read_dicom(
    path: str,
    *,
    axes: str | None = None,
    series_number: str | None = None,
    series_uid: str | None = None,
    series_index: int | None = None,
    return_all_series: bool = False,
    include_private_tags: bool = False,
    force_ras: bool = False,
    revert_scaling: bool = False,
    rescale_type: str = "DV",
    tmp_dir=None,
    **_: Any,
):
    """
    Read a DICOM file, directory, or archive using the shared conversion stack (see *load_dicom_series*).

    Parameters mirror the loader: series selection, optional private tags, RAS orientation, and
    rescaling. Returns one ``(data, metadata)`` pair or a list when ``return_all_series=True``.
    """
    return load_dicom_series(
        path,
        axes=axes,
        series_number=series_number,
        series_uid=series_uid,
        series_index=series_index,
        return_all_series=return_all_series,
        include_private_tags=include_private_tags,
        force_ras=force_ras,
        revert_scaling=revert_scaling,
        rescale_type=rescale_type,
        tmp_dir=tmp_dir,
    )
