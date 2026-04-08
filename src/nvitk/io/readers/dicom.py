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
    Read DICOM source using the same series discovery and conversion
    strategy as the DICOM-to-NIfTI converter, but return numpy arrays
    plus metadata instead of writing files.
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
