from __future__ import annotations

from typing import Any

from ..readers import read_nd2
from ..writers import write_nifti


def nikon2nifti(
    nikon_path: str,
    nifti_path: str,
    *,
    axes: str | None = None,
    **kwargs: Any,
) -> str:
    data, metadata = read_nd2(nikon_path, axes=axes, **kwargs)
    write_nifti(nifti_path, data, metadata=metadata, axes=axes)
    return nifti_path
