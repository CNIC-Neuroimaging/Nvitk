"""
Pickle (``.pkl``) reader for preprocessing sidecars and directly pickled arrays.

nnU-Net / nnssl write ``<name>.pkl`` next to ``<name>.b2nd`` to carry the geometry the array
itself cannot hold, so opening the sidecar is the same request as opening the volume: this
reader delegates to :func:`~nvitk.io.readers.read_b2nd` when the sibling exists. Pickles that
contain an array directly are loaded as-is.

Use :func:`read_pkl` or :func:`nvitk.io.imread` with ``force_type='pkl'``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import UnsupportedFormatError

from .._common import default_nifti_axes, reorder_axes
from .b2nd import read_b2nd

# Keys under which pipelines commonly stash the voxel array inside a pickled dict.
_ARRAY_KEYS = ("data", "image", "array", "arr", "volume")


def _array_from_payload(payload: Any) -> np.ndarray | None:
    """Extract the voxel array from an unpickled object (an array, or a dict holding one)."""
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, dict):
        for key in _ARRAY_KEYS:
            value = payload.get(key)
            if isinstance(value, np.ndarray):
                return value
    return None


def read_pkl(path: str, *, axes: str | None = None, **kwargs: Any):
    """
    Load the volume a ``.pkl`` refers to: the sibling ``.b2nd`` if present, else a pickled array.

    Extra keyword arguments (``channel``, ``squeeze_channel``, ``world``, …) are forwarded to
    :func:`~nvitk.io.readers.read_b2nd` when delegating.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Voxel array and metadata dict.

    Raises
    ------
    UnsupportedFormatError
        If the pickle is a metadata-only sidecar with no companion array file, or holds no array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    sibling = p.with_suffix(".b2nd")
    if sibling.is_file():
        kwargs.pop("properties", None)  # *path* is the sidecar; an override would be contradictory
        return read_b2nd(str(sibling), axes=axes, properties=str(p), **kwargs)

    with p.open("rb") as f:
        payload = pickle.load(f)

    data = _array_from_payload(payload)
    if data is None:
        raise UnsupportedFormatError(
            f"{p.name} holds no voxel array and has no companion '{sibling.name}'. "
            "Preprocessing sidecars are only readable next to the array they describe."
        )

    metadata: dict[str, Any] = {
        "axes": default_nifti_axes(data.ndim),
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "filename": p.name,
        "name": p.stem,
    }
    if isinstance(payload, dict):
        metadata["preprocessing_properties"] = {k: v for k, v in payload.items() if k not in _ARRAY_KEYS}

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
