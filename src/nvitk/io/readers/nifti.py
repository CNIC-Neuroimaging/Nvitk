from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

from .._common import default_nifti_axes, reorder_axes


def read_nifti(path: str, *, axes: str | None = None, **_: Any):
    try:
        import nibabel as nib
    except Exception as exc:
        raise BackendUnavailableError('nibabel is not installed. Please install it with "pip install nibabel".') from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    proxy = nib.load(str(p))
    data = np.asarray(proxy.dataobj)
    metadata: dict[str, Any] = {
        "axes": default_nifti_axes(data.ndim),
        "shape": tuple(data.shape),
        "affine": proxy.affine,
    }

    zooms = proxy.header.get_zooms()[: data.ndim]
    if len(zooms) > 0:
        metadata["x_res"] = float(zooms[0])
    if len(zooms) > 1:
        metadata["y_res"] = float(zooms[1])
    if len(zooms) > 2:
        metadata["z_res"] = float(zooms[2])
    if len(zooms) > 3:
        metadata["t_res"] = float(zooms[3])
        metadata["temporal_resolution"] = float(zooms[3])

    for extension in proxy.header.extensions:
        try:
            try:
                content_bytes = extension.get_content()
            except Exception:
                # Fallback for uncommon extension codes that nibabel cannot decode.
                content_bytes = getattr(extension, "_raw", None)
            if isinstance(content_bytes, (bytes, bytearray)):
                payload = bytes(content_bytes).rstrip(b"\x00")
                extension_metadata = json.loads(payload.decode("utf-8"))
                if isinstance(extension_metadata, dict):
                    if 'axes' in extension_metadata: extension_metadata.pop('axes')
                    if 'shape' in extension_metadata: extension_metadata.pop('shape')
                    if 'affine' in extension_metadata: extension_metadata.pop('affine')
                    if 'x_res' in extension_metadata: extension_metadata.pop('x_res')
                    if 'y_res' in extension_metadata: extension_metadata.pop('y_res')
                    if 'z_res' in extension_metadata: extension_metadata.pop('z_res')
                    if 't_res' in extension_metadata: extension_metadata.pop('t_res')
                    if 'temporal_resolution' in extension_metadata: extension_metadata.pop('temporal_resolution')
                    metadata.update(extension_metadata)
        except Exception:
            continue

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
