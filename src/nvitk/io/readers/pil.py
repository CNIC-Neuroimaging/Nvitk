from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes


def read_pil(path: str, *, axes: str | None = None, **_: Any):
    try:
        from PIL import Image as PILImage
    except Exception as exc:
        raise BackendUnavailableError('Pillow is not installed. Please install it with "pip install pillow".') from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with PILImage.open(str(p)) as img:
        data = np.asarray(img)
        axes_prev = "YX" if data.ndim == 2 else "YXC"
        metadata: dict[str, Any] = {
            "axes": axes_prev,
            "shape": tuple(data.shape),
            "mode": img.mode,
        }

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
