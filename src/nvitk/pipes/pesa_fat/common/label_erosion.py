"""Per-label erosion of PESA-Fat label maps (CT-PET and Dixon stage 2).

The measurement masks are shrunk label by label so their statistics stay inside the
structure (organ motion between CT and PET, partial volume on the Dixon maps); the
``*_VOL`` masks used for volumes are never eroded.
"""

from __future__ import annotations

from typing import Any

from nvitk.core import as_backend_array
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.morphology import erode
from nvitk.types import Image

setup(globals())

log = Logger()


def erode_binary(ref: Image, binary: Any, iterations: int, name: str) -> Any:
    """*binary* eroded by *iterations* 6-connected steps on *ref*'s grid.

    Returned uneroded, with a warning, when erosion would empty it, since an empty
    mask yields no measurement at all.
    """
    binary = as_backend_array(binary) > 0
    if int(iterations) <= 0 or not bool(binary.any()):
        return binary
    shrunk = erode(
        ref.with_data(binary.astype(np.uint8)),
        footprint=1,
        iterations=int(iterations),
        mode="binary",
    ).data > 0
    if not bool(shrunk.any()):
        log.warning(
            f"Eroding {name} by {int(iterations)} iteration(s) would empty it; keeping it as is."
        )
        return binary
    return shrunk


def erode_labels(label_img: Image, labels: dict[str, int], iterations_by_name: dict[str, int]) -> Image:
    """Erode each label of *label_img* named in *iterations_by_name* by its iterations.

    Every label is eroded on its own (other labels count as background), so
    neighbouring labels shrink away from each other too. Names missing from *labels*
    and non-positive iterations are skipped.
    """
    arr = as_backend_array(label_img.data).copy()
    for name, iterations in iterations_by_name.items():
        lid = labels.get(name)
        if lid is None or int(iterations) <= 0:
            continue
        bin_mask = arr == lid
        if not bool(bin_mask.any()):
            continue
        shrunk = erode_binary(label_img, bin_mask, int(iterations), name)
        arr[bin_mask & ~shrunk] = 0
    return label_img.with_data(arr)


__all__ = ["erode_binary", "erode_labels"]
