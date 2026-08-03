"""Mouse TOF CoW Stage-1 / finalize recipe (hardcoded Lab parameters).

Self-contained: Slicer N4ITK CLI + local blood_flood. No nvitk / ANTsPy.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi

from .blood_flood import blood_flood, blood_flood_from_scratch
from .n4 import n4_bias_field_correction

log = logging.getLogger(__name__)

TREE_SPECS: tuple[tuple[str, int], ...] = (
    ("Left ICA", 1),
    ("Right ICA", 2),
    ("Basilar", 3),
)

# Stage-1 recipe (match Napari Lab mouse_tof_cow where units allow).
_N4_SHRINK = 2
_FRANGI_SIGMAS = (0.75, 1.0, 1.5, 2.0, 2.5)
_HYST_LOW = 4.0
_HYST_HIGH = 0.5
_THICKEN = 0
_THIN_PERCENTILE = 55.0
_MIN_CC = 125
_CONNECTIVITY = 3

# Final multilabel expand.
_EXPAND_FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5)
_EXPAND_HYST_LOW = 3.0
_EXPAND_HYST_HIGH = 0.5
_EXPAND_THICKEN = 0
_EXPAND_THIN_PERCENTILE = 25.0
_EXPAND_CONNECTIVITY = 3


def run_stage1(input_volume_node) -> tuple[np.ndarray, np.ndarray]:
    """N4 (Slicer N4ITK CLI) → blood flood from-scratch → label CCs.

    Parameters
    ----------
    input_volume_node
        ``vtkMRMLScalarVolumeNode`` with the 3D TOF intensity.

    Returns
    -------
    labeled, intensity
        Connected-component labels (int32) and the raw (pre-N4) intensity used
        later for the final multilabel expand.
    """
    import slicer

    arr = np.asarray(slicer.util.arrayFromVolume(input_volume_node))
    if arr.ndim != 3:
        raise ValueError(f"Mouse TOF CoW Stage 1 expects a 3D volume, got {arr.ndim}D")
    raw = np.asarray(arr, dtype=np.float32)

    log.info("Mouse TOF CoW Stage 1: N4 bias correction (N4ITKBiasFieldCorrection)")
    corrected = n4_bias_field_correction(
        input_volume_node,
        shrink_factor=_N4_SHRINK,
        spline_distance=None,  # N4ITK default (mm); do not reuse ANTs voxel spline=6
    )

    log.info("Mouse TOF CoW Stage 1: blood flood from-scratch")
    flood = blood_flood_from_scratch(
        corrected,
        frangi_sigmas=_FRANGI_SIGMAS,
        hyst_low_factor=_HYST_LOW,
        hyst_high_factor=_HYST_HIGH,
        thicken_iter=_THICKEN,
        thin_vesselness_percentile=_THIN_PERCENTILE,
        min_cc_voxels=_MIN_CC,
        connectivity=_CONNECTIVITY,
    )
    tree = np.asarray(flood.tree, dtype=bool)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_cc = ndi.label(tree, structure=structure)
    labeled = np.asarray(labeled, dtype=np.int32)
    log.info("Mouse TOF CoW Stage 1 done: %s connected components", n_cc)
    return labeled, raw


def expand_cow_trees(intensity: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Final multilabel blood-flood expand of assigned CoW tree seeds."""
    result = blood_flood(
        np.asarray(intensity, dtype=np.float64),
        np.asarray(markers, dtype=np.int32),
        frangi_sigmas=_EXPAND_FRANGI_SIGMAS,
        hyst_low_factor=_EXPAND_HYST_LOW,
        hyst_high_factor=_EXPAND_HYST_HIGH,
        thicken_iter=_EXPAND_THICKEN,
        thin_vesselness_percentile=_EXPAND_THIN_PERCENTILE,
        connectivity=_EXPAND_CONNECTIVITY,
    )
    labels = np.asarray(result.labels, dtype=np.int32)
    log.info(
        "Mouse TOF CoW final expand: tree_voxels=%s labeled=%s",
        int(np.count_nonzero(result.tree)),
        int(np.count_nonzero(labels)),
    )
    return labels


def ensure_optional_deps() -> None:
    """Raise a clear error if scikit-image / scikit-learn / N4ITK are missing."""
    missing = []
    try:
        import skimage  # noqa: F401
    except ImportError:
        missing.append("scikit-image")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        raise RuntimeError(
            "Mouse TOF CoW is missing: "
            + ", ".join(missing)
            + ". In the Slicer Python console run:\n"
            "  import slicer\n"
            '  slicer.util.pip_install("scikit-image scikit-learn")'
        )
    try:
        import slicer

        if getattr(slicer.modules, "n4itkbiasfieldcorrection", None) is None:
            raise RuntimeError(
                "Slicer built-in module N4ITKBiasFieldCorrection was not found. "
                "Ensure CLI / built-in modules are not disabled in Application Settings."
            )
    except ImportError as exc:
        raise RuntimeError(
            "Mouse TOF CoW N4 requires running inside 3D Slicer."
        ) from exc
