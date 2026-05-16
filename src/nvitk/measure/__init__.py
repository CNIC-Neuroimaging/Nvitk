"""
Image measurements: volume, intensity, SUV, voxel-based overlap, surface
distance, radiomics, and intensity correlation.

Two APIs are provided:

1. **Functional primitives** (the public core). Each one takes a
   :class:`~nvitk.types.Image` (or raw array + spacing) and returns a ``dict``
   (or ``Image`` when it returns derived voxels). They are pure functions
   and backend-aware.
2. **A thin :class:`Measurer` orchestrator** (see :mod:`nvitk.measure.measurer`)
   that binds an (image, mask) pair once, optionally aligns them, and lets you
   chain ``.volume()``, ``.intensity()``, ``.suv()``, ``.voxel_metrics(ref)``
   etc. without retyping the inputs.

Example::

    from nvitk.measure import volume_cc, masked_stats, Measurer

    vol = volume_cc(mask)
    stats = masked_stats(pet, mask, stats=("mean", "max", "p95"))

    m = Measurer(pet, mask).align("raw_to_mask")
    df = (
        m.volume()
         | m.intensity(stats=("mean", "max"))
         | m.suv(kinds=("bw",))
    )
"""

from __future__ import annotations

from .compare import (
    correlation_stats,
    pearson,
    rmse,
    sample_at_physical_points,
    spearman,
)
from .intensity import masked_stats
from .measurer import Measurer
from .radiomics import compute_radiomics, integrated_intensity
from .surface import hausdorff, hausdorff95, mdsd, msd, stdsd, surface_metrics
from .suv import suv_image, suv_stats
from .volume import volume_cc, volume_mm3
from .hemodynamics import (
    mean_flow_ml_s,
    mean_velocity_mm_s,
    pulsatility_index,
    resistivity_index,
    through_plane_velocity_series,
    velocity_mm_s_from_phases,
)
from .voxel import (
    confusion_counts,
    dice,
    fnr,
    fpr,
    jaccard,
    precision,
    recall,
    volume_similarity,
    voxel_metrics,
)

__all__ = [
    "volume_mm3",
    "volume_cc",
    "masked_stats",
    "suv_image",
    "suv_stats",
    "dice",
    "jaccard",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "volume_similarity",
    "confusion_counts",
    "voxel_metrics",
    "hausdorff",
    "hausdorff95",
    "msd",
    "mdsd",
    "stdsd",
    "surface_metrics",
    "integrated_intensity",
    "compute_radiomics",
    "correlation_stats",
    "pearson",
    "spearman",
    "rmse",
    "sample_at_physical_points",
    "Measurer",
    "mean_flow_ml_s",
    "pulsatility_index",
    "resistivity_index",
    "through_plane_velocity_series",
    "velocity_mm_s_from_phases",
]
