"""
Immutable orchestrator that binds an ``(image, mask)`` pair once and exposes
convenience methods returning plain dicts.

Each action (``volume``, ``intensity``, ``suv``, ``voxel_metrics``,
``surface_metrics``, ``radiomics``) is a thin call into the functional
primitives. An optional alignment step via :mod:`nvitk.transform.resampling`
produces a new :class:`Measurer` with the image and mask on a common grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from nvitk.core.array import to_numpy
from nvitk.transform.resampling import resample_mask_to_pet, resample_pet_to_mask, resample_to
from nvitk.types import Image

from .compare import correlation_stats
from .intensity import masked_stats
from .radiomics import compute_radiomics, integrated_intensity
from .surface import surface_metrics
from .suv import suv_image, suv_stats
from .volume import volume_cc, volume_mm3
from .voxel import voxel_metrics


_AlignMethod = "affine"
_AlignDirection = ("raw_to_mask", "mask_to_raw")


@dataclass(frozen=True)
class Measurer:
    """
    Bind an intensity/raw image and a segmentation mask together.

    Instances are immutable; transformations like :meth:`align` return a new
    :class:`Measurer`. All measurement methods return plain :class:`dict`\\s.
    """

    image: Image
    mask: Image

    # -----------------------------------------------------------------
    # Aligners
    # -----------------------------------------------------------------
    def align(
        self,
        direction: str = "mask_to_raw",
        *,
        method: str = "affine",
        order: int | None = None,
    ) -> "Measurer":
        """
        Resample image or mask so both share a grid.

        Parameters
        ----------
        direction
            ``'raw_to_mask'`` (default) -> image resampled to mask's grid.
            ``'mask_to_raw'`` -> mask resampled to image's grid.
        method
            Currently only ``'affine'``.
        order
            Interpolation order. Defaults to 1 for image and 0 for mask.
        """
        if method != "affine":
            raise ValueError(f"Unknown alignment method {method!r}; expected 'affine'.")
        if direction not in _AlignDirection:
            raise ValueError(f"direction must be one of {_AlignDirection}, got {direction!r}.")

        if direction == "raw_to_mask":
            new_image = resample_pet_to_mask(self.image, self.mask, order=1 if order is None else order)
            return Measurer(image=new_image, mask=self.mask)
        new_mask = resample_mask_to_pet(self.mask, self.image, order=0 if order is None else order)
        return Measurer(image=self.image, mask=new_mask)

    def with_mask(self, mask: Image) -> "Measurer":
        return Measurer(image=self.image, mask=mask)

    def with_image(self, image: Image) -> "Measurer":
        return Measurer(image=image, mask=self.mask)

    # -----------------------------------------------------------------
    # Measurements
    # -----------------------------------------------------------------
    def volume(self) -> dict[str, float]:
        return {
            "volume_mm3": volume_mm3(self.mask),
            "volume_cc": volume_cc(self.mask),
        }

    def intensity(self, *, stats: Iterable[str] = ("mean", "median", "max", "p95", "std", "sum")) -> dict[str, float]:
        return masked_stats(self.image, self.mask, stats=stats)

    def suv(
        self,
        *,
        kinds: Iterable[str] = ("bw",),
        stats: Iterable[str] = ("mean", "median", "max", "min", "std", "p95", "p5"),
        philips: bool = True,
        revert_scaling: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        return suv_stats(
            self.image, self.mask,
            metadata=metadata,
            kinds=kinds,
            stats=stats,
            philips=philips,
            revert_scaling=revert_scaling,
        )

    def suv_image(self, *, kind: str = "bw", philips: bool = True, revert_scaling: bool = False) -> Image:
        """Return the full SUV-converted :class:`Image` (unmasked)."""
        out = suv_image(self.image, kind=kind, philips=philips, revert_scaling=revert_scaling)
        assert isinstance(out, Image)
        return out

    def integrated_intensity(self) -> float:
        return integrated_intensity(self.image, self.mask)

    def voxel_metrics(self, reference: Image, *, metrics: Iterable[str] | None = None) -> dict[str, float]:
        return voxel_metrics(reference, self.mask, metrics=metrics)

    def surface_metrics(self, reference: Image, *, metrics: Iterable[str] | None = None) -> dict[str, float]:
        return surface_metrics(reference, self.mask, metrics=metrics)

    def radiomics(self, *, feature_classes: Iterable[str] | None = None) -> dict[str, float]:
        return compute_radiomics(self.image, self.mask, feature_classes=feature_classes)

    def correlation(self, other: Image) -> dict[str, float]:
        """Pearson/Spearman/MAE/RMSE between self.image and *other* on ``self.mask > 0``."""
        a = self.image.data[self.mask.data > 0]
        b = other.data[self.mask.data > 0]
        return correlation_stats(a, b)


__all__ = ["Measurer"]
