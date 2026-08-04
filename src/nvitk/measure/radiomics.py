"""Thin PyRadiomics wrapper with an explicit fallback for ``IntegratedIntensity``."""

from __future__ import annotations

from typing import Any, Iterable

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import ensure_same_shape, resolve_array, resolve_spacing

setup(globals())

try:
    from radiomics import featureextractor as _featureextractor
except ImportError:
    _featureextractor = None


def integrated_intensity(raw: Image | Any, mask: Image | Any) -> float:
    """Integrated intensity of *raw* over ``mask > 0`` (unit-less sum)."""
    ensure_same_shape(raw, mask)
    r = as_backend_array(resolve_array(raw))
    m = as_backend_array(resolve_array(mask))
    out = r[m > 0].sum()
    return float(to_numpy(out))


def compute_radiomics(
    image: Image | Any,
    mask: Image | Any,
    *,
    feature_classes: Iterable[str] | None = None,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """
    Compute PyRadiomics first-order / shape features for (*image*, *mask*).

    Parameters
    ----------
    image
        Intensity image. May be ``None`` only when ``feature_classes == {'shape'}``.
    mask
        Binary/label mask with the same shape as *image*.
    feature_classes
        Iterable of class names. Defaults to ``('firstorder', 'shape')``.
    spacing
        Explicit spacing in mm. Defaults to the mask's :attr:`Image.spacing`.

    Returns
    -------
    dict[str, float]
        Feature names are normalized to ``fo_...`` / ``s_...`` prefixes.
    """
    if _featureextractor is None:
        raise RuntimeError(
            "PyRadiomics is not installed. `pip install pyradiomics` to use compute_radiomics."
        )

    classes = tuple(feature_classes) if feature_classes else ("firstorder", "shape")

    if image is None:
        if set(classes) == {"shape"}:
            image = mask
        else:
            raise ValueError(f"compute_radiomics requires an image for feature classes {classes}.")

    ensure_same_shape(image, mask)
    # pyradiomics: NumPy-only library, so we force the host hop here.
    raw = to_numpy(resolve_array(image))
    lbl = to_numpy(resolve_array(mask))

    try:
        sp = resolve_spacing(mask if isinstance(mask, Image) else image, spacing)
    except ValueError:
        sp = None

    import SimpleITK as sitk

    img = sitk.GetImageFromArray(raw)
    lbl_sitk = sitk.GetImageFromArray(lbl)
    if sp is not None:
        spacing_list = [float(s) for s in sp]
        img.SetSpacing(spacing_list)
        lbl_sitk.SetSpacing(spacing_list)

    extractor = _featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    for cls in classes:
        extractor.enableFeatureClassByName(cls)

    result = extractor.execute(img, lbl_sitk)

    if "firstorder" in classes:
        result["original_firstorder_IntegratedIntensity"] = integrated_intensity(image, mask)

    def _norm(key: str) -> str:
        """Shorten PyRadiomics feature key prefixes for compact export column names."""
        return (
            key.replace("original_shape_", "s_")
            .replace("original_firstorder_", "fo_")
        )

    return {_norm(k): v for k, v in result.items() if k.startswith("original_")}


__all__ = ["integrated_intensity", "compute_radiomics"]
