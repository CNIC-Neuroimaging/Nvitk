"""VMTK image filters on NumPy volumes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._core import array_from_image, image_from_array, run_script

#: ``vmtkimagevesselenhancement`` methods: Frangi, Sato, vesselness-enhancing diffusion (VED,
#: Manniesing 2006) and its multiscale-memory variant.
VESSELNESS_METHODS: tuple[str, ...] = ("frangi", "sato", "ved", "vedm")


def vessel_enhancement(
    volume: np.ndarray,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    method: str = "frangi",
    sigma_min: float = 0.5,
    sigma_max: float = 3.0,
    sigma_steps: int = 5,
    bright_vessels: bool = True,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 5.0,
    iterations: int = 1,
) -> np.ndarray:
    """Multiscale vesselness of an ``(X, Y, Z)`` volume (sigmas in mm).

    ``frangi`` (alpha/beta/gamma), ``sato`` (alpha1/alpha2 fixed at VMTK defaults), ``ved`` /
    ``vedm`` (anisotropic diffusion along the vessels, *iterations* steps; returns the enhanced
    image rather than a [0, 1] response).
    """
    if method not in VESSELNESS_METHODS:
        raise ValueError(f"method must be one of {VESSELNESS_METHODS}")
    img = image_from_array(np.asarray(volume, dtype=np.float32), spacing)
    out = run_script(
        "vmtkImageVesselEnhancement",
        Image=img,
        Method=method,
        SigmaMin=float(sigma_min),
        SigmaMax=float(sigma_max),
        NumberOfSigmaSteps=int(sigma_steps),
        BrightObject=1 if bright_vessels else 0,
        Alpha=float(alpha),
        Beta=float(beta),
        Gamma=float(gamma),
        NumberOfIterations=int(iterations),
    ).Image
    return array_from_image(out).astype(np.float32)


__all__ = ["VESSELNESS_METHODS", "vessel_enhancement"]
