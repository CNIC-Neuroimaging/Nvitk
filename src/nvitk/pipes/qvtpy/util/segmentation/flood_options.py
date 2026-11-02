"""Blood-flood settings of qvtpy stage 4 / 4t: defaults, click options and the matching argv.

Kept apart from :mod:`~nvitk.pipes.qvtpy.util.segmentation.vessel_cd_segmentation` (which does the
flood) so the CLIs (stage 4, stage 4t, ``nvitk-qvtpy``) can declare the options without importing
the segmentation code. The defaults live only here.
"""

from __future__ import annotations

from typing import Any, Callable

import click

#: Settings name -> default. The CLI options are the names with ``-`` for ``_``.
FLOOD_DEFAULTS: dict[str, Any] = {
    "flood_expand": True,
    "flood_normalise_cd": True,
    "flood_hyst_low_factor": 3.0,
    "flood_hyst_high_factor": 0.5,
    "flood_thicken_iter": 0,
    "flood_max_image_frac": 0.08,
    "flood_lr_halfspace_slack": 2,
    "flood_near_radius": 2,
    "flood_aca_step": 2,
    "flood_min_island_mm3": 15.0,
    "flood_protrusion_filter": False,
}

_HELP: dict[str, str] = {
    "flood_expand": "Blood flood: the ACAs / MCAs / PCAs flood a Frangi + hysteresis vessel tree of the CD "
                    "from their threshold masks (comms and venous sinuses only fill their own lumen; the ICAs "
                    "and basilar region-grow instead); what an A1 / M1 / P1 floods beyond its lumen becomes "
                    "its A2 / M2 / P2.",
    "flood_normalise_cd": "Divide the CD by the subject's blood signal (median CD on the ICA / basilar "
                          "centerlines) before the vesselness, so the vessel tree's thresholds mean the "
                          "same across subjects (else: each volume rescaled to its own 99.5th percentile).",
    "flood_hyst_low_factor": "GMM hysteresis low factor (higher -> thinner tree).",
    "flood_hyst_high_factor": "GMM hysteresis high factor (lower -> more seeds, thicker tree).",
    "flood_thicken_iter": "Dilations of the tree inside a high-CD gate (0 = thinnest).",
    "flood_max_image_frac": "Cap on the voxels the growing arteries claim (fraction of the image).",
    "flood_lr_halfspace_slack": "Left/right midline slack (voxels) of the paired MCAs / PCAs.",
    "flood_near_radius": "How far (voxels) the communicating arteries and venous sinuses flood from their own "
                         "mask.",
    "flood_aca_step": "ACAs: voxels each side grows per turn when the two ACAs share their flood, right and "
                      "left in turns.",
    "flood_min_island_mm3": "After the flood, each label's pieces smaller than this (mm³) are dropped.",
    "flood_protrusion_filter": "Remove wart-like surface bumps after the flood (pre-flood voxels protected).",
}


def flood_settings(**kwargs: Any) -> dict[str, Any]:
    """:data:`FLOOD_DEFAULTS` overridden by the non-``None`` *kwargs* (unknown names rejected)."""
    unknown = set(kwargs) - set(FLOOD_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown flood settings: {sorted(unknown)}")
    return {k: (kwargs[k] if kwargs.get(k) is not None else v) for k, v in FLOOD_DEFAULTS.items()}


def flood_cli_args(settings: dict[str, Any]) -> list[str]:
    """The ``--flood-*`` argv of *settings* (:func:`flood_settings`)."""
    out: list[str] = []
    for key, value in flood_settings(**settings).items():
        opt = key.replace("_", "-")
        if isinstance(FLOOD_DEFAULTS[key], bool):
            out.append(f"--{opt}" if value else f"--no-{opt}")
        else:
            out += [f"--{opt}", str(value)]
    return out


def flood_click_options(help_prefix: str = "") -> Callable[[Any], Any]:
    """Decorator adding the blood-flood options (:data:`FLOOD_DEFAULTS`) to a click command; each
    help text starts with *help_prefix* (``"Stage4/4t: "`` in ``nvitk-qvtpy``)."""

    def decorate(func: Any) -> Any:
        for key in reversed(list(FLOOD_DEFAULTS)):
            default = FLOOD_DEFAULTS[key]
            opt = key.replace("_", "-")
            if isinstance(default, bool):
                func = click.option(f"--{opt}/--no-{opt}", default=default, show_default=True,
                                    help=help_prefix + _HELP[key])(func)
            else:
                func = click.option(f"--{opt}", type=type(default), default=default, show_default=True,
                                    help=help_prefix + _HELP[key])(func)
        return func

    return decorate


__all__ = ["FLOOD_DEFAULTS", "flood_cli_args", "flood_click_options", "flood_settings"]
