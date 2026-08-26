"""
Named CT display windows (Hounsfield unit level/width presets).

Description
-----------
CT is the one modality with a physically calibrated intensity scale: Hounsfield units are fixed
by definition (−1000 air, 0 water), so a *fixed* display window means the same tissue contrast on
every scan from every scanner. That is why radiology works in named windows — "brain", "bone",
"lung" — rather than per-image auto-contrast, which changes what you are looking at from case to
case.

A window is stored as ``(level, width)`` in HU, the convention used on the scanner console and in
PACS, and converted to the ``(low, high)`` bounds a display library wants::

    low  = level - width / 2
    high = level + width / 2

Conventions
-----------
Levels and widths are **Hounsfield units**, always. Applying these to non-CT data is meaningless
— MR intensities are arbitrary units with no fixed zero — so callers should check modality
first; :func:`suggest_window` does.

This module is display-only: it never modifies voxels. For *intensity* rescaling that feeds a
model, see :mod:`nvitk.normalization.intensity`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CTWindow:
    """A named CT display window in Hounsfield units."""

    key: str
    title: str
    level: float
    width: float
    description: str = ""

    @property
    def limits(self) -> tuple[float, float]:
        """``(low, high)`` HU bounds — what a viewer's contrast limits expect."""
        half = self.width / 2.0
        return (self.level - half, self.level + half)

    @property
    def label(self) -> str:
        """Display string, e.g. ``"Brain (L 40 / W 80)"``."""
        return f"{self.title} (L {self.level:g} / W {self.width:g})"

    @classmethod
    def from_limits(cls, key: str, title: str, low: float, high: float, description: str = "") -> "CTWindow":
        """Build a window from ``(low, high)`` bounds rather than level/width."""
        return cls(
            key=key,
            title=title,
            level=(float(low) + float(high)) / 2.0,
            width=abs(float(high) - float(low)),
            description=description,
        )


#: Standard diagnostic windows. Values follow common radiological practice; they are display
#: conventions rather than a standard with a single authoritative source, so sites do vary.
CT_WINDOWS: dict[str, CTWindow] = {
    w.key: w
    for w in (
        CTWindow("brain", "Brain", 40, 80,
                 "Grey/white matter differentiation. The default for head CT."),
        CTWindow("stroke", "Stroke / posterior fossa", 35, 30,
                 "Narrow window exaggerating grey-white contrast for early infarct."),
        CTWindow("subdural", "Subdural", 70, 200,
                 "Wide enough to separate an extra-axial collection from adjacent bone."),
        CTWindow("bone", "Bone", 500, 2000,
                 "Cortical detail and fractures; soft tissue is deliberately flat."),
        CTWindow("angio", "CT angiography", 300, 600,
                 "Contrast-filled lumen against vessel wall and surrounding tissue."),
        CTWindow("soft_tissue", "Soft tissue", 50, 400, "General abdominal/soft-tissue review."),
        CTWindow("mediastinum", "Mediastinum", 50, 350, "Mediastinal structures and vessels."),
        CTWindow("lung", "Lung", -600, 1500, "Parenchyma and airways."),
        CTWindow("liver", "Liver", 60, 150,
                 "Narrow window raising the conspicuity of low-contrast lesions."),
        CTWindow("full", "Full range", 1024, 4096,
                 "The whole 12-bit HU range; no tissue emphasis."),
    )
}

#: Window applied when nothing else is chosen for a head CT.
DEFAULT_WINDOW_KEY: str = "brain"

#: Windows most relevant to the vascular work this toolkit is built around, listed first in
#: pickers so the common choice is not buried among abdominal presets.
PREFERRED_ORDER: tuple[str, ...] = (
    "brain", "angio", "stroke", "subdural", "bone", "soft_tissue",
    "mediastinum", "lung", "liver", "full",
)


def window_keys() -> list[str]:
    """Registered keys, vascular/neuro windows first."""
    known = [k for k in PREFERRED_ORDER if k in CT_WINDOWS]
    return known + sorted(k for k in CT_WINDOWS if k not in known)


def get_window(key: str) -> CTWindow:
    """Look up a window by key.

    Raises
    ------
    KeyError
        Listing the valid keys — a typo should not silently fall back to a different window.
    """
    try:
        return CT_WINDOWS[key]
    except KeyError:
        raise KeyError(
            f"Unknown CT window {key!r}. Valid: {', '.join(window_keys())}."
        ) from None


def limits_for(key: str) -> tuple[float, float]:
    """``(low, high)`` HU bounds for the window *key*."""
    return get_window(key).limits


def window_from_limits(low: float, high: float, *, tolerance: float = 1.0) -> CTWindow | None:
    """The registered window matching ``(low, high)``, or ``None`` if it is a custom range.

    Lets a picker show which preset is active after limits were set from elsewhere, and fall
    back to "Custom" honestly rather than snapping to the nearest preset.
    """
    for window in CT_WINDOWS.values():
        window_low, window_high = window.limits
        if abs(window_low - low) <= tolerance and abs(window_high - high) <= tolerance:
            return window
    return None


def looks_like_hounsfield(minimum: float, maximum: float) -> bool:
    """Whether an intensity range plausibly holds Hounsfield units.

    CT reconstructions span roughly −1024 to a few thousand HU, and crucially go **well below
    zero** — which MR magnitude data, being non-negative, never does. That sign asymmetry is the
    cheap discriminator; it is not a modality detector, and callers with real modality metadata
    should prefer it.
    """
    return minimum < -200.0 and maximum > 100.0


def suggest_window(
    modality: str | None = None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> str | None:
    """Suggest a window key, or ``None`` when windowing does not apply.

    Prefers explicit *modality* metadata; falls back to the HU heuristic when the intensity
    range is available. Returns ``None`` for anything not recognisably CT, so a caller can leave
    an MR layer's contrast alone rather than imposing a meaningless HU range on it.
    """
    if modality is not None:
        key = str(modality).strip().lower()
        if key in ("ct", "cta", "ct_angio", "ctangio"):
            return "angio" if key != "ct" else DEFAULT_WINDOW_KEY
        if key:
            return None
    if minimum is not None and maximum is not None and looks_like_hounsfield(minimum, maximum):
        return DEFAULT_WINDOW_KEY
    return None


__all__ = [
    "CT_WINDOWS",
    "DEFAULT_WINDOW_KEY",
    "PREFERRED_ORDER",
    "CTWindow",
    "get_window",
    "limits_for",
    "looks_like_hounsfield",
    "suggest_window",
    "window_from_limits",
    "window_keys",
]
