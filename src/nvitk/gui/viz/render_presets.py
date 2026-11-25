"""3D rendering presets: one click to a sensible volume rendering per modality.

A :class:`RenderPreset` bundles what Napari's layer controls spread over half a dozen widgets:
rendering mode (MIP, attenuated MIP, MinIP, iso-surface, translucent), colormap, contrast (in
the modality's units, e.g. Hounsfield, or as percentiles of the volume), gamma, attenuation,
iso threshold, opacity and blending. Built-in presets cover neuro / body / cardiac CT, CT
angiography, structural MRI, MR angiography (TOF), SWI, 4D-flow magnitude, PET and label maps;
:func:`save_user_preset` stores the current look of a layer as a preset of your own
(``render_presets.json`` beside ``gui.json``).

:func:`apply_preset` works on several layers at once (the selected ones), and
:func:`suggest_preset` guesses one from a layer's name and metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RenderPreset:
    """How to render one kind of volume in 3D."""

    key: str
    label: str
    group: str
    rendering: str = "mip"
    colormap: str = "gray"
    #: ``("hu", lo, hi)`` / ``("abs", lo, hi)`` in the image's own units, or ``("pct", lo, hi)``
    #: percentiles of the volume.
    contrast: tuple[str, float, float] = ("pct", 1.0, 99.5)
    gamma: float = 1.0
    attenuation: float | None = None
    #: Iso threshold, in the same convention as *contrast* (``("hu", 300)`` / ``("pct", 99)``).
    iso: tuple[str, float] | None = None
    opacity: float = 1.0
    blending: str = "translucent_no_depth"
    interpolation: str = "linear"
    #: For label layers (``iso_categorical`` / ``translucent``) instead of the above.
    labels: bool = False
    modalities: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


BUILTIN_PRESETS: tuple[RenderPreset, ...] = (
    RenderPreset("ct_brain", "Neuro CT: brain (MIP)", "CT", "mip", "gray", ("hu", 0, 80),
                 modalities=("ct",), description="Brain window 0-80 HU, maximum intensity."),
    RenderPreset("ct_soft", "Body CT: soft tissue", "CT", "translucent", "gray", ("hu", -160, 240),
                 opacity=0.8, modalities=("ct",), description="Soft-tissue window, translucent."),
    RenderPreset("ct_bone", "CT: bone surface", "CT", "iso", "bone", ("hu", -200, 1500), iso=("hu", 300),
                 modalities=("ct",), description="Iso-surface at 300 HU."),
    RenderPreset("ct_cardiac", "Cardiac CT", "CT", "attenuated_mip", "magma", ("hu", 0, 700), attenuation=0.08,
                 modalities=("ct", "cta"), description="Contrast-filled chambers and vessels, attenuated MIP."),
    RenderPreset("cta_vessels", "CTA: vessels (attenuated MIP)", "CT", "attenuated_mip", "inferno", ("hu", 120, 600),
                 attenuation=0.05, modalities=("cta",), description="Opacified vessels over soft tissue."),
    RenderPreset("cta_iso", "CTA: vessel surface", "CT", "iso", "red", ("hu", 0, 800), iso=("hu", 180),
                 modalities=("cta",), description="Iso-surface at 180 HU (lumen with contrast)."),
    RenderPreset("mra_mip", "MRA / TOF: MIP", "MRI", "mip", "gray", ("pct", 50.0, 99.8),
                 modalities=("tof", "mra"), description="Classic TOF maximum-intensity projection."),
    RenderPreset("mra_iso", "MRA / TOF: vessel surface", "MRI", "iso", "red", ("pct", 50.0, 99.9), iso=("pct", 99.0),
                 modalities=("tof", "mra"), description="Iso-surface at the 99th percentile."),
    RenderPreset("mri_struct", "MRI: structural (attenuated MIP)", "MRI", "attenuated_mip", "gray", ("pct", 1.0, 99.5),
                 attenuation=0.25, modalities=("t1", "t2", "flair", "smri"), description="T1 / T2 / FLAIR overview."),
    RenderPreset("mri_minip", "SWI: minimum intensity", "MRI", "minip", "gray", ("pct", 1.0, 60.0),
                 modalities=("swi",), description="Veins and microbleeds (MinIP)."),
    RenderPreset("flow_mag", "4D flow: magnitude", "MRI", "mip", "viridis", ("pct", 30.0, 99.9),
                 modalities=("4dflow",), description="Magnitude / PC-MRA, MIP."),
    RenderPreset("pet_hot", "PET: SUV (hot)", "PET", "mip", "hot", ("abs", 0.0, 6.0),
                 modalities=("pet",), description="SUV 0-6, hot colormap."),
    RenderPreset("pet_inverse", "PET: inverse MIP", "PET", "mip", "gray_r", ("pct", 0.0, 99.5),
                 blending="translucent", modalities=("pet",), description="Radiology-style inverted MIP."),
    RenderPreset("labels_surface", "Labels: surfaces", "Labels", labels=True, rendering="iso_categorical",
                 description="Each label drawn as an opaque surface."),
    RenderPreset("labels_translucent", "Labels: translucent", "Labels", labels=True, rendering="translucent",
                 opacity=0.6, description="Labels as a translucent volume."),
)

#: Tokens in a layer name / modality that point at a preset.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cta", "angio", "ccta"), "cta_vessels"),
    (("cardiac", "heart", "coronar"), "ct_cardiac"),
    (("pet", "suv", "fdg"), "pet_hot"),
    (("tof", "mra", "angiography"), "mra_mip"),
    (("swi", "qsm", "swan"), "mri_minip"),
    (("4dflow", "4d_flow", "pcmra", "magnitude", "_m.nii"), "flow_mag"),
    (("t1", "t2", "flair", "mprage", "dixon"), "mri_struct"),
    (("ct",), "ct_soft"),
)


def _user_presets_path() -> Path:
    from nvitk.gui.core.prefs import prefs_path

    return prefs_path().with_name("render_presets.json")


def user_presets() -> list[RenderPreset]:
    """Presets saved with :func:`save_user_preset`."""
    try:
        raw = json.loads(_user_presets_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 (none saved, or unreadable)
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        try:
            item = dict(item)
            item["contrast"] = tuple(item.get("contrast") or ("pct", 1.0, 99.5))
            item["iso"] = tuple(item["iso"]) if item.get("iso") else None
            item["modalities"] = tuple(item.get("modalities") or ())
            out.append(RenderPreset(**item))
        except Exception:  # noqa: BLE001
            continue
    return out


def all_presets() -> list[RenderPreset]:
    """Built-in presets, then the user's."""
    return [*BUILTIN_PRESETS, *user_presets()]


def get_preset(key: str) -> RenderPreset | None:
    """The preset named *key*."""
    return next((p for p in all_presets() if p.key == key), None)


def _sample(layer: Any) -> np.ndarray:
    from nvitk.core.array import to_numpy

    data = getattr(layer, "data", None)
    arr = to_numpy(data[0] if getattr(layer, "multiscale", False) else data)
    flat = np.asarray(arr).reshape(-1)
    if flat.size > 500_000:
        flat = flat[:: flat.size // 500_000]
    return flat[np.isfinite(flat)]


def _range(preset: RenderPreset, values: np.ndarray) -> tuple[float, float]:
    mode, lo, hi = preset.contrast
    if mode == "pct" or values.size == 0:
        if values.size == 0:
            return 0.0, 1.0
        lo_v, hi_v = (float(v) for v in np.percentile(values, (lo, hi)))
    else:
        lo_v, hi_v = float(lo), float(hi)
        vmin, vmax = float(values.min()), float(values.max())
        if values.size and (hi_v < vmin or lo_v > vmax):
            # A unit window that misses the data entirely (an MR volume given a CT preset):
            # fall back to the volume's own robust range rather than a black render.
            lo_v, hi_v = (float(v) for v in np.percentile(values, (1.0, 99.5)))
    return (lo_v, hi_v) if hi_v > lo_v else (lo_v, lo_v + 1.0)


def apply_preset(layers: Iterable[Any], preset: RenderPreset | str) -> list[str]:
    """Apply *preset* to every image (or, for label presets, labels) layer in *layers*.

    Returns the names of the layers it changed; others (points, meshes, a label preset on an
    image) are skipped.
    """
    if isinstance(preset, str):
        found = get_preset(preset)
        if found is None:
            raise KeyError(f"Unknown render preset {preset!r}")
        preset = found
    changed: list[str] = []
    for layer in layers:
        kind = type(layer).__name__
        if preset.labels:
            if kind != "Labels":
                continue
            _set(layer, "rendering", preset.rendering)
            _set(layer, "opacity", preset.opacity)
            changed.append(str(layer.name))
            continue
        if kind != "Image" or getattr(layer, "rgb", False):
            continue
        values = _sample(layer)
        lo, hi = _range(preset, values)
        _set(layer, "colormap", preset.colormap)
        _set(layer, "contrast_limits", (lo, hi))
        _set(layer, "gamma", preset.gamma)
        _set(layer, "rendering", preset.rendering)
        _set(layer, "opacity", preset.opacity)
        _set(layer, "blending", preset.blending)
        _set(layer, "interpolation3d", preset.interpolation)
        if preset.attenuation is not None:
            _set(layer, "attenuation", float(preset.attenuation))
        if preset.iso is not None:
            mode, value = preset.iso
            threshold = float(np.percentile(values, value)) if mode == "pct" and values.size else float(value)
            _set(layer, "iso_threshold", threshold)
        try:
            layer.metadata["nvitk_render_preset"] = preset.key
        except Exception:  # noqa: BLE001
            pass
        changed.append(str(layer.name))
    return changed


def _set(layer: Any, attr: str, value: Any) -> None:
    try:
        setattr(layer, attr, value)
    except Exception:  # noqa: BLE001 (an attribute this Napari or layer type lacks)
        pass


def suggest_preset(layer: Any) -> str | None:
    """A preset key for *layer* from its name, source and modality metadata (``None``: no idea)."""
    if layer is None:
        return None
    if type(layer).__name__ == "Labels":
        return "labels_surface"
    meta = getattr(layer, "metadata", {}) or {}
    nmeta = meta.get("nvitk_metadata") if isinstance(meta.get("nvitk_metadata"), dict) else {}
    blob = " ".join(
        str(x).lower()
        for x in (
            getattr(layer, "name", ""),
            getattr(getattr(layer, "source", None), "path", "") or "",
            meta.get("modality", ""), nmeta.get("modality", ""), nmeta.get("series_description", ""),
        )
    )
    for tokens, key in _HINTS:
        if any(t in blob for t in tokens):
            return key
    try:
        from nvitk.viz.ct_windows import looks_like_hounsfield

        values = _sample(layer)
        if values.size and looks_like_hounsfield(float(values.min()), float(values.max())):
            return "ct_soft"
    except Exception:  # noqa: BLE001
        pass
    return "mri_struct"


def preset_from_layer(layer: Any, key: str, label: str) -> RenderPreset:
    """A preset reproducing *layer*'s current look (absolute contrast, as set now)."""
    lo, hi = (float(v) for v in getattr(layer, "contrast_limits", (0.0, 1.0)))
    cmap = getattr(getattr(layer, "colormap", None), "name", "gray")
    iso = getattr(layer, "iso_threshold", None)
    rendering = str(getattr(layer, "rendering", "mip"))
    return RenderPreset(
        key=key, label=label, group="Mine", rendering=rendering, colormap=str(cmap),
        contrast=("hu", lo, hi), gamma=float(getattr(layer, "gamma", 1.0)),
        attenuation=float(getattr(layer, "attenuation", 0.05)) if rendering == "attenuated_mip" else None,
        iso=("hu", float(iso)) if rendering == "iso" and iso is not None else None,
        opacity=float(getattr(layer, "opacity", 1.0)), blending=str(getattr(layer, "blending", "translucent_no_depth")),
    )


def save_user_preset(preset: RenderPreset) -> Path:
    """Add (or replace) *preset* among the user's presets; returns the file written."""
    path = _user_presets_path()
    kept = [p for p in user_presets() if p.key != preset.key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in (*kept, preset)], indent=2), encoding="utf-8")
    return path


def build_preset_menu(menu: Any, layers_getter: Any, *, on_applied: Any = None) -> None:
    """Fill a ``QMenu`` with the presets (grouped), applying the chosen one to ``layers_getter()``.

    The first entry is the suggestion for the active layer; the last saves the active layer's
    current look as a preset.
    """
    from qtpy.QtWidgets import QInputDialog

    def _apply(key: str) -> None:
        layers = list(layers_getter() or [])
        changed = apply_preset(layers, key)
        if on_applied is not None:
            on_applied(key, changed)

    layers = list(layers_getter() or [])
    suggestion = suggest_preset(layers[0]) if layers else None
    if suggestion:
        found = get_preset(suggestion)
        if found is not None:
            action = menu.addAction(f"Suggested: {found.label}")
            action.triggered.connect(lambda _=False, k=found.key: _apply(k))
            menu.addSeparator()
    groups: dict[str, list[RenderPreset]] = {}
    for preset in all_presets():
        groups.setdefault(preset.group, []).append(preset)
    for group, items in groups.items():
        sub = menu.addMenu(group)
        for preset in items:
            action = sub.addAction(preset.label)
            action.setToolTip(preset.description)
            action.triggered.connect(lambda _=False, k=preset.key: _apply(k))
    menu.addSeparator()
    save = menu.addAction("Save the active layer's look as a preset…")

    def _save() -> None:
        current = list(layers_getter() or [])
        if not current:
            return
        name, ok = QInputDialog.getText(menu.parentWidget(), "Save render preset", "Preset name:")
        if ok and str(name).strip():
            key = "user_" + "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
            save_user_preset(preset_from_layer(current[0], key, str(name).strip()))

    save.triggered.connect(_save)
    save.setEnabled(bool(layers))


__all__ = [
    "BUILTIN_PRESETS",
    "RenderPreset",
    "all_presets",
    "apply_preset",
    "build_preset_menu",
    "get_preset",
    "preset_from_layer",
    "save_user_preset",
    "suggest_preset",
    "user_presets",
]
