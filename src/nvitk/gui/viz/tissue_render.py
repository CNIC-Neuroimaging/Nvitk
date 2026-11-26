"""CT as body tissues: a Hounsfield transfer function, tissue layers and a tissue label map.

A CT volume rendered *translucent* takes its colour **and opacity** per sample from the colormap;
:func:`tissue_colormap` builds that colormap from tissue classes, each a Hounsfield range with a
colour and an opacity (fat yellow and faint, muscle red, contrast blood bright red, bone ivory and
dense...), with soft edges (*softness*, in HU) and an overall *density* that scales every opacity.
A tissue's opacity is what **1 cm** of it absorbs, turned into a per-sample opacity from the
volume's voxel size, so the look does not change with the resolution.
The result looks like an anatomical cut-away rather than a grey MIP.

* :data:`TISSUES`: the default classes; :data:`TISSUE_PRESETS`: ready-made looks (realistic body,
  skin and bone, vessels and bone, lungs and airways, bone only).
* :func:`apply_tissue_render` sets a CT image layer's colormap, contrast (the HU window the classes
  span), rendering (translucent) and blending; the settings ride on the layer
  (``metadata["nvitk_tissues"]``) so the panel reopens on them.
* :func:`make_tissue_layers` adds one layer per enabled tissue, **sharing the CT's data** (no copy),
  each showing only its HU band, as a translucent volume or an iso-surface, in a layer folder
  ``Tissues/<CT name>``: toggle tissues on and off, change one tissue's opacity, from the layer list.
* :func:`make_tissue_labels` classifies the voxels into a Labels layer (1 = first enabled tissue...),
  coloured like the tissues, for 2D views and masks.
* :class:`TissueRenderPanel`: the controls (a row per tissue: on, HU from / to, colour, opacity;
  density, softness, gamma, preset, the layer buttons), opened as a pinnable pop-over from the
  render-preset menu (*Body tissues: adjust…*).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

META_KEY = "nvitk_tissues"
PREFS_KEY = "render.tissue_defaults"


@dataclass
class Tissue:
    """One tissue class: a Hounsfield range drawn in one colour."""

    name: str
    lo: float
    hi: float
    color: str
    opacity: float
    enabled: bool = True
    #: Drawn as an iso-surface (at *lo*) when split into its own layer.
    surface: bool = False


#: Default classes, by rising HU (overlaps blend: the more opaque tissue leads). Opacity: the
#: fraction 1 cm of the tissue absorbs.
TISSUES: tuple[Tissue, ...] = (
    Tissue("Lung", -950, -500, "#e8a7a2", 0.03, enabled=False),
    Tissue("Fat", -190, -40, "#e9c46a", 0.05),
    Tissue("Fluid", -10, 18, "#9fc5e8", 0.03, enabled=False),
    Tissue("Soft tissue / muscle", 20, 110, "#b4483a", 0.1),
    Tissue("Contrast blood", 150, 450, "#e0323c", 0.7, enabled=False),
    Tissue("Cancellous bone", 250, 700, "#e8dcc0", 0.7, surface=True),
    Tissue("Cortical bone", 700, 3000, "#fffaf0", 0.97, surface=True),
)

#: Napari's ray step, in voxels (vispy's ``relative_step_size`` default).
RAY_STEP = 0.8

#: Ready-made looks: tissue name -> overrides (``enabled`` / ``opacity`` / ``lo`` / ``hi``), and
#: the panel's globals.
TISSUE_PRESETS: dict[str, dict[str, Any]] = {
    "realistic": {"label": "Realistic body", "density": 0.6, "softness": 30.0, "tissues": {}},
    "skin_bone": {"label": "Skin and bone", "density": 0.7, "softness": 25.0, "tissues": {
        "Fat": {"enabled": False}, "Soft tissue / muscle": {"opacity": 0.03, "lo": -100, "hi": 60}}},
    "vessels_bone": {"label": "Vessels and bone (CTA)", "density": 0.8, "softness": 30.0, "tissues": {
        "Fat": {"enabled": False}, "Soft tissue / muscle": {"opacity": 0.02},
        "Contrast blood": {"enabled": True}}},
    "lungs": {"label": "Lungs and airways", "density": 0.6, "softness": 40.0, "tissues": {
        "Lung": {"enabled": True, "opacity": 0.1}, "Fat": {"enabled": False},
        "Soft tissue / muscle": {"opacity": 0.03}}},
    "bone": {"label": "Bone only", "density": 1.0, "softness": 25.0, "tissues": {
        "Fat": {"enabled": False}, "Soft tissue / muscle": {"enabled": False}}},
}


def default_settings(preset: str = "realistic") -> dict[str, Any]:
    """Settings for *preset*: ``{"tissues": [Tissue dicts], "density", "softness", "gamma", "preset"}``."""
    spec = TISSUE_PRESETS.get(preset, TISSUE_PRESETS["realistic"])
    tissues = []
    for tissue in TISSUES:
        item = asdict(tissue)
        item.update(spec.get("tissues", {}).get(tissue.name, {}))
        tissues.append(item)
    return {"tissues": tissues, "density": float(spec.get("density", 0.6)), "softness": float(spec.get("softness", 30.0)),
            "gamma": 1.0, "preset": preset}


def _rgb(color: str) -> tuple[float, float, float]:
    q = QColor(color)
    return q.redF(), q.greenF(), q.blueF()


def hu_window(settings: dict[str, Any]) -> tuple[float, float]:
    """The HU range the enabled tissues span (plus the soft edges): the layer's contrast limits."""
    on = [t for t in settings["tissues"] if t.get("enabled")]
    if not on:
        return -1000.0, 1000.0
    pad = float(settings.get("softness", 30.0))
    return min(float(t["lo"]) for t in on) - pad, max(float(t["hi"]) for t in on) + pad


def transfer_function(settings: dict[str, Any], *, n: int = 512, only: str | None = None, step_mm: float = 1.0
                      ) -> tuple[np.ndarray, tuple[float, float]]:
    """RGBA samples (``n`` × 4) over the HU window, and that window.

    Each tissue's opacity (per cm, times *density*) becomes the opacity of one ray sample
    *step_mm* long, ramped in and out over *softness* HU at its edges; where tissues overlap,
    colours mix weighted by opacity and the opacity is the largest. *only* keeps one tissue (a
    tissue layer's band).
    """
    lo_w, hi_w = hu_window(settings) if only is None else _single_window(settings, only)
    hu = np.linspace(lo_w, hi_w, n)
    soft = max(float(settings.get("softness", 30.0)), 1e-3)
    density = float(settings.get("density", 0.6))
    gamma = float(settings.get("gamma", 1.0)) or 1.0
    rgb_acc = np.zeros((n, 3))
    w_acc = np.zeros(n)
    alpha = np.zeros(n)
    for tissue in settings["tissues"]:
        if not tissue.get("enabled") or (only is not None and tissue["name"] != only):
            continue
        lo, hi = float(tissue["lo"]), float(tissue["hi"])
        ramp_in = np.clip((hu - (lo - soft / 2)) / soft, 0, 1)
        ramp_out = np.clip(((hi + soft / 2) - hu) / soft, 0, 1)
        per_cm = np.clip(float(tissue["opacity"]) * density, 0.0, 0.9999) ** gamma
        per_sample = 1.0 - (1.0 - per_cm) ** (max(step_mm, 1e-3) / 10.0)
        a = np.minimum(ramp_in, ramp_out) * per_sample
        rgb_acc += np.outer(a, _rgb(tissue["color"]))
        w_acc += a
        alpha = np.maximum(alpha, a)
    rgb = np.where(w_acc[:, None] > 0, rgb_acc / np.maximum(w_acc[:, None], 1e-9), 0.0)
    return np.column_stack([rgb, np.clip(alpha, 0, 1)]).astype(np.float32), (float(lo_w), float(hi_w))


def _single_window(settings: dict[str, Any], name: str) -> tuple[float, float]:
    tissue = next(t for t in settings["tissues"] if t["name"] == name)
    pad = float(settings.get("softness", 30.0))
    return float(tissue["lo"]) - pad, float(tissue["hi"]) + pad


def ray_step_mm(layer: Any) -> float:
    """The length of one ray sample through *layer*, in world units (mm): voxel size × step."""
    try:
        scale = np.abs(np.asarray(layer.scale, dtype=float))[-3:]
        linear = np.asarray(layer.affine.linear_matrix, dtype=float)[-3:, -3:]
        spacing = scale * np.linalg.norm(linear, axis=0)
        return float(np.mean(spacing)) * RAY_STEP
    except Exception:  # noqa: BLE001
        return RAY_STEP


def tissue_colormap(settings: dict[str, Any], *, only: str | None = None, name: str = "nvitk_tissues",
                    step_mm: float = 1.0) -> tuple[Any, tuple[float, float]]:
    """A Napari ``Colormap`` (RGBA, opacity included) and the contrast limits it is drawn over."""
    from napari.utils.colormaps import Colormap

    samples, window = transfer_function(settings, only=only, step_mm=step_mm)
    return Colormap(colors=samples, controls=np.linspace(0.0, 1.0, len(samples)), name=name), window


def is_ct_layer(layer: Any) -> bool:
    """An image layer whose values look like Hounsfield units."""
    if type(layer).__name__ != "Image" or getattr(layer, "rgb", False):
        return False
    try:
        from nvitk.gui.viz.render_presets import _sample
        from nvitk.viz.ct_windows import looks_like_hounsfield

        values = _sample(layer)
        return bool(values.size) and bool(looks_like_hounsfield(float(values.min()), float(values.max())))
    except Exception:  # noqa: BLE001
        return False


def layer_settings(layer: Any) -> dict[str, Any]:
    """The tissue settings stored on *layer*, else the user's default, else the realistic preset."""
    stored = (getattr(layer, "metadata", {}) or {}).get(META_KEY)
    if isinstance(stored, dict) and stored.get("tissues"):
        return copy.deepcopy(stored)
    try:
        from nvitk.gui.core.prefs import load_prefs

        saved = load_prefs().get(PREFS_KEY)
        if isinstance(saved, dict) and saved.get("tissues"):
            return copy.deepcopy(saved)
    except Exception:  # noqa: BLE001
        pass
    return default_settings()


def apply_tissue_render(layer: Any, settings: dict[str, Any] | None = None) -> bool:
    """Draw *layer* (a CT image) through the tissue transfer function; returns whether it could."""
    if type(layer).__name__ != "Image" or getattr(layer, "rgb", False):
        return False
    from nvitk.gui.viz.render_presets import remember_default

    remember_default(layer)
    settings = copy.deepcopy(settings) if settings is not None else layer_settings(layer)
    cmap, window = tissue_colormap(settings, name=f"nvitk_tissues_{id(layer)}", step_mm=ray_step_mm(layer))
    for attr, value in (("rendering", "translucent"), ("colormap", cmap), ("contrast_limits", window),
                        ("gamma", 1.0), ("blending", "translucent_no_depth"), ("opacity", 1.0),
                        ("interpolation3d", "linear")):
        try:
            setattr(layer, attr, value)
        except Exception:  # noqa: BLE001
            pass
    try:
        layer.metadata[META_KEY] = settings
        layer.metadata["nvitk_render_preset"] = "ct_tissues"
    except Exception:  # noqa: BLE001
        pass
    return True


def _tissue_children(viewer: Any, source: Any) -> list[Any]:
    return [ly for ly in viewer.layers
            if (getattr(ly, "metadata", {}) or {}).get("nvitk_tissue_of") == str(getattr(source, "name", ""))]


def make_tissue_layers(viewer: Any, source: Any, settings: dict[str, Any] | None = None) -> list[Any]:
    """One layer per enabled tissue, sharing *source*'s data, in folder ``Tissues/<name>``.

    Layers made earlier for the same source are updated in place (and those of tissues now off
    removed). Volumes draw their band translucent; ``surface`` tissues draw an iso-surface at
    their lower bound in their colour.
    """
    from napari.utils.colormaps import Colormap

    from nvitk.gui.core.layer_folders import _set_folder
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    settings = copy.deepcopy(settings) if settings is not None else layer_settings(source)
    folder = f"Tissues/{source.name}"
    existing = {(ly.metadata or {}).get("nvitk_tissue"): ly for ly in _tissue_children(viewer, source)}
    made = []
    for tissue in settings["tissues"]:
        name = tissue["name"]
        layer = existing.pop(name, None)
        if not tissue.get("enabled"):
            if layer is not None:
                viewer.layers.remove(layer)
            continue
        if tissue.get("surface"):
            r, g, b = _rgb(tissue["color"])
            cmap = Colormap(colors=np.array([[r, g, b, 1.0], [r, g, b, 1.0]]), name=f"nvitk_tissue_{name}_{id(source)}")
            lo, hi = float(tissue["lo"]) - 1.0, float(tissue["hi"])
            props = {"rendering": "iso", "iso_threshold": float(tissue["lo"]), "colormap": cmap,
                     "contrast_limits": (lo, max(hi, lo + 1.0)), "opacity": float(np.clip(tissue["opacity"] * 1.05, 0.05, 1.0)),
                     "blending": "translucent"}
        else:
            cmap, window = tissue_colormap(settings, only=name, name=f"nvitk_tissue_{name}_{id(source)}",
                                           step_mm=ray_step_mm(source))
            props = {"rendering": "translucent", "colormap": cmap, "contrast_limits": window, "opacity": 1.0,
                     "blending": "translucent_no_depth"}
        if layer is None:
            layer = viewer.add_image(source.data, name=f"{name} · {source.name}", **layer_spatial_kwargs(source),
                                     colormap=props.pop("colormap"), contrast_limits=props.pop("contrast_limits"))
            layer.metadata["nvitk_tissue_of"] = str(source.name)
            layer.metadata["nvitk_tissue"] = name
            _set_folder(layer, folder)
        for attr, value in props.items():
            try:
                setattr(layer, attr, value)
            except Exception:  # noqa: BLE001
                pass
        made.append(layer)
    for layer in existing.values():
        viewer.layers.remove(layer)
    return made


def make_tissue_labels(viewer: Any, source: Any, settings: dict[str, Any] | None = None) -> Any:
    """A Labels layer classifying *source*'s voxels into the enabled tissues (label *i* = the
    *i*-th enabled tissue; where ranges overlap, the one centred nearest the value)."""
    from napari.utils.colormaps import DirectLabelColormap

    from nvitk.core.array import to_numpy
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    settings = copy.deepcopy(settings) if settings is not None else layer_settings(source)
    data = to_numpy(source.data)
    labels = np.zeros(data.shape, dtype=np.uint8)
    best = np.full(data.shape, np.inf, dtype=np.float32)
    colors: dict[Any, Any] = {0: np.zeros(4), None: np.zeros(4)}
    names = {}
    on = [t for t in settings["tissues"] if t.get("enabled")]
    for i, tissue in enumerate(on, 1):
        lo, hi = float(tissue["lo"]), float(tissue["hi"])
        inside = (data >= lo) & (data <= hi)
        # Overlapping ranges (contrast blood and cancellous bone): the tissue whose range is centred
        # nearest the voxel's value takes it.
        distance = np.abs(data - (lo + hi) / 2.0).astype(np.float32)
        take = inside & (distance < best)
        labels[take] = i
        best[take] = distance[take]
        colors[i] = np.array([*_rgb(tissue["color"]), 1.0])
        names[i] = tissue["name"]
    del best
    name = f"tissues · {source.name}"
    layer = viewer.add_labels(labels, name=name, **layer_spatial_kwargs(source),
                              colormap=DirectLabelColormap(color_dict=colors), opacity=0.5)
    layer.metadata["nvitk_label_names"] = names
    layer.metadata["nvitk_tissue_of"] = str(source.name)
    return layer


# --------------------------------------------------------------------------- panel


class TissueRenderPanel(QWidget):
    """Controls for one CT layer's tissue rendering (live)."""

    changed = Signal()

    def __init__(self, viewer: Any, layer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._layer = layer
        self._settings = layer_settings(layer)
        self._rows: list[dict[str, Any]] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._apply)
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self._preset = QComboBox()
        for key, spec in TISSUE_PRESETS.items():
            self._preset.addItem(spec["label"], key)
        index = self._preset.findData(self._settings.get("preset", "realistic"))
        self._preset.setCurrentIndex(max(index, 0))
        self._preset.activated.connect(lambda _i: self._load_preset(self._preset.currentData()))
        top.addWidget(QLabel("Look"))
        top.addWidget(self._preset, 1)
        root.addLayout(top)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(2)
        for col, text in enumerate(("", "Tissue", "HU from", "to", "", "Opacity / cm", "Surface")):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold;")
            self._grid.addWidget(label, 0, col)
        root.addLayout(self._grid)
        self._build_rows()

        glob = QGridLayout()
        self._density = self._slider(0, 200, int(self._settings.get("density", 0.6) * 100))
        self._softness = self._slider(1, 300, int(self._settings.get("softness", 30.0)))
        self._gamma = self._slider(20, 300, int(float(self._settings.get("gamma", 1.0)) * 100))
        for row, (text, slider, tip) in enumerate((
            ("Density", self._density, "Scales every tissue's opacity: how much the volume absorbs"),
            ("Softness (HU)", self._softness, "Width of the fade at each tissue's HU edges"),
            ("Gamma", self._gamma, "Opacity curve: below 1 lifts faint tissue, above 1 keeps only the dense"),
        )):
            label = QLabel(text)
            label.setToolTip(tip)
            slider.setToolTip(tip)
            value = QLabel()
            value.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, lab=value, t=text: lab.setText(
                f"{v / 100:.2f}" if t != "Softness (HU)" else f"{v}"))
            slider.valueChanged.emit(slider.value())
            glob.addWidget(label, row, 0)
            glob.addWidget(slider, row, 1)
            glob.addWidget(value, row, 2)
        root.addLayout(glob)

        buttons = QHBoxLayout()
        for text, tip, slot in (
            ("Tissue layers", "One layer per enabled tissue (sharing this volume's data), in a folder: "
                              "toggle and tune each from the layer list", self._make_layers),
            ("Tissue label map", "Classify the voxels into the enabled tissues (a Labels layer)", self._make_labels),
            ("Save as my default", "Use these settings for every CT from now on", self._save_default),
            ("Reset", "Back to the chosen look's tissue settings", lambda: self._load_preset(self._preset.currentData())),
            ("Default rendering", "Leave the tissue look: the layer back to how it was drawn before any preset "
                                  "(as loaded), and this panel closes", self._default_rendering),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        root.addLayout(buttons)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        root.addWidget(self._note)
        self._apply()

    @staticmethod
    def _slider(lo: int, hi: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(max(lo, min(hi, value)))
        return slider

    def _build_rows(self) -> None:
        for row in self._rows:
            for widget in row["widgets"]:
                widget.hide()
                widget.deleteLater()
        self._rows = []
        for i, tissue in enumerate(self._settings["tissues"], 1):
            on = QCheckBox()
            on.setChecked(bool(tissue.get("enabled")))
            name = QLabel(tissue["name"])
            lo = QSpinBox()
            hi = QSpinBox()
            for box, value in ((lo, tissue["lo"]), (hi, tissue["hi"])):
                box.setRange(-1024, 4000)
                box.setSingleStep(10)
                box.setValue(int(value))
                box.setKeyboardTracking(False)
            color = QPushButton()
            color.setFixedWidth(28)
            color.setStyleSheet(f"background-color: {tissue['color']}; border: 1px solid #555;")
            opacity = self._slider(0, 100, int(round(float(tissue["opacity"]) * 100)))
            surface = QCheckBox()
            surface.setChecked(bool(tissue.get("surface")))
            surface.setToolTip("As its own layer: an iso-surface at the lower HU bound (shaded), not a volume")
            widgets = [on, name, lo, hi, color, opacity, surface]
            for col, widget in enumerate(widgets):
                self._grid.addWidget(widget, i, col)
            row = {"widgets": widgets, "tissue": tissue}
            on.toggled.connect(lambda c, t=tissue: self._set(t, "enabled", bool(c)))
            lo.valueChanged.connect(lambda v, t=tissue: self._set(t, "lo", float(v)))
            hi.valueChanged.connect(lambda v, t=tissue: self._set(t, "hi", float(v)))
            opacity.valueChanged.connect(lambda v, t=tissue: self._set(t, "opacity", v / 100.0))
            surface.toggled.connect(lambda c, t=tissue: self._set(t, "surface", bool(c), live=False))
            color.clicked.connect(lambda _=False, t=tissue, b=color: self._pick_color(t, b))
            self._rows.append(row)

    def _set(self, tissue: dict[str, Any], key: str, value: Any, *, live: bool = True) -> None:
        tissue[key] = value
        if live:
            self._timer.start()

    def _pick_color(self, tissue: dict[str, Any], button: QPushButton) -> None:
        color = QColorDialog.getColor(QColor(tissue["color"]), self, f"{tissue['name']} colour")
        if color.isValid():
            tissue["color"] = color.name()
            button.setStyleSheet(f"background-color: {tissue['color']}; border: 1px solid #555;")
            self._timer.start()

    def _load_preset(self, key: str) -> None:
        self._settings = default_settings(str(key or "realistic"))
        self._build_rows()
        for slider, value in ((self._density, self._settings["density"] * 100), (self._softness, self._settings["softness"]),
                              (self._gamma, 100)):
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
            slider.valueChanged.emit(slider.value())
        self._apply()

    def settings(self) -> dict[str, Any]:
        self._settings["density"] = self._density.value() / 100.0
        self._settings["softness"] = float(self._softness.value())
        self._settings["gamma"] = self._gamma.value() / 100.0
        self._settings["preset"] = str(self._preset.currentData() or "realistic")
        return self._settings

    def _default_rendering(self) -> None:
        from nvitk.gui.viz.render_presets import restore_default

        if self._layer is not None:
            restore_default([self._layer])
        window = self.window()
        if window is not None and window is not self:
            window.close()

    def _apply(self) -> None:
        if self._layer is None:
            return
        try:
            apply_tissue_render(self._layer, self.settings())
            children = _tissue_children(self._viewer, self._layer) if self._viewer is not None else []
            if children:
                make_tissue_layers(self._viewer, self._layer, self.settings())
        except Exception as exc:  # noqa: BLE001
            self._note.setText(f"Could not apply: {exc}")
            return
        lo, hi = hu_window(self.settings())
        self._note.setText(f"{self._layer.name}: translucent, window {lo:.0f} to {hi:.0f} HU.")
        self.changed.emit()

    def _make_layers(self) -> None:
        if self._viewer is None:
            return
        made = make_tissue_layers(self._viewer, self._layer, self.settings())
        self._note.setText(f"{len(made)} tissue layer(s) in folder Tissues/{self._layer.name}; they follow these controls.")

    def _make_labels(self) -> None:
        if self._viewer is None:
            return
        layer = make_tissue_labels(self._viewer, self._layer, self.settings())
        self._note.setText(f"Added {layer.name}.")

    def _save_default(self) -> None:
        try:
            from nvitk.gui.core.prefs import save_prefs

            save_prefs({PREFS_KEY: copy.deepcopy(self.settings())})
            self._note.setText("Saved as your default tissue look.")
        except Exception as exc:  # noqa: BLE001
            self._note.setText(f"Could not save: {exc}")


def show_tissue_popover(viewer: Any, layer: Any, global_pos: Any = None) -> Any:
    """The tissue controls for *layer* in a pop-over, pinned (a floating panel; applied at once)."""
    from qtpy.QtGui import QCursor

    from nvitk.gui.core import design
    from nvitk.gui.core.widgets import Popover

    panel = TissueRenderPanel(viewer, layer)
    design.apply_theme(panel)
    parent = viewer.window._qt_window if viewer is not None else None
    popover = Popover(panel, title=f"Body tissues · {layer.name}", parent=parent)
    popover.setAttribute(Qt.WA_DeleteOnClose, True)
    # Pinned from the start: these are tuned while turning the volume, and a plain pop-over closes
    # at the first click on the canvas.
    popover.set_pinned(True)
    popover.show_at_point(global_pos if global_pos is not None else QCursor.pos())
    return popover


__all__ = [
    "TISSUES",
    "TISSUE_PRESETS",
    "Tissue",
    "TissueRenderPanel",
    "apply_tissue_render",
    "default_settings",
    "ray_step_mm",
    "hu_window",
    "is_ct_layer",
    "make_tissue_labels",
    "make_tissue_layers",
    "show_tissue_popover",
    "tissue_colormap",
    "transfer_function",
]
