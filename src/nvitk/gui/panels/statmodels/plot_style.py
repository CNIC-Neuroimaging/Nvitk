"""How a Statmodels figure looks, whatever drew it: the plot pane's Style pop-over.

Every engine has its own plotters (marginal means, forest plots, path diagrams, the anatomical
maps...), so the appearance is not passed into each of them: :func:`apply_style` restyles the
finished figure where the pane shows it, Matplotlib or Plotly alike. One set of choices then holds
for every model and every figure, and travels with the session:

* **palette**: the figure's series colours, remapped in the order they appear (legend order) to
  a named palette (tab10, colour-blind safe, Set2, viridis...); continuous colormaps are left alone;
* **text**: one size for everything (titles, labels, ticks, legend), and the title and axis
  labels overridden when given;
* **marks**: point size and opacity, line width (multipliers of what the plotter chose);
* **frame**: grid, spines (despine), background (white paper or the app's dark), legend placement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: Palettes offered (seaborn / matplotlib names); "" keeps the plotter's own colours.
PALETTES: tuple[tuple[str, str], ...] = (
    ("plotter's own", ""),
    ("tab10", "tab10"),
    ("colour-blind safe", "colorblind"),
    ("deep", "deep"),
    ("muted", "muted"),
    ("Set2", "Set2"),
    ("Dark2", "Dark2"),
    ("Paired", "Paired"),
    ("viridis (ordered)", "viridis"),
    ("magma (ordered)", "magma"),
    ("greys", "Greys"),
)
LEGEND_PLACES: tuple[tuple[str, str], ...] = (
    ("plotter's choice", ""),
    ("best", "best"),
    ("upper right", "upper right"),
    ("upper left", "upper left"),
    ("lower right", "lower right"),
    ("lower left", "lower left"),
    ("outside, right", "outside"),
)
BACKGROUNDS: tuple[tuple[str, str], ...] = (
    ("white (paper)", "white"),
    ("dark (app)", "dark"),
)


@dataclass
class PlotStyle:
    """The pane's appearance choices (see the module)."""

    palette: str = ""
    font_size: float = 0.0  # 0: the plotter's own sizes
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    point_scale: float = 1.0
    point_alpha: float = 0.0  # 0: keep
    line_scale: float = 1.0
    grid: str = ""  # "" keep, "on", "off"
    despine: bool = False
    legend: str = ""
    background: str = "white"
    export_dpi: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "PlotStyle":
        if not isinstance(data, dict):
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_default(self) -> bool:
        return self == PlotStyle()


# --------------------------------------------------------------------------- matplotlib


def _palette_colors(name: str, n: int) -> list[tuple[float, float, float, float]]:
    import matplotlib.colors as mcolors

    try:
        import seaborn as sns

        colors = sns.color_palette(name, n_colors=max(n, 1))
    except Exception:  # noqa: BLE001
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(name)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return [mcolors.to_rgba(c) for c in colors]


def _key(color: Any) -> tuple[float, float, float] | None:
    import matplotlib.colors as mcolors

    try:
        r, g, b, _a = mcolors.to_rgba(color)
    except (ValueError, TypeError):
        return None
    return (round(r, 3), round(g, 3), round(b, 3))


def _series_colors(fig: Any) -> list[tuple[float, float, float]]:
    """The figure's categorical series colours, legend order first, then plotting order; greys
    (reference lines, raw points) are not series."""
    seen: list[tuple[float, float, float]] = []

    def add(color: Any) -> None:
        key = _key(color)
        if key is None or key in seen:
            return
        r, g, b = key
        if max(r, g, b) - min(r, g, b) < 0.06:  # grey / black / white: not a series colour
            return
        seen.append(key)

    for ax in fig.axes:
        legend = ax.get_legend()
        for handle in (legend.legend_handles if legend is not None and hasattr(legend, "legend_handles") else []):
            for getter in ("get_color", "get_facecolor", "get_markerfacecolor"):
                fn = getattr(handle, getter, None)
                if fn is None:
                    continue
                try:
                    value = fn()
                except Exception:  # noqa: BLE001
                    continue
                if hasattr(value, "__len__") and len(value) and not isinstance(value, str) and hasattr(value[0], "__len__"):
                    value = value[0]
                add(value)
                break
    for ax in fig.axes:
        for line in ax.get_lines():
            add(line.get_color())
        for coll in ax.collections:
            if coll.get_array() is not None:
                continue  # mapped through a colormap
            for fc in coll.get_facecolors()[:1]:
                add(fc)
        for patch in ax.patches:
            add(patch.get_facecolor())
    return seen


def _remap_palette(fig: Any, palette: str) -> None:
    import numpy as np

    order = _series_colors(fig)
    if not order:
        return
    new = _palette_colors(palette, len(order))
    mapping = {old: new[i] for i, old in enumerate(order)}

    def swap(color: Any) -> Any:
        key = _key(color)
        if key in mapping:
            r, g, b, _ = mapping[key]
            try:
                alpha = float(np.asarray(color, dtype=float).ravel()[3]) if not isinstance(color, str) and len(np.asarray(color).ravel()) == 4 else 1.0
            except Exception:  # noqa: BLE001
                alpha = 1.0
            return (r, g, b, alpha)
        return color

    def swap_array(colors: Any) -> Any:
        arr = np.asarray(colors, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 4:
            return colors
        return np.array([swap(tuple(c)) for c in arr])

    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_color(swap(line.get_color()))
            line.set_markerfacecolor(swap(line.get_markerfacecolor()))
            line.set_markeredgecolor(swap(line.get_markeredgecolor()))
        for coll in ax.collections:
            if coll.get_array() is not None:
                continue
            try:
                coll.set_facecolors(swap_array(coll.get_facecolors()))
                coll.set_edgecolors(swap_array(coll.get_edgecolors()))
            except Exception:  # noqa: BLE001
                pass
        for patch in ax.patches:
            patch.set_facecolor(swap(patch.get_facecolor()))
            patch.set_edgecolor(swap(patch.get_edgecolor()))
        legend = ax.get_legend()
        if legend is not None:
            for handle in getattr(legend, "legend_handles", []) or []:
                for getter, setter in (("get_color", "set_color"), ("get_facecolor", "set_facecolor"),
                                       ("get_markerfacecolor", "set_markerfacecolor")):
                    if hasattr(handle, getter) and hasattr(handle, setter):
                        try:
                            value = getattr(handle, getter)()
                            if hasattr(value, "__len__") and not isinstance(value, str) and len(value) and hasattr(value[0], "__len__"):
                                getattr(handle, setter)(swap_array(value))
                            else:
                                getattr(handle, setter)(swap(value))
                        except Exception:  # noqa: BLE001
                            pass


def apply_matplotlib(fig: Any, style: PlotStyle) -> None:
    """Restyle a finished Matplotlib *fig* in place."""
    axes = [ax for ax in fig.axes if ax.get_visible()]
    # Colourbars and switched-off axes (a forest's text column) are not plots.
    plots = [ax for ax in axes if not getattr(ax, "_colorbar", None) and ax.get_label() != "<colorbar>" and ax.axison]
    if style.palette:
        try:
            _remap_palette(fig, style.palette)
        except Exception:  # noqa: BLE001 (a figure kind the remap does not understand: leave it)
            pass
    # Texts first: set_title / set_xlabel reset the font, so the sizes come after.
    if plots:
        if style.title:
            if len(plots) == 1:
                plots[0].set_title(style.title)
            else:
                fig.suptitle(style.title, fontsize=style.font_size * 1.25 if style.font_size else None)
        if style.x_label:
            for ax in plots:
                if ax.get_xlabel():
                    ax.set_xlabel(style.x_label)
            if len(plots) == 1:
                plots[0].set_xlabel(style.x_label)
        if style.y_label:
            for ax in plots:
                if ax.get_ylabel():
                    ax.set_ylabel(style.y_label)
            if len(plots) == 1:
                plots[0].set_ylabel(style.y_label)
    for ax in plots:
        if style.point_scale != 1.0 or style.point_alpha:
            for coll in ax.collections:
                sizes = coll.get_sizes() if hasattr(coll, "get_sizes") else None
                if sizes is not None and len(sizes) and style.point_scale != 1.0:
                    coll.set_sizes(sizes * style.point_scale)
                if style.point_alpha and hasattr(coll, "get_offsets") and len(coll.get_offsets()):
                    coll.set_alpha(style.point_alpha)
            for line in ax.get_lines():
                if line.get_marker() not in (None, "None", "", " ") and style.point_scale != 1.0:
                    line.set_markersize(line.get_markersize() * style.point_scale ** 0.5)
        if style.line_scale != 1.0:
            for line in ax.get_lines():
                line.set_linewidth(line.get_linewidth() * style.line_scale)
        if style.grid == "on":
            ax.grid(True, alpha=0.3)
        elif style.grid == "off":
            ax.grid(False)
        if style.despine:
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        if style.font_size:
            size = float(style.font_size)
            ax.title.set_fontsize(size * 1.15)
            ax.xaxis.label.set_fontsize(size)
            ax.yaxis.label.set_fontsize(size)
            ax.tick_params(labelsize=size * 0.9)
            for text in ax.texts:
                text.set_fontsize(min(text.get_fontsize(), size))
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(size * 0.9)
                if legend.get_title() is not None:
                    legend.get_title().set_fontsize(size * 0.9)
        if style.legend:
            legend = ax.get_legend()
            if legend is not None:
                handles = getattr(legend, "legend_handles", None) or []
                labels = [t.get_text() for t in legend.get_texts()]
                title = legend.get_title().get_text() if legend.get_title() is not None else None
                if style.legend == "outside":
                    ax.legend(handles, labels, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                              frameon=False, fontsize=style.font_size * 0.9 if style.font_size else None)
                else:
                    ax.legend(handles, labels, title=title, loc=style.legend,
                              fontsize=style.font_size * 0.9 if style.font_size else None)
    if style.font_size and fig._suptitle is not None:
        fig._suptitle.set_fontsize(style.font_size * 1.25)


def apply_dark(fig: Any) -> None:
    """The app's dark background on a (whitened) figure, text and spines light."""
    from nvitk.gui.core import design

    bg = design.COLOR_BG if hasattr(design, "COLOR_BG") else "#1e1f22"
    fg = design.COLOR_TEXT if hasattr(design, "COLOR_TEXT") else "#e6e6e6"
    fig.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(bg)
            for text in legend.get_texts():
                text.set_color(fg)
    if fig._suptitle is not None:
        fig._suptitle.set_color(fg)


# --------------------------------------------------------------------------- plotly


def apply_plotly(fig: Any, style: PlotStyle) -> None:
    """Restyle a Plotly figure in place (the same choices, in Plotly's terms)."""
    layout: dict[str, Any] = {}
    if style.background == "dark":
        layout["template"] = "plotly_dark"
    if style.font_size:
        layout["font"] = {"size": float(style.font_size)}
    if style.title:
        layout["title"] = {"text": style.title}
    if style.legend:
        places = {"upper right": (0.99, 0.99, "right", "top"), "upper left": (0.01, 0.99, "left", "top"),
                  "lower right": (0.99, 0.01, "right", "bottom"), "lower left": (0.01, 0.01, "left", "bottom")}
        if style.legend in places:
            x, y, xa, ya = places[style.legend]
            layout["legend"] = {"x": x, "y": y, "xanchor": xa, "yanchor": ya}
    if layout:
        fig.update_layout(**layout)
    if style.x_label:
        fig.update_xaxes(title_text=style.x_label)
    if style.y_label:
        fig.update_yaxes(title_text=style.y_label)
    if style.grid in ("on", "off"):
        fig.update_xaxes(showgrid=style.grid == "on")
        fig.update_yaxes(showgrid=style.grid == "on")
    if style.palette:
        import matplotlib.colors as mcolors

        names = []
        for trace in fig.data:
            group = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
            if group not in names:
                names.append(group)
        colors = [mcolors.to_hex(c) for c in _palette_colors(style.palette, len(names))]
        for trace in fig.data:
            group = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
            color = colors[names.index(group)]
            for attr in ("marker", "line"):
                part = getattr(trace, attr, None)
                try:
                    current = part.color if part is not None else None
                except Exception:  # noqa: BLE001
                    continue
                # One colour per trace (a string), not a per-point array or a colorscale.
                if part is not None and (current is None or isinstance(current, str)):
                    try:
                        part.color = color
                    except Exception:  # noqa: BLE001
                        pass
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is not None:
            try:
                if style.point_scale != 1.0 and isinstance(marker.size, (int, float)):
                    marker.size = marker.size * style.point_scale
                if style.point_alpha:
                    marker.opacity = style.point_alpha
            except Exception:  # noqa: BLE001
                pass
        line = getattr(trace, "line", None)
        if line is not None and style.line_scale != 1.0:
            try:
                if isinstance(line.width, (int, float)):
                    line.width = line.width * style.line_scale
            except Exception:  # noqa: BLE001
                pass


def apply_style(fig: Any, style: PlotStyle) -> None:
    """Restyle *fig* (Matplotlib or Plotly) in place; nilearn documents are left as they are."""
    if fig is None or style is None:
        return
    module = type(fig).__module__
    if module.startswith("plotly"):
        apply_plotly(fig, style)
    elif hasattr(fig, "axes") and hasattr(fig, "savefig"):
        apply_matplotlib(fig, style)


__all__ = ["BACKGROUNDS", "LEGEND_PLACES", "PALETTES", "PlotStyle", "StyleForm", "apply_dark", "apply_style"]


# --------------------------------------------------------------------------- the form


@__import__("functools").lru_cache(maxsize=None)
def _style_form_class() -> type:
    """:class:`StyleForm`, built on first use (keeps this module importable without Qt)."""
    from qtpy.QtCore import Signal
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QWidget,
    )

    class StyleForm(QWidget):
        """The Style pop-over: every :class:`PlotStyle` choice; ``changed`` on any edit."""

        changed = Signal(object)

        def __init__(self, style: PlotStyle | None = None, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._building = True
            form = QFormLayout(self)
            form.setContentsMargins(8, 6, 8, 6)
            form.setVerticalSpacing(4)

            def combo(items: tuple[tuple[str, str], ...]) -> QComboBox:
                box = QComboBox()
                for label, key in items:
                    box.addItem(label, key)
                box.currentIndexChanged.connect(self._emit)
                return box

            def spin(lo: float, hi: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
                box = QDoubleSpinBox()
                box.setRange(lo, hi)
                box.setSingleStep(step)
                box.setDecimals(decimals)
                box.valueChanged.connect(self._emit)
                return box

            def line(placeholder: str) -> QLineEdit:
                edit = QLineEdit()
                edit.setPlaceholderText(placeholder)
                edit.editingFinished.connect(self._emit)
                return edit

            self._palette = combo(PALETTES)
            self._palette.setToolTip("Series colours, remapped in legend order; colormaps (maps, heatmaps) are kept.")
            form.addRow("Palette", self._palette)
            self._background = combo(BACKGROUNDS)
            form.addRow("Background", self._background)
            self._font = spin(0, 30, 0.5, 1)
            self._font.setSpecialValueText("plotter's")
            self._font.setToolTip("One text size (pt) for titles, labels, ticks and legend; 0 keeps the plotter's.")
            form.addRow("Text size", self._font)
            self._title = line("plotter's title")
            form.addRow("Title", self._title)
            self._x_label = line("plotter's x label")
            form.addRow("x label", self._x_label)
            self._y_label = line("plotter's y label")
            form.addRow("y label", self._y_label)
            self._points = spin(0.1, 5, 0.1)
            self._alpha = spin(0, 1, 0.05)
            self._alpha.setSpecialValueText("keep")
            marks = QWidget()
            marks_lay = QHBoxLayout(marks)
            marks_lay.setContentsMargins(0, 0, 0, 0)
            marks_lay.addWidget(self._points)
            marks_lay.addWidget(self._alpha)
            self._points.setToolTip("Point size, × the plotter's")
            self._alpha.setToolTip("Point opacity (keep: the plotter's)")
            form.addRow("Points ×, opacity", marks)
            self._lines = spin(0.1, 5, 0.1)
            self._lines.setToolTip("Line width, × the plotter's")
            form.addRow("Lines ×", self._lines)
            self._grid = combo((("plotter's", ""), ("on", "on"), ("off", "off")))
            form.addRow("Grid", self._grid)
            self._despine = QCheckBox("Hide top and right spines")
            self._despine.toggled.connect(self._emit)
            form.addRow("", self._despine)
            self._legend = combo(LEGEND_PLACES)
            form.addRow("Legend", self._legend)
            self._dpi = QSpinBox()
            self._dpi.setRange(72, 1200)
            self._dpi.setSingleStep(50)
            self._dpi.setSuffix(" dpi")
            self._dpi.valueChanged.connect(self._emit)
            form.addRow("Export", self._dpi)
            reset = QPushButton("Reset style")
            reset.clicked.connect(lambda: (self.set_style(PlotStyle()), self._emit()))
            form.addRow("", reset)
            self.set_style(style or PlotStyle())
            self._building = False

        def set_style(self, style: PlotStyle) -> None:
            building, self._building = self._building, True
            try:
                for box, value in ((self._palette, style.palette), (self._background, style.background),
                                   (self._grid, style.grid), (self._legend, style.legend)):
                    index = box.findData(value)
                    box.setCurrentIndex(max(index, 0))
                self._font.setValue(float(style.font_size or 0))
                self._title.setText(style.title)
                self._x_label.setText(style.x_label)
                self._y_label.setText(style.y_label)
                self._points.setValue(float(style.point_scale or 1))
                self._alpha.setValue(float(style.point_alpha or 0))
                self._lines.setValue(float(style.line_scale or 1))
                self._despine.setChecked(bool(style.despine))
                self._dpi.setValue(int(style.export_dpi or 200))
            finally:
                self._building = building

        def style(self) -> PlotStyle:
            return PlotStyle(
                palette=str(self._palette.currentData() or ""), font_size=float(self._font.value()),
                title=self._title.text().strip(), x_label=self._x_label.text().strip(),
                y_label=self._y_label.text().strip(), point_scale=float(self._points.value()),
                point_alpha=float(self._alpha.value()), line_scale=float(self._lines.value()),
                grid=str(self._grid.currentData() or ""), despine=self._despine.isChecked(),
                legend=str(self._legend.currentData() or ""),
                background=str(self._background.currentData() or "white"), export_dpi=int(self._dpi.value()),
            )

        def _emit(self, *_args: Any) -> None:
            if not self._building:
                self.changed.emit(self.style())

    return StyleForm


def StyleForm(*args: Any, **kwargs: Any) -> Any:  # noqa: N802 (a class, built lazily)
    return _style_form_class()(*args, **kwargs)
