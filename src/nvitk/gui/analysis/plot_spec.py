"""What a plot is, independent of how it is drawn: a :class:`PlotSpec` over a DataFrame.

The builder edits a spec; :func:`nvitk.gui.analysis.render_mpl.render` draws one with seaborn.
Specs are plain data, so a plot can be saved with a session, reproduced from a script, or
re-drawn when the table under it changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: kind -> (label, needs y, numeric x expected)
PLOT_KINDS: dict[str, tuple[str, bool, bool]] = {
    "scatter": ("Scatter", True, True),
    "regression": ("Scatter + regression", True, True),
    "line": ("Line", True, True),
    "bar": ("Bar (aggregate)", True, False),
    "box": ("Box", True, False),
    "violin": ("Violin", True, False),
    "strip": ("Strip (points)", True, False),
    "hist": ("Histogram", False, True),
    "kde": ("Density (KDE)", False, True),
    "ecdf": ("ECDF", False, True),
    "heatmap": ("Heatmap (pivot / correlation)", False, False),
    "pair": ("Pair plot", False, True),
}

AGGREGATIONS: tuple[str, ...] = ("mean", "median", "sum", "count", "min", "max", "std")
PALETTES: tuple[str, ...] = ("tab10", "tab20", "Set2", "pastel", "deep", "colorblind", "viridis", "magma", "coolwarm")


@dataclass
class PlotSpec:
    """One plot: kind, columns, aggregation, facets, filter and styling."""

    kind: str = "scatter"
    x: str | None = None
    y: str | None = None
    hue: str | None = None
    col: str | None = None
    agg: str = "mean"
    query: str = ""
    logx: bool = False
    logy: bool = False
    palette: str = "tab10"
    bins: int = 30
    title: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def problems(self, columns: list[str]) -> list[str]:
        """Why this spec cannot be drawn on a frame with *columns* (empty when it can)."""
        out = []
        if self.kind not in PLOT_KINDS:
            out.append(f"unknown plot kind {self.kind!r}")
            return out
        _label, needs_y, _numeric = PLOT_KINDS[self.kind]
        if self.kind not in ("heatmap", "pair") and not self.x:
            out.append("choose an x column")
        if needs_y and not self.y:
            out.append("choose a y column")
        for name in (self.x, self.y, self.hue, self.col):
            if name and name not in columns:
                out.append(f"no column {name!r}")
        return out


__all__ = ["AGGREGATIONS", "PALETTES", "PLOT_KINDS", "PlotSpec"]
