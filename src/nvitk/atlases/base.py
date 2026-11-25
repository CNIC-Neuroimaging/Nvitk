"""Atlas records, label tables, and the registry that lists and loads them across sources.

An *atlas* here is anything you would open over a subject or a template to name or bound
regions: a label image (Harvard-Oxford max-probability, JHU, aparc+aseg), a probabilistic
stack, a template image (MNI152, ANTsXNet's population templates), or a label table on its own
(FreeSurfer's colour LUT, which names a FastSurfer / FreeSurfer output).

Each source module (:mod:`.fsl`, :mod:`.freesurfer`, :mod:`.antsxnet`, :mod:`.nilearn_src`,
:mod:`.configured`) answers two questions cheaply and without network access: *what do you have*
(:func:`list_atlases`) and *where is it* (:func:`load_atlas`). Nothing is downloaded unless the
caller passes ``download=True``.
"""

from __future__ import annotations

import colorsys
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

#: Availability of an atlas on this machine.
INSTALLED = "installed"
DOWNLOADABLE = "downloadable"
MISSING = "missing"

#: Source key -> module implementing ``list_atlases()`` and ``load(info, **kw)``.
SOURCES: dict[str, str] = {
    "fsl": "nvitk.atlases.fsl",
    "freesurfer": "nvitk.atlases.freesurfer",
    "antsxnet": "nvitk.atlases.antsxnet",
    "nilearn": "nvitk.atlases.nilearn_src",
    "configured": "nvitk.atlases.configured",
}


@dataclass(frozen=True)
class Label:
    """One region: id (the voxel value), name, and display colour (0-255 RGB)."""

    id: int
    name: str
    rgb: tuple[int, int, int] = (200, 200, 200)


@dataclass
class AtlasInfo:
    """What a source knows about one atlas before it is loaded.

    ``id`` is ``"<source>:<name>"`` and stable across runs. ``variants`` lists the image choices
    (``"1mm"``, ``"2mm"``, ``"thr25-1mm"``...), the first being the default.
    """

    id: str
    name: str
    source: str
    kind: str  # labels | probabilistic | template | lut
    space: str = ""
    species: str = "human"
    status: str = INSTALLED
    description: str = ""
    citation: str = ""
    variants: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Whether it can be loaded without a download."""
        return self.status == INSTALLED


@dataclass
class AtlasData:
    """A loaded atlas: its files and its label table."""

    info: AtlasInfo
    image: Path | None
    labels: list[Label]
    variant: str = ""
    #: Extra files (a probabilistic stack beside its max-probability image, a brain mask...).
    files: dict[str, Path] = field(default_factory=dict)

    def label_names(self) -> dict[int, str]:
        """``{id: name}``, background included when the atlas names it."""
        return {label.id: label.name for label in self.labels}

    def to_tsv(self) -> str:
        """The label table as TSV (``id name R G B``), FreeSurfer-LUT-like."""
        lines = ["id\tname\tR\tG\tB"]
        lines += [f"{l.id}\t{l.name}\t{l.rgb[0]}\t{l.rgb[1]}\t{l.rgb[2]}" for l in self.labels]
        return "\n".join(lines) + "\n"


def palette(n: int, *, seed: float = 0.11) -> list[tuple[int, int, int]]:
    """*n* distinct, reproducible colours (golden-ratio hue walk) for tables that carry none."""
    out = []
    for i in range(max(n, 0)):
        hue = (seed + i * 0.618033988749895) % 1.0
        sat = 0.55 + 0.25 * ((i * 7) % 3) / 2
        val = 0.85 - 0.15 * ((i * 5) % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def labels_from_names(names: Iterable[tuple[int, str]]) -> list[Label]:
    """Labels for ``(id, name)`` pairs, coloured by :func:`palette` (id 0 stays black)."""
    pairs = list(names)
    colours = palette(len(pairs))
    return [
        Label(int(i), str(name), (0, 0, 0) if int(i) == 0 else colours[k])
        for k, (i, name) in enumerate(pairs)
    ]


def _module(source: str) -> Any:
    return importlib.import_module(SOURCES[source])


def list_atlases(
    source: str | None = None,
    *,
    include_downloadable: bool = False,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[AtlasInfo]:
    """Every atlas the installed sources offer (*source* narrows to one).

    Never downloads. With *include_downloadable*, atlases a source could fetch are listed too
    (status ``"downloadable"``). A source that fails to answer is skipped (and reported to
    *on_error*), so one broken installation does not hide the others.
    """
    keys = [source] if source else list(SOURCES)
    out: list[AtlasInfo] = []
    for key in keys:
        try:
            found = _module(key).list_atlases()
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                on_error(key, exc)
            continue
        out.extend(a for a in found if include_downloadable or a.status == INSTALLED)
    return out


def get_atlas(atlas_id: str, *, include_downloadable: bool = True) -> AtlasInfo:
    """The :class:`AtlasInfo` for *atlas_id* (``KeyError`` when no source offers it)."""
    source = str(atlas_id).split(":", 1)[0]
    if source not in SOURCES:
        raise KeyError(f"Unknown atlas source {source!r} in {atlas_id!r}; sources: {', '.join(SOURCES)}")
    for info in list_atlases(source, include_downloadable=include_downloadable):
        if info.id == atlas_id:
            return info
    raise KeyError(f"No atlas {atlas_id!r} (try `nvitk-atlas list --all`)")


def load_atlas(atlas_id: str, *, variant: str | None = None, download: bool = False) -> AtlasData:
    """Load *atlas_id*: its image path(s) and label table.

    *variant* picks one of :attr:`AtlasInfo.variants` (the first by default). An atlas that is
    only downloadable raises ``FileNotFoundError`` unless *download* is true.
    """
    info = get_atlas(atlas_id)
    if not info.available and not download:
        raise FileNotFoundError(
            f"{atlas_id} is not on this machine; pass download=True (CLI: --download) to fetch it."
        )
    chosen = variant or (info.variants[0] if info.variants else "")
    if info.variants and chosen not in info.variants:
        raise ValueError(f"{atlas_id}: variant {chosen!r} not in {', '.join(info.variants)}")
    return _module(info.source).load(info, variant=chosen, download=download)


__all__ = [
    "AtlasData",
    "AtlasInfo",
    "DOWNLOADABLE",
    "INSTALLED",
    "Label",
    "MISSING",
    "SOURCES",
    "get_atlas",
    "labels_from_names",
    "list_atlases",
    "load_atlas",
    "palette",
]
