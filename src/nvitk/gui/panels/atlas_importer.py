"""Atlas importer (Data & QC module): list the atlases installed sources offer, load them as layers.

Sources are FSL, FreeSurfer / FastSurfer, ANTsXNet, nilearn and the atlases configured in
``settings.json`` (:mod:`nvitk.atlases`). A label atlas loads as a Labels layer coloured by its
own table, with its names registered as the layer's label vocabulary (the label list, the
readout and the layer menu then name each region); a template loads as an image; a
probabilistic atlas loads its max-probability labels, and optionally the 4D probability stack.
Downloadable atlases (ANTsXNet, nilearn) are fetched only when asked.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.widgets import hint_label

_SUITES = {"fsl": "FSL", "freesurfer": "FreeSurfer", "antsxnet": "ANTsPyNet", "nilearn": "nilearn", "configured": "nvitk"}


class AtlasImporterPanel(QWidget):
    """Browse and load atlases."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._infos: dict[str, Any] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)
        row = QHBoxLayout()
        self._source = QComboBox()
        self._source.addItem("All sources", "")
        for key, label in _SUITES.items():
            self._source.addItem(label if key != "configured" else "Configured (settings.json)", key)
        self._downloadable = QCheckBox("include downloadable")
        refresh = QPushButton("↻")
        refresh.clicked.connect(self.refresh)
        row.addWidget(self._source, 1)
        row.addWidget(self._downloadable)
        row.addWidget(refresh)
        root.addLayout(row)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Atlas", "Kind", "Space", "Source"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self._tree.header().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._tree.setRootIsDecorated(False)
        self._tree.currentItemChanged.connect(lambda *_a: self._on_pick())
        from nvitk.gui.core.widgets import resizable  # drag the handle under it to resize

        root.addWidget(resizable(self._tree, height=280, minimum=100, pref_key="atlases.tree"), 1)
        row = QHBoxLayout()
        self._variant = QComboBox()
        self._variant.setToolTip("Resolution / threshold variant")
        self._stack = QCheckBox("probability stack too")
        self._load = QPushButton("Load")
        self._load.clicked.connect(lambda: self._do_load(download=False))
        self._fetch = QPushButton("Download && load")
        self._fetch.clicked.connect(lambda: self._do_load(download=True))
        row.addWidget(self._variant, 1)
        row.addWidget(self._stack)
        row.addWidget(self._load)
        row.addWidget(self._fetch)
        root.addLayout(row)
        self._info = hint_label("")
        root.addWidget(self._info)
        self._source.currentIndexChanged.connect(lambda _i: self.refresh())
        self._downloadable.toggled.connect(lambda _c: self.refresh())
        self._listed = False

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._listed:
            self._listed = True
            self.refresh()

    def refresh(self) -> None:
        from nvitk.atlases import list_atlases

        errors: list[str] = []
        infos = list_atlases(self._source.currentData() or None, include_downloadable=self._downloadable.isChecked(),
                             on_error=lambda s, e: errors.append(f"{s}: {e}"))
        self._tree.clear()
        self._infos = {}
        for info in infos:
            item = QTreeWidgetItem([info.name, info.kind, info.space, _SUITES.get(info.source, info.source)])
            item.setData(0, Qt.UserRole, info.id)
            item.setToolTip(0, f"{info.id}\n{info.description}")
            if not info.available:
                item.setForeground(0, design_brush(design.COLOR_MUTED))
                item.setText(3, item.text(3) + " ↓")
            self._tree.addTopLevelItem(item)
            self._infos[info.id] = info
        self._info.setText(f"{len(infos)} atlas(es)" + (f"; unavailable: {'; '.join(errors)}" if errors else ""))

    def _current(self) -> Any:
        item = self._tree.currentItem()
        return self._infos.get(item.data(0, Qt.UserRole)) if item is not None else None

    def _on_pick(self) -> None:
        info = self._current()
        self._variant.clear()
        if info is None:
            return
        self._variant.addItems(list(info.variants))
        self._variant.setEnabled(bool(info.variants))
        self._stack.setEnabled(info.kind == "probabilistic")
        self._load.setEnabled(info.available)
        self._fetch.setVisible(not info.available)
        self._info.setText(info.description)

    def _do_load(self, *, download: bool) -> None:
        from nvitk.atlases import load_atlas

        info = self._current()
        if info is None:
            return
        try:
            data = load_atlas(info.id, variant=self._variant.currentText() or None, download=download)
            names = load_atlas_layers(self._viewer, data, stack=self._stack.isChecked())
            self._info.setText(f"Loaded {', '.join(names)}" if names else f"{info.name}: a label table only (no image).")
        except Exception as exc:  # noqa: BLE001
            self._info.setText(f"Could not load {info.name}: {exc}")


def design_brush(colour: str) -> Any:
    from qtpy.QtGui import QBrush, QColor

    return QBrush(QColor(colour))


def load_atlas_layers(viewer: Any, data: Any, *, stack: bool = False) -> list[str]:
    """Add *data* (an :class:`~nvitk.atlases.AtlasData`) to *viewer*; returns the new layer names.

    Label atlases become Labels layers with the atlas colours and names; templates images.
    """
    from nvitk.gui.io.napari_io import open_paths_with_nvitk

    made: list[str] = []
    if data.image is None:
        if data.labels:
            from nvitk.atlases.schema import register_atlas_schema

            register_atlas_schema(data)
        return made
    layers = open_paths_with_nvitk(viewer, [data.image]) or []
    for layer in layers:
        if data.info.kind in ("labels", "probabilistic") and data.labels:
            layer = _as_labels(viewer, layer, data)
        layer.name = f"{data.info.name}" + (f" ({data.variant})" if data.variant else "")
        made.append(str(layer.name))
    if stack and "probabilistic" in data.files:
        for layer in open_paths_with_nvitk(viewer, [data.files["probabilistic"]]) or []:
            layer.name = f"{data.info.name} probabilities"
            made.append(str(layer.name))
    return made


def _as_labels(viewer: Any, layer: Any, data: Any) -> Any:
    """Replace an Image layer holding label ids by a Labels layer with the atlas's colours and names."""
    import numpy as np

    from nvitk.atlases.schema import label_colors, register_atlas_schema
    from nvitk.core.array import to_numpy
    from nvitk.gui.core.spatial import layer_spatial_kwargs
    from nvitk.gui.labels.catalog import remember_layer_schema

    if type(layer).__name__ != "Labels":
        ids = to_numpy(layer.data)
        kwargs = layer_spatial_kwargs(layer)
        meta = dict(getattr(layer, "metadata", {}) or {})
        viewer.layers.remove(layer)
        layer = viewer.add_labels(np.asarray(ids).astype(np.int32, copy=False), metadata=meta, **kwargs)
    try:
        from napari.utils.colormaps import DirectLabelColormap

        colours = label_colors(data)
        colours[None] = (0.0, 0.0, 0.0, 0.0)
        layer.colormap = DirectLabelColormap(color_dict=colours)
    except Exception:  # noqa: BLE001 (keep Napari's own colours)
        pass
    remember_layer_schema(layer, register_atlas_schema(data))
    return layer


__all__ = ["AtlasImporterPanel", "load_atlas_layers"]
