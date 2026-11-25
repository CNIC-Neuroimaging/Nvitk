"""Orthogonal slice views of a volume, and their cuts on the 3D canvas.

Modules, in dependency order:

* :mod:`.constants`: plane names, timings, zoom and cache limits, projections.
* :mod:`.geometry`: :class:`AxisView`, :class:`PlaneFrame`, oblique slicing.
* :mod:`.sources`: what is drawn: :class:`RenderSource`, grids, resampling, :class:`SliceCache`, projections.
* :mod:`.render`: contrast, colormaps, label colours, compositing.
* :mod:`.slice_view`: :class:`SliceView`, one interactive slice.
* :mod:`.planes3d`: slice planes, outlines and see-inside clipping on the Napari canvas.
* :mod:`.layout`: :class:`LayoutGrid` (and :class:`PaneGrid`), and the persisted preferences.
* :mod:`.controls3d`: the "3D view" options, as a widget of their own.
* :mod:`.panel`: :class:`OrthoViewerPanel`, which binds the active layer and drives the rest.
"""

from __future__ import annotations

from nvitk.gui.viz.ortho.constants import PROJECTIONS
from nvitk.gui.viz.ortho.planes3d import copy_plane_style, apply_clip, clear_clip, oblique_box_corners, clip_geometry, displayed_axes, remove_ortho_planes, sync_ortho_planes
from nvitk.gui.viz.ortho.sources import flat_axis, is_flat_layer, project_volume, layer_time_axis, layer_time_count, SliceCache
from nvitk.gui.viz.ortho.render import colormap_table, composite, image_rgba, label_rgba_lut, layer_colormap, layer_gamma, slice_to_rgba, blend_overlay, label_lut, slice_to_rgb, volume_contrast
from nvitk.gui.viz.ortho.geometry import AxisView
from nvitk.gui.viz.ortho.panel import OrthoViewerPanel, attach_ortho_dock, open_ortho_views
from nvitk.gui.viz.ortho.slice_view import SliceView
from nvitk.gui.viz.ortho.layout import LayoutGrid, PaneGrid, spec_panes
from nvitk.gui.viz.ortho.controls3d import Controls3D

__all__ = [
    "Controls3D",
    "LayoutGrid",
    "PaneGrid",
    "spec_panes",
    "PROJECTIONS",
    "copy_plane_style",
    "flat_axis",
    "is_flat_layer",
    "project_volume",
    "layer_time_axis",
    "layer_time_count",
    "colormap_table",
    "composite",
    "image_rgba",
    "label_rgba_lut",
    "layer_colormap",
    "layer_gamma",
    "slice_to_rgba",
    "AxisView",
    "OrthoViewerPanel",
    "SliceView",
    "apply_clip",
    "attach_ortho_dock",
    "clear_clip",
    "blend_overlay",
    "oblique_box_corners",
    "clip_geometry",
    "SliceCache",
    "label_lut",
    "open_ortho_views",
    "displayed_axes",
    "remove_ortho_planes",
    "slice_to_rgb",
    "sync_ortho_planes",
    "volume_contrast",
]
