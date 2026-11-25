"""Three orthogonal 2D slice views of a volume, plus 3D planes on the Napari canvas.

The layout a radiologist expects, and the one 3D Slicer opens with: axial,
coronal and sagittal through one shared crosshair, each scrollable on its own,
each showing where the other two are cutting. Scroll a view, click in it, or drag
its slider: the crosshair moves and the other two follow.

The fourth quadrant drives the *main* Napari canvas rather than adding a second
one: with ``Show 3D planes`` on, the three slices are pushed onto the canvas as
``depiction="plane"`` layers at the crosshair, so the 3D view shows the same
three cuts in space. ``Clip volume`` then takes one axis and throws away
everything on one side of its slice, which is how you see *inside* a volume
instead of only at its surface.

The four panes are resizable (:class:`PaneGrid`): drag the gap between two rows
or two columns, double-click a gap to even them out, double-click a view's title
to see that view alone (and again to come back). The sizes are kept for the next
session.

A 2D image (a plain 2D layer, or a positioned slice one voxel thick) is shown in
the first (axial) view when it is the active layer, with the layers on its grid
drawn over it; the other two views stand empty. ``Projection`` swaps the slices of
a volume for a maximum / minimum / mean intensity projection along each view's
axis, over the whole volume or a slab centred on the crosshair.

The implementation lives in the :mod:`nvitk.gui.viz.ortho` package (geometry, sources,
render, slice_view, planes3d, layout, controls3d, panel); this module keeps the old import path
working, private helpers included, for the code and scripts that reach into it.
"""

from __future__ import annotations

from nvitk.gui.viz.ortho.constants import (  # noqa: F401
    _PLANE_NAMES,
    DOCK_OBJECT_NAME,
    _PLANE_LAYER_SUFFIX,
    _BOX_LAYER_SUFFIX,
    BOX_AXIS_COLORS,
    _BOX_EDGE_WIDTH,
    _OVERLAY_OPACITY,
    _CANVAS_SYNC_MS,
    _STYLE_SYNC_MS,
    _REBUILD_SYNC_MS,
    _MIN_ZOOM,
    _MAX_ZOOM,
    _ZOOM_STEP,
    _WHEEL_NOTCH,
    _VIEW_CELLS,
    _PLAY_FPS,
    PROJECTIONS,
    _PROJECTION_SHORT,
    _PROJECTION_CACHE_SIZE,
    _PROJECTION_CHUNK_VOXELS,
    _HANDLE_TOL_PX,
    _HANDLE_MIN_FRACTION,
    _SLICE_CACHE_BUDGET,
    _SLICE_CACHE_TOTAL,
    _OPPOSITE,
    _PLANE_UP_RIGHT,
)
from nvitk.gui.viz.ortho.geometry import (  # noqa: F401
    AxisView,
    _orient_in_plane,
    _PLANE_ORDER,
    _anatomical_views,
    _PLANE_AXES,
    _axis_views,
    _unit,
    PlaneFrame,
    base_frame,
    rotation_about,
    rotate_frame,
    oblique_slice,
    line_direction_px,
)
from nvitk.gui.viz.ortho.sources import (  # noqa: F401
    _resample_budget,
    _RESAMPLE_BUDGET_BYTES,
    RenderSource,
    DRAWN_LAYER_TYPES,
    _grid_key,
    layer_time_axis,
    layer_time_count,
    _spatial_shape,
    _layer_shape,
    is_drawable_layer,
    _is_plain_2d,
    flat_axis,
    is_flat_layer,
    is_volume_layer,
    _layer_volume,
    _same_grid,
    resample_layer_to,
    _axis_index,
    _slice_of,
    project_volume,
    _flat_views,
    _POOL_LOCK,
    SliceCache,
)
from nvitk.gui.viz.ortho.render import (  # noqa: F401
    volume_contrast,
    layer_contrast,
    _grayscale_rgb,
    label_lut,
    _label_rgb,
    _COLORMAP_STEPS,
    layer_colormap,
    layer_gamma,
    colormap_table,
    image_rgba,
    label_rgba_lut,
    _label_rgba,
    label_color_table,
    slice_to_rgba,
    composite,
    _oriented,
    blend_overlay,
    slice_to_rgb,
    layer_interpolation_smooth,
    layer_interpolation_order,
    PaintRun,
    composite_over_rgba,
    composite_runs,
    flatten_runs,
)
from nvitk.gui.viz.ortho.slice_view import (  # noqa: F401
    SliceView,
)
from nvitk.gui.viz.ortho.planes3d import (  # noqa: F401
    _plane_layer_name,
    displayed_axes,
    _plane_geometry,
    remove_ortho_planes,
    _IMAGE_STYLE_ATTRS,
    _LABEL_STYLE_ATTRS,
    copy_plane_style,
    sync_ortho_planes,
    _box_layer_name,
    remove_ortho_boxes,
    slice_box_corners,
    oblique_box_corners,
    _same_corners,
    sync_ortho_boxes,
    _scene_point,
    clip_geometry,
    apply_clip,
    clear_clip,
)
from nvitk.gui.viz.ortho.layout import (  # noqa: F401
    _ORTHO_PREFS_KEY,
    _ortho_prefs,
    _save_ortho_pref,
    _PaneHandle,
    _PaneSplitter,
    PaneGrid,
)
from nvitk.gui.viz.ortho.controls3d import Controls3D  # noqa: F401
from nvitk.gui.viz.ortho.layout import LayoutGrid, spec_panes  # noqa: F401
from nvitk.gui.viz.ortho.panel import (  # noqa: F401
    connect_ortho_panel,
    OrthoViewerPanel,
    open_ortho_views,
    attach_ortho_dock,
)

__all__ = [
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
