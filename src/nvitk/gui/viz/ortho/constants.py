"""Constants shared by the orthogonal views: plane names, timings, zoom, caches, projections."""

from __future__ import annotations




#: Axis codes -> the plane you are looking at when you slice along that axis.
_PLANE_NAMES: dict[str, str] = {
    "S": "Axial", "I": "Axial",
    "A": "Coronal", "P": "Coronal",
    "R": "Sagittal", "L": "Sagittal",
}

#: Name for the layer nvitk adds to the canvas for each 3D slice plane.
DOCK_OBJECT_NAME = "nvitk_ortho_dock"

_PLANE_LAYER_SUFFIX = "_ortho_plane"
_BOX_LAYER_SUFFIX = "_ortho_box"

#: One colour per array axis for the slice outlines, in axis order. Red / green /
#: blue so an outline says which of the three cuts it is without a legend.
BOX_AXIS_COLORS: tuple[str, str, str] = ("#ff4d4d", "#4dd964", "#4d9dff")
_BOX_EDGE_WIDTH = 0.6

#: How strongly an overlay is drawn over the base slice.
_OVERLAY_OPACITY = 0.55

#: Delay before pushing a crosshair move to the 3D canvas. Plane and clipping
#: updates re-upload volumes, so they are coalesced rather than run per step.
_CANVAS_SYNC_MS = 90

#: Coalescing window for colour/opacity bursts.
_STYLE_SYNC_MS = 30

#: Coalescing window for layer-list churn.
_REBUILD_SYNC_MS = 60

#: Zoom bounds and the factor one Ctrl+wheel notch applies. The floor is "fit to
#: the view", which is what the panels do without a zoom at all.
_MIN_ZOOM = 1.0
_MAX_ZOOM = 12.0
#: Per *standard* notch (120 eighths of a degree). Applied in proportion to the
#: wheel's actual delta: high-resolution wheels and touchpads send many small
#: events per notch, and treating each as a full notch is what made a 25 % step
#: race through the whole zoom range in one flick.
_ZOOM_STEP = 1.1
_WHEEL_NOTCH = 120.0

#: Grid cells of the axial, coronal and sagittal views (the 3D card is the fourth).
_VIEW_CELLS = ((0, 0), (0, 1), (1, 0))

#: Default cine rate of the time/energy play button.
_PLAY_FPS = 8

#: Projections the views can show instead of a slice (key -> description).
PROJECTIONS: dict[str, str] = {
    "max": "Maximum intensity projection",
    "min": "Minimum intensity projection",
    "mean": "Average intensity projection",
}
#: Short names, for the menu and the view titles.
_PROJECTION_SHORT: dict[str, str] = {"max": "MIP", "min": "MinIP", "mean": "AIP"}
#: Projections kept per layer (axis × slab), oldest dropped first.
_PROJECTION_CACHE_SIZE = 12
#: Above this many voxels a projection is reduced in parallel chunks.
_PROJECTION_CHUNK_VOXELS = 16_000_000

#: How close to a crosshair line a press must be to grab it, and how far from the
#: centre it must be for that grab to mean "rotate" rather than "move".
_HANDLE_TOL_PX = 10.0
_HANDLE_MIN_FRACTION = 0.55

#: Largest volume for which a per-axis contiguous copy is worth its memory.
#:
#: Measured, rather than guessed. On a 419 MB volume a strided read of the last
#: axis takes 0.94 ms against 0.04 ms from a reordered copy: a real speedup, but
#: one that costs 458 ms to build and only repays after ~500 slices on that axis,
#: while 0.94 ms was already far inside a frame. The old 512 MB ceiling let every
#: layer reorder itself: five layers of a 241 MB study built 2.4 GB of copies on
#: panel open. Small volumes still take the fast path, where it is nearly free.
_SLICE_CACHE_BUDGET = 64 * 1024 * 1024

#: Total reordered bytes allowed across every layer the panel draws. The
#: per-array limit above says nothing about how many arrays there are.
_SLICE_CACHE_TOTAL = 256 * 1024 * 1024


#: Anatomical opposite of each axis code.
_OPPOSITE: dict[str, str] = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}

#: What belongs at the top and at the right of each plane, in neurological
#: convention: the viewer's right is the patient's right, and superior is up.
_PLANE_UP_RIGHT: dict[str, tuple[str, str]] = {
    "Axial": ("A", "R"),
    "Coronal": ("S", "R"),
    "Sagittal": ("S", "A"),
}
