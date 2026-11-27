"""Plane geometry of the orthogonal views: which array axis each view steps along, how it is
oriented on screen, and the oblique planes a dragged crosshair end turns them onto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nvitk.gui.core.orientation import layer_orientation_codes
from nvitk.gui.viz.ortho.constants import (
    _PLANE_NAMES,
    _OPPOSITE,
    _PLANE_UP_RIGHT,
)

@dataclass(frozen=True)
class AxisView:
    """One orthogonal view: which array axis it steps along, and how it is drawn."""

    axis: int
    title: str
    #: Array axis drawn down the image, and whether its direction is reversed.
    rows: int
    flip_rows: bool = False
    #: Array axis drawn across the image, and whether its direction is reversed.
    cols: int = 0
    flip_cols: bool = False

    def to_pixel(self, position: tuple[int, ...] | list[int], shape: tuple[int, ...]) -> tuple[int, int]:
        """Voxel *position* as a (row, column) pixel in this view's rendered slice."""
        row = int(position[self.rows])
        col = int(position[self.cols])
        if self.flip_rows:
            row = int(shape[self.rows]) - 1 - row
        if self.flip_cols:
            col = int(shape[self.cols]) - 1 - col
        return row, col

    def to_voxel(self, row: int, col: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """A (row, column) pixel back to ``(row axis, index, column axis, index)``."""
        r, c = int(row), int(col)
        if self.flip_rows:
            r = int(shape[self.rows]) - 1 - r
        if self.flip_cols:
            c = int(shape[self.cols]) - 1 - c
        return self.rows, r, self.cols, c


def _orient_in_plane(axis: int, codes: str, plane: str) -> tuple[int, bool, int, bool]:
    """Choose which in-plane axis is drawn down and which across, and their signs.

    A raw NumPy slice is drawn in array-axis order, which for a NIfTI volume is
    rarely the way anyone reads that plane: an axial slice of an RAS volume comes
    out with the patient running down the image and anterior to the right. This
    picks the row and column axes from the anatomical codes and flips whichever
    run the wrong way, so up is superior (or anterior, on an axial) and the
    viewer's right is the patient's right.
    """
    rest = [i for i in range(3) if i != axis]
    up, right = _PLANE_UP_RIGHT.get(plane, ("", ""))
    codes_u = codes.upper()

    def code_of(i: int) -> str:
        return codes_u[i] if i < len(codes_u) else ""

    if up and right:
        vertical = [i for i in rest if code_of(i) in (up, _OPPOSITE[up])]
        horizontal = [i for i in rest if code_of(i) in (right, _OPPOSITE[right])]
        if len(vertical) == 1 and len(horizontal) == 1 and vertical[0] != horizontal[0]:
            row_axis, col_axis = vertical[0], horizontal[0]
            # Rows increase downward, so an axis pointing at "up" must be reversed.
            return (
                row_axis,
                code_of(row_axis) == up,
                col_axis,
                code_of(col_axis) == _OPPOSITE[right],
            )
    # No usable codes: fall back to array order, unflipped.
    return rest[0], False, rest[1], False


#: Reading order of the three planes, matching how a viewer like 3D Slicer lays
#: them out. Views whose plane cannot be named keep their array-axis order.
_PLANE_ORDER: dict[str, int] = {"Axial": 0, "Coronal": 1, "Sagittal": 2}


def _anatomical_views(layer: Any) -> list[AxisView]:
    """The three views in one fixed anatomical convention (superior / anterior up,
    the patient's right on the viewer's right), from the axis codes alone, for a
    layer whose affine says nothing usable."""
    codes = layer_orientation_codes(layer) or ""
    views: list[AxisView] = []
    for axis in range(3):
        code = codes[axis].upper() if axis < len(codes) else ""
        plane = _PLANE_NAMES.get(code, f"Axis {axis}")
        label = f"{plane}" + (f"  ·  {code}" if code else "")
        row_axis, flip_rows, col_axis, flip_cols = _orient_in_plane(axis, codes, plane)
        views.append(AxisView(axis=axis, title=label, rows=row_axis, flip_rows=flip_rows,
                              cols=col_axis, flip_cols=flip_cols))
    views.sort(key=lambda v: (_PLANE_ORDER.get(v.title.split(" ")[0], 3), v.axis))
    return views


#: In-plane axes of each plane: the codes drawn down the view and across it.
_PLANE_AXES: dict[str, tuple[str, str]] = {
    "Axial": ("AP", "RL"),
    "Coronal": ("SI", "RL"),
    "Sagittal": ("SI", "AP"),
}


def _axis_views(layer: Any, viewer: Any = None) -> list[AxisView]:
    """The three orthogonal views for *layer*, laid out in the image's own orientation.

    The orientation codes say which way each voxel axis runs, and the views put that
    direction to the right (across) and down (anterior-posterior on the axial plane):
    superior-inferior keeps its letter at the top: an RAS image reads left to right L->R
    and top to bottom P->A, an LPS one (a DICOM series) R->L and A->P, superior up in both.
    When the display has been reoriented (RAS -> LPS by the orientation tools, which
    mirrors the affine and leaves the voxels where they are), the views follow the new
    codes over the voxels' real anatomy, so they mirror as the canvas does. Slicing
    *along* an axis shows the plane perpendicular to it: the superior axis gives axial
    slices, the anterior axis coronal, the left-right axis sagittal. Napari's own view
    settings (Ctrl+T, axis rolls, the camera) are the main window's alone.
    """
    from nvitk.gui.core.orientation import layer_true_orientation_codes

    codes = (layer_orientation_codes(layer) or "").upper()
    true = (layer_true_orientation_codes(layer) or codes).upper()
    if len(codes) != 3 or len(true) != 3:
        return _anatomical_views(layer)
    pair_of = {c: "RL" if c in "RL" else "AP" if c in "AP" else "SI" for c in codes}
    views: list[AxisView] = []
    for axis in range(3):
        code = codes[axis]
        plane = _PLANE_NAMES.get(code, f"Axis {axis}")
        if plane not in _PLANE_AXES:
            return _anatomical_views(layer)
        down_pair, across_pair = _PLANE_AXES[plane]
        rest = [d for d in range(3) if d != axis]
        row = next((d for d in rest if pair_of[codes[d]] == down_pair), None)
        col = next((d for d in rest if pair_of[codes[d]] == across_pair), None)
        if row is None or col is None or row == col:
            return _anatomical_views(layer)
        # A voxel axis runs where its code says unless the display mirrored it.
        kept_row, kept_col = true[row] == codes[row], true[col] == codes[col]
        # Across (and down, on the axial plane): the code's direction to the right /
        # bottom. Superior-inferior: the code's letter at the top.
        flip_rows = kept_row if down_pair == "SI" else not kept_row
        flip_cols = not kept_col
        views.append(AxisView(axis=axis, title=f"{plane}  ·  {code}", rows=row, flip_rows=flip_rows,
                              cols=col, flip_cols=flip_cols))
    views.sort(key=lambda v: (_PLANE_ORDER.get(v.title.split(" ")[0], 3), v.axis))
    return views


def _unit(vector: np.ndarray) -> np.ndarray:
    """*vector* scaled to unit length, or unchanged when it is degenerate."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass(frozen=True)
class PlaneFrame:
    """One view's plane: unit row/column directions and its normal, in millimetres.

    Millimetres, not voxels: on anisotropic spacing a rotation applied to raw
    array-axis vectors shears the plane rather than turning it, and the angle the
    user dragged is not the angle they get.
    """

    row: np.ndarray
    col: np.ndarray
    normal: np.ndarray

    def is_close_to(self, other: PlaneFrame, tol: float = 1e-9) -> bool:
        """True when this frame is the same plane and orientation as *other*."""
        return bool(
            np.allclose(self.row, other.row, atol=tol)
            and np.allclose(self.col, other.col, atol=tol)
            and np.allclose(self.normal, other.normal, atol=tol)
        )


def base_frame(view: AxisView, spacing: Sequence[float]) -> PlaneFrame:
    """The axis-aligned frame *view* starts from, in millimetre space."""
    sp = np.asarray(spacing, dtype=float)[:3]
    row = np.zeros(3)
    row[view.rows] = -1.0 if view.flip_rows else 1.0
    col = np.zeros(3)
    col[view.cols] = -1.0 if view.flip_cols else 1.0
    row_mm = _unit(row * sp)
    col_mm = _unit(col * sp)
    return PlaneFrame(row_mm, col_mm, _unit(np.cross(row_mm, col_mm)))


def rotation_about(axis_mm: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix turning by *angle_rad* about a unit axis."""
    k = _unit(np.asarray(axis_mm, dtype=float))
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * kx + (1.0 - np.cos(angle_rad)) * (kx @ kx)


def rotate_frame(frame: PlaneFrame, axis_mm: np.ndarray, angle_rad: float) -> PlaneFrame:
    """*frame* turned about *axis_mm*, re-normalised against drift."""
    rot = rotation_about(axis_mm, float(angle_rad))
    return PlaneFrame(
        _unit(rot @ frame.row), _unit(rot @ frame.col), _unit(rot @ frame.normal)
    )


def oblique_slice(
    data: np.ndarray,
    *,
    center_vox: Sequence[float],
    anchor_px: tuple[float, float],
    frame: PlaneFrame,
    shape: tuple[int, int],
    steps_mm: tuple[float, float],
    spacing: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """Sample the plane *frame* through *center_vox* onto a ``shape`` pixel grid.

    *anchor_px* is the pixel that lands on *center_vox*. Anchoring on the
    crosshair rather than a corner is what makes an unrotated frame reproduce the
    axis-aligned slice exactly, and keeps the crosshair still on screen while the
    plane turns under it.
    """
    # Slab-parallel on the shared worker pool, bit-identical to SciPy's.
    from nvitk.transform.threaded import map_coordinates

    height, width = int(shape[0]), int(shape[1])
    sp = np.asarray(spacing, dtype=float)[:3]
    centre = np.asarray(center_vox, dtype=float)[:3]
    # A step of one pixel is steps_mm along the frame direction, converted back
    # into voxel indices by the spacing.
    step_row = float(steps_mm[0]) * np.asarray(frame.row, dtype=float) / sp
    step_col = float(steps_mm[1]) * np.asarray(frame.col, dtype=float) / sp
    rows = (np.arange(height, dtype=float) - float(anchor_px[0]))[:, None, None]
    cols = (np.arange(width, dtype=float) - float(anchor_px[1]))[None, :, None]
    coords = centre[None, None, :] + rows * step_row[None, None, :] + cols * step_col[None, None, :]
    return map_coordinates(
        data, np.moveaxis(coords, 2, 0), order=int(order), mode="constant", cval=0.0
    )


def line_direction_px(
    view_frame: PlaneFrame,
    other_normal: np.ndarray,
    steps_mm: tuple[float, float],
) -> tuple[float, float] | None:
    """``(d_row, d_col)`` of another plane's trace across this view, in pixels.

    Two planes meet along ``cross(n_a, n_b)``; drawn on this view that is the line
    marking where the other view is cutting. ``None`` when the planes are parallel
    and there is no trace to draw.
    """
    direction = np.cross(np.asarray(view_frame.normal, float), np.asarray(other_normal, float))
    if float(np.linalg.norm(direction)) <= 1e-9:
        return None
    d_row = float(np.dot(direction, view_frame.row)) / max(float(steps_mm[0]), 1e-9)
    d_col = float(np.dot(direction, view_frame.col)) / max(float(steps_mm[1]), 1e-9)
    if abs(d_row) < 1e-12 and abs(d_col) < 1e-12:
        return None
    norm = float(np.hypot(d_row, d_col))
    return (d_row / norm, d_col / norm)


@dataclass(frozen=True)
class PlaneMap:
    """One view's plane, as a map between its slice pixels and world space.

    Built by the panel from the bound volume's real voxel -> world matrix, the crosshair voxel
    and the view's frame: the same numbers :func:`oblique_slice` samples with, so a point mapped
    here lands on the pixel the slice shows it at, turned planes included. Pixel coordinates are
    voxel-centred (pixel ``(r, c)`` is drawn over ``[r, r + 1) x [c, c + 1)``).
    """

    #: Bound volume voxel -> world, spatial 4x4 (world in the viewer's last three axes).
    matrix: np.ndarray
    #: Crosshair voxel (fractional), and the pixel it is drawn at.
    centre_vox: np.ndarray
    anchor_px: tuple[float, float]
    frame: PlaneFrame
    steps_mm: tuple[float, float]
    spacing: np.ndarray

    def pixel_to_world(self, row: float, col: float) -> np.ndarray:
        """World point (3,) of the fractional, voxel-centred slice pixel ``(row, col)``."""
        sp = np.asarray(self.spacing, dtype=float)
        step_row = float(self.steps_mm[0]) * np.asarray(self.frame.row, dtype=float) / sp
        step_col = float(self.steps_mm[1]) * np.asarray(self.frame.col, dtype=float) / sp
        vox = (np.asarray(self.centre_vox, dtype=float)
               + (float(row) - float(self.anchor_px[0])) * step_row
               + (float(col) - float(self.anchor_px[1])) * step_col)
        return (self.matrix @ np.append(vox, 1.0))[:3]

    def to_plane_mm(self, world: np.ndarray) -> np.ndarray:
        """World points ``(N, 3)`` as millimetre offsets from the crosshair, in the bound
        volume's (spacing-scaled) voxel axes: the frame's own space."""
        world = np.atleast_2d(np.asarray(world, dtype=float))
        inv = np.linalg.inv(self.matrix)
        vox = world @ inv[:3, :3].T + inv[:3, 3]
        return (vox - np.asarray(self.centre_vox, dtype=float)) * np.asarray(self.spacing, dtype=float)

    def mm_to_pixels(self, mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, depth_mm)`` of plane-millimetre points (see :meth:`to_plane_mm`):
        voxel-centred pixels, and the signed distance off the plane."""
        mm = np.atleast_2d(np.asarray(mm, dtype=float))
        rows = float(self.anchor_px[0]) + mm @ np.asarray(self.frame.row, float) / max(float(self.steps_mm[0]), 1e-9)
        cols = float(self.anchor_px[1]) + mm @ np.asarray(self.frame.col, float) / max(float(self.steps_mm[1]), 1e-9)
        return rows, cols, mm @ np.asarray(self.frame.normal, float)

    def world_to_pixels(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, depth_mm)`` of world points ``(N, 3)``."""
        return self.mm_to_pixels(self.to_plane_mm(world))

    def world_normal(self) -> np.ndarray:
        """The plane's normal as a unit world direction."""
        vox = np.asarray(self.frame.normal, dtype=float) / np.asarray(self.spacing, dtype=float)
        return _unit(self.matrix[:3, :3] @ vox)

    def thickness_mm(self) -> float:
        """One voxel step across the plane, in millimetres (the slice's thickness)."""
        n = np.asarray(self.frame.normal, dtype=float) / np.asarray(self.spacing, dtype=float)
        norm = float(np.linalg.norm(n))
        return 1.0 / norm if norm > 1e-12 else 1.0

    def key(self) -> bytes:
        """Identity of the plane: equal keys map every point to the same pixel."""
        parts = [self.matrix, self.centre_vox, np.asarray(self.anchor_px, float), self.frame.row,
                 self.frame.col, self.frame.normal, np.asarray(self.steps_mm, float), self.spacing]
        return b"".join(np.ascontiguousarray(np.asarray(p, dtype=float)).tobytes() for p in parts)
