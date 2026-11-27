"""Transform mode in the slice views: drag a layer in a view's plane, Ctrl+drag to turn it.

See :mod:`nvitk.gui.viz.layer_transform`. A drag here is mapped through the view's
:class:`~nvitk.gui.viz.ortho.geometry.PlaneMap`, so it follows the cursor exactly, turned
(oblique) planes included: the point under the cursor at the press stays under it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt


class OrthoTransformDrag:
    """``SliceView.transform_handler`` for every view of an
    :class:`~nvitk.gui.viz.ortho.panel.OrthoViewerPanel`."""

    def __init__(self, panel: Any) -> None:
        self._panel = panel
        self._session: Any = None
        self._map: Any = None
        self._start: np.ndarray | None = None
        self._rotate = False
        self._normal: np.ndarray | None = None

    def active(self) -> bool:
        """Whether a drag is going on."""
        return self._session is not None

    def targets(self) -> list[Any]:
        from nvitk.gui.viz.layer_transform import transform_targets

        return transform_targets(self._panel._viewer)

    def handle(self, view_index: int, kind: str, pixel: Any, event: Any) -> bool:
        """*kind* ``can`` (hover: would a press move a layer?), ``press``, ``move``, ``release``
        or ``cancel``; *pixel* the cursor as fractional slice-pixel edges ``(row, col)``."""
        if kind == "can":
            return bool(self.targets()) and self._panel.plane_map(view_index) is not None
        if kind == "press":
            return self._press(view_index, pixel, event)
        if self._session is None:
            return False
        if kind == "move":
            self._move(pixel)
            return True
        if kind == "cancel":
            self._session.cancel()
            self._end(finish=False)
            return True
        if kind == "release":
            self._end(finish=True)
            return True
        return False

    def _press(self, view_index: int, pixel: Any, event: Any) -> bool:
        from nvitk.gui.viz.layer_transform import TransformSession, pin_ortho_reference

        if pixel is None or event.button() != Qt.LeftButton:
            return False
        targets = self.targets()
        if not targets:
            return False
        viewer = self._panel._viewer
        # Where the press is, in the world, on the layout the user pressed on: the views are
        # normally already off a layer in Transform mode, and if pinning still re-lays them the
        # point grabbed stays the one under the cursor.
        before = self._panel.plane_map(view_index)
        if before is None:
            return False
        start = before.pixel_to_world(float(pixel[0]) - 0.5, float(pixel[1]) - 0.5)
        pin_ortho_reference(viewer, targets)
        plane = self._panel.plane_map(view_index)
        if plane is None:
            return False
        session = TransformSession(viewer, targets)
        if session.centre is None:
            return False
        self._session, self._map = session, plane
        self._start = start
        self._rotate = bool(event.modifiers() & Qt.ControlModifier)
        self._normal = plane.world_normal()
        return True

    def _move(self, pixel: Any) -> None:
        from nvitk.gui.viz.layer_transform import rotation_about_point, translation

        if pixel is None or self._start is None:
            return
        here = self._map.pixel_to_world(float(pixel[0]) - 0.5, float(pixel[1]) - 0.5)
        session = self._session
        if not self._rotate:
            session.apply(translation(here - self._start))
            return
        # Turn about the view's axis through the layers' centre, by the angle the cursor swept
        # round it (both vectors flattened into the plane).
        n = self._normal
        u0 = self._start - session.centre
        u1 = here - session.centre
        u0 = u0 - np.dot(u0, n) * n
        u1 = u1 - np.dot(u1, n) * n
        if np.linalg.norm(u0) < 1e-6 or np.linalg.norm(u1) < 1e-6:
            return
        angle = float(np.arctan2(np.dot(n, np.cross(u0, u1)), np.dot(u0, u1)))
        session.apply(rotation_about_point(n, angle, session.centre))

    def _end(self, *, finish: bool) -> None:
        session, self._session = self._session, None
        self._map = self._start = self._normal = None
        if finish and session is not None:
            session.finish()


__all__ = ["OrthoTransformDrag"]
