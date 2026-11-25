"""The central viewport: slice views and the 3D canvas in one switchable layout.

* :mod:`.layouts`: the layout presets.
* :mod:`.pane3d`: Napari's viewer under a header with the 3D options pop-over.
* :mod:`.viewport`: :class:`Viewport` and :func:`install_viewport`.
"""

from __future__ import annotations

from nvitk.gui.viewport.layouts import PRESETS, LayoutPreset
from nvitk.gui.viewport.viewport import Viewport, install_viewport

__all__ = ["LayoutPreset", "PRESETS", "Viewport", "install_viewport"]
