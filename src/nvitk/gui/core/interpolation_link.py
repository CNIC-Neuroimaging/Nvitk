"""One interpolation per image layer, in the slice views and the 3D view alike.

Napari keeps two interpolations per Image layer, ``interpolation2d`` and ``interpolation3d``, and
its layer controls change only the one of the display mode the controls think they are in. nvitk's
slice views draw with the 2D one and the 3D pane with the 3D one, so changing "Interpolation" in
the layer controls changed the slice views and left the 3D view as it was.

:func:`install_interpolation_link` keeps the two in step for every image layer: a change of either
is copied to the other. The 3D renderer knows only ``nearest`` and ``linear``, so a smooth 2D mode
(``cubic``, ``spline36``...) becomes ``linear`` in 3D, and a 3D ``linear`` leaves a smooth 2D mode
as it is.
"""

from __future__ import annotations

from typing import Any

_LINKED = "_nvitk_interp_linked"
_BUSY = "_nvitk_interp_syncing"


def _value(mode: Any) -> str:
    return str(getattr(mode, "value", mode) or "nearest").lower()


def _to_3d(mode_2d: str) -> str:
    return "nearest" if mode_2d == "nearest" else "linear"


def _on_2d(layer: Any) -> None:
    if getattr(layer, _BUSY, False):
        return
    wanted = _to_3d(_value(layer.interpolation2d))
    if _value(getattr(layer, "interpolation3d", wanted)) == wanted:
        return
    setattr(layer, _BUSY, True)
    try:
        layer.interpolation3d = wanted
    except Exception:  # noqa: BLE001 (a layer type without a 3D interpolation)
        pass
    finally:
        setattr(layer, _BUSY, False)


def _on_3d(layer: Any) -> None:
    if getattr(layer, _BUSY, False):
        return
    mode_3d = _value(layer.interpolation3d)
    mode_2d = _value(layer.interpolation2d)
    if _to_3d(mode_2d) == mode_3d:
        return  # already consistent (a smooth 2D mode shows as linear in 3D)
    setattr(layer, _BUSY, True)
    try:
        layer.interpolation2d = mode_3d
    except Exception:  # noqa: BLE001
        pass
    finally:
        setattr(layer, _BUSY, False)


def link_layer(layer: Any) -> None:
    """Keep *layer*'s 2D and 3D interpolation in step (no-op for non-image layers)."""
    if type(layer).__name__ != "Image" or getattr(layer, _LINKED, False):
        return
    events = getattr(layer, "events", None)
    if events is None or not hasattr(events, "interpolation2d") or not hasattr(events, "interpolation3d"):
        return
    events.interpolation2d.connect(lambda _e=None, l=layer: _on_2d(l))
    events.interpolation3d.connect(lambda _e=None, l=layer: _on_3d(l))
    setattr(layer, _LINKED, True)
    # Not synced at insertion: Napari's defaults (2D nearest, 3D linear) keep a volume render
    # smooth; from the first change on, the two follow each other.


def install_interpolation_link(viewer: Any) -> None:
    """Link every image layer of *viewer*, present and future."""
    for layer in list(viewer.layers):
        link_layer(layer)
    viewer.layers.events.inserted.connect(lambda event: link_layer(getattr(event, "value", None)))


__all__ = ["install_interpolation_link", "link_layer"]
