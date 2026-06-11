"""Headless matplotlib / PyVista setup for SGE and batch QC."""

from __future__ import annotations

import os


def configure_headless_viz() -> None:
    """Prepare off-screen rendering when no display is available (cluster nodes)."""
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=False)
    except Exception:
        pass
    try:
        import pyvista as pv

        if not os.environ.get("DISPLAY"):
            os.environ["PYVISTA_OFF_SCREEN"] = "true"
            if hasattr(pv, "start_xvfb"):
                try:
                    pv.start_xvfb()
                except Exception:
                    pass
    except ImportError:
        pass


__all__ = ["configure_headless_viz"]
