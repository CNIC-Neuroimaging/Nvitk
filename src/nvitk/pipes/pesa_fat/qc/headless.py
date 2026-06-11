"""Headless matplotlib / PyVista setup for SGE and batch QC."""

from __future__ import annotations

import html
import importlib.util
import os
import time
from functools import lru_cache
from pathlib import Path

from nvitk.core.logger import Logger

log = Logger()

_TRAME_WARNED = False


def _headless_mode() -> bool:
    if os.environ.get("NVITK_HEADLESS", "").lower() in ("1", "true", "yes"):
        return True
    return not bool(os.environ.get("DISPLAY"))


def configure_headless_viz() -> None:
    """Prepare off-screen rendering when no display is available (cluster nodes).

    Must be called before the first ``import pyvista`` / VTK use.
    """
    if _headless_mode():
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
        # Prefer software GL on CPU nodes without /dev/dri access (avoids EGL probing).
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    else:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=False)
    except Exception:
        pass
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        if _headless_mode():
            os.environ["PYVISTA_OFF_SCREEN"] = "true"
        else:
            if hasattr(pv, "start_xvfb"):
                try:
                    pv.start_xvfb()
                except Exception:
                    pass
    except ImportError:
        pass


@lru_cache(maxsize=1)
def trame_export_available() -> bool:
    """Return True when PyVista ``export_html`` dependencies are importable."""
    for name in ("trame", "trame_vtk", "trame_vuetify"):
        if importlib.util.find_spec(name) is None:
            return False
    return True


def warn_if_trame_missing() -> None:
    global _TRAME_WARNED
    if _TRAME_WARNED or trame_export_available():
        return
    _TRAME_WARNED = True
    log.warning(
        "PyVista HTML export needs trame (pip install 'pyvista[jupyter]'); "
        "3D QC embeds will use fallback placeholders."
    )


def write_export_fallback_html(path: Path, message: str, *, retries: int = 3, sleep_s: float = 1.0) -> bool:
    """Write a minimal HTML page when interactive export fails."""
    text = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/></head>"
        f"<body><p>{html.escape(message)}</p></body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, int(retries) + 1):
        try:
            path.write_text(text, encoding="utf-8")
            return True
        except OSError as exc:
            log.warning(
                "write fallback HTML failed (%s) [attempt %d/%d]: %s",
                path,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(float(sleep_s))
    return False


def export_plotter_html(
    pl,
    path: Path,
    *,
    retries: int = 3,
    sleep_s: float = 1.0,
    fallback_message: str | None = None,
) -> bool:
    """Export a PyVista plotter to HTML with retries; optional fallback page on failure."""
    warn_if_trame_missing()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trame_export_available():
        msg = fallback_message or (
            "Interactive 3D export unavailable: install trame (pip install 'pyvista[jupyter]')."
        )
        return write_export_fallback_html(path, msg)

    last_exc: Exception | None = None
    try:
        for attempt in range(1, int(retries) + 1):
            try:
                pl.export_html(str(path))
                return True
            except OSError as exc:
                last_exc = exc
                log.warning(
                    "export_html failed (%s) [attempt %d/%d]: %s",
                    path,
                    attempt,
                    retries,
                    exc,
                )
                if attempt < retries:
                    time.sleep(float(sleep_s))
            except Exception as exc:
                last_exc = exc
                log.warning("export_html failed (%s): %s", path, exc)
                break
    finally:
        try:
            pl.close()
        except Exception:
            pass

    if last_exc is not None:
        log.warning("Giving up exporting %s after %d attempt(s).", path, retries)
        if fallback_message is not None:
            return write_export_fallback_html(
                path,
                f"{fallback_message} ({last_exc!s})",
            )
    return False


__all__ = [
    "configure_headless_viz",
    "export_plotter_html",
    "trame_export_available",
    "warn_if_trame_missing",
    "write_export_fallback_html",
]
