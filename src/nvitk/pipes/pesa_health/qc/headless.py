"""Headless matplotlib / PyVista setup for SGE and batch QC."""

from __future__ import annotations

import base64
import html
import io
import os
import time
from functools import lru_cache
from pathlib import Path

from nvitk.core.logger import Logger

log = Logger()

_TRAME_WARNED = False


def _headless_mode() -> bool:
    """True if ``NVITK_HEADLESS`` is set truthy, or (by default) no ``DISPLAY`` is available."""
    if os.environ.get("NVITK_HEADLESS", "").lower() in ("1", "true", "yes"):
        return True
    return not bool(os.environ.get("DISPLAY"))


def configure_headless_viz() -> None:
    """Prepare off-screen rendering when no display is available (cluster nodes).

    Must be called before the first ``import pyvista`` / VTK use.
    """
    if _headless_mode():
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
        # LIBGL_ALWAYS_SOFTWARE conflicts with VTK EGL offscreen on SGE nodes
        # ("Not allowed to force software rendering when API explicitly selects
        # a hardware device") and causes long /dev/dri permission-denied probes.
        for key in ("LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE", "GALLIUM_DRIVER"):
            os.environ.pop(key, None)
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
            if hasattr(pv, "start_xvfb"):
                try:
                    pv.start_xvfb()
                except Exception:
                    pass
        elif hasattr(pv, "start_xvfb"):
            try:
                pv.start_xvfb()
            except Exception:
                pass
    except ImportError:
        pass


@lru_cache(maxsize=1)
def _trame_export_import_error() -> str | None:
    """Return an error string when PyVista ``export_html`` deps are missing."""
    try:
        from trame_vtk.tools.vtksz2html import write_html  # noqa: F401, PLC0415
        from pyvista.trame import PyVistaLocalView  # noqa: F401, PLC0415
    except ImportError as exc:
        return str(exc)
    return None


@lru_cache(maxsize=1)
def trame_export_available() -> bool:
    """Return True when PyVista ``export_html`` dependencies are importable."""
    return _trame_export_import_error() is None


def warn_if_trame_missing() -> None:
    """Log a one-time warning (via :data:`_TRAME_WARNED`) if trame-vtk isn't available for interactive
    HTML export."""
    global _TRAME_WARNED
    if _TRAME_WARNED or trame_export_available():
        return
    _TRAME_WARNED = True
    detail = _trame_export_import_error() or "unknown import error"
    log.warning(
        "PyVista HTML export needs trame-vtk (pip install 'pyvista[jupyter]' or trame-vtk); "
        "3D QC embeds will use static screenshot fallbacks. (%s)",
        detail,
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


def _write_plotter_screenshot_html(pl, path: Path, *, note: str = "") -> bool:
    """Embed an off-screen plotter screenshot in a minimal HTML page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buf = io.BytesIO()
        pl.screenshot(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        note_html = f"<p>{html.escape(note)}</p>" if note else ""
        text = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
            "<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}"
            "img{display:block;max-width:100%;height:auto;margin:0 auto}</style>"
            "</head><body>"
            f"{note_html}"
            f"<img alt='3D scene' src='data:image/png;base64,{b64}'/>"
            "</body></html>"
        )
        path.write_text(text, encoding="utf-8")
        return True
    except Exception as exc:
        log.warning("screenshot HTML fallback failed (%s): %s", path, exc)
        return False


def export_plotter_html(
    pl,
    path: Path,
    *,
    retries: int = 3,
    sleep_s: float = 1.0,
    fallback_message: str | None = None,
) -> bool:
    """Export a PyVista plotter to HTML with retries; screenshot fallback on failure."""
    warn_if_trame_missing()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _screenshot_or_text(note: str) -> bool:
        """Write a static screenshot HTML page, falling back to a plain-text notice page on failure."""
        if _write_plotter_screenshot_html(pl, path, note=note):
            return True
        return write_export_fallback_html(path, note)

    if not trame_export_available():
        note = fallback_message or (
            "Interactive 3D export unavailable: install trame-vtk "
            "(pip install 'pyvista[jupyter]')."
        )
        try:
            return _screenshot_or_text(note)
        finally:
            try:
                pl.close()
            except Exception:
                pass

    last_exc: Exception | None = None
    ok = False
    try:
        for attempt in range(1, int(retries) + 1):
            try:
                pl.export_html(str(path))
                ok = True
                break
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
        if not ok:
            if last_exc is not None:
                log.warning("Giving up exporting %s after %d attempt(s).", path, retries)
            note = fallback_message or "Interactive 3D export failed."
            if last_exc is not None:
                note = f"{note} ({last_exc!s})"
            ok = _screenshot_or_text(note)
        try:
            pl.close()
        except Exception:
            pass

    return ok


__all__ = [
    "configure_headless_viz",
    "export_plotter_html",
    "trame_export_available",
    "warn_if_trame_missing",
    "write_export_fallback_html",
]
