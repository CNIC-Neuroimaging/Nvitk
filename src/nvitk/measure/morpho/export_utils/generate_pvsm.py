#!/usr/bin/env python3
"""Generate a ParaView state (.pvsm) for a case folder.

Creates two side-by-side render views:
  - View 1: surfaces (opacity 0.3) + centerlines coloured by StenosisBinary
  - View 2: surfaces (opacity 0.3) + centerlines coloured by EnlargementBinary

Edit the CONFIG block below for direct single-case runs.
The batch runner sets CASE_DIR / OUTPUT_PVSM via environment variables or
calls main() after patching the module-level variables directly.
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import sys


# ============================================================================
# CONFIG FOR DIRECT RUN (paths must be supplied at runtime; no host defaults)
# ============================================================================
CASE_DIR = None
OUTPUT_PVSM = None  # None → <CASE_DIR>/<folder_name>.pvsm
QUIET = True
PRINT_FINAL_STATUS = True
# ============================================================================

# Allow overriding via environment variables (used by the batch runner).
CASE_DIR = os.environ.get("GENERATE_PVSM_CASE_DIR", CASE_DIR)
OUTPUT_PVSM = os.environ.get("GENERATE_PVSM_OUTPUT_PVSM", OUTPUT_PVSM) or None


@contextlib.contextmanager
def suppress_console_output(enabled=True):
    """Context manager: redirect stdout/stderr to ``/dev/null`` while active (silences noisy ParaView calls)."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def log(message):
    """Print *message* unless the module-level ``QUIET`` flag is set."""
    if not QUIET:
        print(message)


def find_vtps(folder):
    """Sorted list of ``.vtp`` file paths directly under *folder* (empty if it doesn't exist)."""
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.vtp")))


def extract_vessel_name(filename):
    """Derive a vessel display name from a centerline VTP filename (strips ``_radius`` and old numeric prefixes)."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_radius$", "", name)
    old_style = re.match(r"^\d+_(.+)$", name)
    return old_style.group(1) if old_style else name


def _show_surface(pv, src, view):
    """Display a surface source in a ParaView view as a translucent shaded surface."""
    try:
        with suppress_console_output(QUIET):
            display = pv.Show(src, view)
        try:
            with suppress_console_output(QUIET):
                display.Representation = "Surface"
                display.Opacity = 0.3
        except Exception:
            pass
    except Exception:
        pass


def _show_centerline(pv, src, view, color_array):
    """Display a centerline source in a ParaView view, colored by *color_array* with a visible scalar bar."""
    try:
        with suppress_console_output(QUIET):
            display = pv.Show(src, view)
        try:
            with suppress_console_output(QUIET):
                pv.ColorBy(display, ("POINTS", color_array))
                pv.GetColorTransferFunction(color_array)
                display.RescaleTransferFunctionToDataRange(True)
                display.SetScalarBarVisibility(view, True)
        except Exception:
            try:
                with suppress_console_output(QUIET):
                    display.ColorArrayName = ["POINTS", color_array]
                    display.RescaleTransferFunctionToDataRange(True)
            except Exception:
                pass
        try:
            with suppress_console_output(QUIET):
                display.LineWidth = 5
        except Exception:
            pass
    except Exception as e:
        log(f"Warning: failed to show centerline coloured by {color_array}: {e}")


def main():
    """Direct-run entry point: build and save a ParaView ``.pvsm`` scene for a stage7 case folder."""
    if not CASE_DIR:
        raise SystemExit(
            "Set GENERATE_PVSM_CASE_DIR or CASE_DIR to a stage7 output folder "
            "(<output_root>/<subject>/qvtpy/stage7_morphometrics/)."
        )
    case_dir = os.path.abspath(CASE_DIR)
    if not os.path.isdir(case_dir):
        log(f"Case directory not found: {case_dir}")
        return

    surfaces_dir = os.path.join(case_dir, "surfaces")
    centerlines_dir = os.path.join(case_dir, "centerlines")

    surface_files = find_vtps(surfaces_dir)
    centerline_files = find_vtps(centerlines_dir)

    if not surface_files and not centerline_files:
        log("No .vtp files found under surfaces/ or centerlines/.")
        return

    try:
        with suppress_console_output(QUIET):
            import paraview.simple as pv  # type: ignore
    except Exception as e:
        log(f"Failed to import paraview.simple: {e}")
        return

    try:
        with suppress_console_output(QUIET):
            pv._DisableFirstRenderCameraReset()
    except Exception:
        pass

    with suppress_console_output(QUIET):
        view1 = pv.GetActiveViewOrCreate("RenderView")
        layout = pv.GetLayout()
        layout.SplitHorizontal(0, 0.5)
        view2 = pv.CreateView("RenderView")
        pv.AssignViewToLayout(view=view2, layout=layout, hint=2)

    surface_sources = []
    for f in surface_files:
        try:
            with suppress_console_output(QUIET):
                src = pv.OpenDataFile(f)
                pv.RenameSource(f"{extract_vessel_name(os.path.basename(f))}_surf", src)
                surface_sources.append(src)
        except Exception as e:
            log(f"Warning: failed to load surface {f}: {e}")

    for src in surface_sources:
        _show_surface(pv, src, view1)
        _show_surface(pv, src, view2)

    centerline_sources = []
    for f in centerline_files:
        try:
            with suppress_console_output(QUIET):
                src = pv.OpenDataFile(f)
                pv.RenameSource(f"{extract_vessel_name(os.path.basename(f))}_ctln", src)
                centerline_sources.append(src)
        except Exception as e:
            log(f"Warning: failed to load centerline {f}: {e}")

    for src in centerline_sources:
        _show_centerline(pv, src, view1, "StenosisBinary")
        _show_centerline(pv, src, view2, "EnlargementBinary")

    for view in (view1, view2):
        try:
            with suppress_console_output(QUIET):
                pv.Render(view)
                view.ResetCamera()
        except Exception:
            pass

    for view in (view1, view2):
        try:
            with suppress_console_output(QUIET):
                for source in pv.GetSources().values():
                    if source is None:
                        continue
                    try:
                        display = pv.GetDisplayProperties(source, view)
                        if display and display.Visibility:
                            try:
                                display.RescaleTransferFunctionToDataRange(False, True)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    out_path = OUTPUT_PVSM or os.path.join(case_dir, os.path.basename(case_dir) + ".pvsm")
    try:
        with suppress_console_output(QUIET):
            pv.SaveState(out_path)
        if PRINT_FINAL_STATUS:
            print(f"  [pvsm] saved → {out_path}")
    except Exception as e:
        log(f"Failed to save .pvsm: {e}")


if __name__ == "__main__":
    main()
