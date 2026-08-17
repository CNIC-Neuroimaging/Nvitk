"""Third-party dependency checks for the Mouse TOF Morphometrics Slicer module.

Slicer already ships ``numpy``, ``scipy``, ``vtk`` and ``matplotlib``. The
vendored morphometrics code additionally needs the four packages below. nvitk
itself is **not** a dependency — the pipeline is vendored under ``nvitk_vendor``
precisely because nvitk's install requirements (antspyx, TotalSegmentator,
nnU-Net, a pinned SimpleITK) cannot go into Slicer's Python.
"""

from __future__ import annotations

import importlib

#: ``import name`` → ``pip name`` for everything the vendored pipeline needs.
REQUIRED: dict[str, str] = {
    "pandas": "pandas",
    "nibabel": "nibabel",
    "skimage": "scikit-image",
    "openpyxl": "openpyxl",
}

#: Bundled with Slicer; listed so the UI can report them rather than install them.
BUNDLED: tuple[str, ...] = ("numpy", "scipy", "vtk", "matplotlib")

PIP_INSTALL_ARGS: str = " ".join(REQUIRED.values())


def missing() -> list[str]:
    """pip names of the required packages that are not importable, in install order."""
    return [pip_name for module, pip_name in REQUIRED.items() if not _importable(module)]


def missing_bundled() -> list[str]:
    """Packages that should have shipped with Slicer but are not importable."""
    return [name for name in BUNDLED if not _importable(name)]


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def status_text() -> str:
    """One-line summary for the module's status label."""
    absent = missing()
    broken = missing_bundled()
    if broken:
        return f"Slicer is missing packages it normally bundles: {', '.join(broken)}."
    if absent:
        return "Missing Python packages: " + ", ".join(absent) + " — click Install dependencies."
    return "All dependencies present."


def ensure() -> None:
    """Raise a clear, copy-pasteable error when a required package is absent."""
    absent = missing()
    if absent:
        raise RuntimeError(
            "Mouse TOF Morphometrics is missing: "
            + ", ".join(absent)
            + ".\nClick 'Install dependencies', or in the Slicer Python console run:\n"
            "  import slicer\n"
            f'  slicer.util.pip_install("{" ".join(absent)}")'
        )


def install(packages: list[str] | None = None) -> list[str]:
    """pip-install the missing (or given) packages into Slicer's Python.

    Returns the list that was installed. Raises outside Slicer.
    """
    targets = list(packages) if packages else (missing() or list(REQUIRED.values()))
    if not targets:
        return []
    import slicer  # noqa: PLC0415 - only meaningful inside Slicer

    slicer.util.pip_install(" ".join(targets))
    importlib.invalidate_caches()
    return targets


__all__ = [
    "BUNDLED",
    "PIP_INSTALL_ARGS",
    "REQUIRED",
    "ensure",
    "install",
    "missing",
    "missing_bundled",
    "status_text",
]
