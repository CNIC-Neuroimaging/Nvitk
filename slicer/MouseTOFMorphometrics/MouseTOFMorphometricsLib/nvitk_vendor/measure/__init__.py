"""Vendored ``nvitk.measure`` namespace.

Intentionally empty. Upstream's ``nvitk/measure/__init__.py`` re-exports the
whole measurement API (radiomics, SUV, resampling, …), which would pull
SimpleITK and the I/O stack into Slicer. Only ``morphometrics`` and ``morpho``
are vendored, and they are imported by name.
"""

from __future__ import annotations

__all__: list[str] = []
