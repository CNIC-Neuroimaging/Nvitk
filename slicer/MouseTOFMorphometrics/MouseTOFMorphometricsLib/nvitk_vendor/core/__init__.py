"""NumPy-only stand-ins for ``nvitk.core`` (hand-written; not synced).

nvitk's core layer exists to swap NumPy for CuPy at runtime and to provide the
project's singleton logger. Neither is wanted inside Slicer, so the vendored
algorithm modules are pointed at these much smaller equivalents. The public
names match nvitk's exactly, which is what lets ``vendor_sync.py`` get away with
a plain ``nvitk`` → ``nvitk_vendor`` rename.
"""

from __future__ import annotations

__all__: list[str] = []
