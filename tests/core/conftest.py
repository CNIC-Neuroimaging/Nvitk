from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _reset_backend_state():
    """
    Keep tests independent and easy to read:
    each test starts and ends in numpy mode.
    """
    from nvitk.core import set_global_backend

    set_global_backend("numpy", allow_fallback=True)
    try:
        yield
    finally:
        set_global_backend("numpy", allow_fallback=True)
