"""Import single nvitk modules without running the package ``__init__`` chain.

``import nvitk`` pulls in the database, statistics and GUI stacks. The checks in this folder only
need a few numeric modules, so the parent packages are replaced by empty namespace modules and
the target module is imported directly from ``src/``.
"""

import importlib
import os
import sys
import types

SRC = os.environ.get("NVITK_SRC", os.path.join(os.path.dirname(__file__), "..", "..", "src"))
SRC = os.path.abspath(SRC)
_REAL_PACKAGES = {"nvitk.core"}


def _stub(name):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [os.path.join(SRC, *name.split("."))]
    module.__package__ = name
    sys.modules[name] = module


def load(modname, stub=("nvitk",)):
    """Import *modname* with every parent package (and *stub*) left uninitialised."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    parts = modname.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in _REAL_PACKAGES:
            importlib.import_module(parent)
        else:
            _stub(parent)
    for name in stub:
        _stub(name)
    return importlib.import_module(modname)
