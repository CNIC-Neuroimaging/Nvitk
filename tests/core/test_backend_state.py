from __future__ import annotations

from nvitk.core import (
    available_backends,
    get_backend_modules,
    get_current_backend,
    get_global_backend,
    set_global_backend,
    using,
)


def _has_cupy_backend() -> bool:
    return "cupy" in available_backends()


def test_tutorial_list_available_backends():
    """Tutorial: inspect which backends are usable in this environment."""
    assert "numpy" in available_backends()


def test_tutorial_set_global_backend_and_read_modules():
    """Tutorial: set a global backend and use np/scipy/ndi modules from it."""
    set_global_backend("numpy")
    assert get_global_backend() == "numpy"
    assert get_current_backend() == "numpy"
    mods = get_backend_modules()
    assert mods.xp.__name__ == "numpy"
    assert "scipy" in mods.scipy.__name__
    assert "ndimage" in mods.ndi.__name__


def test_tutorial_request_gpu_with_fallback():
    """
    Tutorial: request cupy; if unavailable, allow_fallback keeps execution on numpy.
    """
    if _has_cupy_backend():
        set_global_backend("cupy")
        assert get_global_backend() == "cupy"
        assert get_backend_modules().xp.__name__.startswith("cupy")
    else:
        set_global_backend("cupy", allow_fallback=True)
        assert get_global_backend() == "numpy"


def test_tutorial_using_context_manager_is_temporary():
    """Tutorial: use `with using(...)` for temporary backend overrides."""
    set_global_backend("numpy")
    start = get_current_backend()
    with using("cupy", allow_fallback=True):
        if _has_cupy_backend():
            assert get_current_backend() == "cupy"
        else:
            assert get_current_backend() == "numpy"
    assert get_current_backend() == start
