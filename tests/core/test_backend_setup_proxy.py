from __future__ import annotations

from nvitk.core import available_backends, register_module_for_backend_updates, set_global_backend, setup, setup_backend_proxy


def _has_cupy_backend() -> bool:
    return "cupy" in available_backends()


def test_tutorial_one_call_setup_injects_np_scipy_ndi():
    """Tutorial: recommended usage -> setup(globals())."""
    module_name = "tests.fake.module.setup"
    fake_globals = {"__name__": module_name}

    setup(fake_globals)
    assert "np" in fake_globals
    assert "scipy" in fake_globals
    assert "ndi" in fake_globals
    assert fake_globals["np"].__name__ == "numpy"


def test_tutorial_setup_globals_follow_backend_switches():
    """Tutorial: module globals are refreshed after backend changes."""
    module_name = "tests.fake.module.refresh"
    fake_globals = {"__name__": module_name}
    setup(fake_globals)

    set_global_backend("numpy")
    assert fake_globals["np"].__name__ == "numpy"

    if _has_cupy_backend():
        set_global_backend("cupy")
        assert fake_globals["np"].__name__.startswith("cupy")
        assert "cupyx" in fake_globals["scipy"].__name__
    else:
        set_global_backend("cupy", allow_fallback=True)
        assert fake_globals["np"].__name__ == "numpy"


def test_tutorial_legacy_two_step_setup_still_works():
    """Tutorial: backward-compatible setup for older modules."""
    module_name = "tests.fake.module.legacy"
    fake_globals = {"__name__": module_name}

    # Historical pattern used in older code:
    setup_backend_proxy(fake_globals)
    register_module_for_backend_updates(module_name, module_globals=fake_globals)

    assert "np" in fake_globals and "scipy" in fake_globals and "ndi" in fake_globals
