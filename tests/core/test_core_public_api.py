from __future__ import annotations

import nvitk.core as core


def test_core_public_api_exports_expected_symbols():
    """Tutorial: these are the main entry points users should import."""
    expected = [
        "available_backends",
        "set_global_backend",
        "get_global_backend",
        "get_current_backend",
        "using",
        "using_backend",
        "setup",
        "setup_backend_proxy",
        "register_module_for_backend_updates",
        "to_numpy",
        "to_cupy",
        "as_backend_array",
        "ensure_same_backend",
    ]
    for symbol in expected:
        assert hasattr(core, symbol), f"missing public symbol: {symbol}"
