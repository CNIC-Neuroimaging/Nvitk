from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

BackendPreference = Literal["auto", "numpy", "cupy", "cupy_required"]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_backend_preference(raw: str | None) -> BackendPreference:
    value = (raw or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "cpu": "numpy",
        "numpy": "numpy",
        "gpu": "cupy",
        "cupy": "cupy",
        "cupy_required": "cupy_required",
        "gpu_required": "cupy_required",
    }
    normalized = aliases.get(value)
    if normalized is None:
        return "auto"
    return normalized 


@dataclass(frozen=True)
class CoreConfig:
    backend_preference: BackendPreference = "auto"
    cuda_device: int | None = None
    warn_on_fallback: bool = True


def load_core_config() -> CoreConfig:
    backend = _normalize_backend_preference(os.getenv("NVITK_BACKEND", "auto"))
    raw_device = os.getenv("NVITK_CUDA_DEVICE", None)
    device: int | None = None
    if raw_device is not None and raw_device.strip() != "":
        try:
            device = int(raw_device)
        except ValueError:
            device = None

    return CoreConfig(
        backend_preference=backend,
        cuda_device=device,
        warn_on_fallback=_env_bool("NVITK_WARN_ON_FALLBACK", True),
    )