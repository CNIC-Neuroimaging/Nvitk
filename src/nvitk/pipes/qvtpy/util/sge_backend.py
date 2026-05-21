"""SGE worker helpers for ``--backend`` / ``NVITK_BACKEND``."""

from __future__ import annotations

import shlex

from nvitk.core.click_backend import sge_backend_env


def sge_backend_cli_args(backend: str = "gpu") -> list[str]:
    return ["--backend", shlex.quote(str(backend).strip().lower())]


def sge_stage_extra_env(src_bind: str, backend: str = "gpu") -> dict[str, str]:
    return sge_backend_env(src_bind, backend)


__all__ = ["sge_backend_cli_args", "sge_stage_extra_env"]
