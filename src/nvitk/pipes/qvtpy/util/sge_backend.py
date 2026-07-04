"""SGE worker helpers for ``--backend`` / ``NVITK_BACKEND``."""

from __future__ import annotations

import shlex

from nvitk.cluster.sge import SgeResources
from nvitk.core.click_backend import sge_backend_env


def sge_backend_cli_args(backend: str = "gpu") -> list[str]:
    return ["--backend", shlex.quote(str(backend).strip().lower())]


def sge_stage_extra_env(src_bind: str, backend: str = "gpu") -> dict[str, str]:
    return sge_backend_env(src_bind, backend)


def sge_backend_is_gpu(backend: str) -> bool:
    return str(backend).strip().lower() == "gpu"


def sge_stage_ngpu(backend: str, *, request_gpu: bool | None = None) -> int:
    """``qsub -l ngpu=…`` count; ``0`` omits the request (CPU job)."""
    from nvitk.pipes.qvtpy import config as cfg

    if request_gpu is False:
        return 0
    if request_gpu is True or sge_backend_is_gpu(backend):
        ngpu = int(cfg.SGE_NGPU)
        return ngpu if ngpu > 0 else 1
    return 0


def sge_stage_use_nv(backend: str, *, request_gpu: bool | None = None) -> bool:
    """Whether the outer ``singularity exec`` should pass ``--nv``."""
    if request_gpu is False:
        return False
    if request_gpu is True or sge_backend_is_gpu(backend):
        return True
    return False


def sge_qvtpy_stage_resources(backend: str, *, request_gpu: bool | None = None) -> SgeResources:
    """Build :class:`~nvitk.cluster.sge.SgeResources` from CLI ``--backend``."""
    from nvitk.pipes.qvtpy import config as cfg

    return SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=sge_stage_ngpu(backend, request_gpu=request_gpu),
        h_vmem=cfg.SGE_H_VMEM,
        queue=cfg.SGE_QUEUE,
    )


__all__ = [
    "sge_backend_cli_args",
    "sge_backend_is_gpu",
    "sge_qvtpy_stage_resources",
    "sge_stage_extra_env",
    "sge_stage_ngpu",
    "sge_stage_use_nv",
]
