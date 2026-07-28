"""SGE worker helpers for ``--backend`` / ``NVITK_BACKEND``."""

from __future__ import annotations

import re
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


_VMEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMG]?)B?$", re.IGNORECASE)


def _vmem_to_mib(value: str) -> float:
    raw = str(value).strip()
    m = _VMEM_RE.match(raw)
    if not m:
        return 0.0
    amount = float(m.group(1))
    unit = (m.group(2) or "M").upper()
    scale = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}.get(unit, 1.0)
    return amount * scale


def sge_qvtpy_array_resources(
    backend: str,
    *,
    include_eicab: bool = False,
    eicab_device: str = "cpu",
    eicab_pe_smp: int | None = None,
) -> tuple[SgeResources, bool]:
    """Padded resources for a per-subject qvtpy array job.

    Uses qvtpy project/account/queue. When *include_eicab* is true, bumps
    ``h_vmem`` / ``ngpu`` / ``pe_smp`` / ``use_nv`` so the shared request covers
    the eICAB task.

    Returns ``(resources, use_nv)``.
    """
    from nvitk.pipes.qvtpy import config as cfg
    from nvitk.segmentation.eicab import config as eicab_cfg

    base = sge_qvtpy_stage_resources(backend)
    use_nv = sge_stage_use_nv(backend)
    if not include_eicab:
        return base, use_nv

    h_vmem = base.h_vmem
    if _vmem_to_mib(eicab_cfg.SGE_H_VMEM) > _vmem_to_mib(h_vmem):
        h_vmem = eicab_cfg.SGE_H_VMEM

    ngpu = max(int(base.ngpu), int(eicab_cfg.SGE_NGPU or 0))
    pe_smp = eicab_pe_smp if eicab_pe_smp is not None else eicab_cfg.SGE_PE_SMP
    if pe_smp is not None:
        pe_smp = int(pe_smp)

    dev = str(eicab_device).strip().lower()
    if dev in {"gpu", "cuda"}:
        use_nv = True
        if ngpu <= 0:
            ngpu = 1

    return (
        SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=ngpu,
            h_vmem=h_vmem,
            queue=cfg.SGE_QUEUE,
            pe_smp=pe_smp,
        ),
        use_nv,
    )


__all__ = [
    "sge_backend_cli_args",
    "sge_backend_is_gpu",
    "sge_qvtpy_array_resources",
    "sge_qvtpy_stage_resources",
    "sge_stage_extra_env",
    "sge_stage_ngpu",
    "sge_stage_use_nv",
]
