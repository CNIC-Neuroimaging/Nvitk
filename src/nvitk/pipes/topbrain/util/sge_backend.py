"""SGE worker helpers for ``--backend`` / ``NVITK_BACKEND`` in the ToPBrain pipeline.

Mirrors :mod:`nvitk.pipes.qvtpy.util.io.sge_backend`, with two differences that follow from
what this pipeline runs:

- stages are **cohort-scoped**, not per-subject. nnU-Net and nnssl own their own per-case
  parallelism, so one job per stage is the right granularity — splitting a training run across
  array tasks would fight the framework rather than help it;
- the deep-learning stages need the *torch* device as well as the nvitk array backend. Those
  are separate axes: ``--backend cpu|gpu`` selects NumPy vs CuPy for our own array work
  (harmonisation, metrics, post-processing), while ``--device`` selects where torch runs.
  :func:`torch_device_for_backend` derives a sensible default so callers rarely pass both.
"""

from __future__ import annotations

import shlex

from nvitk.cluster.sge import SgeResources
from nvitk.core.click_backend import sge_backend_env
from nvitk.core.logger import Logger

log = Logger()


#: Set by ``--sge-project`` for the lifetime of one CLI invocation. Module state rather than a
#: parameter threaded through eight stages' submit functions: it is process-scoped CLI
#: configuration, exactly like the lazily-read ``config`` values it overrides.
_PROJECT_OVERRIDE: str | None = None
_H_VMEM_OVERRIDE: str | None = None


def set_sge_project_override(project: str | None) -> None:
    """Override the configured SGE project for this process.

    The project decides both the queue the job lands in and how ``qsub`` must be asked for a
    GPU (``-l ngpu`` vs ``-l lgpu|sgpu|xsgpu``), so switching between a CPU and a GPU queue
    otherwise means editing ``sge.json`` between runs.
    """
    global _PROJECT_OVERRIDE
    _PROJECT_OVERRIDE = str(project).strip() if project and str(project).strip() else None


def set_sge_h_vmem_override(h_vmem: str | None) -> None:
    """Override the configured ``h_vmem`` for this process.

    On queues that enforce it as an address-space limit, ``h_vmem`` bounds a CUDA process well
    below the GPU's own memory: the failure looks like a GPU out-of-memory with the device
    reporting plenty free. Raising it is therefore an experiment worth running per submission,
    not a value to edit into sge.json between runs.
    """
    global _H_VMEM_OVERRIDE
    _H_VMEM_OVERRIDE = str(h_vmem).strip() if h_vmem and str(h_vmem).strip() else None


def sge_h_vmem() -> str | None:
    """The ``h_vmem`` to request: ``--sge-h-vmem`` if given, else ``sge.json``."""
    from nvitk.pipes.topbrain import config as cfg

    return _H_VMEM_OVERRIDE or cfg.SGE_H_VMEM


def sge_project() -> str | None:
    """The SGE project to submit under: ``--sge-project`` if given, else ``sge.json``."""
    from nvitk.pipes.topbrain import config as cfg

    return _PROJECT_OVERRIDE or cfg.SGE_PROJECT


def sge_backend_cli_args(backend: str = "gpu") -> list[str]:
    """Shell-quoted ``--backend <backend>`` argv fragment for a worker command."""
    return ["--backend", shlex.quote(str(backend).strip().lower())]


def sge_stage_extra_env(src_bind: str, backend: str = "gpu") -> dict[str, str]:
    """Extra environment for an SGE worker running with *backend*.

    Delegates to :func:`~nvitk.core.click_backend.sge_backend_env`.
    """
    return sge_backend_env(src_bind, backend)


def sge_backend_is_gpu(backend: str) -> bool:
    """True if *backend* (case-insensitive) is ``"gpu"``."""
    return str(backend).strip().lower() == "gpu"


def sge_stage_ngpu(backend: str, *, request_gpu: bool | None = None) -> int:
    """``qsub -l ngpu=…`` count; ``0`` omits the request (CPU job)."""
    from nvitk.pipes.topbrain import config as cfg

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
    return bool(request_gpu is True or sge_backend_is_gpu(backend))


def sge_topbrain_stage_resources(
    backend: str,
    *,
    request_gpu: bool | None = None,
    h_vmem: str | None = None,
    pe_smp: int | None = None,
) -> SgeResources:
    """Build :class:`~nvitk.cluster.sge.SgeResources` for one ToPBrain stage.

    Parameters
    ----------
    h_vmem
        Overrides the configured memory request. Preprocessing holds whole angiographic volumes
        in memory (up to 72 M voxels each), so those stages ask for more than the default.
    pe_smp
        Slots for the shared-memory parallel environment. nnU-Net's preprocessing and
        data-augmentation pools are sized from this via ``nnUNet_def_n_proc``.
    """
    from nvitk.pipes.topbrain import config as cfg

    return SgeResources(
        project=sge_project(),
        account=cfg.SGE_ACCOUNT,
        ngpu=sge_stage_ngpu(backend, request_gpu=request_gpu),
        h_vmem=h_vmem if h_vmem is not None else sge_h_vmem(),
        queue=cfg.SGE_QUEUE,
        pe_smp=int(pe_smp) if pe_smp is not None else None,
    )


def torch_device_for_backend(
    backend: str, *, device: str | None = None, remote: bool = False
) -> str:
    """Resolve the torch device for a stage.

    An explicit *device* always wins. Otherwise ``--backend gpu`` implies ``cuda`` **if a CUDA
    device is actually visible**, and falls back to ``cpu`` with a warning rather than dying
    inside torch — a workstation without a GPU should still be able to run the light stages.

    Parameters
    ----------
    remote
        The device is for a job that will run somewhere else. The local probe is then not just
        uninformative but actively wrong, so ``--backend gpu`` is taken at its word.
    """
    if device is not None and str(device).strip():
        return str(device).strip().lower()
    if not sge_backend_is_gpu(backend):
        return "cpu"
    if remote:
        # Building a command for a compute node: this host's hardware says nothing about that
        # one's. Probing here is how a GPU job was submitted with '-device cpu' because the
        # submitting workstation had a broken torch build -- the job then ran, slowly, on the
        # CPU of a node holding a GPU it never touched.
        return "cuda"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # torch missing or CUDA init failed
        log.warning("Could not query CUDA (%s); falling back to torch device 'cpu'.", exc)
        return "cpu"
    log.warning("--backend gpu requested but no CUDA device is visible; using torch 'cpu'.")
    return "cpu"


__all__ = [
    "set_sge_project_override",
    "sge_backend_cli_args",
    "sge_backend_is_gpu",
    "sge_project",
    "sge_stage_extra_env",
    "sge_stage_ngpu",
    "sge_stage_use_nv",
    "sge_topbrain_stage_resources",
    "torch_device_for_backend",
]
