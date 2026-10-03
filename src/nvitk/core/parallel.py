"""
CPU worker budget and a shared thread pool for host-side array work.

Description
-----------
One process-wide answer to "how many CPU threads may nvitk use", plus the pool
that spends them. The GUI exposes it as a preference, the CLI through
``NVITK_WORKERS``, and library code asks :func:`get_worker_count` instead of
guessing.

Threads, not processes: the heavy host kernels nvitk leans on — ``scipy.ndimage``
interpolation, NumPy ufuncs on large arrays, ``scipy.fft`` — release the GIL, so
threads scale on them (measured ~7.5x with 8 threads on ``map_coordinates`` and a
slab-split ``affine_transform`` of a 300x512x512 volume) without pickling
multi-gigabyte volumes into child processes.

Worker specification
--------------------
:func:`resolve_workers` accepts every way a person might say it:

- ``None`` / ``"auto"`` → :data:`DEFAULT_WORKER_FRACTION` of the usable cores;
- an ``int`` ``n > 0`` → exactly ``n`` (clamped to the usable cores);
- ``0`` / ``"all"`` → every usable core;
- a negative ``int`` ``-k`` → all cores but ``k`` (leave some for the desktop);
- a non-integer ``float`` in ``(0, 1)`` or a ``"75%"`` string → that fraction
  of the cores (an integer-valued float such as ``4.0`` is a count).

Nesting
-------
A task running *on* the shared pool that itself calls :func:`parallel_map` runs
its items serially: waiting on the same bounded pool from inside it is how a
saturated pool deadlocks. The outer level already owns the parallelism.

Backends
--------
Workers inherit the submitting thread's backend through
:func:`~nvitk.core.backend.set_thread_backend` (a context variable only). They
must not call :class:`~nvitk.core.backend.using`, which refreshes every module's
``np``/``ndi`` globals process-wide; host kernels inside workers should take their
modules from :func:`~nvitk.core.backend.get_backend_modules` instead.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

from nvitk.core.backend import get_current_backend, set_thread_backend
from nvitk.core.logger import Logger

log = Logger()

T = TypeVar("T")
R = TypeVar("R")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Share of the usable cores taken when nothing more specific was asked for.
#: Three quarters keeps a desktop session responsive while a volume is resampled.
DEFAULT_WORKER_FRACTION = 0.75

#: Environment override, read once when the budget is first needed.
WORKERS_ENV_VAR = "NVITK_WORKERS"

#: Below this many elements a chunked kernel is not worth the hand-off. One
#: oblique 512x416 view slice (~213k samples) is above it: measured, a slab
#: hand-off costs tens of microseconds against milliseconds of interpolation.
MIN_PARALLEL_ELEMENTS = 1 << 16


def physical_memory_bytes() -> int:
    """Installed RAM in bytes (0 when the platform will not say)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return 0

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

_lock = threading.RLock()
_worker_count: int | None = None
_pool: ThreadPoolExecutor | None = None
_pool_size = 0
_local = threading.local()


# ──────────────────────────────────────────────────────────────────────────────
# Worker budget
# ──────────────────────────────────────────────────────────────────────────────


def cpu_count() -> int:
    """Cores this process may actually run on (affinity-aware, never below 1)."""
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(int(os.cpu_count() or 1), 1)


def resolve_workers(spec: int | float | str | None = None) -> int:
    """
    Turn a worker specification into a thread count in ``[1, cpu_count()]``.

    Parameters
    ----------
    spec
        See the module docstring: ``None``/``"auto"``, ``n``, ``0``/``"all"``,
        ``-k``, a fraction, or ``"NN%"``.

    Raises
    ------
    ValueError
        For a string that is none of the accepted spellings.
    """
    cores = cpu_count()
    if spec is None:
        return max(1, int(round(cores * DEFAULT_WORKER_FRACTION)))
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text in ("", "auto", "default"):
            return resolve_workers(None)
        if text in ("all", "max"):
            return cores
        if text.endswith("%"):
            try:
                frac = float(text[:-1]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Bad worker percentage {spec!r}.") from exc
            return max(1, min(cores, int(round(cores * frac))))
        try:
            spec = float(text) if "." in text else int(text)
        except ValueError as exc:
            raise ValueError(
                f"Bad worker spec {spec!r}; use an integer, 'NN%', 'auto' or 'all'."
            ) from exc
    if isinstance(spec, bool):  # bool is an int; refuse the accident explicitly
        raise ValueError("Worker spec must be a number or string, not a bool.")
    if isinstance(spec, float) and not float(spec).is_integer():
        if 0.0 < spec <= 1.0:
            return max(1, min(cores, int(round(cores * spec))))
        raise ValueError(f"Fractional worker spec must be in (0, 1]; got {spec}.")
    n = int(spec)
    if n == 0:
        return cores
    if n < 0:
        return max(1, cores + n)
    return max(1, min(cores, n))


def get_worker_count() -> int:
    """The process-wide CPU thread budget (``$NVITK_WORKERS`` or the default)."""
    global _worker_count
    with _lock:
        if _worker_count is None:
            env = os.environ.get(WORKERS_ENV_VAR, "").strip()
            try:
                _worker_count = resolve_workers(env or None)
            except ValueError:
                log.warning("Ignoring unreadable %s=%r; using the default.", WORKERS_ENV_VAR, env)
                _worker_count = resolve_workers(None)
        return int(_worker_count)


def set_worker_count(spec: int | float | str | None, *, configure_libraries: bool = True) -> int:
    """
    Set the process-wide thread budget and return the resolved count.

    The shared pool is rebuilt on the next use at the new size; work already
    submitted to the old pool finishes there. With *configure_libraries*, the
    thread pools of libraries that keep their own (BLAS/OpenMP, numba, torch,
    SimpleITK) are pointed at the same budget.
    """
    global _worker_count, _pool, _pool_size
    n = resolve_workers(spec)
    with _lock:
        _worker_count = n
        old = _pool if _pool is not None and _pool_size != n else None
        if old is not None:
            _pool = None
            _pool_size = 0
    if old is not None:
        old.shutdown(wait=False, cancel_futures=False)
    if configure_libraries:
        configure_library_threads(n)
    return n


def configure_library_threads(n: int | None = None) -> dict[str, bool]:
    """
    Point third-party thread pools at *n* threads (default: the budget).

    Each library is optional; the result says which ones accepted the setting.
    Environment variables are only *defaulted* for libraries that read them at
    start-up (ITK), so an explicit user setting still wins.
    """
    count = int(n if n is not None else get_worker_count())
    applied: dict[str, bool] = {}

    # ---- BLAS / OpenMP inside NumPy and SciPy ---------------------------------
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=count)
        applied["threadpoolctl"] = True
    except Exception:  # noqa: BLE001 — optional dependency
        applied["threadpoolctl"] = False

    # ---- numba (napari's shape triangulation, some nvitk kernels) ------------
    try:
        import numba

        numba.set_num_threads(max(1, min(count, int(numba.config.NUMBA_NUM_THREADS))))
        applied["numba"] = True
    except Exception:  # noqa: BLE001
        applied["numba"] = False

    # ---- torch intra-op pool (CPU inference) ---------------------------------
    try:
        import sys

        if "torch" in sys.modules:  # never import torch just to configure it
            sys.modules["torch"].set_num_threads(count)
            applied["torch"] = True
        else:
            applied["torch"] = False
    except Exception:  # noqa: BLE001
        applied["torch"] = False

    # ---- SimpleITK / ITK ------------------------------------------------------
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(count))
    try:
        import sys

        if "SimpleITK" in sys.modules:
            sys.modules["SimpleITK"].ProcessObject_SetGlobalDefaultNumberOfThreads(count)
            applied["SimpleITK"] = True
        else:
            applied["SimpleITK"] = False
    except Exception:  # noqa: BLE001
        applied["SimpleITK"] = False
    return applied


# ──────────────────────────────────────────────────────────────────────────────
# Shared pool
# ──────────────────────────────────────────────────────────────────────────────


def shared_thread_pool() -> ThreadPoolExecutor:
    """The process-wide pool, sized to :func:`get_worker_count` (created lazily)."""
    global _pool, _pool_size
    n = get_worker_count()
    with _lock:
        if _pool is None or _pool_size != n:
            _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="nvitk-worker")
            _pool_size = n
        return _pool


def in_worker_thread() -> bool:
    """True when the caller is itself running as a task on the shared pool."""
    return bool(getattr(_local, "in_worker", False))


def _task(func: Callable[[T], R], backend: str) -> Callable[[T], R]:
    """Wrap *func* so it runs with the submitter's backend and the nesting flag set."""

    def _run(item: T) -> R:
        previous = getattr(_local, "in_worker", False)
        _local.in_worker = True
        try:
            set_thread_backend(backend, allow_fallback=True)
            return func(item)
        finally:
            _local.in_worker = previous

    return _run


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int | None = None,
) -> list[R]:
    """
    ``[func(item) for item in items]`` on the shared pool, order preserved.

    Runs serially when there is a single item, a budget of one thread, or the
    call comes from inside a pool task (see *Nesting* in the module docstring).
    Exceptions propagate from the first failing item, as in a plain loop.
    """
    seq: Sequence[T] = list(items)
    budget = get_worker_count() if workers is None else max(1, int(workers))
    if len(seq) <= 1 or budget <= 1 or in_worker_thread():
        return [func(item) for item in seq]
    runner = _task(func, get_current_backend())
    pool = shared_thread_pool()
    if budget >= len(seq):
        return list(pool.map(runner, seq))
    # A narrower budget than the pool: bound the concurrency by batching.
    out: list[R] = []
    for start in range(0, len(seq), budget):
        out.extend(pool.map(runner, seq[start:start + budget]))
    return out


def chunk_bounds(length: int, parts: int, *, min_size: int = 1) -> list[tuple[int, int]]:
    """
    Split ``range(length)`` into at most *parts* contiguous ``(start, stop)`` runs.

    Runs differ in size by at most one, and none is shorter than *min_size*
    unless *length* itself is.
    """
    length = int(length)
    if length <= 0:
        return []
    parts = max(1, min(int(parts), length // max(int(min_size), 1) or 1))
    base, extra = divmod(length, parts)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def worker_summary() -> dict[str, Any]:
    """What the budget currently is, for a status line or a log entry."""
    return {
        "cpu_count": cpu_count(),
        "workers": get_worker_count(),
        "default_fraction": DEFAULT_WORKER_FRACTION,
        "env": os.environ.get(WORKERS_ENV_VAR, ""),
    }


__all__ = [
    "DEFAULT_WORKER_FRACTION",
    "MIN_PARALLEL_ELEMENTS",
    "WORKERS_ENV_VAR",
    "chunk_bounds",
    "configure_library_threads",
    "cpu_count",
    "get_worker_count",
    "in_worker_thread",
    "parallel_map",
    "physical_memory_bytes",
    "resolve_workers",
    "set_worker_count",
    "shared_thread_pool",
    "worker_summary",
]
