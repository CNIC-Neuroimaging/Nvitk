# Core & Backend

`nvitk.core` is the foundation every other module is built on: a NumPy/CuPy dual backend
that lets the rest of the library be written once and run on CPU or GPU, plus the shared
logging, configuration, and error-hierarchy primitives.

## The dual backend

```{code-block} python
from nvitk.core import setup, using, get_current_backend

setup(globals())  # injects backend-aware np, ndi, scipy into this module's globals
print(get_current_backend())

with using("cupy"):
    x = np.asarray([1, 2, 3])  # this block runs on GPU
```

| Function | What it does |
|---|---|
| {func}`~nvitk.core.setup` | One-call setup: injects backend-aware `np`/`ndi`/`scipy` proxy globals into the calling module and registers it so later backend switches refresh those globals automatically. |
| {class}`~nvitk.core.using` | Context-manager **class** for scoped switching — `with using("cupy"): ...`. |
| `using_backend` | Lower-level, generator-based equivalent of `using`. |
| {func}`~nvitk.core.get_current_backend` | The active backend for the current context — thread/async-aware via `contextvars`. |
| `set_global_backend` / `set_default_backend` | Set the process-wide default backend. |
| `available_backends`, `is_cupy_installed`, `is_gpu_available` | Introspect what's actually usable in the current environment. |
| `to_numpy` / `to_cupy`, `is_cupy_array` / `is_numpy_array` | Convert and check array backend without caring which one is active. |

Every registered module's `np`/`scipy`/`ndi` globals are kept in sync after a backend switch
by `BackendProxy` (`nvitk.core.proxy`). The active/default backend can also be set without
touching code, via environment variables read by `nvitk.core.config`:

| Variable | Effect |
|---|---|
| `NVITK_BACKEND` | `auto` (default), `numpy`, or `cupy` — process-wide default. |
| `NVITK_CUDA_DEVICE` | Which CUDA device to use when the backend is `cupy`. |
| `NVITK_WARN_ON_FALLBACK` | Warn (instead of silently succeeding) when a `cupy` request falls back to NumPy because no GPU is available. |

Every GUI and CLI `--backend cpu|gpu` flag across the toolkit (see {doc}`../gui/index` and
the {doc}`Main GUI's GPU toggle <../gui/index>`) is a thin wrapper over this same mechanism,
via `nvitk.core.click_backend`.

## Other `nvitk.core` primitives

| Module | Purpose |
|---|---|
| `nvitk.core.array` | Backend-agnostic array conversion/comparison helpers (`as_backend_array`, `ensure_same_backend`). |
| `nvitk.core.logger` | The shared `Logger` singleton — Rich progress bars, ANSI console output, and file handlers used throughout the CLIs and GUI. |
| `nvitk.core.exceptions` | The `NvitkError` hierarchy (`BackendError`, `BackendUnavailableError`, ...). |
| `nvitk.core.patterns` | A `Singleton` metaclass used by `Logger` and a few other process-wide objects. |

```{seealso}
Full generated reference: [`nvitk.core`](../autoapi/nvitk/core/index).
```
