"""
Shared Click plumbing for the CViT stage CLIs.

Every stage module is runnable on its own (``python -m nvitk.pipes.cvit.stageN …``) — that is
what an SGE job executes — and takes the same ``--<root>`` flags. :func:`root_options` adds them
and :func:`paths_from_options` turns them into a :class:`~nvitk.pipes.cvit.util.paths.CViTPaths`:
explicit flags win, missing ones fall back to ``sge.json``'s ``local_*`` keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import click

from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.paths import CViTPaths

_ROOT_HELP = {
    "data_root": "Labelled input data (read-only).",
    "nnunet_raw": "nnU-Net raw root (nnUNet_raw).",
    "nnunet_preprocessed": "nnU-Net preprocessed root (nnUNet_preprocessed).",
    "nnunet_results": "nnU-Net results root (nnUNet_results).",
    "nnssl_raw": "nnssl raw root.",
    "nnssl_preprocessed": "nnssl preprocessed root.",
    "nnssl_results": "nnssl results root.",
    "corpus_root": "Harmonised unlabelled corpus volumes.",
    "results_root": "Stage outputs (bundles, metrics, probe, predictions, exports).",
    "model_root": "External checkpoints.",
}


def root_options(func: Callable) -> Callable:
    """Add one ``--<root>`` option per :data:`~nvitk.pipes.cvit.util.paths.ROOT_KEYS` entry."""
    for key in reversed(pth.ROOT_KEYS):
        func = click.option(
            f"--{key.replace('_', '-')}", key, type=click.Path(path_type=Path), default=None,
            help=f"{_ROOT_HELP[key]} Default: sge.json pipelines.cvit_paths.local_{key}.",
        )(func)
    return func


def pop_roots(options: dict[str, Any]) -> dict[str, Path | None]:
    """Remove and return the root options from a Click kwargs dict."""
    return {key: options.pop(key, None) for key in pth.ROOT_KEYS}


def paths_from_options(roots: dict[str, Path | None], *, require_config: bool = False) -> CViTPaths:
    """Explicit roots, completed from the ``local_*`` config when available.

    With ``require_config=False`` a root that is neither given nor configured becomes an
    obviously-fake :data:`~nvitk.pipes.cvit.util.paths.UNAVAILABLE_ROOT` path, so a stage that
    only needs some roots (e.g. inference) runs without configuring the rest.
    """
    if require_config:
        return pth.layout_local(**roots)
    filled: dict[str, Path | None] = {}
    for key in pth.ROOT_KEYS:
        value = roots.get(key)
        if value is None:
            value = getattr(pth, f"LOCAL_DEFAULT_{key.upper()}")
        filled[key] = value
    return pth.layout_from_roots(**filled)


def parse_int_list(text: str | None) -> list[int]:
    """``"0,1,2"`` → ``[0, 1, 2]``; empty → ``[]``."""
    return [int(t) for t in str(text or "").replace(" ", "").split(",") if t != ""]


def parse_float_list(text: str | None) -> list[float]:
    return [float(t) for t in str(text or "").replace(" ", "").split(",") if t != ""]


def parse_str_list(text: str | None) -> list[str]:
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


__all__ = [
    "parse_float_list",
    "parse_int_list",
    "parse_str_list",
    "paths_from_options",
    "pop_roots",
    "root_options",
]
