import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *

#: Modules whose import failed, as ``{dotted_name: exception}``. Populated by
#: :func:`recursive_find_python_class` so callers can explain *why* a class was not found
#: instead of reporting it as absent. Also used to report each breakage once per process
#: rather than on every lookup.
SKIPPED_MODULES: dict = {}


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    """Find *class_name* in the package rooted at *folder*.

    nvitk change: modules that cannot be imported are **skipped with a warning** instead of
    aborting the whole scan.

    Upstream imported every module unguarded, so a single optional dependency taking a module
    down made every trainer *sorting after it* invisible — reported as "trainer not found",
    which sends you looking in the wrong place entirely. Concretely: ``simCLR`` pulls in
    ``pl_bolts`` -> ``pytorch_lightning`` -> ``lightning_fabric``, which calls
    ``pkg_resources.declare_namespace``; ``pkg_resources`` was removed in setuptools 81, so that
    import dies and hides ``swinunetr_pretrain``, ``volume_contrastive`` and ``volume_fusion``.

    nnU-Net's own copy of this helper carries the same guard, for the same reason.
    """
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        if not ispkg:
            try:
                m = importlib.import_module(current_module + "." + modname)
            except Exception as exc:
                # Only the requested class matters; an unrelated broken module must not hide
                # the rest of the package.
                dotted = f"{current_module}.{modname}"
                if dotted not in SKIPPED_MODULES:
                    SKIPPED_MODULES[dotted] = exc
                    print(f"[nnssl] skipping {dotted}: {type(exc).__name__}: {exc}")
                continue
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    join(folder, modname), class_name, current_module=next_current_module
                )
            if tr is not None:
                break
    return tr
