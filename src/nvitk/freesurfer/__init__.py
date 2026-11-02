"""FreeSurfer's topology tools for any mask (see :mod:`nvitk.freesurfer.topology`).

:mod:`~nvitk.freesurfer.env` finds the installation and its license, and builds the environment
its programs run in; :mod:`~nvitk.freesurfer.run` runs them (:class:`FsCommand`). The CLI is
``nvitk-freesurfer``, and the GUI's *FreeSurfer* tools appear when an installation is found.
"""

from .env import (
    FreeSurferLicenseError,
    FreeSurferNotFound,
    available,
    freesurfer_home,
    freesurfer_version,
    fs_env,
    fs_license,
    which,
)
from .run import FsCommand, chain

__all__ = [
    "FreeSurferLicenseError",
    "FreeSurferNotFound",
    "FsCommand",
    "available",
    "chain",
    "freesurfer_home",
    "freesurfer_version",
    "fs_env",
    "fs_license",
    "which",
]
