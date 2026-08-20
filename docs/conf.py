# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# -- Mock heavy/optional runtime dependencies ---------------------------------
#
# sphinx-click (unlike autoapi) must actually `import` each CLI module to
# introspect its click.Command tree — and importing nvitk.gui/nvitk.pipes.*
# transitively imports the entire nvitk package (nvitk/__init__.py eagerly
# imports db/types/core/io/stats). Installing the full runtime stack (napari,
# torch, cupy, vtk, SimpleITK, antspyx, rpy2, ...) just to render CLI --help
# text would make the docs build slow and GPU/R-toolchain-dependent, so those
# modules are mocked here instead — MagicMock happily absorbs any attribute
# access or call, which is all a click decorator needs at import time.
_MOCK_MODULES = [
    "napari", "magicgui", "magicgui.widgets", "superqt", "qtpy", "qtpy.QtCore",
    "qtpy.QtWidgets", "qtpy.QtGui", "PyQt6", "PyQt6.QtWebEngineWidgets",
    "vtk", "cv2", "SimpleITK", "pyvista", "trame", "trame_vtk", "kaleido",
    "antspyx", "ants", "antspynet", "totalsegmentator", "nnunetv2", "fireants",
    "nipype", "nipype.interfaces", "nipype.interfaces.fsl", "gudhi", "semopy",
    "xnat", "rpy2", "rpy2.robjects", "rpy2.robjects.packages", "pymer4",
    "great_tables", "formulae", "torch", "torchvision", "cupy", "cutensor",
    "nvidia_nccl", "pydicom", "pyradiomics", "radiomics", "dicom2nifti",
    "nd2", "pylibjpeg", "h5py",
]
for _mod_name in _MOCK_MODULES:
    sys.modules.setdefault(_mod_name, MagicMock())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nvitk"
copyright = "2026, CNIC / Ignacio Marcos Serrano"
author = "Ignacio Marcos Serrano"
# Kept in manual sync with [project].version in ../pyproject.toml (same as
# recipe/recipe.yaml's context.version — that field is static there too, not
# setuptools_scm-driven, despite setuptools_scm being a build-system requirement).
release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx_click",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
]

root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ----------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives, reads cleaner than ```{directive}
    "deflist",       # definition lists (used in the CLI option references)
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

# -- Napoleon (numpydoc-style docstrings) -------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- autoapi -------------------------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
#
# autoapi parses source via static AST analysis rather than importing it, so this
# build does NOT need the GPU/R/ANTs runtime stack installed (torch, cupy, rpy2,
# antspyx, ...) — only the `dev` doc-tooling group from pyproject.toml.

autoapi_dirs = ["../src/nvitk"]
autoapi_type = "python"
autoapi_root = "autoapi"
# False: the exhaustive autoapi tree is linked from docs/api/index.md instead of
# getting its own top-level navbar entry, so there's one coherent "API Reference"
# entry point rather than two. That one manual link is enough to make the whole
# autoapi subtree reachable (avoids "not in any toctree" warnings) since
# autoapi/index itself recursively toctrees every generated page.
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = [
    "*/values.py",       # empty (0-byte) placeholder file
    "*__pycache__*",
    "*/pipes/qvtplus/*",  # vendored third-party MATLAB reference implementation
    # Numeric-prefixed one-off validation scripts and a Singularity sitecustomize hook —
    # not part of the documented public API, and their names (leading digits) don't nest
    # cleanly into the package toctree anyway.
    "*/pipes/pesa_fat/common/validation/*",
    "*/segmentation/eicab/cpu_limit_site/sitecustomize.py",
]

# -- intersphinx ----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "napari": ("https://napari.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "nvitk documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "logo": {"text": "nvitk"},
    "github_url": "https://github.com/ignacio-ms/Nvitk",
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 4,
    "show_nav_level": 1,
    "collapse_navigation": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "default_mode": "auto",  # follows the visitor's OS light/dark preference
}

html_sidebars = {
    "index": [],  # no left sidebar on the homepage — full-width hero + card grid
}
