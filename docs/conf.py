"""Sphinx configuration for MLTK.

Run ``sphinx-build -b html docs docs/_build/html`` from the repo root, or
``make html`` from the docs directory."""
from __future__ import annotations

import os
import sys
from datetime import datetime

# MLTK uses absolute ``import mltk`` internally — make the package
# discoverable when sphinx-build runs from the docs directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mltk  # noqa: E402

# -- Project information -----------------------------------------------------
project = "MLTK"
author = "MLTK contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = mltk.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"MLTK {release}"
html_theme_options = {
    "sidebar_hide_name": False,
}
