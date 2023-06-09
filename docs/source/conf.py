# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

project = "hypll"
copyright = '2023, ""'
author = '""'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_tabs.tabs",
]

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc options:
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_default_flags = [
    "members",
    "undoc-members",
    "special-members",
    "show-inheritance",
]

# Napoleon options:
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None

# Sphinx gallery config:
sphinx_gallery_conf = {
    "examples_dirs": "../../tutorials",
    "gallery_dirs": "tutorials/",
    "filename_pattern": "",
    # TODO(Philipp, 06/23): Figure out how we can build and host tutorials on RTD.
    "plot_gallery": "False",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
