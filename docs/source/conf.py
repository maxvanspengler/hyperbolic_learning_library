# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file

project = 'hypdl'
copyright = '2023, ""'
author = '""'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

sphinx_gallery_conf = {
     'examples_dirs': '../../tutorials',
     'gallery_dirs': 'tutorials/',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
