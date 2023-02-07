# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

import seismolab

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# The short X.Y version.
version = '.'.join(seismolab.__version__.split('.')[:-1])
# The full version, including alpha/beta/rc tags
release = seismolab.__version__

project = 'seismolab'
copyright = '2022, Attila Bodi'
author = 'Attila Bodi'
#release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'numpydoc',
    "nbsphinx",
    "sphinx.ext.autosectionlabel",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
