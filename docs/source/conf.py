# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys, os
#sys.path.insert(0, os.path.abspath('.../src'))

project = 'torchzero'
copyright = '2025, Ivan Nikishev'
author = 'Ivan Nikishev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# https://sphinx-intro-tutorial.readthedocs.io/en/latest/sphinx_extensions.html
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    "myst_nb",

    # 'sphinx_rtd_theme',
]
autosummary_generate = True
autoapi_dirs = ['../../torchzero']
autoapi_type = "python"
# autoapi_ignore = ["*/tensorlist.py"]

# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_options
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# OTHER STUFF I FOUND ON THE INTERNET AND PUT THERE HOPING IT DOES SOMETHING USEFUL
source_suffix = ['.rst', '.md']
master_doc = 'index'