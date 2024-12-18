# THIS FIXES AUTODOC
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


# https://www.sphinx-doc.org/en/master/usage/configuration.html
project = 'torchzero'
author = 'me'
copyright = '%Y'
version = '0.1'
release = '0.1.0'

# https://sphinx-intro-tutorial.readthedocs.io/en/latest/sphinx_extensions.html
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

autosummary_generate = True