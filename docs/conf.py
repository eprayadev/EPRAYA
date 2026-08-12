import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'EPRAYA'
copyright = '2026, Juan S. Castro, Ovidio Almanza, Miguel E. Gámez'
author = 'Juan S. Castro, Ovidio Almanza, Miguel E. Gámez'
release = 'V. 0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',  
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports=["numpy","scipy","matplotlib","ipywidgets","IPython","tkinter","pandas","jax","chex","optax","numba","joblib","threadpoolctl"]
autodoc_typehints = "none"
add_module_names = False
autodoc_preserve_defaults = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
toc_object_entries = False
