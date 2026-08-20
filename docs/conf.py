import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'EPRAYA'
copyright = '2026, Juan S. Castro, Miguel E. Gámez, Ovidio Almanza'
author = 'Juan S. Castro, Miguel E. Gámez, Ovidio Almanza'
release = 'V. 0.2.0'


extensions = ['sphinx.ext.autodoc',  
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'matplotlib.sphinxext.plot_directive']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports=["numpy","scipy","matplotlib","ipywidgets","IPython","tkinter","pandas","jax","chex","optax","numba","joblib","threadpoolctl"]
autodoc_typehints = "none"
add_module_names = False
autodoc_preserve_defaults = True

html_theme = "pydata_sphinx_theme"
html_title = 'EPRAYA-User guide'

html_static_path = ['_static']
html_logo = '_static/logo_.png'
html_favicon = '_static/logo2.ico'
toc_object_entries = False
napoleon_use_rtype = False
html_theme_options = {
    "logo": {
        "text": "EPRAYA","image_light": "logo_.png",
    }
}
