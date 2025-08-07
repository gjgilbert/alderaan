import os
import sys
sys.path.insert(0, os.path.abspath('./../alderaan/'))

import alderaan

# project info
project = 'alderaan'
copyright = '2025, Gregory J. Gilbert'
authors = 'Gregory J. Gilbert & Erik A. Petigura'
release = alderaan.__version__
version = alderaan.__version__

# general configuration
root_doc = 'index'
templates_path = ['_templates']
html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
