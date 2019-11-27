import inspect
import os
import sys

from sphinx.ext import apidoc

# Generate module references
__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
)
output_dir = os.path.abspath(os.path.join(__location__, "..", "docs", "_rst"))
module_dir = os.path.abspath(os.path.join(__location__, "..", "rle_array"))
apidoc_parameters = ["-f", "-e", "-o", output_dir, module_dir]
apidoc.main(apidoc_parameters)

sys.path.append(os.path.abspath(os.path.join(__location__, "sphinxext")))

add_module_names = False
author = "JDA Software, Inc"
copyright = "2019, JDA Software, Inc"
project = "rle-array"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "ignore_missing_refs",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]
html_static_path = ["_static"]
html_theme = "alabaster"
nitpicky = True
templates_path = ["_templates"]
