"""Sphinx configuration for iUNets."""

project = "iUNets"
author = "Christian Etmann"
copyright = "2020–2026, Christian Etmann"
release = "0.2.0"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "furo"
autodoc_typehints = "description"
autoclass_content = "class"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
