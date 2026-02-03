# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import subprocess
import re
import os

def get_latest_tag_or_master():
    """Return latest version tag (vX.Y.Z) if exists, else 'master'."""
    try:
        # Fetch tags sorted by version (requires git >=2.0)
        tags = subprocess.check_output(
            ["git", "tag", "--list", "v*", "--sort=-v:refname"],
            text=True
        ).strip().splitlines()

        # Filter valid semantic version tags
        version_tags = [t for t in tags if re.match(r"^v\d+\.\d+(\.\d+)?$", t)]
        if version_tags:
            return version_tags[0]
    except Exception:
        pass
    return "master"



# -- Project information -----------------------------------------------------

project = "Shoebox"
copyright = "2025, NTNU"
author = "Emir Cem Gezer"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- sphinx-multiversion configuration ---------------------------------------

smv_tag_whitelist = r"^v\d+\.\d+(\.\d+)?$"
smv_branch_whitelist = r"^(main|master)$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+(\.\d+)?$"

smv_latest_version = get_latest_tag_or_master()
smv_rename_latest_version = "latest"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "display_version": False,
}

# Project branding
# Path to logo relative to this configuration directory
html_logo = "../assets/shoeboxpy.png"

# Add custom CSS for NTNU color palette overrides
html_css_files = [
    "css/custom.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Intersphinx mappings for cross-references
intersphinx_mapping = {
    # Second element must be None or a path/objects.inv - empty dict breaks Sphinx >=8
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

todo_include_todos = True

# ensure Sphinx copies that file to the build root
html_extra_path = ["_root"]