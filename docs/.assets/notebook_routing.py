"""Keep ordinary Markdown out of the notebook parser during publication.

The supported notebook sources are Python/Jupyter pairs. mkdocs-jupyter's
Markdown detection invokes Jupytext before applying include/ignore patterns;
that parses mkdocstrings directives as Pandoc div fences and emits warnings.
Route Markdown to MkDocs/mkdocstrings without changing its contents or hiding
warnings. Python and ipynb eligibility still belongs to the notebook plugin.
"""

from pathlib import Path


def on_config(config):
    """Restrict notebook detection before Jupytext can parse API Markdown."""
    plugin = config.plugins.get("mkdocs-jupyter")
    if plugin is None:
        return config
    original = plugin.should_include

    def should_include(file):
        if Path(file.abs_src_path).suffix == ".md":
            return False
        return original(file)

    plugin.should_include = should_include
    return config
