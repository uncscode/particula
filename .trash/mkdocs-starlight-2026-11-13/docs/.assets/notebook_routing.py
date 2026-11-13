"""Keep ordinary Markdown out of the notebook parser during publication.

The supported notebook sources are Python/Jupyter pairs. mkdocs-jupyter's
Markdown detection invokes Jupytext before applying include/ignore patterns;
that parses mkdocstrings directives as Pandoc div fences and emits warnings.
Route Markdown to MkDocs/mkdocstrings without changing its contents or hiding
warnings. Python and ipynb eligibility still belongs to the notebook plugin.
"""

import posixpath
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from mkdocs.utils import get_relative_url


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


def on_page_content(html, page, config, files):
    """Resolve notebook prose links using the same routes as Markdown pages."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def replace(match):
        url = urlsplit(match[2])
        if url.scheme or url.netloc or not url.path:
            return match[0]
        source = posixpath.normpath(
            posixpath.join(
                posixpath.dirname(page.file.src_uri), unquote(url.path)
            )
        )
        target = files.get_file_from_path(source)
        if target is None or target is page.file:
            return match[0]  # preserve downloadable notebook source links
        destination = get_relative_url(target.url, page.url)
        rewritten = urlunsplit(("", "", destination, url.query, url.fragment))
        return f"href={match[1]}{rewritten}{match[1]}"

    return re.sub(r"href=([\"'])(.*?)\1", replace, html)
