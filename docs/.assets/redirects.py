"""Publish compatibility URLs outside navigation and the search index."""

import html
import json
import posixpath
from pathlib import Path, PurePosixPath


def page_url(source):
    """Return the directory-style URL for a docs source path."""
    path = PurePosixPath(source).relative_to("docs")
    if path.name == "index.md":
        return str(path.parent) + "/"
    return str(path.with_suffix("")) + "/"


def on_post_build(config):
    """Write fragment-aware redirects after normal pages/search are built."""
    manifest = Path(__file__).with_name("relocations.json")
    records = json.loads(manifest.read_text())
    site = Path(config.site_dir)
    for old, new in records["moves"].items():
        if old == new or not old.endswith((".md", ".ipynb")):
            continue
        if not new.startswith("docs/"):
            continue
        old_url, new_url = page_url(old), page_url(new)
        # The fast preview does not render notebooks.
        if not (site / new_url / "index.html").is_file():
            if (
                new.endswith(".ipynb")
                and "mkdocs-jupyter" not in config.plugins
            ):
                continue
            raise ValueError(f"Missing redirect destination: {new}")
        destination = posixpath.relpath(new_url, old_url) + "/"
        anchors = {}
        for source, target in records["fragments"].items():
            if source.startswith(old + "#"):
                target_page, fragment = target.split("#", 1)
                anchors[source.split("#", 1)[1]] = (
                    posixpath.relpath(page_url(target_page), old_url)
                    + "/#"
                    + fragment
                )
        out = site / old_url / "index.html"
        if out.exists():
            raise ValueError(f"Redirect would replace a published page: {old}")
        out.parent.mkdir(parents=True, exist_ok=True)
        default = json.dumps(destination)
        mapping = json.dumps(anchors, ensure_ascii=True)
        out.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="robots" content="noindex">'
            "<title>Documentation moved</title>"
            f'<link rel="canonical" href="{html.escape(destination)}">'
            "<script>"
            f"const destinations = {mapping};"
            "const fragment = decodeURIComponent(location.hash.slice(1));"
            f"location.replace(destinations[fragment] || {default}"
            " + location.hash);"
            "</script></head><body>"
            f'<p>This page moved. <a href="{html.escape(destination)}">'
            "Open the current guide</a>.</p></body></html>"
        )
