"""Reject documentation regressions using small independent fixture trees."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "check_docs_structure.py"
SPEC = importlib.util.spec_from_file_location("check_docs_structure", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(root, path, text):
    """Write one fixture file with its parent directories."""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


@pytest.fixture
def tree(tmp_path):
    """Create five sections, an indexed child, and a paired notebook."""
    nav = []
    for title in MODULE.SECTIONS:
        if title == "API":
            nav.append({title: "https://example.org/API/"})
        else:
            path = f"{title.lower()}/index.md"
            write(tmp_path, "docs/" + path, f"# {title}\n")
            nav.append({title: path})
    write(tmp_path, "mkdocs.yml", yaml.safe_dump({"nav": nav}))
    write(tmp_path, "docs/index.md", "# Home\n")
    write(
        tmp_path,
        "docs/.assets/relocations.json",
        json.dumps(
            {
                "moves": {"docs/old.md": "docs/guide/child.md"},
                "fragments": {
                    "docs/old.md#before": "docs/guide/child.md#after"
                },
                "retired": {},
            }
        ),
    )
    write(tmp_path, "docs/guide/index.md", "# Guide\n[child](child.md#after)\n")
    write(tmp_path, "docs/guide/child.md", "# After\n")
    write(tmp_path, "docs/examples/index.md", "# Examples\n[run](run.ipynb)\n")
    write(
        tmp_path,
        "docs/examples/run.ipynb",
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Run"]},
                    {"cell_type": "code", "source": ["result = 1\n"]},
                ]
            }
        ),
    )
    write(tmp_path, "docs/examples/run.py", "result = 1\n")
    return tmp_path


def test_indexes_pairs_and_archives_are_accounted_for(tree):
    """Owning-index links classify pages; archives never enter discovery."""
    write(tree, ".trash/docs-v0.3.0/docs/obsolete.ipynb", "not JSON")
    assert MODULE.check(tree) == []


@pytest.mark.parametrize(
    "change,expected",
    [
        ("nav", "missing navigation destination"),
        ("orphan", "unreachable authored page"),
        ("owner", "unclassified page"),
        ("link", "missing local link"),
        ("anchor", "missing anchor"),
        ("stale", "stale moved link"),
        ("pair", "notebook/source code differs"),
        ("source", "unlinked standalone source"),
    ],
)
def test_structural_regressions_are_rejected(tree, change, expected):
    """Each declared gate fails on its corresponding independent defect."""
    changes = {
        "nav": ("docs/guide/index.md", None),
        "orphan": ("docs/guide/orphan.md", "# Orphan"),
        "owner": ("docs/Features/new.md", "# Wrong owner"),
        "link": ("docs/guide/child.md", "# After\n[bad](missing.md)"),
        "anchor": ("docs/guide/child.md", "# Renamed"),
        "stale": ("docs/guide/child.md", "# After\n[old](../old.md)"),
        "pair": ("docs/examples/run.py", "result = 2"),
        "source": ("docs/examples/hidden.py", "result = 1"),
    }
    path, text = changes[change]
    if text is None:
        (tree / path).unlink()
    else:
        write(tree, path, text)
    assert any(expected in error for error in MODULE.check(tree))


def test_notebook_markdown_and_source_urls_are_checked(tree):
    """Notebook links are checked; code strings and historical URLs are not."""
    path = tree / "docs/examples/run.ipynb"
    notebook = json.loads(path.read_text())
    notebook["cells"][0]["source"] = [
        "# Run\n[bad](missing.md)\n",
        "[current source](https://github.com/Gorkowski/particula/blob/main/missing.py)\n",
        "[historical](https://github.com/uncscode/particula/blob/abc123/old.py)\n",
        "```python\ntext = '[not a link](ignored.md)'\n```\n",
    ]
    path.write_text(json.dumps(notebook))
    errors = MODULE.check(tree)
    assert len(errors) == 2
    assert all("missing local link" in error for error in errors)


def test_percent_source_prose_links_are_checked_independently(tree):
    """Valid notebook prose must not hide stale links in its paired source."""
    write(
        tree,
        "docs/examples/run.py",
        "# %% [markdown]\n"
        "# [old](../old.md)\n"
        "# [missing](missing.md)\n"
        "# [bad anchor](../guide/child.md#missing)\n"
        "# %%\n"
        "# [not prose](ignored.md)\n"
        "result = 1\n",
    )
    errors = MODULE.check(tree)
    assert any("stale moved link" in e for e in errors)
    assert any("missing local link: missing.md" in e for e in errors)
    assert any("missing anchor" in e for e in errors)
    assert all("ignored.md" not in e for e in errors)

    write(
        tree,
        "docs/examples/run.py",
        "# %% [markdown]\n"
        "# [current](../guide/child.md#after)\n"
        "# %%\nresult = 1\n",
    )
    assert MODULE.check(tree) == []


def test_rendered_gates_require_api_redirect_fragments_and_clean_search(tree):
    """Missing generated content or a retired search route fails publication."""
    site = tree / "site"
    pages = {tree / "docs/guide/child.md"}
    manifest = json.loads((tree / "docs/.assets/relocations.json").read_text())
    write(site, "guide/child/index.html", '<h1 id="after">After</h1>')
    write(site, "API/index.html", "API")
    write(site, "old/index.html", '<meta name="robots" content="noindex">')
    write(site, "search/search_index.json", '{"docs": []}')
    assert MODULE.check_published(tree, site, pages, manifest) == []
    write(site, "guide/child/index.html", '<h1 id="changed">Changed</h1>')
    (site / "API/index.html").unlink()
    manifest["retired"] = {"docs/old.md": ".trash/old.md"}
    write(
        site,
        "search/search_index.json",
        '{"docs": [{"location":"old/#before"}]}',
    )
    errors = MODULE.check_published(tree, site, pages, manifest)
    assert any("missing published page: docs/API" in e for e in errors)
    assert any("missing redirect fragment" in e for e in errors)
    assert any("retired source in search" in e for e in errors)


def test_section_tabs_open_their_landing(tree):
    """An otherwise reachable landing must still be the section entry point."""
    path = tree / "mkdocs.yml"
    config = yaml.safe_load(path.read_text())
    config["nav"][0] = {
        "Guide": [{"Child": "guide/child.md"}, {"Guide": "guide/index.md"}]
    }
    path.write_text(yaml.safe_dump(config))
    assert "section landing must be first: Guide" in MODULE.check(tree)


def test_rendered_notebook_urls_and_project_mount(tree):
    """Source-correct notebook links can still be broken in rendered HTML."""
    site = tree / "site"
    page = tree / "docs/examples/run.ipynb"
    write(site, "examples/run/index.html", '<a href="../target.md">bad</a>')
    assert any(
        "broken rendered link" in e
        for e in MODULE.check_rendered_links(tree, site, {page})
    )
    write(
        site,
        "examples/run/index.html",
        '<a href="/particula/guide/child/#after">ok</a>',
    )
    write(site, "guide/child/index.html", '<h1 id="after">After</h1>')
    config = tree / "mkdocs.yml"
    config.write_text(
        config.read_text() + "\nsite_url: https://example.org/particula\n"
    )
    assert MODULE.check_rendered_links(tree, site, {page}) == []
