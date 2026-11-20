"""Reject documentation regressions using small independent fixture trees."""

import json
import sys
from pathlib import Path

import pytest

# Match direct script execution so sibling helpers resolve even when this is
# the first or only test module collected (as in the documentation CI job).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_docs_structure as structure  # noqa: E402


def write(root, path, text):
    """Write one fixture file with its parent directories."""
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


@pytest.fixture
def tree(tmp_path):
    """Create five sections, an indexed child, and a registered Python example."""
    write(tmp_path, "docs/guide/topics/index.md", "# Explore topics\n")
    nav = [
        {
            "label": "Explore topics",
            "items": [{"label": "Topic explorer", "slug": "guide/topics"}],
        }
    ]
    for title in structure.SECTIONS:
        if title == "Studies":
            write(
                tmp_path,
                "docs/studies/scripts.md",
                "# Scripts\n[Studies](index.md)\n",
            )
            nav.append(
                {
                    "label": title,
                    "items": [
                        {
                            "label": "All runnable Python scripts",
                            "slug": "studies/scripts",
                        }
                    ],
                }
            )
        elif title == "API":
            nav.append({"label": title})
        else:
            path = f"{title.lower()}/index.md"
            write(tmp_path, "docs/" + path, f"# {title}\n")
            nav.append(
                {
                    "label": title,
                    "items": [{"label": title, "slug": title.lower()}],
                }
            )
    write(tmp_path, "website/navigation.json", json.dumps(nav))
    write(
        tmp_path,
        "website/examples.json",
        json.dumps({"run": {"source": "docs/studies/run.py"}}),
    )
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
    write(tmp_path, "docs/studies/index.md", "# Examples\n[run](run.ipynb)\n")
    write(
        tmp_path,
        "docs/studies/run.py",
        "# %% [markdown]\n# # Run\n# %%\nresult = 1\n",
    )
    return tmp_path


def test_indexes_scripts_and_archives_are_accounted_for(tree):
    """Owning-index links classify pages; archives never enter discovery."""
    write(tree, ".trash/docs-v0.3.0/docs/obsolete.ipynb", "not JSON")
    assert structure.check(tree) == []


@pytest.mark.parametrize(
    "change,expected",
    [
        ("nav", "missing navigation destination"),
        ("orphan", "unreachable authored page"),
        ("owner", "unclassified page"),
        ("link", "missing local link"),
        ("anchor", "missing anchor"),
        ("stale", "stale moved link"),
        ("registry", "unregistered source"),
        ("source", "unregistered source"),
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
        "registry": ("website/examples.json", "{}"),
        "source": ("docs/studies/hidden.py", "result = 1"),
    }
    path, text = changes[change]
    if text is None:
        (tree / path).unlink()
    else:
        write(tree, path, text)
    assert any(expected in error for error in structure.check(tree))


def test_percent_source_prose_links_are_checked_independently(tree):
    """Valid notebook prose must not hide stale links in its paired source."""
    write(
        tree,
        "docs/studies/run.py",
        "# %% [markdown]\n"
        "# [old](../old.md)\n"
        "# [missing](missing.md)\n"
        "# [bad anchor](../guide/child.md#missing)\n"
        "# %%\n"
        "# [not prose](ignored.md)\n"
        "result = 1\n",
    )
    errors = structure.check(tree)
    assert any("stale moved link" in e for e in errors)
    assert any("missing local link: missing.md" in e for e in errors)
    assert any("missing anchor" in e for e in errors)
    assert all("ignored.md" not in e for e in errors)

    write(
        tree,
        "docs/studies/run.py",
        "# %% [markdown]\n"
        "# [current](../guide/child.md#after)\n"
        "# %%\nresult = 1\n",
    )
    assert structure.check(tree) == []


def test_section_tabs_open_their_landing(tree):
    """An otherwise reachable landing must still be the section entry point."""
    path = tree / "website/navigation.json"
    config = json.loads(path.read_text())
    guide = next(item for item in config if item["label"] == "Guide")
    guide["items"] = [
        {"label": "Child", "slug": "guide/child"},
        {"label": "Guide", "slug": "guide"},
    ]
    path.write_text(json.dumps(config))
    assert "section landing must be first: Guide" in structure.check(tree)


def test_topic_navigation_rejects_unregistered_generated_routes(tree):
    """Generated topics resolve through the explorer; unknown topics fail."""
    write(
        tree,
        "website/topics.json",
        json.dumps({"topics": [{"id": "state"}]}),
    )
    docs = tree / "docs"
    nav = structure.source_navigation(
        [
            {
                "label": "Explore topics",
                "items": [
                    {"label": "Overview", "slug": "guide/topics"},
                    {"label": "State", "slug": "guide/topics/state"},
                    {"label": "Missing", "slug": "guide/topics/missing"},
                ],
            }
        ],
        docs,
    )
    _, errors = structure.check_navigation(
        nav, docs, {docs / "guide/topics/index.md"}
    )
    assert errors == ["missing navigation destination: guide/topics/missing.md"]


def test_rendered_notebook_urls_and_project_mount(tree):
    """Source-correct notebook links can still be broken in rendered HTML."""
    site = tree / "site"
    page = tree / "docs/studies/run.py"
    write(site, "studies/run/index.html", '<a href="../target.md">bad</a>')
    assert any(
        "broken rendered link" in e
        for e in structure.check_rendered_links(tree, site, {page})
    )
    write(
        site,
        "studies/run/index.html",
        '<a href="/particula/guide/child/#after">ok</a>',
    )
    write(site, "guide/child/index.html", '<h1 id="after">After</h1>')
    assert structure.check_rendered_links(tree, site, {page}) == []


def test_notebook_publication_sources_cannot_return(tree):
    """Original notebooks belong only in the excluded reference archive."""
    write(tree, "docs/studies/run.ipynb", "{}")
    assert any(
        "retired notebook publication source" in e
        for e in structure.check(tree)
    )


@pytest.mark.parametrize(
    "declared,noindex,redirect,allowed",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_retired_content_allows_only_declared_redirects(
    tmp_path, declared, noindex, redirect, allowed
):
    """Retirement must coexist with required noindex migration redirects."""
    source = "docs/old.md"
    html = "<h1>Old route</h1>"
    if noindex:
        html += '<meta name="robots" content="noindex">'
    if redirect:
        html += "<html data-particula-redirect>"
    write(tmp_path, "old/index.html", html)
    write(tmp_path, "pagefind/pagefind.js", "")
    manifest = {
        "retired": {source: "archived"},
        "moves": {source: "docs/guide/index.md"} if declared else {},
    }
    errors = structure.check_search(tmp_path, manifest)
    assert (errors == []) is allowed
