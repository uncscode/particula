"""Reject documentation regressions using small independent fixture trees."""

import importlib.util
import json
from pathlib import Path

import pytest

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
    """Create five sections, an indexed child, and a registered Python example."""
    nav = []
    for title in MODULE.SECTIONS:
        if title == "API":
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
        json.dumps({"run": {"source": "docs/examples/run.py"}}),
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
    write(tmp_path, "docs/examples/index.md", "# Examples\n[run](run.ipynb)\n")
    write(
        tmp_path,
        "docs/examples/run.py",
        "# %% [markdown]\n# # Run\n# %%\nresult = 1\n",
    )
    return tmp_path


def test_indexes_scripts_and_archives_are_accounted_for(tree):
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
        "source": ("docs/examples/hidden.py", "result = 1"),
    }
    path, text = changes[change]
    if text is None:
        (tree / path).unlink()
    else:
        write(tree, path, text)
    assert any(expected in error for error in MODULE.check(tree))


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


def test_section_tabs_open_their_landing(tree):
    """An otherwise reachable landing must still be the section entry point."""
    path = tree / "website/navigation.json"
    config = json.loads(path.read_text())
    config[0]["items"] = [
        {"label": "Child", "slug": "guide/child"},
        {"label": "Guide", "slug": "guide"},
    ]
    path.write_text(json.dumps(config))
    assert "section landing must be first: Guide" in MODULE.check(tree)


def test_rendered_notebook_urls_and_project_mount(tree):
    """Source-correct notebook links can still be broken in rendered HTML."""
    site = tree / "site"
    page = tree / "docs/examples/run.py"
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
    assert MODULE.check_rendered_links(tree, site, {page}) == []


def test_notebook_publication_sources_cannot_return(tree):
    """Original notebooks belong only in the excluded reference archive."""
    write(tree, "docs/examples/run.ipynb", "{}")
    assert any(
        "retired notebook publication source" in e for e in MODULE.check(tree)
    )
