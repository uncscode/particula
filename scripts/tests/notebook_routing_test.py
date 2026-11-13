"""Publication routing preserves API Markdown and notebook eligibility."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_api_markdown_never_enters_notebook_parser():
    """API directives remain Markdown; notebook requests retain plugin policy."""
    path = Path(__file__).parents[2] / "docs/.assets/notebook_routing.py"
    spec = importlib.util.spec_from_file_location("notebook_routing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def original(file):
        calls.append(file.abs_src_path)
        return file.abs_src_path.endswith(".ipynb")

    plugin = SimpleNamespace(should_include=original)
    config = SimpleNamespace(plugins={"mkdocs-jupyter": plugin})
    assert module.on_config(config) is config
    assert not plugin.should_include(SimpleNamespace(abs_src_path="API/a.md"))
    assert calls == []
    assert plugin.should_include(SimpleNamespace(abs_src_path="example.ipynb"))
    assert not plugin.should_include(SimpleNamespace(abs_src_path="example.py"))
    assert calls == ["example.ipynb", "example.py"]
    absent = SimpleNamespace(plugins={})
    assert module.on_config(absent) is absent


def test_notebook_links_follow_publication_routes():
    """Cross-topic links use rendered URLs without redirecting downloads."""
    path = Path(__file__).parents[2] / "docs/.assets/notebook_routing.py"
    spec = importlib.util.spec_from_file_location("notebook_links", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = SimpleNamespace(src_uri="examples/topic/run.ipynb")
    target = SimpleNamespace(url="guide/data/")
    sibling = SimpleNamespace(url="examples/topic/other/")
    files = SimpleNamespace(
        get_file_from_path={
            "guide/data.md": target,
            "examples/topic/other.ipynb": sibling,
            "examples/topic/run.ipynb": source,
        }.get
    )
    page = SimpleNamespace(file=source, url="examples/topic/run/")
    html = (
        '<a href="../../guide/data.md#units">guide</a>'
        '<a href="other.ipynb">sibling</a>'
        '<a href="run.ipynb">download</a>'
        '<a href="#local">local</a>'
        '<a href="https://example.org">external</a>'
    )
    actual = module.on_page_content(html, page, None, files)
    assert 'href="../../../guide/data/#units"' in actual
    assert 'href="../other/"' in actual
    assert 'href="run.ipynb"' in actual
    assert 'href="#local"' in actual
    assert 'href="https://example.org"' in actual
