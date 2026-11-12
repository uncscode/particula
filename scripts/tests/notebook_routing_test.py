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
