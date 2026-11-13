"""Exercise source-link translation and saved-output placement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_site as site  # noqa: E402


def test_links_use_base_and_unmigrated_links_are_explicit():
    """Links use base and unmigrated links are explicit."""
    text = "[First run](getting-started/first-simulation.md)\n[Other](processes/wall-loss.md)\n```python\nprint('[x](fake.md)')\n```"
    result = site.rewrite_links(text, "docs/guide/index.md")
    assert "](/particula/guide/getting-started/first-simulation/)" in result
    assert (
        "[Other ↗](https://uncscode.github.io/particula/guide/processes/wall-loss/)"
        in result
    )
    assert "print('[x](fake.md)')" in result


def test_figures_are_visible_and_stdout_is_escaped():
    """Figures are visible and stdout is escaped."""
    result = site.output_markdown(
        [{"image": "plot.png"}, {"text": "<script>bad()</script>"}],
        "/particula/results/demo",
        "Example",
    )
    assert '<img src="/particula/results/demo/plot.png"' in result
    assert "&lt;script&gt;" in result
    assert "<details>" not in result


def test_stale_results_are_separated_from_current_code(tmp_path, monkeypatch):
    """Stale results are separated from current code."""
    source = tmp_path / "docs/example.py"
    source.parent.mkdir()
    source.write_text("# %%\nx = 3\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "source.py").write_text("x = 2\n")
    manifest = {
        "device": "CPU",
        "executed_at": "2026-11-13",
        "revision": "abc123456789",
        "outputs": [[{"text": "2"}]],
    }
    monkeypatch.setattr(site, "ROOT", tmp_path)
    monkeypatch.setattr(site, "SITE", tmp_path / "website")
    monkeypatch.setattr(site, "CONTENT", tmp_path / "content")
    monkeypatch.setattr(site, "bundle", lambda _: (run, manifest))
    monkeypatch.setattr(site, "result_state", lambda *_: "stale")
    site.render_example("demo", {"source": "docs/example.py", "title": "Demo"})
    page = (tmp_path / "content/example.md").read_text()
    assert "refresh needed" in page
    assert (
        page.index("x = 3")
        < page.index("Previous run outputs")
        < page.index('class="result-output"')
    )


def test_route_preserves_case_and_underscores():
    """Route preserves case and underscores."""
    assert (
        site.route("docs/API/particula/particle_data.md")
        == "API/particula/particle_data"
    )
    assert site.route("docs/guide/index.md") == "guide"
    assert site.route("docs/guide/searchindex.md") == "guide/searchindex"
