"""Exercise source-link translation and saved-output placement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_site as site  # noqa: E402


def test_links_use_base_for_complete_site():
    """All migrated links use the local base; fenced code stays untouched."""
    text = "[First run](getting-started/first-simulation.md)\n[Other](processes/wall-loss.md)\n```python\nprint('[x](fake.md)')\n```"
    result = site.rewrite_links(text, "docs/guide/index.md")
    assert "](/particula/guide/getting-started/first-simulation/)" in result
    assert "[Other](/particula/guide/processes/wall-loss/)" in result
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


def test_progress_updates_collapse_without_hiding_scientific_results():
    """One closed disclosure collects stream updates but not scientific text."""
    progress = (
        "Executing Runnable:   9%|▉ | 9289/100000 [00:21<03:35, 421.32it/s]"
    )
    result = site.output_markdown(
        [
            {"text": "Starting simulation\n\r" + progress + "\r"},
            {"text": "\x1b[32m" + progress + "\x1b[0m\nMass conserved: True\n"},
            {"image": "plot.png"},
        ],
        "/particula/results/demo",
        "Example",
    )
    assert result.count("<summary>Execution progress</summary>") == 1
    before, folded = result.split('<details class="execution-progress"')
    assert "Mass conserved: True" in before
    assert "<img " in before
    assert "Executing Runnable" not in before
    assert "Executing Runnable" in folded
    assert " open" not in folded
    assert "\x1b" not in result
    assert "&lt;03:35" in folded


def test_same_site_absolute_links_are_validated_locally(tmp_path):
    """An absolute published URL cannot hide a missing local page or anchor."""
    page = tmp_path / "index.html"
    target = tmp_path / "guide/index.html"
    target.parent.mkdir()
    target.write_text('<h1 id="known">Guide</h1>')
    page.write_text(
        '<a href="https://uncscode.github.io/particula/missing/">Missing</a>'
        '<a href="https://uncscode.github.io/particula/guide/#lost">Lost</a>'
        '<a href="https://uncscode.github.io/particula/guide/#known">Good</a>'
    )
    pages = {p: site.Links(p.read_text()) for p in (page, target)}
    errors = site.check_rendered_links(tmp_path, pages)
    assert len(errors) == 2
    assert any("Missing target" in error for error in errors)
    assert any("Missing anchor" in error for error in errors)


def test_search_and_sitemaps_reject_redirect_and_archive_pages(tmp_path):
    """Inspect compressed search contents, not just index-file existence."""
    import gzip
    import json

    search = tmp_path / "pagefind/fragment"
    search.mkdir(parents=True)
    for key, url in enumerate(("/old/", "/.trash/hidden/")):
        (search / f"{key}.pf_fragment").write_bytes(
            gzip.compress(b"pagefind_dcd" + json.dumps({"url": url}).encode())
        )
    target = tmp_path / "old/index.html"
    target.parent.mkdir()
    target.write_text("<html data-particula-redirect></html>")
    (tmp_path / "sitemap-0.xml").write_text(
        "<urlset><url><loc>https://uncscode.github.io/particula/old/</loc>"
        "</url></urlset>"
    )
    errors = site.check_discovery(
        tmp_path, {target: site.Links(target.read_text())}
    )
    assert any("Archive indexed" in error for error in errors)
    assert any("Redirect indexed" in error for error in errors)
