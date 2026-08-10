"""Regression tests for ADW agent guidance contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    """Read a repository-relative guidance file."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_linter_guidance_matches_particula_ci_scope_and_mutation_safeguards():
    """Require CI-aligned targets and fail-closed mutation-scope guidance."""
    guidance = _read(".opencode/agent/linter.md")

    assert '"field": "worktree_path"' in guidance
    assert "before reading, editing, formatting, or running linters" in guidance
    assert "non-empty repository-relative path" in guidance
    assert "pre-existing changed paths wholly within `target_dir`" in guidance
    assert "newly changed path is outside `target_dir`" in guidance
    assert "ruff check particula/" in guidance
    assert "ruff format particula/ --check" in guidance
    assert "mypy particula/ --ignore-missing-imports" in guidance
    assert "adforge_core/" not in guidance
    assert "adforge_voice/" not in guidance


def test_test_guidance_identifies_wrapper_owned_fallback_coverage_policy():
    """Keep test agents aligned with the wrapper's fallback coverage authority."""
    build_tests_guidance = _read(".opencode/agent/adw-build-tests.md")
    tester_guidance = _read(".opencode/agent/adw-tester.md")
    runner = _read(".opencode/tools/run_pytest.py")

    for guidance in (build_tests_guidance, tester_guidance):
        assert ".opencode/tools/run_pytest.py" in guidance
        assert "80% fallback" in guidance
        assert "Do not invent or pass an explicit coverage threshold" in guidance
    assert "return max(values[0], 80.0) if values else 80.0" in runner
