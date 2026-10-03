"""Regressions for version gating and isolated release-test staging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    """Load the standalone runner without changing package import paths."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "conda_feedstock_runner", SCRIPTS / "conda_feedstock.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("base", "head", "changed"),
    [
        ('__version__ = "0.2.13"', '__version__ = "0.2.14"', True),
        ('__version__ = "0.2.13"', '__version__ = "0.3.0"', True),
        ('__version__ = "0.2.13"', "__version__ = '0.2.13'", False),
        (
            '__version__ = "0.2.13"',
            'import nonexistent\n__version__ = "0.2.13"\nraise RuntimeError()',
            False,
        ),
    ],
)
def test_only_version_value_changes_enable_build(runner, base, head, changed):
    """Other initializer edits do not run the expensive conda job."""
    assert runner.version_changed(base, head) is changed


@pytest.mark.parametrize(
    "source",
    [
        "",
        "__version__ = make_version()",
        "__version__ = 3",
        '__version__ = ""',
        '__version__ = "1"\n__version__ = "2"',
    ],
)
def test_version_gate_rejects_ambiguous_or_nonliteral_versions(runner, source):
    """Unknown version forms fail rather than silently skipping validation."""
    with pytest.raises(ValueError):
        runner.read_version(source)


def test_version_gate_writes_action_output(runner, monkeypatch, tmp_path):
    """The PR comparison forwards the exact changed result to the build job."""
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/mock/git")
    responses = iter(
        ["ancestor-sha\n", '__version__ = "0.2.13"', '__version__ = "0.2.14"']
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.check_version("base-sha", "head-sha")
    assert output.read_text() == "changed=true\n"
    assert commands == [
        ["/mock/git", "merge-base", "base-sha", "head-sha"],
        ["/mock/git", "show", "ancestor-sha:particula/__init__.py"],
        ["/mock/git", "show", "head-sha:particula/__init__.py"],
    ]


def test_staging_retains_integration_and_fixtures_without_application_sources(
    runner,
    tmp_path,
):
    """A release workspace cannot shadow the installed application package."""
    import run_release_tests as release

    source = tmp_path / "source"
    destination = tmp_path / "tests"
    required = (
        "particula/conftest.py",
        "particula/integration_tests/process_test.py",
        "particula/gas/tests/gas_test.py",
        "particula/gas/tests/fixtures/reference.csv",
        "conftest.py",
        "pyproject.toml",
        *release.CPU_EXAMPLES,
    )
    excluded = (
        "particula/__init__.py",
        "particula/gas/__init__.py",
        "particula/gas/species.py",
        "particula/gas/tests/__pycache__/test.pyc",
    )
    for relative in (*required, *excluded):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    release.stage_test_inputs(source, destination)
    assert all((destination / relative).is_file() for relative in required)
    assert not any((destination / relative).exists() for relative in excluded)


def test_missing_example_is_an_error_instead_of_silent_skip(runner, tmp_path):
    """Missing declared release inputs fail during staging."""
    import run_release_tests as release

    source = tmp_path / "source"
    source.mkdir()
    (source / "conftest.py").write_text("")
    (source / "pyproject.toml").write_text("")
    with pytest.raises(FileNotFoundError):
        release.stage_test_inputs(source, tmp_path / "tests")


def test_release_counts_match_pytest_final_categories(runner, capsys):
    """Only terminal results count; successful calls may later be skipped."""
    import run_release_tests as release

    stats = {
        "passed": [object()] * 2,
        "skipped": [object()] * 3,
        "error": [object()],
    }
    session = SimpleNamespace(
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(
                getplugin=lambda _: SimpleNamespace(stats=stats)
            )
        )
    )
    report = release.Report()
    report.pytest_deselected([object()] * 4)
    report.pytest_sessionfinish(session, 1)
    assert report.counts == dict(
        collected=0, deselected=4, passed=2, skipped=3, failed=0, errors=1
    )
    assert '"passed": 2' in capsys.readouterr().out
