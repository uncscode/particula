"""Regressions for version gating and isolated release-test staging."""

from __future__ import annotations

import importlib.util
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


@pytest.mark.parametrize(
    "path",
    [
        "conda/recipe/meta.yaml",
        "conda/recipe/conda_build_config.yaml",
        "scripts/conda_feedstock.py",
        "scripts/run_release_tests.py",
        "scripts/gpu_studies/cases.py",
        "particula/execution/tests/gpu_coupled_test.py",
        "particula/example/tests/future_fixture.py",
        "particula/integration_tests/future_test.py",
        "scripts/check_feedstock_contract.py",
        "scripts/tests/new_test.py",
        ".github/workflows/conda-feedstock.yml",
        ".github/workflows/pypi.yml",
        "pyproject.toml",
        "_particula_build.py",
        "particula_docs.py",
        "docs/guide/data-model/index.md",
        "conftest.py",
        "particula/conftest.py",
        "docs/studies/examples/particle-processes/sources-and-dilution/cpu_dilution.py",
        "examples_tests/dilution_example_test.py",
        "examples_tests/nucleation_example_test.py",
        "examples_tests/condensation_latent_heat_example_test.py",
    ],
)
def test_release_infrastructure_changes_enable_build(runner, path):
    """A recipe or runner fix receives validation without a version bump."""
    assert runner.release_inputs_changed([path])


def test_unrelated_prose_does_not_enable_build(runner):
    """Keep release builds scoped to version and release input changes."""
    assert not runner.release_inputs_changed(["readme.md", "todo_fix.md"])


def test_recipe_only_pr_writes_true_gate(runner, monkeypatch, tmp_path):
    """Exercise the three-dot gate with identical versions and recipe drift."""
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/mock/git")
    responses = iter(
        [
            "ancestor-sha\n",
            '__version__ = "0.2.14"',
            '__version__ = "0.2.14"',
            "conda/recipe/meta.yaml\0",
        ]
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(responses)),
    )
    assert runner.check_version("base", "head", include_release_inputs=True)
    assert output.read_text() == "changed=true\n"


def test_workflow_triggers_cover_release_gate_inputs(runner):
    """Inputs recognized by the gate must first trigger the Actions workflow."""
    workflow = yaml.safe_load(
        (SCRIPTS.parent / ".github/workflows/conda-feedstock.yml").read_text(),
    )
    # PyYAML's YAML 1.1 resolver reads the unquoted Actions `on` key as True.
    events = workflow.get("on", workflow.get(True))
    patterns = events["pull_request"]["paths"]
    for path in (
        *runner.RELEASE_PATHS,
        *runner.study_inputs(SCRIPTS.parent),
        runner.VERSION_PATH,
        "conda/recipe/meta.yaml",
        "scripts/tests/new_test.py",
        "docs/guide/data-model/index.md",
    ):
        assert any(fnmatchcase(path, pattern) for pattern in patterns), path


def test_recipe_declared_inputs_exist_in_staged_source(runner, tmp_path):
    """Exercise the build staging boundary with the actual mirror input list."""
    import check_feedstock_contract as contract

    source = tmp_path / "source"
    required = (
        "particula/__init__.py",
        "particula/conftest.py",
        "particula/tests/example_test.py",
        "pyproject.toml",
        "conftest.py",
        "readme.md",
        "license",
        "_particula_build.py",
        "particula_docs.py",
        "docs/guide/index.md",
        "docs/studies/unlisted.py",
        "docs/data/reference.csv",
        "scripts/fixture.py",
        "scripts/study_downloads.py",
        "website/examples.json",
        "scripts/run_release_tests.py",
        *runner.CPU_EXAMPLES,
        *runner.CPU_EXAMPLE_TESTS,
    )
    for relative in required:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (source / "website/examples.json").write_text(
        '{"demo": {"inputs": ["scripts/fixture.py", "docs/data/reference.csv"]}}'
    )
    excluded = (
        "docs/Features/planning.md",
        "docs/guide/__pycache__/cache.pyc",
        "examples_tests/unlisted_test.py",
        ".opencode/plan.md",
    )
    for relative in excluded:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not a release input")
    staged = tmp_path / "staged"
    runner.stage_source(source, staged)
    recipe = (SCRIPTS.parent / "conda/recipe/meta.yaml").read_text()
    for relative in contract.test_contract(recipe)["source_files"]:
        assert (staged / relative).exists(), relative
    assert not any((staged / relative).exists() for relative in excluded)
    assert all((staged / relative).is_file() for relative in required)


def test_installed_suite_requires_all_package_test_inputs(tmp_path):
    """A missing fixture fails even if all of its test modules were shipped."""
    import run_release_tests as release

    source = tmp_path / "source"
    installed = tmp_path / "installed"
    inputs = (
        "conftest.py",
        "integration_tests/process_test.py",
        "gas/tests/gas_test.py",
        "gas/tests/fixtures/reference.csv",
    )
    for relative in inputs:
        for folder in (source, installed):
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    release.check_installed_test_inputs(source, installed)
    for relative in inputs:
        path = installed / relative
        path.unlink()
        with pytest.raises(FileNotFoundError, match=relative):
            release.check_installed_test_inputs(source, installed)
        path.touch()


def test_missing_package_is_an_error_instead_of_silent_skip(tmp_path):
    """A missing package test tree must not produce an empty release suite."""
    import run_release_tests as release

    with pytest.raises(FileNotFoundError):
        release.check_installed_test_inputs(tmp_path / "absent", tmp_path)


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
