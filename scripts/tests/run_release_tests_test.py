"""Regressions for independent installed-package and CPU-example release gates."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
EXAMPLE_SCRIPTS = (
    "docs/Examples/cpu_dilution.py",
    "docs/Examples/Nucleation/cpu_nucleation.py",
    "docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py",
)
EXAMPLE_TESTS = (
    "examples_tests/dilution_example_test.py",
    "examples_tests/nucleation_example_test.py",
    "examples_tests/condensation_latent_heat_example_test.py",
)
DOMAINS = (
    "dynamics/condensation/",
    "dynamics/coagulation/",
    "dynamics/nucleation/",
    "gas/",
    "particles/",
    "integration_tests/",
    "execution/",
)


@pytest.fixture
def release():
    """Load the CLI without adding the source checkout to sys.path."""
    spec = importlib.util.spec_from_file_location(
        "release_runner_under_test", SCRIPTS / "run_release_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    """Provide declared inputs and unrelated files that must never be staged."""
    root = tmp_path / "source"
    for relative in (
        "conftest.py",
        "pyproject.toml",
        "particula/__init__.py",
        "particula/tests/runtime_test.py",
        *EXAMPLE_SCRIPTS,
        *EXAMPLE_TESTS,
        "docs/Examples/unlisted.py",
        "docs/prose.md",
        "examples_tests/unlisted_test.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    return root


def test_examples_stage_only_three_explicit_test_script_pairs(
    release, source, tmp_path
):
    """Examples do not accidentally stage the package suite or extra docs."""
    destination = tmp_path / "inputs"
    release.stage_test_inputs(source, destination, "examples")
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {"conftest.py", "pyproject.toml", *EXAMPLE_TESTS, *EXAMPLE_SCRIPTS}
    for relative in (*EXAMPLE_TESTS, *EXAMPLE_SCRIPTS):
        assert (destination / relative).read_bytes() == (
            source / relative
        ).read_bytes()


@pytest.mark.parametrize("missing", (*EXAMPLE_SCRIPTS, *EXAMPLE_TESTS))
def test_each_missing_example_input_fails_staging(
    release, source, tmp_path, missing
):
    """Neither a missing executable nor a missing test silently reduces scope."""
    (source / missing).unlink()
    with pytest.raises(FileNotFoundError) as error:
        release.stage_test_inputs(source, tmp_path / "inputs", "examples")
    assert Path(error.value.filename) == source / missing


def test_package_staging_does_not_require_example_inputs(release, tmp_path):
    """A package-only source can be tested with no docs checkout at all."""
    source = tmp_path / "source"
    (source / "particula/tests").mkdir(parents=True)
    for relative in (
        "conftest.py",
        "pyproject.toml",
        "particula/tests/runtime_test.py",
    ):
        (source / relative).write_text("")
    destination = tmp_path / "inputs"
    release.stage_test_inputs(source, destination)
    assert (destination / "particula/tests/runtime_test.py").is_file()
    assert not (destination / "docs").exists()


def test_unknown_suite_rejects_before_staging(release, source, tmp_path):
    """Direct staging callers cannot silently select an unintended suite."""
    destination = tmp_path / "inputs"
    with pytest.raises(ValueError, match="Unknown release suite"):
        release.stage_test_inputs(source, destination, "all")
    assert not destination.exists()


@pytest.mark.parametrize("suite", ("package", "examples"))
@pytest.mark.parametrize("missing", ("conftest.py", "pyproject.toml"))
def test_both_suites_require_warning_bootstrap_and_config(
    release, source, tmp_path, suite, missing
):
    """Missing shared configuration cannot silently change test policy."""
    (source / missing).unlink()
    with pytest.raises(FileNotFoundError) as error:
        release.stage_test_inputs(source, tmp_path / "inputs", suite)
    assert Path(error.value.filename) == source / missing


@pytest.mark.parametrize("suite", ("package", "examples"))
def test_collection_guard_requires_every_suite_domain(release, suite):
    """Nonempty selections still fail if a required domain or example is lost."""
    paths = (
        EXAMPLE_TESTS
        if suite == "examples"
        else tuple(f"particula/{domain}runtime_test.py" for domain in DOMAINS)
    )
    items = [SimpleNamespace(nodeid=f"{path}::test_runtime") for path in paths]
    report = release.Report(suite)
    report.pytest_deselected([object(), object()])
    report.pytest_collection_finish(SimpleNamespace(items=items))
    assert report.counts["collected"] == len(items) + 2
    for index in range(len(items)):
        with pytest.raises(pytest.UsageError, match="No release tests for"):
            release.Report(suite).pytest_collection_finish(
                SimpleNamespace(items=items[:index] + items[index + 1 :])
            )
    with pytest.raises(pytest.UsageError, match="Release selection is empty"):
        release.Report(suite).pytest_collection_finish(
            SimpleNamespace(items=[])
        )


def test_isolated_workers_propagate_suites_failures_and_keep_separate_logs(
    release, source, monkeypatch, capsys
):
    """Worker launch sanitizes pytest injection and preserves both full logs."""
    for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        monkeypatch.setenv(name, "must not leak")
    launches = []

    class Worker:
        def __init__(self, command, **kwargs):
            suite = command[command.index("--suite") + 1]
            workspace = Path(command[command.index("--test-workspace") + 1])
            assert kwargs["cwd"] == workspace
            assert not workspace.is_relative_to(source)
            assert (workspace / "docs").exists() == (suite == "examples")
            assert command[:2] == [sys.executable, "-I"]
            assert command[command.index("--source") + 1] == str(source)
            assert kwargs["stderr"] == subprocess.STDOUT
            assert kwargs["stdout"] == subprocess.PIPE
            for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
                assert name not in kwargs["env"]
            assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
            self.stdout = io.StringIO(f"{suite} output\nfinal line\n")
            launches.append(suite)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 7

    monkeypatch.setattr(release.subprocess, "Popen", Worker)
    for suite in ("package", "examples"):
        assert release.run_installed(source, suite) == 7
    assert launches == ["package", "examples"]
    assert (source / ".artifacts/release-tests.log").read_text() == (
        "package output\nfinal line\n"
    )
    assert (source / ".artifacts/release-examples.log").read_text() == (
        "examples output\nfinal line\n"
    )
    assert capsys.readouterr().out == (
        "package output\nfinal line\nexamples output\nfinal line\n"
    )


@pytest.mark.parametrize("suite_args", ([], ["--suite", "examples"]))
@pytest.mark.parametrize("mode", ("worker", "installed", "wheel"))
def test_cli_propagates_selected_suite_and_failure_status(
    release, source, tmp_path, monkeypatch, suite_args, mode
):
    """All entry paths retain the selected gate and its failing exit status."""
    suite = "examples" if suite_args else "package"
    workspace = tmp_path / "inputs"
    args = ["runner", "--source", str(source), *suite_args]
    if mode == "worker":
        args += ["--test-workspace", str(workspace)]

        def installed_test_main(actual_source, actual_workspace, actual_suite):
            assert (actual_source, actual_workspace, actual_suite) == (
                source,
                workspace,
                suite,
            )
            return 9

        monkeypatch.setattr(release, "installed_test_main", installed_test_main)
    elif mode == "installed":
        args += ["--installed"]

        def run_installed(actual_source, actual_suite):
            assert (actual_source, actual_suite) == (source, suite)
            return 9

        monkeypatch.setattr(release, "run_installed", run_installed)
    else:
        monkeypatch.setattr(
            release.venv.EnvBuilder, "create", lambda *args: None
        )
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if "wheel" in command:
                wheels = Path(command[command.index("--wheel-dir") + 1])
                wheels.mkdir()
                (wheels / "particula-0.2.14-py3-none-any.whl").touch()
            assert kwargs["check"] is ("--installed" not in command)
            return SimpleNamespace(
                returncode=9 if "--installed" in command else 0
            )

        monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", args)
    assert release.main() == 9
    if mode == "wheel":
        assert len(commands) == 3
        assert "--no-deps" in commands[0]
        assert commands[0][-1] == str(source)
        assert commands[1][2:4] == ["pip", "install"]
        assert commands[2][1] == "-I"
        assert commands[2][-2:] == ["--suite", suite]
        assert commands[2][commands[2].index("--source") + 1] == str(source)


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    """Represent distribution metadata without depending on an editable install."""
    origin = tmp_path / "site-packages/particula/__init__.py"
    module = SimpleNamespace(__file__=str(origin), __version__="0.2.14")
    distribution = SimpleNamespace(
        version="0.2.14", locate_file=lambda _: origin
    )
    monkeypatch.setitem(sys.modules, "particula", module)
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda _: distribution
    )
    return module


@pytest.mark.parametrize("failure_step", ("wheel", "install"))
def test_build_or_install_failure_does_not_launch_tests(
    release, source, monkeypatch, failure_step
):
    """A failed fresh environment cannot fall through to another install."""
    monkeypatch.setattr(sys, "argv", ["runner", "--source", str(source)])
    monkeypatch.setattr(release.venv.EnvBuilder, "create", lambda *args: None)
    steps = []

    def run(command, **kwargs):
        step = command[3]
        steps.append(step)
        assert kwargs["check"] is True
        if step == failure_step:
            raise subprocess.CalledProcessError(11, command)
        assert step == "wheel"
        wheels = Path(command[command.index("--wheel-dir") + 1])
        wheels.mkdir()
        (wheels / "particula-0.2.14-py3-none-any.whl").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        release.main()
    assert error.value.returncode == 11
    assert steps == (
        ["wheel"] if failure_step == "wheel" else ["wheel", "install"]
    )


@pytest.mark.parametrize("suite", ("package", "examples"))
def test_installed_pytest_selects_only_its_suite(
    release, installed_package, source, tmp_path, monkeypatch, suite
):
    """The worker executes explicit targets with the corresponding count guard."""
    workspace = tmp_path / "inputs"
    calls = []

    def pip_check(command, **kwargs):
        assert command == [sys.executable, "-m", "pip", "check"]
        assert kwargs["check"] is True
        calls.append("pip")

    def pytest_main(args, plugins):
        calls.append("pytest")
        assert plugins[0].suite == suite
        assert "--import-mode=importlib" in args
        assert args[args.index("--rootdir") + 1] == str(workspace.parent)
        if suite == "examples":
            assert args[-3:] == list(EXAMPLE_TESTS)
            assert not any(arg.startswith("--ignore=") for arg in args)
        else:
            assert args[-1] == "particula"
            assert "--ignore=particula/gpu" in args
        return 6

    monkeypatch.setattr(release.subprocess, "run", pip_check)
    monkeypatch.setattr(pytest, "main", pytest_main)
    assert release.installed_test_main(source, workspace, suite) == 6
    assert calls == ["pip", "pytest"]


@pytest.mark.parametrize("invalid", ("source", "workspace", "docs", "version"))
def test_installed_worker_rejects_invalid_provenance_before_launch(
    release, installed_package, source, tmp_path, monkeypatch, invalid
):
    """Source shadows, documentation leakage, and metadata drift fail closed."""
    workspace = tmp_path / "inputs"
    if invalid in {"source", "workspace"}:
        root = source if invalid == "source" else workspace
        installed_package.__file__ = str(root / "particula/__init__.py")
        message = "imported source checkout"
    elif invalid == "docs":
        (workspace / "docs").mkdir(parents=True)
        message = "must not contain docs"
    else:
        installed_package.__version__ = "wrong"
        message = "differs from installed metadata"

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid workspace must reject before pip or pytest")

    monkeypatch.setattr(release.subprocess, "run", unexpected)
    monkeypatch.setattr(pytest, "main", unexpected)
    with pytest.raises(RuntimeError, match=message):
        release.installed_test_main(source, workspace)
