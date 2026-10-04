"""Source-checkout tests for marker registration and collection boundaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, cast

import pytest
from particula import _pytest_support as pytest_support
from particula import conftest as particula_conftest

_WARP_MARKED_GPU_TESTS = (
    Path("particula/gpu/tests/kernel_exports_test.py"),
    Path("particula/gpu/tests/warp_types_test.py"),
    Path("particula/gpu/dynamics/tests/coagulation_funcs_test.py"),
    Path("particula/gpu/dynamics/tests/condensation_funcs_test.py"),
    Path("particula/gpu/properties/tests/gas_properties_test.py"),
    Path("particula/gpu/properties/tests/particle_properties_test.py"),
)

# These modules have explicit missing-runtime guards. This is a test-module
# loading contract, not support for importing GPU packages without required Warp.
# coagulation_funcs_test.py is covered by the installed-Warp probe above instead:
# its function defaults require constants imported in the Warp-enabled branch.
# Copying that module cannot establish a supported missing-runtime contract.
_MISSING_WARP_MODULE_TESTS = (
    Path("particula/gpu/tests/conversion_test.py"),
    Path("particula/gpu/tests/warp_types_test.py"),
    Path("particula/gpu/properties/tests/gas_properties_test.py"),
    Path("particula/gpu/kernels/tests/environment_test.py"),
    Path("particula/gpu/kernels/tests/coagulation_test.py"),
)


@pytest.fixture(autouse=True)
def _restore_benchmark_option_env() -> Generator[None, None, None]:
    """Restore benchmark opt-in env state after each test."""
    previous = os.environ.get(pytest_support.BENCHMARK_OPTION_ENV_VAR)
    previous_owner = os.environ.get(
        pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR
    )
    yield
    if previous is None:
        os.environ.pop(pytest_support.BENCHMARK_OPTION_ENV_VAR, None)
    else:
        os.environ[pytest_support.BENCHMARK_OPTION_ENV_VAR] = previous
    if previous_owner is None:
        os.environ.pop(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR, None)
        return
    os.environ[pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR] = (
        previous_owner
    )


@dataclass
class _FakeParser:
    """Minimal pytest parser stub for option-registration tests."""

    options: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def addoption(self, name: str, **kwargs: object) -> None:
        """Record option registrations performed by the hook."""
        self.options.append((name, kwargs))


@dataclass
class _FakeConfigureConfig:
    """Minimal pytest config stub for marker-registration tests."""

    marker_lines: list[tuple[str, str]] = field(default_factory=list)

    def addinivalue_line(self, section: str, value: str) -> None:
        """Record ini marker registrations performed by the hook."""
        self.marker_lines.append((section, value))


@dataclass
class _FakeConfig:
    """Minimal pytest config stub for collection-hook tests."""

    benchmark_enabled: bool = False

    def getoption(self, name: str) -> bool:
        """Return the configured benchmark option state."""
        assert name == "--benchmark"
        return self.benchmark_enabled


@dataclass
class _FakeItem:
    """Minimal pytest item stub for collection-hook tests."""

    keywords: set[str]
    markers: list[pytest.MarkDecorator] = field(default_factory=list)

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        """Record markers applied by the collection hook."""
        self.markers.append(marker)


def _load_pyproject_markers() -> list[str]:
    """Load the static pytest marker list from pyproject.toml."""
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)
    return cast(
        list[str], pyproject["tool"]["pytest"]["ini_options"]["markers"]
    )


def test_pytest_configure_registers_expected_gpu_policy_markers() -> None:
    """The hook registers the full shared marker vocabulary."""
    config = _FakeConfigureConfig()

    particula_conftest.pytest_configure(cast(Any, config))

    assert config.marker_lines == [
        ("markers", marker_line)
        for marker_line in particula_conftest.PYTEST_MARKER_LINES
    ]
    assert {line.split(":", 1)[0] for _, line in config.marker_lines} == {
        "slow",
        "performance",
        "benchmark",
        "warp",
        "cuda",
        "gpu_parity",
        "stochastic",
    }


def test_pyproject_marker_list_matches_hook_marker_vocabulary() -> None:
    """Static pytest marker config stays aligned with the hook vocabulary."""
    assert {
        line.split(":", 1)[0].strip() for line in _load_pyproject_markers()
    } == {
        line.split(":", 1)[0].strip()
        for line in particula_conftest.PYTEST_MARKER_LINES
    }


def test_default_collection_leaves_gpu_policy_items_unmodified() -> None:
    """Default collection does not skip non-benchmark GPU policy markers."""
    items = [
        _FakeItem(keywords={"warp"}),
        _FakeItem(keywords={"cuda"}),
        _FakeItem(keywords={"gpu_parity"}),
        _FakeItem(keywords={"stochastic"}),
    ]

    particula_conftest.pytest_collection_modifyitems(
        cast(Any, _FakeConfig()),
        cast(Any, items),
    )

    assert all(item.markers == [] for item in items)


def test_default_collection_only_skips_benchmark_items_even_with_warp() -> None:
    """Mixed benchmark-plus-GPU items still only receive benchmark skipping."""
    benchmark_item = _FakeItem(keywords={"benchmark", "warp"})
    non_benchmark_item = _FakeItem(keywords={"warp", "gpu_parity"})

    particula_conftest.pytest_collection_modifyitems(
        cast(Any, _FakeConfig()),
        [cast(Any, benchmark_item), cast(Any, non_benchmark_item)],
    )

    assert len(benchmark_item.markers) == 1
    marker = benchmark_item.markers[0]
    assert marker.mark.name == "skip"
    assert "--benchmark" in marker.mark.kwargs["reason"]
    assert non_benchmark_item.markers == []


def test_no_extra_device_policy_option_is_registered() -> None:
    """The repository keeps benchmark as the only pytest policy option."""
    parser = _FakeParser()

    particula_conftest.pytest_addoption(cast(Any, parser))

    assert [name for name, _ in parser.options] == ["--benchmark"]


def test_registered_option_is_an_opt_in_boolean() -> None:
    """Benchmark remains a disabled-by-default boolean option with help."""
    parser = _FakeParser()

    particula_conftest.pytest_addoption(cast(Any, parser))

    assert len(parser.options) == 1
    name, options = parser.options[0]
    assert name == "--benchmark"
    assert options["action"] == "store_true"
    assert options["default"] is False
    assert options["help"]


def test_pytest_configure_resets_benchmark_env_when_option_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure should overwrite a stale enabled env state with disabled."""

    @dataclass
    class _FakeConfigureWithOption(_FakeConfigureConfig):
        benchmark_enabled: bool = False

        def getoption(self, name: str) -> bool:
            assert name == "--benchmark"
            return self.benchmark_enabled

    monkeypatch.setenv(pytest_support.BENCHMARK_OPTION_ENV_VAR, "1")
    monkeypatch.setenv(
        pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR,
        str(os.getpid()),
    )
    config = _FakeConfigureWithOption(benchmark_enabled=False)

    particula_conftest.pytest_configure(cast(Any, config))

    assert pytest_support.benchmark_option_enabled_from_env() is False


_COLLECTION_PROBE = textwrap.dedent(
    """
    import importlib
    import json
    import sys
    from pathlib import Path

    if sys.argv[2] == "missing":
        sys.modules["warp"] = None
        try:
            importlib.import_module("warp")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("Warp import was not blocked")

    import pytest

    class Probe:
        def __init__(self):
            self.items = []
            self.reports = []

        def pytest_collectreport(self, report):
            if not report.passed:
                self.reports.append({
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr),
                    "skip_reason": (
                        str(report.longrepr[2]) if report.skipped else None
                    ),
                })

        def pytest_collection_finish(self, session):
            self.items = [
                {
                    "nodeid": item.nodeid,
                    "markers": [mark.name for mark in item.iter_markers()],
                    "skip_reasons": [
                        str(mark.kwargs.get("reason", ""))
                        for mark in item.iter_markers(name="skip")
                    ],
                }
                for item in session.items
            ]

    probe = Probe()
    exit_code = pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider", *sys.argv[3:]],
        plugins=[probe],
    )
    Path(sys.argv[1]).write_text(json.dumps({
        "exit_code": int(exit_code),
        "items": probe.items,
        "reports": probe.reports,
    }), encoding="utf-8")
    """
)


def _collect_in_subprocess(
    targets,
    report_path,
    *,
    missing_warp=False,
    cwd=None,
):
    """Inspect real collected items without running GPU tests or fixtures."""
    environment = os.environ.copy()
    # Isolate nested collection from outer coverage, filters, and plugins.
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(  # noqa: S603 - fixed test-owned collection probe
        [
            sys.executable,
            "-c",
            _COLLECTION_PROBE,
            str(report_path),
            "missing" if missing_warp else "normal",
            *(path.as_posix() for path in targets),
        ],
        cwd=cwd if cwd is not None else Path(__file__).resolve().parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report_path.is_file(), result.stdout + result.stderr
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def normal_collection(tmp_path_factory):
    """Collect original source paths with the required Warp runtime installed."""
    return _collect_in_subprocess(
        _WARP_MARKED_GPU_TESTS,
        tmp_path_factory.mktemp("marker-probe") / "normal.json",
    )


@pytest.fixture(scope="module", params=_MISSING_WARP_MODULE_TESTS, ids=str)
def isolated_missing_warp_collection(request, tmp_path_factory):
    """Probe flat test copies, not imports of their GPU parent packages.

    Preserve root pytest configuration and warning hooks. Helpers remain on the
    existing installed-package import path; no GPU implementation is copied or
    stubbed. Warp is deliberately blocked only in this subprocess to exercise
    the test modules' own guards, not to make the runtime dependency optional.
    """
    root = Path(__file__).resolve().parents[2]
    workspace = tmp_path_factory.mktemp("missing-warp-module")
    for source in (Path("pyproject.toml"), Path("conftest.py"), request.param):
        shutil.copy2(root / source, workspace / source.name)
    report = _collect_in_subprocess(
        (Path(request.param.name),),
        workspace / "missing.json",
        missing_warp=True,
        cwd=workspace,
    )
    return request.param, report


@pytest.mark.parametrize("relative_path", _WARP_MARKED_GPU_TESTS, ids=str)
def test_warp_markers_are_effective_on_collected_items(
    normal_collection,
    relative_path,
) -> None:
    """Runtime pytest items, not source strings, expose Warp selection marks."""
    assert normal_collection["exit_code"] == 0, normal_collection["reports"]
    items = [
        item
        for item in normal_collection["items"]
        if item["nodeid"].split("::", 1)[0] == relative_path.as_posix()
    ]
    assert items, relative_path.as_posix()
    marked = [item for item in items if "warp" in item["markers"]]
    assert marked, relative_path.as_posix()
    # Export tests intentionally mix host-only and Warp-dependent assertions.
    if relative_path.name != "kernel_exports_test.py":
        assert marked == items


def _assert_missing_warp_reason(reason):
    """Require an actionable missing-Warp explanation, not fixed prose."""
    reason = reason.lower()
    assert "warp" in reason, reason
    assert any(
        phrase in reason
        for phrase in (
            "not installed",
            "unavailable",
            "missing",
            "could not import",
        )
    ), reason


def test_isolated_test_module_degrades_to_clean_skips_without_warp(
    isolated_missing_warp_collection,
) -> None:
    """A test module may mark every item skipped or explicitly skip its load.

    Collection errors are never acceptable. This deliberately says nothing
    about importing the full GPU package without its required Warp dependency.
    """
    relative_path, collection = isolated_missing_warp_collection
    reports = collection["reports"]
    assert all(report["outcome"] == "skipped" for report in reports), reports
    items = collection["items"]
    if items:
        assert collection["exit_code"] == pytest.ExitCode.OK, collection
        assert not reports, reports
        for item in items:
            assert item["nodeid"].split("::", 1)[0] == relative_path.name
            assert "warp" in item["markers"], item
            assert "skip" in item["markers"], item
            assert item["skip_reasons"], item
            for reason in item["skip_reasons"]:
                _assert_missing_warp_reason(reason)
    else:
        assert collection["exit_code"] == pytest.ExitCode.NO_TESTS_COLLECTED
        assert len(reports) == 1, collection
        assert reports[0]["nodeid"] == relative_path.name
        _assert_missing_warp_reason(reports[0]["skip_reason"])
