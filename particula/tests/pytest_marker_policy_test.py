"""Tests for Warp marker policy registration and collection boundaries."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, cast

import pytest
from particula import _pytest_support as pytest_support
from particula import conftest as particula_conftest

_BENCHMARK_SKIP_REASON = "GPU benchmarks skipped (pass --benchmark to enable)"

_WARP_MARKED_GPU_TESTS = (
    Path("particula/gpu/tests/kernel_exports_test.py"),
    Path("particula/gpu/tests/warp_types_test.py"),
    Path("particula/gpu/dynamics/tests/coagulation_funcs_test.py"),
    Path("particula/gpu/dynamics/tests/condensation_funcs_test.py"),
    Path("particula/gpu/properties/tests/gas_properties_test.py"),
    Path("particula/gpu/properties/tests/particle_properties_test.py"),
)

_COLLECTION_SAFE_WARP_TESTS = (
    Path("particula/gpu/tests/conversion_test.py"),
    Path("particula/gpu/tests/warp_types_test.py"),
    Path("particula/gpu/dynamics/tests/coagulation_funcs_test.py"),
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

    expected_marker_lines = (
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "performance: marks tests as performance-intensive "
        "(deselect with '-m \"not performance\"')",
        "benchmark: marks tests as GPU benchmarks (enable with '--benchmark')",
        "warp: marks tests as Warp-dependent or Warp-targeted",
        "cuda: marks tests as CUDA-specific or CUDA-if-available",
        "gpu_parity: marks tests as CPU/Warp/CUDA parity validation",
        "stochastic: marks tests as stochastic tolerance-band validation",
    )

    assert config.marker_lines == [
        ("markers", marker_line) for marker_line in expected_marker_lines
    ]
    assert particula_conftest.PYTEST_MARKER_LINES == expected_marker_lines


def test_pyproject_marker_list_matches_hook_marker_vocabulary() -> None:
    """Static pytest marker config stays aligned with the hook vocabulary."""
    assert _load_pyproject_markers() == list(
        particula_conftest.PYTEST_MARKER_LINES
    )


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
    assert marker.mark.kwargs == {"reason": _BENCHMARK_SKIP_REASON}
    assert non_benchmark_item.markers == []


def test_no_extra_device_policy_option_is_registered() -> None:
    """The repository keeps benchmark as the only pytest policy option."""
    parser = _FakeParser()

    particula_conftest.pytest_addoption(cast(Any, parser))

    assert [name for name, _ in parser.options] == ["--benchmark"]


def test_registered_option_help_text_matches_expected_policy_surface() -> None:
    """Option help text remains the documented benchmark-only policy surface."""
    parser = _FakeParser()

    particula_conftest.pytest_addoption(cast(Any, parser))

    assert parser.options == [
        (
            "--benchmark",
            {
                "action": "store_true",
                "default": False,
                "help": "Enable GPU benchmark tests.",
            },
        )
    ]


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


def test_remaining_warp_only_suites_are_explicitly_warp_marked() -> None:
    """Remaining Warp-only suites stay targetable through ``-m warp``."""
    repo_root = Path(__file__).resolve().parents[2]

    for relative_path in _WARP_MARKED_GPU_TESTS:
        contents = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "pytest.mark.warp" in contents, relative_path.as_posix()


def test_warp_marked_suites_avoid_module_level_importorskip() -> None:
    """Warp-marked suites should not rely on eager module-level importorskip."""
    repo_root = Path(__file__).resolve().parents[2]

    for relative_path in _COLLECTION_SAFE_WARP_TESTS:
        contents = (repo_root / relative_path).read_text(encoding="utf-8")
        assert 'pytest.importorskip("warp")' not in contents, (
            relative_path.as_posix()
        )
