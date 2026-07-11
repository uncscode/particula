"""Tests for benchmark gating through registered pytest config hooks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Generator, cast

import pytest
from particula import _pytest_support as pytest_support
from particula import conftest as particula_conftest

_BENCHMARK_SKIP_REASON = "GPU benchmarks skipped (pass --benchmark to enable)"


@pytest.fixture(autouse=True)
def _restore_benchmark_option_env() -> Generator[None, None, None]:
    """Restore benchmark opt-in env state after each test."""
    previous = os.environ.get(pytest_support.BENCHMARK_OPTION_ENV_VAR)
    previous_owner = os.environ.get(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR)
    yield
    if previous is None:
        os.environ.pop(pytest_support.BENCHMARK_OPTION_ENV_VAR, None)
    else:
        os.environ[pytest_support.BENCHMARK_OPTION_ENV_VAR] = previous
    if previous_owner is None:
        os.environ.pop(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR, None)
        return
    os.environ[pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR] = previous_owner


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
    """Minimal pytest config stub for benchmark hook tests."""

    benchmark_enabled: bool

    def getoption(self, name: str) -> bool:
        """Return the configured benchmark option state."""
        assert name == "--benchmark"
        return self.benchmark_enabled


@dataclass
class _FakeItem:
    """Minimal pytest item stub for benchmark hook tests."""

    keywords: set[str]
    markers: list[pytest.MarkDecorator] = field(default_factory=list)

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        """Record markers applied by the collection hook."""
        self.markers.append(marker)


def test_benchmark_option_is_registered() -> None:
    """Pytest adds the benchmark command-line toggle."""
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


def test_benchmark_related_markers_are_registered() -> None:
    """Pytest registers the shared marker vocabulary."""
    config = _FakeConfigureConfig()

    particula_conftest.pytest_configure(cast(Any, config))

    assert config.marker_lines == [
        ("markers", marker_line)
        for marker_line in particula_conftest.PYTEST_MARKER_LINES
    ]


def test_set_benchmark_option_state_round_trips_through_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved benchmark state is shared through the dedicated env var."""
    monkeypatch.delenv(
        pytest_support.BENCHMARK_OPTION_ENV_VAR, raising=False
    )
    monkeypatch.delenv(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR, raising=False)

    pytest_support.set_benchmark_option_state(True)
    assert pytest_support.benchmark_option_enabled_from_env() is True

    pytest_support.set_benchmark_option_state(False)
    assert pytest_support.benchmark_option_enabled_from_env() is False


def test_benchmark_option_enabled_from_env_ignores_inherited_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited benchmark env state from another PID does not opt in locally."""
    monkeypatch.setenv(pytest_support.BENCHMARK_OPTION_ENV_VAR, "1")
    monkeypatch.setenv(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR, "999999")

    assert pytest_support.benchmark_option_enabled_from_env() is False


def test_benchmark_option_enabled_reads_config_like_getoption() -> None:
    """Config-like objects expose the resolved benchmark option state."""
    assert (
        pytest_support.benchmark_option_enabled(
            _FakeConfig(benchmark_enabled=True)
        )
        is True
    )
    assert (
        pytest_support.benchmark_option_enabled(
            _FakeConfig(benchmark_enabled=False)
        )
        is False
    )


def test_benchmark_option_enabled_returns_false_without_callable_getoption() -> (
    None
):
    """Non-config objects default to disabled benchmark mode."""

    class _NoGetOption:
        pass

    class _NonCallableGetOption:
        getoption = True

    assert pytest_support.benchmark_option_enabled(_NoGetOption()) is False
    assert (
        pytest_support.benchmark_option_enabled(_NonCallableGetOption())
        is False
    )


def test_pytest_configure_persists_resolved_benchmark_option_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pytest configure exports the resolved --benchmark option state."""

    @dataclass
    class _FakeConfigureWithOption(_FakeConfigureConfig):
        benchmark_enabled: bool = False

        def getoption(self, name: str) -> bool:
            assert name == "--benchmark"
            return self.benchmark_enabled

    monkeypatch.delenv(
        pytest_support.BENCHMARK_OPTION_ENV_VAR, raising=False
    )
    monkeypatch.delenv(pytest_support.BENCHMARK_OPTION_OWNER_PID_ENV_VAR, raising=False)
    config = _FakeConfigureWithOption(benchmark_enabled=True)

    particula_conftest.pytest_configure(cast(Any, config))

    assert pytest_support.benchmark_option_enabled_from_env() is True


def test_pytest_configure_resets_benchmark_option_state_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pytest configure resets a preexisting opt-in when benchmark is off."""

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


def test_benchmark_collection_hook_skips_benchmark_tests_by_default() -> None:
    """Benchmark-marked items are skipped unless the option is enabled."""
    item = _FakeItem(keywords={"benchmark"})
    particula_conftest.pytest_collection_modifyitems(
        cast(Any, _FakeConfig(benchmark_enabled=False)),
        [cast(Any, item)],
    )
    assert len(item.markers) == 1

    marker = item.markers[0]
    assert isinstance(marker, pytest.MarkDecorator)
    assert marker.mark.name == "skip"
    assert marker.mark.kwargs == {"reason": _BENCHMARK_SKIP_REASON}


def test_benchmark_collection_hook_leaves_benchmark_tests_enabled() -> None:
    """Enabled benchmark mode leaves benchmark items unmodified."""
    item = _FakeItem(keywords={"benchmark"})
    particula_conftest.pytest_collection_modifyitems(
        cast(Any, _FakeConfig(benchmark_enabled=True)),
        [cast(Any, item)],
    )
    assert item.markers == []
