"""Tests for benchmark gating through registered pytest config hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from particula import conftest as particula_conftest

_BENCHMARK_SKIP_REASON = (
    "GPU benchmarks skipped (pass --benchmark to enable)"
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
