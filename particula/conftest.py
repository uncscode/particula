"""Pytest hooks for shared marker registration and benchmark gating.

This module keeps the repository marker vocabulary available even when
``pyproject.toml`` metadata is not loaded and preserves ``--benchmark`` as the
only collection-affecting pytest option.
"""

from __future__ import annotations

import pytest

from particula._pytest_support import (
    PYTEST_MARKER_LINES,
    benchmark_option_enabled,
    set_benchmark_option_state,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register repository-specific pytest command-line options.

    Args:
        parser: Pytest parser receiving option registrations.
    """
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable GPU benchmark tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the shared pytest marker vocabulary.

    Args:
        config: Pytest configuration object receiving marker declarations.
    """
    set_benchmark_option_state(benchmark_option_enabled(config))
    for marker_line in PYTEST_MARKER_LINES:
        config.addinivalue_line("markers", marker_line)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip benchmark-marked tests unless ``--benchmark`` is enabled.

    Args:
        config: Pytest configuration used to read option state.
        items: Collected pytest items that may receive skip markers.
    """
    if benchmark_option_enabled(config):
        return

    skip_benchmark = pytest.mark.skip(
        reason="GPU benchmarks skipped (pass --benchmark to enable)"
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
