"""Pytest configuration for particula test suite.

This file registers custom markers and configures pytest options.
Having markers here ensures they are registered even in environments
where pyproject.toml configuration may not be read (e.g., conda-build).
"""

from __future__ import annotations

import pytest

PYTEST_MARKER_LINES = (
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "performance: marks tests as performance-intensive "
    "(deselect with '-m \"not performance\"')",
    "benchmark: marks tests as GPU benchmarks (enable with '--benchmark')",
    "warp: marks tests as Warp-dependent or Warp-targeted",
    "cuda: marks tests as CUDA-specific or CUDA-if-available",
    "gpu_parity: marks tests as CPU/Warp/CUDA parity validation",
    "stochastic: marks tests as stochastic tolerance-band validation",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line options."""
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable GPU benchmark tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for the test suite."""
    for marker_line in PYTEST_MARKER_LINES:
        config.addinivalue_line("markers", marker_line)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip benchmark-marked tests unless the registered option enables them."""
    if config.getoption("--benchmark"):
        return

    skip_benchmark = pytest.mark.skip(
        reason="GPU benchmarks skipped (pass --benchmark to enable)"
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
