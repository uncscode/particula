"""Pytest configuration for particula test suite.

This file registers custom markers and configures pytest options.
Having markers here ensures they are registered even in environments
where pyproject.toml configuration may not be read (e.g., conda-build).
"""

from __future__ import annotations

import pytest


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
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance-intensive "
        "(deselect with '-m \"not performance\"')",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as GPU benchmarks (enable with '--benchmark')",
    )
