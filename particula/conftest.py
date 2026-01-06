"""Pytest configuration for particula test suite.

This file registers custom markers and configures pytest options.
Having markers here ensures they are registered even in environments
where pyproject.toml configuration may not be read (e.g., conda-build).
"""

import pytest


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
