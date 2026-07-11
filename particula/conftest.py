"""Pytest hooks for shared marker registration and benchmark gating.

This module keeps the repository marker vocabulary available even when
``pyproject.toml`` metadata is not loaded and preserves ``--benchmark`` as the
only collection-affecting pytest option.
"""

from __future__ import annotations

import os

import pytest

BENCHMARK_OPTION_ENV_VAR = "PARTICULA_BENCHMARK_ENABLED"

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


def set_benchmark_option_state(enabled: bool) -> None:
    """Persist the resolved benchmark option state for import-time consumers."""
    os.environ[BENCHMARK_OPTION_ENV_VAR] = "1" if enabled else "0"


def benchmark_option_enabled_from_env() -> bool:
    """Return the resolved benchmark option state from the shared env var."""
    return os.getenv(BENCHMARK_OPTION_ENV_VAR, "0") == "1"


def _benchmark_option_enabled(config: pytest.Config | object) -> bool:
    """Read the resolved benchmark option state from a pytest-like config."""
    getoption = getattr(config, "getoption", None)
    if not callable(getoption):
        return False
    return bool(getoption("--benchmark"))


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
    set_benchmark_option_state(_benchmark_option_enabled(config))
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
    if _benchmark_option_enabled(config):
        return

    skip_benchmark = pytest.mark.skip(
        reason="GPU benchmarks skipped (pass --benchmark to enable)"
    )
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
