"""Shared pytest support helpers for benchmark gating and marker policy.

These helpers are reused by ``conftest.py`` and benchmark-related tests without
importing pytest hook wiring as a helper module.
"""

from __future__ import annotations

import os
from typing import Any

BENCHMARK_OPTION_ENV_VAR = "PARTICULA_BENCHMARK_ENABLED"
BENCHMARK_OPTION_OWNER_PID_ENV_VAR = "PARTICULA_BENCHMARK_OWNER_PID"

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
    """Persist benchmark-option state for the current pytest process only."""
    os.environ[BENCHMARK_OPTION_ENV_VAR] = "1" if enabled else "0"
    os.environ[BENCHMARK_OPTION_OWNER_PID_ENV_VAR] = str(os.getpid())


def benchmark_option_enabled_from_env() -> bool:
    """Return benchmark state only when it was resolved in this process."""
    if os.getenv(BENCHMARK_OPTION_OWNER_PID_ENV_VAR) != str(os.getpid()):
        return False
    return os.getenv(BENCHMARK_OPTION_ENV_VAR, "0") == "1"


def benchmark_option_enabled(config: Any) -> bool:
    """Read benchmark option state from a pytest-like config object."""
    getoption = getattr(config, "getoption", None)
    if not callable(getoption):
        return False
    return bool(getoption("--benchmark"))
