"""Pytest configuration for particula test suite.

This file registers custom markers and configures pytest options.
Having markers here ensures they are registered even in environments
where pyproject.toml configuration may not be read (e.g., conda-build).
"""

from __future__ import annotations

import pytest


def _patch_numpy_copy_mode() -> None:
    """Patch NumPy CopyMode bool for SciPy import compatibility."""
    try:  # pragma: no cover - defensive patch
        from numpy import _globals as _np_globals

        _np_globals._CopyMode.__bool__ = lambda self: False  # type: ignore[method-assign]
    except Exception:  # pragma: no cover  # noqa: S110
        pass


def _patch_scipy_stats() -> None:
    """Stub scipy.stats.lognorm if SciPy import triggers NumPy copy errors."""
    try:  # pragma: no cover - defensive patch
        import sys
        import types

        class _StubLogNorm:
            """Stub of scipy.stats.lognorm that raises when used in tests."""

            def pdf(self, *args, **kwargs):
                """Disallow probability density evaluation in stubbed SciPy."""
                raise RuntimeError("scipy.stats.lognorm stubbed for tests")

            def rvs(self, *args, **kwargs):
                """Disallow random variate sampling in stubbed SciPy."""
                raise RuntimeError("scipy.stats.lognorm stubbed for tests")

        if "scipy" not in sys.modules:
            _scipy_stub = types.ModuleType("scipy")
            sys.modules["scipy"] = _scipy_stub

        _scipy_stats_stub = types.ModuleType("scipy.stats")
        _scipy_stats_stub.lognorm = _StubLogNorm()  # type: ignore[attr-defined]
        sys.modules.setdefault("scipy.stats", _scipy_stats_stub)
        sys.modules["scipy"].stats = sys.modules["scipy.stats"]  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover  # noqa: S110
        pass


_patch_numpy_copy_mode()
_patch_scipy_stats()


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
