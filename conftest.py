"""Pytest bootstrap patches for NumPy/SciPy compatibility.

Applied at repository root so it runs before package imports. This avoids
SciPy import errors triggered by NumPy copy-mode behavior in some test
environments.
"""

from __future__ import annotations

import sys
import types
import warnings


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
        import scipy

        try:
            from scipy import stats as _stats

            _ = _stats.lognorm
            return
        except Exception:  # noqa: S110
            pass
    except Exception:  # noqa: S110
        scipy = None

    try:  # pragma: no cover - defensive patch

        class _StubLogNorm:
            """Stub of scipy.stats.lognorm that raises when used in tests."""

            def pdf(self, *args, **kwargs):
                """Disallow probability density evaluation in stubbed SciPy."""
                raise RuntimeError("scipy.stats.lognorm stubbed for tests")

            def rvs(self, *args, **kwargs):
                """Disallow random variate sampling in stubbed SciPy."""
                raise RuntimeError("scipy.stats.lognorm stubbed for tests")

        if scipy is None:
            _scipy_stub = types.ModuleType("scipy")
            sys.modules.setdefault("scipy", _scipy_stub)

        _scipy_stats_stub = types.ModuleType("scipy.stats")
        _scipy_stats_stub.lognorm = _StubLogNorm()  # type: ignore[attr-defined]
        sys.modules.setdefault("scipy.stats", _scipy_stats_stub)
        sys.modules["scipy"].stats = sys.modules["scipy.stats"]  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover  # noqa: S110
        pass


warnings.filterwarnings(
    "ignore",
    message="The NumPy module was reloaded.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="GasSpecies is deprecated.*",
    category=DeprecationWarning,
)

_patch_numpy_copy_mode()
_patch_scipy_stats()


def pytest_runtest_setup(item) -> None:
    """Ensure deprecation warnings are filtered even under -Werror."""
    warnings.filterwarnings(
        "ignore",
        message="GasSpecies is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
