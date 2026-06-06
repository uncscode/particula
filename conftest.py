"""Pytest bootstrap configuration.

Applied at repository root so it runs before package imports. Filters
noisy warnings that are expected during the test suite.
"""

from __future__ import annotations

import warnings


def _filter_expected_warnings() -> None:
    """Keep known third-party and compatibility warnings out of CI errors."""
    warnings.filterwarnings(
        "ignore",
        message="The NumPy module was reloaded.*",
        category=UserWarning,
        append=False,
    )
    warnings.filterwarnings(
        "ignore",
        message="GasSpecies is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
    warnings.filterwarnings(
        "ignore",
        message="Due to '_pack_', the 'APICLaunchParamRecord' Structure.*",
        category=DeprecationWarning,
        append=False,
    )


_filter_expected_warnings()


def pytest_configure() -> None:
    """Reapply filters after pytest processes command-line warnings."""
    _filter_expected_warnings()


def pytest_runtest_setup(item) -> None:
    """Ensure deprecation warnings are filtered even under -Werror."""
    _filter_expected_warnings()
