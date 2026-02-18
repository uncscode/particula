"""Pytest bootstrap configuration.

Applied at repository root so it runs before package imports. Filters
noisy warnings that are expected during the test suite.
"""

from __future__ import annotations

import warnings

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


def pytest_runtest_setup(item) -> None:
    """Ensure deprecation warnings are filtered even under -Werror."""
    warnings.filterwarnings(
        "ignore",
        message="GasSpecies is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
