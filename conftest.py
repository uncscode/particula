"""Pytest bootstrap configuration.

Applied at repository root so it runs before package imports. Filters
noisy warnings that are expected during the test suite.
"""

from __future__ import annotations

import re
import warnings

import pytest


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
    # chemicals 1.5.2 still requests this pandas CSV parser option. Pandas
    # 3.1 deprecates it for removal in 4.0; property-table values are unchanged.
    warnings.filterwarnings(
        "ignore",
        message=(
            "^The 'float_precision' argument is deprecated\\. "
            "Use the default float precision instead\\.$"
        ),
        category=DeprecationWarning,
        module=r"^chemicals\.data_reader$",
        append=False,
    )


_filter_expected_warnings()


def pytest_configure() -> None:
    """Reapply filters after pytest processes command-line warnings."""
    _filter_expected_warnings()


def pytest_runtest_setup(item) -> None:
    """Ensure deprecation warnings are filtered even under -Werror."""
    _filter_expected_warnings()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items) -> None:
    """Mark explicit CUDA device parameters before marker deselection.

    Fixture parameters such as ``device='cuda:0'`` select actual execution
    targets, whereas the ``warp`` marker alone also covers Warp CPU tests.
    Do not infer a device from test names or arbitrary configuration strings.
    """
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if any(
            "device" in name
            and isinstance(value, str)
            and re.fullmatch(r"cuda(?::\d+)?", value)
            for name, value in parameters.items()
        ):
            item.add_marker(pytest.mark.cuda)
