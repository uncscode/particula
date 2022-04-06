""" testing the distribution discretization
"""

import numpy as np
import pytest
from particula.util.distribution_discretization import discretize


def test_discretize():
    """ testing the discretization utility
    """

    spans = np.linspace(1, 1000, 1000)
    sigma = 1.25
    modes = 100

    assert discretize(
        interval=spans, disttype="lognormal", gsigma=sigma, mode=modes
    ).size == spans.size

    assert np.trapz(discretize(
        interval=spans, disttype="lognormal", gsigma=sigma, mode=modes
    ), spans) == pytest.approx(1, rel=1e-5)

    with pytest.raises(ValueError):
        discretize(
            interval=spans, disttype="linear", gsigma=sigma, mode=modes,
        )


def test_multi_discretize():
    """ testing different modes
    """

    spans = np.linspace(1, 1000, 1000)
    sigma = 1.25
    modes = [100, 200]

    assert discretize(
        interval=spans, disttype="lognormal", gsigma=sigma, mode=modes
    ).size == spans.size

    assert np.trapz(discretize(
        interval=spans, disttype="lognormal", gsigma=sigma, mode=modes
    ), spans) == pytest.approx(1, rel=1e-5)

    with pytest.raises(ValueError):
        discretize(
            interval=spans, disttype="linear", gsigma=sigma, mode=modes,
        )
