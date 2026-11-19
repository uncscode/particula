"""Wiedensohler fractions at the tabulated diameters and domain edges."""

import numpy as np
import pytest
from particula.particles import get_bipolar_charge_probability as fraction


def test_tabulated_charge_states():
    """Check the five published values at one diameter."""
    # TSI Series 3080 manual, Appendix B, fractions at 100 nm.
    np.testing.assert_allclose(
        fraction(100e-9, np.arange(-2, 3)),
        [0.0561, 0.2793, 0.4259, 0.2138, 0.0317],
        atol=5e-5,
    )


def test_broadcast_and_domain_edges():
    """Exercise diameter and state broadcasting at the fit limits."""
    d = np.array([[1e-9], [1e-6]])
    result = fraction(d, [-1, 0, 1])
    assert result.shape == (2, 3)
    np.testing.assert_allclose(
        result[0], 10.0 ** np.array([-2.3197, -0.0003, -2.3484])
    )
    assert np.all(fraction(20e-9, [-2, 2]) > 0)


@pytest.mark.parametrize(
    "d,z",
    [
        (np.nan, 0),
        (np.inf, 0),
        (0, 1),
        (19e-9, -2),
        (1.1e-6, 1),
        (5e-9, 0.5),
        (100e-9, 3),
    ],
)
def test_outside_fit(d, z):
    """Reject unsupported diameter and charge combinations."""
    with pytest.raises(ValueError):
        fraction(d, z)
