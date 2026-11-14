"""Independent checks of LF2013 fit evaluation and omitted-rate boundaries."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from particula.dynamics.ions.attachment import _tables, attachment_coefficient


def test_polynomial_matches_decimal_reference():
    """High precision avoids sharing NumPy's evaluation implementation."""
    for polarity, table in zip((-1, 1), _tables(), strict=True):
        for charge in (-5, -1, 0, 1, 5):
            row = table[charge + 100]
            radius = max(1.5e-9, float(row[-2]) * 1.01)
            with localcontext() as ctx:
                ctx.prec = 60
                x = Decimal(str(radius)).log10()
                exponent = sum(
                    Decimal(str(coefficient)) * x**i
                    for i, coefficient in enumerate(row[1:-4])
                )
                expected = float(Decimal(10) ** exponent)
            np.testing.assert_allclose(
                attachment_coefficient(radius, charge, polarity),
                expected,
                rtol=2e-7,
            )


def test_omitted_repulsive_rate_requires_explicit_bounds():
    """A 3 nm singly charged seed cannot use the omitted same-sign fit."""
    with pytest.raises(ValueError, match="omitted"):
        attachment_coefficient(1.5e-9, 1, 1)
    assert attachment_coefficient(1.5e-9, 1, 1, omitted="lower") == 0
    assert attachment_coefficient(1.5e-9, 1, 1, omitted="upper") == 1e-15
    assert attachment_coefficient(1.5e-9, 1, -1) > 0


@pytest.mark.parametrize(
    "radius,charge,polarity",
    [
        (1e-10, 0, 1),
        (2e-5, 0, 1),
        (1e-8, 0.5, 1),
        (1e-8, 101, 1),
        (1e-8, 0, 0),
        (np.nan, 0, 1),
    ],
)
def test_invalid_domain_rejects_even_with_bounds(radius, charge, polarity):
    """Bounds are not permission to extrapolate unsupported physical inputs."""
    with pytest.raises(ValueError):
        attachment_coefficient(radius, charge, polarity, omitted="lower")
