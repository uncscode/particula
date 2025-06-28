"""
Test cases for the CondensationFactory and CondensationIsothermal classes.
"""

import pytest

from particula.dynamics import (
    CondensationFactory,
    CondensationIsothermal,
)


def test_isothermal_condensation():
    """
    Test the creation of an isothermal condensation strategy.
    """
    factory = CondensationFactory()
    strategy = factory.get_strategy(
        "isothermal",
        {
            "molar_mass": 0.018,
            "molar_mass_units": "kg/mol",
            "diffusion_coefficient": 2e-5,
            "diffusion_coefficient_units": "m^2/s",
            "accommodation_coefficient": 1.0,
        },
    )
    assert isinstance(strategy, CondensationIsothermal)


def test_invalid_condensation_strategy():
    """
    Test that an invalid condensation strategy raises a ValueError.
    """
    factory = CondensationFactory()
    with pytest.raises(ValueError):
        factory.get_strategy("nonexistent", {})
