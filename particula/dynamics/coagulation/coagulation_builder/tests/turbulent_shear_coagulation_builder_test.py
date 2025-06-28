"""Tests for the TurbulentShearCoagulationBuilder class."""

# pylint: disable=duplicate-code

import pytest

# pylint: disable=line-too-long
from particula.dynamics import (
    TurbulentShearCoagulationBuilder,
    TurbulentShearCoagulationStrategy,
)


def test_build_with_valid_parameters():
    """Test that building with valid parameters returns a
    TurbulentShearCoagulationStrategy.
    """
    builder = TurbulentShearCoagulationBuilder()
    builder.set_distribution_type("discrete")
    builder.set_turbulent_dissipation(
        1e-4, turbulent_dissipation_units="m^2/s^3"
    )
    builder.set_fluid_density(1.2, fluid_density_units="kg/m^3")
    strategy = builder.build()
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)


def test_build_missing_required_parameters():
    """Test that building without required parameters raises a ValueError."""
    builder = TurbulentShearCoagulationBuilder()

    with pytest.raises(ValueError):
        builder.build()

    builder.set_distribution_type("discrete")
    with pytest.raises(ValueError):
        builder.build()

    builder.set_turbulent_dissipation(
        1e-4, turbulent_dissipation_units="m^2/s^3"
    )
    with pytest.raises(ValueError):
        builder.build()

    builder.set_fluid_density(1.2, fluid_density_units="kg/m^3")
    # With all required parameters set, it should now succeed:
    strategy = builder.build()
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)
