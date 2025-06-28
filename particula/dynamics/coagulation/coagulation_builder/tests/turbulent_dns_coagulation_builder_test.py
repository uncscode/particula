"""Tests for the TurbulentDNSCoagulationBuilder class."""

# pylint: disable=duplicate-code

import pytest

from particula.dynamics import (
    TurbulentDNSCoagulationBuilder,
    TurbulentDNSCoagulationStrategy,
)


def test_build_with_valid_parameters():
    """Test that building with valid parameters returns a
    TurbulentDNSCoagulationStrategy.
    """
    builder = TurbulentDNSCoagulationBuilder()
    builder.set_distribution_type(
        "discrete", distribution_type_units="dimensionless"
    )
    builder.set_turbulent_dissipation(
        1e-4, turbulent_dissipation_units="m^2/s^3"
    )
    builder.set_fluid_density(1.2, fluid_density_units="kg/m^3")
    builder.set_reynolds_lambda(100, reynolds_lambda_units="dimensionless")
    builder.set_relative_velocity(0.5, "m/s")
    strategy = builder.build()
    assert isinstance(strategy, TurbulentDNSCoagulationStrategy)


def test_build_missing_required_parameters():
    """Test that building without required parameters raises a ValueError."""
    builder = TurbulentDNSCoagulationBuilder()

    with pytest.raises(ValueError):
        builder.build()

    builder.set_distribution_type(
        "discrete", distribution_type_units="dimensionless"
    )
    with pytest.raises(ValueError):
        builder.build()

    builder.set_turbulent_dissipation(
        1e-4, turbulent_dissipation_units="m^2/s^3"
    )
    with pytest.raises(ValueError):
        builder.build()

    builder.set_fluid_density(1.2, fluid_density_units="kg/m^3")
    with pytest.raises(ValueError):
        builder.build()

    builder.set_reynolds_lambda(100, reynolds_lambda_units="dimensionless")
    with pytest.raises(ValueError):
        builder.build()

    builder.set_relative_velocity(0.5, "m/s")

    # With all required parameters present, it should now succeed:
    strategy = builder.build()
    assert isinstance(strategy, TurbulentDNSCoagulationStrategy)
