"""Test for CondensationIsothermalBuilder."""

import pytest

from particula.dynamics import (
    CondensationIsothermal,
    CondensationIsothermalBuilder,
)


def test_build_with_valid_parameters():
    """Test for building CondensationIsothermal with valid parameters."""
    builder = CondensationIsothermalBuilder()
    builder.set_molar_mass(0.018, "kg/mol")
    builder.set_diffusion_coefficient(2e-5, "m^2/s")
    builder.set_accommodation_coefficient(1.0)
    strategy = builder.build()
    assert isinstance(strategy, CondensationIsothermal)


def test_build_missing_required_parameters():
    """Test for building CondensationIsothermal without required parameters."""
    builder = CondensationIsothermalBuilder()
    with pytest.raises(ValueError):
        builder.build()
