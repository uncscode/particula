"""Tests for the isothermal condensation builder."""

import pytest

from particula.dynamics import (
    CondensationIsothermalBuilder,
    CondensationIsothermal,
)


def test_build_with_valid_parameters():
    """Builder returns a valid strategy when all parameters set."""
    builder = CondensationIsothermalBuilder()
    builder.set_molar_mass(0.018, "kg/mol")
    builder.set_diffusion_coefficient(2e-5, "m^2/s")
    builder.set_accommodation_coefficient(1.0)
    strategy = builder.build()
    assert isinstance(strategy, CondensationIsothermal)


def test_build_missing_required_parameters():
    """Builder raises an error when required parameters are missing."""
    builder = CondensationIsothermalBuilder()
    with pytest.raises(ValueError):
        builder.build()
