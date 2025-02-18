"""
Tests for the TurbulentShearCoagulationBuilder class.
"""

import pytest
from particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder import (
    TurbulentShearCoagulationBuilder,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    TurbulentShearCoagulationStrategy,
)

def test_build_with_valid_parameters():
    """
    Test that building with valid parameters returns a TurbulentShearCoagulationStrategy.
    """
    builder = TurbulentShearCoagulationBuilder()
    builder.set_distribution_type("discrete")
    builder.set_turbulent_dissipation(1e-4)
    builder.set_fluid_density(1.2)
    strategy = builder.build()
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)

def test_build_missing_required_parameters():
    """
    Test that building without required parameters raises a ValueError.
    """
    builder = TurbulentShearCoagulationBuilder()

    with pytest.raises(ValueError):
        builder.build()

    builder.set_distribution_type("discrete")
    with pytest.raises(ValueError):
        builder.build()

    builder.set_turbulent_dissipation(1e-4)
    with pytest.raises(ValueError):
        builder.build()

    builder.set_fluid_density(1.2)
    # With all required parameters set, it should now succeed:
    strategy = builder.build()
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)
