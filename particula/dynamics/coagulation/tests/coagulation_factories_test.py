"""Simple tests for CoagulationFactory, verifying that known strategy types
produce the expected strategy classes, and that invalid types raise
the appropriate ValueError.
"""

import pytest

from particula.dynamics import (
    BrownianCoagulationStrategy,
    ChargedCoagulationStrategy,
    CoagulationFactory,
    CombineCoagulationStrategy,
    HardSphereKernelStrategy,
    TurbulentDNSCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
)


def test_brownian_coagulation():
    """Test that the BrownianCoagulationStrategy is created correctly."""
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "brownian", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, BrownianCoagulationStrategy)


def test_charged_coagulation():
    """Test that the ChargedCoagulationStrategy is created correctly."""
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "charged",
        {
            "distribution_type": "discrete",
            "charged_kernel_strategy": HardSphereKernelStrategy(),
        },
    )
    assert isinstance(strategy, ChargedCoagulationStrategy)


def test_turbulent_shear_coagulation():
    """Test that the TurbulentShearCoagulationStrategy is created correctly."""
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "turbulent_shear",
        {
            "distribution_type": "discrete",
            "turbulent_dissipation": 1e-4,
            "turbulent_dissipation_units": "m^2/s^3",
            "fluid_density": 1.2,
            "fluid_density_units": "kg/m^3",
        },
    )
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)


def test_turbulent_dns_coagulation():
    """Test that the TurbulentDNSCoagulationStrategy is created correctly."""
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "turbulent_dns",
        {
            "distribution_type": "discrete",
            "turbulent_dissipation": 1e-3,
            "turbulent_dissipation_units": "m^2/s^3",
            "fluid_density": 1.5,
            "fluid_density_units": "kg/m^3",
            "reynolds_lambda": 100,
            "reynolds_lambda_units": "dimensionless",
            "relative_velocity": 0.5,
            "relative_velocity_units": "m/s",
        },
    )
    assert isinstance(strategy, TurbulentDNSCoagulationStrategy)


def test_combine_coagulation():
    """Test that the CombineCoagulationStrategy is created correctly."""
    factory = CoagulationFactory()

    strategy = factory.get_strategy(
        "combine",
        {
            "strategies": [
                BrownianCoagulationStrategy(distribution_type="discrete"),
                TurbulentShearCoagulationStrategy(
                    distribution_type="discrete",
                    turbulent_dissipation=1e-5,
                    fluid_density=1.1,
                ),
            ],
        },
    )
    assert isinstance(strategy, CombineCoagulationStrategy)


def test_invalid_coagulation():
    """Test that an invalid coagulation strategy raises a ValueError."""
    factory = CoagulationFactory()
    with pytest.raises(ValueError):
        factory.get_strategy("nonexistent", {})
