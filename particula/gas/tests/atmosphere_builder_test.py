"""Tests for the AtmosphereBuilder class."""

import pytest

from particula.gas.atmosphere_builders import AtmosphereBuilder
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_builder_with_species():
    """Test building a Gas object with the GasBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(101325)

    oxygen = GasSpecies(
        name="Oxygen",
        molar_mass=0.032,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=True,
        concentration=0.21,
    )

    nitrogen = GasSpecies(
        name="Nitrogen",
        molar_mass=0.028,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=False,
        concentration=0.79,
    )

    atmo = (
        AtmosphereBuilder()
        .set_temperature(298.15, "K")
        .set_pressure(101325, "Pa")
        .set_more_partitioning_species(oxygen)
        .set_more_gas_only_species(nitrogen)
        .build()
    )

    assert atmo.temperature == 298.15
    assert atmo.total_pressure == 101325
    assert atmo.partitioning_species.get_name() == "Oxygen"
    assert atmo.partitioning_species.get_molar_mass() == 0.032
    assert atmo.gas_only_species.get_name() == "Nitrogen"
    assert atmo.gas_only_species.get_molar_mass() == 0.028
    # total species count (1+1)
    assert len(atmo) == 2


def test_gas_builder_without_species_raises_error():
    """Test that building a Gas object without any species raises an error."""
    builder = AtmosphereBuilder().set_temperature(298.15, "K")
    # No pressure added – dataclass should fail
    with pytest.raises(ValueError):
        builder.build()
