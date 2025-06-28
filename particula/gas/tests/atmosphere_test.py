"""Testing Atmosphere and GasSpecies interaction."""

# pylint: disable=R0801,W0621

import numpy as np
import pytest

from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)

# ------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------


@pytest.fixture()
def vapor_pressure_strategy():
    """Constant vapor pressure strategy fixture."""
    return ConstantVaporPressureStrategy(vapor_pressure=np.array([101325]))


@pytest.fixture()
def oxygen(vapor_pressure_strategy):
    """O₂ GasSpecies fixture."""
    return GasSpecies(
        name="Oxygen",
        molar_mass=0.032,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=True,
        concentration=0.21,
    )


@pytest.fixture()
def hydrogen(vapor_pressure_strategy):
    """H₂ GasSpecies fixture."""
    return GasSpecies(
        name="Hydrogen",
        molar_mass=0.002,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=False,
        concentration=0.79,
    )


@pytest.fixture()
def atmosphere(oxygen, hydrogen):
    """Atmosphere fixture with O₂ and H₂."""
    return Atmosphere(
        temperature=298.15,
        total_pressure=101325,
        partitioning_species=oxygen,
        gas_only_species=hydrogen,
    )


def test_atmosphere_initialization_values(atmosphere):
    """Check stored temperature and pressure values."""
    assert atmosphere.temperature == 298.15
    assert atmosphere.total_pressure == 101325


def test_partitioning_species_initial_length(atmosphere):
    """Initially one partitioning species is present."""
    assert len(atmosphere) == 2


def test_add_partitioning_species_updates_length(
    atmosphere, vapor_pressure_strategy
):
    """Adding another partitioning species increases the length."""
    nitrogen = GasSpecies(
        name="Nitrogen",
        molar_mass=0.028,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=True,
        concentration=0.0,
    )
    atmosphere.add_partitioning_species(nitrogen)
    assert len(atmosphere.partitioning_species) == 2


def test_total_species_count(atmosphere, vapor_pressure_strategy):
    """Total species count equals partitioning + gas-only."""
    nitrogen = GasSpecies(
        name="Nitrogen",
        molar_mass=0.028,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=True,
        concentration=0.0,
    )
    atmosphere.add_partitioning_species(nitrogen)
    # 2 partitioning + 1 gas-only
    assert len(atmosphere) == 3
