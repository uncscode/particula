"""Temporary script to inspect mass-transfer shapes for CondensationIsothermal.

This module boots a seeded system, computes the mass-transfer rate for a single
step, and prints diagnostics about the returned array shapes for quick
investigation. It lives in .trash because it supports ad-hoc experimentation
rather than the shipped API.
"""

import numpy as np

from particula.dynamics.condensation import CondensationIsothermal
from particula.dynamics.condensation.mass_transfer_utils import (
    calc_mass_to_change,
)
from particula.gas import GasSpeciesBuilder, VaporPressureFactory
from particula.particles import (
    ActivityIdealMass,
    ParticleResolvedSpeciatedMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
    SurfaceStrategyVolume,
)

DEFAULT_MOLAR_MASS = 0.018
_VAPOR_STRATEGY = VaporPressureFactory().get_strategy(
    "constant",
    {
        "vapor_pressure": 101325.0,
        "vapor_pressure_units": "Pa",
    },
)


def create_test_system(
    n_particles: int = 220,
    seed: int = 11,
):
    """Build a deterministic particle and gas pair for quick inspection."""
    rng = np.random.default_rng(seed)
    masses = rng.lognormal(mean=-20.0, sigma=0.5, size=(n_particles, 1))
    distribution_strategy = ParticleResolvedSpeciatedMassBuilder().build()
    particle = (
        ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(distribution_strategy)
        .set_activity_strategy(ActivityIdealMass())
        .set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000.0)
        )
        .set_mass(masses, "kg")
        .set_density(np.full_like(masses, 1000.0), "kg/m^3")
        .set_charge(np.zeros_like(masses))
        .set_volume(1.0, "m^3")
        .build()
    )

    gas_species = (
        GasSpeciesBuilder()
        .set_name("water")
        .set_molar_mass(DEFAULT_MOLAR_MASS, "kg/mol")
        .set_vapor_pressure_strategy(_VAPOR_STRATEGY)
        .set_partitioning(True)
        .set_concentration(0.01, "kg/m^3")
        .build()
    )
    return particle, gas_species


particle, gas = create_test_system(220, seed=11)

strategy = CondensationIsothermal(molar_mass=DEFAULT_MOLAR_MASS)
mass_rate = strategy.mass_transfer_rate(
    particle=particle,
    gas_species=gas,
    temperature=298.0,
    pressure=101325.0,
)
mass_rate_array = np.asarray(mass_rate)
particle_mass = particle.get_species_mass()
particle_concentration = particle.get_concentration()

mass_transfer = calc_mass_to_change(
    mass_rate=mass_rate_array,
    time_step=1.0,
    particle_concentration=particle_concentration,
)
print("mass_transfer shape", mass_transfer.shape)
if mass_transfer.ndim == 2:
    first_column = mass_transfer[:, 0]
    mean_column = mass_transfer.mean(axis=1)
    print(
        "max abs diff first vs mean", np.max(np.abs(first_column - mean_column))
    )
    print("max ratio", np.max(np.abs(mean_column / (first_column + 1e-25))))
    print("mean first", np.mean(first_column))
else:
    print("1d mass_transfer")
