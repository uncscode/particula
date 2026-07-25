"""Run the supported bounded CPU-only nucleation example.

The public ``Nucleation`` runnable supports one legacy ``Aerosol`` box with
fixed particle slots. It transfers partitioning gas to particles in place;
zero duration or zero rate paths are no-ops.
"""

import numpy as np
from particula.aerosol import Aerosol
from particula.dynamics import (
    ActivationNucleationBuilder,
    Nucleation,
    NucleationCommitConfig,
    NucleationSourceConfig,
)
from particula.gas import EnvironmentData
from particula.gas.atmosphere import Atmosphere
from particula.gas.gas_data import GasData
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import MassBasedMovingBin
from particula.particles.exhaustion import ExhaustionControls
from particula.particles.particle_data import ParticleData
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume


def _total_mass(particles: ParticleData, gas: GasData) -> np.ndarray:
    """Return concentration-weighted particle plus gas mass by species."""
    return np.sum(
        particles.masses * particles.concentration[..., None], axis=1
    ) + (gas.concentration)


def run_example() -> Aerosol:
    """Create a one-box aerosol and apply a public CPU nucleation step."""
    particles = ParticleData(
        masses=np.zeros((1, 3, 1), dtype=np.float64),
        concentration=np.zeros((1, 3), dtype=np.float64),
        charge=np.zeros((1, 3), dtype=np.float64),
        density=np.array([1000.0], dtype=np.float64),
        volume=np.array([1.0], dtype=np.float64),
    )
    gas = GasData(
        name=["precursor"],
        molar_mass=np.array([0.1], dtype=np.float64),
        concentration=np.array([[1.0e-12]], dtype=np.float64),
        partitioning=np.array([True]),
    )
    gas_only = GasData(
        name=["inert"],
        molar_mass=np.array([0.04], dtype=np.float64),
        concentration=np.array([[2.0e-6]], dtype=np.float64),
        partitioning=np.array([False]),
    )
    particle_facade = ParticleRepresentation.from_data(
        particles,
        strategy=MassBasedMovingBin(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.zeros(3, dtype=np.float64),
    )
    atmosphere = Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=GasSpecies.from_data(
            gas, ConstantVaporPressureStrategy(0.0)
        ),
        gas_only_species=GasSpecies.from_data(
            gas_only, ConstantVaporPressureStrategy(0.0)
        ),
    )
    aerosol = Aerosol(atmosphere, particle_facade)
    strategy = (
        ActivationNucleationBuilder()
        .set_parameters(
            {
                "coefficient": 1.0e-2,
                "coefficient_units": "s^-1",
                "coefficient_provenance": "deterministic documentation example",
                "precursor_number_concentration_lower": 1.0,
                "precursor_number_concentration_lower_units": "1/m^3",
                "precursor_number_concentration_upper": 1.0e30,
                "precursor_number_concentration_upper_units": "1/m^3",
                "temperature_lower": 200.0,
                "temperature_lower_units": "K",
                "temperature_upper": 400.0,
                "temperature_upper_units": "K",
                "injection_composition": [1],
                "formation_diameter": 1.0,
                "formation_diameter_units": "nm",
            }
        )
        .build()
    )
    nucleation = Nucleation(
        NucleationSourceConfig(strategy=strategy, precursor_index=0),
        NucleationCommitConfig(
            maximum_slot_weight=1.0e20,
            source_charge=0.0,
            exhaustion_controls=ExhaustionControls(),
            requested_scale=np.array([1.0], dtype=np.float64),
            minimum_scale=np.array([1.0], dtype=np.float64),
            minimum_volume=np.array([1.0], dtype=np.float64),
        ),
        EnvironmentData(
            temperature=np.array([298.15], dtype=np.float64),
            pressure=np.array([101325.0], dtype=np.float64),
            saturation_ratio=np.array([[1.0]], dtype=np.float64),
        ),
    )
    before = _total_mass(particles, gas)
    result = nucleation.execute(aerosol, time_step=1.0, sub_steps=2)
    assert result is aerosol
    np.testing.assert_allclose(
        _total_mass(particles, gas), before, rtol=1e-12, atol=1e-30
    )
    assert np.any(particles.concentration > 0.0)
    return aerosol


if __name__ == "__main__":
    run_example()
    print("CPU nucleation example completed.")
