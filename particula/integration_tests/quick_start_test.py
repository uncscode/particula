"""Exercise native quick-start workflows without repository documentation files."""

import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)


def test_quick_start_runs():
    """README construction and composed dynamics retain native conservation."""
    particles = par.particles.create_resolved_particle_data(
        masses=[1e-18, 2e-18],
        density=1000.0,
        concentration=[1.0, 1.0],
        volume=1e-6,  # counts and m³
    )
    gas = par.gas.GasData(
        name=["organic"],
        molar_mass=np.array([0.180]),
        concentration=np.array([[1e-6]]),  # kg/m³
        partitioning=np.array([True]),
    )
    environment = par.gas.EnvironmentData(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[1.0]],
    )
    aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)

    strategy = par.dynamics.CondensationIsothermal(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(density=1000.0),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e-3),
        species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
    )
    condensation = par.dynamics.MassCondensation(strategy)
    coagulation = par.dynamics.Coagulation(
        par.dynamics.BrownianCoagulationStrategy("particle_resolved")
    )
    aerosol = (condensation | coagulation).execute(
        aerosol, time_step=10.0, sub_steps=10
    )
    assert isinstance(aerosol, par.Aerosol)
    assert aerosol.particles is particles
    assert aerosol.gas is gas
    assert aerosol.environment is environment
    assert aerosol.particles.distribution_type == "particle_resolved"
    assert aerosol.gas.name == ["organic"]
    assert 0 <= aerosol.gas.concentration[0, 0] < 1e-6
    inventory = (
        np.sum(
            aerosol.particles.masses
            * aerosol.particles.concentration[..., None]
        )
        + aerosol.gas.concentration[0, 0] * 1e-6
    )
    np.testing.assert_allclose(inventory, 1e-12 + 3e-18, rtol=1e-12, atol=1e-30)
    assert np.all(np.isfinite(aerosol.particles.masses))
    assert np.all(aerosol.particles.masses >= 0)


def test_documentation_quick_start_runs():
    """The published PMF example preserves number and material inventory."""
    particles = par.particles.create_lognormal_binned_particle_data(
        distribution_type="discrete",
        radius_bins=np.logspace(-9, -4, 250),
        mode=[100e-9],
        geometric_standard_deviation=[1.2],
        number_concentration=[1e8],
        density=1000.0,
    )
    gas = par.gas.GasData(
        name=["organic"],
        molar_mass=np.array([0.180]),
        concentration=np.array([[100.0]]),  # kg/m³
        partitioning=np.array([True]),
    )
    environment = par.gas.EnvironmentData(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[1.0]],
    )
    aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)
    initial_particle_mass = particles.species_mass_inventory.copy()

    strategy = par.dynamics.CondensationIsothermal(
        molar_mass=gas.molar_mass.copy(),
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
        activity_strategy=par.particles.ActivityIdealMass(),
        surface_strategy=par.particles.SurfaceStrategyVolume(
            surface_tension=0.072, density=1000.0
        ),
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(100.0),
        species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
    )
    process = par.dynamics.MassCondensation(strategy)
    result = process.execute(aerosol, time_step=10.0)
    assert result is aerosol
    particles = aerosol.particles
    assert particles.distribution_type == "discrete"
    np.testing.assert_allclose(particles.number_density, [1e8], rtol=1e-12)
    np.testing.assert_array_equal(particles.volume, [1.0])
    assert 0 <= aerosol.gas.concentration[0, 0] < 100.0
    final_particle_mass = np.sum(
        particles.masses * particles.concentration[..., None]
    )
    initial_particle_mass = initial_particle_mass.item()
    assert final_particle_mass > initial_particle_mass
    np.testing.assert_allclose(
        final_particle_mass + aerosol.gas.concentration[0, 0],
        initial_particle_mass + 100.0,
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.all(np.isfinite(particles.masses))
    assert np.all(particles.masses >= 0)
