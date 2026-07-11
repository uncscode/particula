"""Integration baseline for CPU latent-heat condensation."""

import numpy as np
import particula as par


def _as_float(value: float | np.ndarray) -> float:
    """Convert scalar-like values to float for deterministic assertions."""
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _build_test_aerosol() -> tuple[par.Aerosol, float]:
    """Build a supersaturated single-species water aerosol state."""
    molar_mass_water = 18.015e-3  # kg/mol
    temperature = 298.15  # K
    vapor_pressure_water = par.gas.VaporPressureFactory().get_strategy(
        "water_buck"
    )
    saturation_concentration = vapor_pressure_water.saturation_concentration(
        molar_mass=molar_mass_water,
        temperature=temperature,
    )
    gas_species = (
        par.gas.GasSpeciesBuilder()
        .set_molar_mass(molar_mass_water, "kg/mol")
        .set_vapor_pressure_strategy(vapor_pressure_water)
        .set_concentration(saturation_concentration * 1.03, "kg/m^3")
        .set_name("H2O")
        .set_partitioning(True)
        .build()
    )
    atmosphere = (
        par.gas.AtmosphereBuilder()
        .set_more_partitioning_species(gas_species)
        .set_temperature(temperature, "K")
        .set_pressure(101325.0, "Pa")
        .build()
    )

    particle_radii = np.array([30e-9, 45e-9, 60e-9, 90e-9], dtype=np.float64)
    density = 1000.0  # kg/m^3
    particle_mass = (4.0 / 3.0) * np.pi * particle_radii**3 * density
    particles = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(par.particles.ActivityIdealMass())
        .set_surface_strategy(par.particles.SurfaceStrategyVolume())
        .set_mass(particle_mass.reshape(-1, 1), "kg")
        .set_density(np.array([density], dtype=np.float64), "kg/m^3")
        .set_charge(np.zeros_like(particle_radii, dtype=np.float64))
        .set_volume(1.0e-6, "m^3")
        .build()
    )
    aerosol = par.Aerosol(atmosphere=atmosphere, particles=particles)
    return aerosol, float(saturation_concentration)


def _build_condensation() -> tuple[
    par.dynamics.MassCondensation,
    par.dynamics.CondensationLatentHeat,
]:
    """Build the public latent-heat condensation runnable and strategy."""
    latent_heat_strategy = par.gas.LatentHeatFactory().get_strategy(
        "constant",
        {
            "latent_heat_ref": 2.26e6,
            "latent_heat_ref_units": "J/kg",
        },
    )
    condensation_strategy = par.dynamics.CondensationLatentHeat(
        molar_mass=np.array([18.015e-3], dtype=np.float64),
        diffusion_coefficient=2e-5,
        accommodation_coefficient=1.0,
        latent_heat_strategy=latent_heat_strategy,
        update_gases=True,
    )
    condensation = par.dynamics.MassCondensation(
        condensation_strategy=condensation_strategy
    )
    return condensation, condensation_strategy


def test_condensation_latent_heat_fixture_starts_supersaturated() -> None:
    """The CPU fixture starts above saturation to avoid a no-op baseline."""
    aerosol, saturation_concentration = _build_test_aerosol()

    gas_concentration = _as_float(
        aerosol.atmosphere.partitioning_species.get_concentration()
    )

    assert np.isfinite(saturation_concentration)
    assert np.isfinite(gas_concentration)
    assert gas_concentration > saturation_concentration


def test_condensation_latent_heat_fixture_executes_via_mass_condensation() -> (
    None
):
    """MassCondensation transfers water through the CPU latent-heat path."""
    aerosol, _ = _build_test_aerosol()
    condensation, condensation_strategy = _build_condensation()

    initial_gas_concentration = _as_float(
        aerosol.atmosphere.partitioning_species.get_concentration()
    )
    initial_particle_mass_concentration = _as_float(
        aerosol.particles.get_mass_concentration()
    )

    current = aerosol
    for _ in range(5):
        current = condensation.execute(current, time_step=0.1, sub_steps=1)

    final_gas_concentration = _as_float(
        current.atmosphere.partitioning_species.get_concentration()
    )
    final_particle_mass_concentration = _as_float(
        current.particles.get_mass_concentration()
    )

    assert np.isfinite(initial_gas_concentration)
    assert np.isfinite(final_gas_concentration)
    assert np.isfinite(initial_particle_mass_concentration)
    assert np.isfinite(final_particle_mass_concentration)
    assert np.isfinite(condensation_strategy.last_latent_heat_energy)

    assert (
        final_particle_mass_concentration > initial_particle_mass_concentration
    )
    assert final_gas_concentration < initial_gas_concentration
    assert (
        final_particle_mass_concentration - initial_particle_mass_concentration
        > 0.0
    )
    assert initial_gas_concentration - final_gas_concentration > 0.0
