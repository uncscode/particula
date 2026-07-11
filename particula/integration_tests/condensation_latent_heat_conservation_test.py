"""Integration baseline for CPU latent-heat condensation.

This module exercises the public ``particula as par`` runnable path for a
deterministic, supersaturated, single-species water aerosol.
"""

import numpy as np
import particula as par
import pytest

LATENT_HEAT_WATER = 2.26e6  # J/kg
CONSERVATION_RTOL = 1e-12
CONSERVATION_ATOL = 1e-18


def _as_float(value: float | np.ndarray) -> float:
    """Convert a scalar-like value to ``float`` for deterministic assertions.

    Args:
        value: Scalar or size-1 array-like value to convert to a Python float.

    Returns:
        The scalar value from ``value`` as a Python float.

    Raises:
        AssertionError: If ``value`` has more than one element.
    """
    array_value = np.asarray(value, dtype=np.float64)
    if array_value.ndim == 0:
        return float(array_value)
    if array_value.size != 1:
        raise AssertionError(
            f"Expected scalar or size-1 input, got shape {array_value.shape}."
        )
    return float(array_value.reshape(()))


def _particle_water_inventory(aerosol: par.Aerosol) -> float:
    """Return the total particle-phase water inventory for the fixture.

    Args:
        aerosol: Single-species aerosol state under test.

    Returns:
        Total particle-phase water inventory as a scalar in kg/m^3.

    Raises:
        AssertionError: If the particle mass layout is not a valid
            single-species shape.
    """
    species_mass = np.asarray(
        aerosol.particles.get_species_mass(), dtype=np.float64
    )
    concentration = aerosol.particles.get_concentration()

    if species_mass.ndim == 1:
        water_mass = species_mass
    elif species_mass.ndim == 2 and species_mass.shape[1] == 1:
        water_mass = species_mass[:, 0]
    else:
        raise AssertionError(
            "Expected single-species particle mass with shape "
            f"(n_particles,) or (n_particles, 1), got {species_mass.shape}."
        )

    return float(np.sum(water_mass * concentration, dtype=np.float64))


def _build_test_aerosol() -> tuple[par.Aerosol, float]:
    """Build a supersaturated single-species water aerosol state.

    Returns:
        A tuple containing the configured aerosol and the corresponding water
        saturation concentration in kg/m^3.
    """
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
    """Build the public latent-heat condensation runnable and strategy.

    Returns:
        A tuple containing the public ``MassCondensation`` runnable and its
        ``CondensationLatentHeat`` strategy.
    """
    latent_heat_strategy = par.gas.LatentHeatFactory().get_strategy(
        "constant",
        {
            "latent_heat_ref": LATENT_HEAT_WATER,
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
    """Verify the CPU fixture starts above saturation.

    The precondition prevents this integration baseline from silently becoming
    a no-op transfer case.
    """
    aerosol, saturation_concentration = _build_test_aerosol()

    gas_concentration = _as_float(
        aerosol.atmosphere.partitioning_species.get_concentration()
    )

    assert np.isfinite(saturation_concentration)
    assert np.isfinite(gas_concentration)
    assert gas_concentration > saturation_concentration


def test_as_float_rejects_multi_element_inputs() -> None:
    """Regression: helper rejects multi-element fixture/API drift."""
    with pytest.raises(AssertionError, match="Expected scalar or size-1 input"):
        _as_float(np.array([1.0, 2.0], dtype=np.float64))


@pytest.mark.parametrize(
    "mass_shape",
    [
        (4,),
        (4, 1),
    ],
)
def test_particle_water_inventory_accepts_valid_single_species_shapes(
    mass_shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: helper accepts valid single-species mass layouts."""
    aerosol, _ = _build_test_aerosol()
    species_mass = np.asarray(
        aerosol.particles.get_species_mass(), dtype=np.float64
    )
    reshaped_mass = species_mass.reshape(mass_shape)
    monkeypatch.setattr(
        aerosol.particles,
        "get_species_mass",
        lambda: reshaped_mass,
    )

    expected = float(
        np.sum(
            species_mass.reshape(-1) * aerosol.particles.get_concentration(),
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(_particle_water_inventory(aerosol), expected)


def test_particle_water_inventory_rejects_multi_species_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: helper fails loudly for unexpected multi-species mass."""
    aerosol, _ = _build_test_aerosol()
    species_mass = np.asarray(
        aerosol.particles.get_species_mass(), dtype=np.float64
    )
    monkeypatch.setattr(
        aerosol.particles,
        "get_species_mass",
        lambda: np.column_stack(
            [species_mass.reshape(-1), species_mass.reshape(-1)]
        ),
    )

    with pytest.raises(
        AssertionError,
        match="Expected single-species particle mass",
    ):
        _particle_water_inventory(aerosol)


def test_condensation_latent_heat_fixture_executes_via_mass_condensation() -> (
    None
):
    """Verify MassCondensation transfers water through the CPU path.

    The test confirms that the public latent-heat runnable produces a finite,
    nonzero gas-to-particle transfer, conserves total water inventory, and
    records the final-step latent-heat bookkeeping over a short fixed loop.
    """
    aerosol, _ = _build_test_aerosol()
    condensation, condensation_strategy = _build_condensation()

    initial_particle_water = _particle_water_inventory(aerosol)
    initial_gas_water = _as_float(
        aerosol.atmosphere.partitioning_species.get_concentration()
    )
    initial_total_water = initial_particle_water + initial_gas_water

    current = aerosol
    for _ in range(4):
        current = condensation.execute(current, time_step=0.1, sub_steps=1)

    pre_final_particle_water = _particle_water_inventory(current)
    pre_final_gas_water = _as_float(
        current.atmosphere.partitioning_species.get_concentration()
    )
    current = condensation.execute(current, time_step=0.1, sub_steps=1)

    final_particle_water = _particle_water_inventory(current)
    final_gas_water = _as_float(
        current.atmosphere.partitioning_species.get_concentration()
    )
    final_total_water = final_particle_water + final_gas_water
    final_step_particle_gain = final_particle_water - pre_final_particle_water
    final_step_gas_loss = pre_final_gas_water - final_gas_water
    expected_final_step_energy = final_step_particle_gain * LATENT_HEAT_WATER

    assert np.isfinite(initial_particle_water)
    assert np.isfinite(initial_gas_water)
    assert np.isfinite(pre_final_particle_water)
    assert np.isfinite(pre_final_gas_water)
    assert np.isfinite(final_particle_water)
    assert np.isfinite(final_gas_water)
    assert np.isfinite(condensation_strategy.last_latent_heat_energy)
    assert condensation_strategy.last_latent_heat_energy > 0.0

    assert final_particle_water > initial_particle_water
    assert final_gas_water < initial_gas_water

    np.testing.assert_allclose(
        initial_total_water,
        final_total_water,
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_ATOL,
    )

    assert final_step_particle_gain > 0.0
    np.testing.assert_allclose(
        final_step_particle_gain,
        final_step_gas_loss,
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_ATOL,
    )
    np.testing.assert_allclose(
        condensation_strategy.last_latent_heat_energy,
        expected_final_step_energy,
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_ATOL,
    )
