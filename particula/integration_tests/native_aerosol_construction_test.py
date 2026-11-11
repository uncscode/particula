"""Data-only native construction through replacement and transfer rejection."""

import numpy as np
import pytest
from particula import AerosolBuilder
from particula.gas import EnvironmentData, GasDataBuilder
from particula.gpu import conversion
from particula.particles import (
    create_lognormal_binned_particle_data,
    create_lognormal_resolved_particle_data,
    create_particle_data_from_radii,
    create_resolved_particle_data,
)
from particula.tests.aerosol_replacement_test import assert_unchanged, snapshot


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("mask", [[True, False, True], [False, False, False]])
def test_native_construction_replacement_and_accounting(
    volume, mask, monkeypatch
):
    """Preserve independent per-species quantities and explicit boundaries."""
    from particula._legacy_aerosol import Aerosol as LegacyAerosol
    from particula._legacy_aerosol import AerosolBuilder as LegacyBuilder
    from particula.gas.atmosphere import Atmosphere
    from particula.particles.representation import ParticleRepresentation

    def no_legacy(*args, **kwargs):
        raise AssertionError("native construction must not invoke a facade")

    for cls in (
        LegacyAerosol,
        LegacyBuilder,
        Atmosphere,
        ParticleRepresentation,
    ):
        monkeypatch.setattr(cls, "__init__", no_legacy)
    particles = create_resolved_particle_data(
        masses=[[0.5e-18, 1.5e-18], [2e-18, 3e-18], [0.0, 0.0]],
        density=[1000.0, 1200.0],
        concentration=[8.0, 0.5, 0.0],
        volume=volume,
    )
    gas = (
        GasDataBuilder()
        .set_names(["water", "inert", "organic"])
        .set_molar_mass([0.018, 0.028, 0.1])
        .set_partitioning(mask)
        .set_concentration([1e-8, 7e-8, 3e-8])
        .build()
    )
    environment = EnvironmentData([298.15], [101325.0], [[0.7, 7.0, 1.3]])
    aerosol = (
        AerosolBuilder()
        .set_particles(particles)
        .set_gas(gas)
        .set_environment(environment)
        .build()
    )
    np.testing.assert_allclose(
        particles.species_mass_inventory,
        [[5e-18, 13.5e-18]],
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        particles.species_mass_density,
        [[5e-18 / volume, 13.5e-18 / volume]],
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        gas.concentration * volume,
        [[1e-8 * volume, 7e-8 * volume, 3e-8 * volume]],
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_array_equal(
        aerosol.environment.saturation_ratio, [[0.7, 7.0, 1.3]]
    )
    assert aerosol.particles is particles and aerosol.gas is gas
    replacement = particles.copy()
    aerosol.particles = replacement
    assert aerosol.particles is replacement and aerosol.gas is gas
    current = (aerosol.particles, aerosol.gas, aerosol.environment)
    next_particles = particles.copy()
    next_gas = (
        GasDataBuilder()
        .set_names(["new"])
        .set_molar_mass([0.2])
        .set_partitioning([False])
        .set_concentration([9e-8])
        .build()
    )
    next_environment = EnvironmentData([300.0], [101325.0], [[2.0]])
    with pytest.raises(ValueError):
        aerosol.gas = next_gas
    with pytest.raises(ValueError):
        aerosol.environment = next_environment
    saved = snapshot((*current, next_particles, next_gas, next_environment))
    aerosol.replace_data(
        particles=next_particles, gas=next_gas, environment=next_environment
    )
    assert_unchanged(saved)
    assert aerosol.gas is next_gas and aerosol.environment is next_environment
    bad = next_environment.copy()
    bad.pressure = np.array([])
    saved = snapshot(
        (*current, next_particles, next_gas, next_environment, bad)
    )
    with pytest.raises(ValueError):
        aerosol.replace_data(particles=particles, gas=gas, environment=bad)
    assert aerosol.particles is next_particles and aerosol.gas is next_gas
    assert_unchanged(saved)
    # Tagged native data must reject at the existing explicit transfer boundary.
    monkeypatch.setattr(conversion, "_ensure_warp_available", no_legacy)
    with pytest.raises(ValueError, match="no durable Warp metadata"):
        conversion.to_warp_particle_data(aerosol.particles, device="cpu")


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_native_binned_build_and_sampled_entry(kind):
    """All initializer exports enter the same aggregate construction surface."""
    gas = (
        GasDataBuilder()
        .set_names(["inert"])
        .set_molar_mass([0.028])
        .set_partitioning([False])
        .set_concentration([0.0])
        .build()
    )
    env = EnvironmentData([298.15], [101325.0], [[0.0]])
    for particles in (
        create_particle_data_from_radii(
            radii=[1e-9, 2e-9], concentration=0.0, distribution_type=kind
        ),
        create_lognormal_binned_particle_data(distribution_type=kind),
        create_lognormal_resolved_particle_data(
            particle_resolved_count=10, volume=0.25
        ),
    ):
        aerosol = (
            AerosolBuilder()
            .set_particles(particles)
            .set_gas(gas)
            .set_environment(env)
            .build()
        )
        assert aerosol.particles is particles
        assert np.isfinite(particles.species_mass_inventory).all()
