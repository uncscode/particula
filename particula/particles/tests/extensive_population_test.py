"""Independent raw, intensive and extensive population checks for 0.4."""

from dataclasses import fields, replace

import numpy as np
import numpy.testing as npt
import pytest
from particula.particles import ParticleData, ParticleDataBuilder
from particula.particles._population import (
    number_density_weights,
    write_binned_density,
)


@pytest.mark.parametrize(
    "kind", ["discrete", "particle_resolved", "continuous_pdf"]
)
def test_equal_densities_across_volumes(kind):
    """Integrate each species and node in three represented volumes."""
    volumes = np.array([0.25, 1.0, 4.0])
    pdf = kind == "continuous_pdf"
    ordinates = (
        np.array([2.0e9, 4.0e9, 2.0e9]) if pdf else np.array([1.0, 6.0, 2.0])
    )
    raw = volumes[:, None] * ordinates
    masses = np.array([[1.0, 4.0], [2.0, 1.0], [4.0, 3.0]]) * 1.0e-18
    particles = ParticleData(
        masses=np.broadcast_to(masses, (3, 3, 2)).copy(),
        population=raw,
        charge=np.zeros((3, 3)),
        density=np.ones(2),
        volume=volumes,
        distribution_type=kind,
        radius_grid=np.array([1.0, 2.0, 4.0]) * 1e-9 if pdf else None,
    )
    assert particles.population is raw
    npt.assert_allclose(
        number_density_weights(particles),
        [[1.0, 6.0, 2.0]] * 3,
        rtol=1e-14,
        atol=0,
    )
    npt.assert_allclose(particles.number_density, [9.0, 9.0, 9.0], rtol=1e-14)
    expected = np.array([[21.0, 16.0]] * 3) * 1e-18
    npt.assert_allclose(
        particles.species_mass_density, expected, rtol=1e-14, atol=0
    )
    npt.assert_allclose(
        particles.species_mass_inventory,
        expected * volumes[:, None],
        rtol=1e-14,
        atol=0,
    )
    assert not np.shares_memory(particles.slot_concentration_density, raw)
    assert replace(particles).population is raw
    assert not np.shares_memory(particles.copy().population, raw)
    if not pdf:
        npt.assert_array_equal(raw.sum(axis=1), [2.25, 9.0, 36.0])
    if kind != "particle_resolved":
        with pytest.raises(ValueError, match="population"):
            particles.concentration[:] = 0
        before = raw.copy()
        with pytest.raises(ValueError, match="finite"):
            write_binned_density(
                particles, np.full(raw.shape, np.finfo(float).max)
            )
        npt.assert_array_equal(raw, before)
        write_binned_density(particles, np.ones(raw.shape))
        npt.assert_array_equal(
            raw, np.broadcast_to(volumes[:, None], raw.shape)
        )


def test_legacy_constructor_and_writable_alias():
    """Legacy writes retain buffer identity in compatible boxes."""
    raw = np.array([[2.0]])
    args = dict(
        masses=np.ones((1, 1, 1)),
        charge=np.zeros((1, 1)),
        density=np.ones(1),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    data = ParticleData(concentration=raw, **args)
    assert "population" in ParticleData.concentration.fget.__deprecated__
    assert data.population is data.concentration is raw
    data.concentration += 3
    npt.assert_array_equal(raw, [[5.0]])
    new = np.array([[8.0]])
    data.concentration = new
    assert data.population is new
    assert "concentration" not in [f.name for f in fields(data)]
    with pytest.raises(ValueError, match="either"):
        ParticleData(population=raw, concentration=new, **args)
    args["volume"] = np.array([0.25])
    with pytest.raises(ValueError, match="legacy concentration"):
        ParticleData(concentration=raw, **args)


def test_measured_density_and_ambiguous_builder_inputs():
    """Measure in cm^-3 and convert to represented counts only once."""
    builder = (
        ParticleDataBuilder()
        .set_masses(np.ones((3, 1, 1)))
        .set_density(np.ones(1))
        .set_distribution_type("discrete")
    )
    volumes = np.array([0.25, 1.0, 4.0])
    data = (
        builder.set_concentration(np.array([2.0]), "1/cm^3")
        .set_volume(volumes)
        .build()
    )
    npt.assert_allclose(data.population[:, 0], [5e5, 2e6, 8e6], rtol=1e-14)
    builder.set_population(np.array([2.0]))
    with pytest.raises(ValueError, match="either"):
        builder.build()


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
@pytest.mark.parametrize("volume", [0.0, -1.0, np.inf, np.nan])
def test_direct_tagged_population_rejects_invalid_volume(kind, volume):
    """Explicit storage requires physically meaningful represented air volume."""
    with pytest.raises(ValueError, match="positive and finite"):
        ParticleData(
            masses=np.ones((1, 2, 1)),
            population=np.ones((1, 2)),
            charge=np.zeros((1, 2)),
            density=np.ones(1),
            volume=np.array([volume]),
            distribution_type=kind,
            radius_grid=np.array([1.0, 2.0])
            if kind == "continuous_pdf"
            else None,
        )
