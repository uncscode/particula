"""Independent raw, intensive and extensive population checks for 0.4."""

from dataclasses import fields, replace
from importlib.util import find_spec

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


@pytest.mark.parametrize(
    ("measurement", "units"), [(2e6, "1/m^3"), (2.0, "1/cm^3")]
)
def test_measured_density_and_ambiguous_builder_inputs(measurement, units):
    """Convert density exactly once and exercise the optional unit boundary."""
    builder = (
        ParticleDataBuilder()
        .set_masses(np.ones((3, 1, 1)))
        .set_density(np.ones(1))
        .set_distribution_type("discrete")
    )
    if units == "1/cm^3" and find_spec("pint") is None:
        with pytest.raises(ImportError, match="Install pint"):
            builder.set_concentration(np.array([measurement]), units)
        return
    volumes = np.array([0.25, 1.0, 4.0])
    data = (
        builder.set_concentration(np.array([measurement]), units)
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


@pytest.mark.parametrize("density_first", [True, False])
def test_resolved_builder_density_and_count_conflict(density_first):
    """Distinguish measured densities from legacy absolute resolved counts."""
    builder = (
        ParticleDataBuilder()
        .set_distribution_type("particle_resolved")
        .set_masses(np.ones((3, 1, 1)))
        .set_density(np.ones(1))
    )
    volumes = np.array([0.25, 1.0, 4.0])
    values = np.array([8.0])
    if density_first:
        builder.set_number_density(values)
        built = builder.set_volume(volumes).build()
        npt.assert_array_equal(built.population[:, 0], [2.0, 8.0, 32.0])
        with pytest.raises(ValueError, match="conflict"):
            builder.set_concentration(values, "count")
    else:
        builder.set_concentration(values, "count")
        built = builder.set_volume(volumes).build()
        npt.assert_array_equal(built.population[:, 0], [8.0, 8.0, 8.0])
        with pytest.raises(ValueError, match="conflict"):
            builder.set_number_density(values)


def _sample(kind="discrete", boxes=1):
    return ParticleData(
        masses=np.ones((boxes, 2, 1)),
        population=np.full((boxes, 2), 3.0),
        charge=np.zeros((boxes, 2)),
        density=np.array([1000.0]),
        volume=np.full(boxes, 0.25),
        distribution_type=kind,
        radius_grid=np.array([1e-8, 2e-8])
        if kind == "continuous_pdf"
        else None,
    )


@pytest.mark.parametrize(
    "kind", ["discrete", "particle_resolved", "continuous_pdf"]
)
def test_replace_slot_density_and_volume(kind):
    """Use each new volume once, preserving buffer identity and PDF units."""
    data = _sample(kind, boxes=2)
    raw, volumes = data.population, data.volume
    slots = [[400.0, 200.0], [120.0, 80.0]]
    data.set_slot_concentration_density(slots, volume=[0.5, 2.0])
    assert data.population is raw and data.volume is volumes
    npt.assert_array_equal(volumes, [0.5, 2.0])
    npt.assert_array_equal(raw, [[200, 100], [240, 160]])
    npt.assert_array_equal(data.slot_concentration_density, slots)
    expected = [3e-6, 1e-6] if kind == "continuous_pdf" else [600, 200]
    npt.assert_allclose(data.number_density, expected, atol=0)


def test_replace_slot_density_at_current_volume():
    """A single-box measurement needs no new volume or batch dimension."""
    data = _sample()
    data.volume.flags.writeable = False
    data.set_slot_concentration_density([400, 0])
    npt.assert_array_equal(data.population, [[100, 0]])


def test_set_slot_density_stages_inputs():
    """Inputs may refer to destination buffers, including strided views."""
    data = _sample(boxes=2)
    data.population[:] = [[2, 3], [4, 5]]
    data.set_slot_concentration_density(
        data.population[:, ::-1], volume=data.population[:, 0]
    )
    npt.assert_array_equal(data.volume, [2, 4])
    npt.assert_array_equal(data.population, [[6, 4], [20, 16]])
    data.set_slot_concentration_density([[1, 2], [3, 4]], volume=0.5)
    npt.assert_array_equal(data.population, [[0.5, 1], [1.5, 2]])


@pytest.mark.parametrize(
    ("slots", "volume"),
    [
        ([1, 2], [0.5, 2]),
        ([[1], [1]], 1),
        ([[1, 2]] * 2, [1]),
        ([[1, 2]] * 2, [1, 0]),
        ([[1, 2]] * 2, [1, np.inf]),
        ([[1, -1]] * 2, 2),
        ([[1, np.nan]] * 2, 2),
        ([[1, np.finfo(float).max]] * 2, 2),
        ([[1, 2j]] * 2, 2),
        ([[1, 2]] * 2, True),
    ],
)
def test_set_slot_density_rejects_without_mutation(slots, volume):
    """Invalid values, shape and range preserve both original buffers."""
    data = _sample(boxes=2)
    with pytest.raises((TypeError, ValueError)):
        data.set_slot_concentration_density(slots, volume=volume)
    npt.assert_array_equal(data.population, 3)
    npt.assert_array_equal(data.volume, 0.25)


def test_set_slot_density_rejects_unwritable_storage():
    """Validate even the second destination before publishing the first."""
    data = _sample()
    data.volume.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        data.set_slot_concentration_density([400, 100], volume=0.5)
    npt.assert_array_equal(data.population, 3)
    npt.assert_array_equal(data.volume, 0.25)
    data.charge = data.population
    with pytest.raises(ValueError, match="alias"):
        data.set_slot_concentration_density([400, 100])
    npt.assert_array_equal(data.charge, 3)
