"""Explicit native storage retains the science formerly tested via bridges."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.particles import (
    ParticleData,
    ParticleDataBuilder,
    create_lognormal_binned_particle_data,
    create_lognormal_resolved_particle_data,
    create_particle_data_from_radii,
    create_resolved_particle_data,
)


@pytest.mark.parametrize("kind", ["discrete", "particle_resolved"])
def test_explicit_mass_species_construction(kind):
    """Explicit species axes retain mass, density, counts and scalar charge."""
    masses = np.diag([1.0, 2.0, 3.0])
    density = np.array([1.0, 2.0, 3.0])
    counts = np.array([10.0, 20.0, 30.0])
    if kind == "particle_resolved":
        data = create_resolved_particle_data(
            masses=masses,
            density=density,
            concentration=counts,
            charge=1.0,
            volume=1.0,
        )
    else:
        data = (
            ParticleDataBuilder()
            .set_distribution_type(kind)
            .set_masses(masses)
            .set_density(density)
            .set_concentration(counts, units="1/m^3")
            .set_charge(np.ones(3))
            .set_volume(np.array([1.0]))
            .build()
        )
    assert isinstance(data, ParticleData)
    npt.assert_array_equal(data.masses[0], masses)
    npt.assert_array_equal(data.total_mass[0], [1.0, 2.0, 3.0])
    npt.assert_array_equal(data.density, density)
    npt.assert_array_equal(data.concentration[0], counts)
    npt.assert_array_equal(data.charge[0], [1.0, 1.0, 1.0])
    npt.assert_array_equal(data.volume, [1.0])


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("resolved", [False, True])
def test_native_presets_retain_factory_configurations(configured, resolved):
    """Both preset families retain default and explicit modal construction."""
    parameters = {}
    if configured:
        parameters = dict(
            mode=np.array([100.0, 2000.0]) * 1e-9,
            geometric_standard_deviation=np.array([1.4, 1.5]),
            number_concentration=np.array([1e3, 1e3]),
        )
    if resolved:
        saved = np.random.get_state()
        try:
            data = create_lognormal_resolved_particle_data(
                **parameters,
                particle_resolved_count=1000 if configured else 20000,
            )
        finally:
            np.random.set_state(saved)
        npt.assert_array_equal(
            data.concentration, np.ones_like(data.concentration)
        )
        assert data.n_particles == (1000 if configured else 20000)
    else:
        data = create_lognormal_binned_particle_data(
            **parameters,
            distribution_type="discrete",
        )
        npt.assert_allclose(
            data.number_density,
            [2000 if configured else 1.1e10],
            rtol=1e-12,
            atol=0,
        )
    assert isinstance(data, ParticleData)
    npt.assert_array_equal(data.volume, [1.0])
    assert np.all(np.isfinite(data.masses))
    assert np.all(data.masses > 0)


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("boxes", [1, 2, 3])
def test_explicit_resolved_storage_and_copy(volume, boxes):
    """Own every input lane; retain physical volume and each selected box."""
    masses = np.array([[0.5e-18, 1.5e-18], [2e-18, 3e-18], [0, 0]])
    masses = np.stack([masses * (i + 1) for i in range(boxes)])
    counts = np.array([8.0, 0.5, 0.0])
    charge = np.array([1.0, -1.0, 0.0])
    density = np.array([1000.0, 1200.0])
    data = create_resolved_particle_data(
        masses=masses,
        concentration=counts,
        charge=charge,
        density=density,
        volume=volume,
    )
    copied = data.copy()
    assert data.masses.shape == (boxes, 3, 2)
    for candidate in (data, copied):
        assert candidate.distribution_type == "particle_resolved"
        npt.assert_array_equal(candidate.density, density)
        npt.assert_array_equal(candidate.volume, np.full(boxes, volume))
        for box in range(boxes):
            npt.assert_array_equal(candidate.masses[box], masses[box])
            npt.assert_array_equal(candidate.concentration[box], counts)
            npt.assert_array_equal(candidate.charge[box], charge)
            npt.assert_allclose(
                candidate.slot_concentration_density[box],
                counts / volume,
                rtol=1e-12,
                atol=0,
            )
            npt.assert_allclose(
                candidate.species_mass_inventory[box],
                np.sum(masses[box] * counts[:, None], axis=0),
                rtol=1e-12,
                atol=1e-30,
            )
            npt.assert_array_equal(
                candidate.total_mass[box], masses[box].sum(axis=1)
            )
    for name in ("masses", "concentration", "charge", "density", "volume"):
        assert not np.shares_memory(getattr(copied, name), getattr(data, name))
    for name, source in (
        ("masses", masses),
        ("concentration", counts),
        ("charge", charge),
        ("density", density),
    ):
        assert not np.shares_memory(getattr(data, name), source)
    copied.masses[:] = 0
    npt.assert_array_equal(data.masses, masses)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_explicit_radius_storage(kind):
    """Radius construction retains raw fields and the independent sphere law."""
    radii = np.array([1e-9, 2e-9, 4e-9])
    raw = np.array([1e9, 2e9, 1e9])
    charge = np.array([0.0, -1.0, 2.0])
    data = create_particle_data_from_radii(
        radii=radii,
        concentration=raw,
        charge=charge,
        density=1000.0,
        distribution_type=kind,
    )
    assert data.masses.shape == (1, 3, 1)
    npt.assert_allclose(
        data.masses[0, :, 0],
        4 * np.pi / 3 * radii**3 * 1000,
        rtol=1e-12,
        atol=0,
    )
    npt.assert_allclose(data.radii[0], radii, rtol=1e-12, atol=0)
    npt.assert_array_equal(data.concentration[0], raw)
    npt.assert_array_equal(data.charge[0], charge)
    npt.assert_array_equal(data.density, [1000.0])
    npt.assert_array_equal(data.volume, [1.0])
    if kind == "continuous_pdf":
        npt.assert_array_equal(data.radius_grid, radii)
        expected = np.trapezoid(raw, radii)
    else:
        assert data.radius_grid is None
        expected = raw.sum()
    npt.assert_allclose(data.number_density, [expected], rtol=1e-12, atol=0)
