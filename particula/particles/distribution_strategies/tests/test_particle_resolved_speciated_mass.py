import numpy as np
import pytest

from particula.particles.distribution_strategies import ParticleResolvedSpeciatedMass

particle_resolved_strategy = ParticleResolvedSpeciatedMass()


def test_get_name():
    assert particle_resolved_strategy.get_name() == "ParticleResolvedSpeciatedMass"


def test_get_mass_and_species_mass():
    distribution = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    np.testing.assert_array_equal(
        particle_resolved_strategy.get_species_mass(distribution, densities),
        distribution,
    )
    expected_mass = np.sum(distribution, axis=1)
    np.testing.assert_allclose(
        particle_resolved_strategy.get_mass(distribution, densities), expected_mass
    )


def test_get_radius():
    distribution = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    volumes = np.sum(distribution / densities, axis=1)
    expected = (3 * volumes / (4 * np.pi)) ** (1 / 3)
    np.testing.assert_allclose(
        particle_resolved_strategy.get_radius(distribution, densities), expected
    )


def test_get_total_mass():
    distribution = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 50], dtype=np.float64)
    mass_per_particle = np.sum(distribution, axis=1)
    expected_total = np.sum(mass_per_particle * concentration)
    assert particle_resolved_strategy.get_total_mass(
        distribution, concentration, densities
    ) == pytest.approx(expected_total)


def test_add_mass():
    distribution = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    added_mass = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)

    conc_expand = concentration[:, np.newaxis]
    new_mass = np.divide(
        np.maximum(distribution * conc_expand + added_mass, 0),
        conc_expand,
        out=np.zeros_like(distribution),
        where=conc_expand != 0,
    )
    new_mass_sum = np.sum(new_mass, axis=1)
    expected_conc = np.where(new_mass_sum > 0, concentration, 0)

    new_dist, new_conc = particle_resolved_strategy.add_mass(
        distribution, concentration, densities, added_mass
    )
    np.testing.assert_allclose(new_dist, new_mass)
    np.testing.assert_allclose(new_conc, expected_conc)


def test_add_concentration_append():
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([2.0], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    new_dist, new_conc = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
    )
    np.testing.assert_array_equal(new_dist[-1], added_distribution[0])
    np.testing.assert_array_equal(new_conc, np.array([1.0, 1.0]))


def test_add_concentration_fill_empties():
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([2.0, 0.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    new_dist, new_conc = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
    )
    np.testing.assert_array_equal(new_dist[-1], added_distribution[0])
    np.testing.assert_array_equal(new_conc, np.array([1.0, 0.0, 1.0]))


def test_add_concentration_partial_fill():
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([0.0, 0.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    added_concentration = np.array([1.0, 1.0], dtype=np.float64)
    new_dist, new_conc = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
    )
    np.testing.assert_array_equal(new_dist[0], added_distribution[0])
    np.testing.assert_array_equal(new_dist[-1], added_distribution[1])
    np.testing.assert_array_equal(new_conc, np.array([np.inf, 0.0, np.inf]))


def test_add_concentration_error():
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            distribution.copy(),
            np.array([0.5], dtype=np.float64),
        )


def test_collide_pairs():
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600], [700, 800]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([1, 1, 1, 1], dtype=np.float64)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    expected_mass = distribution.copy()
    small_idx = pairs[:, 0]
    large_idx = pairs[:, 1]
    expected_mass[large_idx, :] += expected_mass[small_idx, :]
    expected_mass[small_idx, :] = 0
    expected_conc = concentration.copy()
    expected_conc[small_idx] = 0

    result_mass, result_conc = particle_resolved_strategy.collide_pairs(
        distribution, concentration, densities, pairs
    )
    np.testing.assert_allclose(result_mass, expected_mass)
    np.testing.assert_allclose(result_conc, expected_conc)
