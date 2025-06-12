import numpy as np
import pytest

from particula.particles.distribution_strategies import RadiiBasedMovingBin

radii_based_strategy = RadiiBasedMovingBin()


def test_get_name():
    assert radii_based_strategy.get_name() == "RadiiBasedMovingBin"


def test_get_mass():
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    expected = (4 / 3) * np.pi * distribution**3 * density
    np.testing.assert_allclose(
        radii_based_strategy.get_mass(distribution, density), expected
    )


def test_get_radius():
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    np.testing.assert_array_equal(
        radii_based_strategy.get_radius(distribution, density), distribution
    )


def test_get_total_mass():
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    expected_masses = (4 / 3) * np.pi * distribution**3 * density
    expected_total = np.sum(expected_masses * concentration)
    assert radii_based_strategy.get_total_mass(
        distribution, concentration, density
    ) == pytest.approx(expected_total)


def test_add_mass():
    distribution = np.array([1, 2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    density = np.array([2, 5, 1], dtype=np.float64)
    added_mass = np.array([10, 20, 30], dtype=np.float64)

    mass_per_particle = added_mass / concentration
    new_volumes = (4 / 3) * np.pi * distribution**3 + mass_per_particle / density
    expected_radii = (3 * new_volumes / (4 * np.pi)) ** (1 / 3)
    new_dist, new_conc = radii_based_strategy.add_mass(
        distribution, concentration, density, added_mass
    )
    np.testing.assert_allclose(new_dist, expected_radii)
    np.testing.assert_array_equal(new_conc, concentration)


def test_add_concentration():
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.5, 0.5], dtype=np.float64)
    expected = concentration + added_concentration
    _, new_conc = radii_based_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
    )
    np.testing.assert_array_equal(new_conc, expected)


def test_add_concentration_distribution_error():
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_add_concentration_shape_error():
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([[1.0, 1.0]], dtype=np.float64),
        )


def test_collide_pairs():
    with pytest.raises(NotImplementedError):
        radii_based_strategy.collide_pairs(
            np.array([1, 2, 3], dtype=np.float64),
            np.array([5], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int64),
        )
