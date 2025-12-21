"""Tests for the :class:`SpeciatedMassMovingBin` distribution strategy."""
# pylint: disable=R0801

import numpy as np
import pytest

from particula.particles.distribution_strategies import SpeciatedMassMovingBin

speciated_mass_strategy = SpeciatedMassMovingBin()


def test_get_name():
    """Test retrieving the class name."""
    assert speciated_mass_strategy.get_name() == "SpeciatedMassMovingBin"


def test_get_species_mass_and_mass():
    """Test species and total mass retrieval."""
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    np.testing.assert_array_equal(
        speciated_mass_strategy.get_species_mass(distribution, densities),
        distribution,
    )
    expected_mass = np.sum(distribution, axis=1)
    np.testing.assert_allclose(
        speciated_mass_strategy.get_mass(distribution, densities), expected_mass
    )


def test_get_radius():
    """Test radius computation."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    volumes = np.sum(distribution / densities, axis=1)
    expected = (3 * volumes / (4 * np.pi)) ** (1 / 3)
    np.testing.assert_allclose(
        speciated_mass_strategy.get_radius(distribution, densities), expected
    )


def test_get_total_mass():
    """Test total mass computation."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 50], dtype=np.float64)
    mass_per_particle = np.sum(distribution, axis=1)
    expected_total = np.sum(mass_per_particle * concentration)
    assert speciated_mass_strategy.get_total_mass(
        distribution, concentration, densities
    ) == pytest.approx(expected_total)


def test_add_mass():
    """Test mass addition per species."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    added_mass = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
    conc_expand = concentration[:, np.newaxis]
    mass_per_particle = np.where(conc_expand > 0, added_mass / conc_expand, 0)
    expected_dist = np.maximum(distribution + mass_per_particle, 0)
    new_dist, new_conc = speciated_mass_strategy.add_mass(
        distribution, concentration, densities, added_mass
    )
    np.testing.assert_allclose(new_dist, expected_dist)
    np.testing.assert_array_equal(new_conc, concentration)


def test_add_concentration():
    """Test concentration addition and charge passthrough."""
    distribution = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    concentration = np.array([5.0, 6.0], dtype=np.float64)
    added_distribution = distribution.copy()
    added_concentration = np.array([1.0, 2.0], dtype=np.float64)
    charge = np.array([0.3, -0.1], dtype=np.float64)
    expected = concentration + added_concentration
    new_dist, new_conc, returned_charge = (
        speciated_mass_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
        )
    )
    np.testing.assert_array_equal(new_dist, distribution)
    np.testing.assert_array_equal(new_conc, expected)
    assert returned_charge is charge
    np.testing.assert_array_equal(returned_charge, charge)


def test_add_concentration_charge_none_returns_none():
    """When charge is None, charge should stay None."""
    distribution = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    concentration = np.array([5.0, 6.0], dtype=np.float64)
    added_distribution = distribution.copy()
    added_concentration = np.array([1.0, 2.0], dtype=np.float64)

    _new_dist, _new_conc, returned_charge = (
        speciated_mass_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )

    assert returned_charge is None


def test_add_concentration_distribution_error():
    """Test distribution shape validation."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        speciated_mass_strategy.add_concentration(
            distribution,
            concentration,
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_add_concentration_shape_error():
    """Test concentration shape validation."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        speciated_mass_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([[1.0, 1.0]], dtype=np.float64),
        )


def test_collide_pairs():
    """Test collide_pairs not implemented."""
    with pytest.raises(NotImplementedError):
        speciated_mass_strategy.collide_pairs(
            np.array([[100, 200], [300, 400]], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
            np.array([[100, 200], [300, 400]], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int64),
        )
