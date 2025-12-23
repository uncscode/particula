"""Tests for the :class:`RadiiBasedMovingBin` distribution strategy."""
# pylint: disable=R0801

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particula.particles.distribution_strategies import RadiiBasedMovingBin

radii_based_strategy = RadiiBasedMovingBin()


def test_get_name():
    """Test retrieving the class name."""
    assert radii_based_strategy.get_name() == "RadiiBasedMovingBin"


def test_get_mass():
    """Test mass calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    expected = (4 / 3) * np.pi * distribution**3 * density
    np.testing.assert_allclose(
        radii_based_strategy.get_mass(distribution, density), expected
    )


def test_get_radius():
    """Test radius retrieval."""
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    np.testing.assert_array_equal(
        radii_based_strategy.get_radius(distribution, density), distribution
    )


def test_get_total_mass():
    """Test total mass computation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)
    density = np.float64(5)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    expected_masses = (4 / 3) * np.pi * distribution**3 * density
    expected_total = np.sum(expected_masses * concentration)
    assert radii_based_strategy.get_total_mass(
        distribution, concentration, density
    ) == pytest.approx(expected_total)


def test_add_mass():
    """Test mass addition updates radii."""
    distribution = np.array([1, 2, 3], dtype=np.float64)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    density = np.array([2, 5, 1], dtype=np.float64)
    added_mass = np.array([10, 20, 30], dtype=np.float64)

    mass_per_particle = added_mass / concentration
    initial_volumes = (4 / 3) * np.pi * distribution**3
    new_volumes = initial_volumes + mass_per_particle / density
    expected_radii = (3 * new_volumes / (4 * np.pi)) ** (1 / 3)
    new_dist, new_conc = radii_based_strategy.add_mass(
        distribution, concentration, density, added_mass
    )
    np.testing.assert_allclose(new_dist, expected_radii)
    np.testing.assert_array_equal(new_conc, concentration)


def test_add_concentration_weighted_charge():
    """Charge is updated by concentration-weighted average."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([1.0, 2.0], dtype=np.float64)
    charge = np.array([0.1, 0.2], dtype=np.float64)
    added_charge = np.array([0.3, -0.2], dtype=np.float64)

    expected_conc = concentration + added_concentration
    expected_charge = np.array([0.2, 0.0])

    new_dist, new_conc, returned_charge = (
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
            added_charge=added_charge,
        )
    )
    np.testing.assert_array_equal(new_dist, distribution)
    np.testing.assert_array_equal(new_conc, expected_conc)
    np.testing.assert_allclose(returned_charge, expected_charge)


def test_add_concentration_zero_total_uses_added_charge():
    """Zero-total bins fall back to added_charge to avoid NaNs."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([0.0, 3.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.0, 1.0], dtype=np.float64)
    charge = np.array([0.4, -0.4], dtype=np.float64)
    added_charge = np.array([1.4, 0.4], dtype=np.float64)
    expected_conc = concentration + added_concentration

    _new_dist, new_conc, returned_charge = (
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
            added_charge=added_charge,
        )
    )

    np.testing.assert_array_equal(new_conc, expected_conc)
    np.testing.assert_allclose(returned_charge, np.array([1.4, -0.2]))


def test_add_concentration_missing_added_charge_preserves_charge():
    """When added_charge is None, existing charge is preserved."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.5, 0.5], dtype=np.float64)
    charge = np.array([0.1, 0.2], dtype=np.float64)

    _new_dist, _new_conc, returned_charge = (
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
        )
    )

    assert returned_charge is charge
    np.testing.assert_array_equal(returned_charge, charge)


def test_add_concentration_charge_none_returns_none():
    """When charge is None, charge return should remain None."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.5, 0.5], dtype=np.float64)

    _new_dist, _new_conc, returned_charge = (
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )

    assert returned_charge is None


def test_add_concentration_distribution_error():
    """Test distribution shape validation."""
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
    """Test concentration shape validation."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([[1.0, 1.0]], dtype=np.float64),
        )


def test_add_concentration_added_charge_shape_error():
    """added_charge must match added_concentration shape."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    charge = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([1.0, 1.0], dtype=np.float64),
            charge=charge,
            added_charge=np.array([[0.1, 0.2]], dtype=np.float64),
        )


def test_add_concentration_charge_shape_error():
    """charge must match concentration shape when provided."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([1.0, 1.0], dtype=np.float64),
            charge=np.array([[0.1, 0.2]], dtype=np.float64),
            added_charge=np.array([0.1, 0.2], dtype=np.float64),
        )


def test_add_concentration_distribution_value_mismatch_error():
    """Distribution mismatch on value triggers ValueError."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        radii_based_strategy.add_concentration(
            distribution,
            concentration,
            np.array([1.0, 2.01], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_collide_pairs():
    """Test collide_pairs not implemented."""
    with pytest.raises(NotImplementedError):
        radii_based_strategy.collide_pairs(
            np.array([1, 2, 3], dtype=np.float64),
            np.array([5], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int64),
        )
