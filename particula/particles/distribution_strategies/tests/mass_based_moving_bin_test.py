"""Tests for the :class:`MassBasedMovingBin` distribution strategy."""

# pylint: disable=R0801

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_mass_based_strategy():
    from particula.particles.distribution_strategies import MassBasedMovingBin

    return MassBasedMovingBin()


mass_based_strategy = _load_mass_based_strategy()


def test_source_file_path():
    """Ensure MassBasedMovingBin resolves to the worktree source."""
    assert (
        "trees/86472619/particula/particles/distribution_strategies/mass_based_moving_bin.py"
        in (mass_based_strategy.add_concentration.__code__.co_filename)
    )


def test_get_name():
    """Test retrieving the class name."""
    assert mass_based_strategy.get_name() == "MassBasedMovingBin"


def test_get_mass():
    """Test that mass equals distribution."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    expected_mass = distribution
    np.testing.assert_array_equal(
        mass_based_strategy.get_mass(distribution, density), expected_mass
    )


def test_get_radius():
    """Test radius calculation."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    expected = (3 * distribution / (4 * np.pi * density)) ** (1 / 3)
    np.testing.assert_allclose(
        mass_based_strategy.get_radius(distribution, density), expected
    )


def test_get_total_mass():
    """Test total mass computation."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    concentration = np.array([1, 2, 3], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    expected_total_mass = np.sum(distribution * concentration)
    assert mass_based_strategy.get_total_mass(
        distribution, concentration, density
    ) == pytest.approx(expected_total_mass)


def test_add_mass():
    """Test mass addition updates distribution."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    concentration = np.array([1, 2, 3], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    added_mass = np.array([10, 20, 30], dtype=np.float64)
    expected_dist = distribution + added_mass
    expected_conc = concentration
    new_dist, new_conc = mass_based_strategy.add_mass(
        distribution, concentration, density, added_mass
    )
    np.testing.assert_array_equal(new_dist, expected_dist)
    np.testing.assert_array_equal(new_conc, expected_conc)


def test_add_concentration_weighted_charge():
    """Charge is updated by concentration-weighted average."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([10.0, 20.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([5.0, 5.0], dtype=np.float64)
    charge = np.array([0.1, -0.2], dtype=np.float64)
    added_charge = np.array([0.0, 0.4], dtype=np.float64)

    expected_conc = concentration + added_concentration
    expected_charge = np.array([0.06666667, -0.08])

    new_dist, new_conc, returned_charge = mass_based_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=charge,
        added_charge=added_charge,
    )

    np.testing.assert_array_equal(new_dist, distribution)
    np.testing.assert_array_equal(new_conc, expected_conc)
    np.testing.assert_allclose(returned_charge, expected_charge)


def test_add_concentration_zero_total_uses_added_charge():
    """Zero-total bins fall back to added_charge to avoid NaNs."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([0.0, 2.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.0, 2.0], dtype=np.float64)
    charge = np.array([0.5, -0.5], dtype=np.float64)
    added_charge = np.array([1.5, 0.5], dtype=np.float64)
    expected_conc = concentration + added_concentration

    _new_dist, new_conc, returned_charge = (
        mass_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
            added_charge=added_charge,
        )
    )

    np.testing.assert_array_equal(new_conc, expected_conc)
    np.testing.assert_allclose(returned_charge, np.array([1.5, 0.0]))


def test_add_concentration_missing_added_charge_preserves_charge():
    """When added_charge is None, existing charge is preserved."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([3.0, 4.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.5, 0.5], dtype=np.float64)
    charge = np.array([0.2, -0.1], dtype=np.float64)

    _new_dist, _new_conc, returned_charge = (
        mass_based_strategy.add_concentration(
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
    """When charge is None, add_concentration should return None charge."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([3.0, 4.0], dtype=np.float64)
    added_distribution = np.array([1.0, 2.0], dtype=np.float64)
    added_concentration = np.array([0.5, 0.5], dtype=np.float64)

    _new_dist, _new_conc, returned_charge = (
        mass_based_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )

    assert returned_charge is None


def test_add_concentration_distribution_error():
    """Test shape validation for distribution input."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        mass_based_strategy.add_concentration(
            distribution,
            concentration,
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_add_concentration_shape_error():
    """Test shape validation for concentration."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        mass_based_strategy.add_concentration(
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
        mass_based_strategy.add_concentration(
            distribution,
            concentration,
            distribution,
            np.array([1.0, 1.0], dtype=np.float64),
            charge=charge,
            added_charge=np.array([[0.1, 0.2]], dtype=np.float64),
        )


def test_add_concentration_charge_shape_error():
    """Charge must match concentration shape when provided."""
    distribution = np.array([1.0, 2.0], dtype=np.float64)
    concentration = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        mass_based_strategy.add_concentration(
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
        mass_based_strategy.add_concentration(
            distribution,
            concentration,
            np.array([1.0, 2.001], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_collide_pairs():
    """Test collide_pairs not implemented."""
    with pytest.raises(NotImplementedError):
        mass_based_strategy.collide_pairs(
            np.array([100, 200, 300], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
            np.array([100, 200, 300], dtype=np.float64),
            np.array([[0, 1]], dtype=np.int64),
        )
