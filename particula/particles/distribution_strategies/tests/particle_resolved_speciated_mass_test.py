"""Tests for ParticleResolvedSpeciatedMass distribution strategy."""
# pylint: disable=R0801

import numpy as np
import pytest

from particula.particles.distribution_strategies import (
    ParticleResolvedSpeciatedMass,
)

particle_resolved_strategy = ParticleResolvedSpeciatedMass()


def test_get_name():
    """Test retrieving the class name."""
    expected = "ParticleResolvedSpeciatedMass"
    assert particle_resolved_strategy.get_name() == expected


def test_get_mass_and_species_mass():
    """Test mass and species mass retrieval."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    np.testing.assert_array_equal(
        particle_resolved_strategy.get_species_mass(distribution, densities),
        distribution,
    )
    expected_mass = np.sum(distribution, axis=1)
    result = particle_resolved_strategy.get_mass(distribution, densities)
    np.testing.assert_allclose(result, expected_mass)


def test_get_radius():
    """Test radius calculation."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    volumes = np.sum(distribution / densities, axis=1)
    expected = (3 * volumes / (4 * np.pi)) ** (1 / 3)
    np.testing.assert_allclose(
        particle_resolved_strategy.get_radius(distribution, densities), expected
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
    assert particle_resolved_strategy.get_total_mass(
        distribution, concentration, densities
    ) == pytest.approx(expected_total)


def test_add_mass():
    """Test mass addition with normalization."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
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
    """Test appending concentration."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([2.0], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    new_dist, new_conc, new_charge = (
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )
    np.testing.assert_array_equal(new_dist[-1], added_distribution[0])
    np.testing.assert_array_equal(new_conc, np.array([1.0, 1.0]))
    assert new_charge is None


def test_add_concentration_fill_empties():
    """Test filling empty bins."""
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([2.0, 0.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    new_dist, new_conc, new_charge = (
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )
    np.testing.assert_array_equal(new_dist, np.array([[1.0, 2.0], [7.0, 8.0]]))
    np.testing.assert_array_equal(new_conc, np.array([1.0, 1.0]))
    assert new_charge is None


def test_add_concentration_partial_fill():
    """Test replacing all bins."""
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([0.0, 0.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    added_concentration = np.array([1.0, 1.0], dtype=np.float64)
    new_dist, new_conc, new_charge = (
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
        )
    )
    np.testing.assert_array_equal(new_dist, added_distribution)
    np.testing.assert_array_equal(new_conc, np.array([1.0, 1.0]))
    assert new_charge is None


def test_add_concentration_with_charge_append():
    """Test appending charged particles adds charge values."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    charge = np.array([0.5], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    added_charge = np.array([2.0], dtype=np.float64)

    _, _, new_charge = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=charge,
        added_charge=added_charge,
    )

    assert new_charge is not None
    np.testing.assert_array_equal(new_charge, np.array([0.5, 2.0]))


def test_add_concentration_fill_empty_with_charge():
    """Test filling empty bins places leading charges."""
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([0.0, 1.0], dtype=np.float64)
    charge = np.array([0.0, -1.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    added_charge = np.array([3.0], dtype=np.float64)

    _, _, new_charge = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=charge,
        added_charge=added_charge,
    )

    assert new_charge is not None
    np.testing.assert_array_equal(new_charge, np.array([3.0, -1.0]))


def test_add_concentration_partial_fill_with_charge():
    """Test partial fill then append charges."""
    distribution = np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64)
    concentration = np.array([0.0, 1.0], dtype=np.float64)
    charge = np.array([0.0, -1.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    added_concentration = np.array([1.0, 1.0], dtype=np.float64)
    added_charge = np.array([4.0, 5.0], dtype=np.float64)

    _, _, new_charge = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=charge,
        added_charge=added_charge,
    )

    assert new_charge is not None
    np.testing.assert_array_equal(new_charge, np.array([4.0, -1.0, 5.0]))


def test_add_concentration_more_empty_bins_than_added():
    """Test when empty_bins_count > added_bins_count with charge."""
    distribution = np.array(
        [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
    )
    concentration = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    charge = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    added_distribution = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    added_concentration = np.array([1.0, 1.0], dtype=np.float64)
    added_charge = np.array([2.0, 3.0], dtype=np.float64)

    new_dist, new_conc, new_charge = (
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=charge,
            added_charge=added_charge,
        )
    )

    # First bin unchanged, bins 1-2 filled, bin 3 remains empty
    expected_dist = np.array(
        [[1.0, 2.0], [7.0, 8.0], [9.0, 10.0], [0.0, 0.0]], dtype=np.float64
    )
    np.testing.assert_array_equal(new_dist, expected_dist)
    np.testing.assert_array_equal(new_conc, np.array([1.0, 1.0, 1.0, 0.0]))
    assert new_charge is not None
    np.testing.assert_array_equal(new_charge, np.array([-1.0, 2.0, 3.0, 0.0]))


def test_add_concentration_charge_defaults_to_zero():
    """Test defaulting charges to zero when omitted."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    charge = np.array([1.0], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)

    _, _, new_charge = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=charge,
    )

    assert new_charge is not None
    np.testing.assert_array_equal(new_charge, np.array([1.0, 0.0]))


def test_add_concentration_charge_shape_mismatch():
    """Test shape mismatch between charge and concentration raises error."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)
    added_charge = np.array([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        particle_resolved_strategy.add_concentration(
            distribution,
            concentration,
            added_distribution,
            added_concentration,
            charge=np.array([0.0], dtype=np.float64),
            added_charge=added_charge,
        )


def test_add_concentration_charge_none_passthrough():
    """Test charge None path returns None unchanged."""
    distribution = np.array([[1.0, 2.0]], dtype=np.float64)
    concentration = np.array([1.0], dtype=np.float64)
    added_distribution = np.array([[3.0, 4.0]], dtype=np.float64)
    added_concentration = np.array([1.0], dtype=np.float64)

    _, _, new_charge = particle_resolved_strategy.add_concentration(
        distribution,
        concentration,
        added_distribution,
        added_concentration,
        charge=None,
    )

    assert new_charge is None


def test_add_concentration_error():
    """Test concentration validation errors."""
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
    """Test pair collisions."""
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

    result_mass, result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs
        )
    )
    np.testing.assert_allclose(result_mass, expected_mass)
    np.testing.assert_allclose(result_conc, expected_conc)
    assert result_charge is None


def test_collide_pairs_with_charge():
    """Test pair collisions conserve charge."""
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600], [700, 800]], dtype=np.float64
    )
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([1, 1, 1, 1], dtype=np.float64)
    charge = np.array([1.0, -2.0, 3.0, 0.0], dtype=np.float64)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    # Expected: charge[1] = 1.0 + (-2.0) = -1.0, charge[3] = 3.0 + 0.0 = 3.0
    # charge[0] = 0, charge[2] = 0
    expected_charge = np.array([0.0, -1.0, 0.0, 3.0], dtype=np.float64)

    _result_mass, _result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs, charge
        )
    )
    assert result_charge is not None
    np.testing.assert_allclose(result_charge, expected_charge)


def test_collide_pairs_charge_one_zero():
    """Test pair collisions when one particle has zero charge."""
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([1, 1], dtype=np.float64)
    charge = np.array([5.0, 0.0], dtype=np.float64)
    pairs = np.array([[0, 1]], dtype=np.int64)

    # Expected: charge[1] = 5.0 + 0.0 = 5.0, charge[0] = 0
    expected_charge = np.array([0.0, 5.0], dtype=np.float64)

    _result_mass, _result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs, charge
        )
    )
    assert result_charge is not None
    np.testing.assert_allclose(result_charge, expected_charge)


def test_collide_pairs_no_charge():
    """Test pair collisions work when charge is None."""
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([1, 1], dtype=np.float64)
    pairs = np.array([[0, 1]], dtype=np.int64)

    _result_mass, _result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs, charge=None
        )
    )
    assert result_charge is None


def test_collide_pairs_zero_charge_optimization():
    """Test that all-zero charges in colliding pairs are a no-op."""
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.array([1, 1], dtype=np.float64)
    charge = np.array([0.0, 0.0], dtype=np.float64)
    pairs = np.array([[0, 1]], dtype=np.int64)

    _result_mass, _result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs, charge
        )
    )
    # Charge should be unchanged (still all zeros)
    assert result_charge is not None
    np.testing.assert_array_equal(result_charge, np.array([0.0, 0.0]))


def test_collide_pairs_1d_distribution_with_charge():
    """Test pair collisions with 1D distribution and charge."""
    distribution = np.array([100, 200, 300, 400], dtype=np.float64)
    densities = np.array([2], dtype=np.float64)
    concentration = np.array([1, 1, 1, 1], dtype=np.float64)
    charge = np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float64)
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    # Expected: charge[1] = 2.0 + (-1.0) = 1.0, charge[3] = 0.5 + 1.5 = 2.0
    expected_charge = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float64)

    _result_mass, _result_conc, result_charge = (
        particle_resolved_strategy.collide_pairs(
            distribution, concentration, densities, pairs, charge
        )
    )
    assert result_charge is not None
    np.testing.assert_allclose(result_charge, expected_charge)
