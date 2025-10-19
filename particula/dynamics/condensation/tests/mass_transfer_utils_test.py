"""Test for mass_transfer_utilsmodule."""

import numpy as np

from particula.dynamics.condensation.mass_transfer_utils import (
    apply_condensation_limit,
    apply_evaporation_limit,
    apply_per_bin_limit,
    calc_mass_to_change,
)


def test_calc_mass_to_change_single():
    """Test mass change calculation for a single particle."""
    mass_rate = np.array([0.1, 0.2])
    time_step = 10.0
    conc = np.array([5.0, 4.0])
    expected = mass_rate * time_step * conc
    result = calc_mass_to_change(mass_rate, time_step, conc)
    np.testing.assert_allclose(result, expected)


def test_calc_mass_to_change_multi():
    """Test mass change calculation for multiple particles."""
    mass_rate = np.array([[0.1, 0.05, 0.02], [0.2, 0.15, 0.1]])
    time_step = 10.0
    conc = np.array([5.0, 4.0])
    expected = mass_rate * time_step * conc[:, None]
    result = calc_mass_to_change(mass_rate, time_step, conc)
    np.testing.assert_allclose(result, expected)


def test_apply_condensation_limit_single():
    """Test condensation limit application for a single particle."""
    mass = np.array([2.0, -1.0])
    gas_mass = np.array([0.5])
    limited, evap_sum, neg_mask = apply_condensation_limit(mass, gas_mass)
    scale = (gas_mass[0] - mass[mass < 0.0].sum()) / mass[mass > 0.0].sum()
    expected = np.where(mass > 0.0, mass * scale, mass)
    np.testing.assert_allclose(limited, expected)
    np.testing.assert_allclose(evap_sum, -1.0)
    np.testing.assert_array_equal(neg_mask, mass < 0.0)


def test_apply_condensation_limit_multi():
    """Test condensation limit application for multiple particles."""
    mass = np.array([[1.0, 0.2], [1.0, -0.1]])
    gas_mass = np.array([1.0, 0.4])
    limited, evap_sum, neg_mask = apply_condensation_limit(
        mass.copy(), gas_mass
    )
    pos_sum = np.where(mass > 0.0, mass, 0.0).sum(axis=0)
    evap = np.where(mass < 0.0, mass, 0.0).sum(axis=0)
    scale = np.ones_like(gas_mass)
    mask = (pos_sum > 0.0) & (pos_sum + evap > gas_mass)
    scale[mask] = (gas_mass[mask] - evap[mask]) / pos_sum[mask]
    expected = np.where(mass > 0.0, mass * scale, mass)
    np.testing.assert_allclose(limited, expected)
    np.testing.assert_allclose(evap_sum, evap)
    np.testing.assert_array_equal(neg_mask, mass < 0.0)


def test_apply_evaporation_limit_single():
    """Test evaporation limit application for a single particle."""
    mass = np.array([-2.0, -1.0])
    particle_mass = np.array([0.5, 1.0])
    conc = np.array([1.0, 2.0])
    evap_sum = mass.sum()
    neg_mask = mass < 0.0
    limited = apply_evaporation_limit(
        mass, particle_mass, conc, evap_sum, neg_mask
    )
    inventory = (particle_mass * conc).sum()
    scale = inventory / (-evap_sum)
    expected = mass * scale
    np.testing.assert_allclose(limited, expected)


def test_apply_evaporation_limit_multi():
    """Test evaporation limit application for multiple particles."""
    mass = np.array([[-4.0, -1.0], [-1.0, -2.0]])
    particle_mass = np.array([[1.0, 0.5], [1.0, 0.5]])
    conc = np.array([1.0, 2.0])
    evap_sum = mass.sum(axis=0)
    neg_mask = mass < 0.0
    limited = apply_evaporation_limit(
        mass, particle_mass, conc, evap_sum, neg_mask
    )
    inventory = (particle_mass * conc[:, None]).sum(axis=0)
    scale = inventory / (-evap_sum)
    expected = mass * scale
    np.testing.assert_allclose(limited, expected)


def test_apply_per_bin_limit_single():
    """Test per-bin limit application for a single particle."""
    mass = np.array([-2.0, -0.5])
    particle_mass = np.array([0.3, 0.4])
    conc = np.array([1.0, 2.0])
    limit = -particle_mass * conc
    expected = np.maximum(mass, limit)
    result = apply_per_bin_limit(mass, particle_mass, conc)
    np.testing.assert_allclose(result, expected)


def test_apply_per_bin_limit_multi():
    """Test per-bin limit application for multiple particles."""
    mass = np.array([[-2.0, -0.2], [0.1, -0.5]])
    particle_mass = np.array([[0.3, 0.4], [0.2, 0.1]])
    conc = np.array([2.0, 1.0])
    limit = -particle_mass * conc[:, None]
    expected = np.maximum(mass, limit)
    result = apply_per_bin_limit(mass, particle_mass, conc)
    np.testing.assert_allclose(result, expected)
