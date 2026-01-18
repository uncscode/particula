"""Tests for the particle activity module.
Replace with real values in the future.
"""

from typing import Optional

import numpy as np
import pytest

from particula.activity.activity_coefficients import bat_activity_coefficients
from particula.activity.ratio import to_molar_mass_ratio
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
    ActivityNonIdealBinary,
)

# Test ActivityNonIdealBinary


def _compute_expected_activity(
    mass_concentration: np.ndarray,
    molar_mass: float,
    oxygen2carbon: float,
    density: float,
    functional_group: Optional[str] = None,
) -> np.ndarray:
    water_moles = mass_concentration[..., 0] / 0.01801528
    organic_moles = mass_concentration[..., 1] / molar_mass
    total_moles = water_moles + organic_moles
    organic_mole_fraction = organic_moles / total_moles
    molar_mass_ratio = to_molar_mass_ratio(molar_mass * 1000.0)
    (
        _activity_water,
        activity_organic,
        _mass_water,
        _mass_organic,
        _gamma_water,
        _gamma_organic,
    ) = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=functional_group,
    )
    return activity_organic


def test_activity_non_ideal_binary_array_consistency_with_bat():
    """Strategy matches BAT for simple binary vector."""
    molar_mass = 0.2
    oxygen2carbon = 0.5
    density = 1400.0
    strategy = ActivityNonIdealBinary(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    mass_concentration = np.array([0.5, 0.5])
    expected = _compute_expected_activity(
        mass_concentration, molar_mass, oxygen2carbon, density
    )
    result = strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_activity_non_ideal_binary_broadcast():
    """Broadcast over 2D arrays matches BAT per row."""
    molar_mass = 0.25
    oxygen2carbon = 0.4
    density = 1200.0
    strategy = ActivityNonIdealBinary(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    mass_concentration = np.array([[0.3, 0.7], [0.8, 0.2]])
    expected = _compute_expected_activity(
        mass_concentration, molar_mass, oxygen2carbon, density
    )
    result = strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_activity_non_ideal_binary_high_dim_broadcast():
    """Higher-dimensional broadcast matches BAT elementwise."""
    molar_mass = 0.18
    oxygen2carbon = 0.3
    density = 1100.0
    strategy = ActivityNonIdealBinary(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    mass_concentration = np.array(
        [
            [[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]],
            [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]],
        ]
    )
    expected = _compute_expected_activity(
        mass_concentration, molar_mass, oxygen2carbon, density
    )
    result = strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_activity_non_ideal_binary_get_name():
    """Returns explicit strategy name."""
    strategy = ActivityNonIdealBinary(
        molar_mass=0.2, oxygen2carbon=0.5, density=1400.0
    )
    assert strategy.get_name() == "ActivityNonIdealBinary"


def test_activity_non_ideal_binary_partial_pressure():
    """Partial pressure scales with activity."""
    strategy = ActivityNonIdealBinary(
        molar_mass=0.2, oxygen2carbon=0.5, density=1400.0
    )
    pure_vapor_pressure = np.array([100.0, 100.0])
    mass_concentration = np.array([[0.6, 0.4], [0.2, 0.8]])
    activity = strategy.activity(mass_concentration)
    partial_pressure = strategy.partial_pressure(
        pure_vapor_pressure=pure_vapor_pressure,
        mass_concentration=mass_concentration,
    )
    np.testing.assert_allclose(partial_pressure, pure_vapor_pressure * activity)


def test_activity_non_ideal_binary_functional_group():
    """Functional group is passed through to BAT helper."""
    molar_mass = 0.22
    oxygen2carbon = 0.6
    density = 1300.0
    functional_group = "carboxylic_acid"
    strategy = ActivityNonIdealBinary(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=functional_group,
    )
    mass_concentration = np.array([0.4, 0.6])
    expected = _compute_expected_activity(
        mass_concentration,
        molar_mass,
        oxygen2carbon,
        density,
        functional_group=functional_group,
    )
    result = strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_activity_non_ideal_binary_invalid_shape_scalar():
    """Scalar input is rejected with clear message."""
    strategy = ActivityNonIdealBinary(
        molar_mass=0.2, oxygen2carbon=0.5, density=1400.0
    )
    with pytest.raises(
        ValueError,
        match=(
            "ActivityNonIdealBinary expects mass_concentration with last "
            "dimension of size 2."
        ),
    ):
        strategy.activity(1.0)


def test_activity_non_ideal_binary_invalid_shape_wrong_last_dim():
    """Non-binary trailing dimension is rejected."""
    strategy = ActivityNonIdealBinary(
        molar_mass=0.2, oxygen2carbon=0.5, density=1400.0
    )
    with pytest.raises(
        ValueError,
        match=(
            "ActivityNonIdealBinary expects mass_concentration with last "
            "dimension of size 2."
        ),
    ):
        strategy.activity(np.array([0.5, 0.3, 0.2]))


def test_activity_non_ideal_binary_zero_total_moles():
    """Zero total moles raises."""
    strategy = ActivityNonIdealBinary(
        molar_mass=0.2, oxygen2carbon=0.5, density=1400.0
    )
    with pytest.raises(
        ValueError,
        match="Total moles must be positive for activity calculation.",
    ):
        strategy.activity(np.zeros((2, 2)))


def test_activity_non_ideal_binary_single_component_limit():
    """Single-component limit matches BAT calculation."""
    molar_mass = 0.21
    oxygen2carbon = 0.55
    density = 1250.0
    strategy = ActivityNonIdealBinary(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    mass_concentration = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = _compute_expected_activity(
        mass_concentration, molar_mass, oxygen2carbon, density
    )
    result = strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


# Test MolarIdealActivity
def test_molar_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = ActivityIdealMolar(molar_mass=50.0)
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_molar_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityIdealMolar(molar_mass=np.array([1.0, 2.0, 3.0]))
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.33333, 0.333333, 0.333333])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4,
    )


# Test MassIdealActivity
def test_mass_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = ActivityIdealMass()
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_mass_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityIdealMass()
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.16666667, 0.33333333, 0.5])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4,
    )


def test_kappa_parameter_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityKappaParameter(
        kappa=np.array([0.1, 0.2, 0.3]),
        density=np.array([1000.0, 2000.0, 3000.0]),
        molar_mass=np.array([1.0, 2.0, 3.0]),
        water_index=0,
    )
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.66666667, 0.33333333, 0.33333333])
    result = activity_strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected_activity, rtol=1e-6)


def test_kappa_parameter_activity_multi_particles():
    """Test activity calculation for multiple particles."""
    activity_strategy = ActivityKappaParameter(
        kappa=np.array([0.0, 0.5]),
        density=np.array([1000.0, 2000.0]),
        molar_mass=np.array([18.0e-3, 200.0e-3]),
        water_index=0,
    )
    mass_concentration = np.array(
        [[100.0, 100.0], [500.0, 100.0], [100.0, 500.0]]
    )
    expected_activity = np.array(
        [[0.8, 0.08256881], [0.95238095, 0.01768173], [0.44444444, 0.31034483]]
    )
    result = activity_strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected_activity, rtol=1e-6)
