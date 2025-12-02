"""Test of overflow and underflow safe functions."""

import warnings

import numpy as np

from particula.util.machine_limit import (
    get_safe_exp,
    get_safe_log,
    get_safe_log10,
    get_safe_power,
)


def test_safe_exp():
    """Test safe_exp function."""
    # Test with positive values
    assert np.allclose(get_safe_exp([1, 2, 3]), np.exp([1, 2, 3]))

    # Test with negative values
    assert np.allclose(get_safe_exp([-1, -2, -3]), np.exp([-1, -2, -3]))

    # Test with large values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(
            get_safe_exp([1000, 2000, 3000]), np.exp([1000, 2000, 3000])
        )
        assert any(item.category == RuntimeWarning for item in w)


def test_safe_log():
    """Test safe_log function."""
    # Test with positive values
    assert np.allclose(get_safe_log([1, 2, 3]), np.log([1, 2, 3]))

    # Test with zero values
    assert np.allclose(
        get_safe_log([0, 0, 0]),
        np.log([np.nextafter(0, 1), np.nextafter(0, 1), np.nextafter(0, 1)]),
    )

    # Test with negative values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(get_safe_log([-1, -2, -3]), np.log([-1, -2, -3]))
        assert any(item.category == RuntimeWarning for item in w)


def test_safe_log10():
    """Test safe_log10 function."""
    # Test with positive values
    assert np.allclose(get_safe_log10([1, 2, 3]), np.log10([1, 2, 3]))

    # Test with zero values
    assert np.allclose(
        get_safe_log10([0, 0, 0]),
        np.log10([np.nextafter(0, 1), np.nextafter(0, 1), np.nextafter(0, 1)]),
    )

    # Test with negative values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(
            get_safe_log10([-1, -2, -3]), np.log10([-1, -2, -3])
        )
        assert any(item.category == RuntimeWarning for item in w)


def test_safe_power():
    """Test get_safe_power function."""
    # Test with typical positive values:
    base = np.array([2, 3, 10])
    exponent = np.array([3, 2, 1])
    # For valid positive inputs, get_safe_power should match np.power.
    assert np.allclose(get_safe_power(base, exponent), np.power(base, exponent))

    # Test for overflow protection:
    # Compute the safe exponent limit for base 10.
    max_exp_input = np.log(np.finfo(np.float64).max)
    safe_exponent = max_exp_input / np.log(10)
    # Use an exponent slightly above the safe limit.
    base_overflow = np.array([10])
    exponent_overflow = np.array([safe_exponent + 1])
    result_overflow = get_safe_power(base_overflow, exponent_overflow)
    expected_overflow = np.array([np.finfo(np.float64).max])
    assert np.allclose(result_overflow, expected_overflow)

    # Test with zero base:
    # raise value error if base is zero and exponent is negative.
    with np.testing.assert_raises(ValueError):
        get_safe_power(0, -1)

    # Test with negative base:
    # raise value error if base is negative and exponent is not an integer.
    with np.testing.assert_raises(ValueError):
        get_safe_power(-1, 0.5)


def test_safe_log_edge_cases():
    """Edge cases for get_safe_log."""
    # Extremely small positive values should return accurate logs.
    small_values = np.array([np.nextafter(0, 1), 1e-300, 1e-40])
    assert np.allclose(get_safe_log(small_values), np.log(small_values))

    # Negative inputs clip to smallest positive without warnings.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_safe_log([-1, -10])
        assert len(w) == 0
    expected = np.log(np.nextafter(0, 1, dtype=np.float64))
    assert np.allclose(result, np.full(2, expected))


def test_safe_log10_edge_cases():
    """Edge cases for get_safe_log10."""
    # Extremely small positive values should return accurate base-10 logs.
    small_values = np.array([np.nextafter(0, 1), 1e-300, 1e-40])
    assert np.allclose(get_safe_log10(small_values), np.log10(small_values))

    # Negative inputs clip to smallest positive without warnings.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = get_safe_log10([-1, -10])
        assert len(w) == 0
    expected = np.log10(np.nextafter(0, 1, dtype=np.float64))
    assert np.allclose(result, np.full(2, expected))


def test_safe_power_edge_cases():
    """Edge cases for get_safe_power."""
    # Very large exponent should clip to machine max
    result_large = get_safe_power(2, 1e308)
    assert np.allclose(result_large, np.array(np.finfo(np.float64).max))

    # Negative base always raises ValueError
    with np.testing.assert_raises(ValueError):
        get_safe_power(-2, 3)

    max_exp_input = np.log(np.finfo(np.float64).max)
    safe_exponent = max_exp_input / np.log(10)

    # Exponent just below overflow threshold should match numpy power
    near_overflow = safe_exponent - 1
    assert np.allclose(
        get_safe_power(10, near_overflow), np.power(10, near_overflow)
    )

    # Exponent well below underflow threshold should underflow to zero
    result_under = get_safe_power(0.5, 1e308)
    assert result_under == 0.0
