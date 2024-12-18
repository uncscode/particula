"""Test of overflow and underflow safe functions."""

import numpy as np
import warnings
from particula.util.machine_limit import safe_exp, safe_log, safe_log10


def test_safe_exp():
    """test safe_exp function."""
    # Test with positive values
    assert np.allclose(safe_exp([1, 2, 3]), np.exp([1, 2, 3]))

    # Test with negative values
    assert np.allclose(safe_exp([-1, -2, -3]), np.exp([-1, -2, -3]))

    # Test with large values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(
            safe_exp([1000, 2000, 3000]), np.exp([1000, 2000, 3000])
        )
        assert any(item.category == RuntimeWarning for item in w)


def test_safe_log():
    """test safe_log function."""
    # Test with positive values
    assert np.allclose(safe_log([1, 2, 3]), np.log([1, 2, 3]))

    # Test with zero values
    assert np.allclose(
        safe_log([0, 0, 0]),
        np.log([np.nextafter(0, 1), np.nextafter(0, 1), np.nextafter(0, 1)]),
    )

    # Test with negative values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(safe_log([-1, -2, -3]), np.log([-1, -2, -3]))
        assert any(item.category == RuntimeWarning for item in w)


def test_safe_log10():
    """test safe_log10 function."""
    # Test with positive values
    assert np.allclose(safe_log10([1, 2, 3]), np.log10([1, 2, 3]))

    # Test with zero values
    assert np.allclose(
        safe_log10([0, 0, 0]),
        np.log10([np.nextafter(0, 1), np.nextafter(0, 1), np.nextafter(0, 1)]),
    )

    # Test with negative values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not np.allclose(safe_log10([-1, -2, -3]), np.log10([-1, -2, -3]))
        assert any(item.category == RuntimeWarning for item in w)
