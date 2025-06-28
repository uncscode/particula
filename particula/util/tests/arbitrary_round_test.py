"""Test the round module."""

import numpy as np

from particula.util.arbitrary_round import get_arbitrary_round


def test_round_arbitrary():
    """Test the round function."""
    # Test single float value
    assert get_arbitrary_round(3.14, base=0.5, mode="round") == 3.0

    # Test list of float values
    assert np.array_equiv(
        get_arbitrary_round([1.2, 2.5, 3.8], base=1.0, mode="round"),
        np.array([1.0, 2.0, 4.0]),
    )

    # Test NumPy array of float values
    assert np.array_equal(
        get_arbitrary_round(np.array([1.2, 2.5, 3.8]), base=1.0, mode="floor"),
        np.array([1.0, 2.0, 3.0]),
    )

    # Test NumPy array of ceil values
    assert np.array_equal(
        get_arbitrary_round(np.array([1.2, 2.5, 3.8]), base=1.0, mode="ceil"),
        np.array([2.0, 3.0, 4.0]),
    )

    # Test rounding to non-integer base
    assert get_arbitrary_round(3.14, base=0.1, mode="round") == 3.1

    # Test rounding mode "round_nonzero"
    assert np.array_equal(
        get_arbitrary_round(
            [0, 0.2, 0.3, 0.6, 3], base=1.0, mode="round", nonzero_edge=True
        ),
        np.array([0, 0.2, 0.3, 1, 3]),
    )

    # Test invalid mode parameter
    try:
        get_arbitrary_round([1.2, 2.5, 3.8], base=1.0, mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
