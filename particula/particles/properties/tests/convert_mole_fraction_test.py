"""Test the function that converts mole fractions to mass fractions."""

from math import isclose

import numpy as np
import pytest

# Import the function you want to test
from particula.particles.properties.convert_mole_fraction import (
    get_mass_fractions_from_moles,
)


@pytest.mark.parametrize(
    "mole_fractions, molecular_weights, expected",
    [
        # 1D example: typical valid data
        (
            np.array([0.2, 0.5, 0.3]),
            np.array([18.0, 44.0, 28.0]),
            "check_sums_to_one",  # We'll check the result in the test
        ),
        # Edge case: all zero mole fractions => should return all zeros
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([0.0, 0.0, 0.0]),
        ),
        # Another 1D with a quick numeric check
        (
            np.array([0.3, 0.7]),
            np.array([18.0, 2.0]),
            "manual_check",  # We'll do an approximate check in the test
        ),
    ],
)
def test_get_mass_fractions_from_moles_1d(
    mole_fractions, molecular_weights, expected
):
    """Test 1D inputs with:
    - typical valid data
    - all zeros
    - approximate numeric check.
    """
    result = get_mass_fractions_from_moles(mole_fractions, molecular_weights)

    assert result.shape == mole_fractions.shape, (
        "Result shape must match input shape."
    )

    # Handle special expected-value cases
    if isinstance(expected, np.ndarray):
        # Direct numeric comparison
        assert np.allclose(result, expected), (
            f"Expected {expected}, got {result}"
        )
    elif expected == "check_sums_to_one":
        # Check sum to ~1
        assert isclose(result.sum(), 1.0, rel_tol=1e-7), (
            f"Sum of mass fractions should be 1, got {result.sum()}"
        )
        # Also ensure nonnegative
        assert np.all(result >= 0), (
            f"Mass fractions must be nonnegative: got {result}"
        )
    elif expected == "manual_check":
        # We can do a quick approximate check
        # For x = [0.3, 0.7], M = [18, 2],
        # total_mass = 0.3*18 + 0.7*2 = 5.4 + 1.4 = 6.8
        # w1 = (0.3*18)/6.8 = 5.4/6.8 = 0.7941..., w2 = 0.2059...
        # They should sum to ~1
        assert np.allclose(result.sum(), 1.0), (
            f"Sum of mass fractions should be 1, got {result.sum()}"
        )
        assert np.allclose(result[0], 5.4 / 6.8, rtol=1e-5), (
            f"First component mismatch, expected ~0.7941, got {result[0]}"
        )
        assert np.allclose(result[1], 1 - (5.4 / 6.8), rtol=1e-5), (
            f"Second component mismatch, expected ~0.2059, got {result[1]}"
        )


def test_get_mass_fractions_from_moles_2d():
    """Test 2D inputs with row-by-row mole fraction conversion."""
    # Example 2D input: each row has 3 components
    x_2d = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.3, 0.3, 0.4],
            [0.0, 0.0, 0.0],  # Edge case: all zero
        ]
    )
    mw_2d = np.array([18.0, 44.0, 28.0])

    result = get_mass_fractions_from_moles(x_2d, mw_2d)

    assert result.shape == x_2d.shape, (
        "Output shape must match the input shape."
    )

    # Row 1 (nonzero): sum should be ~1
    assert np.isclose(result[0].sum(), 1.0, rtol=1e-7), (
        f"Row 1 sum != 1, got {result[0].sum()}"
    )

    # Row 2 (nonzero): sum should be ~1
    assert np.isclose(result[1].sum(), 1.0, rtol=1e-7), (
        f"Row 2 sum != 1, got {result[1].sum()}"
    )

    # Row 3 (all zeros input): should be all zeros
    assert np.allclose(result[2], [0.0, 0.0, 0.0]), (
        f"Row 3 should all be zeros, got {result[2]}"
    )


def test_dimension_mismatch():
    """Test that the function raises an error when shapes are incompatible."""
    x_1d = np.array([0.2, 0.3, 0.5])
    mw_1d_wrong_length = np.array([18.0, 44.0])  # only 2 elements

    with pytest.raises(ValueError) as excinfo:
        get_mass_fractions_from_moles(x_1d, mw_1d_wrong_length)

    assert (
        "shapes" in str(excinfo.value).lower()
        or "dimension" in str(excinfo.value).lower()
    ), "Expected shape/dimension mismatch error."


def test_3d_input_raises():
    """Test that passing a 3D array raises an error."""
    x_3d = np.zeros((2, 2, 2))
    mw_1d = np.array([1.0, 2.0])

    with pytest.raises(ValueError) as excinfo:
        get_mass_fractions_from_moles(x_3d, mw_1d)

    assert "1D or 2D" in str(excinfo.value), "Expected an error for 3D inputs."
