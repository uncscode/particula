"""Tests for validation helpers and the validate_inputs decorator."""

import numpy as np
import pytest
from particula.util.validate_inputs import (
    validate_finite,
    validate_inputs,
    validate_negative,
    validate_nonnegative,
    validate_nonpositive,
    validate_nonzero,
    validate_positive,
)

# Define a sample function to test the decorator
some_dict = {
    "x": "positive",
    "y": "negative",
    "z": "nonpositive",
    "w": "nonnegative",
    "v": "nonzero",
}


@validate_inputs(some_dict)
def sample_function(x, y, z, w, v):
    """A sample function to test the decorator."""
    return x, y, z, w, v


def test_validate_positive():
    """Raises an error when a positive argument is negative."""
    with pytest.raises(ValueError, match="Argument 'x' must be positive."):
        sample_function(x=-1, y=-1, z=0, w=0, v=1)


def test_validate_positive_helper_accepts_positive_arrays():
    """Positive helper should allow positive array values."""
    validate_positive(np.array([1.0, 2.0]), "x")


def test_validate_negative():
    """Raises an error when a negative argument is positive."""
    with pytest.raises(ValueError, match="Argument 'y' must be negative."):
        sample_function(1, 1, 0, 0, 1)


def test_validate_negative_helper_accepts_negative_arrays():
    """Negative helper should allow negative array values."""
    validate_negative(np.array([-1.0, -2.0]), "y")


def test_validate_nonpositive():
    """Raises an error when a nonpositive argument is positive."""
    with pytest.raises(ValueError, match="Argument 'z' must be nonpositive."):
        sample_function(1, -1, 1, 0, 1)


def test_validate_nonpositive_helper_accepts_zero_and_negative_values():
    """Nonpositive helper should allow zero and negative values."""
    validate_nonpositive(np.array([0.0, -1.0]), "z")


def test_validate_nonnegative():
    """Raises an error when a nonnegative argument is negative."""
    with pytest.raises(ValueError, match="Argument 'w' must be nonnegative."):
        sample_function(1, -1, 0, -1, 1)


def test_validate_nonnegative_helper_accepts_zero_and_positive_values():
    """Nonnegative helper should allow zero and positive values."""
    validate_nonnegative(np.array([0.0, 1.0]), "w")


def test_validate_nonzero():
    """Raises an error when a nonzero argument is zero."""
    with pytest.raises(ValueError, match="Argument 'v' must be nonzero."):
        sample_function(1, -1, 0, 0, 0)


def test_validate_nonzero_helper_accepts_nonzero_values():
    """Nonzero helper should allow nonzero scalar and array values."""
    validate_nonzero(np.array([-1.0, 1.0]), "v")


def test_valid_inputs():
    """Raise an error when all arguments are valid."""
    assert sample_function(1, -1, 0, 0, 1) == (1, -1, 0, 0, 1)


def test_valid_finite():
    """Raises an error when a finite argument is infinite."""
    some_dict2 = {"x": "finite"}

    @validate_inputs(some_dict2)
    def sample_function2(x):
        """A sample function to test the decorator."""
        return x

    with pytest.raises(
        ValueError,
        match=r"Argument 'x' must be finite \(no inf or NaN\).",
    ):
        sample_function2(np.nan)


def test_validate_finite_helper_accepts_finite_values():
    """Finite helper should allow finite scalar and array values."""
    validate_finite(np.array([0.0, 1.0]), "x")


def test_none_skips_validation():
    """Test that None values skip validation."""
    some_dict3 = {"x": "positive"}

    @validate_inputs(some_dict3)
    def sample_function3(x=None):
        """Function that allows None input."""
        return x

    assert sample_function3(None) is None


def test_missing_required_argument_preserves_signature_error():
    """Missing required args should raise the wrapped function signature error."""
    with pytest.raises(TypeError, match="missing.*required"):
        sample_function(1, -1, 0, 0)


def test_default_argument_is_validated_when_not_explicitly_passed():
    """Defaulted arguments should still participate in validation."""

    @validate_inputs({"x": "positive"})
    def sample_function4(x=0):
        """Function with an invalid default value for validation."""
        return x

    with pytest.raises(ValueError, match="Argument 'x' must be positive."):
        sample_function4()


def test_default_argument_can_pass_validation():
    """Valid defaults should still allow successful calls."""

    @validate_inputs({"x": "positive"})
    def sample_function5(x=1):
        """Function with a valid default value for validation."""
        return x

    assert sample_function5() == 1


def test_unknown_validation_raises_clear_error():
    """Unknown validation names should raise a descriptive ValueError."""

    @validate_inputs({"x": "mystery"})
    def sample_function6(x):
        """Function with an unsupported validator."""
        return x

    with pytest.raises(
        ValueError,
        match="Unknown validation 'mystery' for argument 'x'.",
    ):
        sample_function6(1)
