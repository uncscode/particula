""" This module contains tests for the validate_inputs decorator.
"""

import pytest

from particula.util.validate_inputs import validate_inputs

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
    """ A sample function to test the decorator """
    return x, y, z, w, v


def test_validate_positive():
    """ Test that the decorator raises an error when a positive argument is negative """
    with pytest.raises(ValueError, match="Argument 'x' must be positive."):
        sample_function(x=-1, y=-1, z=0, w=0, v=1)


def test_validate_negative():
    """ Test that the decorator raises an error when a negative argument is positive """
    with pytest.raises(ValueError, match="Argument 'y' must be negative."):
        sample_function(1, 1, 0, 0, 1)


def test_validate_nonpositive():
    """ Test that the decorator raises an error when a nonpositive argument is positive """
    with pytest.raises(ValueError, match="Argument 'z' must be nonpositive."):
        sample_function(1, -1, 1, 0, 1)


def test_validate_nonnegative():
    """ Test that the decorator raises an error when a nonnegative argument is negative """
    with pytest.raises(ValueError, match="Argument 'w' must be nonnegative."):
        sample_function(1, -1, 0, -1, 1)


def test_validate_nonzero():
    """ Test that the decorator raises an error when a nonzero argument is zero """
    with pytest.raises(ValueError, match="Argument 'v' must be nonzero."):
        sample_function(1, -1, 0, 0, 0)


def test_valid_inputs():
    """" Test that the decorator does not raise an error when all arguments are valid """
    assert sample_function(1, -1, 0, 0, 1) == (1, -1, 0, 0, 1)
