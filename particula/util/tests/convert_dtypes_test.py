"""Unit tests for the convert_dtypes module."""

import numpy as np
import pytest

from particula.util.convert_dtypes import (
    get_coerced_type,
    get_dict_from_list,
    get_shape_check,
    get_values_of_dict,
)


def test_get_coerced_type_success():
    """Scalar and array coercion succeed."""
    assert get_coerced_type(1, float) == 1.0
    array = get_coerced_type([1, 2, 3], np.ndarray)
    assert isinstance(array, np.ndarray)
    assert np.array_equal(array, np.array([1, 2, 3]))
    arr = np.array([1])
    # already ndarray should return same object
    assert get_coerced_type(arr, np.ndarray) is arr


def test_get_coerced_type_failure():
    """Invalid coercion raises ValueError."""
    with pytest.raises(ValueError):
        get_coerced_type("abc", int)


def test_get_dict_from_list_valid():
    """List of strings converted to expected dict."""
    assert get_dict_from_list(["a", "b", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_get_dict_from_list_invalid():
    """Empty or non-string lists raise AssertionError."""
    with pytest.raises(TypeError):
        get_dict_from_list([])
    with pytest.raises(TypeError):
        get_dict_from_list(["a", 1])


def test_get_values_of_dict_success_and_failure():
    """Extract values by keys or raise for missing key."""
    mapping = {"a": 1, "b": 2, "c": 3}
    assert get_values_of_dict(["b", "a"], mapping) == [2, 1]
    with pytest.raises(KeyError):
        get_values_of_dict(["d"], mapping)


def test_get_shape_check_1d():
    """1-D data are expanded to 2-D when header has one item."""
    time = np.arange(3)
    data = np.array([1, 2, 3])
    result = get_shape_check(time, data, ["x"])
    assert result.shape == (3, 1)
    assert np.array_equal(result[:, 0], data)
    with pytest.raises(ValueError):
        get_shape_check(time, data, ["x", "y"])


def test_get_shape_check_2d_and_header():
    """2-D data reshaping and header validation."""
    time = np.arange(3)
    data = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
    reshaped = get_shape_check(time, data, ["a", "b"])
    assert reshaped.shape == (3, 2)
    assert np.array_equal(reshaped, np.array([[1, 4], [2, 5], [3, 6]]))
    with pytest.raises(ValueError):
        get_shape_check(time, np.ones((3, 2)), ["only_one"])
    with pytest.raises(ValueError):
        get_shape_check(time, np.ones((3, 3)), ["a", "b"])


def test_get_shape_check_numpy2_compatibility():
    """NumPy 2.0 scalar conversion uses Python int axis."""
    time = np.arange(5)
    data = np.ones((3, 5))
    reshaped = get_shape_check(time, data, ["a", "b", "c"])
    assert reshaped.shape == (5, 3)
    assert np.array_equal(reshaped, np.ones((5, 3)))

    square = np.ones((5, 5))
    square_result = get_shape_check(time, square, list("abcde"))
    assert square_result.shape == (5, 5)
    assert np.array_equal(square_result, square)
