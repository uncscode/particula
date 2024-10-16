""" testing the utility calculating reduced quantities
"""

import numpy as np
import pytest
from particula.util.reduced_quantity import (
    reduced_value, reduced_self_broadcast)


def test_reduced_value_scalar():
    """ Test that the reduced value is calculated correctly."""
    assert reduced_value(4, 2) == 4 * 2 / (4 + 2), "Failed for scalar inputs"


def test_reduced_value_array():
    """ Test that the reduced value is calculated correctly."""
    alpha = np.array([2, 4, 6])
    beta = np.array([3, 6, 9])
    expected = alpha * beta / (alpha + beta)
    result = reduced_value(alpha, beta)
    assert np.allclose(result, expected), "Failed for array inputs"


def test_reduced_value_zero_division():
    """ Test division by zero handling."""
    alpha = np.array([0, 2, 0])
    beta = np.array([0, 0, 2])
    # Expect zeros where division by zero occurs
    expected = np.array([0, 0, 0])
    result = reduced_value(alpha, beta)
    assert np.array_equal(result, expected), "Failed handling division by zero"


def test_reduced_value_shape_mismatch():
    """ Test error handling for shape mismatch."""
    alpha = np.array([1, 2])
    beta = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        reduced_value(alpha, beta)


def test_reduced_value_negative_values():
    """ Test that the reduced value is calculated correctly."""
    alpha = np.array([-1, -2])
    beta = np.array([-3, -4])
    expected = alpha * beta / (alpha + beta)
    result = reduced_value(alpha, beta)
    assert np.allclose(result, expected), "Failed for negative values"


def test_reduced_value_one_element():
    """ Test with one element in array"""
    alpha = np.array([5])
    beta = np.array([10])
    expected = alpha * beta / (alpha + beta)
    result = reduced_value(alpha, beta)
    assert np.allclose(result, expected), "Failed for single element arrays"


def test_reduced_self_broadcast_typical():
    """ Test that the reduced self broadcast is calculated correctly."""
    alpha_array = np.array([1, 2, 3])
    expected_result = np.array([
        [1 * 1 / (1 + 1), 1 * 2 / (1 + 2), 1 * 3 / (1 + 3)],
        [2 * 1 / (2 + 1), 2 * 2 / (2 + 2), 2 * 3 / (2 + 3)],
        [3 * 1 / (3 + 1), 3 * 2 / (3 + 2), 3 * 3 / (3 + 3)]
    ])
    result = reduced_self_broadcast(alpha_array)
    assert np.allclose(
        result, expected_result), "Test failed for typical input"


def test_reduced_self_broadcast_empty():
    """ Test with empty input"""
    alpha_array = np.array([])
    result = reduced_self_broadcast(alpha_array)
    assert result.shape == (0, 0), "Test failed for empty input"


def test_reduced_self_broadcast_zero_elements():
    """ Test with zero elements"""
    alpha_array = np.array([0, 0, 0])
    expected_result = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    result = reduced_self_broadcast(alpha_array)
    assert np.allclose(
        result, expected_result), "Test failed for zero elements input"


def test_reduced_self_broadcast_one_element():
    """ Test with a single element array"""
    alpha_array = np.array([4])
    expected_result = np.array([
        [4 * 4 / (4 + 4)]
    ])
    result = reduced_self_broadcast(alpha_array)
    assert np.allclose(
        result, expected_result), "Test failed for one element input"


def test_reduced_self_broadcast_negative_elements():
    """ Test with negative elements"""
    alpha_array = np.array([-1, -2, -3])
    expected_result = np.array([
        [-1 * -1 / (-1 - 1), -1 * -2 / (-1 - 2), -1 * -3 / (-1 - 3)],
        [-2 * -1 / (-2 - 1), -2 * -2 / (-2 - 2), -2 * -3 / (-2 - 3)],
        [-3 * -1 / (-3 - 1), -3 * -2 / (-3 - 2), -3 * -3 / (-3 - 3)]
    ])
    result = reduced_self_broadcast(alpha_array)
    assert np.allclose(
        result, expected_result), "Test failed for negative elements input"
