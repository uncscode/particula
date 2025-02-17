"""
Test of the self_broadcast module in the util package.
"""

import numpy as np

from particula.util.self_broadcast import (
    get_pairwise_sum_matrix,
    get_pairwise_diff_matrix,
    get_pairwise_max_matrix,
)


def test_get_pairwise_sum_matrix():
    """
    Test get_pairwise_sum_matrix with a small array input.
    """
    input_array = np.array([1, 2, 3], dtype=np.float64)

    expected_output = np.array(
        [[2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float64
    )

    result = get_pairwise_sum_matrix(input_array)

    assert (
        result.shape == expected_output.shape
    ), f"Expected shape {expected_output.shape}, got {result.shape}"
    assert np.allclose(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


def test_get_pairwise_diff_matrix():
    """
    Test get_pairwise_diff_matrix with a small array input.
    """
    input_array = np.array([1, 2, 3], dtype=np.float64)

    expected_output = np.array(
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float64
    )

    result = get_pairwise_diff_matrix(input_array)

    assert (
        result.shape == expected_output.shape
    ), f"Expected shape {expected_output.shape}, got {result.shape}"
    assert np.allclose(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


def test_get_pairwise_max_matrix():
    """
    Test get_pairwise_max_matrix with a small array input.
    """
    input_array = np.array([1, 2, 3], dtype=np.float64)

    expected_output = np.array(
        [[1, 2, 3], [2, 2, 3], [3, 3, 3]], dtype=np.float64
    )

    result = get_pairwise_max_matrix(input_array)

    assert (
        result.shape == expected_output.shape
    ), f"Expected shape {expected_output.shape}, got {result.shape}"
    assert np.allclose(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


def test_pairwise_functions_edge_cases():
    """
    Test edge cases including empty arrays and single-element arrays.
    """
    empty_array = np.array([], dtype=np.float64)
    single_element_array = np.array([5], dtype=np.float64)

    # Test empty array
    assert get_pairwise_sum_matrix(empty_array).size == 0
    assert get_pairwise_diff_matrix(empty_array).size == 0
    assert get_pairwise_max_matrix(empty_array).size == 0

    # Test single-element array
    assert np.allclose(
        get_pairwise_sum_matrix(single_element_array), np.array([[10.0]])
    )
    assert np.allclose(
        get_pairwise_diff_matrix(single_element_array), np.array([[0.0]])
    )
    assert np.allclose(
        get_pairwise_max_matrix(single_element_array), np.array([[5.0]])
    )


def test_pairwise_functions_large_array():
    """
    Test that the functions work correctly for a larger array.
    """
    input_array = np.arange(10, dtype=np.float64)  # [0, 1, 2, ..., 9]

    sum_matrix = get_pairwise_sum_matrix(input_array)
    diff_matrix = get_pairwise_diff_matrix(input_array)
    max_matrix = get_pairwise_max_matrix(input_array)

    assert sum_matrix.shape == (10, 10)
    assert diff_matrix.shape == (10, 10)
    assert max_matrix.shape == (10, 10)

    # Check specific expected values
    assert sum_matrix[0, 9] == 9  # 0 + 9
    assert diff_matrix[0, 9] == -9  # 0 - 9
    assert max_matrix[0, 9] == 9  # max(0, 9)
    assert sum_matrix[5, 5] == 10  # 5 + 5
    assert diff_matrix[5, 5] == 0  # 5 - 5
    assert max_matrix[5, 5] == 5  # max(5, 5)
