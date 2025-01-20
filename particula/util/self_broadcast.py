"""
Self broadcasting operation to generate a pairwise sum matrix.
"""

import numpy as np
from numpy.typing import NDArray


def get_pairwise_sum_matrix(
    input_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the pairwise sum matrix for a given 1D NumPy array.

    This function performs a self-broadcasting operation to generate a matrix
    where each element (i, j) is computed as:

        output_matrix[i, j] = input_array[i] + input_array[j]

    Arguments:
    ----------
        - input_array : A 1D NumPy array of values.

    Returns:
    --------
        - A 2D NumPy array where each element is the sum of two elements
          from the input array.

    Example:
    --------
    ``` py title="Example"
    input_array = np.array([1, 2, 3])
    get_pairwise_sum_matrix(input_array)
    array([[2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])
    ```
    """
    return input_array[:, np.newaxis] + input_array[np.newaxis, :]


def get_pairwise_diff_matrix(
    input_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the pairwise difference matrix for a given 1D NumPy array.

    This function performs a self-broadcasting operation to generate a matrix
    where each element (i, j) is computed as:

        output_matrix[i, j] = input_array[i] - input_array[j]

    Arguments:
    ----------
        - input_array : A 1D NumPy array of values.

    Returns:
    --------
        - A 2D NumPy array where each element is the difference of two
          elements from the input array.

    Example:
    --------
    ``` py title="Example"
    input_array = np.array([1, 2, 3])
    get_pairwise_diff_matrix(input_array)
    array([[ 0, -1, -2],
           [ 1,  0, -1],
           [ 2,  1,  0]])
    ```
    """
    return input_array[:, np.newaxis] - input_array[np.newaxis, :]


def get_pairwise_max_matrix(
    input_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the pairwise maximum matrix for a given 1D NumPy array.

    This function performs a self-broadcasting operation to generate a matrix
    where each element (i, j) is computed as:

        output_matrix[i, j] = max(input_array[i], input_array[j])

    Arguments:
    ----------
        - input_array : A 1D NumPy array of values.

    Returns:
    --------
        - A 2D NumPy array where each element is the maximum of two
          elements from the input array.

    Example:
    --------
    ``` py title="Example"
    input_array = np.array([1, 2, 3])
    get_pairwise_max_matrix(input_array)
    array([[1, 2, 3],
           [2, 2, 3],
           [3, 3, 3]])
    ```
    """
    return np.maximum(input_array[:, np.newaxis], input_array[np.newaxis, :])
