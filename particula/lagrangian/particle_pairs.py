"""Lagrangian particle pairwise distances and pairwise operations."""

import torch


def remove_duplicates(
        index_pairs: torch.Tensor,
        index_to_remove: int
) -> torch.Tensor:
    """
    Removes duplicate entries from a specified column in a tensor of index
    pairs.

    This function is designed to work with tensors where each row represents a
    pair of indices. It removes rows containing duplicate entries in the
    specified column.

    Parameters:
    - index_pairs (torch.Tensor): A 2D tensor of shape [n, 2], where n is the
        number of index pairs.
    - index_to_remove (int): The column index (0 or 1) from which to remove
        duplicate entries.

    Returns:
    - torch.Tensor: A 2D tensor of index pairs with duplicates removed from
        the specified column.

    Example:
    index_pairs = torch.tensor([[1, 2], [3, 4], [1, 2]])
    index_to_remove = 0
    # Output will be [[1, 2], [3, 4]] assuming column 0 is chosen for removing
        duplicates.
    """
    # Sort index_pairs by the index_to_remove column
    sorted_index_pairs, indices_sorted = torch.sort(
        index_pairs[:, index_to_remove], dim=0)

    # Find unique entries in the index_to_remove column
    # diff_index is True for unique entries
    diff_index = torch.diff(sorted_index_pairs, prepend=torch.tensor([-1])) > 0

    # Select rows from index_pairs that correspond to unique entries
    clean_index = indices_sorted[diff_index]
    return index_pairs[clean_index]


def calculate_pairwise_distance(position: torch.Tensor) -> torch.Tensor:
    """
    need to test this:

    Calculate the pairwise Euclidean distances between points in a given
    position tensor.

    This function computes the pairwise distances between points represented
    in the input tensor. Each row of the input tensor is considered a point in
    n-dimensional space.

    Parameters:
    position (torch.Tensor): A 2D tensor of shape [n_dimensions, n_points]

    Returns:
    torch.Tensor: A 2D tensor of shape [n_points, n_points] containing the
    pairwise Euclidean distances between each pair of points.
    The element at [i, j] in the output tensor represents the distance
    between the i-th and j-th points in the input tensor.

    Example:
    position = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Output will be a 3x3 tensor with the pairwise distances between these
    3 points.
    """
    # Expand position tensor to compute pairwise differences
    detla_position = position.unsqueeze(2) - position.unsqueeze(1)
    # Compute pairwise Euclidean distances
    return torch.sqrt(torch.sum(detla_position**2, dim=0))
