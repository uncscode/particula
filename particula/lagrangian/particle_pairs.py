"""Lagrangian particle pairwise distances and pairwise operations."""

from typing import Tuple
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

    Args:
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

    Args:
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


def validate_pair_distance(
    collision_indices_pairs: torch.Tensor,
    position: torch.Tensor,
    radius: torch.Tensor
) -> torch.Tensor:
    """
    Validates if the Euclidean distances between pairs of points are smaller
    than the sum of their radii.

    Args:
        collision_indices_pairs (torch.Tensor): A tensor containing pairs of
            indices of potentially colliding particles.
        position (torch.Tensor): A 2D tensor of particle positions, where each
            column represents a particle, and each row represents an axis.
        radius (torch.Tensor): A 1D tensor representing the radius of each
            particle.

    Returns:
        torch.Tensor: A tensor containing the indices of the pairs of
            particles that are actually colliding.
    """
    # Fast return if there are no particle pairs
    if collision_indices_pairs.numel() == 0:
        return torch.tensor([], dtype=torch.bool)

    # Calculate 3D distance for each pair of particles
    delta_position = position[:, collision_indices_pairs[:, 0]] \
        - position[:, collision_indices_pairs[:, 1]]
    # Euclidean distance between particles
    distance = torch.sqrt(torch.sum(delta_position**2, axis=0))
    # radius sum of both particles
    distance_threshold = radius[collision_indices_pairs[:, 0]] \
        + radius[collision_indices_pairs[:, 1]]

    # Return the pairs of particles where the distance is less than the sum of
    # their radii
    return collision_indices_pairs[distance < distance_threshold]


def single_axis_sweep_and_prune(
    position_axis: torch.Tensor,
    radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sweep and prune algorithm for collision detection along a single axis.
    This function identifies pairs of particles that are close enough to
    potentially collide.

    Args:
        position_axis (torch.Tensor): The position of particles along a single
            axis.
        radius (torch.Tensor): The radius of particles.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors containing the indices
        of potentially colliding particles.
    """

    # Fast return if there are no particles
    if position_axis.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64), torch.tensor(
            [], dtype=torch.int64)

    # Apply sweep and prune to find pairs of particles that are close enough
    # to collide
    sweep = torch.sort(position_axis)
    sweep_diff = torch.diff(sweep.values)
    radius_sum = radius[sweep.indices[:-1]] + radius[sweep.indices[1:]]

    # Select indices of particles that are close enough to collide
    prune_bool = sweep_diff < radius_sum
    left_overlap_indices = sweep.indices[torch.cat(
        [prune_bool, torch.tensor([False], dtype=torch.bool)])]
    right_overlap_indices = sweep.indices[torch.cat(
        [torch.tensor([False], dtype=torch.bool), prune_bool])]

    return left_overlap_indices, right_overlap_indices


# pylint: disable=too-many-locals
def full_sweep_and_prune(
        position: torch.Tensor,
        radius: torch.Tensor
) -> torch.Tensor:
    """
    Sweep and prune algorithm for collision detection along all three axes
    (x, y, z). This function identifies pairs of particles that are close
    enough to potentially collide in 3D space.

    Args:
        position (torch.Tensor): The 2D tensor of particle positions,
            where each row represents an axis (x, y, z).
        radius (torch.Tensor): The radius of particles.

    Returns:
        torch.Tensor: A tensor containing pairs of indices of potentially
            colliding particles.
    """
    # select only particles with positive radius
    valid_radius = radius > 0
    valid_radius_indices = torch.arange(radius.shape[0])[valid_radius]
    # sweep x axis
    left_x_overlap_shifted, right_x_overlap_shifted = \
        single_axis_sweep_and_prune(
            position_axis=position[0, valid_radius_indices],
            radius=radius[valid_radius_indices]
        )
    # fast return if there are no particles overlapping in x
    if left_x_overlap_shifted.shape[0] == 0:
        return torch.tensor([])
    # unshift from relative valid radius to position index
    left_x_overlap_indices = valid_radius_indices[left_x_overlap_shifted]
    right_x_overlap_indices = valid_radius_indices[right_x_overlap_shifted]

    # cobine left and right indices for next step
    all_overlaps_x = torch.cat(
        [left_x_overlap_indices, right_x_overlap_indices])
    # select unique indices
    indices_x_unique = torch.unique(all_overlaps_x)

    # sweep y axis
    left_y_overlap_shifted, right_y_overlap_shifted = \
        single_axis_sweep_and_prune(
            position_axis=position[1][indices_x_unique],
            radius=radius[indices_x_unique]
        )
    # fast return if there are no particles overlapping in y
    if left_y_overlap_shifted.shape[0] == 0:
        return torch.tensor([])
    # unshift from x relative index to position index
    left_y_overlap_indices = indices_x_unique[left_y_overlap_shifted]
    right_y_overlap_indices = indices_x_unique[right_y_overlap_shifted]

    # combine left and right indices for next step
    all_overlaps_y = torch.cat(
        [left_y_overlap_indices, right_y_overlap_indices])
    # select unique indices
    indices_y_unique = torch.unique(all_overlaps_y)

    # sweep z axis
    left_z_overlap_shifted, right_z_overlap_shifted = \
        single_axis_sweep_and_prune(
            position_axis=position[2][indices_y_unique],
            radius=radius[indices_y_unique]
        )
    # fast return if there are no particles overlapping in z
    if left_z_overlap_shifted.shape[0] == 0:
        return torch.tensor([])
    # unshift from y relative index to position index
    left_z_overlap_indices = indices_y_unique[left_z_overlap_shifted]
    right_z_overlap_indices = indices_y_unique[right_z_overlap_shifted]

    return torch.cat(
        [
            left_z_overlap_indices.unsqueeze(1),
            right_z_overlap_indices.unsqueeze(1),
        ],
        dim=1,
    )


def full_sweep_and_prune_simplified(
    position: torch.Tensor,
    radius: torch.Tensor,
    working_yet: bool = False
) -> torch.Tensor:
    """
    A simplified version of the full sweep and prune algorithm for collision
    written above, it is not working yet. there is an error in the update of
    the indices in the y and z axis.

    Sweep and prune algorithm for collision detection along all three axes
    (x, y, z). This function identifies pairs of particles that are close
    enough to potentially collide in 3D space.

    Args:
        position (torch.Tensor): The 2D tensor of particle positions,
            where each row represents an axis (x, y, z).
        radius (torch.Tensor): The radius of particles.

    Returns:
        torch.Tensor: A tensor containing pairs of indices of potentially
            colliding particles.
    """
    if not working_yet:
        raise NotImplementedError
    if radius.shape[0] == 0:
        return torch.tensor([])

    valid_indices = torch.arange(radius.shape[0])[radius > 0]
    unique_indices = valid_indices

    for axis in range(position.shape[0]):
        if unique_indices.shape[0] == 0:
            break

        left_overlap_indices, right_overlap_indices = \
            single_axis_sweep_and_prune(
                position_axis=position[axis, unique_indices],
                radius=radius[unique_indices]
            )

        if left_overlap_indices.shape[0] == 0:
            return torch.tensor([])

        # Combine indices to form collision pairs, may still have duplicates
        all_overlaps = torch.cat(
            [unique_indices[left_overlap_indices],
             unique_indices[right_overlap_indices]])
        if axis < position.shape[0]:  # not last axis
            unique_indices = torch.unique(all_overlaps)  # remove duplicates

    return torch.cat(
        [
            unique_indices[left_overlap_indices].unsqueeze(1),
            unique_indices[right_overlap_indices].unsqueeze(1),
        ],
        dim=1,
    )
