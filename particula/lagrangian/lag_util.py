"""A collection of utility functions for the Lagrangian model."""
from typing import Tuple
import numpy as np
import torch


def radius(
        mass: torch.Tensor,
        density: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the radius of a sphere given its mass and density using the
    formula for the volume of a sphere.

    This function assumes a uniform density and spherical shape to compute the
    radius based on the mass-density relationship:
    Volume = Mass / Density. The volume of a sphere is given by
    (4/3) * pi * radius^3.

    Parameters:
    - mass (torch.Tensor): A tensor containing the mass of the sphere(s). Can
        be a scalar or a vector.
    - density (torch.Tensor): A tensor containing the density of the sphere(s).
        Can be a scalar or a vector.

    Returns:
    torch.Tensor: A tensor of the same shape as `mass` and `density`
        representing the radius of the sphere(s).

    Note:
    - The function supports broadcasting, so `mass` and `density` can be of
        different shapes, as long as they are broadcastable to a common shape.
    - Units of mass and density should be consistent to obtain a radius in
        meaningful units.
    """
    return torch.pow(3 * mass / (4 * np.pi * density), 1 / 3)


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


def integration_leapfrog(
        position: torch.Tensor,
        velocity: torch.Tensor,
        force: torch.Tensor,
        mass: torch.Tensor,
        time_step: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a single step of leapfrog integration on the position and velocity
    of a particle.

    Leapfrog integration is a numerical method used for solving differential
    equations typically found in molecular dynamics and astrophysics.
    It is symplectic, hence conserves energy over long simulations,
    and is known for its simple implementation and stability over
    large time steps.

    Parameters:
    - position (Tensor): The current position of the particle.
    - velocity (Tensor): The current velocity of the particle.
    - force (Tensor): The current force acting on the particle.
    - mass (float): The mass of the particle.
    - time_step (float): The time step for the integration.

    Returns:
    - tuple: Updated position and velocity of the particle after one time step.

    Reference:
    - https://en.wikipedia.org/wiki/Leapfrog_integration
    """
    # Half step update for velocity
    velocity += 0.5 * time_step * force / mass
    # Full step update for position
    position += time_step * velocity
    # Another half step update for velocity
    velocity += 0.5 * time_step * force / mass
    return position, velocity


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


def find_collisions(
        distance_matrix: torch.Tensor,
        indices: torch.Tensor,
        mass: torch.Tensor,
        k: int = 1
) -> torch.Tensor:
    """
    Find the collision pairs from a distance matrix, given the mass and
    indices of particles.

    This function identifies pairs of particles that are within a certain
    distance threshold (<0), indicating a collision.
    It filters out pairs involving particles with zero mass.

    Parameters:
    distance_matrix (torch.Tensor): A 2D tensor containing the pairwise
        distances between particles.
    indices (torch.Tensor): A 1D tensor containing the indices of the
        particles.
    mass (torch.Tensor): A 1D tensor containing the mass of each particle.
    k (int, optional): The number of closest neighbors to consider for each
        particle. Defaults to 1.

    Returns:
    torch.Tensor: A 2D tensor of shape [n_collisions, 2] containing the
    indices of colliding pairs of particles.

    Note:
    - The function assumes that the diagonal elements of the distance matrix
    (distances of particles to themselves) are less than zero.
    - Particles with zero mass are excluded from the collision pairs.
    """
    # Find the top k closest particles for each particle
    closest_neighbors = torch.topk(
        distance_matrix,
        k=k + 1,
        largest=False,
        sorted=True)

    # Identify collisions and exclude zero-distance (self) and zero-mass
    # particles
    collisions = (closest_neighbors.values < 0) * (mass.unsqueeze(1) > 0)

    # Prepare tensors for collision pairs
    iteration_index = 1  # index of the closest neighbor, only one neighbor is
    # considered for now, this could be put into a loop for multiple neighbors
    # but loops are bad, and any will just collide in the next iteration
    expanded_indices = indices.unsqueeze(1).repeat(1, k + 1)
    expanded_closest_neighbors = closest_neighbors.indices

    # Combine indices to form collision pairs
    collision_indices_pairs = torch.cat(
        [expanded_closest_neighbors.unsqueeze(2),
         expanded_indices.unsqueeze(2)], dim=2)

    # Extract valid collision pairs
    return collision_indices_pairs[
        collisions[:, iteration_index], iteration_index, :].int()


def coalescence(
        velocity: torch.Tensor,
        mass: torch.Tensor,
        collision_indices_pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update mass and velocity of particles based on collision pairs, conserving
    mass and momentum.

    This function processes collision pairs, sorts them to avoid duplicate
    handling, and then updates the mass and velocity of colliding particles
    according to the conservation of mass and momentum.

    Parameters:
    velocity (torch.Tensor): A 2D tensor of shape [n_dimensions, n_particles]
        representing the velocities of particles.
    mass (torch.Tensor): A 1D tensor containing the mass of each particle.
    collision_indices_pairs (torch.Tensor): A 2D tensor containing pairs of
        indices representing colliding particles.
    remove_duplicates_func (function): A function to remove duplicate entries
        from a tensor of index pairs.

    Returns:
    - torch.Tensor: A 2D tensor of shape [n_dimensions, n_particles]
        representing the updated velocities of particles.
    Note:
    - This function modifies the `velocity` and `mass` tensors in-place.
    - It assumes that the mass and momentum are transferred from the right
        particle to the left in each collision pair.
    - The subtraction approach for the right-side particles ensures no mass is
        lost in multi-particle collisions (e.g., A<-B and B<-D).
    """
    # Sort collision pairs and remove duplicate collisions (e.g. A-B and B-A)
    sorted_pairs, _ = torch.sort(collision_indices_pairs, dim=1)
    unique_left_indices = remove_duplicates(sorted_pairs, 0)
    unique_indices = remove_duplicates(unique_left_indices, 1)

    # Update velocities based on conservation of momentum
    total_mass = mass[unique_indices[:, 0]] + mass[unique_indices[:, 1]]
    velocity[:, unique_indices[:, 0]] = (
        mass[unique_indices[:, 0]] * velocity[:, unique_indices[:, 0]] +
        mass[unique_indices[:, 1]] * velocity[:, unique_indices[:, 1]]
    ) / total_mass

    # Subtract mass and velocity from the right-side particles
    velocity[:, unique_indices[:, 1]] -= velocity[:, unique_indices[:, 1]]
    mass[unique_indices[:, 0]] += mass[unique_indices[:, 1]]
    mass[unique_indices[:, 1]] -= mass[unique_indices[:, 1]]

    return velocity, mass

