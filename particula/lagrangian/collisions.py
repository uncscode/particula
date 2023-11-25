"""Operations for handling collisions between particles."""

from typing import Tuple
import torch

from particula.lagrangian import particle_pairs


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
    unique_left_indices = particle_pairs.remove_duplicates(sorted_pairs, 0)
    unique_indices = particle_pairs.remove_duplicates(unique_left_indices, 1)

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
