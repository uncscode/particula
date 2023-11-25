"""Boundary conditions for Lagrangian particle simulations."""

import torch


def wrapped_cube(
        position: torch.Tensor, cube_side: float) -> torch.Tensor:
    """
    Apply cubic boundary conditions with wrap-around, to the position tensor.

    This function modifies positions that exceed the cubic domain side,
    wrapping them around to the opposite side of the domain. It handles both
    positive and negative overflows. The center of the cube is assumed to be
    at zero.

    Parameters:
    - position (torch.Tensor): A tensor representing positions that might
        exceed the domain boundaries. [3, num_particles]
    - cube_side (float): The cube side length of the domain.

    Returns:
    - torch.Tensor: The modified position tensor with boundary conditions
        applied.

    Example:
    position = torch.tensor([...])  # Position tensor
    cube_side = 10.0  # Define the domain
    wrapped_position = boundary.wrapped_cube(position,
        cube_side)
    """
    half_cube_side = cube_side / 2
    # Wrap around for positive overflow
    position = torch.where(
        position > half_cube_side,
        position - cube_side,
        position)

    # Wrap around for negative overflow
    position = torch.where(
        position < -half_cube_side,
        position + cube_side,
        position)
    return position
