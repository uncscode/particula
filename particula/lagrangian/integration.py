"""Integration methods for Lagrangian particle simulations."""

from typing import Tuple
import torch


def leapfrog(
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
