"""Particle self calculation. e.g. radius, surface area, volume, etc."""

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
