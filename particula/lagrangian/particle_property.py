"""Particle self calculation. e.g. radius, surface area, volume, etc."""

from typing import Union
import numpy as np
import torch

from particula.util import friction_factor, dynamic_viscosity, \
    slip_correction, mean_free_path, knudsen_number
from particula.util.input_handling import convert_units


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

    Args:
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


def mass(
        radius: torch.Tensor,
        density: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the mass of a sphere given its radius and density using the
    formula for the volume of a sphere.

    This function assumes a uniform density and spherical shape to compute the
    mass based on the mass-density relationship:
    Volume = Mass / Density. The volume of a sphere is given by
    (4/3) * pi * radius^3.

    Args:
    - radius (torch.Tensor): A tensor containing the radius of the sphere(s).
        Can be a scalar or a vector.
    - density (torch.Tensor): A tensor containing the density of the sphere(s).
        Can be a scalar or a vector.

    Returns:
    torch.Tensor: A tensor of the same shape as `radius` and `density`
        representing the mass of the sphere(s).
    """
    return 4 * np.pi * radius ** 3 * density / 3


def friction_factor_wrapper(
        radius_meter: torch.Tensor,
        temperature_kelvin: torch.Tensor,
        pressure_pascal: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper for the friction factor function.

    Args:
    - radius_meter (torch.Tensor): A tensor containing the radius of the
        sphere(s). Can be a scalar or a vector.


    Returns:
    torch.Tensor: A tensor of the same shape as `radius_meter`,
    """
    # get dynamic viscosity
    dynamic_viscosity_value = dynamic_viscosity.dyn_vis(
        temperature_kelvin=temperature_kelvin,
        pressure_pascal=pressure_pascal,
    )
    # get mean free path
    mean_free_path_meter = mean_free_path.mfp(
        temperature_kelvin=temperature_kelvin,
        pressure_pascal=pressure_pascal,
        dynamic_viscosity=dynamic_viscosity_value,
    )
    # get knudsen number
    knudsen = knudsen_number.knu(
        radius=radius_meter,
        mfp=mean_free_path_meter,
    )
    # get slip correction factor
    slip_correction_factor = slip_correction.scf(
        radius=radius_meter,
        knu=knudsen,
    )

    particle_friction_factor = friction_factor.frifac(
        radius=radius_meter,
        dynamic_viscosity=dynamic_viscosity_value,
        scf_val=slip_correction_factor,
    ).m

    return particle_friction_factor


def generate_particle_masses(
        mean_radius: float,
        std_dev_radius: float,
        density: torch.Tensor,
        num_particles: int,
        radius_input_units: str = "nm",
) -> torch.Tensor:
    """
    Generate particle masses based on a log-normal distribution of particle
        sizes.

    Args:
        mean_size_nm (float): Mean size of the particles in nanometers.
        std_dev_nm (float): Standard deviation of the particle sizes in
            nanometers.
        density (float): Density of the particles in kg/m^3.
        num_particles (int): Number of particles.

    Returns:
        torch.Tensor: An array of particle masses.
    """
    # Convert mean and standard deviation to log-scale
    mean_log = np.log(mean_radius)
    std_dev_log = np.log(std_dev_radius)

    # Sample from log-normal distribution
    radius_normal = torch.distributions.log_normal.LogNormal(
        mean_log, std_dev_log).sample((num_particles,))
    radius_normal = radius_normal * convert_units(
        old=radius_input_units, new='m')  # Convert to meters

    # Calculate mass of each particle
    return mass(radius=radius_normal, density=density)
