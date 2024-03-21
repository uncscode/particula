"""Particle self calculation. e.g. radius, surface area, volume, etc."""

import numpy as np
import torch

from particula.constants import BOLTZMANN_CONSTANT
from particula.util import friction_factor, dynamic_viscosity, \
    slip_correction, mean_free_path, knudsen_number
from particula.util.input_handling import convert_units


def radius_calculation(
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


def mass_calculation(
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
        temperature_kelvin: float,
        pressure_pascal: float,
) -> torch.Tensor:
    """
    Calculate the friction factor for a given radius, temperature, and
    pressure.

    This function wraps several underlying calculations related to
    dynamic viscosity, mean free path, Knudsen number, and slip correction
    factor to compute the particle friction factor.

    Args:
        radius_meter: A tensor representing the radius of the
        sphere(s) in meters. Can be a scalar or a vector.
        temperature_kelvin: A tensor of the temperature in Kelvin.
        pressure_pascal: A tensor of the pressure in Pascals.

    Returns:
        torch.Tensor: A tensor of the same shape as `radius_meter`,
        representing the particle friction factor.
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

    return friction_factor.frifac(
        radius=radius_meter,
        dynamic_viscosity=dynamic_viscosity_value,
        scf_val=slip_correction_factor,
    ).m


def generate_particle_masses(
        mean_radius: float,
        std_dev_radius: float,
        density: torch.Tensor,
        num_particles: int,
        radius_input_units: str = "nm",
) -> torch.Tensor:
    """
    Generate an array of particle masses based on a log-normal distribution of
    particle radii and a given density.

    Args:
        mean_radius (float): Mean radius of the particles. The units are
            specified by `radius_input_units`.
        std_dev_radius (float): Standard deviation of the particle radii. The
            units are specified by `radius_input_units`.
        density (torch.Tensor): Density of the particles in kg/m^3.
        num_particles (int): Number of particles to generate.
        radius_input_units (str, optional): Units of `mean_radius` and
            `std_dev_radius`. Defaults to 'nm' (nanometers).

    Returns:
        torch.Tensor: A tensor of particle masses in kg, corresponding to each
            particle.

    Raises:
        ValueError: If `mean_radius` or `std_dev_radius` are non-positive.
    """
    if mean_radius <= 0 or std_dev_radius <= 0:
        raise ValueError(
            "Mean radius and standard deviation must be positive.")

    # Convert mean and standard deviation from the specified units to meters
    mean_log = np.log(mean_radius)
    std_dev_log = np.log(std_dev_radius)
    if mean_log < 0 or std_dev_log < 0:
        # need to check on this error, not clear why this happens
        raise ValueError(
            "log of Mean radius and standard deviation must be positive for"
            + " torch.distributions.log_normal.LogNormal")

    # Sample radii from the log-normal distribution
    radius_samples = torch.distributions.log_normal.LogNormal(
        mean_log, std_dev_log).sample((num_particles,))
    radius_samples *= torch.asarray(
        convert_units(radius_input_units, "m"))

    # Calculate mass of each particle
    return mass_calculation(radius=radius_samples, density=density)


def thermal_speed(
        temperature_kelvin: float,
        mass_kg: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the thermal speed of a particle based on its temperature and
    mass.

    The thermal speed is computed using the formula: sqrt(8 * k * T / (pi * m))
    where k is the Boltzmann constant, T is the temperature in Kelvin, and m is
    the particle mass in kilograms.

    Args:
        temperature_kelvin (float): Temperature of the environment in Kelvin.
        mass_kg (torch.Tensor): Mass of the particle(s) in kilograms.
            Can be a scalar or a vector.

    Returns:
        torch.Tensor: The thermal speed of the particle(s) in meters per second

    Raises:
        ValueError: If `temperature_kelvin` is less than or equal to zero or
        if any element of `mass_kg` is non-positive.
    """
    if temperature_kelvin <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if torch.any(mass_kg <= 0):
        raise ValueError("All mass values must be positive.")

    return torch.sqrt(8 * BOLTZMANN_CONSTANT.magnitude *
                      temperature_kelvin / (np.pi * mass_kg))


def speed(
    velocity: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the speed of a particle.

    Args:
        velocity (torch.Tensor): Velocity of the particle.

    Returns:
        torch.Tensor: Speed of the particle.
    """
    return torch.sqrt(torch.sum(velocity**2, dim=0))


def random_thermal_velocity(
    temperature_kelvin: float,
    mass_kg: torch.Tensor,
    number_of_particles: int,
    t_type=torch.float,
    random_seed: int = 0,
) -> torch.Tensor:
    """
    Generate a random thermal velocity for each particle.

    Args:
        temperature_kelvin (torch.Tensor): Temperature of the fluid in Kelvin.
        mass_kg (torch.Tensor): Mass of the particle in kilograms.
        number_of_particles (int): Number of particles.

    Returns:
        torch.Tensor: Thermal speed of the particle in meters per second.
    """
    # set the seed
    torch.manual_seed(random_seed)
    # Initialize particle velocities uniformly random
    unit_velocity = torch.rand(3, number_of_particles, dtype=t_type) - 0.5
    # get the speed
    unit_speed = speed(unit_velocity)
    # get the thermal speed
    thermal_particle_speed = thermal_speed(
        temperature_kelvin=temperature_kelvin,
        mass_kg=mass_kg,
    )
    # normalize the unit
    return unit_velocity * thermal_particle_speed / unit_speed


def nearest_match(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    x_new: torch.Tensor,
) -> torch.Tensor:
    """
    Perform nearest neighbor interpolation (on torch objects) to find y-values
    corresponding to new x-values. The function identifies the nearest x-value
    for each value in x_new and returns the corresponding y-value.

    Args:
        x_values (torch.Tensor): The original x-values of shape (n,).
        y_values (torch.Tensor): The original y-values of shape (n,).
            Each y-value corresponds to an x-value.
        x_new (torch.Tensor): The new x-values for which y-values are to be
            interpolated, of shape (m,).

    Returns:
        torch.Tensor: The interpolated y-values of shape (m,). Each value
            corresponds to the nearest match from x_values.
    """
    # Reshape x_values and x_new for broadcasting
    x_values = x_values.unsqueeze(0)
    x_new = x_new.unsqueeze(1)
    # Find the indices of the nearest x-values in x_values for each value in
    # x_new
    idx = torch.argmin(torch.abs(x_values - x_new), dim=1)
    # Return the corresponding y-values for the found indices
    return y_values[idx]
