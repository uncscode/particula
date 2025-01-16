"""
Sedimentation kernel for aerosol particles.

Seinfeld and Pandis (2016), Chapter 13, Equation 13A.4.
"""

from numpy.typing import NDArray
import numpy as np

from particula.particles.properties.settling_velocity import (
    particle_settling_velocity_via_system_state,
)
from particula.util.validate_inputs import validate_inputs


def calculate_collision_efficiency_function(
    radius1: float, radius2: float
) -> float:
    """
    Placeholder function to calculate collision efficiency.

    Args:
    -----
        - radius1 : Radius of the first particle [m].
        - radius2 : Radius of the second particle [m].

    Returns:
    --------
        - Collision efficiency [dimensionless].
    """
    # Implement the actual collision efficiency calculation here
    raise NotImplementedError(
        "Collision efficiency calculation not implemented yet."
    )


def sedimentation_sp2016(
    radius_particle: NDArray[np.float64],
    settling_velocities: NDArray[np.float64],
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]:
    """
    Calculate the sedimentation kernel for aerosol particles.

    Coagulation occurring due to gravitational settling when heavier particles
    settle faster than lighter ones, catching up and colliding with them. The
    coagulation coefficient is proportional to the product of the target area,
    the relative distance swept by the larger particle per unit time, and the
    collision efficiency

    Args:
    -----
        - radius_particle : Array of particle radii [m].
        - settling_velocities : Array of particle settling velocities [m/s].
        - calculate_collision_efficiency : Boolean to calculate collision
            efficiency or use 1.

    Returns:
    --------
        - Sedimentation kernel matrix for aerosol particles [m^3/s].

    References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Chapter 13, Equation 13A.4.
    """
    diameter_matrix = 2 * (
        radius_particle[:, np.newaxis] + radius_particle[np.newaxis, :]
    )
    velocity_diff_matrix = np.abs(
        settling_velocities[:, np.newaxis] - settling_velocities[np.newaxis, :]
    )

    if calculate_collision_efficiency:
        collision_efficiency_matrix = np.vectorize(
            calculate_collision_efficiency_function
        )(radius_particle[:, np.newaxis], radius_particle[np.newaxis, :])
    else:
        collision_efficiency_matrix = np.ones_like(diameter_matrix)

    return (
        np.pi
        / 4
        * diameter_matrix**2
        * velocity_diff_matrix
        * collision_efficiency_matrix
    )


@validate_inputs({"temperature": "positive", "pressure": "positive"})
def sedimentation_sp2016_via_system_state(
    radius_particle: NDArray[np.float64],
    density_particle: NDArray[np.float64],
    temperature: float,
    pressure: float,
    calculate_collision_efficiency: bool = True,
) -> NDArray[np.float64]:
    """
    Calculate the sedimentation kernel for aerosol particles via system state.

    Args:
    -----
        - radius_particle : Array of particle radii [m].
        - density_particle : Array of particle densities [kg/m³].
        - temperature : Temperature of the system [K].
        - pressure : Pressure of the system [Pa].
        - calculate_collision_efficiency : Boolean to calculate collision
            efficiency or use 1.

    Returns:
    --------
        - Sedimentation kernel matrix for aerosol particles [m^3/s].
    """
    settling_velocities = particle_settling_velocity_via_system_state(
        particle_radius=radius_particle,
        particle_density=density_particle,
        temperature=temperature,
        pressure=pressure,
    )

    return sedimentation_sp2016(
        radius_particle=radius_particle,
        settling_velocities=settling_velocities,
        calculate_collision_efficiency=calculate_collision_efficiency,
    )
