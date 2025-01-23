"""
Particle inertia time calculation.
"""
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_density": "positive",
        "fluid_density": "positive",
        "kinematic_viscosity": "positive",
        "relative_velocity": "positive",
    }
)
def get_particle_inertia_time(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: Union[float, NDArray[np.float64]],
    kinematic_viscosity: Union[float, NDArray[np.float64]],
    relative_velocity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the particle inertia time.

    The particle inertia time (τ_p) represents the response time of a particle
    to velocity changes in the surrounding fluid and is given by:

        τ_p = (2 / 9) * (ρ_p / ρ_f) * (r² / (ν f(Re_p)))

    - τ_p : Particle inertia time [s]
    - r (particle_radius) : Particle radius [m]
    - ρ_p (particle_density) : Particle (e.g., droplet, dust) density [kg/m³]
    - ρ_f (fluid_density) : Density of the surrounding fluid (e.g., air, water)
        [kg/m³]
    - ν (kinematic_viscosity) : Kinematic viscosity of the fluid [m²/s]
    - f(Re_p) : Drag correction factor based on the particle Reynolds number
        [-]

    The drag correction factor f(Re_p) is given by:

        f(Re_p) = 1 + 0.15 Re_p^(0.687)

    where the particle Reynolds number is:

        Re_p = (2 r v) / v

    - v (relative_velocity) : Relative velocity between particle and fluid
        [m/s]

    Arguments:
    ----------
        - particle_radius : Radius of the particle [m]
        - particle_density : Density of the particle [kg/m³]
        - fluid_density : Density of the surrounding fluid [kg/m³]
        - kinematic_viscosity : Kinematic viscosity of the fluid [m²/s]
        - relative_velocity : Relative velocity between particle and fluid
            [m/s]

    Returns:
    --------
        - Particle inertia time [s]

    References:
    ----------
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    re_p = (2 * particle_radius * relative_velocity) / kinematic_viscosity
    drag_correction = 1 + 0.15 * re_p**0.687
    return (
        (2 / 9)
        * (particle_density / fluid_density)
        * (particle_radius**2 / (kinematic_viscosity * drag_correction))
    )
